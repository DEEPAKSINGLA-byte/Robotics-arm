"""Offline reference-assisted projection audit using training scenes ONLY.

Estimated effective intrinsics are not a factory calibration certificate.
No test data, model training, or RGB-depth inference is performed.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image
from build_error_review import inside
from camera_geometry import Camera, project_points
from run_experiment2 import file_hash, save_json


def read_pcd(path):
    """Read the exact uncompressed organised XYZRGBL binary format in local OCID."""
    with Path(path).open('rb') as f:
        header = {}
        for _ in range(32):
            line = f.readline().decode('ascii').strip()
            if not line or line.startswith('#'):
                continue
            key, *value = line.split()
            header[key] = value
            if key == 'DATA':
                break
        expected = dict(FIELDS=['x', 'y', 'z', 'rgba', 'label'], SIZE=['4']*5,
                        TYPE=['F', 'F', 'F', 'U', 'U'], COUNT=['1']*5, DATA=['binary'])
        if any(header.get(k) != v for k, v in expected.items()):
            raise ValueError('Unsupported PCD layout; expected OCID uncompressed XYZRGBL')
        w, h = int(header['WIDTH'][0]), int(header['HEIGHT'][0])
        if int(header['POINTS'][0]) != w*h or w <= 0 or h <= 1:
            raise ValueError('Organised image-sized cloud required')
        dtype = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('rgba', '<u4'), ('label', '<u4')])
        payload = f.read()
        if len(payload) != w*h*dtype.itemsize:
            raise ValueError('PCD payload size mismatch')
        return np.frombuffer(payload, dtype=dtype).reshape(h, w), header


def select_training_scenes(manifest, seed=42):
    if manifest['split'] != 'train' or manifest['mode'] != 'sequence':
        raise ValueError('Calibration audit uses sequence training data only')
    groups = defaultdict(lambda: defaultdict(set))
    for row in manifest['rows']:
        key = '/'.join(Path(row['scene']).parts[:3])
        groups[key][row['group']].add(row['scene'])
    rng, selected = random.Random(seed), {}
    for partition, sequences in sorted(groups.items()):
        if len(sequences) < 3:
            raise ValueError('Need three training sequences per partition for fit/check separation')
        chosen = rng.sample(sorted(sequences), 3)
        selected[partition] = [dict(scene=rng.choice(sorted(sequences[g])), group=g,
                                     role='fit' if i < 2 else 'check') for i, g in enumerate(chosen)]
    return selected


def sample_projection(cloud, stride=7):
    v, u = np.indices(cloud.shape)
    points = np.stack([cloud[k] for k in ('x', 'y', 'z')], axis=-1).astype(float)
    good = np.isfinite(points).all(axis=-1) & (points[..., 2] > 0)
    sampled = good & (u % stride == 0) & (v % stride == 0)
    return points, good, points[sampled], np.stack((u[sampled], v[sampled]), axis=-1)


def fit_camera(points, uv, width, height):
    if len(points) < 100:
        raise ValueError('Insufficient valid fitting points')
    xz, yz = points[:, 0]/points[:, 2], points[:, 1]/points[:, 2]
    fx, cx = np.linalg.lstsq(np.column_stack((xz, np.ones(len(points)))), uv[:, 0], rcond=None)[0]
    fy, cy = np.linalg.lstsq(np.column_stack((yz, np.ones(len(points)))), uv[:, 1], rcond=None)[0]
    return Camera(width, height, float(fx), float(fy), float(cx), float(cy))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, default=Path('splits/sequence/train.json'))
    p.add_argument('--ocid-root', type=Path, default=Path('data/OCID-dataset'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('Choose a fresh audit output directory')
    manifest = json.loads(args.manifest.read_text())
    selection = select_training_scenes(manifest, args.seed)
    report, estimates = {}, {}
    for partition, rows in selection.items():
        records, fit_points, fit_uv = [], [], []
        shape = None
        for row in rows:
            relative = Path(row['scene'])
            pcd_path = inside(args.ocid_root, relative.parent.parent/'pcd'/relative.with_suffix('.pcd').name)
            cloud, header = read_pcd(pcd_path)
            if shape is not None and shape != cloud.shape:
                raise ValueError('Image sizes differ within partition')
            shape = cloud.shape
            points, good, sampled, uv = sample_projection(cloud)
            if row['role'] == 'fit':
                fit_points.append(sampled)
                fit_uv.append(uv)
            with Image.open(inside(args.ocid_root, relative)) as im:
                rgb = np.array(im.convert('RGB'))
            with Image.open(inside(args.ocid_root, relative.parent.parent/'depth'/relative.name)) as im:
                depth = np.array(im)
            if depth.shape != shape or rgb.shape != (*shape, 3):
                raise ValueError('RGB/depth/PCD shapes differ')
            rgba = cloud['rgba']
            colors = np.stack(((rgba>>16)&255, (rgba>>8)&255, rgba&255), axis=-1)
            matched = float(np.mean(np.all(rgb == colors, axis=-1)))
            valid_depth = good & (depth > 0)
            if not valid_depth.any():
                raise ValueError('No valid depth reference')
            depth_error = np.abs(depth[valid_depth]/1000.-points[..., 2][valid_depth])
            records.append(dict(**row, pcd_sha256=file_hash(pcd_path), width=shape[1], height=shape[0],
                                rgb_pointcloud_exact_fraction=matched,
                                depth_z_mean_abs_error_m=float(depth_error.mean()),
                                depth_z_p99_abs_error_m=float(np.quantile(depth_error, .99)),
                                valid_point_fraction=float(good.mean()), viewpoint=header.get('VIEWPOINT'),
                                points=sampled, uv=uv))
        camera = fit_camera(np.concatenate(fit_points), np.concatenate(fit_uv), shape[1], shape[0])
        for record in records:
            error = np.linalg.norm(project_points(record.pop('points'), camera)-record.pop('uv'), axis=-1)
            record.update(reprojection_mean_px=float(error.mean()), reprojection_p99_px=float(np.quantile(error, .99)))
        estimates[partition] = dict(camera=camera.to_dict(), status='reference_assisted_effective_pinhole_estimate',
                                    source='Two training-sequence organised PCDs; checked on a third training sequence',
                                    distortion_status='not independently calibrated', robot_transform=None)
        report[partition] = records
        check = next(r for r in records if r['role'] == 'check')
        print(f'{partition}: fx={camera.fx:.3f}, fy={camera.fy:.3f}, cx={camera.cx:.3f}, cy={camera.cy:.3f}; '
              f'held-out training-sequence p99 reprojection error {check["reprojection_p99_px"]:.3f}px', flush=True)
    args.output.mkdir(parents=True)
    save_json(args.output/'estimated_cameras.json', estimates)
    save_json(args.output/'audit.json', dict(seed=args.seed, split='train', fingerprint=manifest['fingerprint'],
              reference_only=True, test_used=False, factory_calibration=False, training_performed=False,
              scene_count=sum(map(len, selection.values())), partitions=report,
              notes=['Depth reference is millimetres; PCD Z checked against depth/1000.',
                     'Reference geometry used only for offline calibration audit, not RGB inference.',
                     'Identity PCD VIEWPOINT is not camera-to-robot calibration.',
                     'Do not interpret effective intrinsics as a physical calibration certificate.']))


if __name__ == '__main__':
    main()
