"""Small real-scene verification after RGB-only Task 3 predictions are saved.

This reads validation labels only for scoring. It never tunes grasp parameters,
and does not measure physical grasp success or claim held-out test accuracy.
"""
import argparse
import base64
import html
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from task3_grasp_proposals import validate_transform, observed_collision_count

REPO = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text())


def verify(runs, data_root, manifest):
    data_root = Path(data_root).resolve()
    index = read(manifest)
    if index.get('split') != 'val' or index.get('mode') != 'sequence':
        raise ValueError('Use the sequence validation manifest, not train or test data.')
    annotations = index['rows']
    records = []
    for run in runs:
        result = read(run/'result.json')
        scene = str(Path(result['image']).resolve().relative_to(data_root))
        matches = [r for r in annotations if r['scene'] == scene and r['sentence'] == result['sentence']]
        target_ids = {r['target_id'] for r in matches}
        if len(target_ids) != 1:
            raise ValueError('Verification requires a unique annotated validation expression')
        source = data_root/scene
        labels = np.array(Image.open(source.parent.parent/'label'/source.name))
        target = labels == target_ids.pop()
        selected = np.array(Image.open(run/'selection/selected_mask.png')) > 0
        overlap = float((target & selected).sum()/max((target | selected).sum(), 1))
        if result['status'] == 'no_candidates':
            records.append(dict(run=str(run.resolve()), sentence=result['sentence'], scene=scene,
                mask_iou=overlap, selected_target=False, proposals=0,
                pose_and_observed_collision_checks_passed=0, contact_seeds_on_annotated_target=0,
                status='no_candidates', robot_execution_ready=False))
            continue
        report = read(run/'grasp_proposals.json')
        with np.load(run/'depth/prediction.npz') as a:
            xyz, valid = a['points_m'], a['mask'].astype(bool)
        valid &= np.isfinite(xyz).all(-1) & (xyz[..., 2] > 0)
        visible = xyz[valid].astype(float)
        tree = cKDTree(visible)
        geometry_passed, contacts_on_target = 0, 0
        c = report['config']
        for proposal in report['proposals']:
            contact_pose = validate_transform(proposal['T_camera_tool_contact'])
            pregrasp = validate_transform(proposal['T_camera_tool_pregrasp'])
            center = np.array(proposal['contact_camera_m'])
            normal = np.array(proposal['outward_normal_camera'])
            np.testing.assert_allclose(contact_pose[:3, 3], center)
            np.testing.assert_allclose(contact_pose[:3, 2], -normal, atol=1e-6)
            np.testing.assert_allclose(pregrasp[:3, 3]-center, normal*c['approach_distance_m'], atol=1e-6)
            collisions = observed_collision_count(visible, tree, center, normal,
                c['cup_radius_m']+c['tool_clearance_m'], c['approach_distance_m'], c['contact_tolerance_m'])
            assert collisions == 0, 'Reported clear path contains observed geometry'
            assert proposal['plane_rms_m'] <= c['max_plane_rms_m']
            assert proposal['footprint_coverage'] >= c['min_footprint_coverage']
            x, y = proposal['contact_pixel_xy']
            assert selected[y, x], 'Contact seed lies outside selected mask'
            contacts_on_target += int(target[y, x])
            geometry_passed += 1
        records.append(dict(run=str(run.resolve()), sentence=result['sentence'], scene=scene,
            mask_iou=overlap, selected_target=overlap >= .5,
            proposals=len(report['proposals']), pose_and_observed_collision_checks_passed=geometry_passed,
            contact_seeds_on_annotated_target=contacts_on_target,
            status=report['status'], robot_execution_ready=result['robot_execution_ready']))
    return dict(scope='Small illustrative validation check; no physical grasp trials or test-set evaluation',
                scenes=len(records), selected_target_count=sum(r['selected_target'] for r in records),
                physical_grasp_success_rate=None, records=records)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--data-root', type=Path, required=True, help='OCID folder containing ARID and YCB scenes')
    p.add_argument('--manifest', type=Path, default=REPO/'splits/sequence/val.json')
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('Choose a new verification folder')
    summary = verify(args.runs, args.data_root, args.manifest)
    args.output.mkdir(parents=True)
    (args.output/'verification.json').write_text(json.dumps(summary, indent=2)+'\n')
    cards = []
    for r in summary['records']:
        preview_path = Path(r['run'])/'grasp_preview.png'
        if not preview_path.is_file():
            preview_path = Path(r['run'])/'selection/selected_mask.png'
        preview = base64.b64encode(preview_path.read_bytes()).decode()
        cards.append(f'<article><h2>{html.escape(r["sentence"])}</h2>'
                     f'<p>Mask overlap: {r["mask_iou"]:.1%}; {r["proposals"]} grasp hypotheses; '
                     f'{r["pose_and_observed_collision_checks_passed"]} pass the implemented geometry checks.</p>'
                     f'<img src="data:image/png;base64,{preview}" alt="Selected object and proposed contacts"></article>')
    page = ('<!doctype html><meta charset="utf-8"><title>Task 3: retained pipeline and grasp proposals</title>'
            '<style>body{font:17px system-ui;max-width:960px;margin:32px auto;padding:0 20px;background:#f3f6fa;color:#172438}'
            'article{padding:24px;background:white;border-radius:12px;margin:24px 0}img{width:100%;max-width:800px}h2{font-size:21px}</style>'
            '<h1>Task 3: image and instruction to grasp proposals</h1>'
            '<p>The existing SAM + SigLIP selector is retained. MoGe predicts the scene geometry from RGB. '
            'Orange dots are geometric suction-contact hypotheses; the arrow marks the best proposal’s approach.</p>'
            f'<p>Correct object selected in {summary["selected_target_count"]}/{summary["scenes"]} illustrative validation scenes. '
            'These examples are a functionality check, not a new benchmark.</p>'
            '<p><b>These proposals have not been physically tested.</b> The default cup diameter is an assumed 20 mm. '
            'Material sealing, hidden surfaces, robot calibration, reachability and complete arm collisions remain unverified.</p>'
            + ''.join(cards))
    (args.output/'report.html').write_text(page)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
