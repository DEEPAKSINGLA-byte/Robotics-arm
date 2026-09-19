"""Find the requested object in an RGB image and suggest suction contacts.

Uses the existing OCID models and environments without changing their weights.
No dataset annotations or measured depth are needed for inference.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from bin_grasp.io import file_hash as sha, save_json as save

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
REPO = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def load_config(path):
    """Model paths are relative to this repository, or can be absolute."""
    config = read(path)
    for key in ('checkpoint', 'sam_path', 'siglip_path', 'depth_head', 'moge_checkpoint'):
        config[key] = str((REPO / config[key]).resolve())
    return config


def select(args):
    import numpy as np
    from PIL import Image
    import torch
    from transformers import AutoModel, AutoProcessor, pipeline
    from bin_grasp.candidates import candidate_inputs
    from experiments.extract_siglip_features import encoder_fingerprint, pooled
    from bin_grasp.model import GroundingModel
    from experiments.run_experiment2 import StableMaskGenerationPipeline, generate_masks, filter_proposals

    if not torch.cuda.is_available():
        raise RuntimeError('Run in a session with GPU access.')
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.backends.mha.set_fastpath_enabled(False)
    plan = load_config(args.config)
    # Check that these are the same models used for the saved results.
    selector_path = Path(plan['checkpoint'])
    if sha(selector_path) != plan['checkpoint_sha256']:
        raise ValueError('The retained selector weights have changed.')
    if sha(Path(plan['sam_path']) / 'model.safetensors') != plan['sam_weights_sha256']:
        raise ValueError('The retained SAM weights have changed.')
    if encoder_fingerprint(plan['siglip_path']) != plan['encoder_fingerprint']:
        raise ValueError('The retained SigLIP weights/preprocessing have changed.')
    checkpoint = torch.load(selector_path, map_location='cpu', weights_only=True)
    args.output.mkdir(parents=True)
    rgb = Image.open(args.image).convert('RGB')
    started = time.perf_counter()
    print('Generating masks with the retained relaxed SAM settings...', flush=True)
    generator = pipeline('mask-generation', model=plan['sam_path'], device=0,
                         dtype=torch.float32, pipeline_class=StableMaskGenerationPipeline)
    generator.model.eval().requires_grad_(False)
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        output = generate_masks(generator, rgb, plan['sam_settings'])
    raw = np.asarray(output['masks'], bool) if len(output['masks']) else np.empty((0, rgb.height, rgb.width), bool)
    quality = np.array([float(x) for x in output['scores']], np.float32)
    f = plan['filtering']
    kept, stats = filter_proposals(raw, quality, (rgb.height, rgb.width), f['min_area'],
                                  f['max_area_fraction'], f['duplicate_iou'], f['max_candidates'])
    masks = raw[kept]
    del generator, output, raw
    gc.collect()
    torch.cuda.empty_cache()
    record = dict(sentence=args.sentence, selected_index=None, candidate_count=len(masks),
                  selector_scores=[], score_interpretation='Ranking scores, not calibrated confidence',
                  checkpoint=str(selector_path), checkpoint_sha256=sha(selector_path),
                  sam_settings=plan['sam_settings'], filtering=f, proposal_statistics=stats,
                  rgb_sha256=sha(args.image), annotations_used=False)
    if len(masks):
        print(f'Choosing from {len(masks)} masks with SigLIP and the retained selector...', flush=True)
        processor = AutoProcessor.from_pretrained(plan['siglip_path'], local_files_only=True)
        encoder = AutoModel.from_pretrained(plan['siglip_path'], local_files_only=True,
                                           dtype=torch.float16).cuda().eval().requires_grad_(False)
        selector = GroundingModel(**checkpoint['config']).cuda().eval().requires_grad_(False)
        selector.load_state_dict(checkpoint['model'])
        crops, geometry = candidate_inputs(np.asarray(rgb), masks)
        images = crops + [rgb]
        pieces = []
        with torch.inference_mode():
            for offset in range(0, len(images), 8):
                batch = processor(images=images[offset:offset+8], return_tensors='pt')
                pieces.append(pooled(encoder.get_image_features(pixel_values=batch['pixel_values'].to('cuda', torch.float16))))
            features = torch.cat(pieces).half().float().cuda()
            token_count = len(processor.tokenizer(args.sentence, truncation=False)['input_ids'])
            if token_count > 64:
                raise ValueError('Sentence exceeds the 64-token training limit; shorten it to avoid silent truncation.')
            tokens = processor(text=[args.sentence], padding='max_length', max_length=64,
                               truncation=True, return_tensors='pt').to('cuda')
            text = pooled(encoder.get_text_features(**tokens)).half().float().cuda()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                scores = selector(visual=features[:-1][None], text=text, scene=features[-1:],
                                  geometry=torch.from_numpy(geometry)[None].cuda(),
                                  valid=torch.ones((1, len(masks)), dtype=torch.bool, device='cuda'))[0]
        record.update(selected_index=int(scores.argmax()), selector_scores=scores.float().cpu().tolist())
    selected = masks[record['selected_index']] if len(masks) else np.zeros((rgb.height, rgb.width), bool)
    Image.fromarray(selected.astype('uint8')*255).save(args.output / 'selected_mask.png')
    np.savez_compressed(args.output / 'candidates.npz', masks=masks, retained_raw_indices=np.array(kept))
    record['selection_stage_seconds'] = time.perf_counter() - started
    save(args.output / 'selection.json', record)
    print(f'Selection saved: candidate {record["selected_index"]}', flush=True)


def assemble(args, start, stage_seconds):
    import cv2
    import numpy as np
    from PIL import Image
    selection = read(args.output / 'selection/selection.json')
    rgb = np.asarray(Image.open(args.image).convert('RGB')).copy()
    mask = np.asarray(Image.open(args.output / 'selection/selected_mask.png')) > 0
    with np.load(args.output / 'depth/prediction.npz') as data:
        xyz = data['points_m']
        valid = data['mask'].astype(bool) & mask & np.isfinite(xyz).all(-1) & (xyz[..., 2] > 0)
        depth = data['depth_m']
        if xyz.shape != (*mask.shape, 3) or depth.shape != mask.shape:
            raise ValueError('Depth and mask dimensions do not match.')
        if not np.allclose(xyz[..., 2][valid], depth[valid], rtol=1e-4, atol=1e-5):
            raise ValueError('The saved depth does not match the 3D points.')
        intrinsics = data['intrinsics']
    points, colors = xyz[valid], rgb[valid]
    np.savez_compressed(args.output / 'selected_object_3d.npz', xyz_camera_m=points,
                        rgb=colors, image_mask=mask, valid_geometry_mask=valid,
                        predicted_normalized_intrinsics=intrinsics)
    # Small standard point-cloud export; all valid points remain in the NPZ.
    indexes = np.linspace(0, max(0, len(points)-1), min(10000, len(points)), dtype=int)
    with (args.output / 'selected_object.ply').open('w') as stream:
        stream.write('ply\nformat ascii 1.0\n' + f'element vertex {len(indexes)}\n' +
                     'property float x\nproperty float y\nproperty float z\n'
                     'property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')
        for i in indexes:
            stream.write(' '.join([*(f'{float(v):.7f}' for v in points[i]), *(str(int(v)) for v in colors[i])]) + '\n')
    preview = rgb.copy()
    preview[mask] = (preview[mask]*.65 + np.array([30, 220, 120])*.35).astype('uint8')
    contours, _ = cv2.findContours(mask.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(preview, contours, -1, (30, 255, 120), 2)
    Image.fromarray(preview).save(args.output / 'selected_object.png')
    ys, xs = np.where(mask)
    result = dict(image=str(args.image), sentence=args.sentence,
                  status='estimated_surface_position' if len(points) else 'no_valid_selected_geometry',
                  selected_candidate=selection['selected_index'], candidate_count=selection['candidate_count'],
                  mask_pixels=int(mask.sum()), valid_surface_points=len(points),
                  valid_depth_fraction=float(valid.sum()/max(mask.sum(), 1)),
                  bbox_xyxy=[int(xs.min()), int(ys.min()), int(xs.max()+1), int(ys.max()+1)] if len(xs) else None,
                  visible_surface_centroid_camera_m=points.mean(0).tolist() if len(points) else None,
                  median_surface_point_camera_m=np.median(points, axis=0).tolist() if len(points) else None,
                  coordinate_frame='camera: +X right, +Y down, +Z forward; metres',
                  position_definition='Mean of predicted visible surface points; not a grasp point or complete-object centre',
                  grasp_pose=None, robot_transform=None,
                  source='Fresh RGB inference; no supplied depth, masks, intrinsics or target ID',
                  weights_changed=False, stage_wall_seconds=stage_seconds,
                  total_wall_seconds=time.perf_counter()-start)
    save(args.output / 'result.json', result)
    print(json.dumps(result, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image', type=Path, required=True)
    p.add_argument('--sentence', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--config', type=Path, default=REPO / 'configs/task3.json')
    p.add_argument('--depth-python', type=Path,
                   help='Python environment with MoGe installed; defaults to models/moge/.venv/bin/python if present')
    p.add_argument('--cup-radius-m', type=float, default=.010, help='Prototype suction cup radius in metres')
    p.add_argument('--camera-to-robot', type=Path, help='Optional calibrated JSON 4x4 T_robot_camera')
    p.add_argument('--stage', choices=['all', 'select'], default='all', help=argparse.SUPPRESS)
    args = p.parse_args()
    args.image, args.output = args.image.resolve(), args.output.resolve()
    args.config = args.config.resolve()
    if not args.image.is_file() or not args.sentence.strip():
        p.error('An existing RGB image and a nonempty sentence are required.')
    if args.output.exists():
        raise FileExistsError('Choose a new output folder; existing results are preserved.')
    if args.stage == 'select':
        select(args)
        return
    from bin_grasp.grasps import GraspConfig, save_grasps, validate_transform, recovery_instructions
    grasp_config = GraspConfig(cup_radius_m=args.cup_radius_m)
    transform = validate_transform(read(args.camera_to_robot)) if args.camera_to_robot else None
    plan = load_config(args.config)
    selector = Path(plan['checkpoint'])
    depth_head = Path(plan['depth_head'])
    local_python = REPO / 'models/moge/.venv/bin/python'
    depth_python = args.depth_python or (local_python if local_python.is_file() else Path(sys.executable))
    # Keep the venv path: resolving its Python symlink would bypass the environment.
    depth_python = depth_python.absolute()
    for path in (selector, depth_head, Path(plan['moge_checkpoint']), depth_python,
                 Path(plan['sam_path']) / 'model.safetensors', Path(plan['siglip_path'])):
        if not path.exists():
            raise FileNotFoundError(f'Missing model or environment: {path}. See docs/task3-pipeline.md.')
    start = time.perf_counter()
    args.output.mkdir(parents=True)
    hashes = {str(path): sha(path) for path in (selector, depth_head)}
    save(args.output / 'configuration.json', dict(selector=str(selector), depth_head=str(depth_head),
         weight_hashes=hashes, image_sha256=sha(args.image), sentence=args.sentence, settings=plan,
         sam='SAM 2.1 Tiny, relaxed settings', encoder='SigLIP 2 Base', depth='MoGe 2 + learned scale head'))
    t = time.perf_counter()
    # Separate processes let the mask models release GPU memory before depth runs.
    subprocess.run([sys.executable, '-B', '-m', 'bin_grasp.pipeline', '--stage', 'select',
                    '--config', str(args.config),
                    '--image', str(args.image), '--sentence', args.sentence,
                    '--output', str(args.output / 'selection')], cwd=REPO, check=True, timeout=180)
    stage_seconds = {'selection_including_loading': time.perf_counter()-t}
    if read(args.output / 'selection/selection.json')['selected_index'] is None:
        save(args.output / 'result.json', dict(image=str(args.image), sentence=args.sentence,
             status='no_candidates', selected_candidate=None, candidate_count=0,
             grasp_pose=None, grasp_status='no_proposal', grasp_proposal_count=0,
             robot_transform=None, robot_execution_ready=False,
             serial_instructions=recovery_instructions(), weights_changed=False,
             stage_wall_seconds=stage_seconds, total_wall_seconds=time.perf_counter()-start))
        return
    t = time.perf_counter()
    subprocess.run([str(depth_python), '-B', '-m', 'bin_grasp.depth',
                    'predict', '--head', str(depth_head), '--image', str(args.image),
                    '--checkpoint', plan['moge_checkpoint'],
                    '--output', str(args.output / 'depth')], cwd=REPO, check=True, timeout=180)
    stage_seconds['fresh_depth_including_loading'] = time.perf_counter()-t
    if hashes != {path: sha(path) for path in hashes}:
        raise RuntimeError('A checkpoint changed during inference.')
    if sha(args.image) != read(args.output / 'configuration.json')['image_sha256']:
        raise RuntimeError('The input image changed during inference.')
    assemble(args, start, stage_seconds)
    report = save_grasps(args.output, grasp_config, transform)
    print(f'Grasp hypotheses: {len(report["proposals"])}; {report["status"]}.', flush=True)


if __name__ == '__main__':
    main()
