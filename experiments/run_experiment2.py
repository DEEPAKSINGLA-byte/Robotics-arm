"""RGB-only SAM proposals + frozen SigLIP + existing selector; validation diagnostics only."""
import argparse
import base64
from collections import Counter, defaultdict
import gc
import json
import math
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image
import torch
import transformers
from transformers import AutoModel, AutoProcessor, pipeline
from transformers.pipelines.mask_generation import MaskGenerationPipeline

from experiments.build_error_review import inside, outline
from experiments.extract_siglip_features import encoder_fingerprint, pooled
from bin_grasp.model import GroundingModel
from bin_grasp.candidates import candidate_inputs
from bin_grasp.io import file_hash, save_json


def tiled_boxes(width, height):
    """Full image plus four overlapping crops (one explicit crop layer)."""
    overlap = int((512 / 1500) * min(width, height))
    crop_w, crop_h = math.ceil((width + overlap) / 2), math.ceil((height + overlap) / 2)
    return [(0, 0, width, height)] + [
        (x, y, min(x + crop_w, width), min(y + crop_h, height))
        for x in (0, crop_w - overlap) for y in (0, crop_h - overlap)]


def generate_masks(generator, rgb, settings):
    """Avoid the installed pipeline's multi-crop path, which filters only masks[0]."""
    if settings['crops_n_layers'] == 0:
        return generator(rgb, **settings)
    if settings['crops_n_layers'] != 1:
        raise ValueError('Explicit crop generation supports only one crop layer')
    from torchvision.ops import batched_nms
    masks, scores, boxes = [], [], []
    local_settings = dict(settings, crops_n_layers=0)
    for x1, y1, x2, y2 in tiled_boxes(rgb.width, rgb.height):
        output = generator(rgb.crop((x1, y1, x2, y2)), **local_settings)
        for raw, score in zip(output['masks'], output['scores']):
            mask = np.asarray(raw, dtype=bool)
            if mask.shape != (y2-y1, x2-x1):
                raise ValueError('Crop mask dimensions differ')
            if not mask.any():
                continue
            # Do not propose objects cut off at an artificial crop boundary.
            if ((x1 > 0 and mask[:, 0].any()) or (x2 < rgb.width and mask[:, -1].any())
                    or (y1 > 0 and mask[0].any()) or (y2 < rgb.height and mask[-1].any())):
                continue
            full = np.zeros((rgb.height, rgb.width), dtype=bool)
            full[y1:y2, x1:x2] = mask
            ys, xs = np.where(full)
            boxes.append([int(xs.min()), int(ys.min()), int(xs.max())+1, int(ys.max())+1])
            masks.append(full)
            scores.append(float(score))
    if not masks:
        return dict(masks=[], scores=[])
    scores = torch.tensor(scores, dtype=torch.float32)
    keep = batched_nms(torch.tensor(boxes, dtype=torch.float32), scores,
                       torch.zeros(len(scores), dtype=torch.long), settings['crops_nms_thresh']).tolist()
    return dict(masks=[masks[i] for i in keep], scores=scores[keep])


class StableMaskGenerationPipeline(MaskGenerationPipeline):
    """TorchVision NMS needs matching FP32 boxes/scores after BF16 inference."""
    def postprocess(self, model_outputs, **kwargs):
        for output in model_outputs:
            output['iou_scores'] = output['iou_scores'].float()
            output['boxes'] = output['boxes'].float()
        return super().postprocess(model_outputs, **kwargs)


def select_rows(index, count, seed):
    if index['split'] != 'val' or index['mode'] != 'sequence':
        raise ValueError('This diagnostic accepts sequence validation only, never test.')
    groups = defaultdict(lambda: defaultdict(list))
    for row in index['rows']:
        groups[row['group']][row['scene']].append(row)
    if not 1 <= count <= sum(len(v) for v in groups.values()):
        raise ValueError('count must fit the number of distinct validation images')
    rng = random.Random(seed)
    names = sorted(groups)
    rng.shuffle(names)
    queues = {}
    for group in names:
        queues[group] = sorted(groups[group])
        rng.shuffle(queues[group])
    result = []
    while len(result) < count:
        for group in names:
            if queues[group]:
                scene = queues[group].pop()
                result.append(rng.choice(groups[group][scene]))
                if len(result) == count:
                    break
    return result


def validate_checkpoint(checkpoint, index):
    if checkpoint['overfit']:
        raise ValueError('Cannot use the debug overfit checkpoint')
    for key in ('cache_id', 'fingerprint', 'mode'):
        if checkpoint[key] != index[key]:
            raise ValueError(f'Checkpoint and validation index differ: {key}')
    if set(checkpoint['training_groups']) & {r['group'] for r in index['rows']}:
        raise ValueError('Checkpoint has seen validation sequences')


def filter_proposals(masks, scores, shape, min_area=32, max_fraction=.95, duplicate_iou=.95, cap=128):
    """Only predicted masks and model quality scores enter this function."""
    if masks.shape != (len(scores), *shape):
        raise ValueError('Invalid SAM mask shape or score count')
    areas = masks.reshape(len(masks), -1).sum(axis=1) if len(masks) else np.array([])
    eligible = [i for i in range(len(masks)) if np.isfinite(scores[i]) and
                min_area <= areas[i] <= shape[0]*shape[1]*max_fraction]
    ranked = sorted(eligible, key=lambda i: (-scores[i], i))
    kept = []
    duplicates = 0
    for i in ranked:
        duplicate = False
        for j in kept:
            intersection = np.logical_and(masks[i], masks[j]).sum()
            union = areas[i] + areas[j] - intersection
            if intersection / max(union, 1) >= duplicate_iou:
                duplicate = True
                break
        if duplicate:
            duplicates += 1
        else:
            kept.append(i)
    truncated = max(0, len(kept)-cap)
    return kept[:cap], dict(sam_candidates=len(masks), rejected_area_or_nonfinite=len(masks)-len(eligible),
                           duplicate_masks_removed=duplicates, capped_masks=truncated, kept_candidates=min(len(kept), cap))


def prepare_candidates(rgb, masks):
    """Exactly the training crop recipe, now derived from predicted masks."""
    crops, geometry = candidate_inputs(rgb, masks)
    return crops, torch.from_numpy(geometry)


def mask_ious(masks, target):
    if not target.any():
        raise ValueError('Evaluation target has an empty mask')
    if not len(masks):
        return np.empty(0, dtype=np.float64)
    intersection = np.logical_and(masks, target).sum(axis=(1, 2))
    union = np.logical_or(masks, target).sum(axis=(1, 2))
    return intersection / np.maximum(union, 1)


def summarize_evaluation(raw_masks, kept, selected, zero_shot, target, threshold=.5):
    ious = mask_ious(raw_masks, target)
    retained = ious[kept]
    best = int(np.argmax(retained)) if len(kept) else None
    selected_iou = float(retained[selected]) if selected is not None else 0.
    best_iou = float(retained[best]) if best is not None else 0.
    available = best_iou >= threshold
    success = selected_iou >= threshold
    return dict(raw_best_iou=float(ious.max()) if len(ious) else 0.,
                best_available_iou=best_iou, best_available_index=best,
                selected_iou=selected_iou, zero_shot_iou=float(retained[zero_shot]) if zero_shot is not None else 0.,
                proposal_available_at_threshold=available, selected_success_at_threshold=success,
                failure='success' if success else 'selection_miss' if available else 'proposal_miss',
                ious=retained.tolist())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--val-features', type=Path, default=Path('features/4764b544cec72bc7/sequence-val-5f243509242f.json'))
    p.add_argument('--sam', type=Path, default=Path('models/sam2.1-hiera-tiny'))
    p.add_argument('--siglip', type=Path, default=Path('models/siglip2-base-patch16-224'))
    p.add_argument('--checkpoint', type=Path, default=Path('runs/regularization-study/baseline/best.pt'))
    p.add_argument('--ocid-root', type=Path, default=Path('data/OCID-dataset'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--count', type=int, default=20)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--points-per-batch', type=int, default=16)
    p.add_argument('--points-per-side', type=int, default=32)
    p.add_argument('--crop-layers', type=int, choices=[0, 1], default=0)
    p.add_argument('--pred-iou-thresh', type=float, default=.8)
    p.add_argument('--stability-score-thresh', type=float, default=.95)
    p.add_argument('--chunk-size', type=int, default=8)
    p.add_argument('--min-area', type=int, default=32)
    p.add_argument('--max-area-fraction', type=float, default=.95)
    p.add_argument('--max-candidates', type=int, default=128)
    p.add_argument('--duplicate-iou', type=float, default=.95)
    p.add_argument('--iou-threshold', type=float, default=.5)
    args = p.parse_args()
    if min(args.count, args.points_per_batch, args.points_per_side, args.chunk_size, args.min_area, args.max_candidates) < 1:
        p.error('Counts, area, and batch sizes must be positive')
    if not all(0 < x <= 1 for x in (args.max_area_fraction, args.duplicate_iou, args.iou_threshold)):
        p.error('Fractions must be in (0,1]')
    if args.crop_layers < 0 or not all(0 <= x <= 1 for x in (args.pred_iou_thresh, args.stability_score_thresh)):
        p.error('crop-layers must be nonnegative; SAM thresholds must be in [0,1]')
    if args.output.exists():
        raise FileExistsError('Choose a new run directory; existing runs are preserved')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('This GPU demo requires CUDA with BF16 support; use an authorized GPU session.')
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.mha.set_fastpath_enabled(False)
    index = json.loads(args.val_features.read_text())
    rows = select_rows(index, args.count, args.seed)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    validate_checkpoint(checkpoint, index)
    if index['recipe']['crop'] != 'masked_gray_square' or index['recipe']['text_length'] != 64:
        raise ValueError('Unsupported encoder preprocessing recipe')
    print('Checking frozen encoder and checkpoint fingerprints...', flush=True)
    if encoder_fingerprint(args.siglip) != index['recipe']['encoder']:
        raise ValueError('Local SigLIP weights differ from the training feature recipe')
    for row in rows:
        if not inside(args.ocid_root, row['scene']).is_file():
            raise FileNotFoundError(f'Connect the dataset drive: {row["scene"]}')
    provenance = dict(seed=args.seed, count=args.count, selection='One random expression per distinct image; balanced sequence groups; not selected by correctness',
                      ocid_root=str(args.ocid_root.resolve()),siglip_path=str(args.siglip.resolve()),
                      val_features=str(args.val_features.resolve()),
                      selected_by_group=dict(Counter(r['group'] for r in rows)),
                      sam_path=str(args.sam.resolve()), sam_weights_sha256=file_hash(args.sam/'model.safetensors'),
                      checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=file_hash(args.checkpoint),
                      cache_id=index['cache_id'], fingerprint=index['fingerprint'],
                      torch=torch.__version__, transformers=transformers.__version__,
                      gpu=torch.cuda.get_device_name(), split='val', mode='sequence',
                      sam_precision='float32 weights, bf16 autocast', siglip_precision='float16', selector_precision='bf16 autocast',
                      crop_implementation='explicit overlapping tiles, artificial-edge rejection, quality-ranked box NMS' if args.crop_layers else 'full image',
                      sam_settings=dict(points_per_batch=args.points_per_batch, points_per_crop=args.points_per_side,
                                        crops_n_layers=args.crop_layers, pred_iou_thresh=args.pred_iou_thresh,
                                        stability_score_thresh=args.stability_score_thresh, crops_nms_thresh=.7),
                      filtering=dict(min_area=args.min_area,max_area_fraction=args.max_area_fraction,duplicate_iou=args.duplicate_iou,max_candidates=args.max_candidates),
                      threshold=args.iou_threshold,
                      notes=['Ground-truth labels are opened only after all predictions are saved.',
                             'Proposal IDs are local predicted indexes, not dataset instance IDs.',
                             'Small balanced diagnostic sample; not the full validation accuracy.',
                             'IoU threshold is our diagnostic convention, not a PDF requirement.',
                             'No training, depth input, or test evaluation.'])
    args.output.mkdir(parents=True)
    (args.output/'proposals').mkdir()
    save_json(args.output/'settings.json', provenance)
    save_json(args.output/'selection.json', rows)
    start = time.perf_counter()
    print('Loading SAM from local files...', flush=True)
    generator = pipeline('mask-generation', model=str(args.sam.resolve()), device=0, dtype=torch.float32,
                         pipeline_class=StableMaskGenerationPipeline)
    generator.model.eval().requires_grad_(False)
    predictions = []
    for n, row in enumerate(rows, 1):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        timer = time.perf_counter()
        rgb = Image.open(inside(args.ocid_root, row['scene'])).convert('RGB')
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            output = generate_masks(generator, rgb, provenance['sam_settings'])
        masks = np.asarray(output['masks'], dtype=bool) if len(output['masks']) else np.empty((0, rgb.height, rgb.width), dtype=bool)
        scores = np.asarray([float(s) for s in output['scores']], dtype=np.float32)
        kept, filtering = filter_proposals(masks, scores, (rgb.height, rgb.width), args.min_area,
                                          args.max_area_fraction, args.duplicate_iou, args.max_candidates)
        torch.cuda.synchronize()
        elapsed = time.perf_counter()-timer
        packed = np.packbits(masks.reshape(len(masks), rgb.height*rgb.width), axis=1)
        np.savez_compressed(args.output/'proposals'/f'{n:03d}.npz', packed=packed,
                            shape=np.array(masks.shape), scores=scores, kept=np.array(kept, dtype=np.int64))
        record = dict(number=n,key=row['key'],scene=row['scene'],sentence=row['sentence'],kept_raw_indexes=kept,
                      proposal_seconds=elapsed,proposal_peak_cuda_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
                      rgb_sha256=file_hash(inside(args.ocid_root,row['scene'])), **filtering)
        predictions.append(record)
        save_json(args.output/'proposal_progress.json', predictions)
        print(f'SAM {n}/{len(rows)}: {len(masks)} proposals -> {len(kept)} kept | {elapsed:.2f}s | GPU peak {record["proposal_peak_cuda_allocated_gib"]:.2f} GiB', flush=True)
    generator.model.to('cpu')
    del generator, output
    gc.collect()
    torch.cuda.empty_cache()
    print('SAM finished. Loading frozen SigLIP and the trained selector...', flush=True)
    encoder = AutoModel.from_pretrained(args.siglip, local_files_only=True, dtype=torch.float16).cuda().eval()
    encoder.requires_grad_(False)
    processor = AutoProcessor.from_pretrained(args.siglip, local_files_only=True)
    selector = GroundingModel(**checkpoint['config']).cuda().eval()
    selector.load_state_dict(checkpoint['model'])
    selector.requires_grad_(False)
    for n, row in enumerate(rows, 1):
        record = predictions[n-1]
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        timer = time.perf_counter()
        masks, kept = load_proposals(args.output, n)
        if kept:
            rgb = np.asarray(Image.open(inside(args.ocid_root, row['scene'])).convert('RGB'))
            crops, geometry = prepare_candidates(rgb, masks[kept])
            images = crops + [Image.fromarray(rgb)]
            pieces = []
            with torch.inference_mode():
                for start_image in range(0, len(images), args.chunk_size):
                    inputs = processor(images=images[start_image:start_image+args.chunk_size], return_tensors='pt')
                    pieces.append(pooled(encoder.get_image_features(pixel_values=inputs['pixel_values'].to('cuda',torch.float16))))
                # Match cached training precision before normalization in the selector.
                features = torch.cat(pieces).half().float().cuda()
                tokens = processor.tokenizer(row['sentence'], truncation=False)['input_ids']
                text_inputs = processor(text=[row['sentence']], padding='max_length', max_length=64, truncation=True, return_tensors='pt').to('cuda')
                text = pooled(encoder.get_text_features(**text_inputs)).half().float().cuda()
                visual = features[:-1].unsqueeze(0)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    scores = selector(visual=visual,text=text,scene=features[-1:].float(),geometry=geometry.unsqueeze(0).cuda(),
                                      valid=torch.ones((1,len(kept)),dtype=torch.bool,device='cuda'))[0]
                similarity = (torch.nn.functional.normalize(visual[0],dim=-1)*torch.nn.functional.normalize(text[0],dim=-1)).sum(-1)
                record.update(selected_index=int(scores.argmax()),zero_shot_index=int(similarity.argmax()),
                              selector_scores=scores.float().cpu().tolist(),similarities=similarity.cpu().tolist(),
                              text_truncated=len(tokens)>64)
        else:
            record.update(selected_index=None,zero_shot_index=None,selector_scores=[],similarities=[],text_truncated=False)
        torch.cuda.synchronize()
        record.update(selection_seconds=time.perf_counter()-timer,
                      selection_peak_cuda_allocated_gib=torch.cuda.max_memory_allocated()/1024**3)
        print(f'Selector {n}/{len(rows)}: chose {record["selected_index"]} | {record["selection_seconds"]:.2f}s', flush=True)
    # Strict phase boundary: predictions are committed before reading any label PNG.
    save_json(args.output/'predictions.json', predictions)
    del encoder, selector
    gc.collect()
    torch.cuda.empty_cache()
    print('Predictions saved. Now opening ground-truth labels for evaluation only...',flush=True)
    cases = []
    for n, (row, record) in enumerate(zip(rows, predictions), 1):
        masks, kept = load_proposals(args.output, n)
        scene = Path(row['scene'])
        label_path = inside(args.ocid_root, scene.parent.parent/'label'/scene.name)
        label = np.asarray(Image.open(label_path))
        if label.shape != masks.shape[1:]:
            raise ValueError('Label and predicted mask dimensions differ')
        target = label == row['target_id']
        metrics = summarize_evaluation(masks, kept, record['selected_index'], record['zero_shot_index'], target, args.iou_threshold)
        record.update(evaluation=metrics,target_id=row['target_id'],group=row['group'])
        correct_edge, _ = outline(target)
        edges = [outline(masks[i])[0] for i in kept]
        cases.append(dict(**record,width=label.shape[1],height=label.shape[0],correct_edge=correct_edge,
                          proposal_edges=edges,image='data:image/png;base64,'+base64.b64encode(inside(args.ocid_root,row['scene']).read_bytes()).decode()))
    summary = dict(samples=len(rows),distinct_images=len({r['scene'] for r in rows}),
                   mean_selected_iou=float(np.mean([r['evaluation']['selected_iou'] for r in predictions])),
                   mean_best_available_iou=float(np.mean([r['evaluation']['best_available_iou'] for r in predictions])),
                   mean_raw_best_iou=float(np.mean([r['evaluation']['raw_best_iou'] for r in predictions])),
                   selected_success_count=sum(r['evaluation']['selected_success_at_threshold'] for r in predictions),
                   proposal_available_count=sum(r['evaluation']['proposal_available_at_threshold'] for r in predictions),
                   raw_proposal_available_count=sum(r['evaluation']['raw_best_iou']>=args.iou_threshold for r in predictions),
                   zero_shot_success_count=sum(r['evaluation']['zero_shot_iou']>=args.iou_threshold for r in predictions),
                   failures=dict(Counter(r['evaluation']['failure'] for r in predictions)),
                   mean_candidates=float(np.mean([r['kept_candidates'] for r in predictions])),
                   mean_proposal_seconds=float(np.mean([r['proposal_seconds'] for r in predictions])),
                   mean_selection_seconds=float(np.mean([r['selection_seconds'] for r in predictions])),
                   peak_cuda_allocated_gib=max(max(r['proposal_peak_cuda_allocated_gib'],r['selection_peak_cuda_allocated_gib']) for r in predictions),
                   threshold=args.iou_threshold,wall_seconds=time.perf_counter()-start,
                   test_evaluated=False,training_performed=False)
    available = summary['proposal_available_count']
    summary['conditional_selection_success'] = summary['selected_success_count']/available if available else None
    save_json(args.output/'evaluation.json', predictions)
    save_json(args.output/'summary.json', summary)
    html = (Path(__file__).resolve().parents[1] / 'templates' / 'experiment2_report_template.html').read_text()
    payload = json.dumps(dict(summary=summary,settings=provenance,cases=cases),allow_nan=False).replace('<','\\u003c')
    (args.output/'report.html').write_text(html.replace('__EXPERIMENT_DATA__',payload))
    print(json.dumps(summary,indent=2),flush=True)
    print('Report:',args.output/'report.html',flush=True)


def load_proposals(output, n):
    with np.load(output/'proposals'/f'{n:03d}.npz',allow_pickle=False) as data:
        shape = tuple(int(v) for v in data['shape'])
        masks = np.unpackbits(data['packed'],axis=1,count=shape[1]*shape[2]).reshape(shape).astype(bool)
        kept = data['kept'].tolist()
    return masks, kept


if __name__ == '__main__':
    main()
