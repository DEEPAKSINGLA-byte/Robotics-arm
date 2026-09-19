"""One fixed full-sequence test evaluation. No fitting, tuning, or checkpoint selection."""
import argparse
from collections import defaultdict
import gc
import json
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image
import torch
from transformers import AutoModel, AutoProcessor, pipeline

from experiments.build_error_review import inside
from bin_grasp.candidates import candidate_inputs
from experiments.extract_siglip_features import encoder_fingerprint, pooled
from bin_grasp.model import GroundingModel
from experiments.prepare_splits import digest
from experiments.run_experiment2 import (StableMaskGenerationPipeline, file_hash, filter_proposals,
                             generate_masks, load_proposals, mask_ious, save_json)


def verify_test_split(documents, summary, checkpoint):
    """Validate full manifests, provenance, and sequence isolation before inference."""
    if set(documents) != {'train', 'val', 'test'} or summary['mode'] != 'sequence':
        raise ValueError('Full sequence train/val/test manifests required')
    for name, doc in documents.items():
        if doc['split'] != name or doc['mode'] != 'sequence' or doc['fingerprint'] != summary['fingerprint']:
            raise ValueError('Split identity or fingerprint mismatch')
        if not doc['rows'] or len({r['key'] for r in doc['rows']}) != len(doc['rows']):
            raise ValueError('Empty split or duplicate expression keys')
    actual = digest(dict(mode='sequence', seed=summary['seed'],
                         splits={k: v['rows'] for k, v in documents.items()}))
    if actual != summary['fingerprint'] or checkpoint['fingerprint'] != actual:
        raise ValueError('Manifests differ from the frozen split fingerprint')
    for key in ('key', 'scene', 'group'):
        groups = [{r[key] for r in documents[s]['rows']} for s in ('train', 'val', 'test')]
        if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
            raise ValueError(f'Split leakage: {key}')
    if checkpoint['overfit'] or checkpoint['mode'] != 'sequence':
        raise ValueError('Debugging or non-sequence checkpoint cannot be used')
    training = {r['group'] for r in documents['train']['rows']}
    if not set(checkpoint['training_groups']).issubset(training):
        raise ValueError('Checkpoint has seen non-training sequences')


def aggregate(rows, results):
    if len(rows) != len(results) or not results:
        raise ValueError('Every expression must be evaluated')
    per_scene, per_group = defaultdict(list), defaultdict(list)
    for row, result in zip(rows, results):
        per_scene[row['scene']].append(result)
        per_group[row['group']].append(result)

    def metrics(items):
        n = len(items)
        correct = sum(r['success'] for r in items)
        available = sum(r['available'] for r in items)
        return dict(expressions=n, success=correct, accuracy=correct/n,
                    available=available, proposal_miss=n-available, selection_miss=available-correct,
                    mean_selected_iou=float(np.mean([r['selected_iou'] for r in items])),
                    conditional_selection_success=correct/available if available else None)

    summary = metrics(results)
    summary.update(images=len(per_scene), scene_macro_accuracy=float(np.mean([metrics(v)['accuracy'] for v in per_scene.values()])),
                   per_group={g: metrics(v) for g, v in sorted(per_group.items())},
                   per_scene={s: metrics(v) for s, v in sorted(per_scene.items())},
                   threshold=.5, split='test', test_evaluated=True, training_performed=False,
                   sample_policy='All expressions on all test images; no correctness-based sampling')
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--model-run', type=Path, default=Path('runs/sam-adaptation-relaxed-300'))
    p.add_argument('--splits', type=Path, default=Path('splits/sequence'))
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--check-only', action='store_true')
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a fresh output directory; do not overwrite a test result')
    settings = json.loads((args.model_run/'settings.json').read_text())
    prep = settings['preparation']
    checkpoint_path = args.model_run/'best.pt'
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    documents = {s: json.loads((args.splits/f'{s}.json').read_text()) for s in ('train', 'val', 'test')}
    split_summary = json.loads((args.splits/'summary.json').read_text())
    verify_test_split(documents, split_summary, checkpoint)
    source = json.loads((Path(prep['validation_run'])/'settings.json').read_text())
    recipe = prep['recipe']
    import transformers
    if (recipe['crop'] != 'masked_gray_square' or recipe['text_length'] != 64
            or recipe['dtype'] != 'float16' or recipe['transformers'] != transformers.__version__):
        raise ValueError('Encoder recipe changed')
    if checkpoint['cache_id'] != prep['cache_id'] or checkpoint['fingerprint'] != prep['fingerprint']:
        raise ValueError('Checkpoint and frozen preparation differ')
    # The final config is taken from the chosen training run, not searched on test.
    sam_settings, filtering = prep['sam_settings'], prep['filtering']
    if sam_settings != dict(points_per_batch=16, points_per_crop=32, crops_n_layers=0,
                            pred_iou_thresh=.7, stability_score_thresh=.9, crops_nms_thresh=.7):
        raise ValueError('Expected the finalized relaxed SAM configuration')
    if prep['threshold'] != .5 or filtering != dict(min_area=32, max_area_fraction=.95, duplicate_iou=.95, max_candidates=128):
        raise ValueError('Expected the finalized filters and IoU threshold')
    sam_path, siglip_path, root = Path(source['sam_path']), Path(source['siglip_path']), Path(source['ocid_root'])
    sam_hash = file_hash(sam_path/'model.safetensors')
    if sam_hash != prep['sam_weights_sha256'] or encoder_fingerprint(siglip_path) != recipe['encoder']:
        raise ValueError('Frozen model weights changed')
    rows = documents['test']['rows']
    scenes = sorted({r['scene'] for r in rows})
    for scene in scenes:
        if not inside(root, scene).is_file():
            raise FileNotFoundError('Connect the dataset drive')
    print(f'Validated untouched sequence split: {len(rows)} expressions / {len(scenes)} images.', flush=True)
    if args.check_only:
        return
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('CUDA with BF16 required')
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.mha.set_fastpath_enabled(False)
    args.output.mkdir(parents=True)
    (args.output/'proposals').mkdir()
    (args.output/'features').mkdir()
    shutil.copyfile(checkpoint_path, args.output/'frozen-selector.pt')
    plan = dict(protocol='full-test-v1', seed=args.seed, expressions=len(rows), images=len(scenes),
                checkpoint=str(checkpoint_path.resolve()), checkpoint_sha256=file_hash(checkpoint_path),
                sam_path=str(sam_path), siglip_path=str(siglip_path), ocid_root=str(root),
                sam_weights_sha256=sam_hash, recipe=recipe, sam_settings=sam_settings, filtering=filtering,
                split_fingerprint=split_summary['fingerprint'], cache_id=checkpoint['cache_id'],
                split='test', threshold=.5, training_performed=False, model_selection_performed=False,
                precision='SAM float32 weights with bf16 autocast; SigLIP fp16; selector bf16 autocast',
                selector_batch_size=32, image_chunk_size=8, text_batch_size=64,
                note='All test expressions. Labels used only after every prediction is committed. Do not tune from this result.')
    save_json(args.output/'evaluation_plan.json', plan)
    save_json(args.output/'selection.json', rows)
    if file_hash(args.output/'frozen-selector.pt') != plan['checkpoint_sha256']:
        raise ValueError('Checkpoint changed while being frozen')
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    generator = pipeline('mask-generation', model=str(sam_path), device=0, dtype=torch.float32,
                         pipeline_class=StableMaskGenerationPipeline)
    generator.model.eval().requires_grad_(False)
    records = {}
    for number, scene in enumerate(scenes, 1):
        timer = time.perf_counter()
        with Image.open(inside(root, scene)) as im:
            rgb = im.convert('RGB')
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            output = generate_masks(generator, rgb, sam_settings)
        masks = np.asarray(output['masks'], dtype=bool) if len(output['masks']) else np.empty((0, rgb.height, rgb.width), bool)
        scores = np.asarray([float(s) for s in output['scores']], dtype=np.float32)
        kept, stats = filter_proposals(masks, scores, (rgb.height, rgb.width), filtering['min_area'],
                                      filtering['max_area_fraction'], filtering['duplicate_iou'], filtering['max_candidates'])
        np.savez_compressed(args.output/'proposals'/f'{number:03d}.npz',
            packed=np.packbits(masks.reshape(len(masks), rgb.height*rgb.width), axis=1),
            shape=np.array(masks.shape), scores=scores, kept=np.array(kept, dtype=np.int64))
        records[scene] = dict(number=number, rgb_sha256=file_hash(inside(root, scene)),
                              proposal_seconds=time.perf_counter()-timer, **stats)
        save_json(args.output/'proposal_progress.json', records)
        print(f'SAM test image {number}/{len(scenes)}: {len(kept)} masks | {records[scene]["proposal_seconds"]:.2f}s', flush=True)
    del generator, output
    gc.collect()
    torch.cuda.empty_cache()
    encoder = AutoModel.from_pretrained(siglip_path, local_files_only=True, dtype=torch.float16).cuda().eval()
    encoder.requires_grad_(False)
    processor = AutoProcessor.from_pretrained(siglip_path, local_files_only=True)
    for i, scene in enumerate(scenes, 1):
        masks, kept = load_proposals(args.output, records[scene]['number'])
        if file_hash(inside(root, scene)) != records[scene]['rgb_sha256']:
            raise ValueError('Image changed between inference phases')
        with Image.open(inside(root, scene)) as im:
            rgb = np.asarray(im.convert('RGB'))
        crops, geometry = candidate_inputs(rgb, masks[kept]) if kept else ([], np.empty((0, 7), np.float32))
        images, pieces = crops+[Image.fromarray(rgb)], []
        with torch.inference_mode():
            for start in range(0, len(images), 8):
                inputs = processor(images=images[start:start+8], return_tensors='pt')
                pieces.append(pooled(encoder.get_image_features(pixel_values=inputs['pixel_values'].to('cuda', torch.float16))))
        features = torch.cat(pieces).half()
        torch.save(dict(visual=features[:-1], scene=features[-1], geometry=torch.from_numpy(geometry)),
                   args.output/'features'/f'{i:03d}.pt')
        print(f'Image features {i}/{len(scenes)}', flush=True)
    texts = np.empty((len(rows), 768), dtype=np.float16)
    truncated = 0
    with torch.inference_mode():
        for start in range(0, len(rows), 64):
            sentences = [r['sentence'] for r in rows[start:start+64]]
            truncated += sum(len(t) > 64 for t in processor.tokenizer(sentences, truncation=False)['input_ids'])
            inputs = processor(text=sentences, padding='max_length', max_length=64, truncation=True, return_tensors='pt').to('cuda')
            texts[start:start+len(sentences)] = pooled(encoder.get_text_features(**inputs)).half().numpy()
            if start % 1024 == 0:
                print(f'Text features {min(start+64, len(rows))}/{len(rows)}', flush=True)
    np.save(args.output/'features'/'text.npy', texts)
    del encoder
    gc.collect()
    torch.cuda.empty_cache()
    selector = GroundingModel(**checkpoint['config']).cuda().eval()
    selector.load_state_dict(checkpoint['model'])
    selector.requires_grad_(False)
    by_scene = defaultdict(list)
    for i, row in enumerate(rows):
        by_scene[row['scene']].append(i)
    predictions = [None]*len(rows)
    with torch.inference_mode():
        for number, scene in enumerate(scenes, 1):
            data = torch.load(args.output/'features'/f'{number:03d}.pt', weights_only=True)
            n = len(data['visual'])
            indexes = by_scene[scene]
            for start in range(0, len(indexes), 32):
                batch = indexes[start:start+32]
                if n:
                    inputs = dict(visual=data['visual'].float().cuda().unsqueeze(0).expand(len(batch), -1, -1),
                                  scene=data['scene'].float().cuda().unsqueeze(0).expand(len(batch), -1),
                                  geometry=data['geometry'].float().cuda().unsqueeze(0).expand(len(batch), -1, -1),
                                  text=torch.from_numpy(texts[batch].astype(np.float32)).cuda(),
                                  valid=torch.ones((len(batch), n), dtype=torch.bool, device='cuda'))
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        scores = selector(**inputs)
                    chosen = scores.argmax(1).cpu().tolist()
                else:
                    chosen = [None]*len(batch)
                for i, pick in zip(batch, chosen):
                    predictions[i] = dict(key=rows[i]['key'], scene=scene, sentence=rows[i]['sentence'], selected_index=pick)
            if number % 10 == 0 or number == len(scenes):
                print(f'Selector test image {number}/{len(scenes)}', flush=True)
    save_json(args.output/'predictions.json', predictions)
    peak = torch.cuda.max_memory_allocated()/1024**3
    del selector
    gc.collect()
    torch.cuda.empty_cache()
    print('All predictions saved. Opening test labels for scoring only.', flush=True)
    results = [None]*len(rows)
    for number, scene in enumerate(scenes, 1):
        masks, kept = load_proposals(args.output, number)
        relative = Path(scene)
        with Image.open(inside(root, relative.parent.parent/'label'/relative.name)) as im:
            labels = np.asarray(im)
        if labels.shape != masks.shape[1:]:
            raise ValueError('Mask and label dimensions differ')
        overlaps = {}
        for i in by_scene[scene]:
            target_id = rows[i]['target_id']
            if target_id not in overlaps:
                overlaps[target_id] = mask_ious(masks[kept], labels == target_id)
            ious = overlaps[target_id]
            pick = predictions[i]['selected_index']
            chosen_iou = float(ious[pick]) if pick is not None else 0.
            best_iou = float(ious.max()) if len(ious) else 0.
            results[i] = dict(key=rows[i]['key'], selected_iou=chosen_iou, best_available_iou=best_iou,
                              success=chosen_iou >= .5, available=best_iou >= .5)
        if number % 20 == 0:
            print(f'Scoring test image {number}/{len(scenes)}', flush=True)
    summary = aggregate(rows, results)
    summary.update(truncated_expressions=truncated, peak_gpu_gib=peak,
                   wall_seconds=time.perf_counter()-started,
                   mean_candidates=float(np.mean([r['kept_candidates'] for r in records.values()])),
                   mean_proposal_seconds=float(np.mean([r['proposal_seconds'] for r in records.values()])))
    save_json(args.output/'evaluation.json', results)
    save_json(args.output/'summary.json', summary)
    if file_hash(checkpoint_path) != plan['checkpoint_sha256'] or file_hash(sam_path/'model.safetensors') != sam_hash:
        raise ValueError('Source model changed during evaluation')
    save_json(args.output/'completed.json', dict(test_evaluated=True, training_performed=False,
              expressions=len(rows), images=len(scenes), model_hashes_unchanged=True))
    print(json.dumps({k: v for k, v in summary.items() if k not in ('per_scene', 'per_group')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
