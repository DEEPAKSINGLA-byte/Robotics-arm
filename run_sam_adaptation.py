"""Prepare SAM-candidate features, then adapt only the existing selector."""
import argparse
from collections import defaultdict
import gc
import json
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image
import torch
from transformers import AutoModel, AutoProcessor, pipeline

from build_error_review import inside
from candidate_inputs import candidate_inputs
from extract_siglip_features import encoder_fingerprint, pooled
from grounding_data import collate
from grounding_model import GroundingModel
from run_experiment2 import (StableMaskGenerationPipeline, file_hash, filter_proposals,
                             load_proposals, mask_ious, save_json, validate_checkpoint)


def check_indexes(train, val):
    if train['split'] != 'train' or val['split'] != 'val':
        raise ValueError('Only training and validation indexes are allowed; never test')
    for key in ('cache_id', 'fingerprint', 'mode', 'recipe'):
        if train[key] != val[key]:
            raise ValueError(f'Feature index mismatch: {key}')
    if train['mode'] != 'sequence':
        raise ValueError('Sequence-separated indexes required')
    for key in ('key', 'scene', 'group'):
        if {r[key] for r in train['rows']} & {r[key] for r in val['rows']}:
            raise ValueError(f'Train/validation overlap: {key}')


def training_rows(index, count, per_scene, seed):
    if index['split'] != 'train' or count < 1 or per_scene < 1:
        raise ValueError('Positive limits and training split required')
    grouped = defaultdict(list)
    for row in index['rows']:
        grouped[row['scene']].append(row)
    if count > len(grouped):
        raise ValueError(f'Only {len(grouped)} training images available')
    rng = random.Random(seed)
    scenes = rng.sample(sorted(grouped), count)
    return [r for s in scenes for r in rng.sample(grouped[s], min(per_scene, len(grouped[s])))]


def positive_loss(scores, positive):
    """Maximize probability of any sufficiently overlapping candidate, not one arbitrary ID."""
    if not positive.any(dim=1).all():
        raise ValueError('Exclude missing-target examples from training loss')
    scores = scores.float()
    return (torch.logsumexp(scores, dim=1) -
            torch.logsumexp(scores.masked_fill(~positive, float('-inf')), dim=1)).mean()


def prepare(args):
    train = json.loads(args.train_features.read_text())
    val = json.loads(args.val_features.read_text())
    check_indexes(train, val)
    initial = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    validate_checkpoint(initial, val)
    settings = json.loads((args.validation_run / 'settings.json').read_text())
    val_rows = json.loads((args.validation_run / 'selection.json').read_text())
    lookup = {r['key']: r for r in val['rows']}
    if not val_rows or any(lookup.get(r['key']) != r for r in val_rows):
        raise ValueError('Saved validation selection is not from the validation index')
    if len({r['scene'] for r in val_rows}) != len(val_rows):
        raise ValueError('Validation selection must use distinct images')
    for key in ('cache_id', 'fingerprint', 'mode', 'split'):
        if settings[key] != val[key]:
            raise ValueError(f'Validation run mismatch: {key}')
    checkpoint_hash = file_hash(args.checkpoint)
    sam_hash = file_hash(args.sam / 'model.safetensors')
    if settings['checkpoint_sha256'] != checkpoint_hash or settings['sam_weights_sha256'] != sam_hash:
        raise ValueError('Use the same baseline checkpoint and SAM as the validation run')
    if settings['threshold'] != .5:
        raise ValueError('This experiment uses the fixed IoU >= 0.5 diagnostic')
    if encoder_fingerprint(args.siglip) != train['recipe']['encoder']:
        raise ValueError('SigLIP differs from the cached text encoder')
    import transformers
    if (train['recipe']['crop'] != 'masked_gray_square' or train['recipe']['text_length'] != 64
            or train['recipe']['dtype'] != 'float16'
            or train['recipe']['transformers'] != transformers.__version__):
        raise ValueError('Unsupported or changed feature recipe')
    rows = {'train': training_rows(train, args.train_scenes, args.expressions_per_scene, args.seed), 'val': val_rows}
    for row in rows['train'] + val_rows:
        if not inside(args.ocid_root, row['scene']).is_file():
            raise FileNotFoundError('Connect the OCID dataset drive')
    predictions = json.loads((args.validation_run / 'predictions.json').read_text())
    if [r['key'] for r in predictions] != [r['key'] for r in val_rows]:
        raise ValueError('Validation predictions and selection differ')
    for row, pred in zip(val_rows, predictions):
        if file_hash(inside(args.ocid_root, row['scene'])) != pred['rgb_sha256']:
            raise ValueError('Validation RGB image changed')
    args.output.mkdir(parents=True)
    (args.output / 'proposals').mkdir()
    (args.output / 'scenes').mkdir()
    provenance = dict(format='sam-adaptation-v1', seed=args.seed, threshold=.5,
                      train_scenes=args.train_scenes, expressions_per_scene=args.expressions_per_scene,
                      validation_run=str(args.validation_run.resolve()),
                      checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=checkpoint_hash,
                      sam_weights_sha256=sam_hash, sam_settings=settings['sam_settings'], filtering=settings['filtering'],
                      recipe=train['recipe'], cache_id=train['cache_id'], fingerprint=train['fingerprint'],
                      mode='sequence', training_groups=sorted({r['group'] for r in rows['train']}),
                      validation_groups=sorted({r['group'] for r in val_rows}),
                      initial_training_groups=initial['training_groups'], test_evaluated=False)
    save_json(args.output / 'settings.json', provenance)
    save_json(args.output / 'selection.json', rows)
    scenes = sorted({r['scene'] for r in rows['train']})
    sources = {}
    generator = pipeline('mask-generation', model=str(args.sam.resolve()), device=0, dtype=torch.float32,
                         pipeline_class=StableMaskGenerationPipeline)
    generator.model.eval().requires_grad_(False)
    for n, scene in enumerate(scenes, 1):
        start = time.perf_counter()
        with Image.open(inside(args.ocid_root, scene)) as im:
            rgb = im.convert('RGB')
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            result = generator(rgb, **settings['sam_settings'])
        masks = np.asarray(result['masks'], dtype=bool) if len(result['masks']) else np.empty((0, rgb.height, rgb.width), bool)
        scores = np.asarray([float(s) for s in result['scores']], dtype=np.float32)
        f = settings['filtering']
        kept, _ = filter_proposals(masks, scores, (rgb.height, rgb.width), f['min_area'],
                                  f['max_area_fraction'], f['duplicate_iou'], f['max_candidates'])
        path = args.output / 'proposals' / f'{n:03d}.npz'
        np.savez_compressed(path, packed=np.packbits(masks.reshape(len(masks), rgb.height*rgb.width), axis=1),
                            shape=np.array(masks.shape), scores=scores, kept=np.array(kept, dtype=np.int64))
        sources[scene] = (args.output, n)
        print(f'SAM training image {n}/{len(scenes)}: {len(kept)} masks | {time.perf_counter()-start:.1f}s', flush=True)
    del generator, result
    gc.collect()
    torch.cuda.empty_cache()
    for n, row in enumerate(val_rows, 1):
        sources[row['scene']] = (args.validation_run, n)
    encoder = AutoModel.from_pretrained(args.siglip, local_files_only=True, dtype=torch.float16).cuda().eval()
    encoder.requires_grad_(False)
    processor = AutoProcessor.from_pretrained(args.siglip, local_files_only=True)
    catalog = {}
    for n, (scene, (folder, number)) in enumerate(sources.items(), 1):
        masks, kept = load_proposals(folder, number)
        masks = masks[kept]
        with Image.open(inside(args.ocid_root, scene)) as im:
            rgb = np.asarray(im.convert('RGB'))
        if len(masks):
            crops, geometry = candidate_inputs(rgb, masks)
        else:
            crops, geometry = [], np.empty((0, 7), np.float32)
        images, pieces = crops + [Image.fromarray(rgb)], []
        with torch.inference_mode():
            for j in range(0, len(images), 8):
                inputs = processor(images=images[j:j+8], return_tensors='pt')
                pieces.append(pooled(encoder.get_image_features(pixel_values=inputs['pixel_values'].to('cuda', torch.float16))))
        features = torch.cat(pieces).half()
        # Labels enter only supervision below; never crops, proposal filtering, or encoder inputs.
        relative = Path(scene)
        with Image.open(inside(args.ocid_root, relative.parent.parent / 'label' / relative.name)) as im:
            labels = np.asarray(im)
        if labels.shape != rgb.shape[:2]:
            raise ValueError('Label/image dimensions differ')
        target_ids = {r['target_id'] for split in rows.values() for r in split if r['scene'] == scene}
        ious = {}
        for target_id in target_ids:
            target = labels == target_id
            if not target.any():
                raise ValueError('Annotation target is absent from label image')
            ious[target_id] = torch.from_numpy(mask_ious(masks, target).astype(np.float32))
        filename = f'scenes/{n:04d}.pt'
        torch.save(dict(visual=features[:-1], scene=features[-1], geometry=torch.from_numpy(geometry), ious=ious), args.output / filename)
        catalog[scene] = dict(file=filename, proposal_sha256=file_hash(folder/'proposals'/f'{number:03d}.npz'),
                              rgb_sha256=file_hash(inside(args.ocid_root, scene)))
        print(f'Features {n}/{len(sources)}: {len(masks)} masks', flush=True)
    del encoder
    gc.collect()
    torch.cuda.empty_cache()
    # Reuse unchanged frozen text features by exact expression key, not row position.
    for split, index, index_path in [('train', train, args.train_features), ('val', val, args.val_features)]:
        positions = {r['key']: i for i, r in enumerate(index['rows'])}
        texts = np.load(index_path.parent / index['text'], mmap_mode='r')
        np.save(args.output / f'{split}-text.npy', np.asarray(texts[[positions[r['key']] for r in rows[split]]], dtype=np.float16))
    save_json(args.output / 'catalog.json', catalog)
    counts = {}
    for split in rows:
        available = sum(bool((torch.load(args.output/catalog[r['scene']]['file'], weights_only=True)['ious'][r['target_id']] >= .5).any()) for r in rows[split])
        counts[split] = dict(expressions=len(rows[split]), available=available, missing=len(rows[split])-available)
    save_json(args.output / 'ready.json', counts)
    print(json.dumps(counts, indent=2), flush=True)


def load_data(folder):
    if not (folder / 'ready.json').is_file():
        raise ValueError('Feature preparation is incomplete; use a new output folder to retry')
    meta = json.loads((folder / 'settings.json').read_text())
    rows = json.loads((folder / 'selection.json').read_text())
    for key in ('key', 'scene', 'group'):
        if {r[key] for r in rows['train']} & {r[key] for r in rows['val']}:
            raise ValueError(f'Data leakage: {key}')
    if set(meta['initial_training_groups']) & {r['group'] for r in rows['val']}:
        raise ValueError('Initial model has seen validation sequences')
    catalog = json.loads((folder / 'catalog.json').read_text())
    records = {s: torch.load(folder / item['file'], weights_only=True) for s, item in catalog.items()}
    samples = {}
    for split in rows:
        texts = np.load(folder / f'{split}-text.npy')
        if texts.shape != (len(rows[split]), 768):
            raise ValueError('Text features do not match selected rows')
        samples[split] = []
        for i, row in enumerate(rows[split]):
            record = records[row['scene']]
            inputs = {k: record[k].float() for k in ('visual', 'scene', 'geometry')}
            inputs['text'] = torch.from_numpy(texts[i].astype(np.float32))
            samples[split].append((inputs, record['ious'][row['target_id']], row['key']))
    return meta, samples


def batches(samples, batch_size, shuffle=False):
    order = torch.randperm(len(samples)).tolist() if shuffle else list(range(len(samples)))
    for start in range(0, len(order), batch_size):
        chosen = [samples[i] for i in order[start:start+batch_size]]
        inputs, _, _ = collate([(s[0], 0, {}) for s in chosen])
        ious = torch.zeros_like(inputs['valid'], dtype=torch.float32)
        for i, sample in enumerate(chosen):
            ious[i, :len(sample[1])] = sample[1]
        yield {k: v.cuda() for k, v in inputs.items()}, ious.cuda(), [s[2] for s in chosen]


@torch.inference_mode()
def evaluate(model, samples, batch_size):
    model.eval()
    nonempty = [s for s in samples if len(s[1])]
    predictions = {s[2]: dict(key=s[2], selected_index=None, iou=0., success=False) for s in samples}
    for inputs, ious, keys in batches(nonempty, batch_size):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            scores = model(**inputs)
        chosen = scores.argmax(1)
        values = ious.gather(1, chosen[:, None]).squeeze(1).cpu().tolist()
        for key, idx, value in zip(keys, chosen.cpu().tolist(), values):
            predictions[key] = dict(key=key, selected_index=idx, iou=value, success=value >= .5)
    predictions = list(predictions.values())
    success = sum(p['success'] for p in predictions)
    available = sum(bool((s[1] >= .5).any()) for s in samples)
    return dict(samples=len(samples), success=success, accuracy=success/len(samples),
                available=available, proposal_miss=len(samples)-available, selection_miss=available-success,
                mean_iou=float(np.mean([p['iou'] for p in predictions]))), predictions


def train(args):
    meta, data = load_data(args.data)
    initial_path = Path(meta['checkpoint'])
    if file_hash(initial_path) != meta['checkpoint_sha256']:
        raise ValueError('Original checkpoint changed')
    initial = torch.load(initial_path, map_location='cpu', weights_only=True)
    model = GroundingModel(**initial['config']).cuda()
    model.load_state_dict(initial['model'])
    usable = [s for s in data['train'] if bool((s[1] >= .5).any())]
    if not usable or not data['val']:
        raise ValueError('Need usable training examples and validation examples')
    args.output.mkdir(parents=True)
    save_json(args.output / 'settings.json', dict(data=str(args.data.resolve()), seed=args.seed, lr=args.lr,
              batch_size=args.batch_size, epochs=args.epochs, patience=args.patience,
              train_usable=len(usable), train_missing=len(data['train'])-len(usable),
              loss='negative log probability of any candidate with IoU >= 0.5',
              frozen=['SAM', 'SigLIP'], test_evaluated=False, preparation=meta))
    baseline, predictions = evaluate(model, data['val'], args.batch_size)
    save_json(args.output / 'baseline.json', baseline)
    save_json(args.output / 'baseline_predictions.json', predictions)
    print(f'Original selector: {baseline["success"]}/{baseline["samples"]}', flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01, fused=True)
    best, stale, history = baseline['accuracy'], 0, []

    def save_best(epoch, metrics, predictions):
        checkpoint = dict(initial, model=model.state_dict(), epoch=epoch, accuracy=metrics['accuracy'],
                          training_groups=sorted(set(initial['training_groups']) | set(meta['training_groups'])),
                          adaptation='sam-adaptation-v1', adaptation_data=str(args.data.resolve()))
        torch.save(checkpoint, args.output / 'best.pt')
        save_json(args.output / 'best_metrics.json', metrics)
        save_json(args.output / 'best_predictions.json', predictions)

    save_best(0, baseline, predictions)
    for epoch in range(1, args.epochs+1):
        torch.cuda.synchronize()
        start = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        model.train()
        loss_sum, seen = 0., 0
        for inputs, ious, _ in batches(usable, args.batch_size, shuffle=True):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                scores = model(**inputs)
            loss = positive_loss(scores, (ious >= .5) & inputs['valid'])
            if not torch.isfinite(loss):
                raise ValueError('Non-finite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            loss_sum += loss.item()*len(ious)
            seen += len(ious)
        metrics, predictions = evaluate(model, data['val'], args.batch_size)
        torch.cuda.synchronize()
        record = dict(epoch=epoch, loss=loss_sum/seen, validation=metrics,
                      epoch_seconds=time.perf_counter()-start,
                      peak_gpu_gib=torch.cuda.max_memory_allocated()/1024**3)
        history.append(record)
        save_json(args.output / 'history.json', history)
        print(f'Epoch {epoch}: {record["epoch_seconds"]:.1f}s | loss {record["loss"]:.4f} | '
              f'validation {metrics["success"]}/{metrics["samples"]} | peak GPU {record["peak_gpu_gib"]:.2f} GiB', flush=True)
        if metrics['accuracy'] > best:
            best, stale = metrics['accuracy'], 0
            save_best(epoch, metrics, predictions)
        else:
            stale += 1
        if stale >= args.patience:
            break
    save_json(args.output / 'summary.json', dict(baseline=baseline, best=json.loads((args.output/'best_metrics.json').read_text()),
                                               epochs_completed=len(history), test_evaluated=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['prepare', 'train'])
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--data', type=Path, help='Prepared feature folder, required for train')
    p.add_argument('--train-scenes', type=int, default=300)
    p.add_argument('--expressions-per-scene', type=int, default=20)
    p.add_argument('--train-features', type=Path, default=Path('features/4764b544cec72bc7/sequence-train-bd9c96bbe427.json'))
    p.add_argument('--val-features', type=Path, default=Path('features/4764b544cec72bc7/sequence-val-5f243509242f.json'))
    p.add_argument('--validation-run', type=Path, default=Path('runs/experiment2-sam21-tiny-100'))
    p.add_argument('--checkpoint', type=Path, default=Path('runs/regularization-study/baseline/best.pt'))
    p.add_argument('--sam', type=Path, default=Path('models/sam2.1-hiera-tiny'))
    p.add_argument('--siglip', type=Path, default=Path('models/siglip2-base-patch16-224'))
    p.add_argument('--ocid-root', type=Path, default=Path('data/OCID-dataset'))
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--patience', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=3e-5)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    if min(args.train_scenes, args.expressions_per_scene, args.epochs, args.patience, args.batch_size) < 1 or not np.isfinite(args.lr) or args.lr <= 0:
        p.error('Counts and learning rate must be positive and finite')
    if args.mode == 'train' and args.data is None:
        p.error('train requires --data')
    if args.output.exists():
        raise FileExistsError('Choose a fresh output directory; existing work is preserved')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('A CUDA GPU supporting BF16 is required')
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.mha.set_fastpath_enabled(False)
    if args.mode == 'prepare':
        prepare(args)
    else:
        train(args)


if __name__ == '__main__':
    main()
