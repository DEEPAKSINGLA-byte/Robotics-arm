"""Extract frozen local SigLIP features, saving each scene only once."""
import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, AutoProcessor

from phase1_dataset import resolve_paths
from candidate_inputs import candidate_inputs
from prepare_splits import digest


def pooled(output):
    # Transformers 4 returns a tensor; installed Transformers 5 returns a record.
    features = output if isinstance(output, torch.Tensor) else output.pooler_output
    if not torch.isfinite(features).all():
        raise ValueError('Non-finite encoder features')
    return F.normalize(features.float(), dim=-1).cpu()


def encoder_fingerprint(path):
    h = hashlib.sha256()
    for file in sorted(Path(path).iterdir()):
        if file.suffix in ('.json', '.model', '.safetensors'):
            h.update(file.name.encode())
            with file.open('rb') as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    h.update(block)
    return h.hexdigest()


def scene_inputs(root, scene):
    paths = resolve_paths(root, scene)
    with Image.open(paths.rgb) as im:
        rgb = np.asarray(im.convert('RGB')).copy()
    with Image.open(paths.label) as im:
        label = np.asarray(im).copy()
    if rgb.shape[:2] != label.shape:
        raise ValueError(f'RGB/label mismatch: {scene}')
    ids = [int(i) for i in np.unique(label) if i != 0]
    masks = np.stack([label == i for i in ids])
    crops, geometry = candidate_inputs(rgb, masks)
    return ids, geometry, crops, Image.fromarray(rgb)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--ocid-root', type=Path, required=True)
    p.add_argument('--model', type=Path, default=Path('models/siglip2-base-patch16-224'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--limit', type=int, default=0, help='Random subset; 0 means complete split')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--chunk-size', type=int, default=8)
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    args = p.parse_args()
    if args.limit < 0 or args.chunk_size < 1:
        p.error('limit must be nonnegative and chunk-size positive')
    torch.set_num_threads(4)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA inaccessible in this session. Run from your terminal or use --device cpu.')
    manifest = json.loads(args.manifest.read_text())
    rows = manifest['rows']
    if args.limit:
        rows = random.Random(args.seed).sample(rows, min(args.limit, len(rows)))
    if not rows:
        raise ValueError('Empty manifest')
    import transformers
    recipe = dict(version=1, encoder=encoder_fingerprint(args.model),
                  transformers=transformers.__version__, dtype='float16' if args.device == 'cuda' else 'float32',
                  crop='masked_gray_square', text_length=64)
    cache_id = digest(recipe)
    output = args.output / cache_id[:16]
    output.mkdir(parents=True, exist_ok=True)
    meta = output / 'recipe.json'
    if meta.exists() and json.loads(meta.read_text()) != recipe:
        raise ValueError('Cache recipe mismatch')
    meta.write_text(json.dumps(recipe, indent=2))
    scenes_dir = output / 'scenes'
    scenes_dir.mkdir(exist_ok=True)
    subset_id = digest([r['key'] for r in rows])[:12]
    index_path = output / f'{manifest["mode"]}-{manifest["split"]}-{subset_id}.json'
    if index_path.exists():
        existing = json.loads(index_path.read_text())
        if (existing['fingerprint'] != manifest['fingerprint'] or existing['rows'] != rows
                or existing['recipe'] != recipe):
            raise ValueError('Existing feature index belongs to changed annotations; choose a new output directory')
        if not (output / existing['text']).is_file() or any(
                not (output / f).is_file() for f in existing['scenes'].values()):
            raise FileNotFoundError('Completed feature index has missing files; use a new output directory')
        print(f'Already complete: {index_path.resolve()}')
        return
    dtype = torch.float16 if args.device == 'cuda' else torch.float32
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    model = AutoModel.from_pretrained(args.model, local_files_only=True, torch_dtype=dtype).to(args.device).eval()
    model.requires_grad_(False)
    if args.device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    def image_features(images):
        pieces = []
        for start in range(0, len(images), args.chunk_size):
            batch = processor(images=images[start:start+args.chunk_size], return_tensors='pt')
            pixels = batch['pixel_values'].to(device=args.device, dtype=dtype)
            with torch.no_grad():
                pieces.append(pooled(model.get_image_features(pixel_values=pixels)))
        return torch.cat(pieces).half()
    scenes = sorted({r['scene'] for r in rows})
    print(f'{len(rows)} expressions; {len(scenes)} unique scenes. Text storage: {len(rows)*768*2/1e6:.1f} MB', flush=True)
    scene_files = {}
    for i, scene in enumerate(scenes):
        file = scenes_dir / (digest(scene) + '.pt')
        scene_files[scene] = str(file.relative_to(output))
        if not file.exists():
            ids, geometry, crops, full = scene_inputs(args.ocid_root, scene)
            visual = image_features(crops + [full])
            record = dict(ids=ids, geometry=torch.from_numpy(geometry), visual=visual[:-1], scene=visual[-1])
            temporary = file.with_suffix('.tmp')
            torch.save(record, temporary)
            temporary.replace(file)
        if (i+1) % 20 == 0 or i+1 == len(scenes):
            print(f'Scenes {i+1}/{len(scenes)}', flush=True)
    text_path = index_path.with_suffix('.npy')
    temporary_text = text_path.with_suffix('.tmp.npy')
    text_array = np.lib.format.open_memmap(temporary_text, mode='w+', dtype=np.float16, shape=(len(rows), 768))
    truncated = 0
    for start in range(0, len(rows), args.chunk_size):
        sentences = [r['sentence'] for r in rows[start:start+args.chunk_size]]
        lengths = processor.tokenizer(sentences, truncation=False)['input_ids']
        truncated += sum(len(ids) > 64 for ids in lengths)
        batch = processor(text=sentences, padding='max_length', max_length=64, truncation=True, return_tensors='pt').to(args.device)
        with torch.no_grad():
            features = pooled(model.get_text_features(**batch))
        text_array[start:start+len(sentences)] = features.numpy().astype(np.float16)
        if start % (args.chunk_size*100) == 0:
            print(f'Text {start+len(sentences)}/{len(rows)}', flush=True)
    text_array.flush()
    del text_array
    temporary_text.replace(text_path)
    peak = torch.cuda.max_memory_allocated()/1024**3 if args.device == 'cuda' else None
    index = dict(version=1, cache_id=cache_id, recipe=recipe, split=manifest['split'], mode=manifest['mode'],
                 fingerprint=manifest['fingerprint'], rows=rows, scenes=scene_files, text=text_path.name,
                 truncated_expressions=truncated, peak_cuda_allocated_gib=peak)
    index_path.write_text(json.dumps(index))
    print(f'Feature index: {index_path.resolve()}\nPeak CUDA allocated GiB: {peak}\nTruncated expressions: {truncated}', flush=True)


if __name__ == '__main__':
    main()
