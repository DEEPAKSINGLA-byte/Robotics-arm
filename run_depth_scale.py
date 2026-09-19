"""Frozen MoGe-2 + RGB-conditioned residual metric scale; no test-set tuning."""
import argparse
import json
import math
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image
import torch
from torch import nn
from experiment_io import file_hash as sha, save_json as _save_json

ROOT = Path(__file__).resolve().parent
DATA = str(ROOT / 'data/OCID-dataset')
VERSION = 1


def save_json(path, value):
    # Preserve the depth experiment's existing newline-terminated JSON format.
    _save_json(path, value, trailing_newline=True)


def catalog(splits):
    catalogs, fingerprints = {}, set()
    for split in ('train', 'val', 'test'):
        manifest = json.loads((Path(splits) / f'{split}.json').read_text())
        if manifest['mode'] != 'sequence' or manifest['split'] != split:
            raise ValueError('Only the established sequence split is supported')
        fingerprints.add(manifest['fingerprint'])
        scenes = {}
        for row in manifest['rows']:
            scene = row['scene']
            if Path(scene).is_absolute() or '..' in Path(scene).parts:
                raise ValueError('Unsafe scene path')
            record = scenes.setdefault(scene, dict(scene=scene, group=row['group'], ids=set()))
            if record['group'] != row['group']:
                raise ValueError('Inconsistent scene group')
            record['ids'].add(row['target_id'])
        catalogs[split] = [{**r, 'ids': sorted(r['ids'])} for _, r in sorted(scenes.items())]
    if len(fingerprints) != 1:
        raise ValueError('Split fingerprints differ')
    for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        for key in ('scene', 'group'):
            if {r[key] for r in catalogs[a]} & {r[key] for r in catalogs[b]}:
                raise ValueError(f'{key} overlap: {a}/{b}')
    return catalogs, fingerprints.pop()


class ScaleHead(nn.Module):
    """Only RGB-derived global features enter this residual log-scale head."""
    def __init__(self, width, hidden=64):
        super().__init__()
        self.register_buffer('mean', torch.zeros(width))
        self.register_buffer('std', torch.ones(width))
        self.net = nn.Sequential(nn.Linear(width, hidden), nn.GELU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features):
        return self.net((features - self.mean) / self.std).squeeze(-1).clamp(-4, 4)


def target_scale(pred, depth_mm):
    gt = depth_mm.astype(np.float64) / 1000
    good = (gt > 0) & np.isfinite(pred) & (pred > 0)
    if good.sum() < 100:
        raise ValueError('Too few valid depth pairs')
    return float(np.log(np.median(gt[good] / pred[good])))


def load_moge(checkpoint, device):
    from moge.model.v2 import MoGeModel
    return MoGeModel.from_pretrained(str(checkpoint)).eval().to(device).requires_grad_(False)


@torch.inference_mode()
def infer_rgb(model, image_path, device, resolution):
    # Hook the existing encoder: same RGB pass, no second encoder and no GT input.
    captured = []
    hook = model.encoder.register_forward_hook(lambda module, args, output: captured.append(output[1].detach()))
    try:
        with Image.open(image_path) as im:
            rgb = np.array(im.convert('RGB'))
        image = torch.from_numpy(rgb).permute(2, 0, 1).to(device).float() / 255
        output = model.infer(image, resolution_level=resolution, use_fp16=device.startswith('cuda'),
                             apply_mask=True, fov_x=None)
    finally:
        hook.remove()
    if len(captured) != 1:
        raise RuntimeError('Unexpected MoGe encoder calls')
    feature = captured[0].float().reshape(-1).cpu().clone()
    if not torch.isfinite(feature).all():
        raise ValueError('Nonfinite RGB feature')
    return output, feature


def exclude_training(catalogs, exclusions):
    """Explicit train-only exclusions; reject typos and val/test scene names."""
    excluded = set(exclusions)
    unknown = excluded - {r['scene'] for r in catalogs['train']}
    if unknown:
        raise ValueError(f'Exclusions must name existing training scenes: {sorted(unknown)}')
    result = {**catalogs, 'train': [r for r in catalogs['train'] if r['scene'] not in excluded]}
    if not result['train']:
        raise ValueError('Cannot exclude every training image')
    return result


def prepare(args):
    catalogs, fingerprint = catalog(args.splits)
    catalogs = exclude_training(catalogs, args.exclude_train_scene)
    print(f'Preparing {len(catalogs["train"])} training / {len(catalogs["val"])} validation images; '
          f'{len(set(args.exclude_train_scene))} explicit training exclusions.', flush=True)
    selected = {}
    for split in ('train', 'val'):
        rows = catalogs[split].copy()
        random.Random(args.seed).shuffle(rows)
        selected[split] = rows[:args.limit] if args.limit else rows
    provenance = dict(version=VERSION, checkpoint_sha256=sha(args.checkpoint),
                      moge_source_sha256=sha(ROOT / 'models/moge/moge/model/v2.py'),
                      script_sha256=sha(__file__), split_fingerprint=fingerprint,
                      use_fp16=args.device.startswith('cuda'), torch_version=str(torch.__version__),
                      resolution=args.resolution, seed=args.seed, limit=args.limit,
                      excluded_train_scenes=sorted(set(args.exclude_train_scene)),
                      input='RGB only; no supplied FOV', depth_units='PNG millimetres',
                      selected=selected)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    plan = out / 'plan.json'
    if plan.exists() and json.loads(plan.read_text()) != provenance:
        raise ValueError('Cache configuration changed; choose a new output directory')
    if not plan.exists() and any(out.iterdir()):
        raise ValueError('Nonempty output without a matching plan')
    save_json(plan, provenance)
    model = None
    for split, rows in selected.items():
        for i, row in enumerate(rows):
            scene = Path(row['scene'])
            paths = {kind: Path(args.data_root) / scene.parent.parent / kind / scene.name
                     for kind in ('rgb', 'depth', 'label')}
            source_hashes = {k: sha(p) for k, p in paths.items()}
            name = f'{split}-{i:04d}'
            record_path = out / f'{name}.pt'
            if record_path.exists():
                cached = torch.load(record_path, weights_only=True)
                if cached['source_hashes'] != source_hashes or cached['row'] != row:
                    raise ValueError('Cached source data changed')
                if split == 'val' and sha(out / f'{name}.npz') != cached['map_sha256']:
                    raise ValueError('Validation cache corrupted')
                print(f'{split} {i+1}/{len(rows)} reused', flush=True)
                continue
            if model is None:
                model = load_moge(args.checkpoint, args.device)
            start = time.perf_counter()
            output, feature = infer_rgb(model, paths['rgb'], args.device, args.resolution)
            pred = output['depth'].float().cpu().numpy()
            # RGB prediction is complete before opening reference depth/labels.
            with Image.open(paths['depth']) as im:
                depth = np.array(im)
            with Image.open(paths['label']) as im:
                objects = np.isin(np.array(im), row['ids'])
            if pred.shape != depth.shape or pred.shape != objects.shape:
                raise ValueError('RGB/depth/label alignment mismatch')
            record = dict(row=row, source_hashes=source_hashes, feature=feature)
            if split == 'train':
                record['target_log_scale'] = target_scale(pred, depth)
            else:
                # Keep exact full-resolution predictions for unaligned validation.
                np.savez_compressed(out / f'{name}.npz', pred=pred, depth_mm=depth, objects=objects)
                record['map_sha256'] = sha(out / f'{name}.npz')
            temp = record_path.with_suffix('.tmp')
            torch.save(record, temp)
            temp.replace(record_path)
            print(f'{split} {i+1}/{len(rows)} | {time.perf_counter()-start:.2f}s', flush=True)
            del output
    save_json(out / 'index.json', dict(provenance=provenance,
              records={s: [f'{s}-{i:04d}.pt' for i in range(len(rows))] for s, rows in selected.items()}))
    print(f'Feature preparation complete: {out / "index.json"}', flush=True)


def metric(pred, depth_mm, objects, factor):
    gt = depth_mm.astype(np.float64) / 1000
    pred = pred.astype(np.float64) * factor
    result = {}
    for name, region in [('whole', np.ones(gt.shape, bool)), ('object', objects)]:
        reference = region & (gt > 0)
        good = reference & np.isfinite(pred) & (pred > 0)
        if not reference.any() or not good.any():
            raise ValueError('Empty validation region or missing predictions')
        p, r = pred[good], gt[good]
        result[name + '_mae_m'] = float(np.abs(p-r).mean())
        result[name + '_absrel'] = float((np.abs(p-r)/r).mean())
        result[name + '_coverage'] = float(good.sum()/reference.sum())
        result[name + '_delta1'] = float((np.maximum(p/r, r/p) < 1.25).sum()/reference.sum())
    return result


def evaluate(maps, factors, rows):
    details = []
    for arrays, factor, row in zip(maps, factors, rows):
        details.append(dict(scene=row['scene'], group=row['group'], factor=float(factor),
                            **metric(*arrays, float(factor))))
    keys = [k for k in details[0] if k.startswith(('whole_', 'object_'))]
    summary = {k: float(np.mean([r[k] for r in details])) for k in keys}
    summary['object_mae_p90_m'] = float(np.percentile([r['object_mae_m'] for r in details], 90))
    return dict(summary=summary, per_image=details)


def train(args):
    torch.manual_seed(args.seed)
    cache = Path(args.data)
    index = json.loads((cache / 'index.json').read_text())
    provenance = index['provenance']
    if provenance['version'] != VERSION:
        raise ValueError('Unsupported cache version')
    records = {s: [torch.load(cache / name, weights_only=True) for name in index['records'][s]]
               for s in ('train', 'val')}
    for s in records:
        if [r['row'] for r in records[s]] != provenance['selected'][s]:
            raise ValueError('Cache rows do not match split plan')
    for key in ('scene', 'group'):
        if {r['row'][key] for r in records['train']} & {r['row'][key] for r in records['val']}:
            raise ValueError('Train/validation leakage')
    x = torch.stack([r['feature'] for r in records['train']]).to(args.device)
    y = torch.tensor([r['target_log_scale'] for r in records['train']], device=args.device)
    vx = torch.stack([r['feature'] for r in records['val']]).to(args.device)
    if len(x) < 2 or not torch.isfinite(y).all():
        raise ValueError('Need at least two valid training images')
    maps = []
    for name, record in zip(index['records']['val'], records['val']):
        path = cache / Path(name).with_suffix('.npz')
        if sha(path) != record['map_sha256']:
            raise ValueError('Validation map fingerprint mismatch')
        with np.load(path) as a:
            maps.append(tuple(a[k] for k in ('pred', 'depth_mm', 'objects')))
    rows = [r['row'] for r in records['val']]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError('Choose an empty training output; previous runs are never overwritten')
    head = ScaleHead(x.shape[1], args.hidden).to(args.device)
    with torch.no_grad():
        head.mean.copy_(x.mean(0))
        head.std.copy_(x.std(0).clamp_min(0.01))
    raw = evaluate(maps, np.ones(len(rows)), rows)
    constant_log = float(np.median(y.cpu().numpy()))
    if abs(constant_log) > 4:
        raise ValueError('Scale outside allowed range; inspect depth units')
    constant = evaluate(maps, np.full(len(rows), math.exp(constant_log)), rows)
    best = raw['summary']['object_mae_m']
    best_epoch, best_kind, history = 0, 'unchanged', []

    def checkpoint(epoch, kind):
        torch.save(dict(state=head.state_dict(), width=x.shape[1], hidden=args.hidden,
                        provenance=provenance, epoch=epoch, kind=kind,
                        validation_object_mae_m=best), out / 'best.pt')

    checkpoint(0, 'unchanged')
    best_report = raw
    with torch.no_grad():
        head.net[-1].bias.fill_(constant_log)
    if constant['summary']['object_mae_m'] < best:
        best = constant['summary']['object_mae_m']
        best_kind, best_report = 'train_constant', constant
        checkpoint(0, best_kind)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    stale = 0
    print(f'Validation object MAE: unchanged {raw["summary"]["object_mae_m"]*100:.2f} cm | '
          f'train-fitted constant {constant["summary"]["object_mae_m"]*100:.2f} cm', flush=True)
    for epoch in range(1, args.epochs+1):
        start = time.perf_counter()
        head.train()
        loss_sum = 0.
        for ids in torch.randperm(len(x), device=args.device).split(args.batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.smooth_l1_loss(head(x[ids]), y[ids], beta=0.1)
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), 1.)
            optimizer.step()
            loss_sum += loss.item() * len(ids)
        if args.device.startswith('cuda'):
            torch.cuda.synchronize()
        training_seconds = time.perf_counter()-start
        head.eval()
        with torch.no_grad():
            factors = head(vx).exp().cpu().numpy()
        report = evaluate(maps, factors, rows)
        score = report['summary']['object_mae_m']
        if score < best:
            best, best_epoch, best_kind, best_report = score, epoch, 'rgb_head', report
            checkpoint(epoch, best_kind)
            stale = 0
        else:
            stale += 1
        seconds = time.perf_counter()-start
        peak = torch.cuda.max_memory_allocated()/2**30 if args.device.startswith('cuda') else 0.
        history.append(dict(epoch=epoch, loss=loss_sum/len(x), seconds=seconds,
                            training_seconds=training_seconds, validation_seconds=seconds-training_seconds,
                            peak_gpu_gib=peak, **report['summary']))
        save_json(out / 'history.json', history)
        print(f'Epoch {epoch}: {seconds:.1f}s (train {training_seconds:.1f}s + validation '
              f'{seconds-training_seconds:.1f}s) | loss {loss_sum/len(x):.4f} | '
              f'object MAE {score*100:.2f} cm | GPU {peak:.2f} GiB', flush=True)
        if stale >= args.patience:
            break
    save_json(out / 'report.json', dict(train_images=len(x), validation_images=len(vx),
              diagnostic_subset=bool(provenance['limit']), selection_metric='unaligned validation object MAE',
              test_evaluated=False, encoder_frozen=True, args=vars(args),
              unchanged=raw, train_constant=constant, constant_factor=math.exp(constant_log),
              best=best_report, best_epoch=best_epoch, best_kind=best_kind,
              epochs_completed=len(history)))
    print(f'Best: {best_kind}, epoch {best_epoch}, object MAE {best*100:.2f} cm. Report: {out / "report.json"}')


def predict(args):
    saved = torch.load(args.head, map_location='cpu', weights_only=True)
    if saved['provenance'].get('use_fp16', True) != args.device.startswith('cuda'):
        raise ValueError('Prediction precision must match feature preparation (CPU versus CUDA)')
    if sha(args.checkpoint) != saved['provenance']['checkpoint_sha256']:
        raise ValueError('MoGe checkpoint does not match training')
    if sha(ROOT / 'models/moge/moge/model/v2.py') != saved['provenance']['moge_source_sha256']:
        raise ValueError('MoGe implementation changed since feature extraction')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError('Prediction output must be empty')
    model = load_moge(args.checkpoint, args.device)
    output, feature = infer_rgb(model, args.image, args.device, saved['provenance']['resolution'])
    head = ScaleHead(saved['width'], saved['hidden'])
    head.load_state_dict(saved['state'])
    head.eval()
    with torch.no_grad():
        factor = float(head(feature.unsqueeze(0)).exp().item())
    # Apply the same scale to XYZ and Z; intrinsics and unit normals do not change.
    np.savez_compressed(out / 'prediction.npz',
        depth_m=output['depth'].float().cpu().numpy()*factor,
        points_m=output['points'].float().cpu().numpy()*factor,
        intrinsics=output['intrinsics'].float().cpu().numpy(), mask=output['mask'].cpu().numpy())
    save_json(out / 'prediction.json', dict(image=str(args.image), factor=factor, head_kind=saved['kind'],
              rgb_only=True, reference_depth_used=False, robot_transform_applied=False,
              warning='Predicted camera-frame geometry; not verified grasp coordinates. Intrinsics are normalized.'))
    print(f'RGB-only prediction saved: {out}, multiplier {factor:.4f}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--data-root', default=DATA)
    p.add_argument('--exclude-train-scene', action='append', default=[],
                   help='Exact training RGB scene path to exclude; recorded in cache provenance')
    p.add_argument('--splits', default=str(ROOT / 'splits/sequence'))
    p.add_argument('--output', required=True)
    p.add_argument('--resolution', type=int, choices=range(10), default=9)
    p.add_argument('--limit', type=int, default=0, help='Images per split for smoke tests only; 0 = all')
    p.add_argument('--seed', type=int, default=42)
    t = sub.add_parser('train')
    t.add_argument('--data', required=True)
    t.add_argument('--output', required=True)
    t.add_argument('--epochs', type=int, default=50)
    t.add_argument('--batch-size', type=int, default=64)
    t.add_argument('--lr', type=float, default=0.0003)
    t.add_argument('--weight-decay', type=float, default=0.01)
    t.add_argument('--hidden', type=int, default=64)
    t.add_argument('--patience', type=int, default=10)
    t.add_argument('--seed', type=int, default=42)
    q = sub.add_parser('predict')
    q.add_argument('--head', required=True)
    q.add_argument('--image', required=True)
    q.add_argument('--output', required=True)
    for command in (p, t, q):
        command.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    for command in (p, q):
        command.add_argument('--checkpoint', default=str(ROOT / 'models/moge-2-vitb-normal/model.pt'))
    args = parser.parse_args()
    for name in ('epochs', 'batch_size', 'hidden', 'patience', 'lr'):
        if hasattr(args, name) and getattr(args, name) <= 0:
            parser.error(f'{name} must be positive')
    if getattr(args, 'limit', 0) < 0:
        parser.error('limit must be nonnegative')
    torch.set_num_threads(4)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable in this session; use a GPU-enabled terminal or --device cpu')
    {'prepare': prepare, 'train': train, 'predict': predict}[args.command](args)


if __name__ == '__main__':
    main()
