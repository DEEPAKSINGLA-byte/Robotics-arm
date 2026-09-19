"""Compare two fixed selectors on validation images excluded from adaptation selection."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from run_experiment2 import load_proposals, save_json


def remaining_index(index, used):
    if index['split'] != 'val' or index['mode'] != 'sequence':
        raise ValueError('Use sequence validation only, never test')
    lookup = {r['key']: r for r in index['rows']}
    if not used or any(lookup.get(r['key']) != r for r in used):
        raise ValueError('Excluded selection must belong to this validation index')
    scenes = {r['scene'] for r in used}
    rows = [r for r in index['rows'] if r['scene'] not in scenes]
    if not rows:
        raise ValueError('No unused validation images remain')
    return dict(index, rows=rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--val-features', type=Path, default=Path('features/4764b544cec72bc7/sequence-val-5f243509242f.json'))
    p.add_argument('--adapted-run', type=Path, default=Path('runs/sam-adaptation-300'))
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('Choose a fresh output folder; existing results are preserved')
    settings = json.loads((args.adapted_run/'settings.json').read_text())
    prepared = Path(settings['data'])
    selection = json.loads((prepared/'selection.json').read_text())
    preparation = settings['preparation']
    index = json.loads(args.val_features.read_text())
    for key in ('cache_id', 'fingerprint', 'mode'):
        if index[key] != preparation[key]:
            raise ValueError(f'Adaptation/validation mismatch: {key}')
    remaining = remaining_index(index, selection['val'])
    count = len({r['scene'] for r in remaining['rows']})
    original_settings = json.loads((Path(preparation['validation_run'])/'settings.json').read_text())
    args.output.mkdir(parents=True)
    manifest = args.output/'remaining-val.json'
    # This manifest supplies rows and provenance only; no cached arrays are read by the evaluator.
    save_json(manifest, remaining)
    save_json(args.output/'comparison_settings.json', dict(images=count, excluded_images=len({r['scene'] for r in selection['val']}),
              seed=args.seed, adapted_run=str(args.adapted_run.resolve()), test_evaluated=False,
              note='Unused images within the same validation sequences; not an independent test set.'))
    checkpoints = {'original': Path(preparation['checkpoint']), 'adapted': args.adapted_run/'best.pt'}
    sam = preparation['sam_settings']
    filtering = preparation['filtering']
    for name, checkpoint in checkpoints.items():
        print(f'Running {name} selector on {count} remaining validation images...', flush=True)
        subprocess.run([sys.executable, '-B', str(Path(__file__).with_name('run_experiment2.py')),
                        '--val-features', str(manifest), '--checkpoint', str(checkpoint),
                        '--count', str(count), '--seed', str(args.seed), '--output', str(args.output/name),
                        '--sam', original_settings['sam_path'], '--siglip', original_settings['siglip_path'],
                        '--ocid-root', original_settings['ocid_root'],
                        '--points-per-batch', str(sam['points_per_batch']), '--points-per-side', str(sam['points_per_crop']),
                        '--min-area', str(filtering['min_area']), '--max-area-fraction', str(filtering['max_area_fraction']),
                        '--max-candidates', str(filtering['max_candidates']), '--duplicate-iou', str(filtering['duplicate_iou'])], check=True)
    old = json.loads((args.output/'original/evaluation.json').read_text())
    new = json.loads((args.output/'adapted/evaluation.json').read_text())
    if len(old) != count or len(new) != count:
        raise ValueError('Incomplete evaluation')
    for n, (a, b) in enumerate(zip(old, new), 1):
        for field in ('key', 'scene', 'sentence', 'rgb_sha256'):
            if a[field] != b[field]:
                raise ValueError(f'Comparison inputs differ: {field}')
        old_masks, old_kept = load_proposals(args.output/'original', n)
        new_masks, new_kept = load_proposals(args.output/'adapted', n)
        if old_kept != new_kept or not np.array_equal(old_masks, new_masks):
            raise ValueError('SAM masks differed between runs: do not interpret this as a selector-only comparison')
    before = [r['evaluation']['selected_success_at_threshold'] for r in old]
    after = [r['evaluation']['selected_success_at_threshold'] for r in new]
    summary = dict(images=count, original_success=sum(before), adapted_success=sum(after),
                   improvement_percentage_points=100*(sum(after)-sum(before))/count,
                   fixed_errors=sum(not a and b for a, b in zip(before, after)),
                   newly_wrong=sum(a and not b for a, b in zip(before, after)),
                   identical_masks_verified=True, test_evaluated=False)
    save_json(args.output/'comparison.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
