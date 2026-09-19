"""Small, fixed validation comparison of automatic SAM settings; no training."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from compare_sam_selectors import remaining_index
from run_experiment2 import select_rows, save_json, file_hash


VARIANTS = {
    'baseline': [],
    'dense64': ['--points-per-side', '64'],
    'crops1': ['--crop-layers', '1'],
    'relaxed': ['--pred-iou-thresh', '0.7', '--stability-score-thresh', '0.9'],
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--count', type=int, default=20)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--val-features', type=Path, default=Path('features/4764b544cec72bc7/sequence-val-5f243509242f.json'))
    p.add_argument('--adapted-run', type=Path, default=Path('runs/sam-adaptation-300'))
    p.add_argument('--variants', nargs='+', choices=list(VARIANTS), default=list(VARIANTS))
    p.add_argument('--baseline-run', type=Path, help='Reuse a completed baseline on the exact selected images and expressions')
    p.add_argument('--check-only', action='store_true', help='Validate setup without creating outputs or running GPU inference')
    args = p.parse_args()
    if len(set(args.variants)) != len(args.variants):
        p.error('Do not repeat variants')
    if 'baseline' not in args.variants and args.baseline_run is None:
        p.error('Omitting baseline requires --baseline-run')
    if 'baseline' in args.variants and args.baseline_run is not None:
        p.error('Choose either a new baseline or --baseline-run, not both')
    if args.output.exists():
        raise FileExistsError('Use a new output folder to preserve results')
    index = json.loads(args.val_features.read_text())
    adapted = json.loads((args.adapted_run/'settings.json').read_text())
    used = json.loads((Path(adapted['data'])/'selection.json').read_text())['val']
    remainder = remaining_index(index, used)
    selected = select_rows(remainder, args.count, args.seed)
    # Freeze exact expressions as well as images across variants.
    subset = dict(index, rows=selected)
    source = json.loads((Path(adapted['preparation']['validation_run'])/'settings.json').read_text())
    variants = {k: v for k, v in VARIANTS.items() if k in args.variants}
    summaries, baseline_rows = {}, None
    if args.baseline_run:
        baseline_settings = json.loads((args.baseline_run/'settings.json').read_text())
        baseline_rows = json.loads((args.baseline_run/'evaluation.json').read_text())
        expected = {r['key']: r for r in selected}
        if len(baseline_rows) != len(selected) or {r['key'] for r in baseline_rows} != set(expected):
            raise ValueError('Reused baseline must cover exactly the selected expressions')
        for row in baseline_rows:
            if any(row[k] != expected[row['key']][k] for k in ('scene', 'sentence')):
                raise ValueError('Baseline image or sentence differs')
        for key in ('cache_id', 'fingerprint', 'mode', 'split'):
            if baseline_settings[key] != index[key]:
                raise ValueError(f'Baseline provenance mismatch: {key}')
        if baseline_settings['checkpoint_sha256'] != file_hash(args.adapted_run/'best.pt'):
            raise ValueError('Baseline used a different selector')
        if baseline_settings['sam_weights_sha256'] != file_hash(Path(source['sam_path'])/'model.safetensors'):
            raise ValueError('Baseline used different SAM weights')
        if baseline_settings['sam_settings'] != dict(points_per_batch=16, points_per_crop=32,
                crops_n_layers=0, pred_iou_thresh=.8, stability_score_thresh=.95, crops_nms_thresh=.7):
            raise ValueError('Reused baseline does not use the original SAM settings')
        if baseline_settings['threshold'] != .5 or baseline_settings['filtering'] != dict(
                min_area=32, max_area_fraction=.95, duplicate_iou=.95, max_candidates=128):
            raise ValueError('Baseline evaluation threshold or candidate filters differ')
        summaries['baseline_reused'] = json.loads((args.baseline_run/'summary.json').read_text())
    if args.check_only:
        print(f'Validated {len(selected)} images; run only {list(variants)}; reused baseline: {args.baseline_run}', flush=True)
        return
    args.output.mkdir(parents=True)
    manifest = args.output/'validation-subset.json'
    save_json(manifest, subset)
    save_json(args.output/'study_settings.json', dict(count=args.count, seed=args.seed, variants=variants,
        baseline_run=str(args.baseline_run.resolve()) if args.baseline_run else None,
        checkpoint=str((args.adapted_run/'best.pt').resolve()),
        selection='Balanced random subset of remaining validation images, independent of correctness',
        note='Exploratory validation tuning; these images are not a held-out test set.',
        training_performed=False, test_evaluated=False))
    env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    for name, overrides in variants.items():
        print(f'=== SAM settings: {name} ===', flush=True)
        subprocess.run([sys.executable, '-B', str(Path(__file__).with_name('run_experiment2.py')),
            '--val-features', str(manifest), '--checkpoint', str(args.adapted_run/'best.pt'),
            '--count', str(args.count), '--seed', str(args.seed), '--output', str(args.output/name),
            '--sam', source['sam_path'], '--siglip', source['siglip_path'], '--ocid-root', source['ocid_root'],
            *overrides], check=True, env=env)
        summary = json.loads((args.output/name/'summary.json').read_text())
        rows = json.loads((args.output/name/'evaluation.json').read_text())
        if baseline_rows is None:
            baseline_rows = rows
            baseline_settings = json.loads((args.output/name/'settings.json').read_text())
        current_settings = json.loads((args.output/name/'settings.json').read_text())
        for key in ('checkpoint_sha256', 'sam_weights_sha256', 'cache_id', 'fingerprint', 'threshold'):
            if current_settings[key] != baseline_settings[key]:
                raise ValueError(f'Uncontrolled change: {key}')
        if len(rows) != len(baseline_rows):
            raise ValueError('Incomplete evaluation')
        reference = {r['key']: r for r in baseline_rows}
        if len(reference) != len(rows) or {r['key'] for r in rows} != set(reference):
            raise ValueError('Variant expression keys differ')
        baseline_rows = [reference[r['key']] for r in rows]
        for a, b in zip(baseline_rows, rows):
            if any(a[k] != b[k] for k in ('key', 'scene', 'sentence', 'rgb_sha256')):
                raise ValueError('Variant images or expressions differ')
        summary['recovered_targets'] = sum(not a['evaluation']['proposal_available_at_threshold'] and b['evaluation']['proposal_available_at_threshold'] for a, b in zip(baseline_rows, rows))
        summary['lost_targets'] = sum(a['evaluation']['proposal_available_at_threshold'] and not b['evaluation']['proposal_available_at_threshold'] for a, b in zip(baseline_rows, rows))
        summary['fixed_selections'] = sum(not a['evaluation']['selected_success_at_threshold'] and b['evaluation']['selected_success_at_threshold'] for a, b in zip(baseline_rows, rows))
        summary['newly_wrong_selections'] = sum(a['evaluation']['selected_success_at_threshold'] and not b['evaluation']['selected_success_at_threshold'] for a, b in zip(baseline_rows, rows))
        summaries[name] = summary
        save_json(args.output/'comparison.json', summaries)
        print(f'{name}: usable target {summary["proposal_available_count"]}/{args.count}; '
              f'correct selection {summary["selected_success_count"]}/{args.count}', flush=True)
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == '__main__':
    main()
