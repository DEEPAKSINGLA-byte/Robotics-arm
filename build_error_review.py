"""Create a self-contained, validation-only visual error review. No model inference."""
import argparse
import base64
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image


def select_errors(index, predictions, count=50, seed=42):
    if index['split'] != 'val':
        raise ValueError('This review tool accepts validation features only; keep test untouched.')
    rows = {r['key']: r for r in index['rows']}
    if len(rows) != len(index['rows']):
        raise ValueError('Duplicate feature keys')
    seen = set()
    grouped = defaultdict(lambda: defaultdict(list))
    for pred in predictions:
        key = pred['key']
        if key in seen or key not in rows:
            raise ValueError('Duplicate or unknown prediction key')
        seen.add(key)
        if type(pred['correct']) is not bool:
            raise ValueError('correct must be a boolean')
        if pred['correct'] != (pred['predicted_index'] == pred['target_index']):
            raise ValueError('Prediction correctness is inconsistent')
        row = rows[key]
        if pred['correct'] != (pred['predicted_scene_instance_id'] == row['target_id']):
            raise ValueError('Prediction IDs are inconsistent with feature targets')
        if not pred['correct']:
            grouped[row['group']][row['scene']].append((row, pred))
    if seen != set(rows):
        raise ValueError('Prediction file does not cover the complete validation index')
    wrong_scenes = sum(len(scenes) for scenes in grouped.values())
    if not 1 <= count <= wrong_scenes:
        raise ValueError(f'Request between 1 and {wrong_scenes} distinct wrong scenes')
    rng = random.Random(seed)
    groups = sorted(grouped)
    rng.shuffle(groups)
    queues = {}
    for group in groups:
        scenes = sorted(grouped[group])
        rng.shuffle(scenes)
        queues[group] = scenes
    chosen = []
    while len(chosen) < count:
        for group in groups:
            if queues[group]:
                scene = queues[group].pop()
                chosen.append(rng.choice(grouped[group][scene]))
                if len(chosen) == count:
                    break
    return chosen, dict(wrong_expressions=sum(len(v) for g in grouped.values() for v in g.values()),
                        wrong_scenes=wrong_scenes,
                        selected_by_group=dict(sorted(Counter(r['group'] for r, _ in chosen).items())))


def inside(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Dataset path escapes root')
    return path


def outline(mask):
    if not mask.any():
        raise ValueError('Selected object has an empty mask')
    padded = np.pad(mask, 1)
    inner = padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:]
    edge = mask & ~inner
    runs = []
    for y in range(edge.shape[0]):
        changes = np.diff(np.pad(edge[y].astype(np.int8), 1))
        for start, end in zip(np.where(changes == 1)[0], np.where(changes == -1)[0]):
            runs.append([int(start), y, int(end - start)])
    ys, xs = np.where(mask)
    return runs, [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def make_case(root, row, pred):
    rgb_path = inside(root, row['scene'])
    rel = Path(row['scene'])
    if rel.parent.name != 'rgb':
        raise ValueError('Expected an rgb scene path')
    label_path = inside(root, rel.parent.parent / 'label' / rel.name)
    with Image.open(rgb_path) as rgb, Image.open(label_path) as label:
        if rgb.size != label.size:
            raise ValueError('RGB and label sizes differ')
        width, height = rgb.size
        labels = np.asarray(label)
        if labels.ndim != 2:
            raise ValueError('Expected a two-dimensional instance label image')
        ids = sorted(int(v) for v in np.unique(labels) if v != 0)
        if ids[pred['target_index']] != row['target_id'] or ids[pred['predicted_index']] != pred['predicted_scene_instance_id']:
            raise ValueError('Candidate index does not match source label IDs')
        correct_edge, correct_box = outline(labels == row['target_id'])
        predicted_edge, predicted_box = outline(labels == pred['predicted_scene_instance_id'])
    return dict(key=row['key'], scene=row['scene'], group=row['group'], sentence=row['sentence'],
                target_id=row['target_id'], predicted_id=pred['predicted_scene_instance_id'],
                candidate_count=len(ids), width=width, height=height,
                image='data:image/png;base64,' + base64.b64encode(rgb_path.read_bytes()).decode(),
                correct_edge=correct_edge, predicted_edge=predicted_edge,
                correct_box=correct_box, predicted_box=predicted_box)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features', type=Path, required=True)
    p.add_argument('--predictions', type=Path, required=True)
    p.add_argument('--ocid-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--count', type=int, default=50)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('Choose a fresh review directory to preserve existing reports')
    index = json.loads(args.features.read_text())
    raw_predictions = args.predictions.read_bytes()
    predictions = [json.loads(line) for line in raw_predictions.splitlines() if line.strip()]
    selected, stats = select_errors(index, predictions, args.count, args.seed)
    cases = []
    for row, pred in selected:
        cases.append(make_case(args.ocid_root, row, pred))
        print(f'Prepared {len(cases)}/{len(selected)} images', flush=True)
    source_hash = hashlib.sha256(raw_predictions).hexdigest()
    report_id = hashlib.sha256(json.dumps([source_hash, args.seed, [c['key'] for c in cases]]).encode()).hexdigest()[:20]
    metadata = dict(report_id=report_id, count=len(cases), seed=args.seed, split='val', mode=index['mode'],
                    cache_id=index['cache_id'], fingerprint=index['fingerprint'],
                    features=str(args.features.resolve()), predictions=str(args.predictions.resolve()),
                    predictions_sha256=source_hash, ocid_root=str(args.ocid_root.resolve()),
                    selection='Round-robin across shuffled sequence groups; random distinct scenes; one random error per scene',
                    caveat='Diversity-focused error sample, not representative error frequencies or a new accuracy estimate.', **stats)
    payload = json.dumps(dict(metadata=metadata, cases=cases), ensure_ascii=True).replace('<', '\\u003c')
    template = (Path(__file__).parent / 'templates' / 'error_review_template.html').read_text()
    if template.count('__REVIEW_DATA__') != 1:
        raise ValueError('Invalid review template')
    args.output.mkdir(parents=True)
    (args.output/'index.html').write_text(template.replace('__REVIEW_DATA__', payload))
    summary = dict(metadata=metadata, cases=[{k:v for k,v in c.items() if k not in ('image','correct_edge','predicted_edge')} for c in cases])
    (args.output/'selection.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(metadata, indent=2))
    print('Open:', args.output/'index.html')


if __name__ == '__main__':
    main()
