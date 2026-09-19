"""Build official expression splits or a separate sequence-disjoint experiment."""
import argparse
import hashlib
import json
import random
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def read_rows(path, source):
    raw = json.loads(Path(path).read_text())
    entries = raw.items() if isinstance(raw, dict) else enumerate(raw)
    rows = []
    for key, row in entries:
        scene = row['scene_path'].replace('\\', '/')
        # Camera views of the same acquisition sequence belong together.
        # Use collection + seq folder, ignoring floor/table/top/bottom views.
        seq = next((part for part in Path(scene).parts if part.startswith('seq')), None)
        if seq is None:
            raise ValueError(f'Cannot identify acquisition sequence: {scene}')
        rows.append(dict(key=f'{source}:{key}', scene=scene,
                         group=f'{Path(scene).parts[0]}/{seq}',
                         sentence=row['sentence'], target_id=int(row['scene_instance_id']),
                         target_class=row['class']))
    return rows


def make_splits(sources, mode, seed):
    if mode == 'official':
        return sources
    rows = sum(sources.values(), [])
    groups = sorted({r['group'] for r in rows})
    if len(groups) < 3:
        raise ValueError('At least three sequence groups are needed')
    random.Random(seed).shuffle(groups)
    nval = max(1, round(len(groups) * .1))
    ntest = max(1, round(len(groups) * .1))
    memberships = {g: ('val' if i < nval else 'test' if i < nval + ntest else 'train')
                   for i, g in enumerate(groups)}
    return {name: [r for r in rows if memberships[r['group']] == name]
            for name in ('train', 'val', 'test')}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--annotations-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mode', choices=['official', 'sequence'], default='sequence')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('Choose a new split output directory')
    sources = {name: read_rows(args.annotations_dir / f'{name}_expressions.json', name)
               for name in ('train', 'val', 'test')}
    splits = make_splits(sources, args.mode, args.seed)
    fingerprint = digest(dict(mode=args.mode, seed=args.seed, splits=splits))
    args.output.mkdir(parents=True)
    summary = dict(mode=args.mode, seed=args.seed, fingerprint=fingerprint, splits={})
    for name, rows in splits.items():
        document = dict(version=1, mode=args.mode, split=name, fingerprint=fingerprint, rows=rows)
        (args.output / f'{name}.json').write_text(json.dumps(document))
        summary['splits'][name] = dict(expressions=len(rows), scenes=len({r['scene'] for r in rows}),
                                      groups=len({r['group'] for r in rows}))
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
