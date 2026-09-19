"""Measure candidate-selection accuracy with counts and transparent subgroups."""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from grounding_data import FeatureDataset, collate
from grounding_model import GroundingModel


@torch.no_grad()
def evaluate(model, loader, device, predictions=None, amp=False):
    if model is not None:
        model.eval()
    groups = defaultdict(lambda: [0, 0])
    total_loss = torch.zeros((),device=device)
    chance = 0.0
    class_complete = 0
    for inputs, target, metadata in loader:
        inputs = {k: v.to(device) for k, v in inputs.items()}
        target = target.to(device)
        if model is None:
            scores = (F.normalize(inputs['visual'], dim=-1) * F.normalize(inputs['text'], dim=-1).unsqueeze(1)).sum(-1)
            scores = scores.masked_fill(~inputs['valid'], float('-inf'))
        else:
            with torch.autocast(device_type=device,dtype=torch.bfloat16,enabled=amp and device=='cuda'):
                scores = model(**inputs)
        total_loss += F.cross_entropy(scores.float(), target, reduction='sum').detach()
        # Transfer predictions/counts once per batch, not once per sample.
        predicted = scores.argmax(-1).cpu().tolist()
        truth = target.cpu().tolist()
        counts = inputs['valid'].sum(-1).cpu().tolist()
        for j, meta in enumerate(metadata):
            correct = int(predicted[j] == truth[j])
            n = counts[j]
            chance += 1/n
            class_complete += meta['class_coverage_complete']
            names = ['all', f'candidates/{"1-5" if n<=5 else "6-10" if n<=10 else "11-20" if n<=20 else "21+"}']
            if meta['duplicate']:
                names.append('known_duplicate_class')
            matched = False
            sentence = meta['sentence'].lower()
            for relation in ('left', 'right', 'behind', 'front', 'above', 'below', 'top', 'bottom', 'near', 'between'):
                if re.search(r'\b' + relation + r'\b', sentence):
                    names.append('relation/' + relation)
                    matched = True
            if not matched:
                names.append('relation/other')
            for name in names:
                groups[name][0] += correct
                groups[name][1] += 1
            if predictions is not None:
                predictions.write(json.dumps(dict(key=meta['key'], predicted_index=int(predicted[j]),
                    target_index=truth[j], predicted_scene_instance_id=meta['candidate_ids'][predicted[j]],
                    correct=bool(correct))) + '\n')
    total = groups['all'][1]
    if not total:
        raise ValueError('No evaluation samples')
    return dict(loss=total_loss.item()/total, accuracy=groups['all'][0]/total,
                mean_selected_mask_iou=groups['all'][0]/total,
                random_choice_expected_accuracy=chance/total,
                class_catalog_complete_samples=class_complete,
                groups={k: dict(correct=v[0], count=v[1], accuracy=v[0]/v[1]) for k,v in sorted(groups.items())},
                notes=['IoU equals accuracy because oracle instance masks are disjoint.',
                       'Relation groups use sentence keywords and can overlap.',
                       'Duplicate-class subgroup includes only duplicates confirmed by annotations in this index.'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features', required=True)
    choice = p.add_mutually_exclusive_group(required=True)
    choice.add_argument('--checkpoint')
    choice.add_argument('--zero-shot', action='store_true')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    p.add_argument('--precision',choices=['auto','fp32','bf16'],default='auto')
    args = p.parse_args()
    torch.set_num_threads(4)
    data = FeatureDataset(args.features)
    model = None
    amp = args.precision=='bf16'
    if args.checkpoint:
        saved = torch.load(args.checkpoint, weights_only=True, map_location='cpu')
        if args.precision=='auto':
            amp = saved.get('precision')=='bf16' and args.device=='cuda'
        if saved['overfit'] and data.index['split'] != 'train':
            raise ValueError('Overfit checkpoint is a debugging result, not a validation/test model')
        for key in ('cache_id', 'mode', 'fingerprint'):
            if saved[key] != data.index[key]:
                raise ValueError(f'Checkpoint/evaluation mismatch: {key}')
        if data.index['mode'] == 'sequence' and data.index['split'] != 'train':
            if set(saved['training_groups']) & {r['group'] for r in data.rows}:
                raise ValueError('Evaluation sequences overlap training')
        model = GroundingModel(**saved['config']).to(args.device)
        model.load_state_dict(saved['model'])
    if amp:
        if args.device!='cuda' or not torch.cuda.is_bf16_supported():
            raise ValueError('BF16 evaluation requires a supported CUDA GPU')
        torch.backends.mha.set_fastpath_enabled(False)
    if args.output.exists() or args.output.with_suffix('.predictions.jsonl').exists():
        raise FileExistsError('Choose a new evaluation output path')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.with_suffix('.predictions.jsonl').open('w') as stream:
        result = evaluate(model, DataLoader(data, batch_size=args.batch_size, collate_fn=collate), args.device, stream, amp=amp)
    result.update(mode=data.index['mode'], split=data.index['split'], samples=len(data),
                  checkpoint=args.checkpoint, features=str(Path(args.features).resolve()))
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
