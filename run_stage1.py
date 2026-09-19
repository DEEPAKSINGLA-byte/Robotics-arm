"""Run complete feature extraction and training with one command."""
import argparse
import subprocess
import sys
from pathlib import Path


def run(script, *arguments, find_index=False):
    command = [sys.executable, '-B', str(Path(__file__).parent / script), *map(str, arguments)]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, text=True)
    index = None
    for line in process.stdout:
        print(line, end='', flush=True)
        for prefix in ('Feature index: ', 'Already complete: '):
            if line.startswith(prefix):
                index = line[len(prefix):].strip()
    if process.wait():
        raise RuntimeError(f'{script} failed; see the message above')
    if find_index and index is None:
        raise RuntimeError('Extractor did not return its feature index')
    return index


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['official','sequence'], default='sequence')
    p.add_argument('--annotations-dir', type=Path, default=Path('data/annotations'))
    p.add_argument('--ocid-root', type=Path, default=Path('data/OCID-dataset'))
    p.add_argument('--model', type=Path, default=Path('models/siglip2-base-patch16-224'))
    p.add_argument('--features', type=Path, default=Path('features'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--feature-storage', choices=['auto','cuda','cpu'], default='auto')
    p.add_argument('--precision', choices=['auto','fp32','bf16'], default='auto')
    p.add_argument('--log-every', type=int, default=50)
    p.add_argument('--init-checkpoint', type=Path, help='Warm start weights with a new optimizer')
    p.add_argument('--hidden',type=int)
    p.add_argument('--layers',type=int)
    p.add_argument('--dropout',type=float)
    p.add_argument('--report-training-fit',action='store_true')
    p.add_argument('--chunk-size', type=int, default=8)
    p.add_argument('--device', choices=['cuda','cpu'], default='cuda')
    p.add_argument('--evaluate-test', action='store_true', help='Evaluate test only when experimental choices are final')
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('Choose a new training output directory')
    split_dir = Path('splits') / args.mode
    if not split_dir.exists():
        run('prepare_splits.py','--annotations-dir',args.annotations_dir,'--mode',args.mode,'--output',split_dir)
    indexes = {}
    for split in ('train','val'):
        indexes[split] = run('extract_siglip_features.py','--manifest',split_dir/f'{split}.json',
            '--ocid-root',args.ocid_root,'--model',args.model,'--output',args.features,
            '--chunk-size',args.chunk_size,'--device',args.device,find_index=True)
    extras = ['--init-checkpoint',args.init_checkpoint] if args.init_checkpoint else []
    for key in ('hidden','layers','dropout'):
        if getattr(args,key) is not None:
            extras += ['--'+key,getattr(args,key)]
    if args.report_training_fit:
        extras += ['--report-training-fit']
    run('train_experiment1.py','--train-features',indexes['train'],'--val-features',indexes['val'],
        '--output',args.output,'--epochs',args.epochs,'--device',args.device,
        '--batch-size',args.batch_size,'--feature-storage',args.feature_storage,
        '--precision',args.precision,'--log-every',args.log_every,*extras)
    if args.evaluate_test:
        index = run('extract_siglip_features.py','--manifest',split_dir/'test.json',
            '--ocid-root',args.ocid_root,'--model',args.model,'--output',args.features,
            '--chunk-size',args.chunk_size,'--device',args.device,find_index=True)
        run('evaluate_experiment1.py','--features',index,'--checkpoint',args.output/'best.pt',
            '--output',args.output/'test.json','--device',args.device)
        run('evaluate_experiment1.py','--features',index,'--zero-shot',
            '--output',args.output/'test-similarity.json','--device',args.device)


if __name__ == '__main__':
    main()
