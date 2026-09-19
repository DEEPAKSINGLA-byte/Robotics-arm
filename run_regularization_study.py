"""Compare one regularization change at a time using validation only."""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-features',type=Path,required=True)
    p.add_argument('--val-features',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--epochs',type=int,default=30)
    p.add_argument('--patience',type=int,default=5)
    args=p.parse_args()
    if args.output.exists():
        raise FileExistsError('Choose a fresh study directory')
    args.output.mkdir(parents=True)
    variants=[('baseline',256,.1),('dropout_030',256,.3),('width_128',128,.1)]
    report=dict(seed=args.seed,max_epochs=args.epochs,patience=args.patience,
                protocol='Fresh initialization; only dropout or width changes; single seed; validation only',runs=[])
    for name,width,dropout in variants:
        directory=args.output/name
        print(f'\nStarting {name}: width={width}, dropout={dropout}',flush=True)
        subprocess.run([sys.executable,'-B',str(Path(__file__).parent/'train_experiment1.py'),
            '--train-features',str(args.train_features),'--val-features',str(args.val_features),
            '--output',str(directory),'--seed',str(args.seed),'--epochs',str(args.epochs),
            '--patience',str(args.patience),'--hidden',str(width),'--dropout',str(dropout),
            '--layers','2','--batch-size','256','--precision','bf16','--lr','0.0003',
            '--log-every','500','--report-training-fit'],check=True)
        summary=json.loads((directory/'summary.json').read_text())
        settings=json.loads((directory/'settings.json').read_text())
        report['runs'].append(dict(name=name,hidden=width,dropout=dropout,
            parameters=settings['trainable_parameters'],checkpoint=str((directory/'best.pt').resolve()),**summary))
        (args.output/'comparison.json').write_text(json.dumps(report,indent=2))
    report['best_by_validation']=max(report['runs'],key=lambda r:r['best_accuracy'])['name']
    (args.output/'comparison.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
