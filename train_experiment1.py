"""Train the small relation model using resident features and timed progress."""
import argparse
import json
import math
import random
import time
from pathlib import Path
import numpy as np
import torch
from grounding_data import FeatureDataset, ResidentFeatures, verify_pair
from grounding_model import GroundingModel
from evaluate_experiment1 import evaluate


def sync(device):
    if device == 'cuda':
        torch.cuda.synchronize()


def resolve_model_config(args, initial=None):
    requested = {k:getattr(args,k) for k in ('hidden','layers','dropout')}
    if initial:
        for key,value in requested.items():
            if value is not None and value != initial['config'][key]:
                raise ValueError(f'Cannot change {key} when loading a checkpoint; start a fresh experiment')
        return dict(initial['config'])
    config = dict(feature_dim=768,hidden=256,layers=2,heads=4,dropout=0.0 if args.overfit else .1)
    config.update({k:v for k,v in requested.items() if v is not None})
    if config['hidden']<=0 or config['hidden']%config['heads']:
        raise ValueError('hidden must be positive and divisible by four attention heads')
    if config['layers']<1 or not 0<=config['dropout']<1:
        raise ValueError('layers must be positive and dropout in [0,1)')
    return config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-features', required=True)
    p.add_argument('--val-features')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--overfit', action='store_true')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--feature-storage', choices=['auto','cuda','cpu'], default='auto')
    p.add_argument('--precision', choices=['auto','fp32','bf16'], default='auto')
    p.add_argument('--log-every', type=int, default=50)
    p.add_argument('--init-checkpoint', type=Path, help='Load learned weights into a NEW optimizer/run')
    p.add_argument('--hidden',type=int,help='Internal width (default: 256)')
    p.add_argument('--layers',type=int,help='Relation layers (default: 2)')
    p.add_argument('--dropout',type=float,help='Dropout probability during training (default: 0.1)')
    p.add_argument('--report-training-fit',action='store_true',help='Evaluate the best checkpoint on training data with dropout disabled')
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--patience', type=int, default=5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', choices=['cuda','cpu'], default='cuda')
    args = p.parse_args()
    if min(args.epochs,args.batch_size,args.patience,args.log_every)<1 or not math.isfinite(args.lr) or args.lr<=0:
        p.error('epochs, batch-size, patience, log-every and lr must be positive and finite')
    if args.overfit and args.val_features:
        p.error('--overfit evaluates its own training subset; omit --val-features')
    if not args.overfit and not args.val_features:
        p.error('Normal training needs --val-features')
    if args.output.exists():
        raise FileExistsError('Choose a fresh run directory to preserve previous results')
    if args.device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable in this session; run from your terminal or use --device cpu')
    supports_bf16 = args.device=='cuda' and torch.cuda.is_bf16_supported()
    if args.precision=='bf16' and not supports_bf16:
        p.error('BF16 requires a supported CUDA GPU; use --precision fp32')
    amp = supports_bf16 and args.precision!='fp32'
    precision = 'bf16' if amp else 'fp32'
    torch.set_num_threads(4)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.mha.set_fastpath_enabled(False)
    train = FeatureDataset(args.train_features)
    if train.index['split']!='train':
        raise ValueError('Use training features for training')
    if args.overfit and len(train)>200:
        raise ValueError('Overfit check requires at most 200 examples')
    val = train if args.overfit else FeatureDataset(args.val_features)
    if not args.overfit:
        verify_pair(train,val)
    initial = None
    if args.init_checkpoint:
        initial = torch.load(args.init_checkpoint,weights_only=True,map_location='cpu')
        for key in ('cache_id','mode','fingerprint'):
            if initial[key]!=train.index[key]:
                raise ValueError(f'Initial checkpoint mismatch: {key}')
        if initial['overfit'] and not args.overfit:
            raise ValueError('Do not use the debugging checkpoint for the full experiment')
        if train.index['mode']=='sequence' and not args.overfit:
            if set(initial['training_groups']) & {r['group'] for r in val.rows}:
                raise ValueError('Initial checkpoint has seen validation sequences')
    config = resolve_model_config(args,initial)
    print(f'Relation model: {config}',flush=True)
    print('Loading feature tables once. No SigLIP inference is needed during training.',flush=True)
    preload_start = time.perf_counter()
    resident = ResidentFeatures(train,args.feature_storage,args.device)
    val_resident = resident if args.overfit else ResidentFeatures(val,args.feature_storage,args.device)
    loader = resident.loader(args.batch_size,shuffle=True,metadata=False)
    validation = val_resident.loader(args.batch_size)
    sync(args.device)
    preload_seconds = time.perf_counter()-preload_start
    model = GroundingModel(**config)
    if initial:
        model.load_state_dict(initial['model'])
        print(f'Warm start from {args.init_checkpoint}; fresh optimizer and epoch count.',flush=True)
    model = model.to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=0 if args.overfit else .01,
                                 fused=args.device=='cuda')
    args.output.mkdir(parents=True)
    settings = {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    settings.update(train_samples=len(train),validation_samples=len(val),model=model.config,
                    actual_precision=precision,feature_locations=[resident.location,val_resident.location],
                    preload_seconds=preload_seconds,trainable_parameters=sum(p.numel() for p in model.parameters()))
    (args.output/'settings.json').write_text(json.dumps(settings,indent=2))
    print(f'Batch {args.batch_size} | {len(loader)} batches/epoch | {precision} | '
          f'preload {preload_seconds:.1f}s',flush=True)
    print('Measuring frozen SigLIP similarity baseline...',flush=True)
    baseline = evaluate(None,validation,args.device)
    (args.output/'zero_shot.json').write_text(json.dumps(baseline,indent=2))
    print(f'Frozen SigLIP similarity accuracy: {baseline["accuracy"]:.2%}',flush=True)
    best,stale,history = -1.0,0,[]

    def checkpoint(epoch,accuracy):
        return dict(model=model.state_dict(),config=model.config,epoch=epoch,accuracy=accuracy,
                    overfit=args.overfit,precision=precision,
                    training_groups=sorted(set(initial['training_groups'] if initial else []) | {r['group'] for r in train.rows}),
                    **{k:train.index[k] for k in ('cache_id','mode','fingerprint')})

    if initial:
        result = evaluate(model,validation,args.device,amp=amp)
        best = result['accuracy']
        torch.save(checkpoint(0,best),args.output/'best.pt')
        (args.output/'best_metrics.json').write_text(json.dumps(result,indent=2))
        print(f'Initial checkpoint validation: {best:.2%}',flush=True)
    for epoch in range(1,args.epochs+1):
        sync(args.device)
        epoch_start = time.perf_counter()
        if args.device=='cuda':
            torch.cuda.reset_peak_memory_stats()
        model.train()
        total_loss = torch.zeros((),device=args.device)
        correct = torch.zeros((),dtype=torch.long,device=args.device)
        total = 0
        print(f'Epoch {epoch}/{args.epochs} started',flush=True)
        for step,(inputs,target,_) in enumerate(loader,1):
            inputs = {k:v.to(args.device) for k,v in inputs.items()}
            target = target.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=args.device,dtype=torch.bfloat16,enabled=amp):
                scores = model(**inputs)
                loss = torch.nn.functional.cross_entropy(scores,target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step()
            total_loss += loss.detach()*len(target)
            correct += (scores.detach().argmax(-1)==target).sum()
            total += len(target)
            if step==1 or step%args.log_every==0 or step==len(loader):
                loss_value = total_loss.item()/total
                if not math.isfinite(loss_value):
                    raise ValueError('Non-finite training loss')
                sync(args.device)
                elapsed = time.perf_counter()-epoch_start
                speed = total/elapsed
                remaining = (len(train)-total)/speed
                print(f'  Batch {step}/{len(loader)} | loss {loss_value:.4f} | '
                      f'{speed:.0f} examples/s | elapsed {elapsed:.1f}s | train ETA {remaining:.1f}s',flush=True)
        sync(args.device)
        train_seconds = time.perf_counter()-epoch_start
        validation_start = time.perf_counter()
        print('  Checking validation...',flush=True)
        result = evaluate(model,validation,args.device,amp=amp)
        sync(args.device)
        validation_seconds = time.perf_counter()-validation_start
        record = dict(epoch=epoch,train_loss=total_loss.item()/total,train_accuracy=correct.item()/total,
                      evaluation_loss=result['loss'],evaluation_accuracy=result['accuracy'],
                      train_seconds=train_seconds,validation_seconds=validation_seconds,
                      epoch_seconds=time.perf_counter()-epoch_start,examples_per_second=total/train_seconds,
                      peak_cuda_allocated_gib=torch.cuda.max_memory_allocated()/1024**3 if args.device=='cuda' else None,
                      peak_cuda_reserved_gib=torch.cuda.max_memory_reserved()/1024**3 if args.device=='cuda' else None)
        history.append(record)
        print(f'Epoch {epoch} finished in {record["epoch_seconds"]:.1f}s '
              f'(training {train_seconds:.1f}s + validation {validation_seconds:.1f}s) | '
              f'train accuracy {record["train_accuracy"]:.2%} | validation {result["accuracy"]:.2%} | '
              f'peak allocated GPU memory {record["peak_cuda_allocated_gib"]} GiB',flush=True)
        if result['accuracy']>best:
            best,stale = result['accuracy'],0
            torch.save(checkpoint(epoch,best),args.output/'best.pt')
            (args.output/'best_metrics.json').write_text(json.dumps(result,indent=2))
        else:
            stale += 1
        (args.output/'history.json').write_text(json.dumps(history,indent=2))
        if args.overfit and best>=.99:
            break
        if not args.overfit and stale>=args.patience:
            print(f'Stopping after {stale} epochs without validation improvement.',flush=True)
            break
    summary = dict(best_accuracy=best,epochs_completed=len(history),debug_overfit=args.overfit,
                   mean_epoch_seconds=sum(r['epoch_seconds'] for r in history)/len(history),
                   precision=precision,peak_cuda_allocated_gib=max((r['peak_cuda_allocated_gib'] or 0) for r in history),
                   overfit_passed=best>=.99 if args.overfit else None)
    if args.report_training_fit:
        print('Measuring the best checkpoint on training data with dropout disabled...',flush=True)
        saved = torch.load(args.output/'best.pt',map_location='cpu',weights_only=True)
        model.load_state_dict(saved['model'])
        fit = evaluate(model,resident.loader(args.batch_size),args.device,amp=amp)
        (args.output/'best_training_metrics.json').write_text(json.dumps(fit,indent=2))
        summary.update(best_training_eval_accuracy=fit['accuracy'],
                       best_generalization_gap=fit['accuracy']-best,best_epoch=saved['epoch'])
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
    if args.overfit and best<.99:
        raise SystemExit('Overfit check did not reach 99%; inspect before a long run.')


if __name__=='__main__':
    main()
