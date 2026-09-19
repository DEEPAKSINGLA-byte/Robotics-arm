# Train the selector with SAM candidates

This is a separate experiment. It starts from the existing best selector and
changes only that selector. SAM and SigLIP stay frozen. Existing checkpoints,
oracle features and the completed 100-image report are not overwritten.

## Why

The selector learned from dataset masks but now receives SAM masks, which can
contain fragments, duplicates and imperfect boundaries. Training on these
candidates tests whether that mismatch contributes to selection errors.
Improvement is not guaranteed.

## Prepare the data

Run from the repository root with the USB dataset connected:

```bash
# Run from the repository folder.
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 -B -m experiments.run_sam_adaptation prepare \
  --train-scenes 300 --expressions-per-scene 20 \
  --output features/sam-adaptation-300
```

The script randomly selects 300 training images, with up to 20 expressions per
image, using seed 42. It generates SAM masks with the same settings as the
100-image validation run. The exact existing validation masks and expressions
are reused; validation images are not resampled based on success or failure.

Image crops and geometry come only from predicted masks. Correct dataset masks
are used only to measure overlap and construct training answers. Frozen text
features are reused by exact expression key. Predicted image features are
computed again with the same frozen SigLIP and crop recipe.

The script verifies training/validation sequence separation, encoder identity,
baseline checkpoint identity, SAM identity and validation image hashes. It never
reads test features. It stores compact masks/features, not copies of RGB images.
`ready.json` is written only after preparation completes and reports coverage.
There is no partial-run resume: if interrupted, choose a new output folder;
keep the partial folder until you decide whether to archive it.

## Train

After preparation completes:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 -B -m experiments.run_sam_adaptation train \
  --data features/sam-adaptation-300 \
  --epochs 20 --batch-size 64 --lr 0.00003 \
  --output runs/sam-adaptation-300
```

Training uses cached features, so the USB is no longer needed. The original
baseline checkpoint must remain available. A fresh optimizer uses a smaller
learning rate than the original training. Each epoch prints loss, total epoch
time (training plus validation), validation successes and peak allocated VRAM.
There is no target GPU-utilization percentage; the selector is small.

An example is trainable if at least one proposal overlaps the correct mask at
IoU >= 0.5. All such proposals count as acceptable answers. The loss increases
their combined probability, avoiding arbitrary penalties when two masks both
describe the target. Examples with no acceptable proposal are skipped during
training and counted in the preparation report. No ground-truth candidate is
inserted to make a missing example artificially solvable.

## Fair comparison

Validation always includes **all 100 expressions**, including SAM misses and
empty proposal sets. Checkpoint selection uses end-to-end successes at IoU >=
0.5, not accuracy on only the easiest examples. No training labels or overlaps
are passed to the model.

Before training, the script evaluates the unchanged selector on these exact
cached candidates. It should reproduce 58/100; inspect any discrepancy before
trusting comparisons. The original 85/100 proposal coverage is unchanged by
selector-only training. The original checkpoint is retained as epoch-zero
`best.pt` unless a later epoch strictly improves validation success. Training
stops after five non-improving epochs by default.

Outputs include `baseline.json`, `baseline_predictions.json`, `history.json`,
`best.pt`, `best_metrics.json`, `best_predictions.json`, `settings.json` and
`summary.json`. The new best checkpoint is also compatible with the existing
`experiments/run_experiment2.py --checkpoint ...` report workflow.

This 100-image validation sample is now used for model selection: improvement
on it is not independent evidence of generalization. Do not repeatedly tune
against the held-out test set. After settling the experiment, evaluate broader
validation coverage and reserve the final test for the final model choice.

## Small setup check

Setup verification passed 21 automated tests and a real GPU trial with two
training images (8 expressions, 5 with usable proposals). It reproduced 58/100
validation successes and 85/100 proposal availability. One close-scoring wrong
prediction picked a different wrong mask in batched evaluation; both had zero
overlap. All success/failure outcomes matched. Mean IoU differed by less than
1e-8 after rounding. The one-epoch trial retained the original checkpoint as
best; it is not an accuracy-improvement experiment.

Use different output names for every run. A preparation run with
`--train-scenes 2 --expressions-per-scene 4` followed by training with
`--epochs 1` checks the pipeline; it cannot establish model quality.

```bash
python3 -B -m unittest discover -p 'test_*.py' -v
```
