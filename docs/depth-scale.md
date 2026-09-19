# RGB-dependent MoGe distance correction

This experiment leaves MoGe-2, SAM, SigLIP and the language selector unchanged.
A new small network reads MoGe's frozen global RGB feature and predicts one
positive multiplier for the whole image. For example, 0.7 changes a predicted
distance of 1 metre to 0.7 metres. This is residual-head training, not full MoGe
fine-tuning. On the completed 199-image validation run, mean object-depth error
fell from 70.04 cm to 4.39 cm. This does not measure physical grasp accuracy.

## Prepare features once

### SSD recovery copy (approved exclusion)

Following USB damage, use the checked SSD copy and explicitly exclude the one
unrecoverable training RGB. This leaves 1,879 training images and preserves all
199 validation images and the test split. Original manifests are not edited; the
exclusion is recorded in the new cache and propagated to trained checkpoints.

```bash
# Run from the repository folder.
models/moge/.venv/bin/python -B -m bin_grasp.depth prepare \
  --data-root /path/to/recovered-ocid \
  --exclude-train-scene ARID20/table/top/seq12/rgb/result_2018-08-21-17-01-40.png \
  --output features/depth-scale-ssd-v1
```

Repeat this exact command after interruption. The old USB-based cache is kept
unchanged; use the new cache path when training. Recovered RGB PNGs may have
different file hashes because they were re-encoded from PCD colour pixels.
The USB is not needed for this command. This is dataset recovery, not USB repair.

### Original complete dataset

Connect the OCID USB dataset. Use the existing MoGe environment, not a new install:

```bash
# Run from the repository folder.
models/moge/.venv/bin/python -B -m bin_grasp.depth prepare \
  --output features/depth-scale-v1
```

This deduplicates expressions into 1,880 training images and 199 validation
images, verifies the existing sequence groups are disjoint from each other and
test, and runs frozen MoGe once per RGB image. Test metadata is checked only for
overlap: no test RGB, depth or predictions are used. Depth labels must be allowed
as training supervision under the competition rules.

Only RGB enters MoGe. Training depth supplies a target log multiplier using the
median of valid pixelwise reference/prediction ratios. Missing depth is excluded;
there is no special depth-boundary denoising in this first experiment. Validation
depth is stored only for scoring, never as a feature or a per-image correction
target. Training images need only features/targets on disk; validation keeps
compressed full-resolution predictions and references for exact pixel scoring.
Allow several hundred MB of free space; actual size depends on compression.

If preparation is interrupted, repeat the identical command. Completed image
records are reused after checking source hashes. Changed settings require a new
cache folder. `--limit 4` is available for setup tests, not meaningful evaluation.

## Train the small correction network

```bash
models/moge/.venv/bin/python -B -m bin_grasp.depth train \
  --data features/depth-scale-v1 \
  --epochs 50 --batch-size 64 --lr 0.0003 \
  --output runs/depth-scale-v1
```

This compares unchanged MoGe, one train-fitted constant multiplier, and the
RGB-dependent head. Input standardization is fitted on training images only.
The head has 64 hidden units, a robust log-scale loss, and a multiplier bounded
to exp(-4)..exp(4). It starts with the training-set constant, uses early stopping
after 10 epochs without improvement, and saves the lowest validation object MAE.
If neither learned alternative beats unchanged MoGe, `best.pt` retains identity
correction. Check `best_kind` in the report instead of assuming adaptation won.

Training uses cached features and is small; low GPU use is expected. Epoch times
separate training from full-resolution CPU validation. Increasing VRAM allocation
is not itself a performance goal. `--device cpu` is also supported for the head.
Existing output folders are never overwritten by a new training run.

Outputs: `best.pt`, `history.json`, and `report.json` with all three comparisons,
per-image errors, sequence identifiers, coverage, relative error and delta1.
Scores are equally weighted per image; object regions are the union of annotated
object IDs, not the language-selected target alone. Selection uses unaligned
validation object MAE. No answer-assisted validation rescaling is performed.
The historical test set has already been evaluated for grounding; this command
does not evaluate it or claim that it is untouched.

## Use the saved correction on RGB alone

Replace `/absolute/path/image.png` with the image to process:

```bash
models/moge/.venv/bin/python -B -m bin_grasp.depth predict \
  --head runs/depth-scale-v1/best.pt \
  --image /absolute/path/image.png \
  --output runs/depth-scale-one-image
```

This command never opens depth or labels. It saves predicted camera-frame depth
and XYZ in metres, a validity mask and MoGe's normalized intrinsics. Both depth
and XYZ receive the same multiplier; intrinsics do not change. Camera calibration,
robot transforms and grasp safety remain separate checks. No existing grasp or
grounding pipeline is automatically switched to this experimental head.

## Setup tests

```bash
models/moge/.venv/bin/python -B -m unittest tests.test_depth_scale
```

An answer-assisted scale diagnostic can motivate this experiment but is not its
achieved accuracy. Only improvements on sequence-separated validation without
reference-derived corrections support a useful learned scale predictor.

### Verified setup (16 September 2026)

Six unit tests passed, covering scale targets, missing depth, identity startup,
feature learning, image deduplication and sequence-leakage rejection. A GPU smoke
test extracted four training and four validation images and trained for two
epochs. It retained unchanged MoGe because the tiny adapted run did not beat it;
these are setup checks, not evidence about full-training accuracy. RGB-only
prediction export was checked for consistent depth and XYZ coordinates. The
eight-image cache occupied about 4 MiB. Completed-record reuse was also tested.
