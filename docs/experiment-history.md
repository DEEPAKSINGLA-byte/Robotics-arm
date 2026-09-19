# OCID Phase 1 and Phase 2 Experiment 1

Status: Phase 1 and the Phase 2 Experiment 1 data interface are implemented.
The frozen SigLIP 2 candidate-selection pipeline is now implemented too.
See "Training the candidate selector" below for commands and measured checks.

## Current Files

- `phase1_dataset.py`: creates and validates one ground-truth expression-mask
  pair at a time.
- `phase2_experiment1.py`: creates every perfect candidate mask in a scene for
  oracle-candidate language grounding.
- `verification_results.json`: compact Phase 1 verification results.
- `requirements.txt`: minimal Python packages.

The OCID dataset and expression JSON files remain on the USB drive. They are
not copied into this repository.

## Purpose

Phase 1 creates a trustworthy interface between the raw OCID files and future
RGB-language models. It does not train a segmentation model and it does not
duplicate the dataset. Its output is a Python dataset object that produces one
correct training sample at a time in memory.

The final Phase 1 sample contains:

- RGB image tensor
- Referring expression as text
- Binary target mask
- Bounding box in `[x1, y1, x2, y2]` format
- Global and scene-local object identifiers
- Class, sequence, and source metadata
- Paths to optional depth and point-cloud supervision

## Storage Constraint

The OCID dataset already occupies about 7 GB. The training annotations contain
259,839 expressions, and many expressions describe the same object. Saving a
mask and resized RGB image for every expression would unnecessarily duplicate
many gigabytes of data.

The implementation therefore follows five storage rules:

1. Never copy the OCID RGB, label, depth, or PCD files.
2. Generate each binary target mask in RAM from the integer label PNG.
3. Do not create transformed-image or token caches on disk.
4. Keep only a small, configurable label cache in RAM.
5. Write only optional, small JSON audit summaries.

The Phase 1 code and documentation should remain far below 1 MB. Runtime RAM is
used temporarily and released when the process exits.

The full training JSON is held in memory for fast random access. This consumes
RAM while the loader is running, but it does not consume additional permanent
storage. Start with `num_workers=0`, because every worker may otherwise hold its
own copy of the annotations.

## Source Data

The implementation reads these existing files directly from the USB drive:

```text
train_expressions.json
val_expressions.json
test_expressions.json
OCID-dataset/OCID-dataset/
    ARID10/
    ARID20/
    YCB10/
```

Within each OCID scene, the relevant modalities are:

```text
rgb/    input colour image
label/  integer object-ID image
depth/  depth in millimetres
pcd/    organized point cloud
```

## Step 1: Resolve Annotation Paths

Each OCID-Ref annotation contains a relative RGB path such as:

```text
ARID20/floor/top/seq06/rgb/result_2018-08-20-11-37-31.png
```

The loader replaces only the `rgb` directory component to resolve the other
modalities:

```text
.../rgb/file.png
.../label/file.png
.../depth/file.png
.../pcd/file.pcd
```

Why: deriving the paths avoids a second manifest containing hundreds of
thousands of repeated absolute paths.

## Step 2: Resolve Global and Local IDs

An annotation has two different identifiers:

```text
instance_id        global identifier across the annotation dataset
scene_instance_id  local integer identifier in the current label image
```

For sample `83152`:

```text
instance_id        5527
scene_instance_id  18
```

The binary target mask is therefore generated as:

```python
target_mask = label_image == 18
```

Why: OCID label images contain small local values such as 0 through 20. The
global value 5527 is metadata and cannot appear as a pixel value in this image.

## Step 3: Generate the Mask On Demand

The 16-bit label PNG is opened only when a sample is requested. A Boolean mask
is constructed in memory:

```python
label = np.asarray(Image.open(label_path))
target_mask = label == scene_instance_id
```

No target-mask PNG is written.

Why: multiple expressions can share the same `(scene_path,
scene_instance_id)`. Dynamic generation avoids storing duplicate masks while
remaining exact and deterministic.

## Step 4: Validate the Sample

Before returning a training sample, the loader checks:

- RGB and label files exist.
- RGB and label dimensions match.
- The scene-local ID occurs in the label image.
- The target mask contains at least one pixel.
- The bounding box has positive width and height.
- The bounded box intersects the image.

Why: a broken path or empty target should fail immediately instead of silently
entering training and damaging the loss calculation.

## Step 5: Interpret the Bounding Box

The downloaded OCID-Ref annotations use:

```text
[x1, y1, x2, y2]
```

This was tested on 500 random expressions. Treating the final two values as
width and height placed 325 of 500 boxes outside the image. Agreement with the
tight box derived from the target mask was:

```text
xyxy interpretation: 0.966 mean box IoU
xywh interpretation: 0.088 mean box IoU
```

Why: using the empirically verified coordinate convention prevents incorrect
crops, metrics, and visual grounding targets.

## Step 6: Convert to Training Tensors

The loader returns:

```text
image        float32 tensor [3, H, W], range 0 to 1
target_mask  Boolean tensor [1, H, W]
bbox         float32 tensor [4]
sentence     raw text
metadata     IDs, classes, paths, and sequence information
```

Raw text is returned instead of pre-tokenized files.

Why: tokenization depends on the language encoder chosen in Phase 2. Delaying
it prevents a large, model-specific token cache and keeps Phase 1 reusable.

## Step 7: Optional Resizing

If a fixed image size is requested:

- RGB is resized with bilinear interpolation.
- The binary mask is resized with nearest-neighbour interpolation.
- Bounding-box coordinates are scaled to the new image size.

Why: bilinear interpolation is appropriate for colour images but corrupts
integer masks. Nearest-neighbour interpolation preserves discrete object
membership.

No random horizontal flip is performed. A flip would invert expressions such
as `left of`, `right of`, `top-left`, and `bottom-right` unless the sentence is
also rewritten correctly.

## Step 8: Optional Depth Supervision

The default model input is RGB only. Depth is loaded only when
`return_depth_supervision=True`.

Why: the problem statement requires RGB-only operation. OCID depth may be used
as training supervision or an evaluation reference, but it must not become a
required deployment input. The PCD path is returned as metadata for later 3D
phases without loading the point cloud in Phase 1.

## Step 9: RAM-Only Label Cache

The default cache holds the eight most recently used label arrays in RAM. It
uses an LRU policy and creates no files.

Why: expressions from the same scene often appear near each other. Reusing the
label array reduces USB reads.

Eight 640 by 480 uint16 labels require only about
5 MB per worker.

Set `label_cache_items=0` to disable this cache or reduce worker memory.

## Step 10: Conservative DataLoader

The provided DataLoader defaults to:

```text
batch_size   4
num_workers  0
shuffle      enabled
```

Why: many workers can each create a separate RAM cache and issue competing
random reads to a USB drive. Start with zero workers, measure throughput, and
increase to one or two only if needed.

## Audit Metrics

The audit mode calculates values in RAM and prints a summary:

```text
mask_inside_bbox_ratio
bbox_vs_tight_bbox_iou
failed sample count
unique target count
unique image count
duplicate expression count
count below 0.98 containment
```

It does not save masks, overlays, CSV files, or contact sheets.

The existing 500-sample validation established:

```text
failed samples                 0
missing scene IDs              0
empty masks                    0
mean mask containment          0.9951
minimum mask containment       0.9671
unique targets                 487
unique images                  402
```

These numbers validate the annotation linkage. They are not model scores,
because both the mask and box are ground-truth annotations.

## Verified Implementation Test

The completed loader was tested directly against the files on the USB drive.
For sample `83152`, it produced an RGB tensor of shape `[3, 480, 640]`, a Boolean
mask of shape `[1, 480, 640]`, and found 8,439 target pixels. Mask containment
inside the supplied box was `0.9987` and box-to-tight-mask IoU was `0.9817`.

A fresh 500-sample audit with seed 42 produced:

```text
evaluated samples               500
failed samples                    0
unique targets                  487
unique images                   402
duplicate expression rows        13
mean mask containment        0.9951
minimum mask containment     0.9671
mean box-to-mask IoU          0.9664
```

A two-item resized batch also passed:

```text
image        [2, 3, 384, 512] float32
target_mask  [2, 1, 384, 512] Boolean
bbox         [2, 4] float32
depth        not loaded
PCD          path only
```

The tests produced no processed RGB images, masks, depth copies, PCD copies,
CSV reports, or preview directories.

## Dataset Split Check

The supplied JSON files were also compared by `(scene_path,
scene_instance_id)`. This distinguishes an expression split from an unseen-scene
split:

```text
split   expressions  unique targets  unique images
train       259,839          17,627          2,298
val          18,342          10,267          1,963
test         27,513          12,409          2,020

pair                 shared targets  shared images
train versus val             10,248          1,962
train versus test            12,390          2,020
val versus test               8,018          1,870
```

Interpretation: these are expression-level splits. They test whether the system
can ground held-out descriptions, but they do not test generalization to unseen
images or unseen target objects. This is not necessarily an annotation error;
it is a property of the provided split. Any reported score must be labelled
accordingly. If unseen-scene performance is required, create a grouped split by
scene or sequence before training.

The split comparison is reproduced without writing files:

```text
python3 phase1_dataset.py --check-splits \
  /path/to/annotations/train_expressions.json \
  /path/to/annotations/val_expressions.json \
  /path/to/annotations/test_expressions.json
```

## Running an Inspection

From a terminal:

```text
python3 phase1_dataset.py \
  --annotations /path/to/annotations/train_expressions.json \
  --ocid-root /path/to/OCID-dataset \
  --inspect-sample 83152
```

This prints tensor shapes and metrics but writes no files.

## Running a Storage-Free Audit

```text
python3 phase1_dataset.py \
  --annotations /path/to/annotations/train_expressions.json \
  --ocid-root /path/to/OCID-dataset \
  --audit 500 \
  --seed 42
```

To preserve only a small summary, add:

```text
--summary-json phase1_audit_summary.json
```

## Using the Dataset During Training

```python
from phase1_dataset import OCIDRefDataset, make_dataloader

dataset = OCIDRefDataset(
    "/path/to/annotations/train_expressions.json",
    "/path/to/OCID-dataset",
    image_size=(512, 384),
    return_depth_supervision=False,
    label_cache_items=8,
)

loader = make_dataloader(
    dataset,
    batch_size=4,
    num_workers=0,
)

batch = next(iter(loader))
```

## What Phase 1 Does Not Do

Phase 1 does not:

- Train an object segmentation model.
- Download pretrained model weights.
- Generate language embeddings.
- Save one processed image or mask per expression.
- Use depth as a required inference input.
- Calculate model IoU or grasp success.

Why: these belong to later phases. Keeping the boundary strict makes data
errors distinguishable from model errors.

## Phase 1 Completion Criteria

Phase 1 is complete when:

1. Train, validation, and test JSON files load.
2. Samples resolve directly to their original USB files.
3. Target masks are generated from `scene_instance_id` without disk output.
4. RGB, mask, box, sentence, and metadata are returned consistently.
5. Random audits contain no unexplained missing or empty targets.
6. Batching works at native and resized resolutions.
7. Split overlap is measured before model training.
8. This README matches the implementation.

## Phase 2 Experiments

Phase 2 is divided into two experiments:

```text
Oracle experiment:
ground-truth candidate masks + expression -> select target mask

Realistic experiment:
RGB -> predicted candidate masks
predicted masks + expression -> select target mask
```

This separation reveals whether an error comes from visual segmentation or
language grounding.

## Phase 2 Experiment 1: Oracle-Candidate Grounding

`phase2_experiment1.py` implements the storage-efficient data interface for the
first experiment:

```text
RGB image + referring expression + every ground-truth candidate mask
                              |
                              v
                  grounding model selects one candidate
                              |
                              v
             compare selected index with target supervision
```

For each expression, the code opens the original integer label PNG and obtains
all foreground IDs except background ID 0. It creates one Boolean candidate
mask per ID in RAM. No candidate-mask files are saved.

The returned `model_inputs` are:

```text
image                 RGB tensor [3, H, W]
sentence              the referring expression
candidate_masks       Boolean tensor [N, H, W]
candidate_boxes       pixel boxes [N, 4]
candidate_geometry    normalized box, centre, and area [N, 7]
```

The true `scene_instance_id`, target index, target class, and target mask are
not model inputs. They are kept separately under `supervision` or
`debug_metadata`. During training, the model predicts one of the N candidate
positions and the target candidate index is used only to calculate the loss.

This separation prevents the correct answer from leaking into the model. The
candidate ID values and ground-truth class names are provided only as debugging
metadata and must not be passed to the grounding model.

Run a single-sample check:

```text
# Run from the repository folder.
python3 phase2_experiment1.py \
  --annotations /path/to/annotations/train_expressions.json \
  --ocid-root /path/to/OCID-dataset \
  --inspect-sample 83152
```

Run a RAM-only interface audit:

```text
python3 phase2_experiment1.py \
  --annotations /path/to/annotations/train_expressions.json \
  --ocid-root /path/to/OCID-dataset \
  --audit 100 \
  --seed 42
```

The audit verifies that every target exists among the candidates, the target
index selects the exact Phase 1 mask, and target supervision does not appear in
`model_inputs`.

This interface audit measures data correctness. Actual candidate-selection
training and evaluation now use the scripts described below.

## Phase 2 Experiment 2: Later Work

Experiment 2 will replace the perfect label-derived candidates with masks
predicted from RGB. It should be started only after Experiment 1 has a reliable
grounding score, so segmentation failures can be measured separately from
language-grounding failures.

Update: the first RGB-only predicted-mask diagnostic is now implemented in
`run_experiment2.py`. See "Experiment 2: automatic masks" at the end of this README.

## Maintenance Rule

Whenever Phase 1 or the Experiment 1 interface changes:

1. Update this Markdown explanation.
2. Run the sample inspection and audit.

This keeps the explanation synchronized with the code without storing large
intermediate artifacts.
## Training the candidate selector

We use the downloaded `google/siglip2-base-patch16-224` model to describe images
and sentences with 768 numbers each. Its weights stay frozen: training changes
only our new, smaller candidate-selection model.

The new files are:

| File | What it does |
| --- | --- |
| `prepare_splits.py` | Separates examples for training and evaluation |
| `extract_siglip_features.py` | Reads USB images and saves compact frozen features |
| `grounding_data.py` | Loads features and handles different candidate counts |
| `grounding_model.py` | Compares objects with the sentence and scores candidates |
| `train_experiment1.py` | Runs the small learning check or normal training |
| `evaluate_experiment1.py` | Measures accuracy and saves candidate predictions |
| `test_grounding.py` | Checks padding, order independence, and split safety |
| `run_stage1.py` | Runs full extraction and training with one command |
| `run_regularization_study.py` | Compares fresh baseline, stronger dropout, and narrower selector |

### What the model sees

For each object, we hide pixels outside its mask with gray, crop to its box,
and pad to a square to preserve its shape. The SigLIP processor resizes this to
224 by 224. The full RGB image is also encoded once to provide scene context.
All crops are temporary RAM objects; no cropped images are saved.

The sentence is encoded with a maximum of 64 tokens, matching the encoder's
text context. Longer sentences are truncated, and the extraction report counts
them. Features are normalized and saved in float16 to reduce storage.

The candidate selector receives object features, sentence features, full-scene
features, and the seven box/centre/area values from the existing interface.
By default, it projects these to 256 numbers and compares them with a two-layer transformer
using four attention heads. It scores each object, with a learned contribution
from the original SigLIP image/text similarity. Highest score wins.

Candidate order carries no learned position label. Empty padding positions are
ignored. Target IDs and class names stay in supervision/evaluation metadata;
they never enter the model's forward function. The selected index can retrieve
the corresponding mask from the original Experiment 1 candidate list.

The whole-scene feature provides context, but it does not guarantee reliable
"behind" reasoning. This remains an experimental model, especially for depth
and occlusion relationships. It uses no ground-truth depth input.

### What is the relation transformer, in plain language?

It is the small part of our model that compares the objects with each other
while taking the sentence into account. Consider: "the box to the left of the
sponge", with two boxes and a sponge in the image.

1. Frozen SigLIP turns each object's appearance and the sentence into numbers.
2. We add the object's position and size. We also provide the full-image feature.
3. The relation transformer exchanges information between the candidate objects,
   the sentence, and the scene. Its attention mechanism learns which other
   information helps score each candidate.
4. The scoring layer produces one score per candidate. The highest-scoring
   candidate supplies the selected mask.

This makes it possible to distinguish boxes using their surroundings, rather
than relying only on how much each crop resembles the words "box". It is not
a hand-written rule that explicitly finds the sponge and checks left/right:
we train it using the correct candidate, and it learns useful comparisons.
Its predictions can still be wrong. Attention alone is not proof of correct
spatial reasoning, and depth-related descriptions remain difficult.

The relation transformer and its input/scoring layers are trained; SigLIP
stays unchanged. This stage is training a new selector on frozen features,
not fine-tuning SigLIP itself.

### Storage and installation

The earlier RAM-only storage rules apply to Phase 1 and candidate masks.
Stage 1 additionally saves compact features in `features/` and small trained
checkpoints in `runs/`. Each scene's candidate features are reused across
expressions. Text features cost about 1.5 KB per expression (roughly 470 MB for
all 305,694 supplied expressions), plus scene features and JSON metadata.
Separate experimental indexes can duplicate text features; reuse the shared
scene cache and check disk space before full extraction.

Run commands from the project directory:

```bash
# Run from the repository folder.
python3 -m pip install -r requirements.txt
```

Your existing CUDA-enabled PyTorch is sufficient. The code was tested with the
installed dependencies; their versions are recorded in the verification report.
Models load only from local files, so extraction needs no network connection.
The code accepts both the tensor output in Transformers 4 and the structured
`pooler_output` returned by the installed Transformers 5 version.

### 1. Prepare the two experiments

```bash
python3 prepare_splits.py --annotations-dir /path/to/annotations \
  --mode official --output splits/official
python3 prepare_splits.py --annotations-dir /path/to/annotations \
  --mode sequence --output splits/sequence
```

The official experiment keeps the supplied expression splits unchanged.
The separate sequence experiment pools their annotations and partitions whole
acquisition groups, about 80/10/10 percent of groups. A group is collection
(ARID10/ARID20/YCB10) plus sequence folder. Different camera views of a sequence
stay together. This is conservative and may hold out more than necessary.
Percentages of expressions can differ because groups have different sizes.
Every manifest records its seed and fingerprint for reproducibility.

Train a separate model for each experiment. Evaluating the official model on
the sequence test split would not be a valid unseen-sequence experiment.
Sequence separation also does not establish unseen physical object identities.
Keep the sequence test set untouched until training choices are finalized.

### 2. Extract features for the 100-example learning check

```bash
python3 extract_siglip_features.py --manifest splits/sequence/train.json \
  --ocid-root /path/to/OCID-dataset \
  --output features --limit 100 --seed 42
```

The command prints a `Feature index:` path. Use that exact JSON path below.
It is namespaced by the encoder hash and extraction settings so incompatible
features cannot be silently mixed. It also reports GPU memory and truncated
sentences. The default image/text chunk size is eight; use `--chunk-size 4`
if other applications are using substantial GPU memory.

Completed scenes are reused after interruption. An interrupted text pass is
recomputed. Keep the source dataset unchanged while using its cache; delete or
choose a new feature output directory explicitly if source images change.

### 3. Check whether the model can learn those examples

Replace `/absolute/path/to/feature-index.json` with the printed path:

```bash
python3 train_experiment1.py \
  --train-features /absolute/path/to/feature-index.json \
  --overfit --epochs 200 --lr 0.001 --output runs/overfit100
```

This deliberately trains and evaluates on the SAME 100 training examples.
Reaching 99% shows that the pipeline can learn this small set; it does not tell
us accuracy on new expressions or scenes. A failed check exits with an error.
The script saves a checkpoint containing only the small selector, not another
copy of SigLIP. Existing run directories are never overwritten.

### 4. Train with separate validation data

The simplest command, after the 100-example check passes, is:

```bash
# Run from the repository folder.
python3 run_stage1.py --mode sequence --output runs/sequence-stage1
```

This extracts every training and validation expression and trains up to 30
epochs. It can take substantially longer than the small learning check.
It prints progress, reuses completed features, and leaves test evaluation for
later. Add `--evaluate-test` only when your experimental choices are settled.
The USB drive must stay connected during feature extraction. After extraction,
training reads only the local feature cache.

To run the individual steps instead:

Extract all training and validation expressions by omitting `--limit`:

```bash
python3 extract_siglip_features.py --manifest splits/sequence/train.json \
  --ocid-root /path/to/OCID-dataset --output features
python3 extract_siglip_features.py --manifest splits/sequence/val.json \
  --ocid-root /path/to/OCID-dataset --output features
```

Use the two newly printed index paths:

```bash
python3 train_experiment1.py --train-features /absolute/path/to/train-index.json \
  --val-features /absolute/path/to/val-index.json \
  --epochs 30 --batch-size 256 --output runs/sequence-stage1
```

Cross-entropy loss teaches the model to give the correct candidate the highest
score. The script measures validation accuracy each epoch, keeps `best.pt`,
and stops after five epochs without improvement. `history.json` records each
epoch; `best_metrics.json` records the selected checkpoint's validation results.
The test set is not used to select a checkpoint.

### 5. Evaluate once training choices are settled

Extract the test split with the same feature command and `splits/sequence/test.json`.
Then use its printed feature index:

```bash
python3 evaluate_experiment1.py --features /absolute/path/to/test-index.json \
  --checkpoint runs/sequence-stage1/best.pt --output runs/sequence-test.json
```

For a useful comparison, score the same candidates using frozen SigLIP
similarity alone:

```bash
python3 evaluate_experiment1.py --features /absolute/path/to/test-index.json \
  --zero-shot --output runs/sequence-test-similarity.json
```

Repeat steps 4 and 5 with `splits/official/` and a separate run directory for
the official expression experiment. The evaluator refuses mismatched encoders,
split fingerprints, and sequence leakage.

### Reading the results

Accuracy is the fraction of expressions selecting the correct candidate.
Reports include counts alongside every percentage, random-choice expected
accuracy, and results by candidate count and relationship keywords. Sentences
can belong to several relation groups. These are simple text categories, not
expert-labelled relationship difficulty classes.

The duplicate-class subgroup counts cases where the annotations in that feature
index confirm two candidates of the target class. Some candidate classes are
unknown, especially in small subsets, so this subgroup can miss duplicates.
`class_catalog_complete_samples` reports how many samples have complete class
coverage. Class metadata is used only for reporting.

Because ground-truth candidate masks do not overlap, a correct selection has
mask IoU 1 and a wrong selection has IoU 0. Thus mean selected-mask IoU equals
selection accuracy in this oracle experiment; it is not an independent measure
of a segmentation model. The predictions JSONL records selected candidate IDs
for inspection or retrieval through the existing mask loader.

### GitHub contents

Commit source code, this README, requirements, tests, and the small verification
summary. `.gitignore` excludes downloaded weights, generated split manifests,
features, runs, checkpoints, dataset copies, environments, secrets, and Python
temporary files. Files already tracked by Git require removing them from its
index; ignoring a filename alone does not untrack it.

### Checks completed on this computer

See `training_verification.json` for the small, shareable record.

- Five code tests passed: padding, candidate order, input/answer separation,
  disjoint sequence grouping, and rejection of incompatible evaluation splits.
- Real extraction succeeded for 100 training expressions (96 scenes) and
  100 validation expressions (61 scenes). No sentences were truncated.
- Peak allocated GPU memory during this extraction was about 0.75 GiB.
  This is PyTorch's allocation measurement; total GPU use can be higher.
- The 2,304,514-parameter selector reached 99/100 correct on its own 100
  training examples after 18 epochs. This passes the intended learning check.
- A separate two-epoch trial using only those 100 training examples obtained
  26/100 on the 100 validation examples. Reloading the saved checkpoint
  reproduced 26/100. Frozen SigLIP similarity alone scored 44/100 there.
  This short trial verifies the workflow and demonstrates no improvement
  over the similarity baseline. This was a workflow check, not the full run.
- Full sequence training has since been completed; see the regularization study
  below. Test-set evaluation has not been run.

The README's first 100-example command has already been run in
`runs/overfit100`; use a new run name to repeat it. Full training features are
already available locally; see the commands below to reuse them.

### References

- [Google SigLIP 2 checkpoint](https://huggingface.co/google/siglip2-base-patch16-224)
- [Transformers SigLIP usage and padding](https://huggingface.co/docs/transformers/model_doc/siglip)

## Faster training and visible progress

The original training loop loaded individual scene files repeatedly from disk:
only 256 scenes could remain in its cache, while shuffled training uses 1,880
scenes. Batch size 32 also gave the GPU little work per update. The new loop
loads each scene once and gathers full batches using tensor operations.

By default, compact scene tables and all sentence features stay on the GPU if
there is enough free memory with a 3 GiB training headroom check. The measured
training table uses 0.87 GiB and validation uses 0.08 GiB. Otherwise `auto`
keeps the feature tables in ordinary RAM. Use `--feature-storage cpu` to force
the RAM option, or `--feature-storage cuda` to require GPU storage.

The default training batch size is now 256. BF16 mixed precision is enabled
automatically on a supported CUDA GPU, and AdamW uses its fused CUDA update.
This reduces computation time while keeping model weights in float32. Use
`--precision fp32` for the previous numerical precision. These performance
changes retain the architecture, feature cache and dataset split. Larger batches
mean fewer optimizer updates per epoch, so the learning trajectory can change.
Compare validation accuracy, not only elapsed time.

Progress now prints the epoch start, first batch, every 50 batches, and last
batch. Each line includes loss, examples per second, elapsed time, and estimated
remaining training time. `--log-every 100` prints less frequently. The first
batch's estimate can be pessimistic because the GPU is warming up.

Each completed epoch prints training time, validation time, their combined
time, both accuracies, and peak allocated GPU memory. These measurements also
go into `history.json`. Epoch time excludes initial loading/baseline measurement
and checkpoint writing; loading time is recorded separately in `settings.json`.

### Measured on the RTX 4060 Laptop GPU

Short benchmark (warmup excluded; not a full training quality comparison):

| Loading and batch | Precision | Examples/second |
| --- | --- | ---: |
| Original disk loading, 32 | FP32 | 2,533 |
| Resident feature tables, 32 | FP32 | 8,774 |
| Resident feature tables, 256 | BF16 | 29,301 |
| Resident feature tables, 512 | BF16 | 27,585 |
| Resident feature tables, 1,024 | BF16 | 25,315 |

Batch 256 was fastest in this short comparison. Increasing the batch consumed
more VRAM but did not increase throughput. GPU utilization cannot be fixed at
70% by a training option; it varies with workload, thermals and other programs.
The aim is completing useful training faster, not filling memory unnecessarily.

One complete verification epoch used all 257,445 training expressions and
24,546 validation expressions, starting from the user's saved epoch-one model:

- Preloading: 1.4 seconds.
- Training: about 9.0 seconds (28,674 expressions/second).
- Validation: about 0.4 seconds.
- Total epoch: 9.4 seconds.
- Peak PyTorch allocated GPU memory: 1.205 GiB.
- Device samples during the run reached 100% utilization and about 1,431 MiB
  total device memory. These are point samples, not an average utilization.
- Validation accuracy: 78.48%, versus 76.73% for the initial weights evaluated
  in BF16 (the earlier FP32 result was 76.74%).

The checks used no test examples. Six code tests passed, including a new test
that resident batches preserve candidate features, target indexes, metadata,
padding, and the last partial batch. The old checkpoints and run directories
were kept. `performance_verification.json` records the measured results.

### Continue training from the checked weights

Your features are already extracted; this command reuses them:

```bash
# Run from the repository folder.
python3 run_stage1.py --mode sequence --output runs/sequence-stage1-fast \
  --epochs 30 --batch-size 256 --log-every 50 \
  --init-checkpoint runs/speed-check/best.pt
```

`--init-checkpoint` reuses learned model weights and starts a new optimizer
and epoch count. It is a warm start, not an exact resume of the old optimizer.
The initial weights are also saved as the new run's best model before updates,
so a run that does not improve validation will keep its starting model.
You can instead use `runs/sequence-stage1/best.pt` to start directly from the
checkpoint you saved before stopping. Omit `--init-checkpoint` for a new model.

To skip even the cached-feature checks and start training directly:

```bash
python3 train_experiment1.py \
  --train-features features/4764b544cec72bc7/sequence-train-bd9c96bbe427.json \
  --val-features features/4764b544cec72bc7/sequence-val-5f243509242f.json \
  --output runs/sequence-stage1-fast --epochs 30 --batch-size 256 \
  --init-checkpoint runs/speed-check/best.pt
```

Choose one of these commands. An existing output directory is never overwritten.
If GPU memory is tight, reduce `--batch-size` or use `--feature-storage cpu`.
All features remain local and compatible with the original frozen SigLIP cache.

Performance implementation references:
[PyTorch performance tuning](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html),
[PyTorch mixed precision](https://docs.pytorch.org/docs/stable/amp.html).

## Checking overfitting: controlled regularization study

The earlier warm-started run in `runs/sequence-stage1-fast` peaked at 79.20%
validation accuracy. By epoch seven, online training accuracy reached 96.27%
while validation accuracy was 77.87%. This is evidence of overfitting: later
training improves performance on familiar examples without improving the
held-out scenes. Early stopping preserved the best checkpoint, not the last one.
These are validation results, not results on the untouched test set.

We test two possible ways to reduce this problem separately:

- Stronger dropout: increase the probability from 0.10 to 0.30. During training,
  dropout temporarily hides some internal signals so the model cannot always
  depend on the same clues. It does not delete dataset examples or candidate
  objects, and it is switched off during evaluation.
- Narrower model: reduce the internal width from 256 to 128 numbers, keeping
  two layers and dropout 0.10. Fewer learned parameters may reduce memorization,
  but can also reduce the model's ability to solve the task.

Neither change guarantees improvement. A fresh default model is trained too,
so a change in initialization/training history is not mistaken for a benefit
from regularization. All three trials use seed 42, the same train/validation
features, batch 256, BF16, learning rate 0.0003, AdamW weight decay 0.01,
at most 30 epochs, and early stopping after five non-improving epochs.
The narrower model has a different parameter shape, so the same seed does not
mean identical initial weights. This is a single-seed exploratory comparison.

Each trial selects its checkpoint by validation accuracy. We then evaluate
that checkpoint on the training set with dropout disabled. This gives a more
comparable train/validation gap than the online training number, which mixes
changing weights and active dropout during an epoch. A smaller gap by itself
is not a win if validation accuracy also falls.

To repeat all three trials, choose a NEW output directory:

```bash
# Run from the repository folder.
python3 run_regularization_study.py \
  --train-features features/4764b544cec72bc7/sequence-train-bd9c96bbe427.json \
  --val-features features/4764b544cec72bc7/sequence-val-5f243509242f.json \
  --output runs/regularization-study-repeat
```

To configure one fresh run, `train_experiment1.py` and `run_stage1.py` now accept
`--hidden 128`, `--layers 2`, `--dropout 0.3`, and `--report-training-fit`.
Use one change at a time for an interpretable comparison. Omit
`--init-checkpoint` when changing the architecture or dropout; incompatible
overrides are rejected. The defaults remain width 256, two layers, dropout 0.10.

The study writes `comparison.json`, plus each trial's `settings.json`,
`history.json`, `best.pt`, `best_metrics.json`, `best_training_metrics.json`,
and `summary.json`. Existing runs and weights are preserved. The test split
is not used for this study and must not be used to choose these settings.

### Measured study results (seed 42)

All training accuracies in this table are evaluations of the selected best
checkpoint, with dropout disabled. The gap is training accuracy minus validation
accuracy, in percentage points. Epoch time is the mean over completed epochs.

| Fresh trial | Parameters | Training | Validation | Gap | Best epoch | Seconds/epoch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Default: width 256, dropout 0.10 | 2,304,514 | 96.00% | 81.72% | 14.28 | 7 | 9.52 |
| Stronger dropout: width 256, dropout 0.30 | 2,304,514 | 97.38% | 79.54% | 17.84 | 11 | 9.56 |
| Narrower: width 128, dropout 0.10 | 726,274 | 95.97% | 79.95% | 16.02 | 10 | 4.90 |

The trials stopped after 12, 16, and 15 epochs respectively. Frozen SigLIP
similarity without the trained selector scored 43.59% on this validation set.
The default architecture won this comparison. Stronger dropout did not improve
validation accuracy or the measured gap; the narrower model was about twice as
fast per epoch but lost 1.77 percentage points of validation accuracy.

Keep the default architecture for now. The best checkpoint from this study is:

```text
runs/regularization-study/baseline/best.pt
```

Its 81.72% exceeds the earlier warm-started run's 79.20%, but that is not evidence
that stronger regularization helped: the winning model uses the original
architecture and dropout. The train/validation gap remains substantial, so
overfitting has not been solved. This single seed does not establish a reliable
ranking across repeated runs. The many expressions share only 1,880 training
images; 257,445 expressions are not 257,445 independent visual scenes.

Seven code tests passed, including configuration checks and rejection of
incompatible checkpoint overrides. Reloading the winning checkpoint through the
standalone evaluator reproduced exactly 20,058 / 24,546 correct (81.72%).
Downloaded weights, feature caches, and all
study runs remain excluded from Git. A small shareable results record is saved
in `regularization_verification.json`.

## Visually reviewing 50 wrong predictions

The review page has been generated at:

```text
outputs/ocid-error-review/index.html
```

Open this HTML file in a browser. It contains all 50 original images, so the
USB drive is not needed after generation. The page starts with the sentence and
unmarked image. Click **Reveal answers** to see green outlines for the correct
object and red outlines for the model's choice, plus close-ups of both objects.
Use **Next**, **Previous**, or the example selector to move between cases.

Choose any error labels that fit, write a short note, and tick **I have reviewed
this example**. Notes are stored locally in that browser when storage is available.
Use **Download my notes** before closing or switching browsers; **Load saved notes**
restores a downloaded JSON file for the same report. Notes are not automatically
written back to this repository or sent to any service. Review totals count only
examples marked reviewed, and categories can overlap.

The baseline checkpoint made 4,488 mistakes across 160 distinct validation
images. This page uses one randomly chosen mistake per selected image, with seed
42, balanced across the seven held-out sequence groups (7 or 8 images per group).
The 50 images are distinct, but images in a sequence are related. This is a
diversity-focused inspection set, not a representative estimate of overall error
frequencies. The test split is not accessed.

The small `selection.json` beside the page records the selected expression keys,
scene paths, correct/predicted IDs, group counts, and prediction-file hash.
Keys beginning with `train:` refer to the original annotation source; the review
uses the regrouped **sequence validation** split, not training examples.

To generate another report, with the USB drive connected, choose a fresh output:

```bash
# Run from the repository folder.
python3 build_error_review.py \
  --features features/4764b544cec72bc7/sequence-val-5f243509242f.json \
  --predictions runs/regularization-study/baseline/reloaded-validation.predictions.jsonl \
  --ocid-root /path/to/OCID-dataset \
  --output runs/error-review-50 --count 50 --seed 42
```

This needs no GPU or model inference. It validates prediction coverage, split,
target IDs, candidate-index mapping, image dimensions, and nonempty masks. The
original images and labels are never changed. Only the selected 50 RGB images
are embedded unmodified in the output; mask boundaries are drawn in the browser.
Generated reports under `runs/` are ignored by Git. Commit the generator,
`error_review_template.html`, and `test_error_review.py`, not dataset images or
personal review notes. Three new tests cover repeatable distinct-scene selection,
balanced groups, invalid inputs, and exact mask boundaries.

## Experiment 2: automatic masks

`run_experiment2.py` connects the local SAM 2.1 Tiny model to frozen SigLIP and
the existing relation selector. It performs inference only; it does not update
any model weights. The dataset's perfect masks are no longer prediction inputs.

### How this works

1. Choose 20 different sequence-validation images, balanced across the seven
   groups, using seed 42. Choose one random sentence per image. Selection does
   not depend on whether the old model got the answer right or wrong.
2. SAM receives each RGB image and creates candidate masks. It receives no
   target IDs, labels, classes, depth, or manually supplied target prompts.
3. Filter the predicted masks using fixed size and near-duplicate rules.
4. Derive each mask's crop, box, centre and area. Use the SAME grey-background,
   square-padded crop recipe as selector training. Recompute SigLIP features:
   the old perfect-mask candidate features cannot represent these new masks.
5. The frozen sentence/image encoders and trained selector choose one mask.
   Save `predictions.json` BEFORE opening any ground-truth label PNG.
6. Only then open labels and calculate selected-mask overlap and whether a
   better target mask existed among the proposals.

SAM runs over the selected images first, then is removed from GPU memory before
loading SigLIP. This limits memory use; reported per-image times exclude model
loading, and the demo is a two-pass batch workflow, not a measured streaming
robot system. No network is needed once the models and USB dataset are present.

### Run the demo

From the repository, choose a fresh output directory:

```bash
# Run from the repository folder.
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 run_experiment2.py \
  --count 20 --output runs/experiment2-sam21-tiny-repeat
```

Use `--count 1` for a quick check. The USB dataset must be connected while
generating the demo. The generated `report.html` embeds the selected RGB images,
so reviewing it afterward needs no USB drive, GPU, or internet connection.

Defaults load:

- `models/sam2.1-hiera-tiny`
- `models/siglip2-base-patch16-224`
- `runs/regularization-study/baseline/best.pt`
- `features/4764b544cec72bc7/sequence-val-5f243509242f.json` for validation
  membership, sentences and training-recipe verification, not old visual features.

The installed environment already supports this workflow. The tested package
versions are recorded in `requirements-experiment2.txt` and each run's settings.
Do not unnecessarily replace a working CUDA installation. The SAM wrapper casts
mask-quality scores and boxes to matching float32 before TorchVision duplicate
removal; this avoids a BF16 dtype mismatch in the installed Transformers pipeline.
Installed libraries and the downloaded weights are not modified.

### Initial settings (not tuned to these 20 answers)

- One image at a time, a 32-by-32 point grid, 16 points per GPU batch.
- No extra crop layers; SAM predicted-quality threshold 0.80, stability 0.95,
  built-in box NMS threshold 0.70.
- Remove predicted masks smaller than 32 pixels or larger than 95% of the image.
- Remove near-identical retained masks at mask IoU >= 0.95, keeping the higher
  SAM-quality score first. Other overlaps remain possible.
- Keep at most 128 candidates, ranked by SAM quality, not by the correct answer.
- SAM uses float32 weights with BF16 autocast; SigLIP uses float16; selector uses
  BF16 autocast. SigLIP outputs are rounded to float16, matching the training cache.

The background is NOT removed using the ground-truth labels. SAM may still
produce table/background regions, object parts, or merged objects. The size and
duplicate filters are starting heuristics, not proof that every retained mask
is an object. The script records proposals before and after our filters so
their effect on target coverage can be checked. "Raw" here means the masks
returned by SAM's automatic pipeline, after SAM's own internal filtering.

### Reading the results

IoU measures how much the selected mask overlaps the correct target mask. It
is not the same as exact candidate-index accuracy with perfect masks. We use
IoU >= 0.50 as an explicit diagnostic threshold, NOT a PDF-specified requirement.

- **Proposal available:** at least one retained mask reaches that target IoU.
- **Success:** the mask actually selected reaches that IoU.
- **Proposal miss:** no retained mask reaches it. Selection alone cannot fix this.
- **Selection miss:** a usable mask exists, but the selector chooses another.
- **Best available IoU:** choose the proposal with the highest true target overlap
  for diagnosis only. This is an oracle upper bound, not a deployable prediction.
- **Conditional selection success:** successes divided by examples with a usable
  proposal. Undefined if no usable proposals exist.
- **Frozen similarity:** selects among the SAME predicted masks without the
  relation selector, as a comparison.

An empty proposal set counts as a failure; it is never silently skipped.
Packed masks in `proposals/` and source image hashes permit rechecking the
results without rerunning SAM. Predicted mask indexes are unrelated to original
dataset instance IDs. Reported timings include per-image reads/processing but
exclude model initialization and the final HTML report generation.

The 20-image, 20-expression balanced sample is a workflow check, not a reliable
full-dataset accuracy estimate. Do not compare its success percentage directly
with 81.72% on all 24,546 validation expressions with perfect masks. No test
data is evaluated. This step still does not implement 3D reasoning or grasping.

### Files saved

Each fresh run contains `settings.json`, `selection.json`, packed predicted masks,
`proposal_progress.json`, `predictions.json` (before labels), `evaluation.json`,
`summary.json`, and `report.html`. All run artifacts remain ignored by Git.
The report lets you inspect the original image, all proposals, the chosen mask,
the best available mask, and each candidate individually.

Tests cover distinct-image selection, checkpoint/split guards, preprocessing
geometry, duplicate filtering, empty proposals, evaluation failure attribution,
and the mixed-precision cleanup fix. Source model documentation:
[SAM 2.1 Tiny](https://huggingface.co/facebook/sam2.1-hiera-tiny),
[SAM 2 automatic mask generation](https://github.com/facebookresearch/sam2#image-prediction).

### First measured result: 20-image diagnostic

Completed run: `runs/experiment2-sam21-tiny-20`.

| Measurement | Result |
| --- | ---: |
| Images / expressions checked | 20 / 20 |
| Usable target proposal before our extra filters (IoU >= 0.50) | 15 / 20 |
| Usable target proposal after our filters | 15 / 20 |
| Selector chose a mask with IoU >= 0.50 | 11 / 20 |
| Missing adequate target proposal | 5 / 20 |
| Adequate proposal existed, but selector missed it | 4 / 20 |
| Frozen SigLIP similarity passed on the same predicted proposals | 6 / 20 |
| Mean selected-mask IoU | 0.508 |
| Mean best-available-mask IoU (oracle diagnostic) | 0.697 |
| Mean retained candidates per image | 24.15 |
| Mean SAM + filtering time per image | 2.210 seconds |
| Mean encoding + selection time per image | 0.094 seconds |
| Peak allocated GPU memory during per-image processing | 0.823 GiB |

On these exact same 20 image/sentence pairs, the earlier saved perfect-mask
predictions were correct in 18 cases. This is a more relevant small-sample
comparison than directly comparing 11/20 with the full-validation 81.72%.
Neither comparison establishes generalization performance from only 20 examples.

The pipeline works end to end for RGB-language mask selection, but automatically
generated masks remain a clear limitation. They can include background texture
and object fragments. The first report includes a case where a printed region
of the floor mat was selected and the requested box had no usable proposal.
Do not interpret a selected SAM mask as proof of a physical graspable object.

Six new tests and all ten existing tests passed (16 total). An independent
read-only audit reconstructed every saved mask, reproduced all per-candidate
IoUs, checked source image hashes and confirmed unchanged SAM/selector weight
hashes. See `experiment2_verification.json` for the compact record. Two initial
failed compatibility smoke runs were preserved; `experiment2-sam21-tiny-smoke-v3`
and the twenty-image run completed. Old runs remain untouched.

The user-facing offline report is also saved at:

```text
outputs/experiment2-sam21-tiny/report.html
```
