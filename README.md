# OCID Phase 1 and Phase 2 Experiment 1

Status: Phase 1 and the Phase 2 Experiment 1 data interface are implemented
and tested directly against the OCID files on the USB drive.

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
  /media/deepak/5415-7117/train_expressions.json \
  /media/deepak/5415-7117/val_expressions.json \
  /media/deepak/5415-7117/test_expressions.json
```

## Running an Inspection

From a terminal:

```text
python3 phase1_dataset.py \
  --annotations /media/deepak/5415-7117/train_expressions.json \
  --ocid-root /media/deepak/5415-7117/OCID-dataset/OCID-dataset \
  --inspect-sample 83152
```

This prints tensor shapes and metrics but writes no files.

## Running a Storage-Free Audit

```text
python3 phase1_dataset.py \
  --annotations /media/deepak/5415-7117/train_expressions.json \
  --ocid-root /media/deepak/5415-7117/OCID-dataset/OCID-dataset \
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
    "/media/deepak/5415-7117/train_expressions.json",
    "/media/deepak/5415-7117/OCID-dataset/OCID-dataset",
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
cd /home/deepak/OCID_Phase1
python3 phase2_experiment1.py \
  --annotations /media/deepak/5415-7117/train_expressions.json \
  --ocid-root /media/deepak/5415-7117/OCID-dataset/OCID-dataset \
  --inspect-sample 83152
```

Run a RAM-only interface audit:

```text
python3 phase2_experiment1.py \
  --annotations /media/deepak/5415-7117/train_expressions.json \
  --ocid-root /media/deepak/5415-7117/OCID-dataset/OCID-dataset \
  --audit 100 \
  --seed 42
```

The audit verifies that every target exists among the candidates, the target
index selects the exact Phase 1 mask, and target supervision does not appear in
`model_inputs`.

Important: this code prepares and validates Experiment 1, but it does not yet
claim language-grounding accuracy. A grounding model must be implemented,
trained using `model_inputs`, and evaluated against
`supervision["target_candidate_index"]` before such a score exists.

## Phase 2 Experiment 2: Later Work

Experiment 2 will replace the perfect label-derived candidates with masks
predicted from RGB. It should be started only after Experiment 1 has a reliable
grounding score, so segmentation failures can be measured separately from
language-grounding failures.

## Maintenance Rule

Whenever Phase 1 or the Experiment 1 interface changes:

1. Update this Markdown explanation.
2. Run the sample inspection and audit.

This keeps the explanation synchronized with the code without storing large
intermediate artifacts.
# Robotics-arm
