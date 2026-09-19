# Final full test evaluation

The chosen configuration is fixed before test scoring:

- SAM 2.1 Tiny, frozen; 32 sampling points per side, no crop layers.
- Mask-quality threshold 0.7; stability threshold 0.9.
- Existing candidate filters: area >= 32 pixels, area fraction <= 0.95,
  mask duplicate IoU 0.95, maximum 128 candidates.
- Frozen SigLIP 2 Base, with the unchanged masked-gray-square crop recipe.
- Selector from `runs/sam-adaptation-relaxed-300/best.pt` (73/99 validation).
- Success threshold: selected mask IoU >= 0.5.

No model, threshold or checkpoint is selected using this evaluation. After
viewing the result, do not tune against these test answers and continue to
describe later results as an untouched test evaluation.

## Coverage and separation

The evaluator processes **all 23,703 expressions on all 220 test images**,
covering seven sequence groups. It generates SAM proposals and image features
once per image, then scores every expression. This is not one expression per
image as in the earlier 99-image validation diagnostic.

Before inference it verifies the full train/validation/test manifests against
their saved fingerprint, checks disjoint keys, images and sequence groups, and
checks that the checkpoint's training groups belong only to the training split.
Existing validation-only scripts continue to reject test data.

## Run

The first authorized full run is `runs/final-test-relaxed`. Do not rerun to pick
a better model or seed. For setup inspection without GPU inference or writing
outputs, use a new nonexistent output name and `--check-only`:

```bash
# Run from the repository folder.
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 -B evaluate_final_test.py \
  --output runs/final-test-preflight --check-only
```

The evaluation command itself is:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 -B evaluate_final_test.py \
  --output runs/final-test-relaxed
```

Existing output folders are rejected. A failed infrastructure run is incomplete,
not a score to compare with successful runs; keep its artifacts for diagnosis.
This implementation does not resume partial runs.

## Audit trail

`evaluation_plan.json` is saved before inference, alongside a byte-identical
copy of the chosen selector. It records settings, precision, hashes and the
full evaluation scope. All RGB-only predictions are committed in
`predictions.json` **before label images are opened**. Ground-truth masks supply
evaluation answers only, never candidate crops, prompts, filters or features.

The run saves compact proposals, frozen image/text features, per-expression
results, and a summary. `completed.json` marks successful completion. Model
hashes are checked again afterward. These generated artifacts are Git-ignored.

## Metrics

- `accuracy`: fraction of all test expressions with selected-mask IoU >= 0.5.
- `proposal_miss`: expressions whose target has no sufficiently overlapping
  retained proposal. Empty candidate sets count as failures.
- `selection_miss`: a usable proposal exists but the selector chooses another.
- `mean_selected_iou`: mean overlap across every expression, including failures.
- `scene_macro_accuracy`: compute accuracy separately per image, then average
  those image accuracies; this avoids images with more expressions dominating.
- `per_group` and `per_scene`: descriptive breakdowns, not tuning targets.

Expressions on the same images are correlated. Do not treat all 23,703 as
independent trials when estimating confidence. Also do not compare this full
expression-weighted test score directly with 73/99 and label the difference an
improvement or regression: the splits and sampling methods differ.

## Completed result

The first full run completed with 15,843/23,703 correct expressions (66.84%).
Proposal misses: 2,358. Selection misses with an available target: 5,502.
Image-macro accuracy: 71.56%. Mean selected IoU: 0.5952. No text truncations.
Measured wall time: 596.1 seconds; peak allocated GPU memory: 0.8554 GiB.
The source selector and SAM hashes remained unchanged. `completed.json` was
written after all scoring completed. This result is now known; the test split
must no longer be described as untouched for future tuning decisions.
