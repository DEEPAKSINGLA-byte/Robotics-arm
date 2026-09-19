# SAM automatic-mask settings comparison

This experiment changes inference settings, not model weights. SAM 2.1 Tiny,
SigLIP and the adapted selector are fixed. Existing default settings are unchanged.

Four settings are declared before inspecting the results:

| Variant | Change from baseline |
| --- | --- |
| baseline | 32 points per side, no crops, quality 0.8, stability 0.95 |
| dense64 | 64 points per side (four times as many point prompts) |
| crops1 | One crop layer, with the same points-per-crop setting |
| relaxed | Quality threshold 0.7 and stability threshold 0.9 |

Other filtering stays fixed, including the 128-candidate cap. The relaxed
variant changes two related filters together; it does not isolate their
individual effects. Crop layers increase processing and memory requirements.
The installed Transformers pipeline's multi-crop forward method filters only
the first crop. Therefore `crops1` explicitly runs the full image plus four
overlapping crops, maps each mask to full-image coordinates, rejects masks
touching artificial crop boundaries, and merges proposals using quality-ranked
box NMS at 0.7. This is our tested tiling wrapper, not an exact reproduction of
Meta's native automatic-mask crop implementation. Crops are processed one at a
time to limit GPU memory. Tests verify all five calls contribute correctly.
The quality threshold is SAM's *predicted* mask quality, not measured IoU against
dataset labels. Evaluation success remains measured IoU >= 0.5 for every run.

## Run

```bash
# Run from the repository folder.
python3 -B run_sam_settings_study.py --count 20 \
  --output runs/sam-settings-study-20
```

Use a fresh output folder for a repeat. The USB dataset and GPU are required.
The script runs offline and generates a self-contained report for each variant.
It first excludes the 100 validation images used for selector adaptation, then
selects a balanced random subset of the remaining validation images. Selection
is independent of correctness; exact images and expressions are fixed across
variants. It verifies unchanged checkpoint hashes and RGB images.

These are exploratory validation comparisons. The remaining images were already
used for the selector comparison and are now being used for settings tuning;
they must not be described as an independent test set. Test data stays untouched.

## Read results

To run only the two promising settings on 99 images and reuse the completed
original-settings baseline with the adapted selector:

```bash
python3 -B run_sam_settings_study.py --count 99 \
  --variants crops1 relaxed \
  --baseline-run runs/selector-comparison-remaining99/adapted \
  --output runs/sam-settings-two-99
```

This launches exactly two evaluations. The saved baseline must match the exact
image/expression set, model weights, original SAM settings and evaluation
threshold. It is matched by expression key even if report ordering differs.
`comparison.json` includes `baseline_reused` for reference, not a third run.
Add `--check-only` to validate configuration without running inference.

`comparison.json` records each variant's:

- Usable target masks, before and after our extra filters.
- Final correct selections with the frozen adapted selector.
- Recovered and lost targets relative to baseline.
- Fixed and newly wrong selections relative to baseline.
- Candidate count, processing time and peak allocated GPU memory.

More proposals are not automatically better: extra fragments may confuse the
selector. Judge final selections as well as target coverage and runtime. A win
on 20 images is a reason for a larger controlled validation check, not proof
that the new settings generalize or that SAM fine-tuning is unnecessary.

The existing evaluator also accepts `--crop-layers`, `--pred-iou-thresh`, and
`--stability-score-thresh` for individual experiments. Its old defaults remain
the same. No SAM fine-tuning code is introduced by this study.

## Measured first study (20 images, seed 42)

| Variant | Usable target | Correct selection | Mean SAM seconds/image |
| --- | --- | --- | --- |
| baseline | 13/20 | 10/20 | 2.25 |
| dense64 | 14/20 | 10/20 | 8.85 |
| crops1 | 16/20 | 13/20 | 10.59 |
| relaxed | 15/20 | 12/20 | 2.62 |

Crops recovered three missing targets, fixed four selections and introduced
one wrong selection. Relaxed filtering recovered two targets and fixed two
selections with no newly wrong cases on this sample. Denser sampling recovered
one target but its selection gain was offset by one new error.

Recommendation: compare relaxed filtering and crops on a larger validation
sample before changing defaults or starting SAM fine-tuning. Relaxed filtering
is the first candidate for a speed-conscious check. Do not combine settings
and assume their gains will add. The original SAM weights and adapted selector
hashes were verified unchanged; 25 unit tests pass. No test data was evaluated.
