# Task 3: current working pipeline and remaining work

Aligned with **CS_AI Problem Statements.pdf**, page 3, “The Bin at the End of
the Belt”. The requested inputs are RGB images and free-form language. The
deliverables include spatial grounding, grasp proposals, verification on
cluttered scenes, serial navigation instructions, and a technical whitepaper.

## What we kept

The existing relaxed SAM 2.1 Tiny, SigLIP 2 Base, trained relation selector,
MoGe 2 and learned distance correction are unchanged. The YOLO-World experiment
is separate and is not used by this runner.

The retained selector is `runs/sam-adaptation-relaxed-300/best.pt`.
The retained depth correction is `runs/depth-scale-ssd-v1/best.pt`.
The historical full grounding test result is 15,843/23,703 expressions,
66.84% at mask IoU >= 0.5. This is object-selection accuracy, not grasp success.

## What was added

One command now takes an image and sentence and runs these stages:

1. Find candidate object outlines using the retained SAM settings.
2. Use SigLIP and the trained relation selector to choose the requested object.
3. Estimate fresh depth and 3D points with MoGe and the existing scale correction.
4. Export the selected visible surface and its approximate camera-frame position.
5. Propose suction contacts on locally flat, sufficiently supported surface patches.
6. Reject candidates whose cup footprint is unsupported or whose approach contains
   another predicted surface. Also reject approaches hidden behind an observed
   surface or with missing depth along the sampled centreline.
7. Save ordered, conditional actions: verify target and preconditions, move to
   pregrasp, approach, activate suction, verify seal, retract, then plan placement.

If there is no supported grasp, the output says to hold, obtain another view,
and plan again. It does not invent a contact from the object's average position.

The suction cup is a **prototype assumption of 20 mm diameter**. Adjust it to real
hardware before interpreting feasibility. The current implementation does not
provide a parallel-jaw grasp planner.

## Setup on another computer

The code is shareable, but the downloaded models and trained checkpoints are
not stored in Git. Use Python 3.10+ and a CUDA GPU. Install the grounding
dependencies from `requirements-experiment2.txt`. This records the versions
used locally; choose the appropriate CUDA wheels for your machine.

The paths in `configs/task3.json` are relative to the repository, even when
running from another folder. Place the following files there, or pass
`--config /path/to/your-config.json` with your own model paths:

- `models/sam2.1-hiera-tiny/`: complete SAM model and processor files.
- `models/siglip2-base-patch16-224/`: complete SigLIP model, tokenizer and processor files.
- `models/moge-2-vitb-normal/model.pt`: MoGe 2 weights.
- `runs/sam-adaptation-relaxed-300/best.pt`: trained selector.
- `runs/depth-scale-ssd-v1/best.pt`: trained depth correction.

For depth, the local setup uses the Microsoft MoGe source under `models/moge`
at revision `74fbce054ebed49800de42d0ad0e83495065719a`, with its own environment
at `models/moge/.venv`. Keep that source version when using the saved depth head:
the depth script checks the MoGe implementation and weight hashes against training.
Install that checkout and its dependencies in the depth environment using
`python -m pip install -e models/moge`. This is a separate setup step, not part of inference.
Use `--depth-python /path/to/environment/bin/python` for another environment.
Without that flag, the runner uses the local MoGe environment if present,
otherwise the Python interpreter used to launch the runner.

The config includes checksums for the retained models. If you deliberately
replace or retrain a model, update its matching config values too. The trained
heads are project artifacts, not public pretrained weights. A fresh clone alone
cannot reproduce the saved predictions without them.

## Run

From the existing project:

```bash
# Run from the repository folder.
python3 -B run_task3.py \
  --image /absolute/path/image.png \
  --sentence 'The blue kleenex on the rear left.' \
  --output /absolute/path/new-result-folder
```

The runner uses the Python that launched it for grounding and a separate process
for depth. Once the models and environments are set up, no training, network
access or dataset annotations are needed. Use a fresh output folder. The one-image demo took about
17 seconds including separate model loads before the lightweight grasp step;
this is a demonstration runner, not an optimized persistent camera service.

Optional settings:

- `--cup-radius-m 0.010`: physical cup radius in metres.
- `--camera-to-robot /absolute/path/T_robot_camera.json`: an actual calibrated
  rigid 4x4 transform. This adds robot-frame pose representations; it does not
  enable execution or establish robot reachability. Do not use an invented transform.

Outputs:

| File | Meaning |
| --- | --- |
| `selected_object.png` | Selected outline over the RGB image |
| `selection/selected_mask.png` | Binary selected mask |
| `selected_object_3d.npz` | Selected predicted surface XYZ in metres and RGB |
| `selected_object.ply` | Small point-cloud export for a 3D viewer |
| `grasp_preview.png` | Orange candidate contacts and an approach arrow |
| `grasp_proposals.json` | Contact/pregrasp poses, checks, rejection reasons and ordered actions |
| `result.json` | Selected object, camera-frame surface centroid and proposal status |
| `configuration.json` | Checkpoints, hashes and inference input |

Scores used to rank contacts are geometric heuristics, not probabilities of
successful suction. `robot_execution_ready` remains false.

## Fast verification

```bash
# Run from the repository folder.
python3 -B -m unittest test_task3_grasps -v
python3 -B verify_task3.py \
  --data-root /path/to/OCID-dataset \
  --runs /absolute/path/result1 /absolute/path/result2 \
  --output /absolute/path/new-verification-folder
```

The verification command needs `splits/sequence/val.json`, or `--manifest` with
the path to that validation manifest. It no longer needs cached features or old
prediction files. The numerical checks cover supported and unsupported contact patches, an
obstacle in the approach cylinder, missing geometry, camera projection, rigid
transforms, invalid inputs and action preconditions. They complete in well under
a second. The second command is an OCID **validation-only** review: it reads
annotations after predictions exist, measures selected-mask overlap, checks
poses and visible collision clearance, and creates a visual report. It does not
run a new dataset-scale experiment or measure physical grasp success.

The completed three-scene check reproduced the retained selector's masks:

| Instruction | Selected-mask IoU | Geometric grasp hypotheses |
| --- | ---: | ---: |
| Blue kleenex on the rear left | 0.941 | 8 |
| Blue box | 0.948 | 8 |
| Cube behind and bottom-right of the box | 0.915 | 8 |

All 24 returned poses passed the implemented pose and predicted-surface collision
checks; their seed pixels lay on the annotated target. This is a functionality
check on three illustrative scenes, not evidence of 100% grasp or grounding
accuracy. Grounding plus fresh depth took roughly 14–17 seconds per scene,
including loading, before the lightweight grasp step. Seven numerical tests passed.

## What remains for the complete problem statement

1. **Validate grasps.** Compare proposals with grasp annotations or a simulator,
   then measure real success where hardware is available. Our current examples
   demonstrate the output format and geometry checks only.
2. **Improve occlusion handling.** The current planner rejects observed blockers
   and requests another view. It does not yet choose which obstructing object to
   remove or build a multi-object removal sequence. Additional views and explicit
   object-support relationships are needed for that claim.
3. **Use actual tool and robot information.** Supply cup/gripper dimensions,
   camera-to-robot calibration, reachability, and full wrist/arm collision checks.
   A thin contact-envelope check cannot replace these.
4. **Test difficult materials and shapes.** Flat predicted geometry does not
   establish an airtight seal, grasp stability, or accurate geometry for shiny,
   transparent, flexible or crushed objects. Add rejection/uncertainty handling
   and a parallel-jaw alternative where appropriate.
5. **Finish the whitepaper with measured evidence.** A technical draft describes
   the implemented coordinate transformations, occlusion checks and known
   failures. It needs grasp-validation results before claiming completion.

The current work is a functional prototype toward Task 3, not a claim of reliable
physical grasping. No robot movements have been issued.
