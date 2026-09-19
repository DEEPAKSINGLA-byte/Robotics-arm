# OCID: finding an object from a sentence

This project takes an RGB image and a sentence like "the blue box behind the
sponge". It selects the object, estimates its visible 3D surface, and suggests
possible suction contacts. It is a Task 3 prototype, not a finished robot controller.

## How it works

1. SAM 2.1 Tiny finds possible object masks.
2. SigLIP 2 Base turns the object crops and sentence into features.
3. A trained relation selector compares the objects and picks one.
4. MoGe 2 estimates depth. A small trained correction adjusts its distance scale.
5. Simple geometry checks look for flat contact patches and visible obstacles.

SAM, SigLIP and MoGe are frozen. Only the selector and distance-correction network
were trained. The current system is unchanged; YOLO-World is not part of it.

## Setup

Use Python 3.10 or newer and a CUDA-capable GPU for the full pipeline. Keep an
already working CUDA environment. For a new grounding environment, install:

```bash
python3 -m pip install -r requirements-experiment2.txt
```

Weights, datasets and environments are not included in Git. The expected local
model folders and trained checkpoints are listed in `configs/task3.json`.
The depth environment must also have MoGe installed. See the
[setup and run guide](docs/task3-pipeline.md) before running a fresh clone.

## Run one image

Run from the repository folder, with a new output directory:

```bash
python3 -B run_task3.py \
  --image /path/to/image.png \
  --sentence "The blue box behind the sponge." \
  --output outputs/blue-box
```

The result folder contains the selected mask, a point cloud, contact/pregrasp
poses, a preview and ordered action suggestions. Model paths and mask settings
come from `configs/task3.json`; no old evaluation run is needed.

The default suction cup is an assumed 20 mm in diameter. These are proposed
contacts, not verified safe grasps. Hidden surfaces, sealing, robot reachability
and full arm collisions still need checking. The code never moves a robot.

## Results

| Check | Recorded result |
| --- | --- |
| Full sequence grounding test | 15,843 / 23,703 expressions correct (66.84%, mask IoU ≥ 0.5) |
| Depth correction, 199 validation images | Mean object-depth error: 70.04 cm → 4.39 cm |
| Three illustrative Task 3 scenes | Correct object in all three; 24 contact proposals passed the implemented geometry checks |

These measure different things. None is a physical grasp-success rate.
The full grounding test has already been used for reporting; later tuning must
not be presented as a new untouched-test result. See the
[full test notes](docs/final-test.md) for the fixed evaluation protocol.

## Quick checks

```bash
python3 -B -m unittest discover -p 'test_*.py'
```

To score saved Task 3 predictions against OCID validation labels:

```bash
python3 -B verify_task3.py \
  --data-root /path/to/OCID-dataset \
  --runs outputs/blue-box \
  --output outputs/verification
```

This also needs the generated `splits/sequence/val.json`, or an explicit
`--manifest` path. Labels are used for checking saved predictions, not inference.

## Code and notes

- `run_task3.py`: main image-and-sentence runner.
- `task3_grasp_proposals.py`: contact proposals and action suggestions.
- `verify_task3.py`: checks saved results against validation annotations.
- `grounding_model.py`, `grounding_data.py`: selector and training batches.
- `candidate_inputs.py`: object crops and position features.
- `camera_geometry.py`, `experiment_io.py`: shared geometry and file helpers.
- `run_depth_scale.py`: depth-correction preparation, training and prediction.
- Other preparation, training and evaluation scripts are kept to reproduce experiments.
- `test_*.py` and `templates/`: fast checks and report layouts.

The [technical draft](docs/task3-technical-draft.md) explains the approach in
simple language. For training details, see [SAM adaptation](docs/sam-adaptation.md),
[depth correction](docs/depth-scale.md), and [mask settings](docs/sam-settings-study.md).
[Experiment history](docs/experiment-history.md) and `reports/verification/`
are older records, not current setup instructions or newly measured results.

## Before pushing

Commit source, tests, configs, requirements, templates and documentation.
The ignore file keeps model weights, datasets, feature caches, generated runs,
environments and secrets out of Git. The weights must be shared separately for
someone else to reproduce the saved model's predictions.
