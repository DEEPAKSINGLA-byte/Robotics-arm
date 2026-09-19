# Technical Report: Finding and Picking an Object from an RGB Image

This report explains our current work for CS/AI Task 3, **“The Bin at the End
of the Belt.”** The main aim is simple: the user gives the system an RGB image
and a sentence such as “pick the blue box behind the bottle.” The system should
find that object, understand approximately where it is in 3D, and suggest a
possible contact point for a suction gripper. This point still needs physical validation.

This is a working prototype. It can find objects and generate grasp suggestions,
but we have not tested the grasps on a real robot. Therefore, the system should
not be treated as ready for physical robot operation.

## 1. What the problem asks us to build

The problem statement asks for a spatial grounding and grasp proposal pipeline
that works on cluttered scenes. “Spatial grounding” means connecting a sentence
to the correct object in the image. For example, if there are three boxes, the
system should understand which one is meant by “the blue box on the right of the
cup.”

The input available during use is:

- one RGB image;
- one sentence describing the required object.

Our current output contains:

- the selected object mask;
- the visible 3D points of that object;
- possible suction contact points;
- a pregrasp pose from which the gripper can approach;
- a simple ordered list of actions for picking the object.

The current system does not control a robot. It also does not yet decide which
other object should be removed first when the requested object is deeply buried.

## 2. Models used in our system

We use four main parts. Each part has one specific job.

### SAM 2.1 Tiny: finding possible objects

SAM creates object masks. A mask is a binary image in which the pixels belonging
to one possible object are marked. SAM does not know which object the sentence
is asking for. It only gives the next part of the system a list of candidates.

We use 32 sampling points per side. Masks that are extremely small, cover almost
the whole image, or are almost duplicates are removed. We keep at most 128
candidates so that later processing remains manageable.

### SigLIP 2 Base: understanding images and text

SigLIP converts an object crop and the input sentence into numerical feature
vectors. These numbers contain information about appearance and language. For
example, they can help identify that a crop looks like a blue box and that the
sentence mentions a blue box.

SigLIP is kept frozen, which means we do not change its trained weights. Each
candidate object is cropped from the RGB image. Pixels outside its mask are made
gray, and the crop is padded into a square before being sent to SigLIP.

### Relation transformer: choosing the requested object

Simple image-text similarity is often not enough. If an image contains two
similar boxes, both may match the word “box.” We therefore use a small trained
relation transformer. It looks at all candidate objects together, along with
their positions, the complete scene and the sentence. This helps it use phrases
such as “left of,” “behind” and “bottom-right.”

The transformer gives one score to each candidate, and the candidate with the
highest score is selected. These scores are ranking values and should not be
read as confidence percentages.

We also tried a separate YOLO-World, MobileSAM and CLIP pipeline on 12 validation
images. It selected 2 objects correctly, while this existing system selected 9
on the same images. Because of this small result, we kept the existing SAM,
SigLIP and relation-transformer system. This test is too small to claim that one
model is always better than another.

### MoGe 2: estimating 3D information from RGB

MoGe estimates depth and a 3D point for each image pixel using only the RGB
image during inference. Its original distance scale was inaccurate on our data,
so an already trained small correction network predicts one scale multiplier
for the complete image. The same multiplier is applied to depth and to the X,
Y and Z coordinates.

Reference depth was used earlier to train this small correction network.
However, reference depth is not given to the system when it makes a prediction.
The runtime input is still only RGB and the sentence.

## 3. Complete working flow

The pipeline works in the following order:

1. SAM creates possible object masks from the RGB image.
2. Very small, very large and duplicate masks are filtered.
3. SigLIP creates features for every candidate, the full image and the sentence.
4. The relation transformer selects the candidate that best matches the complete
   instruction.
5. MoGe predicts the scene depth and 3D points from the RGB image.
6. The selected mask is used to collect the visible 3D points of the object.
7. The grasp module searches these points for locally flat areas that may support
   a suction cup.
8. Unsafe or poorly supported contact points are removed.
9. The remaining grasp suggestions are ranked and saved with ordered actions.

The output is saved as JSON, images and a point-cloud file. This makes it
possible to inspect the result before connecting it to any robot software.

## 4. Coordinate system

The predicted 3D points are first expressed in the camera coordinate frame:

- positive X points toward the right side of the image;
- positive Y points downward;
- positive Z points forward from the camera;
- distances are stored in metres.

To project a 3D point back into the image, we use the camera focal lengths and
centre values predicted by MoGe. In basic form, the horizontal image position
depends on `fx * X / Z + cx`, and the vertical position depends on
`fy * Y / Z + cy`. The implementation also accounts for the image width, height
and pixel-centre convention.

The visible object position reported by the system is the average of the valid
3D points inside its selected mask. This is only the centre of the visible
surface. It is not necessarily the true centre of the full object, especially
when part of the object is hidden.

If a measured camera-to-robot transformation is provided, the camera-frame
grasp pose can be converted to the robot frame using matrix multiplication:

`robot grasp pose = camera-to-robot transform × camera grasp pose`

We currently do not have a verified camera-to-robot calibration in this project.
Therefore, the saved poses should not be sent directly to a real robot.

## 5. How suction grasp points are generated

For the first version, we assume a circular suction cup with a radius of 10 mm.
This is only a configurable test value; the real cup size must be entered before
robot testing.

The algorithm selects points that are away from the edge of the object mask. It
then takes nearby predicted 3D points and fits a small plane to them. A nearly
flat surface is more likely to provide a good suction contact than a sharp edge
or highly curved surface.

A candidate is rejected when:

- there are too few valid 3D points around it;
- the local area is not flat enough;
- the cup would extend outside the selected object;
- part of the required surface is missing;
- predicted scene points block the gripper’s approach;
- the approach goes behind an already observed surface;
- the approach contains unknown depth or leaves the image.

The system assumes an approach distance of 80 mm and adds 5 mm of radial
clearance around the cup. At least 90% of the sampled cup area must be supported
by the selected object. Up to eight grasp suggestions are returned.

The ranking favours well-supported and flatter areas. The ranking number is not
the probability that the grasp will succeed. The system cannot know from shape
alone whether cardboard is porous, plastic is slippery, or a crushed package
can form an airtight seal.

## 6. Occlusion handling and action sequence

Occlusion means that another object or surface hides the requested object or
blocks the gripper’s path. Our current method checks the predicted points of the
complete visible scene. If known geometry lies inside the planned approach
region, that grasp suggestion is rejected.

This only handles visible obstacles. A single RGB image cannot show surfaces
that are completely hidden. If the system finds no acceptable grasp, it returns
the following safe response:

1. do not move the gripper;
2. capture another RGB view;
3. find the requested object again;
4. calculate new grasp suggestions.

When a grasp suggestion exists, the output action list asks the user or robot
software to verify the object, cup size, material, calibration, reachability and
full collision path. It then gives the order: move to the pregrasp pose, approach
the surface, activate suction, verify the seal, and retract. A placement action
is not produced because the input does not give a destination.

The program only produces these instructions as data. It does not publish robot
commands or move any hardware.

## 7. Results obtained so far

The retained object-grounding system was evaluated earlier on 23,703 expressions
from 220 sequence-separated test images. It selected a mask with at least 0.5
IoU for 15,843 expressions, giving 66.84% accuracy. IoU measures how much the
selected mask overlaps the labelled correct object. This is an object-selection
result, not a grasp-success result.

For depth, the average error over object pixels on 199 validation images reduced
from about 70.04 cm before scale correction to 4.39 cm after correction. In a
99-image position review, average object-position error was about 6.24 cm when
the correct object was selected and 25.52 cm when the selection was wrong.

We also ran the complete new pipeline on three cluttered validation scenes:

| Instruction | Selected-mask IoU | Grasp suggestions |
| --- | ---: | ---: |
| Blue tissue box on the rear left | 0.941 | 8 |
| Blue box | 0.948 | 8 |
| Cube behind and bottom-right of the box | 0.915 | 8 |

All 24 saved poses passed our implemented pose and visible-collision checks, and
their contact pixels were located on the labelled target object. These three
examples only confirm that the parts of the program are connected correctly.
They do not prove that the system has 100% accuracy or that the grasps will work
on a robot.

Seven quick software tests also passed. They check a flat graspable surface, an
object that is too small for the cup, an obstacle in the approach path, missing
depth, camera projection, coordinate conversion and invalid input values.

## 8. Main failure cases

The current system can fail in several ways:

- SAM may not generate a usable mask for the requested object.
- The relation transformer may choose the wrong one of two similar objects.
- A correct-looking grasp may be generated on the wrong object after a selection
  mistake.
- Monocular depth can be inaccurate, especially for shiny, transparent or thin
  objects.
- A hidden obstacle cannot be checked from a single RGB view.
- A flat predicted surface may still fail to create a suction seal.
- Cables and flexible objects need a different type of grasp planner.
- The scene may move after the image is captured.
- A calibration error can shift every grasp position.
- A camera-facing approach may still be outside the robot’s reachable workspace.

The position errors measured so far are several centimetres, while the assumed
suction cup radius is only 1 cm. This is a major limitation for real grasping.
Passing the current geometry checks does not remove that error.

## 9. Work still required

The following work is needed before calling Task 3 complete:

1. Test the proposed grasps in a simulator or using a grasp-labelled dataset.
2. If robot hardware is available, measure actual grasp success and failure.
3. Calibrate the camera with the robot and add inverse-kinematics checks.
4. Use the real suction cup or gripper dimensions.
5. Add full gripper, wrist and robot-arm collision checking.
6. Add multiple camera views for hidden surfaces and uncertain depth.
7. Decide which blocking object to remove first when the target is buried.
8. Add another grasp method for curved, porous, transparent and flexible objects.
9. Report grasp success separately from object-selection accuracy.

At present, the project is a useful RGB-to-grasp-proposal prototype. It performs
language grounding, predicts visible 3D geometry and produces inspectable grasp
suggestions. Its outputs still require verification before any physical robot
motion.
