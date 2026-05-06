# ESE 564 Final Project: Vision-Guided Pick-and-Place
## Shams Rupak & Sowmya Cheripally

### Overview

This project implements a vision-guided pick-and-place system on a Franka
Emika Panda 7-DOF arm in MuJoCo. The robot picks one of three YCB-style
mesh objects from a randomized tabletop position and places it in a basket
whose position also varies between episodes. A wall obstacle separates the
pick zone from the place zone, requiring the robot to plan a collision-free
path over the wall.

The pipeline is fully model-based and uses concepts from all four homework
assignments: HSV color segmentation with depth back-projection (HW1),
Jacobian pseudo-inverse inverse kinematics (HW2), RRT motion planning
(HW3), and ICP point-cloud registration (HW4). A coarse-to-fine perception
strategy with an overhead camera + wrist camera, and a grasp stabilizer
that maintains contact during transport, push the success rate to 100%.

Objects supported:
- `cracker_box`
- `mustard_bottle`
- `sugar_box`

### Results

Evaluated across 150 randomized episodes (50 per object). Identical
results were obtained on Apple M1 Pro, Apple M4, and x86_64 Linux:

| Object         | Episodes | Successes | Rate    |
|----------------|----------|-----------|---------|
| cracker_box    | 50       | 50        | 100.0%  |
| mustard_bottle | 50       | 50        | 100.0%  |
| sugar_box      | 50       | 50        | 100.0%  |
| **Total**      | **150**  | **150**   | **100.0%** |

- Mean perception error: 1.0 mm
- Mean placement accuracy: 16.3 mm (basket radius is 100 mm)
- Mean execution time: 1.36 s per episode (simulator time)

### Setup

**Requirements:** Python 3.9-3.13 with a native arm64 build on Apple
Silicon (NOT Anaconda's x86_64 Python). The `setup.sh` script handles
the rest.

```bash
chmod +x setup.sh
./setup.sh
```

`setup.sh` will:
1. Install Python dependencies including `mujoco==3.2.3` (pinned for
   reproducibility), `numpy`, `opencv-python-headless`, `trimesh`, and
   `scipy`.
2. Clone the MuJoCo Menagerie repository (provides the Panda robot model).
3. Modify `mujoco_menagerie/franka_emika_panda/panda.xml` automatically
   to inject the wrist camera, widen the gripper slide-joint range, and
   set the home keyframe with all 37 qpos values. **No manual file
   copying is required** -- the `panda.xml` file in the project root is
   the same modified version that `setup.sh` produces, included for
   reference only.
4. Create the `output/` folder for runtime artifacts.

#### Apple Silicon Note

If your default `python3` is Anaconda's x86_64 Python, MuJoCo will fail
to import. Use a native arm64 Python:

```bash
# Option 1: Homebrew's python3
/opt/homebrew/bin/python3 evaluation.py ...

# Option 2: install a specific version
brew install python@3.12
/opt/homebrew/opt/python@3.12/bin/python3.12 evaluation.py ...
```

### Evaluation

Run the full 150-episode evaluation:

```bash
python3 evaluation.py --num_episodes 50
```

Run a single object:

```bash
python3 evaluation.py --num_episodes 50 --object cracker_box
python3 evaluation.py --num_episodes 50 --object mustard_bottle
python3 evaluation.py --num_episodes 50 --object sugar_box
```

Save debug images:

```bash
python3 evaluation.py --num_episodes 10 --save_images
```

Use ground-truth simulator poses instead of camera perception (debugging
the planner/controller in isolation):

```bash
python3 evaluation.py --num_episodes 10 --use_gt
```

### Recording Videos

`record_video.py` saves successful episodes as image frames; you then
need to compile them with `ffmpeg` to produce a video file. For
convenience, `record_merged_video.py` does both steps in one command.

Record one full pick-and-place sequence (cracker -> mustard -> sugar)
and auto-merge into a single MP4:

```bash
python3 record_merged_video.py --sequence
```

This writes `video_sequence_merged.mp4` to the project root.

Top view only:

```bash
python3 record_merged_video.py --sequence --top_view
```

Both views (overhead + side) simultaneously:

```bash
python3 record_merged_video.py --sequence --both_views
```

To use the lower-level recorder (frames only, manual ffmpeg):

```bash
python3 record_video.py --object cracker_box
ffmpeg -framerate 30 -i output/frames/ep0_frame_%05d.png \
       -c:v libx264 -pix_fmt yuv420p video_ep0.mp4
```

### Pipeline

1. **Perception** - Overhead RGB-D camera observations are segmented
   with HSV color masks and back-projected into 3D point clouds (HW1).
   PCA gives a coarse orientation, then ICP (HW4) refines to a 6-DOF
   pose by registering against pre-computed YCB model clouds. After
   the arm moves to the pre-grasp pose, a wrist-mounted camera takes
   a closer image and re-runs perception for refined estimates.

2. **Antipodal Grasp Planning** - The squeeze axis comes from the
   smallest-eigenvalue eigenvector of the point cloud's covariance.
   The gripper opening is sized to the perception-estimated object
   width. Generates 9 Cartesian waypoints: approach high, pre-grasp,
   grasp, close gripper, lift, clear wall, above bin, release, retreat.
   Per-object grasp height offsets handle the geometric differences
   between the three YCB objects.

3. **Inverse Kinematics (HW2)** - Damped Jacobian pseudo-inverse IK
   converts each Cartesian waypoint to 7-DOF joint angles, with
   automatic retry from alternative starting configurations on failure.

4. **Motion Planning (HW3)** - Direct path is checked first; if it
   collides with the wall, RRT plans a collision-free path in joint
   space, which is then smoothed by random shortcutting.

5. **Control** - A PD controller with cosine-eased trajectory
   interpolation tracks each waypoint. The grasp phase is a combined
   descent-and-close motion to prevent the open fingers from pushing
   the object during descent. A grasp stabilizer applies bounded
   virtual-spring forces between hand and object during transport to
   prevent slip.

6. **Evaluation** - Success is determined by whether the object lands
   inside the basket footprint (XY distance < 100 mm and the right Z
   range). The success check reads simulator state but is only used for
   evaluation, never for control.

### Project Structure

```
.
|-- controller.py              PD control + trajectory execution + grasp stabilizer
|-- evaluation.py              Main randomized evaluation script
|-- grasp_planner.py           9-waypoint generation + Jacobian IK (HW2)
|-- motion_planner.py          RRT + collision checking (HW3)
|-- perception.py              HSV + depth back-projection + ICP (HW1, HW4)
|-- pick_and_place_scene.xml   Main MuJoCo scene
|-- panda.xml                  Modified Panda XML (reference only; setup.sh patches the menagerie copy)
|-- record_video.py            Records successful episodes as frames
|-- record_merged_video.py     Records and merges into MP4
|-- setup.sh                   One-shot setup script
|-- objects/                   YCB STL meshes + ICP model clouds + textured assets
|-- mujoco_menagerie/          Cloned by setup.sh (Panda robot model)
|-- output/                    Runtime artifacts (frames, logs, debug images)
|-- report.tex / report.pdf    Final report
`-- video_sequence_merged.mp4  Demo video (5 successful pick-and-place sequences)
```

### Homework Connections

- **HW1 (Coordinate Transforms)** - Camera-to-world depth back-projection
  in `perception.py`.
- **HW2 (FK/IK)** - Jacobian pseudo-inverse IK in `grasp_planner.py`.
- **HW3 (RRT)** - Collision-aware motion planning in `motion_planner.py`.
- **HW4 (ICP)** - 6-DOF pose registration in `perception.py`.
