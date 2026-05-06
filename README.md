# ESE 564 Final Project: Vision-Guided Pick-and-Place
## Shams Rupak & Sowmya Cheripally

### Overview

This project simulates a Franka Emika Panda robot performing tabletop pick-and-place with YCB-style objects. The scene includes randomized object poses, a randomized basket, and a wall obstacle. The robot uses RGB-D perception, grasp planning, inverse kinematics, RRT-style collision checking, and PD control to pick objects and drop them into the basket.

Objects currently supported:

- `cracker_box`
- `mustard_bottle`
- `sugar_box`

### Setup

Run the setup script first:

```bash
chmod +x setup.sh
./setup.sh
```

The script installs dependencies, downloads MuJoCo Menagerie, and creates the output folders.

### Important Panda XML Step

This project includes a modified Panda robot XML file at:

```text
panda.xml
```

After downloading MuJoCo Menagerie, replace the Menagerie Panda XML with this project’s root-level file:

```bash
cp panda.xml mujoco_menagerie/franka_emika_panda/panda.xml
```

This is needed because the project version adds the wrist camera used by the perception pipeline. If you keep the default `mujoco_menagerie/franka_emika_panda/panda.xml`, the scene may load, but wrist-camera functionality can be missing.

### Evaluation

Run all three objects:

```bash
python3 evaluation.py --num_episodes 50
```

Run one object:

```bash
python3 evaluation.py --num_episodes 50 --object cracker_box
python3 evaluation.py --num_episodes 50 --object mustard_bottle
python3 evaluation.py --num_episodes 50 --object sugar_box
```

Save debug images:

```bash
python3 evaluation.py --num_episodes 10 --save_images
```

### Recording Videos

`record_video.py` records successful episodes as image frame sequences. This is the lower-level recording script. It does not directly create one final merged video file; after running it, you still need a separate video compilation step if you want an `.mp4`.

```bash
python3 record_video.py --object cracker_box
python3 record_video.py --sequence
```

By default, `record_video.py --sequence` records the first 5 successful full sequences in this order:

```text
cracker_box -> mustard_bottle -> sugar_box
```

Record a top-view video frame sequence:

```bash
python3 record_video.py --sequence --top_view
```

Record normal view and top view simultaneously:

```bash
python3 record_video.py --sequence --both_views
```

The generated frames are saved in:

```text
output/frames/
```

For example, one episode is stored as files like:

```text
output/frames/ep0_frame_00000.png
output/frames/ep0_frame_00001.png
...
```

To manually turn one frame sequence into a video, run a separate command such as:

```bash
ffmpeg -framerate 30 -i output/frames/ep0_frame_%05d.png \
       -c:v libx264 -pix_fmt yuv420p video_ep0.mp4
```

### Recording and Merging Videos

`record_merged_video.py` is a convenience wrapper around `record_video.py`. It accepts the same main input options, runs `record_video.py` for you, then automatically merges the successful episode frames into one finished `.mp4`.

Use this script when you want to avoid the extra manual `ffmpeg` step:

```bash
python3 record_merged_video.py --sequence
```

This records successful episodes, merges them, and writes:

```text
video_sequence_merged.mp4
```

Top view only:

```bash
python3 record_merged_video.py --sequence --top_view
```

Output:

```text
video_sequence_merged_top.mp4
```

Both views:

```bash
python3 record_merged_video.py --sequence --both_views
```

Outputs:

```text
video_sequence_merged.mp4
video_sequence_merged_top.mp4
```

In short:

```text
record_video.py         -> records frames only
record_merged_video.py  -> records frames, merges them, and outputs final mp4 files
```

For most final demonstrations, use `record_merged_video.py`.

### Useful Options

Use simulator poses instead of camera perception:

```bash
python3 evaluation.py --num_episodes 50 --use_gt
python3 record_merged_video.py --sequence --use_gt
```

Use camera perception during sequence recording:

```bash
python3 record_merged_video.py --sequence --sequence_perception
```

Change the number of successful episodes:

```bash
python3 record_merged_video.py --sequence --episodes 3
```

### Project Structure

```text
.
├── controller.py                 # PD control, gripper control, trajectory execution
├── evaluation.py                 # Main randomized evaluation script
├── grasp_planner.py              # Waypoints and inverse kinematics
├── motion_planner.py             # Collision-aware path planning helpers
├── perception.py                 # HSV/depth/ICP perception pipeline
├── pick_and_place_scene.xml      # Main MuJoCo scene
├── record_video.py               # Records successful episodes as frames
├── record_merged_video.py        # Records and merges successful episodes into mp4
├── panda.xml                     # Modified Panda XML to copy into MuJoCo Menagerie
├── objects/                      # YCB textured object assets
├── mujoco_menagerie/             # Downloaded robot assets
└── output/                       # Saved frames, images, and generated videos
```

### Pipeline

1. **Perception**: RGB-D camera observations are segmented with HSV masks and projected into 3D.
2. **Grasp Planning**: The planner generates approach, pre-grasp, descend, close, lift, wall-clear, above-bin, release, and retreat waypoints.
3. **Inverse Kinematics**: Cartesian waypoints are converted into Panda joint targets.
4. **Motion Planning**: RRT-style checks are used for collision-aware transitions around the wall.
5. **Control**: A PD controller tracks joint trajectories and commands the gripper.
6. **Evaluation**: Success is checked by whether the object lands inside the basket.

### Notes

- Generated frames are saved in `output/frames/`.
- Merged videos are written to the project root.
- The top-view camera is `overhead_cam` in `pick_and_place_scene.xml`.
- If VS Code cannot preview a generated video, install `ffmpeg` and rerun `record_merged_video.py`; the wrapper will convert the output to a VS Code-friendly codec.
