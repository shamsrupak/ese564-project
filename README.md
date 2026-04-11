# ESE 564 Final Project: Vision-Guided Pick-and-Place
## Shams Rupak & Sowmya Cheripally

### Overview
A Franka Emika Panda 7-DOF robot picks objects from randomized positions on a
tabletop and places them into a basket whose position also varies between episodes.
Uses classical computer vision (HSV segmentation + depth back-projection) for
perception and model-based planning (Jacobian IK + PD control) for manipulation.

### Setup

```bash
# 1. Install dependencies
pip install mujoco numpy opencv-python-headless trimesh scipy

# 2. Clone MuJoCo Menagerie (for the Panda robot model)
git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie.git

# 3. Create output directory
mkdir -p output
```

### Running the Evaluation

```bash
# Run the main evaluation (the professor should run this)
python evaluation.py --num_episodes 10

# Full evaluation (50 episodes)
python evaluation.py --num_episodes 50

# With debug images saved
python evaluation.py --num_episodes 10 --save_images

# Headless mode (no display, for servers)
MUJOCO_GL=egl python evaluation.py --num_episodes 10
```

### Recording Video

```bash
# Record frames from 5 successful episodes
python record_video.py

# Compile frames into video (requires ffmpeg)
ffmpeg -framerate 30 -i output/frames/ep0_frame_%05d.png \
       -c:v libx264 -pix_fmt yuv420p video_ep0.mp4
```

### Project Structure

```
pick_and_place/
  evaluation.py         # Main entry point - runs randomized episodes
  perception.py         # HSV segmentation + depth back-projection + PCA + ICP
  grasp_planner.py      # Grasp waypoints + Jacobian pseudo-inverse IK
  controller.py         # PD controller + trajectory execution
  record_video.py       # Record frames from successful episodes
  report.tex            # Final report (LaTeX source)
  report.pdf            # Final report (compiled PDF)
  README.md             # This file
  mujoco_menagerie/     # Panda robot model (cloned from GitHub)
    franka_emika_panda/
      pick_and_place_scene.xml   # Our scene (table, basket, obstacle, object)
      panda.xml                   # Robot definition
      assets/                     # Robot mesh files
  output/               # Saved images and results
```

### Pipeline

1. **Perception**: Overhead RGB-D camera -> HSV color segmentation -> depth
   back-projection to 3D point cloud -> centroid estimation
2. **Grasp Planning**: Top-down antipodal grasp with 7 Cartesian waypoints
   (approach, pre-grasp, descend, close, lift, transport, release)
3. **Inverse Kinematics**: Damped Jacobian pseudo-inverse IK converts Cartesian
   waypoints to 7-DOF joint angles (with retry from alternative configs)
4. **Controller**: Built-in PD servo + supplementary torques for tight tracking;
   combined descent+settle+close for reliable grasping

### Results
- Success rate: ~82% average across 90 randomized episodes (3 seeds x 30 episodes)
- Perception accuracy: ~50mm total (XY: ~5mm, Z: ~30mm due to top-surface-only view)
- IK accuracy: 2-5mm
- Mean placement accuracy: 12mm (for successful grasps)
- Mean execution time: ~3 seconds per episode

### Homework Connections
- HW1 (Coordinate Transforms): Camera-to-world back-projection
- HW2 (FK/IK): Jacobian pseudo-inverse inverse kinematics
- HW3 (RRT): Motion planning framework (collision checking concepts)
- HW4 (ICP): Point cloud registration framework in perception.py
