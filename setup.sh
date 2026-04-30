#!/bin/bash
# setup.sh - Complete setup script for the pick-and-place project
# Run this ONCE on a fresh machine to set up everything.
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh
#
# This script is idempotent - safe to re-run.

set -e  # stop on any error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PANDA_XML="mujoco_menagerie/franka_emika_panda/panda.xml"

echo "============================================"
echo "ESE 564 Pick-and-Place Project Setup"
echo "Shams Rupak & Sowmya Cheripally"
echo "============================================"
echo ""

# ---- Step 1: Check Python ----
echo "[1/5] Checking Python..."
if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 not found. Install Python 3.8+ first."
    exit 1
fi
PYVER=$(python3 --version)
echo "  Found: $PYVER"

# ---- Step 2: Install dependencies ----
echo ""
echo "[2/5] Installing Python packages..."
pip install mujoco numpy opencv-python-headless trimesh scipy
echo "  Done."

# ---- Step 3: Clone MuJoCo Menagerie ----
echo ""
echo "[3/5] Cloning MuJoCo Menagerie (Panda robot model)..."
if [ -f "$PANDA_XML" ]; then
    echo "  mujoco_menagerie already present, skipping clone."
else
    rm -rf mujoco_menagerie
    git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie.git
fi

# ---- Step 4: Add wrist camera to Panda's hand body ----
# The wrist camera is used for ICP refinement during the approach phase
# (coarse-to-fine perception, as described in the proposal).
# We inject it into the Menagerie panda.xml because we cannot reach into
# included bodies from our scene file.
echo ""
echo "[4/5] Adding wrist camera to panda.xml..."
if grep -q 'name="wrist_cam"' "$PANDA_XML"; then
    echo "  Wrist camera already present, skipping."
else
    # macOS BSD sed and GNU sed differ on the -i flag - detect and use the right one.
    if sed --version >/dev/null 2>&1; then
        SED_INPLACE=(sed -i)        # GNU sed (Linux)
    else
        SED_INPLACE=(sed -i '')     # BSD sed (macOS)
    fi
    "${SED_INPLACE[@]}" \
        's|<body name="hand" pos="0 0 0.107" quat="0.9238795 0 0 -0.3826834">|&\n                      <camera name="wrist_cam" pos="0 0 0.05" xyaxes="0 -1 0 1 0 0" fovy="60"/>|' \
        "$PANDA_XML"

    if ! grep -q 'name="wrist_cam"' "$PANDA_XML"; then
        echo "  ERROR: failed to add wrist camera to panda.xml"
        exit 1
    fi
    echo "  Wrist camera added."
fi

# ---- Step 5: Output directory ----
echo ""
echo "[5/5] Creating output directory..."
mkdir -p output
echo "  Done."

# ---- Verify ----
echo ""
echo "============================================"
echo "Verifying installation..."
echo "============================================"
python3 -c "
import mujoco
import numpy as np
import cv2
print(f'  MuJoCo:  {mujoco.__version__}')
print(f'  NumPy:   {np.__version__}')
print(f'  OpenCV:  {cv2.__version__}')

# Load the scene from the project root (where evaluation.py expects it)
model = mujoco.MjModel.from_xml_path('pick_and_place_scene.xml')
data = mujoco.MjData(model)

cam_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
             for i in range(model.ncam)]
assert 'overhead_cam' in cam_names, 'overhead_cam missing'
assert 'wrist_cam' in cam_names, 'wrist_cam missing - setup failed'

print(f'  Scene:   Loaded ({model.nbody} bodies, {model.njnt} joints)')
print(f'  Cameras: {cam_names}')
print()
print('All systems go!')
"

echo ""
echo "============================================"
echo "To run the project:"
echo "  python3 evaluation.py --num_episodes 5"
echo ""
echo "To run with images saved:"
echo "  python3 evaluation.py --num_episodes 10 --save_images"
echo ""
echo "Headless mode (no display):"
echo "  MUJOCO_GL=egl python3 evaluation.py --num_episodes 10"
echo "============================================"
