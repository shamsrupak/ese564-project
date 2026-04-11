#!/bin/bash
# setup.sh - Complete setup script for the pick-and-place project
# Run this ONCE on a fresh machine to set up everything.
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh

set -e  # stop on any error

echo "============================================"
echo "ESE 564 Pick-and-Place Project Setup"
echo "Shams Rupak & Sowmya Cheripally"
echo "============================================"
echo ""

# ---- Step 1: Check Python ----
echo "[1/4] Checking Python..."
if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 not found. Install Python 3.8+ first."
    exit 1
fi
PYVER=$(python3 --version)
echo "  Found: $PYVER"

# ---- Step 2: Install dependencies ----
echo ""
echo "[2/4] Installing Python packages..."
pip install mujoco numpy opencv-python-headless trimesh scipy
echo "  Done."

# ---- Step 3: Clone MuJoCo Menagerie ----
echo ""
echo "[3/4] Cloning MuJoCo Menagerie (Panda robot model)..."
if [ -d "mujoco_menagerie" ]; then
    echo "  mujoco_menagerie already exists, skipping clone."
else
    git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie.git
fi

# ---- Step 4: Copy scene XML into Menagerie directory ----
echo ""
echo "[4/4] Setting up scene file..."
cp pick_and_place_scene.xml mujoco_menagerie/franka_emika_panda/pick_and_place_scene.xml
mkdir -p output
echo "  Done."

# ---- Verify ----
echo ""
echo "============================================"
echo "Setup complete! Verifying installation..."
echo "============================================"
python3 -c "
import mujoco
import numpy as np
import cv2
print(f'  MuJoCo:  {mujoco.__version__}')
print(f'  NumPy:   {np.__version__}')
print(f'  OpenCV:  {cv2.__version__}')

# Try loading the scene
model = mujoco.MjModel.from_xml_path('mujoco_menagerie/franka_emika_panda/pick_and_place_scene.xml')
data = mujoco.MjData(model)
print(f'  Scene:   Loaded ({model.nbody} bodies, {model.njnt} joints)')
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
