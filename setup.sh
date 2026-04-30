#!/usr/bin/env bash
# setup.sh — one-time setup for the ESE 564 pick-and-place project.
#
# This script:
#   1. Clones MuJoCo Menagerie (if not already present)
#   2. Adds a wrist camera to the Panda's hand body in panda.xml
#      (idempotent — safe to re-run)
#   3. Creates the output/ directory
#
# Usage:
#   bash setup.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PANDA_XML="mujoco_menagerie/franka_emika_panda/panda.xml"

# 1. Clone Menagerie if missing ------------------------------------------------
if [ ! -f "$PANDA_XML" ]; then
  echo "[setup] Cloning MuJoCo Menagerie..."
  rm -rf mujoco_menagerie
  git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie.git
else
  echo "[setup] Menagerie already present, skipping clone."
fi

# 2. Inject wrist camera into panda.xml ----------------------------------------
if grep -q 'name="wrist_cam"' "$PANDA_XML"; then
  echo "[setup] Wrist camera already in panda.xml, skipping."
else
  echo "[setup] Adding wrist camera to panda.xml..."
  # macOS BSD sed and GNU sed differ — detect and use the right one.
  if sed --version >/dev/null 2>&1; then
    SED_INPLACE=(sed -i)        # GNU sed (Linux)
  else
    SED_INPLACE=(sed -i '')     # BSD sed (macOS)
  fi
  "${SED_INPLACE[@]}" \
    's|<body name="hand" pos="0 0 0.107" quat="0.9238795 0 0 -0.3826834">|&\n                      <camera name="wrist_cam" pos="0 0 0.05" xyaxes="0 -1 0 1 0 0" fovy="60"/>|' \
    "$PANDA_XML"

  # Verify the edit landed
  if ! grep -q 'name="wrist_cam"' "$PANDA_XML"; then
    echo "[setup] ERROR: failed to add wrist camera to panda.xml"
    exit 1
  fi
  echo "[setup] Wrist camera added."
fi

# 3. Output directory ----------------------------------------------------------
mkdir -p output

echo "[setup] Done. Run: python3 evaluation.py --num_episodes 5"
