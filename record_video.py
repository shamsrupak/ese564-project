"""
record_video.py - Record frames from 5 successful pick-and-place episodes.

Saves individual frames that can be compiled into a video with ffmpeg:
  ffmpeg -framerate 30 -i frames/ep%d_frame_%04d.png -c:v libx264 -pix_fmt yuv420p video.mp4

Usage:
    python record_video.py
    python record_video.py --object sugar_box
"""
import argparse
import mujoco
import numpy as np
import cv2
import os
import sys
os.environ.setdefault('MUJOCO_GL', 'egl')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from perception import perceive_object
from grasp_planner import compute_grasp_waypoints, compute_joint_targets
from controller import set_gripper
from evaluation import YCB_OBJECTS, randomize_scene

SCENE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pick_and_place_scene.xml")


def choose_active_object():
    parser = argparse.ArgumentParser(description="Record pick-and-place video frames.")
    parser.add_argument("--object", choices=list(YCB_OBJECTS.keys()), default=None,
                        help="Object to record: cracker_box, mustard_bottle, or sugar_box")
    args = parser.parse_args()

    if args.object:
        return args.object

    object_names = list(YCB_OBJECTS.keys())
    print("Choose object to record:")
    for i, name in enumerate(object_names, start=1):
        print(f"  {i}. {name}")

    while True:
        choice = input("Object [1-3]: ").strip()
        if choice in {"1", "2", "3"}:
            return object_names[int(choice) - 1]
        if choice in YCB_OBJECTS:
            return choice
        print("Please enter 1, 2, 3, or an object name.")


ACTIVE_OBJECT = choose_active_object()
OBJ_CFG = YCB_OBJECTS[ACTIVE_OBJECT]

FRAME_DIR = "output/frames"
os.makedirs(FRAME_DIR, exist_ok=True)

model = mujoco.MjModel.from_xml_path(SCENE_PATH)
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, height=480, width=640)

key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
obj_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, OBJ_CFG["body"])
bsk_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "basket")
bj = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "basket_freejoint")]
fj1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint1")
fj2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint2")

extra_kp = np.array([3000, 3000, 3000, 3000, 1500, 1000, 500])
extra_kd = np.array([100, 100, 100, 100, 50, 30, 15])

rng = np.random.default_rng(42)
frame_count = 0
RENDER_EVERY = 10  # render every N simulation steps


def save_frame(ep_num):
    global frame_count
    mujoco.mj_forward(model, data)
    renderer.update_scene(data)
    img = renderer.render()
    cv2.imwrite(f"{FRAME_DIR}/ep{ep_num}_frame_{frame_count:05d}.png",
                cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    frame_count += 1


def run_and_record(ep_num):
    """Run one episode and record frames."""
    global frame_count
    frame_count = 0

    # Setup scene using the same multi-object randomizer as evaluation.py
    gt_obj, gt_bsk = randomize_scene(model, data, rng, ACTIVE_OBJECT)
    for _ in range(50):
        save_frame(ep_num)

    # Perception
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera="overhead_cam")
    rgb = renderer.render().copy()
    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera="overhead_cam")
    depth = renderer.render().copy()
    renderer.disable_depth_rendering()

    obj_r = perceive_object(rgb, depth, model, data, "overhead_cam", OBJ_CFG["color_target"])
    bsk_r = perceive_object(rgb, depth, model, data, "overhead_cam", "red_basket")

    if obj_r is None:
        safe_q = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.0])
        q_start = data.qpos[:7].copy()
        for step in range(800):
            alpha = min(1.0, step / 500)
            alpha = 0.5 * (1 - np.cos(alpha * np.pi))
            q_des = q_start + alpha * (safe_q - q_start)
            data.ctrl[:7] = q_des
            data.qfrc_applied[:7] = extra_kp*(q_des-data.qpos[:7]) - extra_kd*data.qvel[:7]
            set_gripper(data, "open")
            mujoco.mj_step(model, data)
            if step % RENDER_EVERY == 0:
                save_frame(ep_num)
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera="overhead_cam")
        rgb = renderer.render().copy()
        renderer.enable_depth_rendering()
        renderer.update_scene(data, camera="overhead_cam")
        depth = renderer.render().copy()
        renderer.disable_depth_rendering()
        obj_r = perceive_object(rgb, depth, model, data, "overhead_cam", OBJ_CFG["color_target"])
        bsk_r = perceive_object(rgb, depth, model, data, "overhead_cam", "red_basket")

    gt_obj = data.xpos[obj_bid].copy()
    obj_pos = obj_r["position"] if obj_r else gt_obj.copy()
    if obj_r is None:
        obj_pos[2] += OBJ_CFG["flat_half_height"]
    bsk_pos = bsk_r["position"] if bsk_r else gt_bsk.copy()
    R_obj = obj_r["rotation"] if obj_r else None

    # Plan
    waypoints = compute_grasp_waypoints(obj_pos, bsk_pos, R_obj)
    joint_targets = compute_joint_targets(model, data, waypoints)

    if not all(jt["ik_success"] for jt in joint_targets):
        print(f"  Episode {ep_num}: IK failed, skipping")
        return False

    # Execute with frame recording
    for i, jt in enumerate(joint_targets):
        label = jt["label"]
        q_target = jt["q"]
        gripper = jt["gripper"]

        if "2_grasp" in label:
            # Combined descent + close
            q_start = data.qpos[:7].copy()
            q_grasp = q_target

            # Descent
            for step in range(600):
                alpha = min(1.0, step / 450.0)
                alpha = 0.5 * (1 - np.cos(alpha * np.pi))
                q_des = q_start + alpha * (q_grasp - q_start)
                data.ctrl[:7] = q_des
                data.qfrc_applied[:7] = extra_kp*(q_des-data.qpos[:7]) - extra_kd*data.qvel[:7]
                set_gripper(data, "open")
                mujoco.mj_step(model, data)
                if step % RENDER_EVERY == 0:
                    save_frame(ep_num)

            # Settle
            for step in range(300):
                data.ctrl[:7] = q_grasp
                data.qfrc_applied[:7] = extra_kp*(q_grasp-data.qpos[:7]) - extra_kd*data.qvel[:7]
                set_gripper(data, "open")
                mujoco.mj_step(model, data)
                if step % RENDER_EVERY == 0:
                    save_frame(ep_num)

            # Close
            for step in range(1200):
                data.ctrl[:7] = q_grasp
                data.qfrc_applied[:7] = extra_kp*(q_grasp-data.qpos[:7]) - extra_kd*data.qvel[:7]
                cf = min(100.0, 100.0 * step / 400.0)
                set_gripper(data, "close")
                data.qfrc_applied[model.jnt_dofadr[fj1_id]] = -cf
                data.qfrc_applied[model.jnt_dofadr[fj2_id]] = -cf
                mujoco.mj_step(model, data)
                if step % RENDER_EVERY == 0:
                    save_frame(ep_num)

            # Skip the close_gripper waypoint
            i_skip = True

        elif "3_close" in label:
            continue  # already handled in combined grasp

        else:
            # Normal waypoint
            q_start = data.qpos[:7].copy()
            for step in range(1500):
                alpha = min(1.0, step / 900.0)
                alpha = 0.5 * (1 - np.cos(alpha * np.pi))
                q_des = q_start + alpha * (q_target - q_start)
                data.ctrl[:7] = q_des
                data.qfrc_applied[:7] = extra_kp*(q_des-data.qpos[:7]) - extra_kd*data.qvel[:7]
                set_gripper(data, gripper)
                fj1_dof = model.jnt_dofadr[fj1_id]
                fj2_dof = model.jnt_dofadr[fj2_id]
                if gripper == "close":
                    data.qfrc_applied[fj1_dof] = -100.0
                    data.qfrc_applied[fj2_dof] = -100.0
                else:
                    data.qfrc_applied[fj1_dof] = 0.0
                    data.qfrc_applied[fj2_dof] = 0.0
                mujoco.mj_step(model, data)
                if step % RENDER_EVERY == 0:
                    save_frame(ep_num)

    # Check success
    mujoco.mj_forward(model, data)
    obj_f = data.xpos[obj_bid]
    bsk_f = data.xpos[bsk_bid]
    xy_d = np.linalg.norm(obj_f[:2] - bsk_f[:2])
    success = xy_d < 0.10 and obj_f[2] > bsk_f[2] - 0.02 and obj_f[2] < bsk_f[2] + 0.12

    # Save a few extra final frames
    for _ in range(30):
        save_frame(ep_num)

    print(f"  Episode {ep_num}: {'SUCCESS' if success else 'FAIL'} "
          f"({frame_count} frames, xy_dist={xy_d*1000:.0f}mm)")
    return success


# Run until we have 5 successful episodes
print(f"Recording 5 successful {ACTIVE_OBJECT} episodes...")
success_count = 0
ep = 0
while success_count < 5 and ep < 15:
    ok = run_and_record(ep)
    if ok:
        success_count += 1
    ep += 1

print(f"\nDone! {success_count} successful episodes recorded in {FRAME_DIR}/")
print(f"To compile video: ffmpeg -framerate 30 -i {FRAME_DIR}/ep<N>_frame_%05d.png "
      f"-c:v libx264 -pix_fmt yuv420p video.mp4")
