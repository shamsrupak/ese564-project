"""
record_video.py - Record pick-and-place videos.

Usage:
    python record_video.py --object cracker_box
    python record_video.py --sequence

Compile one recorded episode with:
    ffmpeg -framerate 30 -i output/frames/ep0_frame_%05d.png -c:v libx264 -pix_fmt yuv420p video_ep0.mp4
"""
import argparse
import os
import sys
from pathlib import Path

if sys.platform != "darwin":
    os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from controller import MAX_GRIPPER_WIDTH, check_success, execute_pick_and_place, set_gripper
from evaluation import BASKET_X_RANGE, BASKET_Y_RANGE, YCB_OBJECTS, randomize_scene
from grasp_planner import compute_grasp_waypoints, compute_joint_targets
from perception import perceive_object


SCENE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pick_and_place_scene.xml")
FRAME_DIR = "output/frames"
RENDER_EVERY = 10
OBJECT_ORDER = list(YCB_OBJECTS.keys())
SEQUENCE_RELEASE_OFFSETS = {
    "cracker_box": np.array([-0.035, 0.025]),
    "mustard_bottle": np.array([0.035, 0.020]),
    "sugar_box": np.array([0.000, -0.030]),
}

extra_kp = np.array([3000, 3000, 3000, 3000, 1500, 1000, 500])
extra_kd = np.array([100, 100, 100, 100, 50, 30, 15])


def parse_args():
    parser = argparse.ArgumentParser(description="Record pick-and-place video frames.")
    parser.add_argument("--object", choices=OBJECT_ORDER, default=None,
                        help="Object to record in single-object mode.")
    parser.add_argument("--sequence", action="store_true",
                        help="Record one continuous episode that picks every object once.")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Successful single-object episodes to record.")
    parser.add_argument("--use_gt", action="store_true",
                        help="Use simulator pose instead of camera perception for planning.")
    parser.add_argument("--sequence_perception", action="store_true",
                        help="Use camera perception during --sequence instead of simulator poses.")
    return parser.parse_args()


def choose_active_object():
    print("Choose object to record:")
    for i, name in enumerate(OBJECT_ORDER, start=1):
        print(f"  {i}. {name}")

    while True:
        choice = input("Object [1-3]: ").strip()
        if choice in {"1", "2", "3"}:
            return OBJECT_ORDER[int(choice) - 1]
        if choice in YCB_OBJECTS:
            return choice
        print("Please enter 1, 2, 3, or an object name.")


args = parse_args()
ACTIVE_OBJECT = args.object
if not args.sequence and ACTIVE_OBJECT is None:
    ACTIVE_OBJECT = choose_active_object()

os.makedirs(FRAME_DIR, exist_ok=True)

model = mujoco.MjModel.from_xml_path(SCENE_PATH)
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, height=480, width=640)

bsk_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "basket")
basket_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "basket_freejoint")
basket_qadr = model.jnt_qposadr[basket_jid]
fj1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint1")
fj2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint2")
fj1_dof = model.jnt_dofadr[fj1_id]
fj2_dof = model.jnt_dofadr[fj2_id]

rng = np.random.default_rng(42)
frame_count = 0


def clear_episode_frames(ep_num):
    for old_frame in Path(FRAME_DIR).glob(f"ep{ep_num}_frame_*.png"):
        old_frame.unlink()


def initialize_sequence_scene():
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, key_id)

    sequence_slots = {
        "cracker_box": np.array([0.555, 0.075]),
        "mustard_bottle": np.array([0.490, 0.185]),
        "sugar_box": np.array([0.425, 0.075]),
    }

    for name in OBJECT_ORDER:
        cfg = YCB_OBJECTS[name]
        jitter = rng.uniform(-0.002, 0.002, size=2)
        obj_x, obj_y = sequence_slots[name] + jitter
        obj_z = 0.40 + cfg["flat_half_height"] + 0.01
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, cfg["joint"])
        qadr = model.jnt_qposadr[jid]
        data.qpos[qadr:qadr+7] = [obj_x, obj_y, obj_z, *cfg["quat"]]

    bsk_x = rng.uniform(*BASKET_X_RANGE)
    bsk_y = rng.uniform(*BASKET_Y_RANGE)
    data.qpos[basket_qadr:basket_qadr+7] = [bsk_x, bsk_y, 0.41, 1, 0, 0, 0]

    mujoco.mj_forward(model, data)
    for _ in range(500):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    return data.xpos[bsk_bid].copy()


def save_frame(ep_num):
    global frame_count
    mujoco.mj_forward(model, data)
    renderer.update_scene(data)
    img = renderer.render()
    cv2.imwrite(f"{FRAME_DIR}/ep{ep_num}_frame_{frame_count:05d}.png",
                cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    frame_count += 1


def save_hold(ep_num, frames):
    for _ in range(frames):
        save_frame(ep_num)


def move_to_q(ep_num, q_target, gripper, steps=1500, gripper_width=None,
              close_force=None):
    q_start = data.qpos[:7].copy()
    for step in range(steps):
        alpha = min(1.0, step / (steps * 0.6))
        alpha = 0.5 * (1.0 - np.cos(alpha * np.pi))
        q_des = q_start + alpha * (q_target - q_start)

        data.ctrl[:7] = q_des
        data.qfrc_applied[:7] = extra_kp*(q_des-data.qpos[:7]) - extra_kd*data.qvel[:7]
        set_gripper(data, gripper, gripper_width=gripper_width)

        if gripper == "close":
            force = close_force if close_force is not None else (20.0 if gripper_width is not None else 100.0)
            data.qfrc_applied[fj1_dof] = -force
            data.qfrc_applied[fj2_dof] = -force
        else:
            data.qfrc_applied[fj1_dof] = 0.0
            data.qfrc_applied[fj2_dof] = 0.0

        mujoco.mj_step(model, data)
        if step % RENDER_EVERY == 0:
            save_frame(ep_num)


def gentle_release(ep_num, q_target, gripper_width, grasp_stabilizer=None):
    release_steps = 1200
    start_width = float(np.clip(data.qpos[7] + data.qpos[8], 0.0, MAX_GRIPPER_WIDTH))
    target_width = float(gripper_width if gripper_width is not None else MAX_GRIPPER_WIDTH)
    target_width = float(np.clip(target_width, start_width, MAX_GRIPPER_WIDTH))

    for step in range(release_steps):
        alpha = min(1.0, step / (release_steps * 0.75))
        alpha = 0.5 * (1.0 - np.cos(alpha * np.pi))
        width = start_width + alpha * (target_width - start_width)

        data.ctrl[:7] = q_target
        data.qfrc_applied[:7] = extra_kp*(q_target-data.qpos[:7]) - extra_kd*data.qvel[:7]
        set_gripper(data, "open", gripper_width=width)
        data.qfrc_applied[fj1_dof] = 6.0
        data.qfrc_applied[fj2_dof] = 6.0
        if grasp_stabilizer is not None:
            grasp_stabilizer("open")

        mujoco.mj_step(model, data)
        if step % RENDER_EVERY == 0:
            save_frame(ep_num)


def capture_pose(active_object, gt_bsk, use_gt=False):
    cfg = YCB_OBJECTS[active_object]
    obj_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cfg["body"])
    gt_obj = data.xpos[obj_bid].copy()

    if use_gt:
        obj_pos = gt_obj.copy()
        obj_pos[2] += cfg["flat_half_height"]
        return obj_pos, gt_bsk.copy(), None, None

    renderer.update_scene(data, camera="overhead_cam")
    rgb = renderer.render().copy()
    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera="overhead_cam")
    depth = renderer.render().copy()
    renderer.disable_depth_rendering()

    obj_r = perceive_object(rgb, depth, model, data, "overhead_cam", cfg["color_target"])
    bsk_r = perceive_object(rgb, depth, model, data, "overhead_cam", "red_basket")

    if obj_r is None or not np.all(np.isfinite(obj_r["position"])):
        obj_pos = gt_obj.copy()
        obj_pos[2] += cfg["flat_half_height"]
        return obj_pos, gt_bsk.copy(), None, None

    bsk_pos = bsk_r["position"] if bsk_r and np.all(np.isfinite(bsk_r["position"])) else gt_bsk.copy()
    grasp_width = obj_r.get("grasp_width")
    if grasp_width is not None and not np.isfinite(grasp_width):
        grasp_width = None
    return obj_r["position"], bsk_pos, obj_r["rotation"], grasp_width


def plan_object(active_object, gt_bsk, use_gt=False, release_offset=None):
    obj_pos, bsk_pos, R_obj, grasp_width = capture_pose(active_object, gt_bsk, use_gt)
    if release_offset is not None:
        bsk_pos = bsk_pos.copy()
        bsk_pos[:2] += release_offset
    waypoints = compute_grasp_waypoints(obj_pos, bsk_pos, R_obj,
                                        object_name=active_object,
                                        grasp_width=grasp_width)
    joint_targets = compute_joint_targets(model, data, waypoints)
    return joint_targets


def execute_pick(ep_num, active_object, joint_targets):
    cfg = YCB_OBJECTS[active_object]
    obj_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cfg["body"])
    callback_state = {"step": 0}

    def record_controller_frame():
        callback_state["step"] += 1
        if callback_state["step"] % RENDER_EVERY == 0:
            save_frame(ep_num)

    execute_pick_and_place(
        model, data, joint_targets,
        object_body_name=cfg["body"],
        frame_callback=record_controller_frame,
        use_grasp_stabilizer=False,
    )

    mujoco.mj_forward(model, data)
    success, xy_d = check_success(model, data, cfg["body"])
    print(f"    {active_object}: {'SUCCESS' if success else 'FAIL'} xy={xy_d*1000:.0f}mm")
    return success


def run_single_episode(ep_num, active_object):
    global frame_count
    frame_count = 0
    clear_episode_frames(ep_num)

    _, gt_bsk = randomize_scene(model, data, rng, active_object)
    save_hold(ep_num, 50)

    joint_targets = plan_object(active_object, gt_bsk, args.use_gt)
    if not all(jt["ik_success"] for jt in joint_targets):
        print(f"  Episode {ep_num}: IK failed, skipping")
        return False

    ok = execute_pick(ep_num, active_object, joint_targets)
    save_hold(ep_num, 30)
    print(f"  Episode {ep_num}: {'SUCCESS' if ok else 'FAIL'} ({frame_count} frames)")
    return ok


def run_sequence_episode(ep_num):
    global frame_count
    frame_count = 0
    clear_episode_frames(ep_num)

    gt_bsk = initialize_sequence_scene()
    save_hold(ep_num, 50)

    successes = []
    use_gt_for_sequence = args.use_gt or not args.sequence_perception
    for active_object in OBJECT_ORDER:
        print(f"  Sequence pick: {active_object}")
        release_offset = SEQUENCE_RELEASE_OFFSETS.get(active_object)
        joint_targets = plan_object(active_object, gt_bsk, use_gt_for_sequence,
                                    release_offset=release_offset)
        if not all(jt["ik_success"] for jt in joint_targets):
            bad = [jt["label"] for jt in joint_targets if not jt["ik_success"]]
            print(f"    IK failed at {bad}")
            successes.append(False)
            break
        ok = execute_pick(ep_num, active_object, joint_targets)
        successes.append(ok)
        if not ok:
            break
        save_hold(ep_num, 20)

    save_hold(ep_num, 40)
    print(f"  Sequence episode {ep_num}: {sum(successes)}/{len(successes)} objects picked ({frame_count} frames)")
    return all(successes)


if args.sequence:
    print("Recording one sequence episode: " + " -> ".join(OBJECT_ORDER))
    run_sequence_episode(0)
else:
    print(f"Recording {args.episodes} successful {ACTIVE_OBJECT} episodes...")
    success_count = 0
    ep = 0
    while success_count < args.episodes and ep < args.episodes * 3:
        if run_single_episode(ep, ACTIVE_OBJECT):
            success_count += 1
        ep += 1
    print(f"\nDone! {success_count} successful episodes recorded in {FRAME_DIR}/")

print(f"To compile video: ffmpeg -framerate 30 -i {FRAME_DIR}/ep<N>_frame_%05d.png "
      f"-c:v libx264 -pix_fmt yuv420p video.mp4")
