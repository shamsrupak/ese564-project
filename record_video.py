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
from evaluation import BASKET_X_RANGE, BASKET_Y_RANGE, OBJ_X_RANGE, OBJ_Y_RANGE, YCB_OBJECTS, randomize_scene
from grasp_planner import compute_grasp_waypoints, compute_joint_targets
from perception import perceive_object


SCENE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pick_and_place_scene.xml")
TABLE_TOP_Z = 0.40
FRAME_DIR = "output/frames"
RENDER_EVERY = 10
OBJECT_ORDER = list(YCB_OBJECTS.keys())
SEQUENCE_OBJECT_X_RANGE = OBJ_X_RANGE
SEQUENCE_OBJECT_Y_RANGE = OBJ_Y_RANGE
SEQUENCE_MIN_OBJECT_SEPARATION = 0.070
SEQUENCE_OBSTACLE_X_RANGE = (0.44, 0.56)
SEQUENCE_OBSTACLE_Y_RANGE = (-0.14, -0.07)
OBSTACLE_Z = 0.49
SEQUENCE_RELEASE_OFFSETS = {
    "cracker_box": np.array([-0.035, 0.025]),
    "mustard_bottle": np.array([0.000, 0.000]),
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
                        help="Successful episodes to record.")
    parser.add_argument("--use_gt", action="store_true",
                        help="Use simulator pose instead of camera perception for planning.")
    parser.add_argument("--sequence_perception", action="store_true",
                        help="Use camera perception during --sequence instead of simulator poses.")
    view_group = parser.add_mutually_exclusive_group()
    view_group.add_argument("--top_view", action="store_true",
                            help="Record frames from the overhead camera.")
    view_group.add_argument("--both_views", action="store_true",
                            help="Record both the normal view and the overhead camera.")
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
obstacle_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "obstacle")
fj1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint1")
fj2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint2")
fj1_dof = model.jnt_dofadr[fj1_id]
fj2_dof = model.jnt_dofadr[fj2_id]

rng = np.random.default_rng(42)
frame_count = 0


def record_views():
    if args.both_views:
        return (("main", None), ("top", "overhead_cam"))
    if args.top_view:
        return (("top", "overhead_cam"),)
    return (("main", None),)


def frame_path(ep_num, view_name, frame_idx):
    if view_name == "main":
        return f"{FRAME_DIR}/ep{ep_num}_frame_{frame_idx:05d}.png"
    return f"{FRAME_DIR}/ep{ep_num}_{view_name}_frame_{frame_idx:05d}.png"


def clamp_tabletop_object_height(obj_pos, cfg):
    obj_pos = obj_pos.copy()
    top_z = TABLE_TOP_Z + 2.0 * cfg["flat_half_height"]
    obj_pos[2] = np.clip(obj_pos[2], top_z - 0.005, top_z + 0.005)
    return obj_pos


def clear_episode_frames(ep_num):
    for old_frame in Path(FRAME_DIR).glob(f"ep{ep_num}_frame_*.png"):
        old_frame.unlink()
    for old_frame in Path(FRAME_DIR).glob(f"ep{ep_num}_top_frame_*.png"):
        old_frame.unlink()


def sample_sequence_object_positions():
    object_positions = {}
    for name in OBJECT_ORDER:
        for _ in range(200):
            obj_x = rng.uniform(*SEQUENCE_OBJECT_X_RANGE)
            obj_y = rng.uniform(*SEQUENCE_OBJECT_Y_RANGE)
            candidate = np.array([obj_x, obj_y])
            if all(np.linalg.norm(candidate - np.array(pos)) > SEQUENCE_MIN_OBJECT_SEPARATION
                   for pos in object_positions.values()):
                object_positions[name] = (obj_x, obj_y)
                break
        else:
            safe_slots = np.array([
                [0.555, 0.075],
                [0.490, 0.185],
                [0.425, 0.075],
            ])
            safe_slots = safe_slots[rng.permutation(len(safe_slots))]
            return {
                obj_name: tuple(safe_slots[idx] + rng.uniform(-0.004, 0.004, size=2))
                for idx, obj_name in enumerate(OBJECT_ORDER)
            }
    return object_positions


def sample_sequence_basket_position(object_positions):
    for _ in range(200):
        bsk_x = rng.uniform(*BASKET_X_RANGE)
        bsk_y = rng.uniform(*BASKET_Y_RANGE)
        basket_xy = np.array([bsk_x, bsk_y])
        if all(np.linalg.norm(basket_xy - np.array(pos)) > 0.20
               for pos in object_positions.values()):
            return bsk_x, bsk_y
    return rng.uniform(*BASKET_X_RANGE), rng.uniform(*BASKET_Y_RANGE)


def randomize_sequence_obstacle():
    if obstacle_bid < 0:
        return
    model.body_pos[obstacle_bid] = [
        rng.uniform(*SEQUENCE_OBSTACLE_X_RANGE),
        rng.uniform(*SEQUENCE_OBSTACLE_Y_RANGE),
        OBSTACLE_Z,
    ]


def initialize_sequence_scene():
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    randomize_sequence_obstacle()
    object_positions = sample_sequence_object_positions()

    for name in OBJECT_ORDER:
        cfg = YCB_OBJECTS[name]
        obj_x, obj_y = object_positions[name]
        obj_z = 0.40 + cfg["flat_half_height"] + 0.01
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, cfg["joint"])
        qadr = model.jnt_qposadr[jid]
        data.qpos[qadr:qadr+7] = [obj_x, obj_y, obj_z, *cfg["quat"]]

    bsk_x, bsk_y = sample_sequence_basket_position(object_positions)
    data.qpos[basket_qadr:basket_qadr+7] = [bsk_x, bsk_y, 0.41, 1, 0, 0, 0]

    mujoco.mj_forward(model, data)
    for _ in range(500):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    print("    init:",
          " ".join(f"{name}=[{object_positions[name][0]:.3f},{object_positions[name][1]:.3f}]"
                   for name in OBJECT_ORDER),
          f"basket=[{bsk_x:.3f},{bsk_y:.3f}]",
          f"wall=[{model.body_pos[obstacle_bid][0]:.3f},{model.body_pos[obstacle_bid][1]:.3f}]")
    return data.xpos[bsk_bid].copy()


def save_frame(ep_num):
    global frame_count
    mujoco.mj_forward(model, data)
    for view_name, camera_name in record_views():
        if camera_name is None:
            renderer.update_scene(data)
        else:
            renderer.update_scene(data, camera=camera_name)
        img = renderer.render()
        cv2.imwrite(frame_path(ep_num, view_name, frame_count),
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

    obj_pos = clamp_tabletop_object_height(obj_r["position"], cfg)
    bsk_pos = bsk_r["position"] if bsk_r and np.all(np.isfinite(bsk_r["position"])) else gt_bsk.copy()
    grasp_width = obj_r.get("grasp_width")
    if grasp_width is not None and not np.isfinite(grasp_width):
        grasp_width = None
    return obj_pos, bsk_pos, obj_r["rotation"], grasp_width


def plan_object(active_object, gt_bsk, use_gt=False, release_offset=None,
                sequence_mode=False):
    obj_pos, bsk_pos, R_obj, grasp_width = capture_pose(active_object, gt_bsk, use_gt)
    if release_offset is not None:
        bsk_pos = bsk_pos.copy()
        bsk_pos[:2] += release_offset
    waypoint_kwargs = {
        "object_name": active_object,
        "grasp_width": grasp_width,
    }
    if sequence_mode and obstacle_bid >= 0:
        waypoint_kwargs.update({
            "wall_y": float(model.body_pos[obstacle_bid][1]),
            "above_bin_height": 0.24,
            "wall_clearance_z": 0.78,
        })
    waypoints = compute_grasp_waypoints(obj_pos, bsk_pos, R_obj, **waypoint_kwargs)
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
        rrt_max_iterations=800,
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
                                    release_offset=release_offset,
                                    sequence_mode=True)
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
    print("Recording successful sequence episodes: " + " -> ".join(OBJECT_ORDER))
    success_count = 0
    attempt = 0
    max_attempts = args.episodes * 5
    while success_count < args.episodes and attempt < max_attempts:
        print(f"\nSequence attempt {attempt + 1} -> output episode {success_count}")
        if run_sequence_episode(success_count):
            success_count += 1
        attempt += 1
    print(f"\nDone! {success_count} successful sequence episodes recorded in {FRAME_DIR}/")
else:
    print(f"Recording {args.episodes} successful {ACTIVE_OBJECT} episodes...")
    success_count = 0
    ep = 0
    while success_count < args.episodes and ep < args.episodes * 3:
        if run_single_episode(ep, ACTIVE_OBJECT):
            success_count += 1
        ep += 1
    print(f"\nDone! {success_count} successful episodes recorded in {FRAME_DIR}/")

if args.both_views:
    print(f"To compile normal view: ffmpeg -framerate 30 -i {FRAME_DIR}/ep<N>_frame_%05d.png "
          f"-c:v libx264 -pix_fmt yuv420p video.mp4")
    print(f"To compile top view:    ffmpeg -framerate 30 -i {FRAME_DIR}/ep<N>_top_frame_%05d.png "
          f"-c:v libx264 -pix_fmt yuv420p video_top.mp4")
elif args.top_view:
    print(f"To compile top-view video: ffmpeg -framerate 30 -i {FRAME_DIR}/ep<N>_top_frame_%05d.png "
          f"-c:v libx264 -pix_fmt yuv420p video_top.mp4")
else:
    print(f"To compile video: ffmpeg -framerate 30 -i {FRAME_DIR}/ep<N>_frame_%05d.png "
          f"-c:v libx264 -pix_fmt yuv420p video.mp4")
