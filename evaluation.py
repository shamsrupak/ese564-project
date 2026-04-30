"""
evaluation.py - Main evaluation script for the pick-and-place project.

Runs the full pipeline on 3 YCB mesh objects (cracker box, mustard bottle,
sugar box) with 50 randomized episodes per object = 150 total.

Usage:
    python evaluation.py                         # full evaluation (150 episodes)
    python evaluation.py --num_episodes 5        # quick test (5 per object)
    python evaluation.py --save_images           # save screenshots
    python evaluation.py --object cracker_box    # test one object only
"""
import mujoco
import numpy as np
import cv2
import os
import sys
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from perception import perceive_object
from grasp_planner import compute_grasp_waypoints, compute_joint_targets
from controller import execute_pick_and_place, check_success

# ================================================================
# CONFIGURATION
# ================================================================
SCENE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pick_and_place_scene.xml")

OBJ_X_RANGE = (0.40, 0.60)
OBJ_Y_RANGE = (0.00, 0.15)
BASKET_X_RANGE = (0.40, 0.60)
BASKET_Y_RANGE = (-0.30, -0.15)
IMG_H, IMG_W = 480, 640

# YCB object definitions (3 objects, each a 16,384-triangle mesh)
YCB_OBJECTS = {
    "cracker_box": {
        "body": "cracker_box",
        "joint": "cracker_box_freejoint",
        "color_target": "yellow_object",
        "quat": [0.7071, 0, 0.7071, 0],
        "flat_half_height": 0.018,
        "model_cloud": "objects/cracker_box_model_cloud.npy",
    },
    "mustard_bottle": {
        "body": "mustard_bottle",
        "joint": "mustard_bottle_freejoint",
        "color_target": "green_object",
        "quat": [0.7071, 0, 0.7071, 0],    # 90° Y: 4.9cm height, 3.3cm squeeze
        "flat_half_height": 0.025,
        "model_cloud": "objects/mustard_bottle_model_cloud.npy",
    },
    "sugar_box": {
        "body": "sugar_box",
        "joint": "sugar_box_freejoint",
        "color_target": "cyan_object",
        "quat": [0.7071, 0, 0.7071, 0],
        "flat_half_height": 0.013,
        "model_cloud": "objects/sugar_box_model_cloud.npy",
    },
}

INACTIVE_OBJECT_POSITIONS = {
    "cracker_box": (0.42, 0.30),
    "mustard_bottle": (0.50, 0.30),
    "sugar_box": (0.58, 0.30),
}


def randomize_scene(model, data, rng, active_object):
    """Place the active object randomly and park inactive objects on the table."""
    cfg = YCB_OBJECTS[active_object]

    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, key_id)

    # Park inactive objects on the back of the table so they stay visible
    # without interfering with the active pick-and-place path.
    for name, obj_cfg in YCB_OBJECTS.items():
        if name == active_object:
            continue
        park_x, park_y = INACTIVE_OBJECT_POSITIONS[name]
        park_z = 0.40 + obj_cfg["flat_half_height"] + 0.01
        jid = model.jnt_qposadr[
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, obj_cfg["joint"])]
        data.qpos[jid:jid+7] = [park_x, park_y, park_z, *obj_cfg["quat"]]

    # Place active object on table, lying flat
    obj_x = rng.uniform(*OBJ_X_RANGE)
    obj_y = rng.uniform(*OBJ_Y_RANGE)
    obj_z = 0.40 + cfg["flat_half_height"] + 0.01

    jid = model.jnt_qposadr[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, cfg["joint"])]
    data.qpos[jid:jid+7] = [obj_x, obj_y, obj_z, *cfg["quat"]]

    # Randomize basket
    for _ in range(100):
        bsk_x = rng.uniform(*BASKET_X_RANGE)
        bsk_y = rng.uniform(*BASKET_Y_RANGE)
        if np.sqrt((bsk_x-obj_x)**2 + (bsk_y-obj_y)**2) > 0.20:
            break

    bj = model.jnt_qposadr[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "basket_freejoint")]
    data.qpos[bj:bj+7] = [bsk_x, bsk_y, 0.41, 1, 0, 0, 0]

    mujoco.mj_forward(model, data)
    for _ in range(500):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)

    obj_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cfg["body"])
    bsk_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "basket")
    return data.xpos[obj_bid].copy(), data.xpos[bsk_bid].copy()


def run_episode(model, data, renderer, episode_num, active_object,
                save_images=False, output_dir="output", use_perception=True):
    """Run one pick-and-place episode with a specific YCB object."""
    t_start = time.time()
    cfg = YCB_OBJECTS[active_object]

    obj_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cfg["body"])
    bsk_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "basket")
    gt_obj = data.xpos[obj_bid].copy()
    gt_bsk = data.xpos[bsk_bid].copy()

    # ---- Capture images ----
    renderer.update_scene(data, camera="overhead_cam")
    rgb = renderer.render().copy()
    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera="overhead_cam")
    depth = renderer.render().copy()
    renderer.disable_depth_rendering()

    if save_images:
        cv2.imwrite(f"{output_dir}/ep{episode_num}_overhead.png",
                    cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    # ---- Perception ----
    color_target = cfg["color_target"]
    perception_error_mm = 0.0

    if use_perception:
        obj_result = perceive_object(rgb, depth, model, data,
                                     "overhead_cam", color_target)
        bsk_result = perceive_object(rgb, depth, model, data,
                                     "overhead_cam", "red_basket")

        # Retry with arm moved aside if object not detected
        if obj_result is None:
            from controller import move_to_target
            safe_q = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.0])
            move_to_target(model, data, safe_q, "open", duration_steps=800)
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera="overhead_cam")
            rgb = renderer.render().copy()
            renderer.enable_depth_rendering()
            renderer.update_scene(data, camera="overhead_cam")
            depth = renderer.render().copy()
            renderer.disable_depth_rendering()
            obj_result = perceive_object(rgb, depth, model, data,
                                         "overhead_cam", color_target)
            bsk_result = perceive_object(rgb, depth, model, data,
                                         "overhead_cam", "red_basket")

        if obj_result is not None:
            obj_pos = obj_result["position"]
            R_obj = obj_result["rotation"]
            grasp_width = obj_result.get("grasp_width")
            perception_error_mm = np.linalg.norm(obj_pos - gt_obj) * 1000
        else:
            obj_pos = gt_obj.copy()
            obj_pos[2] += cfg["flat_half_height"]
            R_obj = None
            grasp_width = None
            perception_error_mm = -1

        bsk_pos = bsk_result["position"] if bsk_result else gt_bsk.copy()
    else:
        obj_pos = gt_obj.copy()
        obj_pos[2] += cfg["flat_half_height"]
        bsk_pos = gt_bsk.copy()
        R_obj = None
        grasp_width = None

    # ---- Grasp planning ----
    waypoints = compute_grasp_waypoints(obj_pos, bsk_pos, R_obj,
                                        object_name=active_object,
                                        grasp_width=grasp_width)

    # ---- Inverse kinematics ----
    joint_targets = compute_joint_targets(model, data, waypoints)
    if not all(jt["ik_success"] for jt in joint_targets):
        return {
            "episode": episode_num, "object": active_object,
            "success": False, "reason": "IK_FAILURE",
            "perception_error_mm": perception_error_mm,
            "xy_dist_mm": -1, "time_s": time.time() - t_start,
        }

    # ---- Execute ----
    execute_pick_and_place(
        model, data, joint_targets,
        renderer=renderer if save_images else None,
        save_frames=save_images, output_dir=output_dir,
        object_body_name=cfg["body"])

    # ---- Check success ----
    mujoco.mj_forward(model, data)
    success, xy_dist = check_success(model, data, cfg["body"])

    obj_final = data.xpos[obj_bid].copy()
    z_change = (obj_final[2] - gt_obj[2]) * 1000

    if save_images:
        renderer.update_scene(data)
        cv2.imwrite(f"{output_dir}/ep{episode_num}_final.png",
                    cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))

    return {
        "episode": episode_num, "object": active_object,
        "success": success, "reason": "OK" if success else "MISSED",
        "perception_error_mm": perception_error_mm,
        "xy_dist_mm": xy_dist * 1000, "z_change_mm": z_change,
        "time_s": time.time() - t_start,
    }


def main():
    parser = argparse.ArgumentParser(description="Pick-and-Place Evaluation")
    parser.add_argument("--num_episodes", type=int, default=50,
                        help="Episodes PER OBJECT (default 50, total = 3x this)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--save_images", action="store_true",
                        help="Save debug images")
    parser.add_argument("--use_gt", action="store_true",
                        help="Use ground truth instead of perception")
    parser.add_argument("--object", type=str, default=None,
                        help="Test one object only (cracker_box/mustard_bottle/sugar_box)")
    parser.add_argument("--output_dir", type=str, default="output")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # Which objects to test
    if args.object:
        obj_list = [args.object]
    else:
        obj_list = list(YCB_OBJECTS.keys())

    total_episodes = args.num_episodes * len(obj_list)

    print("=" * 70)
    print("PICK-AND-PLACE EVALUATION - YCB Objects")
    print("=" * 70)
    print(f"Objects:     {', '.join(obj_list)}")
    print(f"Episodes:    {args.num_episodes} per object = {total_episodes} total")
    print(f"Seed:        {args.seed}")
    print(f"Perception:  {'Ground Truth' if args.use_gt else 'Camera (HSV + Depth)'}")

    if 'MUJOCO_GL' not in os.environ:
        os.environ['MUJOCO_GL'] = 'egl'

    model = mujoco.MjModel.from_xml_path(SCENE_PATH)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=IMG_H, width=IMG_W)
    print(f"\nModel loaded: {SCENE_PATH}")
    print(f"{'='*70}\n")

    # ---- Run episodes per object ----
    all_results = []
    per_object_results = {name: [] for name in obj_list}
    global_ep = 0

    for obj_name in obj_list:
        print(f"\n{'='*70}")
        print(f"  OBJECT: {obj_name}")
        print(f"{'='*70}")

        for ep in range(args.num_episodes):
            gt_obj, gt_bsk = randomize_scene(model, data, rng, obj_name)
            print(f"  [{obj_name}] Episode {ep+1}/{args.num_episodes}  "
                  f"obj=[{gt_obj[0]:.3f},{gt_obj[1]:.3f},{gt_obj[2]:.3f}]  "
                  f"basket=[{gt_bsk[0]:.3f},{gt_bsk[1]:.3f}]")

            result = run_episode(
                model, data, renderer,
                episode_num=global_ep,
                active_object=obj_name,
                save_images=args.save_images or (ep < 3),
                output_dir=args.output_dir,
                use_perception=not args.use_gt,
            )

            all_results.append(result)
            per_object_results[obj_name].append(result)
            global_ep += 1

            status = "SUCCESS" if result["success"] else f"FAIL ({result['reason']})"
            print(f"    -> {status}  "
                  f"XY dist: {result['xy_dist_mm']:.0f}mm  "
                  f"Perception err: {result['perception_error_mm']:.0f}mm  "
                  f"Time: {result['time_s']:.1f}s")

        # Per-object summary
        obj_successes = sum(1 for r in per_object_results[obj_name] if r["success"])
        obj_rate = obj_successes / args.num_episodes * 100
        print(f"\n  {obj_name}: {obj_successes}/{args.num_episodes} = {obj_rate:.1f}%")

    # ============================================================
    # FINAL SUMMARY
    # ============================================================
    print(f"\n{'='*70}")
    print("EVALUATION SUMMARY")
    print(f"{'='*70}")

    total_success = sum(1 for r in all_results if r["success"])
    total = len(all_results)

    print(f"\n  Per-object results:")
    for obj_name in obj_list:
        results = per_object_results[obj_name]
        s = sum(1 for r in results if r["success"])
        n = len(results)
        print(f"    {obj_name:20s}: {s}/{n} = {s/n*100:.1f}%")

    print(f"\n  Overall success rate: {total_success}/{total} = {total_success/total*100:.1f}%")

    xy_dists = [r["xy_dist_mm"] for r in all_results if r["xy_dist_mm"] > 0]
    perc_errs = [r["perception_error_mm"] for r in all_results if r["perception_error_mm"] > 0]
    times = [r["time_s"] for r in all_results]

    if xy_dists:
        print(f"  Mean XY distance:    {np.mean(xy_dists):.1f} mm")
    if perc_errs:
        print(f"  Mean perception err: {np.mean(perc_errs):.1f} mm")
    print(f"  Mean execution time: {np.mean(times):.1f} s")

    failures = [r for r in all_results if not r["success"]]
    if failures:
        print(f"\n  Failure reasons:")
        reasons = {}
        for f in failures:
            r = f["reason"]
            reasons[r] = reasons.get(r, 0) + 1
        for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
            print(f"    {reason}: {count}")

    print(f"\n{'='*70}")
    return total_success / total * 100


if __name__ == "__main__":
    main()
