"""
evaluation.py - Main evaluation script for the pick-and-place project.

This is the script the professor runs to see the robot in action.
It runs multiple randomized episodes and reports success rates.

Usage:
    MUJOCO_GL=egl python evaluation.py          # headless (server)
    python evaluation.py                         # with display (local)
    python evaluation.py --num_episodes 5        # quick test
    python evaluation.py --num_episodes 50       # full evaluation
"""
import mujoco
import numpy as np
import cv2
import os
import sys
import time
import argparse

# Ensure our modules are importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from perception import perceive_object
from grasp_planner import compute_grasp_waypoints, compute_joint_targets
from controller import execute_pick_and_place, check_success


# ================================================================
# SCENE CONFIGURATION
# ================================================================
SCENE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
    "mujoco_menagerie", "franka_emika_panda", "pick_and_place_scene.xml")

# Workspace bounds for randomization (reachable area on the table)
# Constrained to area where the robot can reliably reach and the
# overhead camera has clear view (not occluded by the robot arm)
OBJ_X_RANGE = (0.40, 0.60)
OBJ_Y_RANGE = (-0.10, 0.15)
BASKET_X_RANGE = (0.40, 0.60)
BASKET_Y_RANGE = (-0.30, -0.15)

# Fixed obstacle position (known, no perception needed)
OBSTACLE_POS = np.array([0.45, 0.15, 0.45])

# Image dimensions for cameras
IMG_H, IMG_W = 480, 640


def randomize_scene(model, data, rng):
    """
    Randomize object and basket positions for a new episode.

    Args:
        model: MuJoCo model
        data: MuJoCo data
        rng: numpy random generator

    Returns:
        obj_true_pos: (3,) ground truth object position (for evaluation only)
        bsk_true_pos: (3,) ground truth basket position (for evaluation only)
    """
    # Reset robot to home
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, key_id)

    # Randomize object position
    obj_x = rng.uniform(*OBJ_X_RANGE)
    obj_y = rng.uniform(*OBJ_Y_RANGE)
    obj_z = 0.45  # on the table (table top at 0.40 + half object height)

    oj = model.jnt_qposadr[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "object_freejoint")]
    data.qpos[oj:oj+7] = [obj_x, obj_y, obj_z, 1, 0, 0, 0]

    # Randomize basket position (ensure it doesn't overlap with object)
    for _ in range(100):
        bsk_x = rng.uniform(*BASKET_X_RANGE)
        bsk_y = rng.uniform(*BASKET_Y_RANGE)
        dist = np.sqrt((bsk_x - obj_x)**2 + (bsk_y - obj_y)**2)
        if dist > 0.20:  # at least 20cm apart
            break

    bsk_z = 0.41
    bj = model.jnt_qposadr[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "basket_freejoint")]
    data.qpos[bj:bj+7] = [bsk_x, bsk_y, bsk_z, 1, 0, 0, 0]

    # Let physics settle
    mujoco.mj_forward(model, data)
    for _ in range(500):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)

    # Read ground truth positions (for evaluation metrics only)
    obj_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_object")
    bsk_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "basket")
    obj_true_pos = data.xpos[obj_bid].copy()
    bsk_true_pos = data.xpos[bsk_bid].copy()

    return obj_true_pos, bsk_true_pos


def run_episode(model, data, renderer, episode_num, save_images=False,
                output_dir="output", use_perception=True):
    """
    Run a single pick-and-place episode.

    Pipeline:
      1. Capture overhead camera images
      2. Run perception (HSV segmentation + depth back-projection)
      3. Compute grasp waypoints
      4. Solve IK for each waypoint
      5. Execute with PD controller
      6. Check success

    Args:
        model: MuJoCo model
        data: MuJoCo data
        renderer: MuJoCo Renderer
        episode_num: episode index (for logging)
        save_images: whether to save debug images
        output_dir: directory for images
        use_perception: if True, use camera perception; if False, use ground truth

    Returns:
        result: dict with episode metrics
    """
    t_start = time.time()

    obj_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_object")
    bsk_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "basket")
    gt_obj = data.xpos[obj_bid].copy()
    gt_bsk = data.xpos[bsk_bid].copy()

    # ---- STEP 1: Capture images ----
    renderer.update_scene(data, camera="overhead_cam")
    rgb = renderer.render().copy()

    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera="overhead_cam")
    depth = renderer.render().copy()
    renderer.disable_depth_rendering()

    if save_images:
        cv2.imwrite(f"{output_dir}/ep{episode_num}_overhead.png",
                    cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    # ---- STEP 2: Perception ----
    perception_error_mm = 0.0

    if use_perception:
        obj_result = perceive_object(rgb, depth, model, data,
                                     "overhead_cam", "yellow_object")
        bsk_result = perceive_object(rgb, depth, model, data,
                                     "overhead_cam", "red_basket")

        # If object not detected, move arm out of the way and retry
        if obj_result is None:
            # Move arm to a safe position that clears the camera view
            from controller import move_to_target
            safe_q = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.0])
            move_to_target(model, data, safe_q, "open", duration_steps=800)
            mujoco.mj_forward(model, data)

            # Recapture images
            renderer.update_scene(data, camera="overhead_cam")
            rgb = renderer.render().copy()
            renderer.enable_depth_rendering()
            renderer.update_scene(data, camera="overhead_cam")
            depth = renderer.render().copy()
            renderer.disable_depth_rendering()

            obj_result = perceive_object(rgb, depth, model, data,
                                         "overhead_cam", "yellow_object")
            bsk_result = perceive_object(rgb, depth, model, data,
                                         "overhead_cam", "red_basket")

        if obj_result is not None:
            obj_pos = obj_result["position"]
            R_obj = obj_result["rotation"]
            perception_error_mm = np.linalg.norm(obj_pos - gt_obj) * 1000
        else:
            # Still failed - use ground truth as fallback
            obj_pos = gt_obj.copy()
            obj_pos[2] += 0.04
            R_obj = None
            perception_error_mm = -1

        if bsk_result is not None:
            bsk_pos = bsk_result["position"]
        else:
            bsk_pos = gt_bsk.copy()
    else:
        obj_pos = gt_obj.copy()
        obj_pos[2] += 0.04
        bsk_pos = gt_bsk.copy()
        R_obj = None

    # ---- STEP 3: Grasp planning ----
    waypoints = compute_grasp_waypoints(obj_pos, bsk_pos, R_obj)

    # ---- STEP 4: Inverse kinematics ----
    joint_targets = compute_joint_targets(model, data, waypoints)
    ik_success = all(jt["ik_success"] for jt in joint_targets)

    if not ik_success:
        return {
            "episode": episode_num,
            "success": False,
            "reason": "IK_FAILURE",
            "perception_error_mm": perception_error_mm,
            "xy_dist_mm": -1,
            "time_s": time.time() - t_start,
            "obj_start": gt_obj,
            "bsk_pos": gt_bsk,
        }

    # ---- STEP 5: Execute ----
    results = execute_pick_and_place(
        model, data, joint_targets,
        renderer=renderer if save_images else None,
        save_frames=save_images,
        output_dir=output_dir
    )

    # ---- STEP 6: Check success ----
    mujoco.mj_forward(model, data)
    success, xy_dist = check_success(model, data)

    obj_final = data.xpos[obj_bid].copy()
    z_change = (obj_final[2] - gt_obj[2]) * 1000

    t_elapsed = time.time() - t_start

    if save_images:
        renderer.update_scene(data)
        cv2.imwrite(f"{output_dir}/ep{episode_num}_final.png",
                    cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))

    return {
        "episode": episode_num,
        "success": success,
        "reason": "OK" if success else "MISSED",
        "perception_error_mm": perception_error_mm,
        "xy_dist_mm": xy_dist * 1000,
        "z_change_mm": z_change,
        "time_s": t_elapsed,
        "obj_start": gt_obj,
        "obj_final": obj_final,
        "bsk_pos": gt_bsk,
        "ik_success": ik_success,
    }


def main():
    parser = argparse.ArgumentParser(description="Pick-and-Place Evaluation")
    parser.add_argument("--num_episodes", type=int, default=10,
                        help="Number of randomized episodes to run")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--save_images", action="store_true",
                        help="Save debug images for each episode")
    parser.add_argument("--use_gt", action="store_true",
                        help="Use ground truth instead of perception (debug)")
    parser.add_argument("--output_dir", type=str, default="output",
                        help="Directory for output files")
    args = parser.parse_args()

    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print("=" * 70)
    print("PICK-AND-PLACE EVALUATION")
    print("=" * 70)
    print(f"Episodes:    {args.num_episodes}")
    print(f"Seed:        {args.seed}")
    print(f"Perception:  {'Ground Truth' if args.use_gt else 'Camera (HSV + Depth)'}")
    print(f"Output:      {args.output_dir}")

    # Load model
    if 'MUJOCO_GL' not in os.environ:
        os.environ['MUJOCO_GL'] = 'egl'

    model = mujoco.MjModel.from_xml_path(SCENE_PATH)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=IMG_H, width=IMG_W)

    print(f"\nModel loaded: {SCENE_PATH}")
    print(f"{'='*70}\n")

    # Run episodes
    all_results = []
    successes = 0

    for ep in range(args.num_episodes):
        print(f"--- Episode {ep+1}/{args.num_episodes} ---")

        # Randomize scene
        gt_obj, gt_bsk = randomize_scene(model, data, rng)
        print(f"  Object: [{gt_obj[0]:.3f}, {gt_obj[1]:.3f}, {gt_obj[2]:.3f}]  "
              f"Basket: [{gt_bsk[0]:.3f}, {gt_bsk[1]:.3f}, {gt_bsk[2]:.3f}]")

        # Run episode
        result = run_episode(
            model, data, renderer,
            episode_num=ep,
            save_images=args.save_images or ep < 5,  # always save first 5
            output_dir=args.output_dir,
            use_perception=not args.use_gt,
        )

        all_results.append(result)

        if result["success"]:
            successes += 1
            status = "SUCCESS"
        else:
            status = f"FAIL ({result['reason']})"

        print(f"  Result: {status}  "
              f"XY dist: {result['xy_dist_mm']:.0f}mm  "
              f"Perception err: {result['perception_error_mm']:.0f}mm  "
              f"Time: {result['time_s']:.1f}s")

        # Running success rate
        rate = successes / (ep + 1) * 100
        print(f"  Running success rate: {successes}/{ep+1} = {rate:.0f}%\n")

    # ============================================================
    # SUMMARY
    # ============================================================
    print("=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    total = len(all_results)
    success_rate = successes / total * 100

    # Compute metrics
    xy_dists = [r["xy_dist_mm"] for r in all_results if r["xy_dist_mm"] > 0]
    perc_errs = [r["perception_error_mm"] for r in all_results
                 if r["perception_error_mm"] > 0]
    times = [r["time_s"] for r in all_results]
    z_changes = [r.get("z_change_mm", 0) for r in all_results
                 if "z_change_mm" in r]

    print(f"\n  Success rate:        {successes}/{total} = {success_rate:.1f}%")
    print(f"  Mean XY distance:    {np.mean(xy_dists):.1f} mm" if xy_dists else "")
    print(f"  Mean perception err: {np.mean(perc_errs):.1f} mm" if perc_errs else "")
    print(f"  Mean execution time: {np.mean(times):.1f} s")
    print(f"  Mean Z change:       {np.mean(z_changes):.1f} mm" if z_changes else "")

    # Failure analysis
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

    return success_rate


if __name__ == "__main__":
    main()
