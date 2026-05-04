"""
controller.py - Joint-space PD controller and trajectory execution.

Implements:
  - PD control with gravity compensation: tau = kp*(q_des - q) + kd*(qdot_des - qdot) + g(q)
  - Trajectory interpolation between waypoints
  - Gripper open/close control
  - Full pick-and-place execution loop
  - Grasp stabilizer for reliable object transport (Sowmya)
  - RRT collision-free motion planning integration (HW3)
"""
import mujoco
import numpy as np


# ================================================================
# PD CONTROLLER
# ================================================================

# Default PD gains for the Panda arm (7 joints)
# Higher gains for base joints (carry more weight), lower for wrist
DEFAULT_KP = np.array([600.0, 600.0, 600.0, 600.0, 250.0, 150.0, 50.0])
DEFAULT_KD = np.array([50.0, 50.0, 50.0, 50.0, 20.0, 20.0, 10.0])
MAX_GRIPPER_WIDTH = 0.10
WIDTH_GRASP_FORCE = 12.0


def apply_finger_close_force(model, data, finger_joint_ids, force, min_q=0.002):
    """Apply inward finger force without pushing slide joints past closed."""
    for joint_id in finger_joint_ids:
        qadr = model.jnt_qposadr[joint_id]
        dof = model.jnt_dofadr[joint_id]
        if force <= 0.0:
            data.qfrc_applied[dof] = 0.0
        elif data.qpos[qadr] > min_q or data.qvel[dof] > 0.0:
            data.qfrc_applied[dof] = -force
        else:
            data.qfrc_applied[dof] = 0.0


def pd_control(data, q_desired, qdot_desired=None, kp=None, kd=None):
    """
    Compute joint torques using PD control with gravity compensation.

    Formula: tau_i = kp_i * (q_des_i - q_i) + kd_i * (qdot_des_i - qdot_i) + g_i
    """
    if kp is None:
        kp = DEFAULT_KP
    if kd is None:
        kd = DEFAULT_KD
    if qdot_desired is None:
        qdot_desired = np.zeros(7)

    q_current = data.qpos[:7]
    qdot_current = data.qvel[:7]

    position_error = q_desired - q_current
    velocity_error = qdot_desired - qdot_current
    gravity_comp = data.qfrc_bias[:7]

    tau = kp * position_error + kd * velocity_error + gravity_comp
    return tau


def gripper_width_to_ctrl(width):
    """Convert total finger gap in meters to the Panda gripper actuator command."""
    width = np.clip(width, 0.0, MAX_GRIPPER_WIDTH)
    return float(width / MAX_GRIPPER_WIDTH * 255.0)


def set_gripper(data, state, gripper_value=None, gripper_width=None):
    """
    Control the gripper (open or close).

    In the Menagerie Panda, actuator 7 (index 7) controls both fingers
    via a tendon. ctrl[7] = 255 means fully open, 0 means fully closed.
    """
    if gripper_value is not None:
        data.ctrl[7] = gripper_value
    elif gripper_width is not None:
        data.ctrl[7] = gripper_width_to_ctrl(gripper_width)
    elif state == "open":
        data.ctrl[7] = 255.0
    elif state == "close":
        data.ctrl[7] = 0.0
    else:
        raise ValueError(f"Unknown gripper state: {state}")


# ================================================================
# TRAJECTORY INTERPOLATION
# ================================================================

def interpolate_joint_trajectory(q_start, q_end, num_steps):
    """Linearly interpolate between two joint configurations."""
    alphas = np.linspace(0.0, 1.0, num_steps)
    trajectory = np.outer(1.0 - alphas, q_start) + np.outer(alphas, q_end)
    return trajectory


# ================================================================
# TRAJECTORY EXECUTION
# ================================================================

def move_to_target(model, data, q_target, gripper_state="open",
                   duration_steps=1500, kp=None, kd=None,
                   grasp_stabilizer=None, gripper_width=None,
                   frame_callback=None):
    """
    Move the robot to a target joint configuration.

    Uses BOTH the built-in position servo (data.ctrl = q_desired) AND
    supplementary torques (data.qfrc_applied) for tight tracking.
    """
    extra_kp = np.array([3000, 3000, 3000, 3000, 1500, 1000, 500])
    extra_kd = np.array([100, 100, 100, 100, 50, 30, 15])

    q_start = data.qpos[:7].copy()

    for step in range(duration_steps):
        # Cosine interpolation for smooth motion
        alpha = min(1.0, step / (duration_steps * 0.6))
        alpha = 0.5 * (1.0 - np.cos(alpha * np.pi))

        q_desired = q_start + alpha * (q_target - q_start)

        # Built-in position servo
        data.ctrl[:7] = q_desired

        # Supplementary torques for tighter tracking
        q_err = q_desired - data.qpos[:7]
        qdot = data.qvel[:7]
        data.qfrc_applied[:7] = extra_kp * q_err - extra_kd * qdot

        # Gripper control
        set_gripper(data, gripper_state, gripper_width=gripper_width)

        # Extra finger forces
        fj1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint1")
        fj2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint2")
        if gripper_state == "close":
            close_force = WIDTH_GRASP_FORCE if gripper_width is not None else 100.0
            min_q = max(0.002, 0.5 * gripper_width - 0.002) if gripper_width is not None else 0.002
            apply_finger_close_force(model, data, (fj1_id, fj2_id), close_force, min_q=min_q)
        else:
            data.qfrc_applied[model.jnt_dofadr[fj1_id]] = 0.0
            data.qfrc_applied[model.jnt_dofadr[fj2_id]] = 0.0

        if grasp_stabilizer is not None:
            grasp_stabilizer(gripper_state)

        mujoco.mj_step(model, data)
        if frame_callback is not None:
            frame_callback()

    # Clear supplementary torques after motion
    data.qfrc_applied[:7] = 0.0

    return True, duration_steps


def execute_pick_and_place(model, data, joint_targets, renderer=None,
                           save_frames=False, output_dir="output",
                           object_body_name=None, frame_callback=None,
                           use_grasp_stabilizer=False):
    """
    Execute the full pick-and-place sequence.

    Key insight: The grasp (descent) and close steps are COMBINED into a
    single motion. The arm descends quickly to the grasp position, then
    immediately starts closing the fingers.

    Includes:
      - Optional grasp stabilizer (Sowmya): applies forces to keep object
        attached to hand during transport
      - RRT collision avoidance (HW3): plans around the obstacle for
        normal waypoint movements
    """
    import cv2
    results = []
    fj1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint1")
    fj2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint2")
    extra_kp = np.array([3000, 3000, 3000, 3000, 1500, 1000, 500])
    extra_kd = np.array([100, 100, 100, 100, 50, 30, 15])
    grasp_offset = {"value": None}

    # ---- Grasp stabilizer setup (Sowmya) ----
    if object_body_name is not None:
        obj_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_body_name)
        obj_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                    f"{object_body_name}_freejoint")
        obj_dof = model.jnt_dofadr[obj_jid]
    else:
        obj_bid = -1
        obj_dof = -1

    hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")

    def _stabilize_grasp(gripper_state):
        """Apply forces to keep the grasped object locked to the hand. (Sowmya)"""
        if not use_grasp_stabilizer or obj_bid < 0:
            return

        if gripper_state != "close":
            data.qfrc_applied[obj_dof:obj_dof+6] = 0.0
            grasp_offset["value"] = None
            return

        hand_pos = data.xpos[hand_id].copy()
        obj_pos = data.xpos[obj_bid].copy()

        if grasp_offset["value"] is None:
            xy_dist = np.linalg.norm(hand_pos[:2] - obj_pos[:2])
            z_gap = hand_pos[2] - obj_pos[2]
            if xy_dist > 0.12 or z_gap < 0.02 or z_gap > 0.20:
                return
            grasp_offset["value"] = obj_pos - hand_pos

        target = hand_pos + grasp_offset["value"]
        obj_vel = data.qvel[obj_dof:obj_dof+3]
        force = 250.0 * (target - obj_pos) - 25.0 * obj_vel
        force = np.clip(force, -60.0, 60.0)
        data.qfrc_applied[obj_dof:obj_dof+3] += force

    active_stabilizer = _stabilize_grasp if use_grasp_stabilizer else None

    # ---- Helper functions ----
    def _save_frame(step_name):
        if save_frames and renderer is not None:
            mujoco.mj_forward(model, data)
            renderer.update_scene(data)
            img = renderer.render()
            cv2.imwrite(f"{output_dir}/exec_{step_name}.png",
                        cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    def _print_status(step_num, total, label, steps):
        mujoco.mj_forward(model, data)
        ee = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")]
        obj_z = data.xpos[obj_bid][2] if obj_bid >= 0 else float("nan")
        fq = data.qpos[7:9]
        print(f"  [{step_num}/{total}] {label:20s} OK  steps={steps:5d}  "
              f"ee=[{ee[0]:.3f},{ee[1]:.3f},{ee[2]:.3f}] "
              f"obj_z={obj_z:.3f} fing=[{fq[0]:.3f},{fq[1]:.3f}]")

    # ---- Main execution loop ----
    i = 0
    total = len(joint_targets)

    while i < total:
        jt = joint_targets[i]
        label = jt["label"]
        q_target = jt["q"]
        gripper = jt["gripper"]
        gripper_width = jt.get("gripper_width")

        # COMBINED GRASP: merge "2_grasp" + "3_close_gripper" into one motion
        if "2_grasp" in label:
            q_grasp = q_target
            q_start = data.qpos[:7].copy()
            open_width = gripper_width
            close_width = joint_targets[i + 1].get("gripper_width") if i + 1 < total else gripper_width

            # Phase 1: Descent (600 steps)
            for step in range(600):
                alpha = min(1.0, step / 450.0)
                alpha = 0.5 * (1.0 - np.cos(alpha * np.pi))
                q_des = q_start + alpha * (q_grasp - q_start)
                data.ctrl[:7] = q_des
                data.qfrc_applied[:7] = extra_kp*(q_des-data.qpos[:7]) - extra_kd*data.qvel[:7]
                set_gripper(data, "open", gripper_width=open_width)
                if active_stabilizer is not None:
                    active_stabilizer("open")
                mujoco.mj_step(model, data)
                if frame_callback is not None:
                    frame_callback()

            # Phase 2: Settle at grasp position, fingers open (300 steps)
            for step in range(300):
                data.ctrl[:7] = q_grasp
                data.qfrc_applied[:7] = extra_kp*(q_grasp-data.qpos[:7]) - extra_kd*data.qvel[:7]
                set_gripper(data, "open", gripper_width=open_width)
                if active_stabilizer is not None:
                    active_stabilizer("open")
                mujoco.mj_step(model, data)
                if frame_callback is not None:
                    frame_callback()

            _print_status(i+1, total, "2_descend", 900)
            _save_frame("descend")

            # Phase 3: Close gripper (1200 steps)
            for step in range(1200):
                data.ctrl[:7] = q_grasp
                data.qfrc_applied[:7] = extra_kp*(q_grasp-data.qpos[:7]) - extra_kd*data.qvel[:7]
                max_cf = WIDTH_GRASP_FORCE if close_width is not None else 100.0
                cf = min(max_cf, max_cf * step / 400.0)
                set_gripper(data, "close", gripper_width=close_width)
                min_q = max(0.002, 0.5 * close_width - 0.002) if close_width is not None else 0.002
                apply_finger_close_force(model, data, (fj1_id, fj2_id), cf, min_q=min_q)
                if active_stabilizer is not None:
                    active_stabilizer("close")
                mujoco.mj_step(model, data)
                if frame_callback is not None:
                    frame_callback()

            _print_status(i+1, total, "3_close", 1200)
            _save_frame("close")

            results.append({"label": "grasp+close", "success": True, "steps": 2100})
            i += 2  # skip the separate close_gripper waypoint
            continue

        elif "close" in label:
            # Standalone close (fallback, normally skipped by combined above)
            for step in range(1000):
                data.ctrl[:7] = q_target
                data.qfrc_applied[:7] = extra_kp*(q_target-data.qpos[:7]) - extra_kd*data.qvel[:7]
                set_gripper(data, "close", gripper_width=gripper_width)
                close_force = WIDTH_GRASP_FORCE if gripper_width is not None else 100.0
                min_q = max(0.002, 0.5 * gripper_width - 0.002) if gripper_width is not None else 0.002
                apply_finger_close_force(model, data, (fj1_id, fj2_id), close_force, min_q=min_q)
                if active_stabilizer is not None:
                    active_stabilizer("close")
                mujoco.mj_step(model, data)
                if frame_callback is not None:
                    frame_callback()
            data.qfrc_applied[:] = 0.0
            results.append({"label": label, "success": True, "steps": 1000})
            _print_status(i+1, total, label, 1000)
            _save_frame(f"{i}_{label}")

        elif "release" in label and gripper == "open":
            # Gentle drop release: hold the hand above the basket and ramp the
            # finger gap open so the object falls free without a snap motion.
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
                open_assist = 0.0
                data.qfrc_applied[model.jnt_dofadr[fj1_id]] = open_assist
                data.qfrc_applied[model.jnt_dofadr[fj2_id]] = open_assist
                if active_stabilizer is not None:
                    active_stabilizer("open")
                mujoco.mj_step(model, data)
                if frame_callback is not None:
                    frame_callback()

            data.qfrc_applied[:] = 0.0
            results.append({"label": label, "success": True, "steps": release_steps})
            _print_status(i+1, total, label, release_steps)
            _save_frame(f"{i}_{label}")

        else:
            # Normal waypoint: use RRT for collision-free path (HW3)
            from motion_planner import plan_path

            q_current = data.qpos[:7].copy()
            path = plan_path(q_current, q_target, model, data)

            if path is None:
                print(f"  [{i+1}/{total}] {label:20s} FAILED  RRT could not find a safe path")
                results.append({"label": label, "success": False, "steps": 0})
                return results

            if len(path) > 2:
                # RRT found a multi-segment path - execute each segment
                total_steps = 0
                for seg_idx in range(len(path) - 1):
                    seg_steps = max(500, 1500 // (len(path) - 1))
                    move_to_target(model, data, path[seg_idx + 1], gripper, seg_steps,
                                   grasp_stabilizer=active_stabilizer,
                                   gripper_width=gripper_width,
                                   frame_callback=frame_callback)
                    total_steps += seg_steps
                results.append({"label": label, "success": True, "steps": total_steps})
            else:
                # Direct path is collision-free (most common case)
                success, steps = move_to_target(model, data, q_target, gripper, 1500,
                                                grasp_stabilizer=active_stabilizer,
                                                gripper_width=gripper_width,
                                                frame_callback=frame_callback)
                results.append({"label": label, "success": success, "steps": steps})

            _print_status(i+1, total, label, results[-1]["steps"])
            _save_frame(f"{i}_{label}")

        i += 1

    return results


def check_success(model, data, object_body_name="cracker_box"):
    """
    Check if the object is inside the basket.

    This reads from the simulator (data.xpos) and is ONLY used for
    evaluation, never during the robot's decision-making.
    """
    obj_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_body_name)
    bsk_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "basket")

    obj_pos = data.xpos[obj_bid]
    bsk_pos = data.xpos[bsk_bid]

    xy_dist = np.linalg.norm(obj_pos[:2] - bsk_pos[:2])
    obj_above_floor = obj_pos[2] - bsk_pos[2]

    success = (xy_dist < 0.10) and (obj_above_floor < 0.12) and (obj_above_floor > -0.02)
    return success, xy_dist
