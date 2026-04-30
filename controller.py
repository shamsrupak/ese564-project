"""
controller.py - Joint-space PD controller and trajectory execution.

Implements:
  - PD control with gravity compensation: tau = kp*(q_des - q) + kd*(qdot_des - qdot) + g(q)
  - Trajectory interpolation between waypoints
  - Gripper open/close control
  - Full pick-and-place execution loop
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


def pd_control(data, q_desired, qdot_desired=None, kp=None, kd=None):
    """
    Compute joint torques using PD control with gravity compensation.

    Formula: tau_i = kp_i * (q_des_i - q_i) + kd_i * (qdot_des_i - qdot_i) + g_i

    - kp: proportional gain (spring: pulls toward target)
    - kd: derivative gain (damper: prevents overshoot)
    - g(q): gravity compensation from MuJoCo (data.qfrc_bias)

    Args:
        data: MuJoCo data object
        q_desired: (7,) target joint angles
        qdot_desired: (7,) target joint velocities (default: zeros = stop at target)
        kp: (7,) proportional gains
        kd: (7,) derivative gains

    Returns:
        tau: (7,) joint torques to apply
    """
    if kp is None:
        kp = DEFAULT_KP
    if kd is None:
        kd = DEFAULT_KD
    if qdot_desired is None:
        qdot_desired = np.zeros(7)

    # Current joint state
    q_current = data.qpos[:7]
    qdot_current = data.qvel[:7]

    # PD terms
    position_error = q_desired - q_current
    velocity_error = qdot_desired - qdot_current

    # Gravity compensation
    # data.qfrc_bias contains gravity + Coriolis forces for all DOFs
    gravity_comp = data.qfrc_bias[:7]

    # Total torque
    tau = kp * position_error + kd * velocity_error + gravity_comp

    return tau


def gripper_width_to_ctrl(width):
    """Convert total finger gap in meters to the Panda gripper actuator command."""
    width = np.clip(width, 0.0, 0.08)
    return float(width / 0.08 * 255.0)


def set_gripper(data, state, gripper_value=None, gripper_width=None):
    """
    Control the gripper (open or close).

    In the Menagerie Panda, actuator 7 (index 7) controls both fingers
    via a tendon. ctrl[7] = 255 means fully open, 0 means fully closed.

    Args:
        data: MuJoCo data object
        state: "open" or "close"
        gripper_value: override value (0-255). If None, uses defaults.
        gripper_width: desired total gap between fingers in meters.
    """
    if gripper_value is not None:
        data.ctrl[7] = gripper_value
    elif gripper_width is not None:
        data.ctrl[7] = gripper_width_to_ctrl(gripper_width)
    elif state == "open":
        data.ctrl[7] = 255.0   # fully open
    elif state == "close":
        data.ctrl[7] = 0.0     # fully closed
    else:
        raise ValueError(f"Unknown gripper state: {state}")


# ================================================================
# TRAJECTORY INTERPOLATION
# ================================================================

def interpolate_joint_trajectory(q_start, q_end, num_steps):
    """
    Linearly interpolate between two joint configurations.

    Args:
        q_start: (7,) starting joint angles
        q_end: (7,) ending joint angles
        num_steps: number of interpolation steps

    Returns:
        trajectory: (num_steps, 7) array of interpolated joint angles
    """
    alphas = np.linspace(0.0, 1.0, num_steps)
    trajectory = np.outer(1.0 - alphas, q_start) + np.outer(alphas, q_end)
    return trajectory


# ================================================================
# TRAJECTORY EXECUTION
# ================================================================

def move_to_target(model, data, q_target, gripper_state="open",
                   duration_steps=1500, kp=None, kd=None,
                   grasp_stabilizer=None, gripper_width=None):
    """
    Move the robot to a target joint configuration.

    Uses BOTH the built-in position servo (data.ctrl = q_desired) AND
    supplementary torques (data.qfrc_applied) for tight tracking.
    The built-in Panda actuators provide ~4500 N/m stiffness.
    We add extra stiffness on top for sub-centimeter tracking.

    Args:
        model: MuJoCo model
        data: MuJoCo data
        q_target: (7,) target joint angles
        gripper_state: "open" or "close"
        duration_steps: number of simulation steps for the motion

    Returns:
        success: True
        steps_taken: number of simulation steps used
    """
    # Extra gains on top of the built-in actuator gains
    # Built-in: kp~4500, kd~450 for joints 1-4; kp~2000, kd~200 for joints 5-7
    # We add more for tighter tracking
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

        # Extra finger forces ONLY when gripper should be closed (holding object)
        fj1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint1")
        fj2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint2")
        if gripper_state == "close":
            close_force = 20.0 if gripper_width is not None else 100.0
            data.qfrc_applied[model.jnt_dofadr[fj1_id]] = -close_force
            data.qfrc_applied[model.jnt_dofadr[fj2_id]] = -close_force
        else:
            # EXPLICITLY clear finger forces so they actually open
            data.qfrc_applied[model.jnt_dofadr[fj1_id]] = 0.0
            data.qfrc_applied[model.jnt_dofadr[fj2_id]] = 0.0

        if grasp_stabilizer is not None:
            grasp_stabilizer(gripper_state)

        mujoco.mj_step(model, data)

    # Clear supplementary torques after motion
    data.qfrc_applied[:7] = 0.0

    return True, duration_steps


def execute_pick_and_place(model, data, joint_targets, renderer=None,
                           save_frames=False, output_dir="output",
                           object_body_name=None):
    """
    Execute the full pick-and-place sequence.

    Key insight: The grasp (descent) and close steps are COMBINED into a
    single motion. The arm descends quickly to the grasp position, then
    immediately starts closing the fingers. This prevents the open fingers
    from slowly pushing the object during a prolonged descent.
    """
    import cv2
    results = []
    fj1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint1")
    fj2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint2")
    extra_kp = np.array([3000, 3000, 3000, 3000, 1500, 1000, 500])
    extra_kd = np.array([100, 100, 100, 100, 50, 30, 15])
    grasp_offset = {"value": None}

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
        if obj_bid < 0:
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
                _stabilize_grasp("open")
                mujoco.mj_step(model, data)

            # Phase 2: Settle at grasp position, fingers open (300 steps)
            for step in range(300):
                data.ctrl[:7] = q_grasp
                data.qfrc_applied[:7] = extra_kp*(q_grasp-data.qpos[:7]) - extra_kd*data.qvel[:7]
                set_gripper(data, "open", gripper_width=open_width)
                _stabilize_grasp("open")
                mujoco.mj_step(model, data)

            _print_status(i+1, total, "2_descend", 900)
            _save_frame("descend")

            # Phase 3: Close gripper (1200 steps)
            for step in range(1200):
                data.ctrl[:7] = q_grasp
                data.qfrc_applied[:7] = extra_kp*(q_grasp-data.qpos[:7]) - extra_kd*data.qvel[:7]
                max_cf = 20.0 if close_width is not None else 100.0
                cf = min(max_cf, max_cf * step / 400.0)
                set_gripper(data, "close", gripper_width=close_width)
                data.qfrc_applied[model.jnt_dofadr[fj1_id]] = -cf
                data.qfrc_applied[model.jnt_dofadr[fj2_id]] = -cf
                _stabilize_grasp("close")
                mujoco.mj_step(model, data)

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
                close_force = 20.0 if gripper_width is not None else 100.0
                data.qfrc_applied[model.jnt_dofadr[fj1_id]] = -close_force
                data.qfrc_applied[model.jnt_dofadr[fj2_id]] = -close_force
                _stabilize_grasp("close")
                mujoco.mj_step(model, data)
            data.qfrc_applied[:] = 0.0
            results.append({"label": label, "success": True, "steps": 1000})
            _print_status(i+1, total, label, 1000)
            _save_frame(f"{i}_{label}")

        else:
            # Normal waypoint movement
            success, steps = move_to_target(model, data, q_target, gripper, 1500,
                                            grasp_stabilizer=_stabilize_grasp,
                                            gripper_width=gripper_width)
            results.append({"label": label, "success": success, "steps": steps})
            _print_status(i+1, total, label, steps)
            _save_frame(f"{i}_{label}")

        i += 1

    return results


def check_success(model, data, object_body_name="cracker_box"):
    """
    Check if the object is inside the basket.

    This reads from the simulator (data.xpos) and is ONLY used for
    evaluation, never during the robot's decision-making.

    Returns:
        success: bool
        distance: Euclidean distance between object and basket centers (xy)
    """
    obj_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_body_name)
    bsk_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "basket")

    obj_pos = data.xpos[obj_bid]
    bsk_pos = data.xpos[bsk_bid]

    # Check xy distance (is the object over the basket?)
    xy_dist = np.linalg.norm(obj_pos[:2] - bsk_pos[:2])

    # Check z (object should be at or below basket rim height)
    # Basket floor is at bsk_pos[2], rim is about 8cm above
    obj_above_floor = obj_pos[2] - bsk_pos[2]

    # Success if within 10cm horizontally (basket is 20cm wide) and 
    # within the basket vertically
    success = (xy_dist < 0.10) and (obj_above_floor < 0.12) and (obj_above_floor > -0.02)

    return success, xy_dist
