"""
grasp_planner.py - Grasp waypoint computation and inverse kinematics.

Given an object pose (from perception) and basket position, computes:
  1. A sequence of 6 Cartesian waypoints for top-down antipodal grasping
  2. Joint-angle targets via iterative Jacobian pseudo-inverse IK (HW2 concepts)

Waypoint sequence:
  1. Pre-grasp:  10cm above object, gripper open, aligned to object
  2. Grasp:      Lower to object height, gripper still open
  3. Close:      Same position, close gripper
  4. Lift:       Raise 15cm with object
  5. Above-bin:  Move above basket, 15cm up
  6. Release:    Lower into basket, open gripper
"""
import mujoco
import numpy as np


# ================================================================
# GRASP WAYPOINT COMPUTATION
# ================================================================

def compute_grasp_orientation(R_obj=None):
    """
    Compute the gripper orientation for a top-down grasp.

    For a top-down grasp:
      - Gripper z-axis points DOWN (into the table) -> [0, 0, -1]
      - Gripper x-axis controls which direction the fingers squeeze from
      - We align fingers to squeeze along the world Y-axis since our
        objects are always spawned upright with Y being the narrow axis

    Args:
        R_obj: (3,3) object orientation (unused for now; objects are upright)

    Returns:
        R_gripper: (3,3) rotation matrix for the gripper
    """
    # Gripper z-axis: pointing down
    gz = np.array([0.0, 0.0, -1.0])

    # The Panda gripper fingers open/close along the LOCAL y-axis of the
    # hand body. We want them to squeeze across the object's narrow side.
    # For our upright objects, squeezing along world-Y works well since
    # object Y-extent (4cm) < gripper max opening (8cm).
    # Gripper x-axis = world x-axis -> fingers squeeze along world y-axis
    gx = np.array([1.0, 0.0, 0.0])
    gy = np.cross(gz, gx)

    R_gripper = np.column_stack([gx, gy, gz])
    return R_gripper


def compute_grasp_waypoints(obj_position, basket_position, R_obj=None,
                            pre_grasp_height=0.12,
                            lift_height=0.18,
                            above_bin_height=0.18,
                            release_height=0.06):
    """
    Compute the 6 Cartesian waypoints for pick-and-place.

    IMPORTANT: All positions are for the "hand" body (wrist), NOT the fingertips.
    The fingertips are approximately 0.058m below the hand body.
    We account for this offset so the fingertips end up at the right height.

    Args:
        obj_position: (3,) object centroid from perception [x, y, z]
        basket_position: (3,) basket centroid from perception [x, y, z]
        R_obj: (3,3) object orientation from perception (optional)
        pre_grasp_height: meters above object for approach
        lift_height: meters to lift after grasping
        above_bin_height: meters above basket for transport
        release_height: meters above basket floor for release

    Returns:
        waypoints: list of dicts, each with:
            "position": (3,) target end-effector position (hand body)
            "orientation": (3,3) target end-effector rotation
            "gripper": "open" or "close"
            "label": description string
    """
    R_grip = compute_grasp_orientation(R_obj)

    # Offset from hand body to fingertip center
    # The hand body is 0.058m above where the fingertips actually grasp
    HAND_TO_FINGERTIP = 0.058

    # We want the fingertips at the object center height
    # Object surface (from perception) is approximately at obj_position[2]
    # The actual center is below the surface
    # Account for ~15mm controller tracking error by targeting deeper
    # Target fingertips at the UPPER portion of the object.
    # Aiming too low causes the object to pop downward when squeezed.
    # Aiming at the upper third gives reliable grip without slippage.
    # perception gives surface height; object center is ~0.04 below surface
    fingertip_grasp_z = obj_position[2] - 0.005

    # Hand position = fingertip position + offset
    hand_grasp_z = fingertip_grasp_z + HAND_TO_FINGERTIP

    waypoints = [
        {
            # First move hand high above the table center to avoid sweeping through objects
            "position": np.array([0.4, 0.0, 0.65]),
            "orientation": R_grip,
            "gripper": "open",
            "label": "0_approach_high",
        },
        {
            "position": np.array([obj_position[0], obj_position[1],
                                  hand_grasp_z + pre_grasp_height]),
            "orientation": R_grip,
            "gripper": "open",
            "label": "1_pre_grasp",
        },
        {
            "position": np.array([obj_position[0], obj_position[1],
                                  hand_grasp_z]),
            "orientation": R_grip,
            "gripper": "open",
            "label": "2_grasp",
        },
        {
            "position": np.array([obj_position[0], obj_position[1],
                                  hand_grasp_z]),
            "orientation": R_grip,
            "gripper": "close",
            "label": "3_close_gripper",
        },
        {
            "position": np.array([obj_position[0], obj_position[1],
                                  hand_grasp_z + lift_height]),
            "orientation": R_grip,
            "gripper": "close",
            "label": "4_lift",
        },
        {
            "position": np.array([basket_position[0], basket_position[1],
                                  basket_position[2] + HAND_TO_FINGERTIP + above_bin_height]),
            "orientation": R_grip,
            "gripper": "close",
            "label": "5_above_bin",
        },
        {
            "position": np.array([basket_position[0], basket_position[1],
                                  basket_position[2] + HAND_TO_FINGERTIP + release_height]),
            "orientation": R_grip,
            "gripper": "open",
            "label": "6_release",
        },
    ]

    return waypoints


# ================================================================
# INVERSE KINEMATICS (extends HW2 concepts)
# ================================================================

def get_ee_pose(model, data):
    """
    Get the current end-effector position and orientation.

    We use the midpoint between the two fingertips as the end-effector point.

    Args:
        model: MuJoCo model
        data: MuJoCo data (after mj_forward)

    Returns:
        ee_pos: (3,) end-effector position in world frame
        ee_rot: (3,3) end-effector rotation matrix
    """
    # Use the "hand" body as our reference
    hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    ee_pos = data.xpos[hand_id].copy()
    ee_rot = data.xmat[hand_id].reshape(3, 3).copy()
    return ee_pos, ee_rot


def compute_ee_jacobian(model, data):
    """
    Compute the 6x7 Jacobian for the end-effector.

    The Jacobian maps joint velocities to end-effector velocities:
      [v]     [J_pos]
      [w]  =  [J_rot]  * qdot

    where v is linear velocity (3,) and w is angular velocity (3,).

    Args:
        model: MuJoCo model
        data: MuJoCo data

    Returns:
        J: (6, 7) Jacobian matrix (position rows + orientation rows)
    """
    hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")

    # MuJoCo computes Jacobians for a point on a body
    # jacp = position Jacobian (3 x nv)
    # jacr = rotation Jacobian (3 x nv)
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacBody(model, data, jacp, jacr, hand_id)

    # Extract only the 7 arm joint columns (ignore finger joints)
    J = np.vstack([jacp[:, :7], jacr[:, :7]])  # (6, 7)
    return J


def orientation_error(R_target, R_current):
    """
    Compute the orientation error between two rotation matrices
    as a 3D rotation vector.

    Uses the axis-angle representation of the error rotation:
      R_error = R_target @ R_current^T
      Then extract the axis-angle from R_error.

    Args:
        R_target: (3,3) desired orientation
        R_current: (3,3) current orientation

    Returns:
        error: (3,) rotation error vector (axis * angle)
    """
    R_err = R_target @ R_current.T

    # Extract angle from trace: trace(R) = 1 + 2*cos(angle)
    trace = np.trace(R_err)
    cos_angle = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    angle = np.arccos(cos_angle)

    if angle < 1e-6:
        # Very small rotation, return zero
        return np.zeros(3)

    # Extract axis from the skew-symmetric part
    axis = np.array([
        R_err[2, 1] - R_err[1, 2],
        R_err[0, 2] - R_err[2, 0],
        R_err[1, 0] - R_err[0, 1],
    ]) / (2.0 * np.sin(angle))

    return axis * angle


def inverse_kinematics(model, data, target_pos, target_rot, q_init=None,
                       max_iters=200, pos_tol=0.005, rot_tol=0.05,
                       step_size=0.3):
    """
    Iterative inverse kinematics using Jacobian pseudo-inverse.
    Extends HW2 IK concepts to the full 7-DOF Panda.

    Algorithm:
      1. Compute current end-effector pose via forward kinematics
      2. Compute 6D error (position + orientation)
      3. Compute Jacobian J (6x7)
      4. Compute delta_q = J_pseudoinverse @ error
      5. Update: q = q + step_size * delta_q
      6. Clamp to joint limits
      7. Repeat until error < tolerance

    Args:
        model: MuJoCo model
        data: MuJoCo data
        target_pos: (3,) desired end-effector position
        target_rot: (3,3) desired end-effector orientation
        q_init: (7,) initial joint angles (None = use current)
        max_iters: maximum iterations
        pos_tol: position tolerance in meters
        rot_tol: rotation tolerance in radians
        step_size: IK step size (damping factor)

    Returns:
        q_result: (7,) joint angles that achieve the target
        success: bool, whether IK converged within tolerance
        pos_error: final position error in meters
    """
    # Save original state so we can restore it later
    original_qpos = data.qpos.copy()
    original_qvel = data.qvel.copy()

    # Initialize joint angles
    if q_init is not None:
        data.qpos[:7] = q_init.copy()
    q = data.qpos[:7].copy()

    for iteration in range(max_iters):
        # Set joints and run forward kinematics
        data.qpos[:7] = q
        mujoco.mj_forward(model, data)

        # Get current end-effector pose
        ee_pos, ee_rot = get_ee_pose(model, data)

        # Compute errors
        pos_err = target_pos - ee_pos                    # (3,)
        rot_err = orientation_error(target_rot, ee_rot)  # (3,)

        pos_err_norm = np.linalg.norm(pos_err)
        rot_err_norm = np.linalg.norm(rot_err)

        # Check convergence
        if pos_err_norm < pos_tol and rot_err_norm < rot_tol:
            # Restore original state
            data.qpos[:] = original_qpos
            data.qvel[:] = original_qvel
            mujoco.mj_forward(model, data)
            return q.copy(), True, pos_err_norm

        # Combine into 6D error
        error_6d = np.concatenate([pos_err, rot_err])  # (6,)

        # Compute Jacobian
        J = compute_ee_jacobian(model, data)  # (6, 7)

        # Damped pseudo-inverse for numerical stability
        # J_pinv = J^T (J J^T + lambda*I)^{-1}
        damping = 0.01
        JJT = J @ J.T + damping * np.eye(6)
        J_pinv = J.T @ np.linalg.inv(JJT)  # (7, 6)

        # Compute joint update
        delta_q = J_pinv @ error_6d  # (7,)

        # Update joints with step size
        q = q + step_size * delta_q

        # Clamp to joint limits
        for i in range(7):
            jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                        f"joint{i+1}")
            lo, hi = model.jnt_range[jnt_id]
            q[i] = np.clip(q[i], lo, hi)

    # Did not converge
    data.qpos[:] = original_qpos
    data.qvel[:] = original_qvel
    mujoco.mj_forward(model, data)
    return q.copy(), False, pos_err_norm


def compute_joint_targets(model, data, waypoints):
    """
    Convert a list of Cartesian waypoints to joint-angle targets using IK.

    Args:
        model: MuJoCo model
        data: MuJoCo data
        waypoints: list of dicts from compute_grasp_waypoints

    Returns:
        joint_targets: list of dicts with:
            "q": (7,) joint angles
            "gripper": "open" or "close"
            "label": description
            "ik_success": bool
            "ik_error": position error in mm
    """
    # Start from current robot configuration
    q_current = data.qpos[:7].copy()

    # Alternative starting configs for IK retry
    alt_configs = [
        np.array([0.0, 0.0, 0.0, -1.57, 0.0, 1.57, -0.785]),  # home
        np.array([0.0, -0.3, 0.0, -2.0, 0.0, 1.7, 0.0]),       # tucked
        np.array([0.3, 0.0, -0.3, -1.5, 0.0, 1.5, 0.5]),       # offset
    ]

    joint_targets = []

    for wp in waypoints:
        # Try IK with current config first
        q_result, success, pos_error = inverse_kinematics(
            model, data,
            target_pos=wp["position"],
            target_rot=wp["orientation"],
            q_init=q_current,
            max_iters=300,
            pos_tol=0.005,
            rot_tol=0.1,
        )

        # If failed, retry with alternative starting configs
        if not success:
            for alt_q in alt_configs:
                q_retry, s_retry, e_retry = inverse_kinematics(
                    model, data,
                    target_pos=wp["position"],
                    target_rot=wp["orientation"],
                    q_init=alt_q,
                    max_iters=500,
                    pos_tol=0.005,
                    rot_tol=0.1,
                )
                if s_retry:
                    q_result, success, pos_error = q_retry, s_retry, e_retry
                    break

        joint_targets.append({
            "q": q_result,
            "gripper": wp["gripper"],
            "label": wp["label"],
            "ik_success": success,
            "ik_error_mm": pos_error * 1000,
            "target_pos": wp["position"],
        })

        q_current = q_result

    return joint_targets
