"""
motion_planner.py - RRT collision-free motion planning for the Panda.

Directly extends our HW3 RRT implementation to plan in the Panda's 7-DOF
configuration space while avoiding collisions with the obstacle and basket.

Functions adapted from HW3:
  - sample_point (HW3 Problem 2.1)
  - get_nearest (HW3 Problem 2.2)
  - extend (HW3 Problem 2.3)
  - is_collision_free (HW3 Problem 2.4)
  - rrt / panda_rrt (HW3 Problem 2.5)
  - smooth_path (HW3 Problem 2.6)
"""
import mujoco
import numpy as np


def get_joint_positions(model, data):
    """Get the current 7-DOF arm joint positions."""
    return data.qpos[:7].copy()


def set_joint_positions(model, data, q):
    """Set the 7-DOF arm joint positions and update forward kinematics."""
    data.qpos[:7] = q
    mujoco.mj_forward(model, data)


def sample_point(model, q_goal, rng, epsilon=0.15):
    """
    Sample a random configuration in joint space.
    (HW3 Problem 2.1: sample_point)

    With probability epsilon, return the goal configuration (goal bias).
    Otherwise, sample uniformly within joint limits.
    """
    if rng.random() < epsilon:
        return q_goal.copy()

    # Get joint limits from the model
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i+1}")
                 for i in range(7)]
    joint_ranges = np.array([model.jnt_range[jid] for jid in joint_ids])
    q_sample = rng.uniform(joint_ranges[:, 0], joint_ranges[:, 1])
    return q_sample


def get_nearest(nodes, q):
    """
    Find the nearest node in the tree to the query point.
    (HW3 Problem 2.2: get_nearest)

    Uses Euclidean distance in joint space.
    """
    distances = np.linalg.norm(nodes - q, axis=1)
    nearest_idx = np.argmin(distances)
    return nodes[nearest_idx]


def extend(q_sample, q_near, step_size=0.15):
    """
    Extend from q_near toward q_sample by at most step_size.
    (HW3 Problem 2.3: extend)
    """
    dist = np.linalg.norm(q_sample - q_near)
    if dist <= step_size:
        return q_sample.copy()
    direction = (q_sample - q_near) / dist
    return q_near + step_size * direction


def check_robot_obstacle_collision(model, data):
    """
    Check if the robot is colliding with the obstacle or basket.

    Returns True if there IS a collision (path is NOT free).
    Filters contacts to only check robot-vs-obstacle pairs,
    ignoring expected contacts like object-on-table.
    """
    # Get obstacle geom id
    obstacle_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "obstacle_geom")

    # Get robot geom ids (all geoms belonging to robot bodies link0-hand)
    robot_body_names = [
        "link0", "link1", "link2", "link3", "link4",
        "link5", "link6", "link7", "hand",
        "left_finger", "right_finger"
    ]
    robot_body_ids = set()
    for name in robot_body_names:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid >= 0:
            robot_body_ids.add(bid)

    # Check all contacts
    for i in range(data.ncon):
        contact = data.contact[i]
        geom1 = contact.geom1
        geom2 = contact.geom2

        # Check if one geom is the obstacle and the other belongs to robot
        body1 = model.geom_bodyid[geom1]
        body2 = model.geom_bodyid[geom2]

        if geom1 == obstacle_id and body2 in robot_body_ids:
            return True
        if geom2 == obstacle_id and body1 in robot_body_ids:
            return True

    return False


def is_collision_free(model, data, q1, q2, max_step_size=0.05):
    """
    Check if the straight-line path from q1 to q2 is collision-free.
    (HW3 Problem 2.4: is_collision_free)

    Interpolates between q1 and q2 and checks for robot-obstacle
    collisions at each step using MuJoCo's collision detection.
    """
    # Save current state
    base_q = get_joint_positions(model, data)

    dist = np.linalg.norm(q2 - q1)
    if dist < 1e-6:
        set_joint_positions(model, data, base_q)
        return True

    step_size = min(max_step_size, dist)
    num_steps = max(int(dist / step_size), 1)
    direction = (q2 - q1) / dist

    result = True
    for i in range(num_steps + 1):
        alpha = i / num_steps
        q_check = q1 + alpha * (q2 - q1)
        set_joint_positions(model, data, q_check)

        if check_robot_obstacle_collision(model, data):
            result = False
            break

    # Restore original state
    set_joint_positions(model, data, base_q)
    return result


def retrace_plan(q_goal, tree):
    """
    Trace back through the tree from the goal to the root.
    (HW3 Problem 2.5 helper: retrace_plan)

    Returns the path as a list of numpy arrays from start to goal.
    """
    v = tuple(q_goal)
    plan = []
    while v is not None:
        plan.append(np.array(v))
        v = tree[v]
    return plan[::-1]


def rrt(q_init, q_goal, model, data, rng=None,
        threshold=0.05, step_size=0.15, max_iterations=2000):
    """
    RRT motion planner for the Panda arm.
    (HW3 Problem 2.5: panda_rrt)

    Finds a collision-free path from q_init to q_goal in the Panda's
    7-DOF configuration space, avoiding the obstacle.

    Args:
        q_init: (7,) start joint configuration
        q_goal: (7,) goal joint configuration
        model: MuJoCo model
        data: MuJoCo data
        rng: numpy random generator
        threshold: distance threshold to goal
        step_size: max extension step in joint space
        max_iterations: max RRT iterations

    Returns:
        path: list of (7,) numpy arrays from q_init to near q_goal,
              or None if no path found within max_iterations
    """
    if rng is None:
        rng = np.random.default_rng()

    # Check if direct path is collision-free (skip RRT if so)
    if is_collision_free(model, data, q_init, q_goal):
        return [q_init, q_goal]

    # Build RRT tree
    tree = {tuple(q_init): None}

    for iteration in range(max_iterations):
        q_sample = sample_point(model, q_goal, rng)
        q_near = get_nearest(np.array(list(tree.keys())), q_sample)
        q_new = extend(q_sample, q_near, step_size)

        if is_collision_free(model, data, q_near, q_new):
            tree[tuple(q_new)] = tuple(q_near)

            if np.linalg.norm(q_new - q_goal) < threshold:
                # Reached the goal
                tree[tuple(q_goal)] = tuple(q_new)
                path = retrace_plan(q_goal, tree)
                return path

    # Failed to find a path
    print(f"  [RRT] Failed after {max_iterations} iterations")
    return None


def smooth_path(path, model, data, max_iters=200):
    """
    Smooth a path by attempting to shortcut random pairs of waypoints.
    (HW3 Problem 2.6: smooth_path)

    Randomly picks two waypoints and checks if the direct path between
    them is collision-free. If so, removes the intermediate waypoints.
    """
    if len(path) <= 2:
        return path

    rng = np.random.default_rng(42)
    smoothed = list(path)

    for _ in range(max_iters):
        if len(smoothed) <= 2:
            break
        i = rng.integers(0, len(smoothed) - 2)
        j = rng.integers(i + 2, len(smoothed))
        if is_collision_free(model, data, smoothed[i], smoothed[j]):
            smoothed = smoothed[:i+1] + smoothed[j:]

    return smoothed


def plan_path(q_start, q_goal, model, data, rng=None, max_iterations=2000):
    """
    High-level path planning function.

    First checks if direct path is collision-free (common case).
    If not, runs RRT and smooths the result.

    Args:
        q_start: (7,) start configuration
        q_goal: (7,) goal configuration
        model: MuJoCo model
        data: MuJoCo data
        rng: numpy random generator

    Returns:
        path: list of (7,) configurations, or [q_start, q_goal] if
              direct path is free, or None if planning fails
    """
    if rng is None:
        rng = np.random.default_rng()

    # Try direct path first (most common case - no obstacle in the way)
    if is_collision_free(model, data, q_start, q_goal):
        return [q_start, q_goal]

    # Run RRT
    path = rrt(q_start, q_goal, model, data, rng, max_iterations=max_iterations)
    if path is None:
        return None

    # Smooth the path
    path = smooth_path(path, model, data)

    return path
