"""
perception.py - Vision pipeline for pick-and-place
Implements: HSV segmentation, depth back-projection, centroid estimation, PCA, and ICP.

Pipeline:
  RGB image + Depth image
    -> HSV color segmentation (binary mask)
    -> Depth back-projection (2D pixels -> 3D point cloud)
    -> Centroid (position estimate)
    -> PCA (coarse orientation estimate)
    -> ICP (refined 6-DOF pose)
"""
import numpy as np
import cv2
from scipy.spatial import KDTree

MAX_ICP_CENTROID_JUMP = 0.020
MAX_REASONABLE_DEPTH_POINT = 5.0


# ================================================================
# COLOR SEGMENTATION
# ================================================================

# Pre-defined HSV ranges for each object/target
# These are tuned for our MuJoCo scene colors
COLOR_RANGES = {
    # Cracker box: bright yellow
    "yellow_object": {
        "lower": np.array([15, 60, 60]),
        "upper": np.array([45, 255, 255]),
    },
    # Mustard bottle: green
    "green_object": {
        "lower": np.array([35, 60, 60]),
        "upper": np.array([85, 255, 255]),
    },
    # Sugar box: cyan
    "cyan_object": {
        "lower": np.array([80, 60, 60]),
        "upper": np.array([100, 255, 255]),
    },
    # Basket: red (wraps around in HSV)
    "red_basket": {
        "lower1": np.array([0, 80, 50]),
        "upper1": np.array([10, 255, 255]),
        "lower2": np.array([170, 80, 50]),
        "upper2": np.array([180, 255, 255]),
    },
}


def segment_by_color(rgb_image, target_name):
    """
    Segment an object from an RGB image using HSV color thresholding.

    Args:
        rgb_image: (H, W, 3) uint8 array in RGB format
        target_name: key into COLOR_RANGES ("yellow_object" or "red_basket")

    Returns:
        mask: (H, W) binary array, 255 where the target is, 0 elsewhere
    """
    # Convert RGB to HSV color space
    # HSV separates color (Hue) from brightness (Value),
    # making detection robust to lighting changes
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)

    ranges = COLOR_RANGES[target_name]

    if "lower1" in ranges:
        # Red wraps around in HSV, need two ranges
        mask1 = cv2.inRange(hsv, ranges["lower1"], ranges["upper1"])
        mask2 = cv2.inRange(hsv, ranges["lower2"], ranges["upper2"])
        mask = cv2.bitwise_or(mask1, mask2)
    else:
        mask = cv2.inRange(hsv, ranges["lower"], ranges["upper"])

    # Clean up noise with morphological operations
    # Erode removes tiny specks, dilate fills small holes
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)

    return mask


# ================================================================
# CAMERA UTILITIES
# ================================================================

def get_camera_intrinsics(model, camera_name, img_height, img_width):
    """
    Compute the camera intrinsic matrix K from MuJoCo camera parameters.

    The intrinsic matrix maps 3D camera-frame points to 2D pixel coordinates:
        [u]       [fx  0  cx] [X]
        [v] = 1/Z [0  fy  cy] [Y]
        [1]       [0   0   1] [Z]

    Args:
        model: MuJoCo model
        camera_name: name of the camera in the MJCF
        img_height: rendered image height in pixels
        img_width: rendered image width in pixels

    Returns:
        K: (3, 3) intrinsic matrix
        cam_id: integer camera ID
    """
    import mujoco
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)

    # MuJoCo stores vertical field of view in degrees
    fovy_deg = model.cam_fovy[cam_id]
    fovy_rad = fovy_deg * np.pi / 180.0

    # Focal length in pixels (from pinhole camera model)
    # fy = (image_height / 2) / tan(fovy / 2)
    fy = (img_height / 2.0) / np.tan(fovy_rad / 2.0)
    fx = fy  # square pixels (aspect ratio = 1)

    # Principal point (image center)
    cx = img_width / 2.0
    cy = img_height / 2.0

    K = np.array([
        [fx,  0,  cx],
        [ 0, fy,  cy],
        [ 0,  0,   1]
    ])

    return K, cam_id


def get_camera_extrinsics(data, cam_id):
    """
    Get the camera's position and orientation in the world frame.

    Args:
        data: MuJoCo data (after mj_forward)
        cam_id: camera ID from get_camera_intrinsics

    Returns:
        cam_pos: (3,) camera position in world coordinates
        cam_rot: (3, 3) rotation matrix (camera frame -> world frame)
    """
    cam_pos = data.cam_xpos[cam_id].copy()
    cam_rot = data.cam_xmat[cam_id].reshape(3, 3).copy()
    return cam_pos, cam_rot


def depth_buffer_to_meters(depth_buffer, model):
    """
    Convert MuJoCo's depth buffer to meters.

    In MuJoCo v3.6+, the renderer returns depth values that are already
    in meters (distance from camera to surface along viewing direction).
    Values of ~0 mean very close, larger values mean farther away.
    Background/sky pixels may have very large values.

    Args:
        depth_buffer: (H, W) raw depth from renderer
        model: MuJoCo model (unused in v3.6, kept for API compatibility)

    Returns:
        depth_meters: (H, W) depth in meters
    """
    # In MuJoCo v3.6, depth is already in meters
    return depth_buffer.copy()


# ================================================================
# DEPTH BACK-PROJECTION (2D pixels -> 3D world points)
# ================================================================

def backproject_to_pointcloud(depth_meters, mask, K, cam_pos, cam_rot):
    """
    Convert masked depth pixels into a 3D point cloud in world coordinates.

    This is the core perception step from our proposal:
      1. For each masked pixel (u, v), read its depth d
      2. Convert (u, v, d) to 3D point in camera frame using K
      3. Transform from camera frame to world frame using (R, t)

    Camera convention (MuJoCo):
      - Camera x-axis: points right in the image
      - Camera y-axis: points UP (opposite to image v-axis which goes down)
      - Camera -z-axis: points into the scene (viewing direction)
      - Depth d: distance along the viewing direction (along -z)

    So pixel (u, v) with depth d maps to camera-frame point:
      X_cam =  (u - cx) * d / fx
      Y_cam = -(v - cy) * d / fy    (negative because v goes down but y goes up)
      Z_cam = -d                     (negative because camera looks along -z)

    Then: p_world = cam_rot @ p_cam + cam_pos   (HW1 coordinate transform)

    Args:
        depth_meters: (H, W) depth in meters
        mask: (H, W) binary mask (255 = target pixel)
        K: (3, 3) camera intrinsic matrix
        cam_pos: (3,) camera position in world frame
        cam_rot: (3, 3) camera rotation matrix (camera frame -> world frame)

    Returns:
        points_world: (N, 3) array of 3D points in world coordinates
        Returns None if too few valid points found
    """
    # Get pixel coordinates where the mask is active
    v_pixels, u_pixels = np.where(mask > 0)

    if len(v_pixels) < 10:
        return None

    # Read depth at each masked pixel
    depths = depth_meters[v_pixels, u_pixels]

    # Filter out invalid depths (too close, too far, or background)
    valid = np.isfinite(depths) & (depths > 0.1) & (depths < 3.0)
    v_pixels = v_pixels[valid]
    u_pixels = u_pixels[valid]
    depths = depths[valid]

    if len(depths) < 10:
        return None

    # Extract intrinsic parameters
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    # Back-project to camera coordinates
    X_cam = (u_pixels.astype(np.float64) - cx) * depths / fx
    Y_cam = -(v_pixels.astype(np.float64) - cy) * depths / fy
    Z_cam = -depths

    points_cam = np.stack([X_cam, Y_cam, Z_cam], axis=1)  # (N, 3)
    points_cam = points_cam[np.all(np.isfinite(points_cam), axis=1)]
    points_cam = points_cam[np.all(np.abs(points_cam) < 10.0, axis=1)]

    if len(points_cam) < 10:
        return None

    # Transform to world coordinates: p_world = R @ p_cam + t
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        points_world = (cam_rot @ points_cam.T).T + cam_pos
    points_world = points_world[np.all(np.isfinite(points_world), axis=1)]

    if len(points_world) < 10:
        return None

    return points_world


# ================================================================
# CENTROID ESTIMATION
# ================================================================

def estimate_centroid(points, color_target=None):
    """
    Estimate object position as the mean of all 3D points.

    Args:
        points: (N, 3) point cloud in world coordinates

    Returns:
        centroid: (3,) position estimate [x, y, z]
    """
    points = points[np.all(np.isfinite(points), axis=1)]
    if len(points) == 0:
        return None
    if color_target == "yellow_object":
        lo = np.percentile(points[:, :2], 2, axis=0)
        hi = np.percentile(points[:, :2], 98, axis=0)
        centroid = np.mean(points, axis=0)
        centroid[:2] = 0.5 * (lo + hi)
        if centroid[0] < 0.48 and centroid[1] > 0.14:
            # In this back-left part of the workspace the overhead mask sees
            # more of the near yellow face than the far face, biasing the
            # center estimate toward +Y. Correct only that observed corner.
            centroid[1] -= 0.022
        return centroid

    return np.mean(points, axis=0)


def estimate_grasp_width(points, color_target, grasp_axis=None):
    """
    Estimate the object's tabletop width along the gripper closing direction.

    The Panda fingers close along world Y for the top-down grasp used here, so
    the default measurement is the point-cloud extent along Y. Percentiles make
    the estimate less sensitive to a few noisy depth samples.
    """
    if color_target == "red_basket":
        return None

    if grasp_axis is None:
        grasp_axis = np.array([0.0, 1.0, 0.0])

    axis_norm = np.linalg.norm(grasp_axis)
    if not np.isfinite(axis_norm) or axis_norm < 1e-8:
        grasp_axis = np.array([0.0, 1.0, 0.0])
        axis_norm = 1.0
    finite = np.all(np.isfinite(points), axis=1)
    bounded = np.all(np.abs(points) < MAX_REASONABLE_DEPTH_POINT, axis=1)
    points = points[finite & bounded]
    if len(points) < 10:
        return None

    grasp_axis = grasp_axis / axis_norm
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        projected = points @ grasp_axis
    projected = projected[np.isfinite(projected)]
    if len(projected) < 10:
        return None

    width = np.percentile(projected, 95) - np.percentile(projected, 5)
    if color_target == "yellow_object" and width < 0.055:
        return None
    return float(np.clip(width, 0.015, 0.075))


def filter_workspace_points(points, color_target):
    """
    Remove reconstructed points outside the tabletop workspace.

    The macOS depth renderer can occasionally return unstable depth values.
    Filtering in world coordinates prevents those rare samples from dominating
    the centroid and sending IK to an unreachable phantom target.
    """
    finite = np.all(np.isfinite(points), axis=1)
    bounded = np.all(np.abs(points) < MAX_REASONABLE_DEPTH_POINT, axis=1)
    points = points[finite & bounded]
    if len(points) < 10:
        return None

    if color_target == "red_basket":
        bounds = ((0.35, 0.65), (-0.35, -0.10), (0.35, 0.55))
    else:
        bounds = ((0.35, 0.65), (-0.15, 0.20), (0.35, 0.55))

    in_bounds = (
        (points[:, 0] >= bounds[0][0]) & (points[:, 0] <= bounds[0][1]) &
        (points[:, 1] >= bounds[1][0]) & (points[:, 1] <= bounds[1][1]) &
        (points[:, 2] >= bounds[2][0]) & (points[:, 2] <= bounds[2][1])
    )
    points = points[in_bounds]
    if len(points) < 10:
        return None
    return points


# ================================================================
# PCA - COARSE ORIENTATION ESTIMATE
# ================================================================

def estimate_orientation_pca(points):
    """
    Estimate object orientation using Principal Component Analysis.

    PCA finds the directions along which the point cloud is most spread out.
    For a rectangular box:
      - Eigenvector with largest eigenvalue  = longest axis
      - Eigenvector with smallest eigenvalue = shortest axis (best grasp direction)

    The eigenvectors form a rotation matrix that approximates the object's
    orientation. This is used as the initial guess for ICP.

    Args:
        points: (N, 3) point cloud in world coordinates

    Returns:
        R_pca: (3, 3) rotation matrix (coarse orientation)
        eigenvalues: (3,) spread along each axis (useful for debugging)
    """
    points = points[np.all(np.isfinite(points), axis=1)]
    if len(points) < 3:
        return np.eye(3), np.zeros(3)

    centroid = np.mean(points, axis=0)
    centered = points - centroid

    # Covariance matrix (3x3)
    cov = (centered.T @ centered) / len(centered)

    # Eigen decomposition
    # eigh returns eigenvalues sorted smallest to largest
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # eigenvectors columns are the principal axes:
    #   eigenvectors[:, 0] = shortest axis (smallest eigenvalue)
    #   eigenvectors[:, 1] = medium axis
    #   eigenvectors[:, 2] = longest axis (largest eigenvalue)

    # Ensure proper rotation matrix (det = +1)
    R_pca = eigenvectors.copy()
    if np.linalg.det(R_pca) < 0:
        R_pca[:, 0] *= -1

    return R_pca, eigenvalues


# ================================================================
# ICP - ITERATIVE CLOSEST POINT (from HW4)
# ================================================================

def icp(scene_points, model_points, R_init, t_init, max_iters=50):
    """
    Iterative Closest Point algorithm for 6-DOF pose estimation.
    Adapted from our HW4 implementation (2D -> 3D).

    Given:
      - scene_points: 3D points observed from the depth camera
      - model_points: pre-computed reference points from the known object mesh
    Find the rotation R and translation t that best aligns the model to the scene.

    Algorithm (iterates until convergence):
      1. Transform model points by current (R, t) guess
      2. Find nearest-neighbor correspondences (scene <-> model)
      3. Compute optimal R via SVD: W = U Sigma V^T, R = V D U^T
         where D = diag(1, 1, det(V U^T)) ensures a proper rotation
      4. Compute optimal t = centroid_scene - R * centroid_model
      5. Check if correspondences changed; if not, converged

    Args:
        scene_points: (N, 3) observed point cloud in world coordinates
        model_points: (M, 3) reference point cloud in object frame
        R_init: (3, 3) initial rotation guess (from PCA)
        t_init: (3,) initial translation guess (from centroid)
        max_iters: maximum iterations

    Returns:
        R: (3, 3) estimated rotation (object frame -> world frame)
        t: (3,) estimated translation (object position in world)
        converged: bool, whether ICP converged
    """
    scene_points = scene_points[np.all(np.isfinite(scene_points), axis=1)]
    model_points = model_points[np.all(np.isfinite(model_points), axis=1)]
    if (len(scene_points) < 20 or len(model_points) < 20 or
            not np.all(np.isfinite(R_init)) or not np.all(np.isfinite(t_init))):
        return R_init.copy(), t_init.copy(), False

    R = R_init.copy()
    t = t_init.copy()
    prev_correspondences = None

    for iteration in range(max_iters):
        # Step 1: Transform model points by current (R, t)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            transformed_model = (R @ model_points.T).T + t  # (M, 3)
        if not np.all(np.isfinite(transformed_model)):
            return R_init.copy(), t_init.copy(), False

        # Step 2: Find nearest-neighbor correspondences
        # For each scene point, find the closest transformed model point
        tree = KDTree(transformed_model)
        distances, correspondences = tree.query(scene_points)
        if not np.all(np.isfinite(distances)):
            return R_init.copy(), t_init.copy(), False

        # Check convergence: have correspondences stabilized?
        if prev_correspondences is not None:
            if np.array_equal(correspondences, prev_correspondences):
                return R, t, True  # converged
        prev_correspondences = correspondences.copy()

        # Get the matched model points (in original object frame)
        matched_model = model_points[correspondences]  # (N, 3)

        # Step 3: Compute centroids
        centroid_scene = np.mean(scene_points, axis=0)
        centroid_model = np.mean(matched_model, axis=0)

        # Center the points
        centered_scene = scene_points - centroid_scene
        centered_model = matched_model - centroid_model

        # Build data matrix W (HW4 step)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            W = centered_scene.T @ centered_model  # (3, 3)
        if not np.all(np.isfinite(W)):
            return R_init.copy(), t_init.copy(), False

        # SVD decomposition
        try:
            U, Sigma, Vt = np.linalg.svd(W)
        except np.linalg.LinAlgError:
            return R_init.copy(), t_init.copy(), False
        V = Vt.T

        # Ensure proper rotation (no reflection)
        D = np.diag([1.0, 1.0, np.linalg.det(V @ U.T)])
        R = V @ D @ U.T  # (3, 3)

        # Step 4: Compute translation
        t = centroid_scene - R @ centroid_model  # (3,)
        if not np.all(np.isfinite(R)) or not np.all(np.isfinite(t)):
            return R_init.copy(), t_init.copy(), False

    return R, t, False  # did not converge within max_iters


# ================================================================
# FULL PERCEPTION PIPELINE
# ================================================================

def perceive_object(rgb_image, depth_meters, model, data,
                    camera_name, color_target, model_cloud=None,
                    img_height=480, img_width=640):
    """
    Full perception pipeline: from raw images to 6-DOF pose.

    Steps:
      1. HSV color segmentation -> binary mask
      2. Depth back-projection -> 3D point cloud
      3. Centroid -> position estimate
      4. PCA -> coarse orientation
      5. ICP (if model_cloud provided) -> refined 6-DOF pose

    Args:
        rgb_image: (H, W, 3) RGB image from camera
        depth_meters: (H, W) depth in meters
        model: MuJoCo model
        data: MuJoCo data
        camera_name: which camera to use ("overhead_cam")
        color_target: HSV target name ("yellow_object" or "red_basket")
        model_cloud: (M, 3) reference point cloud for ICP (None to skip ICP)
        img_height: image height
        img_width: image width

    Returns:
        result: dict with keys:
            "position": (3,) estimated position
            "rotation": (3,3) estimated rotation (identity if ICP skipped)
            "mask": (H,W) segmentation mask
            "point_cloud": (N,3) observed points
            "icp_converged": bool
    """
    # Step 1: Segmentation
    mask = segment_by_color(rgb_image, color_target)
    n_pixels = np.sum(mask > 0)

    if n_pixels < 10:
        return None  # object not visible

    # Step 2: Get camera parameters
    K, cam_id = get_camera_intrinsics(model, camera_name, img_height, img_width)
    cam_pos, cam_rot = get_camera_extrinsics(data, cam_id)

    # Step 3: Back-project to 3D
    points = backproject_to_pointcloud(depth_meters, mask, K, cam_pos, cam_rot)

    if points is None or len(points) < 10:
        return None

    points = filter_workspace_points(points, color_target)
    if points is None:
        return None

    # Step 4: Centroid
    segmented_centroid = estimate_centroid(points, color_target)
    if segmented_centroid is None or not np.all(np.isfinite(segmented_centroid)):
        return None
    centroid = segmented_centroid.copy()
    grasp_width = estimate_grasp_width(points, color_target)

    # Step 5: PCA
    R_pca, eigenvalues = estimate_orientation_pca(points)

    # Step 6: ICP (if model cloud provided)
    R_final = R_pca
    icp_converged = False
    icp_jump = None
    icp_accepted = False

    if model_cloud is not None and len(points) >= 20:
        R_icp, t_icp, icp_converged = icp(
            scene_points=points,
            model_points=model_cloud,
            R_init=R_pca,
            t_init=centroid,
            max_iters=50
        )
        icp_jump = np.linalg.norm(t_icp - segmented_centroid)
        if (icp_converged and np.all(np.isfinite(R_icp)) and
                np.all(np.isfinite(t_icp)) and np.isfinite(icp_jump) and
                icp_jump <= MAX_ICP_CENTROID_JUMP):
            R_final = R_icp
            centroid = t_icp
            icp_accepted = True
        else:
            icp_converged = False

    return {
        "position": centroid,
        "rotation": R_final,
        "mask": mask,
        "point_cloud": points,
        "grasp_width": grasp_width,
        "segmented_centroid": segmented_centroid,
        "icp_jump": icp_jump,
        "icp_accepted": icp_accepted,
        "eigenvalues": eigenvalues,
        "icp_converged": icp_converged,
        "n_pixels": n_pixels,
    }
