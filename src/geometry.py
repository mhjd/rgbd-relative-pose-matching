import cv2
import math
import numpy as np
from dataset import get_content

DEPTH_SCALE = 5000.0

# Freiburg 1 RGB intrinsics: https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats
FX = 517.3
FY = 516.5
CX = 318.6
CY = 255.3
CAMERA_MATRIX = np.array([
    [FX, 0.0, CX],
    [0.0, FY, CY],
    [0.0, 0.0, 1.0],
])

# Assume undistorted RGB images.
DIST_COEFFS = np.zeros((4, 1))

def backproject_pixel_to_3d(u, v, depth_value):
    """
    Convert a pixel and its depth into a 3D point in the camera coordinate frame.
    """
    z = depth_value / DEPTH_SCALE
    x = (u - CX) * z / FX
    y = (v - CY) * z / FY
    return (x, y, z)

def build_3d_2d_correspondences(matches, keypoints_i, keypoints_j, depth_image):
    object_points = []
    image_points = []
    for feature_match in matches:
        u, v = keypoints_i[feature_match.queryIdx].pt
        u = int(round(u))
        v = int(round(v))
        if v < 0 or v >= depth_image.shape[0] or u < 0 or u >= depth_image.shape[1]:
            # invalid keypoint
            continue
        depth_value = depth_image[v, u]
        if depth_value == 0:
            continue
        object_points.append(backproject_pixel_to_3d(u, v, depth_value))
        image_points.append(keypoints_j[feature_match.trainIdx].pt)
    return object_points, image_points

def estimate_pose_pnp(object_points, image_points):
    """
    Estimate the camera motion from 3D-to-2D correspondences.

    success indicates whether PnP-RANSAC found a pose.
    rvec is the estimated rotation.
    tvec is the estimated translation.
    inliers are the correspondences kept by RANSAC as geometrically consistent.
    """
    if len(object_points) < 4:
        return False, None, None, None
    object_points = np.array(object_points, dtype=np.float32)
    image_points = np.array(image_points, dtype=np.float32)
    success, rvec, tvec, inliers = cv2.solvePnPRansac(object_points, image_points, CAMERA_MATRIX, DIST_COEFFS)
    return success, rvec, tvec, inliers

def quaternion_to_rotation_matrix(qx, qy, qz, qw):
    """
    Convert a ground-truth quaternion into a 3x3 rotation matrix.
    """
    q = np.array([qx, qy, qz, qw], dtype=float)
    # Normalize the quaternion before converting it to a rotation matrix.
    q = q / np.linalg.norm(q)
    x, y, z, w = q
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y],
    ])

def get_groundtruth_pose(frame):
    """
    Build the 4x4 camera-to-world pose matrix stored in the ground truth.
    """
    _, _, groundtruth_line, _, _ = frame
    tx, ty, tz, qx, qy, qz, qw = [float(value) for value in get_content(groundtruth_line)]
    # Start from an identity matrix and fill the rotation and translation blocks.
    pose = np.eye(4)
    pose[:3, :3] = quaternion_to_rotation_matrix(qx, qy, qz, qw)
    pose[:3, 3] = [tx, ty, tz]
    return pose

def get_rotation_angle(rotation_matrix):
    """
    Convert a rotation matrix into a rotation angle in degrees.
    """
    # For a 3D rotation matrix, this formula gives the cosine of the rotation angle.
    cos_angle = (np.trace(rotation_matrix) - 1) / 2
    cos_angle = max(-1, min(1, cos_angle))
    return math.degrees(math.acos(cos_angle))

def get_groundtruth_relative_pose(frame_pair):
    """
    Convert two ground-truth camera poses into the relative pose from frame i to frame j.
    """
    frame_i, frame_j = frame_pair
    pose_i = get_groundtruth_pose(frame_i)
    pose_j = get_groundtruth_pose(frame_j)
    # np.linalg.inv(pose_j) converts world coordinates back into camera j coordinates.
    relative_pose = np.linalg.inv(pose_j) @ pose_i
    return relative_pose

def get_groundtruth_motion(frame_pair):
    """
    Return the ground-truth translation norm and rotation angle for one frame pair.
    """
    relative_pose = get_groundtruth_relative_pose(frame_pair)
    translation_norm = np.linalg.norm(relative_pose[:3, 3])
    rotation_angle = get_rotation_angle(relative_pose[:3, :3])
    return translation_norm, rotation_angle

def get_pnp_pose_matrix(rvec, tvec):
    """
    Convert OpenCV PnP output into a 4x4 pose matrix.
    """
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    # Start from an identity matrix and fill the rotation and translation blocks.
    pose = np.eye(4)
    pose[:3, :3] = rotation_matrix
    pose[:3, 3] = tvec.ravel()
    return pose

def get_pose_error(estimated_pose, groundtruth_pose):
    """
    Compare estimated and ground-truth poses with translation and rotation errors.
    """
    translation_error = np.linalg.norm(estimated_pose[:3, 3] - groundtruth_pose[:3, 3])
    rotation_error_matrix = estimated_pose[:3, :3] @ groundtruth_pose[:3, :3].T
    rotation_error = get_rotation_angle(rotation_error_matrix)
    return translation_error, rotation_error
