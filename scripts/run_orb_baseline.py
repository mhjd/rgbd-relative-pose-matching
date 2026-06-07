import sys
sys.path.append("src")

import cv2
import numpy as np
from dataset import get_rgb_path_from_frame, get_depth_path_from_frame, get_data_content, synchronize_frames, make_frame_pairs
from geometry import build_3d_2d_correspondences, estimate_pose_pnp, get_groundtruth_motion, get_groundtruth_relative_pose, get_pnp_pose_matrix, get_pose_error

FRAME_GAPS = [1, 2, 5, 10, 20, 50, 100]

def get_orb_features(frame, orb, features_cache):
    rgb_path = get_rgb_path_from_frame(frame)
    if rgb_path in features_cache:
        return features_cache[rgb_path]
    rgb_image = cv2.imread(rgb_path, cv2.IMREAD_GRAYSCALE)
    if rgb_image is None:
        print(f"Could not read RGB image : {rgb_path}")
        sys.exit(1)
    keypoints, descriptors = orb.detectAndCompute(rgb_image, None)
    features_cache[rgb_path] = (keypoints, descriptors)
    return keypoints, descriptors

def get_depth_image(frame, depth_cache):
    depth_path = get_depth_path_from_frame(frame)
    if depth_path in depth_cache:
        return depth_cache[depth_path]
    depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth_image is None:
        print(f"Could not read depth image : {depth_path}")
        sys.exit(1)
    depth_cache[depth_path] = depth_image
    return depth_image

def match_orb_pair(frame_pair, orb, matcher, features_cache, depth_cache):
    """
    Match ORB descriptors between two frames.

    Return named matching statistics for one frame pair.
    """
    frame_i, frame_j = frame_pair
    keypoints_i, descriptors_i = get_orb_features(frame_i, orb, features_cache)
    keypoints_j, descriptors_j = get_orb_features(frame_j, orb, features_cache)
    if descriptors_i is None or descriptors_j is None:
        return {
            "keypoints_i_count": len(keypoints_i),
            "keypoints_j_count": len(keypoints_j),
            "match_count": 0,
            "correspondence_count": 0,
            "object_points": [],
            "image_points": [],
            "best_match_distance": None,
        }
    matches = matcher.match(descriptors_i, descriptors_j)
    matches = sorted(matches, key=lambda orb_match: orb_match.distance)
    depth_i = get_depth_image(frame_i, depth_cache)
    object_points, image_points = build_3d_2d_correspondences(matches, keypoints_i, keypoints_j, depth_i)
    best_match_distance = None
    if len(matches) > 0:
        best_match_distance = matches[0].distance
    return {
        "keypoints_i_count": len(keypoints_i),
        "keypoints_j_count": len(keypoints_j),
        "match_count": len(matches),
        "correspondence_count": len(object_points),
        "object_points": object_points,
        "image_points": image_points,
        "best_match_distance": best_match_distance,
    }

def inspect_orb_pairs(frame_pairs, frame_gap, orb, matcher, features_cache, depth_cache):
    """
    Run ORB matching and PnP-RANSAC on all pairs for one frame gap.
    """
    keypoints_i_counts = []
    keypoints_j_counts = []
    match_counts = []
    correspondence_counts = []
    matching_failed_pairs = 0
    pnp_success_count = 0
    pnp_failed_count = 0
    pnp_inlier_counts = []
    pnp_inlier_ratios = []
    translation_errors = []
    rotation_errors = []
    groundtruth_translation_norms = []
    groundtruth_rotation_angles = []
    for frame_pair in frame_pairs:
        translation_norm, rotation_angle = get_groundtruth_motion(frame_pair)
        groundtruth_translation_norms.append(translation_norm)
        groundtruth_rotation_angles.append(rotation_angle)
        orb_result = match_orb_pair(frame_pair, orb, matcher, features_cache, depth_cache)
        keypoints_i_counts.append(orb_result["keypoints_i_count"])
        keypoints_j_counts.append(orb_result["keypoints_j_count"])
        match_counts.append(orb_result["match_count"])
        correspondence_counts.append(orb_result["correspondence_count"])
        if orb_result["best_match_distance"] is None:
            matching_failed_pairs += 1
        success, rvec, tvec, inliers = estimate_pose_pnp(orb_result["object_points"], orb_result["image_points"])
        if success:
            inlier_count = len(inliers)
            pnp_success_count += 1
            pnp_inlier_counts.append(inlier_count)
            pnp_inlier_ratios.append(inlier_count / orb_result["correspondence_count"])
            estimated_pose = get_pnp_pose_matrix(rvec, tvec)
            groundtruth_pose = get_groundtruth_relative_pose(frame_pair)
            translation_error, rotation_error = get_pose_error(estimated_pose, groundtruth_pose)
            translation_errors.append(translation_error)
            rotation_errors.append(rotation_error)
        else:
            pnp_failed_count += 1
    result = {
        "gap": frame_gap,
        "pairs": len(frame_pairs),
        "matching_failed": matching_failed_pairs,
        "mean_keypoints_i": sum(keypoints_i_counts) / len(keypoints_i_counts),
        "mean_keypoints_j": sum(keypoints_j_counts) / len(keypoints_j_counts),
        "mean_matches": sum(match_counts) / len(match_counts),
        "mean_correspondences": sum(correspondence_counts) / len(correspondence_counts),
        "mean_gt_translation": sum(groundtruth_translation_norms) / len(groundtruth_translation_norms),
        "mean_gt_rotation": sum(groundtruth_rotation_angles) / len(groundtruth_rotation_angles),
        "pnp_success": pnp_success_count,
        "pnp_failed": pnp_failed_count,
        "mean_pnp_inliers": None,
        "mean_pnp_inlier_ratio": None,
        "mean_translation_error": None,
        "median_translation_error": None,
        "p95_translation_error": None,
        "max_translation_error": None,
        "mean_rotation_error": None,
        "median_rotation_error": None,
        "p95_rotation_error": None,
        "max_rotation_error": None,
    }
    if len(pnp_inlier_counts) > 0:
        result["mean_pnp_inliers"] = sum(pnp_inlier_counts) / len(pnp_inlier_counts)
        result["mean_pnp_inlier_ratio"] = sum(pnp_inlier_ratios) / len(pnp_inlier_ratios)
        result["mean_translation_error"] = sum(translation_errors) / len(translation_errors)
        result["median_translation_error"] = np.median(translation_errors)
        result["p95_translation_error"] = np.percentile(translation_errors, 95)
        result["max_translation_error"] = max(translation_errors)
        result["mean_rotation_error"] = sum(rotation_errors) / len(rotation_errors)
        result["median_rotation_error"] = np.median(rotation_errors)
        result["p95_rotation_error"] = np.percentile(rotation_errors, 95)
        result["max_rotation_error"] = max(rotation_errors)
    return result

def format_float(value, decimals):
    if value is None:
        return "NA"
    return f"{value:.{decimals}f}"

def print_results_table(results):
    print(
        f"{'gap':>3} | "
        f"{'pairs':>5} | "
        f"{'gt_t':>5} | "
        f"{'gt_r':>5} | "
        f"{'matches':>7} | "
        f"{'corr':>6} | "
        f"{'pnp_ok':>6} | "
        f"{'pnp_fail':>8} | "
        f"{'inl_ratio':>9} | "
        f"{'t_med':>5} | "
        f"{'t_p95':>6} | "
        f"{'r_med':>5} | "
        f"{'r_p95':>6}"
    )
    for result in results:
        print(
            f"{result['gap']:>3} | "
            f"{result['pairs']:>5} | "
            f"{format_float(result['mean_gt_translation'], 3):>5} | "
            f"{format_float(result['mean_gt_rotation'], 2):>5} | "
            f"{format_float(result['mean_matches'], 1):>7} | "
            f"{format_float(result['mean_correspondences'], 1):>6} | "
            f"{result['pnp_success']:>6} | "
            f"{result['pnp_failed']:>8} | "
            f"{format_float(result['mean_pnp_inlier_ratio'], 3):>9} | "
            f"{format_float(result['median_translation_error'], 3):>5} | "
            f"{format_float(result['p95_translation_error'], 3):>6} | "
            f"{format_float(result['median_rotation_error'], 2):>5} | "
            f"{format_float(result['p95_rotation_error'], 2):>6}"
        )

rgb = get_data_content("rgb.txt")
depth = get_data_content("depth.txt")
groundtruth = get_data_content("groundtruth.txt")
synchronized_frames, _, _ = synchronize_frames(rgb, depth, groundtruth)
orb = cv2.ORB_create()
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
features_cache = {}
depth_cache = {}

results = []
for frame_gap in FRAME_GAPS:
    frame_pairs = make_frame_pairs(synchronized_frames, frame_gap)
    results.append(inspect_orb_pairs(frame_pairs, frame_gap, orb, matcher, features_cache, depth_cache))

print(f"Synchronized frames : {len(synchronized_frames)}")
print_results_table(results)
