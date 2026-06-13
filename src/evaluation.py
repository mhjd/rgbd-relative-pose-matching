import numpy as np

from dataset import make_frame_pairs
from experiment_results import FrameGapStatistics
from geometry import (
    estimate_pose_pnp,
    get_groundtruth_motion,
    get_groundtruth_relative_pose,
    get_pnp_pose_matrix,
    get_pose_error,
)

FRAME_GAPS = [1, 2, 5, 10, 20, 50, 100]


def evaluate_frame_gaps(synchronized_frames, match_pair, frame_gaps=FRAME_GAPS, max_pairs_per_gap=None, show_progress=False):
    results = []
    for frame_gap in frame_gaps:
        frame_pairs = make_frame_pairs(synchronized_frames, frame_gap)
        if max_pairs_per_gap is not None:
            frame_pairs = frame_pairs[:max_pairs_per_gap]
        if show_progress:
            print(f"Evaluating gap {frame_gap} with {len(frame_pairs)} pairs")
        results.append(evaluate_frame_pairs(frame_pairs, frame_gap, match_pair))
    return results


def evaluate_frame_pairs(frame_pairs, frame_gap, match_pair):
    """
    Run matching and PnP-RANSAC on all pairs for one frame gap.
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
        match_result = match_pair(frame_pair)
        keypoints_i_counts.append(match_result["keypoints_i_count"])
        keypoints_j_counts.append(match_result["keypoints_j_count"])
        match_counts.append(match_result["match_count"])
        correspondence_counts.append(match_result["correspondence_count"])
        if match_result["best_match_distance"] is None:
            matching_failed_pairs += 1
        success, rvec, tvec, inliers = estimate_pose_pnp(match_result["object_points"], match_result["image_points"])
        if success:
            inlier_count = len(inliers)
            pnp_success_count += 1
            pnp_inlier_counts.append(inlier_count)
            pnp_inlier_ratios.append(inlier_count / match_result["correspondence_count"])
            estimated_pose = get_pnp_pose_matrix(rvec, tvec)
            groundtruth_pose = get_groundtruth_relative_pose(frame_pair)
            translation_error, rotation_error = get_pose_error(estimated_pose, groundtruth_pose)
            translation_errors.append(translation_error)
            rotation_errors.append(rotation_error)
        else:
            pnp_failed_count += 1
    result = FrameGapStatistics(frame_gap, len(frame_pairs))
    result.matching_failed = matching_failed_pairs
    result.mean_keypoints_i = sum(keypoints_i_counts) / len(keypoints_i_counts)
    result.mean_keypoints_j = sum(keypoints_j_counts) / len(keypoints_j_counts)
    result.mean_matches = sum(match_counts) / len(match_counts)
    result.mean_correspondences = sum(correspondence_counts) / len(correspondence_counts)
    result.mean_gt_translation = sum(groundtruth_translation_norms) / len(groundtruth_translation_norms)
    result.mean_gt_rotation = sum(groundtruth_rotation_angles) / len(groundtruth_rotation_angles)
    result.pnp_success = pnp_success_count
    result.pnp_failed = pnp_failed_count
    if len(pnp_inlier_counts) > 0:
        result.mean_pnp_inliers = sum(pnp_inlier_counts) / len(pnp_inlier_counts)
        result.mean_pnp_inlier_ratio = sum(pnp_inlier_ratios) / len(pnp_inlier_ratios)
        result.mean_translation_error = sum(translation_errors) / len(translation_errors)
        result.median_translation_error = np.median(translation_errors)
        result.p95_translation_error = np.percentile(translation_errors, 95)
        result.max_translation_error = max(translation_errors)
        result.mean_rotation_error = sum(rotation_errors) / len(rotation_errors)
        result.median_rotation_error = np.median(rotation_errors)
        result.p95_rotation_error = np.percentile(rotation_errors, 95)
        result.max_rotation_error = max(rotation_errors)
    return result
