from dataclasses import dataclass
import time

import numpy as np

from dataset import get_depth_path_from_frame, get_rgb_path_from_frame, get_timestamp, make_frame_pairs
from experiment_results import FrameGapStatistics
from geometry import (
    estimate_pose_pnp,
    get_groundtruth_motion,
    get_groundtruth_relative_pose,
    get_pnp_pose_matrix,
    get_pose_error,
)

FRAME_GAPS = [1, 2, 5, 10, 20, 50, 100]


@dataclass
class MatchResult:
    keypoints_i_count: int
    keypoints_j_count: int
    match_count: int
    correspondence_count: int
    object_points: object
    image_points: object
    best_match_distance: float | None


def to_json_float(value):
    if value is None:
        return None
    return float(value)


def to_json_int(value):
    if value is None:
        return None
    return int(value)


def frame_to_json(frame):
    rgb_line, depth_line, groundtruth_line, rgb_depth_time_diff, rgb_groundtruth_time_diff = frame
    return {
        "rgb_timestamp": to_json_float(get_timestamp(rgb_line)),
        "depth_timestamp": to_json_float(get_timestamp(depth_line)),
        "groundtruth_timestamp": to_json_float(get_timestamp(groundtruth_line)),
        "rgb_path": get_rgb_path_from_frame(frame),
        "depth_path": get_depth_path_from_frame(frame),
        "rgb_depth_time_diff": to_json_float(rgb_depth_time_diff),
        "rgb_groundtruth_time_diff": to_json_float(rgb_groundtruth_time_diff),
    }


def build_pair_result_record(
    method_name,
    frame_gap,
    pair_index,
    frame_pair,
    match_result,
    groundtruth_translation,
    groundtruth_rotation,
    pnp_success,
    pnp_inlier_count,
    pnp_inlier_ratio,
    translation_error,
    rotation_error,
):
    frame_i, frame_j = frame_pair
    return {
        "method": method_name,
        "gap": frame_gap,
        "pair_index": pair_index,
        "frame_i": frame_to_json(frame_i),
        "frame_j": frame_to_json(frame_j),
        "keypoints_i": to_json_int(match_result.keypoints_i_count),
        "keypoints_j": to_json_int(match_result.keypoints_j_count),
        "matches": to_json_int(match_result.match_count),
        "correspondences": to_json_int(match_result.correspondence_count),
        "best_match_distance": to_json_float(match_result.best_match_distance),
        "groundtruth_translation": to_json_float(groundtruth_translation),
        "groundtruth_rotation": to_json_float(groundtruth_rotation),
        "pnp_success": bool(pnp_success),
        "pnp_inliers": to_json_int(pnp_inlier_count),
        "pnp_inlier_ratio": to_json_float(pnp_inlier_ratio),
        "translation_error": to_json_float(translation_error),
        "rotation_error": to_json_float(rotation_error),
    }


def evaluate_frame_gaps(
    synchronized_frames,
    match_pair,
    frame_gaps=FRAME_GAPS,
    max_pairs_per_gap=None,
    show_progress=False,
    method_name=None,
    pair_result_writer=None,
):
    results = []
    for frame_gap in frame_gaps:
        frame_pairs = make_frame_pairs(synchronized_frames, frame_gap)
        if max_pairs_per_gap is not None:
            frame_pairs = frame_pairs[:max_pairs_per_gap]
        if show_progress:
            print(f"Evaluating gap {frame_gap} with {len(frame_pairs)} pairs")
        start_time = time.perf_counter()
        results.append(evaluate_frame_pairs(frame_pairs, frame_gap, match_pair, method_name, pair_result_writer))
        if show_progress:
            elapsed_time = time.perf_counter() - start_time
            print(f"Finished gap {frame_gap} in {elapsed_time:.2f}s\n")
    return results


def evaluate_frame_pairs(frame_pairs, frame_gap, match_pair, method_name=None, pair_result_writer=None):
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
    for pair_index, frame_pair in enumerate(frame_pairs):
        translation_norm, rotation_angle = get_groundtruth_motion(frame_pair)
        groundtruth_translation_norms.append(translation_norm)
        groundtruth_rotation_angles.append(rotation_angle)
        match_result = match_pair(frame_pair)
        keypoints_i_counts.append(match_result.keypoints_i_count)
        keypoints_j_counts.append(match_result.keypoints_j_count)
        match_counts.append(match_result.match_count)
        correspondence_counts.append(match_result.correspondence_count)
        if match_result.best_match_distance is None:
            matching_failed_pairs += 1
        success, rvec, tvec, inliers = estimate_pose_pnp(match_result.object_points, match_result.image_points)
        inlier_count = None
        inlier_ratio = None
        translation_error = None
        rotation_error = None
        if success:
            inlier_count = len(inliers)
            inlier_ratio = inlier_count / match_result.correspondence_count
            pnp_success_count += 1
            pnp_inlier_counts.append(inlier_count)
            pnp_inlier_ratios.append(inlier_ratio)
            estimated_pose = get_pnp_pose_matrix(rvec, tvec)
            groundtruth_pose = get_groundtruth_relative_pose(frame_pair)
            translation_error, rotation_error = get_pose_error(estimated_pose, groundtruth_pose)
            translation_errors.append(translation_error)
            rotation_errors.append(rotation_error)
        else:
            pnp_failed_count += 1
        if pair_result_writer is not None:
            pair_result_writer(build_pair_result_record(
                method_name,
                frame_gap,
                pair_index,
                frame_pair,
                match_result,
                translation_norm,
                rotation_angle,
                success,
                inlier_count,
                inlier_ratio,
                translation_error,
                rotation_error,
            ))
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
