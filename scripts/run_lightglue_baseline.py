import sys
sys.path.append("src")

import torch
from dataset import get_depth_image, get_rgb_path_from_frame, get_synchronized_frames
from evaluation import evaluate_frame_gaps
from experiment_results import print_experiment_summary
from geometry import build_3d_2d_correspondences_from_pixels
from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, rbd

MAX_PAIRS_PER_GAP = 20


def get_lightglue_features(frame, extractor, device, features_cache):
    rgb_path = get_rgb_path_from_frame(frame)
    if rgb_path in features_cache:
        return features_cache[rgb_path]
    image = load_image(rgb_path).to(device)
    with torch.inference_mode():
        features = extractor.extract(image)
    features_cache[rgb_path] = features
    return features


def match_lightglue_pair(frame_pair, extractor, matcher, device, features_cache, depth_cache):
    """
    Match SuperPoint features with LightGlue between two frames.

    Return named matching statistics for one frame pair.
    """
    frame_i, frame_j = frame_pair
    features_i = get_lightglue_features(frame_i, extractor, device, features_cache)
    features_j = get_lightglue_features(frame_j, extractor, device, features_cache)

    with torch.inference_mode():
        matches = matcher({"image0": features_i, "image1": features_j})

    features_i = rbd(features_i)
    features_j = rbd(features_j)
    matches = rbd(matches)

    matched_indices = matches["matches"]
    points_i = features_i["keypoints"][matched_indices[:, 0]]
    points_j = features_j["keypoints"][matched_indices[:, 1]]

    depth_i = get_depth_image(frame_i, depth_cache)
    object_points, image_points = build_3d_2d_correspondences_from_pixels(points_i, points_j, depth_i)

    best_match_distance = None
    if len(matched_indices) > 0:
        best_match_distance = 0

    return {
        "keypoints_i_count": len(features_i["keypoints"]),
        "keypoints_j_count": len(features_j["keypoints"]),
        "match_count": len(matched_indices),
        "correspondence_count": len(object_points),
        "object_points": object_points,
        "image_points": image_points,
        "best_match_distance": best_match_distance,
    }


device = "mps" if torch.backends.mps.is_available() else "cpu"
synchronized_frames = get_synchronized_frames()
extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
matcher = LightGlue(features="superpoint").eval().to(device)
features_cache = {}
depth_cache = {}

print(f"Device : {device}")
results = evaluate_frame_gaps(
    synchronized_frames,
    lambda frame_pair: match_lightglue_pair(frame_pair, extractor, matcher, device, features_cache, depth_cache),
    max_pairs_per_gap=MAX_PAIRS_PER_GAP,
    show_progress=True,
)

print_experiment_summary(synchronized_frames, results)
