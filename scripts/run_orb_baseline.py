import sys
sys.path.append("src")

import cv2
from dataset import get_rgb_path_from_frame, get_depth_image, get_synchronized_frames
from evaluation import evaluate_frame_gaps
from geometry import build_3d_2d_correspondences
from experiment_results import print_experiment_summary

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

synchronized_frames = get_synchronized_frames()
orb = cv2.ORB_create()
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
features_cache = {}
depth_cache = {}

results = evaluate_frame_gaps(
    synchronized_frames,
    lambda frame_pair: match_orb_pair(frame_pair, orb, matcher, features_cache, depth_cache),
)

print_experiment_summary(synchronized_frames, results)
