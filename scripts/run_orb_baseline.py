import sys
sys.path.append("src")

import cv2
from dataset import DATA_PATH, SEQUENCE, get_content, get_data_content, synchronize_frames, make_frame_pairs

FRAME_GAPS = [1, 2, 5, 10]
DEPTH_SCALE = 5000.0
FX = 517.3
FY = 516.5
CX = 318.6
CY = 255.3

def get_rgb_path_from_frame(frame):
    rgb_line, _, _, _, _ = frame
    return DATA_PATH + SEQUENCE + get_content(rgb_line)[0]

def get_depth_path_from_frame(frame):
    _, depth_line, _, _, _ = frame
    return DATA_PATH + SEQUENCE + get_content(depth_line)[0]

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
    for orb_match in matches:
        u, v = keypoints_i[orb_match.queryIdx].pt
        u = int(round(u))
        v = int(round(v))
        if v < 0 or v >= depth_image.shape[0] or u < 0 or u >= depth_image.shape[1]:
            # invalid keypoint
            continue
        depth_value = depth_image[v, u]
        if depth_value == 0:
            continue
        object_points.append(backproject_pixel_to_3d(u, v, depth_value))
        image_points.append(keypoints_j[orb_match.trainIdx].pt)
    return object_points, image_points

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
        "best_match_distance": best_match_distance,
    }

def inspect_orb_pairs(frame_pairs, frame_gap, orb, matcher, features_cache, depth_cache):
    """
    Run ORB matching on all pairs for one frame gap and print summary stats.
    """
    keypoints_i_counts = []
    keypoints_j_counts = []
    match_counts = []
    correspondence_counts = []
    failed_pairs = 0
    for frame_pair in frame_pairs:
        orb_result = match_orb_pair(frame_pair, orb, matcher, features_cache, depth_cache)
        keypoints_i_counts.append(orb_result["keypoints_i_count"])
        keypoints_j_counts.append(orb_result["keypoints_j_count"])
        match_counts.append(orb_result["match_count"])
        correspondence_counts.append(orb_result["correspondence_count"])
        if orb_result["best_match_distance"] is None:
            failed_pairs += 1
    print(f"ORB gap {frame_gap} pairs : {len(frame_pairs)}")
    print(f"ORB gap {frame_gap} failed pairs : {failed_pairs}")
    print(f"ORB gap {frame_gap} mean keypoints i : {sum(keypoints_i_counts) / len(keypoints_i_counts)}")
    print(f"ORB gap {frame_gap} mean keypoints j : {sum(keypoints_j_counts) / len(keypoints_j_counts)}")
    print(f"ORB gap {frame_gap} mean raw matches : {sum(match_counts) / len(match_counts)}")
    print(f"ORB gap {frame_gap} mean 3D-to-2D correspondences : {sum(correspondence_counts) / len(correspondence_counts)}")

rgb = get_data_content("rgb.txt")
depth = get_data_content("depth.txt")
groundtruth = get_data_content("groundtruth.txt")
synchronized_frames, _, _ = synchronize_frames(rgb, depth, groundtruth)
orb = cv2.ORB_create()
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
features_cache = {}
depth_cache = {}

print(f"Synchronized frames : {len(synchronized_frames)}")

for frame_gap in FRAME_GAPS:
    frame_pairs = make_frame_pairs(synchronized_frames, frame_gap)
    inspect_orb_pairs(frame_pairs, frame_gap, orb, matcher, features_cache, depth_cache)
