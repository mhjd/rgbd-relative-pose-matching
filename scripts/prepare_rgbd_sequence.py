import sys
sys.path.append("src")

import cv2
from dataset import DATA_PATH, SEQUENCE, get_content, get_data_content, synchronize_frames, make_frame_pairs

DEPTH_SCALE = 5000.0
FRAME_GAPS = [1, 2, 5, 10]

def check_sanity(data, expected_number_element, data_name):
    for _, content in data:
        if len(content) != expected_number_element:
            print(f"Data name : {data_name}")
            print(content)
            sys.exit(1)
    print(f"{data_name} have the expected number of elements in its content")

def inspect_first_synchronized_frame(synchronized_frames):
    rgb_line, depth_line, _, _, _ = synchronized_frames[0]
    rgb_path = DATA_PATH + SEQUENCE + get_content(rgb_line)[0]
    depth_path = DATA_PATH + SEQUENCE + get_content(depth_line)[0]
    rgb_image = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if rgb_image is None:
        print(f"Could not read RGB image : {rgb_path}")
        sys.exit(1)
    if depth_image is None:
        print(f"Could not read depth image : {depth_path}")
        sys.exit(1)
    valid_depth = depth_image[depth_image > 0] / DEPTH_SCALE
    print(f"First synchronized RGB path : {rgb_path}")
    print(f"First synchronized depth path : {depth_path}")
    print(f"RGB image shape : {rgb_image.shape}")
    print(f"Depth image shape : {depth_image.shape}")
    print(f"Depth image dtype : {depth_image.dtype}")
    print(f"Valid depth min meters : {valid_depth.min()}")
    print(f"Valid depth max meters : {valid_depth.max()}")

rgb = get_data_content("rgb.txt")
depth = get_data_content("depth.txt")
groundtruth = get_data_content("groundtruth.txt")

print(f"RGB entries : {len(rgb)}")
print(f"Depth entries : {len(depth)}")
print(f"Ground truth entries : {len(groundtruth)}")

print(f"First RGB : {rgb[0]}")
print(f"First Depth : {depth[0]}")
print(f"First Ground truth : {groundtruth[0]}")

check_sanity(rgb, 1, "rgb")
check_sanity(depth,1, "depth")
check_sanity(groundtruth, 7, "ground truth")

synchronized_frames, dropped_without_depth, dropped_without_groundtruth = synchronize_frames(rgb, depth, groundtruth)
rgb_depth_time_diffs = [rgb_depth_time_diff for _, _, _, rgb_depth_time_diff, _ in synchronized_frames]
rgb_groundtruth_time_diffs = [rgb_groundtruth_time_diff for _, _, _, _, rgb_groundtruth_time_diff in synchronized_frames]

print(f"Synchronized frames : {len(synchronized_frames)}")
print(f"Dropped RGB without depth : {dropped_without_depth}")
print(f"Dropped RGB without ground truth : {dropped_without_groundtruth}")
print(f"Min RGB-depth time diff : {min(rgb_depth_time_diffs)}")
print(f"Max RGB-depth time diff : {max(rgb_depth_time_diffs)}")
print(f"Min RGB-ground truth time diff : {min(rgb_groundtruth_time_diffs)}")
print(f"Max RGB-ground truth time diff : {max(rgb_groundtruth_time_diffs)}")

inspect_first_synchronized_frame(synchronized_frames)

for frame_gap in FRAME_GAPS:
    frame_pairs = make_frame_pairs(synchronized_frames, frame_gap)
    print(f"Frame pairs with gap {frame_gap} : {len(frame_pairs)}")
