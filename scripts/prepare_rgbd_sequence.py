import sys
import cv2

DATA_PATH = "data/"
SEQUENCE = "rgbd_dataset_freiburg1_xyz/"
MAX_RGB_DEPTH_TIMESTAMP_DIFF = 0.02
MAX_RGB_GROUNDTRUTH_TIMESTAMP_DIFF = 0.05
DEPTH_SCALE = 5000.0
FRAME_GAPS = [1, 2, 5, 10]

def get_timestamp(data_line):
    return data_line[0]
def get_content(data_line):
    return data_line[1]

def get_data_content(data_filename):
    data_content = []
    with open(DATA_PATH + SEQUENCE + data_filename , "r") as data_file:
       for line in data_file:
           if line[0] != '#':
               parsed_line = line.split()
               timestamp_str =  parsed_line[0]
               content = parsed_line[1:]
               data_content.append((float(timestamp_str), content))
    return data_content

def check_sanity(data, expected_number_element, data_name):
    for _, content in data:
        if len(content) != expected_number_element: 
            print(f"Data name : {data_name}")
            print(content)
            sys.exit(1)
    print(f"{data_name} have the expected number of elements in its content")
            
    
def find_closest_by_timestamp(reference_timestamp, candidate_data, candidate_index, max_timestamp_diff):
    while candidate_index < len(candidate_data) and get_timestamp(candidate_data[candidate_index]) < reference_timestamp:
        candidate_index += 1
    possible_candidates = []
    if candidate_index < len(candidate_data):
        possible_candidates.append(candidate_data[candidate_index])
    if candidate_index > 0:
        possible_candidates.append(candidate_data[candidate_index - 1])
    if len(possible_candidates) == 0:
        return None, candidate_index, None
    best_candidate = min(possible_candidates, key=lambda candidate_line: abs(get_timestamp(candidate_line) - reference_timestamp))
    time_diff = get_timestamp(best_candidate) - reference_timestamp
    if abs(time_diff) > max_timestamp_diff:
        return None, candidate_index, time_diff
    return best_candidate, candidate_index, time_diff

def synchronize_frames(rgb, depth, groundtruth):
    depth_index = 0
    groundtruth_index = 0
    synchronized_frames = []
    dropped_without_depth = 0
    dropped_without_groundtruth = 0
    for rgb_line in rgb:
        rgb_timestamp = get_timestamp(rgb_line)
        depth_line, depth_index, rgb_depth_time_diff = find_closest_by_timestamp(rgb_timestamp, depth, depth_index, MAX_RGB_DEPTH_TIMESTAMP_DIFF)
        groundtruth_line, groundtruth_index, rgb_groundtruth_time_diff = find_closest_by_timestamp(rgb_timestamp, groundtruth, groundtruth_index, MAX_RGB_GROUNDTRUTH_TIMESTAMP_DIFF)
        if depth_line is None:
            dropped_without_depth += 1
            continue
        if groundtruth_line is None:
            dropped_without_groundtruth += 1
            continue
        synchronized_frames.append((rgb_line, depth_line, groundtruth_line, rgb_depth_time_diff, rgb_groundtruth_time_diff))
    return synchronized_frames, dropped_without_depth, dropped_without_groundtruth

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

def make_frame_pairs(synchronized_frames, frame_gap):
    frame_pairs = []
    for frame_index in range(len(synchronized_frames) - frame_gap):
        frame_pairs.append((synchronized_frames[frame_index], synchronized_frames[frame_index + frame_gap]))
    return frame_pairs

            
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
