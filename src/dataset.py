DATA_PATH = "data/"
SEQUENCE = "rgbd_dataset_freiburg1_xyz/"
MAX_RGB_DEPTH_TIMESTAMP_DIFF = 0.02
MAX_RGB_GROUNDTRUTH_TIMESTAMP_DIFF = 0.05

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
    # Keep the candidate with the smallest timestamp distance from the reference.
    best_candidate = min(possible_candidates, key=lambda candidate_line: abs(get_timestamp(candidate_line) - reference_timestamp))
    time_diff = get_timestamp(best_candidate) - reference_timestamp
    if abs(time_diff) > max_timestamp_diff:
        return None, candidate_index, time_diff
    return best_candidate, candidate_index, time_diff

def synchronize_frames(rgb, depth, groundtruth):
    """
    Find the closest depth and ground truth for each RGB image.
    
    Measurements were not captured at exactly the same time, so timestamps
    must be associated by temporal proximity.
    """
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

def make_frame_pairs(synchronized_frames, frame_gap):
    frame_pairs = []
    for frame_index in range(len(synchronized_frames) - frame_gap):
        frame_pairs.append((synchronized_frames[frame_index], synchronized_frames[frame_index + frame_gap]))
    return frame_pairs
