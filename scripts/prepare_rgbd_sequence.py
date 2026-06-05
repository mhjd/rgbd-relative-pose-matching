import sys

DATA_PATH = "data/"
SEQUENCE = "rgbd_dataset_freiburg1_xyz/"
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
check_sanity(depth, 1, "depth")
check_sanity(groundtruth, 7, "ground truth")
