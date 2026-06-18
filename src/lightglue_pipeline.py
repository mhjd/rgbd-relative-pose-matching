import time

import torch
from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, rbd

from dataset import get_depth_image, get_rgb_path_from_frame
from evaluation import MatchResult
from geometry import build_3d_2d_correspondences_from_pixels


class LightGlueMatcher:
    def __init__(self, device, max_num_keypoints=2048):
        self.device = device
        self.max_num_keypoints = max_num_keypoints
        self.extractor = SuperPoint(max_num_keypoints=max_num_keypoints).eval().to(device)
        self.matcher = LightGlue(features="superpoint").eval().to(device)
        self.features_cache = {}
        self.depth_cache = {}
        self.image_loading_time = 0.0
        self.superpoint_extraction_time = 0.0
        self.lightglue_matching_time = 0.0
        self.correspondence_time = 0.0

    def synchronize_device(self):
        if self.device == "mps":
            torch.mps.synchronize()
        elif self.device == "cuda":
            torch.cuda.synchronize()

    def get_features(self, frame):
        rgb_path = get_rgb_path_from_frame(frame)
        if rgb_path in self.features_cache:
            return self.features_cache[rgb_path]
        start_time = time.perf_counter()
        image = load_image(rgb_path).to(self.device)
        self.synchronize_device()
        self.image_loading_time += time.perf_counter() - start_time

        self.synchronize_device()
        start_time = time.perf_counter()
        with torch.inference_mode():
            features = self.extractor.extract(image)
        self.synchronize_device()
        self.superpoint_extraction_time += time.perf_counter() - start_time
        self.features_cache[rgb_path] = features
        return features

    def match_pair(self, frame_pair):
        """
        Match SuperPoint features with LightGlue between two frames.

        Return named matching statistics for one frame pair.
        """
        frame_i, frame_j = frame_pair
        features_i = self.get_features(frame_i)
        features_j = self.get_features(frame_j)

        self.synchronize_device()
        start_time = time.perf_counter()
        with torch.inference_mode():
            matches = self.matcher({"image0": features_i, "image1": features_j})
        self.synchronize_device()
        self.lightglue_matching_time += time.perf_counter() - start_time

        features_i = rbd(features_i)
        features_j = rbd(features_j)
        matches = rbd(matches)

        matched_indices = matches["matches"]
        points_i = features_i["keypoints"][matched_indices[:, 0]]
        points_j = features_j["keypoints"][matched_indices[:, 1]]

        start_time = time.perf_counter()
        depth_i = get_depth_image(frame_i, self.depth_cache)
        object_points, image_points = build_3d_2d_correspondences_from_pixels(points_i, points_j, depth_i)
        self.correspondence_time += time.perf_counter() - start_time

        best_match_distance = None
        if len(matched_indices) > 0:
            best_match_distance = 0

        return MatchResult(
            keypoints_i_count=len(features_i["keypoints"]),
            keypoints_j_count=len(features_j["keypoints"]),
            match_count=len(matched_indices),
            correspondence_count=len(object_points),
            object_points=object_points,
            image_points=image_points,
            best_match_distance=best_match_distance,
        )

    def print_timing_summary(self, title=None):
        if title is not None:
            print(title)
        print("Timing summary")
        print(f"  Image loading         : {self.image_loading_time:.2f}s")
        print(f"  SuperPoint extraction : {self.superpoint_extraction_time:.2f}s")
        print(f"  LightGlue matching    : {self.lightglue_matching_time:.2f}s")
        print(f"  Depth/correspondences : {self.correspondence_time:.2f}s\n")
