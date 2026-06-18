import sys
import time

import cv2

from dataset import get_depth_image, get_rgb_path_from_frame
from evaluation import MatchResult
from geometry import build_3d_2d_correspondences


class OrbMatcher:
    def __init__(self):
        self.orb = cv2.ORB_create()
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.features_cache = {}
        self.depth_cache = {}
        self.image_loading_time = 0.0
        self.orb_extraction_time = 0.0
        self.orb_matching_time = 0.0
        self.correspondence_time = 0.0

    def get_features(self, frame):
        rgb_path = get_rgb_path_from_frame(frame)
        if rgb_path in self.features_cache:
            return self.features_cache[rgb_path]
        start_time = time.perf_counter()
        rgb_image = cv2.imread(rgb_path, cv2.IMREAD_GRAYSCALE)
        self.image_loading_time += time.perf_counter() - start_time
        if rgb_image is None:
            print(f"Could not read RGB image : {rgb_path}")
            sys.exit(1)
        start_time = time.perf_counter()
        keypoints, descriptors = self.orb.detectAndCompute(rgb_image, None)
        self.orb_extraction_time += time.perf_counter() - start_time
        self.features_cache[rgb_path] = (keypoints, descriptors)
        return keypoints, descriptors

    def match_pair(self, frame_pair):
        """
        Match ORB descriptors between two frames.

        Return named matching statistics for one frame pair.
        """
        frame_i, frame_j = frame_pair
        keypoints_i, descriptors_i = self.get_features(frame_i)
        keypoints_j, descriptors_j = self.get_features(frame_j)
        if descriptors_i is None or descriptors_j is None:
            return MatchResult(
                keypoints_i_count=len(keypoints_i),
                keypoints_j_count=len(keypoints_j),
                match_count=0,
                correspondence_count=0,
                object_points=[],
                image_points=[],
                best_match_distance=None,
            )
        start_time = time.perf_counter()
        matches = self.matcher.match(descriptors_i, descriptors_j)
        matches = sorted(matches, key=lambda orb_match: orb_match.distance)
        self.orb_matching_time += time.perf_counter() - start_time

        start_time = time.perf_counter()
        depth_i = get_depth_image(frame_i, self.depth_cache)
        object_points, image_points = build_3d_2d_correspondences(matches, keypoints_i, keypoints_j, depth_i)
        self.correspondence_time += time.perf_counter() - start_time
        best_match_distance = None
        if len(matches) > 0:
            best_match_distance = matches[0].distance
        return MatchResult(
            keypoints_i_count=len(keypoints_i),
            keypoints_j_count=len(keypoints_j),
            match_count=len(matches),
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
        print(f"  ORB extraction        : {self.orb_extraction_time:.2f}s")
        print(f"  ORB matching          : {self.orb_matching_time:.2f}s")
        print(f"  Depth/correspondences : {self.correspondence_time:.2f}s\n")
