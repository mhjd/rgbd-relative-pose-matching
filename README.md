# RGB-D Relative Pose Matching

This project compares classical and learned feature matching methods for estimating camera motion from RGB-D image pairs. It uses ORB as a classical baseline and SuperPoint + LightGlue as a learned matching pipeline, with the goal of studying how both methods behave when the motion between frames increases.

## Goal

Estimating the motion of a camera between two observations is a common geometric task in computer vision and robotics. It is useful whenever a system needs to understand how its viewpoint changed while observing a scene.

This project studies this problem using RGB-D data. Each RGB-D frame contains a color image and a depth image. The color image shows what the camera sees. The depth image gives the distance from the camera to the visible surfaces.

The goal is to estimate how the camera moved from the first frame to the second. In geometric vision, this is called estimating the relative pose between the two frames. The relative pose contains the rotation and translation of the camera between the two observations.

To estimate this motion, the system first needs to find visual correspondences between the two color images. A correspondence means that a point seen in the first image is matched with the same physical point in the second image.

This project compares a classical correspondence method, ORB, with a learned correspondence pipeline, SuperPoint and LightGlue. ORB is a hand-designed feature detector and descriptor. SuperPoint and LightGlue use neural networks to detect, describe, and match image points. Both methods are evaluated on the same frame pairs and with the same geometric pose-estimation backend, so that the comparison focuses on the usefulness of the visual correspondences.

## Pipeline

For each pair of frames, the project follows the same geometric pipeline for both matching methods.

```
RGB-D frame i and RGB frame j
-> feature matching
-> convert matched pixels from frame i into 3D points using depth
-> 3D-to-2D PnP-RANSAC
-> estimated camera motion from i to j
-> comparison with ground truth
```

Feature matching first produces correspondences between points in the two color images. The depth map from the first frame is then used to convert matched 2D pixels from frame `i` into 3D points. The corresponding points in frame `j` remain 2D image observations.

PnP-RANSAC is the geometric step that estimates the camera motion from these 3D-to-2D correspondences. The PnP part estimates the camera pose that best projects the 3D points from frame `i` onto their matched 2D locations in frame `j`. The RANSAC part makes this estimation robust by rejecting matches that are not geometrically consistent with the dominant camera motion.

This design avoids requiring valid depth in both frames for every match. It also keeps the geometric backend identical for ORB and SuperPoint + LightGlue, which makes the comparison focus on the matching stage.

## Experimental Design

The experiment varies the temporal gap between paired frames. A small gap corresponds to a small camera motion, while a larger gap usually creates a harder matching and pose-estimation problem.

For each frame gap, ORB and SuperPoint + LightGlue are evaluated on the same frame pairs. Both methods use the same depth maps, the same camera intrinsics, the same PnP-RANSAC backend, and the same evaluation metrics. This keeps the geometric part of the pipeline fixed and makes the comparison focus on the visual correspondences produced by each method.

The analysis will report matching statistics, PnP success and failure cases, relative pose error, and runtime. This is intended to show not only which method produces more matches, but which method produces matches that remain useful for camera motion estimation as the frame gap increases.

## Current Status

This repository is a work in progress.

At this stage, the project implements the data preparation and ORB baseline parts of the study. RGB, depth, and ground-truth measurements are synchronized by timestamp. Frame pairs are generated for several temporal gaps. ORB matches are converted into 3D-to-2D correspondences using valid depth, and PnP-RANSAC is used to estimate relative camera motion.

The current metrics report matching statistics, PnP success and failure counts, inlier ratios, ground-truth motion magnitude, and pose errors for the ORB baseline.

The learned matching pipeline based on SuperPoint + LightGlue is planned next. The goal is to evaluate it with the same frame pairs, depth data, PnP backend, and metrics used for ORB.
