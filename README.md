# RGB-D Relative Pose Matching

Minimal research project comparing ORB and SuperPoint + LightGlue for RGB-D relative pose estimation on the TUM RGB-D dataset.

Core question: *How do classical and learned feature matchers degrade under increasing inter-frame motion?*

Pipeline:
```
RGB-D frame pair -> feature matching -> 3D-2D PnP-RANSAC -> relative pose error
```

