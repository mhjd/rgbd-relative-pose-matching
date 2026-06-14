import sys
sys.path.append("src")

import torch
from dataset import get_synchronized_frames
from evaluation import evaluate_frame_gaps
from experiment_results import print_experiment_summary
from lightglue_pipeline import LightGlueMatcher

MAX_PAIRS_PER_GAP = 20

device = "mps" if torch.backends.mps.is_available() else "cpu"
synchronized_frames = get_synchronized_frames()
lightglue_matcher = LightGlueMatcher(device)

print(f"Device : {device}")
results = evaluate_frame_gaps(
    synchronized_frames,
    lightglue_matcher.match_pair,
    max_pairs_per_gap=MAX_PAIRS_PER_GAP,
    show_progress=True,
)

print_experiment_summary(synchronized_frames, results)
lightglue_matcher.print_timing_summary()
