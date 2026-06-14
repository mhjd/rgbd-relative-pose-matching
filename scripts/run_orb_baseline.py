import sys
sys.path.append("src")

from dataset import get_synchronized_frames
from evaluation import evaluate_frame_gaps
from experiment_results import print_experiment_summary
from orb_pipeline import OrbMatcher

synchronized_frames = get_synchronized_frames()
orb_matcher = OrbMatcher()

results = evaluate_frame_gaps(
    synchronized_frames,
    orb_matcher.match_pair,
)

print_experiment_summary(synchronized_frames, results)
orb_matcher.print_timing_summary()
