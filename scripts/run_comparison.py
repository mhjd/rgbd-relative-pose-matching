import sys
sys.path.append("src")

import time

import torch
from dataset import get_synchronized_frames
from evaluation import FRAME_GAPS, evaluate_frame_gaps
from experiment_results import format_float
from lightglue_pipeline import LightGlueMatcher
from orb_pipeline import OrbMatcher

MAX_PAIRS_PER_GAP = None
LIGHTGLUE_MAX_KEYPOINTS = 512


def print_comparison_table(method_results, title=None):
    if title is not None:
        print(title)
    print(
        f"{'method':>10} | "
        f"{'gap':>3} | "
        f"{'pairs':>5} | "
        f"{'gt_t':>5} | "
        f"{'gt_r':>5} | "
        f"{'matches':>7} | "
        f"{'corr':>6} | "
        f"{'pnp_ok':>6} | "
        f"{'pnp_fail':>8} | "
        f"{'inl_ratio':>9} | "
        f"{'t_med':>5} | "
        f"{'t_p95':>6} | "
        f"{'r_med':>5} | "
        f"{'r_p95':>6}"
    )
    rows = [
        (method_name, result)
        for method_name, results in method_results
        for result in results
    ]
    for row_index, (method_name, result) in enumerate(rows):
        line_end = "\n\n" if row_index == len(rows) - 1 else "\n"
        print(
            f"{method_name:>10} | "
            f"{result.gap:>3} | "
            f"{result.pairs:>5} | "
            f"{format_float(result.mean_gt_translation, 3):>5} | "
            f"{format_float(result.mean_gt_rotation, 2):>5} | "
            f"{format_float(result.mean_matches, 1):>7} | "
            f"{format_float(result.mean_correspondences, 1):>6} | "
            f"{result.pnp_success:>6} | "
            f"{result.pnp_failed:>8} | "
            f"{format_float(result.mean_pnp_inlier_ratio, 3):>9} | "
            f"{format_float(result.median_translation_error, 3):>5} | "
            f"{format_float(result.p95_translation_error, 3):>6} | "
            f"{format_float(result.median_rotation_error, 2):>5} | "
            f"{format_float(result.p95_rotation_error, 2):>6}",
            end=line_end,
        )


script_start_time = time.perf_counter()
device = "mps" if torch.backends.mps.is_available() else "cpu"
synchronized_frames = get_synchronized_frames()
orb_matcher = OrbMatcher()
lightglue_matcher = LightGlueMatcher(device, max_num_keypoints=LIGHTGLUE_MAX_KEYPOINTS)

print(f"Device : {device}")
print(f"LightGlue max keypoints : {LIGHTGLUE_MAX_KEYPOINTS}")
print(f"Synchronized frames : {len(synchronized_frames)}\n")

orb_results = []
lightglue_results = []

for frame_gap in FRAME_GAPS:
    print(f"Evaluating ORB gap {frame_gap}")
    orb_results.extend(evaluate_frame_gaps(
        synchronized_frames,
        orb_matcher.match_pair,
        frame_gaps=[frame_gap],
        max_pairs_per_gap=MAX_PAIRS_PER_GAP,
        show_progress=True,
    ))

    print(f"Evaluating SuperPoint + LightGlue gap {frame_gap}")
    lightglue_results.extend(evaluate_frame_gaps(
        synchronized_frames,
        lightglue_matcher.match_pair,
        frame_gaps=[frame_gap],
        max_pairs_per_gap=MAX_PAIRS_PER_GAP,
        show_progress=True,
    ))

    print_comparison_table(
        [
            ("ORB", orb_results),
            ("LightGlue", lightglue_results),
        ],
        title=f"Intermediate comparison through gap {frame_gap}",
    )

    orb_matcher.print_timing_summary("ORB intermediate timings")
    lightglue_matcher.print_timing_summary("SuperPoint + LightGlue intermediate timings")

print_comparison_table(
    [
        ("ORB", orb_results),
        ("LightGlue", lightglue_results),
    ],
    title="Comparison",
)

orb_matcher.print_timing_summary("ORB")
lightglue_matcher.print_timing_summary("SuperPoint + LightGlue")
print(f"Total wall-clock time : {time.perf_counter() - script_start_time:.2f}s\n")
