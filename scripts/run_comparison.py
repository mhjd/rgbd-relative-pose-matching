import sys
sys.path.append("src")

from datetime import datetime
import json
from pathlib import Path
import time

import torch
from dataset import SEQUENCE, get_synchronized_frames
from evaluation import FRAME_GAPS, evaluate_frame_gaps
from experiment_results import format_float
from lightglue_pipeline import LightGlueMatcher
from orb_pipeline import OrbMatcher

MAX_PAIRS_PER_GAP = None
LIGHTGLUE_MAX_KEYPOINTS = 512


def create_comparison_output_dir():
    outputs_dir = Path("outputs")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = outputs_dir / f"comparison_{timestamp}"
    suffix = 2
    while output_dir.exists():
        output_dir = outputs_dir / f"comparison_{timestamp}_{suffix}"
        suffix += 1
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def write_json_file(path, data):
    with open(path, "w", encoding="utf-8") as output_file:
        json.dump(data, output_file, indent=2)
        output_file.write("\n")


def read_json_file(path):
    with open(path, "r", encoding="utf-8") as input_file:
        return json.load(input_file)


def write_jsonl_record(output_file, record):
    json.dump(record, output_file)
    output_file.write("\n")
    output_file.flush()


def build_timing_metadata(orb_matcher, lightglue_matcher, total_wall_clock_time):
    return {
        "orb": {
            "image_loading": orb_matcher.image_loading_time,
            "orb_extraction": orb_matcher.orb_extraction_time,
            "orb_matching": orb_matcher.orb_matching_time,
            "depth_correspondences": orb_matcher.correspondence_time,
        },
        "lightglue": {
            "image_loading": lightglue_matcher.image_loading_time,
            "superpoint_extraction": lightglue_matcher.superpoint_extraction_time,
            "lightglue_matching": lightglue_matcher.lightglue_matching_time,
            "depth_correspondences": lightglue_matcher.correspondence_time,
        },
        "total_wall_clock": total_wall_clock_time,
    }


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
output_dir = create_comparison_output_dir()
pairs_path = output_dir / "pairs.jsonl"
metadata_path = output_dir / "metadata.json"
write_json_file(metadata_path, {
    "created_at": datetime.now().isoformat(timespec="seconds"),
    "sequence": SEQUENCE.rstrip("/"),
    "frame_gaps": FRAME_GAPS,
    "max_pairs_per_gap": MAX_PAIRS_PER_GAP,
    "lightglue_max_keypoints": LIGHTGLUE_MAX_KEYPOINTS,
    "device": device,
    "synchronized_frame_count": len(synchronized_frames),
    "pair_results_file": pairs_path.name,
})

print(f"Device : {device}")
print(f"LightGlue max keypoints : {LIGHTGLUE_MAX_KEYPOINTS}")
print(f"Synchronized frames : {len(synchronized_frames)}\n")
print(f"Output directory : {output_dir}\n")

orb_results = []
lightglue_results = []

with open(pairs_path, "w", encoding="utf-8") as pairs_file:
    def write_pair_result(record):
        write_jsonl_record(pairs_file, record)

    for frame_gap in FRAME_GAPS:
        print(f"Evaluating ORB gap {frame_gap}")
        orb_results.extend(evaluate_frame_gaps(
            synchronized_frames,
            orb_matcher.match_pair,
            frame_gaps=[frame_gap],
            max_pairs_per_gap=MAX_PAIRS_PER_GAP,
            show_progress=True,
            method_name="ORB",
            pair_result_writer=write_pair_result,
        ))

        print(f"Evaluating SuperPoint + LightGlue gap {frame_gap}")
        lightglue_results.extend(evaluate_frame_gaps(
            synchronized_frames,
            lightglue_matcher.match_pair,
            frame_gaps=[frame_gap],
            max_pairs_per_gap=MAX_PAIRS_PER_GAP,
            show_progress=True,
            method_name="LightGlue",
            pair_result_writer=write_pair_result,
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
total_wall_clock_time = time.perf_counter() - script_start_time
metadata = read_json_file(metadata_path)
metadata["timings"] = build_timing_metadata(orb_matcher, lightglue_matcher, total_wall_clock_time)
write_json_file(metadata_path, metadata)
print(f"Total wall-clock time : {total_wall_clock_time:.2f}s\n")
