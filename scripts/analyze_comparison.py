import sys
sys.path.append("src")

import argparse
import json
import os
from pathlib import Path
from tempfile import gettempdir

cache_dir = Path(gettempdir()) / "rgbd-relative-pose-matching-cache"
os.environ.setdefault("MPLCONFIGDIR", str(cache_dir / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiment_results import FrameGapStatistics, format_float

METHOD_ORDER = ["ORB", "LightGlue"]


def get_latest_comparison_output_dir():
    output_dirs = sorted(Path("outputs").glob("comparison_*"), key=lambda path: path.stat().st_mtime)
    if len(output_dirs) == 0:
        raise FileNotFoundError("No comparison output directory found in outputs/")
    return output_dirs[-1]


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as input_file:
        for line in input_file:
            records.append(json.loads(line))
    return records


def load_json(path):
    with open(path, "r", encoding="utf-8") as input_file:
        return json.load(input_file)


def mean(values):
    return sum(values) / len(values)


def summarize_records(records, frame_gap):
    result = FrameGapStatistics(frame_gap, len(records))
    result.matching_failed = sum(record["best_match_distance"] is None for record in records)
    result.mean_keypoints_i = mean([record["keypoints_i"] for record in records])
    result.mean_keypoints_j = mean([record["keypoints_j"] for record in records])
    result.mean_matches = mean([record["matches"] for record in records])
    result.mean_correspondences = mean([record["correspondences"] for record in records])
    result.mean_gt_translation = mean([record["groundtruth_translation"] for record in records])
    result.mean_gt_rotation = mean([record["groundtruth_rotation"] for record in records])

    successful_records = [record for record in records if record["pnp_success"]]
    result.pnp_success = len(successful_records)
    result.pnp_failed = len(records) - len(successful_records)
    if len(successful_records) > 0:
        pnp_inliers = [record["pnp_inliers"] for record in successful_records]
        pnp_inlier_ratios = [record["pnp_inlier_ratio"] for record in successful_records]
        translation_errors = [record["translation_error"] for record in successful_records]
        rotation_errors = [record["rotation_error"] for record in successful_records]
        result.mean_pnp_inliers = mean(pnp_inliers)
        result.mean_pnp_inlier_ratio = mean(pnp_inlier_ratios)
        result.mean_translation_error = mean(translation_errors)
        result.median_translation_error = float(np.median(translation_errors))
        result.p95_translation_error = float(np.percentile(translation_errors, 95))
        result.max_translation_error = max(translation_errors)
        result.mean_rotation_error = mean(rotation_errors)
        result.median_rotation_error = float(np.median(rotation_errors))
        result.p95_rotation_error = float(np.percentile(rotation_errors, 95))
        result.max_rotation_error = max(rotation_errors)
    return result


def group_records(records):
    grouped_records = {}
    for record in records:
        key = (record["method"], record["gap"])
        grouped_records.setdefault(key, []).append(record)
    return grouped_records


def build_method_results(records):
    grouped_records = group_records(records)
    methods = [method for method in METHOD_ORDER if any(key[0] == method for key in grouped_records)]
    methods.extend(sorted({key[0] for key in grouped_records} - set(methods)))
    method_results = []
    for method in methods:
        frame_gaps = sorted(key[1] for key in grouped_records if key[0] == method)
        results = [
            summarize_records(grouped_records[(method, frame_gap)], frame_gap)
            for frame_gap in frame_gaps
        ]
        method_results.append((method, results))
    return method_results


def statistics_to_json(method_results):
    summary = []
    for method_name, results in method_results:
        for result in results:
            summary.append({
                "method": method_name,
                "gap": result.gap,
                "pairs": result.pairs,
                "matching_failed": result.matching_failed,
                "mean_keypoints_i": result.mean_keypoints_i,
                "mean_keypoints_j": result.mean_keypoints_j,
                "mean_matches": result.mean_matches,
                "mean_correspondences": result.mean_correspondences,
                "mean_gt_translation": result.mean_gt_translation,
                "mean_gt_rotation": result.mean_gt_rotation,
                "pnp_success": result.pnp_success,
                "pnp_failed": result.pnp_failed,
                "mean_pnp_inliers": result.mean_pnp_inliers,
                "mean_pnp_inlier_ratio": result.mean_pnp_inlier_ratio,
                "mean_translation_error": result.mean_translation_error,
                "median_translation_error": result.median_translation_error,
                "p95_translation_error": result.p95_translation_error,
                "max_translation_error": result.max_translation_error,
                "mean_rotation_error": result.mean_rotation_error,
                "median_rotation_error": result.median_rotation_error,
                "p95_rotation_error": result.p95_rotation_error,
                "max_rotation_error": result.max_rotation_error,
            })
    return summary


def write_summary(path, summary):
    with open(path, "w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, indent=2)
        output_file.write("\n")


def get_method_series(method_results, metric_name):
    series = {}
    for method_name, results in method_results:
        series[method_name] = {
            "gaps": [result.gap for result in results],
            "values": [getattr(result, metric_name) for result in results],
        }
    return series


def plot_rotation_metric_vs_gap(method_results, metric_name, ylabel, title, filename):
    colors = {
        "ORB": "#4C78A8",
        "LightGlue": "#F58518",
    }
    metric_series = get_method_series(method_results, metric_name)
    fig, ax = plt.subplots(figsize=(8, 4))

    for method_name, series in metric_series.items():
        ax.plot(
            series["gaps"],
            series["values"],
            marker="o",
            linewidth=2,
            color=colors.get(method_name),
            label=method_name,
        )
    first_method = next(iter(metric_series))
    ax.set_xscale("log")
    ax.set_xticks(metric_series[first_method]["gaps"])
    ax.set_xticklabels([str(gap) for gap in metric_series[first_method]["gaps"]])
    ax.set_xlabel("Frame gap")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    figure_path = output_dir / filename
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)
    return figure_path


def plot_rotation_error_figures(method_results, output_dir):
    return [
        plot_rotation_metric_vs_gap(
            method_results,
            "median_rotation_error",
            "Median rotation error (deg)",
            "Median rotation error under increasing frame gap",
            "median_rotation_error_vs_gap.png",
        ),
        plot_rotation_metric_vs_gap(
            method_results,
            "mean_rotation_error",
            "Mean rotation error (deg)",
            "Mean rotation error under increasing frame gap",
            "mean_rotation_error_vs_gap.png",
        ),
        plot_rotation_metric_vs_gap(
            method_results,
            "p95_rotation_error",
            "95th percentile rotation error (deg)",
            "95th percentile rotation error under increasing frame gap",
            "p95_rotation_error_vs_gap.png",
        ),
    ]


def plot_pnp_failures_vs_gap(method_results, output_dir):
    colors = {
        "ORB": "#4C78A8",
        "LightGlue": "#F58518",
    }
    fig, ax = plt.subplots(figsize=(8, 4))
    gap_positions = np.arange(len(method_results[0][1]))
    bar_width = 0.35

    for method_index, (method_name, results) in enumerate(method_results):
        offsets = gap_positions + (method_index - 0.5) * bar_width
        ax.bar(
            offsets,
            [result.pnp_failed for result in results],
            width=bar_width,
            color=colors.get(method_name),
            label=method_name,
        )

    ax.set_xticks(gap_positions)
    ax.set_xticklabels([str(result.gap) for result in method_results[0][1]])
    ax.set_xlabel("Frame gap")
    ax.set_ylabel("PnP failures")
    ax.set_title("PnP failures under increasing frame gap")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    figure_path = output_dir / "pnp_failures_vs_gap.png"
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)
    return figure_path


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


def print_timing_summary(metadata):
    timings = metadata.get("timings")
    if timings is None:
        print("Timing metadata is not available for this run.\n")
        return

    orb_timings = timings["orb"]
    lightglue_timings = timings["lightglue"]
    print("ORB")
    print("Timing summary")
    print(f"  Image loading         : {orb_timings['image_loading']:.2f}s")
    print(f"  ORB extraction        : {orb_timings['orb_extraction']:.2f}s")
    print(f"  ORB matching          : {orb_timings['orb_matching']:.2f}s")
    print(f"  Depth/correspondences : {orb_timings['depth_correspondences']:.2f}s\n")

    print("SuperPoint + LightGlue")
    print("Timing summary")
    print(f"  Image loading         : {lightglue_timings['image_loading']:.2f}s")
    print(f"  SuperPoint extraction : {lightglue_timings['superpoint_extraction']:.2f}s")
    print(f"  LightGlue matching    : {lightglue_timings['lightglue_matching']:.2f}s")
    print(f"  Depth/correspondences : {lightglue_timings['depth_correspondences']:.2f}s\n")
    print(f"Total wall-clock time : {timings['total_wall_clock']:.2f}s\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "output_dir",
        nargs="?",
        type=Path,
        default=None,
        help="Comparison output directory. Defaults to the latest outputs/comparison_* directory.",
    )
    return parser.parse_args()


args = parse_args()
output_dir = args.output_dir or get_latest_comparison_output_dir()
pairs_path = output_dir / "pairs.jsonl"
metadata_path = output_dir / "metadata.json"
summary_path = output_dir / "summary.json"
metadata = load_json(metadata_path)
records = load_jsonl(pairs_path)
method_results = build_method_results(records)
write_summary(summary_path, statistics_to_json(method_results))
rotation_figure_paths = plot_rotation_error_figures(method_results, output_dir)
pnp_failures_figure_path = plot_pnp_failures_vs_gap(method_results, output_dir)

print(f"Analyzing output directory : {output_dir}")
print(f"Pair records : {len(records)}")
print(f"Summary file : {summary_path}\n")
for figure_path in rotation_figure_paths:
    print(f"Rotation error figure : {figure_path}")
print(f"PnP failures figure : {pnp_failures_figure_path}")
print()
print_comparison_table(method_results, title="Comparison")
print_timing_summary(metadata)
