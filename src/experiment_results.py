class FrameGapStatistics:
    def __init__(self, frame_gap, pair_count):
        self.gap = frame_gap
        self.pairs = pair_count
        self.matching_failed = None
        self.mean_keypoints_i = None
        self.mean_keypoints_j = None
        self.mean_matches = None
        self.mean_correspondences = None
        self.mean_gt_translation = None
        self.mean_gt_rotation = None
        self.pnp_success = None
        self.pnp_failed = None
        self.mean_pnp_inliers = None
        self.mean_pnp_inlier_ratio = None
        self.mean_translation_error = None
        self.median_translation_error = None
        self.p95_translation_error = None
        self.max_translation_error = None
        self.mean_rotation_error = None
        self.median_rotation_error = None
        self.p95_rotation_error = None
        self.max_rotation_error = None

def format_float(value, decimals):
    if value is None:
        return "NA"
    return f"{value:.{decimals}f}"

def print_experiment_summary(synchronized_frames, results):
    print(f"Synchronized frames : {len(synchronized_frames)}")
    print_frame_gap_statistics_table(results)

def print_frame_gap_statistics_table(results):
    print(
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
    for result in results:
        print(
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
            f"{format_float(result.p95_rotation_error, 2):>6}"
        )
