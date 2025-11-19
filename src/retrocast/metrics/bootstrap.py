from collections import defaultdict
from collections.abc import Callable
from typing import Any, TypeVar

import numpy as np

from retrocast.models.evaluation import TargetEvaluation
from retrocast.models.stats import MetricResult, ReliabilityFlag, StratifiedMetric

T = TypeVar("T")


def check_reliability(n: int, p: float) -> ReliabilityFlag:
    """
    Checks rules of thumb for statistical reliability.

    Rules:
    1. N >= 30: Central Limit Theorem kicks in.
    2. np > 5 and n(1-p) > 5: Valid for binary proportions (avoiding boundary effects).
    """
    if n < 30:
        return ReliabilityFlag(code="LOW_N", message=f"Small sample size (N={n} < 30). CIs may be unstable.")

    # Check for extreme probabilities (too close to 0 or 1 for the sample size)
    # If p=0 or p=1, the bootstrap collapses to a single point, which is technically
    # accurate for the sample but terrible for inference.
    successes = n * p
    failures = n * (1 - p)

    if successes < 5 or failures < 5:
        return ReliabilityFlag(
            code="EXTREME_P", message=f"Extreme value (p={p:.1%}) for N={n}. Boundary effects likely."
        )

    return ReliabilityFlag(code="OK", message="Reliable.")


def _bootstrap_1d(data: np.ndarray, n_boot: int, alpha: float, seed: int) -> MetricResult:
    """Internal numpy-optimized bootstrap for 1D array."""
    n = len(data)
    if n == 0:
        return MetricResult(
            value=0.0,
            ci_lower=0.0,
            ci_upper=0.0,
            n_samples=0,
            reliability=ReliabilityFlag(code="LOW_N", message="No data."),
        )

    rng = np.random.default_rng(seed)

    # Resample indices: (n_boot, n)
    indices = rng.integers(0, n, (n_boot, n))

    # Compute means for all samples at once
    # data[indices] creates a (n_boot, n) array of values
    resampled_means = np.mean(data[indices], axis=1)

    # Calculate point estimate first
    value = float(np.mean(data))
    reliability = check_reliability(n, value)

    return MetricResult(
        value=float(np.mean(data)),
        ci_lower=float(np.percentile(resampled_means, 100 * alpha / 2)),
        ci_upper=float(np.percentile(resampled_means, 100 * (1 - alpha / 2))),
        n_samples=n,
        reliability=reliability,
    )


def compute_metric_with_ci(
    targets: list[TargetEvaluation],
    extractor: Callable[[TargetEvaluation], float],
    metric_name: str,
    group_by: Callable[[TargetEvaluation], Any] | None = None,
    n_boot: int = 10000,
    seed: int = 42,
) -> StratifiedMetric:
    """
    Computes a metric with CIs, optionally stratified.
    """
    # 1. Overall
    values_overall = np.array([extractor(t) for t in targets])
    overall_res = _bootstrap_1d(values_overall, n_boot, 0.05, seed)

    # 2. Stratified
    by_group = {}
    if group_by:
        grouped = defaultdict(list)
        for t in targets:
            key = group_by(t)
            val = extractor(t)
            grouped[key].append(val)

        for key, vals in grouped.items():
            # Use a deterministic seed variant for each group to stabilize small-N noise
            # (seed + hash of key)
            group_seed = seed + abs(hash(key)) % 10000
            by_group[key] = _bootstrap_1d(np.array(vals), n_boot, 0.05, group_seed)

    return StratifiedMetric(metric_name=metric_name, overall=overall_res, by_group=by_group)


# --- Extractor Helpers ---


def get_is_solvable(t: TargetEvaluation) -> float:
    return 1.0 if t.is_solvable else 0.0


def make_get_top_k(k: int) -> Callable[[TargetEvaluation], float]:
    def _get_top_k(t: TargetEvaluation) -> float:
        return 1.0 if (t.gt_rank is not None and t.gt_rank <= k) else 0.0

    return _get_top_k
