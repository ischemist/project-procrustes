from collections import defaultdict
from collections.abc import Callable
from typing import Any, TypeVar

import numpy as np

from retrocast.models.evaluation import TargetEvaluation
from retrocast.models.stats import MetricResult, ModelComparison, ReliabilityFlag, StratifiedMetric

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
            # Skip None keys (e.g., unsolvable targets with no stratification length)
            if key is None:
                continue
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
        return 1.0 if (t.acceptable_rank is not None and t.acceptable_rank <= k) else 0.0

    return _get_top_k


def compute_paired_difference(
    targets_a: list[TargetEvaluation],
    targets_b: list[TargetEvaluation],
    metric_extractor: Callable[[TargetEvaluation], float],
    model_a_name: str,
    model_b_name: str,
    metric_name: str,
    n_boot: int = 10000,
    seed: int = 42,
) -> ModelComparison:
    """
    Computes the paired difference (B - A) with bootstrap CI.
    Assumes targets_a and targets_b are aligned (same target IDs in same order).
    """
    # 1. Align Data
    # We must ensure we are comparing Target X to Target X.
    # Convert to dict for safety, then align.
    dict_a = {t.target_id: t for t in targets_a}
    dict_b = {t.target_id: t for t in targets_b}

    # Intersection keys (should be all of them, but let's be safe)
    common_ids = sorted(list(set(dict_a.keys()) & set(dict_b.keys())))

    if len(common_ids) == 0:
        raise ValueError("No common targets found between models.")

    # Extract metric vectors
    vec_a = np.array([metric_extractor(dict_a[tid]) for tid in common_ids])
    vec_b = np.array([metric_extractor(dict_b[tid]) for tid in common_ids])

    # 2. Bootstrap the Difference
    n = len(vec_a)
    rng = np.random.default_rng(seed)

    # Calculate observed difference
    diff_obs = float(np.mean(vec_b) - np.mean(vec_a))

    # Resample indices (n_boot, n)
    indices = rng.integers(0, n, (n_boot, n))

    # Compute means for A and B using the SAME indices (This is the "Paired" part)
    means_a = np.mean(vec_a[indices], axis=1)
    means_b = np.mean(vec_b[indices], axis=1)

    # Distribution of differences
    diffs = means_b - means_a

    # 3. CI & Significance
    ci_lower = float(np.percentile(diffs, 2.5))
    ci_upper = float(np.percentile(diffs, 97.5))

    # Significant if 0 is not in the interval
    is_significant = not (ci_lower <= 0 <= ci_upper)

    return ModelComparison(
        metric=metric_name,
        model_a=model_a_name,
        model_b=model_b_name,
        diff_mean=diff_obs,
        diff_ci_lower=ci_lower,
        diff_ci_upper=ci_upper,
        is_significant=is_significant,
    )


def get_bootstrap_distribution(
    targets: list[TargetEvaluation], extractor: Callable[[TargetEvaluation], float], n_boot: int = 10000, seed: int = 42
) -> np.ndarray:
    """
    Returns the raw array of bootstrap means. Shape: (n_boot,)
    Useful for probabilistic ranking and advanced hypothesis testing.
    """
    values = np.array([extractor(t) for t in targets])
    n = len(values)
    if n == 0:
        return np.zeros(n_boot)

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, (n_boot, n))

    # Calculate means for all 10k samples
    return np.mean(values[indices], axis=1)
