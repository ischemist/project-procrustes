from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

import numpy as np
import numpy.typing as npt

from retrocast.models.analysis import MetricSummary, ReliabilityFlag
from retrocast.models.evaluation import TargetResult, Tier

T = TypeVar("T")


@dataclass(frozen=True)
class StratifiedMetricSummary:
    metric_name: str
    overall: MetricSummary
    by_stratum: dict[str, MetricSummary]


def check_reliability(n: int, p: float) -> ReliabilityFlag:
    if n < 30:
        return ReliabilityFlag(code="LOW_N", message=f"Small sample size (N={n} < 30). CIs may be unstable.")

    successes = n * p
    failures = n * (1 - p)
    if successes < 5 or failures < 5:
        return ReliabilityFlag(
            code="EXTREME_P",
            message=f"Extreme value (p={p:.1%}) for N={n}. Boundary effects likely.",
        )

    return ReliabilityFlag(code="OK", message="Reliable.")


def summarize_values(
    values: Sequence[float],
    *,
    n_boot: int = 10000,
    seed: int = 42,
    alpha: float = 0.05,
) -> MetricSummary:
    data: npt.NDArray[np.float64] = np.array(values, dtype=np.float64)
    if len(data) == 0:
        return MetricSummary(
            value=0.0,
            count=0,
            reliability=ReliabilityFlag(code="LOW_N", message="No data."),
        )

    distribution = _bootstrap_mean(data, n_boot=n_boot, seed=seed)
    value = float(np.mean(data))
    return MetricSummary(
        value=value,
        count=len(data),
        ci_low=float(np.percentile(distribution, 100 * alpha / 2)),
        ci_high=float(np.percentile(distribution, 100 * (1 - alpha / 2))),
        reliability=check_reliability(len(data), value),
    )


def compute_metric_with_ci(
    targets: Sequence[T],
    extractor: Callable[[T], float],
    metric_name: str,
    *,
    stratify_by: Callable[[T], Any] | None = None,
    n_boot: int = 10000,
    seed: int = 42,
) -> StratifiedMetricSummary:
    overall = summarize_values([extractor(target) for target in targets], n_boot=n_boot, seed=seed)
    by_stratum = {}
    if stratify_by is not None:
        strata: dict[str, list[float]] = defaultdict(list)
        for target in targets:
            stratum = stratify_by(target)
            if stratum is not None:
                strata[str(stratum)].append(extractor(target))
        for index, (stratum, values) in enumerate(strata.items()):
            by_stratum[stratum] = summarize_values(values, n_boot=n_boot, seed=seed + index + 1)
    return StratifiedMetricSummary(metric_name=metric_name, overall=overall, by_stratum=by_stratum)


def get_is_solvable(target: TargetResult) -> float:
    return 1.0 if any(candidate.satisfies_solv(Tier.ZERO) for candidate in target.candidates) else 0.0


def make_get_top_k(k: int) -> Callable[[TargetResult], float]:
    def get_top_k(target: TargetResult) -> float:
        ranked = sorted(target.candidates, key=lambda candidate: candidate.rank)
        candidates = [candidate for candidate in ranked if candidate.satisfies_task()]
        return 1.0 if any(candidate.matches_acceptable for candidate in candidates[:k]) else 0.0

    return get_top_k


def get_bootstrap_distribution(
    targets: Sequence[T],
    extractor: Callable[[T], float],
    *,
    n_boot: int = 10000,
    seed: int = 42,
) -> npt.NDArray[np.float64]:
    values: npt.NDArray[np.float64] = np.array([extractor(target) for target in targets], dtype=np.float64)
    if len(values) == 0:
        return np.zeros(n_boot, dtype=np.float64)

    return _bootstrap_mean(values, n_boot=n_boot, seed=seed)


def _bootstrap_mean(values: npt.NDArray[np.float64], *, n_boot: int, seed: int) -> npt.NDArray[np.float64]:
    if len(values) == 0:
        return np.zeros(n_boot, dtype=np.float64)

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), (n_boot, len(values)))
    return np.mean(values[indices], axis=1)
