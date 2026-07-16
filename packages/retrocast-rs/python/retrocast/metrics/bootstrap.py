import json
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

from retrocast.models.analysis import MetricSummary, ReliabilityFlag
from retrocast.models.evaluation import TargetResult

T = TypeVar("T")


class BootstrapDistribution(list[float]):
    @property
    def shape(self) -> tuple[int]:
        return (len(self),)


@dataclass(frozen=True)
class StratifiedMetricSummary:
    metric_name: str
    overall: MetricSummary
    by_stratum: dict[str, MetricSummary]


def check_reliability(n: int, p: float) -> ReliabilityFlag:
    from retrocast import _native

    return ReliabilityFlag.model_validate_json(_native.reliability_flag_json(n, p))


def summarize_values(
    values: Sequence[float],
    *,
    n_boot: int = 10000,
    seed: int = 42,
    alpha: float = 0.05,
    reliability: bool = True,
) -> MetricSummary:
    from retrocast import _native

    return MetricSummary.model_validate_json(
        _native.summarize_values_json(
            json.dumps(list(values), separators=(",", ":")),
            n_boot,
            seed,
            alpha=alpha,
            reliability=reliability,
        )
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
    from retrocast import _native

    return _native.target_metric_json(target.model_dump_json(exclude_none=True), "is_solvable")


def make_get_top_k(k: int) -> Callable[[TargetResult], float]:
    def get_top_k(target: TargetResult) -> float:
        from retrocast import _native

        return _native.target_metric_json(target.model_dump_json(exclude_none=True), "top_k", k=k)

    return get_top_k


def get_bootstrap_distribution(
    targets: Sequence[T],
    extractor: Callable[[T], float],
    *,
    n_boot: int = 10000,
    seed: int = 42,
) -> BootstrapDistribution:
    from retrocast import _native

    values = [extractor(target) for target in targets]
    raw = _native.bootstrap_distribution_json(
        json.dumps(values, separators=(",", ":")),
        n_boot,
        seed,
    )
    return BootstrapDistribution(json.loads(raw))
