from __future__ import annotations

from collections.abc import Callable, Sequence

from retrocast.v2.metrics.analysis import summarize_targets
from retrocast.v2.models.analysis import AnalysisReport, MetricSummary
from retrocast.v2.models.evaluation import Evaluation, TargetResult


def analyze(
    evaluation: Evaluation,
    *,
    ks: Sequence[int] = (1, 5, 10, 50),
    stratify_by: Callable[[TargetResult], str | None] | None = None,
    n_boot: int = 10000,
    seed: int = 42,
) -> AnalysisReport:
    targets = list(evaluation.targets.values())
    metrics = summarize_targets(
        targets,
        tiers=evaluation.tiers,
        ks=ks,
        metric_label=evaluation.metric_label,
        n_boot=n_boot,
        seed=seed,
    )

    by_stratum: dict[str, dict[str, MetricSummary]] = {}
    if stratify_by is not None or any(_default_route_depth_stratum(target) is not None for target in targets):
        strata: dict[str, list[TargetResult]] = {}
        for target in targets:
            stratum = _default_route_depth_stratum(target) if stratify_by is None else stratify_by(target)
            if stratum is None:
                continue
            strata.setdefault(stratum, []).append(target)
        by_stratum = {
            stratum: summarize_targets(
                stratum_targets,
                tiers=evaluation.tiers,
                ks=ks,
                metric_label=evaluation.metric_label,
                n_boot=n_boot,
                seed=seed,
            )
            for stratum, stratum_targets in strata.items()
        }

    return AnalysisReport(metrics=metrics, by_stratum=by_stratum)


def _default_route_depth_stratum(target: TargetResult) -> str | None:
    if target.target.acceptable_routes:
        return f"depth {target.target.acceptable_routes[0].depth()}"
    route_depth = target.effective_constraints.route_depth
    if route_depth is None:
        return None
    return f"depth {route_depth}"
