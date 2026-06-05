from __future__ import annotations

from collections.abc import Callable, Sequence

from retrocast.metrics.analysis import summarize_targets
from retrocast.models.analysis import AnalysisReport, RuntimeSummary
from retrocast.models.evaluation import Evaluation, TargetResult
from retrocast.models.task import RouteDepthConstraint


def analyze(
    evaluation: Evaluation,
    *,
    ks: Sequence[int] = (1, 5, 10, 50),
    prefix_depths: Sequence[int] = (1, 2, 3),
    stratify_by: Callable[[TargetResult], str | None] | None = None,
    n_boot: int = 10000,
    seed: int = 42,
) -> AnalysisReport:
    targets = list(evaluation.targets.values())
    metrics = summarize_targets(
        targets,
        tiers=evaluation.tiers,
        ks=ks,
        prefix_depths=prefix_depths,
        metric_label=evaluation.metric_label,
        acceptable_match_level=evaluation.acceptable_match_level,
        n_boot=n_boot,
        seed=seed,
    )

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
            prefix_depths=prefix_depths,
            metric_label=evaluation.metric_label,
            acceptable_match_level=evaluation.acceptable_match_level,
            n_boot=n_boot,
            seed=seed,
        )
        for stratum, stratum_targets in strata.items()
    }

    return AnalysisReport(
        metrics=metrics,
        by_stratum=by_stratum,
        bootstrap_resamples=n_boot,
        runtime=_runtime_summary(targets),
    )


def _runtime_summary(targets: list[TargetResult]) -> RuntimeSummary:
    wall_times = [target.wall_time for target in targets if target.wall_time is not None]
    cpu_times = [target.cpu_time for target in targets if target.cpu_time is not None]
    total_wall_time = sum(wall_times) if wall_times else None
    total_cpu_time = sum(cpu_times) if cpu_times else None
    return RuntimeSummary(
        total_wall_time=total_wall_time,
        mean_wall_time=total_wall_time / len(wall_times) if total_wall_time is not None else None,
        total_cpu_time=total_cpu_time,
        mean_cpu_time=total_cpu_time / len(cpu_times) if total_cpu_time is not None else None,
        timed_target_count=max(len(wall_times), len(cpu_times)),
    )


def _default_route_depth_stratum(target: TargetResult) -> str | None:
    if target.target.acceptable_routes:
        return f"depth {target.target.acceptable_routes[0].depth()}"
    for constraint in target.effective_constraints:
        if isinstance(constraint, RouteDepthConstraint):
            return f"depth {constraint.max_depth}"
    return None
