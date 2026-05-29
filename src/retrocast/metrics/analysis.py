from __future__ import annotations

from collections.abc import Sequence

from retrocast.metrics.bootstrap import summarize_values
from retrocast.models.analysis import MetricSummary
from retrocast.models.evaluation import TargetResult, Tier


def summarize_targets(
    targets: Sequence[TargetResult],
    *,
    tiers: Sequence[Tier],
    ks: Sequence[int],
    metric_label: str = "task",
    n_boot: int = 10000,
    seed: int = 42,
) -> dict[str, MetricSummary]:
    metrics: dict[str, MetricSummary] = {}
    for tier in tiers:
        metric_suffix = str(int(tier))
        metrics[f"solv_{metric_suffix}[{metric_label}]_rate"] = _solv_rate(targets, tier, n_boot=n_boot, seed=seed)
        metrics[f"mrr_solv_{metric_suffix}[{metric_label}]"] = _mrr_solv(targets, tier, n_boot=n_boot, seed=seed)

    reconstruction_targets = [target for target in targets if target.target.acceptable_routes]
    if reconstruction_targets:
        for k in sorted(set(ks)):
            metrics[f"acceptable_reconstruction_top_{k}[{metric_label}]"] = _top_k_reconstruction(
                reconstruction_targets,
                k,
                n_boot=n_boot,
                seed=seed,
            )
    return metrics


def _solv_rate(targets: Sequence[TargetResult], tier: Tier, *, n_boot: int, seed: int) -> MetricSummary:
    values = []
    for target in targets:
        values.append(1.0 if any(candidate.satisfies_solv(tier) for candidate in target.candidates) else 0.0)
    return summarize_values(values, n_boot=n_boot, seed=seed)


def _mrr_solv(targets: Sequence[TargetResult], tier: Tier, *, n_boot: int, seed: int) -> MetricSummary:
    values = []
    for target in targets:
        candidates = sorted(target.candidates, key=lambda candidate: candidate.rank)
        reciprocal_rank = 0.0
        for candidate in candidates:
            if candidate.satisfies_solv(tier):
                reciprocal_rank = 1.0 / candidate.rank
                break
        values.append(reciprocal_rank)
    return summarize_values(values, n_boot=n_boot, seed=seed)


def _top_k_reconstruction(targets: Sequence[TargetResult], k: int, *, n_boot: int, seed: int) -> MetricSummary:
    values = []
    for target in targets:
        ranked = sorted(target.candidates, key=lambda candidate: candidate.rank)
        candidates = [candidate for candidate in ranked if candidate.satisfies_task()]
        values.append(1.0 if any(candidate.matches_acceptable for candidate in candidates[:k]) else 0.0)
    return summarize_values(values, n_boot=n_boot, seed=seed)
