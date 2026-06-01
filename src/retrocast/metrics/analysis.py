from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol

from retrocast.metrics.bootstrap import summarize_values
from retrocast.models.analysis import MetricSummary
from retrocast.models.evaluation import ScoredCandidate, TargetResult, Tier
from retrocast.models.route import InChIKeyLevel, Route, RoutePath

CandidatePredicate = Callable[[ScoredCandidate, Tier], bool]


class CandidateMetric(Protocol):
    def __call__(
        self,
        targets: Sequence[TargetResult],
        tier: Tier,
        predicate: CandidatePredicate,
        *,
        n_boot: int,
        seed: int,
    ) -> MetricSummary: ...


def summarize_targets(
    targets: Sequence[TargetResult],
    *,
    tiers: Sequence[Tier],
    ks: Sequence[int],
    prefix_depths: Sequence[int] = (1, 2, 3),
    metric_label: str = "task",
    acceptable_match_level: InChIKeyLevel = InChIKeyLevel.FULL,
    n_boot: int = 10000,
    seed: int = 42,
) -> dict[str, MetricSummary]:
    metrics: dict[str, MetricSummary] = {}
    for tier in tiers:
        metric_suffix = str(int(tier))
        metric_specs: tuple[tuple[str, CandidateMetric, CandidatePredicate], ...] = (
            (f"tier_{metric_suffix}_validity_rate", _candidate_rate, ScoredCandidate.satisfies_validity),
            (f"mrr_tier_{metric_suffix}", _candidate_mrr, ScoredCandidate.satisfies_validity),
            (f"solv_{metric_suffix}[{metric_label}]_rate", _candidate_rate, ScoredCandidate.satisfies_solv),
            (f"mrr_solv_{metric_suffix}[{metric_label}]", _candidate_mrr, ScoredCandidate.satisfies_solv),
        )
        for name, summarize, predicate in metric_specs:
            metrics[name] = summarize(targets, tier, predicate, n_boot=n_boot, seed=seed)

    reconstruction_targets = [target for target in targets if target.target.acceptable_routes]
    if reconstruction_targets:

        def reconstruction_summary(values: Sequence[float], *, reliability: bool = True) -> MetricSummary:
            return summarize_values(values, n_boot=n_boot, seed=seed, reliability=reliability)

        for k in sorted(set(ks)):
            metrics[f"acceptable_reconstruction_top_{k}[{metric_label}]"] = reconstruction_summary(
                _top_k_reconstruction_values(reconstruction_targets, k)
            )
            metrics[f"acceptable_root_reconstruction_top_{k}[{metric_label}]"] = reconstruction_summary(
                _top_k_root_reconstruction_values(reconstruction_targets, k, acceptable_match_level)
            )
            metrics[f"acceptable_reconstruction_given_root_top_{k}[{metric_label}]"] = reconstruction_summary(
                _top_k_reconstruction_given_root_values(reconstruction_targets, k, acceptable_match_level)
            )
            metrics[f"distinct_root_reactions_top_{k}[{metric_label}]"] = reconstruction_summary(
                _distinct_root_reaction_values(reconstruction_targets, k, acceptable_match_level),
                reliability=False,
            )
            for depth in sorted(set(prefix_depths)):
                metrics[f"acceptable_prefix_reconstruction_depth_{depth}_top_{k}[{metric_label}]"] = (
                    reconstruction_summary(
                        _top_k_prefix_reconstruction_values(reconstruction_targets, k, depth, acceptable_match_level)
                    )
                )
    return metrics


def _candidate_rate(
    targets: Sequence[TargetResult],
    tier: Tier,
    predicate: CandidatePredicate,
    *,
    n_boot: int,
    seed: int,
) -> MetricSummary:
    values = []
    for target in targets:
        values.append(1.0 if any(predicate(candidate, tier) for candidate in target.candidates) else 0.0)
    return summarize_values(values, n_boot=n_boot, seed=seed)


def _candidate_mrr(
    targets: Sequence[TargetResult],
    tier: Tier,
    predicate: CandidatePredicate,
    *,
    n_boot: int,
    seed: int,
) -> MetricSummary:
    values = []
    for target in targets:
        candidates = sorted(target.candidates, key=lambda candidate: candidate.rank)
        reciprocal_rank = 0.0
        for candidate in candidates:
            if predicate(candidate, tier):
                reciprocal_rank = 1.0 / candidate.rank
                break
        values.append(reciprocal_rank)
    return summarize_values(values, n_boot=n_boot, seed=seed)


def _top_k_reconstruction_values(targets: Sequence[TargetResult], k: int) -> list[float]:
    values = []
    for target in targets:
        candidates = _task_satisfying_top_k(target, k)
        values.append(1.0 if any(candidate.matches_acceptable for candidate in candidates) else 0.0)
    return values


def _top_k_root_reconstruction_values(
    targets: Sequence[TargetResult],
    k: int,
    match_level: InChIKeyLevel,
) -> list[float]:
    return [1.0 if _target_has_root_hit(target, k, match_level) else 0.0 for target in targets]


def _top_k_reconstruction_given_root_values(
    targets: Sequence[TargetResult],
    k: int,
    match_level: InChIKeyLevel,
) -> list[float]:
    values = []
    for target in targets:
        if not _target_has_root_hit(target, k, match_level):
            continue
        candidates = _task_satisfying_top_k(target, k)
        values.append(1.0 if any(candidate.matches_acceptable for candidate in candidates) else 0.0)
    return values


def _top_k_prefix_reconstruction_values(
    targets: Sequence[TargetResult],
    k: int,
    depth: int,
    match_level: InChIKeyLevel,
) -> list[float]:
    return [1.0 if _target_has_prefix_hit(target, k, depth, match_level) else 0.0 for target in targets]


def _distinct_root_reaction_values(
    targets: Sequence[TargetResult],
    k: int,
    match_level: InChIKeyLevel,
) -> list[float]:
    values = []
    for target in targets:
        root_signatures = {
            signature
            for candidate in _task_satisfying_top_k(target, k)
            if (signature := _root_reaction_signature(candidate.route, match_level)) is not None
        }
        values.append(float(len(root_signatures)))
    return values


def _target_has_root_hit(target: TargetResult, k: int, match_level: InChIKeyLevel) -> bool:
    acceptable_roots = {
        signature
        for route in target.target.acceptable_routes
        if (signature := _root_reaction_signature(route, match_level)) is not None
    }
    if not acceptable_roots:
        return False

    candidate_roots = {
        signature
        for candidate in _task_satisfying_top_k(target, k)
        if (signature := _root_reaction_signature(candidate.route, match_level)) is not None
    }
    return bool(candidate_roots & acceptable_roots)


def _target_has_prefix_hit(target: TargetResult, k: int, depth: int, match_level: InChIKeyLevel) -> bool:
    acceptable_prefixes = {route.signature(match_level, depth=depth) for route in target.target.acceptable_routes}
    candidate_prefixes = {
        candidate.route.signature(match_level, depth=depth)
        for candidate in _task_satisfying_top_k(target, k)
        if candidate.route is not None
    }
    return bool(candidate_prefixes & acceptable_prefixes)


def _task_satisfying_top_k(target: TargetResult, k: int) -> list[ScoredCandidate]:
    ranked = sorted(target.candidates, key=lambda candidate: candidate.rank)
    return [candidate for candidate in ranked if candidate.satisfies_task()][:k]


def _root_reaction_signature(route: Route | None, match_level: InChIKeyLevel) -> str | None:
    if route is None:
        return None
    try:
        return route.reaction_at(RoutePath.root_reaction()).signature(match_level)
    except KeyError:
        return None
