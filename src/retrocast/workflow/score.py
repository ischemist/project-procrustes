import logging

from retrocast.chem import InchiKeyLevel, reduce_inchikey
from retrocast.io.data import RoutesDict
from retrocast.metrics.similarity import find_acceptable_match
from retrocast.metrics.solvability import (
    get_reaction_tier_failure_codes,
    get_route_tier_failure_codes,
    is_route_solved,
)
from retrocast.models.benchmark import BenchmarkSet, ExecutionStats
from retrocast.models.candidates import CandidateRecord, CandidateRecordsDict
from retrocast.models.chem import Route
from retrocast.models.evaluation import EvaluationResults, ScoredCandidate, TargetEvaluation, tier_rank_key
from retrocast.models.validity import (
    IMPLEMENTED_VALIDITY_TIERS,
    CheckResult,
    ConstraintResult,
    MetricScope,
    ReactionValidity,
    RouteValidity,
    ScopeId,
    StockTerminationConstraint,
    TierResult,
    ValidityTier,
)
from retrocast.typing import InchiKeyStr

logger = logging.getLogger(__name__)


def _make_tier_result(tier: ValidityTier, failure_codes: list[str]) -> TierResult:
    if not failure_codes:
        return TierResult(tier=tier, status="pass")
    return TierResult(
        tier=tier,
        status="fail",
        checks=[CheckResult(code=code, status="fail") for code in failure_codes],
    )


def _get_missing_stock_leaves(
    route: Route,
    stock: set[InchiKeyStr],
    match_level: InchiKeyLevel,
) -> list[InchiKeyStr]:
    missing = []
    for leaf in route.leaves:
        key = leaf.inchikey if match_level == InchiKeyLevel.FULL else reduce_inchikey(leaf.inchikey, match_level)
        if key not in stock:
            missing.append(leaf.inchikey)
    return sorted(set(missing))


def _make_stock_constraint_result(missing_leaves: list[InchiKeyStr]) -> ConstraintResult:
    if not missing_leaves:
        return ConstraintResult(status="pass")
    return ConstraintResult(
        status="fail",
        checks=[
            CheckResult(
                code="constraint.stock_termination.missing_leaf",
                status="fail",
                details={"missing_leaf_inchikeys": missing_leaves},
            )
        ],
    )


def _score_route_candidate(
    candidate: CandidateRecord,
    *,
    acceptable_sigs: list[str],
    stock_inchikeys: set[InchiKeyStr],
    match_level: InchiKeyLevel,
) -> tuple[ScoredCandidate, dict[int, bool], dict[str, bool], int | None]:
    if candidate.route is None:
        failure = candidate.adapter_failure
        check = CheckResult(
            code=failure.code if failure else "adapter.unknown_failure",
            status="fail",
            message=failure.message if failure else None,
            details=failure.context if failure else {},
        )
        return (
            ScoredCandidate(
                rank=candidate.rank,
                adapter_failure=failure,
                validity=RouteValidity(tiers={0: TierResult(tier=0, status="fail", checks=[check])}),
                constraint_results={"stock": ConstraintResult(status="not_evaluated")},
            ),
            {tier: False for tier in IMPLEMENTED_VALIDITY_TIERS},
            {"stock": False},
            None,
        )

    route = candidate.route
    stock_terminated = is_route_solved(route, stock_inchikeys, match_level=match_level)
    route_tiers = {
        tier: _make_tier_result(tier, get_route_tier_failure_codes(route, tier))
        for tier in sorted(IMPLEMENTED_VALIDITY_TIERS)
    }
    tier_passes = {tier: result.status == "pass" for tier, result in route_tiers.items()}
    reaction_validity: list[ReactionValidity] = []
    for reaction in route.iter_reactions():
        reaction_tiers = {
            tier: _make_tier_result(tier, get_reaction_tier_failure_codes(reaction, tier))
            for tier in sorted(IMPLEMENTED_VALIDITY_TIERS)
        }
        reaction_validity.append(
            ReactionValidity(
                reaction_id=reaction.reaction_id,
                tiers=reaction_tiers,
            )
        )
    route_validity = RouteValidity(tiers=route_tiers, reactions=reaction_validity)
    constraint_results = {
        "stock": _make_stock_constraint_result(_get_missing_stock_leaves(route, stock_inchikeys, match_level))
    }
    matched_idx = find_acceptable_match(route, acceptable_sigs, match_level=match_level)
    return (
        ScoredCandidate(
            rank=candidate.rank,
            route=route,
            validity=route_validity,
            constraint_results=constraint_results,
            matches_acceptable=matched_idx is not None,
            matched_acceptable_index=matched_idx,
        ),
        tier_passes,
        {"stock": stock_terminated},
        matched_idx,
    )


def _records_from_routes(predictions: RoutesDict) -> CandidateRecordsDict:
    return {
        target_id: [CandidateRecord(rank=rank, route=route) for rank, route in enumerate(routes, start=1)]
        for target_id, routes in predictions.items()
    }


def score_routes(
    benchmark: BenchmarkSet,
    predictions: RoutesDict,
    stock: set[InchiKeyStr],
    stock_name: str,
    model_name: str,
    execution_stats: ExecutionStats | None = None,
    match_level: InchiKeyLevel = InchiKeyLevel.FULL,
) -> EvaluationResults:
    """
    Score benchmark-keyed route predictions.

    Route-only artifacts contain successful adapted routes, so they are first
    lifted into candidate records with contiguous ranks.
    """
    return score_candidate_records(
        benchmark=benchmark,
        candidates=_records_from_routes(predictions),
        stock=stock,
        stock_name=stock_name,
        model_name=model_name,
        execution_stats=execution_stats,
        match_level=match_level,
        denominator_type="route_only",
    )


def score_candidate_records(
    benchmark: BenchmarkSet,
    candidates: CandidateRecordsDict,
    stock: set[InchiKeyStr],
    stock_name: str,
    model_name: str,
    execution_stats: ExecutionStats | None = None,
    match_level: InchiKeyLevel = InchiKeyLevel.FULL,
    denominator_type: str = "complete",
) -> EvaluationResults:
    """
    Score candidate records against a benchmark and stock.

    Candidate records preserve the planner's original target-local ranks,
    including slots that failed adaptation. `denominator_type` is written to
    metadata so downstream analysis can distinguish complete candidate
    denominators from route-only artifacts.
    """
    logger.info(f"Scoring {model_name} on {benchmark.name}...")

    if execution_stats:
        logger.info(f"Runtime stats available for {len(execution_stats.wall_time)} targets")

    # Check if benchmark has any acceptable routes
    has_acceptable_routes = any(len(target.acceptable_routes) > 0 for target in benchmark.targets.values())

    stock_scope = MetricScope(
        id="stock",
        constraints=[
            StockTerminationConstraint(
                stock_name=stock_name,
                match_level=match_level.value,
            )
        ],
    )

    eval_results = EvaluationResults(
        model_name=model_name,
        benchmark_name=benchmark.name,
        stock_name=stock_name,
        has_acceptable_routes=has_acceptable_routes,
        metric_scopes=[stock_scope],
    )

    # Pre-normalize stock if using a non-default match level
    if match_level != InchiKeyLevel.FULL:
        stock_inchikeys = {reduce_inchikey(k, match_level) for k in stock}
    else:
        stock_inchikeys = stock

    # Iterate Targets (The Denominator)
    n_candidates = 0
    for target_id, target in benchmark.targets.items():
        target_candidates = candidates.get(target_id, [])
        n_candidates += len(target_candidates)

        # Pre-compute acceptable route signatures
        acceptable_sigs = [
            route.get_structural_signature(match_level=match_level) for route in target.acceptable_routes
        ]

        scored_candidates = []
        first_reconstruction_rank = None
        first_valid_ranks: dict[str, int | None] = {
            tier_rank_key(tier): None for tier in sorted(IMPLEMENTED_VALIDITY_TIERS)
        }
        first_solv_ranks: dict[ScopeId, dict[str, int | None]] = {
            "stock": {tier_rank_key(tier): None for tier in sorted(IMPLEMENTED_VALIDITY_TIERS)}
        }
        # Counter for the "Effective Rank" (only increments on solvable routes)
        effective_rank_counter = 1

        for candidate in target_candidates:
            scored_candidate, tier_passes, scope_passes, matched_idx = _score_route_candidate(
                candidate,
                acceptable_sigs=acceptable_sigs,
                stock_inchikeys=stock_inchikeys,
                match_level=match_level,
            )
            for tier, passed in tier_passes.items():
                tier_key = tier_rank_key(tier)
                if passed and first_valid_ranks[tier_key] is None:
                    first_valid_ranks[tier_key] = candidate.rank
                if passed and scope_passes["stock"] and first_solv_ranks["stock"][tier_key] is None:
                    first_solv_ranks["stock"][tier_key] = candidate.rank

            if scope_passes["stock"]:
                if matched_idx is not None and first_reconstruction_rank is None:
                    first_reconstruction_rank = effective_rank_counter
                effective_rank_counter += 1

            scored_candidates.append(scored_candidate)

        # Summary for this target
        has_stock_terminated_route = any(
            candidate.constraint_results["stock"].status == "pass" for candidate in scored_candidates
        )
        # Always stratify by primary acceptable route (benchmark ground truth)
        source_route = target.primary_route

        # Extract runtime metrics if available
        wall_time = execution_stats.wall_time.get(target_id) if execution_stats else None
        cpu_time = execution_stats.cpu_time.get(target_id) if execution_stats else None

        t_eval = TargetEvaluation(
            target_id=target_id,
            candidates=scored_candidates,
            has_stock_terminated_route=has_stock_terminated_route,
            first_valid_ranks=first_valid_ranks,
            first_solv_ranks=first_solv_ranks,
            first_reconstruction_ranks={"stock": first_reconstruction_rank},
            stratification_length=source_route.length if source_route else None,
            stratification_is_convergent=source_route.has_convergent_reaction if source_route else None,
            wall_time=wall_time,
            cpu_time=cpu_time,
        )

        eval_results.results[target_id] = t_eval

    eval_results.metadata["scoring_denominator"] = {
        "type": denominator_type,
        "n_candidates_scored": n_candidates,
    }

    return eval_results
