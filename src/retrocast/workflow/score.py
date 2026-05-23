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
from retrocast.models.evaluation import EvaluationResults, ScoredCandidate, TargetEvaluation
from retrocast.models.validity import (
    IMPLEMENTED_VALIDITY_TIERS,
    CheckResult,
    ConstraintResult,
    MetricScope,
    ReactionValidity,
    RouteValidity,
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
    route,
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
    reactions_by_index: dict[int, ReactionValidity] = {}
    for reaction_index, reaction in enumerate(route.iter_reactions(), start=1):
        reaction_tiers = {
            tier: _make_tier_result(tier, get_reaction_tier_failure_codes(reaction, tier))
            for tier in sorted(IMPLEMENTED_VALIDITY_TIERS)
        }
        reactions_by_index[reaction_index] = ReactionValidity(
            reaction_index=reaction_index,
            product_smiles=reaction.product.smiles,
            product_inchikey=reaction.product.inchikey,
            reactant_smiles=[reactant.smiles for reactant in reaction.step.reactants],
            reactant_inchikeys=[reactant.inchikey for reactant in reaction.step.reactants],
            tiers=reaction_tiers,
        )
    route_validity = RouteValidity(tiers=route_tiers, reactions=list(reactions_by_index.values()))
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


def score_model(
    benchmark: BenchmarkSet,
    predictions: RoutesDict,
    stock: set[InchiKeyStr],
    stock_name: str,
    model_name: str,
    execution_stats: ExecutionStats | None = None,
    match_level: InchiKeyLevel = InchiKeyLevel.FULL,
) -> EvaluationResults:
    """
    Scores model predictions against a benchmark.

    This function evaluates each predicted route for:
    1. Solvability: Are all starting materials in stock?
    2. Acceptability: Does the route match any acceptable route?

    Stratification is based on the matched acceptable route if found,
    otherwise falls back to the primary acceptable route (benchmark ground truth).

    Args:
        benchmark: The benchmark set with acceptable routes
        predictions: Model predictions (target_id -> list of routes)
        stock: Set of available stock InChIKeys
        stock_name: Name of the stock set
        model_name: Name of the model being evaluated
        execution_stats: Optional runtime statistics for predictions
        match_level: Level of InChI key matching specificity:
            - None or FULL: Exact matching (default)
            - NO_STEREO: Ignore stereochemistry
            - CONNECTIVITY: Match on molecular skeleton only

    Returns:
        Evaluation results with per-target scoring and matched route metadata
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
        acceptable_rank = None
        tier_validity_ranks = {tier: None for tier in sorted(IMPLEMENTED_VALIDITY_TIERS)}
        solv_ranks = {"stock": {tier: None for tier in sorted(IMPLEMENTED_VALIDITY_TIERS)}}
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
                if passed and tier_validity_ranks[tier] is None:
                    tier_validity_ranks[tier] = candidate.rank
                if passed and scope_passes["stock"] and solv_ranks["stock"][tier] is None:
                    solv_ranks["stock"][tier] = candidate.rank

            if scope_passes["stock"]:
                if matched_idx is not None and acceptable_rank is None:
                    acceptable_rank = effective_rank_counter
                effective_rank_counter += 1

            scored_candidates.append(scored_candidate)

        # Summary for this target
        has_stock_terminated_route = any(
            candidate.constraint_results["stock"].status == "pass" for candidate in scored_candidates
        )
        has_tier_0_valid_route = tier_validity_ranks.get(0) is not None
        is_solv_0 = solv_ranks["stock"].get(0) is not None

        # Always stratify by primary acceptable route (benchmark ground truth)
        source_route = target.primary_route

        # Extract runtime metrics if available
        wall_time = execution_stats.wall_time.get(target_id) if execution_stats else None
        cpu_time = execution_stats.cpu_time.get(target_id) if execution_stats else None

        t_eval = TargetEvaluation(
            target_id=target_id,
            candidates=scored_candidates,
            has_stock_terminated_route=has_stock_terminated_route,
            has_tier_0_valid_route=has_tier_0_valid_route,
            is_solv_0=is_solv_0,
            acceptable_rank=acceptable_rank,
            tier_validity_ranks=tier_validity_ranks,
            solv_ranks=solv_ranks,
            top_k_ranks={"stock": acceptable_rank},
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
