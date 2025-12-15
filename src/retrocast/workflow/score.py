import logging

from retrocast.chem import InchiKeyLevel, reduce_inchikey
from retrocast.io.data import RoutesDict
from retrocast.metrics.similarity import find_acceptable_match
from retrocast.metrics.solvability import is_route_solved
from retrocast.models.benchmark import BenchmarkSet, ExecutionStats
from retrocast.models.evaluation import EvaluationResults, ScoredRoute, TargetEvaluation
from retrocast.typing import InchiKeyStr

logger = logging.getLogger(__name__)


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
    logger.info(f"Scoring {model_name} on {benchmark.name}...")

    if execution_stats:
        logger.info(f"Runtime stats available for {len(execution_stats.wall_time)} targets")

    # Check if benchmark has any acceptable routes
    has_acceptable_routes = any(len(target.acceptable_routes) > 0 for target in benchmark.targets.values())

    eval_results = EvaluationResults(
        model_name=model_name,
        benchmark_name=benchmark.name,
        stock_name=stock_name,
        has_acceptable_routes=has_acceptable_routes,
    )

    # Pre-normalize stock if using a non-default match level
    if match_level != InchiKeyLevel.FULL:
        stock_inchikeys = {reduce_inchikey(k, match_level) for k in stock}
    else:
        stock_inchikeys = stock

    # Iterate Targets (The Denominator)
    for target_id, target in benchmark.targets.items():
        predicted_routes = predictions.get(target_id, [])

        # Pre-compute acceptable route signatures
        acceptable_sigs = [route.get_signature(match_level=match_level) for route in target.acceptable_routes]

        scored_routes = []
        acceptable_rank = None
        # Counter for the "Effective Rank" (only increments on solvable routes)
        effective_rank_counter = 1

        for route in predicted_routes:
            # 1. Metric: Solvability
            solved = is_route_solved(route, stock_inchikeys, match_level=match_level)

            # 2. Metric: Acceptability (matches any acceptable route?)
            matched_idx = find_acceptable_match(route, acceptable_sigs, match_level=match_level)

            if solved:
                if matched_idx is not None and acceptable_rank is None:
                    acceptable_rank = effective_rank_counter
                effective_rank_counter += 1

            # Store pre-computed flags for fast stats later
            scored_routes.append(
                ScoredRoute(
                    rank=route.rank,
                    is_solved=solved,
                    matches_acceptable=matched_idx is not None,
                    matched_acceptable_index=matched_idx,
                )
            )

        # Summary for this target
        is_solvable = any(r.is_solved for r in scored_routes)

        # Always stratify by primary acceptable route (benchmark ground truth)
        source_route = target.primary_route

        # Extract runtime metrics if available
        wall_time = execution_stats.wall_time.get(target_id) if execution_stats else None
        cpu_time = execution_stats.cpu_time.get(target_id) if execution_stats else None

        t_eval = TargetEvaluation(
            target_id=target_id,
            routes=scored_routes,
            is_solvable=is_solvable,
            acceptable_rank=acceptable_rank,
            stratification_length=source_route.length if source_route else None,
            stratification_is_convergent=source_route.has_convergent_reaction if source_route else None,
            wall_time=wall_time,
            cpu_time=cpu_time,
        )

        eval_results.results[target_id] = t_eval

    return eval_results
