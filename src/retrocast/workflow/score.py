import logging

from retrocast.io.data import RoutesDict
from retrocast.metrics.similarity import find_acceptable_match
from retrocast.metrics.solvability import is_route_solved
from retrocast.models.benchmark import BenchmarkSet
from retrocast.models.evaluation import EvaluationResults, ScoredRoute, TargetEvaluation
from retrocast.typing import InchiKeyStr

logger = logging.getLogger(__name__)


def score_model(
    benchmark: BenchmarkSet, predictions: RoutesDict, stock: set[InchiKeyStr], stock_name: str, model_name: str
) -> EvaluationResults:
    """
    Scores model predictions against a benchmark.

    This function evaluates each predicted route for:
    1. Solvability: Are all starting materials in stock?
    2. Acceptability: Does the route match any acceptable route?

    Stratification is based on the properties of the MATCHED acceptable route,
    not pre-computed target metadata.

    Args:
        benchmark: The benchmark set with acceptable routes
        predictions: Model predictions (target_id -> list of routes)
        stock: Set of available stock InChIKeys
        stock_name: Name of the stock set
        model_name: Name of the model being evaluated

    Returns:
        Evaluation results with per-target scoring and matched route metadata
    """
    logger.info(f"Scoring {model_name} on {benchmark.name}...")

    # Check if benchmark has any acceptable routes
    has_acceptable_routes = any(len(target.acceptable_routes) > 0 for target in benchmark.targets.values())

    eval_results = EvaluationResults(
        model_name=model_name,
        benchmark_name=benchmark.name,
        stock_name=stock_name,
        has_acceptable_routes=has_acceptable_routes,
    )

    # Iterate Targets (The Denominator)
    for target_id, target in benchmark.targets.items():
        predicted_routes = predictions.get(target_id, [])

        # Pre-compute acceptable route signatures
        acceptable_sigs = [route.get_signature() for route in target.acceptable_routes]

        scored_routes = []
        acceptable_rank = None
        # Counter for the "Effective Rank" (only increments on solvable routes)
        effective_rank_counter = 1

        for route in predicted_routes:
            # 1. Metric: Solvability
            solved = is_route_solved(route, stock)

            # 2. Metric: Acceptability (matches any acceptable route?)
            matched_idx = find_acceptable_match(route, acceptable_sigs)

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

        # Extract matched route properties for stratification
        first_solved_match = next((r for r in scored_routes if r.is_solved and r.matches_acceptable), None)

        if first_solved_match and first_solved_match.matched_acceptable_index is not None:
            matched_route = target.acceptable_routes[first_solved_match.matched_acceptable_index]
            matched_length = matched_route.length
            matched_convergent = matched_route.has_convergent_reaction
        else:
            matched_length = None
            matched_convergent = None

        t_eval = TargetEvaluation(
            target_id=target_id,
            routes=scored_routes,
            is_solvable=is_solvable,
            acceptable_rank=acceptable_rank,
            matched_route_length=matched_length,
            matched_route_is_convergent=matched_convergent,
        )

        eval_results.results[target_id] = t_eval

    return eval_results
