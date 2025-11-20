from retrocast.io.routes import RoutesDict
from retrocast.metrics.similarity import is_exact_match
from retrocast.metrics.solvability import is_route_solved
from retrocast.models.benchmark import BenchmarkSet
from retrocast.models.evaluation import EvaluationResults, ScoredRoute, TargetEvaluation
from retrocast.typing import SmilesStr
from retrocast.utils.logging import logger


def score_model(
    benchmark: BenchmarkSet, predictions: RoutesDict, stock: set[SmilesStr], stock_name: str, model_name: str
) -> EvaluationResults:
    logger.info(f"Scoring {model_name} on {benchmark.name}...")

    eval_results = EvaluationResults(model_name=model_name, benchmark_name=benchmark.name, stock_name=stock_name)

    # Iterate Targets (The Denominator)
    for target_id, target in benchmark.targets.items():
        predicted_routes = predictions.get(target_id, [])

        # Pre-compute GT signature
        gt_sig = target.ground_truth.get_signature() if target.ground_truth else None

        scored_routes = []
        found_gt_rank = None
        # Counter for the "Effective Rank" (only increments on solvable routes)
        effective_rank_counter = 1

        for route in predicted_routes:
            # 1. Metric: Solvability
            solved = is_route_solved(route, stock)

            # 2. Metric: Similarity (Exact Match)
            match = (gt_sig is not None) and is_exact_match(route, gt_sig)

            if solved:
                if match and found_gt_rank is None:
                    found_gt_rank = effective_rank_counter
                effective_rank_counter += 1

            # Store pre-computed flags for fast stats later
            scored_routes.append(ScoredRoute(rank=route.rank, is_solved=solved, is_gt_match=match))

        # Summary for this target
        is_solvable = any(r.is_solved for r in scored_routes)

        t_eval = TargetEvaluation(
            target_id=target_id,
            routes=scored_routes,
            is_solvable=is_solvable,
            gt_rank=found_gt_rank,
            route_length=target.route_length,
            is_convergent=target.is_convergent,
        )

        eval_results.results[target_id] = t_eval

    return eval_results
