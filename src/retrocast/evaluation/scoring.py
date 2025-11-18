"""
Per-target evaluation (scoring) for retrosynthesis routes.

This module provides functions to evaluate predicted routes against:
1. Stock availability (solvability)
2. Experimental routes (if available)
"""

from retrocast.models.chem import EvaluationResults, Route, TargetEvaluation
from retrocast.utils.logging import logger


def is_route_solved(route: Route, stock: set[str]) -> bool:
    """
    Check if a route is solved (all leaves are in stock).

    Args:
        route: Route object to check.
        stock: Set of SMILES strings representing available building blocks.

    Returns:
        True if all leaf molecules are in the stock, False otherwise.
    """
    leaves = route.leaves
    return all(leaf.smiles in stock for leaf in leaves)


def find_experimental_route_rank(
    predicted_routes: list[Route], experimental_route: Route, stock: set[str]
) -> int | None:
    """
    Find the rank of the experimental route among solved predicted routes.

    Only routes that are solved (all leaves in stock) are considered when computing rank.
    Rank is 1-indexed (rank=1 is the top prediction).

    Args:
        predicted_routes: List of predicted routes, already sorted by model rank.
        experimental_route: The experimental route to find.
        stock: Set of SMILES strings representing available building blocks.

    Returns:
        Rank (1-indexed) if found among solved routes, None otherwise.
    """
    experimental_sig = experimental_route.get_signature()

    rank = 1
    for route in predicted_routes:
        # Only consider solved routes
        if not is_route_solved(route, stock):
            continue

        if route.get_signature() == experimental_sig:
            return rank

        rank += 1

    return None


def evaluate_predictions(
    predictions: dict[str, list[Route]],
    stocks: dict[str, set[str]],
    experimental_routes: dict[str, Route] | None,
    model_name: str,
    dataset_name: str,
) -> EvaluationResults:
    """
    Evaluate predicted routes against stocks and experimental routes.

    Args:
        predictions: Map of target_id to list of predicted routes (sorted by rank).
        stocks: Map of stock_name to set of SMILES strings.
        experimental_routes: Optional map of target_id to experimental route.
        model_name: Name of the model being evaluated.
        dataset_name: Name of the dataset being evaluated.

    Returns:
        EvaluationResults object with all metrics computed.
    """
    stock_names = list(stocks.keys())
    results = EvaluationResults(
        model_name=model_name,
        dataset_name=dataset_name,
        stock_names=stock_names,
        targets={},
    )

    for target_id, routes in predictions.items():
        target_eval = TargetEvaluation(target_id=target_id)

        # Store experimental route length if available
        if experimental_routes and target_id in experimental_routes:
            exp_route = experimental_routes[target_id]
            target_eval.experimental_route_length = exp_route.depth

        # For each stock, compute metrics
        for stock_name, stock in stocks.items():
            # Check if any route is solved
            solved_routes = [r for r in routes if is_route_solved(r, stock)]
            num_solved = len(solved_routes)
            is_solvable = num_solved > 0

            target_eval.solvability[stock_name] = is_solvable
            target_eval.num_solved_routes[stock_name] = num_solved

            # Find experimental route rank if available
            if experimental_routes and target_id in experimental_routes:
                exp_route = experimental_routes[target_id]
                rank = find_experimental_route_rank(routes, exp_route, stock)
                target_eval.experimental_route_rank[stock_name] = rank
            else:
                target_eval.experimental_route_rank[stock_name] = None

        results.targets[target_id] = target_eval

    # Log summary
    for stock_name in stock_names:
        num_solvable = sum(1 for t in results.targets.values() if t.solvability[stock_name])
        logger.info(f"{stock_name}: {num_solvable}/{len(results.targets)} targets solvable")

    return results
