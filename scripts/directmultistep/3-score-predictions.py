"""
Score DMS predictions against stock availability and experimental routes.

This script evaluates processed predictions to compute:
1. Solvability - whether routes can be solved with available building blocks
2. Experimental route rank - rank of experimental route among solved predictions

Usage:
    uv run scripts/directmultistep/3-score-predictions.py
"""

from pathlib import Path

from retrocast.evaluation import evaluate_predictions
from retrocast.io import load_routes, load_stock, save_evaluation_results
from retrocast.utils.logging import logger

base_dir = Path(__file__).resolve().parents[2]
PROCESSED_DIR = base_dir / "data" / "processed"
PAROUTES_DIR = base_dir / "data" / "paroutes"
SCORED_DIR = base_dir / "data" / "scored"


def main():
    model_name = "dms-flash-fp16"

    # Load stocks
    stocks = {
        "n1-stock": load_stock(PAROUTES_DIR / "n1-stock.txt"),
        "n5-stock": load_stock(PAROUTES_DIR / "n5-stock.txt"),
    }

    for dataset in ["n1", "n5"]:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Evaluating {model_name} on {dataset}")
        logger.info(f"{'=' * 60}")

        # Load predicted routes
        predictions_path = PROCESSED_DIR / model_name / dataset / "results.json.gz"
        predictions = load_routes(predictions_path)

        # Load experimental routes
        experimental_path = PAROUTES_DIR / "processed" / f"{dataset}-routes.json.gz"
        experimental_routes_dict = load_routes(experimental_path)
        # Extract first route for each target (experimental route)
        experimental_routes = {target_id: routes[0] for target_id, routes in experimental_routes_dict.items() if routes}

        # Evaluate
        results = evaluate_predictions(
            predictions=predictions,
            stocks=stocks,
            experimental_routes=experimental_routes,
            model_name=model_name,
            dataset_name=f"paroutes-{dataset}",
        )

        # Save results
        output_path = SCORED_DIR / model_name / dataset / "evaluation.json.gz"
        save_evaluation_results(results, output_path)

        logger.info(f"Completed evaluation for {dataset}")


if __name__ == "__main__":
    main()
