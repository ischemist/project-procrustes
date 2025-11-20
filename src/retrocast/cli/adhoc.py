import sys
from pathlib import Path
from typing import Any

from retrocast.api import score_predictions
from retrocast.io.files import save_json_gz
from retrocast.io.loaders import load_benchmark
from retrocast.io.routes import load_routes
from retrocast.utils.logging import logger


def handle_score_file(args: Any) -> None:
    """
    Handler for 'retrocast score-file'.
    Scores predictions from a specific file against a specific benchmark file.
    """
    benchmark_path = Path(args.benchmark)
    routes_path = Path(args.routes)
    stock_path = Path(args.stock)
    output_path = Path(args.output)

    if not benchmark_path.exists():
        logger.error(f"Benchmark file not found: {benchmark_path}")
        sys.exit(1)
    if not routes_path.exists():
        logger.error(f"Routes file not found: {routes_path}")
        sys.exit(1)
    if not stock_path.exists():
        logger.error(f"Stock file not found: {stock_path}")
        sys.exit(1)

    try:
        # Load inputs
        benchmark = load_benchmark(benchmark_path)
        routes = load_routes(routes_path)

        # Run Scoring via API
        results = score_predictions(
            benchmark=benchmark,
            predictions=routes,
            stock=stock_path,
            model_name=args.model_name,
        )

        # Save
        save_json_gz(results, output_path)
        logger.info(f"Scoring complete. Results saved to {output_path}")

    except Exception as e:
        logger.critical(f"Scoring failed: {e}", exc_info=True)
        sys.exit(1)
