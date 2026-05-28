from __future__ import annotations

from pathlib import Path

from rich.console import Console

from retrocast.cli.progress import create_cli_progress, quiet_info_logs
from retrocast.io import load_json_gz, load_stock_file
from retrocast.utils.logging import configure_script_logging, logger
from retrocast.v2.adapters import AiZynthFinderAdapter
from retrocast.v2.io import load_benchmark
from retrocast.v2.metrics import TaskConstraintChecker
from retrocast.v2.models import Tier
from retrocast.v2.workflow import (
    adapt_candidates,
    adapt_routes,
    collect_candidates,
    collect_routes,
    ingest_candidates,
    ingest_routes,
    score,
)

RAW_PATH = Path("data/retrocast/2-raw/aizynthfinder-4.4.1-mcts-iter100-depth6/mkt-cnv-160/results.json.gz")
BENCHMARK_PATH = Path("data/retrocast/1-benchmarks/definitions/mkt-cnv-160.json.gz")
STOCKS_DIR = Path("data/retrocast/1-benchmarks/stocks")


def main() -> None:
    configure_script_logging()

    console = Console()
    raw_payload = load_json_gz(RAW_PATH)
    task = load_benchmark(BENCHMARK_PATH)
    stock_name = task.default_constraints.stock
    stock = load_stock_file(STOCKS_DIR / f"{stock_name}.csv.gz") if stock_name is not None else set()
    adapter = AiZynthFinderAdapter()
    logger.info("loaded raw payload and benchmark: targets=%s", len(task.targets))

    routes = []
    candidates = []
    with quiet_info_logs("retrocast"), create_cli_progress(console=console, unit="target") as progress:
        task_id = progress.add_task("adapting targets", total=len(task.targets))
        for target_id, target in task.targets.items():
            source_key = target_id if target_id in raw_payload else target.smiles
            if source_key not in raw_payload:
                logger.warning("missing payload for target_id=%s smiles=%s", target_id, target.smiles)
                progress.advance(task_id)
                continue
            target_payload = raw_payload[source_key]
            routes.extend(adapt_routes(target_payload, adapter, target=target, source_key=source_key))
            candidates.extend(adapt_candidates(target_payload, adapter, target=target, source_key=source_key))
            progress.advance(task_id)
    logger.info("adapted routes=%s candidates=%s", len(routes), len(candidates))

    collected_routes = collect_routes(routes, task)
    collected_candidates = collect_candidates(candidates, task)
    logger.info(
        "collected routes=%s candidates=%s",
        sum(len(items) for items in collected_routes.values()),
        sum(len(items) for items in collected_candidates.values()),
    )

    ingested_routes = ingest_routes(raw_payload, adapter, task)
    ingested_candidates = ingest_candidates(raw_payload, adapter, task)
    logger.info("ingest_routes: %s", sum(len(items) for items in ingested_routes.values()))
    logger.info("ingest_candidates: %s", sum(len(items) for items in ingested_candidates.values()))

    evaluation = score(
        collected_candidates,
        task,
        route_tier_checkers=[],
        constraint_checker=TaskConstraintChecker.stock_termination(stock=stock, stock_name=stock_name),
    )
    solv_zero_count = sum(
        any(candidate.satisfies_solv(Tier.ZERO) for candidate in target_result.candidates)
        for target_result in evaluation.targets.values()
    )
    logger.info("score: solv_zero_targets=%s/%s", solv_zero_count, len(evaluation.targets))


if __name__ == "__main__":
    main()
