from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console

from retrocast.adapters import AiZynthFinderAdapter
from retrocast.cli.progress import create_cli_progress, quiet_info_logs
from retrocast.io import load_benchmark, load_json_gz, load_stock_file
from retrocast.metrics import RequiredLeavesChecker, RouteDepthChecker, StockTerminationChecker
from retrocast.utils.logging import configure_script_logging, logger
from retrocast.workflow import (
    adapt_candidates,
    adapt_routes,
    analyze,
    collect_candidates,
    collect_routes,
    ingest_candidates,
    ingest_routes,
    score,
)

RAW_PATH = Path("data/retrocast/2-raw/aizynthfinder-4.4.1-mcts-iter100-depth6/mkt-cnv-160/results.json.gz")
BENCHMARK_PATH = Path("data/retrocast/1-benchmarks/definitions/mkt-cnv-160.json.gz")
STOCK_NAME = "n5-stock"
STOCK_PATH = Path("data/retrocast/1-benchmarks/stocks/n5-stock.csv.gz")


def main() -> None:
    args = parse_args()
    configure_script_logging()

    console = Console()
    raw_payload = load_json_gz(args.raw_path)
    task = load_benchmark(args.benchmark_path)
    stock = load_stock_file(args.stock_path)
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
        tier_checkers=[],
        constraint_checkers=[
            StockTerminationChecker(stocks={args.stock_name: stock}),
            RequiredLeavesChecker(),
            RouteDepthChecker(),
        ],
    )
    report = analyze(evaluation, ks=(1, 10))
    logger.info("score: targets=%s", len(evaluation.targets))
    for name, metric in report.metrics.items():
        ci_low = "n/a" if metric.ci_low is None else f"{metric.ci_low:.3f}"
        ci_high = "n/a" if metric.ci_high is None else f"{metric.ci_high:.3f}"
        logger.info(
            "analyze: %s=%.3f ci95=[%s, %s] count=%s",
            name,
            metric.value,
            ci_low,
            ci_high,
            metric.count,
        )
    if not report.by_stratum:
        logger.info("analyze: no stratified metrics")
    for stratum, metrics in report.by_stratum.items():
        for name, metric in metrics.items():
            ci_low = "n/a" if metric.ci_low is None else f"{metric.ci_low:.3f}"
            ci_high = "n/a" if metric.ci_high is None else f"{metric.ci_high:.3f}"
            logger.info(
                "analyze: stratum=%s metric=%s value=%.3f ci95=[%s, %s] count=%s",
                stratum,
                name,
                metric.value,
                ci_low,
                ci_high,
                metric.count,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the library API example against local schema-2 artifacts.")
    parser.add_argument("--raw-path", type=Path, default=RAW_PATH)
    parser.add_argument("--benchmark-path", type=Path, default=BENCHMARK_PATH)
    parser.add_argument("--stock-name", default=STOCK_NAME)
    parser.add_argument("--stock-path", type=Path, default=STOCK_PATH)
    return parser.parse_args()


if __name__ == "__main__":
    main()
