"""
audit route-pattern embeddings between a paroutes training release and query routes.

usage:
    uv run scripts/paroutes/training-set-prep/05-audit-route-embeddings.py
"""

from __future__ import annotations

from typing import Literal

from rich.console import Console

from retrocast.chem import InChIKeyLevel
from retrocast.cli.progress import step_progress
from retrocast.curation.training.embedding_audit import build_route_embedding_audit
from retrocast.curation.training.embedding_report import render_route_embedding_audit_markdown
from retrocast.curation.training.route_release import adapt_training_routes
from retrocast.io import load_benchmark, load_training_route_records, save_jsonl_gz
from retrocast.models.route import Route
from retrocast.paths import benchmark_definitions_dir, paroutes_assets_dir, paroutes_training_release_file
from retrocast.utils.logging import configure_script_logging, logger

RELEASE_VERSION = "v2026-06-05"
RELEASE_NAME = "route-holdout-n1-n5"
RELEASE_PATH = paroutes_training_release_file(RELEASE_VERSION, RELEASE_NAME)
BENCHMARKS = {"mkt-cnv-160": benchmark_definitions_dir() / "mkt-cnv-160.json.gz"}
RAW_HOLDOUTS = ("n1", "n5")
ROUTE_SELECTION: Literal["primary", "all"] = "primary"
MATCH_LEVEL = InChIKeyLevel.FULL
INCLUDE_PARTIAL = True
PARTIAL_MIN_REACTIONS = 2
ALLOW_LEAF_EXTENSION = True
OUTPUT_PATH = RELEASE_PATH.parent / "route-embedding-audit.md"
LEDGER_OUTPUT_PATH = RELEASE_PATH.parent / "route-embedding-ledger.jsonl.gz"


def main() -> None:
    configure_script_logging()
    queries: dict[str, dict[str, Route]] = {}
    with step_progress(console=Console(), total=len(BENCHMARKS) + len(RAW_HOLDOUTS) + 3, transient=True) as step:
        for source, path in BENCHMARKS.items():
            with step(f"{source}: load benchmark queries"):
                benchmark_queries: dict[str, Route] = {}
                for target in load_benchmark(path).targets.values():
                    routes = target.acceptable_routes if ROUTE_SELECTION == "all" else target.acceptable_routes[:1]
                    for index, route in enumerate(routes, start=1):
                        query_id = target.id if len(routes) == 1 else f"{target.id}:{index}"
                        benchmark_queries[query_id] = route
                queries[source] = benchmark_queries
            logger.info("loaded %s query routes from %s", len(benchmark_queries), path)

        for dataset in RAW_HOLDOUTS:
            path = paroutes_assets_dir() / f"{dataset}-routes.json.gz"
            with step(f"{dataset}: adapt holdout routes"):
                adaptation = adapt_training_routes(path, dataset=dataset, show_progress=False)
                queries[f"{dataset}-routes"] = {
                    f"{dataset}-{route.source.raw_index + 1:05d}": route.route for route in adaptation.routes
                }
            logger.info("loaded %s adapted %s query routes from %s", adaptation.stats.adapted_routes, dataset, path)

        with step(f"{RELEASE_NAME}: load training routes"):
            container_records = load_training_route_records(RELEASE_PATH)
        with step(f"{RELEASE_NAME}: build embedding audit"):
            audit = build_route_embedding_audit(
                release_name=RELEASE_NAME,
                container_records=container_records,
                queries_by_source=queries,
                match_level=MATCH_LEVEL,
                allow_leaf_extension=ALLOW_LEAF_EXTENSION,
                include_partial=INCLUDE_PARTIAL,
                partial_min_reactions=PARTIAL_MIN_REACTIONS,
            )
        with step(f"{RELEASE_NAME}: write audit outputs"):
            OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
            LEDGER_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
            OUTPUT_PATH.write_text(
                render_route_embedding_audit_markdown(audit, container_label="training"), encoding="utf-8"
            )
            ledger_rows = save_jsonl_gz(audit.ledger_rows, LEDGER_OUTPUT_PATH)
    logger.info(
        "wrote route embedding audit to %s and %s (%s ledger rows)", OUTPUT_PATH, LEDGER_OUTPUT_PATH, ledger_rows
    )


if __name__ == "__main__":
    main()
