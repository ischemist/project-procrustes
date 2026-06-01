"""
audit route-pattern embeddings between a paroutes training release and query routes.

usage:
    uv run scripts/paroutes/training-set-prep/05-audit-route-embeddings.py
"""

from __future__ import annotations

from pathlib import Path

from retrocast.chem import InChIKeyLevel
from retrocast.curation.training import adapt_training_routes
from retrocast.curation.training.embedding_audit import (
    BenchmarkRouteSelection,
    QueryRoute,
    benchmark_query_routes,
    build_route_embedding_audit,
)
from retrocast.curation.training.embedding_report import render_route_embedding_audit_markdown
from retrocast.io import load_benchmark, load_training_route_records, save_jsonl_gz
from retrocast.paths import benchmark_definitions_dir, paroutes_assets_dir, paroutes_training_release_file
from retrocast.utils.logging import configure_script_logging, logger

RELEASE_VERSION = "v2026-05-12"
RELEASE_NAME = "route-holdout-n1-n5"
RELEASE_PATH = paroutes_training_release_file(RELEASE_VERSION, RELEASE_NAME)
BENCHMARK_PATHS = (benchmark_definitions_dir() / "mkt-cnv-160.json.gz",)
RAW_HOLDOUTS = ("n1", "n5")
ROUTE_SELECTION: BenchmarkRouteSelection = "primary"
MATCH_LEVEL = InChIKeyLevel.FULL
INCLUDE_PARTIAL = True
PARTIAL_MIN_REACTIONS = 2
ALLOW_LEAF_EXTENSION = True
SHOW_PROGRESS = True
OUTPUT_PATH = RELEASE_PATH.parent / "route-embedding-audit.md"
LEDGER_OUTPUT_PATH = RELEASE_PATH.parent / "route-embedding-ledger.jsonl.gz"


def main() -> None:
    configure_script_logging()
    queries = _load_queries()
    training_records = load_training_route_records(RELEASE_PATH)
    audit = build_route_embedding_audit(
        release_name=RELEASE_NAME,
        training_records=training_records,
        queries_by_source=queries,
        match_level=MATCH_LEVEL,
        allow_leaf_extension=ALLOW_LEAF_EXTENSION,
        include_partial=INCLUDE_PARTIAL,
        partial_min_reactions=PARTIAL_MIN_REACTIONS,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    LEDGER_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(render_route_embedding_audit_markdown(audit), encoding="utf-8")
    ledger_rows = save_jsonl_gz(audit.ledger_rows, LEDGER_OUTPUT_PATH)
    logger.info(
        "wrote route embedding audit to %s and %s (%s ledger rows)", OUTPUT_PATH, LEDGER_OUTPUT_PATH, ledger_rows
    )


def _load_queries() -> dict[str, list[QueryRoute]]:
    queries: dict[str, list[QueryRoute]] = {}
    for path in BENCHMARK_PATHS:
        source = _artifact_name(path)
        queries[source] = _load_benchmark_queries(path, source)

    for dataset in RAW_HOLDOUTS:
        source = f"{dataset}-routes"
        queries[source] = _load_raw_paroutes_queries(dataset=dataset)
    return queries


def _load_benchmark_queries(path: Path, source: str) -> list[QueryRoute]:
    benchmark = load_benchmark(path)
    queries = benchmark_query_routes(
        benchmark,
        source=source,
        route_selection=ROUTE_SELECTION,
    )
    logger.info("loaded %s query routes from %s", len(queries), path)
    return queries


def _load_raw_paroutes_queries(*, dataset: str) -> list[QueryRoute]:
    path = paroutes_assets_dir() / f"{dataset}-routes.json.gz"
    adaptation = adapt_training_routes(path, dataset=dataset, show_progress=SHOW_PROGRESS)
    logger.info("loaded %s adapted %s query routes from %s", adaptation.stats.adapted_routes, dataset, path)
    return [
        QueryRoute(
            source=f"{dataset}-routes",
            id=f"{dataset}-{route.source.raw_index + 1:05d}",
            route=route.route,
        )
        for route in adaptation.routes
    ]


def _artifact_name(path: Path) -> str:
    name = path.name
    for suffix in (".json.gz", ".jsonl.gz", ".json"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


if __name__ == "__main__":
    main()
