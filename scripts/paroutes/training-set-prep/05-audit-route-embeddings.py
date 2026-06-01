"""
audit route-pattern embeddings between a paroutes training release and query routes.

usage:
    uv run scripts/paroutes/training-set-prep/05-audit-route-embeddings.py
    uv run scripts/paroutes/training-set-prep/05-audit-route-embeddings.py --include-partial
    uv run scripts/paroutes/training-set-prep/05-audit-route-embeddings.py --release-path path/to/all.jsonl.gz
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

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
from retrocast.utils.logging import configure_script_logging, logger

BASE_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = BASE_DIR / "data" / "retrocast"
RAW_DIR = DATA_DIR / "0-assets" / "paroutes"
BENCHMARK_DIR = DATA_DIR / "1-benchmarks" / "definitions"
RELEASE_VERSION = "v2026-05-12"
DEFAULT_RELEASE_PATH = (
    DATA_DIR / "releases" / "paroutes-training-sets" / RELEASE_VERSION / "route-holdout-n1-n5" / "all.jsonl.gz"
)
DEFAULT_BENCHMARK_PATHS = (BENCHMARK_DIR / "mkt-cnv-160.json.gz",)
DEFAULT_RAW_HOLDOUTS = ("n1", "n5")


def main() -> None:
    configure_script_logging()
    parser = argparse.ArgumentParser(description="audit route embeddings in a paroutes training release.")
    parser.add_argument(
        "--release-path",
        type=Path,
        default=DEFAULT_RELEASE_PATH,
        help=f"training route release all.jsonl.gz. default: {DEFAULT_RELEASE_PATH}",
    )
    parser.add_argument(
        "--benchmark-path",
        action="append",
        type=Path,
        default=None,
        help="benchmark definition to audit. may be passed more than once. default: mkt-cnv-160.",
    )
    parser.add_argument(
        "--route-selection",
        choices=("primary", "all"),
        default="primary",
        help="which acceptable benchmark routes to audit. default: primary.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=RAW_DIR,
        help=f"raw paroutes asset directory. default: {RAW_DIR}",
    )
    parser.add_argument(
        "--raw-holdout",
        action="append",
        choices=DEFAULT_RAW_HOLDOUTS,
        default=None,
        help="raw paroutes holdout to audit. may be passed more than once. default: n1 and n5.",
    )
    parser.add_argument(
        "--skip-raw-holdouts",
        action="store_true",
        help="only audit benchmark query routes.",
    )
    parser.add_argument(
        "--match-level",
        choices=tuple(level.value for level in InChIKeyLevel),
        default=InChIKeyLevel.FULL.value,
        help="molecule identity level. default: full.",
    )
    parser.add_argument(
        "--include-partial",
        action="store_true",
        help="also report non-root query subroute embeddings.",
    )
    parser.add_argument(
        "--partial-min-reactions",
        type=int,
        default=2,
        help="minimum reactions in a partial query subroute. default: 2.",
    )
    parser.add_argument(
        "--disallow-leaf-extension",
        action="store_true",
        help="require query leaves to also be leaves in the training route.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="markdown report path. default: <release-dir>/route-embedding-audit.md.",
    )
    parser.add_argument(
        "--ledger-output-path",
        type=Path,
        default=None,
        help="compressed jsonl ledger path. default: <release-dir>/route-embedding-ledger.jsonl.gz.",
    )
    parser.add_argument("--no-progress", action="store_true", help="disable progress bars.")
    args = parser.parse_args()

    queries = _load_queries(args)
    training_records = load_training_route_records(args.release_path)
    audit = build_route_embedding_audit(
        release_name=args.release_path.parent.name,
        training_records=training_records,
        queries_by_source=queries,
        match_level=InChIKeyLevel(args.match_level),
        allow_leaf_extension=not args.disallow_leaf_extension,
        include_partial=args.include_partial,
        partial_min_reactions=args.partial_min_reactions,
    )

    output_path = args.output_path or args.release_path.parent / "route-embedding-audit.md"
    ledger_output_path = args.ledger_output_path or args.release_path.parent / "route-embedding-ledger.jsonl.gz"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_route_embedding_audit_markdown(audit), encoding="utf-8")
    ledger_rows = save_jsonl_gz(audit.ledger_rows, ledger_output_path)
    logger.info(
        "wrote route embedding audit to %s and %s (%s ledger rows)", output_path, ledger_output_path, ledger_rows
    )


def _load_queries(args: argparse.Namespace) -> dict[str, list[QueryRoute]]:
    queries: dict[str, list[QueryRoute]] = {}
    benchmark_paths = args.benchmark_path or list(DEFAULT_BENCHMARK_PATHS)
    for path in benchmark_paths:
        source = _artifact_name(path)
        queries[source] = _load_benchmark_queries(path, source, route_selection=args.route_selection)

    if args.skip_raw_holdouts:
        return queries

    for dataset in args.raw_holdout or list(DEFAULT_RAW_HOLDOUTS):
        source = f"{dataset}-routes"
        queries[source] = _load_raw_paroutes_queries(
            raw_dir=args.raw_dir,
            dataset=dataset,
            show_progress=not args.no_progress,
        )
    return queries


def _load_benchmark_queries(path: Path, source: str, *, route_selection: str) -> list[QueryRoute]:
    benchmark = load_benchmark(path)
    queries = benchmark_query_routes(
        benchmark,
        source=source,
        route_selection=cast(BenchmarkRouteSelection, route_selection),
    )
    logger.info("loaded %s query routes from %s", len(queries), path)
    return queries


def _load_raw_paroutes_queries(*, raw_dir: Path, dataset: str, show_progress: bool) -> list[QueryRoute]:
    path = raw_dir / f"{dataset}-routes.json.gz"
    adaptation = adapt_training_routes(path, dataset=dataset, show_progress=show_progress)
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
