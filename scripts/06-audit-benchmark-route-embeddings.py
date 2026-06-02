"""
audit route-pattern embeddings within benchmark acceptable routes.

usage:
    uv run scripts/06-audit-benchmark-route-embeddings.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

from retrocast.chem import InChIKeyLevel
from retrocast.curation.training.embedding_audit import (
    RouteEmbeddingAudit,
    build_route_embedding_audit,
    build_training_embedding_index,
)
from retrocast.curation.training.embedding_report import render_route_embedding_audit_markdown
from retrocast.curation.training.records import TrainingRouteRecord
from retrocast.io import load_benchmark, save_jsonl_gz
from retrocast.paths import benchmark_definitions_dir
from retrocast.utils.logging import configure_script_logging, logger

DEFAULT_BENCHMARKS = ("mkt-cnv-160", "mkt-lin-500", "ref-lng-84", "uspto-190")


def main() -> None:
    configure_script_logging()
    args = parse_args()
    match_level = InChIKeyLevel(args.match_level)
    audits: list[RouteEmbeddingAudit] = []
    all_records: list[TrainingRouteRecord] = []

    for benchmark_name in args.benchmarks:
        records = load_benchmark_records(benchmark_name, route_selection=args.route_selection)
        all_records.extend(records)
        audits.append(
            build_route_embedding_audit(
                release_name=benchmark_name,
                training_records=records,
                queries_by_source={benchmark_name: {record.id: record.route for record in records}},
                match_level=match_level,
                allow_leaf_extension=args.allow_leaf_extension,
                include_partial=True,
                partial_min_reactions=args.partial_min_reactions,
                exclude_query_containers=True,
            )
        )
        logger.info("loaded %s acceptable routes from %s", len(records), benchmark_name)

    index = build_training_embedding_index(all_records, match_level)
    audit = merge_audits(
        "benchmark-route-embeddings",
        audits,
        total_routes=len(all_records),
        route_signatures=len(index.route_signatures),
        reaction_signatures=len(index.reaction_signatures),
    )
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "benchmark-route-embedding-audit.md"
    ledger_path = output_dir / "benchmark-route-embedding-ledger.jsonl.gz"
    report_path.write_text(
        render_route_embedding_audit_markdown(audit, container_label="benchmark"),
        encoding="utf-8",
    )
    n_rows = save_jsonl_gz(audit.ledger_rows, ledger_path)
    logger.info("wrote %s and %s (%s ledger rows)", report_path, ledger_path, n_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="audit route embeddings among acceptable routes inside benchmarks")
    parser.add_argument("--benchmarks", nargs="+", default=list(DEFAULT_BENCHMARKS))
    parser.add_argument("--route-selection", choices=("primary", "all"), default="primary")
    parser.add_argument(
        "--match-level", choices=[level.value for level in InChIKeyLevel], default=InChIKeyLevel.FULL.value
    )
    parser.add_argument("--partial-min-reactions", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=Path("data/retrocast/audits"))
    parser.add_argument("--no-leaf-extension", dest="allow_leaf_extension", action="store_false")
    parser.set_defaults(allow_leaf_extension=True)
    return parser.parse_args()


def load_benchmark_records(
    benchmark_name: str,
    *,
    route_selection: Literal["primary", "all"],
) -> list[TrainingRouteRecord]:
    benchmark = load_benchmark(benchmark_definitions_dir() / f"{benchmark_name}.json.gz")
    records: list[TrainingRouteRecord] = []
    for target in benchmark.targets.values():
        routes = target.acceptable_routes[:1] if route_selection == "primary" else target.acceptable_routes
        for index, route in enumerate(routes, start=1):
            route_id = target.id if len(routes) == 1 else f"{target.id}:{index}"
            records.append(TrainingRouteRecord(id=route_id, split="training", route=route))
    return records


def merge_audits(
    release_name: str,
    audits: list[RouteEmbeddingAudit],
    *,
    total_routes: int,
    route_signatures: int,
    reaction_signatures: int,
) -> RouteEmbeddingAudit:
    if not audits:
        raise ValueError("merge_audits requires at least one audit")
    first = audits[0]
    return RouteEmbeddingAudit(
        release_name=release_name,
        match_level=first.match_level,
        allow_leaf_extension=first.allow_leaf_extension,
        partial_min_reactions=first.partial_min_reactions,
        training_routes=total_routes,
        training_route_signatures=route_signatures,
        training_reaction_signatures=reaction_signatures,
        query_sets=tuple(query_set for audit in audits for query_set in audit.query_sets),
        ledger_rows=tuple(row for audit in audits for row in audit.ledger_rows),
    )


if __name__ == "__main__":
    main()
