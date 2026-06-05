"""
Create task-constraint variants from canonical PaRoutes benchmarks.

Usage:
    uv run scripts/paroutes/benchmark-prep/05-create-constraint-variants.py
    uv run scripts/paroutes/benchmark-prep/05-create-constraint-variants.py --source mkt-cnv-160 --variants leaf leaf-depth
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

from retrocast.io import create_manifest, load_benchmark, save_benchmark
from retrocast.models import (
    STOCK_TERMINATION,
    Benchmark,
    RequiredLeavesConstraint,
    RouteDepthConstraint,
    TaskConstraint,
)
from retrocast.models.route import MoleculeView, Route
from retrocast.typing import SmilesStr
from retrocast.utils.logging import configure_script_logging, logger

BASE_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = BASE_DIR / "data" / "retrocast"
DEF_DIR = DATA_DIR / "1-benchmarks" / "definitions"

Variant = Literal["depth", "leaf", "leaf-depth"]


def select_required_leaf(route: Route) -> MoleculeView:
    leaves = route.leaves()
    if not leaves:
        raise ValueError("route has no leaves")
    return max(leaves, key=lambda leaf: (leaf.path.depth(), len(str(leaf.value.smiles)), str(leaf.value.smiles)))


def create_variant(source: Benchmark, *, source_name: str, variant: Variant) -> Benchmark:
    if not any(constraint.kind == STOCK_TERMINATION for constraint in source.default_constraints):
        raise ValueError(f"{source_name}: source benchmark must define a default stock")

    constraints: dict[str, list[TaskConstraint]] = {}
    required_leaf_ids: dict[str, str] = {}

    for target in source.targets.values():
        if not target.acceptable_routes:
            raise ValueError(f"{target.id}: target has no acceptable routes")

        route = target.acceptable_routes[0]
        leaf = select_required_leaf(route)
        required_leaf_ids[target.id] = leaf.id()
        target_constraints = []
        if "leaf" in variant:
            target_constraints.append(RequiredLeavesConstraint(smiles=[SmilesStr(leaf.value.smiles)]))
        if "depth" in variant:
            target_constraints.append(RouteDepthConstraint(max_depth=route.depth()))
        constraints[target.id] = target_constraints

    return Benchmark(
        name=f"{source_name}-{variant}",
        description=description_for_variant(source, variant),
        targets={target.id: target.model_copy(deep=True) for target in source.targets.values()},
        default_constraints=[constraint.model_copy(deep=True) for constraint in source.default_constraints],
        constraints=constraints,
        annotations={
            **source.annotations,
            "source_benchmark": source.name,
            "constraint_variant": variant,
            "required_leaf_selection": "deepest leaf; ties broken by longer smiles, then lexical smiles",
            "required_leaf_ids": required_leaf_ids if "leaf" in variant else {},
        },
    )


def description_for_variant(source: Benchmark, variant: Variant) -> str:
    suffix = {
        "depth": "with target-specific route-depth constraints.",
        "leaf": "with target-specific required-leaf constraints.",
        "leaf-depth": "with target-specific required-leaf and route-depth constraints.",
    }[variant]
    return f"{source.description.rstrip()} Derived from {source.name} {suffix}".strip()


def save_variant(source_path: Path, benchmark: Benchmark, *, variant: Variant, output_dir: Path) -> Path:
    output_path = output_dir / f"{benchmark.name}.json.gz"
    save_benchmark(benchmark, output_path)

    manifest = create_manifest(
        action="scripts/paroutes/benchmark-prep/05-create-constraint-variants",
        sources=[source_path],
        outputs=[(output_path, benchmark, "benchmark")],
        root_dir=BASE_DIR / "data",
        parameters={"source": source_path.name.removesuffix(".json.gz"), "variant": variant},
        statistics={
            "n_targets": len(benchmark.targets),
            "n_constrained_targets": len(benchmark.constraints),
            "metric_label": benchmark.derived_metric_label(),
        },
    )
    manifest_path = output_dir / f"{benchmark.name}.manifest.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2, exclude_none=True), encoding="utf-8")
    logger.info("created %s", output_path)
    return output_path


def main() -> None:
    configure_script_logging()
    parser = argparse.ArgumentParser(description="create benchmark variants with explicit task constraints.")
    parser.add_argument("--source", default="mkt-cnv-160", help="source benchmark name in data/retrocast definitions.")
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=("depth", "leaf", "leaf-depth"),
        default=("depth", "leaf", "leaf-depth"),
        help="constraint variants to materialize.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEF_DIR, help="directory for generated benchmark artifacts.")
    args = parser.parse_args()

    source_path = DEF_DIR / f"{args.source}.json.gz"
    source = load_benchmark(source_path)
    for variant in args.variants:
        benchmark = create_variant(source, source_name=args.source, variant=variant)
        save_variant(source_path, benchmark, variant=variant, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
