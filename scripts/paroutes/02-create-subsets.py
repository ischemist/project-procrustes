"""
Creates the canonical evaluation subsets from the full PaRoutes datasets.

Subsets created:
1. stratified-linear-600 (100 routes each for depth 2-7)
2. stratified-convergent-250 (50 routes each for depth 2-6)
3. random-n5-100 (legacy support)

Usage:
    uv run scripts/paroutes/02-create-subsets.py
"""

from pathlib import Path

from retrocast.curation.filtering import filter_by_route_type, merge_target_pools
from retrocast.curation.sampling import sample_random, sample_stratified
from retrocast.io.files import save_json_gz
from retrocast.io.loaders import load_benchmark
from retrocast.io.manifests import create_manifest
from retrocast.models.benchmark import BenchmarkSet
from retrocast.utils.logging import logger

# CONFIG
BASE_DIR = Path(__file__).resolve().parents[2]
DEF_DIR = BASE_DIR / "data" / "1-benchmarks" / "definitions"

# The seed that defines the benchmark forever.
CANONICAL_SEED = 42


def create_subset(name: str, targets: list, source_paths: list[Path], stock_name: str, description: str) -> None:
    """Helper to assemble, save, and manifest a subset."""
    subset = BenchmarkSet(name=name, description=description, stock_name=stock_name, targets={t.id: t for t in targets})

    out_path = DEF_DIR / f"{name}.json.gz"
    save_json_gz(subset, out_path)

    # Create manifest
    manifest_path = DEF_DIR / f"{name}.manifest.json"
    manifest = create_manifest(
        action="create_subset",
        sources=source_paths,
        outputs=[(out_path, subset)],
        parameters={"seed": CANONICAL_SEED, "name": name},
        statistics={"n_targets": len(subset.targets)},
    )

    with open(manifest_path, "w") as f:
        f.write(manifest.model_dump_json(indent=2))

    logger.info(f"Created {name} with {len(subset.targets)} targets.")


def main():
    # 1. Load Universe
    n1_path = DEF_DIR / "paroutes-n1-full.json.gz"
    n5_path = DEF_DIR / "paroutes-n5-full.json.gz"

    if not n5_path.exists() or not n1_path.exists():
        logger.error("Full datasets not found. Run 01-cast-paroutes.py first.")
        return

    n5 = load_benchmark(n5_path)
    n1 = load_benchmark(n1_path)

    # 2. Prepare Pools
    # We use n5 as primary, n1 as fallback
    pool_linear = merge_target_pools(filter_by_route_type(n5, "linear"), filter_by_route_type(n1, "linear"))

    pool_convergent = merge_target_pools(filter_by_route_type(n5, "convergent"), filter_by_route_type(n1, "convergent"))

    logger.info(f"Pool size - Linear: {len(pool_linear)}, Convergent: {len(pool_convergent)}")

    # 3. Create Stratified Linear
    # 100 routes for lengths 2-7
    linear_counts = {d: 100 for d in range(2, 8)}

    targets_linear = sample_stratified(
        pool_linear, group_fn=lambda t: t.route_length, counts=linear_counts, seed=CANONICAL_SEED
    )

    create_subset(
        name="stratified-linear-600",
        targets=targets_linear,
        source_paths=[n5_path, n1_path],
        stock_name="n5-stock",  # Using n5 stock as standard
        description="Stratified set of 600 linear routes (100 each for lengths 2-7).",
    )

    # 4. Create Stratified Convergent
    # 50 routes for lengths 2-6
    convergent_counts = {d: 50 for d in range(2, 7)}

    targets_convergent = sample_stratified(
        pool_convergent, group_fn=lambda t: t.route_length, counts=convergent_counts, seed=CANONICAL_SEED
    )

    create_subset(
        name="stratified-convergent-250",
        targets=targets_convergent,
        source_paths=[n5_path, n1_path],
        stock_name="n5-stock",
        description="Stratified set of 250 convergent routes (50 each for lengths 2-6).",
    )

    # 5. Create Random Legacy Set (from n5 only, to match PaRoutes spirit)
    n5_pool = list(n5.targets.values())
    targets_random = sample_random(n5_pool, 100, seed=CANONICAL_SEED)

    create_subset(
        name="random-n5-100",
        targets=targets_random,
        source_paths=[n5_path],
        stock_name="n5-stock",  # Uses its own stock
        description="Random sample of 100 routes from n5 (legacy comparison).",
    )


if __name__ == "__main__":
    main()
