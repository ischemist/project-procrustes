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

from retrocast.curation.filtering import clean_and_prioritize_pools, filter_by_route_type
from retrocast.curation.sampling import sample_random, sample_stratified_priority
from retrocast.io.files import save_json_gz
from retrocast.io.loaders import load_benchmark
from retrocast.io.manifests import create_manifest
from retrocast.models.benchmark import BenchmarkSet
from retrocast.utils.logging import logger

BASE_DIR = Path(__file__).resolve().parents[2]


def create_subset(
    name: str, targets: list, source_paths: list[Path], stock_name: str, description: str, out_dir: Path, seed: int
) -> None:
    """Helper to assemble, save, and manifest a subset."""
    subset = BenchmarkSet(name=name, description=description, stock_name=stock_name, targets={t.id: t for t in targets})

    out_path = out_dir / f"{name}.json.gz"
    save_json_gz(subset, out_path)

    # Create manifest
    manifest_path = out_dir / f"{name}.manifest.json"
    manifest = create_manifest(
        action="create_subset",
        sources=source_paths,
        outputs=[(out_path, subset)],
        parameters={"seed": seed, "name": name},
        statistics={"n_targets": len(subset.targets)},
    )

    with open(manifest_path, "w") as f:
        f.write(manifest.model_dump_json(indent=2))

    logger.info(f"Created {name} with {len(subset.targets)} targets.")


def main():
    seeds = [
        # 299792458,
        # 19910806,
        # 20260317,
        # 17760704,
        # 17890304,
        # 42,
        # 20251030,
        # 662607015,
        # 20180329,
        # 20170612,
        # 20180818,
        # 20151225,
        # 19690721,
        # 20160310,
        # 19450716,
    ]
    seeds = [20251030]
    for CANONICAL_SEED in seeds:
        # CANONICAL_SEED = 42
        DEF_DIR = BASE_DIR / "data" / "1-benchmarks" / "definitions"

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
        n5_linear, n1_linear = clean_and_prioritize_pools(
            filter_by_route_type(n5, "linear"), filter_by_route_type(n1, "linear")
        )

        n5_conv, n1_conv = clean_and_prioritize_pools(
            filter_by_route_type(n5, "convergent"), filter_by_route_type(n1, "convergent")
        )

        # 3. Create Stratified Linear
        # 100 routes for lengths 2-7
        linear_counts = {d: 100 for d in range(2, 8)}

        targets_linear = sample_stratified_priority(
            pools=[n5_linear, n1_linear], group_fn=lambda t: t.route_length, counts=linear_counts, seed=CANONICAL_SEED
        )

        create_subset(
            name=f"stratified-linear-600-seed={CANONICAL_SEED}",
            targets=targets_linear,
            source_paths=[n5_path, n1_path],
            stock_name="n5-stock",  # Using n5 stock as standard
            description="Stratified set of 600 linear routes (100 each for lengths 2-7).",
            out_dir=DEF_DIR,
            seed=CANONICAL_SEED,
        )

        # 4. Create Stratified Convergent
        # 50 routes for lengths 2-6
        convergent_counts = {d: 100 for d in range(2, 6)}
        convergent_counts[6] = 50

        targets_convergent = sample_stratified_priority(
            pools=[n5_linear, n1_linear],
            group_fn=lambda t: t.route_length,
            counts=convergent_counts,
            seed=CANONICAL_SEED,
        )

        create_subset(
            name=f"stratified-convergent-450-seed={CANONICAL_SEED}",
            targets=targets_convergent,
            source_paths=[n5_path, n1_path],
            stock_name="n5-stock",
            description="Stratified set of 450 convergent routes (100 each for lengths 2-5, 50 for length 6).",
            out_dir=DEF_DIR,
            seed=CANONICAL_SEED,
        )

        # 5. Create Random Legacy Set (from n5 only, to match PaRoutes spirit)
        n5_pool = list(n5.targets.values())
        for n in [10, 100, 250, 500, 1000]:
            targets_random = sample_random(n5_pool, n, seed=CANONICAL_SEED)

            create_subset(
                name=f"random-n5-{n}-seed={CANONICAL_SEED}",
                targets=targets_random,
                source_paths=[n5_path],
                stock_name="n5-stock",  # Uses its own stock
                description=f"Random sample of {n} routes from n5 (legacy comparison).",
                out_dir=DEF_DIR,
                seed=CANONICAL_SEED,
            )


if __name__ == "__main__":
    main()
