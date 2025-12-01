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
from retrocast.io import create_manifest, load_benchmark, load_stock_file, save_json_gz
from retrocast.models.benchmark import create_benchmark
from retrocast.utils.logging import configure_script_logging, logger

BASE_DIR = Path(__file__).resolve().parents[2]
STOCKS_DIR = BASE_DIR / "data" / "1-benchmarks" / "stocks"


def create_subset(
    name: str, targets: list, source_paths: list[Path], stock_name: str, description: str, out_dir: Path, seed: int
) -> None:
    """Helper to assemble, save, and manifest a subset."""
    # Load the stock for validation
    stock_path = STOCKS_DIR / f"{stock_name}.csv.gz"
    stock = load_stock_file(stock_path)

    subset = create_benchmark(
        name=name, description=description, stock=stock, stock_name=stock_name, targets={t.id: t for t in targets}
    )

    out_path = out_dir / f"{name}.json.gz"
    save_json_gz(subset, out_path)

    # Create manifest
    manifest_path = out_dir / f"{name}.manifest.json"
    manifest = create_manifest(
        action="scripts/paroutes/02-create-subsets",
        sources=source_paths + [stock_path],
        outputs=[(out_path, subset, "benchmark")],
        root_dir=BASE_DIR / "data",
        parameters={"seed": seed, "name": name},
        statistics={"n_targets": len(subset.targets)},
    )

    with open(manifest_path, "w") as f:
        f.write(manifest.model_dump_json(indent=2))

    logger.info(f"Created {name} with {len(subset.targets)} targets.")


def main():
    configure_script_logging()
    # seeds = [299792458, 19910806, 20260317, 17760704, 17890304, 42, 20251030, 662607015, 20180329,
    # 20170612, 20180818, 20151225, 19690721, 20160310, 19450716] # fmt:skip

    REFLIN_SEED = 17890304
    REFCNV_SEED = 662607015
    MKTLIN_SEED = 19450716
    MKTCNV_SEED = 20180329

    DEF_DIR = BASE_DIR / "data" / "1-benchmarks" / "definitions"

    # Reference Routes (full paroutes)
    n1_path = DEF_DIR / "paroutes-n1-full-pruned.json.gz"
    n5_path = DEF_DIR / "paroutes-n5-full-pruned.json.gz"

    if not n5_path.exists() or not n1_path.exists():
        logger.error("Full datasets not found. Run 01-cast-paroutes.py first.")
        return

    n5 = load_benchmark(n5_path)
    n5_linear = filter_by_route_type(n5, "linear")
    n5_conv = filter_by_route_type(n5, "convergent")

    # 3. Create Stratified Linear
    # 100 routes for lengths 2-7
    linear_counts = {d: 100 for d in range(2, 8)}

    targets_linear = sample_stratified_priority(
        pools=[n5_linear], group_fn=lambda t: t.route_length, counts=linear_counts, seed=REFLIN_SEED
    )

    create_subset(
        name="ref-lin-600",
        targets=targets_linear,
        source_paths=[n5_path],
        stock_name="n5-stock",
        description="Stratified set of 600 linear routes (100 each for lengths 2-7).",
        out_dir=DEF_DIR,
        seed=REFLIN_SEED,
    )

    # 4. Create Stratified Convergent
    convergent_counts = {d: 100 for d in range(2, 6)}

    targets_convergent = sample_stratified_priority(
        pools=[n5_conv],
        group_fn=lambda t: t.route_length,
        counts=convergent_counts,
        seed=REFCNV_SEED,
    )

    create_subset(
        name="ref-cnv-400",
        targets=targets_convergent,
        source_paths=[n5_path],
        stock_name="n5-stock",
        description="Stratified set of 400 convergent routes (100 each for lengths 2-5).",
        out_dir=DEF_DIR,
        seed=REFCNV_SEED,
    )

    n1 = load_benchmark(n1_path)
    n5_pool, n1_pool = clean_and_prioritize_pools(list(n5.targets.values()), list(n1.targets.values()))
    long_counts = {d: 100 for d in range(8, 11)}
    # note - there are much fewer than 100 routes for such lengths, so 100 acts as "take all"
    target_long = sample_stratified_priority(
        pools=[n5_pool, n1_pool],
        group_fn=lambda t: t.route_length,
        counts=long_counts,
        seed=42,  # seed doesn't really matter since we're taking all routes for lengths 8-10
    )
    create_subset(
        name="ref-lng-84",
        targets=target_long,
        source_paths=[n5_path, n1_path],
        stock_name="n1-n5-stock",
        description="84 targets with extra long (8-10 steps) ground truth routes.",
        out_dir=DEF_DIR,
        seed=42,
    )

    # 5. Create Random Legacy Set
    n5_pool = list(n5.targets.values())
    for n in [50, 100, 250, 500, 1000, 2000]:
        targets_random = sample_random(n5_pool, n, seed=42)

        create_subset(
            name=f"random-n5-{n}",
            targets=targets_random,
            source_paths=[n5_path],
            stock_name="n5-stock",
            description=f"Random sample of {n} routes from n5 (legacy comparison).",
            out_dir=DEF_DIR,
            seed=42,
        )
    # ------------ PaRoutes with Buyables Enforced ----------
    n5_path = DEF_DIR / "paroutes-n5-full-buyables-pruned.json.gz"

    if not n5_path.exists():
        logger.error("Full datasets not found. Run 01-cast-paroutes.py first.")
        return

    n5 = load_benchmark(n5_path)
    n5_linear = filter_by_route_type(n5, "linear")
    n5_conv = filter_by_route_type(n5, "convergent")
    # 3. Create Stratified Linear
    linear_counts = {d: 100 for d in range(2, 7)}
    targets_linear = sample_stratified_priority(
        pools=[n5_linear], group_fn=lambda t: t.route_length, counts=linear_counts, seed=MKTLIN_SEED
    )

    create_subset(
        name="mkt-lin-500",
        targets=targets_linear,
        source_paths=[n5_path],
        stock_name="buyables-stock",
        description="Stratified set of 500 linear routes (100 each for lengths 2-6) that are solvable with buyables stock set.",
        out_dir=DEF_DIR,
        seed=MKTLIN_SEED,
    )

    # 4. Create Stratified Convergent
    convergent_counts = {d: 40 for d in range(2, 6)}

    targets_convergent = sample_stratified_priority(
        pools=[n5_conv],
        group_fn=lambda t: t.route_length,
        counts=convergent_counts,
        seed=MKTCNV_SEED,
    )

    create_subset(
        name="mkt-cnv-160",
        targets=targets_convergent,
        source_paths=[n5_path],
        stock_name="buyables-stock",
        description="Stratified set of 160 convergent routes (40 each for lengths 2-5) that are solvable with buyables stock set.",
        out_dir=DEF_DIR,
        seed=MKTCNV_SEED,
    )


if __name__ == "__main__":
    main()
