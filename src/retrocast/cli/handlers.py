from pathlib import Path
from typing import Any

from retrocast.adapters.factory import get_adapter
from retrocast.io.files import load_json_gz
from retrocast.io.loaders import load_benchmark
from retrocast.io.manifests import create_manifest
from retrocast.utils.logging import logger
from retrocast.workflow import ingest

# from retrocast.workflow import score, analyze  # To be implemented in Phase 3/4


def _get_paths(config: dict, args: Any) -> dict[str, Path]:
    """Helper to resolve standard paths based on config."""
    base = Path(config.get("data_dir", "data"))
    return {
        "benchmarks": base / "benchmarks" / "definitions",
        "raw": base / "raw",
        "processed": base / "processed",
        "scored": base / "scored",
        "results": base / "results",
    }


def handle_ingest(args: Any, config: dict[str, Any]) -> None:
    paths = _get_paths(config, args)

    # 1. Load Benchmark Definition
    bench_file = paths["benchmarks"] / f"{args.benchmark}.json.gz"
    benchmark = load_benchmark(bench_file)

    # 2. Load Adapter
    # Config should look like: models: { "dms-flash": { "adapter": "dms", ... } }
    model_conf = config.get("models", {}).get(args.model)
    if not model_conf:
        raise ValueError(f"Model '{args.model}' not found in config")

    adapter = get_adapter(model_conf["adapter"])

    # 3. Load Raw Data
    # The adapter might need to know how to read the file,
    # but usually we just load the JSON/Pickle payload here.
    logger.info(f"Loading raw data from {args.raw_file}...")
    if args.raw_file.suffix == ".gz":
        raw_data = load_json_gz(args.raw_file)
    else:
        # Handle pickle or other formats if necessary, or delegate to adapter helper
        # For now, assume JSON/Pickle loading happens here or inside a helper
        import pickle

        with open(args.raw_file, "rb") as f:
            raw_data = pickle.load(f)

    # 4. Run Workflow
    processed_routes, out_path, stats = ingest.ingest_model_predictions(
        model_name=args.model,
        benchmark=benchmark,
        raw_data=raw_data,
        adapter=adapter,
        output_dir=paths["processed"],
        anonymize=not args.no_anonymize,
        sampling_strategy=args.sampling,
        sample_k=args.k,
    )

    # 5. Create Manifest
    manifest = create_manifest(
        action="ingest",
        sources=[args.raw_file, bench_file],
        outputs=[(out_path, processed_routes)],
        parameters={"model": args.model, "benchmark": args.benchmark, "sampling": args.sampling, "k": args.k},
        statistics=stats,
    )

    manifest_path = out_path.with_name("manifest.json")
    with open(manifest_path, "w") as f:
        f.write(manifest.model_dump_json(indent=2))

    logger.info(f"Ingestion complete. Manifest saved to {manifest_path}")


def handle_score(args: Any, config: dict[str, Any]) -> None:
    # Placeholder for Phase 3
    logger.info("Scoring functionality coming in Phase 3...")
    pass


def handle_analyze(args: Any, config: dict[str, Any]) -> None:
    # Placeholder for Phase 4
    logger.info("Analysis functionality coming in Phase 4...")
    pass
