import sys
from pathlib import Path
from typing import Any

from retrocast.adapters.factory import get_adapter
from retrocast.curation.sampling import SAMPLING_STRATEGIES
from retrocast.io.files import load_json_gz
from retrocast.io.loaders import load_benchmark
from retrocast.io.manifests import create_manifest
from retrocast.utils.logging import logger
from retrocast.workflow import ingest


def _get_paths(config: dict) -> dict[str, Path]:
    """Resolve standard directory layout."""
    # default to current dir/data if not specified
    base = Path(config.get("data_dir", "data"))
    return {
        "benchmarks": base / "1-benchmarks" / "definitions",
        "raw": base / "2-raw",
        "processed": base / "3-processed",
    }


def _resolve_models(args: Any, config: dict) -> list[str]:
    """Determine which models to process."""
    defined_models = list(config.get("models", {}).keys())

    if args.all_models:
        return defined_models

    if args.model:
        if args.model not in defined_models:
            logger.error(f"Model '{args.model}' not defined in config.")
            sys.exit(1)
        return [args.model]

    logger.error("Must specify --model or --all-models")
    sys.exit(1)


def _resolve_benchmarks(args: Any, paths: dict[str, Path]) -> list[str]:
    """Determine which benchmarks to process by looking at files."""
    # Find all valid benchmark definition files
    avail_files = list(paths["benchmarks"].glob("*.json.gz"))
    avail_names = [p.name.replace(".json.gz", "") for p in avail_files]

    if args.all_datasets:
        return avail_names

    if args.dataset:
        if args.dataset not in avail_names:
            logger.error(f"Benchmark '{args.dataset}' not found in {paths['benchmarks']}")
            sys.exit(1)
        return [args.dataset]

    logger.error("Must specify --dataset or --all-datasets")
    sys.exit(1)


def _ingest_single(model_name: str, benchmark_name: str, config: dict, paths: dict, args: Any) -> None:
    """The core logic for a single run."""
    model_conf = config["models"][model_name]

    # 1. Resolve Raw File Path
    # Convention: data/raw/{model}/{benchmark}/{filename}
    raw_filename = model_conf.get("raw_results_filename", "results.json.gz")
    raw_path = paths["raw"] / model_name / benchmark_name / raw_filename

    if not raw_path.exists():
        logger.warning(f"Skipping {model_name}/{benchmark_name}: File not found at {raw_path}")
        return

    # 2. Resolve Sampling
    # CLI overrides Config overrides None
    strategy = args.sampling_strategy
    k = args.k

    if not strategy:
        # Fallback to config
        samp_conf = model_conf.get("sampling")
        if samp_conf:
            strategy = samp_conf.get("strategy")
            k = samp_conf.get("k")

    if strategy and strategy not in SAMPLING_STRATEGIES:
        logger.error(f"Invalid sampling strategy: {strategy}")
        return

    # 3. Load Artifacts
    try:
        benchmark = load_benchmark(paths["benchmarks"] / f"{benchmark_name}.json.gz")
        adapter = get_adapter(model_conf["adapter"])

        # Handle loading based on extension (assuming JSON/GZ mostly)
        # If you have pickles, you might want a helper here
        if raw_path.suffix == ".gz":
            raw_data = load_json_gz(raw_path)
        else:
            raise NotImplementedError("Unsupported file format")

        # 4. Run Workflow
        processed_routes, out_path, stats = ingest.ingest_model_predictions(
            model_name=model_name,
            benchmark=benchmark,
            raw_data=raw_data,
            adapter=adapter,
            output_dir=paths["processed"],
            anonymize=args.anonymize,
            sampling_strategy=strategy,
            sample_k=k,
        )

        # 5. Manifest
        manifest = create_manifest(
            action="ingest",
            sources=[raw_path, paths["benchmarks"] / f"{benchmark_name}.json.gz"],
            outputs=[(out_path, processed_routes)],
            parameters={"model": model_name, "benchmark": benchmark_name, "sampling": strategy, "k": k},
            statistics=stats,
        )

        manifest_path = out_path.with_name("manifest.json")
        with open(manifest_path, "w") as f:
            f.write(manifest.model_dump_json(indent=2))

    except Exception as e:
        logger.error(f"Failed to ingest {model_name} on {benchmark_name}: {e}", exc_info=True)


def handle_ingest(args: Any, config: dict[str, Any]) -> None:
    paths = _get_paths(config)
    models = _resolve_models(args, config)
    benchmarks = _resolve_benchmarks(args, paths)

    logger.info(f"Queued ingestion: {len(models)} models x {len(benchmarks)} benchmarks.")

    for model in models:
        for bench in benchmarks:
            _ingest_single(model, bench, config, paths, args)


def handle_list(config: dict[str, Any]) -> None:
    """List available models."""
    models = config.get("models", {})
    print(f"Found {len(models)} models in config:")
    for name, conf in models.items():
        print(f"  - {name} (adapter: {conf.get('adapter')})")


def handle_info(config: dict[str, Any], model_name: str) -> None:
    """Show details for a model."""
    conf = config.get("models", {}).get(model_name)
    if not conf:
        logger.error(f"Model {model_name} not found.")
        return
    import yaml

    print(yaml.dump({model_name: conf}))


# Placeholder handlers
def handle_score(args: Any, config: dict[str, Any]) -> None:
    pass


def handle_analyze(args: Any, config: dict[str, Any]) -> None:
    pass
