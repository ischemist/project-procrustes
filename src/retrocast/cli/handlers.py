import logging
import sys
from pathlib import Path
from typing import Any

from rich.console import Console

from retrocast.adapters.factory import get_adapter
from retrocast.curation.sampling import SAMPLING_STRATEGIES
from retrocast.io.blob import load_json_gz, save_json_gz
from retrocast.io.data import load_benchmark, load_routes, load_stock_file
from retrocast.io.provenance import create_manifest
from retrocast.models.evaluation import EvaluationResults
from retrocast.visualization.report import create_single_model_summary_table, generate_markdown_report
from retrocast.workflow import analyze, ingest, score

console = Console()
logger = logging.getLogger(__name__)


def _get_paths(config: dict) -> dict[str, Path]:
    """Resolve standard directory layout."""
    base = Path(config.get("data_dir", "data"))
    return {
        "benchmarks": base / "1-benchmarks" / "definitions",
        "stocks": base / "1-benchmarks" / "stocks",
        "raw": base / "2-raw",
        "processed": base / "3-processed",
        "scored": base / "4-scored",
        "results": base / "5-results",
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
    avail_files = list(paths["benchmarks"].glob("*.json.gz"))
    avail_names = [p.name.replace(".json.gz", "") for p in avail_files]

    if hasattr(args, "all_datasets") and args.all_datasets:
        return avail_names

    if hasattr(args, "dataset") and args.dataset:
        if args.dataset not in avail_names:
            logger.error(f"Benchmark '{args.dataset}' not found in {paths['benchmarks']}")
            sys.exit(1)
        return [args.dataset]

    logger.error("Must specify --dataset or --all-datasets")
    sys.exit(1)


# --- INGESTION ---


def _ingest_single(model_name: str, benchmark_name: str, config: dict, paths: dict, args: Any) -> None:
    """The core logic for ingestion."""
    model_conf = config["models"][model_name]

    # Convention: data/raw/{model}/{benchmark}/{filename}
    raw_filename = model_conf.get("raw_results_filename", "results.json.gz")
    raw_path = paths["raw"] / model_name / benchmark_name / raw_filename

    if not raw_path.exists():
        logger.warning(f"Skipping {model_name}/{benchmark_name}: File not found at {raw_path}")
        return

    # Resolve Sampling
    strategy = getattr(args, "sampling_strategy", None)
    k = getattr(args, "k", None)

    if not strategy:
        samp_conf = model_conf.get("sampling")
        if samp_conf:
            strategy = samp_conf.get("strategy")
            k = samp_conf.get("k")

    if strategy and strategy not in SAMPLING_STRATEGIES:
        logger.error(f"Invalid sampling strategy: {strategy}")
        return

    try:
        benchmark = load_benchmark(paths["benchmarks"] / f"{benchmark_name}.json.gz")
        adapter = get_adapter(model_conf["adapter"])

        if raw_path.suffix == ".gz":
            raw_data = load_json_gz(raw_path)
        else:
            raise NotImplementedError("Unsupported file format (only .json.gz supported currently)")

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


# --- SCORING ---


def _score_single(model_name: str, benchmark_name: str, paths: dict, args: Any) -> None:
    bench_path = paths["benchmarks"] / f"{benchmark_name}.json.gz"
    routes_path = paths["processed"] / benchmark_name / model_name / "routes.json.gz"

    if not routes_path.exists():
        logger.warning(f"Skipping score for {model_name}/{benchmark_name}: Routes not found. Run ingest first.")
        return

    try:
        benchmark = load_benchmark(bench_path)

        # Determine Stock
        # 1. CLI Arg -> 2. Benchmark Def -> 3. Fail
        stock_name = getattr(args, "stock", None) or benchmark.stock_name
        if not stock_name:
            logger.error(f"Skipping {benchmark_name}: No stock specified in definition or CLI.")
            return

        stock_path = paths["stocks"] / f"{stock_name}.txt"
        if not stock_path.exists():
            logger.error(f"Stock file missing: {stock_path}")
            return

        stock_set = load_stock_file(stock_path)
        predictions = load_routes(routes_path)

        eval_results = score.score_model(
            benchmark=benchmark, predictions=predictions, stock=stock_set, stock_name=stock_name, model_name=model_name
        )

        # Save Output: data/4-scored/{benchmark}/{model}/{stock}/evaluation.json.gz
        output_dir = paths["scored"] / benchmark_name / model_name / stock_name
        output_dir.mkdir(parents=True, exist_ok=True)

        out_path = output_dir / "evaluation.json.gz"
        save_json_gz(eval_results, out_path)

        # Manifest
        manifest = create_manifest(
            action="score_model",
            sources=[bench_path, routes_path, stock_path],
            outputs=[(out_path, eval_results)],
            parameters={"model": model_name, "benchmark": benchmark_name, "stock": stock_name},
            statistics={
                "n_targets": len(eval_results.results),
                "n_solvable": sum(1 for r in eval_results.results.values() if r.is_solvable),
            },
        )

        with open(output_dir / "manifest.json", "w") as f:
            f.write(manifest.model_dump_json(indent=2))

        logger.info(f"Scored {model_name} on {benchmark_name} (Stock: {stock_name}). Saved to {out_path}")

    except Exception as e:
        logger.error(f"Failed to score {model_name} on {benchmark_name}: {e}", exc_info=True)


def handle_score(args: Any, config: dict[str, Any]) -> None:
    paths = _get_paths(config)
    models = _resolve_models(args, config)
    benchmarks = _resolve_benchmarks(args, paths)

    logger.info(f"Queued scoring: {len(models)} models x {len(benchmarks)} benchmarks.")

    for model in models:
        for bench in benchmarks:
            _score_single(model, bench, paths, args)


# --- ANALYSIS ---


def _analyze_single(model_name: str, benchmark_name: str, paths: dict, args: Any) -> None:
    # We need to know WHICH stock was used for scoring.
    # If CLI arg provided, use it. Else, check directory for single entry.
    stock_arg = getattr(args, "stock", None)
    scored_base = paths["scored"] / benchmark_name / model_name

    if not scored_base.exists():
        logger.warning(f"Skipping analysis for {model_name}/{benchmark_name}: No scored data found.")
        return

    if stock_arg:
        stocks_to_process = [stock_arg]
    else:
        # Auto-discover scored stocks
        stocks_to_process = [d.name for d in scored_base.iterdir() if d.is_dir()]
        if not stocks_to_process:
            logger.warning(f"No stock directories found in {scored_base}")
            return

    for stock_name in stocks_to_process:
        score_path = scored_base / stock_name / "evaluation.json.gz"
        if not score_path.exists():
            logger.warning(f"Missing evaluation file: {score_path}")
            continue

        try:
            logger.info(f"Analyzing {model_name} | {benchmark_name} | {stock_name}...")

            # Load
            raw_data = load_json_gz(score_path)
            eval_results = EvaluationResults.model_validate(raw_data)

            # Compute (delegated to workflow)
            final_stats = analyze.compute_model_statistics(eval_results)

            # Save
            output_dir = paths["results"] / benchmark_name / model_name / stock_name
            output_dir.mkdir(parents=True, exist_ok=True)
            save_json_gz(final_stats, output_dir / "statistics.json.gz")

            # Report (Markdown)
            report = generate_markdown_report(final_stats, visible_k=args.top_k)
            with open(output_dir / "report.md", "w") as f:
                f.write(report)

            # Visualization (HTML)
            if args.make_plots:
                from retrocast.visualization.plots import plot_diagnostics

                fig = plot_diagnostics(final_stats)
                fig.write_html(output_dir / "diagnostics.html", include_plotlyjs="cdn", auto_open=False)

            # CLI Feedback (Rich Table)
            console.print()
            console.print(create_single_model_summary_table(final_stats))
            console.print(f"\n[dim]Full report saved to: {output_dir}[/]\n")

        except Exception as e:
            logger.error(f"Failed analysis for {model_name} ({stock_name}): {e}", exc_info=True)


def handle_analyze(args: Any, config: dict[str, Any]) -> None:
    paths = _get_paths(config)
    models = _resolve_models(args, config)
    benchmarks = _resolve_benchmarks(args, paths)

    logger.info(f"Queued analysis: {len(models)} models x {len(benchmarks)} benchmarks.")

    for model in models:
        for bench in benchmarks:
            _analyze_single(model, bench, paths, args)


# --- UTILS ---


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
