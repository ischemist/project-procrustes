import logging
import os
import sys
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

from retrocast.adapters.resolve import resolve_adapter, resolve_raw_results_filename
from retrocast.chem import InchiKeyLevel
from retrocast.curation.sampling import SAMPLING_STRATEGIES
from retrocast.io.blob import load_json_gz, save_json_gz
from retrocast.io.data import load_benchmark, load_execution_stats, load_routes, load_stock_file
from retrocast.io.provenance import create_manifest
from retrocast.models.evaluation import EvaluationResults
from retrocast.models.provenance import VerificationReport
from retrocast.paths import DEFAULT_DATA_DIR, ENV_VAR_NAME, get_paths
from retrocast.visualization.report import create_single_model_summary_table, generate_markdown_report
from retrocast.workflow import analyze, ingest, score, verify

console = Console()
logger = logging.getLogger(__name__)


def _get_paths(config: dict) -> dict[str, Path]:
    """Resolve standard directory layout."""
    base = Path(config.get("data_dir", DEFAULT_DATA_DIR))
    return get_paths(base)


def _resolve_models(args: Any, paths: dict, stage: str) -> list[str]:
    """Determine which models to process via filesystem discovery.

    Args:
        args: CLI arguments with --model or --all-models
        paths: Resolved paths dict from _get_paths
        stage: One of "ingest", "score", or "analyze" — controls where to scan

    Returns:
        List of model names (folder names)
    """
    # Single model specified
    if hasattr(args, "model") and args.model:
        return [args.model]

    # All models discovery
    if hasattr(args, "all_models") and args.all_models:
        if stage == "ingest":
            # Scan raw/*/*/manifest.json for models with directives.adapter
            raw_dir = paths["raw"]
            discovered = set()
            for manifest_path in raw_dir.glob("*/*/manifest.json"):
                # manifest_path is data/2-raw/{model}/{benchmark}/manifest.json
                model_name = manifest_path.parent.parent.name
                try:
                    import json

                    with open(manifest_path) as f:
                        manifest = json.load(f)
                    if manifest.get("directives", {}).get("adapter"):
                        discovered.add(model_name)
                except Exception as e:
                    logger.debug(f"Skipping malformed manifest {manifest_path}: {e}")
                    continue
            if not discovered:
                logger.warning(f"No models with manifests found in {raw_dir}")
            return sorted(discovered)

        elif stage == "score":
            # Scan processed/*/ for model directories
            processed_dir = paths["processed"]
            if not processed_dir.exists():
                logger.warning(f"Processed directory does not exist: {processed_dir}")
                return []
            # Get unique model names from processed/{benchmark}/{model}/
            discovered = set()
            for benchmark_dir in processed_dir.iterdir():
                if benchmark_dir.is_dir():
                    for model_dir in benchmark_dir.iterdir():
                        if model_dir.is_dir():
                            discovered.add(model_dir.name)
            return sorted(discovered)

        elif stage == "analyze":
            # Scan scored/*/ for model directories
            scored_dir = paths["scored"]
            if not scored_dir.exists():
                logger.warning(f"Scored directory does not exist: {scored_dir}")
                return []
            # Get unique model names from scored/{benchmark}/{model}/
            discovered = set()
            for benchmark_dir in scored_dir.iterdir():
                if benchmark_dir.is_dir():
                    for model_dir in benchmark_dir.iterdir():
                        if model_dir.is_dir():
                            discovered.add(model_dir.name)
            return sorted(discovered)

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


def _ingest_single(model_name: str, benchmark_name: str, paths: dict, args: Any) -> None:
    """The core logic for ingestion with dynamic adapter resolution."""
    raw_dir = paths["raw"] / model_name / benchmark_name

    if not raw_dir.exists():
        logger.warning(f"Skipping {model_name}/{benchmark_name}: Directory not found at {raw_dir}")
        return

    # Resolve adapter via new hierarchy (CLI > manifest > fail)
    try:
        adapter, source = resolve_adapter(
            cli_adapter=getattr(args, "adapter", None),
            raw_dir=raw_dir,
            model_name=model_name,
        )
        logger.info(f"Resolved adapter from {source} for {model_name}/{benchmark_name}")
    except Exception as e:
        logger.error(f"Skipping {model_name}/{benchmark_name}: {e}")
        return

    # Resolve raw results filename from manifest directives
    raw_filename = resolve_raw_results_filename(raw_dir=raw_dir)
    raw_path = raw_dir / raw_filename

    if not raw_path.exists():
        logger.warning(f"Skipping {model_name}/{benchmark_name}: File not found at {raw_path}")
        return

    # Resolve Sampling (CLI only, no config fallback)
    strategy = getattr(args, "sampling_strategy", None)
    k = getattr(args, "k", None)

    if strategy and strategy not in SAMPLING_STRATEGIES:
        logger.error(f"Invalid sampling strategy: {strategy}")
        return

    ignore_stereo = getattr(args, "ignore_stereo", False)

    try:
        benchmark = load_benchmark(paths["benchmarks"] / f"{benchmark_name}.json.gz")

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
            ignore_stereo=ignore_stereo,
        )

        manifest = create_manifest(
            action="ingest",
            sources=[raw_path, paths["benchmarks"] / f"{benchmark_name}.json.gz"],
            outputs=[(out_path, processed_routes, "predictions")],
            root_dir=paths["raw"].parent,  # The 'data/' directory
            parameters={"model": model_name, "benchmark": benchmark_name, "sampling": strategy, "k": k},
            statistics=stats.to_manifest_dict(),
        )

        manifest_path = out_path.with_name("manifest.json")
        with open(manifest_path, "w") as f:
            f.write(manifest.model_dump_json(indent=2))

    except Exception as e:
        logger.error(f"Failed to ingest {model_name} on {benchmark_name}: {e}", exc_info=True)


def handle_ingest(args: Any, config: dict[str, Any]) -> None:
    paths = _get_paths(config)
    models = _resolve_models(args, paths, stage="ingest")
    benchmarks = _resolve_benchmarks(args, paths)

    logger.info(f"Queued ingestion: {len(models)} models x {len(benchmarks)} benchmarks.")

    for model in models:
        for bench in benchmarks:
            _ingest_single(model, bench, paths, args)


# --- SCORING ---


def _score_single(model_name: str, benchmark_name: str, paths: dict, args: Any) -> None:
    bench_path = paths["benchmarks"] / f"{benchmark_name}.json.gz"
    routes_path = paths["processed"] / benchmark_name / model_name / "routes.json.gz"

    if not routes_path.exists():
        logger.warning(f"Skipping score for {model_name}/{benchmark_name}: Routes not found. Run ingest first.")
        return

    ignore_stereo = getattr(args, "ignore_stereo", False)
    match_level = InchiKeyLevel.NO_STEREO if ignore_stereo else InchiKeyLevel.FULL

    try:
        benchmark = load_benchmark(bench_path)

        # Determine Stock
        # 1. CLI Arg -> 2. Benchmark Def -> 3. Fail
        stock_name = getattr(args, "stock", None) or benchmark.stock_name
        if not stock_name:
            logger.error(f"Skipping {benchmark_name}: No stock specified in definition or CLI.")
            return

        stock_path = paths["stocks"] / f"{stock_name}.csv.gz"
        if not stock_path.exists():
            logger.error(f"Stock file missing: {stock_path}")
            return

        stock_set = load_stock_file(stock_path, return_as="inchikey")
        predictions = load_routes(routes_path)

        # Load execution stats if available
        execution_stats = None
        exec_stats_path = paths["raw"] / model_name / benchmark_name / "execution_stats.json.gz"
        if exec_stats_path.exists():
            try:
                execution_stats = load_execution_stats(exec_stats_path)
                logger.info(f"Loaded execution stats from {exec_stats_path}")
            except Exception as e:
                logger.warning(f"Failed to load execution stats from {exec_stats_path}: {e}")

        eval_results = score.score_model(
            benchmark=benchmark,
            predictions=predictions,
            stock=stock_set,
            stock_name=stock_name,
            model_name=model_name,
            execution_stats=execution_stats,
            match_level=match_level,
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
            outputs=[(out_path, eval_results, "unknown")],
            root_dir=paths["raw"].parent,  # The 'data/' directory
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
    models = _resolve_models(args, paths, stage="score")
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

            console.print()
            console.print(create_single_model_summary_table(final_stats, visible_k=args.top_k))
            console.print(f"\n[dim]Full report saved to: {output_dir}[/]\n")

        except Exception as e:
            logger.error(f"Failed analysis for {model_name} ({stock_name}): {e}", exc_info=True)


def handle_analyze(args: Any, config: dict[str, Any]) -> None:
    paths = _get_paths(config)
    models = _resolve_models(args, paths, stage="analyze")
    benchmarks = _resolve_benchmarks(args, paths)

    logger.info(f"Queued analysis: {len(models)} models x {len(benchmarks)} benchmarks.")

    for model in models:
        for bench in benchmarks:
            _analyze_single(model, bench, paths, args)


# --- VERIFICATION ---
EXPLANATORY_SECTIONS = {
    "Primary Artifact": "[bold]Primary Artifact[/bold]: An input file not generated by this workflow (e.g., raw data). Its integrity is a precondition.",
    "Phase 1": "[bold]Phase 1 - Manifest Chain Consistency[/bold]: Checks the 'paper trail' to ensure the logical flow of data between steps is unbroken.",
    "Phase 2": "[bold]Phase 2 - On-Disk File Integrity[/bold]: Checks the 'physical evidence' to verify that every file on disk matches its hash record in the manifests.",
    "Graph Discovery": "[bold]Graph Discovery[/bold]: The process of finding all manifests linked to the target, building a complete picture of the data's lineage.",
}


def _render_report(report: VerificationReport) -> None:
    """Pretty prints a verification report that is intelligent about its context."""
    color = "green" if report.is_valid else "red"
    title = f"Verification Report for [bold]{report.manifest_path}[/]"
    lines = []

    # --- Pre-scan to determine context ---
    categories_present = set()
    for issue in report.issues:
        if issue.category:
            categories_present.add(issue.category)

    # Show overview if multiple phases or phase1 is present
    if len(categories_present) > 1 or "phase1" in categories_present:
        lines.append("[bold]Verification Process Overview:[/bold]\n")
        if "graph" in categories_present:
            lines.append(EXPLANATORY_SECTIONS["Graph Discovery"])
        lines.append(EXPLANATORY_SECTIONS["Primary Artifact"])
        if "phase1" in categories_present:
            lines.append(EXPLANATORY_SECTIONS["Phase 1"])
        if "phase2" in categories_present:
            lines.append(EXPLANATORY_SECTIONS["Phase 2"])
        lines.append("")  # spacer

    # Show issues
    if report.issues:
        lines.append("[bold]Issues Found:[/bold]\n")
        for issue in report.issues:
            level_display = {
                "PASS": "[green]✓ PASS[/green]",
                "INFO": "[blue]ℹ INFO[/blue]",
                "WARN": "[yellow]⚠ WARN[/yellow]",
                "FAIL": "[red]✗ FAIL[/red]",
            }.get(issue.level, issue.level)

            category_str = f" [{issue.category}]" if issue.category else ""
            lines.append(f"{level_display}{category_str}: {issue.message}")
            lines.append(f"  Path: [dim]{issue.path}[/dim]\n")
    else:
        lines.append("[green]✓ No issues found[/green]")

    console.print(Panel("\n".join(lines), title=title, border_style=color))


def handle_verify(args: Any, config: dict[str, Any]) -> None:
    """Verify data integrity and lineage."""
    data_dir = Path(config.get("data_dir", DEFAULT_DATA_DIR))

    if args.all:
        # Discover all manifests in data directory
        manifest_paths = list(data_dir.glob("**/manifest.json"))
        if len(manifest_paths) == 0:
            logger.warning("No manifests found in data directory")
        else:
            logger.info(f"Discovered {len(manifest_paths)} manifests for verification")
    else:
        target = Path(args.target)
        if target.is_dir():
            manifest_path = target / "manifest.json"
            if not manifest_path.exists():
                logger.error(f"No manifest.json found in {target}")
                sys.exit(1)
            manifest_paths = [manifest_path]
        elif target.name == "manifest.json":
            manifest_paths = [target]
        else:
            logger.error(f"Target must be a directory or manifest.json file: {target}")
            sys.exit(1)

    all_reports = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Verifying manifests...", total=len(manifest_paths))

        for manifest_path in manifest_paths:
            try:
                report = verify.verify_manifest(
                    manifest_path=manifest_path,
                    root_dir=data_dir,
                    deep=args.deep,
                    lenient=not args.strict,
                )
                all_reports.append(report)
                _render_report(report)
            except Exception as e:
                console.print(f"[red]Failed to verify {manifest_path}: {e}[/]")
                logger.error(f"Verification error for {manifest_path}", exc_info=True)

            progress.advance(task)

    # Summary
    console.print()
    total = len(all_reports)
    valid_count = sum(1 for r in all_reports if r.is_valid)
    if valid_count == total:
        console.print(f"[bold green]✓ All {total} manifests verified successfully.[/]")
    else:
        console.print(f"[bold yellow]⚠ {total - valid_count} of {total} manifests failed verification.[/]")
        console.print("\n[bold red]❌ Verification failed for one or more manifests.[/]")
        sys.exit(1)


# --- UTILS ---


def handle_list(config: dict[str, Any]) -> None:
    """List discovered models from raw data manifests."""
    paths = _get_paths(config)
    raw_dir = paths["raw"]

    if not raw_dir.exists():
        console.print(f"[yellow]Raw data directory does not exist: {raw_dir}[/]")
        return

    # Discover models with manifests
    discovered = {}  # model -> [(benchmark, adapter)]
    for manifest_path in raw_dir.glob("*/*/manifest.json"):
        # manifest_path is data/2-raw/{model}/{benchmark}/manifest.json
        model_name = manifest_path.parent.parent.name
        benchmark_name = manifest_path.parent.name
        try:
            import json

            with open(manifest_path) as f:
                manifest = json.load(f)
            adapter = manifest.get("directives", {}).get("adapter")
            if adapter:
                if model_name not in discovered:
                    discovered[model_name] = []
                discovered[model_name].append((benchmark_name, adapter))
        except Exception as e:
            logger.debug(f"Skipping malformed manifest {manifest_path}: {e}")
            continue

    if not discovered:
        console.print(f"\n[yellow]No models with manifests found in {raw_dir}[/]")
        console.print("\nModels must have a manifest.json with directives.adapter in their raw data directories.")
        return

    console.print(f"\n[bold]Discovered {len(discovered)} models in {raw_dir}:[/bold]\n")
    for model_name in sorted(discovered.keys()):
        benchmarks_info = discovered[model_name]
        adapters = set(a for _, a in benchmarks_info)
        adapter_str = ", ".join(sorted(adapters))
        console.print(f"  [cyan]{model_name}[/cyan]")
        console.print(f"    Adapter(s): {adapter_str}")
        console.print(f"    Benchmarks: {len(benchmarks_info)}")

    console.print()


def handle_config(args: Any, config: dict[str, Any]) -> None:
    """Show resolved configuration and paths."""
    paths = _get_paths(config)
    data_dir = Path(config.get("data_dir", DEFAULT_DATA_DIR))
    source = config.get("_data_dir_source", "unknown")

    console.print()
    console.print("[bold]RetroCast Configuration[/bold]")
    console.print("=" * 40)

    # Data directory info
    console.print(f"\n[bold]Data directory:[/bold] {data_dir.resolve()}")
    console.print(f"  Source: {source}")

    # Environment variable
    env_value = os.environ.get(ENV_VAR_NAME)
    env_status = env_value if env_value else "[dim]not set[/dim]"
    console.print("\n[bold]Environment:[/bold]")
    console.print(f"  {ENV_VAR_NAME}: {env_status}")

    # Resolved paths
    console.print("\n[bold]Resolved paths:[/bold]")
    max_key_len = max(len(k) for k in paths)
    for name, path in paths.items():
        exists_marker = "[green]exists[/green]" if path.exists() else "[dim]missing[/dim]"
        console.print(f"  {name:<{max_key_len}}: {path} ({exists_marker})")

    console.print()
