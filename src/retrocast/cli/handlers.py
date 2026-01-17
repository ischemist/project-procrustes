import logging
import os
import sys
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

from retrocast.adapters.factory import get_adapter
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

    ignore_stereo = getattr(args, "ignore_stereo", False)

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

            console.print()
            console.print(create_single_model_summary_table(final_stats, visible_k=args.top_k))
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

    # --- Render the report ---
    icons = {"PASS": "[green]✓[/]", "FAIL": "[red]✗[/]", "WARN": "[yellow]![/]", "INFO": "[cyan]i[/]"}

    for issue in report.issues:
        # 1. Check if it's a main header
        if issue.category == "header":
            if lines and lines[-1] != "":  # Add a blank line for spacing if needed
                lines.append("")
            lines.append(f"[bold cyan][{issue.message}][/]")
            continue  # CRITICAL: Do not process this line further

        # 2. Check if it's a sub-header (context for a group of checks)
        if issue.category == "context":
            lines.append(f"[dim]{issue.message}[/dim]")
            continue  # CRITICAL: Do not process this line further

        # 3. If it's none of the above, it's a standard check result line
        icon = icons.get(issue.level, "[dim]?[/]")
        lines.append(f"  {icon} [dim]{issue.path}[/]: {issue.message}")

    content = "\n".join(lines).strip()
    panel = Panel(content, title=title, border_style=color, expand=False, padding=(1, 2))
    console.print(panel)


def handle_verify(args: Any, config: dict[str, Any]) -> None:
    """Handler for the 'verify' command."""
    paths = _get_paths(config)
    root_dir = paths["raw"].parent

    # Determine lenient mode: default is True (lenient), --strict sets it to False
    lenient = not getattr(args, "strict", False)

    manifests_to_check = []
    output_only_manifests = set()  # Track which manifests should only check outputs

    if args.all:
        mode_desc = "strict" if not lenient else "lenient"
        logger.info(f"Scanning for manifests in 1-benchmarks, 2-raw, 3-processed, and 4-scored... (mode: {mode_desc})")
        # Scan workflow folders (check both input and output hashes)
        for folder in [paths["raw"], paths["processed"], paths["scored"]]:
            if folder.exists():
                manifests_to_check.extend(folder.glob("**/*manifest.json"))

        # Also scan benchmark folders (only check output hashes)
        for folder in [paths["benchmarks"], paths["stocks"]]:
            if folder.exists():
                benchmark_manifests = list(folder.glob("**/*manifest.json"))
                manifests_to_check.extend(benchmark_manifests)
                output_only_manifests.update(benchmark_manifests)

        manifests_to_check = sorted(manifests_to_check)
    elif args.target:
        manifests_to_check = [Path(args.target)]

    if not manifests_to_check:
        logger.warning("No manifests found to verify.")
        return

    # Different behavior for --all vs single manifest
    if args.all:
        # Batch mode: single progress bar and summary at the end
        logger.info(f"Verifying {len(manifests_to_check)} manifest(s)...")

        passed_manifests = []
        failed_manifests = []
        warnings_count = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]Verifying manifests...", total=len(manifests_to_check))

            for m_path in manifests_to_check:
                progress.update(task, description=f"[cyan]Verifying {m_path.relative_to(root_dir)}...")

                # Check if this manifest should only verify output files
                output_only = m_path in output_only_manifests
                report = verify.verify_manifest(
                    m_path, root_dir=root_dir, deep=args.deep, output_only=output_only, lenient=lenient
                )

                if report.is_valid:
                    passed_manifests.append(m_path.relative_to(root_dir))
                    # Count warnings even in passed manifests
                    warnings_count += sum(1 for issue in report.issues if issue.level == "WARN")
                else:
                    failed_manifests.append((m_path.relative_to(root_dir), report))

                progress.advance(task)

        # Print summary
        console.print()
        console.print("[bold]Verification Summary[/bold]")
        console.print(f"  [green]✓[/] Passed: {len(passed_manifests)} manifest(s)")
        console.print(f"  [red]✗[/] Failed: {len(failed_manifests)} manifest(s)")
        if warnings_count > 0:
            console.print(
                f"  [yellow]![/] Warnings: {warnings_count} missing file(s) (use --strict to require all files)"
            )

        if failed_manifests:
            console.print()
            console.print("[bold red]Failed Manifests:[/bold red]")
            for failed_path, report in failed_manifests:
                console.print(f"\n[bold]→ {failed_path}[/bold]")
                # Show only FAIL-level issues in summary
                fail_issues = [issue for issue in report.issues if issue.level == "FAIL"]
                for issue in fail_issues[:5]:  # Show first 5 failures per manifest
                    console.print(f"  [red]✗[/] {issue.path}: {issue.message}")
                if len(fail_issues) > 5:
                    console.print(f"  [dim]... and {len(fail_issues) - 5} more failure(s)[/dim]")

            console.print("\n[bold red]❌ Overall verification failed.[/]")
            sys.exit(1)
        else:
            console.print("\n[bold green]✅ All manifests verified successfully![/]")
    else:
        # Single manifest mode: show detailed report
        mode_desc = "strict" if not lenient else "lenient"
        logger.info(f"Verifying manifest... (mode: {mode_desc})")
        overall_valid = True
        has_warnings = False
        for m_path in manifests_to_check:
            report = verify.verify_manifest(m_path, root_dir=root_dir, deep=args.deep, lenient=lenient)
            _render_report(report)
            if not report.is_valid:
                overall_valid = False
            if any(issue.level == "WARN" for issue in report.issues):
                has_warnings = True

        if overall_valid:
            if has_warnings and lenient:
                console.print("\n[bold green]✅ Overall verification successful![/]")
                console.print("[dim]Note: Some files are missing but all present files have valid hashes.[/]")
                console.print("[dim]Use --strict to require all referenced files to be present.[/]")
            else:
                console.print("\n[bold green]✅ Overall verification successful![/]")
        else:
            console.print("\n[bold red]❌ Verification failed for one or more manifests.[/]")
            sys.exit(1)


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
