from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console

from retrocast import __version__
from retrocast.cli.progress import create_cli_progress, estimate_raw_route_entries, quiet_info_logs
from retrocast.exceptions import RetroCastException
from retrocast.io.blob import load_json_artifact
from retrocast.io.data import load_stock_file
from retrocast.paths import (
    DEFAULT_DATA_DIR,
    ENV_VAR_NAME,
    get_data_dir_source,
    get_paths,
    resolve_data_dir,
    validate_directory_name,
    validate_filename,
)
from retrocast.typing import InChIKeyStr
from retrocast.utils.logging import configure_script_logging
from retrocast.v2.adapters import ADAPTER_TYPES, DEPRECATED_ADAPTER_SLUGS, get_adapter, normalize_adapter_slug
from retrocast.v2.adapters.base import AdaptMode
from retrocast.v2.cli.manifest import manifest_sidecar_path, write_manifest
from retrocast.v2.cli.report import create_analysis_table, generate_markdown_report
from retrocast.v2.io import (
    load_benchmark,
    load_candidates,
    load_collected_candidates,
    load_evaluation,
    save_analysis_report,
    save_candidates,
    save_collected_candidates,
    save_evaluation,
)
from retrocast.v2.metrics.constraints import TaskConstraintChecker
from retrocast.v2.models import Candidate
from retrocast.v2.models.route import InChIKeyLevel
from retrocast.v2.models.task import Benchmark
from retrocast.v2.workflow import (
    adapt_candidates,
    analyze,
    collect_candidates,
    ingest_candidates,
    score,
)
from retrocast.v2.workflow.stats import candidate_statistics, collected_candidate_statistics, evaluation_statistics

logger = logging.getLogger(__name__)
console = Console()


def main() -> None:
    configure_script_logging(use_rich=True)
    parser = _build_parser()
    args = parser.parse_args()

    try:
        if args.command == "adapt":
            handle_adapt(args)
        elif args.command == "collect":
            handle_collect(args)
        elif args.command == "list-adapters":
            handle_list_adapters()
        else:
            config = _load_config(args.config)
            config_data_dir = config.get("data_dir")
            data_dir = resolve_data_dir(cli_arg=getattr(args, "data_dir", None), config_value=config_data_dir)
            config["data_dir"] = str(data_dir)
            config["_data_dir_source"] = get_data_dir_source(
                cli_arg=getattr(args, "data_dir", None), config_value=config_data_dir
            )
            if args.command == "config":
                handle_config(config)
            elif args.command == "ingest":
                handle_ingest(args, config)
            elif args.command == "score":
                handle_score(args, config)
            elif args.command == "analyze":
                handle_analyze(args, config)
    except RetroCastException as exc:
        logger.error("Command failed: %s", exc, exc_info=True)
        sys.exit(1)
    except Exception as exc:
        logger.error("Command failed: %s", exc, exc_info=True)
        sys.exit(1)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"Retrocast schema v2 CLI v{__version__}")
    parser.add_argument("--config", type=Path, default=Path("retrocast-config.yaml"), help="Path to config file")
    parser.add_argument("--data-dir", type=Path, help=f"Override data directory, or use {ENV_VAR_NAME}")
    parser.add_argument("--version", "-V", action="version", version=f"retrocast {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("config", help="Show resolved configuration and paths")
    subparsers.add_parser("list-adapters", help="List available schema v2 adapters")

    adapt = subparsers.add_parser("adapt", help="Adapt raw planner output into v2 candidates")
    adapt.add_argument("--input", required=True, type=Path)
    adapt.add_argument("--output", required=True, type=Path)
    adapt.add_argument("--adapter", required=True)
    adapt.add_argument("--mode", choices=["strict", "prune"], default="strict")

    collect = subparsers.add_parser("collect", help="Collect v2 candidates by benchmark target")
    collect.add_argument("--input", required=True, type=Path)
    collect.add_argument("--benchmark", required=True, type=Path)
    collect.add_argument("--output", required=True, type=Path)

    ingest = subparsers.add_parser("ingest", help="Project-mode v2 adapt plus collect")
    _add_model_dataset_args(ingest)
    ingest.add_argument("--adapter", help="Override adapter from raw manifest")
    ingest.add_argument("--mode", choices=["strict", "prune"], default="strict")
    ingest.add_argument("--no-progress", action="store_true", help="Disable progress bars during ingestion")

    score_parser = subparsers.add_parser("score", help="Score v2 processed candidates")
    _add_model_dataset_args(score_parser)
    score_parser.add_argument("--stock", help="Override stock name")
    score_parser.add_argument("--ignore-stereo", action="store_true", help="Use stereo-agnostic stock matching")

    analyze_parser = subparsers.add_parser("analyze", help="Analyze v2 evaluation artifacts")
    _add_model_dataset_args(analyze_parser)
    analyze_parser.add_argument("--stock", help="Specific stock to analyze")
    analyze_parser.add_argument("--top-k", nargs="+", type=int, default=[1, 3, 5, 10, 20, 50, 100])
    analyze_parser.add_argument("--n-boot", type=int, default=10000)

    return parser


def _add_model_dataset_args(parser: argparse.ArgumentParser) -> None:
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--model")
    model_group.add_argument("--all-models", action="store_true")
    dataset_group = parser.add_mutually_exclusive_group(required=True)
    dataset_group.add_argument("--dataset")
    dataset_group.add_argument("--all-datasets", action="store_true")


def _load_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        return {}
    with open(config_path, encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def handle_adapt(args: argparse.Namespace) -> None:
    adapter = get_adapter(args.adapter)
    raw_payload = load_json_artifact(args.input)
    mode: AdaptMode = args.mode
    root_dir = args.output.parent.resolve()
    candidates = adapt_candidates(raw_payload, adapter, mode=mode)
    save_candidates(candidates, args.output)
    stats = candidate_statistics(candidates).to_manifest_dict()

    write_manifest(
        manifest_sidecar_path(args.output),
        action="[cli:v2]adapt",
        sources=[args.input],
        outputs=[args.output],
        root_dir=root_dir,
        parameters={
            "adapter": normalize_adapter_slug(args.adapter),
            "mode": mode,
        },
        statistics=stats,
    )
    logger.info("Adapted v2 candidates to %s", args.output)


def handle_collect(args: argparse.Namespace) -> None:
    task = load_benchmark(args.benchmark)
    collected = collect_candidates(load_candidates(args.input), task)
    save_collected_candidates(collected, args.output)
    stats = collected_candidate_statistics(collected).to_manifest_dict()

    write_manifest(
        manifest_sidecar_path(args.output),
        action="[cli:v2]collect",
        sources=[args.input, args.benchmark],
        outputs=[args.output],
        root_dir=args.output.parent.resolve(),
        statistics=stats,
    )
    logger.info("Collected v2 candidates to %s", args.output)


def handle_ingest(args: argparse.Namespace, config: dict[str, Any]) -> None:
    paths = get_paths(Path(config.get("data_dir", DEFAULT_DATA_DIR)))
    models = _resolve_models(args, paths, stage="ingest")
    benchmarks = _resolve_benchmarks(args, paths)
    show_progress = not getattr(args, "no_progress", False)
    total_jobs = len(models) * len(benchmarks)

    if show_progress and total_jobs > 1:
        with create_cli_progress(console=console, unit="jobs") as progress, quiet_info_logs("retrocast"):
            task_id = progress.add_task("Ingesting model/dataset jobs", total=total_jobs)
            for model_name in models:
                for benchmark_name in benchmarks:
                    _ingest_one(model_name, benchmark_name, paths, args, show_route_progress=False)
                    progress.advance(task_id)
        return

    for model_name in models:
        for benchmark_name in benchmarks:
            _ingest_one(model_name, benchmark_name, paths, args, show_route_progress=show_progress)


def _ingest_one(
    model_name: str,
    benchmark_name: str,
    paths: dict[str, Path],
    args: argparse.Namespace,
    *,
    show_route_progress: bool,
) -> None:
    raw_dir = paths["raw"] / model_name / benchmark_name
    if not raw_dir.exists():
        logger.warning("Skipping %s/%s: raw directory missing", model_name, benchmark_name)
        return
    adapter_name = args.adapter or _manifest_directive(raw_dir / "manifest.json", "adapter")
    if adapter_name is None:
        logger.warning("Skipping %s/%s: no adapter in CLI or manifest", model_name, benchmark_name)
        return
    raw_filename = _manifest_directive(raw_dir / "manifest.json", "raw_results_filename") or "results.json.gz"
    raw_path = raw_dir / raw_filename
    if not raw_path.exists():
        logger.warning("Skipping %s/%s: raw file missing at %s", model_name, benchmark_name, raw_path)
        return

    task_path = paths["benchmarks"] / f"{benchmark_name}.json.gz"
    task = load_benchmark(task_path)
    raw_payload = load_json_artifact(raw_path)
    adapter = get_adapter(adapter_name)
    output_dir = paths["processed"] / benchmark_name / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_total = estimate_raw_route_entries(
        raw_payload,
        input_kind="target-keyed-provider-output",
        benchmark_targets=task.targets,
    )

    with _route_progress(
        enabled=show_route_progress,
        description=f"Ingesting {model_name}/{benchmark_name}",
        total=progress_total,
    ) as advance_progress:
        collected = ingest_candidates(
            raw_payload=raw_payload,
            adapter=adapter,
            task=task,
            mode=args.mode,
            progress_callback=advance_progress,
        )

    output_path = output_dir / "candidates.json.gz"
    save_collected_candidates(collected, output_path)
    stats = collected_candidate_statistics(collected).to_manifest_dict()

    write_manifest(
        output_dir / "manifest.json",
        action="ingest:v2",
        sources=[raw_path, task_path],
        outputs=[output_path],
        root_dir=paths["raw"].parent.resolve(),
        parameters={
            "model": model_name,
            "benchmark": benchmark_name,
            "adapter": normalize_adapter_slug(adapter_name),
            "mode": args.mode,
        },
        statistics=stats,
    )
    logger.info("Ingested %s/%s to %s", model_name, benchmark_name, output_path)


def handle_score(args: argparse.Namespace, config: dict[str, Any]) -> None:
    paths = get_paths(Path(config.get("data_dir", DEFAULT_DATA_DIR)))
    for model_name in _resolve_models(args, paths, stage="score"):
        for benchmark_name in _resolve_benchmarks(args, paths):
            _score_one(model_name, benchmark_name, paths, args)


def _score_one(model_name: str, benchmark_name: str, paths: dict[str, Path], args: argparse.Namespace) -> None:
    task_path = paths["benchmarks"] / f"{benchmark_name}.json.gz"
    candidates_path = paths["processed"] / benchmark_name / model_name / "candidates.json.gz"
    if not candidates_path.exists():
        logger.warning("Skipping %s/%s: no processed candidates", model_name, benchmark_name)
        return

    predictions = load_collected_candidates(candidates_path)
    task = load_benchmark(task_path)
    stock_registry, stock_paths, output_label = _load_stock_registry(task, paths, stock_override=args.stock)
    match_level = InChIKeyLevel.NO_STEREO if args.ignore_stereo else InChIKeyLevel.FULL
    evaluation = score(
        predictions=predictions,
        task=task,
        constraint_checker=TaskConstraintChecker(
            stocks=stock_registry,
            match_level=match_level,
        ),
        acceptable_match_level=match_level,
    )

    output_dir = paths["scored"] / benchmark_name / model_name / output_label
    output_path = output_dir / "evaluation.json.gz"
    save_evaluation(evaluation, output_path)
    sources = [task_path, candidates_path, *stock_paths]
    write_manifest(
        output_dir / "manifest.json",
        action="score:v2",
        sources=sources,
        outputs=[output_path],
        root_dir=paths["raw"].parent.resolve(),
        parameters={
            "model": model_name,
            "benchmark": benchmark_name,
            "stock": output_label,
            "ignore_stereo": args.ignore_stereo,
        },
        statistics=evaluation_statistics(evaluation),
    )
    logger.info("Scored %s/%s using %s to %s", model_name, benchmark_name, output_label, output_path)


def handle_analyze(args: argparse.Namespace, config: dict[str, Any]) -> None:
    paths = get_paths(Path(config.get("data_dir", DEFAULT_DATA_DIR)))
    for model_name in _resolve_models(args, paths, stage="analyze"):
        for benchmark_name in _resolve_benchmarks(args, paths):
            _analyze_one(model_name, benchmark_name, paths, args)


def _analyze_one(model_name: str, benchmark_name: str, paths: dict[str, Path], args: argparse.Namespace) -> None:
    scored_base = paths["scored"] / benchmark_name / model_name
    if not scored_base.exists():
        logger.warning("Skipping %s/%s: no scored directory", model_name, benchmark_name)
        return
    stock_names = [args.stock] if args.stock else [path.name for path in scored_base.iterdir() if path.is_dir()]
    for stock_name in stock_names:
        evaluation_path = scored_base / stock_name / "evaluation.json.gz"
        if not evaluation_path.exists():
            logger.warning("Skipping missing evaluation %s", evaluation_path)
            continue
        evaluation = load_evaluation(evaluation_path)
        report = analyze(evaluation, ks=args.top_k, n_boot=args.n_boot)
        output_dir = paths["results"] / benchmark_name / model_name / stock_name
        analysis_path = output_dir / "analysis.json.gz"
        markdown_path = output_dir / "report.md"
        save_analysis_report(report, analysis_path)
        markdown_path.write_text(
            generate_markdown_report(
                report, title=f"Evaluation Report: {model_name} / {benchmark_name} / {stock_name}"
            ),
            encoding="utf-8",
        )
        write_manifest(
            output_dir / "manifest.json",
            action="analyze:v2",
            sources=[evaluation_path],
            outputs=[analysis_path, markdown_path],
            root_dir=paths["raw"].parent.resolve(),
            parameters={
                "model": model_name,
                "benchmark": benchmark_name,
                "stock": stock_name,
                "top_k": args.top_k,
                "n_boot": args.n_boot,
            },
            statistics={"n_metrics": len(report.metrics), "n_strata": len(report.by_stratum)},
        )
        console.print(
            create_analysis_table(report, title=f"Analysis Results: {model_name} / {benchmark_name} / {stock_name}")
        )
        console.print(f"\n[dim]Full report saved to: {output_dir}[/]\n")


def handle_config(config: dict[str, Any]) -> None:
    data_dir = Path(config.get("data_dir", DEFAULT_DATA_DIR))
    paths = get_paths(data_dir)
    console.print()
    console.print("[bold]RetroCast Schema V2 Configuration[/bold]")
    console.print("=" * 40)
    console.print(f"\n[bold]Data directory:[/bold] {data_dir.resolve()}")
    console.print(f"  Source: {config.get('_data_dir_source', 'unknown')}")
    console.print("\n[bold]Resolved paths:[/bold]")
    max_key_len = max(len(name) for name in paths)
    for name, path in paths.items():
        exists_marker = "[green]exists[/green]" if path.exists() else "[dim]missing[/dim]"
        console.print(f"  {name:<{max_key_len}}: {path} ({exists_marker})")
    console.print()


def handle_list_adapters() -> None:
    console.print("Available schema v2 adapters:")
    for name in sorted(ADAPTER_TYPES):
        console.print(f"  - {name}")
    aliases = ", ".join(f"{alias} -> {canonical}" for alias, canonical in sorted(DEPRECATED_ADAPTER_SLUGS.items()))
    if aliases:
        console.print(f"Deprecated aliases: {aliases}")


def _resolve_models(args: argparse.Namespace, paths: dict[str, Path], *, stage: str) -> list[str]:
    if getattr(args, "model", None):
        return [validate_directory_name(args.model, param_name="model")]
    if stage == "ingest":
        base = paths["raw"]
    elif stage == "score":
        base = paths["processed"]
    else:
        base = paths["scored"]
    models = set()
    if base.exists():
        for benchmark_dir in base.iterdir():
            if not benchmark_dir.is_dir():
                continue
            if stage == "ingest":
                models.add(benchmark_dir.name)
            else:
                for model_dir in benchmark_dir.iterdir():
                    if model_dir.is_dir():
                        models.add(model_dir.name)
    return sorted(validate_directory_name(model, param_name="model") for model in models)


def _resolve_benchmarks(args: argparse.Namespace, paths: dict[str, Path]) -> list[str]:
    if getattr(args, "dataset", None):
        return [validate_filename(args.dataset, param_name="dataset")]
    if not paths["benchmarks"].exists():
        return []
    return sorted(
        validate_filename(path.name.removesuffix(".json.gz"), param_name="dataset")
        for path in paths["benchmarks"].glob("*.json.gz")
    )


def _manifest_directive(path: Path, key: str) -> str | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    value = payload.get("directives", {}).get(key)
    return str(value) if value is not None else None


def _load_stock_registry(
    task: Benchmark,
    paths: dict[str, Path],
    *,
    stock_override: str | None,
) -> tuple[dict[str, set[InChIKeyStr]], list[Path], str]:
    stock_names = _effective_stock_names(task)
    if stock_override is not None:
        stock_path = paths["stocks"] / f"{stock_override}.csv.gz"
        stock = set(load_stock_file(stock_path, return_as="inchikey"))
        registry_names = stock_names or {stock_override}
        return {stock_name: stock for stock_name in registry_names}, [stock_path], stock_override

    stock_paths = []
    stock_registry = {}
    for stock_name in sorted(stock_names):
        stock_path = paths["stocks"] / f"{stock_name}.csv.gz"
        stock_registry[stock_name] = set(load_stock_file(stock_path, return_as="inchikey"))
        stock_paths.append(stock_path)

    return stock_registry, stock_paths, task.derived_metric_label()


def _effective_stock_names(task: Benchmark) -> set[str]:
    stock_names = set()
    for target_id in task.targets:
        stock = task.effective_constraints(target_id).stock
        if stock is not None:
            stock_names.add(stock)
    return stock_names


@contextmanager
def _route_progress(
    *,
    enabled: bool,
    description: str,
    total: int | None,
) -> Iterator[Callable[[], None] | None]:
    if not enabled:
        yield None
        return

    with create_cli_progress(console=console, unit="routes") as progress, quiet_info_logs("retrocast"):
        task_id = progress.add_task(description, total=total)
        yield lambda: progress.advance(task_id)


def _flatten_candidates(candidates_by_target: dict[str, list[Candidate]]) -> list[Candidate]:
    return [candidate for candidates in candidates_by_target.values() for candidate in candidates]


if __name__ == "__main__":
    main()
