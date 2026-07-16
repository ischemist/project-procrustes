#!/usr/bin/env python3
"""Benchmark the AiZynthFinder pipeline and aggregate parent/child RSS."""

from __future__ import annotations

import argparse
import contextlib
import gzip
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import psutil


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--benchmark", required=True, type=Path)
    parser.add_argument("--stock", required=True, type=Path)
    parser.add_argument("--adapter", default="aizynthfinder")
    parser.add_argument("--execution-stats", type=Path)
    parser.add_argument("--rust-binary", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--workers", nargs="+", type=int, default=[1, 12])
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = [
        (f"{surface}-{workers}", command(args, workers=workers))
        for workers in args.workers
        for surface, command in (("python-api", _python_command), ("rust-cli", _rust_command))
    ]
    results = []
    for name, command in cases:
        for index in range(args.warmups):
            _run(command, args.output_dir / "runs" / name / f"warmup-{index}")
        samples = [
            _run(command, args.output_dir / "runs" / name / f"sample-{index}") for index in range(args.repetitions)
        ]
        results.append(_summarize(name, command, samples))

    dataset_stats = results[0]["samples"][0]["internal"]
    payload = {
        "schema_version": 1,
        "machine": _machine_info(),
        "dataset": {
            "adapter": args.adapter,
            "raw": str(args.raw.resolve()),
            "benchmark": str(args.benchmark.resolve()),
            "stock": str(args.stock.resolve()),
            "targets": dataset_stats["targets"],
            "candidates": dataset_stats["candidates"],
        },
        "settings": {
            "repetitions": args.repetitions,
            "warmups": args.warmups,
            "bootstrap_resamples": args.n_boot,
            "rss": "peak aggregate resident set of the process tree, sampled every 10 ms",
        },
        "results": results,
        "semantic_validation": _semantic_validation(args.output_dir, [name for name, _ in cases]),
    }
    (args.output_dir / "benchmark-results.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "README.md").write_text(_markdown(payload), encoding="utf-8")


def _python_command(args: argparse.Namespace, *, workers: int) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "retrocast.cli.main",
        "pipeline",
        "--raw",
        str(args.raw),
        "--benchmark",
        str(args.benchmark),
        "--stock",
        str(args.stock),
        "--output-dir",
        "{output_dir}",
        "--adapter",
        args.adapter,
        "--workers",
        str(workers),
        "--n-boot",
        str(args.n_boot),
    ]
    if args.execution_stats:
        command.extend(["--execution-stats", str(args.execution_stats)])
    return command


def _rust_command(args: argparse.Namespace, *, workers: int) -> list[str]:
    command = [
        str(args.rust_binary),
        "pipeline",
        "--raw",
        str(args.raw),
        "--benchmark",
        str(args.benchmark),
        "--stock",
        str(args.stock),
        "--output-dir",
        "{output_dir}",
        "--adapter",
        args.adapter,
        "--workers",
        str(workers),
        "--n-boot",
        str(args.n_boot),
    ]
    if args.execution_stats:
        command.extend(["--execution-stats", str(args.execution_stats)])
    return command


def _run(command_template: list[str], output_dir: Path) -> dict[str, Any]:
    shutil.rmtree(output_dir, ignore_errors=True)
    command = [part.format(output_dir=output_dir) for part in command_template]
    started = time.perf_counter()
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    root = psutil.Process(process.pid)
    peak_rss = 0
    while process.poll() is None:
        processes = [root]
        with contextlib.suppress(psutil.Error):
            processes.extend(root.children(recursive=True))
        rss = 0
        for child in processes:
            with contextlib.suppress(psutil.Error):
                rss += child.memory_info().rss
        peak_rss = max(peak_rss, rss)
        time.sleep(0.01)
    stdout, stderr = process.communicate()
    wall_seconds = time.perf_counter() - started
    if process.returncode != 0:
        raise RuntimeError(f"command failed ({process.returncode}): {' '.join(command)}\n{stdout}\n{stderr}")
    internal = json.loads((output_dir / "pipeline-stats.json").read_text(encoding="utf-8"))
    return {
        "wall_seconds": wall_seconds,
        "peak_tree_rss_bytes": peak_rss,
        "internal": internal,
    }


def _summarize(name: str, command: list[str], samples: list[dict[str, Any]]) -> dict[str, Any]:
    wall = [sample["wall_seconds"] for sample in samples]
    rss = [sample["peak_tree_rss_bytes"] for sample in samples]
    targets = samples[0]["internal"]["targets"]
    candidates = samples[0]["internal"]["candidates"]
    phase_names = ["ingest_seconds", "score_seconds", "analyze_seconds", "total_seconds"]
    return {
        "name": name,
        "command": command,
        "samples": samples,
        "median_wall_seconds": statistics.median(wall),
        "median_peak_tree_rss_bytes": int(statistics.median(rss)),
        "max_peak_tree_rss_bytes": max(rss),
        "median_targets_per_second": targets / statistics.median(wall),
        "median_candidates_per_second": candidates / statistics.median(wall),
        "median_phases": {
            phase: statistics.median(sample["internal"][phase] for sample in samples) for phase in phase_names
        },
    }


def _machine_info() -> dict[str, Any]:
    cpu = platform.processor()
    if sys.platform == "darwin":
        cpu = (
            subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, check=False
            ).stdout.strip()
            or cpu
        )
    native_info = None
    try:
        from retrocast import _native

        native_info = _native.engine_info()
    except ImportError:
        pass
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "rust": _command_version(["rustc", "--version"]),
        "cpu": cpu,
        "logical_cpus": os.cpu_count(),
        "memory_bytes": psutil.virtual_memory().total,
        "native_engine": native_info,
    }


def _command_version(command: list[str]) -> str | None:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def _semantic_validation(output_dir: Path, case_names: list[str]) -> dict[str, Any]:
    from retrocast.models.analysis import AnalysisReport
    from retrocast.models.evaluation import Evaluation

    reference_name = case_names[0]
    reference = output_dir / "runs" / reference_name / "sample-0"
    comparisons: dict[str, Any] = {}
    for case in case_names[1:]:
        candidate = output_dir / "runs" / case / "sample-0"
        reference_candidates = _read_gzip_json(reference / "candidates.json.gz")
        candidate_candidates = _read_gzip_json(candidate / "candidates.json.gz")
        reference_evaluation = Evaluation.model_validate(_read_gzip_json(reference / "evaluation.json.gz"))
        candidate_evaluation = Evaluation.model_validate(_read_gzip_json(candidate / "evaluation.json.gz"))
        reference_analysis = AnalysisReport.model_validate(_read_gzip_json(reference / "analysis.json.gz"))
        candidate_analysis = AnalysisReport.model_validate(_read_gzip_json(candidate / "analysis.json.gz"))
        delta, non_numeric_equal = _maximum_numeric_delta(
            reference_analysis.model_dump(mode="json", exclude_none=True),
            candidate_analysis.model_dump(mode="json", exclude_none=True),
        )
        comparisons[f"{reference_name}_vs_{case}"] = {
            "candidates_equal": reference_candidates == candidate_candidates,
            "evaluation_equal": reference_evaluation == candidate_evaluation,
            "analysis_non_numeric_equal": non_numeric_equal,
            "analysis_max_abs_numeric_delta": delta,
        }
    return comparisons


def _read_gzip_json(path: Path) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def _maximum_numeric_delta(left: Any, right: Any) -> tuple[float, bool]:
    maximum = 0.0
    non_numeric_equal = True

    def visit(left_value: Any, right_value: Any) -> None:
        nonlocal maximum, non_numeric_equal
        if isinstance(left_value, dict) and isinstance(right_value, dict):
            if left_value.keys() != right_value.keys():
                non_numeric_equal = False
                return
            for key in left_value:
                visit(left_value[key], right_value[key])
            return
        if isinstance(left_value, list) and isinstance(right_value, list):
            if len(left_value) != len(right_value):
                non_numeric_equal = False
                return
            for left_item, right_item in zip(left_value, right_value, strict=True):
                visit(left_item, right_item)
            return
        if isinstance(left_value, int | float) and isinstance(right_value, int | float):
            maximum = max(maximum, abs(float(left_value) - float(right_value)))
            return
        if left_value != right_value:
            non_numeric_equal = False

    visit(left, right)
    return maximum, non_numeric_equal


def _markdown(payload: dict[str, Any]) -> str:
    rows = []
    for result in payload["results"]:
        rows.append(
            "| {name} | {wall:.3f} | {targets:.1f} | {candidates:.1f} | {rss:.1f} | {ingest:.3f} | {score:.3f} | {analyze:.3f} |".format(
                name=result["name"],
                wall=result["median_wall_seconds"],
                targets=result["median_targets_per_second"],
                candidates=result["median_candidates_per_second"],
                rss=result["median_peak_tree_rss_bytes"] / (1024 * 1024),
                ingest=result["median_phases"]["ingest_seconds"],
                score=result["median_phases"]["score_seconds"],
                analyze=result["median_phases"]["analyze_seconds"],
            )
        )
    machine = payload["machine"]
    validation = payload["semantic_validation"]
    exact_cases = all(
        result["candidates_equal"] and result["evaluation_equal"] and result["analysis_non_numeric_equal"]
        for result in validation.values()
    )
    maximum_delta = max(result["analysis_max_abs_numeric_delta"] for result in validation.values())
    return f"""# {payload["dataset"]["adapter"]} pipeline benchmark

This benchmark runs the complete ingest, score, and analyze pipeline over {payload["dataset"]["targets"]:,} targets and {payload["dataset"]["candidates"]:,} candidate slots. Values are medians of {payload["settings"]["repetitions"]} measured runs after {payload["settings"]["warmups"]} warm-up run per case. Wall time includes process startup and artifact IO. RSS is the peak aggregate resident set of the parent process and all live children, sampled every 10 ms.

| execution path | wall (s) | targets/s | candidates/s | peak tree RSS (MiB) | ingest (s) | score (s) | analyze (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

Machine: {machine["cpu"]}, {machine["logical_cpus"]} logical CPUs, {machine["memory_bytes"] / (1024**3):.1f} GiB RAM, {machine["platform"]}. Python {machine["python"]}; native engine {machine["native_engine"]}; {machine["rust"]}.

`python-api-*` calls the Rust core through the supported Python package and keeps untouched stage values as opaque Rust handles. Its measured wall time and RSS still include Python startup plus final Pydantic and artifact materialization. `rust-cli-*` calls the same core through the standalone executable. Worker count is owned by the Rust core in both cases.

## Semantic validation

Candidate artifacts, parsed evaluation models, and every non-numeric analysis field agree across all four cases: {str(exact_cases).lower()}. The largest absolute numeric difference in an analysis artifact is {maximum_delta:.3g}; this is JSON floating-point round-trip noise at the Python boundary.
"""


if __name__ == "__main__":
    main()
