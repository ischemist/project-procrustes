from __future__ import annotations

import logging
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager
from typing import Any

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    ProgressColumn,
    Task,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.text import Text


class RateColumn(ProgressColumn):
    """Render task throughput using a domain-specific unit."""

    def __init__(self, unit: str) -> None:
        super().__init__()
        self._unit = unit

    def render(self, task: Task) -> Text:
        if task.speed is None:
            return Text(f"-- {self._unit}/s", style="progress.data.speed")
        return Text(f"{task.speed:.1f} {self._unit}/s", style="progress.data.speed")


def create_cli_progress(
    *,
    console: Console,
    unit: str,
    transient: bool = False,
) -> Progress:
    return Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("elapsed"),
        TimeElapsedColumn(),
        TextColumn("eta"),
        TimeRemainingColumn(),
        RateColumn(unit),
        console=console,
        transient=transient,
    )


@contextmanager
def step_progress(
    *,
    console: Console,
    total: int,
    unit: str = "steps",
    transient: bool = False,
) -> Iterator[Callable[[str], AbstractContextManager[None]]]:
    with create_cli_progress(console=console, unit=unit, transient=transient) as progress:
        task_id = progress.add_task("starting", total=total)

        @contextmanager
        def step(description: str) -> Iterator[None]:
            progress.update(task_id, description=description)
            yield
            progress.advance(task_id)

        yield step


def estimate_raw_route_entries(
    raw_data: Any,
    *,
    input_kind: str,
    benchmark_targets: Mapping[str, Any] | None = None,
    max_entries_per_target: int | None = None,
) -> int | None:
    if input_kind in {"provider_output", "provider-output"}:
        return _capped_len(_route_collection_len(raw_data), max_entries_per_target)

    if not isinstance(raw_data, Mapping) or benchmark_targets is None:
        return None

    total = 0
    for target_id, target in benchmark_targets.items():
        if target_id in raw_data:
            payload = raw_data[target_id]
        elif target.smiles in raw_data:
            payload = raw_data[target.smiles]
        else:
            continue

        payload_count = _route_collection_len(payload)
        if payload_count is None:
            return None
        capped_count = _capped_len(payload_count, max_entries_per_target)
        if capped_count is None:
            return None
        total += capped_count
    return total


def _capped_len(value: int | None, limit: int | None) -> int | None:
    if value is None:
        return None
    if limit is None:
        return value
    return min(value, limit)


def _route_collection_len(payload: Any) -> int | None:
    if hasattr(payload, "routes"):
        routes = payload.routes
        return len(routes) if isinstance(routes, Sequence) and not isinstance(routes, str | bytes | bytearray) else None

    if isinstance(payload, Sequence) and not isinstance(payload, str | bytes | bytearray):
        return len(payload)

    return None


@contextmanager
def quiet_info_logs(*logger_names: str) -> Iterator[None]:
    """Hide normal info logs while a live progress bar owns the terminal."""

    original_levels: dict[str, int] = {}
    for logger_name in logger_names:
        target_logger = logging.getLogger(logger_name)
        original_levels[logger_name] = target_logger.level
        target_logger.setLevel(logging.WARNING)
    try:
        yield
    finally:
        for logger_name, level in original_levels.items():
            logging.getLogger(logger_name).setLevel(level)
