from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
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


def create_cli_progress(*, console: Console, unit: str) -> Progress:
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
    )


def estimate_raw_route_entries(
    raw_data: Any,
    *,
    input_kind: str,
    benchmark_targets: Mapping[str, Any] | None = None,
) -> int | None:
    if input_kind in {"provider_output", "provider-output"}:
        return _route_collection_len(raw_data)

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
        total += payload_count
    return total


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
