from __future__ import annotations

import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ExecutionStats:
    wall_time: dict[str, float] = field(default_factory=dict)
    cpu_time: dict[str, float] = field(default_factory=dict)


class ExecutionTimer:
    def __init__(self) -> None:
        self.wall_times: dict[str, float] = {}
        self.cpu_times: dict[str, float] = {}

    @contextmanager
    def measure(self, target_id: str) -> Generator[None, None, None]:
        wall_start = time.perf_counter()
        cpu_start = time.thread_time()
        try:
            yield
        finally:
            self.wall_times[target_id] = time.perf_counter() - wall_start
            self.cpu_times[target_id] = time.thread_time() - cpu_start

    def to_model(self) -> ExecutionStats:
        return ExecutionStats(wall_time=dict(self.wall_times), cpu_time=dict(self.cpu_times))
