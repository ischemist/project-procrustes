import time
from collections.abc import Generator
from contextlib import contextmanager

from retrocast.models.benchmark import ExecutionStats


class ExecutionTimer:
    """
    accumulates wall/cpu time for targets.
    usage:
        timer = ExecutionTimer()
        with timer.measure("target_1"):
             model.predict(...)
        stats = timer.to_model()
    """

    def __init__(self) -> None:
        self.wall_times: dict[str, float] = {}
        self.cpu_times: dict[str, float] = {}

    @contextmanager
    def measure(self, target_id: str) -> Generator[None, None, None]:
        t_wall_start = time.perf_counter()
        t_cpu_start = time.process_time()
        try:
            yield
        finally:
            t_wall_end = time.perf_counter()
            t_cpu_end = time.process_time()
            self.wall_times[target_id] = t_wall_end - t_wall_start
            self.cpu_times[target_id] = t_cpu_end - t_cpu_start

    def to_model(self) -> ExecutionStats:
        return ExecutionStats(wall_time=self.wall_times, cpu_time=self.cpu_times)
