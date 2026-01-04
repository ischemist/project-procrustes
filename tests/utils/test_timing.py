from __future__ import annotations

import time

import pytest

from retrocast.models.benchmark import ExecutionStats
from retrocast.utils.timing import ExecutionTimer


@pytest.mark.unit
class TestExecutionTimer:
    def test_single_measurement(self):
        """Basic smoke test: measure one block."""
        timer = ExecutionTimer()

        with timer.measure("task_1"):
            time.sleep(0.01)

        assert "task_1" in timer.wall_times
        assert "task_1" in timer.cpu_times
        assert timer.wall_times["task_1"] > 0
        assert timer.cpu_times["task_1"] >= 0

    def test_multiple_targets(self):
        """Core use case: timing different pipeline stages."""
        timer = ExecutionTimer()

        with timer.measure("load_data"):
            time.sleep(0.01)

        with timer.measure("predict"):
            time.sleep(0.01)

        with timer.measure("score"):
            time.sleep(0.01)

        assert len(timer.wall_times) == 3
        assert len(timer.cpu_times) == 3
        assert "load_data" in timer.wall_times
        assert "predict" in timer.wall_times
        assert "score" in timer.wall_times
        assert all(t > 0 for t in timer.wall_times.values())
        assert all(t >= 0 for t in timer.cpu_times.values())

    def test_overwrite_behavior(self):
        """Measuring same target twice overwrites previous time."""
        timer = ExecutionTimer()

        # First measurement
        with timer.measure("target_1"):
            time.sleep(0.01)
        first_time = timer.wall_times["target_1"]

        # Second measurement (shorter)
        with timer.measure("target_1"):
            time.sleep(0.005)
        second_time = timer.wall_times["target_1"]

        # Should have exactly one entry, and second should be different (shorter)
        assert len(timer.wall_times) == 1
        assert second_time != first_time
        assert second_time < first_time

    def test_nested_measurements(self):
        """Nested contexts record independently."""
        timer = ExecutionTimer()

        with timer.measure("outer"):
            time.sleep(0.01)
            with timer.measure("inner"):
                time.sleep(0.01)
            time.sleep(0.01)

        assert len(timer.wall_times) == 2
        assert "outer" in timer.wall_times
        assert "inner" in timer.wall_times
        # Outer should be >= inner (outer includes inner plus extra sleep)
        assert timer.wall_times["outer"] >= timer.wall_times["inner"]

    def test_exception_safety(self):
        """Time recorded even if block raises exception."""
        timer = ExecutionTimer()

        with pytest.raises(ValueError, match="test error"), timer.measure("failing_task"):
            time.sleep(0.01)
            raise ValueError("test error")

        # Time should still be recorded despite exception
        assert "failing_task" in timer.wall_times
        assert "failing_task" in timer.cpu_times
        assert timer.wall_times["failing_task"] > 0

    def test_empty_timer_to_model(self):
        """Empty timer produces empty ExecutionStats."""
        timer = ExecutionTimer()
        stats = timer.to_model()

        assert isinstance(stats, ExecutionStats)
        assert stats.wall_time == {}
        assert stats.cpu_time == {}

    def test_to_model_contract(self):
        """to_model() creates valid ExecutionStats with correct fields."""
        timer = ExecutionTimer()

        with timer.measure("task_a"):
            time.sleep(0.01)

        with timer.measure("task_b"):
            time.sleep(0.01)

        stats = timer.to_model()

        assert isinstance(stats, ExecutionStats)
        assert stats.wall_time == timer.wall_times
        assert stats.cpu_time == timer.cpu_times
        assert "task_a" in stats.wall_time
        assert "task_b" in stats.wall_time
        assert "task_a" in stats.cpu_time
        assert "task_b" in stats.cpu_time

    def test_wall_vs_cpu_time(self):
        """Wall time >= CPU time for sleep-dominated workload."""
        timer = ExecutionTimer()

        with timer.measure("sleep_task"):
            time.sleep(0.02)

        # Sleep increases wall time but not CPU time
        # Wall should be significantly greater than CPU
        wall = timer.wall_times["sleep_task"]
        cpu = timer.cpu_times["sleep_task"]

        assert wall >= cpu
        # For sleep-dominated work, wall should be much greater
        # Using a conservative threshold to avoid flakiness
        assert wall > 0.015  # Should be ~20ms
        # CPU time should be minimal for pure sleep
        assert cpu < 0.01  # Should be < 10ms
