"""
Tests for retrocast.cli.compare.

Philosophy: test the logic that can silently misbehave.
- _resolve_statistics_path: pure path arithmetic, no I/O — wrong result means a
  model is silently skipped, which is hard to catch otherwise.
- _load_sources: integration over the loading loop using real statistics.json.gz
  files in tmp_path. Stops before plotly is touched.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from retrocast.cli.compare import _load_sources, _resolve_statistics_path
from retrocast.io.blob import save_json_gz

# =============================================================================
# _resolve_statistics_path
# =============================================================================


@pytest.mark.unit
class TestResolveStatisticsPath:
    """Pure path arithmetic — no I/O, no fixtures."""

    def test_explicit_absolute_statistics_path(self):
        """Explicit absolute 'statistics' key is returned as-is."""
        entry = {"name": "my-model", "statistics": "/data/results/statistics.json.gz"}
        result = _resolve_statistics_path(entry, source_root=None, benchmark="b", stock="s", yaml_dir=Path("/cfg"))
        assert result == Path("/data/results/statistics.json.gz")

    def test_explicit_relative_statistics_path(self):
        """Explicit relative 'statistics' key is resolved against yaml_dir."""
        entry = {"name": "my-model", "statistics": "../results/statistics.json.gz"}
        yaml_dir = Path("/project/comparisons")
        result = _resolve_statistics_path(entry, source_root=None, benchmark="b", stock="s", yaml_dir=yaml_dir)
        assert result == Path("/project/results/statistics.json.gz")

    def test_shorthand_absolute_root(self):
        """Shorthand with absolute root builds the expected 5-results path."""
        entry = {"name": "dms-explorer-xl"}
        source_root = Path("/data/retrocast")
        result = _resolve_statistics_path(
            entry, source_root, benchmark="mkt-cnv-160", stock="buyables-stock", yaml_dir=Path("/cfg")
        )
        assert result == Path("/data/retrocast/5-results/mkt-cnv-160/dms-explorer-xl/buyables-stock/statistics.json.gz")

    def test_shorthand_relative_root_resolved_against_yaml_dir(self, tmp_path):
        """Relative root is resolved against yaml_dir before building the path."""
        entry = {"name": "model-x"}
        yaml_dir = tmp_path / "comparisons"
        # source_root already resolved by _load_sources before calling this function
        source_root = (yaml_dir / "../data/retrocast").resolve()
        result = _resolve_statistics_path(entry, source_root, benchmark="pharma", stock="n5-stock", yaml_dir=yaml_dir)
        expected = source_root / "5-results" / "pharma" / "model-x" / "n5-stock" / "statistics.json.gz"
        assert result == expected

    def test_no_root_no_statistics_raises_value_error(self):
        """Missing both 'statistics' and source root raises ValueError mentioning the model name."""
        entry = {"name": "orphan-model"}
        with pytest.raises(ValueError, match="orphan-model"):
            _resolve_statistics_path(entry, source_root=None, benchmark="b", stock="s", yaml_dir=Path("/cfg"))


# =============================================================================
# _load_sources
# =============================================================================


def _write_stats(path: Path, stats) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_json_gz(stats, path)


def _make_cfg(tmp_path: Path, sources: list[dict], benchmark: str = "pharma", stock: str = "n5-stock") -> dict:
    return {"benchmark": benchmark, "stock": stock, "sources": sources}


@pytest.mark.integration
class TestLoadSources:
    """Integration tests for _load_sources using real statistics.json.gz files."""

    def test_single_model_shorthand_root(self, tmp_path, synthetic_statistics_factory):
        """Shorthand root: one model is loaded with correct hourly_cost and model_config."""
        stats = synthetic_statistics_factory("model-a", "pharma", "n5-stock")
        _write_stats(
            tmp_path / "5-results" / "pharma" / "model-a" / "n5-stock" / "statistics.json.gz",
            stats,
        )

        cfg = _make_cfg(
            tmp_path,
            sources=[
                {
                    "root": str(tmp_path),
                    "models": [
                        {"name": "model-a", "hourly_cost": 1.29, "legend": "Model A", "short": "A", "color": "#abc"}
                    ],
                }
            ],
        )

        stats_list, hourly_costs, model_config, model_groups = _load_sources(cfg, yaml_dir=tmp_path)

        assert len(stats_list) == 1
        assert stats_list[0].model_name == "model-a"
        assert hourly_costs == {"model-a": 1.29}
        assert model_config["model-a"]["legend"] == "Model A"
        assert model_config["model-a"]["short"] == "A"
        assert model_config["model-a"]["color"] == "#abc"
        assert model_groups == {}

    def test_multiple_models_across_two_sources(self, tmp_path, synthetic_statistics_factory):
        """Models spread across two source blocks are all loaded."""
        for name in ["model-a", "model-b"]:
            stats = synthetic_statistics_factory(name, "pharma", "n5-stock")
            _write_stats(tmp_path / "5-results" / "pharma" / name / "n5-stock" / "statistics.json.gz", stats)

        cfg = _make_cfg(
            tmp_path,
            sources=[
                {"root": str(tmp_path), "models": [{"name": "model-a"}]},
                {"root": str(tmp_path), "models": [{"name": "model-b"}]},
            ],
        )

        stats_list, _, _, _ = _load_sources(cfg, yaml_dir=tmp_path)

        assert {s.model_name for s in stats_list} == {"model-a", "model-b"}

    def test_explicit_statistics_path(self, tmp_path, synthetic_statistics_factory):
        """Explicit 'statistics' key loads from the given path regardless of root."""
        stats_file = tmp_path / "custom" / "location" / "statistics.json.gz"
        stats = synthetic_statistics_factory("my-model", "pharma", "n5-stock")
        _write_stats(stats_file, stats)

        cfg = _make_cfg(
            tmp_path,
            sources=[
                {
                    "models": [{"name": "my-model", "statistics": str(stats_file)}],
                }
            ],
        )

        stats_list, _, _, _ = _load_sources(cfg, yaml_dir=tmp_path)

        assert len(stats_list) == 1
        assert stats_list[0].model_name == "my-model"

    def test_missing_file_is_skipped_not_an_error(self, tmp_path, synthetic_statistics_factory):
        """A missing statistics file is skipped; the other models still load."""
        stats = synthetic_statistics_factory("model-good", "pharma", "n5-stock")
        _write_stats(tmp_path / "5-results" / "pharma" / "model-good" / "n5-stock" / "statistics.json.gz", stats)

        cfg = _make_cfg(
            tmp_path,
            sources=[
                {
                    "root": str(tmp_path),
                    "models": [{"name": "model-missing"}, {"name": "model-good"}],
                }
            ],
        )

        stats_list, _, _, _ = _load_sources(cfg, yaml_dir=tmp_path)

        assert len(stats_list) == 1
        assert stats_list[0].model_name == "model-good"

    def test_group_key_populates_model_groups(self, tmp_path, synthetic_statistics_factory):
        """A model entry with 'group' key appears in the returned model_groups dict."""
        stats = synthetic_statistics_factory("model-a", "pharma", "n5-stock")
        _write_stats(tmp_path / "5-results" / "pharma" / "model-a" / "n5-stock" / "statistics.json.gz", stats)

        cfg = _make_cfg(
            tmp_path,
            sources=[
                {
                    "root": str(tmp_path),
                    "models": [{"name": "model-a", "group": "my-group"}],
                }
            ],
        )

        _, _, _, model_groups = _load_sources(cfg, yaml_dir=tmp_path)

        assert model_groups == {"model-a": "my-group"}

    def test_model_config_defaults_when_keys_absent(self, tmp_path, synthetic_statistics_factory):
        """legend, short, and color fall back to sensible defaults when not specified."""
        stats = synthetic_statistics_factory("model-a", "pharma", "n5-stock")
        _write_stats(tmp_path / "5-results" / "pharma" / "model-a" / "n5-stock" / "statistics.json.gz", stats)

        cfg = _make_cfg(
            tmp_path,
            sources=[
                {
                    "root": str(tmp_path),
                    "models": [{"name": "model-a"}],
                }
            ],
        )

        _, _, model_config, _ = _load_sources(cfg, yaml_dir=tmp_path)

        assert model_config["model-a"]["legend"] == "model-a"
        assert model_config["model-a"]["short"] == "model-a"
        assert model_config["model-a"]["color"] == "#888888"
