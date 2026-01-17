"""Tests for retrocast.paths module."""

from __future__ import annotations

from pathlib import Path

import pytest

from retrocast.paths import (
    DEFAULT_DATA_DIR,
    ENV_VAR_NAME,
    check_migration_needed,
    get_data_dir_source,
    get_paths,
    resolve_data_dir,
)


@pytest.mark.unit
class TestResolveDataDir:
    """Tests for resolve_data_dir function."""

    def test_cli_arg_takes_highest_priority(self, monkeypatch):
        """CLI argument should override env var and config."""
        monkeypatch.setenv(ENV_VAR_NAME, "/env/path")

        result = resolve_data_dir(cli_arg="/cli/path", config_value="/config/path")

        assert result == Path("/cli/path")

    def test_env_var_takes_second_priority(self, monkeypatch):
        """Env var should override config when no CLI arg provided."""
        monkeypatch.setenv(ENV_VAR_NAME, "/env/path")

        result = resolve_data_dir(cli_arg=None, config_value="/config/path")

        assert result == Path("/env/path")

    def test_config_value_takes_third_priority(self, monkeypatch):
        """Config value should be used when no CLI arg or env var."""
        monkeypatch.delenv(ENV_VAR_NAME, raising=False)

        result = resolve_data_dir(cli_arg=None, config_value="/config/path")

        assert result == Path("/config/path")

    def test_default_used_when_no_overrides(self, monkeypatch):
        """Default should be used when no CLI, env, or config."""
        monkeypatch.delenv(ENV_VAR_NAME, raising=False)

        result = resolve_data_dir(cli_arg=None, config_value=None)

        assert result == DEFAULT_DATA_DIR

    def test_default_is_data_retrocast(self):
        """Verify default is data/retrocast."""
        assert Path("data/retrocast") == DEFAULT_DATA_DIR

    def test_accepts_path_objects(self, monkeypatch):
        """Should accept Path objects, not just strings."""
        monkeypatch.delenv(ENV_VAR_NAME, raising=False)

        result = resolve_data_dir(cli_arg=Path("/cli/path"))

        assert result == Path("/cli/path")

    def test_env_var_empty_string_is_falsy(self, monkeypatch):
        """Empty env var should fall through to config/default."""
        monkeypatch.setenv(ENV_VAR_NAME, "")

        result = resolve_data_dir(cli_arg=None, config_value="/config/path")

        # Empty string is falsy, so falls through to config
        assert result == Path("/config/path")


@pytest.mark.unit
class TestGetPaths:
    """Tests for get_paths function."""

    def test_returns_expected_keys(self):
        """Should return all expected path keys."""
        result = get_paths(Path("/data"))

        expected_keys = {"benchmarks", "stocks", "raw", "processed", "scored", "results"}
        assert set(result.keys()) == expected_keys

    def test_paths_relative_to_data_dir(self):
        """All paths should be relative to provided data directory."""
        data_dir = Path("/custom/data/dir")
        result = get_paths(data_dir)

        assert result["benchmarks"] == data_dir / "1-benchmarks" / "definitions"
        assert result["stocks"] == data_dir / "1-benchmarks" / "stocks"
        assert result["raw"] == data_dir / "2-raw"
        assert result["processed"] == data_dir / "3-processed"
        assert result["scored"] == data_dir / "4-scored"
        assert result["results"] == data_dir / "5-results"

    def test_works_with_relative_path(self):
        """Should work with relative paths."""
        result = get_paths(Path("relative/path"))

        assert result["raw"] == Path("relative/path/2-raw")


@pytest.mark.unit
class TestCheckMigrationNeeded:
    """Tests for check_migration_needed function."""

    def test_no_warning_when_not_using_default(self, tmp_path):
        """Should not warn when using custom data directory."""
        # Create legacy data
        legacy = tmp_path / "data" / "1-benchmarks"
        legacy.mkdir(parents=True)

        # Using custom path, not default
        custom_path = tmp_path / "custom"

        result = check_migration_needed(custom_path)

        assert result is None

    def test_no_warning_when_legacy_doesnt_exist(self, tmp_path, monkeypatch):
        """Should not warn when no data at legacy location."""
        # Change to tmp_path so relative paths resolve correctly
        monkeypatch.chdir(tmp_path)

        result = check_migration_needed(DEFAULT_DATA_DIR)

        assert result is None

    def test_no_warning_when_new_location_exists(self, tmp_path, monkeypatch):
        """Should not warn when data exists at new location."""
        monkeypatch.chdir(tmp_path)

        # Create both legacy and new
        (tmp_path / "data" / "1-benchmarks").mkdir(parents=True)
        (tmp_path / "data" / "retrocast" / "1-benchmarks").mkdir(parents=True)

        result = check_migration_needed(DEFAULT_DATA_DIR)

        assert result is None

    def test_warning_when_legacy_exists_new_missing(self, tmp_path, monkeypatch):
        """Should warn when data at legacy but not at new location."""
        monkeypatch.chdir(tmp_path)

        # Create only legacy
        (tmp_path / "data" / "1-benchmarks").mkdir(parents=True)

        result = check_migration_needed(DEFAULT_DATA_DIR)

        assert result is not None
        assert "legacy location" in result
        assert "data/" in result
        assert ENV_VAR_NAME in result


@pytest.mark.unit
class TestGetDataDirSource:
    """Tests for get_data_dir_source function."""

    def test_cli_source(self, monkeypatch):
        """Should report CLI source when cli_arg provided."""
        monkeypatch.setenv(ENV_VAR_NAME, "/env/path")

        result = get_data_dir_source(cli_arg="/cli/path", config_value="/config/path")

        assert "CLI" in result
        assert "--data-dir" in result

    def test_env_source(self, monkeypatch):
        """Should report env source when env var set."""
        monkeypatch.setenv(ENV_VAR_NAME, "/env/path")

        result = get_data_dir_source(cli_arg=None, config_value="/config/path")

        assert "environment" in result
        assert ENV_VAR_NAME in result

    def test_config_source(self, monkeypatch):
        """Should report config source when config value used."""
        monkeypatch.delenv(ENV_VAR_NAME, raising=False)

        result = get_data_dir_source(cli_arg=None, config_value="/config/path")

        assert "config" in result

    def test_default_source(self, monkeypatch):
        """Should report default when no overrides."""
        monkeypatch.delenv(ENV_VAR_NAME, raising=False)

        result = get_data_dir_source(cli_arg=None, config_value=None)

        assert result == "default"
