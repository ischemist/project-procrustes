"""Tests for retrocast.paths module."""

from __future__ import annotations

from pathlib import Path

import pytest

from retrocast.exceptions import SecurityError
from retrocast.paths import (
    DEFAULT_DATA_DIR,
    ENV_VAR_NAME,
    check_migration_needed,
    ensure_path_within_root,
    get_data_dir_source,
    get_paths,
    resolve_data_dir,
    validate_directory_name,
    validate_filename,
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


@pytest.mark.unit
class TestValidateFilename:
    """Tests for validate_filename function."""

    def test_valid_filename_returns_unchanged(self):
        """Valid filenames should be returned unchanged."""
        assert validate_filename("results.json.gz") == "results.json.gz"
        assert validate_filename("model_v1.json") == "model_v1.json"
        assert validate_filename("data-file.txt") == "data-file.txt"

    def test_valid_unicode_filename(self):
        """Unicode filenames without separators should be valid."""
        assert validate_filename("файл.json") == "файл.json"
        assert validate_filename("文件.gz") == "文件.gz"

    def test_rejects_forward_slash(self):
        """Should reject forward slashes."""
        with pytest.raises(SecurityError, match="path separator"):
            validate_filename("path/to/file.json")

    def test_rejects_backslash(self):
        """Should reject backslashes."""
        with pytest.raises(SecurityError, match="path separator"):
            validate_filename("path\\to\\file.json")

    def test_rejects_parent_directory_traversal(self):
        """Should reject .. (parent directory)."""
        with pytest.raises(SecurityError, match="path separator"):
            validate_filename("../../../etc/passwd")

    def test_rejects_standalone_dotdot(self):
        """Should reject standalone .."""
        with pytest.raises(SecurityError, match="parent directory"):
            validate_filename("..")

    def test_rejects_standalone_dot(self):
        """Should reject standalone ."""
        with pytest.raises(SecurityError, match="current directory"):
            validate_filename(".")

    def test_rejects_dot_prefix(self):
        """Should reject ./ prefix."""
        with pytest.raises(SecurityError, match="path separator"):
            validate_filename("./file.json")

    def test_rejects_dot_suffix(self):
        """Should reject /. suffix."""
        with pytest.raises(SecurityError, match="path separator"):
            validate_filename("path/.")

    def test_rejects_dotdot_prefix(self):
        """Should reject ../ prefix."""
        with pytest.raises(SecurityError, match="path separator"):
            validate_filename("../file.json")

    def test_rejects_dotdot_suffix(self):
        """Should reject /.. suffix."""
        with pytest.raises(SecurityError, match="path separator"):
            validate_filename("path/..")

    def test_rejects_null_bytes(self):
        """Should reject null bytes."""
        with pytest.raises(SecurityError, match="null bytes"):
            validate_filename("file\x00.txt")

    def test_includes_param_name_in_error(self):
        """Error message should include parameter name."""
        with pytest.raises(SecurityError, match="raw_results_filename"):
            validate_filename("../file.json", param_name="raw_results_filename")

    def test_includes_filename_in_error(self):
        """Error message should include the problematic filename."""
        with pytest.raises(SecurityError, match="../../../etc/passwd"):
            validate_filename("../../../etc/passwd")


@pytest.mark.unit
class TestValidateDirectoryName:
    """Tests for validate_directory_name function."""

    def test_valid_directory_name(self):
        """Valid directory names should be accepted."""
        assert validate_directory_name("model_v1") == "model_v1"
        assert validate_directory_name("test-data") == "test-data"

    def test_rejects_traversal(self):
        """Should reject path traversal."""
        with pytest.raises(SecurityError):
            validate_directory_name("../../etc")

    def test_uses_param_name_in_error(self):
        """Should use the provided param_name in error."""
        with pytest.raises(SecurityError, match="model"):
            validate_directory_name("../etc", param_name="model")


@pytest.mark.unit
class TestEnsurePathWithinRoot:
    """Tests for ensure_path_within_root function."""

    def test_path_within_root_returns_resolved(self, tmp_path):
        """Valid paths should return resolved path."""
        root = tmp_path / "data"
        root.mkdir()
        path = root / "subdir" / "file.txt"

        result = ensure_path_within_root(path, root)

        assert result == path.resolve()

    def test_relative_path_within_root(self, tmp_path):
        """Relative paths within root should be valid."""
        root = tmp_path / "data"
        root.mkdir()
        subdir = root / "subdir"
        subdir.mkdir()

        # Change to subdirectory and use relative path
        import os

        original_cwd = os.getcwd()
        try:
            os.chdir(subdir)
            path = Path("..") / "other" / "file.txt"
            (root / "other").mkdir()
            (root / "other" / "file.txt").write_text("test")

            result = ensure_path_within_root(path, root)

            assert result == (root / "other" / "file.txt").resolve()
        finally:
            os.chdir(original_cwd)

    def test_rejects_traversal_outside_root(self, tmp_path):
        """Should reject paths that escape root."""
        root = tmp_path / "data"
        root.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "secret.txt").write_text("secret")

        path = root / ".." / "outside" / "secret.txt"

        with pytest.raises(SecurityError, match="escapes root directory"):
            ensure_path_within_root(path, root)

    def test_rejects_absolute_path_outside_root(self, tmp_path):
        """Should reject absolute paths outside root."""
        root = tmp_path / "data"
        root.mkdir()

        with pytest.raises(SecurityError, match="escapes root directory"):
            ensure_path_within_root(Path("/etc/passwd"), root)

    def test_description_in_error_message(self, tmp_path):
        """Error message should include path description."""
        root = tmp_path / "data"
        root.mkdir()

        path = root / ".." / "etc"

        with pytest.raises(SecurityError, match="raw data directory"):
            ensure_path_within_root(path, root, description="raw data directory")
