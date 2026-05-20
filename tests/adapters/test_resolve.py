import json

import pytest

from retrocast._warnings import RetroCastFutureWarning
from retrocast.adapters import ADAPTER_TYPES
from retrocast.adapters.resolve import (
    DEFAULT_RAW_RESULTS_FILENAME,
    _read_manifest_directives,
    resolve_adapter,
    resolve_raw_results_filename,
)
from retrocast.exceptions import AdapterResolutionError


@pytest.mark.unit
class TestReadManifestDirectives:
    def test_missing_manifest_returns_empty_dict(self, tmp_path, caplog):
        with caplog.at_level("DEBUG"):
            directives = _read_manifest_directives(tmp_path)

        assert directives == {}
        assert "No manifest.json found" in caplog.text

    def test_malformed_manifest_returns_empty_dict(self, tmp_path, caplog):
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text("{not valid json")

        with caplog.at_level("WARNING"):
            directives = _read_manifest_directives(tmp_path)

        assert directives == {}
        assert "Failed to read manifest" in caplog.text

    def test_non_mapping_directives_returns_empty_dict(self, tmp_path, caplog):
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps({"directives": ["not", "a", "dict"]}))

        with caplog.at_level("WARNING"):
            directives = _read_manifest_directives(tmp_path)

        assert directives == {}
        assert "Manifest directives is not a dict" in caplog.text


@pytest.mark.unit
class TestResolveAdapter:
    def test_resolve_adapter_prefers_cli_override(self, tmp_path):
        adapter, source = resolve_adapter(
            cli_adapter="aizynthfinder",
            raw_dir=tmp_path,
            model_name="test-model",
        )

        assert isinstance(adapter, ADAPTER_TYPES["aizynthfinder"])
        assert source == "cli"

    def test_resolve_adapter_warns_for_deprecated_cli_override(self, tmp_path):
        with pytest.warns(RetroCastFutureWarning, match="aizynth"):
            adapter, source = resolve_adapter(
                cli_adapter="aizynth",
                raw_dir=tmp_path,
                model_name="test-model",
            )

        assert isinstance(adapter, ADAPTER_TYPES["aizynthfinder"])
        assert source == "cli"

    def test_resolve_adapter_uses_manifest_when_cli_is_absent(self, tmp_path):
        (tmp_path / "manifest.json").write_text(json.dumps({"directives": {"adapter": "askcos"}}))

        adapter, source = resolve_adapter(
            cli_adapter=None,
            raw_dir=tmp_path,
            model_name="test-model",
        )

        assert isinstance(adapter, ADAPTER_TYPES["askcos"])
        assert source == "manifest"

    def test_resolve_adapter_warns_for_deprecated_manifest_adapter(self, tmp_path):
        (tmp_path / "manifest.json").write_text(json.dumps({"directives": {"adapter": "dms"}}))

        with pytest.warns(RetroCastFutureWarning, match="dms"):
            adapter, source = resolve_adapter(
                cli_adapter=None,
                raw_dir=tmp_path,
                model_name="test-model",
            )

        assert isinstance(adapter, ADAPTER_TYPES["directmultistep"])
        assert source == "manifest"

    def test_resolve_adapter_invalid_cli_override_raises(self, tmp_path):
        with pytest.raises(AdapterResolutionError) as exc_info:
            resolve_adapter(
                cli_adapter="nope",
                raw_dir=tmp_path,
                model_name="test-model",
            )

        assert exc_info.value.code == "adapter.unknown"
        assert exc_info.value.context["source"] == "cli"

    def test_resolve_adapter_invalid_manifest_adapter_raises(self, tmp_path):
        (tmp_path / "manifest.json").write_text(json.dumps({"directives": {"adapter": "nope"}}))

        with pytest.raises(AdapterResolutionError) as exc_info:
            resolve_adapter(
                cli_adapter=None,
                raw_dir=tmp_path,
                model_name="test-model",
            )

        assert exc_info.value.code == "adapter.unknown"
        assert exc_info.value.context["source"] == "manifest"
        assert exc_info.value.context["adapter"] == "nope"

    def test_resolve_adapter_missing_configuration_raises(self, tmp_path):
        with pytest.raises(AdapterResolutionError) as exc_info:
            resolve_adapter(
                cli_adapter=None,
                raw_dir=tmp_path,
                model_name="test-model",
            )

        assert exc_info.value.code == "adapter.resolution_missing"
        assert exc_info.value.context == {"model": "test-model", "raw_dir": str(tmp_path)}


@pytest.mark.unit
class TestResolveRawResultsFilename:
    def test_uses_default_filename_when_manifest_has_no_override(self, tmp_path):
        assert resolve_raw_results_filename(raw_dir=tmp_path) == DEFAULT_RAW_RESULTS_FILENAME

    def test_uses_manifest_override_and_logs_it(self, tmp_path, caplog):
        (tmp_path / "manifest.json").write_text(json.dumps({"directives": {"raw_results_filename": "valid.json.gz"}}))

        with caplog.at_level("DEBUG"):
            filename = resolve_raw_results_filename(raw_dir=tmp_path)

        assert filename == "valid.json.gz"
        assert "Using raw_results_filename 'valid.json.gz'" in caplog.text
