from __future__ import annotations

import importlib
import importlib.metadata

import pytest

import retrocast._version as version_module
from retrocast._warnings import RetroCastFutureWarning, warn_deprecated


@pytest.mark.unit
@pytest.mark.parametrize(
    ("raw", "normalized"),
    [
        ("1.2a3", "1.2.0a3"),
        ("1.2.post4", "1.2.0.post4"),
        ("1.2.dev5+local", "1.2.0.dev5+local"),
    ],
)
def test_version_normalization_keeps_explicit_patch(raw: str, normalized: str) -> None:
    assert version_module._normalize_version_with_patch(raw) == normalized


@pytest.mark.unit
def test_version_falls_back_when_package_metadata_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing_version(_name: str) -> str:
        raise importlib.metadata.PackageNotFoundError

    monkeypatch.setattr(importlib.metadata, "version", missing_version)
    reloaded = importlib.reload(version_module)

    assert reloaded.__version__ == "0.0.0.dev0+unknown"
    importlib.reload(version_module)


@pytest.mark.unit
def test_warn_deprecated_emits_consistent_public_api_message() -> None:
    with pytest.warns(RetroCastFutureWarning) as warnings:
        warn_deprecated(old="old_fn", new="new_fn", remove_in="1.0", note="Use batch mode.")

    assert (
        str(warnings[0].message)
        == "old_fn is deprecated and will be removed in 1.0. Use new_fn instead. Use batch mode."
    )
