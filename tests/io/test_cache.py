from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from retrocast.io.cache import CacheCodec, json_type_cache, local_cache


@dataclass(frozen=True)
class CachedValue:
    name: str
    count: int


@pytest.mark.integration
def test_local_cache_returns_cached_value_for_same_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RETROCAST_CACHE_DIR", str(tmp_path / "cache"))
    calls = 0

    @local_cache(
        namespace="test-values",
        key=lambda value: {"value": value},
        codec=json_type_cache(CachedValue),
    )
    def compute(value: str) -> CachedValue:
        nonlocal calls
        calls += 1
        return CachedValue(name=value, count=calls)

    assert compute("same") == CachedValue(name="same", count=1)
    assert compute("same") == CachedValue(name="same", count=1)
    assert calls == 1


@pytest.mark.integration
def test_local_cache_recomputes_when_key_changes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RETROCAST_CACHE_DIR", str(tmp_path / "cache"))
    calls = 0

    @local_cache(
        namespace="test-values",
        key=lambda value: {"value": value},
        codec=json_type_cache(CachedValue),
    )
    def compute(value: str) -> CachedValue:
        nonlocal calls
        calls += 1
        return CachedValue(name=value, count=calls)

    assert compute("first") == CachedValue(name="first", count=1)
    assert compute("second") == CachedValue(name="second", count=2)
    assert calls == 2


@pytest.mark.integration
def test_local_cache_ignores_invalid_cached_payload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RETROCAST_CACHE_DIR", str(tmp_path / "cache"))
    calls = 0

    @local_cache(
        namespace="test-values",
        key=lambda value: {"value": value},
        codec=CacheCodec(
            load=lambda cache_root: (_ for _ in ()).throw(ValueError(f"bad cache at {cache_root}")),
            save=lambda _cache_root, _value: None,
        ),
    )
    def compute(value: str) -> str:
        nonlocal calls
        calls += 1
        return value

    assert compute("same") == "same"
    assert compute("same") == "same"
    assert calls == 2


@pytest.mark.integration
def test_local_cache_rejects_non_json_key_values(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RETROCAST_CACHE_DIR", str(tmp_path / "cache"))

    @local_cache(
        namespace="test-values",
        key=lambda: {"bad": object()},
        codec=json_type_cache(CachedValue),
    )
    def compute() -> CachedValue:
        return CachedValue(name="unused", count=0)

    with pytest.raises(TypeError, match="cache key must be JSON-serializable"):
        compute()


@pytest.mark.integration
def test_local_cache_ignores_non_mapping_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RETROCAST_CACHE_DIR", str(tmp_path / "cache"))
    calls = 0

    @local_cache(
        namespace="test-values",
        key=lambda: {"value": "same"},
        codec=json_type_cache(CachedValue),
    )
    def compute() -> CachedValue:
        nonlocal calls
        calls += 1
        return CachedValue(name="same", count=calls)

    first = compute()
    cache_roots = list((tmp_path / "cache" / "test-values").iterdir())
    (cache_roots[0] / "manifest.json").write_text('["not", "a", "manifest"]', encoding="utf-8")

    assert first == CachedValue(name="same", count=1)
    assert compute() == CachedValue(name="same", count=2)
