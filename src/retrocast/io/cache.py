from __future__ import annotations

import functools
import json
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, TypeVar

from pydantic import TypeAdapter, ValidationError

from retrocast.exceptions import RetroCastException
from retrocast.hashing import hash_json
from retrocast.io.blob import load_json_gz, save_json_gz
from retrocast.paths import resolve_cache_dir

logger = logging.getLogger(__name__)

T = TypeVar("T")
CacheKey = Mapping[str, Any]


@dataclass(frozen=True)
class CacheCodec(Generic[T]):
    load: Callable[[Path], T]
    save: Callable[[Path, T], None]


def json_type_cache(value_type: Any) -> CacheCodec[Any]:
    adapter = TypeAdapter(value_type)

    def load(cache_root: Path) -> Any:
        return adapter.validate_python(load_json_gz(cache_root / "value.json.gz"))

    def save(cache_root: Path, value: Any) -> None:
        payload = adapter.dump_python(value, mode="json")
        save_json_gz(payload, cache_root / "value.json.gz")

    return CacheCodec(load=load, save=save)


def local_cache(
    *,
    namespace: str,
    key: Callable[..., CacheKey],
    codec: CacheCodec[T],
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    def decorate(compute: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(compute)
        def cached(*args: Any, **kwargs: Any) -> T:
            cache_key = {"namespace": namespace, **dict(key(*args, **kwargs))}
            cache_parent = resolve_cache_dir(namespace)
            cache_root = _cache_root(cache_parent, cache_key)
            cached_value = _load(cache_root, cache_key, codec.load)
            if cached_value is not None:
                logger.info("loaded cached %s from %s", namespace, cache_root)
                return cached_value

            value = compute(*args, **kwargs)
            _write(cache_root, cache_key, codec.save, value)
            logger.info("cached %s in %s", namespace, cache_root)
            return value

        return cached

    return decorate


def _load(cache_root: Path, cache_key: CacheKey, load: Callable[[Path], T]) -> T | None:
    manifest_path = cache_root / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("cache_key") != dict(cache_key):
            raise ValueError("cache key mismatch")
        return load(cache_root)
    except (OSError, ValueError, ValidationError, RetroCastException) as exc:
        logger.warning("ignoring invalid cache at %s: %s", cache_root, exc)
        return None


def _write(cache_root: Path, cache_key: CacheKey, save: Callable[[Path, T], None], value: T) -> None:
    cache_root.mkdir(parents=True, exist_ok=True)
    save(cache_root, value)
    manifest = {"cache_key": dict(cache_key)}
    (cache_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def _cache_root(cache_dir: Path, cache_key: CacheKey) -> Path:
    digest = hash_json(cache_key)
    return cache_dir / digest[:16]
