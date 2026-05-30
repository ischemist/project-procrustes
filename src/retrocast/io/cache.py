from __future__ import annotations

import functools
import json
import logging
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, TypeVar

from retrocast.hashing import hash_json
from retrocast.paths import resolve_cache_dir

logger = logging.getLogger(__name__)

T = TypeVar("T")
CacheKey = Mapping[str, Any]
USE_DEFAULT_CACHE_DIR = object()


def local_cache(
    *,
    namespace: str,
    key: Callable[..., CacheKey],
    load: Callable[[Path], T],
    save: Callable[[Path, T], None],
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    def decorate(compute: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(compute)
        def cached(*args: Any, **kwargs: Any) -> T:
            cache_dir = kwargs.pop("cache_dir", USE_DEFAULT_CACHE_DIR)
            refresh_cache = kwargs.pop("refresh_cache", False)
            if cache_dir is None:
                return compute(*args, **kwargs)

            cache_key = {"namespace": namespace, **dict(key(*args, **kwargs))}
            cache_parent = resolve_cache_dir(namespace) if cache_dir is USE_DEFAULT_CACHE_DIR else Path(cache_dir)
            cache_root = _cache_root(cache_parent, cache_key)
            if not refresh_cache:
                cached_value = _load(cache_root, cache_key, load)
                if cached_value is not None:
                    logger.info("loaded cached %s from %s", namespace, cache_root)
                    return cached_value

            value = compute(*args, **kwargs)
            _write(cache_root, cache_key, save, value)
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
    except Exception as exc:
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
