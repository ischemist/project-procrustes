from __future__ import annotations

from pathlib import Path
from typing import Any


def hash_file(path: Path) -> str:
    from retrocast import native

    return native.hash_file(str(path))


def hash_json(value: Any) -> str:
    from retrocast import native

    return native.hash_json(value)
