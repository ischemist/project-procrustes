from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def hash_file(path: Path) -> str:
    sha256 = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            sha256.update(chunk)
    return sha256.hexdigest()


def hash_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()
