"""Version utilities for retrocast - isolated to avoid circular imports."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from packaging.version import Version


def _normalize_version_with_patch(version_str: str) -> str:
    """Ensure version always has explicit major.minor.micro format (e.g., 0.3.0.dev16 not 0.3.dev16)."""
    v = Version(version_str)
    # Reconstruct with explicit patch version
    base = f"{v.major}.{v.minor}.{v.micro}"

    # Add pre-release, post-release, dev, local parts if present
    parts = [base]
    if v.pre:
        parts.append(f"{v.pre[0]}{v.pre[1]}")
    if v.post is not None:
        parts.append(f".post{v.post}")
    if v.dev is not None:
        parts.append(f".dev{v.dev}")
    if v.local:
        parts.append(f"+{v.local}")

    return "".join(parts)


try:
    __version__ = _normalize_version_with_patch(version("retrocast"))
except PackageNotFoundError:
    # Package not installed (running from source without editable install)
    __version__ = "0.0.0.dev0+unknown"
