from __future__ import annotations

import warnings
from typing import ClassVar


class RetroCastFutureWarning(FutureWarning):
    """Warning category for public RetroCast API migrations."""


def warn_deprecated(
    *,
    old: str,
    new: str | None = None,
    remove_in: str,
    note: str | None = None,
    stacklevel: int = 2,
) -> None:
    """Emit a consistent warning for public API migrations."""
    message = f"{old} is deprecated and will be removed in {remove_in}."
    if new is not None:
        message += f" Use {new} instead."
    if note is not None:
        message += f" {note}"
    warnings.warn(message, RetroCastFutureWarning, stacklevel=stacklevel)


class DeprecatedFieldAccessMixin:
    _deprecated_fields: ClassVar[dict[str, tuple[str, str | None, str | None]]] = {}

    def __getattribute__(self, name: str):
        deprecated_fields = object.__getattribute__(self, "_deprecated_fields")
        if name in deprecated_fields:
            old, new, note = deprecated_fields[name]
            warn_deprecated(old=old, new=new, remove_in="0.3.0", note=note, stacklevel=3)
        return super().__getattribute__(name)
