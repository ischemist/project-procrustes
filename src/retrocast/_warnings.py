from __future__ import annotations

import warnings


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
