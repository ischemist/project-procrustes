from __future__ import annotations

import logging

from retrocast.exceptions import RetroCastException


def format_cli_error(error: RetroCastException) -> str:
    """Format an expected RetroCast failure for a CLI boundary."""
    bits = [f"{error.message} [{error.code}]"]
    if error.context:
        safe_context = ", ".join(
            f"{key}={str(value).replace(chr(10), ' ').replace(chr(13), ' ')}"
            for key, value in sorted(error.context.items())
        )
        bits.append(f"({safe_context})")
    return " ".join(bits)


def log_expected_error(
    logger: logging.Logger, message: str, error: RetroCastException, *, exc_info: bool = False
) -> None:
    logger.error(f"{message}: {format_cli_error(error)}", exc_info=exc_info)
