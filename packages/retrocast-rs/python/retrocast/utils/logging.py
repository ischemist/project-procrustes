import logging
import os
import warnings
from typing import TextIO

from retrocast._warnings import RetroCastFutureWarning

logger = logging.getLogger("retrocast")
_ORIGINAL_SHOWWARNING = warnings.showwarning

# Default to silence. This ensures that if someone imports retrocast
# but doesn't configure logging, they don't get spammed.
logger.addHandler(logging.NullHandler())


def configure_script_logging(use_rich: bool = True, log_level: str = "INFO") -> None:
    """
    Configures logging for CLI scripts/applications.
    Call this at the start of your `main()` functions.
    """
    log_level = os.environ.get("RETROCAST_LOG", log_level).upper()

    if use_rich:
        from rich.console import Console
        from rich.logging import RichHandler

        logging.basicConfig(
            level=log_level,
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler(console=Console(), show_level=True, show_path=True, markup=True)],
        )
    else:
        # Fallback standard logging
        logging.basicConfig(
            level=log_level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )

    # Ensure our library logger respects the level
    logger.setLevel(log_level)
    _route_retrocast_warnings_to_logger()


def _route_retrocast_warnings_to_logger() -> None:
    """Render public RetroCast warnings through the configured CLI logger."""

    def showwarning(
        message: Warning | str,
        category: type[Warning],
        filename: str,
        lineno: int,
        file: TextIO | None = None,
        line: str | None = None,
    ) -> None:
        if issubclass(category, RetroCastFutureWarning):
            logger.warning(str(message))
            return
        _ORIGINAL_SHOWWARNING(message, category, filename, lineno, file=file, line=line)

    warnings.showwarning = showwarning  # ty: ignore[invalid-assignment]
