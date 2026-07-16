import logging
import warnings

from retrocast._warnings import RetroCastFutureWarning
from retrocast.utils.logging import configure_script_logging


def test_configure_script_logging_routes_retrocast_future_warnings_to_logger(caplog):
    original_showwarning = warnings.showwarning
    try:
        configure_script_logging(use_rich=False)

        with caplog.at_level(logging.WARNING, logger="retrocast"):
            warnings.warn("deprecated adapter slug", RetroCastFutureWarning, stacklevel=1)
    finally:
        warnings.showwarning = original_showwarning

    assert "deprecated adapter slug" in caplog.text
