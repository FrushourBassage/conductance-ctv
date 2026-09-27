"""Process-wide logging configuration.

Ref: repository-wide engineering convention; the manuscript reports no logging.
"""

from __future__ import annotations

import logging
import os
import sys

_DEFAULT_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
_CONFIGURED = False


def configure_logging(level: str | int | None = None, stream: object | None = None) -> None:
    """Install a single root handler. Safe to call repeatedly."""
    global _CONFIGURED
    resolved = level if level is not None else os.environ.get("CONDUCTANCE_CTV_LOG_LEVEL", "INFO")
    handler = logging.StreamHandler(sys.stderr if stream is None else stream)  # type: ignore[arg-type]
    handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT))
    root = logging.getLogger("conductance_ctv")
    if not _CONFIGURED:
        root.handlers.clear()
        root.addHandler(handler)
        root.propagate = False
        _CONFIGURED = True
    root.setLevel(resolved)


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger under the package root."""
    if not name.startswith("conductance_ctv"):
        name = f"conductance_ctv.{name}"
    return logging.getLogger(name)
