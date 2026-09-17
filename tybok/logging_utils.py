"""Shared logging setup for the runtime service processes.

Runtime components log through ``tybok.<component>`` loggers. The entry
points (``worker`` / ``gateway`` / ``serve``) call :func:`setup_logging` once so
a single stderr handler with a consistent bracketed
``[timestamp] [logger] [LEVEL] message`` format is installed. Library modules
never configure the root logger at import time, so embedding the engines in
another application leaves that application's logging configuration untouched.
"""

from __future__ import annotations

import logging

# Bracketed line format, second resolution (the C++ gateway mirrors it, see
# ``gateway_cpp/src/log.h``). Sub-second precision is deliberately absent: the only
# place it matters is an inference's latency, which is logged as a value in its own line.
_FORMAT = "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: int = logging.INFO) -> None:
    """Install the shared root handler (idempotent) and set the root level."""
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=level, format=_FORMAT, datefmt=_DATE_FORMAT)
    else:
        root.setLevel(level)
