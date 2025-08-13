from __future__ import annotations

import logging
from typing import Optional


def setup_logger(name: str = "pspm", level: int = logging.INFO) -> logging.Logger:
    """Create and configure a module-level logger."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(level)
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    logger.propagate = False
    return logger
