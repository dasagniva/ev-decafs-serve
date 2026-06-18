"""Logging setup for evdecafs_serve.

Library code only attaches a handler if the root logger has none configured (e.g. when a
module is imported directly, outside of an application that already configures logging).
Unlike the research repo's version, this never writes to a cwd-relative ``logs/`` directory —
that pattern doesn't survive being an installed package serving HTTP requests.
"""

from __future__ import annotations

import logging


def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logging.getLogger().handlers and not logger.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(name)s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    return logger
