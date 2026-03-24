"""Central logging setup for the API process (works alongside uvicorn)."""
from __future__ import annotations

import logging
import os


def configure_logging() -> None:
    level_name = os.environ.get("LOG_LEVEL", "INFO").strip().upper()
    level = getattr(logging, level_name, logging.INFO)
    fmt = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
    datefmt = "%Y-%m-%dT%H:%M:%S"
    kwargs = {"level": level, "format": fmt, "datefmt": datefmt}
    try:
        logging.basicConfig(**kwargs, force=True)
    except TypeError:
        logging.basicConfig(**kwargs)
