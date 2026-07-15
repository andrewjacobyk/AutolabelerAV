"""Central logging configuration.

Every log line is emitted through the standard ``logging`` module so
that both the GUI (via a queue handler) and the console see the same
stream.
"""

from __future__ import annotations

import logging
import logging.handlers
import queue
import sys
from pathlib import Path
from typing import Optional

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"

# A process-wide queue that the GUI reads from to display log lines.
GUI_LOG_QUEUE: "queue.Queue[str]" = queue.Queue(maxsize=10_000)


class _GuiQueueHandler(logging.Handler):
    """Handler that pushes formatted records into ``GUI_LOG_QUEUE``."""

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover
        try:
            msg = self.format(record)
            try:
                GUI_LOG_QUEUE.put_nowait(msg)
            except queue.Full:
                # Drop the oldest entry to make room.
                try:
                    GUI_LOG_QUEUE.get_nowait()
                    GUI_LOG_QUEUE.put_nowait(msg)
                except queue.Empty:
                    pass
        except Exception:  # noqa: BLE001
            self.handleError(record)


def setup_logging(
    level: str = "INFO",
    log_dir: Optional[Path] = None,
    console: bool = True,
) -> None:
    """Configure the root logger.

    Safe to call multiple times; existing handlers are cleared first.
    """
    root = logging.getLogger()
    root.setLevel(level.upper())

    for h in list(root.handlers):
        root.removeHandler(h)

    fmt = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FMT)

    if console:
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setFormatter(fmt)
        root.addHandler(ch)

    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(
            log_dir / "vla_pipeline.log",
            maxBytes=2_000_000,
            backupCount=5,
            encoding="utf-8",
        )
        fh.setFormatter(fmt)
        root.addHandler(fh)

    gh = _GuiQueueHandler()
    gh.setFormatter(fmt)
    root.addHandler(gh)

    # Hugging Face + HTTP clients are extremely chatty at INFO and drown
    # out our own messages.  Keep warnings/errors only.
    for noisy in (
        "httpx",
        "httpcore",
        "huggingface_hub",
        "huggingface_hub.file_download",
        "huggingface_hub.utils._http",
        "transformers",
        "transformers.modeling_utils",
        "transformers.configuration_utils",
        "urllib3",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
