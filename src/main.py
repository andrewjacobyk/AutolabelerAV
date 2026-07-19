"""Entry point for the VLM Pipeline GUI."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Optional .env support (safe if the file is missing).
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except Exception:  # pragma: no cover
    pass

from .core.config import AppConfig, DEFAULT_CONFIG_PATH
from .utils.logger import get_logger, setup_logging
from .utils.paths import PROJECT_ROOT


def _parse_args(argv):
    p = argparse.ArgumentParser("vlm-pipeline")
    p.add_argument("--config", default=str(DEFAULT_CONFIG_PATH),
                   help="Path to config.yaml")
    p.add_argument("--headless", action="store_true",
                   help="Skip GUI (useful for CI smoke tests).")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv or sys.argv[1:])

    cfg = AppConfig.load(args.config)
    cfg.ensure_paths()

    setup_logging(
        level=cfg.get("logging.level", "INFO"),
        log_dir=cfg.path_for("logs_dir"),
        console=bool(cfg.get("logging.console", True)),
    )
    log = get_logger(__name__)
    log.info("VLM Pipeline starting (project=%s)", PROJECT_ROOT)

    # Route HuggingFace cache into the project's models folder.
    models_dir = cfg.path_for("models_dir")
    os.environ.setdefault("HF_HOME", str(models_dir))
    os.environ.setdefault("HF_HUB_CACHE", str(models_dir / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(models_dir / "transformers"))

    # Download-stability defaults (also set by run.bat / run.sh but we
    # want them applied when the app is launched from an IDE too).
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

    # Forward HUGGINGFACE_TOKEN -> HF_TOKEN so authenticated requests
    # (higher rate limits + gated model downloads) work automatically.
    token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
    if token:
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGINGFACE_TOKEN"] = token

    if args.headless:
        log.info("Headless mode -> exiting after config load.")
        return 0

    from .gui.app import run
    run(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
