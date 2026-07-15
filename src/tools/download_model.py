"""CLI: download a model's weights to the local HF cache."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Mirror the env defaults from main.py / run.bat.
os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parents[2] / "data" / "models"))
os.environ.setdefault("HF_HUB_CACHE", str(Path(os.environ["HF_HOME"]) / "hub"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

from ..core.config import AppConfig
from ..core.model_hub import ensure_cached
from ..core.resources import estimate, format_short
from ..utils.logger import get_logger, setup_logging

log = get_logger(__name__)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Download VLM weights to local cache")
    p.add_argument("model_id", help="Key from config.yaml models: block")
    p.add_argument("--config", default="config.yaml")
    args = p.parse_args(argv)

    setup_logging(level="INFO", console=True)
    cfg = AppConfig.load(args.config)
    spec = cfg.model_spec(args.model_id)
    est = estimate(args.model_id, "fp16", family=spec.get("family"))

    print(f"Model : {args.model_id}")
    print(f"HF id : {spec['hf_id']}")
    print(f"Est.  : {format_short(est)}")
    print()

    path = ensure_cached(spec["hf_id"], spec.get("revision"),
                         progress=lambda m: print(f"  {m}"))
    print(f"\nCached at: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
