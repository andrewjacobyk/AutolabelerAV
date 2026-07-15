"""CLI: print the VRAM / disk resource table."""

from __future__ import annotations

import argparse

from ..core.config import AppConfig
from ..core.resources import format_table


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Print VLM resource estimates")
    p.add_argument("--config", default="config.yaml")
    args = p.parse_args(argv)

    cfg = AppConfig.load(args.config)
    models = cfg.get("models", {}) or {}
    print(format_table(models))
    print()
    print("Columns = estimated peak VRAM (GB) at each precision.")
    print("n/a = precision not supported by that model backend.")
    print("Disk column omitted; all models cache fp16 shards (~2 bytes/param).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
