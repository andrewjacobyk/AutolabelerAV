"""Minimal smoke test - imports every module and validates config.

Run with the venv activated:  python _smoke.py
"""

from __future__ import annotations

import importlib
import sys
import traceback


MODULES = [
    "src",
    "src.utils.logger",
    "src.utils.paths",
    "src.utils.gpu",
    "src.core.config",
    "src.core.dataset",
    "src.core.video",
    "src.core.inference",
    "src.core.finetune",
    "src.core.validate",
    "src.core.resources",
    "src.core.vlm",
    "src.core.vlm.base",
    "src.core.vlm.moondream",
    "src.core.vlm.florence",
    "src.core.vlm.smolvlm",
    "src.core.vlm.blip",
    "src.core.vlm.generic",
    "src.core.vlm.paligemma",
    "src.core.vlm.qwenvl",
    "src.core.vlm.internvl",
    "src.core.vlm.minicpm",
    "src.gui.widgets",
    "src.gui.tab_extract",
    "src.gui.tab_inference",
    "src.gui.tab_finetune",
    "src.gui.tab_validate",
    "src.gui.tab_settings",
    "src.gui.app",
    "src.main",
]


def main() -> int:
    ok = fail = 0
    for m in MODULES:
        try:
            importlib.import_module(m)
            print(f"[OK]   {m}")
            ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"[FAIL] {m}: {e.__class__.__name__}: {e}")
            traceback.print_exc()
            fail += 1

    # Validate the config + factory can resolve every declared model.
    print("\nValidating config -> load_model factory ...")
    try:
        from src.core.config import AppConfig
        from src.core.vlm import load_model
        from src.core.resources import estimate, format_table
        cfg = AppConfig.load()
        for mid in cfg.all_model_ids():
            spec = cfg.model_spec(mid)
            instance = load_model(mid, spec)
            print(f"[OK]   {mid:32s} -> {type(instance).__name__}")

        print("\nResource estimates (VRAM in GB):\n")
        print(format_table(cfg.get("models", {}) or {}))
        cfg_ok = True
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] config validation: {e}")
        traceback.print_exc()
        cfg_ok = False

    print(f"\nImports: {ok} ok, {fail} failed. Config resolves: {cfg_ok}")
    return 0 if (fail == 0 and cfg_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
