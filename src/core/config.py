"""Configuration loading / persistence."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import yaml

from ..utils.paths import PROJECT_ROOT, ensure_dir, resolve

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


@dataclass
class AppConfig:
    """Thin wrapper around the loaded YAML mapping."""

    data: Dict[str, Any] = field(default_factory=dict)
    path: Path = DEFAULT_CONFIG_PATH

    # ------------------------------------------------------------------
    # Loading / saving
    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path: Path | str | None = None) -> "AppConfig":
        cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
        if not cfg_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {cfg_path}")
        with cfg_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return cls(data=data, path=cfg_path)

    def save(self, path: Path | str | None = None) -> Path:
        out = Path(path) if path else self.path
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(self.data, fh, sort_keys=False, allow_unicode=True)
        self.path = out
        return out

    def copy(self) -> "AppConfig":
        return AppConfig(data=copy.deepcopy(self.data), path=self.path)

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------
    def get(self, dotted: str, default: Any = None) -> Any:
        cur: Any = self.data
        for part in dotted.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                return default
        return cur

    def set(self, dotted: str, value: Any) -> None:
        cur: Dict[str, Any] = self.data
        parts = dotted.split(".")
        for part in parts[:-1]:
            cur = cur.setdefault(part, {})
            if not isinstance(cur, dict):
                raise TypeError(f"Path {dotted!r} traverses non-dict node")
        cur[parts[-1]] = value

    # ------------------------------------------------------------------
    # Common paths (resolved absolute)
    # ------------------------------------------------------------------
    def path_for(self, key: str) -> Path:
        raw = self.get(f"paths.{key}")
        if raw is None:
            raise KeyError(f"paths.{key} not configured")
        return resolve(raw)

    def ensure_paths(self) -> None:
        for key in (
            "videos_dir",
            "frames_dir",
            "outputs_dir",
            "datasets_dir",
            "models_dir",
            "logs_dir",
        ):
            raw = self.get(f"paths.{key}")
            if raw:
                ensure_dir(raw)

    # ------------------------------------------------------------------
    # Model helpers
    # ------------------------------------------------------------------
    def model_spec(self, model_id: str) -> Dict[str, Any]:
        # Look up directly instead of going through the dotted-key
        # helper -- model ids may legitimately contain ``.``
        models = self.get("models", {}) or {}
        spec = models.get(model_id)
        if not spec:
            raise KeyError(
                f"Model {model_id!r} is not defined in config.models"
            )
        return spec

    def local_model_ids(self) -> list[str]:
        models = self.get("models", {}) or {}
        return [k for k, v in models.items() if v.get("kind", "local") == "local"]

    def all_model_ids(self) -> list[str]:
        return list((self.get("models", {}) or {}).keys())
