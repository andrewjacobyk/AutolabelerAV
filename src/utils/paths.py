"""Filesystem helpers."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve(p: str | Path) -> Path:
    """Resolve a path against the project root when it is relative."""
    path = Path(p)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def ensure_dir(p: str | Path) -> Path:
    path = resolve(p)
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_stem(name: str) -> str:
    """Sanitize a name for use as a folder/file stem."""
    bad = '<>:"/\\|?*'
    out = "".join("_" if c in bad else c for c in name).strip()
    return out or "unnamed"
