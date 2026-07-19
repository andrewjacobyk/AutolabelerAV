"""Reliable Hugging Face model download / cache management.

The default ``from_pretrained`` path can look unstable on Windows
because huggingface_hub may use the *xet* transfer backend, which shows
two overlapping progress bars ("Downloading" + "Reconstructing") and
can restart from scratch when interrupted.

This module forces the classic HTTP downloader (``snapshot_download``)
with automatic resume so partial files are picked up cleanly.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Callable, Optional

from ..utils.logger import get_logger

log = get_logger(__name__)

ProgressCb = Callable[[str], None]
_MAX_ATTEMPTS = 4


def _apply_download_env() -> None:
    """Best-effort download defaults (also set in run.bat / main.py)."""
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")
    # Classic HTTP downloader -- avoids the confusing xet reconstruction UI.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")


def cache_dir() -> Path:
    return Path(os.environ.get("HF_HUB_CACHE", "data/models/hub"))


def is_cached(hf_id: str, revision: Optional[str] = None) -> bool:
    """Return True when a complete snapshot exists in the local cache."""
    try:
        from huggingface_hub import repo_info
        info = repo_info(hf_id, revision=revision)
        rev = info.sha
        snap = cache_dir() / f"models--{hf_id.replace('/', '--')}" / "snapshots" / rev
        if not snap.exists():
            return False
        # A complete snapshot always has config.json.
        return (snap / "config.json").exists()
    except Exception:
        return False


def ensure_cached(
    hf_id: str,
    revision: Optional[str] = None,
    *,
    progress: Optional[ProgressCb] = None,
) -> Path:
    """Download (or resume) every file for ``hf_id`` and return the cache path.

    Safe to call multiple times; already-cached files are skipped.
    Retries on transient network errors (common on large shards).
    """
    _apply_download_env()

    if is_cached(hf_id, revision):
        log.info("Model %s already cached locally.", hf_id)
        if progress:
            progress(f"{hf_id}: already cached")
        return _snapshot_path(hf_id, revision)

    from huggingface_hub import snapshot_download

    log.info("Downloading %s (revision=%s) ...", hf_id, revision or "main")
    if progress:
        progress(f"Downloading {hf_id} ... (large models can take 10-30 min)")

    kwargs = dict(repo_id=hf_id)
    if revision:
        kwargs["revision"] = revision

    last_err: Optional[Exception] = None
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            started = time.time()
            path = Path(snapshot_download(**kwargs))
            elapsed = time.time() - started
            log.info("Download complete for %s in %.0fs -> %s", hf_id, elapsed, path)
            if progress:
                progress(f"Download complete ({elapsed:.0f}s)")
            return path
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            wait = min(30, 5 * attempt)
            log.warning(
                "Download attempt %d/%d for %s failed: %s. Retrying in %ds ...",
                attempt, _MAX_ATTEMPTS, hf_id, exc, wait,
            )
            if progress:
                progress(f"Download interrupted, retrying ({attempt}/{_MAX_ATTEMPTS}) ...")
            time.sleep(wait)

    raise RuntimeError(
        f"Failed to download {hf_id} after {_MAX_ATTEMPTS} attempts: {last_err}"
    ) from last_err


def _snapshot_path(hf_id: str, revision: Optional[str]) -> Path:
    from huggingface_hub import repo_info
    info = repo_info(hf_id, revision=revision)
    return cache_dir() / f"models--{hf_id.replace('/', '--')}" / "snapshots" / info.sha


def download_from_spec(spec: dict, progress: Optional[ProgressCb] = None) -> Path:
    """Convenience wrapper that reads ``hf_id`` / ``revision`` from a config spec."""
    return ensure_cached(
        spec["hf_id"],
        spec.get("revision"),
        progress=progress,
    )


def cache_status(hf_id: str, revision: Optional[str] = None) -> str:
    """Return ``cached`` or ``missing`` for a Hugging Face model id."""
    return "cached" if is_cached(hf_id, revision) else "missing"


def cache_status_for_config(models: dict) -> Dict[str, str]:
    """Map every config model key to ``cached`` / ``missing``."""
    out: Dict[str, str] = {}
    for mid, spec in (models or {}).items():
        out[mid] = cache_status(spec.get("hf_id", ""), spec.get("revision"))
    return out
