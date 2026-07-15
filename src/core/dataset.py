"""Dataset / JSON persistence.

The pipeline stores one JSON per video with the following schema::

    {
      "schema_version": 1,
      "video": {
        "name": "vacation.mp4",
        "path": "data/videos/vacation.mp4",
        "duration_sec": 187.4,
        "fps": 29.97,
        "frame_count": 5615,
        "width": 1920,
        "height": 1080
      },
      "extraction": {
        "frames_per_minute": 6,
        "image_format": "jpg",
        "resize_long_side": 1024,
        "frames_dir": "data/frames/vacation",
        "num_frames": 19
      },
      "model": {
        "id": "moondream2",
        "hf_id": "vikhyatk/moondream2",
        "kind": "local"
      },
      "prompt": "Describe this scene...",
      "generated_at": "2026-07-14T22:41:03Z",
      "frames": [
        {
          "index": 0,
          "source_frame": 0,
          "timestamp_sec": 0.0,
          "timestamp_hhmmss": "00:00:00.000",
          "file": "vacation_000000.jpg",
          "description": "A wide beach at sunset with two people walking."
        },
        ...
      ]
    }
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..utils.logger import get_logger
from ..utils.paths import ensure_dir

log = get_logger(__name__)

SCHEMA_VERSION = 1


def new_document(
    video_meta: Dict[str, Any],
    extraction_meta: Dict[str, Any],
    model_meta: Dict[str, Any],
    prompt: str,
) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "video": video_meta,
        "extraction": extraction_meta,
        "model": model_meta,
        "prompt": prompt,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "frames": [],
    }


def add_frame(doc: Dict[str, Any], frame: Dict[str, Any]) -> None:
    doc.setdefault("frames", []).append(frame)


def save(doc: Dict[str, Any], out_path: Path | str) -> Path:
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, ensure_ascii=False)
    log.info("Wrote %d frames to %s", len(doc.get("frames", [])), out_path)
    return out_path


def load(path: Path | str) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Fine-tuning dataset conversion
# ---------------------------------------------------------------------------
def to_training_pairs(
    doc: Dict[str, Any],
    frames_root: Path | str,
) -> List[Dict[str, Any]]:
    """Convert one output document into ``[{image, text}, ...]`` samples."""
    frames_dir = Path(doc.get("extraction", {}).get("frames_dir") or frames_root)
    out: List[Dict[str, Any]] = []
    for f in doc.get("frames", []):
        desc = (f.get("description") or "").strip()
        if not desc:
            continue
        img_path = frames_dir / f["file"]
        if not img_path.exists():
            continue
        out.append({
            "image_path": str(img_path),
            "text": desc,
            "video": doc.get("video", {}).get("name"),
            "timestamp_sec": f.get("timestamp_sec"),
        })
    return out


def merge_documents_to_dataset(
    docs: List[Dict[str, Any]],
    output_jsonl: Path | str,
    frames_root: Path | str,
) -> int:
    """Write a JSONL dataset combining all provided output documents."""
    output_jsonl = Path(output_jsonl)
    ensure_dir(output_jsonl.parent)
    n = 0
    with output_jsonl.open("w", encoding="utf-8") as fh:
        for doc in docs:
            for pair in to_training_pairs(doc, frames_root):
                fh.write(json.dumps(pair, ensure_ascii=False) + "\n")
                n += 1
    log.info("Wrote %d training pairs to %s", n, output_jsonl)
    return n


def iter_output_files(outputs_dir: Path | str) -> List[Path]:
    """Return every ``*.json`` file below ``outputs_dir``."""
    root = Path(outputs_dir)
    if not root.exists():
        return []
    return sorted(root.glob("*.json"))
