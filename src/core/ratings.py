"""Human rating persistence for model-vs-model comparisons."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..utils.logger import get_logger
from ..utils.paths import ensure_dir, resolve
from . import dataset as ds
from .validate import _align

log = get_logger(__name__)

RATINGS_SCHEMA = 1


def ratings_dir(outputs_dir: Path | str) -> Path:
    return ensure_dir(resolve(outputs_dir) / "ratings")


def comparison_key(model_a_id: str, model_b_id: str, video_stem: str) -> str:
    return f"{video_stem}__{model_a_id}__vs__{model_b_id}"


def ratings_path(
    outputs_dir: Path | str,
    model_a_id: str,
    model_b_id: str,
    video_stem: str,
) -> Path:
    return ratings_dir(outputs_dir) / f"{comparison_key(model_a_id, model_b_id, video_stem)}.json"


def load_comparison(path: Path | str) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def save_comparison(doc: Dict[str, Any], path: Path | str) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, ensure_ascii=False)
    return path


def _avg_inference_sec(doc: Dict[str, Any]) -> Optional[float]:
    """Return average seconds per frame from timing metadata or frame records."""
    timing = doc.get("timing") or {}
    avg = timing.get("avg_sec_per_frame")
    if avg is not None:
        return float(avg)

    frames = doc.get("frames") or []
    secs = [
        float(f["inference_sec"])
        for f in frames
        if f.get("inference_sec") is not None
    ]
    if secs:
        return sum(secs) / len(secs)

    elapsed = doc.get("elapsed_sec")
    if elapsed is not None and frames:
        return float(elapsed) / len(frames)
    return None


def _model_run_meta(doc: Dict[str, Any]) -> Dict[str, Any]:
    model = doc.get("model") or {}
    avg_sec = _avg_inference_sec(doc)
    return {
        "prompt": (doc.get("prompt") or "").strip(),
        "avg_inference_sec": round(avg_sec, 3) if avg_sec is not None else None,
        "precision": model.get("precision"),
        "hf_id": model.get("hf_id"),
    }


def _collect_frames_dirs(*docs: Dict[str, Any], video_stem: str) -> List[str]:
    """Gather every plausible frames directory for image lookup."""
    dirs: List[str] = []
    seen: set[str] = set()

    def add(raw: str | Path | None) -> None:
        if not raw:
            return
        p = resolve(raw)
        key = str(p).lower()
        if key not in seen:
            seen.add(key)
            dirs.append(str(p))

    for doc in docs:
        add((doc.get("extraction") or {}).get("frames_dir"))
        out_dir = (doc.get("extraction") or {}).get("output_dir")
        add(out_dir)

    if video_stem:
        add(Path("data/frames") / video_stem)

    return dirs


def build_comparison_session(
    json_a: Path | str,
    json_b: Path | str,
) -> Dict[str, Any]:
    """Align two output JSONs and prepare a review session."""
    doc_a = ds.load(json_a)
    doc_b = ds.load(json_b)
    model_a_id = (doc_a.get("model") or {}).get("id", "model_a")
    model_b_id = (doc_b.get("model") or {}).get("id", "model_b")
    video_stem = Path(json_a).stem
    frames_dirs = _collect_frames_dirs(doc_a, doc_b, video_stem=video_stem)
    frames_dir = frames_dirs[0] if frames_dirs else None

    aligned = _align(doc_a, doc_b)
    existing_ratings: Dict[str, Dict[str, Any]] = {}

    out_path = ratings_path(
        Path(json_a).parent,
        model_a_id,
        model_b_id,
        video_stem,
    )
    if out_path.exists():
        prev = load_comparison(out_path)
        for row in prev.get("comparisons", []):
            existing_ratings[row.get("file", "")] = row

    comparisons: List[Dict[str, Any]] = []
    for ref_frame, hyp_frame in aligned:
        fname = hyp_frame.get("file") or ref_frame.get("file")
        prev = existing_ratings.get(fname, {})
        comparisons.append({
            "file": fname,
            "index": hyp_frame.get("index", ref_frame.get("index")),
            "timestamp_sec": hyp_frame.get("timestamp_sec"),
            "timestamp_hhmmss": hyp_frame.get("timestamp_hhmmss"),
            "description_a": ref_frame.get("description", ""),
            "description_b": hyp_frame.get("description", ""),
            "rating_a": prev.get("rating_a"),
            "rating_b": prev.get("rating_b"),
            "rated_at": prev.get("rated_at"),
        })

    return {
        "schema_version": RATINGS_SCHEMA,
        "model_a": {
            "id": model_a_id,
            "json_path": str(resolve(json_a)),
            **_model_run_meta(doc_a),
        },
        "model_b": {
            "id": model_b_id,
            "json_path": str(resolve(json_b)),
            **_model_run_meta(doc_b),
        },
        "video_stem": video_stem,
        "frames_dir": frames_dir,
        "frames_dirs": frames_dirs,
        "comparisons": comparisons,
    }


def frame_image_path(session: Dict[str, Any], index: int) -> Optional[Path]:
    comps = session.get("comparisons", [])
    if index < 0 or index >= len(comps):
        return None
    fname = comps[index].get("file")
    if not fname:
        return None

    candidates: List[Path] = []
    seen: set[str] = set()

    def add_dir(raw: str | Path | None) -> None:
        if not raw:
            return
        p = resolve(raw)
        key = str(p).lower()
        if key in seen:
            return
        seen.add(key)
        candidates.append(p)

    for d in session.get("frames_dirs") or []:
        add_dir(d)
    add_dir(session.get("frames_dir"))

    video_stem = session.get("video_stem")
    if video_stem:
        add_dir(Path("data/frames") / video_stem)

    for directory in candidates:
        path = directory / fname
        if path.is_file():
            return path
    return None


def list_output_jsons(folder: Path | str) -> Dict[str, Path]:
    """Map video stem -> JSON path for top-level output files in a folder."""
    root = resolve(folder)
    if not root.is_dir():
        return {}
    return {
        p.stem: p
        for p in sorted(root.glob("*.json"))
        if p.is_file()
    }


def list_synced_json_pairs(
    dir_a: Path | str,
    dir_b: Path | str,
) -> List[Tuple[Path, Path]]:
    """Return (json_a, json_b) pairs with matching filenames in both folders."""
    a_map = list_output_jsons(dir_a)
    b_map = list_output_jsons(dir_b)
    stems = sorted(set(a_map) & set(b_map))
    return [(a_map[s], b_map[s]) for s in stems]


def find_paired_json(
    json_path: Path | str,
    other_dir: Path | str,
) -> Optional[Path]:
    """Find the JSON with the same stem in ``other_dir``."""
    stem = Path(json_path).stem
    return list_output_jsons(other_dir).get(stem)


def pair_index_for_json(
    pairs: List[Tuple[Path, Path]],
    json_path: Path | str,
) -> int:
    """Return the index of ``json_path`` in ``pairs``, or -1."""
    target = resolve(json_path)
    for i, (a, _) in enumerate(pairs):
        if resolve(a) == target:
            return i
    return -1


def save_frame_rating(
    session: Dict[str, Any],
    index: int,
    rating_a: int,
    rating_b: int,
    outputs_dir: Path | str,
) -> Path:
    """Persist ratings for one frame (0-5 each) and sync back to source JSONs."""
    rating_a = max(0, min(5, int(rating_a)))
    rating_b = max(0, min(5, int(rating_b)))
    comps = session.setdefault("comparisons", [])
    if index < 0 or index >= len(comps):
        raise IndexError(f"Frame index {index} out of range")

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    row = comps[index]
    row["rating_a"] = rating_a
    row["rating_b"] = rating_b
    row["rated_at"] = now

    model_a_id = session["model_a"]["id"]
    model_b_id = session["model_b"]["id"]
    video_stem = session["video_stem"]

    session["summary"] = summarize_session(session)

    out_path = ratings_path(outputs_dir, model_a_id, model_b_id, video_stem)
    save_comparison(session, out_path)

    _patch_source_json(session["model_a"]["json_path"], row["file"], rating_a)
    _patch_source_json(session["model_b"]["json_path"], row["file"], rating_b)

    log.info(
        "Saved ratings for %s: %s=%d, %s=%d",
        row.get("file"), model_a_id, rating_a, model_b_id, rating_b,
    )
    return out_path


def _patch_source_json(json_path: str, frame_file: str, rating: int) -> None:
    path = Path(json_path)
    if not path.exists():
        return
    doc = ds.load(path)
    for frame in doc.get("frames", []):
        if frame.get("file") == frame_file:
            frame["human_rating"] = rating
            frame["human_rated_at"] = datetime.now(timezone.utc).isoformat(
                timespec="seconds"
            )
            break
    ds.save(doc, path)


def summarize_session(session: Dict[str, Any]) -> Dict[str, Any]:
    comps = session.get("comparisons", [])
    rated = [c for c in comps if c.get("rating_a") is not None and c.get("rating_b") is not None]
    if not rated:
        return {
            "n_total": len(comps),
            "n_rated": 0,
            "avg_rating_a": None,
            "avg_rating_b": None,
            "wins_a": 0,
            "wins_b": 0,
            "ties": 0,
        }

    avg_a = sum(c["rating_a"] for c in rated) / len(rated)
    avg_b = sum(c["rating_b"] for c in rated) / len(rated)
    wins_a = sum(1 for c in rated if c["rating_a"] > c["rating_b"])
    wins_b = sum(1 for c in rated if c["rating_b"] > c["rating_a"])
    ties = sum(1 for c in rated if c["rating_a"] == c["rating_b"])

    return {
        "n_total": len(comps),
        "n_rated": len(rated),
        "avg_rating_a": round(avg_a, 3),
        "avg_rating_b": round(avg_b, 3),
        "wins_a": wins_a,
        "wins_b": wins_b,
        "ties": ties,
    }


def aggregate_ratings(
    outputs_dir: Path | str,
    model_a_id: str,
    model_b_id: str,
) -> Dict[str, Any]:
    """Average human ratings across every comparison file for a model pair."""
    root = ratings_dir(outputs_dir)
    if not root.exists():
        return {"n_sessions": 0, "n_rated_frames": 0}

    suffix = f"__{model_a_id}__vs__{model_b_id}.json"
    files = sorted(root.glob(f"*{suffix}"))
    all_a: List[float] = []
    all_b: List[float] = []
    sessions = 0

    for fp in files:
        doc = load_comparison(fp)
        summary = doc.get("summary") or summarize_session(doc)
        if summary.get("n_rated", 0) == 0:
            continue
        sessions += 1
        for row in doc.get("comparisons", []):
            if row.get("rating_a") is not None and row.get("rating_b") is not None:
                all_a.append(float(row["rating_a"]))
                all_b.append(float(row["rating_b"]))

    return {
        "model_a": model_a_id,
        "model_b": model_b_id,
        "n_sessions": sessions,
        "n_rated_frames": len(all_a),
        "avg_rating_a": round(sum(all_a) / len(all_a), 3) if all_a else None,
        "avg_rating_b": round(sum(all_b) / len(all_b), 3) if all_b else None,
        "rating_files": [str(f) for f in files],
    }
