"""Inference orchestration.

Takes an extracted-frames directory (with ``manifest.json``), runs a
VLM against every frame, and writes the resulting JSON per-video.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..utils.logger import get_logger
from ..utils.paths import ensure_dir, resolve
from . import dataset as ds
from .model_hub import download_from_spec
from .video import load_manifest
from .vlm import load_model
from .vlm.base import VLMBase

log = get_logger(__name__)

ProgressCb = Callable[[int, int, str], None]


def run_inference_on_frames(
    frames_dir: Path | str,
    output_json: Path | str,
    model_id: str,
    model_spec: Dict[str, Any],
    prompt: str,
    precision: str = "fp16",
    max_new_tokens: int = 96,
    progress: Optional[ProgressCb] = None,
    should_stop: Optional[Callable[[], bool]] = None,
    model: Optional[VLMBase] = None,
) -> Path:
    """Run inference against a folder of extracted frames.

    ``frames_dir`` must contain the ``manifest.json`` produced by
    :func:`core.video.extract_frames`.
    """
    frames_dir = resolve(frames_dir)
    manifest = load_manifest(frames_dir)
    if manifest is None:
        raise FileNotFoundError(f"No manifest.json in {frames_dir}")

    video_meta = manifest.get("video", {}) or {}
    video_name = Path(video_meta.get("path", "unknown")).name

    doc = ds.new_document(
        video_meta={
            "name": video_name,
            "path": video_meta.get("path"),
            "duration_sec": video_meta.get("duration_sec"),
            "fps": video_meta.get("fps"),
            "frame_count": video_meta.get("frame_count"),
            "width": video_meta.get("width"),
            "height": video_meta.get("height"),
        },
        extraction_meta={
            "frames_per_minute": manifest.get("frames_per_minute"),
            "frames_dir": str(frames_dir),
            "num_frames": len(manifest.get("frames", [])),
        },
        model_meta={
            "id": model_id,
            "hf_id": model_spec.get("hf_id"),
            "kind": model_spec.get("kind", "local"),
            "provider": model_spec.get("provider"),
            "precision": precision,
        },
        prompt=prompt,
    )

    own_model = model is None
    if own_model:
        model = load_model(model_id, model_spec, precision=precision,
                           max_new_tokens=max_new_tokens)
        log.info("Ensuring weights are cached for %s ...", model_spec.get("hf_id"))
        download_from_spec(model_spec)
        model.load()

    frames = manifest.get("frames", [])
    total = len(frames)
    started = time.time()

    try:
        for i, frame in enumerate(frames):
            if should_stop and should_stop():
                log.warning("Inference cancelled by user at frame %d/%d", i, total)
                break
            img_path = frames_dir / frame["file"]
            if not img_path.exists():
                log.warning("Missing frame file: %s", img_path)
                continue
            try:
                image = VLMBase.open_image(img_path)
                t0 = time.perf_counter()
                description = model.describe(image, prompt)
                infer_sec = time.perf_counter() - t0
            except Exception as exc:  # noqa: BLE001
                log.exception("Inference failed for %s: %s", img_path.name, exc)
                description = f"[error: {exc}]"
                infer_sec = 0.0

            entry = dict(frame)
            entry["description"] = description
            entry["inference_sec"] = round(infer_sec, 3)
            ds.add_frame(doc, entry)

            if progress:
                progress(i + 1, total, f"{video_name}: {frame.get('timestamp_hhmmss', '')}")

        elapsed = time.time() - started
        n_done = len(doc.get("frames", []))
        doc["elapsed_sec"] = round(elapsed, 3)
        doc["timing"] = {
            "elapsed_sec": round(elapsed, 3),
            "num_frames": n_done,
            "avg_sec_per_frame": round(elapsed / n_done, 3) if n_done else 0.0,
        }
        ds.save(doc, output_json)
        log.info(
            "Inference done in %.1fs (avg %.2fs/frame) -> %s",
            elapsed, doc["timing"]["avg_sec_per_frame"], output_json,
        )
        return Path(output_json)
    finally:
        if own_model and model is not None:
            model.unload()


def run_inference_on_many(
    frames_root: Path | str,
    outputs_dir: Path | str,
    model_id: str,
    model_spec: Dict[str, Any],
    prompt: str,
    precision: str = "fp16",
    max_new_tokens: int = 96,
    progress: Optional[ProgressCb] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> List[Path]:
    """Run inference over every frame-subfolder inside ``frames_root``."""
    frames_root = resolve(frames_root)
    outputs_dir = ensure_dir(outputs_dir)

    # Load model once and reuse it across all videos.
    model = load_model(model_id, model_spec, precision=precision,
                       max_new_tokens=max_new_tokens)
    log.info("Ensuring weights are cached for %s ...", model_spec.get("hf_id"))
    download_from_spec(model_spec)
    model.load()

    subdirs = [p for p in sorted(frames_root.iterdir())
               if p.is_dir() and (p / "manifest.json").exists()]

    written: List[Path] = []
    try:
        for i, sub in enumerate(subdirs, 1):
            if should_stop and should_stop():
                break
            out_json = Path(outputs_dir) / f"{sub.name}.json"
            log.info("[%d/%d] Inference on %s -> %s", i, len(subdirs), sub, out_json)
            written.append(
                run_inference_on_frames(
                    sub, out_json,
                    model_id=model_id, model_spec=model_spec,
                    prompt=prompt, precision=precision,
                    max_new_tokens=max_new_tokens,
                    progress=progress, should_stop=should_stop,
                    model=model,
                )
            )
    finally:
        model.unload()
    return written
