"""Video frame extraction.

The primary API is :func:`extract_frames` which pulls N frames per
minute from a video file, optionally resizes them, and writes them to a
per-video subdirectory together with a ``manifest.json`` describing
each saved frame.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Iterable, List, Optional

import cv2

from ..utils.logger import get_logger
from ..utils.paths import ensure_dir, resolve, safe_stem

log = get_logger(__name__)

SUPPORTED_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".wmv", ".flv"}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class FrameRecord:
    """Metadata for one extracted frame."""

    index: int                  # Index within the extracted sequence (0-based)
    source_frame: int           # Original frame index in the video
    timestamp_sec: float        # Position in the source video, in seconds
    timestamp_hhmmss: str       # Human-readable time code
    file: str                   # Filename (relative to the frames folder)
    width: int
    height: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class VideoInfo:
    path: Path
    fps: float
    frame_count: int
    duration_sec: float
    width: int
    height: int

    def to_dict(self) -> dict:
        return {
            "path": str(self.path),
            "fps": self.fps,
            "frame_count": self.frame_count,
            "duration_sec": self.duration_sec,
            "width": self.width,
            "height": self.height,
        }


@dataclass
class ExtractionResult:
    video: VideoInfo
    output_dir: Path
    frames: List[FrameRecord]
    frames_per_minute: float
    elapsed_sec: float

    def to_dict(self) -> dict:
        return {
            "video": self.video.to_dict(),
            "output_dir": str(self.output_dir),
            "frames_per_minute": self.frames_per_minute,
            "elapsed_sec": self.elapsed_sec,
            "frames": [f.to_dict() for f in self.frames],
        }


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------
def probe(video_path: Path | str) -> VideoInfo:
    """Return basic metadata about a video without decoding it."""
    p = Path(video_path)
    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {p}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        dur = (n / fps) if fps > 0 else 0.0
        return VideoInfo(path=p, fps=fps, frame_count=n, duration_sec=dur,
                         width=w, height=h)
    finally:
        cap.release()


def _hhmmss(sec: float) -> str:
    if not math.isfinite(sec) or sec < 0:
        sec = 0.0
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec - h * 3600 - m * 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def _resize_long_side(img, long_side: Optional[int]):
    if not long_side or long_side <= 0:
        return img
    h, w = img.shape[:2]
    if max(h, w) <= long_side:
        return img
    if w >= h:
        new_w = long_side
        new_h = int(round(h * long_side / w))
    else:
        new_h = long_side
        new_w = int(round(w * long_side / h))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def iter_video_files(root: Path | str) -> Iterable[Path]:
    root = Path(root)
    if root.is_file() and root.suffix.lower() in SUPPORTED_EXTS:
        yield root
        return
    if not root.exists():
        return
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
            yield p


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------
ProgressCb = Callable[[int, int, str], None]


def extract_frames(
    video_path: Path | str,
    output_root: Path | str,
    frames_per_minute: float = 6.0,
    image_format: str = "jpg",
    jpeg_quality: int = 92,
    resize_long_side: Optional[int] = 1024,
    progress: Optional[ProgressCb] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> ExtractionResult:
    """Extract ``frames_per_minute`` frames from ``video_path``.

    Parameters
    ----------
    video_path : path to a single video file
    output_root : directory that will contain a subfolder per video
    frames_per_minute : how many frames to keep per minute of source video
    image_format : ``jpg`` or ``png``
    jpeg_quality : 1..100 (only used for jpg)
    resize_long_side : if set, resize so the longest side equals this value
    progress : callback ``(done, total, message)``
    should_stop : callback returning ``True`` to abort mid-extraction
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    image_format = image_format.lower().strip(".")
    if image_format not in {"jpg", "jpeg", "png"}:
        raise ValueError(f"Unsupported image format: {image_format}")
    ext = "jpg" if image_format in {"jpg", "jpeg"} else "png"

    info = probe(video_path)
    if info.fps <= 0 or info.frame_count <= 0:
        raise IOError(f"Invalid video (fps={info.fps}, frames={info.frame_count})")

    # Compute stride (source frames between two kept frames).
    frames_per_second_target = frames_per_minute / 60.0
    if frames_per_second_target <= 0:
        raise ValueError("frames_per_minute must be > 0")
    stride = max(1, int(round(info.fps / frames_per_second_target)))
    expected = max(1, info.frame_count // stride)

    stem = safe_stem(video_path.stem)
    out_dir = ensure_dir(Path(output_root) / stem)

    log.info(
        "Extracting frames: %s (fps=%.2f, frames=%d, stride=%d, target=%d)",
        video_path.name, info.fps, info.frame_count, stride, expected,
    )

    encode_params: list[int] = []
    if ext == "jpg":
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
    else:
        encode_params = [int(cv2.IMWRITE_PNG_COMPRESSION), 3]

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    records: List[FrameRecord] = []
    started = time.time()
    saved = 0
    src_idx = 0

    try:
        # Seek by frame index rather than time to stay deterministic
        while True:
            if should_stop and should_stop():
                log.warning("Extraction cancelled by user at frame %d", src_idx)
                break

            cap.set(cv2.CAP_PROP_POS_FRAMES, src_idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            frame = _resize_long_side(frame, resize_long_side)
            h, w = frame.shape[:2]
            ts = src_idx / info.fps
            fname = f"{stem}_{saved:06d}.{ext}"
            fpath = out_dir / fname

            ok_write, buf = cv2.imencode(f".{ext}", frame, encode_params)
            if not ok_write:
                log.warning("Failed to encode frame %d", src_idx)
            else:
                buf.tofile(str(fpath))  # unicode-safe path on Windows

            rec = FrameRecord(
                index=saved,
                source_frame=src_idx,
                timestamp_sec=round(ts, 4),
                timestamp_hhmmss=_hhmmss(ts),
                file=fname,
                width=w,
                height=h,
            )
            records.append(rec)
            saved += 1

            if progress:
                progress(saved, expected, f"{video_path.name} @ {_hhmmss(ts)}")

            src_idx += stride
            if src_idx >= info.frame_count:
                break
    finally:
        cap.release()

    elapsed = time.time() - started
    result = ExtractionResult(
        video=info,
        output_dir=out_dir,
        frames=records,
        frames_per_minute=frames_per_minute,
        elapsed_sec=elapsed,
    )

    # Persist manifest for downstream steps
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(result.to_dict(), fh, indent=2, ensure_ascii=False)

    log.info(
        "Extracted %d frames from %s in %.1fs -> %s",
        saved, video_path.name, elapsed, out_dir,
    )
    return result


def extract_folder(
    videos_root: Path | str,
    output_root: Path | str,
    frames_per_minute: float = 6.0,
    image_format: str = "jpg",
    jpeg_quality: int = 92,
    resize_long_side: Optional[int] = 1024,
    progress: Optional[ProgressCb] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> List[ExtractionResult]:
    """Extract frames from every video found under ``videos_root``."""
    results: List[ExtractionResult] = []
    videos = list(iter_video_files(videos_root))
    for i, v in enumerate(videos, 1):
        if should_stop and should_stop():
            break
        log.info("[%d/%d] Processing %s", i, len(videos), v.name)
        res = extract_frames(
            v,
            output_root=output_root,
            frames_per_minute=frames_per_minute,
            image_format=image_format,
            jpeg_quality=jpeg_quality,
            resize_long_side=resize_long_side,
            progress=progress,
            should_stop=should_stop,
        )
        results.append(res)
    return results


def load_manifest(frames_dir: Path | str) -> Optional[dict]:
    """Load ``manifest.json`` from a frames folder if present."""
    p = resolve(frames_dir) / "manifest.json"
    if not p.exists():
        return None
    with p.open("r", encoding="utf-8") as fh:
        return json.load(fh)
