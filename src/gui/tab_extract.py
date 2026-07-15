"""GUI tab: Video -> Frames extraction."""

from __future__ import annotations

import tkinter as tk
from pathlib import Path

import customtkinter as ctk

from ..core.config import AppConfig
from ..core.video import extract_folder, extract_frames, iter_video_files, probe
from ..utils.logger import get_logger
from ..utils.paths import resolve
from .widgets import PathPicker, ProgressPanel, Worker

log = get_logger(__name__)


class ExtractTab(ctk.CTkFrame):
    def __init__(self, master, config: AppConfig, **kwargs):
        super().__init__(master, **kwargs)
        self.config_obj = config
        self.worker = Worker()

        self._build()

    # ------------------------------------------------------------------
    def _build(self) -> None:
        header = ctk.CTkLabel(
            self, text="Step 1 - Extract frames from video",
            font=ctk.CTkFont(size=18, weight="bold"),
        )
        header.pack(anchor="w", padx=16, pady=(16, 4))

        subtitle = ctk.CTkLabel(
            self,
            text=("Because video is highly autocorrelated in time, we keep only "
                  "a handful of frames per minute for inference."),
            justify="left",
        )
        subtitle.pack(anchor="w", padx=16, pady=(0, 12))

        # ----- Paths ---------------------------------------------------
        pf = ctk.CTkFrame(self)
        pf.pack(fill="x", padx=16, pady=6)

        self.pick_input = PathPicker(
            pf, label="Video or folder:", kind="dir",
            initial=str(self.config_obj.path_for("videos_dir")),
        )
        self.pick_input.pack(fill="x", padx=6, pady=4)

        self.pick_out = PathPicker(
            pf, label="Frames output dir:", kind="dir",
            initial=str(self.config_obj.path_for("frames_dir")),
        )
        self.pick_out.pack(fill="x", padx=6, pady=4)

        row = ctk.CTkFrame(pf, fg_color="transparent")
        row.pack(fill="x", padx=6, pady=6)
        row.grid_columnconfigure((1, 3, 5), weight=1)

        ctk.CTkLabel(row, text="Frames per minute:").grid(row=0, column=0, sticky="w", padx=4)
        self.fpm_var = tk.StringVar(
            value=str(self.config_obj.get("extraction.frames_per_minute", 6))
        )
        ctk.CTkEntry(row, textvariable=self.fpm_var, width=90).grid(
            row=0, column=1, sticky="w", padx=(0, 12)
        )

        ctk.CTkLabel(row, text="Format:").grid(row=0, column=2, sticky="w", padx=4)
        self.fmt_var = tk.StringVar(
            value=self.config_obj.get("extraction.image_format", "jpg")
        )
        ctk.CTkOptionMenu(row, values=["jpg", "png"], variable=self.fmt_var,
                          width=90).grid(row=0, column=3, sticky="w", padx=(0, 12))

        ctk.CTkLabel(row, text="Long side (px):").grid(row=0, column=4, sticky="w", padx=4)
        self.long_var = tk.StringVar(
            value=str(self.config_obj.get("extraction.resize_long_side") or "1024")
        )
        ctk.CTkEntry(row, textvariable=self.long_var, width=90).grid(
            row=0, column=5, sticky="w"
        )

        # ----- Buttons -------------------------------------------------
        bf = ctk.CTkFrame(self, fg_color="transparent")
        bf.pack(fill="x", padx=16, pady=(4, 6))
        self.btn_probe = ctk.CTkButton(bf, text="Probe", width=110, command=self._on_probe)
        self.btn_probe.pack(side="left", padx=4)
        self.btn_run = ctk.CTkButton(bf, text="Extract", width=140, command=self._on_run)
        self.btn_run.pack(side="left", padx=4)
        self.btn_stop = ctk.CTkButton(bf, text="Stop", width=100, fg_color="#B14545",
                                      hover_color="#8f2f2f", command=self._on_stop,
                                      state="disabled")
        self.btn_stop.pack(side="left", padx=4)

        # ----- Progress ------------------------------------------------
        self.progress = ProgressPanel(self)
        self.progress.pack(fill="x", padx=16, pady=(6, 6))

        # ----- Summary text (list of detected videos) -----------------
        self.summary = ctk.CTkTextbox(self, height=140, wrap="word",
                                      font=("Consolas", 11))
        self.summary.pack(fill="both", expand=True, padx=16, pady=(0, 8))
        self.summary.configure(state="disabled")

    # ------------------------------------------------------------------
    def _write_summary(self, text: str) -> None:
        self.summary.configure(state="normal")
        self.summary.delete("1.0", "end")
        self.summary.insert("end", text)
        self.summary.configure(state="disabled")

    def _on_probe(self) -> None:
        src = resolve(self.pick_input.get())
        vids = list(iter_video_files(src))
        if not vids:
            self._write_summary(f"No videos found under {src}")
            return
        lines = [f"Found {len(vids)} video(s):"]
        for v in vids:
            try:
                info = probe(v)
                lines.append(
                    f"  * {v.name}  |  {info.width}x{info.height}  |  "
                    f"{info.fps:.2f} fps  |  {info.frame_count} frames  |  "
                    f"{info.duration_sec:.1f} s"
                )
            except Exception as e:  # noqa: BLE001
                lines.append(f"  * {v.name}  [error: {e}]")
        self._write_summary("\n".join(lines))

    def _parse_int(self, s: str, default: int) -> int:
        try:
            return int(float(s))
        except (TypeError, ValueError):
            return default

    def _parse_float(self, s: str, default: float) -> float:
        try:
            return float(s)
        except (TypeError, ValueError):
            return default

    def _on_run(self) -> None:
        src = resolve(self.pick_input.get())
        out = resolve(self.pick_out.get())
        fpm = self._parse_float(self.fpm_var.get(), 6.0)
        fmt = self.fmt_var.get()
        long_side_raw = (self.long_var.get() or "").strip()
        long_side = self._parse_int(long_side_raw, 0) if long_side_raw else 0

        vids = list(iter_video_files(src))
        if not vids:
            self._write_summary(f"No videos found under {src}")
            return

        self.btn_run.configure(state="disabled")
        self.btn_probe.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.progress.reset("Starting...")

        def _progress(done: int, total: int, msg: str) -> None:
            self.after(0, lambda: self.progress.set_progress(done, total, msg))

        def _job():
            extract_folder(
                videos_root=src,
                output_root=out,
                frames_per_minute=fpm,
                image_format=fmt,
                jpeg_quality=int(self.config_obj.get("extraction.jpeg_quality", 92)),
                resize_long_side=long_side or None,
                progress=_progress,
                should_stop=self.worker.should_stop,
            )

        def _done(err):
            self.after(0, lambda: self._on_done(err))

        self.worker.start(_job, on_done=_done)

    def _on_stop(self) -> None:
        self.worker.request_stop()
        self.progress.set_status("Stopping...")

    def _on_done(self, err) -> None:
        self.btn_run.configure(state="normal")
        self.btn_probe.configure(state="normal")
        self.btn_stop.configure(state="disabled")
        if err is not None:
            self.progress.set_status(f"Error: {err}")
            log.exception("Extraction failed", exc_info=err)
        else:
            self.progress.set_status("Done")
