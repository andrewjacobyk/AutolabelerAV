"""GUI tab: run VLM inference over extracted frames."""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import messagebox

import customtkinter as ctk

from ..core.config import AppConfig
from ..core.inference import run_inference_on_frames, run_inference_on_many
from ..core.model_hub import cache_status_for_config, download_from_spec, is_cached
from ..core.resources import estimate, format_short, format_table
from ..utils.gpu import read_gpus
from ..utils.logger import get_logger
from ..utils.paths import resolve
from .widgets import PathPicker, ProgressPanel, Worker

log = get_logger(__name__)


class InferenceTab(ctk.CTkFrame):
    def __init__(self, master, config: AppConfig, **kwargs):
        super().__init__(master, **kwargs)
        self.config_obj = config
        self.worker = Worker()
        self._last_timing: dict | None = None
        self._build()

    # ------------------------------------------------------------------
    def _build(self) -> None:
        ctk.CTkLabel(
            self, text="Step 2 - Describe frames with a VLM",
            font=ctk.CTkFont(size=18, weight="bold"),
        ).pack(anchor="w", padx=16, pady=(16, 4))

        ctk.CTkLabel(
            self,
            text=("Loads an open-weight vision-language model and generates "
                  "one description per frame. Results are stored as JSON."),
            justify="left",
        ).pack(anchor="w", padx=16, pady=(0, 12))

        pf = ctk.CTkFrame(self)
        pf.pack(fill="x", padx=16, pady=6)

        self.pick_frames = PathPicker(
            pf, label="Frames folder:", kind="dir",
            initial=str(self.config_obj.path_for("frames_dir")),
        )
        self.pick_frames.pack(fill="x", padx=6, pady=4)

        self.pick_out = PathPicker(
            pf, label="Outputs folder (JSON):", kind="dir",
            initial=str(self.config_obj.path_for("outputs_dir")),
        )
        self.pick_out.pack(fill="x", padx=6, pady=4)

        # ----- Model selector -----------------------------------------
        row1 = ctk.CTkFrame(pf, fg_color="transparent")
        row1.pack(fill="x", padx=6, pady=(6, 2))
        row1.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(row1, text="Model:", width=110, anchor="w").grid(
            row=0, column=0, sticky="w", padx=4
        )
        self.model_var = tk.StringVar(
            value=self.config_obj.get("inference.active_model", "moondream2")
        )
        model_ids = self.config_obj.all_model_ids()
        self.model_menu = ctk.CTkOptionMenu(
            row1, values=model_ids, variable=self.model_var,
            command=self._on_model_change,
        )
        self.model_menu.grid(row=0, column=1, sticky="ew", padx=4)

        ctk.CTkLabel(row1, text="Precision:", width=90, anchor="w").grid(
            row=0, column=2, sticky="w", padx=(12, 4)
        )
        self.prec_var = tk.StringVar(
            value=self.config_obj.get("inference.precision", "fp16")
        )
        ctk.CTkOptionMenu(
            row1, values=["fp16", "bf16", "fp32", "int8", "int4"],
            variable=self.prec_var, width=110, command=self._on_prec_change,
        ).grid(row=0, column=3, padx=4)

        ctk.CTkLabel(row1, text="Max tokens:", width=100, anchor="w").grid(
            row=0, column=4, sticky="w", padx=(12, 4)
        )
        self.tokens_var = tk.StringVar(
            value=str(self.config_obj.get("inference.max_new_tokens", 96))
        )
        ctk.CTkEntry(row1, textvariable=self.tokens_var, width=70).grid(
            row=0, column=5, padx=4
        )

        # ----- Prompt --------------------------------------------------
        ctk.CTkLabel(pf, text="Prompt:").pack(anchor="w", padx=10, pady=(8, 0))
        self.prompt_txt = ctk.CTkTextbox(pf, height=80, wrap="word")
        self.prompt_txt.pack(fill="x", padx=8, pady=(0, 8))
        self.prompt_txt.insert(
            "1.0", self.config_obj.get("inference.prompt", "Describe this scene.")
        )

        # ----- Model info + resource estimate -------------------------
        self.model_info = ctk.CTkLabel(
            pf, text="", anchor="w", justify="left",
            text_color=("#555555", "#B0B0B0"),
        )
        self.model_info.pack(anchor="w", padx=10, pady=(0, 2))

        self.resource_info = ctk.CTkLabel(
            pf, text="", anchor="w", justify="left",
            text_color=("#2b6fd5", "#7BAAF7"),
            font=ctk.CTkFont(size=12, weight="bold"),
        )
        self.resource_info.pack(anchor="w", padx=10, pady=(0, 4))

        # ----- Local cache status for all models --------------------
        cache_hdr = ctk.CTkFrame(pf, fg_color="transparent")
        cache_hdr.pack(fill="x", padx=10, pady=(4, 0))
        ctk.CTkLabel(
            cache_hdr, text="Local weights:",
            font=ctk.CTkFont(size=12, weight="bold"),
        ).pack(side="left")
        ctk.CTkButton(
            cache_hdr, text="Refresh", width=70, height=24,
            command=self._refresh_cache_list,
        ).pack(side="right", padx=4)

        self.cache_list = ctk.CTkTextbox(pf, height=72, font=("Consolas", 10))
        self.cache_list.pack(fill="x", padx=8, pady=(2, 6))
        self.cache_list.configure(state="disabled")

        # ----- Timing from last inference run -----------------------
        self.timing_info = ctk.CTkLabel(
            pf, text="", anchor="w", justify="left",
            text_color=("#1a7f4b", "#5fd68a"),
            font=ctk.CTkFont(size=12, weight="bold"),
        )
        self.timing_info.pack(anchor="w", padx=10, pady=(0, 6))

        self._refresh_model_info()
        self._refresh_cache_list()

        # ----- Buttons -------------------------------------------------
        bf = ctk.CTkFrame(self, fg_color="transparent")
        bf.pack(fill="x", padx=16, pady=(4, 6))
        self.btn_single = ctk.CTkButton(
            bf, text="Run on single folder", width=180,
            command=lambda: self._run(False),
        )
        self.btn_single.pack(side="left", padx=4)
        self.btn_batch = ctk.CTkButton(
            bf, text="Run on ALL subfolders", width=200,
            command=lambda: self._run(True),
        )
        self.btn_batch.pack(side="left", padx=4)
        self.btn_download = ctk.CTkButton(
            bf, text="Download weights only", width=170, command=self._on_download,
        )
        self.btn_download.pack(side="left", padx=4)
        self.btn_resources = ctk.CTkButton(
            bf, text="Resource table...", width=130, command=self._show_resource_table,
        )
        self.btn_resources.pack(side="left", padx=4)
        self.btn_stop = ctk.CTkButton(
            bf, text="Stop", width=90, fg_color="#B14545",
            hover_color="#8f2f2f", state="disabled", command=self._on_stop,
        )
        self.btn_stop.pack(side="left", padx=4)

        self.progress = ProgressPanel(self)
        self.progress.pack(fill="x", padx=16, pady=(4, 8))

    # ------------------------------------------------------------------
    def _on_model_change(self, *_):
        self._refresh_model_info()

    def _on_prec_change(self, *_):
        self._refresh_model_info()

    def _refresh_model_info(self) -> None:
        model_id = self.model_var.get()
        try:
            spec = self.config_obj.model_spec(model_id)
        except Exception as e:
            self.model_info.configure(text=f"[unknown model: {e}]")
            self.resource_info.configure(text="")
            return

        family = spec.get("family", "?")
        info = (
            f"HF id = {spec.get('hf_id')}"
            + (f"  |  revision = {spec['revision']}" if spec.get("revision") else "")
            + f"  |  family = {family}"
        )
        self.model_info.configure(text=info)

        est = estimate(model_id, self.prec_var.get(), family=family)
        gpus = read_gpus()
        vram_gb = gpus[0].total_mb / 1024 if gpus else 0.0

        if not est.supported:
            colour = ("#B14545", "#F08A8A")
            text = f"[!]  {est.note}"
        elif vram_gb and not est.fits_in(vram_gb):
            colour = ("#B14545", "#F08A8A")
            text = (
                f"[!]  This won't fit on your GPU: "
                f"{format_short(est)}  vs installed {vram_gb:.1f} GB VRAM"
            )
        else:
            colour = ("#2b6fd5", "#7BAAF7")
            text = f"Estimated: {format_short(est)}"
            if gpus:
                text += f"  (GPU {vram_gb:.1f} GB total)"

        self.resource_info.configure(text=text, text_color=colour)

        # Highlight selected model cache state in the status line.
        hf_id = spec.get("hf_id", "")
        cached = is_cached(hf_id, spec.get("revision"))
        status = "downloaded locally" if cached else "needs download"
        self.model_info.configure(text=info + f"  |  weights: {status}")

    def _refresh_cache_list(self) -> None:
        models = self.config_obj.get("models", {}) or {}
        status = cache_status_for_config(models)
        lines = []
        for mid in self.config_obj.all_model_ids():
            st = status.get(mid, "missing")
            icon = "[OK] " if st == "cached" else "[--] "
            lines.append(f"{icon}{mid}")
        cached_n = sum(1 for s in status.values() if s == "cached")
        header = f"{cached_n}/{len(lines)} models ready locally\n"
        self.cache_list.configure(state="normal")
        self.cache_list.delete("1.0", "end")
        self.cache_list.insert("end", header + "\n".join(lines))
        self.cache_list.configure(state="disabled")

    def _set_timing_display(self, timing: dict | None) -> None:
        if not timing:
            self.timing_info.configure(text="")
            return
        elapsed = timing.get("elapsed_sec", 0)
        n = timing.get("num_frames", 0)
        avg = timing.get("avg_sec_per_frame", 0)
        self.timing_info.configure(
            text=(
                f"Last run: {elapsed:.1f}s total  |  {n} frame(s)  |  "
                f"avg {avg:.2f}s per image"
            )
        )

    def _show_resource_table(self) -> None:
        models = self.config_obj.get("models", {}) or {}
        table = format_table(models)
        win = ctk.CTkToplevel(self)
        win.title("VRAM / disk estimates per model")
        win.geometry("820x520")
        txt = ctk.CTkTextbox(win, font=("Consolas", 11))
        txt.pack(fill="both", expand=True, padx=10, pady=10)
        txt.insert("end", table + "\n\n")
        txt.insert(
            "end",
            "Columns show estimated peak VRAM (GB) at each precision.\n"
            "'n/a' = precision not supported by that backend.\n"
            "Add ~0.5 GB headroom for the CUDA driver on an 8 GB card.\n"
            "Disk is always the fp16 weight size; int4/int8 still download fp16 shards.\n",
        )
        txt.configure(state="disabled")

    def _on_download(self) -> None:
        model_id = self.model_var.get()
        try:
            spec = self.config_obj.model_spec(model_id)
        except Exception as e:
            messagebox.showerror("Config error", str(e))
            return

        self.btn_download.configure(state="disabled")
        self.btn_single.configure(state="disabled")
        self.btn_batch.configure(state="disabled")
        self.progress.reset(f"Downloading {model_id} ...")

        def _progress(msg: str) -> None:
            self.after(0, lambda: self.progress.set_status(msg))

        def _job():
            download_from_spec(spec, progress=_progress)

        def _done(err):
            self.after(0, lambda: self._on_download_done(err))

        self.worker.start(_job, on_done=_done)

    def _on_download_done(self, err) -> None:
        self.btn_download.configure(state="normal")
        self.btn_single.configure(state="normal")
        self.btn_batch.configure(state="normal")
        if err is not None:
            log.exception("Download failed", exc_info=err)
            self.progress.set_status(f"Download error: {err}")
            messagebox.showerror(
                "Download failed",
                f"{err}\n\nTips:\n"
                "- Let the download finish; closing the app restarts it.\n"
                "- Set HUGGINGFACE_TOKEN in Settings for faster downloads.\n"
                "- Enable Windows Developer Mode to fix symlink warnings.\n"
                "- Check logs/vla_pipeline.log for details.",
            )
        else:
            self.progress.set_status("Download complete")
        self._refresh_cache_list()
        self._refresh_model_info()

    # ------------------------------------------------------------------
    def _run(self, batch: bool) -> None:
        frames = resolve(self.pick_frames.get())
        outputs = resolve(self.pick_out.get())
        outputs.mkdir(parents=True, exist_ok=True)
        model_id = self.model_var.get()
        try:
            spec = self.config_obj.model_spec(model_id)
        except Exception as e:
            self.progress.set_status(f"Config error: {e}")
            return
        prompt = self.prompt_txt.get("1.0", "end").strip()
        precision = self.prec_var.get()
        try:
            max_new_tokens = int(float(self.tokens_var.get()))
        except (TypeError, ValueError):
            max_new_tokens = 96

        # -- Frames-folder sanity check -------------------------------
        if batch:
            if not frames.exists() or not any(
                p.is_dir() and (p / "manifest.json").exists()
                for p in frames.iterdir()
            ):
                messagebox.showerror(
                    "No frames found",
                    f"'{frames}' does not contain any per-video subfolder with a "
                    "manifest.json.\n\nExtract frames in Tab 1 first, then point "
                    "this picker at the parent frames folder.",
                )
                return
        else:
            if not (frames / "manifest.json").exists():
                messagebox.showerror(
                    "Missing manifest.json",
                    f"'{frames}' does not contain a manifest.json.\n\n"
                    "For a single-folder run, pick the extracted-frames "
                    "subfolder of ONE video (e.g. data/frames/my_video), "
                    "not the parent frames directory.",
                )
                return

        # -- Preflight resource check ---------------------------------
        est = estimate(model_id, precision, family=spec.get("family"))
        gpus = read_gpus()
        vram_gb = gpus[0].total_mb / 1024 if gpus else 0.0

        if not est.supported:
            messagebox.showerror("Unsupported precision", est.note)
            return

        if vram_gb and not est.fits_in(vram_gb):
            proceed = messagebox.askyesno(
                title="Model may not fit in VRAM",
                message=(
                    f"'{model_id}' at {precision} needs about "
                    f"{est.vram_gb:.1f} GB of VRAM,\n"
                    f"but your GPU only has {vram_gb:.1f} GB.\n\n"
                    "Loading will most likely fail with OutOfMemory.\n"
                    "Recommended: switch precision to int4 or pick a smaller "
                    "model.\n\nContinue anyway?"
                ),
                icon="warning",
            )
            if not proceed:
                self.progress.set_status("Cancelled - model too large.")
                return

        # -- Kick off the worker --------------------------------------
        self.progress.reset("Loading model...")
        self.btn_single.configure(state="disabled")
        self.btn_batch.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self._run_batch = batch
        self._run_frames = frames
        self._run_outputs = outputs

        def _progress(done: int, total: int, msg: str) -> None:
            self.after(0, lambda: self.progress.set_progress(done, total, msg))

        def _job():
            if batch:
                return run_inference_on_many(
                    frames_root=frames,
                    outputs_dir=outputs,
                    model_id=model_id,
                    model_spec=spec,
                    prompt=prompt,
                    precision=precision,
                    max_new_tokens=max_new_tokens,
                    progress=_progress,
                    should_stop=self.worker.should_stop,
                )
            out_json = Path(outputs) / f"{Path(frames).name}.json"
            run_inference_on_frames(
                frames_dir=frames,
                output_json=out_json,
                model_id=model_id,
                model_spec=spec,
                prompt=prompt,
                precision=precision,
                max_new_tokens=max_new_tokens,
                progress=_progress,
                should_stop=self.worker.should_stop,
            )
            return out_json

        def _done(err):
            self.after(0, lambda: self._on_done(err))

        self.worker.start(_job, on_done=_done)

    def _on_stop(self) -> None:
        self.worker.request_stop()
        self.progress.set_status("Stopping...")

    def _on_done(self, err) -> None:
        self.btn_single.configure(state="normal")
        self.btn_batch.configure(state="normal")
        self.btn_stop.configure(state="disabled")
        if err is not None:
            log.exception("Inference failed", exc_info=err)
            self.progress.set_status(f"Error: {err}")
            self._set_timing_display(None)
        else:
            self.progress.set_status("Done")
            self._load_timing_from_outputs()

    def _load_timing_from_outputs(self) -> None:
        """Read timing stats from the JSON written by the last inference run."""
        from ..core import dataset as ds

        try:
            if getattr(self, "_run_batch", False):
                outputs = Path(self._run_outputs)
                total_sec = 0.0
                total_frames = 0
                for jp in sorted(outputs.glob("*.json")):
                    if "ratings" in jp.parts or "reports" in jp.parts:
                        continue
                    doc = ds.load(jp)
                    t = doc.get("timing") or {}
                    total_sec += float(t.get("elapsed_sec", doc.get("elapsed_sec", 0)))
                    total_frames += int(t.get("num_frames", len(doc.get("frames", []))))
                if total_frames:
                    self._set_timing_display({
                        "elapsed_sec": round(total_sec, 3),
                        "num_frames": total_frames,
                        "avg_sec_per_frame": round(total_sec / total_frames, 3),
                    })
                return

            frames = Path(self._run_frames)
            out_json = Path(self._run_outputs) / f"{frames.name}.json"
            if out_json.exists():
                doc = ds.load(out_json)
                timing = doc.get("timing")
                if timing:
                    self._set_timing_display(timing)
                elif doc.get("elapsed_sec"):
                    n = len(doc.get("frames", []))
                    elapsed = float(doc["elapsed_sec"])
                    self._set_timing_display({
                        "elapsed_sec": elapsed,
                        "num_frames": n,
                        "avg_sec_per_frame": round(elapsed / n, 3) if n else 0,
                    })
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not read timing from output JSON: %s", exc)
