"""GUI tab: validation — auto metrics + visual human review."""

from __future__ import annotations

import json
import tkinter as tk
from pathlib import Path
from tkinter import messagebox

import customtkinter as ctk
from PIL import Image

from ..core.config import AppConfig
from ..core.ratings import (
    aggregate_ratings,
    build_comparison_session,
    find_paired_json,
    frame_image_path,
    list_synced_json_pairs,
    pair_index_for_json,
    save_frame_rating,
    summarize_session,
)
from ..core.validate import validate
from ..utils.logger import get_logger
from ..utils.paths import ensure_dir, resolve
from .widgets import PathPicker, ProgressPanel, Worker

log = get_logger(__name__)


class ValidateTab(ctk.CTkFrame):
    def __init__(self, master, config: AppConfig, **kwargs):
        super().__init__(master, **kwargs)
        self.config_obj = config
        self.worker = Worker()
        self._last_result = None
        self._review_session: dict | None = None
        self._review_index = 0
        self._ctk_image = None
        self._pil_image = None
        self._sync_pairs: list[tuple[Path, Path]] = []
        self._sync_index = -1
        self._sync_busy = False
        self._build()

    # ------------------------------------------------------------------
    def _build(self) -> None:
        ctk.CTkLabel(
            self, text="Step 4 - Validation",
            font=ctk.CTkFont(size=18, weight="bold"),
        ).pack(anchor="w", padx=16, pady=(16, 4))

        self.tabs = ctk.CTkTabview(self)
        self.tabs.pack(fill="both", expand=True, padx=12, pady=(4, 8))
        self.tabs.add("Auto metrics")
        self.tabs.add("Visual review")

        self._build_auto_tab(self.tabs.tab("Auto metrics"))
        self._build_review_tab(self.tabs.tab("Visual review"))

    # ==================================================================
    # Auto metrics (BLEU / ROUGE / cosine)
    # ==================================================================
    def _build_auto_tab(self, parent) -> None:
        ctk.CTkLabel(
            parent, justify="left",
            text="Compare two output JSON files with automatic text metrics.",
        ).pack(anchor="w", padx=8, pady=(8, 4))

        pf = ctk.CTkFrame(parent)
        pf.pack(fill="x", padx=8, pady=4)
        jf = [("JSON", "*.json")]
        self.pick_ref = PathPicker(
            pf, label="Reference JSON (model A):", kind="openfile",
            pattern=jf, initial=str(self.config_obj.path_for("outputs_dir")),
        )
        self.pick_ref.pack(fill="x", padx=6, pady=4)
        self.pick_hyp = PathPicker(
            pf, label="Hypothesis JSON (model B):", kind="openfile",
            pattern=jf, initial=str(self.config_obj.path_for("outputs_dir")),
        )
        self.pick_hyp.pack(fill="x", padx=6, pady=4)

        row = ctk.CTkFrame(pf, fg_color="transparent")
        row.pack(fill="x", padx=6, pady=6)
        self.chk_bleu = ctk.CTkCheckBox(row, text="BLEU-4")
        self.chk_rouge = ctk.CTkCheckBox(row, text="ROUGE-L")
        self.chk_cos = ctk.CTkCheckBox(row, text="Cosine (MiniLM)")
        for w in (self.chk_bleu, self.chk_rouge, self.chk_cos):
            w.select()
            w.pack(side="left", padx=8)

        bf = ctk.CTkFrame(parent, fg_color="transparent")
        bf.pack(fill="x", padx=8, pady=4)
        self.btn_run = ctk.CTkButton(bf, text="Run validation", width=150,
                                       command=self._on_run_metrics)
        self.btn_run.pack(side="left", padx=4)
        self.btn_export = ctk.CTkButton(bf, text="Export report...", width=140,
                                        command=self._on_export)
        self.btn_export.pack(side="left", padx=4)

        self.progress_auto = ProgressPanel(parent)
        self.progress_auto.pack(fill="x", padx=8, pady=4)

        self.results = ctk.CTkTextbox(parent, wrap="word", font=("Consolas", 11),
                                      height=200)
        self.results.pack(fill="both", expand=True, padx=8, pady=(4, 8))
        self.results.configure(state="disabled")

    # ==================================================================
    # Visual review (image + human ratings 0-5)
    # ==================================================================
    def _build_review_tab(self, parent) -> None:
        ctk.CTkLabel(
            parent, justify="left",
            text=("Load two model output JSONs for the same video, browse frames "
                  "with the reference image, and rate each description 0-5."),
        ).pack(anchor="w", padx=8, pady=(8, 4))

        pf = ctk.CTkFrame(parent)
        pf.pack(fill="x", padx=8, pady=4)
        outputs_root = str(self.config_obj.path_for("outputs_dir"))
        self.pick_out_a = PathPicker(
            pf, label="Model A outputs folder:", kind="dir",
            initial=outputs_root,
        )
        self.pick_out_a.pack(fill="x", padx=6, pady=4)
        self.pick_out_b = PathPicker(
            pf, label="Model B outputs folder:", kind="dir",
            initial=outputs_root,
        )
        self.pick_out_b.pack(fill="x", padx=6, pady=4)

        sync_row = ctk.CTkFrame(pf, fg_color="transparent")
        sync_row.pack(fill="x", padx=6, pady=(0, 4))
        self.chk_sync = ctk.CTkCheckBox(
            sync_row, text="Sync outputs (auto-pair JSONs by filename)",
            command=self._on_sync_toggle,
        )
        self.chk_sync.select()
        self.chk_sync.pack(side="left", padx=4)

        jf = [("JSON", "*.json")]
        self.pick_rev_a = PathPicker(
            pf, label="Model A JSON:", kind="openfile", pattern=jf,
            initial=outputs_root,
        )
        self.pick_rev_a.pack(fill="x", padx=6, pady=4)
        self.pick_rev_b = PathPicker(
            pf, label="Model B JSON:", kind="openfile", pattern=jf,
            initial=outputs_root,
        )
        self.pick_rev_b.pack(fill="x", padx=6, pady=4)
        self.pick_rev_a.var.trace_add("write", lambda *_: self._on_json_a_changed())
        self.pick_out_a.var.trace_add("write", lambda *_: self._on_sync_folders_changed())
        self.pick_out_b.var.trace_add("write", lambda *_: self._on_sync_folders_changed())

        bf = ctk.CTkFrame(parent, fg_color="transparent")
        bf.pack(fill="x", padx=8, pady=4)
        ctk.CTkButton(bf, text="Load comparison", width=150,
                        command=self._on_load_review).pack(side="left", padx=4)
        ctk.CTkButton(bf, text="Aggregate all videos", width=160,
                        command=self._on_aggregate).pack(side="left", padx=4)

        self.video_nav = ctk.CTkFrame(parent, fg_color="transparent")
        self.video_nav.pack(fill="x", padx=8, pady=(0, 4))
        ctk.CTkButton(
            self.video_nav, text="◀ Prev video", width=110,
            command=lambda: self._step_sync_video(-1),
        ).pack(side="left", padx=4)
        self.video_counter = ctk.CTkLabel(self.video_nav, text="")
        self.video_counter.pack(side="left", padx=8)
        ctk.CTkButton(
            self.video_nav, text="Next video ▶", width=110,
            command=lambda: self._step_sync_video(1),
        ).pack(side="left", padx=4)
        self.video_nav.pack_forget()

        self._review_body = ctk.CTkFrame(parent)
        self._review_body.pack(fill="both", expand=True, padx=8, pady=4)
        body = self._review_body
        body.grid_columnconfigure(0, weight=1)
        body.grid_columnconfigure(1, weight=2)
        body.grid_rowconfigure(0, weight=1)

        # Left: image preview
        left = ctk.CTkFrame(body)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        self.img_label = ctk.CTkLabel(left, text="(no image)", width=360, height=270)
        self.img_label.pack(padx=8, pady=8)

        nav = ctk.CTkFrame(left, fg_color="transparent")
        nav.pack(fill="x", padx=8, pady=4)
        ctk.CTkButton(nav, text="Prev", width=70, command=self._prev_frame).pack(
            side="left", padx=4
        )
        self.frame_counter = ctk.CTkLabel(nav, text="0 / 0")
        self.frame_counter.pack(side="left", padx=8)
        ctk.CTkButton(nav, text="Next", width=70, command=self._next_frame).pack(
            side="left", padx=4
        )

        self.frame_meta = ctk.CTkLabel(left, text="", anchor="w", justify="left",
                                     font=("Consolas", 10))
        self.frame_meta.pack(anchor="w", padx=10, pady=(0, 8))

        # Right: descriptions + ratings
        right = ctk.CTkFrame(body)
        right.grid(row=0, column=1, sticky="nsew")
        self.lbl_model_a = ctk.CTkLabel(right, text="Model A", anchor="w",
                                        font=ctk.CTkFont(weight="bold"))
        self.lbl_model_a.pack(anchor="w", padx=10, pady=(8, 0))
        self.lbl_meta_a = ctk.CTkLabel(
            right, text="", anchor="w", justify="left", wraplength=520,
            font=("Consolas", 10),
            text_color=("#555555", "#B0B0B0"),
        )
        self.lbl_meta_a.pack(anchor="w", padx=10, pady=(0, 4))
        self.txt_desc_a = ctk.CTkTextbox(right, height=80, wrap="word")
        self.txt_desc_a.pack(fill="x", padx=10, pady=4)
        self.txt_desc_a.configure(state="disabled")

        self.lbl_model_b = ctk.CTkLabel(right, text="Model B", anchor="w",
                                        font=ctk.CTkFont(weight="bold"))
        self.lbl_model_b.pack(anchor="w", padx=10, pady=(8, 0))
        self.lbl_meta_b = ctk.CTkLabel(
            right, text="", anchor="w", justify="left", wraplength=520,
            font=("Consolas", 10),
            text_color=("#555555", "#B0B0B0"),
        )
        self.lbl_meta_b.pack(anchor="w", padx=10, pady=(0, 4))
        self.txt_desc_b = ctk.CTkTextbox(right, height=80, wrap="word")
        self.txt_desc_b.pack(fill="x", padx=10, pady=4)
        self.txt_desc_b.configure(state="disabled")

        rate_row = ctk.CTkFrame(right, fg_color="transparent")
        rate_row.pack(fill="x", padx=10, pady=8)
        ctk.CTkLabel(rate_row, text="Rate A (0-5):").grid(row=0, column=0, padx=4)
        self.rating_a_var = tk.IntVar(value=3)
        self.slider_a = ctk.CTkSlider(rate_row, from_=0, to=5, number_of_steps=5,
                                      variable=self.rating_a_var, width=180)
        self.slider_a.grid(row=0, column=1, padx=4)
        self.lbl_rating_a = ctk.CTkLabel(rate_row, text="3", width=24)
        self.lbl_rating_a.grid(row=0, column=2, padx=4)
        self.slider_a.configure(command=lambda v: self.lbl_rating_a.configure(
            text=str(int(round(float(v))))))

        ctk.CTkLabel(rate_row, text="Rate B (0-5):").grid(row=1, column=0, padx=4, pady=6)
        self.rating_b_var = tk.IntVar(value=3)
        self.slider_b = ctk.CTkSlider(rate_row, from_=0, to=5, number_of_steps=5,
                                      variable=self.rating_b_var, width=180)
        self.slider_b.grid(row=1, column=1, padx=4, pady=6)
        self.lbl_rating_b = ctk.CTkLabel(rate_row, text="3", width=24)
        self.lbl_rating_b.grid(row=1, column=2, padx=4, pady=6)
        self.slider_b.configure(command=lambda v: self.lbl_rating_b.configure(
            text=str(int(round(float(v))))))

        ctk.CTkButton(right, text="Save rating for this frame", width=200,
                      command=self._save_rating).pack(anchor="w", padx=10, pady=4)

        self.review_summary = ctk.CTkLabel(
            right, text="", anchor="w", justify="left",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=("#2b6fd5", "#7BAAF7"),
        )
        self.review_summary.pack(anchor="w", padx=10, pady=(8, 8))

        self.after(100, self._try_auto_load_from_folders)

    # ------------------------------------------------------------------
    # Auto metrics handlers
    # ------------------------------------------------------------------
    def _write_results(self, text: str) -> None:
        self.results.configure(state="normal")
        self.results.delete("1.0", "end")
        self.results.insert("end", text)
        self.results.configure(state="disabled")

    def _on_run_metrics(self) -> None:
        ref = resolve(self.pick_ref.get())
        hyp = resolve(self.pick_hyp.get())
        metrics = []
        if self.chk_bleu.get():
            metrics.append("bleu")
        if self.chk_rouge.get():
            metrics.append("rougeL")
        if self.chk_cos.get():
            metrics.append("cosine")
        if not metrics:
            self._write_results("Select at least one metric.")
            return

        self.progress_auto.reset("Computing metrics...")
        self.btn_run.configure(state="disabled")

        def _job():
            res = validate(
                ref, hyp, metrics,
                embed_model=self.config_obj.get(
                    "validation.sentence_embed_model",
                    "sentence-transformers/all-MiniLM-L6-v2",
                ),
            )
            self._last_result = res
            self.after(0, lambda: self._render_metrics(res))

        def _done(err):
            self.after(0, lambda: self._on_metrics_done(err))

        self.worker.start(_job, on_done=_done)

    def _render_metrics(self, res) -> None:
        def fmt(x): return "n/a" if x is None else f"{x:.4f}"
        lines = [
            f"Pairs compared: {res.n_pairs}",
            f"BLEU-4  : {fmt(res.bleu)}",
            f"ROUGE-L : {fmt(res.rougeL)}",
            f"Cosine  : {fmt(res.cosine)}",
            "",
            "First 5 samples:",
        ]
        for row in (res.per_frame or [])[:5]:
            lines.append("-" * 60)
            lines.append(f"  file: {row.get('file')} @ t={row.get('timestamp_sec')}s")
            lines.append(f"  ref:  {row.get('reference')}")
            lines.append(f"  hyp:  {row.get('hypothesis')}")
        self._write_results("\n".join(lines))

    def _on_metrics_done(self, err) -> None:
        self.btn_run.configure(state="normal")
        if err is not None:
            log.exception("Validation failed", exc_info=err)
            self.progress_auto.set_status(f"Error: {err}")
        else:
            self.progress_auto.set_status("Done")

    def _on_export(self) -> None:
        if not self._last_result:
            self._write_results("Run validation first.")
            return
        out_dir = ensure_dir(self.config_obj.path_for("outputs_dir") / "reports")
        out = out_dir / "validation_report.json"
        with out.open("w", encoding="utf-8") as fh:
            json.dump(self._last_result.to_dict(), fh, indent=2, ensure_ascii=False)
        self.progress_auto.set_status(f"Report saved: {out}")

    # ------------------------------------------------------------------
    # Visual review handlers
    # ------------------------------------------------------------------
    def _sync_enabled(self) -> bool:
        return bool(self.chk_sync.get())

    def _rebuild_sync_pairs(self) -> None:
        if not self._sync_enabled():
            self._sync_pairs = []
            self._sync_index = -1
            self._update_video_nav()
            return
        dir_a = self.pick_out_a.get()
        dir_b = self.pick_out_b.get()
        if not dir_a or not dir_b:
            self._sync_pairs = []
            self._sync_index = -1
            self._update_video_nav()
            return
        self._sync_pairs = list_synced_json_pairs(dir_a, dir_b)
        current = self.pick_rev_a.get()
        if current:
            idx = pair_index_for_json(self._sync_pairs, current)
            self._sync_index = idx if idx >= 0 else 0
        elif self._sync_pairs:
            self._sync_index = 0
        else:
            self._sync_index = -1
        self._update_video_nav()

    def _update_video_nav(self) -> None:
        if self._sync_enabled() and self._sync_pairs:
            if not self.video_nav.winfo_ismapped():
                self.video_nav.pack(fill="x", padx=8, pady=(0, 4),
                                   before=self._review_body)
            idx = max(0, min(self._sync_index, len(self._sync_pairs) - 1))
            stem = self._sync_pairs[idx][0].stem
            self.video_counter.configure(
                text=f"Video {idx + 1} / {len(self._sync_pairs)}  ({stem})"
            )
        else:
            self.video_nav.pack_forget()
            self.video_counter.configure(text="")

    def _on_sync_toggle(self) -> None:
        self._rebuild_sync_pairs()
        if self._sync_enabled():
            self._try_auto_load_from_folders()

    def _on_sync_folders_changed(self) -> None:
        if self._sync_busy:
            return
        self._rebuild_sync_pairs()
        if self._sync_enabled():
            self._try_auto_load_from_folders()

    def _both_output_folders_set(self) -> bool:
        return bool(self.pick_out_a.get().strip() and self.pick_out_b.get().strip())

    def _try_auto_load_from_folders(self) -> None:
        """When sync is on and both output folders are set, load the first pair."""
        if not self._sync_enabled() or not self._both_output_folders_set():
            return
        self._rebuild_sync_pairs()
        if not self._sync_pairs:
            return
        idx = self._sync_index if 0 <= self._sync_index < len(self._sync_pairs) else 0
        self._apply_sync_pair(idx, auto_load=True)

    @staticmethod
    def _format_model_header(side: str, meta: dict) -> str:
        model_id = meta.get("id", side)
        parts = [f"Model {side}: {model_id}"]
        precision = meta.get("precision")
        if precision:
            parts.append(str(precision))
        avg = meta.get("avg_inference_sec")
        if avg is not None:
            parts.append(f"avg {avg:.2f}s/frame")
        return "  |  ".join(parts)

    @staticmethod
    def _format_model_meta(meta: dict) -> str:
        prompt = (meta.get("prompt") or "").strip()
        if not prompt:
            return "Prompt: (not recorded in JSON)"
        if len(prompt) > 240:
            prompt = prompt[:237] + "..."
        return f"Prompt: {prompt}"

    def _on_json_a_changed(self) -> None:
        if self._sync_busy or not self._sync_enabled():
            return
        self._sync_pair_from_a(auto_load=False)

    def _sync_pair_from_a(self, *, auto_load: bool) -> None:
        json_a = self.pick_rev_a.get()
        if not json_a:
            return
        self._rebuild_sync_pairs()
        idx = pair_index_for_json(self._sync_pairs, json_a)
        if idx < 0:
            paired = find_paired_json(json_a, self.pick_out_b.get())
            if paired is None:
                self._update_video_nav()
                return
            self._sync_busy = True
            try:
                self.pick_rev_b.set(str(paired))
            finally:
                self._sync_busy = False
            idx = pair_index_for_json(self._sync_pairs, json_a)
        if idx >= 0:
            self._sync_index = idx
            self._update_video_nav()
        if auto_load:
            self._on_load_review()

    def _apply_sync_pair(self, index: int, *, auto_load: bool) -> None:
        if not self._sync_pairs:
            return
        index = max(0, min(index, len(self._sync_pairs) - 1))
        json_a, json_b = self._sync_pairs[index]
        self._sync_index = index
        self._sync_busy = True
        try:
            self.pick_rev_a.set(str(json_a))
            self.pick_rev_b.set(str(json_b))
        finally:
            self._sync_busy = False
        self._update_video_nav()
        if auto_load:
            self._on_load_review()

    def _step_sync_video(self, delta: int) -> None:
        if not self._sync_pairs:
            self._rebuild_sync_pairs()
        if not self._sync_pairs:
            messagebox.showinfo(
                "No synced videos",
                "Enable Sync outputs and choose two output folders that contain "
                "matching JSON filenames.",
            )
            return
        start = self._sync_index if self._sync_index >= 0 else 0
        nxt = (start + delta) % len(self._sync_pairs)
        self._apply_sync_pair(nxt, auto_load=True)

    def _on_load_review(self) -> None:
        if self._sync_enabled():
            self._sync_pair_from_a(auto_load=False)
        try:
            session = build_comparison_session(
                self.pick_rev_a.get(), self.pick_rev_b.get()
            )
        except Exception as e:
            messagebox.showerror("Load failed", str(e))
            return
        if not session.get("comparisons"):
            messagebox.showwarning("No frames", "No aligned frames between the two JSONs.")
            return

        self._review_session = session
        self._review_index = 0
        meta_a = session["model_a"]
        meta_b = session["model_b"]
        self.lbl_model_a.configure(text=self._format_model_header("A", meta_a))
        self.lbl_model_b.configure(text=self._format_model_header("B", meta_b))
        self.lbl_meta_a.configure(text=self._format_model_meta(meta_a))
        self.lbl_meta_b.configure(text=self._format_model_meta(meta_b))
        self._show_review_frame()
        self._update_review_summary()

    def _show_review_frame(self) -> None:
        if not self._review_session:
            return
        comps = self._review_session["comparisons"]
        n = len(comps)
        if n == 0:
            return
        idx = max(0, min(self._review_index, n - 1))
        self._review_index = idx
        row = comps[idx]

        self.frame_counter.configure(text=f"{idx + 1} / {n}")
        self.frame_meta.configure(
            text=f"file: {row.get('file')}\n"
                 f"time: {row.get('timestamp_hhmmss')} ({row.get('timestamp_sec')}s)"
        )

        # Descriptions (before image so a preview failure cannot block text)
        for tb, key in ((self.txt_desc_a, "description_a"),
                        (self.txt_desc_b, "description_b")):
            tb.configure(state="normal")
            tb.delete("1.0", "end")
            tb.insert("1.0", row.get(key, ""))
            tb.configure(state="disabled")

        # Image — CTkImage requires a PIL.Image reference kept alive.
        img_path = frame_image_path(self._review_session, idx)
        self._pil_image = None
        self._ctk_image = None
        if img_path and img_path.is_file():
            try:
                pil = Image.open(img_path).convert("RGB")
                pil.thumbnail((360, 270), Image.Resampling.LANCZOS)
                self._pil_image = pil
                self._ctk_image = ctk.CTkImage(
                    light_image=self._pil_image,
                    dark_image=self._pil_image,
                    size=self._pil_image.size,
                )
                self.img_label.configure(image=self._ctk_image, text="")
            except Exception as exc:  # noqa: BLE001
                log.warning("Failed to load preview image %s: %s", img_path, exc)
                self.img_label.configure(
                    image=None,
                    text=f"(image load error)\n{row.get('file')}\n{exc}",
                )
        else:
            tried = self._review_session.get("frames_dirs") or []
            hint = tried[0] if tried else "(no frames_dir in JSON)"
            self.img_label.configure(
                image=None,
                text=f"(image not found)\n{row.get('file')}\n{hint}",
            )

        # Existing ratings
        ra = row.get("rating_a")
        rb = row.get("rating_b")
        if ra is not None:
            self.rating_a_var.set(int(ra))
            self.lbl_rating_a.configure(text=str(int(ra)))
        if rb is not None:
            self.rating_b_var.set(int(rb))
            self.lbl_rating_b.configure(text=str(int(rb)))

    def _prev_frame(self) -> None:
        if self._review_session and self._review_index > 0:
            self._review_index -= 1
            self._show_review_frame()

    def _next_frame(self) -> None:
        if self._review_session:
            n = len(self._review_session["comparisons"])
            if self._review_index < n - 1:
                self._review_index += 1
                self._show_review_frame()

    def _save_rating(self) -> None:
        if not self._review_session:
            messagebox.showinfo("Load first", "Load a comparison before rating.")
            return
        ra = int(round(self.rating_a_var.get()))
        rb = int(round(self.rating_b_var.get()))
        try:
            save_frame_rating(
                self._review_session,
                self._review_index,
                ra, rb,
                self.config_obj.path_for("outputs_dir"),
            )
        except Exception as e:
            messagebox.showerror("Save failed", str(e))
            return
        self._update_review_summary()
        messagebox.showinfo("Saved", f"Rating saved: A={ra}, B={rb}")

    def _update_review_summary(self) -> None:
        if not self._review_session:
            return
        s = summarize_session(self._review_session)
        model_a = self._review_session["model_a"]["id"]
        model_b = self._review_session["model_b"]["id"]
        if s["n_rated"] == 0:
            text = f"Rated {s['n_rated']}/{s['n_total']} frames in this video."
        else:
            text = (
                f"This video: {s['n_rated']}/{s['n_total']} rated  |  "
                f"avg {model_a}={s['avg_rating_a']:.2f}  "
                f"avg {model_b}={s['avg_rating_b']:.2f}  |  "
                f"wins A={s['wins_a']}  B={s['wins_b']}  ties={s['ties']}"
            )
        self.review_summary.configure(text=text)

    def _on_aggregate(self) -> None:
        if not self._review_session:
            messagebox.showinfo("Load first", "Load a comparison to identify the model pair.")
            return
        model_a = self._review_session["model_a"]["id"]
        model_b = self._review_session["model_b"]["id"]
        agg = aggregate_ratings(
            self.config_obj.path_for("outputs_dir"), model_a, model_b
        )
        if agg.get("n_rated_frames", 0) == 0:
            messagebox.showinfo(
                "No ratings yet",
                f"No rated frames found for {model_a} vs {model_b}.\n"
                "Rate some frames first, then try again.",
            )
            return
        msg = (
            f"Aggregate across {agg['n_sessions']} video(s), "
            f"{agg['n_rated_frames']} rated frame(s):\n\n"
            f"  {model_a}: avg {agg['avg_rating_a']:.2f}\n"
            f"  {model_b}: avg {agg['avg_rating_b']:.2f}"
        )
        messagebox.showinfo("Aggregate ratings", msg)
