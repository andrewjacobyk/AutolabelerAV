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
    frame_image_path,
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
        jf = [("JSON", "*.json")]
        self.pick_rev_a = PathPicker(
            pf, label="Model A JSON:", kind="openfile", pattern=jf,
            initial=str(self.config_obj.path_for("outputs_dir")),
        )
        self.pick_rev_a.pack(fill="x", padx=6, pady=4)
        self.pick_rev_b = PathPicker(
            pf, label="Model B JSON:", kind="openfile", pattern=jf,
            initial=str(self.config_obj.path_for("outputs_dir")),
        )
        self.pick_rev_b.pack(fill="x", padx=6, pady=4)

        bf = ctk.CTkFrame(parent, fg_color="transparent")
        bf.pack(fill="x", padx=8, pady=4)
        ctk.CTkButton(bf, text="Load comparison", width=150,
                        command=self._on_load_review).pack(side="left", padx=4)
        ctk.CTkButton(bf, text="Aggregate all videos", width=160,
                        command=self._on_aggregate).pack(side="left", padx=4)

        body = ctk.CTkFrame(parent)
        body.pack(fill="both", expand=True, padx=8, pady=4)
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
        self.txt_desc_a = ctk.CTkTextbox(right, height=80, wrap="word")
        self.txt_desc_a.pack(fill="x", padx=10, pady=4)
        self.txt_desc_a.configure(state="disabled")

        self.lbl_model_b = ctk.CTkLabel(right, text="Model B", anchor="w",
                                        font=ctk.CTkFont(weight="bold"))
        self.lbl_model_b.pack(anchor="w", padx=10, pady=(8, 0))
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
    def _on_load_review(self) -> None:
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
        model_a = session["model_a"]["id"]
        model_b = session["model_b"]["id"]
        self.lbl_model_a.configure(text=f"Model A: {model_a}")
        self.lbl_model_b.configure(text=f"Model B: {model_b}")
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

        # Image
        img_path = frame_image_path(self._review_session, idx)
        if img_path and img_path.exists():
            pil = Image.open(img_path).convert("RGB")
            max_w, max_h = 360, 270
            pil.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
            self._ctk_image = ctk.CTkImage(light_image=pil, dark_image=pil,
                                           size=pil.size)
            self.img_label.configure(image=self._ctk_image, text="")
        else:
            self._ctk_image = None
            self.img_label.configure(image=None, text="(image not found)")

        # Descriptions
        for tb, key in ((self.txt_desc_a, "description_a"),
                        (self.txt_desc_b, "description_b")):
            tb.configure(state="normal")
            tb.delete("1.0", "end")
            tb.insert("1.0", row.get(key, ""))
            tb.configure(state="disabled")

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
