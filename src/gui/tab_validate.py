"""GUI tab: validation of a hypothesis output vs. reference."""

from __future__ import annotations

import json
import tkinter as tk
from pathlib import Path

import customtkinter as ctk

from ..core.config import AppConfig
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
        self._build()

    # ------------------------------------------------------------------
    def _build(self) -> None:
        ctk.CTkLabel(
            self, text="Step 4 - Validation",
            font=ctk.CTkFont(size=18, weight="bold"),
        ).pack(anchor="w", padx=16, pady=(16, 4))
        ctk.CTkLabel(
            self, justify="left",
            text=("Compare a candidate output document against a reference one "
                  "(BLEU-4 / ROUGE-L / MiniLM cosine similarity)."),
        ).pack(anchor="w", padx=16, pady=(0, 12))

        pf = ctk.CTkFrame(self)
        pf.pack(fill="x", padx=16, pady=6)
        json_filter = [("JSON", "*.json")]
        self.pick_ref = PathPicker(pf, label="Reference JSON:", kind="openfile",
                                   pattern=json_filter,
                                   initial=str(self.config_obj.path_for("outputs_dir")))
        self.pick_ref.pack(fill="x", padx=6, pady=4)
        self.pick_hyp = PathPicker(pf, label="Hypothesis JSON:", kind="openfile",
                                   pattern=json_filter,
                                   initial=str(self.config_obj.path_for("outputs_dir")))
        self.pick_hyp.pack(fill="x", padx=6, pady=4)

        row = ctk.CTkFrame(pf, fg_color="transparent")
        row.pack(fill="x", padx=6, pady=6)
        self.chk_bleu = ctk.CTkCheckBox(row, text="BLEU-4")
        self.chk_rouge = ctk.CTkCheckBox(row, text="ROUGE-L")
        self.chk_cos = ctk.CTkCheckBox(row, text="Cosine (MiniLM)")
        for w in (self.chk_bleu, self.chk_rouge, self.chk_cos):
            w.select()
            w.pack(side="left", padx=8)

        bf = ctk.CTkFrame(self, fg_color="transparent")
        bf.pack(fill="x", padx=16, pady=(4, 6))
        self.btn_run = ctk.CTkButton(bf, text="Run validation", width=170, command=self._on_run)
        self.btn_run.pack(side="left", padx=4)
        self.btn_export = ctk.CTkButton(bf, text="Export report...", width=150, command=self._on_export)
        self.btn_export.pack(side="left", padx=4)

        self.progress = ProgressPanel(self)
        self.progress.pack(fill="x", padx=16, pady=(4, 8))

        self.results = ctk.CTkTextbox(self, wrap="word", font=("Consolas", 11))
        self.results.pack(fill="both", expand=True, padx=16, pady=(0, 8))
        self.results.configure(state="disabled")

        self._last_result = None

    # ------------------------------------------------------------------
    def _write(self, text: str) -> None:
        self.results.configure(state="normal")
        self.results.delete("1.0", "end")
        self.results.insert("end", text)
        self.results.configure(state="disabled")

    def _on_run(self) -> None:
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
            self._write("Select at least one metric.")
            return

        self.progress.reset("Computing metrics...")
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
            self.after(0, lambda: self._render_result(res))

        def _done(err):
            self.after(0, lambda: self._on_done(err))

        self.worker.start(_job, on_done=_done)

    def _render_result(self, res) -> None:
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
        self._write("\n".join(lines))

    def _on_done(self, err) -> None:
        self.btn_run.configure(state="normal")
        if err is not None:
            log.exception("Validation failed", exc_info=err)
            self.progress.set_status(f"Error: {err}")
        else:
            self.progress.set_status("Done")

    def _on_export(self) -> None:
        if not self._last_result:
            self._write("Run validation first.")
            return
        out_dir = ensure_dir(self.config_obj.path_for("outputs_dir") / "reports")
        out = out_dir / "validation_report.json"
        with out.open("w", encoding="utf-8") as fh:
            json.dump(self._last_result.to_dict(), fh, indent=2, ensure_ascii=False)
        self.progress.set_status(f"Report saved: {out}")
