"""GUI tab: LoRA fine-tuning."""

from __future__ import annotations

import tkinter as tk
from pathlib import Path

import customtkinter as ctk

from ..core import dataset as ds
from ..core.config import AppConfig
from ..core.finetune import FineTuneConfig, check_support, run_finetune
from ..utils.logger import get_logger
from ..utils.paths import ensure_dir, resolve
from .widgets import PathPicker, ProgressPanel, Worker

log = get_logger(__name__)


class FineTuneTab(ctk.CTkFrame):
    def __init__(self, master, config: AppConfig, **kwargs):
        super().__init__(master, **kwargs)
        self.config_obj = config
        self.worker = Worker()
        self._build()

    # ------------------------------------------------------------------
    def _build(self) -> None:
        ctk.CTkLabel(
            self, text="Step 3 - Fine-tune (LoRA)",
            font=ctk.CTkFont(size=18, weight="bold"),
        ).pack(anchor="w", padx=16, pady=(16, 4))
        ctk.CTkLabel(
            self, justify="left",
            text=("Use inference outputs as a supervised dataset. "
                  "Only local models with a documented training path are supported."),
        ).pack(anchor="w", padx=16, pady=(0, 12))

        pf = ctk.CTkFrame(self)
        pf.pack(fill="x", padx=16, pady=6)

        self.pick_outputs = PathPicker(
            pf, label="Inference outputs folder:", kind="dir",
            initial=str(self.config_obj.path_for("outputs_dir")),
        )
        self.pick_outputs.pack(fill="x", padx=6, pady=4)

        self.pick_datasets = PathPicker(
            pf, label="Dataset (JSONL) output:", kind="dir",
            initial=str(self.config_obj.path_for("datasets_dir")),
        )
        self.pick_datasets.pack(fill="x", padx=6, pady=4)

        self.pick_frames = PathPicker(
            pf, label="Frames root (for images):", kind="dir",
            initial=str(self.config_obj.path_for("frames_dir")),
        )
        self.pick_frames.pack(fill="x", padx=6, pady=4)

        row = ctk.CTkFrame(pf, fg_color="transparent")
        row.pack(fill="x", padx=6, pady=(6, 4))
        row.grid_columnconfigure((1, 3, 5), weight=1)

        ctk.CTkLabel(row, text="Base model:").grid(row=0, column=0, sticky="w", padx=4)
        self.base_var = tk.StringVar(
            value=self.config_obj.get("finetune.base_model", "moondream2")
        )
        ctk.CTkOptionMenu(row, values=self.config_obj.local_model_ids(),
                          variable=self.base_var).grid(row=0, column=1, sticky="ew", padx=(0, 12))

        ctk.CTkLabel(row, text="Epochs:").grid(row=0, column=2, sticky="w", padx=4)
        self.epochs_var = tk.StringVar(
            value=str(self.config_obj.get("finetune.epochs", 3))
        )
        ctk.CTkEntry(row, textvariable=self.epochs_var, width=90).grid(
            row=0, column=3, sticky="w", padx=(0, 12)
        )

        ctk.CTkLabel(row, text="Learning rate:").grid(row=0, column=4, sticky="w", padx=4)
        self.lr_var = tk.StringVar(
            value=str(self.config_obj.get("finetune.learning_rate", 1e-4))
        )
        ctk.CTkEntry(row, textvariable=self.lr_var, width=100).grid(row=0, column=5, sticky="w")

        row2 = ctk.CTkFrame(pf, fg_color="transparent")
        row2.pack(fill="x", padx=6, pady=(2, 6))
        row2.grid_columnconfigure((1, 3, 5, 7), weight=1)

        ctk.CTkLabel(row2, text="LoRA r:").grid(row=0, column=0, sticky="w", padx=4)
        self.r_var = tk.StringVar(value=str(self.config_obj.get("finetune.lora_r", 8)))
        ctk.CTkEntry(row2, textvariable=self.r_var, width=70).grid(row=0, column=1, padx=(0, 12))

        ctk.CTkLabel(row2, text="alpha:").grid(row=0, column=2, sticky="w", padx=4)
        self.alpha_var = tk.StringVar(value=str(self.config_obj.get("finetune.lora_alpha", 16)))
        ctk.CTkEntry(row2, textvariable=self.alpha_var, width=70).grid(row=0, column=3, padx=(0, 12))

        ctk.CTkLabel(row2, text="dropout:").grid(row=0, column=4, sticky="w", padx=4)
        self.drop_var = tk.StringVar(value=str(self.config_obj.get("finetune.lora_dropout", 0.05)))
        ctk.CTkEntry(row2, textvariable=self.drop_var, width=70).grid(row=0, column=5, padx=(0, 12))

        ctk.CTkLabel(row2, text="grad accum:").grid(row=0, column=6, sticky="w", padx=4)
        self.ga_var = tk.StringVar(value=str(self.config_obj.get("finetune.gradient_accumulation", 8)))
        ctk.CTkEntry(row2, textvariable=self.ga_var, width=70).grid(row=0, column=7, padx=(0, 4))

        bf = ctk.CTkFrame(self, fg_color="transparent")
        bf.pack(fill="x", padx=16, pady=(4, 6))
        self.btn_build = ctk.CTkButton(
            bf, text="1. Build dataset from outputs", width=230, command=self._on_build_dataset
        )
        self.btn_build.pack(side="left", padx=4)
        self.btn_train = ctk.CTkButton(
            bf, text="2. Start fine-tune", width=170, command=self._on_train
        )
        self.btn_train.pack(side="left", padx=4)
        self.btn_stop = ctk.CTkButton(
            bf, text="Stop", width=90, fg_color="#B14545",
            hover_color="#8f2f2f", state="disabled", command=self._on_stop,
        )
        self.btn_stop.pack(side="left", padx=4)

        self.progress = ProgressPanel(self)
        self.progress.pack(fill="x", padx=16, pady=(4, 8))

        self.info = ctk.CTkTextbox(self, height=140, wrap="word",
                                   font=("Consolas", 11))
        self.info.pack(fill="both", expand=True, padx=16, pady=(0, 8))
        self.info.configure(state="disabled")

    # ------------------------------------------------------------------
    def _log_info(self, text: str) -> None:
        self.info.configure(state="normal")
        self.info.insert("end", text + "\n")
        self.info.see("end")
        self.info.configure(state="disabled")

    def _on_build_dataset(self) -> None:
        outputs_dir = resolve(self.pick_outputs.get())
        datasets_dir = ensure_dir(self.pick_datasets.get())
        frames_root = resolve(self.pick_frames.get())

        docs = [ds.load(p) for p in ds.iter_output_files(outputs_dir)]
        if not docs:
            self._log_info(f"No JSON outputs found under {outputs_dir}")
            return
        jsonl = datasets_dir / "train.jsonl"
        n = ds.merge_documents_to_dataset(docs, jsonl, frames_root)
        self._log_info(f"Built {n} pairs -> {jsonl}")

    def _on_train(self) -> None:
        model_id = self.base_var.get()
        try:
            spec = self.config_obj.model_spec(model_id)
        except Exception as e:
            self._log_info(f"Config error: {e}")
            return
        err = check_support(spec)
        if err:
            self._log_info(f"Unsupported: {err}")
            return

        datasets_dir = resolve(self.pick_datasets.get())
        jsonl = datasets_dir / "train.jsonl"
        if not jsonl.exists():
            self._log_info(f"Dataset not found: {jsonl}. Build it first.")
            return

        try:
            cfg = FineTuneConfig(
                base_model=model_id,
                model_spec=spec,
                dataset_jsonl=jsonl,
                output_dir=resolve(self.config_obj.get("finetune.output_dir",
                                                     "data/models/finetuned")),
                lora_r=int(float(self.r_var.get())),
                lora_alpha=int(float(self.alpha_var.get())),
                lora_dropout=float(self.drop_var.get()),
                epochs=int(float(self.epochs_var.get())),
                learning_rate=float(self.lr_var.get()),
                gradient_accumulation=int(float(self.ga_var.get())),
            )
        except ValueError as e:
            self._log_info(f"Invalid hyperparameter: {e}")
            return

        self.progress.reset("Loading base model...")
        self.btn_train.configure(state="disabled")
        self.btn_build.configure(state="disabled")
        self.btn_stop.configure(state="normal")

        def _progress(done: int, total: int, msg: str) -> None:
            self.after(0, lambda: self.progress.set_progress(done, total, msg))

        def _job():
            out = run_finetune(cfg, progress=_progress,
                               should_stop=self.worker.should_stop)
            self.after(0, lambda: self._log_info(f"Fine-tune saved -> {out}"))

        def _done(err):
            self.after(0, lambda: self._on_done(err))

        self.worker.start(_job, on_done=_done)

    def _on_stop(self) -> None:
        self.worker.request_stop()
        self.progress.set_status("Stopping...")

    def _on_done(self, err) -> None:
        self.btn_train.configure(state="normal")
        self.btn_build.configure(state="normal")
        self.btn_stop.configure(state="disabled")
        if err is not None:
            log.exception("Fine-tune failed", exc_info=err)
            self.progress.set_status(f"Error: {err}")
            self._log_info(f"Error: {err}")
        else:
            self.progress.set_status("Done")
