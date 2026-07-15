"""GUI tab: settings & environment info."""

from __future__ import annotations

import os
import tkinter as tk
from pathlib import Path

import customtkinter as ctk

from ..core.config import AppConfig
from ..core.resources import format_table
from ..utils.gpu import describe_torch, read_system
from ..utils.logger import get_logger
from .widgets import PathPicker

log = get_logger(__name__)


class SettingsTab(ctk.CTkFrame):
    def __init__(self, master, config: AppConfig, **kwargs):
        super().__init__(master, **kwargs)
        self.config_obj = config
        self._build()

    # ------------------------------------------------------------------
    def _build(self) -> None:
        ctk.CTkLabel(
            self, text="Settings",
            font=ctk.CTkFont(size=18, weight="bold"),
        ).pack(anchor="w", padx=16, pady=(16, 4))

        # --- Paths -----------------------------------------------------
        pf = ctk.CTkFrame(self)
        pf.pack(fill="x", padx=16, pady=(6, 6))
        ctk.CTkLabel(pf, text="Paths",
                     font=ctk.CTkFont(size=14, weight="bold")).pack(
            anchor="w", padx=8, pady=(6, 0))
        self.pickers = {}
        for key, label in [
            ("videos_dir",  "Videos:"),
            ("frames_dir",  "Frames:"),
            ("outputs_dir", "Outputs:"),
            ("datasets_dir", "Datasets:"),
            ("models_dir",  "Models cache:"),
            ("logs_dir",    "Logs:"),
        ]:
            pk = PathPicker(pf, label=label, kind="dir",
                            initial=str(self.config_obj.path_for(key)))
            pk.pack(fill="x", padx=6, pady=2)
            self.pickers[key] = pk

        # --- Hugging Face access token --------------------------------
        af = ctk.CTkFrame(self)
        af.pack(fill="x", padx=16, pady=(6, 6))
        ctk.CTkLabel(af, text="Hugging Face access (session-only)",
                     font=ctk.CTkFont(size=14, weight="bold")).pack(
            anchor="w", padx=8, pady=(6, 0))
        ctk.CTkLabel(af, justify="left",
                     text=("Only needed to download gated open-weight models "
                           "(e.g. PaliGemma). For persistence, copy "
                           ".env.example -> .env and set HUGGINGFACE_TOKEN there."),
                     text_color=("#555555", "#B0B0B0")
                     ).pack(anchor="w", padx=8, pady=(0, 6))

        self.key_vars = {}
        row = ctk.CTkFrame(af, fg_color="transparent")
        row.pack(fill="x", padx=6, pady=2)
        ctk.CTkLabel(row, text="HuggingFace token:", width=170, anchor="w").pack(side="left", padx=4)
        var = tk.StringVar(value=os.getenv("HUGGINGFACE_TOKEN", ""))
        ctk.CTkEntry(row, textvariable=var, show="*").pack(
            side="left", fill="x", expand=True, padx=4
        )
        self.key_vars["HUGGINGFACE_TOKEN"] = var

        # --- Buttons ---------------------------------------------------
        bf = ctk.CTkFrame(self, fg_color="transparent")
        bf.pack(fill="x", padx=16, pady=(6, 6))
        ctk.CTkButton(bf, text="Apply", width=110, command=self._on_apply).pack(side="left", padx=4)
        ctk.CTkButton(bf, text="Save to config.yaml", width=170, command=self._on_save).pack(side="left", padx=4)

        # --- Environment + resources ---------------------------------
        info = ctk.CTkFrame(self)
        info.pack(fill="both", expand=True, padx=16, pady=(6, 8))
        ctk.CTkLabel(info, text="Environment",
                     font=ctk.CTkFont(size=14, weight="bold")).pack(
            anchor="w", padx=8, pady=(6, 0))
        self.env_txt = ctk.CTkTextbox(info, font=("Consolas", 11), wrap="word",
                                      height=100)
        self.env_txt.pack(fill="x", padx=6, pady=6)

        ctk.CTkLabel(info, text="Estimated VRAM by model x precision",
                     font=ctk.CTkFont(size=14, weight="bold")).pack(
            anchor="w", padx=8, pady=(4, 0))
        ctk.CTkLabel(
            info, justify="left",
            text=("Weights + KV cache + CUDA overhead, approximate. "
                  "'n/a' = precision not wired up for that family."),
            text_color=("#555555", "#B0B0B0"),
        ).pack(anchor="w", padx=8, pady=(0, 4))
        self.res_txt = ctk.CTkTextbox(info, font=("Consolas", 11), wrap="none")
        self.res_txt.pack(fill="both", expand=True, padx=6, pady=(0, 6))
        self._refresh_resources()

        self._refresh_env()
        self.after(3000, self._tick_env)

    # ------------------------------------------------------------------
    def _refresh_env(self) -> None:
        s = read_system()
        parts = [describe_torch(),
                 f"CPU: {s.cpu_pct:.0f}%   RAM: {s.ram_used_gb:.1f} / {s.ram_total_gb:.1f} GB"]
        for g in s.gpus:
            parts.append(
                f"GPU{g.index} {g.name} | VRAM {g.used_mb}/{g.total_mb} MB "
                f"({g.used_pct:.0f}%) | util {g.utilization}%"
            )
        if not s.gpus:
            parts.append("No NVIDIA GPU detected (or NVML unavailable).")
        self.env_txt.configure(state="normal")
        self.env_txt.delete("1.0", "end")
        self.env_txt.insert("end", "\n".join(parts))
        self.env_txt.configure(state="disabled")

    def _refresh_resources(self) -> None:
        models = self.config_obj.get("models", {}) or {}
        table = format_table(models)
        self.res_txt.configure(state="normal")
        self.res_txt.delete("1.0", "end")
        self.res_txt.insert("end", table)
        self.res_txt.configure(state="disabled")

    def _tick_env(self) -> None:
        self._refresh_env()
        self.after(3000, self._tick_env)

    def _on_apply(self) -> None:
        for k, pk in self.pickers.items():
            self.config_obj.set(f"paths.{k}", pk.get())
        for env_name, var in self.key_vars.items():
            val = var.get().strip()
            if val:
                os.environ[env_name] = val
        log.info("Settings applied for current session.")

    def _on_save(self) -> None:
        self._on_apply()
        path = self.config_obj.save()
        log.info("Config saved to %s", path)
