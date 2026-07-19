"""Main application window."""

from __future__ import annotations

import tkinter as tk

import customtkinter as ctk

from ..core.config import AppConfig
from ..utils.gpu import describe_torch, read_system
from ..utils.logger import get_logger
from .tab_extract import ExtractTab
from .tab_finetune import FineTuneTab
from .tab_inference import InferenceTab
from .tab_settings import SettingsTab
from .tab_validate import ValidateTab
from .widgets import LogConsole

log = get_logger(__name__)


class App(ctk.CTk):
    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self.config_obj = config
        self.title("VLM Pipeline - Video Scene Captioning")
        self.geometry("1180x820")
        self.minsize(1000, 700)

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self._build_layout()
        self.after(2000, self._tick_status)

    # ------------------------------------------------------------------
    def _build_layout(self) -> None:
        # ---- Top status bar ------------------------------------------
        top = ctk.CTkFrame(self, corner_radius=0, height=44)
        top.pack(side="top", fill="x")
        top.pack_propagate(False)

        title = ctk.CTkLabel(top, text="VLM Pipeline",
                             font=ctk.CTkFont(size=16, weight="bold"))
        title.pack(side="left", padx=14)

        self.status_var = tk.StringVar(value="")
        status = ctk.CTkLabel(top, textvariable=self.status_var,
                              font=("Consolas", 11), anchor="e")
        status.pack(side="right", padx=14)

        # ---- Tab view ------------------------------------------------
        self.tabs = ctk.CTkTabview(self)
        self.tabs.pack(fill="both", expand=True, padx=10, pady=(6, 4))

        self.tabs.add("1. Extract")
        self.tabs.add("2. Inference")
        self.tabs.add("3. Fine-tune")
        self.tabs.add("4. Validate")
        self.tabs.add("Settings")

        self.extract_tab = ExtractTab(self.tabs.tab("1. Extract"), self.config_obj)
        self.extract_tab.pack(fill="both", expand=True)

        self.inference_tab = InferenceTab(self.tabs.tab("2. Inference"), self.config_obj)
        self.inference_tab.pack(fill="both", expand=True)

        self.finetune_tab = FineTuneTab(self.tabs.tab("3. Fine-tune"), self.config_obj)
        self.finetune_tab.pack(fill="both", expand=True)

        self.validate_tab = ValidateTab(self.tabs.tab("4. Validate"), self.config_obj)
        self.validate_tab.pack(fill="both", expand=True)

        self.settings_tab = SettingsTab(self.tabs.tab("Settings"), self.config_obj)
        self.settings_tab.pack(fill="both", expand=True)

        # ---- Log console at bottom -----------------------------------
        self.log_console = LogConsole(self, height=180)
        self.log_console.pack(fill="x", padx=10, pady=(4, 8))

    # ------------------------------------------------------------------
    def _tick_status(self) -> None:
        s = read_system()
        parts = [describe_torch()]
        parts.append(f"CPU {s.cpu_pct:.0f}%  |  RAM {s.ram_used_gb:.1f}/{s.ram_total_gb:.1f}GB")
        for g in s.gpus:
            parts.append(f"{g.name.split()[-1] if g.name else 'GPU'} "
                         f"{g.used_mb}/{g.total_mb}MB ({g.used_pct:.0f}%)")
        self.status_var.set("   |   ".join(parts))
        self.after(2000, self._tick_status)


def run(config: AppConfig) -> None:
    app = App(config)
    app.mainloop()
