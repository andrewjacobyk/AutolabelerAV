"""Reusable CustomTkinter widgets used across tabs."""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from tkinter import filedialog
from typing import Callable, Iterable, Optional

import customtkinter as ctk

from ..utils.logger import GUI_LOG_QUEUE


# ---------------------------------------------------------------------------
# Log console -- polls the global GUI_LOG_QUEUE and appends lines.
# ---------------------------------------------------------------------------
class LogConsole(ctk.CTkFrame):
    def __init__(self, master, height: int = 180, **kwargs):
        super().__init__(master, **kwargs)
        self.textbox = ctk.CTkTextbox(self, height=height, wrap="word",
                                      font=("Consolas", 11))
        self.textbox.pack(fill="both", expand=True, padx=6, pady=6)
        self.textbox.configure(state="disabled")
        self._poll()

    def _poll(self) -> None:
        drained = 0
        try:
            while drained < 200:
                msg = GUI_LOG_QUEUE.get_nowait()
                self._append(msg)
                drained += 1
        except queue.Empty:
            pass
        self.after(120, self._poll)

    def _append(self, line: str) -> None:
        self.textbox.configure(state="normal")
        self.textbox.insert("end", line + "\n")
        self.textbox.see("end")
        self.textbox.configure(state="disabled")


# ---------------------------------------------------------------------------
# Progress bar with textual status line.
# ---------------------------------------------------------------------------
class ProgressPanel(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.status_var = tk.StringVar(value="Idle")
        self.label = ctk.CTkLabel(self, textvariable=self.status_var, anchor="w")
        self.label.pack(fill="x", padx=8, pady=(6, 0))
        self.bar = ctk.CTkProgressBar(self)
        self.bar.pack(fill="x", padx=8, pady=(2, 6))
        self.bar.set(0.0)

    def set_status(self, message: str) -> None:
        self.status_var.set(message)

    def set_progress(self, done: int, total: int, message: str = "") -> None:
        pct = 0.0 if total <= 0 else max(0.0, min(1.0, done / total))
        self.bar.set(pct)
        if message:
            self.set_status(f"{done}/{total} - {message}")
        else:
            self.set_status(f"{done}/{total}")

    def reset(self, status: str = "Idle") -> None:
        self.bar.set(0.0)
        self.status_var.set(status)


# ---------------------------------------------------------------------------
# Path picker (Entry + Browse button)
# ---------------------------------------------------------------------------
class PathPicker(ctk.CTkFrame):
    def __init__(self, master, *, label: str, initial: str = "",
                 kind: str = "dir", pattern: Optional[Iterable[tuple[str, str]]] = None,
                 **kwargs):
        super().__init__(master, **kwargs)
        self.kind = kind
        self.pattern = pattern or [("All files", "*.*")]
        self.var = tk.StringVar(value=initial)

        ctk.CTkLabel(self, text=label, width=150, anchor="w").grid(
            row=0, column=0, sticky="w", padx=(6, 4), pady=4
        )
        self.entry = ctk.CTkEntry(self, textvariable=self.var)
        self.entry.grid(row=0, column=1, sticky="ew", padx=4, pady=4)
        self.btn = ctk.CTkButton(self, text="Browse...", width=90, command=self._pick)
        self.btn.grid(row=0, column=2, padx=(4, 6), pady=4)
        self.grid_columnconfigure(1, weight=1)

    def _pick(self) -> None:
        if self.kind == "dir":
            path = filedialog.askdirectory(initialdir=self.var.get() or ".")
        elif self.kind == "openfile":
            path = filedialog.askopenfilename(initialdir=".",
                                              filetypes=list(self.pattern))
        elif self.kind == "savefile":
            path = filedialog.asksaveasfilename(initialdir=".",
                                                filetypes=list(self.pattern))
        else:
            path = ""
        if path:
            self.var.set(path)

    def get(self) -> str:
        return self.var.get().strip()

    def set(self, value: str) -> None:
        self.var.set(value)


# ---------------------------------------------------------------------------
# Worker thread wrapper. Publishes progress via a callable.
# ---------------------------------------------------------------------------
class Worker:
    def __init__(self) -> None:
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def start(self, target: Callable[..., None], *args, on_done: Optional[Callable[[Optional[BaseException]], None]] = None) -> bool:
        if self._thread and self._thread.is_alive():
            return False
        self._stop.clear()

        def _runner():
            err: Optional[BaseException] = None
            try:
                target(*args)
            except BaseException as e:  # noqa: BLE001
                err = e
            finally:
                if on_done is not None:
                    on_done(err)

        self._thread = threading.Thread(target=_runner, daemon=True)
        self._thread.start()
        return True

    def request_stop(self) -> None:
        self._stop.set()

    def should_stop(self) -> bool:
        return self._stop.is_set()

    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())
