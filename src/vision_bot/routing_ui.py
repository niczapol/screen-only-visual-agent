from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable


class RoutingControlPanel:
    def __init__(self, on_toggle: Callable[[bool], None] | None = None) -> None:
        self.root = tk.Tk()
        self.root.title("Vision Mining Router")
        self.root.geometry("320x220")
        self.enabled = tk.BooleanVar(value=False)
        self.on_toggle = on_toggle
        self.ore_status_label: ttk.Label | None = None
        self.status_label: ttk.Label | None = None

        self._build_ui()

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        main = ttk.Frame(self.root, padding=12)
        main.grid(row=0, column=0, sticky="nsew")

        ttk.Label(main, text="Mining route controller", font=("Segoe UI", 12, "bold")).grid(row=0, column=0, sticky="w")

        toggle = ttk.Checkbutton(
            main,
            text="Включить передвижение",
            variable=self.enabled,
            command=self._handle_toggle,
        )
        toggle.grid(row=1, column=0, pady=(10, 8), sticky="w")

        ttk.Label(main, text="Обнаружение руды на миникарте:").grid(row=2, column=0, sticky="w")
        self.ore_status_label = ttk.Label(main, text="Нет", foreground="green")
        self.ore_status_label.grid(row=3, column=0, pady=(4, 10), sticky="w")

        self.status_label = ttk.Label(main, text="Готов", foreground="gray")
        self.status_label.grid(row=4, column=0, pady=(0, 10), sticky="w")

        ttk.Button(main, text="Закрыть", command=self.root.destroy).grid(row=5, column=0, sticky="w")

    def _handle_toggle(self) -> None:
        if self.on_toggle is not None:
            self.on_toggle(self.enabled.get())

    def set_ore_detected(self, detected: bool) -> None:
        if self.ore_status_label is not None:
            self.ore_status_label.config(text="Да" if detected else "Нет")

    def set_status(self, text: str) -> None:
        if self.status_label is not None:
            self.status_label.config(text=text)

    def start(self) -> None:
        self.root.mainloop()
