"""
First-launch dependency checker.

Imports of heavy third-party packages (mediapipe, torch, customtkinter, …) are
deferred. We import-probe each one against requirements.txt and, if any are
missing, show a stdlib-only tkinter popup asking the user whether to install
them. The popup uses the standard library so the check itself cannot fail.

Call ensure_dependencies() at the very top of the main script, before any
third-party imports.
"""
from __future__ import annotations

import importlib
import os
import re
import subprocess
import sys
import tkinter as tk
from tkinter import scrolledtext, ttk
from typing import List, Tuple


# Map pip distribution name → import name when they differ.
# Anything not listed is assumed to import under the same name.
PIP_TO_IMPORT = {
    "opencv-python": "cv2",
    "Pillow": "PIL",
    "scikit-learn": "sklearn",
    "openai-whisper": "whisper",
    "faster-whisper": "faster_whisper",
    "sentence-transformers": "sentence_transformers",
    "umap-learn": "umap",
    "huggingface-hub": "huggingface_hub",
    "moviepy": "moviepy",
    "SpeechRecognition": "speech_recognition",
    "datamapplot": "datamapplot",
    "PyYAML": "yaml",
    "python-dateutil": "dateutil",
}


def _parse_requirements(path: str) -> List[Tuple[str, str]]:
    """Return list of (pip_name, raw_requirement_line) from a requirements.txt."""
    out: List[Tuple[str, str]] = []
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or line.startswith("-"):
                continue
            # Strip extras and version spec — keep just the project name
            name = re.split(r"[<>=!;\s\[]", line, maxsplit=1)[0].strip()
            if name:
                out.append((name, line))
    return out


def _is_installed(pip_name: str) -> bool:
    """Try to import the package. Returns True on success."""
    import_name = PIP_TO_IMPORT.get(pip_name, pip_name.replace("-", "_"))
    try:
        importlib.import_module(import_name)
        return True
    except Exception:
        return False


def _missing(requirements_path: str) -> List[Tuple[str, str]]:
    return [(name, line) for name, line in _parse_requirements(requirements_path)
            if not _is_installed(name)]


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

class _DependencyPrompt:
    """A small tkinter popup: 'Install required modules? [Yes] [No]'
    with an expandable details panel listing every missing package."""

    def __init__(self, missing: List[Tuple[str, str]]):
        self.missing = missing
        self.choice: str = "no"
        self._details_visible = False

        self.root = tk.Tk()
        self.root.title("VideoHandTracker — Missing dependencies")
        self.root.geometry("520x230")
        self.root.minsize(520, 230)
        try:
            self.root.iconbitmap(default="")
        except Exception:
            pass

        wrapper = ttk.Frame(self.root, padding=18)
        wrapper.pack(fill="both", expand=True)

        title = ttk.Label(
            wrapper,
            text="Install required modules?",
            font=("Segoe UI", 13, "bold"),
        )
        title.pack(anchor="w")

        body = ttk.Label(
            wrapper,
            text=(f"{len(missing)} required package(s) are missing. Install them "
                  "now using pip into the current Python environment?"),
            wraplength=470, justify="left",
        )
        body.pack(anchor="w", pady=(6, 12))

        self.toggle_btn = ttk.Button(
            wrapper, text="▶ Show details", command=self._toggle_details, width=18,
        )
        self.toggle_btn.pack(anchor="w")

        self.details_frame = ttk.Frame(wrapper)
        self.details_text = scrolledtext.ScrolledText(
            self.details_frame, height=8, wrap="word",
            font=("Consolas", 9),
        )
        for _, line in missing:
            self.details_text.insert("end", line + "\n")
        self.details_text.configure(state="disabled")
        self.details_text.pack(fill="both", expand=True)

        button_row = ttk.Frame(wrapper)
        button_row.pack(side="bottom", fill="x", pady=(12, 0))
        ttk.Button(button_row, text="No, exit", command=self._on_no, width=12).pack(side="right")
        ttk.Button(button_row, text="Yes, install", command=self._on_yes, width=14).pack(side="right", padx=(0, 8))

        self.root.protocol("WM_DELETE_WINDOW", self._on_no)
        self.root.update_idletasks()
        self._center()

    def _center(self):
        self.root.update_idletasks()
        w = self.root.winfo_width()
        h = self.root.winfo_height()
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        self.root.geometry(f"{w}x{h}+{(sw - w) // 2}+{(sh - h) // 2}")

    def _toggle_details(self):
        self._details_visible = not self._details_visible
        if self._details_visible:
            self.toggle_btn.configure(text="▼ Hide details")
            self.details_frame.pack(fill="both", expand=True, pady=(8, 0))
            self.root.geometry("520x430")
        else:
            self.toggle_btn.configure(text="▶ Show details")
            self.details_frame.pack_forget()
            self.root.geometry("520x230")

    def _on_yes(self):
        self.choice = "yes"
        self.root.destroy()

    def _on_no(self):
        self.choice = "no"
        self.root.destroy()

    def run(self) -> str:
        self.root.mainloop()
        return self.choice


def _install_progress_window(missing: List[Tuple[str, str]]) -> bool:
    """Run pip install with a live progress popup. Returns True on success."""
    win = tk.Tk()
    win.title("Installing dependencies…")
    win.geometry("560x360")

    frame = ttk.Frame(win, padding=16)
    frame.pack(fill="both", expand=True)
    ttk.Label(frame, text="Installing required packages…",
              font=("Segoe UI", 12, "bold")).pack(anchor="w")
    status = ttk.Label(frame, text="Starting pip…", foreground="#555")
    status.pack(anchor="w", pady=(4, 8))

    log = scrolledtext.ScrolledText(frame, height=14, font=("Consolas", 9))
    log.pack(fill="both", expand=True)

    win.update()

    def append(line: str):
        log.insert("end", line)
        log.see("end")
        win.update()

    cmd = [sys.executable, "-m", "pip", "install"] + [line for _, line in missing]
    append("$ " + " ".join(cmd) + "\n\n")

    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, universal_newlines=True,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            append(line)
            # Update the short status line with the most recent meaningful line
            short = line.strip()
            if short and not short.startswith("    "):
                status.configure(text=short[:80])
                win.update()
        rc = proc.wait()
    except Exception as e:
        append(f"\nInstall command failed to launch: {e}\n")
        rc = 1

    if rc == 0:
        status.configure(text="Install complete. You can close this window.",
                         foreground="#0a7d2b")
        ttk.Button(frame, text="Continue", command=win.destroy).pack(pady=(8, 0))
    else:
        status.configure(text=f"pip exited with code {rc}.",
                         foreground="#a8071a")
        ttk.Button(frame, text="Close", command=win.destroy).pack(pady=(8, 0))

    win.mainloop()
    return rc == 0


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def ensure_dependencies(requirements_path: str | None = None) -> None:
    """Probe each requirements.txt entry. Prompt the user if any are missing.
    Exits the process cleanly if the user declines or install fails."""
    if requirements_path is None:
        here = os.path.dirname(os.path.abspath(__file__))
        requirements_path = os.path.join(here, "requirements.txt")

    missing = _missing(requirements_path)
    if not missing:
        return  # Happy path — everything importable

    print(f"[DependencyCheck] {len(missing)} packages missing:",
          ", ".join(n for n, _ in missing))

    prompt = _DependencyPrompt(missing)
    choice = prompt.run()

    if choice != "yes":
        print("[DependencyCheck] User declined install — exiting.")
        sys.exit(0)

    ok = _install_progress_window(missing)
    if not ok:
        print("[DependencyCheck] Install failed; exiting.")
        sys.exit(1)

    # Re-check after install. If anything is still missing, warn and let the
    # main app continue — pip may have installed binaries that need a restart.
    still_missing = _missing(requirements_path)
    if still_missing:
        print("[DependencyCheck] Still missing after install:",
              ", ".join(n for n, _ in still_missing))
        print("[DependencyCheck] You may need to restart the application.")
