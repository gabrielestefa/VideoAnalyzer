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


def _is_in_venv() -> bool:
    """True when running inside a virtual environment (venv, virtualenv, conda)."""
    return (
        sys.prefix != sys.base_prefix
        or os.environ.get("VIRTUAL_ENV") is not None
        or os.environ.get("CONDA_DEFAULT_ENV") is not None
    )


def _find_uv() -> "str | None":
    """Return the path to the uv executable, or None if not on PATH."""
    import shutil
    return shutil.which("uv")


def _relaunch_via_run_bat(here: str) -> bool:
    """
    Launch run.bat in the project directory (which creates .venv, installs
    packages, and restarts the app inside the venv) then exit this process.
    Returns False only if run.bat cannot be found.
    """
    run_bat = os.path.join(here, "run.bat")
    if not os.path.exists(run_bat):
        return False

    # Show a brief non-blocking notice before handing off
    notice = tk.Tk()
    notice.title("VideoAnalyzer — First-time setup")
    notice.geometry("420x130")
    notice.resizable(False, False)
    try:
        notice.iconbitmap(default="")
    except Exception:
        pass
    f = ttk.Frame(notice, padding=20)
    f.pack(fill="both", expand=True)
    ttk.Label(f, text="Setting up environment…",
              font=("Segoe UI", 12, "bold")).pack(anchor="w")
    ttk.Label(f, text="Installing packages and restarting the app.\nThis window will close automatically.",
              justify="left", foreground="#555").pack(anchor="w", pady=(6, 0))
    notice.update()

    subprocess.Popen(["cmd", "/c", run_bat], cwd=here)

    notice.after(1800, notice.destroy)
    notice.mainloop()
    sys.exit(0)


def _run_installer_with_progress(cmd, append, set_status, set_current, win):
    """
    Stream installer output into the UI, parsing uv's
        Downloading <name> (<size>)
        Downloaded  <name>
    lines into a dedicated 'currently downloading' status row with elapsed time.

    Returns the process exit code.
    """
    import re
    import time as _time

    downloading_re = re.compile(r"Downloading\s+(\S+)\s+\(([^)]+)\)")
    downloaded_re = re.compile(r"Downloaded\s+(\S+)")

    current_pkg: "str | None" = None
    current_size: "str | None" = None
    current_start: float = 0.0

    def refresh_current(done: bool = False):
        if current_pkg is None:
            set_current("")
            return
        elapsed = int(_time.time() - current_start)
        if done:
            set_current(f"✓ {current_pkg} ({current_size}) — done in {elapsed}s")
        else:
            set_current(f"⬇ {current_pkg} ({current_size}) — {elapsed}s elapsed")

    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, universal_newlines=True,
        )
        assert proc.stdout is not None

        # Tick the elapsed-time row even when no new output arrives
        def tick():
            if current_pkg is not None:
                refresh_current(done=False)
            win.after(500, tick)
        win.after(500, tick)

        for line in proc.stdout:
            append(line)

            m = downloading_re.search(line)
            if m:
                current_pkg = m.group(1)
                current_size = m.group(2)
                current_start = _time.time()
                refresh_current(done=False)
            else:
                m2 = downloaded_re.search(line)
                if m2 and m2.group(1) == current_pkg:
                    refresh_current(done=True)
                    current_pkg = None
                    current_size = None

            short = line.strip()
            if short and not short.startswith("    "):
                set_status(short[:80])
            win.update()

        return proc.wait()

    except Exception as e:
        append(f"\nInstall command failed to launch: {e}\n")
        return 1


def _install_progress_window(missing: List[Tuple[str, str]]) -> bool:
    """Install missing packages with a live progress popup. Returns True on success."""
    win = tk.Tk()
    win.title("Installing dependencies…")
    win.geometry("580x400")

    frame = ttk.Frame(win, padding=16)
    frame.pack(fill="both", expand=True)
    ttk.Label(frame, text="Installing required packages…",
              font=("Segoe UI", 12, "bold")).pack(anchor="w")

    current = ttk.Label(frame, text="", foreground="#0a4d8c",
                        font=("Segoe UI", 10, "bold"))
    current.pack(anchor="w", pady=(6, 2))

    status = ttk.Label(frame, text="Starting installer…", foreground="#555")
    status.pack(anchor="w", pady=(0, 8))

    log = scrolledtext.ScrolledText(frame, height=14, font=("Consolas", 9))
    log.pack(fill="both", expand=True)

    win.update()

    def append(line: str):
        log.insert("end", line)
        log.see("end")
    def set_status(text: str):
        status.configure(text=text)
    def set_current(text: str):
        current.configure(text=text)

    uv = _find_uv()
    if uv:
        cmd = [uv, "pip", "install", "--python", sys.executable] + [line for _, line in missing]
    else:
        cmd = [sys.executable, "-m", "pip", "install"] + [line for _, line in missing]

    append("$ " + " ".join(cmd) + "\n\n")

    rc = _run_installer_with_progress(cmd, append, set_status, set_current, win)

    if rc == 0:
        set_current("")
        status.configure(text="Install complete. You can close this window.",
                         foreground="#0a7d2b")
        ttk.Button(frame, text="Continue", command=win.destroy).pack(pady=(8, 0))
    else:
        status.configure(text=f"Installer exited with code {rc}.",
                         foreground="#a8071a")
        ttk.Button(frame, text="Close", command=win.destroy).pack(pady=(8, 0))

    win.mainloop()
    return rc == 0


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def _detect_cuda_variant() -> str:
    """Return 'cu124' if an NVIDIA GPU is detected, 'cpu' otherwise."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode == 0 and r.stdout.strip():
            return "cu124"
    except Exception:
        pass
    return "cpu"


def _install_torch_window(variant: str) -> bool:
    """Install PyTorch with a live progress popup. Returns True on success."""
    if variant == "cu124":
        label = "NVIDIA GPU detected — installing PyTorch with CUDA 12.4 support.\nSupports RTX 20 / 30 / 40 / 50 series  (PyTorch ≥ 2.6.0)"
        packages = ["torch>=2.6.0", "torchvision", "torchaudio"]
        extra = ["--index-url", "https://download.pytorch.org/whl/cu124"]
    else:
        label = "No NVIDIA GPU detected �� installing CPU-only PyTorch."
        packages = ["torch>=2.6.0", "torchvision", "torchaudio"]
        extra = []

    win = tk.Tk()
    win.title("Installing PyTorch…")
    win.geometry("580x420")
    frame = ttk.Frame(win, padding=16)
    frame.pack(fill="both", expand=True)
    ttk.Label(frame, text="Installing PyTorch…",
              font=("Segoe UI", 12, "bold")).pack(anchor="w")
    ttk.Label(frame, text=label, justify="left",
              foreground="#555").pack(anchor="w", pady=(4, 8))

    current = ttk.Label(frame, text="", foreground="#0a4d8c",
                        font=("Segoe UI", 10, "bold"))
    current.pack(anchor="w", pady=(0, 2))

    status = ttk.Label(frame, text="Starting…", foreground="#555")
    status.pack(anchor="w", pady=(0, 4))
    log = scrolledtext.ScrolledText(frame, height=12, font=("Consolas", 9))
    log.pack(fill="both", expand=True)
    win.update()

    def append(line: str):
        log.insert("end", line)
        log.see("end")
    def set_status(text: str):
        status.configure(text=text)
    def set_current(text: str):
        current.configure(text=text)

    uv = _find_uv()
    if uv:
        cmd = [uv, "pip", "install", "--python", sys.executable] + packages + extra
    else:
        cmd = [sys.executable, "-m", "pip", "install"] + packages + extra

    append("$ " + " ".join(cmd) + "\n\n")
    rc = _run_installer_with_progress(cmd, append, set_status, set_current, win)

    if rc == 0:
        set_current("")
        status.configure(text="PyTorch installed successfully.", foreground="#0a7d2b")
        ttk.Button(frame, text="Continue", command=win.destroy).pack(pady=(8, 0))
    else:
        status.configure(text=f"Installation failed (exit {rc}).", foreground="#a8071a")
        ttk.Button(frame, text="Close", command=win.destroy).pack(pady=(8, 0))
    win.mainloop()
    return rc == 0


def _ensure_torch() -> None:
    """
    Check PyTorch separately from requirements.txt (its index URL varies by GPU).
    - Missing entirely  → detect GPU, install correct variant with progress UI.
    - CPU torch + GPU   → warn the user and offer to reinstall as CUDA build.
    - CUDA torch + GPU  → happy path, return immediately.
    """
    try:
        import torch  # noqa: F401
        cuda_ok = torch.cuda.is_available()
        if cuda_ok:
            return  # CUDA torch working — nothing to do

        # Torch present but CUDA unavailable.  Check whether a physical GPU exists.
        variant = _detect_cuda_variant()
        if variant == "cpu":
            return  # No GPU — CPU torch is correct

        # GPU found but torch has no CUDA support.
        root = tk.Tk()
        root.withdraw()
        from tkinter import messagebox
        reinstall = messagebox.askyesno(
            "PyTorch — GPU not enabled",
            "An NVIDIA GPU was detected but PyTorch is running in CPU-only mode.\n\n"
            "This significantly reduces transcription and analysis speed.\n\n"
            "Reinstall PyTorch with CUDA 12.4 support now?\n"
            "(Supports RTX 20 / 30 / 40 / 50 series)",
        )
        root.destroy()
        if reinstall:
            _install_torch_window("cu124")
        return

    except ImportError:
        pass

    # Torch not installed at all.
    variant = _detect_cuda_variant()
    prompt_text = (
        "PyTorch is not installed.\n\n"
        + ("An NVIDIA GPU was detected.\nInstall PyTorch with CUDA 12.4 support?\n(RTX 20 / 30 / 40 / 50 series)"
           if variant == "cu124"
           else "No NVIDIA GPU detected.\nInstall CPU-only PyTorch?")
    )
    root = tk.Tk()
    root.withdraw()
    from tkinter import messagebox
    install = messagebox.askyesno("Install PyTorch?", prompt_text)
    root.destroy()
    if install:
        ok = _install_torch_window(variant)
        if not ok:
            print("[DependencyCheck] PyTorch install failed; some features will be unavailable.")


def ensure_dependencies(requirements_path: str | None = None) -> None:
    """Probe each requirements.txt entry. Prompt the user if any are missing.

    Decision tree:
      • All present          → return immediately (happy path).
      • Missing, not in venv → launch run.bat (creates venv + installs + restarts)
                               and exit this process.  run.bat not found → fall through.
      • Missing, in venv     → show install prompt and run uv/pip directly.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    if requirements_path is None:
        requirements_path = os.path.join(here, "requirements.txt")

    # Silent venv re-exec: if we're not currently running the project's .venv
    # interpreter but it exists on disk, swap to it immediately.  This makes
    # `python hand_heatmap_modern.py` work the same as `run.bat` without any
    # popups, even when launched from a system or uv-managed Python.
    #
    # We use subprocess instead of os.execv because Windows execv mangles
    # arguments containing spaces (e.g. "C:\Users\Corsair Carbide\...") by
    # failing to quote them in the rebuilt command line.
    venv_python = os.path.join(here, ".venv", "Scripts", "python.exe")
    current_python = os.path.normcase(os.path.abspath(sys.executable))
    if (os.path.exists(venv_python)
            and os.path.normcase(os.path.abspath(venv_python)) != current_python):
        try:
            result = subprocess.run([venv_python, *sys.argv])
            sys.exit(result.returncode)
        except Exception as e:
            print(f"[DependencyCheck] Failed to relaunch in venv: {e}")
            # Fall through and continue in the current interpreter

    missing = _missing(requirements_path)
    if not missing:
        return

    print(f"[DependencyCheck] {len(missing)} packages missing:",
          ", ".join(n for n, _ in missing))

    # Not inside a venv AND no .venv exists yet: hand off to run.bat to do
    # first-time setup (creates .venv with correct Python, installs packages).
    if not _is_in_venv():
        if _relaunch_via_run_bat(here):
            return  # _relaunch_via_run_bat calls sys.exit — never reached
        # run.bat not found: fall through and try a direct in-place install.

    prompt = _DependencyPrompt(missing)
    choice = prompt.run()

    if choice != "yes":
        print("[DependencyCheck] User declined install — exiting.")
        sys.exit(0)

    ok = _install_progress_window(missing)
    if not ok:
        print("[DependencyCheck] Install failed; exiting.")
        sys.exit(1)

    # Re-check after install. Warn if anything is still absent — a restart
    # may be needed for packages that register entry-points or C extensions.
    still_missing = _missing(requirements_path)
    if still_missing:
        print("[DependencyCheck] Still missing after install:",
              ", ".join(n for n, _ in still_missing))
        print("[DependencyCheck] You may need to restart the application.")

    # PyTorch is not in requirements.txt because its index URL varies by GPU.
    # Check it separately here.
    _ensure_torch()
