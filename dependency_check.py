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


# ---------------------------------------------------------------------------
# RTMW whole-body pose stack (mmpose + mmcv + mmdet + mmengine).
# Heavy (~2 GB) so we install it via openmim, which finds wheels matching
# the installed torch+CUDA combo.  Failure is non-fatal — the engine falls
# back to MediaPipe (hands only).
# ---------------------------------------------------------------------------

def _ensure_mmpose() -> None:
    """If mmpose isn't installed, offer a one-click install via mim.

    Skipped silently when the selected hardware profile doesn't need RTMW
    (e.g. CPU-only or sub-4 GB VRAM machines — MediaPipe is the right
    choice there and we shouldn't push a 2 GB download.)
    """
    try:
        import mmpose  # noqa: F401
        return  # already installed — happy path
    except ImportError:
        pass

    # Profile gate: only ask if RTMW is actually part of this profile's stack.
    try:
        from gpu_profile import select_profile, required_components
        comps = required_components(select_profile())
        if "rtmw" not in comps:
            print("[DependencyCheck] Profile does not require RTMW; skipping install prompt.")
            return
    except Exception as e:
        print(f"[DependencyCheck] Profile check failed ({e}); falling through to prompt.")

    root = tk.Tk()
    root.withdraw()
    from tkinter import messagebox
    install = messagebox.askyesno(
        "Install whole-body pose? (RTMW)",
        "RTMW provides whole-body pose estimation:\n"
        " - 17 body keypoints (arms, legs, torso, head)\n"
        " - 21 + 21 hand keypoints\n"
        " - 68 face keypoints\n\n"
        "Install now?\n"
        "Size: ~2 GB.  Time: 3-10 minutes.\n\n"
        "Without it, only MediaPipe (hands only) is available.",
    )
    root.destroy()
    if not install:
        print("[DependencyCheck] User declined RTMW install.")
        return

    ok = _install_mmpose_window()
    if not ok:
        print("[DependencyCheck] RTMW install failed; engine will use MediaPipe.")


def _install_mmpose_window() -> bool:
    """Install the OpenMMLab stack via openmim with a progress popup."""
    venv_python = sys.executable
    venv_dir = os.path.dirname(venv_python)
    mim_exe = os.path.join(venv_dir, "mim.exe")

    win = tk.Tk()
    win.title("Installing RTMW (whole-body pose)…")
    win.geometry("580x420")
    frame = ttk.Frame(win, padding=16)
    frame.pack(fill="both", expand=True)
    ttk.Label(frame, text="Installing RTMW whole-body pose stack",
              font=("Segoe UI", 12, "bold")).pack(anchor="w")
    ttk.Label(frame,
              text=("Pulls wheels matching the installed torch+CUDA combo via "
                    "openmim. Expect ~2 GB and 3-10 min."),
              justify="left", foreground="#555", wraplength=540
              ).pack(anchor="w", pady=(4, 8))
    current = ttk.Label(frame, text="", foreground="#0a4d8c",
                        font=("Segoe UI", 10, "bold"))
    current.pack(anchor="w", pady=(0, 2))
    status = ttk.Label(frame, text="Starting…", foreground="#555")
    status.pack(anchor="w", pady=(0, 4))
    log = scrolledtext.ScrolledText(frame, height=12, font=("Consolas", 9))
    log.pack(fill="both", expand=True)
    win.update()

    def append(line: str):
        log.insert("end", line); log.see("end")
    def set_status(text: str):
        status.configure(text=text)
    def set_current(text: str):
        current.configure(text=text)

    # Step 1 — openmim itself.  Use uv when available for speed.
    uv = _find_uv()
    if uv:
        cmd_mim = [uv, "pip", "install", "--python", venv_python, "openmim"]
    else:
        cmd_mim = [venv_python, "-m", "pip", "install", "openmim"]
    append("$ " + " ".join(cmd_mim) + "\n")
    rc = _run_installer_with_progress(cmd_mim, append, set_status, set_current, win)
    if rc != 0:
        status.configure(text="openmim install failed.", foreground="#a8071a")
        ttk.Button(frame, text="Close", command=win.destroy).pack(pady=(8, 0))
        win.mainloop()
        return False

    if not os.path.exists(mim_exe):
        # uv installs may put scripts in a different location; fall back to
        # invoking mim through python -m.
        mim_call = [venv_python, "-m", "mim"]
    else:
        mim_call = [mim_exe]

    # mmcv-lite is the right pick for this project — see install_rtmw.bat
    # for the full reasoning.  TL;DR: no prebuilt full-mmcv wheels for our
    # torch+CUDA+python combo on Windows, and source build needs the CUDA
    # Toolkit.  Inference works fine on mmcv-lite; GPU acceleration is
    # provided by PyTorch directly.
    # mmpose --no-deps + manual runtime deps: skips chumpy (deprecated SMPL
    # lib whose setup.py breaks in pip's build-isolation env).  RTMW only
    # does 2D keypoint extraction, so chumpy is never touched at runtime.
    steps = [
        ("mmengine", mim_call + ["install", "mmengine"]),
        ("mmcv-lite", [venv_python, "-m", "pip", "install", "mmcv-lite>=2.0.0"]),
        ("mmdet", mim_call + ["install", "mmdet>=3.2.0,<4.0"]),
        ("mmpose (without SMPL deps)",
            [venv_python, "-m", "pip", "install", "--no-deps", "mmpose>=1.3.0,<2.0"]),
        ("mmpose runtime deps",
            [venv_python, "-m", "pip", "install", "xtcocotools", "json_tricks", "munkres"]),
    ]
    for label, cmd in steps:
        set_current(f"Installing {label}…")
        win.update()
        append("\n$ " + " ".join(cmd) + "\n")
        rc = _run_installer_with_progress(cmd, append, set_status, set_current, win)
        if rc != 0:
            status.configure(text=f"Install of {label} failed (exit {rc}).",
                             foreground="#a8071a")
            ttk.Button(frame, text="Close", command=win.destroy).pack(pady=(8, 0))
            win.mainloop()
            return False

    set_current("")
    status.configure(text="RTMW installed!  Set VIDEOANALYZER_BACKEND=rtmw to use it.",
                     foreground="#0a7d2b")
    ttk.Button(frame, text="Continue", command=win.destroy).pack(pady=(8, 0))
    win.mainloop()
    return True


# ---------------------------------------------------------------------------
# Model assets — small files (~15 MB) Google hosts on a stable CDN.  We
# auto-download on first launch so a fresh clone of the repo "just works"
# without the user having to chase model files manually.
# ---------------------------------------------------------------------------

# (filename, url, required, sha256_prefix_or_None)
_REQUIRED_MODELS = [
    (
        "gesture_recognizer.task",
        "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
        "gesture_recognizer/float16/1/gesture_recognizer.task",
        True,
    ),
    (
        "universal_sentence_encoder.tflite",
        "https://storage.googleapis.com/mediapipe-tasks/text_embedder/"
        "universal_sentence_encoder.tflite",
        False,  # optional — text similarity gracefully degrades without it
    ),
]


def _download_with_progress(url: str, dest_path: str, label: str) -> bool:
    """
    Download a file with a small tk progress dialog.  Returns True on success.
    Uses stdlib urllib so no extra deps are required.
    """
    import urllib.request
    import urllib.error

    win = tk.Tk()
    win.title(f"Downloading {label}…")
    win.geometry("520x150")
    win.resizable(False, False)
    f = ttk.Frame(win, padding=18)
    f.pack(fill="both", expand=True)
    ttk.Label(f, text=f"Downloading {label}", font=("Segoe UI", 11, "bold")).pack(anchor="w")
    sub = ttk.Label(f, text="Starting…", foreground="#555")
    sub.pack(anchor="w", pady=(4, 8))
    bar = ttk.Progressbar(f, orient="horizontal", mode="determinate", maximum=100)
    bar.pack(fill="x")
    win.update()

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    tmp_path = dest_path + ".part"

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "VideoAnalyzer-setup/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            total = int(resp.headers.get("Content-Length", "0"))
            done = 0
            chunk_size = 64 * 1024
            with open(tmp_path, "wb") as out:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    out.write(chunk)
                    done += len(chunk)
                    if total:
                        pct = int(done / total * 100)
                        bar["value"] = pct
                        sub.configure(text=f"{done/1024/1024:.1f} / {total/1024/1024:.1f} MiB ({pct}%)")
                    else:
                        sub.configure(text=f"{done/1024/1024:.1f} MiB")
                    win.update()
        os.replace(tmp_path, dest_path)
        win.destroy()
        return True
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        print(f"[DependencyCheck] Download failed for {label}: {e}")
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        sub.configure(text=f"Failed: {e}", foreground="#a8071a")
        ttk.Button(f, text="Close", command=win.destroy).pack(pady=(8, 0))
        win.mainloop()
        return False


def _ensure_models() -> None:
    """
    Make sure the MediaPipe model assets are present in ./Models/.
    Required models block the launch on failure; optional ones just warn.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(here, "Models")

    for filename, url, required in _REQUIRED_MODELS:
        dest = os.path.join(models_dir, filename)
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            continue
        print(f"[DependencyCheck] Missing model: {filename}")
        ok = _download_with_progress(url, dest, label=filename)
        if not ok:
            if required:
                print(f"[DependencyCheck] FATAL: required model {filename} could not be downloaded")
                root = tk.Tk()
                root.withdraw()
                from tkinter import messagebox
                messagebox.showerror(
                    "Model download failed",
                    f"Could not download {filename}.\n\n"
                    f"Check your internet connection and try again.\n\n"
                    f"You can also download it manually from:\n{url}\n"
                    f"and place it in:\n{models_dir}",
                )
                root.destroy()
                sys.exit(1)
            else:
                print(f"[DependencyCheck] Optional model {filename} unavailable; continuing.")


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
    if missing:
        print(f"[DependencyCheck] {len(missing)} packages missing:",
              ", ".join(n for n, _ in missing))

        # Not inside a venv AND no .venv exists yet: hand off to run.bat to
        # do first-time setup (creates .venv with correct Python, installs).
        if not _is_in_venv():
            if _relaunch_via_run_bat(here):
                return  # _relaunch_via_run_bat calls sys.exit — never reached
            # run.bat not found: fall through and try a direct install.

        prompt = _DependencyPrompt(missing)
        choice = prompt.run()

        if choice != "yes":
            print("[DependencyCheck] User declined install — exiting.")
            sys.exit(0)

        ok = _install_progress_window(missing)
        if not ok:
            print("[DependencyCheck] Install failed; exiting.")
            sys.exit(1)

        # Re-check after install. Warn if anything is still absent — a
        # restart may be needed for entry-points / C extensions.
        still_missing = _missing(requirements_path)
        if still_missing:
            print("[DependencyCheck] Still missing after install:",
                  ", ".join(n for n, _ in still_missing))
            print("[DependencyCheck] You may need to restart the application.")

    # These three checks run on EVERY launch (they no-op silently when their
    # target is present), so existing venvs created before later phases pick
    # up the new requirements without the user having to delete .venv.
    #
    # PyTorch — index URL varies by GPU, so not in requirements.txt.
    _ensure_torch()
    # MediaPipe model assets — auto-downloaded so a fresh clone "just works".
    _ensure_models()
    # RTMW whole-body pose stack — opt-in, prompts on first detect.
    _ensure_mmpose()
