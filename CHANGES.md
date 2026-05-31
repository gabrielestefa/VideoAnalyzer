# VideoAnalyzer — Session Changes

Summary of fixes and improvements applied during this session.

---

## 1. Setup & installation (one-click flow)

Goal: a brand-new Windows machine should be able to clone the repo and launch
the app without any manual Python / pip / venv steps.

### Files added
- **`run.bat`** — single entry point. First launch creates the venv and
  installs everything; subsequent launches start the app directly.
- **`detect_gpu.py`** — stdlib-only helper that calls `nvidia-smi` and prints
  `cu124` (NVIDIA GPU present) or `cpu` (no GPU). Used by both `setup.bat`
  and the runtime dependency check.

### Files rewritten
- **`setup.bat`** — switched to a sequential `goto`-based flow because the
  earlier nested `if (..) else (..)` blocks were being corrupted by the
  Windows batch parser (both GPU branches were running, special characters in
  `echo` lines were mis-tokenised, etc.). The new version:
  1. Detects `uv` (preferred) or falls back to system Python + plain venv.
  2. Pins Python **3.11** when uv is available — wider package wheel support
     than 3.13/3.14.
  3. Installs `requirements.txt` (everything except PyTorch).
  4. Calls `detect_gpu.py` and installs the matching PyTorch (`cu124` for
     RTX 20 / 30 / 40 / 50 series, or CPU build).

- **`run.bat`** — also `goto`-based for the same reason. If `.venv` exists,
  launches the app; otherwise calls `setup.bat /silent` then launches.

### Files updated
- **`requirements.txt`**
  - `mediapipe<0.12` (was `<0.11`, blocked the current release).
  - PyTorch install hint refers to the `cu124` index.
  - **`hf_xet` intentionally excluded** — its chunked transfer protocol
    produces jumpy progress and doesn't support byte-level resume of partial
    downloads. Plain HTTPS is slower but smoother and resumable.

---

## 2. Runtime dependency check

The app already had `dependency_check.py` running at the top of
`hand_heatmap_modern.py`. Several bugs and gaps were fixed.

### `dependency_check.py` — what was wrong & how it was fixed

| Issue | Fix |
|---|---|
| Called `pip install` against uv-managed Python → `externally-managed-environment` error | Detect `_is_in_venv()` and, if not in a venv, hand off to `run.bat` |
| `os.execv()` mangled paths with spaces (`C:\Users\Corsair Carbide\...`) when re-launching in the venv | Replaced with `subprocess.run(...)` + `sys.exit()` (proper Windows quoting) |
| First-time setup popup appeared **every** launch when run from system Python | Silent venv re-exec at the top of `ensure_dependencies()`: if `.venv\Scripts\python.exe` exists and we're not running it, swap to it transparently |
| PyTorch wasn't checked at all (lives outside `requirements.txt` because its index URL is GPU-dependent) | New `_ensure_torch()` function — handles three cases: missing, CPU build + GPU detected, CUDA build + GPU |
| Install progress window only showed the most recent log line | Added a dedicated **"currently downloading"** row that parses uv's `Downloading <pkg> (<size>)` / `Downloaded <pkg>` output and shows live elapsed seconds per package |

---

## 3. Whisper transcription robustness

### Background
The repo already had:
- The `batch_size` fix (use `BatchedInferencePipeline` on CUDA, not
  `WhisperModel.transcribe`) — Part 1 of `fix.md`.
- Basic stall timeout around the worker subprocess — Part 2 of `fix.md`.

Parts 3 & 4 from `fix.md` (chunk-based transcription with progress
preservation) appear to already be in `_persistent_whisper_worker` and its
parent loop in `video_analysis_engine.py`, so we did not re-implement them.

### Bugs found & fixed during testing

**Worker killed mid-download, falls back to CPU**
- Symptom: `ERROR Whisper worker did not signal ready within 120s` while the
  3 GB `large-v3` model was still downloading. Program then ran on CPU.
- Root cause: the 120 s deadline was a *total* timer, but `WhisperModel(...)`
  blocks silently during the download — no queue messages, so the parent
  killed the worker.
- Fix:
  1. Worker now spawns a **background heartbeat thread** that emits a
     progress message every 5 s during the download. Heartbeat scans the
     HuggingFace cache (`~/.cache/huggingface/hub/models--*whisper*/blobs/`)
     and reports `XXX / ~3090 MiB (NN%)`, so the user sees real download
     progress, not just elapsed time.
  2. Parent loop reset to a **silence-based** timeout: 120 s of *no
     messages* (not 120 s total). Any heartbeat resets the deadline, so an
     actively-progressing download never gets killed. A separate
     `worker_proc.is_alive()` check catches the genuine death case.

### File touched
- **`video_analysis_engine.py`** — `_spawn()` (parent timeout loop) and
  `_persistent_whisper_worker` (heartbeat thread + progress reporting).

---

## 4. GPU support — RTX 20 / 30 / 40 / 50 series

All four generations now work from a single install path:

- Index URL: `https://download.pytorch.org/whl/cu124` (CUDA 12.4 — backward
  compatible with Turing through Blackwell).
- PyTorch version pinned to `>=2.6.0` — required for RTX 50 (Blackwell)
  kernel support; older torch installs but runs Blackwell in fallback mode.
- Detection happens twice for safety: once in `setup.bat` (via
  `detect_gpu.py`), once at runtime (`_ensure_torch()` in
  `dependency_check.py`) — the runtime check can also detect a CPU-only
  torch install on a GPU machine and offer to reinstall.

---

## 5. Files added / modified

```
.
├── CHANGES.md                  ← this file
├── detect_gpu.py               ← NEW — stdlib GPU detector
├── run.bat                     ← NEW — one-click launcher
├── setup.bat                   ← REWRITTEN — sequential, parser-safe
├── requirements.txt            ← UPDATED — mediapipe bump, hf_xet excluded
├── dependency_check.py         ← UPDATED — venv re-exec, torch handling,
│                                  per-package install progress
├── video_analysis_engine.py    ← UPDATED — heartbeat + silence timeout
└── fix.md                      ← original Whisper fix plan (unchanged)
```

---

## 6. How to verify a clean install

```powershell
# 1. Wipe the venv (and optionally the HF cache for a true cold start)
Remove-Item -Recurse -Force .venv
# Remove-Item -Recurse -Force "$env:USERPROFILE\.cache\huggingface"

# 2. Launch — this triggers setup.bat then starts the app
.\run.bat
```

Expected behaviour:
- One GPU branch runs in `setup.bat` (either CUDA or CPU, not both).
- After setup finishes, the app launches automatically (no second prompt).
- On first transcription, the UI shows live download progress in MiB and %.
- Killing the app mid-download and relaunching resumes from roughly the same
  byte offset (now that `hf_xet` is excluded).

---

## 7. Known limitations

- **First Whisper download is ~3 GB at ~4 MB/s on typical home connections**
  (~13 min). Cached permanently after that.
- **Heartbeat tick interval is 5 s.** At 4 MB/s that's ~20 MiB (~0.6%) per
  tick, so progress updates look smooth. On much faster links you'll see
  bigger per-tick jumps.
- **Python 3.13/3.14 are not supported.** `mediapipe`, `hdbscan`, and a few
  transformer libraries don't publish wheels for those versions yet. `uv`
  handles this automatically by downloading 3.11; the fallback path warns
  if a system Python ≥3.13 is detected.
