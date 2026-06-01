@echo off
REM ---------------------------------------------------------------
REM  VideoAnalyzer setup.  Sequential flow, no nested if-blocks.
REM  Pass /silent to suppress the pause at the end (used by run.bat).
REM ---------------------------------------------------------------

set SILENT=0
if /i "%1"=="/silent" set SILENT=1

echo ========================================
echo  VideoAnalyzer Setup
echo ========================================
echo.

REM --- Step 1: Detect uv ---
where uv >nul 2>&1
if errorlevel 1 goto NO_UV
goto HAVE_UV

:HAVE_UV
echo [uv detected] Creating virtual environment with Python 3.11...
uv venv .venv --python 3.11
if errorlevel 1 goto FAIL_VENV
set USE_UV=1
goto VENV_READY

:NO_UV
echo [uv not found] Falling back to system Python + venv.
python --version >nul 2>&1
if errorlevel 1 goto NO_PYTHON
if not exist .venv python -m venv .venv
if errorlevel 1 goto FAIL_VENV
set USE_UV=0
goto VENV_READY

:VENV_READY
echo.
echo Installing base dependencies...
if "%USE_UV%"=="1" goto BASE_UV
goto BASE_PIP

:BASE_UV
uv pip install -r requirements.txt --python .venv\Scripts\python.exe
if errorlevel 1 goto FAIL_BASE
goto BASE_DONE

:BASE_PIP
.venv\Scripts\python.exe -m pip install --upgrade pip -q
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto FAIL_BASE
goto BASE_DONE

:BASE_DONE
echo.
echo Detecting GPU...

REM --- Step 2: Detect GPU via helper, writing result to a temp file ---
REM (Avoids the for/f + backticks + parens combination that breaks the parser.)
.venv\Scripts\python.exe detect_gpu.py > "%TEMP%\va_gpu.txt" 2>nul
set /p CUDA_VARIANT=<"%TEMP%\va_gpu.txt"
del "%TEMP%\va_gpu.txt" >nul 2>&1
if "%CUDA_VARIANT%"=="" set CUDA_VARIANT=cpu

if "%CUDA_VARIANT%"=="cu124" goto INSTALL_CUDA
goto INSTALL_CPU

:INSTALL_CUDA
echo [GPU] NVIDIA GPU detected - installing PyTorch with CUDA 12.4 support.
echo       Supports RTX 20 / 30 / 40 / 50 series  (PyTorch 2.6.0 or newer)
echo.
if "%USE_UV%"=="1" goto CUDA_UV
goto CUDA_PIP

:CUDA_UV
uv pip install --python .venv\Scripts\python.exe "torch>=2.6.0" torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
if errorlevel 1 goto FAIL_TORCH
goto RTMW_START

:CUDA_PIP
.venv\Scripts\python.exe -m pip install "torch>=2.6.0" torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
if errorlevel 1 goto FAIL_TORCH
goto RTMW_START

:INSTALL_CPU
echo [GPU] No NVIDIA GPU detected - installing CPU-only PyTorch.
echo.
if "%USE_UV%"=="1" goto CPU_UV
goto CPU_PIP

:CPU_UV
uv pip install --python .venv\Scripts\python.exe "torch>=2.6.0" torchvision torchaudio
if errorlevel 1 goto FAIL_TORCH
goto RTMW_START

:CPU_PIP
.venv\Scripts\python.exe -m pip install "torch>=2.6.0" torchvision torchaudio
if errorlevel 1 goto FAIL_TORCH
goto RTMW_START

REM ===============================================================
REM  Step 4: Profile-gated optional components
REM  Asks gpu_profile.py which heavy stacks this machine should install.
REM  CPU-only / very-low-VRAM machines skip the 2 GB mmpose download.
REM ===============================================================

:RTMW_START
.venv\Scripts\python.exe gpu_profile.py components > "%TEMP%\va_comps.txt" 2>nul
set /p VA_COMPONENTS=<"%TEMP%\va_comps.txt"
del "%TEMP%\va_comps.txt" >nul 2>&1
if "%VA_COMPONENTS%"=="" set VA_COMPONENTS=
echo.
echo Profile components for this machine: %VA_COMPONENTS%

REM Check whether RTMW is needed; otherwise skip the install entirely.
echo %VA_COMPONENTS% | findstr /C:"rtmw" >nul
if errorlevel 1 goto SKIP_RTMW

echo Installing whole-body pose stack (RTMW)...
echo This adds about 2 GB and takes 3-10 minutes on first run.
echo If it fails, the app still works using MediaPipe (hands only).
echo.

.venv\Scripts\python.exe -m pip install -q openmim
if errorlevel 1 goto RTMW_WARN

.venv\Scripts\mim.exe install mmengine
if errorlevel 1 goto RTMW_WARN

REM mmcv-lite: pure-Python build (no CUDA C++ ops).  OpenMMLab does not
REM publish full-mmcv wheels for torch 2.6 + cu124 + py3.11 + Windows, and
REM source-building requires the full CUDA Toolkit.  mmcv-lite is enough
REM for RTMW inference; GPU acceleration still happens via PyTorch.
.venv\Scripts\python.exe -m pip install "mmcv-lite>=2.0.0"
if errorlevel 1 goto RTMW_WARN

.venv\Scripts\mim.exe install "mmdet>=3.2.0,<4.0"
if errorlevel 1 goto RTMW_WARN

REM mmpose --no-deps skips chumpy (broken build, SMPL-only dep we don't use).
.venv\Scripts\python.exe -m pip install --no-deps "mmpose>=1.3.0,<2.0"
if errorlevel 1 goto RTMW_WARN
.venv\Scripts\python.exe -m pip install xtcocotools json_tricks munkres
if errorlevel 1 goto RTMW_WARN

REM Install mmcv._ext no-op stub so mmpose transformer heads can import
REM (mmcv-lite ships mmcv.ops but not the CUDA C extension it expects).
.venv\Scripts\python.exe install_mmcv_ext_stub.py
if errorlevel 1 goto RTMW_WARN

echo.
echo [RTMW] Installed successfully.
goto DONE

:SKIP_RTMW
echo [RTMW] Not required for this hardware profile. Skipping (~2 GB saved).
goto DONE

:RTMW_WARN
echo.
echo ========================================
echo  WARNING: RTMW install failed.
echo  App will still work using MediaPipe (hands only).
echo  Retry later with:  install_rtmw.bat
echo ========================================
goto DONE

REM ===============================================================
REM  Outcome labels
REM ===============================================================

:DONE
echo.
echo ========================================
echo  Setup complete!  GPU variant: %CUDA_VARIANT%
echo ========================================
echo.
if "%SILENT%"=="0" pause
exit /b 0

:NO_PYTHON
echo ERROR: Python not found.
echo Install uv (recommended): https://docs.astral.sh/uv/getting-started/installation/
echo Or install Python 3.11 from https://python.org
if "%SILENT%"=="0" pause
exit /b 1

:FAIL_VENV
echo ERROR: Failed to create virtual environment.
if "%SILENT%"=="0" pause
exit /b 1

:FAIL_BASE
echo.
echo ERROR: Base dependency installation failed.
if "%SILENT%"=="0" pause
exit /b 1

:FAIL_TORCH
echo.
echo ERROR: PyTorch installation failed.
echo If this is an RTX 50 series card, ensure your NVIDIA driver supports CUDA 12.4+.
echo Driver download: https://www.nvidia.com/drivers
if "%SILENT%"=="0" pause
exit /b 1
