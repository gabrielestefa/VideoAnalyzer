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
goto DONE

:CUDA_PIP
.venv\Scripts\python.exe -m pip install "torch>=2.6.0" torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
if errorlevel 1 goto FAIL_TORCH
goto DONE

:INSTALL_CPU
echo [GPU] No NVIDIA GPU detected - installing CPU-only PyTorch.
echo.
if "%USE_UV%"=="1" goto CPU_UV
goto CPU_PIP

:CPU_UV
uv pip install --python .venv\Scripts\python.exe "torch>=2.6.0" torchvision torchaudio
if errorlevel 1 goto FAIL_TORCH
goto DONE

:CPU_PIP
.venv\Scripts\python.exe -m pip install "torch>=2.6.0" torchvision torchaudio
if errorlevel 1 goto FAIL_TORCH
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
