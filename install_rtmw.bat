@echo off
REM ---------------------------------------------------------------
REM  Standalone RTMW installer.  Run this if:
REM    - setup.bat installed RTMW but it failed (network blip, etc.)
REM    - You have an existing .venv that predates the RTMW work
REM    - You opted out of RTMW originally and changed your mind
REM
REM  After success:
REM    set VIDEOANALYZER_BACKEND=rtmw
REM    run.bat
REM ---------------------------------------------------------------

echo ========================================
echo  RTMW (whole-body pose) installer
echo ========================================
echo.

if not exist .venv\Scripts\python.exe goto NO_VENV

echo Installing openmim...
.venv\Scripts\python.exe -m pip install -q openmim
if errorlevel 1 goto FAIL

echo Installing mmengine...
.venv\Scripts\mim.exe install mmengine
if errorlevel 1 goto FAIL

REM mmcv-lite: pure-Python build of mmcv with CUDA C++ ops stripped out.
REM Reason: OpenMMLab does not publish prebuilt mmcv wheels for the
REM torch 2.6 + cu124 + py3.11 + Windows combo this project targets, and
REM source-building full mmcv requires the CUDA Toolkit (CUDA_HOME).
REM mmcv-lite is pure Python, installs in seconds, and is sufficient for
REM RTMW *inference* (the only thing this project does).  The actual GPU
REM acceleration still comes from PyTorch's CUDA kernels at the layer level.
echo Installing mmcv-lite...
.venv\Scripts\python.exe -m pip install "mmcv-lite>=2.0.0"
if errorlevel 1 goto FAIL

echo Installing mmdet...
.venv\Scripts\mim.exe install "mmdet>=3.2.0,<4.0"
if errorlevel 1 goto FAIL

REM mmpose: install --no-deps to skip chumpy (a deprecated SMPL mesh lib
REM whose setup.py has a broken pip-in-build-env import).  Chumpy is only
REM needed for 3D mesh tasks; RTMW does 2D keypoints and never touches it.
REM Then manually add mmpose's pure-Python runtime deps that we actually use.
echo Installing mmpose (without chumpy SMPL dep)...
.venv\Scripts\python.exe -m pip install --no-deps "mmpose>=1.3.0,<2.0"
if errorlevel 1 goto FAIL
.venv\Scripts\python.exe -m pip install xtcocotools json_tricks munkres
if errorlevel 1 goto FAIL

REM mmcv-lite ships mmcv.ops which eagerly tries to load CUDA C extensions
REM from mmcv._ext (which doesn't exist in -lite).  Install a no-op stub
REM so mmpose's transformer heads import cleanly.  RTMW never calls these.
echo Installing mmcv._ext stub...
.venv\Scripts\python.exe install_mmcv_ext_stub.py
if errorlevel 1 goto FAIL

echo.
echo ========================================
echo  RTMW installed successfully!
echo.
echo  To enable whole-body pose:
echo    set VIDEOANALYZER_BACKEND=rtmw
echo    run.bat
echo ========================================
echo.
pause
exit /b 0

:NO_VENV
echo ERROR: No .venv found at .venv\Scripts\python.exe
echo Run setup.bat or run.bat first to create the virtual environment.
pause
exit /b 1

:FAIL
echo.
echo ERROR: An install step failed. Check output above.
echo Common causes:
echo   - No internet connection
echo   - mmcv source build failure (install Visual Studio Build Tools)
echo   - torch / CUDA mismatch (re-run setup.bat to reinstall torch)
pause
exit /b 1
