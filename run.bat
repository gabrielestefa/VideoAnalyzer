@echo off
REM ---------------------------------------------------------------
REM  One-click launcher.
REM  First run: creates .venv and installs all dependencies.
REM  Subsequent runs: launches the app directly.
REM ---------------------------------------------------------------

if exist .venv\Scripts\python.exe goto LAUNCH

echo First-time setup - this will take a few minutes...
echo.
call "%~dp0setup.bat" /silent
if errorlevel 1 goto SETUP_FAILED

:LAUNCH
echo Starting VideoAnalyzer...
.venv\Scripts\python.exe hand_heatmap_modern.py
exit /b %errorlevel%

:SETUP_FAILED
echo.
echo Setup failed. Please check the error above.
pause
exit /b 1
