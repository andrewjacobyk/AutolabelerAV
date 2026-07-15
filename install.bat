@echo off
REM ============================================================
REM VLA Pipeline - Windows Installer
REM Creates a Python virtual environment and installs all deps.
REM
REM All output is captured in logs\install.log.  On any failure
REM the window stays open so you can read the error.
REM ============================================================

setlocal enableextensions
cd /d "%~dp0"

if not exist "logs" mkdir "logs" >nul 2>&1
set "LOG=logs\install.log"
> "%LOG%" echo === VLA Pipeline install log ^| %DATE% %TIME% ===

echo.
echo === VLA Pipeline - Environment Setup ===
echo (full log: %LOG%)
echo.

REM =============================================================
REM  1. Locate a supported Python interpreter (3.10 - 3.13)
REM =============================================================
set "PYCMD="
where py >nul 2>&1
if errorlevel 1 goto :no_py_launcher

call :try_py 3.12
if defined PYCMD goto :have_py
call :try_py 3.11
if defined PYCMD goto :have_py
call :try_py 3.13
if defined PYCMD goto :have_py
call :try_py 3.10
if defined PYCMD goto :have_py

:no_py_launcher
where python >nul 2>&1
if errorlevel 1 goto :err_no_python
set "PYCMD=python"

:have_py
echo Using Python: %PYCMD%
%PYCMD% --version
%PYCMD% --version >> "%LOG%" 2>&1

REM =============================================================
REM  2. Create / re-use the virtual environment
REM =============================================================
if exist ".venv\Scripts\python.exe" goto :venv_ready

echo [1/4] Creating virtual environment in .venv ...
%PYCMD% -m venv .venv >> "%LOG%" 2>&1
if errorlevel 1 goto :err_venv
goto :venv_ready

:venv_ready
set "VPY=.venv\Scripts\python.exe"
if not exist "%VPY%" goto :err_venv_missing

REM =============================================================
REM  3. Upgrade pip / setuptools / wheel (non-fatal)
REM =============================================================
echo.
echo [2/4] Upgrading pip / setuptools / wheel ...
"%VPY%" -m pip install --disable-pip-version-check --upgrade pip setuptools wheel >> "%LOG%" 2>&1
if errorlevel 1 echo [WARN] pip upgrade returned non-zero, continuing.

REM =============================================================
REM  4. Install PyTorch (try CUDA 12.6, then 11.8, then CPU)
REM =============================================================
echo.
echo [3/4] Installing PyTorch (trying CUDA 12.6 first) ...
"%VPY%" -m pip install --disable-pip-version-check --index-url https://download.pytorch.org/whl/cu126 torch torchvision >> "%LOG%" 2>&1
if not errorlevel 1 goto :torch_ok

echo [WARN] CUDA 12.6 wheels unavailable, trying CUDA 11.8 ...
"%VPY%" -m pip install --disable-pip-version-check --index-url https://download.pytorch.org/whl/cu118 torch torchvision >> "%LOG%" 2>&1
if not errorlevel 1 goto :torch_ok

echo [WARN] CUDA install failed, falling back to CPU-only torch ...
"%VPY%" -m pip install --disable-pip-version-check torch torchvision >> "%LOG%" 2>&1
if errorlevel 1 goto :err_torch

:torch_ok
"%VPY%" -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'cuda_ok', torch.cuda.is_available())"
"%VPY%" -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'cuda_ok', torch.cuda.is_available())" >> "%LOG%" 2>&1

REM =============================================================
REM  5. Install remaining project requirements
REM =============================================================
echo.
echo [4/4] Installing project requirements ...
"%VPY%" -m pip install --disable-pip-version-check -r requirements.txt >> "%LOG%" 2>&1
if errorlevel 1 goto :err_req

REM =============================================================
REM  6. Data folders
REM =============================================================
if not exist "data\videos"   mkdir "data\videos"
if not exist "data\frames"   mkdir "data\frames"
if not exist "data\outputs"  mkdir "data\outputs"
if not exist "data\datasets" mkdir "data\datasets"
if not exist "data\models"   mkdir "data\models"
if not exist "logs"          mkdir "logs"

echo.
echo === Installation complete ===
echo Launch the app with:   run.bat
echo Full install log:      %LOG%
echo.
pause
endlocal
exit /b 0

REM =============================================================
REM  Helpers
REM =============================================================
:try_py
REM Usage: call :try_py 3.12
if defined PYCMD goto :eof
py -%1 -c "import sys" >nul 2>&1
if errorlevel 1 goto :eof
set "PYCMD=py -%1"
goto :eof

:err_no_python
echo [ERROR] No Python 3.10 / 3.11 / 3.12 / 3.13 found on this machine.
echo         Please install Python 3.12 from https://www.python.org/downloads/
goto :fail

:err_venv
echo [ERROR] Failed to create the .venv virtual environment.
echo         Check %LOG% for the underlying pip/venv message.
goto :fail

:err_venv_missing
echo [ERROR] .venv exists but %VPY% is missing.
echo         Delete the .venv folder and re-run install.bat.
goto :fail

:err_torch
echo [ERROR] Could not install PyTorch (neither CUDA nor CPU wheels worked).
echo         Check %LOG% and your internet / proxy settings.
goto :fail

:err_req
echo [ERROR] Failed installing requirements.txt.
echo         The last ~40 lines of %LOG% usually show which package failed.
goto :fail

:fail
echo.
echo === INSTALL FAILED ===
echo Full log:  %LOG%
echo.
pause
endlocal
exit /b 1
