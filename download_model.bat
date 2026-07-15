@echo off
REM Download model weights without launching the GUI.
REM Usage:  download_model.bat moondream2
setlocal
cd /d "%~dp0"

if "%~1"=="" (
    echo Usage: download_model.bat ^<model_id^>
    echo Example: download_model.bat moondream2
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Run install.bat first.
    exit /b 1
)

set "HF_HOME=%~dp0data\models"
set "TRANSFORMERS_CACHE=%HF_HOME%\transformers"
set "HF_HUB_CACHE=%HF_HOME%\hub"
set "HF_HUB_DISABLE_SYMLINKS_WARNING=1"
set "HF_HUB_DOWNLOAD_TIMEOUT=60"
set "HF_HUB_DISABLE_XET=1"
set "HF_HUB_ENABLE_HF_TRANSFER=0"

".venv\Scripts\python.exe" -m src.tools.download_model %*
endlocal
