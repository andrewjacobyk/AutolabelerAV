@echo off
REM ============================================================
REM VLA Pipeline - Launcher
REM ============================================================
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found. Run install.bat first.
    exit /b 1
)

call ".venv\Scripts\activate.bat"

REM ---- Hugging Face cache: keep everything inside the workspace ----
set "HF_HOME=%~dp0data\models"
set "TRANSFORMERS_CACHE=%HF_HOME%\transformers"
set "HF_HUB_CACHE=%HF_HOME%\hub"

REM ---- Download stability -----------------------------------------
REM Silence the noisy "your filesystem doesn't support symlinks"
REM warning we get on Windows without Developer Mode.
set "HF_HUB_DISABLE_SYMLINKS_WARNING=1"

REM More generous timeouts for large shard downloads (defaults are 10s).
set "HF_HUB_DOWNLOAD_TIMEOUT=60"
set "HF_HUB_ETAG_TIMEOUT=30"

REM Force the classic HTTP downloader.  The newer xet-transfer path
REM shows two overlapping progress bars ("Downloading" +
REM "Reconstructing") that can look like a stalled download; the
REM classic downloader is slower but predictable.
set "HF_HUB_DISABLE_XET=1"
set "HF_HUB_ENABLE_HF_TRANSFER=0"

python -m src.main %*
endlocal
