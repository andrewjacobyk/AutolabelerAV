#!/usr/bin/env bash
# ============================================================
# VLM Pipeline - Linux/macOS installer
# ============================================================
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs
LOG="logs/install.log"
: > "$LOG"

PYCMD=""
for c in python3.12 python3.11 python3.13 python3.10 python3; do
    if command -v "$c" >/dev/null 2>&1; then PYCMD="$c"; break; fi
done
if [[ -z "$PYCMD" ]]; then
    echo "[ERROR] Python 3.10-3.13 not found." >&2
    exit 1
fi
echo "Using Python: $($PYCMD --version)"

if [[ ! -d ".venv" ]]; then
    echo "[1/4] Creating virtual environment ..."
    "$PYCMD" -m venv .venv | tee -a "$LOG"
fi
VPY=".venv/bin/python"

echo "[2/4] Upgrading pip / setuptools / wheel ..."
"$VPY" -m pip install --disable-pip-version-check --upgrade pip setuptools wheel | tee -a "$LOG"

echo "[3/4] Installing PyTorch (CUDA 12.6 -> 11.8 -> CPU) ..."
if ! "$VPY" -m pip install --disable-pip-version-check \
        --index-url https://download.pytorch.org/whl/cu126 \
        torch torchvision 2>&1 | tee -a "$LOG" ; then
    echo "[WARN] cu126 wheels unavailable, trying cu118 ..." | tee -a "$LOG"
    if ! "$VPY" -m pip install --disable-pip-version-check \
            --index-url https://download.pytorch.org/whl/cu118 \
            torch torchvision 2>&1 | tee -a "$LOG" ; then
        echo "[WARN] CUDA wheels unavailable, falling back to CPU-only ..." | tee -a "$LOG"
        "$VPY" -m pip install --disable-pip-version-check torch torchvision | tee -a "$LOG"
    fi
fi

"$VPY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'cuda_ok', torch.cuda.is_available())" | tee -a "$LOG"

echo "[4/4] Installing project requirements ..."
"$VPY" -m pip install --disable-pip-version-check -r requirements.txt | tee -a "$LOG"

mkdir -p data/videos data/frames data/outputs data/datasets data/models logs

echo
echo "=== Installation complete ==="
echo "Activate:  source .venv/bin/activate"
echo "Launch:    ./run.sh"
echo "Full log:  $LOG"
