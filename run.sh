#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ ! -f ".venv/bin/activate" ]]; then
    echo "[ERROR] Virtual environment missing. Run ./install.sh first." >&2
    exit 1
fi
# shellcheck disable=SC1091
source .venv/bin/activate

# HF cache inside the workspace.
export HF_HOME="$(pwd)/data/models"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_HUB_CACHE="$HF_HOME/hub"

# Download stability.
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
export HF_HUB_DOWNLOAD_TIMEOUT=60
export HF_HUB_ETAG_TIMEOUT=30
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0

python -m src.main "$@"
