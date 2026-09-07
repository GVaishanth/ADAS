#!/usr/bin/env bash
#
# setup_env.sh
# ============
# Recreates the Python environment and installs the ML stack, then (optionally)
# retrains + evaluates the four comparison backbones.
#
# This exists because the heavy, regenerable artefacts (the virtualenv and the
# trained checkpoints) are NOT kept in the workspace — they are build outputs.
# Run this once on any machine with Python 3.10+ to rebuild them.
#
# Usage:
#   bash setup_env.sh                      # env + deps only
#   bash setup_env.sh --with-training      # env + deps + retrain the 4 backbones
#
set -euo pipefail
cd "$(dirname "$0")"

PY=${PY:-python3}
VENV=venv

echo "[1/3] Creating virtualenv at $VENV ..."
"$PY" -m venv "$VENV"
./"$VENV"/bin/pip install --upgrade pip -q
echo "[2/3] Installing requirements.txt (this is the bulk of the download) ..."
./"$VENV"/bin/pip install -r requirements.txt

echo "[3/3] Verifying imports ..."
./"$VENV"/bin/python - <<'PY'
import numpy, cv2, pandas, torch, torchvision, timm, ultralytics, transformers, sklearn, fastapi, uvicorn
print("   OK  -> torch", torch.__version__, "| timm", timm.__version__, "| ultralytics", ultralytics.__version__)
PY

if [ "${1:-}" = "--with-training" ]; then
  echo ""
  echo "Training + evaluating the four transformer backbones (CPU/GPU aware) ..."
  ./"$VENV"/bin/python datasets/prepare_current.py
  ./"$VENV"/bin/python -m algorithms._shared.run_comparison --pretrained --epochs 30
  ./"$VENV"/bin/python results/make_comparison_chart.py
fi

echo ""
echo "Done. To run the server:"
echo "  ./$VENV/bin/python -m uvicorn integration.api.server:app --host 127.0.0.1 --port 7000"
echo "  # then open http://localhost:7000/dashboard"
