#!/usr/bin/env bash
#
# build_frontend.sh — build the merged ADAS frontend into ADAS_frontend/dist.
# The FastAPI backend serves that build as the single frontend on port 7000.
#
#   bash build_frontend.sh
#
set -euo pipefail
cd "$(dirname "$0")/.."      # -> ADAS_testing/
FRONTEND=ADAS_frontend

if [ ! -d "$FRONTEND/node_modules" ]; then
  echo "[1/2] npm install ..."
  (cd "$FRONTEND" && npm install --no-audit --no-fund)
fi
echo "[2/2] npm run build ..."
(cd "$FRONTEND" && npm run build)
echo "Done. Built $FRONTEND/dist — served by the backend at http://localhost:7000"
