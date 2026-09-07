# ====================================================================
#  ADAS — one-shot PowerShell setup + run (Windows)
#  Run from anywhere; it handles the venv, deps, frontend build and
#  launches the server on http://localhost:7000
# ====================================================================

# ---- 1. Paths ----
$Root       = "C:\Users\RC\Downloads\ADAS_testing_2"
$ADAS       = Join-Path $Root "ADAS_testing"        # holds venv + ADAS project
$Backend    = Join-Path $ADAS "ADAS"                # the `integration` package lives here
$Frontend   = Join-Path $Root "ADAS_frontend"
$Venv       = Join-Path $ADAS "venv"
$Py         = Join-Path $Venv "Scripts\python.exe"

# ---- 2. Create venv + install backend deps (first run) ----
if (-not (Test-Path $Py)) {
    Write-Host "[1/4] Creating venv + installing backend deps..." -ForegroundColor Cyan
    python -m venv $Venv
    & $Py -m pip install --upgrade pip | Out-Null
    & $Py -m pip install -r (Join-Path $Backend "requirements.txt")
} else {
    Write-Host "[1/4] venv already present — skipping install." -ForegroundColor Cyan
}

# ---- 3. Build the frontend (first run, or anytime after UI edits) ----
$Inline = Join-Path $Frontend "dist\index_inline.html"
if (-not (Test-Path $Inline)) {
    Write-Host "[2/4] Building frontend (npm install + build + inline)..." -ForegroundColor Cyan
    Push-Location $Frontend
    if (-not (Test-Path node_modules)) { npm install --no-audit --no-fund }
    npm run build | Out-Null
    & $Py make_inline.py
    Pop-Location
} else {
    Write-Host "[2/4] Frontend build present — skipping." -ForegroundColor Cyan
}

# ---- 4. Optional: train the four-backbone weather models if weights absent ----
$Weights = Join-Path $Backend "models\vit\best.pt"
if (-not (Test-Path $Weights)) {
    Write-Host "[3/4] No trained weights yet. Training will take a while." -ForegroundColor Yellow
    Write-Host "      (Skip this to just preview the UI; model tabs will say 'unavailable')." -ForegroundColor Yellow
    $Answer = Read-Host "Train now? (y/N)"
    if ($Answer -eq "y") {
        Push-Location $Backend
        & $Py -m algorithms._shared.run_comparison --epochs 30 --pretrained --augment
        Pop-Location
    }
} else {
    Write-Host "[3/4] Weights present — skipping training." -ForegroundColor Cyan
}

# ---- 5. Launch the backend on localhost:7000 ----
Write-Host "[4/4] Starting server at http://localhost:7000 ..." -ForegroundColor Green
Push-Location $Backend
& $Py -m uvicorn integration.api.server:app --host 127.0.0.1 --port 7000
