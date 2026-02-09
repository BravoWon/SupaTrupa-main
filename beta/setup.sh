#!/bin/bash
# SupaTrupa Beta - One-Command Setup
# Installs all dependencies for backend and frontend

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=========================================="
echo "  SupaTrupa Beta v0.1.0 - Setup"
echo "=========================================="
echo ""

# --- Check prerequisites ---
echo "[1/5] Checking prerequisites..."

check_command() {
    if ! command -v "$1" &> /dev/null; then
        echo "  ERROR: $1 is required but not found."
        echo "  $2"
        exit 1
    else
        echo "  OK: $1 ($($1 --version 2>&1 | head -1))"
    fi
}

check_command python3 "Install Python 3.9+ from https://python.org"
check_command node "Install Node.js 20+ from https://nodejs.org or use nvm"

# Check pnpm (may need corepack)
if ! command -v pnpm &> /dev/null; then
    echo "  pnpm not found, enabling via corepack..."
    corepack enable 2>/dev/null || {
        echo "  ERROR: pnpm is required. Install via: npm install -g pnpm"
        exit 1
    }
fi
echo "  OK: pnpm ($(pnpm --version 2>&1 | head -1))"
echo ""

# --- Backend setup ---
echo "[2/5] Setting up Python backend..."
cd "$PROJECT_ROOT/backend"

if [ ! -d ".venv" ]; then
    echo "  Creating virtual environment..."
    python3 -m venv .venv
fi

echo "  Activating venv and installing dependencies..."
source .venv/bin/activate
pip install -e ".[all]" --quiet 2>&1 | tail -3
echo "  Backend dependencies installed."
echo ""

# --- Frontend setup ---
echo "[3/5] Setting up React frontend..."
cd "$PROJECT_ROOT/frontend"
pnpm install --silent 2>&1 | tail -3
echo "  Frontend dependencies installed."
echo ""

# --- Frontend .env ---
echo "[4/5] Configuring frontend environment..."
ENV_FILE="$PROJECT_ROOT/frontend/client/.env"
if [ ! -f "$ENV_FILE" ]; then
    echo "VITE_API_URL=http://localhost:8000" > "$ENV_FILE"
    echo "  Created $ENV_FILE"
else
    echo "  $ENV_FILE already exists"
fi
echo ""

# --- Verification ---
echo "[5/5] Verifying installation..."
cd "$PROJECT_ROOT/backend"
source .venv/bin/activate
python3 -c "
from jones_framework.core.condition_state import ConditionState
from jones_framework.perception.tda_pipeline import TDAPipeline
from jones_framework.perception.regime_classifier import RegimeClassifier
print('  Python imports: OK')
" 2>/dev/null || echo "  WARNING: Python import check failed"

cd "$PROJECT_ROOT/frontend"
if pnpm check --silent 2>/dev/null; then
    echo "  TypeScript check: OK"
else
    echo "  WARNING: TypeScript check had issues (may still work)"
fi
echo ""

# --- Done ---
echo "=========================================="
echo "  Setup Complete!"
echo "=========================================="
echo ""
echo "  To start the platform:"
echo ""
echo "  Terminal 1 (backend):"
echo "    cd backend"
echo "    source .venv/bin/activate"
echo "    uvicorn jones_framework.api.server:app --reload --port 8000"
echo ""
echo "  Terminal 2 (frontend):"
echo "    cd frontend"
echo "    pnpm dev"
echo ""
echo "  Then open: http://localhost:3000"
echo ""
echo "  Run tests: bash scripts/consumer-tests.sh"
echo "  Full test plan: docs/TEST_PLAN.md"
echo ""
