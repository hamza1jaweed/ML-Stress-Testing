#!/usr/bin/env bash
# Start backend (FastAPI) and frontend (Vite) in parallel for local dev.
# Usage: ./run.sh   — Ctrl+C stops both.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

cleanup() {
  echo ""
  echo "Shutting down..."
  kill 0 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# --- backend -----------------------------------------------------------
if [ ! -d ".venv" ]; then
  echo "Creating Python virtualenv..."
  python3 -m venv .venv || python -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate 2>/dev/null || source .venv/Scripts/activate

pip install -q -r backend/requirements.txt
echo "Starting FastAPI on :8000..."
uvicorn backend.main:app --reload --port 8000 &

# --- frontend ----------------------------------------------------------
cd frontend
if [ ! -d "node_modules" ]; then
  echo "Installing frontend dependencies..."
  npm install
fi
echo "Starting Vite on :5173..."
npm run dev &

wait
