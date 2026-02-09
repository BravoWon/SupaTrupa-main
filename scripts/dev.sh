#!/bin/bash
# Development server script - starts both backend and frontend

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "Starting Unified Activity:State Platform..."

# Start backend in background
echo "Starting Python API server on port 8000..."
cd "$PROJECT_ROOT/backend"
uvicorn jones_framework.api.server:app --reload --port 8000 &
BACKEND_PID=$!

# Start frontend
echo "Starting React frontend on port 5173..."
cd "$PROJECT_ROOT/frontend"
pnpm dev &
FRONTEND_PID=$!

# Trap to cleanup on exit
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null" EXIT

echo ""
echo "Services started:"
echo "  Backend API:  http://localhost:8000"
echo "  Frontend:     http://localhost:5173"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for both processes
wait
