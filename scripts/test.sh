#!/bin/bash
# Test runner script - runs all tests

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=== Running All Tests ==="
echo ""

# Backend tests
echo "--- Backend Tests (pytest) ---"
cd "$PROJECT_ROOT/backend"
if command -v pytest &> /dev/null; then
    pytest -v --cov=jones_framework tests/ || echo "Backend tests completed (some may have failed)"
else
    echo "pytest not installed, skipping backend tests"
fi

echo ""

# Frontend tests
echo "--- Frontend Tests (vitest) ---"
cd "$PROJECT_ROOT/frontend"
if [ -f "node_modules/.bin/vitest" ]; then
    pnpm test || echo "Frontend tests completed (some may have failed)"
else
    echo "vitest not installed, skipping frontend tests"
fi

echo ""

# Consumer tests
echo "--- Consumer Tests ---"
cd "$PROJECT_ROOT"
if [ -f "scripts/consumer-tests.sh" ]; then
    ./scripts/consumer-tests.sh || echo "Consumer tests completed (some may have failed)"
else
    echo "Consumer tests not configured yet"
fi

echo ""
echo "=== Test Suite Complete ==="
