#!/bin/bash
# SupaTrupa Beta - Quick Verification
# Checks that backend is running and key endpoints respond

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

BASE_URL="http://localhost:8000"
PASS=0
FAIL=0

echo "=========================================="
echo "  SupaTrupa Beta - Quick Verify"
echo "=========================================="
echo ""

# --- Backend health ---
echo "1. Backend Health"
if curl -s "$BASE_URL/health" > /dev/null 2>&1; then
    echo "   PASS: Backend is running"
    PASS=$((PASS + 1))
else
    echo "   FAIL: Backend not responding at $BASE_URL"
    echo "   Start with: cd backend && source .venv/bin/activate && uvicorn jones_framework.api.server:app --reload --port 8000"
    FAIL=$((FAIL + 1))
fi

# --- API status ---
echo "2. API Status"
STATUS=$(curl -s "$BASE_URL/api/v1/status" 2>/dev/null)
if [ -n "$STATUS" ]; then
    REGIME=$(echo "$STATUS" | python3 -c "import sys,json; print(json.load(sys.stdin).get('active_regime','?'))" 2>/dev/null)
    EXPERTS=$(echo "$STATUS" | python3 -c "import sys,json; print(json.load(sys.stdin).get('num_experts','?'))" 2>/dev/null)
    echo "   PASS: Active regime=$REGIME, Experts=$EXPERTS"
    PASS=$((PASS + 1))
else
    echo "   FAIL: No response"
    FAIL=$((FAIL + 1))
fi

# --- Classification ---
echo "3. Classification Endpoint"
RESULT=$(curl -s -X POST "$BASE_URL/api/v1/classify" \
    -H "Content-Type: application/json" \
    -d '{"point_cloud":[[15,120,45,8500,3200],[18,140,52,9200,3400],[20,110,38,7800,3100],[12,160,60,10000,3600],[16,130,48,8800,3300]]}' 2>/dev/null)
if [ -n "$RESULT" ]; then
    REGIME_ID=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('regime_id','?'))" 2>/dev/null)
    echo "   PASS: regime_id=$REGIME_ID"
    PASS=$((PASS + 1))
else
    echo "   FAIL: No response"
    FAIL=$((FAIL + 1))
fi

# --- Dashboard Summary ---
echo "4. Dashboard Summary"
RESULT=$(curl -s -X POST "$BASE_URL/api/v1/dashboard/summary" \
    -H "Content-Type: application/json" \
    -d '{"records":[{"wob":15,"rpm":120,"rop":45,"torque":8500,"spp":3200},{"wob":18,"rpm":140,"rop":52,"torque":9200,"spp":3400},{"wob":20,"rpm":110,"rop":38,"torque":7800,"spp":3100}]}' 2>/dev/null)
if [ -n "$RESULT" ]; then
    DISPLAY=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('regime_display','?'))" 2>/dev/null)
    echo "   PASS: regime_display=$DISPLAY"
    PASS=$((PASS + 1))
else
    echo "   FAIL: No response"
    FAIL=$((FAIL + 1))
fi

# --- Frontend build ---
echo "5. Frontend Build"
if [ -d "$PROJECT_ROOT/frontend/dist" ]; then
    SIZE=$(du -sh "$PROJECT_ROOT/frontend/dist" | cut -f1)
    echo "   PASS: dist/ exists ($SIZE)"
    PASS=$((PASS + 1))
else
    echo "   SKIP: No build yet (run: cd frontend && pnpm build)"
fi

echo ""
echo "=========================================="
echo "  Results: $PASS passed, $FAIL failed"
echo "=========================================="

if [ $FAIL -gt 0 ]; then
    echo "  Some checks failed. See docs/TEST_PLAN.md for troubleshooting."
    exit 1
else
    echo "  All checks passed. System is ready for testing."
fi
