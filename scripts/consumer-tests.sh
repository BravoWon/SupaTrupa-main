#!/bin/bash
# Consumer Tests - End-to-end validation of the unified platform
# Covers representative endpoints from all 16 API categories (15 tests)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

BASE_URL="http://localhost:8000"
PASS=0
FAIL=0
SKIP=0

# Reusable sample data
POINT_CLOUD='[[15,120,45,8500,3200],[18,140,52,9200,3400],[20,110,38,7800,3100],[12,160,60,10000,3600],[16,130,48,8800,3300]]'
DRILL_RECORDS='[{"wob":15,"rpm":120,"rop":45,"torque":8500,"spp":3200},{"wob":18,"rpm":140,"rop":52,"torque":9200,"spp":3400},{"wob":20,"rpm":110,"rop":38,"torque":7800,"spp":3100},{"wob":12,"rpm":160,"rop":60,"torque":10000,"spp":3600},{"wob":16,"rpm":130,"rop":48,"torque":8800,"spp":3300}]'

echo "=== Consumer Tests (15 endpoint groups) ==="
echo "Testing end-to-end functionality..."
echo ""

# Helper: check if backend is running
check_backend() {
    curl -s "$BASE_URL/health" > /dev/null 2>&1
}

# Helper: POST JSON and verify response has expected field
test_post() {
    local name="$1"
    local endpoint="$2"
    local body="$3"
    local expected_field="$4"

    if ! check_backend; then
        echo "  SKIP: Backend not running"
        SKIP=$((SKIP + 1))
        return
    fi

    local result
    result=$(curl -s -w "\n%{http_code}" -X POST "$BASE_URL$endpoint" \
        -H "Content-Type: application/json" \
        -d "$body" 2>/dev/null)
    local http_code
    http_code=$(echo "$result" | tail -1)
    local response
    response=$(echo "$result" | sed '$d')

    if [ "$http_code" = "200" ]; then
        local field_value
        field_value=$(echo "$response" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('$expected_field', 'MISSING'))" 2>/dev/null || echo "PARSE_ERROR")
        if [ "$field_value" != "MISSING" ] && [ "$field_value" != "PARSE_ERROR" ]; then
            echo "  PASS: $expected_field = $field_value"
            PASS=$((PASS + 1))
        else
            echo "  FAIL: Response missing '$expected_field' field"
            FAIL=$((FAIL + 1))
        fi
    else
        echo "  FAIL: HTTP $http_code"
        FAIL=$((FAIL + 1))
    fi
}

# Helper: GET and verify response has expected field
test_get() {
    local name="$1"
    local endpoint="$2"
    local expected_field="$3"

    if ! check_backend; then
        echo "  SKIP: Backend not running"
        SKIP=$((SKIP + 1))
        return
    fi

    local result
    result=$(curl -s -w "\n%{http_code}" "$BASE_URL$endpoint" 2>/dev/null)
    local http_code
    http_code=$(echo "$result" | tail -1)
    local response
    response=$(echo "$result" | sed '$d')

    if [ "$http_code" = "200" ]; then
        local field_value
        field_value=$(echo "$response" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('$expected_field', 'MISSING'))" 2>/dev/null || echo "PARSE_ERROR")
        if [ "$field_value" != "MISSING" ] && [ "$field_value" != "PARSE_ERROR" ]; then
            echo "  PASS: $expected_field = $field_value"
            PASS=$((PASS + 1))
        else
            echo "  FAIL: Response missing '$expected_field' field"
            FAIL=$((FAIL + 1))
        fi
    else
        echo "  FAIL: HTTP $http_code"
        FAIL=$((FAIL + 1))
    fi
}

# --- Test 1: Health ---
echo "Test 1: Backend Health Check"
test_get "health" "/health" "status"
echo ""

# --- Test 2: API Status ---
echo "Test 2: API Status"
test_get "status" "/api/v1/status" "framework_available"
echo ""

# --- Test 3: Regime Classification ---
echo "Test 3: Regime Classification"
test_post "classify" "/api/v1/classify" \
    "{\"point_cloud\": $POINT_CLOUD}" "regime_id"
echo ""

# --- Test 4: TDA Features ---
echo "Test 4: TDA Feature Extraction"
test_post "tda_features" "/api/v1/tda/features" \
    "{\"point_cloud\": $POINT_CLOUD}" "betti_0"
echo ""

# --- Test 5: Drilling Ingest ---
echo "Test 5: Drilling Ingest"
test_post "drilling_ingest" "/api/v1/drilling/ingest" \
    "{\"records\": $DRILL_RECORDS}" "regime"
echo ""

# --- Test 6: Parameter Resonance Network ---
echo "Test 6: Parameter Resonance Network"
test_post "network_compute" "/api/v1/network/compute" \
    "{\"records\": $DRILL_RECORDS}" "nodes"
echo ""

# --- Test 7: Curvature Field (Geometry/Sensitivity) ---
echo "Test 7: Curvature Field"
test_post "curvature_field" "/api/v1/geometry/curvature-field" \
    "{\"records\": $DRILL_RECORDS, \"grid_size\": 5}" "points"
echo ""

# --- Test 8: Regime Fingerprint ---
echo "Test 8: Regime Fingerprint"
test_post "fingerprint" "/api/v1/tda/fingerprint" \
    "{\"point_cloud\": $POINT_CLOUD}" "fingerprint"
echo ""

# --- Test 9: Topology Forecast ---
echo "Test 9: Topology Forecast"
test_post "forecast" "/api/v1/tda/forecast" \
    "{\"signatures\": [{\"betti_0\":3,\"betti_1\":1,\"entropy_h0\":0.5,\"entropy_h1\":0.3,\"max_lifetime_h0\":2.1,\"max_lifetime_h1\":1.2,\"mean_lifetime_h0\":0.8,\"mean_lifetime_h1\":0.4,\"n_features_h0\":5,\"n_features_h1\":2},{\"betti_0\":4,\"betti_1\":1,\"entropy_h0\":0.6,\"entropy_h1\":0.35,\"max_lifetime_h0\":2.3,\"max_lifetime_h1\":1.3,\"mean_lifetime_h0\":0.9,\"mean_lifetime_h1\":0.45,\"n_features_h0\":6,\"n_features_h1\":2}], \"horizon\": 3}" "forecast"
echo ""

# --- Test 10: Shadow Attractor Analysis ---
echo "Test 10: Shadow Attractor Analysis"
test_post "shadow_attractor" "/api/v1/shadow/attractor" \
    "{\"records\": $DRILL_RECORDS}" "attractor_type"
echo ""

# --- Test 11: Advisory Recommend ---
echo "Test 11: Advisory Recommend"
test_post "advisory_recommend" "/api/v1/advisory/recommend" \
    "{\"records\": $DRILL_RECORDS, \"target_regime\": \"OPTIMAL\"}" "steps"
echo ""

# --- Test 12: Field Register ---
echo "Test 12: Field Register"
test_post "field_register" "/api/v1/field/register" \
    "{\"well_name\": \"Test-Well-A\", \"records\": $DRILL_RECORDS}" "well_name"
echo ""

# --- Test 13: Dashboard Summary ---
echo "Test 13: Dashboard Summary"
test_post "dashboard_summary" "/api/v1/dashboard/summary" \
    "{\"records\": $DRILL_RECORDS}" "regime"
echo ""

# --- Test 14: Frontend Build ---
echo "Test 14: Frontend Build"
cd "$PROJECT_ROOT/frontend"
if [ -d "dist" ]; then
    echo "  PASS: Frontend build exists ($(du -sh dist | cut -f1))"
    PASS=$((PASS + 1))
else
    echo "  INFO: No production build (run: cd frontend && pnpm build)"
    SKIP=$((SKIP + 1))
fi
echo ""

# --- Test 15: Shared Types Compilation ---
echo "Test 15: Shared Types Compilation"
cd "$PROJECT_ROOT"
if [ -f "shared/types/index.ts" ]; then
    if command -v tsc &> /dev/null; then
        if tsc --noEmit shared/types/index.ts 2>/dev/null; then
            echo "  PASS: Shared types compile successfully"
            PASS=$((PASS + 1))
        else
            echo "  WARN: Type errors in shared types"
            FAIL=$((FAIL + 1))
        fi
    else
        echo "  SKIP: TypeScript compiler not available"
        SKIP=$((SKIP + 1))
    fi
else
    echo "  FAIL: Shared types file not found"
    FAIL=$((FAIL + 1))
fi
echo ""

# --- Summary ---
TOTAL=$((PASS + FAIL + SKIP))
echo "=== Consumer Tests Complete ==="
echo "Results: $PASS passed, $FAIL failed, $SKIP skipped (out of $TOTAL)"
echo ""

if [ $FAIL -gt 0 ]; then
    exit 1
fi
