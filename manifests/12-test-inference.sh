#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Inference smoke test — Gemma 270M on RHOAI 3.4 (GPU time-slicing)
# Run after deploy.sh completes and InferenceService reaches Ready status.
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../config/config.env"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

log_info()    { echo -e "${CYAN}[INFO]${NC} $*"; }
log_success() { echo -e "${GREEN}[OK]${NC} $*"; }
log_error()   { echo -e "${RED}[FAIL]${NC} $*"; }
log_section() { echo -e "\n${BOLD}${CYAN}━━━ $* ━━━${NC}"; }

# --- Get endpoint URL from InferenceService status ---
log_info "Getting endpoint URL from InferenceService..."
ENDPOINT_URL=$(oc get inferenceservice gemma-270m \
    -n "${NAMESPACE}" \
    -o jsonpath='{.status.url}' 2>/dev/null || echo "")

if [ -z "${ENDPOINT_URL}" ]; then
    log_info "URL not in status — checking Route..."
    ENDPOINT_URL=$(oc get route -n "${NAMESPACE}" \
        -o jsonpath='{.items[0].spec.host}' 2>/dev/null || echo "")
    if [ -n "${ENDPOINT_URL}" ]; then
        ENDPOINT_URL="https://${ENDPOINT_URL}"
    fi
fi

if [ -z "${ENDPOINT_URL}" ]; then
    log_error "Cannot get endpoint URL"
    echo "  → Check: oc get inferenceservice gemma-270m -n ${NAMESPACE}"
    echo "  → For port-forward set: ENDPOINT_URL=http://localhost:8080"
    echo ""
    read -rp "Enter endpoint URL manually (e.g. http://localhost:8080): " ENDPOINT_URL
fi

log_info "Endpoint: ${ENDPOINT_URL}"

timed_curl() {
    local start end elapsed
    start=$(date +%s%N)
    curl --insecure "$@"
    end=$(date +%s%N)
    elapsed=$(( (end - start) / 1000000 ))
    echo -e "\n${YELLOW}Response time: ${elapsed} ms${NC}" >&2
}

echo ""
echo -e "${BOLD}══════════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}        Inference Test — Gemma 3 270M (GPU time-slicing)      ${NC}"
echo -e "${BOLD}══════════════════════════════════════════════════════════════${NC}"

# =============================================================================
# TEST 1: GET /v1/models
# =============================================================================
log_section "Test 1: Available models (GET /v1/models)"

echo -e "${CYAN}Request:${NC} GET ${ENDPOINT_URL}/v1/models"
echo "---"
MODELS_RESPONSE=$(timed_curl \
    --silent \
    --fail \
    --max-time 30 \
    "${ENDPOINT_URL}/v1/models" \
    || echo '{"error": "request failed"}')

echo "${MODELS_RESPONSE}" | python3 -m json.tool 2>/dev/null || echo "${MODELS_RESPONSE}"

if echo "${MODELS_RESPONSE}" | grep -q "gemma"; then
    log_success "Model 'google/gemma-3-270m' is loaded and responding"
else
    log_error "Model not responding or name mismatch"
fi

# =============================================================================
# TEST 2: GPU time-slicing explanation
# =============================================================================
log_section "Test 2: GPU time-slicing question (POST /v1/completions)"

PROMPT="Explain GPU time-slicing in one sentence."
echo -e "${CYAN}Prompt:${NC} ${PROMPT}"
echo "---"

RESPONSE=$(timed_curl \
    --silent \
    --fail \
    --max-time 60 \
    --header "Content-Type: application/json" \
    --data "{
        \"model\": \"${MODEL_NAME}\",
        \"prompt\": \"${PROMPT}\",
        \"max_tokens\": 128,
        \"temperature\": 0.7
    }" \
    "${ENDPOINT_URL}/v1/completions" \
    || echo '{"error": "request failed"}')

ANSWER=$(echo "${RESPONSE}" | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    print(data['choices'][0]['text'])
except:
    print('[Failed to parse response]')
" 2>/dev/null || echo "${RESPONSE}")

echo -e "${GREEN}Response:${NC}"
echo "${ANSWER}"

# =============================================================================
# TEST 3: OpenShift AI question
# =============================================================================
log_section "Test 3: OpenShift AI question (POST /v1/completions)"

PROMPT3="What is OpenShift AI and what are its main use cases?"
echo -e "${CYAN}Prompt:${NC} ${PROMPT3}"
echo "---"

RESPONSE3=$(timed_curl \
    --silent \
    --fail \
    --max-time 60 \
    --header "Content-Type: application/json" \
    --data "{
        \"model\": \"${MODEL_NAME}\",
        \"prompt\": \"${PROMPT3}\",
        \"max_tokens\": 128,
        \"temperature\": 0.7
    }" \
    "${ENDPOINT_URL}/v1/completions" \
    || echo '{"error": "request failed"}')

ANSWER3=$(echo "${RESPONSE3}" | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    print(data['choices'][0]['text'])
except:
    print('[Failed to parse response]')
" 2>/dev/null || echo "${RESPONSE3}")

echo -e "${GREEN}Response:${NC}"
echo "${ANSWER3}"

# =============================================================================
# Summary
# =============================================================================
echo ""
echo -e "${BOLD}══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}${BOLD}  Inference tests complete                                    ${NC}"
echo -e "${BOLD}══════════════════════════════════════════════════════════════${NC}"
echo ""
echo "To run more tests:"
echo "  curl -k ${ENDPOINT_URL}/v1/completions \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"model\":\"${MODEL_NAME}\",\"prompt\":\"Your prompt here\",\"max_tokens\":128}'"
echo ""
