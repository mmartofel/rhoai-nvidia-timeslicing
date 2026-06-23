#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Port-forward Gemma 270M InferenceService to localhost for local access
# Usage: ./scripts/port-forward.sh [LOCAL_PORT]
# Default local port: 8080
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${REPO_DIR}/config/config.env"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

log_info()    { echo -e "${CYAN}[INFO]${NC} $*"; }
log_success() { echo -e "${GREEN}[OK]${NC} $*"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error()   { echo -e "${RED}[FAIL]${NC} $*" >&2; }

LOCAL_PORT="${1:-8080}"

echo ""
echo -e "${BOLD}══════════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}  Port-Forward: gemma-270m → localhost:${LOCAL_PORT}${NC}"
echo -e "${BOLD}══════════════════════════════════════════════════════════════${NC}"
echo ""

log_info "Looking for Service for 'gemma-270m' in namespace '${NAMESPACE}'..."

SERVICE_NAME=""
for LABEL_SELECTOR in \
    "serving.kserve.io/inferenceservice=gemma-270m" \
    "app=gemma-270m" \
    "component=predictor"; do

    FOUND=$(oc get service -n "${NAMESPACE}" \
        -l "${LABEL_SELECTOR}" \
        --no-headers 2>/dev/null | awk '{print $1}' | grep -v "knative\|metrics" | head -1 || echo "")
    if [ -n "${FOUND}" ]; then
        SERVICE_NAME="${FOUND}"
        log_success "Found Service: '${SERVICE_NAME}' (via label: ${LABEL_SELECTOR})"
        break
    fi
done

if [ -z "${SERVICE_NAME}" ]; then
    log_warn "Could not find Service automatically. Available Services in namespace '${NAMESPACE}':"
    echo ""
    oc get service -n "${NAMESPACE}" 2>/dev/null || true
    echo ""
    read -rp "Enter Service name for port-forward: " SERVICE_NAME
fi

if [ -z "${SERVICE_NAME}" ]; then
    log_error "No Service name provided — cannot start port-forward"
    exit 1
fi

SERVICE_PORT=$(oc get service "${SERVICE_NAME}" -n "${NAMESPACE}" \
    -o jsonpath='{.spec.ports[0].port}' 2>/dev/null || echo "8080")

log_info "Service: ${SERVICE_NAME} (port: ${SERVICE_PORT})"
log_info "Running: oc port-forward service/${SERVICE_NAME} ${LOCAL_PORT}:${SERVICE_PORT} -n ${NAMESPACE}"

echo ""
echo -e "${BOLD}━━━ Once port-forward is running, use these commands: ━━━${NC}"
echo ""
echo -e "${CYAN}# List models:${NC}"
echo "  curl http://localhost:${LOCAL_PORT}/v1/models | python3 -m json.tool"
echo ""
echo -e "${CYAN}# Inference request:${NC}"
echo "  curl http://localhost:${LOCAL_PORT}/v1/completions \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{"
echo "      \"model\": \"${MODEL_NAME}\","
echo "      \"prompt\": \"Explain GPU time-slicing in one sentence.\","
echo "      \"max_tokens\": 128"
echo "    }' | python3 -m json.tool"
echo ""
echo -e "${CYAN}# Health check:${NC}"
echo "  curl http://localhost:${LOCAL_PORT}/health"
echo ""
echo -e "${YELLOW}Ctrl+C to stop port-forward${NC}"
echo ""

exec oc port-forward \
    "service/${SERVICE_NAME}" \
    "${LOCAL_PORT}:${SERVICE_PORT}" \
    -n "${NAMESPACE}"
