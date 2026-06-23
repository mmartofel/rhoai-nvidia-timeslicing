#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Status check for Gemma 270M deployment (5× time-sliced T4) on RHOAI 3.4
# Usage: ./scripts/status.sh [--watch]
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
log_error()   { echo -e "${RED}[FAIL]${NC} $*"; }
log_section() { echo -e "\n${BOLD}${CYAN}━━━ $* ━━━${NC}"; }

WATCH_MODE=false
[ "${1:-}" = "--watch" ] && WATCH_MODE=true

print_status() {

echo ""
echo -e "${BOLD}╔══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║   Gemma 270M — GPU Time-Slicing — $(date '+%Y-%m-%d %H:%M:%S')   ║${NC}"
echo -e "${BOLD}╚══════════════════════════════════════════════════════════════════╝${NC}"

# =============================================================================
# InferenceService status
# =============================================================================
log_section "InferenceService"

if ! oc get inferenceservice gemma-270m -n "${NAMESPACE}" &>/dev/null; then
    log_error "InferenceService 'gemma-270m' not found in namespace '${NAMESPACE}'"
    echo "  → Run deployment: ./scripts/deploy.sh"
    return 1
fi

oc get inferenceservice gemma-270m -n "${NAMESPACE}" \
    -o custom-columns=\
'NAME:.metadata.name,READY:.status.conditions[?(@.type=="Ready")].status,URL:.status.url' \
    2>/dev/null || oc get inferenceservice gemma-270m -n "${NAMESPACE}"

echo ""
log_info "Status conditions:"
oc get inferenceservice gemma-270m -n "${NAMESPACE}" \
    -o jsonpath='{range .status.conditions[*]}  {.type}: {.status} ({.reason}) — {.message}{"\n"}{end}' \
    2>/dev/null || true

# =============================================================================
# Pod status
# =============================================================================
log_section "Pods in namespace '${NAMESPACE}'"

oc get pods -n "${NAMESPACE}" -o wide --no-headers 2>/dev/null | \
    awk 'BEGIN{printf "%-50s %-12s %-8s %-40s\n","POD","STATUS","READY","NODE"}
         {printf "%-50s %-12s %-8s %-40s\n",$1,$4,$2,$7}'

echo ""
log_info "Time-slicing validation (all ${GPU_REPLICAS} pods should land on the same GPU node):"

NODES_LIST=$(oc get pods -n "${NAMESPACE}" -o wide --no-headers 2>/dev/null | \
    grep -v "Completed\|Evicted\|Error" | awk '{print $7}' | sort || true)
UNIQUE_NODES=$(echo "${NODES_LIST}" | grep -v '^$' | sort -u | wc -l | tr -d ' ')
TOTAL_PODS=$(echo "${NODES_LIST}" | grep -v '^$' | wc -l | tr -d ' ')

if [ "${TOTAL_PODS}" -eq 0 ]; then
    log_warn "No predictor pods yet — deployment may still be starting"
elif [ "${UNIQUE_NODES}" -eq 1 ] && [ "${TOTAL_PODS}" -ge "${GPU_REPLICAS}" ]; then
    GPU_NODE=$(echo "${NODES_LIST}" | grep -v '^$' | sort -u)
    log_success "Time-slicing OK: ${TOTAL_PODS}/${GPU_REPLICAS} pods on node '${GPU_NODE}'"
elif [ "${TOTAL_PODS}" -lt "${GPU_REPLICAS}" ]; then
    log_warn "${TOTAL_PODS}/${GPU_REPLICAS} pods running — still scheduling"
else
    log_warn "${TOTAL_PODS} pods across ${UNIQUE_NODES} nodes — expected 1 node for time-slicing"
fi

# =============================================================================
# GPU usage per running pod
# =============================================================================
log_section "GPU usage (nvidia-smi)"

RUNNING_PODS=$(oc get pods -n "${NAMESPACE}" --no-headers 2>/dev/null | \
    grep "Running" | awk '{print $1}' || true)

if [ -z "${RUNNING_PODS}" ]; then
    log_warn "No Running pods — skipping GPU check"
else
    for POD in ${RUNNING_PODS}; do
        POD_NODE=$(oc get pod "${POD}" -n "${NAMESPACE}" \
            -o jsonpath='{.spec.nodeName}' 2>/dev/null || echo "unknown")
        GPU_INFO=$(oc exec "${POD}" -n "${NAMESPACE}" -- \
            nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu \
            --format=csv,noheader,nounits 2>/dev/null || echo "nvidia-smi unavailable")
        echo -e "  ${CYAN}${POD}${NC} (${POD_NODE})"
        echo "    GPU: ${GPU_INFO}"
    done
fi

# =============================================================================
# Recent warning events
# =============================================================================
log_section "Warning events (last 10 min)"

oc get events -n "${NAMESPACE}" \
    --sort-by='.lastTimestamp' \
    --field-selector 'type!=Normal' \
    2>/dev/null | tail -20 || true

echo ""
log_info "Full events: oc get events -n ${NAMESPACE} --sort-by=.lastTimestamp"

} # end print_status

if [ "${WATCH_MODE}" = true ]; then
    log_info "Watch mode (--watch). Refreshing every 30s. Ctrl+C to stop."
    while true; do
        clear
        print_status || true
        sleep 30
    done
else
    print_status
fi
