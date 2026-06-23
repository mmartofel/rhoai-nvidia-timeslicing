#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Remove Gemma 270M deployment from the cluster
# Usage:
#   ./scripts/undeploy.sh                   # delete resources, keep namespace
#   ./scripts/undeploy.sh --delete-namespace # delete everything including namespace
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

DELETE_NAMESPACE=false
for arg in "$@"; do
    [ "${arg}" = "--delete-namespace" ] && DELETE_NAMESPACE=true
done

echo ""
echo -e "${BOLD}${RED}══════════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}${RED}  Removing Gemma 270M deployment from RHOAI                   ${NC}"
echo -e "${BOLD}${RED}══════════════════════════════════════════════════════════════${NC}"
echo ""

log_warn "Namespace:  ${NAMESPACE}"
log_warn "Resources to remove: InferenceService, ServingRuntime, Playground,"
log_warn "  storage-config Secret, s3-data-connection Secret, huggingface-token Secret,"
log_warn "  s3-data-connection-sa ServiceAccount, transfer Jobs"
if [ "${DELETE_NAMESPACE}" = true ]; then
    log_warn "  + namespace '${NAMESPACE}' (ALL resources)"
fi
echo ""
read -rp "Proceed with removal? [y/N]: " CONFIRM
case "${CONFIRM:-N}" in
    [Yy]*) ;;
    *) echo "Cancelled."; exit 0 ;;
esac

# --- InferenceService ---
log_info "Deleting InferenceService 'gemma-270m'..."
if oc get inferenceservice gemma-270m -n "${NAMESPACE}" &>/dev/null; then
    oc delete inferenceservice gemma-270m -n "${NAMESPACE}"
    log_success "InferenceService 'gemma-270m' deleted"
else
    log_warn "InferenceService 'gemma-270m' not found — skipping"
fi

# --- ServingRuntime ---
log_info "Deleting ServingRuntime 'vllm-cuda-runtime'..."
if oc get servingruntime vllm-cuda-runtime -n "${NAMESPACE}" &>/dev/null; then
    oc delete servingruntime vllm-cuda-runtime -n "${NAMESPACE}"
    log_success "ServingRuntime 'vllm-cuda-runtime' deleted"
else
    log_warn "ServingRuntime 'vllm-cuda-runtime' not found — skipping"
fi

# --- Playground ---
log_info "Deleting Playground 'lsd-genai-playground'..."
if oc get llamastackdistribution lsd-genai-playground -n "${NAMESPACE}" &>/dev/null 2>&1; then
    oc delete llamastackdistribution lsd-genai-playground -n "${NAMESPACE}"
    log_success "Playground 'lsd-genai-playground' deleted"
else
    log_warn "Playground 'lsd-genai-playground' not found — skipping"
fi

# --- Secrets ---
for SECRET in storage-config s3-data-connection hf-token; do
    log_info "Deleting Secret '${SECRET}'..."
    if oc get secret "${SECRET}" -n "${NAMESPACE}" &>/dev/null; then
        oc delete secret "${SECRET}" -n "${NAMESPACE}"
        log_success "Secret '${SECRET}' deleted"
    else
        log_warn "Secret '${SECRET}' not found — skipping"
    fi
done

# --- ServiceAccount ---
log_info "Deleting ServiceAccount 's3-data-connection-sa'..."
if oc get serviceaccount s3-data-connection-sa -n "${NAMESPACE}" &>/dev/null; then
    oc delete serviceaccount s3-data-connection-sa -n "${NAMESPACE}"
    log_success "ServiceAccount 's3-data-connection-sa' deleted"
else
    log_warn "ServiceAccount 's3-data-connection-sa' not found — skipping"
fi

# --- Jobs ---
log_info "Cleaning up Jobs..."
oc delete job gemma-model-transfer -n "${NAMESPACE}" --ignore-not-found &>/dev/null && \
    log_success "Job 'gemma-model-transfer' deleted" || true

# --- Stray Error pods ---
STRAY_PODS=$(oc get pods -n "${NAMESPACE}" --no-headers 2>/dev/null | \
    awk '$4 == "Error" || $4 == "OOMKilled" || $4 == "CrashLoopBackOff" {print $1}' || true)
if [ -n "${STRAY_PODS}" ]; then
    log_info "Deleting stray pods in Error/OOMKilled/CrashLoopBackOff state..."
    echo "${STRAY_PODS}" | xargs oc delete pod -n "${NAMESPACE}" --ignore-not-found
    log_success "Stray pods deleted"
fi

# --- Namespace ---
if [ "${DELETE_NAMESPACE}" = false ]; then
    echo ""
    read -rp "Also delete namespace '${NAMESPACE}'? This removes ALL remaining resources. [y/N]: " DEL_NS
    case "${DEL_NS:-N}" in
        [Yy]*) DELETE_NAMESPACE=true ;;
    esac
fi

if [ "${DELETE_NAMESPACE}" = true ]; then
    log_info "Deleting namespace '${NAMESPACE}'..."
    if oc get namespace "${NAMESPACE}" &>/dev/null; then
        oc delete namespace "${NAMESPACE}"
        log_success "Namespace '${NAMESPACE}' deleted"
        echo ""
        log_info "To redeploy from scratch: ./scripts/deploy.sh"
    else
        log_warn "Namespace '${NAMESPACE}' not found — skipping"
    fi
else
    log_info "Namespace '${NAMESPACE}' kept"
    echo ""
    log_info "To redeploy: ./scripts/deploy.sh --skip-transfer"
fi

echo ""
log_success "Undeploy complete"
