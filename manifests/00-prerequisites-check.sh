#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Prerequisites check for Gemma 270M on RHOAI 3.4 (GPU time-slicing)
# Run before deploy.sh or standalone: ./manifests/00-prerequisites-check.sh
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../config/config.env"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

ERRORS=0

log_info()    { echo -e "${CYAN}[INFO]${NC} $*"; }
log_success() { echo -e "${GREEN}[OK]${NC} $*"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error()   { echo -e "${RED}[FAIL]${NC} $*"; ERRORS=$((ERRORS + 1)); }

echo ""
echo -e "${BOLD}══════════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}  Prerequisites check: Gemma 270M on RHOAI 3.4                ${NC}"
echo -e "${BOLD}  GPU time-slicing — zenek-hqxqx cluster                      ${NC}"
echo -e "${BOLD}══════════════════════════════════════════════════════════════${NC}"
echo ""

# --- 1. oc CLI ---
log_info "Checking oc CLI..."
if ! command -v oc &>/dev/null; then
    log_error "'oc' not found in PATH"
    echo "  → Install OpenShift CLI: https://docs.openshift.com/container-platform/latest/cli_reference/openshift_cli/getting-started-cli.html"
    ERRORS=$((ERRORS + 1))
else
    log_success "'oc' CLI available: $(oc version --client --short 2>/dev/null || oc version --client 2>/dev/null | head -1)"
fi

# --- 2. Cluster login ---
log_info "Checking cluster login..."
if ! oc whoami &>/dev/null; then
    log_error "Not logged in to OpenShift cluster"
    echo "  → Run: oc login <cluster-url>"
    ERRORS=$((ERRORS + 1))
else
    CURRENT_USER=$(oc whoami)
    CURRENT_SERVER=$(oc whoami --show-server 2>/dev/null || echo "unknown")
    log_success "Logged in as: ${CURRENT_USER} @ ${CURRENT_SERVER}"
fi

if [ "${ERRORS}" -gt 0 ]; then
    echo ""
    log_error "Critical errors before cluster connection. Fix them and retry."
    exit 1
fi

# --- 3. RHOAI DataScienceCluster ---
log_info "Checking RHOAI installation (DataScienceCluster)..."
if ! oc get crd datascienceclusters.datasciencecluster.opendatahub.io &>/dev/null; then
    log_error "CRD DataScienceCluster not found — RHOAI may not be installed"
else
    DSC_COUNT=$(oc get datasciencecluster --all-namespaces --no-headers 2>/dev/null | wc -l | tr -d ' ')
    if [ "${DSC_COUNT}" -eq 0 ]; then
        log_error "No DataScienceCluster found"
    else
        DSC_NAME=$(oc get datasciencecluster --all-namespaces --no-headers 2>/dev/null | awk '{print $2}' | head -1)
        DSC_NS=$(oc get datasciencecluster --all-namespaces --no-headers 2>/dev/null | awk '{print $1}' | head -1)
        RHOAI_VERSION=$(oc get csv -n redhat-ods-operator --no-headers 2>/dev/null | grep -i "rhods\|rhoai\|opendatahub" | awk '{print $5}' | head -1 || echo "unknown")
        log_success "DataScienceCluster '${DSC_NAME}' (ns: ${DSC_NS}) — RHOAI version: ${RHOAI_VERSION}"
    fi
fi

# --- 4. KServe CRD ---
log_info "Checking KServe CRD (InferenceService)..."
if oc get crd inferenceservices.serving.kserve.io &>/dev/null; then
    log_success "CRD inferenceservices.serving.kserve.io exists"
else
    log_error "CRD inferenceservices.serving.kserve.io not found"
    echo "  → KServe must be enabled in RHOAI DataScienceCluster"
    echo "  → Check: oc get datasciencecluster -o yaml | grep -A3 kserve"
fi

# --- 5. GPU nodes ---
log_info "Checking GPU nodes (nvidia.com/gpu.present=true)..."
GPU_NODES=$(oc get nodes -l 'nvidia.com/gpu.present=true' --no-headers 2>/dev/null | wc -l | tr -d ' ')
if [ "${GPU_NODES}" -eq 0 ]; then
    log_error "No nodes with label nvidia.com/gpu.present=true"
    echo "  → Check: oc get nodes --show-labels | grep nvidia"
    echo "  → NFD and GPU Operator must be installed and running"
else
    log_success "Found ${GPU_NODES} GPU node(s)"
    oc get nodes -l 'nvidia.com/gpu.present=true' --no-headers 2>/dev/null | \
        awk '{printf "    → Node: %-50s Status: %s\n", $1, $2}'
fi

# --- 6. GPU time-slicing verification ---
log_info "Checking GPU time-slicing (nvidia.com/gpu allocatable)..."
GPU_COUNT=$(oc get nodes -l 'nvidia.com/gpu.present=true' \
    -o jsonpath='{.items[0].status.allocatable.nvidia\.com/gpu}' 2>/dev/null || echo "0")
if [ "${GPU_COUNT}" -eq 5 ] 2>/dev/null; then
    log_success "Time-slicing active: ${GPU_COUNT} virtual GPUs per node"
elif [ "${GPU_COUNT}" -eq 1 ] 2>/dev/null; then
    log_warn "GPU present but time-slicing not configured (allocatable: ${GPU_COUNT})"
    echo "  → Run scripts/setup.sh first to enable time-slicing"
else
    log_warn "GPU allocatable: '${GPU_COUNT}' — run scripts/setup.sh to verify time-slicing"
fi

# --- 7. GPU Operator ---
log_info "Checking GPU Operator (nvidia-gpu-operator)..."
GPU_OP_NS="nvidia-gpu-operator"
if ! oc get namespace "${GPU_OP_NS}" &>/dev/null; then
    GPU_OP_NS=$(oc get namespace --no-headers 2>/dev/null | awk '{print $1}' | grep -i "gpu\|nvidia" | head -1 || echo "")
    if [ -z "${GPU_OP_NS}" ]; then
        log_warn "GPU Operator namespace not found (expected: nvidia-gpu-operator)"
        echo "  → Check: oc get namespaces | grep -i gpu"
    fi
fi

if [ -n "${GPU_OP_NS}" ]; then
    RUNNING_PODS=$(oc get pods -n "${GPU_OP_NS}" --no-headers 2>/dev/null | grep -c "Running" || echo "0")
    TOTAL_PODS=$(oc get pods -n "${GPU_OP_NS}" --no-headers 2>/dev/null | wc -l | tr -d ' ')
    if [ "${RUNNING_PODS}" -gt 0 ]; then
        log_success "GPU Operator (ns: ${GPU_OP_NS}): ${RUNNING_PODS}/${TOTAL_PODS} pods Running"
    else
        log_error "GPU Operator (ns: ${GPU_OP_NS}): no pods in Running state"
        echo "  → Check: oc get pods -n ${GPU_OP_NS}"
    fi
fi

# --- 8. AcceleratorProfile CRD ---
log_info "Checking AcceleratorProfile CRD..."
if ! oc get crd acceleratorprofiles.dashboard.opendatahub.io &>/dev/null; then
    log_warn "CRD AcceleratorProfile not found — GPU profile won't register in Dashboard"
    echo "  → deploy.sh will apply AcceleratorProfile after RHOAI Dashboard is installed"
else
    if oc get acceleratorprofile nvidia-t4 -n redhat-ods-applications &>/dev/null; then
        log_success "AcceleratorProfile 'nvidia-t4' already exists in redhat-ods-applications"
    else
        log_warn "AcceleratorProfile 'nvidia-t4' not found — will be created by deploy.sh (step 6)"
    fi
fi

# --- 9. envsubst ---
log_info "Checking envsubst..."
if ! command -v envsubst &>/dev/null; then
    log_error "'envsubst' not found — required for template rendering"
    echo "  → macOS: brew install gettext"
    echo "  → Linux: apt-get install gettext / yum install gettext"
else
    log_success "'envsubst' available"
fi

# --- Summary ---
echo ""
echo -e "${BOLD}══════════════════════════════════════════════════════════════${NC}"
if [ "${ERRORS}" -eq 0 ]; then
    echo -e "${GREEN}${BOLD}  All prerequisites met — ready to run deploy.sh             ${NC}"
else
    echo -e "${RED}${BOLD}  Found ${ERRORS} error(s). Fix them before deploying.          ${NC}"
fi
echo -e "${BOLD}══════════════════════════════════════════════════════════════${NC}"
echo ""

[ "${ERRORS}" -eq 0 ] || exit 1
