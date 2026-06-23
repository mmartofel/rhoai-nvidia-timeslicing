#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# setup.sh — Configure NVIDIA GPU time-slicing on the GPU Operator
#
# Run this ONCE (or after GPU Operator updates) to enable 5 virtual GPUs
# per physical T4. Idempotent — safe to re-run.
#
# Prerequisites:
#   - oc CLI logged in to zenek-hqxqx cluster
#   - NVIDIA GPU Operator installed in nvidia-gpu-operator namespace
#
# After setup.sh succeeds, run scripts/deploy.sh to deploy the Gemma model.
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

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
log_step()    { echo -e "\n${BOLD}${CYAN}▶ $*${NC}"; }

die() {
    log_error "$*"
    exit 1
}

echo ""
echo -e "${BOLD}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║   NVIDIA GPU Time-Slicing Setup — RHOAI 3.4                  ║${NC}"
echo -e "${BOLD}║   Cluster: zenek-hqxqx (us-east-2)                           ║${NC}"
echo -e "${BOLD}║   Target:  5 virtual GPUs per physical T4 node               ║${NC}"
echo -e "${BOLD}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# =============================================================================
# Step 1: Verify cluster access and GPU Operator
# =============================================================================
log_step "Step 1/4: Verifying cluster access"

if ! command -v oc &>/dev/null; then
    die "'oc' CLI not found in PATH. Install it first."
fi

if ! oc whoami &>/dev/null; then
    die "Not logged in to OpenShift. Run: oc login <cluster-url>"
fi

log_success "Logged in as $(oc whoami) @ $(oc whoami --show-server 2>/dev/null || echo 'unknown')"

if ! oc get namespace nvidia-gpu-operator &>/dev/null; then
    die "Namespace 'nvidia-gpu-operator' not found. Is the NVIDIA GPU Operator installed?"
fi

CLUSTER_POLICY=$(oc get clusterpolicy gpu-cluster-policy -n nvidia-gpu-operator \
    -o jsonpath='{.metadata.name}' 2>/dev/null || echo "")
if [ -z "${CLUSTER_POLICY}" ]; then
    die "ClusterPolicy 'gpu-cluster-policy' not found in nvidia-gpu-operator."
fi

log_success "GPU Operator and ClusterPolicy present"

# =============================================================================
# Step 2: Apply time-slicing ConfigMap
# =============================================================================
log_step "Step 2/4: Applying time-slicing ConfigMap (5 replicas per GPU)"

oc apply -f "${REPO_DIR}/manifests/02-time-slicing-config.yaml"
log_success "time-slicing-config ConfigMap applied to nvidia-gpu-operator"

# =============================================================================
# Step 3: Patch ClusterPolicy to activate time-slicing
# =============================================================================
log_step "Step 3/4: Patching ClusterPolicy to reference time-slicing-config"

oc patch clusterpolicy gpu-cluster-policy \
    -n nvidia-gpu-operator \
    --type=merge \
    -p '{
      "spec": {
        "devicePlugin": {
          "config": {
            "name": "time-slicing-config",
            "default": "any"
          }
        }
      }
    }'

log_success "ClusterPolicy patched"

log_info "Waiting for nvidia-device-plugin DaemonSet rollout (may take 60-90s)..."
sleep 10  # give the operator time to react to the patch

if ! oc rollout status daemonset/nvidia-device-plugin-daemonset \
    -n nvidia-gpu-operator \
    --timeout=180s 2>/dev/null; then
    log_warn "Device plugin DaemonSet rollout timeout — checking status manually..."
    oc get daemonset nvidia-device-plugin-daemonset -n nvidia-gpu-operator 2>/dev/null || true
    log_warn "Continuing — nodes may still be updating. Re-run scripts/setup.sh to verify."
else
    log_success "nvidia-device-plugin-daemonset rollout complete"
fi

# Give nodes a moment to update their allocatable resources
log_info "Waiting 15s for node resource advertisements to update..."
sleep 15

# =============================================================================
# Step 4: Verify time-slicing is active
# =============================================================================
log_step "Step 4/4: Verifying time-slicing (nvidia.com/gpu allocatable per node)"

echo ""
echo -e "${BOLD}GPU node status:${NC}"
oc get nodes -l nvidia.com/gpu.present=true \
    -o custom-columns='NAME:.metadata.name,STATUS:.status.conditions[-1].type,GPU:.status.allocatable.nvidia\.com/gpu'
echo ""

# Check that at least one node shows 5 GPUs
GPU_COUNT=$(oc get nodes -l 'nvidia.com/gpu.present=true' \
    -o jsonpath='{.items[0].status.allocatable.nvidia\.com/gpu}' 2>/dev/null || echo "0")

if [ "${GPU_COUNT}" = "5" ]; then
    echo -e "${GREEN}${BOLD}"
    echo "  ✅ Time-slicing active: ${GPU_COUNT} virtual GPUs per node"
    echo "     Each of the 5 pods in deploy.sh will claim 1 virtual GPU,"
    echo "     sharing the same physical T4 via kernel-level time-slicing."
    echo -e "${NC}"
elif [ "${GPU_COUNT}" -gt 1 ] 2>/dev/null; then
    log_warn "Allocatable GPUs: ${GPU_COUNT} (expected 5). Nodes may still be updating."
    echo "  → Wait 60s and re-check: oc get nodes -l nvidia.com/gpu.present=true -o custom-columns='NAME:.metadata.name,GPU:.status.allocatable.nvidia\.com/gpu'"
else
    log_warn "Allocatable GPUs: '${GPU_COUNT}'. Time-slicing may not be active yet."
    echo "  → Check GPU Operator pods: oc get pods -n nvidia-gpu-operator"
    echo "  → Check device plugin config: oc get cm time-slicing-config -n nvidia-gpu-operator -o yaml"
    echo "  → Nodes may need 1-2 minutes to advertise updated resources"
fi

echo ""
echo -e "${BOLD}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║  Setup complete. Next step:                                  ║${NC}"
echo -e "${BOLD}║                                                              ║${NC}"
echo -e "${BOLD}║    export HF_TOKEN=hf_xxxxxxxxxxxx                           ║${NC}"
echo -e "${BOLD}║    # (or set it in config/config.env)                        ║${NC}"
echo -e "${BOLD}║    ./scripts/deploy.sh                                       ║${NC}"
echo -e "${BOLD}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
