#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# deploy.sh — Deploy 7× Gemma 270M on RHOAI 3.4 (GPU time-slicing)
#
# Flow:
#   0. Prerequisites check
#   1. Validate HuggingFace token
#   2. Create namespace 'time-slicing' (RHOAI Dashboard visible)
#   3. Deploy MinIO (cluster-local model storage)
#   4. Create secrets (HF token, S3 credentials, KServe SA)
#   5. Apply AcceleratorProfile (NVIDIA T4)
#   6. Run model transfer Job (google/gemma-3-270m → MinIO)
#   7. Deploy ServingRuntime + InferenceService (5 replicas, 1 GPU each)
#   8. Deploy Playground (LlamaStackDistribution for RHOAI Gen AI Studio)
#   9. Run inference smoke test
#
# Usage: ./scripts/deploy.sh [--skip-prereqs] [--skip-transfer] [--no-test]
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${REPO_DIR}/config/config.env"

# Defaults for variables that may be absent in older config.env files
: "${GPU_REPLICAS:=5}"
: "${GPU_MEMORY_UTILIZATION:=0.14}"
: "${MAX_MODEL_LEN:=1024}"
: "${NUM_GPU_BLOCKS_OVERRIDE:=200}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

SKIP_PREREQS=false
SKIP_TRANSFER=false
NO_TEST=false

for arg in "$@"; do
    case "${arg}" in
        --skip-prereqs)  SKIP_PREREQS=true ;;
        --skip-transfer) SKIP_TRANSFER=true ;;
        --no-test)       NO_TEST=true ;;
    esac
done

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
echo -e "${BOLD}║   Gemma 270M deployment — RHOAI 3.4 / GPU Time-Slicing       ║${NC}"
echo -e "${BOLD}║   ${GPU_REPLICAS} InferenceService replicas on a single T4 node          ║${NC}"
echo -e "${BOLD}║   Model storage: MinIO (cluster-local S3)                    ║${NC}"
echo -e "${BOLD}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# =============================================================================
# Step 0: Prerequisites
# =============================================================================
log_step "Step 0/9: Prerequisites check"

if [ "${SKIP_PREREQS}" = true ]; then
    log_warn "Skipping prereqs check (--skip-prereqs)"
else
    bash "${REPO_DIR}/manifests/00-prerequisites-check.sh" || \
        die "Prerequisites check failed. Fix errors and retry."
fi

# =============================================================================
# Step 1: HuggingFace token
# =============================================================================
log_step "Step 1/9: HuggingFace token validation"

if [ "${SKIP_TRANSFER}" = true ]; then
    log_info "Skipping model transfer (--skip-transfer) — HF token not required"
elif [ -z "${HF_TOKEN:-}" ]; then
    log_warn "HF_TOKEN not set in config/config.env"
    echo ""
    read -rsp "Enter HF_TOKEN (hidden input): " HF_TOKEN
    echo ""
    if [ -z "${HF_TOKEN}" ]; then
        die "HF_TOKEN is required to download google/gemma-3-270m (gated model)"
    fi
    log_success "HF_TOKEN provided interactively"
else
    log_success "HF_TOKEN set (length: ${#HF_TOKEN} chars)"
fi

export HF_TOKEN NAMESPACE MINIO_ACCESS_KEY MINIO_SECRET_KEY MINIO_ENDPOINT MINIO_BUCKET \
       MODEL_NAME MODEL_S3_PATH GPU_REPLICAS GPU_MEMORY_UTILIZATION MAX_MODEL_LEN NUM_GPU_BLOCKS_OVERRIDE

# =============================================================================
# Step 2: Namespace
# =============================================================================
log_step "Step 2/9: Creating namespace '${NAMESPACE}'"

NS_PHASE=$(oc get namespace "${NAMESPACE}" -o jsonpath='{.status.phase}' 2>/dev/null || echo "")
if [ "${NS_PHASE}" = "Terminating" ]; then
    log_warn "Namespace '${NAMESPACE}' is Terminating — waiting (max 120s)..."
    WAIT_NS=0
    until [ "${WAIT_NS}" -ge 120 ] || ! oc get namespace "${NAMESPACE}" &>/dev/null; do
        WAIT_NS=$((WAIT_NS + 5))
        sleep 5
    done
    oc get namespace "${NAMESPACE}" &>/dev/null && \
        die "Namespace '${NAMESPACE}' still exists — check: oc get namespace ${NAMESPACE}"
fi

oc apply -f "${REPO_DIR}/manifests/01-namespace.yaml"
log_success "Namespace '${NAMESPACE}' ready (visible in RHOAI Dashboard as 'GPU Time-Slicing Demo')"

# =============================================================================
# Step 3: MinIO
# =============================================================================
log_step "Step 3/9: MinIO in namespace '${MINIO_NAMESPACE}'"

MINIO_READY=$(oc get deployment minio -n "${MINIO_NAMESPACE}" \
    -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo "0")

if [ "${MINIO_READY:-0}" -ge 1 ] 2>/dev/null; then
    log_info "MinIO already running in '${MINIO_NAMESPACE}' — verifying credentials..."
    MINIO_POD=$(oc get pod -n "${MINIO_NAMESPACE}" -l app=minio \
        -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
    if [ -n "${MINIO_POD}" ]; then
        if oc exec "${MINIO_POD}" -n "${MINIO_NAMESPACE}" -- \
            mc alias set check http://localhost:9000 \
                "${MINIO_ACCESS_KEY}" "${MINIO_SECRET_KEY}" &>/dev/null; then
            log_success "MinIO accessible with configured credentials — skipping deployment"
        else
            die "MinIO is running but credentials in config.env do not match. Update MINIO_ACCESS_KEY/MINIO_SECRET_KEY to match the existing MinIO, or remove the existing deployment first."
        fi
    else
        log_warn "MinIO deployment is ready but no pod found — proceeding to bucket step"
    fi
else
    oc apply -f "${REPO_DIR}/manifests/03-minio.yaml"

    log_info "Waiting for MinIO to be ready (max 120s)..."
    if ! oc rollout status deployment/minio -n "${MINIO_NAMESPACE}" --timeout=120s; then
        log_error "MinIO did not become ready in 120s"
        echo "  oc get pods -n ${MINIO_NAMESPACE}"
        echo "  oc logs deployment/minio -n ${MINIO_NAMESPACE}"
        die "Check MinIO logs above"
    fi
    log_success "MinIO ready: ${MINIO_ENDPOINT}"
fi

# =============================================================================
# Step 4: Secrets
# =============================================================================
log_step "Step 4/9: Creating secrets"

if ! command -v envsubst &>/dev/null; then
    die "'envsubst' not found. Install: brew install gettext (macOS) or apt-get install gettext (Linux)"
fi

envsubst '${NAMESPACE} ${HF_TOKEN}' \
    < "${REPO_DIR}/manifests/04-hf-secret.yaml.template" | oc apply -f -
log_success "Secret 'hf-token' applied"

envsubst '${NAMESPACE} ${MINIO_ACCESS_KEY} ${MINIO_SECRET_KEY} ${MINIO_ENDPOINT} ${MINIO_BUCKET} ${MODEL_S3_PATH}' \
    < "${REPO_DIR}/manifests/05-s3-connection.yaml.template" | oc apply -f -
log_success "Secret 's3-data-connection' applied"

envsubst '${NAMESPACE} ${MINIO_ACCESS_KEY} ${MINIO_SECRET_KEY} ${MINIO_ENDPOINT} ${MINIO_BUCKET}' \
    < "${REPO_DIR}/manifests/10-kserve-model-sa.yaml.template" | oc apply -f -
log_success "Secret 'storage-config' and ServiceAccount 's3-data-connection-sa' applied"

# =============================================================================
# Step 5: AcceleratorProfile
# =============================================================================
log_step "Step 5/9: AcceleratorProfile (NVIDIA T4)"

oc apply -f "${REPO_DIR}/manifests/07-accelerator-profile.yaml" 2>/dev/null || \
    log_warn "AcceleratorProfile could not be applied (CRD may not exist yet — continuing)"
log_success "AcceleratorProfile applied"

# =============================================================================
# Step 6: Model transfer (HuggingFace → MinIO)
# =============================================================================
log_step "Step 6/9: Model transfer google/gemma-3-270m → MinIO"

if [ "${SKIP_TRANSFER}" = true ]; then
    log_warn "Skipping model transfer (--skip-transfer)"
else
    MODEL_EXISTS=false
    MINIO_POD=$(oc get pod -n "${MINIO_NAMESPACE}" -l app=minio \
        -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")

    if [ -n "${MINIO_POD}" ]; then
        if oc exec "${MINIO_POD}" -n "${MINIO_NAMESPACE}" -- \
            ls "/data/${MINIO_BUCKET}/${MODEL_S3_PATH}/config.json" &>/dev/null 2>&1; then
            MODEL_EXISTS=true
        fi
    fi

    if [ "${MODEL_EXISTS}" = true ]; then
        log_success "Model already in MinIO (s3://${MINIO_BUCKET}/${MODEL_S3_PATH}/) — skipping transfer"
    else
        log_info "Model not found in MinIO — starting transfer Job..."
        log_warn "Downloading ~540 MB from HuggingFace — may take up to 30 minutes"

        oc delete job gemma-model-transfer -n "${NAMESPACE}" --ignore-not-found &>/dev/null

        envsubst '${NAMESPACE}' \
            < "${REPO_DIR}/manifests/06-model-transfer-job.yaml.template" | oc apply -f -
        log_success "Job 'gemma-model-transfer' started"

        log_info "Waiting for transfer to complete (timeout: $((TRANSFER_TIMEOUT_SECONDS / 60)) minutes)..."
        log_info "Monitor progress: oc logs -f job/gemma-model-transfer -n ${NAMESPACE}"
        echo ""

        if ! oc wait --for=condition=complete job/gemma-model-transfer \
            -n "${NAMESPACE}" \
            --timeout="${TRANSFER_TIMEOUT_SECONDS}s"; then
            log_error "Transfer Job did not complete in time"
            echo "  Logs:"
            oc logs job/gemma-model-transfer -n "${NAMESPACE}" --tail=50 || true
            echo ""
            echo "  Diagnose:"
            echo "    oc describe job gemma-model-transfer -n ${NAMESPACE}"
            echo "    oc logs job/gemma-model-transfer -n ${NAMESPACE}"
            die "Model transfer failed. Check logs above."
        fi

        log_success "Model transferred to MinIO: s3://${MINIO_BUCKET}/${MODEL_S3_PATH}/"
    fi
fi

# =============================================================================
# Step 7: ServingRuntime + InferenceService
# =============================================================================
log_step "Step 7/9: Deploying ServingRuntime and InferenceService (${GPU_REPLICAS} replicas)"

oc delete servingruntime vllm-timeslicing-runtime -n "${NAMESPACE}" \
    --ignore-not-found &>/dev/null && \
    log_info "Removed old custom ServingRuntime 'vllm-timeslicing-runtime'" || true

log_info "Processing vllm-cuda-runtime-template from redhat-ods-applications..."
if ! oc get template vllm-cuda-runtime-template -n redhat-ods-applications &>/dev/null; then
    die "Template 'vllm-cuda-runtime-template' not found in redhat-ods-applications. Is RHOAI installed?"
fi

oc process vllm-cuda-runtime-template -n redhat-ods-applications \
    --output yaml | oc apply -n "${NAMESPACE}" -f -
log_success "ServingRuntime 'vllm-cuda-runtime' applied"

oc annotate servingruntime vllm-cuda-runtime -n "${NAMESPACE}" \
    "opendatahub.io/template-name=vllm-cuda-runtime-template" \
    "opendatahub.io/template-display-name=vLLM NVIDIA GPU ServingRuntime for KServe" \
    --overwrite &>/dev/null
log_success "ServingRuntime RHOAI Dashboard annotations set"

oc patch servingruntime vllm-cuda-runtime -n "${NAMESPACE}" \
    --type=json \
    -p '[
      {"op":"add","path":"/spec/containers/0/args/-","value":"--gpu-memory-utilization"},
      {"op":"add","path":"/spec/containers/0/args/-","value":"'"${GPU_MEMORY_UTILIZATION}"'"},
      {"op":"add","path":"/spec/containers/0/args/-","value":"--max-model-len"},
      {"op":"add","path":"/spec/containers/0/args/-","value":"'"${MAX_MODEL_LEN}"'"},
      {"op":"add","path":"/spec/containers/0/args/-","value":"--no-enable-chunked-prefill"},
      {"op":"add","path":"/spec/containers/0/args/-","value":"--enforce-eager"},
      {"op":"add","path":"/spec/containers/0/args/-","value":"--attention-backend"},
      {"op":"add","path":"/spec/containers/0/args/-","value":"FLEX_ATTENTION"},
      {"op":"add","path":"/spec/containers/0/args/-","value":"--num-gpu-blocks-override"},
      {"op":"add","path":"/spec/containers/0/args/-","value":"'"${NUM_GPU_BLOCKS_OVERRIDE}"'"}
    ]' &>/dev/null
log_success "ServingRuntime patched: --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION} --max-model-len ${MAX_MODEL_LEN} --no-enable-chunked-prefill --enforce-eager --attention-backend FLEX_ATTENTION --num-gpu-blocks-override ${NUM_GPU_BLOCKS_OVERRIDE}"

oc apply -f "${REPO_DIR}/manifests/09-inference-service.yaml"
log_success "InferenceService 'gemma-270m' applied"

# The connection-isvc webhook fires only on CREATE, not UPDATE — safe to patch storage.key
# after creation. KServe's pod-mutator reads storage-config secret when this key is set.
oc patch inferenceservice gemma-270m -n "${NAMESPACE}" \
    --type=merge \
    -p '{"spec":{"predictor":{"model":{"storage":{"key":"s3-data-connection"}}}}}' &>/dev/null
log_success "InferenceService patched: storage.key=s3-data-connection (KServe S3 credentials)"

echo ""
log_info "Deployment configuration:"
echo "  Model:            s3://${MINIO_BUCKET}/${MODEL_S3_PATH}/"
echo "  Namespace:        ${NAMESPACE}"
echo "  Replicas:         ${GPU_REPLICAS} (one per time-sliced virtual GPU)"
echo "  GPU per pod:      1 virtual (time-sliced T4)"
echo "  GPU util:         ${GPU_MEMORY_UTILIZATION} (conservative for coexistence)"
echo "  Max context len:  ${MAX_MODEL_LEN} tokens"
echo ""

log_info "Waiting for InferenceService Ready (timeout: $((DEPLOY_TIMEOUT_SECONDS / 60)) minutes)..."
log_info "Monitor: ./scripts/status.sh --watch"
echo ""

START_TIME=$(date +%s)
while true; do
    ELAPSED=$(( $(date +%s) - START_TIME ))

    if [ "${ELAPSED}" -gt "${DEPLOY_TIMEOUT_SECONDS}" ]; then
        log_error "Timeout after $((DEPLOY_TIMEOUT_SECONDS / 60)) minutes"
        echo "  oc get inferenceservice gemma-270m -n ${NAMESPACE}"
        echo "  oc get pods -n ${NAMESPACE} -o wide"
        echo "  oc describe inferenceservice gemma-270m -n ${NAMESPACE}"
        exit 1
    fi

    READY_STATUS=$(oc get inferenceservice gemma-270m \
        -n "${NAMESPACE}" \
        -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}' \
        2>/dev/null || echo "Unknown")

    case "${READY_STATUS}" in
        "True")
            log_success "InferenceService reached Ready status after ${ELAPSED}s"
            break
            ;;
        "False")
            REASON=$(oc get inferenceservice gemma-270m \
                -n "${NAMESPACE}" \
                -o jsonpath='{.status.conditions[?(@.type=="Ready")].reason}' \
                2>/dev/null || echo "")
            log_warn "[${ELAPSED}s] NotReady — Reason: ${REASON:-waiting for resources}"
            ;;
        *)
            PODS_RUNNING=$(oc get pods -n "${NAMESPACE}" --no-headers 2>/dev/null | \
                grep -c "Running" || echo "0")
            log_info "[${ELAPSED}s] Status: ${READY_STATUS} — Pods Running: ${PODS_RUNNING}/${GPU_REPLICAS}"
            ;;
    esac

    sleep "${POLL_INTERVAL_SECONDS}"
done

# =============================================================================
# Step 8: Playground
# =============================================================================
log_step "Step 8/9: Deploying Playground (LlamaStackDistribution)"

INFERENCE_ENDPOINT=$(oc get inferenceservice gemma-270m \
    -n "${NAMESPACE}" \
    -o jsonpath='{.status.url}' 2>/dev/null || echo "")

if [ -z "${INFERENCE_ENDPOINT}" ]; then
    log_warn "InferenceService URL not available yet — using placeholder for Playground"
    INFERENCE_ENDPOINT="https://gemma-270m-predictor-${NAMESPACE}.apps.zenek-hqxqx.example.com"
fi

export INFERENCE_ENDPOINT
envsubst '${NAMESPACE} ${INFERENCE_ENDPOINT}' \
    < "${REPO_DIR}/manifests/11-playground.yaml.template" | oc apply -f -
log_success "Playground 'lsd-genai-playground' applied"
log_info "Playground available in RHOAI: Gen AI Studio → Playground"

# =============================================================================
# Step 9: Summary + smoke test
# =============================================================================
ENDPOINT_URL=$(oc get inferenceservice gemma-270m \
    -n "${NAMESPACE}" \
    -o jsonpath='{.status.url}' 2>/dev/null || echo "")

echo ""
echo -e "${BOLD}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║  DEPLOYMENT COMPLETE                                         ║${NC}"
echo -e "${BOLD}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BOLD}RHOAI Dashboard:${NC}"
echo "  → Projects → time-slicing (GPU Time-Slicing Demo)"
echo "  → Models → gemma-270m (${GPU_REPLICAS} running replicas)"
echo "  → Gen AI Studio → Playground → ${MODEL_NAME}"
echo ""
if [ -n "${ENDPOINT_URL}" ]; then
    echo -e "${GREEN}Inference endpoint:${NC} ${ENDPOINT_URL}"
    echo ""
    echo "  curl -k ${ENDPOINT_URL}/v1/models"
    echo "  curl -k ${ENDPOINT_URL}/v1/completions \\"
    echo "    -H 'Content-Type: application/json' \\"
    echo "    -d '{\"model\":\"${MODEL_NAME}\",\"prompt\":\"Explain GPU time-slicing.\",\"max_tokens\":128}'"
else
    log_warn "Endpoint URL not yet available — check: oc get inferenceservice gemma-270m -n ${NAMESPACE}"
fi
echo ""

log_step "Step 9/9: Inference smoke test"
if [ "${NO_TEST}" = false ]; then
    read -rp "Run inference smoke test? [Y/n]: " RUN_TESTS
    case "${RUN_TESTS:-Y}" in
        [Yy]* | "")
            bash "${REPO_DIR}/manifests/12-test-inference.sh"
            ;;
        *)
            log_info "Skipped. Run manually: ./manifests/12-test-inference.sh"
            ;;
    esac
fi
