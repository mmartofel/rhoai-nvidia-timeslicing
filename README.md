# NVIDIA GPU Time-Slicing on RHOAI 3.4

Demonstrates GPU time-slicing on OpenShift AI (RHOAI) 3.4 using a single NVIDIA T4 node. Time-slicing splits one physical GPU into 7 virtual GPUs. 5 of them are used — running 5 independent vLLM instances concurrently, each serving `google/gemma-3-270m`.

The project is visible in the RHOAI Dashboard, the inference endpoint is registered as a model server, and the model is available in the Gen AI Studio Playground.

## How it works

```
Physical T4 GPU (16 GB VRAM)
         │
         │  NVIDIA time-slicing (GPU Operator ConfigMap)
         │  migStrategy: none, replicas: 7
         │
    ┌────┴─────────────────────────────────┐
    │  7 × virtual nvidia.com/gpu          │
    │  (kernel-level time-multiplexed)     │
    └────┬────┬────┬────┬────┬─────────────┘
         │    │    │    │    │
      Pod 1  Pod 2  Pod 3  Pod 4  Pod 5
      vLLM   vLLM   vLLM   vLLM   vLLM
      Gemma  Gemma  Gemma  Gemma  Gemma
         │    │    │    │    │
         └────┴────┴─────────┘
                    │
          KServe InferenceService
          (load-balances across 5 pods)
                    │
            OpenAI-compatible API
```

Each pod requests `nvidia.com/gpu: 1` (one virtual slice). With 7 virtual GPUs per node, all 5 pods land on the same physical T4. 5 replicas are used instead of 7 — T4 memory pressure during simultaneous startup makes 7 unstable (see [Troubleshooting](#troubleshooting)).

Conservative vLLM settings are required for T4 (Turing, 64 KB shared memory per block):

| Flag | Value | Reason |
|---|---|---|
| `--gpu-memory-utilization` | `0.14` | 1/7 VRAM budget per pod — fits 5+ pods on 16 GB |
| `--max-model-len` | `1024` | Keeps KV cache allocation small |
| `--no-enable-chunked-prefill` | — | Triton chunked-prefill kernel exceeds T4 shared memory |
| `--enforce-eager` | — | Disables CUDA graph capture (avoids profiling OOM) |
| `--attention-backend FLEX_ATTENTION` | — | Triton attention kernel needs 80 KB; T4 limit is 64 KB |
| `--num-gpu-blocks-override 200` | — | Bypasses memory profiling assertion that fails when sibling pods free GPU memory mid-profile |

## Requirements

| Requirement | Details |
|---|---|
| RHOAI | 3.4 with KServe enabled |
| NVIDIA GPU Operator | Installed in `nvidia-gpu-operator` namespace |
| GPU nodes | T4 nodes with `nvidia.com/gpu.present=true` label (already provisioned on `zenek-hqxqx`) |
| Node Feature Discovery | Running for GPU labelling |
| CLI tools | `oc`, `envsubst` (brew install gettext on macOS), `curl` |
| HuggingFace token | Required — Gemma is a gated model (accept terms at huggingface.co/google/gemma-3-270m) |

## Quick start

```bash
# 1. Clone and enter the repo
git clone <repo-url>
cd rhoai-nvidia-timeslicing

# 2. Configure (set your HuggingFace token)
cp config/config.env.example config/config.env
# Edit config/config.env and fill in HF_TOKEN

# 3. Configure GPU time-slicing (run once, or after GPU Operator updates)
./scripts/setup.sh

# 4. Deploy Gemma and the Playground
./scripts/deploy.sh
```

## RHOAI Dashboard

After `./scripts/deploy.sh` completes:

| Dashboard Location | What you see |
|---|---|
| **Projects** | `GPU Time-Slicing Demo` project (namespace: `time-slicing`) |
| **Project → Models** | `gemma-270m` model server with 5 running replicas |
| **Gen AI Studio → Playground** | `google/gemma-3-270m` model available |

## Management

```bash
# Check deployment status (IS, pods, GPU usage, events)
./scripts/status.sh

# Watch status in a loop (30s refresh)
./scripts/status.sh --watch

# Local access via port-forward (localhost:8080)
./scripts/port-forward.sh

# Remove deployment (keep namespace — useful to redeploy with --skip-transfer)
./scripts/undeploy.sh

# Remove everything including the namespace (start completely fresh)
./scripts/undeploy.sh --delete-namespace

# Run inference smoke test manually
./manifests/12-test-inference.sh

# Check GPU allocation on nodes
oc get nodes -l nvidia.com/gpu.present=true \
  -o custom-columns='NAME:.metadata.name,STATUS:.status.conditions[-1].type,GPU:.status.allocatable.nvidia\.com/gpu'

# Remove time-slicing config (resets GPU Operator to default)
oc patch clusterpolicy gpu-cluster-policy -n nvidia-gpu-operator \
  --type=merge -p '{"spec":{"devicePlugin":{"config":{"name":"","default":""}}}}'
oc delete configmap time-slicing-config -n nvidia-gpu-operator
```

## API usage

The model is served under the InferenceService name `gemma-270m` (set via `--served-model-name={{.Name}}`), **not** the HuggingFace ID `google/gemma-3-270m`. Use `"model": "gemma-270m"` in all API calls.

`google/gemma-3-270m` is a **base completion model** with no built-in chat template. A minimal Gemma-format template is provided in `manifests/13-chat-template.yaml` and mounted into vLLM automatically by `deploy.sh`. This makes `/v1/chat/completions` and the Playground work, though response quality is limited (base model, not instruction-tuned). For proper assistant-style chat, switch to `google/gemma-3-270m-it`.

```bash
# Get the endpoint URL
ENDPOINT=$(oc get inferenceservice gemma-270m -n time-slicing -o jsonpath='{.status.url}')

# List models
curl -k ${ENDPOINT}/v1/models

# Text completion
curl -k ${ENDPOINT}/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-270m",
    "prompt": "Explain GPU time-slicing in one sentence.",
    "max_tokens": 128,
    "temperature": 0.7
  }'
```

## Repository structure

```
rhoai-nvidia-timeslicing/
├── README.md
├── CLAUDE.md
├── .gitignore
├── config/
│   ├── config.env.example         # Template — version-controlled
│   └── config.env                 # Your config with HF_TOKEN — gitignored
├── manifests/
│   ├── 00-prerequisites-check.sh  # Pre-flight validation
│   ├── 01-namespace.yaml          # Namespace: time-slicing (RHOAI-visible)
│   ├── 02-time-slicing-config.yaml # GPU Operator ConfigMap (7 replicas/GPU)
│   ├── 03-minio.yaml              # MinIO S3 in rhoai-model-registries
│   ├── 04-hf-secret.yaml.template # HuggingFace token secret (name: hf-token)
│   ├── 05-s3-connection.yaml.template # MinIO connection (uri type, no managed label)
│   ├── 06-model-transfer-job.yaml.template # Job: HF → MinIO transfer
│   ├── 07-accelerator-profile.yaml # NVIDIA T4 AcceleratorProfile
│   ├── 08-serving-runtime.yaml    # Reference only — deploy.sh uses RHOAI template
│   ├── 09-inference-service.yaml  # KServe InferenceService (5 replicas)
│   ├── 10-kserve-model-sa.yaml.template # storage-config Secret + SA for KServe
│   ├── 11-playground.yaml.template # LlamaStackDistribution (Playground UI)
│   ├── 12-test-inference.sh       # Inference smoke test
│   └── 13-chat-template.yaml      # Gemma chat template ConfigMap (base model workaround)
└── scripts/
    ├── setup.sh                   # Configure GPU time-slicing (run once)
    ├── deploy.sh                  # Deploy Gemma + Playground (9 steps)
    ├── status.sh                  # Show IS status, pods, GPU usage, events
    ├── port-forward.sh            # localhost:8080 → gemma-270m service
    └── undeploy.sh                # Remove deployment (optionally delete namespace)
```

## Troubleshooting

**storage-initializer fails: `S3 authentication failed` / `Unable to locate credentials`**

The ODH controller overwrote `storage-config`. This happens if `s3-data-connection` has `opendatahub.io/managed: "true"` — ODH sees a `uri`-type connection and sets `"type": ""` in `storage-config`. Fix immediately:

```bash
# Verify the damage
oc get secret storage-config -n time-slicing \
  -o jsonpath='{.data.s3-data-connection}' | base64 -d

# Recreate storage-config with correct type
oc delete secret storage-config -n time-slicing --ignore-not-found
oc create secret generic storage-config -n time-slicing \
  --from-literal=s3-data-connection='{
    "access_key_id":"minio",
    "bucket":"gemma-models",
    "endpoint_url":"http://minio.rhoai-model-registries.svc.cluster.local:9000",
    "region":"us-east-1",
    "secret_access_key":"minio123",
    "type":"s3"
  }'

# Restart pods to pick up the new secret
oc rollout restart deployment/gemma-270m-predictor -n time-slicing
```

**Pods stuck in Pending:**
```bash
oc describe pod -n time-slicing -l serving.kserve.io/inferenceservice=gemma-270m | grep -A10 Events
# Common cause: time-slicing not active (run scripts/setup.sh first) or CPU resource pressure
```

**InferenceService stuck in NotReady:**
```bash
oc describe inferenceservice gemma-270m -n time-slicing
oc get pods -n time-slicing -o wide
# Check storage-initializer logs
oc logs -n time-slicing <pod-name> -c storage-initializer
```

**Model transfer Job fails:**
```bash
oc logs job/gemma-model-transfer -n time-slicing
# Common causes: HF_TOKEN invalid, Gemma terms not accepted on HuggingFace, network timeout
```

**Playground shows no model:**
```bash
oc logs -n time-slicing -l llamastack.io/distribution=lsd-genai-playground
# The llama-stack server must be able to reach the InferenceService endpoint
```

**GPU time-slicing not active after scripts/setup.sh:**
```bash
oc get nodes -l nvidia.com/gpu.present=true \
  -o custom-columns='NAME:.metadata.name,GPU:.status.allocatable.nvidia\.com/gpu'
# If GPU column shows "1" instead of "7", wait 60-90s for device plugin to restart
oc rollout status daemonset/nvidia-device-plugin-daemonset -n nvidia-gpu-operator
```

**vLLM pods crash with `OutOfResources: shared memory`:**

T4 Triton attention kernel requires 80 KB shared memory; T4 hardware limit is 64 KB. Ensure the ServingRuntime has `--attention-backend FLEX_ATTENTION`. Check:
```bash
oc get servingruntime vllm-cuda-runtime -n time-slicing \
  -o jsonpath='{.spec.containers[0].args}'
```

**vLLM pods crash with `AssertionError: Error in memory profiling`:**

Time-sliced GPUs can violate vLLM's memory profiling assertion (another pod freeing memory looks like a gain). Ensure the ServingRuntime has `--num-gpu-blocks-override 200`.
