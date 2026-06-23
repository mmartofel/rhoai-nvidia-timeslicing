# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Scripts and manifests for demonstrating NVIDIA GPU time-slicing on RHOAI 3.4 on the `zenek-hqxqx` OCP cluster (us-east-2). Time-slicing splits each physical T4 GPU into 7 virtual GPUs. 5 vLLM instances run concurrently (not 7 — see replica count rationale below). Each instance serves `google/gemma-3-270m` and is deployed as a KServe `InferenceService` replica.

The deployment is fully integrated with RHOAI Dashboard: the `time-slicing` namespace appears as a project, the model server is visible in the project's Models view, and the model is available in Gen AI Studio Playground via `LlamaStackDistribution`.

## Cluster context

- Cluster: `zenek-hqxqx`, region `us-east-2`
- 4 GPU worker nodes (T4, time-sliced 7× each → `nvidia.com/gpu: 7` per node)
- No non-GPU worker nodes; master nodes have `NoSchedule` taint
- GPU nodes are CPU-constrained — keep CPU requests at `100m` per pod
- **DAS operator removed 2026-06-22** — do not reinstall without coordinating; its `failurePolicy: Fail` webhook blocks all pod creation cluster-wide if the operator is deleted without removing the webhook first
- RHOAI 3.4 with KServe (standard `InferenceService`, not llm-d/LLMInferenceService)

## Prerequisites

- `oc` CLI logged into the `zenek-hqxqx` OCP cluster
- `HF_TOKEN` environment variable or `config/config.env` with a valid Hugging Face token
- NVIDIA GPU Operator installed in `nvidia-gpu-operator` namespace

## Workflow

```bash
cp config/config.env.example config/config.env
# Edit config/config.env: set HF_TOKEN

./scripts/setup.sh      # Configure GPU time-slicing on the GPU Operator (run once)
./scripts/deploy.sh     # Deploy MinIO, model transfer, InferenceService, Playground

./scripts/status.sh     # Check IS status, pods, GPU usage, warning events
./scripts/status.sh --watch            # Same, auto-refresh every 30s
./scripts/port-forward.sh              # localhost:8080 → gemma-270m
./scripts/undeploy.sh                  # Remove deployment, keep namespace
./scripts/undeploy.sh --delete-namespace  # Full teardown (use to start fresh)
```

## Key design decisions

**Time-slicing config**: The `time-slicing-config` ConfigMap (in `manifests/02-time-slicing-config.yaml`, applied to `nvidia-gpu-operator`) sets 7 virtual GPU replicas per physical GPU with `migStrategy: none`. Activated by patching `ClusterPolicy/gpu-cluster-policy` in `scripts/setup.sh`. Makes `nvidia.com/gpu: 7` allocatable on each node.

**Why not MIG**: T4 GPUs do not support MIG hardware partitioning. MIG requires A100 or H100. Time-slicing with `migStrategy: none` is the correct approach for T4.

**5 replicas, not 7**: With 7 pods starting simultaneously on one 16 GB physical GPU, pods race during memory profiling — one pod freeing memory looks like a "gain" to another pod's profiling assertion. `--num-gpu-blocks-override 200` bypasses profiling but 7 pods also hit hard memory pressure during model loading. 5 replicas is stable; 7 causes Init:CrashLoopBackOff storms at startup.

**GPU memory and vLLM T4 constraints**: T4 GPUs (Turing, compute capability 7.5) have a 64 KB shared memory limit per block. Several vLLM v1 defaults are incompatible with T4 and must be overridden in the ServingRuntime:

| Flag | Value | Why |
|---|---|---|
| `--gpu-memory-utilization` | `0.14` | 1/7 of VRAM — fits 5+ pods on one 16 GB T4 |
| `--max-model-len` | `1024` | Limits KV cache allocation |
| `--no-enable-chunked-prefill` | — | Chunked-prefill Triton kernel exceeds T4 64 KB shared mem |
| `--enforce-eager` | — | Disables CUDA graph capture (avoids profiling OOM during startup) |
| `--attention-backend FLEX_ATTENTION` | — | Default `TRITON_ATTN` backend requires 80 KB shared mem; T4 limit is 64 KB — causes `OutOfResources` during INFERENCE. Must use CLI flag, not env var (`VLLM_ATTENTION_BACKEND` is unrecognized by this vLLM build) |
| `--num-gpu-blocks-override 200` | — | vLLM v1 memory profiling asserts free memory cannot increase mid-profile; on time-sliced GPUs a sibling pod freeing memory violates this. Bypass with a fixed block count |

**Gemma 3 does not support float16**: Adding `--dtype float16` causes a hard validation error (`gemma3_text does not support float16, use bfloat16 or float32`). Do not add this flag. vLLM auto-selects float32 on T4 as a bfloat16 fallback.

**Gemma 3 base model chat template**: `google/gemma-3-270m` has no built-in chat template (transformers v4.44+). Without one, `/v1/chat/completions` returns `400 BadRequestError`. A minimal Gemma-format template is provided in `manifests/13-chat-template.yaml` (ConfigMap `gemma-chat-template`), mounted into the ServingRuntime at `/tmp/chat-template/chat.jinja` and activated via `--chat-template`. This makes the Playground (LlamaStack) work. Response quality is limited — the base model continues text rather than answering as an assistant. For proper chat, switch to `google/gemma-3-270m-it`.

**Model name in API calls**: The model is served as `gemma-270m` (the InferenceService name), set via `--served-model-name={{.Name}}` in the ServingRuntime. Always use `"model": "gemma-270m"` in API calls — not the HuggingFace ID `google/gemma-3-270m`.

**Model storage**: `google/gemma-3-270m` is downloaded from HuggingFace to MinIO once (via `06-model-transfer-job.yaml.template`). The InferenceService reads from `s3://gemma-models/gemma-3-270m/`. This eliminates HF API dependency at pod startup.

**S3 credentials via storage-config**: The InferenceService uses `spec.predictor.model.storage.key: s3-data-connection`. KServe reads the `storage-config` secret (key `s3-data-connection`, JSON blob with MinIO credentials) and injects it as `STORAGE_CONFIG` env var into the storage-initializer. This avoids the CredentialBuilder path (SA annotation → individual `secretKeyRef` env vars) that causes `spec.initContainers[0].env[2].valueFrom: Invalid value` conflicts under KServe's `reinvocationPolicy: IfNeeded` when the ODH pod webhook also modifies the pod.

**ODH controller and storage-config — critical**: The `s3-data-connection` secret must NOT have `opendatahub.io/managed: "true"`. If it does, the ODH controller sees the `uri`-type connection, creates/updates `storage-config` with `"type": ""`, and KServe's pod-mutator rejects it: `storage type must be one of [s3, hdfs, webhdfs]. storage type [] is not supported`. `opendatahub.io/dashboard: "true"` alone is sufficient for Dashboard visibility without triggering ODH reconciliation.

- `storage-config` secret must also have no `managed: "true"` label — ODH adds it back if the connection is managed, and will keep setting `"type": ""`.
- If ODH overwrites or deletes `storage-config`, recreate it directly:
  ```bash
  oc delete secret storage-config -n time-slicing --ignore-not-found
  oc create secret generic storage-config -n time-slicing \
    --from-literal=s3-data-connection='{"access_key_id":"minio","bucket":"gemma-models","endpoint_url":"http://minio.rhoai-model-registries.svc.cluster.local:9000","region":"us-east-1","secret_access_key":"minio123","type":"s3"}'
  oc rollout restart deployment/gemma-270m-predictor -n time-slicing
  ```

**`connection-isvc` webhook fires on CREATE only**: The RHOAI webhook at path `/platform-connection-isvc` reads `opendatahub.io/connections` annotation on IS CREATE and sets `storageUri`. It does not fire on UPDATE. `deploy.sh` patches `spec.predictor.model.storage.key: s3-data-connection` immediately after `oc apply -f 09-inference-service.yaml` — the webhook won't strip it because it only fires on CREATE.

**ServingRuntime source**: The `vllm-cuda-runtime` ServingRuntime is provisioned from the RHOAI-native template (`vllm-cuda-runtime-template` in `redhat-ods-applications`), not from `manifests/08-serving-runtime.yaml`. `deploy.sh` runs `oc process | oc apply` and then patches in the T4-specific args. `manifests/08-serving-runtime.yaml` documents the final patched state for reference only.

**RHOAI Dashboard visibility**: Namespace labelled `opendatahub.io/dashboard: "true"` + `modelmesh-enabled: "false"`. InferenceService labelled `opendatahub.io/dashboard: "true"`. LlamaStackDistribution for Playground.

## Namespaces

| Namespace | Purpose |
|---|---|
| `time-slicing` | Gemma InferenceService, secrets, ServingRuntime, Playground |
| `rhoai-model-registries` | MinIO S3 model storage |
| `nvidia-gpu-operator` | GPU Operator, device plugin DaemonSet, time-slicing ConfigMap |
| `redhat-ods-applications` | AcceleratorProfile (must live here for RHOAI Dashboard) |

## Scripts

`scripts/` contains all operational scripts. Each sources `config/config.env` via `REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"`.

| Script | Purpose |
|---|---|
| `scripts/setup.sh` | Configure GPU time-slicing on the GPU Operator (run once) |
| `scripts/deploy.sh` | Full 9-step deployment |
| `scripts/status.sh [--watch]` | IS status, pods, nvidia-smi per pod, warning events |
| `scripts/port-forward.sh [PORT]` | Port-forward to localhost (default 8080) |
| `scripts/undeploy.sh [--delete-namespace]` | Teardown with optional namespace deletion |

## Config pattern

All tunables are in `config/config.env` (gitignored). The `config/config.env.example` is the versioned template. `scripts/deploy.sh` sources `config/config.env` and uses `envsubst` to render `*.template` files before applying with `oc apply`.

New variables added to `config.env.example` may be absent in older `config.env` copies. `deploy.sh` sets safe defaults after sourcing:
```bash
: "${GPU_REPLICAS:=5}"
: "${GPU_MEMORY_UTILIZATION:=0.14}"
: "${MAX_MODEL_LEN:=1024}"
: "${NUM_GPU_BLOCKS_OVERRIDE:=200}"
```

When calling `envsubst`, always pass an explicit variable list to avoid corrupting llama-stack config syntax (e.g. `${env.VLLM_API_TOKEN_1:=fake}`) in the Playground template:
```bash
envsubst '${NAMESPACE} ${INFERENCE_ENDPOINT}' < manifests/11-playground.yaml.template | oc apply -f -
```
