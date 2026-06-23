# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Scripts and manifests for demonstrating NVIDIA GPU time-slicing on RHOAI 3.4 on the `zenek-hqxqx` OCP cluster (us-east-2). Time-slicing splits each physical T4 GPU into 7 virtual GPUs, allowing 7 independent vLLM instances to run concurrently. Each instance serves `google/gemma-3-270m` and is deployed as a KServe `InferenceService` replica.

The deployment is fully integrated with RHOAI Dashboard: the `time-slicing` namespace appears as a project, the model server is visible in the project's Models view, and the model is available in Gen AI Studio Playground via `LlamaStackDistribution`.

## Cluster context

- Cluster: `zenek-hqxqx`, region `us-east-2`
- 3 GPU worker nodes (T4, time-sliced 7× each → `nvidia.com/gpu: 7` per node)
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

**GPU memory and vLLM T4 constraints**: T4 GPUs (Turing, compute capability 7.5) have a 64 KB shared memory limit per block. vLLM 0.18 default CUDA graph and chunked prefill Triton kernels require 80 KB — impossible on T4. Three flags are required in the ServingRuntime: `--gpu-memory-utilization 0.14`, `--max-model-len 1024`, `--no-enable-chunked-prefill`, `--enforce-eager`. The 0.14 ceiling is derived from 1/7 of total VRAM so all 7 time-sliced instances fit simultaneously. `--enforce-eager` disables CUDA graph capture (which also calls the over-limit Triton kernel during profiling).

**Gemma 3 does not support float16**: Adding `--dtype float16` to vLLM causes a hard validation error (`gemma3_text does not support float16, use bfloat16 or float32`). Do not add this flag. vLLM auto-selects float32 on T4 as a bfloat16 fallback.

**Model storage**: `google/gemma-3-270m` is downloaded from HuggingFace to MinIO once (via `06-model-transfer-job.yaml.template`). The InferenceService reads from `s3://gemma-models/gemma-3-270m/`. This eliminates HF API dependency at pod startup.

**S3 credentials via storage-config**: The InferenceService uses `spec.predictor.model.storage.key: s3-data-connection` (no `storageUri`). KServe reads the `storage-config` secret (key `s3-data-connection`, JSON blob with MinIO credentials) and injects it as a single `STORAGE_CONFIG` env var into the storage-initializer. This avoids the CredentialBuilder path (SA annotation → individual `secretKeyRef` env vars) that causes `spec.initContainers[0].env[2].valueFrom: Invalid value` conflicts under KServe's `reinvocationPolicy: IfNeeded` when the ODH pod webhook also modifies the pod.

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

When calling `envsubst`, always pass an explicit variable list to avoid corrupting llama-stack config syntax (e.g. `${env.VLLM_API_TOKEN_1:=fake}`) in the Playground template:
```bash
envsubst '${NAMESPACE} ${INFERENCE_ENDPOINT}' < manifests/11-playground.yaml.template | oc apply -f -
```
