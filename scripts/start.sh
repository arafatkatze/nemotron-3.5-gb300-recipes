#!/usr/bin/env bash
set -euo pipefail
recipe_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
state="${1:-candidate}"
case "$state" in
  candidate) state_args=(--mamba-ssm-cache-dtype float16 --enable-mamba-cache-stochastic-rounding --mamba-cache-philox-rounds 5) ;;
  control) state_args=(--mamba-ssm-cache-dtype float32) ;;
  *) echo 'Usage: scripts/start.sh [candidate|control] [--dry-run]' >&2; exit 2 ;;
esac
model_path="${MODEL_PATH:-$recipe_root/model}"
results_path="${RESULTS_PATH:-$recipe_root/runs}"
container_name="${CONTAINER_NAME:-nemotron35-recipes}"
image='vllm/vllm-openai@sha256:fd33bec1578f312a01547cc6b9691a5154c3e8bcfb76d2b35acb657bfc4eee8f'
cmd=(docker run -d --name "$container_name" --gpus all --memory 360g --shm-size 16g
  -p 127.0.0.1:8015:8015 -v "$model_path:/model:ro" -v "$recipe_root:/recipe:ro"
  -v "$results_path:/results" -v nemotron35-recipe-cache:/root/.cache
  -e OMP_NUM_THREADS=1 -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1
  --entrypoint vllm "$image" serve /model --served-model-name nemotron-3.5-super-vl
  --host 0.0.0.0 --port 8015 --trust-remote-code --max-model-len 16384
  --max-num-seqs 256 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.80
  --no-enable-prefix-caching --limit-mm-per-prompt '{"image":0,"video":0,"audio":0}'
  --reasoning-parser nemotron_v3 --seed 1234 --async-scheduling --mamba-backend flashinfer
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}' "${state_args[@]}")
if [[ "${2:-}" == '--dry-run' ]]; then printf '%q ' "${cmd[@]}"; printf '\n'; exit 0; fi
if [[ ! -f "$model_path/config.json" ]]; then echo "Download the pinned model first: $model_path" >&2; exit 1; fi
if docker container inspect "$container_name" >/dev/null 2>&1; then
  echo "Container $container_name already exists; inspect and remove that exact container before switching arms." >&2; exit 1
fi
mkdir -p "$results_path"
"${cmd[@]}"
echo "Wait for readiness: curl --fail http://127.0.0.1:8015/health"
