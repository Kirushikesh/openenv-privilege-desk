#!/bin/bash
set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <model-name>"
    echo "  e.g.: $0 Qwen/Qwen3-1.7B"
    exit 1
fi

MODEL_NAME="$1"
MODEL_SAFE=$(echo "$MODEL_NAME" | tr '/:' '--' | tr -cd '[:alnum:]_-')
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_ID="${TIMESTAMP}_${MODEL_SAFE}"

OUTPUT_DIR="./outputs/${RUN_ID}"
mkdir -p "${OUTPUT_DIR}"

# Pick a random free port to avoid collision when multiple jobs land on the same node
MASTER_PORT=$(( 29500 + RANDOM % 1000 ))
UV_BIN=$(which uv)

echo "Model     : ${MODEL_NAME}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Job name  : ${RUN_ID}"
echo "Port      : ${MASTER_PORT}"
echo "uv        : ${UV_BIN}"

export CUDA_LAUNCH_BLOCKING=1

bsub -q normal -M 128 -n 1 \
  -gpu "num=2:gmodel=NVIDIAA100_SXM4_80GB" \
  -o "${OUTPUT_DIR}/output.txt" \
  -e "${OUTPUT_DIR}/error.txt" \
  -J "${RUN_ID}" \
  -env "MASTER_PORT=${MASTER_PORT},VLLM_USE_V1=0" \
  "${UV_BIN}" run python training/train_grpo_openenv.py \
    --model-id     "${MODEL_NAME}" \
    --dataset-size 1000 \
    --output-dir   "${OUTPUT_DIR}" \
    --report-to    "tensorboard" \
    --seed         42 \
    --debug
