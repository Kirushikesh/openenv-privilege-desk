#!/bin/bash
set -e

# Start environment server
echo "Starting environment server..."
nohup uv run python -m server.app > training/http_logs1.txt 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# Wait for server startup
echo "Waiting for server to be ready..."
sleep 60

# Verify server is running
curl -f http://localhost:8001/health || {
    echo "Server failed to start"
    kill $SERVER_PID
    exit 1
}

echo "Server ready. Starting training..."

# Run training
# uv run python training/train_grpo.py \
#   --model-id "Qwen/Qwen3.5-2B" \
#   --env-url "http://localhost:8001" \
#   --phase all \
#   --episodes-per-phase 32 \
#   --num-generations 8 \
#   --lora-rank 16 \
#   --output-dir ./outputs/grpo_run1

uv run python training/train_grpo.py \
  --model-id Qwen/Qwen3-1.7B \
  --env-url "http://localhost:8001" \
  --use-vllm --vllm-mode colocate \
  --episodes-per-phase 32 \
  --num-generations 8 \
  --output-dir ./outputs/grpo_run2

# Cleanup
echo "Training complete. Shutting down server..."
kill $SERVER_PID
