#!/usr/bin/env bash

set -euo pipefail

TRITON_PID=""
CLIENT_PID=""

tritonserver \
    --model-repository=/models \
    --model-control-mode=explicit \
    --allow-http=true \
    --allow-grpc=true \
    --allow-metrics=true \
    --log-verbose=1 \
    --log-file=/logs/triton.log \
    --exit-on-error=false \
    --strict-model-config=false \
    --backend-config=python,shm-default-byte-size=1048576 &
TRITON_PID=$!

cleanup() {
    local exit_code=$?
    trap - EXIT SIGINT SIGTERM

    if [[ -n "${CLIENT_PID}" ]] && kill -0 "${CLIENT_PID}" 2>/dev/null; then
        kill "${CLIENT_PID}" 2>/dev/null || true
        wait "${CLIENT_PID}" 2>/dev/null || true
    fi

    if [[ -n "${TRITON_PID}" ]] && kill -0 "${TRITON_PID}" 2>/dev/null; then
        kill "${TRITON_PID}" 2>/dev/null || true
        wait "${TRITON_PID}" 2>/dev/null || true
    fi

    echo "Cleanup complete!"
    exit "${exit_code}"
}

trap cleanup EXIT SIGINT SIGTERM

# Wait for triton server to be ready
while true; do
    if ! kill -0 "${TRITON_PID}" 2>/dev/null; then
        echo "Triton Server exited before becoming ready."
        wait "${TRITON_PID}" 2>/dev/null || true
        exit 1
    fi

    status_code=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/v2/health/ready || echo "000")
    if [ "$status_code" -eq 200 ]; then
        echo "Triton Server is ready!"
        break
    else
        echo "Triton Server not ready yet. Waiting..."
        sleep 2
    fi
done

# Start Triton client application
export PLAYER_BACKEND_URL="http://172.17.0.1:8702"
export RUST_LOG=INFO
cd /app && ./client &
CLIENT_PID=$!

# Wait for cargo process to finish
wait "${CLIENT_PID}"
