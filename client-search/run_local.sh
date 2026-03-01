#!/bin/bash

# Run triton server
docker compose -f ../docker-compose-triton.yml up -d

# Wait for triton server to be ready
while true; do
    status_code=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/v2/health/ready)
    if [ "$status_code" -eq 200 ]; then
        echo "Triton Server is ready!"
        break
    else
        echo "Triton Server not ready yet. Waiting..."
        sleep 2
    fi
done

# Start Triton client application
export RUST_LOG=INFO
cd client && cargo run --release &
CARGO_PID=$!

# Define cleanup function
cleanup() {
    kill $CARGO_PID 2>/dev/null
    wait $CARGO_PID 2>/dev/null
    
    # Stop Triton Server
    docker compose -f ../docker-compose-triton.yml down
    exit
}

# Trap SIGINT and SIGTERM
trap cleanup SIGINT SIGTERM EXIT

# Wait for cargo process to finish
wait $CARGO_PID