#!/bin/bash

# Run backend server in dev mode
cd ./backend && uv run uvicorn main:app --reload --host 0.0.0.0 --port 8702 &
BACKEND_PID=$!

# Run frontend in dev mode
cd frontend && npm run dev &
FRONTEND_PID=$!

# Define cleanup function
cleanup() {
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    wait $BACKEND_PID $FRONTEND_PID 2>/dev/null
    exit
}

# Trap SIGINT and SIGTERM
trap cleanup SIGINT SIGTERM EXIT

# Keep script running
wait 