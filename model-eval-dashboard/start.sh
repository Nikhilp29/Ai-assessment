#!/bin/bash
# Runs FastAPI (internal port 8000) and Streamlit (public port) together.
# PUBLIC_PORT uses $PORT if the host provides one (Render), else 7860
# (Hugging Face Spaces' fixed port, if ever deployed there instead).
set -e
PUBLIC_PORT="${PORT:-7860}"

echo "Starting FastAPI backend on port 8000..."
uvicorn app.main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

echo "Waiting for backend health check..."
for i in $(seq 1 30); do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "Backend is up."
        break
    fi
    sleep 1
done

echo "Starting Streamlit frontend on port $PUBLIC_PORT..."
export DASHBOARD_API_URL="http://localhost:8000"
streamlit run frontend/streamlit_app.py \
    --server.port "$PUBLIC_PORT" \
    --server.address 0.0.0.0 \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection false

kill $BACKEND_PID
