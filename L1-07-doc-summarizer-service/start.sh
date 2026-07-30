#!/bin/bash
# This container runs BOTH processes:
#   - FastAPI/uvicorn on internal port 8000 (never exposed externally)
#   - Streamlit on the platform's public port (what visitors actually see)
# Streamlit talks to FastAPI over localhost, same as running them as two
# separate terminals locally.
#
# The public port varies by host:
#   - Render (and most PaaS hosts) inject a $PORT env var to bind to
#   - Hugging Face Spaces (Docker SDK) expects a fixed port, 7860
# PUBLIC_PORT below defaults to 7860 so it works unchanged on either.
set -e
PUBLIC_PORT="${PORT:-7860}"

echo "Starting FastAPI backend on port 8000..."
uvicorn app.main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Wait for the backend to actually be ready instead of a fixed sleep
echo "Waiting for backend health check..."
for i in $(seq 1 30); do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "Backend is up."
        break
    fi
    sleep 1
done

echo "Starting Streamlit frontend on port $PUBLIC_PORT..."
export SUMMARIZER_API_URL="http://localhost:8000"
streamlit run frontend/streamlit_app.py \
    --server.port "$PUBLIC_PORT" \
    --server.address 0.0.0.0 \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection false

# If streamlit exits, kill the backend too so the container actually stops
kill $BACKEND_PID
