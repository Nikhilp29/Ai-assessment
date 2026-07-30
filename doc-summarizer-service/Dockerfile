FROM python:3.11-slim

WORKDIR /app

# System deps: curl is needed by start.sh's health-check loop
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY frontend/ ./frontend/
COPY sample_docs/ ./sample_docs/
COPY start.sh .
RUN chmod +x start.sh

# Hugging Face Spaces (Docker SDK) requires a writable HOME + Streamlit
# config dir, and expects the app on port 7860.
ENV HOME=/app \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
EXPOSE 7860

CMD ["./start.sh"]
