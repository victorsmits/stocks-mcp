FROM python:3.13-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY my_server.py ./

RUN pip install --no-cache-dir --break-system-packages \
    "fastmcp>=2.2.5" \
    "yfinance>=0.2.50"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -sf http://localhost:8000/sse || exit 1

CMD ["python", "my_server.py"]