FROM python:3.13-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml my_server.py ./

# Installer les dépendances dans le Python système
RUN pip install --no-cache-dir --break-system-packages \
    "fastmcp>=2.2.5" \
    "aiohttp>=3.11.0" \
    "beautifulsoup4>=4.13.4"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -sf http://localhost:8000/sse || exit 1

# FastMCP SSE natif — supergateway supprimé
CMD ["python", "my_server.py"]