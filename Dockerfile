# ── Base image Python 3.13 slim ───────────────────────────────────────────────
FROM python:3.13-slim

# Dépendances système
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Installer Node.js 20 (pour supergateway)
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Installer supergateway (stdio → SSE bridge pour Claude.ai)
RUN npm install -g supergateway

WORKDIR /app

# Copier les fichiers du projet
COPY pyproject.toml my_server.py ./

# Installer les dépendances directement dans le Python système
# (nécessaire car supergateway appelle 'python' sans virtualenv)
RUN pip install --no-cache-dir --break-system-packages \
    "fastmcp>=2.2.5" \
    "aiohttp>=3.11.0" \
    "beautifulsoup4>=4.13.4"

# Port SSE exposé
EXPOSE 8000

# Healthcheck sur le endpoint SSE
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -sf http://localhost:8000/sse || exit 1

# supergateway encapsule my_server.py (stdio) et l'expose en SSE HTTP
CMD ["supergateway", \
     "--stdio", "python my_server.py", \
     "--port", "8000", \
     "--baseUrl", "http://localhost:8000", \
     "--ssePath", "/sse", \
     "--messagePath", "/message"]