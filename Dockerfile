# ─── Stage 1 : build de l'environnement Python ───────────────────────────────
FROM python:3.12-slim AS base

# Dépendances système
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Installer Node.js 20 (pour supergateway)
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Installer supergateway (stdio → SSE/HTTP bridge)
RUN npm install -g supergateway

WORKDIR /app

# Cloner le projet
RUN git clone https://github.com/natifridman/stocks-mcp.git .

# Installer les dépendances Python via pip (fallback si pas de lock file uv)
RUN pip install --no-cache-dir fastmcp requests

# Vérifier que le server existe bien
RUN ls -la my_server.py

# ─── Stage final ─────────────────────────────────────────────────────────────
FROM base AS final

WORKDIR /app

# Port SSE exposé
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Lancement : supergateway enveloppe le server stdio et l'expose en SSE
CMD ["supergateway", \
     "--stdio", "python my_server.py", \
     "--port", "8000", \
     "--baseUrl", "http://localhost:8000", \
     "--ssePath", "/sse", \
     "--messagePath", "/message"]
