FROM python:3.11-slim

WORKDIR /app

# System deps:
#   curl          — health checks
#   git           — uvx needs this to fetch MCP servers from git+https URLs
#   ca-certs/gnupg — NodeSource repo signing
#   nodejs (20)   — npx-launched MCP servers
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates gnupg git \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ src/

# Tarot deck downloader. The deck itself lives under ./data/tarot/ which is
# bind-mounted from the host as a persistent volume, so a fresh clone
# downloads once and rebuilds reuse the existing cards. The bot triggers
# this script in a background thread on startup if any cards are missing.
COPY scripts/download_tarot.py scripts/download_tarot.py

# Create logs directory
RUN mkdir -p logs

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Run with gunicorn for production. gthread workers let a single process
# hold many concurrent connections — required for the /admin/live SSE
# endpoint, which would otherwise pin the only sync worker and freeze the
# rest of the admin UI. The bot's signal/MCP/LLM async work runs on its
# own loop in a separate thread (set up in build_app), so changing the
# Flask worker class doesn't affect that.
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--worker-class", "gthread", "--threads", "8", "--timeout", "120", "src.main:create_gunicorn_app()"]
