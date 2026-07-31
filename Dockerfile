FROM python:3.11-slim

WORKDIR /app

# System deps:
#   curl          — health checks
#   git           — uvx needs this to fetch MCP servers from git+https URLs
#   ca-certs/gnupg — NodeSource repo signing
#   nodejs (20)   — npx-launched MCP servers
#   ffmpeg        — transcodes inbound Signal voice notes (AAC/m4a) to the
#                   mono 16k mp3 that audio-capable models actually decode;
#                   without it, audio input silently degrades to text-only
#   fonts-noto-cjk — required by the I Ching composer to render hexagram
#                   names in Chinese; falls back to PIL default otherwise
#   fonts-dejavu-core — Latin serif/sans for tarot + I Ching titles
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates gnupg git ffmpeg \
        fonts-noto-cjk fonts-dejavu-core \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

# Pre-install npx-based MCP servers so first-call doesn't trigger an
# `npm install` that prints to stdout — npm's progress messages corrupt
# the MCP stdio JSONRPC stream and either spam parse errors or push the
# 30s startup-handshake budget over the line, causing brave-search to
# fail on cold-cache container starts. Install once at image build,
# then invoke the resulting binary directly (db rows use the binary
# names with empty args, no `npx`).
#
# mcp-pyodide ships a Pyodide-in-Node Python sandbox the writer LLM
# uses for real computation (correlations, regressions, options
# pricing, custom indicators). Pinned because Pyodide bundles change
# behavior across versions and the bot's quant directives in
# ask_command.py are written against a specific package set.
RUN npm install -g \
        @brave/brave-search-mcp-server@latest \
        mcp-pyodide@1.0.0

# Ships the SDK pin for uvx-launched MCP servers that haven't migrated to the
# 2.x SDK. Deliberately NOT set as a global ENV: servers opt in individually by
# pointing UV_CONSTRAINT at this path in their own env config, so a server that
# has caught up resolves freely. See mcp-server-constraints.txt.
COPY mcp-server-constraints.txt .

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

# Drop root: everything from here on (gunicorn, MCP stdio subprocesses,
# LLM tool calls) runs as an unprivileged user, so a compromise of any
# of those paths doesn't get root-in-container. UID 1000 matches the
# host owner of the bind-mounted data/ and logs/ dirs.
RUN useradd --uid 1000 --create-home --shell /usr/sbin/nologin botuser \
    && chown -R botuser:botuser /app
USER botuser

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
