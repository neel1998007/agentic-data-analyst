# ── Backend Dockerfile ────────────────────────────────────────────────────────
# Builds the FastAPI application image.
FROM python:3.11-slim

# ── Environment ───────────────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── System dependencies ───────────────────────────────────────────────────────
# gcc / g++  — compile C extensions pulled in by uvicorn[standard]
#              (httptools, uvloop).
# libpq-dev  — included for completeness; psycopg2-binary is a pre-built
#              wheel so it is not strictly required, but keeps the image
#              forward-compatible if the binary wheel ever falls back to
#              source build.
# libgomp1   — OpenMP runtime for sentence-transformers / torch CPU threading.
# curl       — health-check probes and container debugging.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        libpq-dev \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ───────────────────────────────────────────────────────
# Copy only requirements.txt first to maximise Docker layer cache reuse.
# A code-only rebuild will skip this (slow) step entirely.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application source ────────────────────────────────────────────────────────
COPY src/      src/
COPY scripts/  scripts/

# ── Port ──────────────────────────────────────────────────────────────────────
EXPOSE 8000

# ── Default command ───────────────────────────────────────────────────────────
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
