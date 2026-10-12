# Cortex container image
#
# Base: official Python 3.12 slim image (Debian). Note: Red Hat UBI 9 was tried
# first, but it ships SQLite 3.34 and ChromaDB requires SQLite >= 3.35.
#
# The image runs as a non-root user and is OpenShift-compatible: OpenShift
# starts containers with a random user ID that belongs to group 0, so every
# folder the app writes to is made writable by group 0.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/app \
    ANONYMIZED_TELEMETRY=False \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# 1. Dependencies first, so a code change doesn't reinstall everything.
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# 2. Download Chroma's ONNX MiniLM embedding model at build time (into
#    $HOME/.cache/chroma), so the container doesn't need internet to embed documents.
RUN python -c "from chromadb.utils import embedding_functions as ef; ef.ONNXMiniLM_L6_V2()(['warm up'])"

# 3. Application code.
COPY . .

# 4. Non-root user 1001 in group 0. Group 0 owns /app and has the same
#    permissions as the owner, so a random OpenShift UID (also in group 0)
#    can write chroma_db/, temp_uploads/ and the model cache.
RUN useradd --uid 1001 --gid 0 --home-dir /app --no-create-home appuser && \
    mkdir -p chroma_db temp_uploads && \
    chgrp -R 0 /app && \
    chmod -R g=u /app
USER 1001

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

# Needs ANTHROPIC_API_KEY at runtime:
#   docker run -p 8501:8501 -e ANTHROPIC_API_KEY=... cortex
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
