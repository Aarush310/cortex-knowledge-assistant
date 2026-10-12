# Cortex — container image
#
# Base: official Python 3.12 slim (Debian). Red Hat UBI 9 was evaluated first,
# but it ships SQLite 3.34 and ChromaDB requires SQLite >= 3.35.
#
# OpenShift-compatible: OpenShift runs containers as a random UID in group 0,
# so everything the app writes to is group-0 writable and root is never needed.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/app \
    CORTEX_DATA_DIR=/app/var/chroma_db \
    ANONYMIZED_TELEMETRY=False \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# Dependencies first so code changes reuse this cached layer.
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Pre-download the local ONNX embedding model so the container starts without
# fetching it (and works on clusters without internet access).
RUN python -c "from chromadb.utils import embedding_functions as ef; ef.ONNXMiniLM_L6_V2()(['warm-up'])"

COPY . .

# Non-root user 1001 in group 0; group 0 gets the owner's permissions.
RUN useradd --uid 1001 --gid 0 --home-dir /app --no-create-home appuser && \
    mkdir -p /app/var && \
    chgrp -R 0 /app && \
    chmod -R g=u /app
USER 1001

# Vector store lives here — mount a volume / PVC to keep it.
VOLUME ["/app/var"]
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "app.py", \
     "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
