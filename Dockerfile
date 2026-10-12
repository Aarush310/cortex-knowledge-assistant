# Cortex container image
#
# Base: Red Hat Universal Base Image (UBI 9) with Python 3.12. UBI images run
# as a non-root user (1001) and are designed for OpenShift, which starts
# containers with a random user ID that belongs to group 0.
FROM registry.access.redhat.com/ubi9/python-312:latest

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    ANONYMIZED_TELEMETRY=False \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /opt/app-root/src

# 1. Dependencies first, so a code change doesn't reinstall everything.
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# 2. Download Chroma's ONNX MiniLM embedding model at build time, so the
#    container doesn't need internet access to embed documents.
RUN python -c "from chromadb.utils import embedding_functions as ef; ef.ONNXMiniLM_L6_V2()(['warm up'])"

# 3. Application code.
COPY . .

# OpenShift runs the container as a random UID in group 0, so every folder the
# app writes to (chroma_db/, temp_uploads/, the model cache) must be writable
# by group 0, not just by user 1001.
USER 0
RUN mkdir -p chroma_db temp_uploads && \
    chgrp -R 0 /opt/app-root/src && \
    chmod -R g=u /opt/app-root/src
USER 1001

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

# Needs ANTHROPIC_API_KEY at runtime:
#   docker run -p 8501:8501 -e ANTHROPIC_API_KEY=... cortex
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
