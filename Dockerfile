# Cortex — container image
# Built on Red Hat's Universal Base Image (UBI 9) Python 3.11 image, so the same
# image runs under Docker, Podman and OpenShift (arbitrary non-root UID, group 0).
FROM registry.access.redhat.com/ubi9/python-311:latest

WORKDIR /opt/app-root/src

# Dependencies first so this layer is cached between code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the local ONNX embedding model at build time so the container
# starts without needing to fetch it (and works on air-gapped clusters).
RUN python -c "from chromadb.utils import embedding_functions as ef; ef.ONNXMiniLM_L6_V2()(['warm-up'])"

COPY . .

ENV CORTEX_DATA_DIR=/opt/app-root/data \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# OpenShift runs containers as a random UID in group 0: make app dirs group-writable.
USER 0
RUN mkdir -p "$CORTEX_DATA_DIR" && fix-permissions /opt/app-root -P
USER 1001

VOLUME ["/opt/app-root/data"]
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", \
     "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
