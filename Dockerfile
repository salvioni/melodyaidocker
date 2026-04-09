# ─── Base image ───────────────────────────────────────────────────────────────
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# ─── System dependencies ───────────────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    curl \
    git \
    build-essential \
    ffmpeg \
    cmake \
    libboost-filesystem-dev \
    libboost-program-options-dev \
    libboost-regex-dev \
    libmad0-dev \
    libid3tag0-dev \
    libsndfile1-dev \
    libgd-dev \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install audiowaveform from source
RUN git clone https://github.com/bbc/audiowaveform.git /tmp/audiowaveform \
    && cd /tmp/audiowaveform \
    && cmake -D ENABLE_TESTS=0 . \
    && make -j$(nproc) \
    && make install \
    && rm -rf /tmp/audiowaveform

# Make python3.12 the default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# ─── Install uv ───────────────────────────────────────────────────────────────
RUN curl -Ls https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# ─── Working directory ────────────────────────────────────────────────────────
WORKDIR /app

# ─── Install Python dependencies ──────────────────────────────────────────────
COPY requirements-base.txt .

RUN uv venv .venv --python 3.12 \
    && . .venv/bin/activate \
    && uv pip install -r requirements-base.txt \
    && uv pip install onnxruntime

ENV PATH="/app/.venv/bin:$PATH"
ENV VIRTUAL_ENV="/app/.venv"

# ─── Copy application source ──────────────────────────────────────────────────
COPY melody/ ./melody/
COPY tests.py .

# ─── Runtime environment variables ───────────────────────────────────────────
ENV TEMP_DIR=/tmp/melody
ENV MELODY_DEVICE=auto
ENV MELODY_SEPARATION_BACKEND=auto

# ─── Expose port ──────────────────────────────────────────────────────────────
EXPOSE 8001

# ─── Health check ─────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# ─── Entrypoint ───────────────────────────────────────────────────────────────
CMD ["uvicorn", "melody.server:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]
