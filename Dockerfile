# ============================================================
# FastLiaison AI Models Gateway — CPU-only
# ============================================================

# ─────────────────────────────────────────────────────────────
# Stage 1: builder — installs all Python packages
# ─────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

ARG DEBIAN_FRONTEND=noninteractive

# Install build tools and system dependencies for compiling packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    git \
    curl \
    python3-dev \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /install

# Upgrade pip, setuptools, and wheel
RUN python3 -m pip install --upgrade pip setuptools wheel

# ── PyTorch CPU-only
RUN pip install --no-cache-dir \
    torch \
    torchvision \
    torchaudio --index-url https://download.pytorch.org/whl/cpu

# ── Core dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    pydantic \
    python-multipart \
    python-dotenv

# ── Video / Audio / ML packages
RUN pip install --no-cache-dir \
    opencv-python-headless \
    mediapipe==0.10.13 \
    openai-whisper \
    librosa \
    moviepy \
    soundfile \
    numba \
    pillow \
    numpy \
    scipy

# ── Transformers / NLP
RUN pip install --no-cache-dir \
    transformers \
    sentencepiece \
    tokenizers

# ── ML / XAI
RUN pip install --no-cache-dir \
    scikit-learn \
    joblib \
    pandas \
    lightgbm \
    matplotlib \
    seaborn

# ── Chatbot / AI mentor
RUN pip install --no-cache-dir \
    langchain \
    langchain-openai \
    openai \
    google-generativeai \
    pdfplumber

# ── Misc
RUN pip install --no-cache-dir \
    httpx \
    plotly

# ─────────────────────────────────────────────────────────────
# Stage 2: runtime — lean final image
# ─────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

ARG DEBIAN_FRONTEND=noninteractive

# Runtime OS dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg \
    libgomp1 \
    curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy project source
WORKDIR /app
COPY . .

# ─────────────────────────────────────────────────────────────
# Environment variables
# ─────────────────────────────────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=""
ENV HF_HOME=/app/model_cache/huggingface
ENV TRANSFORMERS_CACHE=/app/model_cache/huggingface
ENV HF_DATASETS_CACHE=/app/model_cache/huggingface/datasets
ENV XDG_CACHE_HOME=/app/model_cache
ENV NUMBA_CACHE_DIR=/tmp/numba_cache
ENV TOKENIZERS_PARALLELISM=false
ENV GLOG_minloglevel=2

# ─────────────────────────────────────────────────────────────
# Expose, healthcheck, entrypoint
# ─────────────────────────────────────────────────────────────
EXPOSE 8001

HEALTHCHECK \
    --interval=60s \
    --timeout=20s \
    --start-period=180s \
    --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

WORKDIR /app/gateway

CMD ["uvicorn", "main:app", \
     "--host", "0.0.0.0", \
     "--port", "8001", \
     "--workers", "1", \
     "--timeout-keep-alive", "300", \
     "--log-level", "info"]
