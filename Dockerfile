# ============================================================
# FastLiaison AI Models Gateway — CPU-only
# ============================================================

# ─────────────────────────────────────────────────────────────
# Stage 1: builder — installs all Python packages
# ─────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

ARG DEBIAN_FRONTEND=noninteractive

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
    libgomp1 \
    protobuf-compiler \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /install

RUN python3 -m pip install --upgrade pip setuptools wheel

# ── PyTorch CPU-only
RUN pip install --no-cache-dir \
    torch \
    torchvision \
    torchaudio --index-url https://download.pytorch.org/whl/cpu

# ── Core dependencies
RUN pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    pydantic \
    python-multipart \
    python-dotenv

# ── OpenCV MUST be pinned before mediapipe is installed
#    mediapipe 0.10.13 is tested against opencv 4.8.x
# ── Pin protobuf FIRST — mediapipe 0.10.x requires protobuf<5
RUN pip install --no-cache-dir \
    "protobuf>=4.25.3,<5.0.0"

# ── Pin numpy before opencv pulls in numpy 2.x
RUN pip install --no-cache-dir \
    "numpy>=1.21.2,<2.0.0"

# ── OpenCV pinned — must be before mediapipe
RUN pip install --no-cache-dir \
    "opencv-python-headless==4.8.1.78"

# ── mediapipe after all its dependencies are locked
RUN pip install --no-cache-dir \
    "mediapipe==0.10.13"

# ── Video / Audio / ML packages
#    Install mediapipe AFTER opencv so it uses the pinned version
RUN pip install --no-cache-dir \
    "openai-whisper" \
    "librosa" \
    "moviepy==1.0.3" \
    "imageio-ffmpeg" \
    "soundfile" \
    "numba" \
    "pillow" \
    "scipy"

# ── Transformers / NLP
RUN pip install --no-cache-dir \
    "transformers>=4.35.0" \
    "sentencepiece" \
    "tokenizers"

# ── ML / XAI
RUN pip install --no-cache-dir \
    "scikit-learn" \
    "joblib" \
    "pandas" \
    "lightgbm" \
    "matplotlib" \
    "seaborn"

# ── Chatbot / AI mentor
RUN pip install --no-cache-dir \
    "langchain" \
    "langchain-openai" \
    "openai" \
    "google-generativeai" \
    "pdfplumber"

# ── Misc
RUN pip install --no-cache-dir \
    "httpx" \
    "plotly"

# ─────────────────────────────────────────────────────────────
# Stage 2: runtime — lean final image
# ─────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

ARG DEBIAN_FRONTEND=noninteractive

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
# Suppress mediapipe graph logging noise
ENV GLOG_logtostderr=0
ENV MEDIAPIPE_DISABLE_GPU=1

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