# ============================================================
# FastLiaison AI Models Gateway
# Target:  DigitalOcean Droplet — Ubuntu 22.04
# Runtime: CPU-only, single-stage build
#
# WHY SINGLE-STAGE:
#   mediapipe bundles its own protobuf/opencv native .so files
#   inside the wheel. Multi-stage COPY misses them and causes
#   the "Failed to parse graph" error seen at runtime.
#   Single-stage avoids this entirely.
#
# Recommended Droplet: 16GB RAM / 4 vCPU ($96/mo)
# ============================================================

FROM python:3.11-slim

ARG DEBIAN_FRONTEND=noninteractive

LABEL maintainer="FastLiaison"
LABEL description="FastLiaison AI Gateway — all models, CPU-only"
LABEL version="1.1.0"

# ── System dependencies ───────────────────────────────────────
# Everything mediapipe, OpenCV, librosa, moviepy need at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Build tools (needed for some pip wheels)
    build-essential \
    gcc \
    g++ \
    # OpenCV runtime libs
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    # ffmpeg: moviepy audio extraction from MP4
    ffmpeg \
    # OpenMP: scikit-learn / LightGBM parallel jobs
    libgomp1 \
    # git: some pip packages require it at install time
    git \
    # curl: healthcheck
    curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# ── Step 1: Pin protobuf BEFORE anything else ─────────────────
# mediapipe 0.10.x requires protobuf >= 4.25, < 5.0
# Installing this first prevents pip from pulling protobuf 5.x
# which breaks mediapipe's internal graph parsing completely
RUN pip install --no-cache-dir "protobuf>=4.25.3,<5.0.0"

# ── Step 2: Pin numpy BEFORE opencv or mediapipe ──────────────
# mediapipe 0.10.9 is not compatible with numpy 2.x
RUN pip install --no-cache-dir "numpy>=1.23.0,<2.0.0"

# ── Step 3: PyTorch CPU-only ──────────────────────────────────
# Must specify the CPU index URL or pip pulls CUDA wheels (~2GB wasted)
RUN pip install --no-cache-dir \
    torch==2.2.2 \
    torchvision==0.17.2 \
    torchaudio==2.2.2 \
    --index-url https://download.pytorch.org/whl/cpu

# ── Step 4: OpenCV — pin before mediapipe ─────────────────────
# mediapipe 0.10.9 is tested against opencv-python 4.8.x
# Do NOT use opencv-python (has GUI deps) — use headless
RUN pip install --no-cache-dir "opencv-python-headless==4.8.1.78"

# ── Step 5: MediaPipe ─────────────────────────────────────────
# 0.10.9 is the last version before the internal graph proto
# format changed. 0.10.13+ causes "Failed to parse graph" on
# CPU-only Linux Docker containers.
RUN pip install --no-cache-dir "mediapipe==0.10.9"

# ── Step 6: Validate mediapipe before continuing ──────────────
# GLOG env vars set HERE (before the check) to suppress the
# proto graph dump that floods the output. These are NOT errors —
# mediapipe just prints its internal pipeline to stderr by default.
# Setting them here silences it during both build validation
# and at runtime.
ENV GLOG_minloglevel=3
ENV GLOG_logtostderr=0
ENV MEDIAPIPE_DISABLE_GPU=1

RUN python3 -c "\
import os; \
os.environ['GLOG_minloglevel'] = '3'; \
os.environ['GLOG_logtostderr'] = '0'; \
import cv2, mediapipe as mp; \
print('OpenCV:', cv2.__version__); \
print('MediaPipe:', mp.__version__); \
fm = mp.solutions.face_mesh.FaceMesh( \
    max_num_faces=1, \
    refine_landmarks=False, \
    min_detection_confidence=0.5, \
    min_tracking_confidence=0.5 \
); \
fm.close(); \
print('FaceMesh: OK') \
"

# ── Step 7: Audio / video processing ─────────────────────────
RUN pip install --no-cache-dir \
    "openai-whisper==20231117" \
    "librosa==0.10.1" \
    "moviepy==1.0.3" \
    "imageio-ffmpeg" \
    "soundfile==0.12.1" \
    "numba==0.59.1" \
    "pillow==10.3.0" \
    "scipy==1.13.0"

# ── Step 8: Transformers / NLP ────────────────────────────────
# BERT email classifier + MMIA NLP (RoBERTa, BART)
RUN pip install --no-cache-dir \
    "transformers==4.41.2" \
    "sentencepiece==0.2.0" \
    "tokenizers==0.19.1"

# ── Step 9: FastAPI gateway + web framework ───────────────────
RUN pip install --no-cache-dir \
    "fastapi==0.111.0" \
    "uvicorn[standard]==0.29.0" \
    "pydantic==2.7.1" \
    "python-multipart==0.0.9" \
    "python-dotenv==1.0.1" \
    "httpx==0.27.0"

# ── Step 10: ML / XAI / Career path ──────────────────────────
RUN pip install --no-cache-dir \
    "scikit-learn==1.5.0" \
    "joblib==1.4.2" \
    "pandas==2.2.2" \
    "lightgbm==4.3.0" \
    "matplotlib==3.9.0" \
    "seaborn==0.13.2" \
    "plotly==5.22.0"

# ── Step 11: AI mentor chatbot ────────────────────────────────
RUN pip install --no-cache-dir \
    "langchain==0.2.5" \
    "langchain-openai==0.1.8" \
    "openai==1.30.5" \
    "google-generativeai==0.5.4" \
    "pdfplumber==0.11.0"

# ── Copy project source ───────────────────────────────────────
COPY . .

# ── Environment variables ─────────────────────────────────────

# Python
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Force CPU — prevents torch from scanning for CUDA on startup
ENV CUDA_VISIBLE_DEVICES=""

# HuggingFace — persisted via Docker volume
ENV HF_HOME=/app/model_cache/huggingface
ENV TRANSFORMERS_CACHE=/app/model_cache/huggingface
ENV HF_DATASETS_CACHE=/app/model_cache/huggingface/datasets

# Whisper cache
ENV XDG_CACHE_HOME=/app/model_cache

# Numba (librosa dependency) — needs writable cache dir
ENV NUMBA_CACHE_DIR=/tmp/numba_cache

# Suppress HuggingFace tokenizer warning
ENV TOKENIZERS_PARALLELISM=false
# GLOG_minloglevel=3, GLOG_logtostderr=0, MEDIAPIPE_DISABLE_GPU=1
# already set above before the mediapipe validation step

# ── Port ──────────────────────────────────────────────────────
EXPOSE 8001

# ── Healthcheck ───────────────────────────────────────────────
# start-period=180s: container starts fast but first MMIA/BERT
# request triggers lazy model loading (60-120s on CPU)
HEALTHCHECK \
    --interval=60s \
    --timeout=20s \
    --start-period=180s \
    --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# ── Entrypoint ────────────────────────────────────────────────
# Must run from /app/gateway so relative imports resolve correctly
WORKDIR /app/gateway

# Single worker — PyTorch/MediaPipe models are NOT fork-safe
# timeout-keep-alive=300 handles long video analysis requests
CMD ["uvicorn", "main:app", \
     "--host", "0.0.0.0", \
     "--port", "8001", \
     "--workers", "1", \
     "--timeout-keep-alive", "300", \
     "--log-level", "info"]