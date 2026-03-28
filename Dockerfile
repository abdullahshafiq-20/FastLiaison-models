# ============================================================
# FastLiaison AI Models Gateway
# Target:  DigitalOcean Droplet — Ubuntu 22.04
# Runtime: CPU-only (no CUDA)
# Recommended Droplet: 16GB RAM / 4 vCPU / 160GB SSD ($96/mo)
#                      Minimum:  8GB RAM / 2 vCPU  ($48/mo)
# ============================================================

# ─────────────────────────────────────────────────────────────
# Stage 1: builder — installs all Python packages
# Kept separate so the final image has no compilers/headers
# ─────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    git \
    curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /install

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# ── PyTorch CPU-only FIRST ────────────────────────────────────
# Must come before anything else that might pull CUDA wheels (~2GB saved)
RUN pip install --no-cache-dir \
    torch==2.2.2 \
    torchvision==0.17.2 \
    torchaudio==2.2.2 \
    --index-url https://download.pytorch.org/whl/cpu

# ── Gateway core ──────────────────────────────────────────────
RUN pip install --no-cache-dir \
    "fastapi==0.111.0" \
    "uvicorn[standard]==0.29.0" \
    "pydantic==2.7.1" \
    "python-multipart==0.0.9" \
    "python-dotenv==1.0.1"

# ── MMIA: video emotion + audio transcription ─────────────────
RUN pip install --no-cache-dir \
    "opencv-python-headless==4.9.0.80" \
    "mediapipe==0.10.14" \
    "openai-whisper==20231117" \
    "librosa==0.10.1" \
    "moviepy==1.0.3" \
    "soundfile==0.12.1" \
    "numba==0.59.1" \
    "pillow==10.3.0" \
    "numpy==1.26.4" \
    "scipy==1.13.0"

# ── Transformers: BERT email classifier + MMIA NLP ────────────
RUN pip install --no-cache-dir \
    "transformers==4.41.2" \
    "sentencepiece==0.2.0" \
    "tokenizers==0.19.1"

# ── XAI recommendations + predictive career path ──────────────
RUN pip install --no-cache-dir \
    "scikit-learn==1.5.0" \
    "joblib==1.4.2" \
    "pandas==2.2.2" \
    "lightgbm==4.3.0" \
    "matplotlib==3.9.0" \
    "seaborn==0.13.2"

# ── AI mentor chatbot ─────────────────────────────────────────
RUN pip install --no-cache-dir \
    "langchain==0.2.5" \
    "langchain-openai==0.1.8" \
    "openai==1.30.5" \
    "google-generativeai==0.5.4" \
    "pdfplumber==0.11.0"

# ── Misc shared ───────────────────────────────────────────────
RUN pip install --no-cache-dir \
    "httpx==0.27.0" \
    "plotly==5.22.0"

# ─────────────────────────────────────────────────────────────
# Stage 2: runtime — lean final image, no compilers
# ─────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

ARG DEBIAN_FRONTEND=noninteractive

LABEL maintainer="FastLiaison"
LABEL description="FastLiaison AI Gateway — all models, CPU-only"
LABEL version="1.0.0"

# ── Runtime OS dependencies only ──────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    # OpenCV headless runtime libs
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    # ffmpeg: moviepy needs this for audio extraction from MP4
    ffmpeg \
    # OpenMP: parallel jobs in scikit-learn / LightGBM
    libgomp1 \
    # healthcheck inside container
    curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ── Pull all installed packages from builder stage ────────────
COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=builder /usr/local/bin /usr/local/bin

# ── Copy project source ───────────────────────────────────────
WORKDIR /app
COPY . .

# ─────────────────────────────────────────────────────────────
# Environment variables
# ─────────────────────────────────────────────────────────────

# Python
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Force CPU — prevents torch from scanning for CUDA on startup
ENV CUDA_VISIBLE_DEVICES=""

# HuggingFace: store downloaded models in a named Docker volume
# (prevents re-downloading RoBERTa + BART on every container restart)
ENV HF_HOME=/app/model_cache/huggingface
ENV TRANSFORMERS_CACHE=/app/model_cache/huggingface
ENV HF_DATASETS_CACHE=/app/model_cache/huggingface/datasets

# Whisper: same volume, different subdirectory
ENV XDG_CACHE_HOME=/app/model_cache

# Numba (used by librosa) needs a writable cache dir
ENV NUMBA_CACHE_DIR=/tmp/numba_cache

# Suppress noisy startup warnings
ENV TOKENIZERS_PARALLELISM=false
ENV GLOG_minloglevel=2

# ─────────────────────────────────────────────────────────────
# Expose, healthcheck, entrypoint
# ─────────────────────────────────────────────────────────────
EXPOSE 8001

# start-period=180s: gateway starts fast (~2s), but first request
# to MMIA/BERT will trigger lazy model loading (~60-120s on CPU)
HEALTHCHECK \
    --interval=60s \
    --timeout=20s \
    --start-period=180s \
    --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

WORKDIR /app/gateway

# Single worker is required — PyTorch models are not fork-safe.
# timeout-keep-alive=300 handles long video analysis requests.
CMD ["uvicorn", "main:app", \
     "--host", "0.0.0.0", \
     "--port", "8001", \
     "--workers", "1", \
     "--timeout-keep-alive", "300", \
     "--log-level", "info"]