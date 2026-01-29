# ============================================
# MedQuery Dockerfile
# ============================================
# Multi-stage build for production deployment
# ============================================

# ==========================================
# Stage 1: Builder
# ==========================================
FROM python:3.11-slim AS builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install dependencies
WORKDIR /build
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# ==========================================
# Stage 2: Production
# ==========================================
FROM python:3.11-slim AS production

# Labels
LABEL maintainer="MedQuery Team" \
    description="Production RAG System for Clinical Documentation" \
    version="0.1.0"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PATH="/opt/venv/bin:$PATH" \
    # Application defaults
    APP_ENV=production \
    API_HOST=0.0.0.0 \
    API_PORT=8000 \
    # Model cache directory
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface/transformers \
    SENTENCE_TRANSFORMERS_HOME=/app/.cache/sentence_transformers

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libpq5 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd --gid 1000 medquery && \
    useradd --uid 1000 --gid medquery --shell /bin/bash --create-home medquery

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Set working directory
WORKDIR /app

# Create necessary directories
RUN mkdir -p /app/src /app/uploads /app/logs /app/.cache && \
    chown -R medquery:medquery /app

# Copy application code
COPY --chown=medquery:medquery src/ /app/src/
COPY --chown=medquery:medquery scripts/ /app/scripts/
COPY --chown=medquery:medquery datasets/ /app/datasets/

# Switch to non-root user
USER medquery

# Pre-download the embedding model (optional, can be done at runtime)
# RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

# ==========================================
# Stage 3: Development (optional)
# ==========================================
FROM production AS development

USER root

# Install development dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Copy development requirements
COPY --chown=medquery:medquery pyproject.toml /app/
RUN pip install -e ".[dev,test]"

USER medquery

# Development command with auto-reload
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
