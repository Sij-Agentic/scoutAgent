# Multi-stage Dockerfile for ScoutAgent
# Stage 1: Build stage for compiling dependencies
FROM python:3.11-slim AS builder

# Install system dependencies needed for compilation
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    libjpeg-dev \
    libpng-dev \
    libfreetype6-dev \
    liblcms2-dev \
    libwebp-dev \
    libtiff5-dev \
    libopenjp2-7-dev \
    libharfbuzz-dev \
    libfribidi-dev \
    libxcb1-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install Python dependencies
COPY requirements-docker.txt /tmp/requirements-docker.txt
COPY scout_agent/requirements_llm.txt /tmp/requirements_llm.txt

# Install requirements with optimizations
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /tmp/requirements-docker.txt && \
    pip install --no-cache-dir -r /tmp/requirements_llm.txt

# Stage 2: Runtime stage
FROM python:3.11-slim

# Install essential runtime dependencies
RUN apt-get update && apt-get install -y \
    libxml2 \
    libxslt1.1 \
    libjpeg62-turbo \
    libpng16-16 \
    libfreetype6 \
    liblcms2-2 \
    libwebp7 \
    libtiff6 \
    libopenjp2-7 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy the ScoutAgent application
COPY scout_agent/ ./scout_agent/

# Copy startup script
COPY start_services.sh ./start_services.sh

# Copy API services
COPY api_service.py ./api_service.py
COPY worker_service.py ./worker_service.py

# Create directories for output and logs
RUN mkdir -p /app/output /app/logs /app/temp

# Set environment variables (runtime secrets will be injected by Fargate)
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV SCOUT_LLM_DEFAULT_BACKEND=deepseek
ENV SCOUT_LLM_DEFAULT_MODEL=deepseek-chat

# NOTE: .env is not copied into the image; Fargate task env/Secrets Manager should provide runtime vars

# Create non-root user for security
RUN groupadd -r scoutagent && useradd -r -g scoutagent scoutagent
RUN chown -R scoutagent:scoutagent /app
USER scoutagent

# Expose port (for future API service)
EXPOSE 8000

# Default command (can be overridden)
CMD ["./start_services.sh", "--help"]
