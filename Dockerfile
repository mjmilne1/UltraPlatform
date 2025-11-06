# ============================================================================
# ANYA AI ASSISTANT - PRODUCTION DOCKERFILE
# ============================================================================

FROM python:3.11-slim as base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY modules/anya/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY modules/ modules/

# Create non-root user
RUN useradd -m -u 1000 anya && chown -R anya:anya /app
USER anya

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Run application
CMD ["python", "-m", "uvicorn", "modules.anya.integration.anya_orchestrator:app", "--host", "0.0.0.0", "--port", "8000"]
