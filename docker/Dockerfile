# syntax=docker/dockerfile:1
# Multi-stage build. One image serves both compose services (api runs uvicorn via the
# entrypoint; mlflow runs `mlflow server`).

# ---------------------------------------------------------------------------
# Builder: resolve + install dependencies into /app/.venv using uv.
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

RUN pip install --no-cache-dir uv

# Use the base image's system Python for the venv so the interpreter the venv points at
# also exists in the (identical) runtime base — avoids a broken venv across stages.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON=python3.11 \
    UV_PYTHON_PREFERENCE=only-system

WORKDIR /app

# Install third-party deps first (cached unless pyproject/lock change).
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

# Then install the project itself. README.md is required by the build backend
# (pyproject `readme = "README.md"`), so it must be present before the project install.
COPY README.md ./
COPY src ./src
COPY configs ./configs
COPY scripts ./scripts
RUN uv sync --frozen --no-dev

# ---------------------------------------------------------------------------
# Runtime: slim image with just the venv + app, running as a non-root user.
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /mlflow \
    && chown appuser:appuser /mlflow   # fresh named volume inherits this ownership

WORKDIR /app
COPY --from=builder --chown=appuser:appuser /app /app
COPY --chown=appuser:appuser docker/entrypoint.sh /app/docker/entrypoint.sh
RUN chmod +x /app/docker/entrypoint.sh

USER appuser
EXPOSE 8000

# Liveness via the API's own /health (no curl needed in the image).
HEALTHCHECK --interval=15s --timeout=5s --start-period=180s --retries=10 \
    CMD python -c "import sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)" || exit 1

# Default command = serve the API. The mlflow service overrides this in compose.
CMD ["uvicorn", "evdecafs_serve.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
