# ── Backend: FastAPI + TensorFlow inference ──
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# ── Layer caching optimisation ──
# Copy only requirements first so Docker re-uses this layer
# on every rebuild that does NOT change requirements.txt.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy application source ──
# .dockerignore excludes: venv/, data/, .git/, mlruns/, etc.
COPY app.py .
COPY model/ ./model/

# ── Non-root user (production security best practice) ──
RUN adduser --disabled-password --gecos "" appuser
USER appuser

# ── Runtime ──
EXPOSE 8000

# Use 2 Uvicorn workers for lightweight production serving.
# For high-traffic, scale replicas in ECS instead of workers here.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]