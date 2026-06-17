FROM python:3.11-slim

WORKDIR /app

# Prefer CPU-only torch to avoid pulling CUDA libraries
# So sentence-transformers doesn't pull the
# default CUDA-enabled build (which adds ~900MB of unused GPU libraries
# on a CPU-only deployment).
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the embedding model at build time (cached layer)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy application code
COPY app/ app/
COPY data/ data/
COPY config/ config/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]