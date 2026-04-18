# FastAPI application with MediaPipe/OpenCV support
FROM python:3.11-slim

WORKDIR /app

# System dependencies for MediaPipe/OpenCV + FFmpeg (codec normalisation, VFR fix, rotation)
RUN apt-get update && apt-get install -y \
    libgl1 libglib2.0-0 \
    ffmpeg libavcodec-extra \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies (locked for reproducibility; matches CI)
COPY requirements.txt requirements.lock ./
RUN pip install --no-cache-dir -r requirements.lock

# Application code
COPY . .

# Pre-download MediaPipe pose model so analysis overlay works on first request (no runtime download)
RUN python scripts/download_pose_model.py

# Pre-seed ChromaDB during build — no NBA API calls at runtime. Build has network;
# if NBA API succeeds we get full roster; if it fails we use fallback. Either way
# chroma_db is baked into the image so startup is instant and error-free.
RUN python -c "\
import os; os.makedirs('/app/chroma_db', exist_ok=True); \
from app.db_seeder import seed_database; \
import chromadb; \
seed_database(chromadb.PersistentClient(path='/app/chroma_db')) \
"

EXPOSE 8000

# Single gunicorn worker with uvicorn kernel.
# -w 1: one process on the 1 GB Fly shared-cpu-1x machine (MediaPipe is not
#        safely reentrant across workers; keep at 1).
# --timeout 120: long enough for a 5 s clip round-trip including MediaPipe
#                VIDEO decode on cold frames.
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", \
     "-w", "1", \
     "-b", "0.0.0.0:8000", \
     "--timeout", "120", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "app.main:app"]
