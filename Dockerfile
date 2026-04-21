# FastAPI application with MediaPipe/OpenCV support
FROM python:3.11-slim

WORKDIR /app

# System dependencies for MediaPipe/OpenCV + FFmpeg (codec normalisation, VFR fix, rotation).
#
# Why each package:
#   libgl1            -> OpenGL desktop runtime; needed by cv2.imshow paths and some
#                        OpenCV codecs even in headless mode.
#   libglib2.0-0      -> GLib runtime; OpenCV depends on it transitively.
#   libegl1, libgles2 -> EGL + OpenGL ES 2.0 runtimes. MediaPipe Tasks API
#                        (mediapipe.tasks.vision.PoseLandmarker) dlopen()s
#                        libEGL.so.1 + libGLESv2.so.2 during landmarker creation
#                        even on CPU-only inference. Without these the /v1/analyze/
#                        gym/video endpoint fails with:
#                        "libGLESv2.so.2: cannot open shared object file".
#   libgomp1          -> OpenMP runtime; MediaPipe + numpy use it for parallelism.
#   ffmpeg, libavcodec-extra -> H.264/VP9/WebM decode + the FFmpeg preprocess
#                               step (rotation/VFR fix) before pose extraction.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libegl1 libgles2 libgomp1 \
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
RUN python -c "import os, chromadb; from app.db_seeder import seed_database; os.makedirs('/app/chroma_db', exist_ok=True); seed_database(chromadb.PersistentClient(path='/app/chroma_db'))"

EXPOSE 8000

# Single gunicorn worker with uvicorn kernel.
# -w 1: one process on the 1 GB Fly shared-cpu-1x machine (MediaPipe is not
#        safely reentrant across workers; keep at 1).
# --timeout 300: gym/basketball video analyze can exceed 120s (MediaPipe + FFmpeg +
#                occasional codec edge cases). Hitting the default 120s kills the worker
#                mid-request (see Fly logs: WORKER TIMEOUT).
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", \
    "-w", "1", \
    "-b", "0.0.0.0:8000", \
    "--timeout", "300", \
    "--access-logfile", "-", \
    "--error-logfile", "-", \
    "app.main:app"]
