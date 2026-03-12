# FastAPI application with MediaPipe/OpenCV support
FROM python:3.11-slim

WORKDIR /app

# System dependencies for MediaPipe/OpenCV
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY . .

# Pre-download MediaPipe pose model so analysis overlay works on first request (no runtime download)
RUN python scripts/download_pose_model.py

# Pre-seed ChromaDB during build — no NBA API calls at runtime. Build has network;
# if NBA API succeeds we get full roster; if it fails we use fallback. Either way
# chroma_db is baked into the image so startup is instant and error-free.
RUN python -c "\
import os; os.makedirs('/app/chroma_db', exist_ok=True); \
from db_seeder import seed_database; \
import chromadb; \
seed_database(chromadb.PersistentClient(path='/app/chroma_db')) \
"

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
