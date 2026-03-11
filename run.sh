#!/bin/bash
# Run the Apex Oracle server using the project's virtual environment.
# This ensures cv2, mediapipe, etc. are found (they're installed in .venv).
# Exclude .venv from file watching — uvicorn requires an absolute path to match
# the watcher's paths and prevent constant reloads when pip/IDE touches venv files.
cd "$(dirname "$0")"
PROJECT_ROOT="$(pwd)"
.venv/bin/python -m uvicorn main:app --reload \
  --reload-exclude "$PROJECT_ROOT/.venv"
