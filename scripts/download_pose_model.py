"""Download MediaPipe pose model during Docker build."""
from pathlib import Path
import urllib.request
import ssl

p = Path("/app/pose_landmarker_heavy.task")
if not p.exists():
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
        str(p),
    )
    print("Pose model downloaded")
else:
    print("Pose model exists")
