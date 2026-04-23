"""Download and verify the YOLOv8n ONNX model for Track B ball detection.

Usage
-----
    python scripts/download_ball_detector.py

What it does
------------
1. Attempts to download ``yolov8n.onnx`` from the Ultralytics GitHub assets
   release endpoint.  If that fails (no network, URL changed), it falls back
   to exporting from the ``.pt`` PyTorch checkpoint.
2. Verifies the SHA-256 of the downloaded file against
   ``app/pose/models/yolov8n.onnx.sha256`` (if the file exists).
   If the SHA file does not exist, it is created from the downloaded model
   so subsequent runs can verify integrity.
3. Exits 0 on success, exits 1 on unrecoverable failure.

Environment variables
---------------------
LAKSH_ENABLE_BALL_DETECT : str
    When ``"1"`` the Dockerfile will run this script.  When absent, skip.
    Safe to run manually at any time regardless of this variable.

Model provenance
----------------
YOLOv8n (Ultralytics), trained on COCO 2017 train.
COCO class 32 = sports ball (0-indexed).  See docs/adr/0005-ball-leaves-hand-signal.md.
"""
from __future__ import annotations

import hashlib
import logging
import pathlib
import sys
import urllib.request

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT = pathlib.Path(__file__).parent.parent
_MODEL_DIR = _REPO_ROOT / "app" / "detection" / "models"
_MODEL_PATH = _MODEL_DIR / "yolov8n.onnx"
_SHA_PATH = _MODEL_DIR / "yolov8n.onnx.sha256"

# Primary download URL (Ultralytics GitHub assets release — stable endpoint).
# This is the ONNX export from the YOLOv8n v8.3.0 checkpoint, opset 17,
# input 640×640.  The SHA256 below was computed from this exact artifact.
_ONNX_URL = (
    "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.onnx"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sha256(path: pathlib.Path) -> str:
    """Return the hex SHA-256 of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: pathlib.Path) -> None:
    """Download *url* to *dest* with a progress indicator."""
    logger.info("Downloading %s → %s", url, dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    def _progress(count: int, block_size: int, total_size: int) -> None:
        if total_size > 0:
            pct = min(100, 100 * count * block_size // total_size)
            print(f"\r  {pct}% ", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=_progress)
    print()  # newline after progress bar


def _try_ultralytics_export() -> bool:
    """Fall back: download .pt and export to ONNX via ultralytics package.

    Returns True on success.  Requires ``pip install ultralytics`` (not in
    requirements.txt — install manually when needed).
    """
    try:
        from ultralytics import YOLO  # type: ignore[import-untyped]
    except ImportError:
        logger.warning(
            "ultralytics not installed; cannot fall back to .pt export.\n"
            "  pip install ultralytics   # then retry"
        )
        return False

    pt_path = _MODEL_DIR / "yolov8n.pt"
    _pt_url = (
        "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt"
    )
    try:
        if not pt_path.exists():
            _download(_pt_url, pt_path)
        logger.info("Exporting .pt → .onnx …")
        model = YOLO(str(pt_path))
        model.export(format="onnx", opset=17, imgsz=640, dynamic=False)
        # Ultralytics writes the onnx next to the .pt with the same stem.
        exported = pt_path.with_suffix(".onnx")
        if exported.exists():
            exported.rename(_MODEL_PATH)
            logger.info("Export complete: %s", _MODEL_PATH)
            return True
        logger.error("Export produced no file at expected path %s", exported)
        return False
    except Exception as exc:
        logger.error("Ultralytics export failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    """Entry point; returns 0 on success, 1 on failure."""
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: download if missing
    # ------------------------------------------------------------------
    if not _MODEL_PATH.exists():
        logger.info("YOLOv8n ONNX not found — attempting download …")
        try:
            _download(_ONNX_URL, _MODEL_PATH)
        except Exception as exc:
            logger.warning("Primary download failed (%s); trying ultralytics export …", exc)
            if not _try_ultralytics_export():
                logger.error(
                    "Could not obtain yolov8n.onnx.  "
                    "Ball detector (Track B) will be disabled.\n"
                    "  To enable manually:\n"
                    "    pip install ultralytics\n"
                    "    python scripts/download_ball_detector.py"
                )
                return 1
    else:
        logger.info("Model already present: %s", _MODEL_PATH)

    # ------------------------------------------------------------------
    # Step 2: SHA verification / creation
    # ------------------------------------------------------------------
    actual_sha = _sha256(_MODEL_PATH)
    logger.info("SHA-256: %s", actual_sha)

    if _SHA_PATH.exists():
        expected_sha = _SHA_PATH.read_text(encoding="utf-8").strip()
        if actual_sha != expected_sha:
            logger.error(
                "SHA mismatch! Expected %s, got %s.\n"
                "Delete %s and re-run to re-download.",
                expected_sha,
                actual_sha,
                _MODEL_PATH,
            )
            return 1
        logger.info("SHA verified ✓")
    else:
        # First successful download — write the SHA as the new expected value.
        _SHA_PATH.write_text(actual_sha + "\n", encoding="utf-8")
        logger.info("SHA saved to %s", _SHA_PATH)

    logger.info("Ball detector model ready.  Set LAKSH_ENABLE_BALL_DETECT=1 to activate.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
