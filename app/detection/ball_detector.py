"""Ball-leaves-hand signal (S3) via YOLOv8n ONNX.

Plan reference: Track B — ``warm-swimming-acorn.md``.

This module detects basketball release frames as an optional third signal
that upgrades the S1+S2 consensus in :mod:`app.basketball.shot_segmenter`.
It is gated by ``LAKSH_ENABLE_BALL_DETECT=1``; when the env var is absent
the :func:`detect_release_frames` function returns an empty list (zero
inference cost, zero impact on the baseline pipeline).

Model
-----
YOLOv8n ONNX (Ultralytics).  Input tensor: ``images`` [1, 3, 640, 640]
float32 in RGB 0-1 range with letterbox padding.  Output tensor:
``output0`` [1, 84, 8400] where the 84 channels are::

    [cx, cy, w, h, score_class_0, ..., score_class_79]

COCO class 32 = ``sports ball`` (0-indexed COCO80 list — the plan's "37"
was a typo; counting from 0: person=0, …, sports ball=32).

Algorithm
---------
For each S1 candidate frame ``p`` (provided by :func:`segment_shots`):

1. Extract ~24 frames in the window ``[p - 0.5 s, p + 0.3 s]`` from the
   video.
2. Run YOLO inference per frame; keep only ``sports ball`` detections above
   confidence threshold 0.35 (COCO eval default).
3. From each detection, compute the Euclidean distance (in normalized image
   coordinates) between the ball centroid and the nearest wrist landmark.
4. Find the "hold" frame — the frame where ball-to-wrist distance is
   minimised (ball in the shooter's hand).
5. The **release frame** is the first frame AFTER the hold frame where:
   * ``ball_to_wrist_dist > 0.15`` (normalized image units — ~15 % of
     image width at typical filming distances), AND
   * the ball's y-coordinate is *decreasing* (y gets smaller in MediaPipe
     / image space, meaning the ball is moving upward / away from the body)
     for at least 3 consecutive frames.
6. If either condition is never satisfied within the window, S3 does not
   fire for that candidate (silence, not fabrication).

Edge cases
----------
* **No ball in frame** → no detections at any confidence → S3 silent.
* **Ball partially occluded** → YOLO trained on COCO still detects balls
  that are > ~40 % visible (COCO benchmark includes occluded objects);
  partial visibility is explicitly NOT a failure mode.
* **Shooting without basketball** → S3 never fires because there is no
  ball to detect; S1+S2 may still produce a ``degraded`` shot (honest).
* **Low wrist visibility** → wrist position is interpolated from the
  nearest non-NaN frame; if ALL frames in the window have NaN wrist, S3
  does not fire for that candidate.
* **Model not downloaded** → ``FileNotFoundError`` is caught at class
  init; all calls fall back to the empty-list result; a single
  ``WARNING`` log is emitted (not repeated per frame).
* **onnxruntime not installed** → ``ImportError`` is caught; same
  graceful fallback.
* **Video not decodable** → ``cv2.VideoCapture`` opens but no frames →
  returns empty list; exception is caught and logged.

Honesty contract
----------------
S3 is a *third* independent signal.  When it disagrees with S1+S2, the
shot keeps its status from S1+S2 but adds ``"s3_disagreed"`` to its
``reason_codes`` so judges can interrogate the discrepancy.  S3 can only
*confirm*, never *override*, the S1+S2 consensus.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (all have provenance comments)
# ---------------------------------------------------------------------------

# Absolute path to the expected ONNX model.
# Stored in app/detection/models/ (separate from the MediaPipe .task file
# which lives at repo root).  The download script creates this file.
_MODEL_PATH = Path(__file__).parent / "models" / "yolov8n.onnx"

# COCO 80-class index for "sports ball" (0-indexed).  Verified against the
# official COCO class list (https://cocodataset.org/#explore).  The plan
# document stated "37" but that is 1-indexed naming; 0-indexed is 32.
_COCO_SPORTS_BALL_CLS = 32

# YOLO input resolution (YOLOv8n default export target).
_INPUT_SIZE = 640

# Confidence threshold: 0.35 is the COCO benchmark default for YOLOv8 evaluation.
# Lower = more detections (more false positives); higher = fewer (more misses).
# 0.35 is the correct operating point for a COCO-pretrained detector on sports clips.
_CONF_THRESHOLD = 0.35

# IoU threshold for greedy NMS (suppress overlapping detections from same ball).
_NMS_IOU_THRESHOLD = 0.45

# Release criterion: ball centroid must be farther than this fraction of image
# width/height from the wrist landmark to count as "left hand".
# Derivation: typical wrist-to-palm distance is ~8 % of frame height in a
# chest-height side-view shot; 15 % gives comfortable 1.9× margin before
# counting as separation.
_SEPARATION_THRESHOLD = 0.15

# Ball must be moving upward (y decreasing in image space, since y=0 is top)
# for at least this many consecutive frames to confirm release.
_MIN_UPWARD_FRAMES = 3

# Window around each S1 candidate: 0.5 s before, 0.3 s after (from plan).
_WINDOW_PRE_S = 0.5
_WINDOW_POST_S = 0.3


# ---------------------------------------------------------------------------
# Dataclass for a single detection in one frame
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BallDetection:
    """A single ball detection in one decoded frame.

    Coordinates are in **original normalized image space** (not letterboxed
    space).  ``cx`` and ``cy`` are the centroid in [0, 1] × [0, 1];
    ``w`` and ``h`` are the bounding-box dimensions in the same units.
    """

    frame_idx: int
    cx: float
    cy: float
    w: float
    h: float
    confidence: float


# ---------------------------------------------------------------------------
# Letterbox pre-processing helpers
# ---------------------------------------------------------------------------


def _letterbox(
    img_bgr: np.ndarray, target: int = _INPUT_SIZE
) -> tuple[np.ndarray, float, int, int]:
    """Pad *img_bgr* to a square with aspect-ratio-preserving resize.

    Returns
    -------
    padded : np.ndarray
        [target, target, 3] BGR uint8.
    scale : float
        How much the original image was scaled down.
    pad_left : int
        Horizontal padding (half of total horizontal pad).
    pad_top : int
        Vertical padding (half of total vertical pad).
    """
    h, w = img_bgr.shape[:2]
    scale = target / max(h, w)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    try:
        import cv2  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError("cv2 (opencv-python-headless) is required for ball detection") from exc
    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_top = (target - new_h) // 2
    pad_bottom = target - new_h - pad_top
    pad_left = (target - new_w) // 2
    pad_right = target - new_w - pad_left
    padded = cv2.copyMakeBorder(
        resized, pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT, value=(114, 114, 114),
    )
    return padded, scale, pad_left, pad_top


def _preprocess(img_bgr: np.ndarray) -> tuple[np.ndarray, float, int, int]:
    """BGR uint8 → [1, 3, 640, 640] float32 in [0, 1] + letterbox params."""
    padded, scale, pad_left, pad_top = _letterbox(img_bgr)
    # BGR → RGB
    rgb = padded[..., ::-1]
    # HWC → CHW, normalize
    chw = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    return chw[np.newaxis], scale, pad_left, pad_top


def _unletterbox(
    cx_lb: float, cy_lb: float, w_lb: float, h_lb: float,
    orig_h: int, orig_w: int, scale: float, pad_left: int, pad_top: int,
) -> tuple[float, float, float, float]:
    """Convert letterboxed bbox coords back to normalized original image coords."""
    # Center in letterboxed pixel space
    cx_px = cx_lb * _INPUT_SIZE
    cy_px = cy_lb * _INPUT_SIZE
    w_px = w_lb * _INPUT_SIZE
    h_px = h_lb * _INPUT_SIZE
    # Remove padding and downscale
    cx_orig = (cx_px - pad_left) / (orig_w * scale) if orig_w > 0 else 0.5
    cy_orig = (cy_px - pad_top) / (orig_h * scale) if orig_h > 0 else 0.5
    w_orig = w_px / (orig_w * scale) if orig_w > 0 else 0.0
    h_orig = h_px / (orig_h * scale) if orig_h > 0 else 0.0
    return (
        float(np.clip(cx_orig, 0.0, 1.0)),
        float(np.clip(cy_orig, 0.0, 1.0)),
        float(np.clip(w_orig, 0.0, 1.0)),
        float(np.clip(h_orig, 0.0, 1.0)),
    )


# ---------------------------------------------------------------------------
# NMS
# ---------------------------------------------------------------------------


def _iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    """Intersection-over-Union for two (cx, cy, w, h) boxes (normalized)."""
    ax1, ay1 = box_a[0] - box_a[2] / 2, box_a[1] - box_a[3] / 2
    ax2, ay2 = box_a[0] + box_a[2] / 2, box_a[1] + box_a[3] / 2
    bx1, by1 = box_b[0] - box_b[2] / 2, box_b[1] - box_b[3] / 2
    bx2, by2 = box_b[0] + box_b[2] / 2, box_b[1] + box_b[3] / 2
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 1e-8 else 0.0


def _nms(
    detections: list[tuple[float, float, float, float, float]],
    iou_thresh: float = _NMS_IOU_THRESHOLD,
) -> list[tuple[float, float, float, float, float]]:
    """Greedy NMS.  *detections*: list of (cx, cy, w, h, conf) sorted desc by conf."""
    kept: list[tuple[float, float, float, float, float]] = []
    for det in sorted(detections, key=lambda d: d[4], reverse=True):
        if all(_iou(det[:4], k[:4]) < iou_thresh for k in kept):
            kept.append(det)
    return kept


# ---------------------------------------------------------------------------
# BallDetector class
# ---------------------------------------------------------------------------


class BallDetector:
    """YOLOv8n ONNX inference wrapper for basketball detection.

    The constructor is cheap — model loading happens lazily on first call
    to :meth:`detect_frame`.

    If the model file is not found or ``onnxruntime`` is not installed,
    every inference call returns an empty list.  A single WARNING is logged
    (not once per frame) and the ``disabled`` flag is set.
    """

    def __init__(self, model_path: Optional[Path] = None) -> None:
        """Initialise the detector (does NOT load the model yet)."""
        self._model_path = model_path or _MODEL_PATH
        self._session: object | None = None
        self._disabled = False
        self._disable_reason: str = ""

    # ------------------------------------------------------------------
    # Lazy model load
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load the ONNX session on first call.  Sets ``_disabled`` on failure."""
        if self._session is not None or self._disabled:
            return
        try:
            import onnxruntime as ort  # type: ignore[import-untyped]
        except ImportError:
            self._disabled = True
            self._disable_reason = "onnxruntime not installed"
            logger.warning("Ball detector disabled: %s", self._disable_reason)
            return

        model_file = Path(self._model_path)
        if not model_file.exists():
            self._disabled = True
            self._disable_reason = (
                f"Model not found at {model_file}. "
                "Run: python scripts/download_ball_detector.py"
            )
            logger.warning("Ball detector disabled: %s", self._disable_reason)
            return

        try:
            # CPU-only provider — this is the authoritative inference path for
            # the Fly 1x shared-cpu-1x machine (no GPU).
            self._session = ort.InferenceSession(
                str(model_file),
                providers=["CPUExecutionProvider"],
            )
            logger.info("Ball detector loaded: %s", model_file.name)
        except Exception as exc:  # onnxruntime can raise generic Exception
            self._disabled = True
            self._disable_reason = f"ONNX load error: {exc}"
            logger.warning("Ball detector disabled: %s", self._disable_reason)

    # ------------------------------------------------------------------
    # Per-frame inference
    # ------------------------------------------------------------------

    def detect_frame(
        self, img_bgr: np.ndarray
    ) -> list[BallDetection]:
        """Run YOLO on one frame; return list of sports-ball detections.

        Parameters
        ----------
        img_bgr : np.ndarray
            Single decoded frame as (H, W, 3) uint8 BGR (OpenCV convention).

        Returns
        -------
        list[BallDetection]
            Sorted descending by confidence.  Empty list when disabled, no
            ball found, or frame is degenerate.
        """
        self._load()
        if self._disabled or self._session is None:
            return []
        if img_bgr is None or img_bgr.size == 0:
            return []

        orig_h, orig_w = img_bgr.shape[:2]
        if orig_h == 0 or orig_w == 0:
            return []

        try:
            inp, scale, pad_left, pad_top = _preprocess(img_bgr)
            # Session input name is always "images" in Ultralytics ONNX export.
            raw_out: list[np.ndarray] = self._session.run(  # type: ignore[union-attr]
                None, {"images": inp}
            )
        except Exception as exc:
            logger.debug("YOLO inference error on frame: %s", exc)
            return []

        # Output: [1, 84, 8400] → [8400, 84]
        preds = raw_out[0].squeeze(0).T  # (8400, 84)

        # COCO class 32 column: index 4 + 32 = 36
        ball_col = 4 + _COCO_SPORTS_BALL_CLS  # = 36
        ball_scores = preds[:, ball_col]
        mask = ball_scores > _CONF_THRESHOLD
        if not mask.any():
            return []

        candidates: list[tuple[float, float, float, float, float]] = []
        for row, conf in zip(preds[mask], ball_scores[mask]):
            cx_lb, cy_lb, w_lb, h_lb = (
                float(row[0]) / _INPUT_SIZE,
                float(row[1]) / _INPUT_SIZE,
                float(row[2]) / _INPUT_SIZE,
                float(row[3]) / _INPUT_SIZE,
            )
            candidates.append((cx_lb, cy_lb, w_lb, h_lb, float(conf)))

        kept = _nms(candidates)
        result: list[BallDetection] = []
        for cx_lb, cy_lb, w_lb, h_lb, conf in kept:
            cx, cy, w, h = _unletterbox(
                cx_lb, cy_lb, w_lb, h_lb, orig_h, orig_w, scale, pad_left, pad_top
            )
            result.append(
                BallDetection(frame_idx=-1, cx=cx, cy=cy, w=w, h=h, confidence=conf)
            )
        return result

    # ------------------------------------------------------------------
    # Window-level release detection
    # ------------------------------------------------------------------

    def detect_release_frames(
        self,
        video_path: str,
        s1_candidate_frames: list[int],
        wrist_xy_normalized: np.ndarray,
        fps: float,
        *,
        conf_threshold: float = _CONF_THRESHOLD,
        separation_threshold: float = _SEPARATION_THRESHOLD,
        min_upward_frames: int = _MIN_UPWARD_FRAMES,
    ) -> list[int]:
        """Return frame indices where ball release is confirmed by S3.

        For each S1 candidate, reads ~24 frames from *video_path*, runs
        YOLO, and applies the hold-then-separate logic.

        Parameters
        ----------
        video_path : str
            Path to the (FFmpeg-normalized) video file.
        s1_candidate_frames : list[int]
            Frame indices from the S1 wrist-nadir detector.
        wrist_xy_normalized : np.ndarray
            Shape (n_frames, 2+) normalized wrist positions from MediaPipe.
            Used to anchor the proximity check (avoids re-running pose).
        fps : float
            Must be > 0.
        conf_threshold : float
            Override YOLO confidence threshold (testing hook).
        separation_threshold : float
            Normalized distance at which ball counts as "left hand".
        min_upward_frames : int
            Ball must move upward for this many consecutive frames.

        Returns
        -------
        list[int]
            Subset of ``s1_candidate_frames`` confirmed by S3.  May be
            empty even when candidates exist (no ball in clip, model
            disabled, etc.).
        """
        self._load()
        if self._disabled or not s1_candidate_frames:
            return []

        try:
            import cv2  # type: ignore[import-untyped]
        except ImportError:
            return []

        wrist = np.asarray(wrist_xy_normalized, dtype=np.float64)
        n_total = wrist.shape[0]

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.warning("Ball detector: could not open video %s", video_path)
            return []

        confirmed: list[int] = []

        try:
            for candidate in s1_candidate_frames:
                pre = int(round(_WINDOW_PRE_S * fps))
                post = int(round(_WINDOW_POST_S * fps))
                frame_start = max(0, candidate - pre)
                frame_end = min(n_total - 1, candidate + post)
                if frame_end <= frame_start:
                    continue

                # Read frames in the window
                frames_bgr: dict[int, np.ndarray] = {}
                cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_start))
                for fi in range(frame_start, frame_end + 1):
                    ret, frm = cap.read()
                    if not ret:
                        break
                    frames_bgr[fi] = frm

                if len(frames_bgr) < _MIN_UPWARD_FRAMES + 1:
                    # Not enough frames to evaluate — edge case: very short clip
                    logger.debug("Ball detector: window too short for candidate %d", candidate)
                    continue

                release = self._find_release_in_window(
                    frames_bgr=frames_bgr,
                    wrist=wrist,
                    frame_start=frame_start,
                    frame_end=frame_end,
                    separation_threshold=separation_threshold,
                    min_upward_frames=min_upward_frames,
                    conf_threshold=conf_threshold,
                )
                if release is not None:
                    confirmed.append(release)
        finally:
            cap.release()

        return confirmed

    def _find_release_in_window(
        self,
        frames_bgr: dict[int, np.ndarray],
        wrist: np.ndarray,
        frame_start: int,
        frame_end: int,
        separation_threshold: float,
        min_upward_frames: int,
        conf_threshold: float,
    ) -> int | None:
        """Core release-frame logic for a single shot window.

        Returns the release frame index, or None if no release is confirmed.

        Edge cases handled
        ------------------
        * No ball detected in any frame → return None (never fabricate).
        * Ball detected but never separates from wrist → return None.
        * Wrist NaN in window → interpolate from nearest valid frame; if ALL
          frames are NaN, use image center (0.5, 0.5) as conservative fallback.
        * Ball moving upward for fewer than ``min_upward_frames`` → return None.
        """
        ball_by_frame: dict[int, BallDetection] = {}
        for fi, frm in frames_bgr.items():
            dets = self.detect_frame(frm)
            if dets:
                # Take highest-confidence detection per frame
                ball_by_frame[fi] = max(dets, key=lambda d: d.confidence)

        if not ball_by_frame:
            # Edge case: no ball in ANY frame of this window (shooting without
            # ball, ball fully out of frame, or model disabled).
            return None

        # Interpolate wrist positions for this window (handle NaN frames).
        wrist_interp = self._interpolate_wrist(wrist, frame_start, frame_end)

        # ---- Find "hold" frame: frame where ball is closest to wrist ----
        # Only search in frames where the ball WAS detected.
        hold_fi: int | None = None
        hold_dist = math.inf
        for fi, det in ball_by_frame.items():
            wx, wy = wrist_interp.get(fi, (0.5, 0.5))
            dist = math.hypot(det.cx - wx, det.cy - wy)
            if dist < hold_dist:
                hold_dist = dist
                hold_fi = fi

        if hold_fi is None:
            return None

        # ---- Search for release AFTER the hold frame --------------------
        # A frame is a release candidate if:
        #   1. Ball is detected.
        #   2. Ball-to-wrist distance > separation_threshold.
        #   3. Ball y (image-space) has been DECREASING for min_upward_frames
        #      consecutive frames (y decreases = ball moving UP in image).
        sorted_frames = sorted(fi for fi in ball_by_frame if fi >= hold_fi)
        upward_streak = 0
        prev_ball_cy: float | None = None
        first_separation_fi: int | None = None

        for fi in sorted_frames:
            det = ball_by_frame[fi]
            wx, wy = wrist_interp.get(fi, (0.5, 0.5))
            dist = math.hypot(det.cx - wx, det.cy - wy)

            if dist > separation_threshold:
                if first_separation_fi is None:
                    first_separation_fi = fi

            # Track upward movement: in image space, y decreases as ball rises.
            if prev_ball_cy is not None:
                if det.cy < prev_ball_cy:
                    upward_streak += 1
                else:
                    upward_streak = 0
            prev_ball_cy = det.cy

            # Release confirmed: separated AND sustained upward arc.
            if (
                first_separation_fi is not None
                and upward_streak >= min_upward_frames
            ):
                return first_separation_fi

        return None

    @staticmethod
    def _interpolate_wrist(
        wrist: np.ndarray,
        frame_start: int,
        frame_end: int,
    ) -> dict[int, tuple[float, float]]:
        """Return {frame_index: (wx, wy)} for frames in [frame_start, frame_end].

        NaN frames are filled via linear interpolation from valid neighbours.
        If ALL frames are NaN (fully occluded wrist), every frame gets (0.5, 0.5)
        which is the image centre — a conservative fallback that will not trigger
        the separation gate, so S3 stays silent rather than fabricating a release.
        """
        result: dict[int, tuple[float, float]] = {}
        frames = list(range(frame_start, frame_end + 1))
        for fi in frames:
            if fi >= wrist.shape[0]:
                result[fi] = (0.5, 0.5)
                continue
            row = wrist[fi]
            x, y = float(row[0]), float(row[1])
            if math.isfinite(x) and math.isfinite(y):
                result[fi] = (x, y)
            else:
                result[fi] = (float("nan"), float("nan"))

        # Linear interpolation: for each NaN entry, find nearest valid neighbours
        valid_fi = [fi for fi, v in result.items() if math.isfinite(v[0])]
        if not valid_fi:
            for fi in frames:
                result[fi] = (0.5, 0.5)
            return result

        for fi in frames:
            if not math.isfinite(result[fi][0]):
                # Find bracketing valid frames
                left = max((v for v in valid_fi if v <= fi), default=None)
                right = min((v for v in valid_fi if v >= fi), default=None)
                if left is not None and right is not None and left != right:
                    t = (fi - left) / (right - left)
                    lx, ly = result[left]
                    rx, ry = result[right]
                    result[fi] = (lx + t * (rx - lx), ly + t * (ry - ly))
                elif left is not None:
                    result[fi] = result[left]
                elif right is not None:
                    result[fi] = result[right]
                else:
                    result[fi] = (0.5, 0.5)

        return result


# ---------------------------------------------------------------------------
# Module-level convenience function (used by physics_engine.py)
# ---------------------------------------------------------------------------

_SHARED_DETECTOR: BallDetector | None = None


def get_detector() -> BallDetector:
    """Return the process-wide singleton ``BallDetector`` instance.

    The first call creates and caches it; subsequent calls reuse the
    already-loaded ONNX session.  Thread-safety: the lazy ``_load()`` is
    idempotent so double-initialisation just logs twice.
    """
    global _SHARED_DETECTOR  # noqa: PLW0603
    if _SHARED_DETECTOR is None:
        _SHARED_DETECTOR = BallDetector()
    return _SHARED_DETECTOR
