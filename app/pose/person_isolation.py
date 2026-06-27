"""
P2 — optional person ROI before 2D pose (offline eval / hard clips).

Uses **bundled OpenCV Haar cascades** (upper-body, then full-body fallback) for a
zero–extra-weight detection path, plus **OpenCV TrackerMIL** for temporal continuity.
This is a **research-grade heuristic**, not a SOTA detector: expect misses on hard
angles, occlusion, or clutter. When no person box is found, the pipeline falls back
to the **full frame** (same as pre-P2 behaviour).

Landmarks/keypoints are always reported in **normalized coordinates of the full
working frame** (after ``preprocess_frame_max720``), so metrics stay comparable to
non-isolated runs.

**RTMPose:** ``rtmlib.Body`` still runs YOLOX on the **crop**—P2 stacks Haar+MIL in front for
experiments on crowded frames, not for a minimal single-detector pipeline.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Public mode id for CLI / provenance (extend with dnn_ssd_v1, etc. later).
PERSON_ISOLATION_MODE_HAAR_MIL_V1 = "haar_mil_v1"

_VALID_MODES = frozenset({PERSON_ISOLATION_MODE_HAAR_MIL_V1})


def normalize_person_isolation_mode(mode: str | None) -> str | None:
    if mode is None or (isinstance(mode, str) and not mode.strip()):
        return None
    key = mode.strip().lower().replace("-", "_")
    if key not in _VALID_MODES:
        raise ValueError(
            f"Unknown person_isolation mode {mode!r}. Supported: {sorted(_VALID_MODES)}"
        )
    return key


def unmap_normalized_xy_from_crop(
    x_crop: float,
    y_crop: float,
    x0: int,
    y0: int,
    wc: int,
    hc: int,
    full_w: int,
    full_h: int,
) -> tuple[float, float]:
    """Map normalized coords in the crop back to normalized coords on the full frame."""
    if full_w <= 0 or full_h <= 0 or wc <= 0 or hc <= 0:
        return x_crop, y_crop
    px = x_crop * wc + x0
    py = y_crop * hc + y0
    return px / full_w, py / full_h


def clamp_roi_xyxy(
    x0: int, y0: int, x1: int, y1: int, w: int, h: int, *, min_side: int = 48
) -> tuple[int, int, int, int]:
    x0 = int(np.clip(x0, 0, max(0, w - 1)))
    y0 = int(np.clip(y0, 0, max(0, h - 1)))
    x1 = int(np.clip(x1, x0 + 1, w))
    y1 = int(np.clip(y1, y0 + 1, h))
    if x1 - x0 < min_side:
        pad = min_side - (x1 - x0)
        x0 = max(0, x0 - pad // 2)
        x1 = min(w, x0 + min_side)
        x0 = max(0, x1 - min_side)
    if y1 - y0 < min_side:
        pad = min_side - (y1 - y0)
        y0 = max(0, y0 - pad // 2)
        y1 = min(h, y0 + min_side)
        y0 = max(0, y1 - min_side)
    return x0, y0, x1, y1


def expand_xyxy(
    x0: int, y0: int, x1: int, y1: int, w: int, h: int, margin: float
) -> tuple[int, int, int, int]:
    m = int(round(margin * max(x1 - x0, y1 - y0, 1)))
    return x0 - m, y0 - m, x1 + m, y1 + m


def _largest_haar_box(
    gray: np.ndarray,
    cascade: cv2.CascadeClassifier,
    *,
    min_side: int,
    scale_factor: float = 1.06,
    min_neighbors: int = 4,
) -> tuple[int, int, int, int] | None:
    boxes = cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=(min_side, min_side * 2),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )
    if boxes is None or len(boxes) == 0:
        return None
    best = max(boxes, key=lambda b: int(b[2]) * int(b[3]))
    x, y, bw, bh = int(best[0]), int(best[1]), int(best[2]), int(best[3])
    return x, y, x + bw, y + bh


@dataclass
class HaarMilPersonIsolation:
    """
    Per-clip stateful ROI provider. Call :meth:`start_clip` before each decode pass.

    ``step`` returns ``(x0, y0, x1, y1)`` in pixels on the current BGR frame.

    Counters: ``frames_full_frame_fallback`` counts frames whose ROI is the full image
    (Haar miss, coast between Haar attempts when no tracker, or tracker lost).
    ``haar_detection_attempts`` counts actual Haar runs (at most ~1 per ``redetect_every_n_frames``
    when no person is ever detected — not once per video frame).
    """

    redetect_every_n_frames: int = 12
    margin_frac: float = 0.12
    min_side_px: int = 48
    implementation_id: str = "opencv_haar_upperbody_fullbody_tracker_mil_v1"

    _tracker: Any = field(default=None, repr=False)
    _frame_index: int = field(default=0, repr=False)
    _cascade_ub: cv2.CascadeClassifier = field(init=False, repr=False)
    _cascade_fb: cv2.CascadeClassifier = field(init=False, repr=False)

    frames_full_frame_fallback: int = 0
    haar_detection_attempts: int = 0
    tracker_update_failures: int = 0

    def __post_init__(self) -> None:
        ub_path = cv2.data.haarcascades + "haarcascade_upperbody.xml"  # type: ignore[attr-defined]  # cv2.data exists at runtime; not in stubs
        fb_path = cv2.data.haarcascades + "haarcascade_fullbody.xml"  # type: ignore[attr-defined]  # cv2.data exists at runtime; not in stubs
        self._cascade_ub = cv2.CascadeClassifier(ub_path)
        self._cascade_fb = cv2.CascadeClassifier(fb_path)
        if self._cascade_ub.empty() or self._cascade_fb.empty():
            raise RuntimeError("OpenCV Haar cascade XML missing — check opencv-python-headless install")

    def start_clip(self) -> None:
        self._tracker = None
        self._frame_index = 0
        self.frames_full_frame_fallback = 0
        self.haar_detection_attempts = 0
        self.tracker_update_failures = 0

    def stats_dict(self, *, mode: str) -> dict[str, Any]:
        return {
            "mode": mode,
            "implementation_id": self.implementation_id,
            "redetect_every_n_frames": self.redetect_every_n_frames,
            "margin_frac": self.margin_frac,
            "min_side_px": self.min_side_px,
            "frames_full_frame_fallback": self.frames_full_frame_fallback,
            "haar_detection_attempts": self.haar_detection_attempts,
            # Same as haar_detection_attempts; kept for older JSONL readers.
            "redetect_events": self.haar_detection_attempts,
            "tracker_update_failures": self.tracker_update_failures,
        }

    def _detect_largest(self, bgr: np.ndarray) -> tuple[int, int, int, int] | None:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        h, w = gray.shape[:2]
        box = _largest_haar_box(
            gray, self._cascade_ub, min_side=self.min_side_px
        ) or _largest_haar_box(gray, self._cascade_fb, min_side=self.min_side_px)
        if box is None:
            return None
        x0, y0, x1, y1 = box
        x0, y0, x1, y1 = expand_xyxy(x0, y0, x1, y1, w, h, self.margin_frac)
        return clamp_roi_xyxy(x0, y0, x1, y1, w, h, min_side=self.min_side_px)

    def _init_tracker(self, bgr: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> bool:
        self._tracker = cv2.TrackerMIL_create()  # type: ignore[attr-defined]  # exists at runtime; not in cv2 stubs
        bw, bh = x1 - x0, y1 - y0
        bbox = (float(x0), float(y0), float(bw), float(bh))
        try:
            self._tracker.init(bgr, bbox)
        except cv2.error as e:
            logger.debug("TrackerMIL init failed: %s", e)
            self._tracker = None
            return False
        return True

    def step(self, bgr: np.ndarray) -> tuple[int, int, int, int]:
        h, w = bgr.shape[:2]
        R = max(1, self.redetect_every_n_frames)

        # When there is no tracker (Haar never hit, or we cleared it), only run Haar every R
        # frames — not every frame. Otherwise `_tracker is None` forces redetect forever and
        # `haar_detection_attempts` equals frame count (misleading).
        no_tracker_run_haar = self._tracker is None and (self._frame_index % R == 0)
        tracker_periodic_haar = (
            self._tracker is not None
            and self._frame_index > 0
            and (self._frame_index % R == 0)
        )

        if no_tracker_run_haar or tracker_periodic_haar:
            box = self._detect_largest(bgr)
            self.haar_detection_attempts += 1
            if box is None:
                self.frames_full_frame_fallback += 1
                self._tracker = None
                self._frame_index += 1
                return 0, 0, w, h
            x0, y0, x1, y1 = box
            if not self._init_tracker(bgr, x0, y0, x1, y1):
                self.frames_full_frame_fallback += 1
                self._tracker = None
                self._frame_index += 1
                return 0, 0, w, h
            self._frame_index += 1
            return x0, y0, x1, y1

        if self._tracker is None:
            # Between Haar attempts: full frame, no extra Haar cost.
            self.frames_full_frame_fallback += 1
            self._frame_index += 1
            return 0, 0, w, h

        ok, tbox = self._tracker.update(bgr)
        if not ok:
            self.tracker_update_failures += 1
            box = self._detect_largest(bgr)
            self.haar_detection_attempts += 1
            if box is None:
                self.frames_full_frame_fallback += 1
                self._tracker = None
                self._frame_index += 1
                return 0, 0, w, h
            x0, y0, x1, y1 = box
            if not self._init_tracker(bgr, x0, y0, x1, y1):
                self.frames_full_frame_fallback += 1
                self._tracker = None
                self._frame_index += 1
                return 0, 0, w, h
            self._frame_index += 1
            return x0, y0, x1, y1

        fx, fy, fbw, fbh = tbox
        x0, y0 = int(round(fx)), int(round(fy))
        x1, y1 = int(round(fx + fbw)), int(round(fy + fbh))
        x0, y0, x1, y1 = clamp_roi_xyxy(x0, y0, x1, y1, w, h, min_side=self.min_side_px)
        self._frame_index += 1
        return x0, y0, x1, y1


def create_person_isolation(mode: str | None) -> HaarMilPersonIsolation | None:
    m = normalize_person_isolation_mode(mode)
    if m is None:
        return None
    if m == PERSON_ISOLATION_MODE_HAAR_MIL_V1:
        return HaarMilPersonIsolation()
    raise ValueError(f"Unhandled person_isolation mode {m!r}")
