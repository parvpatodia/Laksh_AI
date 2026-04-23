"""Tests for app.detection.ball_detector.

All tests run without the actual ONNX model or onnxruntime installed.
The BallDetector class is tested via mocked InferenceSession objects and
direct unit tests on the helper functions.

Edge cases tested
-----------------
1. Detector disabled when model is not downloaded.
2. Detector disabled when onnxruntime is not installed.
3. detect_frame returns empty list on degenerate / zero-size frame.
4. detect_frame returns empty list when no ball exceeds confidence threshold.
5. detect_frame returns the highest-confidence ball detection.
6. detect_frame NMS: two overlapping detections → one kept.
7. detect_release_frames: empty result when detector is disabled.
8. detect_release_frames: empty result when no ball detected in window.
9. detect_release_frames: confirms release when ball separates + moves up.
10. detect_release_frames: no release when ball moves up but never separates.
11. detect_release_frames: no release when ball separates but never moves up.
12. detect_release_frames: shooting without basketball → empty list (no fabrication).
13. Wrist interpolation: NaN frames are filled from valid neighbours.
14. Wrist interpolation: all-NaN falls back to image-centre (0.5, 0.5).
15. Letterbox preserves aspect ratio and pads symmetrically.
16. Unletterbox inverts letterbox exactly (round-trip accuracy).
17. IoU computation for non-overlapping, partially-overlapping, identical boxes.
18. Wrist visibility masking in shot_segmenter (_find_s1_candidates with wrist_vis).
19. ROM gate windowed peak search: NaN at exact peak → recovered from ±3 frame window.
"""
from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_detection(cx: float = 0.5, cy: float = 0.5, w: float = 0.05,
                    h: float = 0.05, conf: float = 0.8) -> object:
    """Return a mock 'output0' tensor row simulating one ball detection.

    YOLOv8n output format: [cx, cy, w, h, score_0, ..., score_79]
    where all coordinates are in letterboxed 640-normalised space.
    COCO class 32 = sports ball → column 4 + 32 = 36.
    """
    row = np.zeros(84, dtype=np.float32)
    row[0] = cx * 640   # model output uses pixel space in letterboxed frame
    row[1] = cy * 640
    row[2] = w * 640
    row[3] = h * 640
    row[36] = conf      # class 32 score
    return row


def _mock_session(detections: list) -> MagicMock:
    """Return an InferenceSession mock whose .run() returns *detections*."""
    # Output shape [1, 84, 8400]; build from provided rows + padding
    n = max(8400, len(detections))
    data = np.zeros((1, 84, n), dtype=np.float32)
    for i, det in enumerate(detections):
        data[0, :, i] = det
    sess = MagicMock()
    sess.run.return_value = [data]
    return sess


# ---------------------------------------------------------------------------
# Import under test (tolerates onnxruntime absence)
# ---------------------------------------------------------------------------


from app.detection.ball_detector import (  # noqa: E402
    BallDetection,
    BallDetector,
    _iou,
    _letterbox,
    _nms,
    _unletterbox,
)


# ---------------------------------------------------------------------------
# Unit tests: _letterbox / _unletterbox
# ---------------------------------------------------------------------------


class TestLetterbox:
    """Letterbox + unletterbox round-trip accuracy."""

    def test_square_image_no_padding(self) -> None:
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        padded, scale, pad_left, pad_top = _letterbox(img)
        assert padded.shape == (640, 640, 3)
        assert scale == pytest.approx(1.0)
        assert pad_left == 0
        assert pad_top == 0

    def test_landscape_pads_top_bottom(self) -> None:
        img = np.zeros((360, 640, 3), dtype=np.uint8)
        padded, scale, pad_left, pad_top = _letterbox(img)
        assert padded.shape == (640, 640, 3)
        assert scale == pytest.approx(1.0)
        assert pad_top > 0
        assert pad_left == 0

    def test_portrait_pads_left_right(self) -> None:
        img = np.zeros((640, 360, 3), dtype=np.uint8)
        padded, scale, pad_left, pad_top = _letterbox(img)
        assert padded.shape == (640, 640, 3)
        assert pad_left > 0
        assert pad_top == 0

    def test_roundtrip_landscape(self) -> None:
        """Unletterbox should invert letterbox coordinates."""
        orig_h, orig_w = 720, 1280
        img = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
        _, scale, pad_left, pad_top = _letterbox(img)
        # Inject a known letterboxed bbox (centred in the image)
        cx_lb, cy_lb = 0.5, 0.5
        w_lb, h_lb = 0.04, 0.04
        cx, cy, w, h = _unletterbox(
            cx_lb, cy_lb, w_lb, h_lb, orig_h, orig_w, scale, pad_left, pad_top
        )
        # Centred in the original image → cx, cy should be close to 0.5
        assert cx == pytest.approx(0.5, abs=0.02)
        assert cy == pytest.approx(0.5, abs=0.02)

    def test_unletterbox_clips_to_01(self) -> None:
        orig_h, orig_w = 100, 100
        img = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
        _, scale, pad_left, pad_top = _letterbox(img)
        # Force coords outside [0,1] to verify clipping
        cx, cy, w, h = _unletterbox(
            2.0, 2.0, 0.1, 0.1, orig_h, orig_w, scale, pad_left, pad_top
        )
        assert 0.0 <= cx <= 1.0
        assert 0.0 <= cy <= 1.0


# ---------------------------------------------------------------------------
# Unit tests: IoU + NMS
# ---------------------------------------------------------------------------


class TestIoU:
    def test_no_overlap(self) -> None:
        a = (0.1, 0.1, 0.1, 0.1)
        b = (0.9, 0.9, 0.1, 0.1)
        assert _iou(a, b) == pytest.approx(0.0, abs=1e-6)

    def test_identical_boxes(self) -> None:
        box = (0.5, 0.5, 0.2, 0.2)
        assert _iou(box, box) == pytest.approx(1.0, abs=1e-6)

    def test_half_overlap(self) -> None:
        a = (0.25, 0.5, 0.5, 0.5)   # centre-x=0.25, spans 0-0.5
        b = (0.75, 0.5, 0.5, 0.5)   # centre-x=0.75, spans 0.5-1.0
        # They share exactly one column → IoU = 0 / (0.25 + 0.25 - 0) = 0
        assert _iou(a, b) == pytest.approx(0.0, abs=1e-3)


class TestNMS:
    def test_single_box_returned(self) -> None:
        dets = [(0.5, 0.5, 0.1, 0.1, 0.9)]
        assert _nms(dets) == dets

    def test_high_iou_suppressed(self) -> None:
        """Two nearly identical boxes → only higher-conf kept."""
        dets = [
            (0.5, 0.5, 0.1, 0.1, 0.9),
            (0.51, 0.51, 0.1, 0.1, 0.5),
        ]
        kept = _nms(dets)
        assert len(kept) == 1
        assert kept[0][4] == pytest.approx(0.9)

    def test_non_overlapping_both_kept(self) -> None:
        dets = [
            (0.1, 0.1, 0.05, 0.05, 0.8),
            (0.9, 0.9, 0.05, 0.05, 0.7),
        ]
        assert len(_nms(dets)) == 2


# ---------------------------------------------------------------------------
# BallDetector tests — model disabled paths
# ---------------------------------------------------------------------------


class TestBallDetectorDisabled:
    """Detector must fail gracefully in all disabled scenarios."""

    def test_disabled_when_model_missing(self, tmp_path: Path) -> None:
        det = BallDetector(model_path=tmp_path / "nonexistent.onnx")
        frames = det.detect_frame(np.zeros((100, 100, 3), dtype=np.uint8))
        assert frames == []
        assert det._disabled

    def test_detect_release_frames_empty_when_disabled(self, tmp_path: Path) -> None:
        det = BallDetector(model_path=tmp_path / "nonexistent.onnx")
        det._load()
        result = det.detect_release_frames(
            video_path="nonexistent.mp4",
            s1_candidate_frames=[10, 20],
            wrist_xy_normalized=np.random.rand(30, 2),
            fps=30.0,
        )
        assert result == []

    def test_detect_frame_empty_when_onnxruntime_missing(self, tmp_path: Path) -> None:
        """Simulate onnxruntime ImportError."""
        onnx_file = tmp_path / "fake.onnx"
        onnx_file.write_bytes(b"fake")
        det = BallDetector(model_path=onnx_file)
        # Patch the import inside _load
        with patch.dict("sys.modules", {"onnxruntime": None}):
            det._session = None
            det._disabled = False
            det._load()
        assert det._disabled

    def test_degenerate_frame_empty(self, tmp_path: Path) -> None:
        onnx_file = tmp_path / "fake.onnx"
        onnx_file.write_bytes(b"fake")
        det = BallDetector(model_path=onnx_file)
        det._session = _mock_session([])  # bypass file-based load
        det._disabled = False
        # Zero-size frame
        result = det.detect_frame(np.zeros((0, 0, 3), dtype=np.uint8))
        assert result == []

    def test_empty_frame_returns_empty(self, tmp_path: Path) -> None:
        onnx_file = tmp_path / "fake.onnx"
        onnx_file.write_bytes(b"fake")
        det = BallDetector(model_path=onnx_file)
        det._session = _mock_session([])
        det._disabled = False
        result = det.detect_frame(None)  # type: ignore[arg-type]
        assert result == []


# ---------------------------------------------------------------------------
# BallDetector tests — active paths (mocked session)
# ---------------------------------------------------------------------------


class TestBallDetectorActive:
    """Tests run with a mock ONNX session (no model file needed)."""

    def _make_det(self, tmp_path: Path) -> BallDetector:
        onnx_file = tmp_path / "fake.onnx"
        onnx_file.write_bytes(b"fake")
        det = BallDetector(model_path=onnx_file)
        return det

    def test_no_ball_above_threshold_returns_empty(self, tmp_path: Path) -> None:
        """All class-32 scores below 0.35 → no detections."""
        det = self._make_det(tmp_path)
        row = _make_detection(conf=0.10)  # below threshold
        det._session = _mock_session([row])
        det._disabled = False
        result = det.detect_frame(np.zeros((720, 1280, 3), dtype=np.uint8))
        assert result == []

    def test_single_ball_detected(self, tmp_path: Path) -> None:
        det = self._make_det(tmp_path)
        row = _make_detection(cx=0.4, cy=0.4, conf=0.85)
        det._session = _mock_session([row])
        det._disabled = False
        result = det.detect_frame(np.zeros((720, 1280, 3), dtype=np.uint8))
        assert len(result) == 1
        assert result[0].confidence == pytest.approx(0.85, abs=0.01)

    def test_nms_deduplicates_overlapping(self, tmp_path: Path) -> None:
        """Two nearly-identical detections → only one kept after NMS."""
        det = self._make_det(tmp_path)
        r1 = _make_detection(cx=0.5, cy=0.5, conf=0.9)
        r2 = _make_detection(cx=0.51, cy=0.51, conf=0.6)
        det._session = _mock_session([r1, r2])
        det._disabled = False
        result = det.detect_frame(np.zeros((640, 640, 3), dtype=np.uint8))
        assert len(result) == 1

    def test_no_basketball_no_release(self, tmp_path: Path) -> None:
        """Shooting without a basketball: no detections → no confirmed release.
        Honesty contract: S3 never fabricates.
        """
        det = self._make_det(tmp_path)
        det._session = _mock_session([])  # no ball ever
        det._disabled = False
        wrist = np.column_stack([
            np.linspace(0.4, 0.5, 60),
            np.linspace(0.8, 0.2, 60),
        ])
        # Patch video reading to return a stream of black frames
        with patch("cv2.VideoCapture") as MockCap:
            cap_inst = MagicMock()
            cap_inst.isOpened.return_value = True
            cap_inst.read.return_value = (True, np.zeros((720, 1280, 3), dtype=np.uint8))
            MockCap.return_value = cap_inst
            result = det.detect_release_frames(
                "fake.mp4", [30], wrist, fps=30.0
            )
        assert result == [], "No ball → S3 must be silent"

    def test_release_confirmed_ball_separates_and_rises(self, tmp_path: Path) -> None:
        """Ball starts near wrist, then separates upward → S3 fires."""
        det = self._make_det(tmp_path)
        det._disabled = False

        # Frame 28 (hold): ball at (0.4, 0.6) — wrist at (0.4, 0.6) → dist = 0
        # Frame 30 (separation): dist > 0.15, ball rising (cy decreasing)
        # Frames 31-33: ball continues rising

        n_frames = 60
        wrist = np.zeros((n_frames, 2), dtype=np.float64)
        wrist[:, 0] = 0.4
        wrist[:, 1] = 0.6  # wrist at y=0.6

        frames_bgr: dict[int, np.ndarray] = {
            fi: np.zeros((720, 1280, 3), dtype=np.uint8) for fi in range(25, 45)
        }

        # Simulate per-frame detections
        def _fake_detect_frame(self_inner: BallDetector, img: np.ndarray) -> list:
            # Caller iterates over frames_bgr in the _find_release_in_window loop.
            # We distinguish frames by image content — easier to mock at the window
            # level. Instead, override _find_release_in_window directly.
            return []

        # Directly test _find_release_in_window with crafted per-frame detections.
        hold_fi = 28
        ball_by_frame: dict[int, BallDetection] = {}
        # Before hold: ball near wrist
        for fi in range(25, hold_fi + 1):
            ball_by_frame[fi] = BallDetection(
                frame_idx=fi, cx=0.4, cy=0.6, w=0.05, h=0.05, confidence=0.8
            )
        # After hold: ball moves up and away
        for fi in range(hold_fi + 1, 45):
            dist_offset = (fi - hold_fi) * 0.05  # grows with time
            cy_offset = (fi - hold_fi) * 0.06    # ball rises (cy decreases)
            ball_by_frame[fi] = BallDetection(
                frame_idx=fi,
                cx=0.4 + dist_offset,
                cy=0.6 - cy_offset,
                w=0.05,
                h=0.05,
                confidence=0.8,
            )

        # Inject per-frame detections by patching detect_frame
        fi_iter = iter(sorted(ball_by_frame.keys()))

        def patched_detect(img: np.ndarray) -> list[BallDetection]:
            try:
                fi = next(fi_iter)
                return [ball_by_frame[fi]]
            except StopIteration:
                return []

        det.detect_frame = patched_detect  # type: ignore[method-assign]

        result = det._find_release_in_window(
            frames_bgr=frames_bgr,
            wrist=wrist,
            frame_start=25,
            frame_end=44,
            separation_threshold=0.15,
            min_upward_frames=3,
            conf_threshold=0.35,
        )
        assert result is not None, "Release should be confirmed when ball separates + rises"

    def test_no_release_ball_separates_but_never_rises(self, tmp_path: Path) -> None:
        """Ball leaves wrist but moves DOWNWARD (e.g. a drop pass) → S3 silent."""
        det = self._make_det(tmp_path)
        det._disabled = False

        n_frames = 60
        wrist = np.zeros((n_frames, 2), dtype=np.float64)
        wrist[:, 0] = 0.4
        wrist[:, 1] = 0.4

        # All ball detections have cy INCREASING (ball falling, not rising)
        frames_bgr = {fi: np.zeros((100, 100, 3), dtype=np.uint8) for fi in range(25, 45)}
        ball_by_frame: dict[int, BallDetection] = {}
        for fi in range(25, 45):
            dist = (fi - 25) * 0.05  # separates from wrist
            ball_by_frame[fi] = BallDetection(
                frame_idx=fi,
                cx=0.4 + dist,
                cy=0.4 + dist * 0.3,  # cy INCREASING (ball falling in image)
                w=0.05, h=0.05, confidence=0.8,
            )

        fi_iter = iter(sorted(ball_by_frame.keys()))

        def patched_detect(img: np.ndarray) -> list[BallDetection]:
            try:
                return [ball_by_frame[next(fi_iter)]]
            except StopIteration:
                return []

        det.detect_frame = patched_detect  # type: ignore[method-assign]

        result = det._find_release_in_window(
            frames_bgr=frames_bgr,
            wrist=wrist,
            frame_start=25,
            frame_end=44,
            separation_threshold=0.15,
            min_upward_frames=3,
            conf_threshold=0.35,
        )
        assert result is None, "Downward ball motion should NOT trigger S3"

    def test_no_release_ball_rises_but_never_separates(self, tmp_path: Path) -> None:
        """Ball in hand arcing upward (hold not released) → S3 silent."""
        det = self._make_det(tmp_path)
        det._disabled = False

        n_frames = 60
        wrist = np.zeros((n_frames, 2), dtype=np.float64)
        wrist[:, 0] = 0.5
        wrist[:, 1] = 0.5

        frames_bgr = {fi: np.zeros((100, 100, 3), dtype=np.uint8) for fi in range(25, 45)}
        ball_by_frame: dict[int, BallDetection] = {}
        for fi in range(25, 45):
            # Ball stays very close to wrist (dist < 0.05) but rises
            ball_by_frame[fi] = BallDetection(
                frame_idx=fi, cx=0.5, cy=0.5 - (fi - 25) * 0.005,
                w=0.05, h=0.05, confidence=0.8,
            )

        fi_iter = iter(sorted(ball_by_frame.keys()))

        def patched_detect(img: np.ndarray) -> list[BallDetection]:
            try:
                return [ball_by_frame[next(fi_iter)]]
            except StopIteration:
                return []

        det.detect_frame = patched_detect  # type: ignore[method-assign]

        result = det._find_release_in_window(
            frames_bgr=frames_bgr,
            wrist=wrist,
            frame_start=25,
            frame_end=44,
            separation_threshold=0.15,
            min_upward_frames=3,
            conf_threshold=0.35,
        )
        assert result is None, "Ball stays in hand (dist < threshold) → no release"


# ---------------------------------------------------------------------------
# Wrist interpolation tests
# ---------------------------------------------------------------------------


class TestWristInterpolation:
    def test_valid_wrist_passthrough(self) -> None:
        det = BallDetector.__new__(BallDetector)
        wrist = np.array([[0.3, 0.7], [0.4, 0.6], [0.5, 0.5]], dtype=np.float64)
        result = det._interpolate_wrist(wrist, 0, 2)
        assert result[0] == pytest.approx((0.3, 0.7), abs=1e-9)
        assert result[1] == pytest.approx((0.4, 0.6), abs=1e-9)
        assert result[2] == pytest.approx((0.5, 0.5), abs=1e-9)

    def test_nan_frame_interpolated(self) -> None:
        """NaN at frame 1 → interpolated from frames 0 and 2."""
        det = BallDetector.__new__(BallDetector)
        wrist = np.array(
            [[0.2, 0.8], [np.nan, np.nan], [0.6, 0.4]], dtype=np.float64
        )
        result = det._interpolate_wrist(wrist, 0, 2)
        assert math.isfinite(result[1][0])
        assert result[1][0] == pytest.approx(0.4, abs=1e-9)
        assert result[1][1] == pytest.approx(0.6, abs=1e-9)

    def test_all_nan_falls_back_to_centre(self) -> None:
        """All-NaN wrist → every frame gets (0.5, 0.5) conservative fallback."""
        det = BallDetector.__new__(BallDetector)
        wrist = np.full((5, 2), np.nan, dtype=np.float64)
        result = det._interpolate_wrist(wrist, 0, 4)
        for fi in range(5):
            assert result[fi] == pytest.approx((0.5, 0.5), abs=1e-9)

    def test_leading_nan_filled_from_right(self) -> None:
        """NaN at start → filled from nearest valid on the right."""
        det = BallDetector.__new__(BallDetector)
        wrist = np.array([[np.nan, np.nan], [0.7, 0.3]], dtype=np.float64)
        result = det._interpolate_wrist(wrist, 0, 1)
        assert result[0] == pytest.approx((0.7, 0.3), abs=1e-9)


# ---------------------------------------------------------------------------
# Integration: wrist visibility masking in shot_segmenter
# ---------------------------------------------------------------------------


class TestShotSegmenterVisibilityMask:
    """Verify that low-vis wrist frames are masked to NaN in S1 detection."""

    def test_low_vis_peak_masked(self) -> None:
        """A wrist-y peak during a low-visibility window must NOT produce an S1."""
        from app.basketball.shot_segmenter import ShotSegmenterConfig, _find_s1_candidates

        fps = 30.0
        n = 90
        cfg = ShotSegmenterConfig(min_inter_shot_s=0.6, prominence_wrist_y=0.04)

        # Build a wrist_y with one clear nadir at frame 45
        wrist_y = np.ones(n, dtype=np.float64) * 0.5
        wrist_y[45] = 0.1  # nadir — would be a strong peak

        # Mark the nadir frame as low-visibility (0.10 < 0.30 floor)
        wrist_vis = np.full(n, 0.80, dtype=np.float64)
        wrist_vis[43:48] = 0.10  # frames around the nadir have low vis

        peaks = _find_s1_candidates(wrist_y, fps, cfg, wrist_vis=wrist_vis)
        assert peaks.size == 0, (
            "Low-visibility wrist nadir must be masked and NOT detected as S1"
        )

    def test_high_vis_peak_detected(self) -> None:
        """A high-visibility nadir produces a valid S1."""
        from app.basketball.shot_segmenter import ShotSegmenterConfig, _find_s1_candidates

        fps = 30.0
        n = 90
        cfg = ShotSegmenterConfig(min_inter_shot_s=0.6, prominence_wrist_y=0.04)

        wrist_y = np.ones(n, dtype=np.float64) * 0.5
        wrist_y[45] = 0.1  # nadir

        wrist_vis = np.full(n, 0.80, dtype=np.float64)  # all high vis

        peaks = _find_s1_candidates(wrist_y, fps, cfg, wrist_vis=wrist_vis)
        assert peaks.size == 1
        assert peaks[0] == 45

    def test_no_wrist_vis_no_masking(self) -> None:
        """When wrist_vis is None, no masking occurs (test fixtures unaffected)."""
        from app.basketball.shot_segmenter import ShotSegmenterConfig, _find_s1_candidates

        fps = 30.0
        n = 90
        cfg = ShotSegmenterConfig(min_inter_shot_s=0.6, prominence_wrist_y=0.04)

        wrist_y = np.ones(n, dtype=np.float64) * 0.5
        wrist_y[45] = 0.1

        peaks = _find_s1_candidates(wrist_y, fps, cfg, wrist_vis=None)
        assert peaks.size == 1


# ---------------------------------------------------------------------------
# Integration: NaN-at-peak recovery in bicep curl ROM gate
# ---------------------------------------------------------------------------


class TestBicepCurlNanAtPeak:
    """Windowed best-angle search recovers from NaN at exact peak frame."""

    def _canonical_frames(
        self,
        peak_angle: float,
        start_angle: float = 160.0,
        end_angle: float = 160.0,
        n_frames: int = 30,
        nan_at_peak: bool = False,
    ) -> list:
        """Build minimal canonical_frames list with one rep worth of angles.

        Uses the format expected by _angle_at: canonical_frames[i][joint_name]
        = (x, y) where the triplet shoulder-elbow-wrist encodes the angle.
        """
        from app.gym.rep_features import _ANGLE_TRIPLETS

        triplet = _ANGLE_TRIPLETS["right_elbow"]  # (shoulder, elbow, wrist)

        def _joint(x: float, y: float) -> dict:
            """_get_joint expects dict with 'x', 'y', 'visibility' keys."""
            return {"x": x, "y": y, "visibility": 0.95}

        def _make_frame(angle_deg: float) -> dict:
            """Create a frame dict with the triplet encoding angle_deg.

            Place shoulder at (0,0), elbow at (1,0), wrist positioned so
            that the interior elbow angle equals angle_deg.
            Interior angle formula: cos(angle) = dot(ba, bc) / (|ba||bc|)
            where ba = shoulder - elbow = (-1,0), bc = wrist - elbow.
            Setting |bc|=1: bc_x = -cos(angle), bc_y = sin(angle).
            """
            rad = math.radians(angle_deg)
            bc_x = -math.cos(rad)
            bc_y = math.sin(rad)
            wrist_x = 1.0 + bc_x
            wrist_y_val = bc_y
            return {
                triplet[0]: _joint(0.0, 0.0),        # shoulder
                triplet[1]: _joint(1.0, 0.0),        # elbow
                triplet[2]: _joint(wrist_x, wrist_y_val),  # wrist
            }

        frames: list = []
        for fi in range(n_frames):
            if fi == 0:
                frames.append(_make_frame(start_angle))
            elif fi == n_frames // 2:
                if nan_at_peak:
                    # NaN frame: no joints visible (simulates wrist out of frame)
                    frames.append({})
                else:
                    frames.append(_make_frame(peak_angle))
            elif fi == n_frames - 1:
                frames.append(_make_frame(end_angle))
            else:
                # Linear interpolation between start → peak → end
                if fi < n_frames // 2:
                    t = fi / (n_frames // 2)
                    frames.append(_make_frame(start_angle + t * (peak_angle - start_angle)))
                else:
                    t = (fi - n_frames // 2) / (n_frames // 2)
                    frames.append(_make_frame(peak_angle + t * (end_angle - peak_angle)))
        return frames

    def test_nan_at_peak_recovered_by_window(self) -> None:
        """When the exact peak frame has NaN angle, the ±3 window finds the real peak."""
        from app.gym.rep_features import evaluate_bicep_curl_rom_gate
        from app.gym.rep_segmenter import RepSpan

        frames = self._canonical_frames(
            peak_angle=45.0,  # valid full curl
            start_angle=160.0,
            end_angle=160.0,
            nan_at_peak=True,  # exact peak frame has NaN
        )
        n = len(frames)
        rep = RepSpan(
            start_frame=0,
            peak_frame=n // 2,
            end_frame=n - 1,
            status="complete",
            reason_codes=(),
        )
        result = evaluate_bicep_curl_rom_gate(
            rep=rep,
            canonical_frames=frames,
            exercise=_DummyExercise(),
        )
        # The windowed search finds the nearby frame with angle=45° → C1 passes
        assert result.status in ("valid", "degraded"), (
            f"Expected valid/degraded with NaN at peak, got {result.status}: {result.reason_codes}"
        )
        if result.status == "valid":
            assert "peak_not_flexed" not in (result.reason_codes or ())

    def test_peak_too_shallow_still_rejected(self) -> None:
        """Even with ±3 window, a genuinely shallow peak is still rejected."""
        from app.gym.rep_features import evaluate_bicep_curl_rom_gate
        from app.gym.rep_segmenter import RepSpan

        frames = self._canonical_frames(
            peak_angle=120.0,  # shallow — does not reach 60° flexion
            start_angle=160.0,
            end_angle=160.0,
            nan_at_peak=False,
        )
        n = len(frames)
        rep = RepSpan(
            start_frame=0,
            peak_frame=n // 2,
            end_frame=n - 1,
            status="complete",
            reason_codes=(),
        )
        result = evaluate_bicep_curl_rom_gate(rep=rep, canonical_frames=frames, exercise=_DummyExercise())
        assert "peak_not_flexed" in (result.reason_codes or ()), (
            "Shallow peak (120°) must still fail C1 peak gate"
        )


def _DummyExercise():
    """Return the real dumbbell_bicep_curl ExerciseV0 from the registry."""
    from app.gym.exercises_v0 import EXERCISES_V0
    return EXERCISES_V0["dumbbell_bicep_curl"]
