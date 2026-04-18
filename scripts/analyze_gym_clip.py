#!/usr/bin/env python3
"""End-to-end gym clip analysis: pose -> rep segmentation -> per-rep features -> calibration.

The script supports two input modes so it can be used both in production
(with a real video) and in testing / offline demos (with pre-extracted frames):

  --video PATH       Run MediaPipe on the video then proceed.
  --frames-json PATH Read a JSON file containing pre-extracted canonical frames
                     (``{"fps": 30.0, "frames": [{"joint": {...}}, null, ...]}``)
                     and skip the pose-extraction step.  Useful for reproducible
                     tests and for demo environments where MediaPipe is unavailable.

Output is a single JSON blob written to ``--out`` (or stdout) with the structure::

  {
    "schema_version": "1.0.0",
    "exercise_id": "back_squat",
    "source": "video" | "frames_json",
    "fps": 30.0,
    "n_frames": 150,
    "segment": { ...SegmentResult.to_dict()... },
    "feature_vectors": [ ...RepFeatureVector.to_dict()... ],
    "calibration": {
      "exercise_id": "...",
      "evidence_status": "uncalibrated_v0",
      "evidence_source": null,
      "per_rep": [
        { "rep_index": 0, "fields": { "rep_duration_s": {...}, ... } },
        ...
      ]
    }
  }

Usage examples::

  # Full video run
  python scripts/analyze_gym_clip.py \\
      --exercise-id back_squat --video evaluation/gym_clips/squat_001.mp4

  # Offline / test run (no MediaPipe)
  python scripts/analyze_gym_clip.py \\
      --exercise-id push_up --frames-json /tmp/push_up_frames.json

  # Pretty-print to stdout
  python scripts/analyze_gym_clip.py --exercise-id plank --video clip.mp4 --pretty

Exit codes
----------
* ``0`` — success (reps detected or zero-rep result written).
* ``1`` — validation error (unknown exercise-id, missing required args, etc.).
* ``2`` — file I/O or pose-extraction error.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.gym.calibration_v0 import (  # noqa: E402
    apply_calibration,
    load_calibration_v0,
)
from app.gym.exercises_v0 import get_exercise, validate_exercise_id  # noqa: E402
from app.gym.pose_adapter import (  # noqa: E402
    frames_json_to_canonical_frames,
)
from app.gym.rep_features import (  # noqa: E402
    RepFeaturesConfig,
    feature_vectors_from_segment,
)
from app.gym.rep_segmenter import SegmenterConfig, segment_reps  # noqa: E402

ANALYZE_SCHEMA_VERSION = "1.0.0"
DEFAULT_CALIBRATION_CONFIG = REPO_ROOT / "evaluation" / "gym_calibration_v0.json"


def _load_frames_json(path: Path) -> tuple[float, list]:
    """Parse ``--frames-json`` input; return ``(fps, canonical_frames)``."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise RuntimeError(f"cannot read frames-json {path}: {e}") from e
    if not isinstance(payload, dict):
        raise RuntimeError("frames-json must be a JSON object with 'fps' and 'frames'")
    fps = payload.get("fps")
    frames_raw = payload.get("frames")
    if fps is None or not isinstance(frames_raw, list):
        raise RuntimeError("frames-json must contain 'fps' (number) and 'frames' (list)")
    try:
        fps = float(fps)
    except (TypeError, ValueError) as e:
        raise RuntimeError(f"frames-json 'fps' must be numeric: {e}") from e
    if fps <= 0:
        raise RuntimeError(f"frames-json 'fps' must be > 0, got {fps}")
    canonical = frames_json_to_canonical_frames(frames_raw)
    return fps, canonical


def _load_video(
    video_path: Path,
    *,
    multipass: bool,
    person_isolation: str | None,
) -> tuple[float, list]:
    """Run MediaPipe on ``video_path``; return ``(fps, canonical_frames)``."""
    try:
        from app.gym.pose_adapter import extract_canonical_frames
    except ImportError as e:
        raise RuntimeError(
            f"MediaPipe not available ({e}); use --frames-json for offline mode"
        ) from e
    try:
        fps, frames = extract_canonical_frames(
            video_path, multipass=multipass, person_isolation=person_isolation
        )
    except Exception as e:
        raise RuntimeError(f"pose extraction failed: {e}") from e
    return fps, frames


def _build_output(
    exercise_id: str,
    source: str,
    fps: float,
    canonical_frames: list,
    calibration_path: Path,
    seg_config: SegmenterConfig | None,
    feat_config: RepFeaturesConfig | None,
) -> dict[str, Any]:
    """Run segmentation + feature extraction + calibration; build output dict."""
    exercise = get_exercise(exercise_id)
    if exercise is None:
        raise RuntimeError(f"unknown exercise_id {exercise_id!r}")

    from app.gym.rep_features import extract_rep_signal

    signal, _miss = extract_rep_signal(canonical_frames, exercise)

    seg_result = segment_reps(
        signal=signal,
        fps=fps,
        exercise=exercise,
        config=seg_config,
    )

    feature_vectors = feature_vectors_from_segment(
        segment=seg_result,
        canonical_frames=canonical_frames,
        exercise=exercise,
        config=feat_config,
    )

    # Calibration
    cal_manifest = load_calibration_v0(calibration_path)
    cal_entry = cal_manifest.get(exercise_id)

    per_rep_cal: list[dict[str, Any]] = []
    if cal_entry is not None:
        for fv in feature_vectors:
            per_rep_cal.append(
                {
                    "rep_index": fv.rep_index,
                    "fields": apply_calibration(cal_entry, fv),
                }
            )
    cal_block: dict[str, Any] = {
        "exercise_id": exercise_id,
        "evidence_status": cal_entry.evidence_status if cal_entry else "no_config",
        "evidence_source": cal_entry.evidence_source if cal_entry else None,
        "comparable_fields": list(cal_entry.comparable_fields) if cal_entry else [],
        "per_rep": per_rep_cal,
    }

    return {
        "schema_version": ANALYZE_SCHEMA_VERSION,
        "exercise_id": exercise_id,
        "source": source,
        "fps": fps,
        "n_frames": len(canonical_frames),
        "segment": seg_result.to_dict(),
        "feature_vectors": [fv.to_dict() for fv in feature_vectors],
        "calibration": cal_block,
    }


def main() -> int:
    """Entry point."""
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    input_group = ap.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--video", type=Path, help="Video file (runs MediaPipe)")
    input_group.add_argument(
        "--frames-json",
        type=Path,
        dest="frames_json",
        help="Pre-extracted frames JSON (skips MediaPipe; for tests/offline demos)",
    )
    ap.add_argument(
        "--exercise-id",
        required=True,
        help="Frozen exercise ID (e.g. back_squat, bench_press, plank)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write JSON output to this file (default: stdout)",
    )
    ap.add_argument(
        "--pretty",
        action="store_true",
        help="Indent JSON output (pretty-print)",
    )
    ap.add_argument(
        "--calibration-config",
        type=Path,
        default=DEFAULT_CALIBRATION_CONFIG,
        dest="calibration_config",
        help=f"Calibration config JSON (default: {DEFAULT_CALIBRATION_CONFIG.relative_to(REPO_ROOT)})",
    )
    ap.add_argument(
        "--multipass",
        action="store_true",
        help="MediaPipe multipass (baseline/gamma/denoise) — only used with --video",
    )
    ap.add_argument(
        "--person-isolation",
        default=None,
        dest="person_isolation",
        help="Person-isolation mode (e.g. haar_mil_v1) — only used with --video",
    )
    args = ap.parse_args()

    # Validate exercise-id eagerly.
    err = validate_exercise_id(args.exercise_id)
    if err:
        print(json.dumps({"ok": False, "error": err}), file=sys.stderr)
        return 1
    if get_exercise(args.exercise_id) is None:
        print(
            json.dumps({"ok": False, "error": f"reserved token {args.exercise_id!r} is not analysable"}),
            file=sys.stderr,
        )
        return 1

    # Validate calibration config path.
    if not args.calibration_config.is_file():
        print(
            json.dumps(
                {"ok": False, "error": f"calibration config not found: {args.calibration_config}"}
            ),
            file=sys.stderr,
        )
        return 2

    # Load frames.
    if args.frames_json is not None:
        try:
            fps, canonical_frames = _load_frames_json(args.frames_json)
            source = "frames_json"
        except RuntimeError as e:
            print(json.dumps({"ok": False, "error": str(e)}), file=sys.stderr)
            return 2
    else:
        if not args.video.is_file():
            print(
                json.dumps({"ok": False, "error": f"video not found: {args.video}"}),
                file=sys.stderr,
            )
            return 2
        try:
            fps, canonical_frames = _load_video(
                args.video,
                multipass=args.multipass,
                person_isolation=args.person_isolation,
            )
            source = "video"
        except RuntimeError as e:
            print(json.dumps({"ok": False, "error": str(e)}), file=sys.stderr)
            return 2

    # Run pipeline.
    try:
        result = _build_output(
            exercise_id=args.exercise_id,
            source=source,
            fps=fps,
            canonical_frames=canonical_frames,
            calibration_path=args.calibration_config,
            seg_config=None,
            feat_config=None,
        )
    except RuntimeError as e:
        print(json.dumps({"ok": False, "error": str(e)}), file=sys.stderr)
        return 1
    except Exception as e:
        print(
            json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"}),
            file=sys.stderr,
        )
        return 2

    indent = 2 if args.pretty else None
    out_str = json.dumps(result, indent=indent)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(out_str + "\n", encoding="utf-8")
        print(
            json.dumps(
                {
                    "ok": True,
                    "written": str(args.out),
                    "exercise_id": result["exercise_id"],
                    "n_frames": result["n_frames"],
                    "n_reps": len(result["feature_vectors"]),
                    "fps": result["fps"],
                }
            )
        )
    else:
        print(out_str)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
