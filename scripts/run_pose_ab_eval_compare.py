#!/usr/bin/env python3
"""
Orchestrate gym pose A/B eval (MediaPipe + optional RTMPose) and JSONL comparison.

Designed for **local / CI-with-deps** runs: MediaPipe is required; RTMPose is best-effort
if ``requirements-pose-optional.txt`` is installed (may download ONNX on first run).

Exit codes:
  0 — MediaPipe eval OK; compare printed if both JSONL exist; RTMPose may be skipped.
  1 — MediaPipe eval failed, or compare failed when both outputs exist.
  2 — MediaPipe OK but RTMPose requested and failed (compare skipped); see stderr JSON.

Use ``--skip-rtmpose`` for MediaPipe-only smoke (still writes compare if you pass two existing files — not this path).

Examples:
  python scripts/run_pose_ab_eval_compare.py --manifest evaluation/gym_manifest.csv
  python scripts/run_pose_ab_eval_compare.py --manifest evaluation/gym_manifest.csv --multipass
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_eval(
    backend: str,
    manifest: Path,
    out: Path,
    multipass: bool,
) -> tuple[int, str]:
    cmd: list[str] = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "eval_pose_baseline.py"),
        "--manifest",
        str(manifest),
        "--out",
        str(out),
        "--backend",
        backend,
    ]
    if multipass:
        cmd.append("--multipass")
    r = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    tail = ""
    if r.stdout:
        tail += r.stdout[-4000:]
    if r.stderr:
        tail += r.stderr[-4000:]
    return r.returncode, tail


def _run_compare(a: Path, b: Path, per_clip: Path | None) -> tuple[int, str]:
    cmd: list[str] = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "compare_pose_baseline_jsonl.py"),
        "--a",
        str(a),
        "--b",
        str(b),
        "--label-a",
        "mediapipe",
        "--label-b",
        "rtmpose",
    ]
    if per_clip is not None:
        cmd.extend(["--per-clip-out", str(per_clip)])
    r = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def main() -> int:
    ap = argparse.ArgumentParser(description="Run pose A/B eval + compare (P1b orchestration).")
    ap.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "evaluation" / "gym_manifest.csv",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "evaluation",
        help="Directory for JSONL outputs (default: evaluation/)",
    )
    ap.add_argument("--multipass", action="store_true")
    ap.add_argument(
        "--skip-rtmpose",
        action="store_true",
        help="Only run MediaPipe (no second backend, no compare).",
    )
    ap.add_argument(
        "--per-clip-out",
        type=Path,
        default=None,
        help="Optional: write per-clip diff JSONL (can be large for big manifests)",
    )
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_mp = args.out_dir / "pose_ab_mediapipe.jsonl"
    out_rtm = args.out_dir / "pose_ab_rtmpose.jsonl"

    report: dict = {"manifest": str(args.manifest.resolve()), "outputs": {}}

    rc_mp, log_mp = _run_eval("mediapipe", args.manifest, out_mp, args.multipass)
    report["mediapipe_exit_code"] = rc_mp
    report["outputs"]["mediapipe"] = str(out_mp.resolve())
    if rc_mp != 0:
        print(
            json.dumps(
                {
                    "error": "mediapipe_eval_failed",
                    "exit_code": rc_mp,
                    "log_tail": log_mp[-2500:],
                },
                indent=2,
            )
        )
        return 1

    if args.skip_rtmpose:
        print(json.dumps({**report, "note": "skip_rtmpose: compare not run"}, indent=2))
        return 0

    rc_rtm, log_rtm = _run_eval("rtmpose", args.manifest, out_rtm, args.multipass)
    report["rtmpose_exit_code"] = rc_rtm
    report["outputs"]["rtmpose"] = str(out_rtm.resolve())

    if rc_rtm != 0:
        print(
            json.dumps(
                {
                    **report,
                    "warning": "rtmpose_eval_failed_compare_skipped",
                    "rtmpose_log_tail": log_rtm[-2500:],
                    "hint": "pip install -r requirements-pose-optional.txt; first run may need network",
                },
                indent=2,
            )
        )
        return 2

    rc_c, out_c = _run_compare(out_mp, out_rtm, args.per_clip_out)
    report["compare_exit_code"] = rc_c
    if args.per_clip_out is not None:
        report["outputs"]["per_clip_diff"] = str(args.per_clip_out.resolve())
    if rc_c != 0:
        print(json.dumps({"error": "compare_failed", "exit_code": rc_c, "log": out_c[-2000:]}, indent=2))
        return 1

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from app.pose.scorecard_command import suggest_eval_scorecard_header_command

    report["scorecard_header_suggested_command"] = suggest_eval_scorecard_header_command(
        repo_root=REPO_ROOT,
        manifest_path=args.manifest.resolve(),
        jsonl_paths=[out_mp.resolve(), out_rtm.resolve()],
        python_exe=sys.executable,
    )
    report["scorecard_header_note"] = (
        "Run from repo root; paste JSONL hashes into scorecards / PR bundles (scorecard_schema_version 1.1.0)."
    )

    print(out_c.strip())
    print(json.dumps({"orchestration_report": report}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
