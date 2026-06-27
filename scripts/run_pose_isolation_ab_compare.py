#!/usr/bin/env python3
"""
P2 — run the same manifest twice (full frame vs ``--person-isolation haar_mil_v1``) and compare JSONL.

L0 only: same semantics as ``compare_pose_baseline_jsonl.py`` plus ``p2_l0`` multi-person tallies.

Examples:
  python scripts/run_pose_isolation_ab_compare.py --manifest evaluation/gym_manifest.csv
  python scripts/run_pose_isolation_ab_compare.py --manifest evaluation/gym_manifest.csv --backend mediapipe --out-dir evaluation
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_eval(manifest: Path, out: Path, backend: str, multipass: bool, person_isolation: str | None) -> tuple[int, str]:
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
    if person_isolation:
        cmd.extend(["--person-isolation", person_isolation])
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
        "full_frame",
        "--label-b",
        "haar_mil_v1",
    ]
    if per_clip is not None:
        cmd.extend(["--per-clip-out", str(per_clip)])
    r = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="P2: MediaPipe (or RTMPose) full frame vs person-isolation JSONL + L0 compare."
    )
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
    ap.add_argument("--backend", type=str, default="mediapipe", help="mediapipe | rtmpose")
    ap.add_argument("--multipass", action="store_true")
    ap.add_argument(
        "--per-clip-out",
        type=Path,
        default=None,
        help="Optional per-clip diff JSONL (intersection clips)",
    )
    ap.add_argument(
        "--isolation-mode",
        type=str,
        default="haar_mil_v1",
        help="Passed to eval_pose_baseline --person-isolation for run B",
    )
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.manifest.stem
    out_full = args.out_dir / f"{stem}_pose_full.jsonl"
    out_iso = args.out_dir / f"{stem}_pose_{args.isolation_mode}.jsonl"

    report: dict = {
        "manifest": str(args.manifest.resolve()),
        "backend": args.backend,
        "outputs": {"full_frame": str(out_full.resolve()), "isolation": str(out_iso.resolve())},
    }

    rc_f, log_f = _run_eval(args.manifest, out_full, args.backend, args.multipass, person_isolation=None)
    report["full_frame_exit_code"] = rc_f
    if rc_f != 0:
        print(
            json.dumps(
                {"error": "eval_full_frame_failed", "exit_code": rc_f, "log_tail": log_f[-3000:]},
                indent=2,
            )
        )
        return 1

    rc_i, log_i = _run_eval(
        args.manifest, out_iso, args.backend, args.multipass, person_isolation=args.isolation_mode
    )
    report["isolation_exit_code"] = rc_i
    if rc_i != 0:
        print(
            json.dumps(
                {
                    "error": "eval_isolation_failed",
                    "exit_code": rc_i,
                    "log_tail": log_i[-3000:],
                },
                indent=2,
            )
        )
        return 1

    rc_c, out_c = _run_compare(out_full, out_iso, args.per_clip_out)
    report["compare_exit_code"] = rc_c
    if args.per_clip_out is not None:
        report["outputs"]["per_clip_diff"] = str(args.per_clip_out.resolve())

    if rc_c != 0:
        print(json.dumps({"error": "compare_failed", "exit_code": rc_c, "log": out_c[-2500:]}, indent=2))
        return 1

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from app.pose.scorecard_command import suggest_eval_scorecard_header_command

    report["scorecard_header_suggested_command"] = suggest_eval_scorecard_header_command(
        repo_root=REPO_ROOT,
        manifest_path=args.manifest.resolve(),
        jsonl_paths=[out_full.resolve(), out_iso.resolve()],
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
