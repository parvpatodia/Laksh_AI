"""
Static checks for gym pose eval (no video decode, no model inference).

Use before long A/B runs to see missing deps, absent ``.task`` file, or manifest issues.

Native extensions (OpenCV, MediaPipe, ONNX Runtime, rtmlib) are probed in a **subprocess**
so a bad binary or sandboxed environment cannot take down the reporting process (e.g.
onnxruntime has been observed to SIGSEGV under restricted sandboxes while the parent
stays healthy if the import runs in a child).
"""
from __future__ import annotations

import hashlib
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

# Modules backed by native code — always checked via subprocess when ``find_spec`` hits.
_HEAVY_MODULES = frozenset({"cv2", "mediapipe", "onnxruntime", "rtmlib"})

_NOT_INSTALLED = (
    "module not found for this interpreter — install project requirements into the same "
    "Python you use for `make` / scripts (see Makefile `PYTHON ?= python3`)."
)


def _dependency_block(import_ok: bool, version_or_failure: str | None) -> dict[str, Any]:
    """
    JSON block for one optional dependency.

    On failure, ``version`` is always null and diagnostics go in ``probe_error`` so we never
    overload ``version`` with exception text or signal messages.
    """
    if import_ok:
        return {
            "import_ok": True,
            "version": version_or_failure or "unknown",
            "probe_error": None,
        }
    if version_or_failure is None:
        return {"import_ok": False, "version": None, "probe_error": _NOT_INSTALLED}
    return {"import_ok": False, "version": None, "probe_error": version_or_failure}


def _import_ok(name: str) -> tuple[bool, str | None]:
    spec = importlib.util.find_spec(name)
    if spec is None:
        return False, None
    if name in _HEAVY_MODULES:
        return _import_ok_subprocess(name)
    try:
        mod = importlib.import_module(name)
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"
    ver = getattr(mod, "__version__", None)
    return True, str(ver) if ver is not None else "unknown"


def _import_ok_subprocess(name: str) -> tuple[bool, str | None]:
    """Import ``name`` in a child process; survive extension crashes in the child."""
    code = (
        "import importlib,sys\n"
        f"try:\n"
        f"    m = importlib.import_module({name!r})\n"
        f"    v = getattr(m, '__version__', None)\n"
        f"    sys.stdout.write(v if v is not None else 'unknown')\n"
        f"except Exception as e:\n"
        f"    sys.stderr.write(f'{{type(e).__name__}}: {{e}}')\n"
        f"    sys.exit(1)\n"
    )
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=90,
        )
    except subprocess.TimeoutExpired:
        return False, "TimeoutExpired: import probe exceeded 90s"
    except OSError as e:
        return False, f"{type(e).__name__}: {e}"

    if proc.returncode == 0:
        out = (proc.stdout or "").strip()
        return True, out if out else "unknown"

    err = (proc.stderr or "").strip() or (proc.stdout or "").strip()
    if proc.returncode < 0:
        sig = -proc.returncode
        hint = f"child killed by signal {sig}" + (f" ({err})" if err else "")
        return False, hint
    return False, err or f"exit {proc.returncode}"


def collect_eval_readiness(
    *,
    manifest_path: Path | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """
    Build a JSON-serializable readiness report.

    ``mediapipe_gym_eval_minimal`` is True when OpenCV + MediaPipe import and the heavy
    ``.task`` file exist (same gate as actually running the landmarker). FFmpeg is
    **recommended** for comparable metrics but not required to start a run.
    """
    root = repo_root or Path(__file__).resolve().parent.parent.parent

    ffmpeg = shutil.which("ffmpeg") is not None
    ok_cv2, cv2_ver = _import_ok("cv2")
    ok_mp, mp_ver = _import_ok("mediapipe")

    from app.pose.expected_artifacts import POSE_LANDMARKER_HEAVY_TASK_SHA256
    from app.pose.mediapipe_common import default_model_path

    task_path = default_model_path()
    task_present = task_path.is_file()
    task_sha256: str | None = None
    task_sha256_matches_expected: bool | None = None
    if task_present:
        h = hashlib.sha256()
        with open(task_path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        task_sha256 = h.hexdigest()
        task_sha256_matches_expected = task_sha256 == POSE_LANDMARKER_HEAVY_TASK_SHA256

    ok_ort, ort_ver = _import_ok("onnxruntime")
    ok_rtm, rtm_ver = _import_ok("rtmlib")

    notes: list[str] = []
    if not ffmpeg:
        notes.append(
            "FFmpeg not on PATH — runs use source decode; metrics may differ from H.264-normalized CI."
        )
    if not task_present:
        notes.append(
            f"Missing pose task file at {task_path.name} — run: "
            f"python3 scripts/download_pose_model.py (same interpreter as Makefile PYTHON)"
        )
    elif task_sha256_matches_expected is False:
        notes.append(
            "pose_landmarker_heavy.task SHA-256 does not match expected — "
            "re-download with scripts/download_pose_model.py or update "
            "app/pose/expected_artifacts.py if MediaPipe rotated the blob."
        )
    if ok_rtm and ok_ort:
        notes.append(
            "rtmlib + onnxruntime present — RTMPose path may download ONNX zips on first run (network)."
        )

    manifest_block: dict[str, Any] | None = None
    if manifest_path is not None:
        mp = Path(manifest_path)
        if not mp.is_file():
            manifest_block = {
                "path": str(mp.resolve()),
                "load_ok": False,
                "error": "manifest file not found",
            }
        else:
            try:
                from app.pose.gym_manifest import load_gym_manifest, summarize_manifest_path_status

                jobs = load_gym_manifest(mp, root)
                stat = summarize_manifest_path_status(jobs)
                manifest_block = {
                    "path": str(mp.resolve()),
                    "load_ok": True,
                    "clips_in_manifest": stat["clips_in_manifest"],
                    "files_present": stat["files_present"],
                    "files_missing": stat["files_missing"],
                    "missing_paths_sample": stat["missing_paths_sample"],
                }
            except ValueError as e:
                manifest_block = {
                    "path": str(mp.resolve()),
                    "load_ok": False,
                    "error": str(e),
                }

    # Require matching SHA when file exists (wrong blob => not minimal)
    minimal_mp = bool(
        ok_cv2
        and ok_mp
        and task_present
        and (task_sha256_matches_expected is True)
    )

    report: dict[str, Any] = {
        "report_purpose": "static_readiness_no_inference",
        "report_schema_version": "1.2.0",
        "interpreter": sys.executable,
        "repo_root": str(root.resolve()),
        "ffmpeg_on_path": ffmpeg,
        "opencv": _dependency_block(ok_cv2, cv2_ver),
        "mediapipe": _dependency_block(ok_mp, mp_ver),
        "pose_landmarker_task": {
            "path": str(task_path.resolve()),
            "present": task_present,
            "expected_sha256": POSE_LANDMARKER_HEAVY_TASK_SHA256,
            "sha256": task_sha256,
            "sha256_matches_expected": task_sha256_matches_expected,
        },
        "onnxruntime": _dependency_block(ok_ort, ort_ver),
        "rtmlib": _dependency_block(ok_rtm, rtm_ver),
        "mediapipe_gym_eval_minimal": minimal_mp,
        "rtmpose_stack_imports_ok": bool(ok_rtm and ok_ort and ok_cv2),
        "notes": notes,
    }
    if manifest_block is not None:
        report["gym_manifest"] = manifest_block

    return report
