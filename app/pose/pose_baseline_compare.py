"""
Pairwise comparison of two pose baseline JSONL runs (P1b — L0 pipeline A vs B).

This compares **reported pipeline outputs** on the same ``clip_id`` keys. It does **not**
establish ground-truth pose accuracy without labeled keyframes (see POSE_UPGRADE_EXECUTION_PLAN §5).
"""
from __future__ import annotations

import json
import logging
import statistics
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MULTIPLE_PEOPLE_REASON = "multiple_people_detected"


def _has_reason_code(row: dict[str, Any], code: str) -> bool:
    codes = row.get("reason_codes")
    if not isinstance(codes, list):
        return False
    return code in codes


def _haar_attempts_from_row(row: dict[str, Any]) -> int | None:
    prov = row.get("provenance")
    if not isinstance(prov, dict):
        return None
    iso = prov.get("person_isolation")
    if not isinstance(iso, dict):
        return None
    v = iso.get("haar_detection_attempts")
    if v is None:
        v = iso.get("redetect_events")
    try:
        return int(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def load_pose_baseline_rows(path: Path) -> dict[str, dict[str, Any]]:
    """
    Load JSONL; index by ``clip_id``. Duplicate ``clip_id`` rows: **last** wins, with a warning.
    """
    rows: dict[str, dict[str, Any]] = {}
    dup = 0
    with path.open(encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_num}: invalid JSON: {e}") from e
            cid = obj.get("clip_id")
            if not cid:
                raise ValueError(f"{path}:{line_num}: missing clip_id")
            cid = str(cid)
            if cid in rows:
                dup += 1
            rows[cid] = obj
    if dup:
        logger.warning("%s: %d duplicate clip_id row(s); kept last occurrence each", path.name, dup)
    return rows


def compare_pose_baseline_rows(
    a: dict[str, dict[str, Any]],
    b: dict[str, dict[str, Any]],
    *,
    label_a: str = "A",
    label_b: str = "B",
) -> dict[str, Any]:
    """
    Build an aggregate comparison for clips present in **either** side.

    Uses intersection for delta statistics only where both ``ok`` are true and
    ``detection_rate`` is present.
    """
    ids = sorted(set(a.keys()) | set(b.keys()))
    both_ok: list[str] = []
    only_a_ok: list[str] = []
    only_b_ok: list[str] = []
    both_fail: list[str] = []
    only_in_a: list[str] = []
    only_in_b: list[str] = []

    deltas: list[float] = []
    usable_a = usable_b = 0
    usable_flip_to_b: list[str] = []
    usable_flip_to_a: list[str] = []
    ffmpeg_mismatch: list[str] = []

    for cid in ids:
        ra = a.get(cid)
        rb = b.get(cid)
        if ra is None:
            only_in_b.append(cid)
            continue
        if rb is None:
            only_in_a.append(cid)
            continue

        oka = bool(ra.get("ok"))
        okb = bool(rb.get("ok"))
        if oka and okb:
            both_ok.append(cid)
            fa = ra.get("ffmpeg_preprocess_applied")
            fb = rb.get("ffmpeg_preprocess_applied")
            if fa is not None and fb is not None and bool(fa) != bool(fb):
                ffmpeg_mismatch.append(cid)
            da = float(ra.get("detection_rate") or 0.0)
            db = float(rb.get("detection_rate") or 0.0)
            deltas.append(db - da)
            ua = bool(ra.get("pose_usable_heuristic"))
            ub = bool(rb.get("pose_usable_heuristic"))
            if ua:
                usable_a += 1
            if ub:
                usable_b += 1
            if not ua and ub:
                usable_flip_to_b.append(cid)
            if ua and not ub:
                usable_flip_to_a.append(cid)
        elif oka and not okb:
            only_a_ok.append(cid)
        elif okb and not oka:
            only_b_ok.append(cid)
        else:
            both_fail.append(cid)

    n_delta = len(deltas)
    mean_a = mean_b = None
    mean_delta = median_delta = min_delta = max_delta = None
    if n_delta:
        mean_a = round(sum(float(a[cid].get("detection_rate") or 0.0) for cid in both_ok) / n_delta, 4)
        mean_b = round(sum(float(b[cid].get("detection_rate") or 0.0) for cid in both_ok) / n_delta, 4)
        mean_delta = round(sum(deltas) / n_delta, 4)
        median_delta = round(float(statistics.median(deltas)), 4)
        min_delta = round(float(min(deltas)), 4)
        max_delta = round(float(max(deltas)), 4)

    confound_notes: list[str] = []
    if ffmpeg_mismatch:
        confound_notes.append(
            f"{len(ffmpeg_mismatch)} clip(s) have differing ffmpeg_preprocess_applied "
            "between runs — detection_rate deltas may mix decode-path effects, not only backbone."
        )

    p2_multi_a = p2_multi_b = 0
    p2_cleared_b_vs_a: list[str] = []
    p2_introduced_b_vs_a: list[str] = []
    for cid in both_ok:
        ra_i = a[cid]
        rb_i = b[cid]
        ma = _has_reason_code(ra_i, MULTIPLE_PEOPLE_REASON)
        mb = _has_reason_code(rb_i, MULTIPLE_PEOPLE_REASON)
        if ma:
            p2_multi_a += 1
        if mb:
            p2_multi_b += 1
        if ma and not mb:
            p2_cleared_b_vs_a.append(cid)
        if not ma and mb:
            p2_introduced_b_vs_a.append(cid)

    return {
        "comparison_purpose": (
            "L0: same-clip pipeline output comparison; not labeled keypoint accuracy"
        ),
        "label_a": label_a,
        "label_b": label_b,
        "clip_ids_total_union": len(ids),
        "clip_ids_only_in_a": only_in_a,
        "clip_ids_only_in_b": only_in_b,
        "clips_both_ok": len(both_ok),
        "clips_only_ok_a": len(only_a_ok),
        "clips_only_ok_b": len(only_b_ok),
        "clips_both_failed": len(both_fail),
        "only_ok_a_clip_ids_sample": only_a_ok[:10],
        "only_ok_b_clip_ids_sample": only_b_ok[:10],
        "both_failed_clip_ids_sample": both_fail[:10],
        "confound_notes": confound_notes,
        "intersection_both_ok": {
            "mean_detection_rate_a": mean_a,
            "mean_detection_rate_b": mean_b,
            "mean_delta_detection_rate_b_minus_a": mean_delta,
            "median_delta_detection_rate_b_minus_a": median_delta,
            "min_delta_detection_rate_b_minus_a": min_delta,
            "max_delta_detection_rate_b_minus_a": max_delta,
            "clips_ffmpeg_preprocess_mismatch": len(ffmpeg_mismatch),
            "clip_ids_ffmpeg_preprocess_mismatch_sample": ffmpeg_mismatch[:15],
            "usable_heuristic_count_a": usable_a,
            "usable_heuristic_count_b": usable_b,
            "usable_gained_b_vs_a_clip_ids_sample": usable_flip_to_b[:15],
            "usable_lost_b_vs_a_clip_ids_sample": usable_flip_to_a[:15],
        },
        "p2_l0": {
            "comparison_note": (
                "L0 P2: counts of multiple_people_detected in reason_codes on clips where both runs "
                "ok=True. cleared_b_vs_a = A had the flag, B did not (e.g. full-frame vs ROI crop)."
            ),
            "clips_both_ok": len(both_ok),
            "multiple_people_detected_count_a": p2_multi_a,
            "multiple_people_detected_count_b": p2_multi_b,
            "n_cleared_multi_person_b_vs_a": len(p2_cleared_b_vs_a),
            "n_introduced_multi_person_b_vs_a": len(p2_introduced_b_vs_a),
            "cleared_multi_person_b_vs_a_clip_ids_sample": p2_cleared_b_vs_a[:20],
            "introduced_multi_person_b_vs_a_clip_ids_sample": p2_introduced_b_vs_a[:20],
        },
    }


def per_clip_diff_rows(
    a: dict[str, dict[str, Any]],
    b: dict[str, dict[str, Any]],
    *,
    label_a: str = "A",
    label_b: str = "B",
) -> list[dict[str, Any]]:
    """One dict per clip_id in union with compact numeric diff (both sides present)."""
    out: list[dict[str, Any]] = []
    for cid in sorted(set(a.keys()) & set(b.keys())):
        ra, rb = a[cid], b[cid]
        fa, fb = ra.get("ffmpeg_preprocess_applied"), rb.get("ffmpeg_preprocess_applied")
        ffmis = (
            fa is not None
            and fb is not None
            and bool(fa) != bool(fb)
        )
        out.append(
            {
                "clip_id": cid,
                "ok_a": bool(ra.get("ok")),
                "ok_b": bool(rb.get("ok")),
                "detection_rate_a": ra.get("detection_rate"),
                "detection_rate_b": rb.get("detection_rate"),
                "delta_detection_rate_b_minus_a": (
                    round(float(rb.get("detection_rate") or 0.0) - float(ra.get("detection_rate") or 0.0), 4)
                    if ra.get("ok") and rb.get("ok")
                    else None
                ),
                "pose_usable_a": ra.get("pose_usable_heuristic"),
                "pose_usable_b": rb.get("pose_usable_heuristic"),
                "ffmpeg_preprocess_a": fa,
                "ffmpeg_preprocess_b": fb,
                "ffmpeg_preprocess_mismatch": ffmis,
                "backend_a": ra.get("backend"),
                "backend_b": rb.get("backend"),
                "max_people_seen_a": ra.get("max_people_seen"),
                "max_people_seen_b": rb.get("max_people_seen"),
                "multiple_people_detected_a": _has_reason_code(ra, MULTIPLE_PEOPLE_REASON),
                "multiple_people_detected_b": _has_reason_code(rb, MULTIPLE_PEOPLE_REASON),
                "haar_detection_attempts_a": _haar_attempts_from_row(ra),
                "haar_detection_attempts_b": _haar_attempts_from_row(rb),
            }
        )
    return out
