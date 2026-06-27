"""Cloudflare R2 client (S3-compatible, boto3 backend).

All interaction with R2 goes through this module so the rest of the app
never imports boto3 directly.  The client is built lazily on first use and
cached for the process lifetime -- boto3 clients are thread-safe for reads
and the single-worker Fly machine means there is no concurrent writer
contention.

Environment variables (set via ``fly secrets set`` on the server, or
``.env.local`` locally):

.. code-block:: text

    R2_ACCOUNT_ID   - Cloudflare account ID (hex string)
    R2_ACCESS_KEY   - R2 API token access key
    R2_SECRET       - R2 API token secret
    R2_BUCKET       - Bucket name, e.g. "laksh-clips"
    R2_PUBLIC_BASE  - Optional public URL prefix for the bucket
                      (set if the bucket has a custom domain or Workers route).
                      When absent, only signed URLs are used.

Key layout::

    clips/{sha256_of_raw_bytes}/raw.webm   - raw clip
    clips/{sha256_of_raw_bytes}/result.json - canonical analysis result
    clips/{sha256_of_raw_bytes}/meta.json  - {sport_id, exercise_id, created_at}

Lifecycle: a Cloudflare R2 lifecycle rule deletes ``clips/*`` after 7 days.
No PII is stored -- the sha256 is the only identifier.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any

import boto3
from botocore.client import Config

#: Signed-URL TTL in seconds (24 h).
SIGNED_URL_TTL: int = 86_400

#: Maximum clip size enforced server-side (50 MB).
MAX_CLIP_BYTES: int = 50 * 1024 * 1024


# ---------------------------------------------------------------------------
# Client factory
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _get_client() -> "boto3.client":  # type: ignore[name-defined]
    """Return a cached boto3 S3 client pointed at Cloudflare R2.

    Raises ``RuntimeError`` if required env vars are absent so the error
    surfaces at first use (import time is too early; env may not be loaded).
    """
    account_id = _require_env("R2_ACCOUNT_ID")
    access_key = _require_env("R2_ACCESS_KEY")
    secret = _require_env("R2_SECRET")

    endpoint = f"https://{account_id}.r2.cloudflarestorage.com"
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret,
        config=Config(signature_version="s3v4"),
        region_name="auto",  # R2 ignores region but boto3 requires a value
    )


def _require_env(key: str) -> str:
    """Return env var or raise ``RuntimeError``."""
    val = os.environ.get(key)
    if not val:
        raise RuntimeError(
            f"Missing required environment variable: {key}. "
            "Set it via `fly secrets set` or in `.env.local`."
        )
    return val


def _bucket() -> str:
    """Return the configured R2 bucket name."""
    return _require_env("R2_BUCKET")


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def sha256_hex(data: bytes) -> str:
    """Return lowercase hex SHA-256 of *data*."""
    return hashlib.sha256(data).hexdigest()


def clip_prefix(sha: str) -> str:
    """Return the R2 key prefix for a clip identified by *sha*."""
    return f"clips/{sha}"


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------


def upload_clip(raw_bytes: bytes) -> str:
    """Upload a raw WebM clip to R2 and return its SHA-256 hex.

    Parameters
    ----------
    raw_bytes:
        Raw video bytes from MediaRecorder.  Must be <= :data:`MAX_CLIP_BYTES`.

    Returns
    -------
    str
        SHA-256 hex of *raw_bytes*, also the clip's permanent identifier.

    Raises
    ------
    ValueError
        If *raw_bytes* exceeds :data:`MAX_CLIP_BYTES`.
    """
    if len(raw_bytes) > MAX_CLIP_BYTES:
        raise ValueError(
            f"Clip too large: {len(raw_bytes)} bytes > {MAX_CLIP_BYTES} limit"
        )
    sha = sha256_hex(raw_bytes)
    key = f"{clip_prefix(sha)}/raw.webm"
    _get_client().put_object(
        Bucket=_bucket(),
        Key=key,
        Body=raw_bytes,
        ContentType="video/webm",
    )
    return sha


def upload_result(sha: str, result: dict[str, Any]) -> None:
    """Upload the canonical analysis result JSON for *sha*.

    Parameters
    ----------
    sha:
        Clip identifier (returned by :func:`upload_clip`).
    result:
        Dict matching ``AnalyzeResponseModel`` (will be JSON-serialised).
    """
    key = f"{clip_prefix(sha)}/result.json"
    body = json.dumps(result, ensure_ascii=False, indent=2).encode()
    _get_client().put_object(
        Bucket=_bucket(),
        Key=key,
        Body=body,
        ContentType="application/json",
    )


def upload_meta(sha: str, sport_id: str, exercise_id: str) -> None:
    """Upload a lightweight metadata JSON for *sha*.

    Stores ``{sport_id, exercise_id, created_at}`` so lifecycle rules and
    audit scripts can inspect clips without downloading the full result.
    """
    key = f"{clip_prefix(sha)}/meta.json"
    meta = {
        "sha": sha,
        "sport_id": sport_id,
        "exercise_id": exercise_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    body = json.dumps(meta, ensure_ascii=False).encode()
    _get_client().put_object(
        Bucket=_bucket(),
        Key=key,
        Body=body,
        ContentType="application/json",
    )


# ---------------------------------------------------------------------------
# Download / signed URLs
# ---------------------------------------------------------------------------


def signed_url(sha: str, object_type: str = "raw.webm") -> str:
    """Return a 24-hour signed GET URL for *sha*/*object_type*.

    Parameters
    ----------
    sha:
        Clip identifier.
    object_type:
        One of ``"raw.webm"``, ``"result.json"``, ``"meta.json"``.

    Returns
    -------
    str
        HTTPS signed URL valid for :data:`SIGNED_URL_TTL` seconds.
    """
    key = f"{clip_prefix(sha)}/{object_type}"
    return _get_client().generate_presigned_url(
        "get_object",
        Params={"Bucket": _bucket(), "Key": key},
        ExpiresIn=SIGNED_URL_TTL,
    )


def download_result(sha: str) -> dict[str, Any]:
    """Download and parse the canonical result JSON for *sha*.

    Returns
    -------
    dict
        Parsed result, or raises ``KeyError`` if the key does not exist.
    """
    key = f"{clip_prefix(sha)}/result.json"
    response = _get_client().get_object(Bucket=_bucket(), Key=key)
    return json.loads(response["Body"].read())


def clip_exists(sha: str) -> bool:
    """Return True if a raw clip for *sha* exists in R2."""
    import botocore.exceptions

    key = f"{clip_prefix(sha)}/raw.webm"
    try:
        _get_client().head_object(Bucket=_bucket(), Key=key)
        return True
    except botocore.exceptions.ClientError as exc:
        if exc.response["Error"]["Code"] in ("404", "NoSuchKey"):
            return False
        raise
