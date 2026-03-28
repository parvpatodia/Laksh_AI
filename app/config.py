"""
Centralised application configuration via Pydantic Settings.

Previously every module called os.environ.get() independently with
inconsistent defaults. This is now the single source for all env-driven
config — validated at startup, typed, and documented.

Usage:
    from app.config import settings
    settings.gemini_api_key  # str
    settings.max_upload_bytes  # int
"""
from __future__ import annotations

import os
from pathlib import Path
from functools import lru_cache

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Default CORS origins — override in production via CORS_ORIGINS env var
_DEFAULT_CORS_ORIGINS = [
    "https://lakshai-production.up.railway.app",
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:8000",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:8000",
]


class Settings:
    """
    Application settings loaded from environment variables.
    All fields have documented defaults so the server can start locally
    without a .env file (except gemini_api_key, which is required for analysis).
    """

    def __init__(self) -> None:
        # --- Required ---
        self.gemini_api_key: str = os.environ.get("GEMINI_API_KEY", "")

        # --- Logging ---
        self.log_level: str = os.environ.get("LOG_LEVEL", "INFO").strip().upper()

        # --- CORS ---
        cors_raw = os.environ.get("CORS_ORIGINS", "")
        if cors_raw.strip():
            self.cors_origins: list[str] = [o.strip() for o in cors_raw.split(",") if o.strip()]
        else:
            self.cors_origins = list(_DEFAULT_CORS_ORIGINS)

        # --- Upload limits ---
        # 200 MB default — large enough for phone video, small enough to avoid abuse
        self.max_upload_bytes: int = int(os.environ.get("MAX_UPLOAD_BYTES", str(200 * 1024 * 1024)))

        # --- ChromaDB ---
        # Default: <repo_root>/chroma_db. Override on read-only filesystems (Railway, etc.)
        default_chroma = str(_REPO_ROOT / "chroma_db")
        self.chroma_persist_dir: str = os.environ.get("CHROMA_PERSIST_DIR", default_chroma)

        # --- NBA API ---
        self.nba_api_timeout: int = int(os.environ.get("NBA_API_TIMEOUT", "90"))
        self.nba_api_retries: int = int(os.environ.get("NBA_API_RETRIES", "2"))
        self.nba_api_delay_sec: float = float(os.environ.get("NBA_API_DELAY_SEC", "0.6"))

    def validate(self) -> None:
        """
        Assert required config is present. Called once in lifespan startup so
        the server fails loudly at boot rather than silently on first request.
        """
        if not self.gemini_api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. "
                "Set it in .env or as an environment variable before starting the server."
            )
        if self.max_upload_bytes < 1024 * 1024:
            raise RuntimeError(
                f"MAX_UPLOAD_BYTES={self.max_upload_bytes} is unreasonably small (< 1 MB)."
            )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the singleton Settings instance (cached after first call)."""
    s = Settings()
    return s


# Module-level singleton — import this in all consumers
settings = get_settings()
