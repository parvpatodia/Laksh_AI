"""
Stable API response contract for clients and integration tests.

Bump when adding/removing top-level JSON fields on primary routes (breaking change).
Patch-level changes (new optional fields) may stay on the same minor version per team policy.
"""
from __future__ import annotations

# Analyze-video and related JSON responses include this key for client pinning.
API_SCHEMA_VERSION = "1.0.0"
