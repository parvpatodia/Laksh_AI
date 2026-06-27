"""Object-storage helpers.

Currently wraps Cloudflare R2 (S3-compatible).  All public surface uses
signed URLs so raw clips are never directly reachable without a time-limited
token.
"""
