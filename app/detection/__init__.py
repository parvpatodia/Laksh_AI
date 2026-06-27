"""Optional detection models (Track B: ball-leaves-hand S3 signal).

All code in this sub-package is gated by LAKSH_ENABLE_BALL_DETECT=1.
When the env var is absent the package can be imported safely; inference
methods return None / empty lists so callers need no conditional branches.
"""
