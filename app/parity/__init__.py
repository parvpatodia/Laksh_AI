"""Realtime-vs-canonical parity analysis.

The parity probe is the live analog of the offline ADR 0002 canonical gate.
When a judge performs a movement in the browser:

1. The browser computes *realtime_preview* ghost metrics from MediaPipe
   landmark coordinates in JavaScript.
2. The server runs the *canonical_backend* pipeline on the same captured
   frames (heavy pose model + full Python analyzer).
3. :func:`~app.parity.realtime.compare_feature_vectors` computes the
   per-field absolute delta between the two vectors and returns a
   ``parity_probe`` block embedded in the v1 response envelope.

This makes the numerical agreement between the real-time UX and the
authoritative analysis explicit and auditable by judges.
"""
