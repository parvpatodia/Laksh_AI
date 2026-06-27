"""You.com-grounded coaching.

Coaching FAULTS are detected deterministically from measured reps (the rule +
evidence are shown). This package GROUNDS the remediation for each fault in real
sources retrieved via the You.com Search API, attaching resolvable citations.
It is gated by ``YOU_API_KEY`` (+ optional ``LAKSH_COACHING_GROUNDING=off``);
without a key it degrades to the deterministic cue with no citations, so the
live demo never breaks.
"""
