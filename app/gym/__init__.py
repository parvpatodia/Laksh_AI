"""Gym vertical: exercise taxonomy and rep-level analysis scaffolding.

Separate package from :mod:`app.pose` because exercises are a product-level
taxonomy, not a pose-estimation concern. Pose modules consume exercise
metadata (e.g. ``rep_signal_joint``) but do not own it.

See ``GOALS.md`` Milestone 1 and ``docs/product-grade_laksh_roadmap_*.plan.md``
Phase D for scope.
"""
