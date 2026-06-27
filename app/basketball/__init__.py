"""Basketball-specific analysis helpers.

Submodules
----------
* :mod:`app.basketball.shot_segmenter` — multi-signal consensus release detection
  that returns 0..N :class:`ShotSpan`s for a clip, each confirmed by >=2
  independent signals per the design in plan `warm-swimming-acorn.md` A1.
"""
