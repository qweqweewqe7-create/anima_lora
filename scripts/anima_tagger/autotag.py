"""Moved to ``anime_tools.tagger.cli.autotag`` (curation split Phase 1, 2026-08-30).

Forwarding shell kept for one release so ``make …`` targets and
``make daemon-run ARGS="scripts/anima_tagger/autotag.py …"`` keep working; removed in Phase 3.
"""

from library._moved import forward

forward(__name__, "anime_tools.tagger.cli.autotag", run=__name__ == "__main__")
