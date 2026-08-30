"""Moved to ``anime_tools.captions.index`` (curation split Phase 1, 2026-08-30).

Forwarding shell kept for one release so ``make …`` targets and
``make daemon-run ARGS="scripts/preprocess/build_caption_index.py …"`` keep working; removed in Phase 3.
"""

from library._moved import forward

forward(__name__, "anime_tools.captions.index", run=__name__ == "__main__")
