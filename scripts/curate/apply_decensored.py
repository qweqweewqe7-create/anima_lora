"""Moved to ``anime_tools.grouping.cli.apply_decensored`` (curation split Phase 2, 2026-08-30).

Forwarding shell kept for one release so ``make …`` targets and
``make daemon-run ARGS="scripts/curate/apply_decensored.py …"`` keep working; removed in Phase 3.
"""

from library._moved import forward

forward(
    __name__, "anime_tools.grouping.cli.apply_decensored", run=__name__ == "__main__"
)
