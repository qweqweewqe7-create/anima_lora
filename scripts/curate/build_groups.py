"""Moved to ``anime_tools.grouping.cli.build_groups`` (curation split Phase 2, 2026-08-30).

Forwarding shell kept for one release so ``make …`` targets and
``make daemon-run ARGS="scripts/curate/build_groups.py …"`` keep working; removed in Phase 3.
"""

from library._moved import forward

forward(__name__, "anime_tools.grouping.cli.build_groups", run=__name__ == "__main__")
