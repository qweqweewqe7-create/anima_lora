"""Moved to ``anime_tools.masking.cli.probe_nms_pairs`` (curation split Phase 2, 2026-08-30).

Forwarding shell kept for one release so ``make …`` targets and
``make daemon-run ARGS="scripts/preprocess/probe_nms_pairs.py …"`` keep working; removed in Phase 3.
"""

from library._moved import forward

forward(__name__, "anime_tools.masking.cli.probe_nms_pairs", run=__name__ == "__main__")
