"""Moved to ``anime_tools.stages.cli.audit_apply_curated`` (curation split Phase 1, 2026-08-30).

Forwarding shell kept for one release so ``make …`` targets and
``make daemon-run ARGS="scripts/preprocess/audit_apply_curated.py …"`` keep working; removed in Phase 3.
"""

from library._moved import forward

forward(
    __name__, "anime_tools.stages.cli.audit_apply_curated", run=__name__ == "__main__"
)
