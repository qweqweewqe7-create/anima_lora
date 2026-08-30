"""Import-path forwarding for modules moved to ``anime_tools`` (curation split
Phase 1, 2026-08-30 — ``docs/proposal/curation_repo_split.md``). Every shim is
deleted in Phase 3; new code imports ``anime_tools.*`` directly."""

from __future__ import annotations

import importlib
import os
import sys
import warnings
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]


def alias(old: str, new: str) -> None:
    """Make ``import old`` resolve to the ``new`` module object (same object, so
    monkeypatching through either path hits the real thing)."""
    warnings.warn(
        f"{old} moved to {new} (anime_tools split); this shim goes away in Phase 3",
        DeprecationWarning,
        stacklevel=3,
    )
    sys.modules[old] = importlib.import_module(new)


def forward(old: str, new: str, *, run: bool) -> None:
    """Script shell: pin the curation home to this checkout (so bare relative
    defaults such as ``models/…`` keep resolving here rather than the CWD), then
    either run the moved ``main()`` (``python scripts/…`` / ``-m``) or alias the
    module for importers."""
    os.environ.setdefault("ANIMA_HOME", str(_REPO_ROOT))
    if run:
        mod = importlib.import_module(new)
        sys.exit(mod.main())
    alias(old, new)
