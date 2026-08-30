"""Vendored Perception Encoder (PE) vision tower — now owned by
``anime_tools.vision.pe`` (curation split Phase 2, 2026-08-30) so the package's
PE-Spatial grouping embedder works standalone. The trainer keeps importing it
from here (REPA, CMMD, the PE feature cache, IP-Adapter bench); this module is
a permanent re-export, not a deprecation shim. Same module object, so
monkeypatching through either path hits the real thing.
"""

from __future__ import annotations

import importlib
import sys

sys.modules[__name__] = importlib.import_module("anime_tools.vision.pe")
