"""Moved to ``anime_tools._hf`` (curation side of the ``anime_tools``
split — the tagger's gated-backbone fetch needs it and must not import the
trainer). Re-exported here for trainer callers; shim goes away in Phase 3."""

from anime_tools._hf import (  # noqa: F401
    ensure_hf_timeouts,
    hf_download,
    hf_file_cached,
)
