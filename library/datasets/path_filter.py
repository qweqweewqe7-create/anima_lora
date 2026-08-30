"""Moved to ``library.captioning.path_filter`` (curation side of the
``anime_tools`` split — the ``path_pattern`` glob semantics are shared by the
training subsets and every curation stage). Re-exported here for trainer
callers; this shim goes away in split Phase 3."""

from library.captioning.path_filter import filter_paths_by_glob  # noqa: F401
