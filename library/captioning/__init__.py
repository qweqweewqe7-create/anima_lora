"""Captioning — MOVED to ``anime_tools`` (curation split Phase 1, 2026-08-30).

Every submodule here is a forwarding shim onto its new home
(``anime_tools.captions.*`` / ``anime_tools.tagger.*`` / ``anime_tools.stages.captions``);
``AnimaTagger`` stays reachable lazily for one release. Import ``anime_tools``
directly in new code — the shims go away in Phase 3.
"""

__all__ = ["AnimaTagger"]


def __getattr__(name: str):
    if name == "AnimaTagger":
        from anime_tools.tagger.tagger import AnimaTagger

        return AnimaTagger
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
