# Anima Tagger — moved to `anime_tools`

This document moved with the code in the curation split (Phase 1, 2026-08-30):
**https://github.com/sorryhyun/anime_tools/blob/main/docs/anima_tagger.md**
(sibling checkout: `../anime_tools/docs/anima_tagger.md`).

The tagger itself lives at `anime_tools.tagger` (`AnimaTagger`, dbv4 backend, sidecar head); the trainer reaches it via `anima_lora.captioning.AnimaTagger` (the `library.captioning` shim was deleted in Phase 3, 2026-08-30). `make tagger*` / `make autotag` forward to `python -m anime_tools.tagger.cli…`.
