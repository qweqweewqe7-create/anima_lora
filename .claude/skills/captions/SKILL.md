---
name: captions
description: Caption pipeline — position-clause grammar (never hand-split a caption), make caption-autotag modes, make caption-position (v2 rewrite rules and gates), and the preprocess-stage wiring for both. Load before parsing/editing captions or caption code, running either target, or touching the caption preprocess stages.
---

# Caption pipeline (trainer-side wiring)

The caption code moved to the **`anime_tools`** package (curation split Phase 1,
2026-08-30 — https://github.com/sorryhyun/anime_tools, sibling checkout
`../anime_tools`). The full skill — grammar details, `--caption_drop_groups`
resolution order, autotag modes, the v2 position-clause move rules and gates,
the tuning defaults — lives there: `../anime_tools/.claude/skills/captions/SKILL.md`,
evidence in `../anime_tools/docs/position_captions.md`. **Read it before
editing caption code.** What stays trainer-side is below.

## The one rule

`<flat tag bag>. On the left, akita neru, yellow eyes. On the right, kasane teto.`
— the **period** delimits clauses, commas separate tags *inside* one. A plain
`caption.split(",")` silently corrupts clauses. **Never hand-split a caption**:
`anime_tools.captions.position_clauses` (`parse_caption` / `compose_caption`)
is the single grammar; `anime_tools.captions.shuffle` is the training-time
shuffle / `@no-artist` grammar (`library.anima.training` re-exports it).

## Trainer targets (all forward to `anime_tools` CLIs)

| Target | Package entry (`python -m …`; `run()` exports `ANIMA_HOME`) | Notes |
|---|---|---|
| `make caption-autotag` | `anime_tools.stages.cli.autotag_captions` | dry-run default; `--mode missing\|merge\|overwrite`; `ARGS="--apply"` then **`make preprocess-te`** |
| `make caption-position` | `anime_tools.stages.cli.position_captions` | SAM3 → tagger → v2 rewrite; dry-run default, GPU — route through the daemon |
| `make preprocess-captions` | `anime_tools.stages.cli.correct_captions` | corrected mirror + `.variants.txt` under `post_image_dataset/resized/`; `--caption_drop_groups` |
| `make caption-index` | `anime_tools.captions.index` | `post_image_dataset/captions/caption_index.json` |
| `make autotag` / `make tagger*` | `anime_tools.tagger.cli.*` | single-image / vocab build / dbv4 ckpt |

Stage wiring (`scripts/tasks/preprocess.py`): autotag runs **first** (right after
resize, `--apply`), then position clauses, then correction/variants, then TE —
chain order pinned by `tests/test_preprocess_tasks.py`. Caption edits do **not**
invalidate TE caches (existence-only skip) — always re-run `make preprocess-te`
after an `--apply`; a stale `.variants.txt` keeps training the old caption.

`configs/clause_vocabulary.yaml` is the user-editable clause policy; the package
ships an identical default used when the file is absent from the curation home.
