# Curation split — tagger / masking / grouping / caption polishing → `sorryhyun/anime_tools`

Status: **Phases 0–3 complete (2026-08-30); Phase 3b (bench/docs relocation) planned.**
`github.com/sorryhyun/anime_tools` (package `anime_tools`, git dependency — no PyPI) owns the
caption grammar + polishing stages, the Anima Tagger, masking (SAM3 / MIT / merge), grouping
(PE-Spatial near-twin features → `groups.json`) and the vendored PE vision tower. The trainer
depends on it via the default-on `anime-tools-git` uv group (pinned rev in
`[tool.uv.sources]`; `uv sync --no-group anime-tools-git --group anime-tools-dev` for a live
`../anime_tools` loop). The old trainer import paths and forwarding shells were deleted in
Phase 3 — `make` targets invoke `python -m anime_tools.…`. Contract: `../anime_tools/docs/contract.md`.
The per-phase logs, target layout and move manifest were removed from this file once done —
see the trainer commits `bc427186`…`874b34d9` and the package history.

Decisions that changed during execution (recorded so they aren't re-litigated):
- **PE-Spatial is owned by `anime_tools`** (`anime_tools/vision/pe.py`; reverses the Phase-0
  "PE stays in the trainer" seam) so a standalone install groups with the same embedder. The
  trainer still needs the tower for REPA / CMMD / PE caching → `library/models/pe.py` is a
  **permanent** re-export, *not* a Phase-3 shim; the `library/vision/{encoder,encoders,buckets}`
  registry stays trainer-side. Embedder dtype pinned to bf16 (the pre-split default) so the
  `$NEAR_TWIN_CACHE` stays valid.
- **GUI panels stay in the trainer** (no `gui` extra). They reach the package only through
  the `autotag_server` stdio protocol, daemon jobs and the torch-free grammar; there is no
  standalone curation GUI (non-goal), so moving Qt + i18n bought nothing. MCP was considered
  and rejected — it would replace a process boundary the daemon bridge already covers.
- Both GitHub repos are **private until the trainer branch merges** — flip with
  `gh repo edit sorryhyun/anime_tools --visibility public --accept-visibility-change-consequences`
  (same for `ComfyUI-Anima-Tagger`) first, or end users' `uv sync` fails on the git dep.

## The one rule: dependency direction

```
anima_lora (trainer)  ──depends on──▶  anime_tools            never the reverse
```

The trainer may import from the curation package. The curation package must **never**
import `library.anima`, `library.models`, `library.training`, `networks`, or `train.py`.
If a curation feature needs something from the trainer, the fix is to move that leaf into
the curation package (or duplicate a ≤50-line pure helper), not to import it.

This is the shape the code already mostly has — the audit below found **zero** imports from
the DiT/VAE/adapter side inside the candidate set, and only ~10 trainer-side sites importing
*into* it.

### The PE-Core seam (the one real design decision)

The live tagger is **dbv4** (`animetimm/caformer_b36.dbv4-full` via `timm.create_model`,
default checkpoint since 2026-08-27); the sidecar head reads dbv4's hidden feature. The
`load_pe_encoder` imports in `anima_tagger.py` are reached only by the legacy `"pe"`
backend. So the tagger does **not** need PE-Core.

The PE-Core / PE-Spatial loader (`library/vision/{encoder,encoders,buckets}.py`) is used by
REPA, validation CMMD, `preprocess/pe.py` (FEI feature cache), the FeRA/Hydra router
(`networks/lora_anima/factory.py`) — all trainer — and, on the curation side, only by
**grouping** (`grouping.py:221`, PE-Spatial) and the near-twins tool (already trainer-side in
`easycontrol_adapters/`). (`vision/buckets.py` is the PE patch-bucket spec, *not* the free-fit
training bucketing in `library/datasets/buckets.py` — no overlap.)

Decision: **PE stays in the trainer.** Grouping takes its embedder as an injected callable
(`embed(images) -> features`, plus the `pe_features` cache layout which moves with grouping
since it is grouping's own cache, not the trainer's `{stem}_anima_pe.safetensors`). The
trainer keeps a thin `make curate-group` that wires PE-Spatial in; a standalone
`anime-tools group` CLI accepts any embedder (dbv4 backbone features are a reasonable
default so the package works without the trainer installed — to be benched, not assumed
equivalent). The legacy `"pe"` tagger backend is **not** moved: it is dropped from the
package (checkpoint format stays readable by a trainer-side shim for one release).

Result: the curation package's base install shares **no module** with the trainer —
`timm + transformers + safetensors + PIL + numpy`. The earlier idea of making the PE loader
the package's shared core is withdrawn.

## Phase 3 — delete shims (done 2026-08-30, same day as Phase 2)

Done ahead of the "one release later" schedule because the trainer branch is unmerged and
nothing outside it ever installed the shims. Removed `library/_moved.py` and every shim/shell
(`library/captioning/*`, the `library/preprocess` / `library/vision` / `library/datasets`
aliases, `scripts/anima_tagger/`, `scripts/curate/`, the `scripts/preprocess/*` caption / mask
/ probe / audit shells); `scripts/tasks/{masking,curate,preprocess}.py` and
`easycontrol_adapters/region/prep.py` invoke `python -m anime_tools.…` (daemon `--queue`
routing unchanged — `anima_daemon.cli._label_for` already labels `-m` jobs);
`tests/test_curation_boundary.py` shrank to the task wrappers; CLAUDE.md §Curation rewritten.
Kept: `library/models/pe.py` (permanent re-export), the `library/vision` registry,
`anima_lora.captioning.AnimaTagger` (façade, reads `anime_tools.tagger`). Deps: `sam3` stays
direct (`bench/position_captions/probe_autocaption.py`), `albumentations` stays
(`project/finished/sr`), `segmentation-models-pytorch` dropped (rides on `anime-tools[masking]`).
Package rev bumped to `a3b3a4a`. Still open from the original list: the Windows offline
tarball (GH #92) must vendor a checkout of the pinned `anime_tools` rev
(`filter="data"`, no symlinks) — a release-process item, tracked in
[[project_gh92_windows_backend_default_group]].

## Phase 3b — relocate tagger-only bench / docs / tests (audit 2026-08-30)

Everything below imports only `anime_tools.*` (+ stdlib/torch) and is curation-shaped, so it
moves to the package under the same one rule. Everything *not* listed stays on purpose.

**Move:**
- `bench/tagger_external/{calibration_check,probe_position_rescore}.py` + `README.md` +
  `results/` → `../anime_tools/bench/tagger_external/`. The package has no `bench/` yet —
  add a minimal `bench/_common.py` (the `result.json` envelope + run-dir helper; copy, don't
  import `bench._common` from the trainer). `probe_position_rescore.py:35` still imports
  `bench.tagger_external.run_bench`, which was archived 2026-08-30 — fix (inline
  `collect_external` / `load_external` or drop the probe) **before** moving.
- `bench/sam3_soft_prompt/*` → `../anime_tools/bench/sam3_soft_prompt/`. One trainer leak:
  `library.preprocess._dataset` (the dataset walker) — swap for `anime_tools._walk` (the
  contract's ≤50-line-helper rule), then it is clean.
- Docs → `../anime_tools/docs/`: `docs/experimental/soft_prompt_for_sam.md` and
  `docs/proposal/sam3_soft_prompt_expansion.md` (the shipped `caption-position` subject
  detector lives in `anime_tools.stages.instance_detection`), and
  `docs/findings/tagger_label_sharing_heads.md` (closed tagger-head finding; keep its pointer
  to the trainer's `_archive/anima_tagger_training/pe_backend_removed_2026_08_30/`).
  Leave 3-line "moved" pointers behind for one release, then delete them together with the
  existing pointers (`docs/experimental/{anima_tagger,position_captions}.md`,
  `docs/proposal/tagger_caformer_backend.md`).
- `tests/test_grouped_loss_negweight.py` (imports only `anime_tools.captions.group_router`)
  → package tests, unless `test_caption_drop_groups.py` / `test_tag_groups.py` already cover it.
- Gitignored history: `_archive/{anima_tagger_training,tagger_eval,tagger_factored_head}`
  (~2.1 GB) → the package's `_archive/`. No dependency effect either way; update the
  `bench/tagger_external/README.md` pointer if it moves.

**Stays (tagger used as a *judge* of trainer output — trainer→package direction, allowed):**
- `bench/{readback,tag_dropout,position_captions,region}/` — generate with the DiT
  (`anima_lora`, `library.inference`, `bench._anima`), score with the tagger.
- `docs/proposal/{tag_readback_reward,tag_dropout_mechanism,turbo_caption_ranking,
  phash_edit_position_clauses}.md`, `docs/findings/pe_registers_no_patch_outliers.md` —
  training / RWR / turbo / register lines that consume tagger scores or the PE tower.
- `scripts/toolkits/build_randoms.py`, `easycontrol_adapters/tools/{near_twins,
  phash_edit_pairs,subject_edit_pairs}.py` — trainer / EasyControl consumers.
- `gui/tabs/_autotag.py`, `gui/tabs/preprocess/captions.py` — GUI stays (decided above).
- `scripts/tasks/downloads.py` tagger-checkpoint fetch (`sorryhyun/anima-tagger/dbv4`) —
  `make download-models` needs it; it already reads `anime_tools.tagger.dbv4_meta`. A later
  refactor may call a package-side `download_tagger()`, but that is not a move.
- `configs/clause_vocabulary.yaml` — the user-editable override of the package default
  (`scripts/update.py` preserves it); not a duplicate.
- `tests/{test_caption_variant_sidecars,test_curation_walk_parity,test_curation_boundary,
  test_caption_shuffle,test_read_caption}.py` — cross-boundary or trainer-side.

Gate: same as Phases 1–2 — byte-identical bench outputs / passing tests before and after;
bump the pinned rev + `uv lock` in the trainer when the package gains `bench/`.

## Open risks

- **Two-repo dev loop.** Every curation change spans a path dependency; a contract change is a
  two-PR change — that's the point. Remember to bump the pinned rev (+ `uv lock`) whenever the
  package changes, and never pin `onnxruntime` in the package (conflicts with the trainer's
  `onnxruntime-gpu`; the CTD gate falls back to `cv2.dnn`).
- **Version skew for users.** The trainer should pin `anime_tools>=X,<Y` once the package is
  tagged, and the GUI system dialog should show both versions; a skewed install must fail at
  import with a clear message, not at step 3000.
- **Daemon.** `anime_tools` CLIs are plain scripts; the trainer's `make … --queue` keeps routing
  them through the daemon. No daemon client in the curation repo.

## Non-goals

- No functional change to any stage, the tagger, or the grammar during the move (Phases 1–2
  were byte-identical-output refactors; the hash checks were the gate).
- Not moving resize / VAE / TE / PE-feature *caching* — those are trainer caches.
- Not touching `sigma_lowres`, `preprocess-reconcile`, or `preprocess-demote` (latent-side).
- Not building a standalone curation GUI — the panels stay hosted by the trainer GUI.
