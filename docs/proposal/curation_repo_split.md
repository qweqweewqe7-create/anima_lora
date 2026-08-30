# Curation split — tagger / masking / grouping / caption polishing → `sorryhyun/anime_tools`

Status: **Phases 0–2 complete (2026-08-30); Phase 3 pending (one release out).**
`github.com/sorryhyun/anime_tools` (package `anime_tools`, git dependency — no PyPI) owns the
caption grammar + polishing stages, the Anima Tagger, masking (SAM3 / MIT / merge), grouping
(PE-Spatial near-twin features → `groups.json`) and the vendored PE vision tower. The trainer
depends on it via the default-on `anime-tools-git` uv group (pinned rev in
`[tool.uv.sources]`; `uv sync --no-group anime-tools-git --group anime-tools-dev` for a live
`../anime_tools` loop). Old trainer import paths are `DeprecationWarning` shims
(`library/_moved.py`) and forwarding shells. Contract: `../anime_tools/docs/contract.md`.
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

## Phase 3 — delete shims (one release after Phase 2)

- Remove `library/_moved.py` and every shim/shell that uses it: `library/captioning/*`,
  `library/preprocess/{caption_variants,autotag,position_captions,…}.py`,
  `library/vision/{pe_features,pe_matching,grouping_embedder}.py`, `library/datasets/grouping.py`,
  `scripts/anima_tagger/*.py`, `scripts/curate/*.py`, `scripts/preprocess/{autotag_captions,
  position_captions,correct_captions,generate_masks,generate_masks_mit,merge_masks,probe_*}.py`.
  **Keep** `library/models/pe.py` (permanent re-export) and `library/vision/{encoder,encoders,
  buckets,data,resampler}.py` (trainer registry).
- `scripts/tasks/{tagger,masking,curate}.py` switch from script paths to `-m anime_tools.…`
  module invocations (the `make` target names and `--queue` daemon routing stay).
- `tests/test_curation_boundary.py` shrinks to the task wrappers (or is retired in favor of
  the package's `test_boundary.py`).
- `CLAUDE.md` §Captions / §Preprocessing rewritten to pointers; memory notes referencing moved
  paths (`project_caption_index_shared_artifact`, `project_tagger_dbv4_backend`,
  `project_tagger_resident_mmap_ram_budget`) get their paths updated.
- Trainer `pyproject.toml`: decide whether `sam3` / `segmentation-models-pytorch` /
  `albumentations` stay as direct deps (still used by `bench/`, `easycontrol_adapters/`) or
  ride only on `anime-tools[masking]`.
- Windows tarball (GH #92): the offline tarball must vendor a checkout of the pinned
  `anime_tools` rev (`filter="data"`, no symlinks — [[project_gh92_windows_backend_default_group]]).

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
