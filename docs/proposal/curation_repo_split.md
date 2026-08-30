# Curation split — tagger / masking / grouping / caption polishing → `sorryhyun/anime_tools`

Status: **Phase 1 complete (2026-08-30).** `github.com/sorryhyun/anime_tools` exists (private until the trainer branch merges — flip with `gh repo edit sorryhyun/anime_tools --visibility public --accept-visibility-change-consequences`, same for `ComfyUI-Anima-Tagger`); captions + tagger + stages live there, the trainer depends on it, old paths are shims. Contract: `../anime_tools/docs/contract.md`. Next: Phase 2 (masking + grouping + GUI panels).

Phase 1 log:
- 2026-08-30 — repo created (`anime_tools/{captions,tagger,stages}` + `_env/_walk/_hf/path_filter`
  copies; `captions/data/clause_vocabulary.yaml` packaged as the default the trainer's
  `configs/` copy overrides). 13 tests, 4 docs, the `captions` skill and the contract moved.
  Move was scripted (copy + ordered import-rewrite rules), no hand edits to bodies.
- Trainer: `library/_moved.py` `alias()` shims (sys.modules aliasing — same object, so
  monkeypatching either path works) for every moved `library.*` module; `forward()` shells for
  `scripts/preprocess/{8}.py` + `scripts/anima_tagger/*.py` that pin `ANIMA_HOME` and call the
  package `main()` — `make` argv shapes unchanged (pinned by `tests/test_preprocess_tasks.py`).
  In-process consumers (GUI, bench, easycontrol, tasks, façade) import `anime_tools` directly.
- Dependency: git rev pinned in `[tool.uv.sources]` through a default-on `anime-tools-git`
  group; `anime-tools-dev` (path, editable) conflicts with it — uv insists the package sit in
  the group, not in `dependencies` (same shape as `cuda-windows`/`rocm-windows`).
- `curation_home()` fallback changed from "checkout root" to **CWD** (no checkout once
  installed); the trainer's `run()` exports `ANIMA_HOME` so nothing moved.
- Guard caught two Phase-2 leaks the old manifest-prefix rule hid: `generate_masks{,_mit}.py`
  imported `library.preprocess.walk_images` → `anime_tools._walk`.
- Tagger node extracted to `github.com/sorryhyun/ComfyUI-Anima-Tagger` (`_vendor` dropped,
  `pip install anime-tools[tagger]`); `sync_vendor.py` no longer has a tagger target.
- Exit gate: both suites green (trainer 1405 fast + slow, package 285); caption master +
  mirrors + `.variants.txt` byte-identical on the live dataset except `mikozin/11841806`, which
  rewrites on **every** run pre-split too (unseeded variant draw — pre-existing);
  `caption_index.json` `image_meta`/`groups` identical vs a `main` worktree run. TE caches not
  regenerated (their producer `library/preprocess/text.py` only changed an import).


Phase 0 log:
- 2026-08-30 — guard test `tests/test_curation_boundary.py` landed (task d). It caught three
  lazy `library.anima` imports the original audit missed: the caption shuffle grammar
  (`NO_ARTIST_SENTINEL` + prefix/shuffle helpers) now lives in `library/captioning/shuffle.py`
  (trainer re-exports from `library.anima.training`), and the tokenizer-only loads in
  `correct_captions.py` / `position_captions.py` go through `library/captioning/tokenizers.py`,
  which takes tokenizer **directories** — the trainer wrapper (`scripts/tasks/preprocess.py`)
  resolves `.safetensors` → bundled config dir via `library.anima.weights.qwen3_tokenizer_dir`.
- 2026-08-30 — (b) grouping takes an injected `Embedder` (`library/vision/grouping_embedder.py`
  is the trainer's PE-Spatial one; CLI `--embedder module:callable`); the legacy `"pe"` tagger
  backend is **deleted** (`anima_tagger_model.py`, the PE cache builders, `bench/tagger_external/
  run_bench.py` → `_archive/anima_tagger_training/pe_backend_removed_2026_08_30/`); the tagger
  node's `_vendor` no longer ships `library/vision`.
- 2026-08-30 — (c) `path_filter.py` + `hf_download.py` moved under `library/captioning/` (shims
  left); `library/captioning/{_env,_walk}.py` are the curation-side home/path/logging + walker
  copies (parity-tested). The guard now forbids **every** non-manifest `library.*` import and
  passes. Model-path resolution decided: `ANIME_TOOLS_HOME` → `ANIMA_HOME` → checkout root.
- 2026-08-30 — (a) `anime_tools_contract.md` written. **Phase 0 exit met: tests green, no
  behavior change** (1757 passed).

## Why

`anima_lora` is two products in one tree:

1. **The trainer** — DiT + adapters + flow-matching loop + inference stacks. Depends on the
   dataset *caches* (`post_image_dataset/lora/*.npz|.safetensors`) and the caption master.
2. **Dataset curation** — Anima Tagger (+ dbv4 backend), SAM3/MIT masking, PE-Spatial
   grouping (`scripts/curate/`, near-twins), caption polishing (autotag, position clauses,
   correction, tag-drop groups, multiview audit, caption index). Produces the caption master
   and sidecars the trainer reads.

The second product is ~12k lines (`library/captioning/` alone is 17 modules), has its own
model zoo (PE-Core, PE-Spatial, SAM3, MIT, dbv4), its own heavy deps (`sam3`,
`segmentation_models_pytorch`, `albumentations`, `cv2`, `onnxruntime`), its own GUI surface
(~1.5k lines across `gui/tabs/_autotag.py`, `_caption_editor.py`, `_image_overlays.py`,
`preprocess/{captions,masking}.py`), ~18 tests, four experimental docs, a skill, and a
ComfyUI node (`custom_nodes/comfyui-anima-tagger/`) that today vendors `library/` +
`networks/` just to run the tagger. None of it touches the DiT, VAE, or `networks/`.

Splitting it out buys: a trainer install that doesn't drag `sam3`/`smp`/`cv2` (relevant to
the Windows tarball, GH #92); a curation tool usable on datasets destined for *other*
trainers (the Arca/CN/JP users already ask for the tagger standalone); the tagger node
depending on a package instead of `vendor-sync`; and a cleaner review surface for both.

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

## Target layout

```
anime_tools/                      (new repo `github.com/sorryhyun/anime_tools`, package `anime_tools`, uv-managed, py3.13; git-dependency, not PyPI)
  captions/                        # CORE (torch-free part) + [tagger] extra
    position_clauses.py                        ← library/captioning/position_clauses.py   (grammar, torch-free)
    caption_layout.py clause_rewrite.py clause_vocabulary.py correction.py
    tag_groups.py tag_rules.py tag_drop_groups.py taxonomy.py group_router.py
    index.py                                   ← scripts/preprocess/build_caption_index.py (caption_index.json)
    variants.py                                ← library/preprocess/caption_variants.py
  tagger/                          # [tagger] extra
    model.py data.py tagger.py dbv4_backend.py dbv4_meta.py feature_cache.py readback.py
                                               ← library/captioning/anima_tagger*.py, dbv4_*, feature_cache, readback
    cli/                                       ← scripts/anima_tagger/*  (autotag, autotag_server, predict, train_sidecar, build_dbv4_ckpt, derive_groups, vocab, constants)
  masking/                         # [masking] extra (sam3, smp, albumentations, cv2)
    sam.py mit.py merge.py                     ← scripts/preprocess/{generate_masks,generate_masks_mit,merge_masks}.py
    probe_sam.py probe_nms.py                  ← scripts/preprocess/{probe_sam_masks,probe_nms_pairs}.py
  grouping/                        # [grouping] extra — embedder injected, no PE dependency
    groups.py                                  ← library/datasets/grouping.py
    features.py matching.py                    ← library/vision/{pe_features,pe_matching}.py
    build_groups.py match_decensored.py apply_decensored.py   ← scripts/curate/*
  stages/                          # the caption-master stages (orchestration, were library/preprocess/*)
    autotag.py position_captions.py multiview_audit.py multiview_sheet.py instance_detection.py
    _walk.py                                   ← library/preprocess/_dataset.py::walk_images (duplicate; tiny)
  gui/                             # [gui] extra — PySide6 panels the trainer GUI mounts
    autotag.py caption_editor.py image_overlays.py captions_section.py masking_section.py
  tasks.py                         # `make`-style dispatcher: autotag, mask, mask-clean, curate-group, tagger-*, caption-*
  tests/                           # the ~18 tests listed in §Manifest
  docs/                            # anima_tagger.md, position_captions.md, multiview_audit.md, tagger_caformer_backend.md (proposal)
  skills/captions/                 # the `captions` skill moves with the grammar
```

Extras in `pyproject.toml`: `tagger`, `masking`, `grouping`, `gui`, `all`. **The base install
is `timm + transformers + safetensors + PIL + numpy`** (grammar + dbv4 tagger; torch comes
with timm). `anima_lora` depends on `anime_tools` (base) for the grammar and lists
`anime_tools[tagger,gui]` under its own `gui`/`dev` extras only.

**Distribution is a git dependency, not PyPI** (the repo is the product, like
`kohya-ss/sd-scripts`): release pins `anime_tools @ git+https://github.com/sorryhyun/anime_tools@<tag>`
in `anima_lora/pyproject.toml`; dev uses `[tool.uv.sources] anime_tools = { path = "../anime_tools",
editable = true }` so uncommitted edits are live. `uv sync` handles both, so end users see no
change. No wheel build, no PyPI name to defend.

### What stays in the trainer

- `library/preprocess/{images,latents,text,pe,uncond,reconcile,resize_preview,_progress}.py`
  — resize / VAE / TE / PE-feature caching. These produce the *trainer's* caches; they stay.
  `pe.py` untouched.
- `library/datasets/*` (free-fit buckets, `CachedDataset`, subsets, `mask_dir` alpha-mask
  loading) — the trainer *consumes* masks; the format is the contract (§Sidecar contract).
- `library/vision/{encoder,encoders,buckets}.py` — the PE-Core/PE-Spatial loader stays (see
  §seam); `library/vision/__init__.py` stops re-exporting `pe_features`/`pe_matching`.
- `library/training/{validation,cmmd,repa}.py`, `preprocess/pe.py`, `networks/lora_anima/factory.py`
  — untouched.
- `easycontrol_adapters/tools/near_twins/` — stays; imports grouping from the package and passes
  its own PE-Spatial bundle.
- `library/runtime/hf_download.py` — 116 lines, both sides need it; **duplicate** into
  `anime_tools/_hf.py` rather than create a dependency.
- `library/datasets/image_utils.py::IMAGE_TRANSFORMS` — the tagger only needs the 3-line
  transform; inline it in `anime_tools/tagger/data.py`.
- `anima_lora.captioning` façade namespace — becomes a lazy re-export of
  `anime_tools.tagger.AnimaTagger` (keeps embedder API stable; raises a clear
  "install anime_tools[tagger]" ImportError otherwise).

## Move manifest

| From (anima_lora) | To (anime_tools) | Extra | Notes |
|---|---|---|---|
| `library/vision/{pe_features,pe_matching}.py` | `grouping/{features,matching}.py` | grouping | grouping's own feature cache; embedder injected |
| `library/captioning/position_clauses.py` | `captions/position_clauses.py` | core | torch-free; **single grammar, never forked** |
| `library/captioning/{caption_layout,clause_rewrite,clause_vocabulary,correction,tag_groups,tag_rules,tag_drop_groups,taxonomy,group_router}.py` | `captions/` | core | |
| `library/captioning/preprocess.py` | `stages/captions.py` | core | drops `library.preprocess._dataset` import → `stages/_walk.py` |
| `library/preprocess/caption_variants.py` | `captions/variants.py` | core | trainer reads variant sidecars — see contract |
| `scripts/preprocess/build_caption_index.py` | `captions/index.py` + CLI | core | `caption_index.json` is a contract artifact |
| `library/captioning/{anima_tagger,anima_tagger_model,anima_tagger_data,dbv4_backend,dbv4_meta,feature_cache,readback}.py` | `tagger/` | tagger | legacy `"pe"` backend + `_bundle`/`_bundle_aux` dropped; `anima_tagger_data.py` PE feature-cache builders go to `_archive/` unless `bench/readback` still needs them |
| `scripts/anima_tagger/*` | `tagger/cli/` | tagger | `autotag_server.py` keeps its HTTP contract (GUI + node use it) |
| `scripts/preprocess/{autotag_captions,position_captions,correct_captions,review_position_captions,ab_position_captions,audit_multiview,audit_apply_curated}.py` | `stages/` + CLI | tagger | thin shells; bodies already in `library/preprocess/{autotag,position_captions,multiview_audit}.py` which move too |
| `library/preprocess/{autotag,position_captions,multiview_audit,multiview_sheet,instance_detection}.py` | `stages/` | tagger | |
| `scripts/preprocess/{generate_masks,generate_masks_mit,merge_masks,probe_sam_masks,probe_nms_pairs}.py` | `masking/` | masking | `filter_paths_by_glob` (from `library.datasets.subsets`) → duplicate the glob helper |
| `library/datasets/grouping.py`, `scripts/curate/*` | `grouping/` | grouping | |
| `scripts/tasks/{tagger,masking,curate}.py` | `anime_tools/tasks.py` | — | `make autotag|mask|mask-clean|curate-group|tagger*|test-tagger|caption-*` become `anime-tools <target>`; trainer `Makefile` keeps thin forwarding aliases for one release |
| `gui/tabs/{_autotag,_caption_editor,_image_overlays}.py`, `gui/tabs/preprocess/{captions,masking}.py` | `gui/` | gui | trainer's `PreprocessingTab` mounts them if importable, hides the sections otherwise |
| `tests/test_{autotag_captions,caption_correction,caption_drop_groups,caption_index,caption_shuffle,caption_variant_sidecars,colorize_caption,grouping_grid_match,position_captions,read_caption,tagger_*,tag_groups}.py` | `tests/` | — | `test_grouped_loss_negweight`, `test_repa_pe_sidecar`, `test_timestep_mask_mixed_rank` **stay** (trainer-side consumers) |
| `bench/{tagger_external,readback,position_captions}/` | `bench/` | — | `bench/region/` stays (EasyControl consumer) but imports the tagger from the package |
| `docs/experimental/{anima_tagger,position_captions,multiview_audit}.md`, `docs/proposal/tagger_caformer_backend.md` | `docs/` | — | trainer docs keep one-line pointers |
| `.claude/skills/captions/` | `skills/captions/` | — | trainer `CLAUDE.md` §Captions shrinks to the grammar warning + pointer |
| `custom_nodes/comfyui-anima-tagger/` | **its own repo** (like the other extracted nodes) | — | `pyproject` depends on `anime_tools[tagger]`; `_vendor/` deleted; drop from `sync_vendor.py` |

### Cross-boundary import fix-ups (trainer side, exhaustive from the audit)

| Site | Change |
|---|---|
| `library/datasets/grouping.py:179,221` | moves; call site becomes `build_groups(embed=pe_spatial_embedder)` wired by the trainer's `make curate-group` |
| `easycontrol_adapters/tools/near_twins/__main__.py:83,413` | keeps its `load_pe_encoder`; imports matching from `anime_tools.grouping` |
| `library/vision/__init__.py` | stays; drops `pe_features`/`pe_matching` re-exports (shim for one release) |
| `library/captioning/__init__.py` | same shim pattern → `anime_tools.captions` / `.tagger` |
| `gui/tabs/preprocess/{tab,knobs}.py`, `gui/system_dialog.py`, `gui/tabs/image_tab.py` | grammar + tagger via package; sections mounted conditionally |
| `gui/i18n/{ko,cn,ja}.py` | strings for the moved panels move with them (`anime_tools/gui/i18n/`); the `translator` agent's surface list gains the new path |
| `easycontrol_adapters/tools/near_twins/`, `easycontrol_adapters/region/prep.py` | tagger/grouping via package |
| `anima_lora/__init__.py` | `captioning` namespace → lazy import of `anime_tools.tagger` |
| `scripts/experimental_tasks/inference.py`, `bench/*` (11 files) | mechanical rename |

## Sidecar contract (the real API)

Everything the trainer reads from the curation side is a **file**. Freeze and document these
in `anime_tools/docs/contract.md` *before* moving code; both repos' tests pin them.

| Artifact | Producer | Consumer | Format |
|---|---|---|---|
| `image_dataset/**/{stem}.txt` | autotag / position / correction stages, caption editor | `preprocess-te` (`library/preprocess/text.py`) | caption grammar: `<tag, tag, …>. <Position clause>. …` — parsed only via `position_clauses.parse_caption` |
| `{stem}.variants.txt` (caption variants) | `captions/variants.py` | TE caching + train-time variant pick | current sidecar format, unchanged |
| `caption_index.json` | `captions/index.py` | `make caption-index` consumers, grouping, tag-dropout bench | see [[project_caption_index_shared_artifact]] — sampling policy stays out |
| `masks/{stem}.png` (alpha masks) | `masking/` | `library/datasets/subsets.py` (`mask_dir`) | 8-bit L PNG, same size as resized image |
| `groups.json` / decensor match tables | `grouping/` | grouped-loss neg-weight, near-twins descriptor blueprints | current schema |
| grouping feature cache (`pe_features` layout) | `grouping/features.py` | grouping only | curation-private; **not** the trainer's `{stem}_anima_pe.safetensors`, which stays trainer-only |

Nothing in the contract is written by one side and parsed by code living on the other side
except the caption grammar — which is why `position_clauses.py` moves and the trainer
imports it from the package.

## Phases

**Phase 0 — contract + seam (no repo yet).** In `anima_lora`: (a) write `contract.md`;
(b) make grouping take the embedder as an injected callable instead of calling
`load_pe_encoder` itself (`grouping.py:221`); delete the legacy `"pe"` tagger backend
(`anima_tagger.py::_init_pe_backend/_bundle/_bundle_aux`, PE builders in
`anima_tagger_data.py:360,460`) or fence it behind a trainer-side shim; (c) inline `IMAGE_TRANSFORMS` + the glob
helper; (d) add a CI guard test: `anime_tools`-destined modules must not import
`library.{anima,models,training}` / `networks` (grep-based, lands now so the boundary can't
regress before the move). Everything stays importable at old paths. **Exit: tests green, no
behavior change.**

**Phase 1 — new repo, core + tagger.** Create `anime_tools` with `vision/`, `captions/`,
`tagger/`, `stages/`; move the manifest rows for those; add `anime_tools` as a git
dependency in `anima_lora/pyproject.toml` (`[tool.uv.sources]` path override for dev); trainer imports switch; old `library/vision` + `library/captioning`
become shims that warn. Tagger node repo extracted, `_vendor` dropped. **Exit: `make
test-unit` green in both repos; `make preprocess` end-to-end on the live dataset produces
byte-identical `.txt` + TE caches vs. pre-split (hash the caption master before/after).**

**Phase 2 — masking + grouping + GUI.** Move the remaining extras; trainer GUI mounts the
panels conditionally; `make mask|autotag|curate-group|…` forward to `anime-tools` with a
deprecation line. Windows tarball (GH #92): the git dependency resolves at `uv sync` like any
other; the offline tarball vendors a checkout of the pinned tag (`filter="data"`, no symlinks —
same rules as [[project_gh92_windows_backend_default_group]]). **Exit: GUI smoke in 4 languages (translator agent re-syncs the moved
strings); `make mask` byte-identical masks on a fixed 20-image sample.**

**Phase 3 — delete shims.** One release after Phase 2: remove `library/vision`,
`library/captioning`, the Makefile aliases; `CLAUDE.md` §Captions / §Preprocessing rewritten
to pointers; memory notes that reference moved paths (`project_caption_index_shared_artifact`,
`project_tagger_dbv4_backend`, `project_tagger_resident_mmap_ram_budget`) get their paths
updated.

## Risks / open questions

- **Two-repo dev loop.** Every caption/tagger change now spans a path dependency. Mitigation:
  `uv` path source in dev so uncommitted `anime_tools` edits are live; a `make
  test-both` alias. Accept that a contract change is a two-PR change — that's the point.
- **Version skew for users.** The trainer must pin `anime_tools>=X,<Y` and the GUI should
  show both versions in the system dialog. A skewed install fails at import with a clear
  message, not at step 3000.
- **Model-path resolution.** The tagger checkpoint (`models/captioners/anima-tagger-dbv4`) and
  the gated dbv4 backbone resolve via `library.env.anima_home()` + `hf_download` today. The
  curation package needs its own `ANIMA_CURATE_MODELS` (default `~/.anima/models`) and the
  trainer passes its dir explicitly. **Decided in Phase 0** — see the contract §4 (`ANIME_TOOLS_HOME` / `ANIME_TOOLS_MODELS`).
- **Grouping without PE.** A standalone `anime-tools group` run (no trainer installed) needs
  an embedder; dbv4 backbone features are the candidate default but the grid-match grouping was
  tuned on PE-Spatial — bench before calling them interchangeable
  ([[project_region_v5_slack_pairs_face]] / `tests/test_grouping_grid_match.py` fixtures).
- **Daemon.** Autotag/mask jobs currently go through the trainer's daemon (`make daemon-run`).
  `anime_tools` CLIs are plain scripts; the trainer's `make autotag --queue` alias keeps
  routing them through the daemon. No daemon client in the curation repo for now.
- **i18n.** ~200 strings move. The `translator` agent's surface list must include the new
  repo or the KR/CN/JP GUI silently falls back to English (real user base — see
  [[project_user_community_audience]]).
- **Name / account.** Settled 2026-08-29: repo `sorryhyun/anime_tools` (GitHub-unique via the
  owner prefix; `muggletools` was the runner-up). The trainer's current remote is
  `n0va39/anima_lora` — publishing the new repo under `sorryhyun` means the public pair lives
  under that account going forward.

## Non-goals

- No functional change to any stage, the tagger, or the grammar during the move (Phases 1–2
  are byte-identical-output refactors; the hash checks are the gate).
- Not moving resize / VAE / TE / PE-feature *caching* — those are trainer caches.
- Not touching `sigma_lowres`, `preprocess-reconcile`, or `preprocess-demote` (latent-side).
- Not building a standalone curation GUI — the panels stay hosted by the trainer GUI.
