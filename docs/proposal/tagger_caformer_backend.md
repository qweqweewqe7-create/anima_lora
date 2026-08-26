# Anima Tagger → external dbv4 backbone: archive the training pipeline, keep the contract

Status: **Phases 1–2 LANDED (2026-08-27)**, artist scope dropped by decision.
Phase 0 (the measurement) is the reason this document exists.

**2026-08-27 results** — `config.json["backend"]="dbv4"` behind the unchanged
`AnimaTagger.predict()` contract (`library/captioning/dbv4_backend.py`,
`make tagger-dbv4` → `models/captioners/anima-tagger-dbv4/`, sidecar via
`daemon-run scripts/anima_tagger/train_sidecar.py`):

- **Sidecar** (linear on caformer's 3072-d MLP-head hidden, 238 rows =
  118 copyright + 36 OC characters + 84 danbooru-renamed generals; **no
  `@artist`** — the user decided artist attribution is not a tagger goal):
  copyright macro-F1 **0.815** / mAP 0.92 (v5 0.638 → gate passed), OC
  characters 0.889 / 0.98, renamed generals 0.40 / 0.61. People-count: the
  plain count-tag rule on dbv4's count tags scores **0.943** vs the sidecar
  head 0.929 vs v5 0.885 — the rule is shipped as authoritative (it is also
  consistent with the emitted `Ngirls` tags); the head's softmax is exposed
  as `people_count_scores` only.
- **Phase 1 gate** (`bench/position_captions/probe_autocaption.py`, same-day
  v5 baseline on today's 338-GT-image population, 168 hair / 45 character
  positions): hair-position accuracy **0.566 → 0.750**, character-position
  **0.822 → 0.911**, count 0.757 (shared SAM3 detections). The proposal's
  "≥ 0.80 hair" was set against the 0.30 12-sheet number; on this larger
  population the remaining 42 misses are adjacent-shade confusions
  (grey↔white, blue/purple→black on dark art), not a backend defect.
- Two backend-specific rules were needed and are unit-pinned
  (`tests/test_tagger_dbv4_backend.py`): pure-`softmax` groups only emit a
  winner that clears its own threshold (dbv4 was never CE-trained on our
  groups — the first live run emitted `@aak`/`loli`/`uncensored` off a −30
  logit), and the `original`-copyright OC/artist consistency rule is skipped
  (no artist is ever emitted, so it killed 9/12 mignon OC hits).
- **Default flipped** (same day): `DEFAULT_TAGGER_DIR` →
  `models/captioners/anima-tagger-dbv4`, `TAGGER_HF_SUBFOLDER` → `dbv4/`
  (our vocab/rules/groups/thresholds/sidecar only — the backbone is fetched
  from the gated upstream repo); ComfyUI node, `make download-tagger`,
  caption-index vocab default and the node's `_vendor` tree follow.
- **Phase 3 calibration** (`bench/tagger_external/calibration_check.py`,
  `results/20260827-*-dbv4-calib`): head-tier ECE **0.019** (gate ≤ 0.05
  PASS; mid 0.006, tail 0.002 — pooled ECE is dominated by the negative mass;
  above 0.5 the backbone is mildly over-confident, conf−acc +0.08…+0.16).
  Threshold transfer: only 55 % of head tags have the card `best_threshold`
  within ±0.10 of our val-optimal (gate ≥ 80 % FAIL) **but** mean card−val
  = +0.007 and median |Δ| = 0.09 — no systematic bias, just 0.05-grid noise
  on few-positive tags. Verdict: keep the card thresholds; recalibrating on
  791 val images would re-introduce the v4 hair-trigger problem. Sidecar
  rows: ECE 0.002 (val-calibrated by construction).
- **Phase 4 readback** (`bench/readback/results/20260827-0827-dbv4`, logits
  off the sidecar feature cache): shuffled-caption win-rate **1.000**, AUROC
  **1.000**, recall@1 row/col 0.979 / 0.987 (v3-era 0.991 / 0.98) → PASS;
  the read-back instrument is strictly stronger on the new backend.
- Still open: the `caption-position` knob resweep (`bag_relax` /
  `_EDGE_CLEAR`, soft-prompt proposal A3), Phase 5 (RWR), Phase 6 (archive
  the PE training pipeline + 158 GB caches), and uploading the `dbv4/`
  subfolder to `sorryhyun/anima-tagger` so the auto-fetch resolves.

## TL;DR

The in-house Anima Tagger (frozen PE-Core + PE-Spatial, linear head, 2.5k-tag
vocab, ~16k training images) is **2–2.5× worse than an off-the-shelf danbooru
tagger on our own held-out split**, on every slice, including the tail tags we
built it for. Measured 2026-08-26 (`bench/tagger_external/`, 791 val images,
intersection vocab = 2,182 of our 2,532 tags = 92.6 % of GT positives, mean AP
= threshold-free headline):

| mAP | anima-tagger-v5 | `caformer_b36.dbv4-full` | `convnextv2_huge.dbv4-full` |
|---|---:|---:|---:|
| all (1,920 supported tags) | 0.297 | **0.633** | **0.719** |
| character | 0.619 | 0.964 | 0.968 |
| count (`1girl`…) | 0.591 | 0.853 | 0.922 |
| general | 0.276 | 0.612 | 0.703 |
| tail (train freq < 200) | 0.285 | 0.630 | 0.723 |
| rating, 4-way acc | 0.833 | 0.905 | 0.923 |
| params / TFLOP / input | 2×PE (~1.3 B) | 134 M / 0.13 / 384² | 693 M / 1.20 / 512² |

Ours wins 22 of 840 tags with val support ≥ 5, and those are caption-convention
artifacts (`breasts`, `nude`, `skirt` sit at base rate for *both* — rules.yaml
rewrites), not real wins. On the **position-caption crops** (the saved SAM3
crops of the 12 hand-written GT sheets, re-scored pixel-identical):
hair-colour-per-crop ours **3/10** (7 crops abstain — the `hair_color`
softmax-when-solo gate returns `None` on multi-person crops), caformer
**10/10**, huge 8/10; GT character kept ours 3/6, both externals 4/6 (the two
misses are characters absent from the dbv4 vocab). Counterbalanced L/R
binding renders: 48/48 for all three.

Decision proposed: **archive the tagger *training* pipeline and its 158 GB of
feature caches; keep the tagger *inference contract* and re-implement it on
`caformer_b36.dbv4-full`** (9× cheaper than the Huge, identical on
characters, best on the crop task), with a small sidecar head for the three
things dbv4 cannot say (`@artist`, copyright, dataset-only characters). Then
— and only if it calibrates — promote it from "captioner" to "training
signal" through the already-validated read-back instrument.

## Why this is not a head-tweak problem

The gap is a *backbone* gap. dbv4 is a full backbone fine-tuned on all of
danbooru (12,476 tags, ~5 M images); ours is a linear probe on frozen
retrieval features. Two lines already tried to close this from the head side
and were refuted — label-sharing heads (issue #94, `docs/findings/
tagger_label_sharing_heads.md`: factored + full = linear) and the spatial-L
headroom line (this week's v6-spatialL: val macro-F1 0.2496, +0.01). The
frozen-PE ceiling is what the external number exposes. Its own held-out
micro-F1 (0.689 caformer / 0.697 huge) matches what we measure on our images
(0.674 / 0.716), so this is not danbooru memorisation of our val set either.

## What dbv4 cannot do (the sidecar scope)

Unmatched from our vocab: **92 `@artist`** (the whole roster — dbv4 has no
artist category), **118 copyright** (no copyright category either), **36
characters** (dataset OCs like `chiyo (ane naru mono)`), 84 general (mostly
rules.yaml renames / our rating literals), 19 deprecated. Plus two head-shaped
things: our **people-count head** (`1girl_1boy`, …) and our **rating band**
(`safe/sensitive/nsfw/explicit` — a rename of danbooru's four, mapped
1:1 in the bench; not a gap).

Everything else our stack does on top of the score vector — `GroupRouter`
softmax groups (`groups.yaml`), `rules.yaml` renames, `character_floor`, the
OC-suffix/artist consistency rule, position-clause eligibility — is
**post-processing of a `{tag: prob}` dict** and applies unchanged to any
backend. That is the design property this proposal leans on.

## Design

### Keep: the `AnimaTagger.predict()` contract

`predict(pil) -> {rating, rating_scores, people_count?, scores, kept, groups}`
is what every consumer reads: `library/preprocess/autotag.py`,
`library/preprocess/position_captions.py` (`tag_fn=tagger.predict`),
DirectEdit ψ_src (`scripts/experimental_tasks/inference.py`, comfy node),
`library/captioning/readback.py`, the GUI autotag tab (`gui/tabs/_autotag.py`
via `autotag_server.py`), `scripts/inference_server.py`, and the
`comfyui-anima-tagger` node. None of them change.

### Replace: the backend behind it

```
library/captioning/anima_tagger.py
  AnimaTagger              # facade, unchanged public surface
    ├─ backend: Dbv4Backend    # NEW — timm caformer_b36 (or huge), pad-white→384, sigmoid
    │     scores over the dbv4 vocab, name-normalised (`_`→space), rating 4-way
    └─ sidecar: SidecarHead    # NEW — linear head on the backend's pooled pre-logit
          features; emits ONLY tags dbv4 lacks (artist / copyright / OC chars /
          people_count); trained on our 16k images with cached features
  GroupRouter / rules / floors  # unchanged, applied to the merged score dict
```

The merged `scores` dict is `dbv4 tags ∪ sidecar tags`, with our vocab as the
key space (dbv4 names normalised through the existing `rules.yaml` map — the
same `rename_recovery` the eval bench uses). `vocab.json` / `groups.yaml` /
`rules.yaml` stay as **data**; the training-time machinery that produced them
(`build_vocab`, `derive_groups`) stays as *curation scripts* since they read
captions, not features.

Backend choice: **caformer_b36 default**, `convnextv2_huge` as an opt-in for
batch autotag where +0.09 mAP is worth 9× compute. Both are gated GPL-3.0
repos — see Risks.

### Sidecar head — "tune a little bit"

Not a fine-tune of dbv4 (692 M / 134 M params on 16 k images would
over-fit and destroy the danbooru prior that is the whole point). Instead the
same recipe we already have, on a better trunk: cache the backend's pooled
pre-logit feature per training image (caformer_b36: one 384² forward, 16 k
images ≈ minutes), train a linear head **only on the sidecar label set**
(artist softmax + sentinel, copyright BCE, OC characters BCE, people-count
CE), calibrate per-tag thresholds on val. `train_cached.py` / `calibrate.py`
already do exactly this against PE features; the change is the feature
source and the label mask.

Optional second step if the sidecar is weak: unfreeze dbv4's **classifier
layer only** and train it jointly with the sidecar rows appended, at LR 1e-4,
with the original 12,476 rows regularised toward their initial weights
(L2-SP). This is the only "make caformer more advanced" move that cannot
lose the danbooru prior. Anything deeper needs a much larger dataset than we
have and is out of scope.

### Calibration study — the gate for training use

A tagger used as a *training signal* needs confidences that mean something,
not just a good ranking. Measure on our val split, per frequency tier:

- reliability diagram + ECE of the raw sigmoid (dbv4 was trained with
  label smoothing? — `meta.json` says `drop_rate 0.2`, no smoothing flag;
  check empirically, do not assume);
- per-tag `best_threshold` from its card vs our val-optimal threshold —
  if the two agree within ±0.1 on head tags, its thresholds transfer and we
  do not need to recalibrate on our data;
- the *shuffled-caption drop* and *true-vs-random AUROC* controls from
  `bench/readback/run_bench.py`, which are the validated definition of "this
  confidence tracks caption adherence" in this repo.

### Training-time use — "tag adherence boost"

Do **not** invent a new objective. The repo already has a validated
instrument for exactly this: `library/captioning/readback.py::TagReadback`
(Read-It-Back, arXiv 2607.11886) — mean log σ(tag prob) over the caption's
content tags, **group-relative only** (same caption, N images; absolute
values across captions carry a language-prior term). Phase 0a of
`tag_readback_reward.md` passed on the v3 tagger: shuffled-caption drop
0.991, AUROC 0.98, transfers to generated images (AUROC 0.98–1.00 on turbo
renders). Its limits are also recorded there: blind to the text/pose axis
(chance on turbo teacher-vs-student), and the FM-val trap
(`project_closed_lines_rollup`: never gate a training change on FM-MSE —
judge by CMMD + held-out read-back + eyeball).

The rebuild is: `TagReadback` takes its logits from the new backend (it
needs only a `{tag: logit}` per image and the content-tag mask — the
artist/metadata masking stays). Then the consumers the readback proposal
already names light up with a much stronger judge: `dave_mod_bestofn q_tag`,
soup ingredient gating, seed selection, and the **RWR artist LoRA**
(reward-weighted regression — per-sample loss weights from read-back on the
model's own renders, ReST grow/improve over the existing CFM loop). That is
the "tag adherence boost for LoRA training" path with the least new risk,
because the estimator and phase discipline are inherited from PR #67.

A direct differentiable tag loss (tagger on the 1-step x₀ estimate) is
listed as a **later** option, not a phase: it needs a VAE decode per step
(the tagger itself is cheap at 384²), is REPA-adjacent in shape (feature
alignment on turbo was refuted — `project_turbo_repa_phase0_drift`), and its
failure mode (the LoRA learns to please the tagger rather than the caption)
is exactly what the group-relative RWR formulation avoids.

## Phases and gates

| Phase | What | Gate | Cost |
|---|---|---|---|
| **0** ✅ | `bench/tagger_external/run_bench.py` + `probe_position_rescore.py` | done — numbers above | — |
| **1** | `Dbv4Backend` behind `AnimaTagger`; parity test that `predict()` returns the same keys/types; wire `--tagger_backend {pe,dbv4}` in `autotag` / `position_captions` / inference server | re-run `bench/position_captions/probe_autocaption.py` end-to-end (SAM3 + tagger): `hair_position_accuracy` 0.30 → **≥ 0.80**, `char_position_accuracy` ≥ 0.57; `make caption-autotag` dry-run review sheet on 50 images, no new caption-grammar breakage (`position_clauses` unit tests green) | ~1 day |
| **2** | sidecar head on cached caformer features (artist / copyright / OC chars / people-count) | artist argmax acc + copyright macro-F1 **≥ v5** on the same val split (v5 copyright mean val-F1 0.638; artist acc must be measured — its softmax group is excluded from the F1 sweep); people-count acc ≥ v5 | ~half day + minutes of GPU |
| **3** | calibration study (ECE, threshold transfer, readback controls) | ECE ≤ 0.05 on head tier, card thresholds within ±0.1 of val-optimal on ≥ 80 % of head tags; readback shuffled-drop ≥ 0.90, AUROC ≥ 0.80 (the existing gates) | ~half day |
| **4** | `TagReadback` on the new backend; re-run `bench/readback/run_bench.py` real-data + turbo-render arms | match or beat the v3 numbers (0.991 / 0.98); turbo checkpoint-spread ratio reported, not gated | ~half day |
| **5** | RWR artist LoRA per `tag_readback_reward.md` with the new judge | that proposal's gates (CMMD non-regression + held-out read-back lift + eyeball); **not** FM-val | as budgeted there |
| **6** | archive | after Phase 2 passes: see below | ~1 day |

Phase 1 alone pays for the whole thing (position captions + autotag). Phases
3–5 are the "use it for LoRA training" bet and each has a kill switch.

## Archive plan (Phase 6)

Move to `_archive/anima_tagger_training/` (untracked tier, per the
project-lifecycle convention): `scripts/anima_tagger/{train_cached,
train_common, calibrate, caches, embed_tags}.py`, the `build_features` /
`train` / `calibrate` modes of `cli.py`, `make tagger` /
`make preprocess-tagger` targets, the training tests
(`test_anima_tagger_{cached_dataset,dual_encoder,label_embed,pe_cache_batching}.py`,
`test_tagger_spatial_headroom.py`, `test_tagger_calibration_and_strokes.py`),
the v2/v3/v5/v6 checkpoints (keep v5 in `models/captioners/` until Phase 2
is green — it is the artist/copyright fallback), and
`docs/experimental/anima_tagger.md` §Training pipeline (the doc is rewritten
around the backend/sidecar split). Reclaim **158 GB**:
`post_image_dataset/anima_tagger/` (42 GB, incl. the dead 32 GB spatial-L
dir) + `anima_tagger_stroked/` (116 GB). Note the mmap-resident RAM budget
memory (`project_tagger_resident_mmap_ram_budget`) becomes moot.

Keep in tree: `library/captioning/{anima_tagger, position_clauses,
tag_rules, taxonomy, tag_groups, readback, correction}.py`,
`scripts/anima_tagger/{vocab, derive_groups, predict, autotag,
autotag_server, eval_metrics}.py`, `bench/tagger_external/`, the
`GroupRouter` (moves out of `train_common.py` into `library/captioning/`).

Also close: `project_tagger_dual_hardrouted` / `project_tagger_v5_stroke_aug`
memories become historical; the "spatial-L headroom" line is superseded, not
refuted (its premise — the PE trunk is the ceiling — is confirmed from the
other direction).

## Risks

- **Licence.** Both dbv4 repos are **GPL-3.0** and gated (access approved
  for the Huge and caformer; `convformer_b36` still pending). This repo is
  MIT. Loading GPL weights at runtime does not relicense our code, but a
  ComfyUI node that *bundles* or auto-downloads them, and any fine-tuned
  sidecar checkpoint that *includes* dbv4 weights, must ship as GPL. Plan:
  never vendor the weights, auto-download from HF under the user's own token
  (the gate makes the user accept the terms), and ship the sidecar as a
  **separate** safetensors that contains only our head. Confirm with the
  animetimm terms before Phase 1 lands in the node.
- **Danbooru rating semantics ≠ ours.** `questionable→nsfw`,
  `general→safe` mapped 1:1; our corpus is 67 % explicit and the bench says
  dbv4 already beats us on it (0.905 vs 0.833), but the sidecar can carry a
  4-way rating head trained on our labels if the mapping leaks.
- **Name-space drift.** dbv4 uses danbooru names as of 2025-10; our
  `rules.yaml` renames and `booru id-space collision`
  (`project_booru_id_space_collision`) both bite when joining vocabularies.
  The bench's `align_vocab` is the single join point — keep it that way.
- **Calibration may fail (Phase 3).** Then the backend is still the better
  *captioner* (Phases 1–2 stand) and training use falls back to the v3-era
  read-back — nothing regresses.
- **`convformer_b36`** was the user's first "lighter" candidate; caformer is
  the same family (MetaFormer, 134 M) and is the one we could access. Re-run
  `run_bench.py --external_repo animetimm/convformer_b36.dbv4-full
  --external_arch convformer_b36 --external_img_size 384` when the gate
  opens; it is one command.

## Reuse inventory (verified by code-reading 2026-08-26)

- `bench/tagger_external/run_bench.py` — intersection-vocab eval,
  `align_vocab` join, `load_external` (timm arch + safetensors, no imgutils
  dependency), `collect_external` (pad-white-square → bicubic → ImageNet
  norm, matches the card's `preprocess.json`).
- `bench/tagger_external/probe_position_rescore.py` — crop-level rescoring
  of saved probe artifacts.
- `scripts/anima_tagger/train_cached.py` + `calibrate.py` — the sidecar
  trainer, modulo feature source + label mask.
- `library/captioning/readback.py` — `TagReadback`; needs a logits provider,
  nothing else.
- `bench/readback/run_bench.py`, `render_turbo.py` — Phase 3/4 validity
  gates as-is.
- `bench/position_captions/probe_autocaption.py` — Phase 1 gate as-is.
