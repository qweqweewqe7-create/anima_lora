# SteerPE — text-steerable PE (SteerViT recipe on PE-Spatial)

Status: **Phase 0 run 2026-08-25 — concept steering emerges at this dataset's
scale; attribute steering does not (unsupervised).** No consumer wired yet.

Origin: *Steerable Visual Representations* (Ruthardt & Gaur et al., arXiv
2604.02327). The shipped SteerViT checkpoint (DINOv2-B, RoBERTa, photo data) was
probed first and rejected for our use — its softmax-over-patches heatmap is
winner-take-all, so it could not light two views of one girl or serve as an
identity witness for the multi-view audit. What we kept is the *recipe*:
zero-init tanh-gated cross-attention (vision → text) interleaved into a frozen
ViT, trained with a referring-segmentation proxy.

| File | Role |
|---|---|
| `networks/methods/steer_pe.py` | `SteerPE` — frozen `PEVisionTransformer` driven block-by-block, `GatedCrossAttention` at every other block, `TextConnector` (Qwen3-TE → PE width), per-patch `seg_head`; `set_gate_scale(ω)` |
| `bench/steer_pe/run_bench.py` | train + eval in one job (`make daemon-run ARGS="--label p0 bench/steer_pe/run_bench.py --label p0 --steps 3000"`) |
| `tests/test_steer_pe.py` | zero-gate bit-exactness, gate gradient, frozen tower, adapter round-trip |

Deviations from the paper: keys/values from the **Anima Qwen3-0.6B TE** (speaks
booru tags), and **per-patch sigmoid BCE** instead of patch softmax (binary
per-instance SAM3 masks are the supervision; BCE lets one prompt light every
instance). Supervision is entirely in-house: the region prep's SAM3 masks —
`masks`→"the girl", `masks_boy`→"the boy", `masks_head`→"the face",
`masks_person`→"a person" — 4 488 pairs, split by artist (15 % held out).

## Phase 0 result (`bench/steer_pe/results/20260825-1713-p0/`)

PE-Spatial-B16-512, 6 CA layers, 16.0M adapter params, 3 000 steps bs 8, ~5 min
on a 5070 Ti. Held-out patch-level PR-AUC vs SAM3 masks:

| prompt | n | steered | gate ω=0 (unsteered tower, same head) | swapped prompt |
|---|---|---|---|---|
| the girl | 118 | **0.983** | 0.949 | 0.590 |
| the boy | 11 | **0.971** | 0.396 | 0.164 |
| the face | 120 | **0.973** | 0.610 | 0.342 |
| a person | 74 | 0.863 | 0.821 | 0.679 |

Pair images (girl + boy masks, n=10): share of predicted mass inside the girl
mask is **0.86 under "the girl" vs 0.03 under "the boy"**; inside the boy mask
0.85 vs 0.01. Gates stay small (|tanh α| ≈ 0.01–0.03) yet decisive — same
shape as the paper's Fig. 7.

- **Text steering is real and prompt-driven** (swapped-prompt collapse is the
  paper's Tab. 6 control). The unsteered tower already localises *a* figure
  (0.95 on "the girl"), so the gain is in *which* figure — boy/face/person.
- **BCE lights every instance**: on the audit's tagger-only `multiple views`
  sheet (`ama_mitsuki/5828774`, SAM3 found zero boxes) "the girl" lights both
  views. On the other zero-box audit images "the boy" finds the partner's hands
  / arm. Caveat: on a headless-jeans view "the boy" also fires (boy masks are
  only 171 of 4 488 pairs) and "the face" picks a speech bubble when no face is
  visible — no abstain behaviour was trained.
- **Attribute steering did not emerge**: on 14 `2girls` images with two hair
  colours in the caption, the maps for "the girl with X hair" vs "… Y hair" have
  cosine **0.99999** — the CA keys only on the four trained concept words. This
  is a data statement, not a ceiling: nothing in the mix ties a prompt to *one
  of two* girls.

## Phase 3 slice — skytnt/anime-segmentation (2026-08-25, no lift)

Tier B trial on a 1/10 slice of `skytnt/anime-segmentation` (CC0): the full
`imgs`/`masks` real-GT set (1 111) + `fg-00` (2 000 RGBA foregrounds, alpha →
mask, composited on a random flat colour), mounted at
`/media/sorryhyun/새 볼륨/dataset/anime_segmentation`. Bench flags:
`--anime_seg_mode eval` (control: SAM3-only training, scored on the real GT) vs
`train` (adds 2 653 pairs under `a person`). Same 3 000 steps / seed.

| | ctrl `20260825-1752-p3-ctrl` | +aseg `20260825-1758-p3-aseg` |
|---|---|---|
| anime-seg real GT, "a person" (n=120) steered / ω=0 / swapped | **0.993** / 0.983 / 0.804 | 0.992 / 0.988 / 0.753 |
| SAM3 held-out girl / boy / face / person | 0.983 / 0.969 / 0.972 / 0.861 | 0.983 / 0.978 / 0.976 / 0.870 |
| pair share girl-under-girl / girl-under-boy | 0.859 / 0.030 | 0.857 / 0.045 |
| "the girl" swapped-prompt PR-AUC | 0.588 | 0.775 (worse discrimination) |

- The SAM3-distilled tower already scores **0.993** on the teacher-independent
  character-foreground GT — the "a character" concept is saturated by in-house
  pseudo-labels; the 2.6k extra pairs move nothing outside noise (boy n=11) and
  the zero-box audit sheet is unchanged (the headless-jeans "the boy" false
  fire is marginally *wider*).
- What the set is good for: a **teacher-independent eval** of the whole-figure
  concept (kept as `pr_auc_anime_seg`, `--anime_seg_mode eval`), not training.
  Its only label is binary foreground, so it cannot touch the Phase 1 gap
  (attribute binding) — the proposal's Phase 3 kill fires for this set: stop
  collecting single-concept sets; the next lever is still attribute-bound
  instance pairs (Phase 1).

### Where SAM3 fails on that GT (`bench/steer_pe/sam3_gap.py`, `20260825-1812-gap`)

SAM3 (thr 0.4) on all 1 111 real-GT images, IoU of the per-prompt union mask:

| prompt | zero-box | IoU<0.5 | mean IoU |
|---|---|---|---|
| `girl` | 312 (28 %) | 323 | 0.66 |
| `person` | 35 | 46 | 0.89 |
| `anime character` | 1 | 13 | 0.93 |
| best of 3 | 0 | **6 (0.5 %)** | 0.935 |

- The dominant "failure" is the **prompt**: `girl` alone (the region prep's
  concept) returns nothing on 28 % (male / chibi / non-human figures);
  `anime character` grounds nearly everything → use it as the whole-figure
  fallback prompt in `generate_masks` / region prep.
- The 6 genuine best-of-3 failures are **mask fragmentation** on extreme
  close-ups / deformed styles (speckled SAM3 masks), one 1 %-of-canvas distant
  figure, one figure behind a vase — not the zero-box / multi-view / attribute
  cases this line is about. On those 6, the Phase-3 control SteerPE (`a
  person`) beats SAM3 on 5 (IoU 0.72 vs 0.40, +0.32) because its 32×32 map
  covers the blob whole; on the rest it trails (0.88 vs 0.935) — a coarse
  fragmentation witness / box-prompt source, not a mask replacement.
- **The local copy was pruned (2026-08-25)** to the 29 pairs with SAM3
  best-of-3 IoU < 0.7 (`kept.json` in the dataset dir; `fg/` removed) — the
  only rows worth keeping as a hard-case eval / training garnish. Re-download
  the full set if a broader real-GT eval is ever needed again.

## Tier A — in-house pseudo-labels (2026-08-25)

Proposal `docs/proposal/steer_pe_anime.md` Tier A, run end-to-end in one
evening. Data (`post_image_dataset/steer_pe/`, untracked):

| Source | Script | Rows |
|---|---|---|
| A.1 multi-girl instances (SAM3 mask + crop-tagger tags, rival mask) | `bench/steer_pe/dump_instance_pairs.py --mode multi` | 453 imgs → 1 080 instances → 2 160 rows |
| A.1/A.3 solo attributes + character names (text-only, Phase-0 girl mask) | `… --mode solo` | 2 624 attr + 1 488 name |
| abstain negatives (absent hair/eye colour, wrong name) | both modes | 5 018 |
| A.2 caption-gated SAM3 concept sweep (`concepts.yaml`, ~75 concepts + hair/eyes/hands always) | `bench/steer_pe/sweep_concepts.py` | 31 k positive + 25 k negative (12.9 k SAM3 no-det, 12 k absent-tag) |

Bench additions: `--pairs_manifest` (JSONL row contract, `negative` /
`rival_mask` fields), `pr_auc_by_kind`, `abstain`, `wrong_instance`,
`--holdout_prompts` (zero-shot word test), `--drop_kinds`, `--kind_weight`.

### Results (`20260825-2104-tierA`, `20260825-2117-tierA-attr`)

Both 6 000 steps / 12 min, held-out by artist. The second run drops
`attr_solo` from training and oversamples `attr_multi` ×4; its swapped-prompt
control is the corrected one (a *different concept*, not the other phrasing of
the same concept — the first run's concept `swapped` column is meaningless).

| Kind (held-out) | steered | gate 0 | swapped (run 2) |
|---|---|---|---|
| the girl / boy / face | 0.98 / 0.91 / 0.96 | 0.80 / 0.20 / 0.54 | 0.78 / 0.23 / 0.34 |
| hair / hands / eyes | 0.97 / 0.74 / 0.54 | 0.64 / 0.02 / 0.01 | 0.30 / 0.04 / – |
| school uniform / swimsuit / speech bubble / sky / bag | 0.95 / 0.85 / 0.91 / 0.88 / 0.88 | 0.69 / 0.21 / 0.07 / 0.22 / 0.08 | – / 0.14 / 0.12 / – / – |
| cat ears / animal ears / thighs / breasts / bed | 0.83 / 0.72 / 0.79 / 0.72 / 0.76 | 0.20 / 0.12 / 0.09 / 0.05 / 0.16 | – / 0.15 / – / – / – |
| small things: choker / collar / necklace / navel / nipples | 0.13 / 0.01 / 0.08 / 0.06 / 0.18 | ~0 | – |
| **attr_multi** (the attribute test) | **0.43–0.47** | 0.34–0.39 | 0.44–0.48 |
| zero-shot held-out words umbrella / glasses / window | 0.34 / 0.14 / 0.09 | 0.12 / 0.00 / 0.07 | – |

- **Vocabulary (Phase 2 goal): PASS.** Most concepts with ≥ ~50 training
  masks land 0.7–0.97 with gate-0 near zero and swapped-prompt ≤ 0.3 — it is
  the text. Speech bubble 0.91 is a free win for the region/mask pipelines.
  Failures are the sub-patch objects (16 px patches at 512²) — expected, and
  the pixel-decoder (Phase 4c) is the only fix.
- **Abstain: PASS.** Mean prob 0.03–0.04 under an absent attribute / wrong
  character name vs 0.17–0.33 under a present one.
- **Attribute binding (Phase 1 gate): FAIL, twice.** `wrong_instance`
  share-correct 0.475 → 0.458 (chance), rival/own ≈ 1.0, attribute-map cosine
  0.987 → 0.961. Dropping the solo shortcut rows and ×4 oversampling did not
  move it. The instance sheet shows correct SAM3 targets and a heat map that
  floods every girl regardless of prompt. Caveat before calling the mechanism
  dead: only 2 160 multi-instance rows exist (proposal kill line is 30 k+), and
  their tags are mostly pose / anatomy (`standing, navel, nipples`) because the
  clause vocabulary gates hair / eye colour on the caption listing both — the
  data is thin *and* rarely visually discriminative.
- **Zero-shot words: weak.** Slightly above frozen, far from the paper's
  claim; with ~75 trained concepts the tower is still mostly a lookup table.
- Character names 0.93–0.95 but gate-0 already 0.74–0.82 and swapped 0.83 —
  on solo images the name is just "the girl"; no evidence the name itself is
  read.

### Verdict / what to use

Ship-able as a **fixed-vocabulary anime concept grounder** (girl / boy / face
/ person / hair / hands / clothing / ears / bubbles …) with a trustworthy
abstain — the practical consumer set from the proposal (audit witness, SAM3
box-prompt fallback, region `focus not found` rescue). Do **not** use it for
"which girl" questions. If attribute binding is ever retried, the lever is
data, not training knobs: multi-instance pairs whose prompts differ in a
*visible* attribute (rerun the dump with `--ungated_identity`, or pair by hair
colour from the crop tagger's raw scores) at ≥ 10 k rows.

Adapter: `bench/steer_pe/results/20260825-2104-tierA/steer_pe_adapter.safetensors`
(run 1 — the better concept numbers; run 2 traded concept accuracy for the
attribute oversampling that did not pay).

## Original next-phase note (superseded by Tier A above)

Attribute supervision is the missing ingredient and it also exists in-house:
the position-caption pipeline's per-instance boxes/masks carry identity tags
(hair / eyes / name). Pairs of the form ("the girl with black hair", instance
mask) on multi-girl images would test whether the steered tower can read
identity per instance *without cropping* — the multi-view audit's headless-crop
failure — and give the audit a fourth witness. Consumers to decide between:
audit identity witness, SAM3 box-prompt fallback on zero-box images, or a
text-steered reference encoder. Memory: `steervit-probe`.
