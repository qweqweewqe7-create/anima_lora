# SteerPE — an anime-native, text-steerable vision tower (project proposal)

Status: **Tier A RUN (2026-08-25)** — results in
[`docs/experimental/steer_pe.md`](../experimental/steer_pe.md) §Tier A.
Phase 2 (vocabulary) passes, abstain passes, **Phase 1 (attribute binding)
fails at the data we have** (2 160 multi-instance rows, chance-level
wrong-instance share); zero-shot words weak. Practical outcome: a
fixed-vocabulary anime concept grounder. Phases 3–4 unrun. Reuse inventory verified by code-reading; external datasets are listed
as *candidates to verify* — none has been downloaded or licence-checked yet.

## TL;DR

Take the SteerViT recipe (zero-init tanh-gated cross-attention interleaved
into a frozen ViT, trained on a referring-segmentation proxy; arXiv
2604.02327) and grow it into **the** prompt-conditioned vision tower for this
repo: frozen **PE-Spatial** + keys from the **Anima Qwen3 TE**, trained on
**anime-domain referring masks** — first our own SAM3 / position-caption
pseudo-labels, then public anime segmentation & detection sets.

Phase 0 already showed the mechanism works at toy scale (4.5k SAM3 pairs,
5 min): held-out patch PR-AUC 0.97–0.98 on girl / boy / face with a clean
swapped-prompt collapse and 0.86-vs-0.03 girl/boy discrimination on pair
images. What it did *not* show is anything beyond the four trained words —
"the girl with black hair" is ignored (cos 0.99999 to "blonde hair"). The
claim of this proposal is that **the ceiling is data, not mechanism**, and the
data exists.

What this is **not**: a SAM3 replacement in mask quality. Phase 0 was SAM3
distilled onto a 32×32 grid; whatever we train next inherits its teachers'
errors. The bet is on three things SAM3 does not give us — (1) grounding that
survives the anime cases where SAM3's `girl` prompt returns nothing (headless
/ back-view figures, two-view sheets: recall on all 3 zero-box audit images in
Phase 0), (2) **booru-vocabulary prompts** (attributes, character names) via
the Qwen TE, and (3) features the tagger and adapters can consume directly,
since the tower is the same PE they already read.

## Why PE + Qwen, not the SteerViT checkpoint

Measured 2026-08-25 (memory `steervit-probe`): the shipped DINOv2-B model
steers on anime ("the girl" PR-AUC 0.91–0.94 vs SAM3 masks) but its softmax
heatmap is winner-take-all — one view of a two-view sheet, 0.00 / 0.94 mass
split on the audit's `unsure` pairs — and RoBERTa knows no booru tags. Our
recipe changes exactly those two things: **per-patch sigmoid BCE** (binary
instance masks are the supervision; one prompt may light every instance) and
**Qwen3-0.6B keys** (the text encoder the DiT is already conditioned on, so
the tower speaks the dataset's own caption language). Both are shipped in
`networks/methods/steer_pe.py`.

## Reuse inventory (live in-tree)

| Piece | Where | State |
|---|---|---|
| `SteerPE` (gated CA, connector, seg head, `set_gate_scale`) | `networks/methods/steer_pe.py` | shipped, tested (`tests/test_steer_pe.py`) |
| train + eval bench (PR-AUC / gate-0 / swapped / pair-steer / sheets) | `bench/steer_pe/run_bench.py` | shipped; single daemon job |
| Frozen PE-Spatial-B16-512 / PE-Core-L14-336 loaders | `library/vision/encoders.py` | shipped |
| Qwen3 TE loader | `library/anima/weights.py::load_qwen3_text_encoder` | shipped |
| SAM3 concept masks: girl / girl−boy / boy / person / head | `post_image_dataset/easycontrol/region/masks*` (4 488 pairs) | exist |
| Speech-bubble / text masks (SAM3 + MIT) | `post_image_dataset/masks` (873) | exist — a free "speech bubble" / "text" concept |
| Per-instance boxes + masks **with identity tags** (hair / eyes / name) | position-caption pipeline (`library/preprocess/position_captions.py`, `instance_detection.py`) | exist as a pipeline; pairs need to be dumped once |
| Caption master with booru tags for every image | `image_dataset/*.txt`, `caption_index.json` | exist |
| Multi-view audit (the first consumer) | `library/preprocess/multiview_audit.py` | shipped; witness slot open |
| Tagger (dual PE encoder, hard-routed) | `library/captioning/anima_tagger*.py` | shipped; reads frozen PE caches |

## Data plan — the actual project

Ordered by cost; each tier is a separate training mix so its contribution is
measurable.

### Tier A — in-house pseudo-labels (zero new annotation)

1. **Attribute-bound instances** (the Phase 0 gap). Dump the position-caption
   pipeline's per-instance SAM3 masks together with the identity tags the
   tagger read off each crop → pairs like ("the girl with black hair and red
   eyes", mask of *that* instance) on every multi-girl image. Also the
   reverse polarity: ("the girl with blonde hair", empty mask) when no such
   instance exists — the abstain behaviour Phase 0 lacked.
2. **Broad concept sweep with SAM3.** Run `scripts/preprocess/generate_masks.py`
   over the 3 008-image corpus (+ the `retrieved/` crawl pool) with a prompt
   list drawn from the tag vocabulary that SAM3 can ground: body parts (hair,
   eyes, hands, legs, feet, breasts…), clothing (skirt, school uniform,
   swimsuit, thighhighs…), objects (sword, umbrella, cat, food…), scene
   (window, bed, sky, text, speech bubble). Only prompts whose tag is *in the
   image's caption* are run, so the caption gates false grounding; a prompt
   that returns nothing yields a negative pair.
3. **Character names.** Same as (1) with the character tag as the prompt —
   the PODS-style instance test the paper won with descriptive prompts.

Expected scale: 3k images × ~10 prompts ≈ 30k pairs, a few GPU-hours of SAM3.

### Tier B — public anime segmentation / detection (verify each before use)

| Candidate | What it gives | Notes |
|---|---|---|
| Manga109 | bodies, faces, frames, text boxes on manga pages | licence is research-only — check; pages are the multi-view / multi-panel case we care about |
| skytnt/anime-segmentation (HF) | character foreground masks | **tried 2026-08-25 (1/10 slice, CC0): no lift** — SAM3-distill already 0.993 on its real GT; kept as eval only (`--anime_seg_mode eval`), see `steer_pe.md` |
| AniSeg | anime character + face instance masks | small; good for the instance-level face test |
| DanbooRegion | region maps for illustrations | no semantics — usable only as boundary supervision / a superpixel prior |
| animeface / anime-face-detector training sets | face boxes + landmarks | box → coarse mask; "the face" at scale |
| AnimeRun / cartoon segmentation sets | segmentation on animated frames | different style; hold out as an OOD eval, not train |

These need a one-time inventory pass (download, licence, label schema, overlap
with our booru ids — see memory `booru-id-space-collision` before joining
anything by id). Anything research-only stays out of shipped weights.

### Tier C — self-training (later)

Once Tier A+B trains, SteerPE's own heatmaps become the **box prompt** for SAM3
on the images where SAM3's text grounding failed, yielding pixel masks for
exactly the hard cases; those go back into the mix. This is the loop that could
genuinely push past SAM3's *coverage* on anime (never its mask fidelity).

## Phases, gates, kill criteria

| Phase | Work | Gate (pass → next) | Kill |
|---|---|---|---|
| **1 — attributes** | Tier A.1 only; retrain; add attribute PR-AUC + "right girl vs wrong girl" share on multi-girl held-out | wrong-girl share ≤ 0.2 of right-girl (Phase 0 girl/boy was 0.03/0.86); empty-mask prompts abstain (mean prob < 0.1) | attribute maps still ≥ 0.95 cosine to each other after 30k+ pairs → mechanism cannot bind attributes at PE-B scale; close the line |
| **2 — vocabulary** | Tier A.2–A.3; PR-AUC per concept family; zero-shot test on *held-out prompt words* (train without "umbrella", test it) | held-out-word PR-AUC ≥ 0.6 on ≥ half the families (the paper's zero-shot claim, our domain) | no zero-shot transfer → the tower is a lookup table of trained words; still usable for the fixed-vocabulary consumers, but drop the "referring" framing |
| **3 — external data** | Tier B (anime-segmentation slice: **killed for that set**, no lift); ablate each set's contribution on the Phase 1/2 metrics + Manga109 panel test | measurable lift on the hard cases (zero-box audit images, multi-panel) | no lift → in-house pseudo-labels are sufficient; stop collecting |
| **4 — consumers** | (a) multi-view audit witness + SAM3 box-prompt fallback; (b) tagger crop-free instance attributes; (c) optional pixel decoder (upsample head) for a real anime referring-segmenter | (a): audit finds the 3 known zero-box sheets and does not flip any `count-explained` row; (b): identity agreement on the audit's headless crops ≥ crop-based | per consumer |

Phase 1 is one daemon job on top of a pair-dump script; Phases 2–3 are days,
not weeks. Phase 4c (pixel decoder) is the only piece that makes this a
detector/segmenter in its own right and is deliberately last.

## Metrics that must appear in every phase

- Held-out **by artist** PR-AUC per prompt, with the two controls from Phase 0:
  gate ω=0 (how much is the frozen tower) and swapped prompt (is it text).
- **Wrong-instance share** on multi-instance images (the attribute test).
- **Abstain**: mean probability under a prompt whose concept is absent.
- **Base-tower preservation**: the tagger's held-out F1 when fed ω=1 steered
  features vs plain PE — the paper's "representation quality" axis. A tower
  that steers but breaks the tagger fails the project.
- The four audit images + `5847152`/`5847168` as a fixed qualitative sheet.

## Risks

- **Teacher ceiling.** Everything in Tier A is SAM3 output; systematic SAM3
  errors (the degenerate whole-canvas proposals documented in
  `multiview_audit.md` §5) become training targets. Mitigation: the same
  `dedupe_detections` fill-ratio fix, caption-gated prompts, Tier B ground
  truth.
- **Patch resolution.** 32×32 at 512² (16 px) is fine for "where" and useless
  for "exact boundary". Do not let a consumer that needs pixels adopt this
  without Phase 4c.
- **Cache invalidation.** Steered features are per-prompt; the tagger's
  cached PE features stay prompt-agnostic (ω=0 is bit-exact PE, tested). Any
  consumer that wants steered features pays one PE forward per prompt at use
  time — design consumers around that, never cache steered tokens by stem.
- **Boy / male recall.** 171 boy masks in Phase 0 and "the boy" fires on a
  headless-jeans view. Tier A.2 must oversample male prompts; the region
  pipeline's own note (SAM3 male prompts miss most anime partners) means the
  teacher is weakest exactly here.
- **Licences.** Tier B sets are research-licensed in part; shipped weights
  must be trainable from Tier A + permissive subsets alone.

## Relation to existing lines

- Multi-view audit: this is the fourth witness the audit doc left open, and
  the SAM3-fallback its `--part_prompts` escalation tried and rejected.
- Region task: a possible mask-fallback for `focus not found` images; not a
  paint-mask source.
- Tagger: candidate crop-free instance attributes; must not regress the
  cached-feature design.
- IP-adapter / EasyControl reference conditioning: a text-steered reference
  encoder is the long-range payoff but has no live consumer today — explicitly
  out of scope until Phase 4.
