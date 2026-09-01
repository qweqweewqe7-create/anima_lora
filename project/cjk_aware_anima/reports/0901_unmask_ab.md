# Unmask A/B — text masks off + OCR-quoted captions (2026-09-01, PROVISIONAL)

**Status: provisional — s42 readouts only; s7/s1234 renders still in the
daemon queue. Verdict to be confirmed or revised on the full grid (K3).**

## Goal (the line's final reframe, set this day)

The CJK line's ultimate criterion is **not** KO-prompt UX and not a
text-rendering feature: it is that **manga data trains healthily with the
text masks OFF** — captions that carry the in-image text make text pixels
attributable, so unmasked training neither corrupts general quality nor
teaches unconditional text spam. MT/substitution is not a direction for this
goal (it destroys the surface), and the coarse-tag arm (`japanese text` alone,
unmasked) was already tried before this line and lost to masking — settled,
do not re-run. The loss curve cannot see this property; the readout is
render-level (spam rate + adherence + quality), never fm loss.

## Design

Shard: `sincos` (350 imgs, 133 with text masks; heavy typeset text, some
pages Chinese-typeset). Plain LoRA dim32 / lr 2e-5 / 8 epochs / batch 1,
identical in both arms; latents shared bit-identically.

| arm | loss mask | captions | TE encode |
|---|---|---|---|
| A | text masks ON (production) | production captions | stock caches |
| C | **OFF** (`masked_loss=false`) | + OCR quote tags (73 imgs, 148 lines) | **synthjako2 ext pack** re-cache |

`base` = no-LoRA reference renders.

Pipeline (all landed this day):

- `datasets/ocr_text_captions.py` — mask-complement CCs, dilation-merged to
  bubble level, padded crops → batched manga-ocr with mean-token logprobs
  (reuses `manga_text.MangaOCR`), gates: logprob > −0.3, area ≥ 400 mask-px,
  junk regex. Output: sidecar captions + `ocr_records_sincos.jsonl`.
  Quality after bubble-merge: median line 12 chars, full sentences
  (`「いや、ちがう...八奈見さん、ここじゃまずいって!」`).
- `datasets/cache_te_ext.py` — temp mirror (symlinked resized imgs, corrected
  captions + variants with quote tags appended via `parse_caption` /
  `compose_caption`), then the standard `cache_text_embeddings` stage with a
  tokenize strategy routing the T5 side through `HybridT5Encoder` and the
  synthjako2 rows appended to the adapter embed. 351 caches; sanity: an OCR'd
  caption fires 35 ext rows; CJK-free captions encode bit-identically to
  stock (copied-through imgs are an in-run control).
- `configs/gui-methods/custom/cjk_unmask_{a,c}.toml` — the two arms;
  arm C redirects `text_cache_dir` only (latents shared with A).
- Eval prompts: `assets/unmask_eval_prompts.txt` — 8 text-free rows
  (row 8 `comic, 2koma` asks layout, still no text), rendered per arm at
  seeds 42/7/1234 (`output/tests/cjk_unmask_eval/`).

Both training runs cache-verified at launch (arm C log: "Skipping Qwen3 …
all text-encoder outputs cached", zero latent re-encodes — the ext
conditioning is what trained). Note inference conditioning is *identical*
across arms (stock encoder, no CJK in eval prompts): every A↔C render
difference is weights-only.

## s42 readouts (2 of 8 rows inspected; single seed — treat accordingly)

- **Row 6, maid/cafe** (model eyeball): tag adherence perfect in all three
  arms. base **spams large decorative gibberish latin signage** on a
  text-free prompt ("CIRGO MO, TAFE"); arm A ≈ base (same spam, near-same
  composition — masked training leaves the base's text habit untouched, as
  expected). **Arm C drops the decorative spam**: text only as small
  chalkboard-menu scribbles (diegetic), and the cafe interior is the most
  fully realized of the three. Composition diverges from base far more than
  A does — consistent with unmasking admitting more training signal.
- **Row 7, upper body/portrait** (user eyeball): base and arm A both
  hallucinate **cat ears** not in the prompt; **arm C does not** — the
  adherence win at this seed is C's.

## Provisional verdict

At s42, the unmasked + OCR-captioned arm shows **no quality degradation and
no text-spam increase** — the two failure modes the settled coarse-tag
experiment produced — and on both inspected rows it *beats* A on spurious
content (decorative gibberish, hallucinated cat ears). This despite the
enablers being individually mediocre: OCR lines are gappy fragments of the
real pages (60→73/133 imgs pass gates), and the synthjako2 pack's semantic
performance on names is known-weak. The bar was "not worse than masking";
s42 reads "possibly better".

## Caveats / owed before a real verdict

1. Single seed, 2/8 rows read closely — K3 says multi-seed before believing
   any render claim. s7/s1234 pending in the queue.
2. No quantitative spam count yet (tagger-judge pass deferred by user call —
   eyeball first).
3. Positive control unrun: does C render *asked-for* text
   (`japanese text, 「…」` prompt through the ext encoder — needs merge or
   run_bench-style encode)? Not required for the unmask-health gate, but it
   is the attribution mechanism's direct signature.
4. Confound to keep honest: C differs from A in TWO ways (mask off AND
   captions/encoding). If the full grid confirms health, the decomposition
   arm (mask off + production captions) is what the settled coarse-tag
   experiment already approximates — but on *this* shard/recipe it was not
   re-run. Any surprise should re-check against that.
5. sincos has Chinese-typeset pages; OCR wrote them as kanji strings. Fine
   for attribution, wrong as transcription.
