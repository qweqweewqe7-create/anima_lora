# CJK-aware Anima — findings

Every settled verdict of the line in one place, with the evidence pointer.
Read this before proposing anything: most of the obvious levers are measured
and closed. Companion files: [`deliverables.md`](deliverables.md) (what exists
and where) · [`plan.md`](plan.md) (what remains). The dated reports carry the
full tables: [`reports/0816_phase2.md`](reports/0816_phase2.md) (Phase 2b gates +
2c first pass + corpus work), [`reports/0827_names_synth.md`](reports/0827_names_synth.md)
(name register, §8 JA-context, §9 attn term),
[`reports/0830_adapter_lora.md`](reports/0830_adapter_lora.md) (plan3 adapter
LoRA, closed). Dataset-side numbers: [`datasets/README.md`](datasets/README.md).

## 1. The problem and the opening (Phase 0 probe, 2026-08-15)

Anima conditions on two streams: Qwen3 hidden states (content) and a T5-id
query table that cross-attends into them through the 6-block LLM Adapter. Qwen
reads JA natively; the T5 side is a stock EN SentencePiece that collapses
native JA to `▁ <unk> </s>` — conditioning dies with it (cos ~0.02 vs the
all-EN reference, discrimination 0.91 = no prompt-specific signal). Feeding the
T5 side an **EN translation** of the same JA prompt recovers cos 0.69–0.76 with
healthy discrimination (~0.11). So the T5 stream needs to be *semantic*, not
Japanese, and the broken piece is small, isolated, and has a working reference
(`t5en`) to imitate. `bench/cjk_adapter/results/20260815-1836/`.

**Ruled out by the probe** (do not reopen): romanization (`t5rom` cos ~0.06 —
non-unk is not enough); reverse routing (EN on Qwen, JA on T5: cos ~0.07/0.15,
disc 0.92 — Qwen is the content channel); cleverer zero-shot maps (a 0.75
anchor-holdout cos still gave ~0.05 end-to-end — the gap is contextual, not
map quality, which is what makes training necessary); a fresh SentencePiece
over T5 (borrowing Qwen tokenization keeps source/query streams token-aligned
over CJK spans); mT5 / any vocab swap (ext rows are already Qwen BPE pieces;
rare names fall to char pieces in every tokenizer).

## 2. The design that holds (Phase 2b, settled 2026-08-16)

Ext vocab = Qwen's CJK tokens mapped into the T5 embedding space (anchor
init), appended as new rows; distill the new rows so the student
`adapter(qwen(ja), t5ext(ja))` matches the teacher `adapter(qwen(ja), t5(en_mt))`.
Shared Qwen side isolates the broken piece; trainable surface = new rows only,
so **EN prompts are bit-identical by construction** (G1, unit-tested).

Settled by gates G0b/G0/G1/G2/G3/G4 and confirmed at 2c scale:

- `param=global` — a shared low-rank + per-dim diagonal + scalar-gain
  correction over the ext rows; `global_row`'s 1,892 free rows buy nothing
  end-to-end.
- `loss=span` — segment-mean cosine per aligned span. Flat cosine is a
  **control, not a gate**: G5's oracle argmin of the span objective scores 0.13
  on the old `cos_vs_en ≥ 0.6` bar, and both flat probes showed buying flat
  points costs readout-space alignment (`flat`-trained near-disc 0.914).
- The readout that measures anything is the **real-query attention readout**
  (`build_query_bank.py`, DiT image-token queries at 2–3 σ, centered by
  `fit_centers`); random query directions are refused. `recovery_attn` is a
  **mix statistic** — the readout floor is register-dependent by 100× (G3).
- Teacher ceiling for `tags` is 0.823 readout recovery; the design reaches
  ~87% of the addressable signal on the corpus holdout.
- Trust hedging is a non-lever (G4b): dropping `mt_unverified` spans is
  *worse* on every column at 10⁴ pairs — noisy supervision beats none.
- Cost model: the surface saturates in ~20 GPU-min; the binding constraint is
  span-carrying ext rows, never compute.
- Two contracts: max-pad 512 with no `crossattn_seqlens` masking on both arms;
  GPU work through the daemon.

## 3. What the render grid taught (Phase 2c, 2026-08-16 → 08-17)

The 20-prompt same-seed grid (`en / ja_t5en / ja_ext`) splits exactly along
**supervision density**: the teacher is at EN parity everywhere; the student
transfers high-visit tag content (t1 school, t2 maid) and collapses on
thin/zero-visit content (t3 armor `鎧`:0, names `博`:2, prose function words
0 by construction). `gates/coverage.py` is the diagnostic; identity-carrying
tokens want O(100+) visits (`教室`:39 renders a classroom, `霊夢`:37 does not
render Reimu).

Corpus levers, each measured once and closed:

| lever | result |
|---|---|
| **D1-wide** (3,008 → 16,128 captions, 45,230 pairs) | buys **visits, not vocabulary** — 500+ band 381 → 756, rows visited flat at ~6,400, no `v=0` token moves. More captions multiply the same glossary. |
| **D1-pairs tail fill** (`danbooru-ja-tag-pair`, CC0) | buys vocabulary: 5,248 tags filled, unmapped segments 42,530 → 13,714. |
| **D1-pairs item 2** (community names as arbiter candidates + widened `--mt` rebuild) | 4,438 wordings moved, 0 pinned regressions, unmapped → 878 (−94%); grid moves on the coverage-bound prompts. The **polysemy class** (`bow`→蝶結び back-translates to the sense, not the string) is unwinnable by F1 → human review. |
| **D1-words** (katakana loanword chosen over native kanji: `armor`→アーマー not 鎧, 119 tags) | half are Chinese and correctly rejected (`bed`→床 = *floor*) → a human review axis, not automatic. |
| **D2** commentary (73k native JA, 9,068 paired) | span-less → **inert under `loss=span`**; +13% only in its own register under `attn`. A register/promo-filter problem, not access. |
| **manga_text** OCR corpus | rejected as distill material (duplicates D4, OCR noise arrives MT-laundered); kept for the glyph line's geometry. |

Invariants the glossary work established: **verified ≠ Japanese** (棕毛 /
藍眼睛 back-translate perfectly and are Chinese — keep both script filters and
kana-first ranking); **selection is pure post-processing** (`--reselect`, ~1 s);
**a glossary rebuild requires `--mt`** (the CPU-only path drops the
back-translation layer and regressed 1,991 wordings — tried and reverted);
leaving danbooru tags latin is strictly worse (routes to original spiece rows,
trains *no* ext rows); Wikidata covers 0/89 artists (handles are not entities).

## 4. Names: the failure that closed two lines (2026-08-27 → 08-30)

Question: can text pairs alone make a rare kanji character name render in a
full-JA prompt? **No — with every lever measured.**

| arm | what it tested | full-JA Reimu | verdict |
|---|---|---|---|
| `names` register | pin names that occur in captions | ✗ | `博麗霊夢` occurs 3× in 60k; nothing to pin |
| `synth` (EN-context) | 177k minted pairs, rarest kanji → 300 visits | ✗ (r1 mixed ✓) | rows learn **context-specifically** |
| `synth_bal` | rebalanced draw | ✗ (r1 ✓) | same pack, only neighbour language differs |
| `synthja` (§8) | 261k pairs, visits bought *in JA context*, coverage complete (Asuka 0 under-floor rows) | ✗ | "thin visits" **falsified** for names |
| `synthja_attn` (§9) | `attn:1.0,span:0.5` sequence objective | worse; r1 gain lost | objective lever **spent** |
| `lora16` (plan3) | rank-16 ext-gated LoRA on adapter self-attn q/k/v/o + cross q | ✗; r1 regresses | capacity real but spent on *smearing* (recovery_attn 0.90 → 0.45) |
| `lora16_reg` (plan3) | + `attn:0.25` regulariser | ✗; r1 hair pink | every metric restored (recovery_attn 0.96, names closer to teacher than rows-only), render moves halfway back to `synthja` |

Reading: the rows learn (recovery ≥ 0.90 in every arm) and the adapter has
composition capacity (poses, Miku, Asuka's suit, no strays in `lora16`) — but
**nothing in the distillation target contains the composition of a rare kanji
name**. The teacher is the frozen adapter reading EN pieces; matching it per
span or per token on 3k synthetic names in swapped-in contexts does not
transfer to `博麗霊夢` in an all-new-row context. Miku works in full JA because
`ミク` is visited inside real JA tag captions. What still works everywhere:
tag registers, quotes (`quote_preserved` cos 0.988), katakana names, and the
**mixed register** (JA name + EN tags, r1) — which is what users actually type.

Do not re-propose for this failure: more pairs / JESC / STAIR / D2 growth
(inert under span, and §8 falsified visits), mT5, rank/MLP/token-gate/attn-
weight sweeps (no instrument can rank them — §5), or 2-iii full adapter
finetune (forks the adapter for every user, gives up the EN guarantee).
The only unexplored lever is a **different target**: a real-caption corpus
where the name co-occurs with its own attributes, or a DiT-side signal.
That is [`plan.md`](plan.md).

## 5. Metrics: what can and cannot see the name failure

- `recovery_attn` is **saturated and blind in both directions** — 0.90 (rows)
  and 0.96 (reg) render the same missing Reimu; 0.45 (`lora16`) rendered a
  different wrong one. The eyeball grid is the gate; distill metrics are
  health checks.
- The **adapter-space name residual** (`residual_probe.py`: `Δ = pool(full) −
  pool(name-stripped)`, margin = own-character cos − best other) *does* witness
  "no identity was written": Reimu's student margin is 0.03–0.07 in every arm
  while the teacher separates the three characters at 0.78 and Miku sits at
  0.37–0.46. Spearman ρ 0.72 / AUC 0.94 against the eyeball labels
  (`assets/grid_labels.json`, 31 points) — but **0/6 within-prompt
  concordance**: the LoRA arms raise the residual while the render worsens
  (Goodharted by the very objective). So: a cheap **floor gate** (margin ≈ 0 ⇒
  don't render) and diagnostic, never an arm selector.
- Arm-vs-arm selection needs a **DiT-side read**; the turbo-4-step + Tagger
  render scorer is the one to build (velocity probe unnecessary if it lands).

## 6. Glyph side (plan2 Phase 0, 2026-08-17)

Char-row separability measured on the union of qwen 1-char rows + char-map
rows (39,632 rows): the common-kanji class is cleanly separated and
`param=global` *widens* it (13/15 confusable pairs more separated post-training;
kanji pairs >0.9 drop 93,786 → 51,588). One real collision class was an **init
artifact**: char-map rows were the *mean* of UTF-8 byte-token embeddings, so
527 char pairs whose byte triples are permutations (鯰/鰯) were bit-identical.
Fixed at the source (position-weighted pooling for colliding multisets, 527 →
0 duplicates); every pack trained from `synthja` on starts from the fixed
asset. VAE glyph ceiling and register census were skipped by user call
(qwen-image proves the VAE family carries glyphs). `assets/separability_phase02*.json`.

## 7. Deployment facts (settled, not yet built)

The artifact **cannot be a LoRA** — new rows appended to `llm_adapter.embed
[32128, 1024]` (a shape change) plus a tokenizer mapping (behaviour, not
weights). It ships as `ext_embed.safetensors + .json` (release asset, CNS-γ
pattern). Bake-in into a forked DiT is rejected (breaks stock ComfyUI, which
hardcodes 32128 in `comfy/ldm/anima/model.py`, and still can't carry the
tokenizer). ComfyUI needs one node wrapping the CLIP's t5xxl tokenize path +
an object patch on the adapter embed (forward-hook-not-override invariant);
endgame is upstream to core. Details in [`deliverables.md`](deliverables.md#ship-contract).
