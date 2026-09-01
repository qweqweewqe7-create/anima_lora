# CJK-aware Anima — word-row minting (ko3 / "plan3", 2026-09-01)

*Follows [`plan_ko2.md`](plan_ko2.md). Attacks the layer the corpus levers
cannot reach: the frozen consumer (llm_adapter + DiT cross-attn) converges in
embedding space but never **composes** a multi-row CJK surface into one
rendered identity (plan3-adapter verdict "metrics restorable, renders never
compose"; ko2's 쌍둥이/n1 misses). Minting sidesteps composition instead of
teaching it: one appended ext row per curated surface, emitted as a single
query slot by greedy longest-match, distilled with the existing span loss
while every base row stays frozen.*

## Smoke evidence (2026-09-01, all runs same-day)

Two iterations on 10 KO surfaces (레이무/하쿠레이/동방프로젝트, 아스카/소류/
랑그레이/에반게리온, 쌍둥이, 무녀복, 커플룩), 1,273-pair filtered corpus,
base 58,968 rows frozen via `--tunable_rows_from`:

| condition (identical span set, identical loss) | span loss |
|---|---|
| pre-mint — spelled-out encoding, joint synthjako2 pack | 0.325 |
| mint init — pooled mean of constituent rows | 0.561 |
| mint trained — smoke2, `--span_focus_from`, 3k steps | **0.230** |

(`bench/cjk_adapter/measure_minted_span_baseline.py`; envelopes
`bench/cjk_distill/results/20260901-1418-2c-mint-smoke2`, grids
`bench/cjk_adapter/results/20260901-14*-mint-smoke*-grid`.)

**F1 — mechanics are clean.** Word rows fire only on their surfaces: 17/20
grid prompts render **pixel-identical** to the joint pack, EN encode path is
bit-exact by construction (word match lives inside CJK runs only).

**F2 — loss focus is mandatory.** smoke1 trained the minted rows under the
ordinary batch loss: flat (0.198→0.201) — minted spans are ~10% of the span
mass, and the visible render changes came mostly from the slot-count change,
with faces degrading. `--span_focus_from` (zero background span weights)
turned the same rows into real learners: 0.500→0.189 train, renders healed.

**F3 — a trained single row beats the composition it replaced** in embedding
space: 0.230 vs 0.325. The pooled init does not (0.561) — training is what
earns the row, mean-init is not a shortcut.

**F4 — renders: attribute binding materializes; identity needs data; abstract
concepts overshoot.**
- n2 아스카 (211 pairs): the strongest result — joint's "three schoolgirls"
  became a clean single girl, red twin-tails, serious expression. Not yet the
  canonical plugsuit Asuka, but every prompted attribute binds.
- t5 쌍둥이 (1.8k pairs): "matched pair" now binds (mirrored, same design) but
  the pair rendered as robots — the row over-shot off the human manifold.
- n1 레이무 (18–27 pairs): no identity movement. Consistent with the visit
  floor: the row trained (loss fell) but had ~25 distinct contexts.

## Phases

**M1 — data densification per target (the n1 fix).** The `synth_names.py`
machinery exists for exactly this: caption-swap minting to a visit floor.
Generate ≥~200 caption-swapped pairs per minted *name* (레이무 tier), and
widen pair coverage for minted *loanword tags* (그레이스케일 has 1 pair —
below any floor). Re-run the n1/n2 probe; gate: n1 moves or the name tier is
declared data-bound at a measured floor.

*RUN 2026-09-01 (`2c-mint-m1`) — data works where the target is an
attribute/style; identity is blocked by drift, not data.* Eojeol guard landed
first (`is_hangul_char` + boundary check in `_encode_cjk_words`, unit-tested);
그레이스케일 minted as an 11th row; corpus densified via `synth_names --only`
(레이무 27→203 pairs) and the new `synth_tags --lang ko --rows-from`
(그레이스케일 1→301, 커플룩 2→301, 무녀복 10→311; register `tags_synth_ko`);
`mint_corpus.py` commits the smoke's filter rule (2,348 pairs; the spaced wiki
label 동방 프로젝트 respaced to the eval surface 동방프로젝트 in synth pairs
only). Readouts (envelope `20260901-1507-2c-mint-m1`, grids `1518/1529/1531`):

- Embedding: F3 bar holds at scale — trained 0.227 vs pre-mint composition
  0.321 (1,326 pairs); per-surface: reimu family 0.202 vs 0.227, tag tier
  0.741→0.218. Every minted row learns.
- Structure: EN + all 16 non-target KO prompts **pixel-identical** to the
  shipped synthjako2 grid; diffs are exactly the 4 minted-target prompts.
- Renders: **c3 그레이스케일 binds** (monochrome 90s style — the 1→301
  densification did it); t5 recovers *human* twins (robot overshoot gone),
  matching semi-binds (color-swapped outfits). **n1 does not land identity**
  at 203 pairs and — the load-bearing observation — ko_ext renders drift
  off-manifold (sketch/chibi/doodle styles) across seeds 42/7/1234 while the
  EN arm renders canonical Reimu at the same seeds. n2's smoke win also
  destabilizes across seeds (blonde sketch / chibi sticker).

Verdict: the n1 miss cannot be declared data-bound — **drift confounds it**.
The smoke's t5-only M2 diagnosis generalizes to the name tier: pure span
focus (background weight 0) lets the 11 rows push any render they touch off
the manifold. M2 is therefore the blocking phase, and the name-tier floor
question is re-judged after an anti-drift arm on this densified corpus.

**M2 — anti-drift (the t5 fix).** Candidates, cheapest first: (a) mixed
focus — background spans at small weight (0.05–0.1) instead of 0, so the
surrounding scene anchors the row on-manifold; (b) init-anchor penalty
(`‖row − init‖`); (c) cap per-row lr/steps. One arm each on the smoke corpus,
gate on t5 rendering *human* twins while n2 keeps its win.

**M3 — scale.** Only after M1+M2 gates: KO name families (500) + a curated
loanword-tag list, allocation visit-floor-driven exactly like
`synth_names.py`. JA (3,000 families) after KO validates — JA has no spaces,
so longest-match needs a boundary guard (see risks).

**M4 — gates (K3 discipline: multi-seed before believing any single render).**
1. EN bit-exact — structural, but keep the test.
2. Non-target KO prompts pixel-identical to the shipped pack (the F1 control,
   now as an explicit gate).
3. JA grids untouched (word rows cannot fire on JA text unless minted).
4. Per-target render probe: each minted name/tag gets its eval prompt;
   ≥3 seeds on any marginal call.
5. Embedding: minted-span loss ≤ pre-mint composition baseline (the F3 bar).

**Ship shape.** Word rows ride the existing pack format — appended
`ext_embed` rows + `mapping["word"]` — so `run_bench`/cache/distill already
consume them (`HybridT5Encoder.from_mapping`). Integration item before ship:
verify the production loaders (inference.py sidecar auto-discovery, ComfyUI
adapter node) rebuild the encoder from the pack JSON rather than a stale
mapping copy; if any consumer hardcodes 58,968 rows, it fails loud on load —
audit first.

## Risks

1. **Longest-match overtriggering** — a minted surface matching inside an
   unrelated longer word. KO: require the match to start at an eojeol
   boundary (space/punct/BOS) — particles attach only at the end, so this is
   cheap and sufficient. JA: no spaces; defer until the KO gate passes, then
   guard with the name lexicon's own tokenization (match only when the
   surrounding chars are not mid-word kana/kanji runs — design owed).
2. **Abstract-concept rows** (쌍둥이 tier) may be net-negative even after M2 —
   scope discipline: names and loanword tags are the shipped tier; abstract
   tags stay experimental until a render gate passes.
3. **Row budget** — trivial (+10 rows ≈ 40 KB); thousands of families ≈ a few
   MB on a 241 MB pack. Not a constraint.
4. **The smoke's thin-data result (n1) might be a floor, not a lack** — M1
   exists to distinguish those; if 200+ pairs still do not move identity, the
   name tier hits the same consumer wall and only C (glossary substitution at
   inference) remains for names.

## Relation to standing verdicts

- plan3 adapter-LoRA stays closed (this trains rows, not the adapter).
- JESC/STAIR/D2/desc_ko prose verdicts untouched — minting is span-shaped by
  construction.
- The C fallback (inference-time glossary substitution of known tags to EN
  tokens) is *complementary*, shares the longest-match layer, and remains the
  zero-training floor if M1 fails.

## Code landed with the smoke (2026-09-01)

- `bench/cjk_adapter/ext_vocab.py` — `mapping["word"]` + `_encode_cjk_words`
  greedy longest-match (backward compatible; no word map = bit-identical).
- `bench/cjk_adapter/mint_words.py` — mint rows onto a trained pack.
- `scripts/distill_cjk/config.py`, `distill.py` — `--tunable_rows_from`
  (freeze base rows), `--span_focus_from` (concentrate span loss on minted
  spans).
- `bench/cjk_adapter/measure_minted_span_baseline.py` — the three-condition
  span-loss comparison.

~~Note the eojeol-boundary guard (risk 1) is NOT yet implemented — M1 must not
scale the word list before it lands.~~ Landed 2026-09-01 with M1
(`ext_vocab.is_hangul_char`, boundary check in `_encode_cjk_words`,
`tests/test_cjk_distill.py` word-match tests).
