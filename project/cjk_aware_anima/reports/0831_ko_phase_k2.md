# K2 — cache_ko + joint synthjako retrain (plan_ko.md Phase K2)

*2026-08-31, after the K1.5 round-2 sign-off. Cache staged + joint pack
trained; K3 grids pending (this report grows their verdicts).*

## Staging

`cache_ko` staged separately (44 GB, 94 GB left free — inside budget):
61,994 train + 500 holdout over `tags_ko 16,746 / tags_alt_ko 16,746 /
names_ko 16,290 / names_synth_ko 12,212` (train). JA `cache_synth3` untouched
— `distill --cache_dir` now accepts a comma list (`CachedPairs` concatenates
records, each resolving shards against its own dir; staging still writes one
dir). Equivalence smoke single==list; 23 unit tests green.

## Joint retrain (`bench/cjk_distill/results/20260831-1732-2c-synthjako-v1/`)

v4 recipe verbatim (span/global/provenance, 12k steps, batch 32) +
`--init_pack cjk_vocab_pack_synthja_v4` warm start + KO registers at
sampling 0.55 each → KO ≈ 28% of a batch (JA effective mass 87.3k under
`names_synth_ja:0.2`; KO 62.0k × 0.55 = 34.1k). `--eval_limit 1000` — the
holdout concat is JA-then-KO and eval slices the first N, so 256 would have
scored JA only. 45 GPU-min. Output `output/ckpt/cjk_vocab_pack_synthjako`.

| metric | v4 (JA-only holdout, first 256) | jako (full 1000, JA+KO) |
|---|---|---|
| final span loss | 0.0694 | 0.0868 |
| disc_far | 0.089 | 0.087 |
| tags student attn | 0.560 | 0.528 |
| tags_alt student attn | 0.520 | 0.530 |
| names student attn / recovery | 0.934 / 0.879 | 0.894 / 0.780 |
| names_synth_ja student attn | 0.680 | 0.666 |
| tags_ko / tags_alt_ko student attn | — | 0.463 / 0.468 |
| names_ko student attn | — | 0.955 |
| names_synth_ko student attn | — | 0.628 |

**Read with the sample caveat**: v4's per-register numbers come from the
first 256 holdout pairs, jako's from all 500 JA (+500 KO) — different draws
of the same holdout, so small JA deltas are partly composition. `names`
0.879→0.780 recovery is the one worth watching; the K3 same-seed grid is the
actual gate. KO `tags_*` student ≈0.46–0.47 lands near the JA tags band
(0.52–0.56) from rows that were 0.061 zero-shot at K0; the low KO "teacher"
column (0.08) is expected — the stock-spiece teacher is unk-walled on hangul,
which is why `recovery` >2 there is not comparable to JA.

## K3 grids (running)

- `2c-synthjako-ja-grid` (`20260831-1827`) — **metric-clean vs v4**
  (`20260831-1309`, same seed): per-prompt `ja_ext` cos_vs_en deltas all
  within ±0.013, both directions, no register systematically down;
  discrimination `ja_ext` 0.129→0.128 (p1-vs-p2 0.248→0.240), `en`/`ja_t5en`
  bit-identical as expected. Render eyeball (gate 2 proper) pending user.
- `2c-synthjako-ja-mixed-grid` (`20260831-1843`) — **metric-clean vs
  `20260831-1324`**: worst `ja_ext` delta −0.010 (r3_reimu_full), all others
  ≤±0.010; `ja_t5en` bit-identical; discrimination `ja_ext` 0.130→0.129.
- `ko-k3-recovery-grid` (`20260831-1848`) vs K0's `20260831-1346` yardstick:
  **do not read the flat `cos_vs_en` as recovery** — the shipped JA v4 pack
  sits at the same flat level (0.05–0.18) on its own grid and renders fine;
  the student's tokenization differs from the reference by construction (the
  same objection that demoted L_flat). What the grid does show: `ko_ext`
  moved off the K0 floor on **every t\* prompt** (0.061→0.105 mean, +60–90%
  per prompt; t5 0.074→0.138), q\* 0.085→0.112, while s\* prose stayed at
  floor (untrained register, expected) and n\* stayed flat (full-KO names =
  expected fails, plan §5b territory). Discrimination `ko_ext` 0.144 stays
  healthy (en 0.110). The honest recovery instrument remains the holdout
  attn readout (tags_ko 0.46–0.47, just under the JA tags band 0.52–0.56)
  and the **rendered t\*/c\* grid — gate 3 is the user eyeball on
  `ko_ext` vs `ko_t5en` there.**
- G1 EN bit-exact: unit suite green (construction-level, pack-independent).
