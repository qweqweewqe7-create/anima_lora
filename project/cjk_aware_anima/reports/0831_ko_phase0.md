# K0 — KO gap sizing (plan_ko.md Phase K0)

*2026-08-31. One CPU pass + one daemon grid on the shipped `synthja_v4` pack.
No go/no-go — this fixes the yardstick for K1/K2. Run:
`bench/cjk_adapter/results/20260831-1346-ko-phase0-grid/` (80 renders, seed 42,
30 steps, cfg 3.5). Prompts: [`assets/ko_eval_prompts.json`](../assets/ko_eval_prompts.json)
— the 20 JA ids in Arca-Live-register KO; per-row wording decisions recorded in
the file's `_comment` (소녀 1명 over latin `1girl`; 갑옷 chosen, 아머 axis
deferred to the K1 glossary review).*

Arm naming: the plan's `ko_unk` is the bench's `ko_native` (stock T5),
`ko_ext` = synthja_v4 pack with zero-shot hangul rows, `ko_t5en` = teacher.

## 1. Adapter-output readout (`cos_vs_en`, mean per register)

| register | ko_native (unk) | ko_ext (zero-shot) | ko_t5en (teacher) |
|---|---|---|---|
| t* tags (6) | 0.088 | **0.061** | **0.806** |
| s* prose (5) | 0.090 | 0.062 | 0.734 |
| q* quotes (3) | 0.115 | 0.085 | 0.821 |
| n* names (3) | 0.100 | 0.074 | 0.815 |
| c* comp (3) | 0.088 | 0.087 | 0.795 |
| **all (20)** | 0.094 | **0.072** (range .047–.106) | **0.786** |

Discrimination (mean over all prompt pairs; ~1.0 = collapsed):
`en` 0.110 · `ko_t5en` 0.111 · `ko_ext` **0.146** · `ko_native` 0.372.

Reading:

- **The hangul rows are inert, as plan_ko.md expected** — `ko_ext` cos 0.072,
  far under the 0.3 bar that would have let K2 start from a smaller corpus.
  The JA-trained `param=global` correction does **not** transfer semantic
  alignment across scripts to rows it never visited. → **Full K1 corpus at the
  planned ≈60k-pair shape.**
- One surprise vs the plan's "disc ≈ 0.9" guess: `ko_ext` discrimination is
  *healthy* (0.146 ≈ the EN baseline's 0.110). Anchor-init rows already carry
  prompt-*specific* signal — it just points nowhere semantically. The failure
  is alignment, not collapse; consistent with findings §1 ("the gap is
  contextual, not map quality").
- `ko_t5en` 0.786 sits in the JA teacher band → the same teacher ceiling
  applies; the K3 recovery gate target is unchanged.
- `ko_native` disc 0.372 is only a *partial* wall (digits/latin survive the
  stock T5; unk = 9–17 of ~23–43 nonpad).
- Eyeball (t1): `en` and `ko_t5en` render the same faithful
  classroom/sailor-uniform image; `ko_native` and `ko_ext` both fall to
  generic off-prompt multi-character scenes.

## 2. Coverage baseline (`gates/coverage.py --lang ko`, floor 5)

Against `pairs_synth.jsonl` registers `tags,tags_alt` (33,766 pairs, 18,957
unique span texts, 3,469 / 58,968 rows visited): **every KO content row in all
20 prompts is at 0 visits** (19–40 ext rows per prompt, 100 % v=0), and
`unk=0` everywhere — hangul routes cleanly to ext rows. This is the number K1
must move; the K1 exit gate is "no user-facing KO tag token under floor".

(`coverage.py` gained a `--lang` flag for this — it hardcoded `entry["ja"]`.)

## 3. Spacing probe (plan risk 2) — CLEARED

`검은 머리` encodes to the identical ext-row sequence (검 58475 / 은 32388 /
머 58865 / 리 32285) bare, after `", "`, after a latin tag (`1girl, `),
mid-list, and in prose — separators land on T5-side `▁ ,` pieces and the
intra-tag space folds into the run without perturbing the hangul rows. So
`tags_ko` trains exactly the rows users hit. Side-confirmation of the plan's
tokenisation note: KO content words are char-row heavy (`머리` = 머+리;
whole-syllable pieces like 소/은/이 come from the T5-overlap band).

## Verdict

K0 done. Zero-shot KO ≈ the JA Phase-0 floor; teacher path healthy; rows
addressable and untrained. Proceed to **K1 at the full planned corpus shape**
(lexicon rebuild `--langs ja ko` → glossary `--lang ko` with mandatory `--mt`
→ human review → pairs + capped `names_synth_ko`), then K2 joint retrain
(`synthjako`, KO 25–30 % sampling, separate `cache_ko`).
