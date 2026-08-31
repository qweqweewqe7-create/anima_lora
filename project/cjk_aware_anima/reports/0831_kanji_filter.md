# 0831 — allowed-kanji filter (joyo+jinmeiyo+whitelist) over the JA corpus

*User call: limit JA to 상용한자/인명한자. Measured first, then shipped as a
veto-only guard with a reviewed whitelist. Code: `datasets/kanji_allow.py` +
`tag_glossary.han_allowed`/`is_japanese`; mechanism note in
`datasets/README.md`.*

## Census that shaped the design (63k-pair corpus, pre-filter)

- Kanji outside joyo+jinmeiyo: **1.01% of occurrences, heavy-tailed** — top-5
  = 71%, top-100 = 96% of the out-of-set mass.
- The **head is community register**, not noise: 膣 7134×, 掴 6334×, 舐, 跪,
  肛, 狐, 繋 all clear the ~300-visit render floor. A raw joyo/jinmeiyo cut
  would delete the corpus's own vocabulary (and `博麗霊夢` — the canonical
  never-composes name — is 100% joyo, so the cut would not touch the actual
  name failure).
- What the boundary *does* catch: zh leakage the old guards missed — kana-
  bearing zh wordings passed on their kana (结月ゆかり), and Shift-JIS
  encodes some simplified forms (崩坏's 坏, 海梦's 梦 are JIS level 2).

## The shipped rule

`ALLOWED` = joyo 2,137 + jinmeiyo 815 (permissive page scrape, variants in)
\+ `HYOGAI_KEEP` 63 (top-100 out-of-set minus 37 hand-classified zh chars)
\+ `NAME_RESCUE` 8 (tail chars carried only by real names: 燐朧驤棠掟杠鰐鼠)
\+ `EVAL_KEEP` 俯瞰 (§5a eval-pinned vocab, invisible to the census at 0 visits)
\+ `DIFF_RESCUE` 47 (audited from the first run's full glossary diff — NSFW/core
vocab 痙攣/叩/嚢/繍/咳/絨毯/筐/墟/牢/榴/翅/憑…, official-JA names
碇/竈/謐/聲/奘/霍娥…, and HSR/WuWa JA localizations that keep hanzi 霄霖瑾芬)
= **3,072 chars**.

Semantics: **veto-only, wording-level, at glossary/fill selection** —
`han_allowed()` rejects a candidate only for a disallowed Han char; latin and
symbol wordings (`:d`, `!?`, `OL`, `3D`) pass vacuously and stay untranslated
(user call, see memory `feedback_emoticon_tags_stay_latin`). `is_japanese()`
keeps the strict form on name paths only. A rejected wording falls to the next
source or EN passthrough — never row deletion, so coverage is unchanged.

## Result (rebuilt 2026-08-31, chain: glossary reselect → build_pairs →
synth_names → synth_tags, all daemon jobs)

- **Glossary diff vs signed-off build: 22 wordings** — all zh removals or
  improvements: 崩坏→崩壊, 崩坏3→崩壊3rd, 僵尸→キョンシー, 香取バツーン赛→
  鹿取抜雲斎, 波可娜・费雷尼→プルクラ・フェリーニ, 冰螢術士→氷蛍術師,
  鐵血工造→鉄血工造; 机械腿/米哈游/无限大/同班同學2 → EN passthrough.
  **0 emoticon/latin tags touched; 0 general-axis regressions.**
- **Corpus census: 0 out-of-ALLOWED occurrences** in pairs.jsonl (63,241,
  D2 intact at 9,068), pairs_synth.jsonl (238,455) and pairs_synth_tags.jsonl
  (240,674). ext rows visited 6,545 (was 6,634 — the delta is the zh rows).
- Spot-restorations all hold: 聲の形 (fallback was zh 声之形), 碇シンジ
  (fallback was ミサシン), 竈門禰豆子, 静謐のハサン, 手榴弾 (fallback was zh
  手雷), 痙攣, 刺繍, 飛霄.

## Known limits (by design)

- A zh wording built entirely from allowed chars (馬剃天愛星, 声之形,
  黑曜星魔 — 黑 is a jinmeiyo variant) passes the char filter; the kana-first
  ranking catches most (声之形 loses to 聲の形), the rest are override
  material. `itzpapa (genshin impact)` → 黑曜星魔 (1×) is the one such row in
  this diff — override candidate.
- Tags whose only JA wordings were zh now pass through EN (otter costume,
  mechanical legs, leona (league of legends)); proper JA wordings for these
  are `tag_overrides.json` material, not filter material.
- Retrain: `2c-synthja-v4-kanjifilter` → `output/ckpt/cjk_vocab_pack_synthja_v4`
  (v3 recipe: span/global/trust-provenance, 12k steps, register sampling
  `names_synth_ja:0.2,names_synth:0.5`, span scale `names_synth:en_pinned=0.3`,
  cache_synth3 restaged). Acceptance = the 2c surface as in plan §3-2.

## Distill result (`bench/cjk_distill/results/20260831-1221-2c-synthja-v4-kanjifilter/`)

Holdouts are different draws over different corpora — read directions, not
decimals (train pairs 227,230 vs v3's 251,651; the delta is the filtered rows).

| metric | v3-tags | **v4-kanjifilter** |
|---|---|---|
| final span loss | 0.0892 | 0.0694 |
| held-out span | 0.086 | 0.093 |
| disc_far (gate ≤0.2) | 0.092 | **0.089** |
| recovery_attn | 1.025 | 1.015 |
| cos_student_vs_en_attn | 0.688 | 0.695 |
| `names` attn recovery | 0.768 | **0.879** |
| rows visited | 3,667 | 3,535 |

Every health metric in the v3 band or better; no register regressed. The
remaining acceptance is the rendered same-seed grid (plan §3-2) — eyeball gate,
n1/n2/r3 full-JA names stay expected fails.
