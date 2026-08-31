# CJK-aware Anima — report 0831: corpus rebuild (name-axis fix + `, ` joiner) → `synthja_v2`

*Written 2026-08-31. Triggered by the user reading `spotcheck.md` after the
glossary sign-off and spotting `D1/pepper0/5126752/names` with
`grani (arknights)` left English while `arknights` → アークナイツ, and every
JA side joined with `、`. Both turned out to be corpus-wide. Digest goes to
[`findings.md`](../findings.md) §3; code notes in
[`datasets/README.md`](../datasets/README.md).*

Status: **`cjk_vocab_pack_synthja_v2` replaces `synthja` as the v1 ship
candidate** — no regression on any grid prompt, visible gains on t2/t3 and on
the name prompts a1/a2/r2/m1, every distill-side gate improved.

## 1. The two bugs

**(a) Name axis from the wrong index.** `tag_glossary.py` took a tag's axis
only from `caption_index.json["groups"]`, which indexes `image_dataset`
(3,065 images), while the glossary and pairs are built over the D1-wide
roots (16,883 captions). Any character/copyright tag that never occurs in
`image_dataset` fell to `general` — and general tags go through **MT, which
renders a name as words**: `ame (mignon)` → 雨（可愛い）, `grani (arknights)`
→ グラニ（アークナイト）, `shiro (mignon)` → シロ（ミニョン）. Two
consequences: the `names` register silently kept those names EN
(`en_pinned`, `n_missing` stayed 0 — 6,149 of 14,959 name pairs), and the
`tags`/`tags_alt` registers trained the rows on the junk MT wording — the
exact G4b failure the review is meant to catch, invisible to it because
these rows had no rival source to disagree with.

Fix: `resolve_axis` = caption index → wiki dump `category_name` → artist-OC
form (`name (handle)` with `@handle` an artist in the corpus) → `@` handle →
general; entries record `axis_src`. Names never reach MT.

**(b) The joiner was an ext row.** Every student string joined tags with
`、` (U+3001), which lies in `ext_vocab._CJK_RANGES` (CJK punctuation), so the
delimiter itself was a trained ext row on every pair, while the teacher saw
the native T5 `,`. Users type `, ` in every language's prompt box. Fix:
`build_pairs.pick_joiner` — `", "` 80 % / `、` 20 % per pair (seeded),
stored in the record's `"joiner"` field, read by
`scripts/distill_cjk/data.py::_ja_span_chars` (absent field ⇒ legacy `、`).
The tag-style eval prompts (`assets/ja_eval_prompts*.json`) were re-expressed
with `, `; prose prompts keep `、` as punctuation.

## 2. What the rebuild changed

Glossary (`--reselect` from the pre-fix build — CPU, MT candidates banked):

| | before | after |
|---|---|---|
| tags | 14,753 | 15,142 (corpus grew since 08-17) |
| axis moves general → character / copyright / artist | — | 2,138 / 540 / 5 (wiki) + 33 OCs (artist rule) |
| name wordings changed | — | 2,400 (1,701 wiki, 298 wiki_han, 396 → `unresolved` = EN passthrough) |
| **general-axis wordings changed** | — | **0** → the signed-off review still stands |

Corpus: `pairs.jsonl` 63,242 (tags/tags_alt 16,883 each, names 16,502, D2
9,068, D6 2×1,953); `pairs_synth.jsonl` 262,852 with `names_synth_ja`
175,610 / `names_synth` 24,000. Joiner mix on span registers: `, ` 199,857 /
`、` 50,021 (prose registers carry none). `cache_synth2` restaged from
scratch (171 GB, ~70 min). The pepper0 record now reads
`sensitive, 1girl, グラニ, アークナイツ, @pepper0, …` with
`{"en": "grani (arknights)", "ja": "グラニ", "via": "wiki"}`.

## 3. Distill (`2c-synthja-v2`, settled recipe, warm from scratch)

`bench/cjk_distill/results/20260831-0022-2c-synthja-v2/` vs
`20260828-2128-2c-synthja/`. Holdouts are different draws over different
corpora, so read directions, not third decimals.

| metric | synthja | **synthja_v2** |
|---|---|---|
| final span loss | 0.0924 | 0.0921 |
| `cos_student_vs_en_attn` (the readout) | 0.529 | **0.674** |
| `recovery_attn` | 0.901 | **1.042** |
| attn recovery — tags / tags_alt | 0.536 / 0.659 | **1.125 / 1.168** |
| attn recovery — names / names_synth / names_synth_ja | 0.947 / 0.988 / 0.919 | 0.830 / 0.831 / 1.043 |
| `cos_student_vs_en` by register — names | 0.183 | **0.340** |
| `discrimination_far` (gate ≤ 0.2) | 0.099 | 0.089 |
| `discrimination_near_attn` | 0.980 | 0.802 |
| flat `recovery` (control, not a gate) | 0.101 | 0.031 |

The names-register attn recovery dips because its *native* baseline moved
(0.58 → 0.70: with `, ` the untrained rows already sit closer to the teacher,
so there is less to recover); the student's absolute readout on names rose
(`cos_student_vs_en` 0.18 → 0.34). The flat `recovery` drop is the control
findings §2 warns about — it moved opposite to every readout-space number.

## 4. Grids (same seed 42, `en` / `ja_t5en` / `ja_ext`)

`bench/cjk_adapter/results/20260831-0110-2c-synthja-v2-grid/` and
`…-0126-2c-synthja-v2-mixed-grid/` vs the 08-28 pair. NB the old `ja_ext`
column was rendered on `、`-joined prompts, the new on `, `-joined ones — the
prompt the student sees changed with the register.

| prompt | old `ja_ext` | **v2 `ja_ext`** |
|---|---|---|
| t1 school | clean | clean |
| t2 maid | maid, tray missing | maid + apron + tray-ish, closer to teacher |
| t3 armor | red-cloak figure, no armour | knight silhouette + sword; armour still weak (`鎧` 193 visits, under the 300 floor) |
| t4 cat / t5 twins / t6 boy | ok | ok, unchanged |
| q1 / q3 quotes | fail (eval-only register) | fail, as expected |
| n1 Reimu full-JA | wrong character | wrong character (expected fail, out of v1) |
| n2 Asuka full-JA | red-haired other | **orange hair + red plugsuit** |
| c1 / c3 | ok | ok |
| c2 wide | figure in green field, no sunflowers | same |
| r1 Reimu mixed | miko outfit, wrong face | miko outfit, wrong face |
| r2 Reimu inverse | miko, no bow | **red bow + black hair = Reimu** |
| r3 Reimu full | wrong character | wrong character (expected fail) |
| a1 Asuka mixed | different red-haired girl | **Asuka (plugsuit, eyepatch)** |
| a2 Asuka inverse | Asuka-ish | **Asuka, closer to teacher** |
| m1 Miku full | Miku, wrong pose | **Miku singing with mic** |
| m2 Miku mixed | Miku | Miku |

Grid discrimination (`ja_ext`): tags grid p1-vs-p2 0.197 → 0.231 (teacher
0.223); mixed grid 0.076 → 0.095, mean-over-pairs 0.120 → 0.132.

Coverage (`gates/coverage.py`, floor 300, registers
tags,tags_alt,names,names_synth_ja): no new zero-visit user-facing token on
t1–t6; `緻` (t2 緻密な背景) was already 0 before the rebuild (report 0816
§), `鎧` 193 / `照明` 2 / `巫` 188 are the plan §5a list, unchanged.

## 5. Verdict and what carries forward

- **Ship `synthja_v2`.** Gates: G1 unit test unchanged (rows-only surface);
  far-disc 0.089 ≤ 0.2; no tag prompt regressed; coverage unchanged.
- The name-prompt gains (a1, r2, m1) come from the *names* register finally
  seeing the 6,149 previously-unswapped names plus the 2,400 corrected
  wordings — text pairs, no adapter change. They do **not** reopen the
  rare-kanji line: n1/r3 (博麗霊夢 full-JA) still fail exactly as findings §4
  records.
- Plan §5a (under-floor general tags, `鎧`/`照明`/`巫`) is still the next
  cheap corpus item; t3 shows it is the remaining tag-side gap.
- Pre-0831 numbers (`synthja` band, 08-16/08-28 grids) are not directly
  comparable to anything built from here: different joiner, different
  prompts, different holdout.
- Korean (`plan_ko.md`) inherits both fixes; the disk constraint there is
  relaxed — the volume has ~150 GB free after this restage.

## 6. Same-day §5a arm: `tags_synth_ja` → `synthja_v3`

`datasets/synth_tags.py` (new): prompt-driven under-floor targeting — split
the tag-style eval prompts (t*/c*) into aligned (en, ja) segs, keep the 10
whose ext rows sit under the 300 floor (`緻密な背景`/`接写の肖像`/`俯瞰` at 0,
`劇的な照明` 2, `1990年代アニメ風` 14, `風になびく髪` 48, ひまわり畑 54,
`図書館` 164, `銀の鎧` 193, `柔らかい照明`), mint 2,249 captions by
substituting the pinned wording (`via: eval_pinned`, trust 1.0) into real
`image_dataset` templates composed through the glossary. Corpus
`pairs_synth_tags.jsonl` 265,101 → `cache_synth3` → distill
`2c-synthja-v3-tags` (register share 2 %) → both grids.

Distill: loss 0.092→0.089, readout 0.674→0.688, far-disc 0.092 (holdout is a
fresh draw; `tags_synth_ja` absent from it at 0.85 % of the corpus — the grid
is its gate). Grids vs v2, same seed:

- **Moved, exactly the targeted vocabulary**: c1 (actual close-up profile
  portrait, wind-blown hair), c2 (vast field + tiny distant figure + sky),
  c3 (greyscale figure instead of a smear), t2 (indoors + detailed interior),
  t6 (teacher's composition). Names untouched: r2/a1/a2/m1/m2 hold, r1 gains
  black hair; mixed-grid discrimination flat.
- **Not passed: t3 armor** — `銀の鎧` needed only 107 pairs to reach the 300
  floor from 193 and the render still has no knight/armor; the block is
  plausibly phrase-binding (or the floor itself), not zero visits. t1/t5
  composition wobbles slightly (t1 loses the desk; candidate cause: the 136
  `図書館` pairs biasing indoor scenes bookshelf-ward).

Verdict: **`synthja_v3` supersedes v2** (all v2 gains kept + the c*/t2/t6
vocabulary; one open failure, no regression that outweighs it). Open item:
rerun `synth_tags.py` at a higher floor / per-target for the armor family
(`--extra-terms`, e.g. 騎士/城 alongside 銀の鎧) before concluding §5a is
exhausted for t3.
