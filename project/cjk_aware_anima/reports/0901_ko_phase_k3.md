# K3 — v2 retrain on the r3/r4 wording + gate ledger

*2026-09-01. Records the 2026-08-31 evening chain that cleared K1.5's owed
item (full `cache_ko` re-stage + `synthjako` retrain after the round-3/4
glossary overhaul) and the K3 gate state, including the two JA grids that
died with the daemon shutdown and were re-queued this morning.*

## The v2 chain (2026-08-31 evening, verified from on-disk mtimes)

| when | what |
|---|---|
| 20:42 | overrides + glossary final — 345 overrides (r4 vetoes, MT-eaten emoticon tags), `collisions_accepted_ko.json` |
| 20:43–44 | pairs rebuilt (`pairs_ko` / `pairs_synth_ko` / `names_synth_ko`, coverage gate green) |
| 20:46–21:04 | `cache_ko` re-staged **in full** (44 GB — the stager is positionally keyed and 65.8 % of student text changed) |
| 21:04–21:58 | **`2c-synthjako-v2`** (`bench/cjk_distill/results/20260831-2104-2c-synthjako-v2/`): v4 recipe + `--init_pack synthja_v4`, `--cache_dir cache_synth3,cache_ko`, 289,209 pairs, sampling unchanged. Output `output/ckpt/cjk_vocab_pack_synthjako` — **the on-disk pack is now v2** |
| 22:01 | three K3 grids submitted; KO recovery grid completed 22:22; **both JA grids died with the daemon shutdown (0 renders)** |

The 22:28 commit (`99a158e2`) committed the 20:42 override/collision state, so
the trained pack matches the committed wording — plan_ko K1.5's "Owed"
paragraph is cleared by this chain.

JA regrids re-queued 2026-09-01 08:09 (daemon jobs `20260901-080944-1b998c`
ja-grid, `-4ebca1` ja-mixed-grid); verdicts appended below when done.

## Holdout, v1 → v2 (final eval, student attn)

Same instrument, but the KO holdout pairs themselves were reworded with the
corpus — KO deltas are partly the corpus, not the pack. The KO `teacher`
column stays unk-walled on hangul (0.088→0.010), so `recovery` remains
non-comparable to JA (K2 report caveat, unchanged).

| register | v1 | v2 | read |
|---|---|---|---|
| tags / tags_alt | 0.528 / 0.530 | 0.523 / 0.521 | JA unchanged (cache untouched) |
| names | 0.894 | 0.894 | unchanged; the K2 recovery-dip watch item did not worsen |
| names_synth_ja | 0.666 | 0.663 | unchanged |
| **tags_ko / tags_alt_ko** | 0.463 / 0.468 | **0.370 / 0.384** | **the watch item** — see below |
| names_ko | 0.955 | 0.946 | unchanged |
| names_synth_ko | 0.628 | 0.674 | improved |

Final span loss 0.0868 → 0.1008, disc_far 0.087 → 0.086.

**tags_ko band gap.** Under the reworded corpus the student sits ~0.15 under
the JA tags band (0.52–0.56) instead of ~0.07. Two candidate readings, not
separable from this run alone: (a) the r3/r4 wording is *harder* (more
distinct KO surface forms after de-collisioning — 336 collision groups broken
up means fewer repeated targets), or (b) the redrawn holdout slice. The grid
shows no corresponding regression (below), so this is a *watch*, not a fail;
re-read after G2/G5. If it persists, the in-plan levers are the
`mt_unverified` tail review by occurrence (`datasets/audit_glossary.py`), the
3 pending collision groups, and optionally a KO-sampling arm — **not** the
closed lines (mT5 / adapter-LoRA / MT-prompt tuning / more synth volume).

## Gate 3 — KO recovery grid v2 (`20260831-2201-ko-k3-recovery-grid-v2`)

Per-prompt flat `cos_vs_en` for `ko_ext`, K0 → v1 → v2 (flat cos is *not* the
recovery instrument — K2 report; shown to confirm v2 ≈ v1, i.e. the reworded
corpus did not regress the grid):

| prompt | K0 | v1 | v2 |
|---|---|---|---|
| t1 school | 0.051 | 0.087 | 0.087 |
| t2 maid | 0.055 | 0.097 | 0.093 |
| t3 armor | 0.058 | 0.105 | 0.114 |
| t4 cat | 0.061 | 0.100 | 0.116 |
| t5 two | 0.074 | 0.138 | 0.140 |
| t6 boy | 0.068 | 0.104 | 0.117 |
| q1/q2/q3 | 0.084/0.083/0.088 | 0.133/0.067/0.135 | 0.136/0.069/0.127 |
| n1/n2/n3 | 0.068/0.083/0.069 | 0.074/0.085/0.068 | 0.068/0.076/0.076 |
| c1/c2/c3 | 0.062/0.091/0.106 | 0.079/0.087/0.124 | 0.077/0.087/0.126 |

s\* prose stays at floor (untrained register, expected); n\* flat (full-KO
names = expected fails, plan.md §5b territory). Discrimination `ko_ext` 0.144
(en 0.110), unchanged from v1.

**Render eyeball: DONE — user checked the v2 grid and signed off
(2026-09-01).** Gate 3 is green.

## Gate ledger (2026-09-01 morning)

| gate | status |
|---|---|
| G1 EN bit-exact | **GREEN** (unit suite, construction-level, pack-independent) |
| G2 JA non-regression | holdout-level green (JA registers unchanged v1→v2); **v2 same-seed render grids owed** — the 22:01 jobs died with the daemon before rendering, so the earlier eyeball was necessarily on the v1 grids. Re-queued; compare vs the v4 twins (`20260831-1309` / `-1324`, ±0.013 band) + quick n1/n2 eyeball |
| G3 KO recovery | **GREEN** — metric level v2 ≈ v1, render eyeball signed off (user, 2026-09-01) |
| G4 coverage | **GREEN** (K1.5 r3/r4 gate re-run; disc_far 0.086 train / 0.144 grid) |
| G5 register drift | **OWED** — 300 hand-typed Arca-Live-register KO prompts (not glossary-composed) vs the composed holdout; gap > the JA D7 gap is the signal. CPU + one eval job |

## Remaining before K4

1. G2 v2 grids finish → metric compare vs the v4 twins + n1/n2 eyeball.
2. G5 (the only gate with no measurement at all — and the user base *is* the
   Arca register, so it is the real ship risk).
3. tags_ko band-gap watch: re-read after 1–2; levers listed above.

## v2 JA grid verdicts (re-run 2026-09-01 morning)

- **`2c-synthjako-v2-ja-grid`** (`20260901-0809`) vs v4 (`20260831-1309`,
  same seed): `en`/`ja_t5en` bit-identical; `ja_ext` within ±0.013 on every
  t\*/n\*/m\* prompt; **two q\* prompts marginally over the band** —
  q1_quote_sign −0.015, q2_quote_shop −0.016 (the v1 grid was in band, so
  ~−0.005 of this is v1→v2). Discrimination `ja_ext` 0.129→0.127
  (p1-vs-p2 0.248→0.240) — unchanged. Quote registers are eval-only and the
  deltas are flat-cos on the weakest instrument; **the q1/q2 renders join
  n1/n2 in the eyeball** rather than being called a metric fail.
- **`2c-synthjako-v2-ja-mixed-grid`** (`20260901-0825`) vs `20260831-1324`:
  fully in band (worst `ja_ext` delta −0.010 @ r1_reimu_mixed); `ja_t5en`
  bit-identical; discrimination unchanged (0.130→0.129).

**Gate 2 metric level: GREEN on the v2 pack** (t\*/n\*/m\* clean, q1/q2
marginal-and-flagged).

### Render eyeball (Claude, 2026-09-01) — and what the ±0.013 band actually means

First, an instrument correction: **the flat-cos band does not imply visual
sameness.** t1's `ja_ext` was in band on every grid, yet v4 renders a
full-body seifuku girl sitting on a desk while v1/v2 render a close-up
blazer-and-red-tie girl — the composition flipped *at v1 already*, inside the
band (chaos-floor territory: tiny embedding deltas flip same-seed
compositions). The environment is clean — `en` renders are pixel-identical
across the 08-31 and 09-01 runs — so all `ja_ext` differences are pack
effects. The honest baseline for "did the r3/r4 rewording hurt JA" is
therefore **v1** (user-eyeballed and accepted), not pixel-sameness with v4.

Per-prompt verdicts vs v1 (v4 shown for context):

| prompt | v4 → v1 → v2 | verdict |
|---|---|---|
| t1 school | seifuku full-body → blazer close-up → blazer close-up (flatter shading) | **PASS** — composition held v1→v2 |
| n1 hakurei | same wrong-identity pink-haired girl in all three (known kanji-name miss) | **PASS** — unchanged |
| q1 sign | wooden object overhead → raised sign-ish → girl holding a *book*, no sign | mild drift (quote semantics lost) |
| q2 shop | clean interior → painterly bookstore → **degenerate mosaic blocks** | **regression** (quality collapse; all three miss the "storefront" semantics — that part is old) |
| n2 asuka | red hair + red suit → red hair + red suit (plus extra figure) → **purple-haired man**, red outfit only | **regression** (name semantics lost; tracks the names attn dip) |

Both regressions sit on the registers most prone to single-seed composition
flips (quotes are eval-only n=4; n2 is a kanji full-name). To separate
systematic-vs-chaos-flip, a **seed probe** ran: n2+q2 at seeds 43/44/45 on v4
and v2 (`n2q2-probe-{v4,v2}-s{43,44,45}`; the v1 pack file was overwritten by
v2 at the same output name, so v1 cannot be re-rendered).

**Seed-probe verdict: both "regressions" are chaos flips, not systematic.**

- **n2 asuka**: v4 keeps red-hair identity only 1/3 seeds (s43 red ✓, s44
  white-haired figure, s45 purple-haired); v2 gets 2/3 (s43 red **with the
  interface headset**, s44 black-haired man, s45 red-pink ✓). Identity on a
  kanji full-name is a per-seed coin flip in *both* packs — the seed-42
  purple-haired man was v2's unlucky draw, not a drift.
- **q2 shop**: v4 also degenerates at some seeds (s44 = the same pixel-mosaic
  failure mode; s45 an odd isometric diorama; only s43 clean). v2 gives
  degenerate/muddy at all three, so it is *maybe* marginally worse on this
  prompt, but the instability itself is a shared property of the untrained
  quote register, not a v2 regression.

**Gate 2 verdict: PASS** — core tags stable v1→v2, JA holdout unchanged, and
the two flagged prompts are within both packs' natural per-seed variance.

### KO recovery grid — independent render read (Claude, 2026-09-01)

Supplementary to the user's sign-off; `ko_ext` vs `ko_t5en` on the v2 grid,
with the JA v2 grid as the cross-language control:

| prompt | verdict on `ko_ext` |
|---|---|
| t1 school | **binds** — 교복/교실/미소/앉음 all land (blazer variant, clean) |
| t3 armor | 검/night mood land; **기사/은색 갑옷 lost** (black bodysuit) — but JA `ja_ext` *also* fails armor (red-robed swordsman) → **shared ext-pack ceiling, not KO-specific** |
| t5 two | 소녀 2명/벚꽃/봄/야외/전신 land; 쌍둥이 weak (different hair), 커플룩 approximate; style drifts chibi with a text artifact |
| c1 closeup | **binds** — closeup portrait composes (freckles missing) |
| c3 mono | **흑백/그레이스케일 lost** (full-color retro cel, composition off) — JA `ja_ext` renders a *proper* monochrome scene → **KO-specific binding gap** |

Reading: KO recovery is real (pure-KO input composes correct scenes where the
zero-shot baseline was noise) but **uneven at the mid-frequency tag tier** —
consistent with tags_ko attn sitting under the JA band. The KO-specific
misses to chase (흑백/그레이스케일, 쌍둥이) are glossary/visit-floor targets,
i.e. exactly the `mt_unverified`-tail / §5a-widening lever — while t3-style
misses are shared with JA and are not a KO corpus problem.

### The `mt_unverified` tail is mostly a self-inflicted arbitration loss

Follow-up to the "external dataset" question (2026-09-01): no new public KO
source beats what is already in-repo — the KR KB
(`models/danbooru_tags_classified.csv`, the same Localsmile
danbooru_KR_wiki dataset the GUI uses) covers **3,169 of the 4,264
`mt_unverified` tags (97.3 % of tail occurrences) with a `키워드` field the
glossary discarded.** Mechanism (`tag_glossary.py::choose`): candidates are
arbitrated by back-translation F1 against the EN *tag name*, which is
structurally unable to verify booru jargon (백합→"lily"≠`yuri`,
파이즈리→"titjob"≠`paizuri`) — so exactly where the KB matters most, its
candidate is rejected and the fallback hands the win to the **unarbitrated MT
string**. All three r3 poster-child errors (paizuri/cuffs/yuri) had the
correct answer sitting in the KB the whole time; the r3/r4 hand-overrides
reproduced KB keywords verbatim.

Blanket KB-first is wrong, though — above the review floor MT is usually
right and the KB keyword would regress (`blonde hair` 금발→노란 머리,
`swimsuit` 수영복→비키니, `grey hair`→은발). The value is band-dependent:

- **occ ≥ 100** (117 tags / 93.8k occ): MT mostly fine (implicitly covered by
  r1–r4 review); KB flip would be net-negative unexamined.
- **occ < 100** (2,556 tags / 27.8k occ): MT degrades into semantic howlers
  (`improvised gag`→즉흥 개그 "improv bit" vs KB 임시 재갈; `skinny
  dipping`→니하이 수영 vs KB 알몸 수영); KB usually correct. Known KB failure
  class: rare character tags whose keyword is the *series* name
  (`elsa granhilte`→리제로) — occ≈1 noise.

**Proposed r5 rule** (not yet applied — user sign-off owed): below the floor,
a KB candidate outranks `mt_unverified` (`via: kb_unverified`); above the
floor, nothing auto-flips and the 117 disagreements go to user review.
Decision artifact: `datasets/assets/tag_glossary_review_ko_r5_kb.md`
(A = 117 pick-one rows, B = 2,556 auto-flip preview). If accepted:
`choose()` fallback change + glossary re-arbitration (CPU, candidates are
cached in the JSON) → the standard reselect→pairs→cache_ko→retrain loop
(~1.5 h wall, same as the r3/r4 round).

### `desc_ko` — the KB's description translations as a prose register

Second extraction from the same KB (user request 2026-09-01): the KB rows
are natural-Korean summary translations of the danbooru wiki bodies, i.e. a
per-tag EN↔KO **domain prose** parallel corpus that already sits on disk.
Built: `datasets/desc_pairs.py` →
`post_image_dataset/cjk_distill/pairs_desc_ko.jsonl` — **11,631 pairs**
(EN wiki first sentence, `[[link]]` markup stripped ↔ KB description minus
the `[분류]` prefix and `키워드:` suffix), restricted to glossary/corpus
tags, deduped on shared descriptions. Sample:
`wiping tears` — "The act of removing tears from one's face, typically by
hand, sleeve, or tissue." ↔ "손, 소매, 휴지 등으로 얼굴의 눈물을 닦아내는
행위." Rows are span-less and ride the D2-`commentary` whole-sequence path
(`via: kb_desc`), so no loss-side change is needed.

Why this is not the closed JESC/STAIR line: that verdict was about *generic
off-domain* parallel prose; these are caption-shaped visual descriptions in
community terminology, aligned per tag, hitting the two holes nothing else
covers — the s\* prose prompts (at floor since K0, "untrained register,
expected") and tag vocabulary inside real sentence syntax
(particles/spacing — the G5 register axis). Still gated as a **trial arm**,
not a bulk ship; KO side keeps its full description (sometimes one sentence
longer than EN — same looseness precedent as commentary rows; a
first-sentence-truncation knob is the fallback if the loose rows hurt).

### Proposed next loop (two arms, clean attribution)

1. User reviews r5 section A (117 rows, pick MT-vs-KB) + spot-checks B.
2. **Arm A** — r5 glossary only: `choose()` rule + re-arbitrate → reselect →
   pairs → cache_ko re-stage → retrain (~45 GPU-min). Gates: tags_ko band,
   K3 grids, JA non-regression.
3. **Arm B** — r5 + `desc_ko` (sampled low, ~0.1): same loop + the new
   register. Gate: **s\* prompts off the floor** with tags_ko/JA unchanged
   vs Arm A. Distinct `--out` names — never overwrite the shipped pack.
