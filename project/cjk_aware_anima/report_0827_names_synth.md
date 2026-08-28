# CJK-aware Anima — Phase 2c item (b): synthetic name register (2026-08-27/28)

*Line home: [`plan.md`](plan.md) · ledger [`done.md`](done.md) · prior verdicts
[`report_0816_phase2.md`](report_0816_phase2.md).*

Question under test: **can text pairs alone make a kanji character name render
in a JA prompt?** (the user's question, 2026-08-27). Answer so far: **the rows
learn, but context-specifically** — visits bought in EN context transfer to
mixed prompts and not to full-JA prompts. The JA-context arm that decides the
question is built and not yet trained (§5).

## 1. Restart state (2026-08-27)

The line's Aug-17 state lived on the unmerged branch `cjk-tagpair-tailfill`;
its 14 CJK files + `scripts/distill_cjk/{config,distill}.py` +
`tests/test_cjk_glossary.py` were checked out onto main's working tree
(uncommitted as of this report). The queued-but-never-run Aug-17 step —
retrain with `--train_registers tags,tags_alt,names` on the collision-fixed
ext init — is what ran first.

Stager fix on the way: `scripts/distill_cjk/cache.py` staged at 26 pairs/s with
the GPU idle (the "GIL-free tokenizer thread" premise was wrong — the hybrid
ext encoder + span alignment are pure Python). Replaced with a process pool
(`--encode_workers`, default `min(8, nproc-2)`), bit-identical on 320 pairs,
**67 pairs/s** measured on the daemon.

## 2. Arm A — `names` register alone (`cjk_vocab_pack_names`)

`bench/cjk_distill/results/20260827-2112-2c-names`, grids
`bench/cjk_adapter/results/20260827-2133-2c-names-grid`, `-2149-…-mixed-grid`.

| | item2 (Aug 17) | names |
|---|---|---|
| `recovery_attn` | 0.921 | 0.895 |
| `cos_student_vs_en_attn` | 0.615 | 0.614 |
| `cos_native_vs_en_attn` (floor) | 0.022 | 0.115 |

Recovery drop is the holdout mix (names pairs are mostly readable without ext
rows → floor rises), not a regression. **Renders unchanged**: n1/r1 Reimu still
a generic miko; t3 armor still no armor. Root cause made exact:
**`博麗霊夢` occurs 3× in the 60k corpus** — the `names` register can only pin
names that already occur in captions, and Reimu is not in the retrieved pool
(`博`:19, `麗`:44 visits vs the ~300 render floor). Katakana names (ミク, アスカ)
render because their captions are plentiful; the script is irrelevant.

## 3. Arm B — synthetic EN-context names (`cjk_vocab_pack_synth`)

Builder: [`datasets/synth_names.py`](datasets/synth_names.py).

- **Name source**: danbooru wiki 2026 snapshot (`other_names`, `post_count`,
  `category_name`) — carries full names the tag-pair set lacks (`博麗霊夢` vs
  `霊夢`). The field is polluted with event/meme tags (`初音ミク誕生祭2025`,
  `るしあ大好きだよ`, `レイマリ`); measured rule: **wiki ∩ tag-pair(2024) is
  canonical** almost without exception, expanded by kanji-only supersets
  (`博麗霊夢` ⊃ `霊夢`) and substrings (`魔理沙` ⊂ `霧雨魔理沙`). Guards:
  `tag_pairs.japanese_names` + digit drop. The naive "longest = full name"
  rule tie-broke `レイマリ` over `博麗霊夢` — do not reintroduce it.
- **Targets**: pool characters ∪ wiki `post_count ≥ 3000` ∪ eval names =
  3,000 (1,623 kanji primaries). Copyright: pool co-occurrence → parenthetical
  → first `[[copyright]]` wiki-body link.
- **Templates**: real retrieved captions with a character segment; the
  character (and copyright) swapped on both sides, everything else EN
  (`en_pinned`) — `build_names`' exact-context rule.
- **Rarity-weighted allocation**: per name 24–200 captions, greedy so the
  rarest primary-name kanji reaches `--floor 300` given the base corpus's
  visits → 177,188 pairs. Coverage: `博` 19→300+, `麗` 44→300+, `惣` 1→300+.

Results (`…/20260827-2327-2c-synth`, grids `20260828-0006-2c-synth-grid`,
`-0022-…-mixed-grid`):

- **r1 (JA name + EN tags)**: first render with the red bow + red-white miko
  outfit at a shrine — closest any ext arm has come to Reimu.
- **n1/r3 (full JA)**: *regressed* to a generic red-haired girl; m1 Miku and
  n2 Asuka regressed too. Diagnosis: `names_synth` was 75% of the corpus,
  ~30 `en_pinned` spans per pair at weight 1.0 vs one name span; `tags` got
  ~2.5× fewer gradient visits than Arm A.

## 4. Arm C — rebalanced draw (`cjk_vocab_pack_synth_bal`)

Two distill-time knobs added (no cache rebuild): `--register_sampling
names_synth:0.15` (batch share 75%→~35%) and `--register_span_scale
names_synth:en_pinned=0.3`. `…/20260828-00xx-2c-synth-bal`, grids
`20260828-0112-2c-synth-bal-grid`, `-0128-…-mixed-grid`.

- Tag registers back to normal (t1/t2 clean, m1 Miku recovered).
- **n1 full-JA Reimu still fails; r1 mixed keeps its partial gain.** Same
  pack, only the neighbour language differs → **the name rows are learned
  context-specifically**: the 6-block adapter's output for a row depends on
  its neighbours, and synth only ever showed `博麗霊夢` between EN tokens.
  Miku works in full JA because `ミク` is visited inside JA tag captions.

This is the actual finding of the night: *visits are necessary, and the
context they arrive in has to match the prompt register.*

## 5. Next arm — JA-context synthesis (BUILT, NOT TRAINED)

`synth_names.py --context ja|en|both` (default `ja`): non-name segments
composed through the glossary via `build_pairs.compose` (exactly the `tags`
register), names/copyrights pinned to the chosen wording → register
`names_synth_ja`; `both` adds 8 EN-context captions per name on top. Dry-run
verified; the corpus build + cache (`cache_synth2`, ~70 min) + distill (12k
steps, sampling `names_synth_ja:0.2,names_synth:0.5`) + grids were about to be
queued when the user stopped to plan (2026-08-28 01:40).

Command sequence (all through the daemon):

```
uv run python project/cjk_aware_anima/datasets/synth_names.py --context both
uv run python project/cjk_aware_anima/gates/coverage.py --pack output/ckpt/cjk_vocab_pack_names.json \
    --pairs post_image_dataset/cjk_distill/pairs_synth.jsonl --registers tags,tags_alt,names,names_synth_ja --floor 300
P=post_image_dataset/cjk_distill/pairs_synth.jsonl; C=post_image_dataset/cjk_distill/cache_synth2
make daemon-run ARGS="--queue -m scripts.distill_cjk.cache --pairs $P --cache_dir $C --holdout 500"
make daemon-run ARGS="--queue -m scripts.distill_cjk.distill --pairs $P --cache_dir $C --loss span --steps 12000 \
    --batch_size 32 --param global --trust provenance \
    --train_registers tags,tags_alt,names,names_synth,names_synth_ja \
    --register_sampling names_synth_ja:0.2,names_synth:0.5 --register_span_scale names_synth:en_pinned=0.3 \
    --out output/ckpt/cjk_vocab_pack_synthja --label 2c-synthja"
make daemon-run ARGS="--queue bench/cjk_adapter/run_bench.py --ext --ext_prefix output/ckpt/cjk_vocab_pack_synthja \
    --prompts project/cjk_aware_anima/assets/ja_eval_prompts.json --arms en,ja_t5en,ja_ext --label 2c-synthja-grid"
make daemon-run ARGS="--queue bench/cjk_adapter/run_bench.py --ext --ext_prefix output/ckpt/cjk_vocab_pack_synthja \
    --prompts project/cjk_aware_anima/assets/ja_eval_prompts_names_mixed.json --arms en,ja_t5en,ja_ext --label 2c-synthja-mixed-grid"
```

**Decision rule**: n1/r3 full-JA renders Reimu (red bow, red-white miko,
black hair) *and* r1 keeps its gain *and* t1/t2 stay clean → "text pairs
close the name register", proceed to §6.1. Full-JA still fails with `博`/`麗`
above floor in JA context → the objective, not the data, is the block
(sequence term, §6.3); do not buy more pairs.

## 6. Plan after the JA-context verdict

1. **General tags by the same recipe** (`鎧`:23, `照明`:3, `巫`:72) — extend
   `synth_names.py` to substitute/insert under-floor *general* tags from the
   `coverage.py` under-floor list into JA-context templates. Text-only, CPU,
   same rarity-weighted allocation. Not names-specific: this is the "targeted
   caption widening" item (c) done without crawling.
2. **Glossary sign-off** (unchanged hard blocker): `tag_glossary_review.md` →
   `tag_overrides.json`. Known junk found on the way: `touhou → おは東方`
   (should be `東方`/`東方Project`; the synth register bypasses it via the wiki
   ∩ tag-pair rule, the `tags` register does not). Alt pools such as
   `detached sleeves → 袖だけ霊夢` are *filtered by `alt_pool`* — not a bug.
3. **Sequence-term decision** for span-less registers (prose/quotes): `attn`
   is the measured candidate. Only after this do D2 (73k ready) → D3 → JESC
   matter; doujin OCR pairs are never needed for the encoder side.
4. **Phase 3 ship** when the grid passes: encoder + strategy shim out of
   `bench/`, sidecar as release asset, ComfyUI touchpoint. The synth registers
   ship *inside the pack*, nothing to distribute.
5. The glyph line ([`plan2.md`](plan2.md)) stays behind Phase 3; AnimeText
   (`deepghs/AnimeText`, detection-only, its test split is what
   `manga_text.py` already used) and PP-OCRv6 are its Phase-1 inputs, not
   this line's.

## 7. Uncommitted working tree (2026-08-28)

- Restored from the branch: `project/cjk_aware_anima/**`,
  `bench/cjk_adapter/ext_vocab.py`, `scripts/distill_cjk/{config,distill}.py`,
  `tests/test_cjk_glossary.py`.
- New: `datasets/synth_names.py`, this report.
- Modified: `scripts/distill_cjk/cache.py` (process pool), `data.py`
  (ctor args kept), `config.py`/`distill.py` (`--register_sampling`,
  `--register_span_scale`; `tests/test_cjk_distill.py` 23 passed).
- Artifacts: `output/ckpt/cjk_vocab_pack_{names,synth,synth_bal}`,
  `post_image_dataset/cjk_distill/{cache,cache_synth}`, `names_synth.jsonl`,
  `pairs_synth.jsonl` (EN-context build; re-run `--context both` before §5).

## 8. JA-context verdict (2026-08-28 evening) — **objective, not data**

§5 ran as written (`--context both` → 261,391 pairs: `names_synth_ja` 177,202 +
`names_synth` 24,000 + base; coverage gate: `博`/`麗` above the 300 floor in
JA context, n2 fully covered, only `巫`:176 short). Distill
`bench/cjk_distill/results/20260828-2128-2c-synthja` (recovery_attn 0.901,
`names_synth_ja` recovery 0.95-class like the other name registers); grids
`bench/cjk_adapter/results/20260828-2211-2c-synthja-grid`,
`-2228-2c-synthja-mixed-grid`.

| prompt | synth_bal | synthja | rule |
|---|---|---|---|
| n1 / r3 full-JA Reimu | blue-hair generic | **purple-hair generic** — no bow, no miko, no black hair | ✗ |
| r1 mixed Reimu | blonde miko | black-hair miko, red-white — partial gain kept | ✓ |
| m1 Miku full-JA | ok | ok | ✓ |
| n2 Asuka full-JA | fails | fails (red hair, no plugsuit) — **0 under-floor rows** | ✗ |
| t1 / t2 | clean | clean | ✓ |

Visits were bought *in the matching context* and full-JA still fails; n2
fails with complete coverage. The "thin visits" hypothesis is falsified for
the name register. Per the §5 rule: **do not buy more pairs** — D2 / STAIR /
JESC stay blocked (span-less → inert under `loss=span`, exactly the D2
result). The block is the objective: the span term does not pull a name span
toward the teacher when its neighbours are JA rows (`cos_student_vs_en`
by register: `names_synth_ja` 0.145 < `names_synth` 0.221). Next step is
§6.3 — add the `attn` sequence term and re-distill **on the same
`cache_synth2`** (kept, ~155 G; no rebuild needed), then rerun both grids.

Disk note: `cache/` (32 G) and `cache_synth/` (140 G) were deleted 2026-08-28
to make room; `pairs.jsonl` / `names_synth.jsonl` / `pairs_synth.jsonl` kept,
so either rebuilds from the daemon (~15 min / ~1 h).

## 9. `attn` sequence-term arm (2026-08-29) — **negative; objective lever spent**

Same corpus / `cache_synth2` / registers / sampling as `synthja`, only
`--loss attn:1.0,span:0.5` (probe blocks 0,13,27, real query bank).
`bench/cjk_distill/results/20260828-2239-2c-synthja-attn` (recovery_attn
0.826 vs 0.901 span-only; per-register cos identical), grids
`bench/cjk_adapter/results/20260828-2337-2c-synthja-attn-grid`,
`-2353-2c-synthja-attn-mixed-grid`.

| prompt | span | attn |
|---|---|---|
| n1 / r3 full-JA Reimu | purple-hair generic | headless crop — worse |
| r1 mixed Reimu | black-hair miko (partial) | orange hair — gain lost |
| n2 Asuka | red-hair girl | bearded man |
| m1 Miku / t2 | ok | ok (t1 grows a stray figure) |
| s1 prose | vague | vague — no lift in attn's own register |

Both levers the plan held for the name register — matched-context data (§8)
and the sequence objective (this) — are spent without moving full-JA names.
Two conclusions: (1) `recovery_attn` does not witness the failure (0.90 and
0.83 both render nothing) — the grid is the gate from here; (2) the shape is
*rows learn, the frozen 6-block adapter does not compose all-new rows into
the teacher's output*. That is the capacity signal plan.md's 2-ii
escalation was gated on. Next: **either** a small ext-id-gated LoRA on the
adapter's T5-side blocks (EN stays bit-identical by construction; same
cache/distill), **or** ship Phase 3 with the `synthja` span pack and scope
rare kanji names out of v1. mT5 vocab swap and JESC/STAIR/D2 do not touch
the mechanism — do not re-propose them for this failure.
