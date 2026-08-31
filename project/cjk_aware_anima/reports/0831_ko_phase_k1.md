# K1 — KO corpus (plan_ko.md Phase K1) — built, pending K1.5 sign-off

*2026-08-31. All six K1 steps done in one day; the corpus exists and passes
the tag-register coverage gate. K2 is blocked on the K1.5 user inspection.*

## What was built

| Step | Artifact | Numbers |
|---|---|---|
| 1. Lexicon | `assets/wikidata_lexicon.json` (langs ja+ko) | 355/497 chars + 48/57 franchises with `ko`; JA keys stable except 1 upstream edit (園田 海未→園田海未, space) + entity list grew with the newer caption index (1314→1343) |
| 2. Glossary | `assets/tag_glossary_ko.json` | 15,142 tags, **99.66% occ coverage**, unresolved 0.3%. Via (occ): wiki/KB-verified 27.7%, mt_verified 25.1%, **mt_unverified 33.9%**, override 4.3% (14, user round 1), rating 2.5% |
| 3. Review r1 | applied 2026-08-31 (exemplars 솔로/단발/니삭스; rating 건전/약후방/후방주의/성인용; 14 overrides) | via `--reselect`, no GPU re-buy |
| 4. Pairs | `post_image_dataset/cjk_distill/pairs_ko.jsonl` | 50,190 (tags_ko 16,883 / tags_alt_ko 16,883 / names_ko 16,424); joiner `", "` only; `lang:"ko"` on every record; D2/D6 skipped |
| 5. Synth | `names_synth_ko.jsonl` + merged `pairs_synth_ko.jsonl` | 12,304 over top-500 characters (`--max-names 500 --floor 60 --max-per-name 40`) — plan's ≈13k cap |
| 6. Gate | `gates/coverage.py --lang ko` on all 4 KO registers | below |

Total 62,494 pairs ≈ **41 GB** cache estimate at 0.66 MB/pair — inside the K2
budget (138 GB free on the dataset volume, measured).

## The KR KB (the find that changed the plan)

`models/danbooru_tags_classified.csv` (Localsmile/danbooru_KR_wiki_tag_search,
already fetched by `make download-danbooru-tags`; user pointer 2026-08-31):
114k danbooru tags with Korean taxonomy + a `키워드:` field carrying
community-register names — 66,866 rows with hangul keywords, covering **83.6%
of our general tag types (95.7% of occurrences)** and 2,973/3,630 characters.
It is the KO analog of the JA tag-pair set the plan assumed didn't exist
(verified: none on HF). Wired as candidate source `src: "kb"` competing
through the same back-translation arbitration, plus a name tier in `build()`
and the name-family source in `synth_names.py`.

## Coverage gate (K0 baseline: 100% of rows at 0 visits)

62,494 pairs, 29,244 unique span texts, 1,604 rows visited. **Every t*/q*/n*/c*
prompt fully covered** except one token; misses:

| prompt | v=0 tokens | class |
|---|---|---|
| t5_tags_two | 똑 (똑같은 옷) | **tag register — K1.5 decision** (glossary chose 맞춤 옷차림 mt_unverified; KB alt 커플룩 at same F1) |
| s1/s2/s3/s5 | 횡·텅·젊·있고·그리고·만들 | prose function rows — s* is the untrained D2/D7 register, expected (same as JA v1) |
| q3 | 라고 | quote grammar row — quotes are eval-only by design |

Per plan K1 step 6: the t5 row goes to a §5a-style fix (override or targeted
widening) **before K2**, folded into K1.5.

## Known review classes for K1.5 (from spotcheck + review file)

`assets/tag_glossary_review_ko.md` (top-200 with back-translation evidence)
+ `post_image_dataset/cjk_distill/spotcheck_ko.md` (~200 pairs). Two classes:

1. **Literal MT beat community wording on F1** (the arbiter-blind polysemy
   class, JA's `bow`): `looking at viewer`→시청자를 바라봄 (정면 응시/카메라
   시선 lost; eval prompt says 카메라 응시), `blush`→얼굴 붉어짐 (홍조),
   `hetero`→이성애자 (노멀), `pussy`→여성 성기 (보지),
   `matching outfits`→맞춤 옷차림 (커플룩), `presenting`→제시.
2. **Outright semantic errors spotted in spotcheck**: `mole`→주근깨
   (=freckles; should be 점), `highleg`→긴 스커트 (=long skirt),
   `prone bone`→엎드린 자세의 뼈, `after vaginal`→질 후에,
   `bow`→보우 (ribbon sense missing — the JA polysemy twin).

## Ops notes

- The daemon's default stall watchdog (~120s) killed the first `--mt` job mid
  HF download ("Fetching 4 files" prints nothing) — `--stall-timeout 0` on
  download-heavy first runs.
- `HAN_WIDE` in `tag_glossary.py` has a latent quirk (compat range starts at
  unified U+8C48 → swallows hangul); JA left bit-identical, KO uses `HAN_KO`.
- MT wall time: ~2.6h total on the CPU-streamed 7B (10,654 forward + 28,502
  back-translations, ~151–225 items/min), caches in `assets/.mtcache/`.

## Next

**K1.5 (user)**: review round 2 over the two classes above + spotcheck +
the 똑같은 옷 decision → `tag_overrides_ko.json` → `--reselect` + pairs
rebuild (CPU, minutes). Then **K2**: stage `cache_ko`, joint retrain
`synthjako` warm-started from synthja_v4, KO 25–30% sampling.

## Review round 2 (K1.5, applied 2026-08-31)

User audit over the review file + spotcheck: 16 semantic overrides (홍조,
카메라 응시, 리본, 점, 하이레그, 섹스 후, 상납 자세, 엎드린 뒤치기, 커플룩,
질내사정, 노모, 애액, 옷 입은 여성과 나체의 남성, 가슴 노출, 고무줄에 살이
눌린, 뒤치기) + all 40 `* thighhighs` variants moved off MT's 니하이 onto the
니삭스 exemplar (색깔 있는것들도 수정). `tag_overrides_ko.json` now 71 entries
= 9.6% of occurrences. Applied via `--reselect` (prior banked as
`tag_glossary_ko.pre_round2.json`) + pairs/synth rebuild — same 62,494-pair
corpus, no GPU.

The t5 똑같은 옷 decision: **커플룩** on both sides — glossary override *and*
the `ko_eval_prompts.json` t5 wording (the corpus can only visit the register
it trains). Gate now fully green: every t*/q*/n*/c* prompt at v=0/v<5 = 0;
remaining misses are the expected s* prose rows + q3 라고 (untrained
registers, same as JA v1).
