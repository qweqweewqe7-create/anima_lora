# position_captions — probes for position-aware auto-captioning

> **Both probes came back green and the feature shipped** as
> `make caption-position` (v2: the caption is rewritten, each bound tag moved
> out of the flat bag). These are the Phase-0 feasibility probes, kept for the
> record — the live doc is
> [`docs/experimental/position_captions.md`](../../docs/experimental/position_captions.md)
> and the retired design proposal is `_archive/proposals/position_captions.md`.

Feasibility probes for expanding the auto-caption system with per-instance
positional clauses in the dataset's existing hand-written convention
(`On the left, akita neru, yellow eyes. On the middle, hatsune miku, …`).
Motivation: the flat tag bag can't express which attribute belongs to which
character (`2girls, blonde hair, black hair` — who is blonde?), and the same
ambiguity hits `1girl, multiple views` sheets (e.g.
`channel_(caststation)/8090164`: maid/bunny/bikini/swimsuit tags all unbound).

Populations at time of writing: 258 multi-girl captions (12 with hand-written
clauses = ground truth), ~350 multiple-views captions.

## Probe A — `probe_binding.py`

Does the **base model** hear positional clauses at all? Renders no-LoRA images
from counterbalanced prompts (`…, blonde hair, black hair. On the left, blonde
hair. On the right, black hair.`), splits each render into left/right halves,
and asks the tagger which hair color wins on each side.

- Headline metric: `side_accuracy` (chance = 0.5). The two sides of one image
  are anti-correlated (both colors usually render; only the assignment
  varies), so read the side-level number, not `image_accuracy`.
- If accuracy is near chance, positional captions would be teaching the model
  a new capability from ~260 images — temper expectations. Well above chance
  → captions reinforce an existing capability.

```
make daemon-run ARGS="bench/position_captions/probe_binding.py --label binding"
```

## Probe B — `probe_autocaption.py`

The pipeline prototype: SAM3 `girl` instances (per-instance boxes + scores,
IoU-deduped, ordered left→right) → padded crops → Anima Tagger per crop →
composed positional clause. Gates on **detected instance count ≥ 2**, not the
girls-count tag, so multiple-views sheets are first-class.

Default input = the 12 ground-truth images + the 8090164 showcase. Metrics:

> **`discover_gt` is poisoned since the 2026-08-19 corpus-wide apply — always
> pass `--images`.** It treats any resized caption containing "on the left" as
> ground truth, and after `caption-position --apply` nearly every derived
> caption has clauses: a default run now scores 338 "GT" images that are the
> pipeline's *own output*, not hand labels (`n_gt_images: 338` instead of 12).
> Those numbers measure agreement-with-the-previous-pipeline and are not
> comparable to the 12-sheet gate. The hand-GT 12 survived the apply untouched
> (they already had clauses, so `is_candidate` skipped them — master and
> resized still match), and they are:
> `butterchalk/butterchalk2`, `channel_(caststation)/{channel1,channel5,channel6}`,
> `chicke_iii/chicke2`, `hews/{4615461,6321464,6776696,6795089}`,
> `ootomo_takuji/takuji3`, `rurudo/rurudo1`, `sweetonedollar/sweetonedollar5`
> (all `post_image_dataset/resized/<rel>.png`).


- `count_accuracy` — detected instances vs hand-written clause count.
- `char_position_accuracy` — GT clause names a character the tagger vocab
  knows → did the crop at that position keep it?
- `hair_position_accuracy` — GT clause names a hair color → did the crop's
  hair-color group winner match?
- `proposed` clause strings + saved crops in the run dir for eyeballing.

```
make daemon-run ARGS="bench/position_captions/probe_autocaption.py --label autocaption"
```

Results land in `results/<YYYYMMDD-HHMM>-<label>/` (gitignored) with the
standard `result.json` envelope.
