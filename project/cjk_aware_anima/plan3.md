# CJK-aware Anima — plan3: adapter capacity (ext-gated LoRA on the LLM Adapter)

*Line home: [`plan.md`](plan.md) is the vocab-pack line (rows only, adapter
frozen). This document promotes its **2-ii escalation rung** into its own
line, now that the capacity signal it was gated on has appeared
([`report_0827_names_synth.md`](report_0827_names_synth.md) §8–§9).
Written 2026-08-29.*

Status: **PROPOSED** — one arm, grid-gated, with the fallback pre-committed
(§Decision). Nothing here needs a cache rebuild: `cache_synth2` (261k pairs,
JA-context names, ~170 G) is the training set as-is.

## What forced this rung

Two nights closed both levers plan.md held for the name register:

| lever | arm | full-JA Reimu / Asuka | verdict |
|---|---|---|---|
| matched-context data | `synthja` (§8): `博`/`麗` > 300 visits *in JA context*, Asuka 0 under-floor rows | fail / fail | data falsified |
| sequence objective | `synthja_attn` (§9): `attn:1.0,span:0.5` | worse / worse (r1 gain lost) | objective falsified |

Everything else the pack buys is intact across both arms (tag registers,
quotes, Miku, mixed-prompt names). What never moves is a name whose *whole
neighbourhood* is new rows.

Mechanism, stated precisely enough to attack: the LLM Adapter is 6 blocks,
each **self-attention over the T5-id stream** (`self_attn=True`,
`library/anima/models.py:1394`) + cross-attention into Qwen + MLP. The
self-attention is where a query token's output depends on its neighbours —
and it was pretrained on EN pieces only. Rows-only distillation asks *new
inputs* to make *frozen* self-attention weights reproduce the teacher's
composition of `hakurei reimu` when every key/value around the name is also a
new row. The rows learn (recovery 0.90+ in every arm); the composition does
not. Miku works because `ミク` is a common piece visited inside real JA
captions the adapter already half-handles. That is a capacity limit in the
adapter, and it is the only thing left un-tried on this failure.

Two things §9 taught that shape the gates below:

1. **`recovery_attn` is blind to this failure** (0.90 and 0.83 both render
   nothing). The eyeball grid is the gate; distill metrics are health checks.
2. Adding gradient pressure on the *sequence* readout without capacity made
   renders worse — so capacity is added with the **span** loss that already
   holds every other register, not with `attn`.

## Goal, stated as the invariant it creates

A JA prompt made entirely of ext rows composes a rare kanji name the way the
EN teacher does — **while every EN prompt stays bit-identical to the base
model**, and the artifact is an ordinary LoRA that rides the existing Anima
Adapter Loader alongside the vocab pack (plan.md deployment §: "always
alongside the vocab pack, never instead of it").

The EN invariant is kept **by construction, not by regression testing**:
the LoRA delta is multiplied by a per-sequence gate `g ∈ {0,1}` = "this
sequence contains at least one ext id (≥ 32128)". An EN prompt has `g = 0`
and never touches the delta, so bit-exactness is a property of the wiring.
The EN-regression gate plan.md demanded for 2-ii collapses to a unit test.

## Design

### Placement

LoRA (rank `r`, α = r, zero-init B) on the adapter's per-block Linears, all 6
blocks:

| target | why |
|---|---|
| `self_attn.{q,k,v,o}_proj` | the neighbour-mixing path — the measured failure |
| `cross_attn.q_proj` | how a (new) query reads Qwen; plan.md's original 2-ii target |
| MLP | **off** in the first arm (adds params without a hypothesis) |
| `cross_attn.{k,v}_proj` | **off** — they read the Qwen side, which is unchanged |
| `in_proj` | Identity (1024→1024) — nothing to adapt |

Rank **16** first (6 blocks × 5 Linears × 2×1024×16 ≈ 1.0 M params — same
order as the ext table's global map, ~4 MB bf16). Rank is the only knob to
sweep if the first arm is *directionally* right but weak.

### Gate

`g = any(ids >= T5_TABLE_SIZE, dim=seq)` per sequence, broadcast over the
delta: `y = W x + g · (B A x) · (α/r)`. Two consequences worth writing down:

- **Mixed prompts get the delta on every token**, EN tokens included. That
  is intended: the EN tokens *around* a JA name are part of the context the
  adapter has to compose, and mixed prompts (r1/a1/m2) are a register users
  actually type. The bit-exact guarantee is for *pure* EN sequences.
- Batches in training are all-ext by construction (every pair is a JA
  student), so `g = 1` throughout training; the gate is an inference-side
  contract, exercised by the unit test, not by the loss.

Token-level gating (delta only on ext-id query positions) is the stricter
alternative; it is **not** the first arm because self-attention's k/v at EN
positions are exactly what a JA query must learn to read. Keep it as the
fallback if mixed-prompt EN tags visibly degrade.

### Training

Joint: ext table (`--param global`, as every arm since 0816) **and** the
adapter LoRA, one AdamW, same loss/data as `synthja`:

```
--loss span --steps 12000 --batch_size 32 --param global --trust provenance
--train_registers tags,tags_alt,names,names_synth,names_synth_ja
--register_sampling names_synth_ja:0.2,names_synth:0.5
--register_span_scale names_synth:en_pinned=0.3
--adapter_lora r=16 --adapter_lora_targets self_qkvo,cross_q
--adapter_lora_lr 1e-4
```

Only the last line is new. LR for the LoRA is 10× below the table's 1e-3:
the adapter is pretrained and the delta should be a correction, not a
re-fit; the table LR stays where it is measured. The teacher is unchanged
(t5en arm through the *frozen* adapter — the cache already stores it), so the
LoRA is distilled toward the base adapter's own EN composition. That is the
point: the student adapter learns to do with ext rows what the base does
with EN pieces, no new target enters.

Warm start from `cjk_vocab_pack_synthja` rows (already the best span pack)
rather than the anchor init, so the arm measures *capacity added*, not
capacity-plus-relearning. Cold-init is the control if the warm arm is
ambiguous.

### Code

- `scripts/distill_cjk/adapter_lora.py` (new): `AdapterLoRA` — wraps the
  targeted `nn.Linear`s in a forward-hook style delta (do **not** override
  `forward`; the ComfyUI Adapter Loader's `forward_hook`-not-override
  invariant applies here too so the same module ships), owns the gate,
  `parameters()`, `state_dict()` in standard `lora_A/lora_B` naming under
  `llm_adapter.blocks.{i}.…` keys so the existing loader consumes it.
- `scripts/distill_cjk/config.py`: `--adapter_lora r=<int>`,
  `--adapter_lora_targets`, `--adapter_lora_lr`, `--init_pack <prefix>`
  (warm start). `ss_`-style metadata into the pack JSON `training` dict
  (`adapter_lora: {r, targets, lr}`) — the misplaced-key-silent-default
  lesson from turbo applies: verify arms via metadata, not argv.
- `scripts/distill_cjk/distill.py::train_arm`: build → attach → add the
  LoRA params as a second param group → train. `save_vocab_pack` writes a
  sibling `<out>.adapter_lora.safetensors` when present.
- `bench/cjk_adapter/run_bench.py`: `--adapter_lora <path>` loads the
  sibling onto `anima.llm_adapter` before encoding (mirror of the ext-table
  append at line ~318). The grid is the gate, so this lands with the arm.
- `tests/test_cjk_adapter_lora.py`: (a) **EN bit-exact** — random EN id
  batch through base vs LoRA-wrapped adapter with random non-zero B:
  `torch.equal`; (b) gate flips on one ext id; (c) state-dict key naming
  round-trips through the Adapter Loader's key parser.

## Phases

### Phase 0 — wiring (CPU, ~½ day)

`adapter_lora.py` + config + test. Gate: unit test green; a 200-step smoke on
the daemon shows both param groups moving and pack + sidecar written.

### Phase 1 — the arm (~1 GPU-h)

`cjk_vocab_pack_synthja_lora16` (warm from `synthja`), then both grids
(`ja_eval_prompts.json`, `ja_eval_prompts_names_mixed.json`, arms
`en,ja_t5en,ja_ext`). Cost: distill ~40 min + grids ~20 min, all daemon.

**Gate (all four, eyeball, this arm vs `synthja` span pack):**

1. **n1 / r3** full-JA Reimu: black hair, red bow, red-white miko, shrine.
   n2 Asuka: red plugsuit *or* orange twin-tails — either is a first.
2. **r1** keeps its partial gain (black-hair miko) — no regression.
3. **t1 / t2 / m1** clean; **no stray figures** (the §9 tell).
4. Mixed-prompt EN tags still land (r1's `red and white miko outfit`,
   m2's `vocaloid`) — the gate lets the delta touch EN tokens, this is
   where that shows if it hurts.

Health (not gates): span loss ≤ `synthja`'s 0.092; recovery_attn in the
0.85–0.92 band the rows-only arms occupy; `cos_native_vs_en_attn` floor
unchanged (the LoRA must not lift the *native* arm — if it does, it is
smoothing the readout, not composing).

### Phase 2 — only if Phase 1 is directionally right

- Rank sweep {8, 32} if names appear but weakly.
- MLP on if self/cross only gets bow-but-no-outfit style partials.
- Token-level gate if mixed-prompt EN tags degraded in gate 4.
- Cold-init control if the warm arm's gain could be the rows, not the LoRA
  (it cannot be — rows alone are `synthja` — but record it once).

Each is one arm, one grid. No corpus work anywhere in this line.

### Ship (folds into plan.md Phase 3)

The pack becomes three files: `.safetensors` (rows) + `.json` (mapping,
provenance, training) + `.adapter_lora.safetensors`. The LoRA rides the
Anima Adapter Loader unchanged (standard keys, forward-hook delta); the
gate travels as a tiny wrapper the loader already has room for (it already
patches `llm_adapter` for the ext rows). Documentation states the
dependency both ways: LoRA without the pack is a no-op (`g = 0` always);
pack without the LoRA is the §8 behaviour.

## Decision (pre-committed)

- Gate passes → ship Phase 3 **with** the LoRA; rare kanji names are in
  scope for v1. Then, and only then, the prose registers (D2 → STAIR →
  JESC) become worth sizing, and against *this* adapter, not the frozen one.
- Gate fails on 1 with 2–4 clean → Phase 2, at most two arms.
- Gate fails on 1 *and* Phase 2 doesn't move it → **ship Phase 3 with
  `cjk_vocab_pack_synthja` as is** and scope rare kanji names out of v1
  (users type `hakurei reimu` latin — the mixed register r1 already works).
  The line then closes; the next rung would be plan.md 2-iii (full adapter
  finetune), which is **not proposed**: it forks the adapter for every
  user and gives up the EN guarantee.

## Not in this line

- **mT5 / any vocab swap.** Ext rows are already Qwen BPE pieces
  (multi-char, token-aligned with the Qwen stream, exact anchor init);
  rare names fall to char pieces in any tokenizer. Does not touch the
  mechanism.
- **More pairs, JESC, STAIR, D2 growth.** §8 falsified visits as the cause;
  span-less corpora are inert under the loss this arm uses.
- **`attn` loss.** §9. Health metric only.
- **Lexicon single-row names.** Subsumed: if the adapter can compose, it
  doesn't need them; if it can't, a single row still sits in an
  all-new-row context.

## Risks

1. **The LoRA learns to read the teacher's *pad/attention-sink* structure
   rather than compose** — would show as native-arm floor rising
   (`cos_native_vs_en_attn`). Health check above; fallback = token gate.
2. **Mixed-prompt EN regression** (gate 4). Fallback = token-level gate.
3. **Ship surface**: the Adapter Loader must apply a gated delta; if its
   patching can't express the gate, the pack node grows one wrapper. Not a
   research risk; noted so it isn't discovered at publish time.
4. **Over-fitting the 3,000 synth names.** Holdout is register-mixed; the
   grid's n3 (place name) and c-prompts are the no-name controls — if they
   drift from `synthja`, the delta is doing more than name composition.
