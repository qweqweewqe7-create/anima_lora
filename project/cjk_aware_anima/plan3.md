# CJK-aware Anima — plan3: adapter capacity (ext-gated LoRA on the LLM Adapter)

*Line home: [`plan.md`](plan.md) is the vocab-pack line (rows only, adapter
frozen). This document promotes its **2-ii escalation rung** into its own
line, now that the capacity signal it was gated on has appeared
([`report_0827_names_synth.md`](report_0827_names_synth.md) §8–§9).
Written 2026-08-29.*

Status: **Phase 1 RUN (2026-08-30) — gate 1 failed, gate 2 regressed →
Phase 2, one arm** (§Phase 1 result, §Phase 2). Phase 0 wiring landed in
`3d604a9a`. Nothing here needs a cache rebuild: `cache_synth2` (261k pairs,
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
0.85–0.92 band the rows-only arms occupy. ~~`cos_native_vs_en_attn` floor
unchanged~~ — **vacuous under the sequence gate**: the native arm has no ext
id, so `g = 0` and it is bit-identical by construction. Risk 1 ("smoothing
the readout, not composing") has to be read off `recovery_attn` — per
register, `attn_by_register` in the distill envelope.

### Phase 1 result (2026-08-30)

Arm `cjk_vocab_pack_synthja_lora16` (daemon `20260830-181525-ded029`,
envelope `bench/cjk_distill/results/20260830-1815-plan3-lora16`; grids
`bench/cjk_adapter/results/20260830-1907-plan3-lora16-grid`,
`-1924-…-mixed-grid`). Metadata verified: r=16, α=16, 5 targets, lr 1e-4,
`init_pack=synthja`, sidecar 3.9 MB.

| | `synthja` | `lora16` | band |
|---|---|---|---|
| held-out span | 0.085 | **0.041** | ≤ 0.092 ✓ |
| `recovery_attn` | 0.901 | **0.449** | 0.85–0.92 ✗ |
| held-out attn loss | 0.236 | 0.458 | |
| attn cos→EN, names / names_synth / names_synth_ja | .923 / .913 / .452 | **.666 / .355 / .225** | teacher .942 / .921 / .494 |
| attn cos→EN, tags / tags_alt / commentary | .404 / .469 / .976 | .424 / .450 / .984 | unchanged |

The flat `recovery` sat at 0.107 the whole run (0.101 baseline = `synthja`'s
final, i.e. the warm start loaded) — that metric is capped by tokenization
and is uninformative here, as `distill.py:198` already says. The honest one
moved **down, and only on the name registers**, while span loss (held-out
too, so not overfitting the 3k synth names) halved. Mechanism: `loss_span`
is a segment-mean cosine per aligned span — invariant to how a name's content
is distributed across its positions — and a LoRA on self-attention can
satisfy it by *smearing* the neighbourhood's content across tokens. Rows
alone cannot mix neighbours, which is why `synthja` scored 0.90 on the same
metric.

Grid, `ja_ext` arm, vs `synthja`:

| gate | verdict |
|---|---|
| 1 · n1/r3 Reimu | ✗ — red-haired girl, white cap, forest; no black hair / bow / miko |
| 1 · n2 Asuka | ~ red suit + visor (arguably a plugsuit — first time), no twin-tails |
| 2 · r1 keeps black-hair miko | ✗ **regression** — miko outfit kept, hair turned red |
| 3 · t1/t2/m1 clean, no strays | ✓ — t1/t2 clean, m1 *improved* (mic, pose), zero stray figures (`synthja` n2 had one) |
| 4 · mixed EN tags land | ✓ — r1 miko, m2 vocaloid, a1 red plugsuit (a1 twin-tails lost) |

The visual tell is the metric tell: **every Reimu prompt renders red hair**
— `red bow / red-and-white` from the name's neighbourhood lands on the
subject. So the LoRA does add composition capacity (no strays, Miku better,
Asuka gains a suit) but span-only training lets it spend that capacity on
redistribution rather than on reproducing the teacher's per-token structure.

### Phase 2 — regularise the readout (one arm, decided 2026-08-30)

Phase 1 is directionally right (capacity shows) and fails on *how* the
capacity is spent, so the Phase 2 menu as first written does not apply:
rank {8, 32} and MLP add more of the same un-constrained freedom; the
token-level gate answers a gate-4 failure that did not fire.

**Arm `cjk_vocab_pack_synthja_lora16_reg`**: identical recipe, loss
`span:1.0,attn:0.25`, warm from `synthja` rows. The attn term is the
position-wise, sink-aware sequence readout — it charges the smear the pooled
span cosine is blind to, so the delta has to compose rather than
redistribute. This deliberately reverses §Not-in-this-line's "no `attn`
loss": §9's evidence was rows-only, where extra sequence pressure had no
capacity to absorb it and made renders worse; with the LoRA the failure is
the mirror image. `config.py` currently refuses `--adapter_lora` together
with the attn loss on the §9 reasoning — relax that guard (it is a
health-metric interaction, not a wiring hazard) and note why.

Pass: `recovery_attn` on the name registers back ≥ 0.85 **and** r1's hair
is black again (gate 2 restored) — then read gate 1. Weight 0.25 is the
only knob; 0.5 was §9's value and is the one retry if 0.25 leaves
`recovery_attn` < 0.85 with span still ≤ 0.092.

Deferred, only if the reg arm passes gate 2 but not gate 1: rank 32 (the
delta is then composing, and may simply be weak). No corpus work anywhere in
this line.

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
- Gate fails on 1 with 2–4 clean → Phase 2, at most two arms. *(Phase 1
  outcome: fails 1, regresses 2, 3–4 clean → Phase 2 as rewritten above —
  the reg arm plus at most one weight retry.)*
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
- **`attn` loss as the objective.** §9. Health metric only — but as a
  *regulariser* alongside span it is the Phase 2 arm (see there for why §9
  does not transfer to the capacity setting).
- **Lexicon single-row names.** Subsumed: if the adapter can compose, it
  doesn't need them; if it can't, a single row still sits in an
  all-new-row context.

## Risks

1. **The LoRA learns to satisfy the pooled span objective by smearing
   rather than composing** — *this fired in Phase 1* (`recovery_attn`
   0.90 → 0.45 on names, red-hair bleed in every Reimu render). The
   native-floor check written here originally cannot see it (gate = 0 on
   the native arm); `attn_by_register` can. Fallback = attn regulariser
   (Phase 2), then token gate.
2. **Mixed-prompt EN regression** (gate 4). Fallback = token-level gate.
3. **Ship surface**: the Adapter Loader must apply a gated delta; if its
   patching can't express the gate, the pack node grows one wrapper. Not a
   research risk; noted so it isn't discovered at publish time.
4. **Over-fitting the 3,000 synth names.** Holdout is register-mixed; the
   grid's n3 (place name) and c-prompts are the no-name controls — if they
   drift from `synthja`, the delta is doing more than name composition.
