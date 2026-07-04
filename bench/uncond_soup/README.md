# uncond_soup — does soup-with-uncond really help?

Phase 0 of the Self-Soupervision question (Fuller, Green & Shelhamer,
arXiv:2602.02890): model soups whose ingredients are prepared by a
**label-free inter-training** stage. For a flow-matching DiT the natural
"SSL" is unconditional training — the FM objective needs no labels; the
caption is the only label in this pipeline — so the recipe maps to:

```
Anima base (stock)
  └─ uncond inter-train on a bounded 3-artist pool     caption_dropout_rate=1.0
       ├─ captioned fine-tune, target artist, seed 1   ┐
       ├─ captioned fine-tune, target artist, seed 2   ├─ "self" family
       └─ captioned fine-tune, target artist, seed 3   ┘
                                        └── uniform ΔW soup = self:soup
Anima base (stock)
       ├─ captioned fine-tune, target artist, seed 1   ┐
       ├─ captioned fine-tune, target artist, seed 2   ├─ "base" family (control)
       └─ captioned fine-tune, target artist, seed 3   ┘
                                        └── uniform ΔW soup = base:soup
```

The two families share fine-tuning seeds (paired data order + FM training
noise), so the only difference is the unconditional initialization. This also
makes the bench a seed-lottery mitigation test (arXiv:2606.20536, "The FID
Lottery"): souping over training seeds averages over the training lottery,
and every gate below is judged against the sampling-noise floor
`sigma_within` rather than raw deltas.

## Scope guards

- **Bounded pool**: exactly 3 artists for the uncond phase (the target artist
  should be one of them — matching the paper's "inter-train on unlabeled data
  from the task" setting). Supervision = single artist.
- **Plain LoRA only** (the lora.toml default: `down_init="weight_svd"`,
  optional T-LoRA). SVD-Down matters here: it pins every seed's A-init to
  top-r(W₀), putting the ingredients in the shared-init/different-stochasticity
  regime where linear mode connectivity (LMC) is empirically strongest.
  Hydra/chimera/ortho checkpoints are refused by the soup builder.
- **Soup is at the ΔW level, exactly** — a weighted average of rank-r deltas
  is representable without approximation as one rank-N·r LoRA by block
  concatenation (`soup.py`). Parameterwise (A/B) averaging is wrong and is
  never done: `avg(B)@avg(A) ≠ avg(B@A)`, and weight_svd's randomized-SVD
  sign ambiguity would make row-wise A averaging cancel.

## Files

| File | Role |
|---|---|
| `launch.py` | Emits the 7 training commands (1 pool + 2×3 fine-tunes). Prints; `--write runs.sh` to save. Never launches. |
| `soup.py` | Exact ΔW-average soup / λ-interpolation builder (also a library for `probe.py`). Folds alpha and channel-scaling `inv_scale` per ingredient; output has `alpha = rank` (scale 1) and carries soup provenance in metadata. |
| `probe.py` | CMMD-scores all arms against the target artist's held-out split and reduces to the gates. Reuses `bench/seed_lottery/probe.py`'s trainer-faithful split + compile + CMMD loop. |

## Running

```bash
# 1. Plan the runs (pick 3 artist dirs under post_image_dataset/resized/)
python bench/uncond_soup/launch.py --pool_artists artistA artistB artistC

# 2. Run/queue the 7 printed train.py commands (pool first — the self family
#    fine-tunes load its checkpoint via --network_weights).

# 3. Score (the launch output prints this line with paths filled in)
python bench/uncond_soup/probe.py --method lora --preset default \
    --target_artist artistA \
    --self_ckpts output/ckpt/uncondsoup_self_s100{1,2,3}.safetensors \
    --base_ckpts output/ckpt/uncondsoup_base_s100{1,2,3}.safetensors \
    --pool_ckpt output/ckpt/uncondsoup_pool.safetensors \
    --validation_split_num 16 --validation_seed 42
```

`--validation_split_num` / `--validation_seed` must match between launch and
probe — that is what makes the held-out set the one no run ever saw. The
uncond phase still requires TE caches on disk (embeddings are loaded, then
dropped rows are swapped to the T5("") sidecar, which `train.py` stages
automatically when caption dropout is on).

## Gates (CMMD, lower = better)

1. **LMC** — every λ=0.5 pair midpoint within a family must satisfy
   `cmmd(mid) ≤ mean(endpoints) + sigma_within`. If this fails, souping's
   premise is broken and the verdict short-circuits to `LMC_FAILS`
   (fix init sharing before re-running — e.g. branch fine-tunes from one
   saved init, or pin the weight_svd RNG).
2. **SOUP** — `soup ≤ best ingredient + sigma_within` per family (souping at
   least doesn't lose to seed-fishing).
3. **UNCOND (headline)** — `base_soup − self_soup > gate_mult·sigma_within`
   → `UNCOND_HELPS`; symmetric for `UNCOND_HURTS`; else inconclusive.

## Caveats / non-goals

- **Not compute-matched**: the self family gets the pool pre-train for free.
  A `UNCOND_HELPS` here licenses a Phase 1 with an equal-budget control
  (e.g. base family trained `pool_epochs` longer), not a shipping decision.
- The expected effect size is modest when all fine-tune data is already
  captioned — the paper's headline gains came from *shifted/unlabeled target*
  data. The interesting follow-up if Phase 0 passes: pools of genuinely
  uncaptioned images.
- `sigma_within` here is the *sampling* floor (K val seeds per fixed arm).
  The between-training-seed floor is `bench/seed_lottery/`'s job; with N=3
  ingredients this bench can't estimate it reliably and doesn't try.
- Memory guard: unconditional training pushes content into the
  style/unconditional pathway (see `project_lora_crossattn_learns_labeled_only`)
  and reshapes the CFG-uncond branch. Eyeball the contact sheet, not just the
  numbers.
