# Trajectory-resolved latent statistics — effective token usage measured *during* generation

Status: **DRAFT 2026-07-23 — nothing built. Phase 0 (passive recorder) gates
Phase 1 (atlas); Phase 2 (intactness gauge) is the actual product; Phase 3
(interventions) is explicitly speculative and gated on Phase 1/2 evidence.**

## Premise

The Anima DiT works in the Qwen Image VAE latent space (`z_dim = 16`,
per-channel `latents_mean` / `latents_std` shipped in the model config). The
anime domain is latent-sparse: flat cel shading, line art, and uniform
backgrounds mean most spatial tokens of a *finished* image carry near-zero
unique information under quantization. Static corpus statistics (quantize the
cached `.npz` latents, measure per-token / per-channel entropy) would confirm
that — but they are **reconstruction** statistics of clean images. They say
nothing about *when* along the σ trajectory each token's information
materializes, and the one intervention this repo already tried on the strength
of redundancy signals — the deferred-foveated merge — died precisely because
its damage was invisible to final-result framing until benched hard:

> the periphery soft blur is **constitutive** (P4t: no tail treatment recovers
> it) — `docs/inference/foveated.md`, archived 2026-07-03.

A region denoised at reduced token count never receives its high-frequency
detail. The failure lives in the **process**, not the endpoint: final-image
metrics on the fovea looked fine, the periphery *trace* was what flatlined.

This proposal inverts the order of operations. Before proposing any new
efficiency intervention, build the measurement layer:

1. **Measure statistics while generation runs** — per-step, per-token,
   per-channel — using signals the loop already computes for free.
2. Derive an **effective token usage profile** `E(σ)`: at each noise level,
   what fraction of tokens is still actively receiving information?
3. Turn the same traces into a **trajectory-intactness gauge**: a candidate
   intervention must keep the *whole generation process* statistically intact
   relative to baseline, not merely score well on the final image. "Intact on
   overall generation process, not only on final result."

Everything in Phases 0–2 is observability. No generation behavior changes.

## Prior art in-tree (why this is cheap)

- **Free per-step signals already exist.** The foveated runner accumulated
  `cfgdelta` (per-cell |v_cond − v_uncond|) and `x0var` (x̂₀ Laplacian energy)
  during early steps with "zero extra models, zero extra forwards"
  (`docs/inference/foveated.md` §Mechanism-2, `networks/foveated.py`). The
  recorder generalizes exactly this accumulation — full-trajectory, dumped to
  disk, no mask built, nothing acted on.
- **Process-level probes are an established pattern.** The main loop already
  carries capability probes that manipulate the *trajectory* to answer
  process questions: `ANIMA_TEXT_KNOCKOUT_SIGMA`, `ANIMA_FREEZE_GUIDANCE_SIGMA`
  (`library/inference/generation.py` ~L897–913). This proposal adds the
  passive complement: observe the trajectory instead of perturbing it.
- **Channel calibration precedent.** `docs/optimizations/channel_scaling.md` /
  `_archive/bench/channel_stats/channel_dominance_analysis.md` did the
  per-channel statistics program for DiT *activations*. This is the analog for
  the VAE latent trajectory.
- **The gauge has a validation target on day one.** The archived foveation
  runner still works (off by default). A trajectory gauge that cannot flag
  foveation's periphery flatline — a *known* constitutive process defect —
  is falsified immediately.

## The statistics

All computed on `x̂₀ = latents − σ·v` (the per-step denoised estimate, already
formed at `generation.py` L510/L1067 — flow-matching, so this is one FMA we
get for free), normalized per channel with the VAE's `latents_mean` /
`latents_std` so "a bit" means the same thing in every channel. "Token" =
16-px patch cell = a `2×2` latent-pixel cell aggregated to the DiT patch grid,
so maps are directly comparable to attention/token-count reasoning.

Per step `i` (σ = `sigmas[i]`), per token `p`:

| trace | definition | what it answers |
|---|---|---|
| **code(p, i)** | k-bit uniform quantization code of x̂₀ (default k=4/channel, knob) | the quantized-VAE view: which discrete cell the token estimate is in |
| **commit(p)** | last step where `code(p, ·)` changed | when did this token's content lock? |
| **activity(p, i)** | ‖x̂₀(p, i) − x̂₀(p, i−1)‖ | is information still flowing into this token? |
| **hf(p, i)** | Laplacian energy of x̂₀ in the token's neighborhood | foveation's x0var, kept for continuity — the trace that flatlined |
| **guide(p, i)** | ‖v_cond − v_uncond‖ per token (CFG runs only) | foveation's cfgdelta — where the prompt is steering |
| **cbits(c, i)** | entropy of channel c's code histogram over all tokens | per-channel information ramp — which of the 16 channels the anime domain actually uses, and when |

Derived scalars:

- **E(σ)** — effective token usage: fraction of tokens with
  `activity(p, i) > τ` (τ from the noise floor at σ→0). The headline curve.
- **commit-CDF** — distribution of `commit(p)` over σ. If 60 % of tokens have
  committed by σ = 0.4 on the anime corpus, that is the quantified headroom
  for *any* late-step token intervention — and if the CDF is flat, the whole
  efficiency thesis dies cheaply, in Phase 1, without shipping anything.

Storage: one `.npz` sidecar per generation (fp16 maps + a small JSON header:
seed, prompt hash, resolution, sampler, step count, k). At 28 steps × ~4 k
tokens × 6 traces ≈ a few MB — fine for bench runs, which is the only
intended producer.

## Phases

### Phase 0 — passive recorder (`--traj_stats`)

A small observer class, `library/inference/traj_stats.py`, threaded through
the denoise loops the same way `smc_cfg` is (constructed in the engine from
`args`, `None` when off):

- `record(i, sigma, latents, noise_pred, uncond_noise_pred=None)` called once
  per step, after the CFG combine, before the sampler step.
- Hook sites: `generate_body` main loop (`library/inference/generation.py`
  ~L916), the tiled loop (~L393), and `networks/spectrum.py::spectrum_denoise`
  (so Spectrum-composed runs are measurable too — the gauge's "known-good"
  arm needs this).
- **Invariants** (each pinned by a test, per the Tier-1.5 contract):
  - Recorder on/off produces **bit-identical** latents (pure observation; all
    stats math on `.float()` *copies*, never in-place, and never feeding back).
  - 5D discipline: in-loop latents are `(B, C, 1, H, W)` — the recorder
    squeezes **dim 2 explicitly** (never bare `squeeze()`; see the CLAUDE.md
    dim-2 invariant) and asserts 4D before any FFT/Laplacian helper.
  - Off by default; zero allocations when off (`None` short-circuit, same as
    the other plug-ins).
- Overhead budget: < 2 % step time at 1024² (quantize + diff + one 3×3 conv,
  all on-GPU, dumped async at end). Measured in the Phase 0 bench.

### Phase 1 — the atlas (anime-domain trajectory statistics)

`bench/traj_stats/run_bench.py` (shares `bench/_common.py`, drops the
standard `result.json` envelope into `bench/traj_stats/results/`):

1. **Generation arm**: seed × prompt grid over anime prompts via the recorder
   (routed through `make gen` so it queues behind training instead of
   OOM-colliding). Aggregate `E(σ)`, commit-CDF, `cbits(c, ·)`.
2. **Real-image arm**: DirectEdit inversion
   (`library/inference/editing/directedit.py`) replays a *real* cached-corpus
   image as a trajectory — recorder on the inversion pass gives the same
   traces for ground-truth anime images, tying the during-generation
   statistics back to the static corpus-cache statistics (which this bench
   also computes from `post_image_dataset/lora/*.npz` as the zero-cost
   baseline column).
3. Deliverable: a short report answering (a) how front-loaded is token
   commitment in this domain, (b) which channels carry it, (c) does generation
   match inversion (if not, generated trajectories are off-manifold and the
   corpus stats don't transfer — important negative result on its own).

### Phase 2 — the trajectory-intactness gauge

The product. For a candidate intervention X and a fixed (prompt, seed, steps):

    D(X) = per-σ divergence profile between X's traces and baseline's traces
           — token-wise x̂₀ RMSE(σ), |E_X(σ) − E_base(σ)|, hf-trace delta,
           commit-CDF shift — reported as a *curve*, not one scalar.

Calibration, both directions:

- **Known-bad**: the archived foveation runner (`--fovea_sigma_c 0.75`). The
  gauge must show the periphery `hf` trace flatlining below σ_c and a
  commit-CDF hole — i.e. it must rediscover P4t from traces alone, *without
  looking at the final image*. This is the acceptance test.
- **Known-good**: Spectrum and SMC-CFG composes (shipped, quality-neutral by
  their own benches) must show small, structureless D. If the gauge flags
  shipped-good methods, τ / normalization get re-tuned before anyone uses it.

Output: `bench/traj_stats/gauge.py --baseline <dir> --candidate <dir>` →
verdict bands (calibrated in this phase, provisional: intact · perturbed ·
process-broken), riding the same `result.json` envelope.

### Phase 3 — interventions (gated, speculative)

Only if Phase 1 shows exploitable structure (front-loaded commit-CDF, skewed
`cbits`), ranked by the Phase 2 gauge before any quality bench is spent:

- **Entropy-aware tier routing** (training-side, safest): feed per-image
  latent entropy into `choose_edge` (`library/datasets/buckets.py` L116) so
  flat cel-style images train a tier down. No inference-time process risk at
  all — the gauge is irrelevant here, corpus stats from Phase 1 suffice.
- **Committed-token compute reuse** (inference-side): below a σ threshold,
  reuse the previous step's velocity for tokens whose code has been stable
  for m steps (a delta-token / cache-skip scheme — tokens keep their
  *identity and resolution*, unlike merging; staleness is bounded and
  refreshable, unlike foveation's constitutive pooling). Gauge-gated.
- **σ-scheduled channel truncation**: if `cbits` shows late-σ channels idle,
  drop them from the stats/guidance path — *measurement first*.

Explicitly **not** proposed: re-running token *merging* with better selection.
P3/P4t closed that line — the defect was the mechanism (reduced-resolution
denoising), not the mask, and a better redundancy prior does not change that.

## Non-goals / scope guards

- Phases 0–2 never change a generated image. The recorder is falsifiable on
  bit-exactness and ships with that test.
- No new models, no extra forwards, no training-loop involvement (the
  training-side Phase 3 item consumes Phase 1 *outputs*, not the recorder).
- `k`, τ, and verdict bands are bench-calibrated knobs, not shipped defaults —
  nothing here touches `configs/base.toml`.
- Tiled path support is best-effort in Phase 0 (record post-blend only);
  per-tile traces are out of scope.

## Falsifiers (cheap exits)

1. Phase 1 commit-CDF ≈ uniform in σ → no late-step headroom → Phase 3
   inference items die; only tier routing survives (and only if the *static*
   corpus stats are skewed).
2. Generation-arm vs inversion-arm traces disagree structurally → corpus
   statistics don't transfer to generation → restrict all claims to img2img /
   editing paths.
3. Gauge can't separate foveation (known-bad) from Spectrum (known-good) →
   the trace set is insufficient; stop before anyone trusts it.
