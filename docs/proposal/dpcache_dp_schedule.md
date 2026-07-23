# DPCache — DP-planned global skip schedule as a third Spectrum schedule mode

Status: **DRAFT 2026-07-23 — nothing built. Phase 0 (offline paper-check on
existing counterfactual data) gates Phase 1 (planner + `--spectrum_schedule dp`);
Phase 2 (matched-compute A/B vs `window`/`sea`) is the ship gate; the hybrid
dp+sea mode is explicitly speculative and gated on Phase 2 evidence.**

Paper: DPCache (Cui et al., Alibaba, arXiv:2602.22654v2) — training-free
diffusion acceleration that plans the cache schedule *globally*: calibrate a few
full-step runs, build a **Path-Aware Cost Tensor** `C[i,j,k]` (cumulative L1
feature-prediction error of skipping from step `j` to `k` given preceding key
step `i`), then dynamic-programming-select the optimal `K` key timesteps for a
fixed compute budget. Full forwards only at key steps; skipped steps get
forecast features. Claims 4.87× on FLUX with better quality than
TeaCache/TaylorSeer/SpeCa and much higher fidelity-to-baseline (PSNR +2.4,
LPIPS −0.11).

## Premise

Spectrum's when-to-skip decision now has two modes (`docs/inference/spectrum.md`):
the content-blind **growing window** (default) and the content-adaptive **SEA
trigger** (`--spectrum_schedule sea`, grafted from SeaCache —
`docs/findings/seacache_sea_decision_metric.md`). Both are *local* policies: the
window is a fixed cadence, SEA is a greedy accumulate-and-fire threshold. Neither
sees the global structure of the trajectory, and the SEA finding closed with a
concrete symptom of that blindness:

> the σ<0.45 tail carries ≈0% of SEA-weighted skip-cost … yet the blind schedule
> force-computes the last 3 steps — a concrete reallocation the adaptive trigger
> could exploit.

DPCache is precisely the reallocation machine: given a compute budget `K`, the DP
finds the provably-optimal *placement* of the `K` real forwards under the cost
model — it will abandon the dead tail and spend those forwards on critical
transitions, something neither a growing window nor a greedy threshold can do
(the threshold only reacts once cost has already accumulated; it cannot *move*
a forward earlier).

The integration follows the exact pattern the SEA graft validated: **swap only
the when-to-skip decision, keep everything else** — Chebyshev forecasting, the
per-step head recompute (`t_embedder → final_layer → unpatchify`), warmup/stop
forcing. DPCache's reuse is forecast-style (each skipped step gets a fresh,
distinct predicted feature, not a frozen output) and the paper states the
framework is predictor-agnostic, so Spectrum's forecaster slots in directly.
Because the per-step `noise_pred` reconstruction still runs every step, the
sampler-boundary plug-ins (SMC-CFG, mod-guidance) compose unchanged — the same
composition argument already on record for SEA.

Operational side-benefit: the schedule is computed **once, offline** (the DP is
`O(KT²)`, negligible), keyed on schedule geometry exactly like
`output/spectrum_sea_delta.json`. It *replaces* SEA's auto-δ matched-compute
binary search with something cleaner — specify the budget `K` directly, get the
optimal placement for it. Zero per-step overhead, fully deterministic,
compile-friendly (the skip pattern is known before step 0).

## The open question this proposal actually decides

DP-planned (global, content-blind) vs SEA (local, content-adaptive) is a genuine
empirical toss-up on Anima, and our own data cuts both ways:

- **For DP**: the paper beats the locally-adaptive methods (TeaCache, SpeCa) on
  exactly the fidelity metrics where global planning should win, and its
  calibration ablation shows the schedule is content-agnostic (1 sample ≈ 11
  samples ≈ different prompt source → identical schedule).
- **For SEA**: Phase 1 of the SeaCache eval proved there IS real per-prompt
  signal on Anima — step-stratified, SEA-filtered input distance predicts *which
  prompt* is costlier to skip at a given step (ρ +0.51). A fixed global schedule
  cannot capture that, by construction.

These are orthogonal axes (step *placement* vs prompt *adaptivity*), so the
ceiling is probably a hybrid — but the cheap, decision-grade question is whether
globally-optimal placement alone beats the shipped modes at matched compute.

## Prior art in-tree (why this is cheap)

- **The decision-vs-reuse split is already established.** The SEA graft
  (`networks/spectrum.py` + `networks/spectrum_sea.py`) proved the scheduler is
  a clean seam: `spectrum_denoise` picks refresh-vs-forecast per step behind one
  predicate (`schedule == "sea"` branch, ~L303). A `dp` mode is a precomputed
  boolean mask consulted at the same seam — strictly simpler than SEA (no
  per-step FFT, no accumulator state).
- **The cost-measurement harness already exists.**
  `_archive/bench/spectrum_sea/phase1_counterfactual.py` computes the **true
  counterfactual skip-cost** — actually cache the step (forecast both branches,
  run only the head), CFG-combine, measure x̂₀ deviation vs full compute. That
  is a per-step slice of exactly the quantity PACT wants; extending it from
  single-step skips to `(i, j, k)` segments is the calibration stage.
- **The evaluation discipline is already written down.** Step-stratification
  (detrend the shared σ-envelope before correlating), sanity-check estimators on
  monotone-only synthetic input, matched-compute comparison, eyeball/CMMD ship
  gate — all from `seacache_sea_decision_metric.md`. This proposal inherits
  them wholesale.
- **The prior-inversion record says: verify, don't trust.** CTCal and the
  spectral-fraction metric both inverted published priors on Anima
  (`ctcal_premise_inverted_on_anima.md`, `spectral_fraction_metric_inverts.md`).
  Two of DPCache's premises get re-verified here rather than assumed
  (open-loop calibration fidelity, content-agnosticism — see Phase 1).

## Known weaknesses of the paper (and how the phases address them)

1. **PACT is open-loop.** Calibration measures skip errors along the *undrifted*
   full-step trajectory; at inference, forecast errors feed back into `x_t`.
   The cumulative-L1 segments and one-predecessor conditioning only partially
   model closed-loop drift. → Phase 1 builds PACT from the **true
   counterfactual** harness instead of the paper's open-loop replay, and
   Phase 0 measures how much the two disagree before any planner code lands.
2. **Content-agnosticism is a claim about FLUX/HunyuanVideo, not Anima** — and
   our SEA result already shows per-prompt cost variance exists here. → Phase 1
   re-runs the paper's calibration ablation on Anima (does the planned schedule
   change across disjoint calibration prompt sets?). If schedules diverge,
   content-agnostic planning is out and only the hybrid survives.
3. **Less headroom at Anima's step counts.** The paper plans `K` out of `T=50`;
   Anima typically samples 24–30 steps, so the planner has fewer degrees of
   freedom and the margin over the window shrinks. → Phase 0 quantifies the
   margin on paper before anything is built.
4. **"Beats the full-step baseline" (+0.028 ImageReward)** is reward-metric
   noise/bias, not free quality. Ignored; the fidelity numbers (PSNR/LPIPS) are
   the credible result.

## Phases

### Phase 0 — offline paper-check (no inference-path code)

Rebuild the cost structure from recorded trajectories and run the DP on paper.
`bench/spectrum_dp/run_bench.py` (shares `bench/_common.py`, standard
`result.json` envelope):

1. Record full-step trajectories (features + `x_t`) for the SEA bench's prompt
   set at the shipped geometry (24–30 steps, CFG, 1024²).
2. Build **open-loop PACT** exactly per the paper (Chebyshev-forecast segment
   `j→k` from features at `i, j`; cumulative L1 over skipped steps, final-layer
   features only — `T≈28` makes the tensor trivially small).
3. DP-select schedules for `K` matched to the window's and SEA's realized
   post-warmup refresh counts (the matched-compute discipline from the SEA
   eval).
4. Score all three schedules under the **true single-step counterfactual cost**
   already measured by `phase1_counterfactual.py` (re-run, not reused — same
   harness, current checkpoint).

**Gate**: DP's planned schedule must beat the window's realized schedule on
summed counterfactual cost at matched `K`, and at least tie SEA's. If it can't
win on its own cost model's home turf, stop — nothing ships, total cost is one
bench.

Deliverable also includes the **tail check**: does the planner reallocate the
σ<0.45 force-computed steps as the SEA finding predicted? (If yes, that's the
mechanism; if it wins *without* touching the tail, the win is coming from
somewhere unmodeled — investigate before Phase 1.)

### Phase 1 — closed-loop PACT + the planner + `--spectrum_schedule dp`

Only on a Phase-0 pass:

- **Closed-loop PACT**: extend the counterfactual harness from single-step
  skips to `(i, j, k)` segments — actually forecast across the segment with the
  drifted `x_t` feeding back, measure cumulative x̂₀ deviation. Compare against
  the open-loop tensor (this is the paper-weakness-1 measurement). Use
  whichever the Phase-0 scoring says is more faithful.
- **Content-agnosticism ablation on Anima**: build PACT from disjoint
  calibration prompt sets (n=1 / n=5 / n=11, anime prompts vs DrawBench-style);
  compare planned schedules. Identical → one cached schedule per geometry, like
  the paper. Divergent → content-agnostic planning is falsified on Anima
  (recorded as a finding either way).
- **Planner + integration**: `networks/spectrum_dp.py` (PACT build + DP +
  schedule cache, mirroring `spectrum_sea.py`'s role), a `dp` arm in
  `spectrum_denoise`'s decision seam, schedule cached to
  `output/spectrum_dp_schedule.json` keyed on the same geometry tuple as the
  SEA δ (steps / warmup / stop / K / cfg / sampler / H×W — not the prompt).
  Calibration runs route through `make gen` (daemon) like every other bench.
- **Invariants** (pinned by tests, per the Tier-1.5 contract):
  - `dp` mode changes *only* which steps refresh — forecast + head recompute
    still run every step (the SMC-CFG/mod-guidance composition invariant, same
    test shape as SEA's).
  - Warmup and stop-caching forcing are respected: `M ≥ warmup` initial steps
    and the final forced steps are constraints on the DP, not overridable by it
    (relaxing the tail forcing is a *separate, explicit* knob — it is the
    experiment, not a silent default).
  - Missing/stale schedule cache → hard fall back to `window`, never a partial
    schedule.

### Phase 2 — matched-compute A/B (ship gate)

Three-arm comparison at identical realized refresh counts: `window` vs `sea`
vs `dp`, over the SEA eval's prompt grid. Fidelity-to-baseline (x̂₀ RMSE / CMMD
vs the unaccelerated run) as primary, eyeball A/B as the gate — the same
protocol that shipped (and nearly didn't hold back) the SEA node mirror. `dp`
ships as opt-in only on a win or clear tie-with-simplification; `window` stays
the default regardless.

### Phase 3 — hybrid dp+sea (speculative, gated on Phase 2)

Only if Phase 2 shows `dp` and `sea` winning on *different* prompts/steps
(i.e. the orthogonality thesis holds empirically): DP-planned baseline schedule
+ SEA-triggered *extra* refreshes on top (never removals — the plan is the
floor, the trigger only adds). Budget accounting and the δ/K interaction get
designed then, not now.

## Non-goals / scope guards

- **Not a Spectrum replacement and not a new reuse policy.** Chebyshev
  forecasting, head recompute, cond/uncond-separate forecasters are untouched.
  Adopting DPCache's TaylorSeer predictor or final-layer-only caching is out of
  scope (Spectrum's single-hook capture already has the same memory profile).
- **No per-prompt planning.** If content-agnosticism fails on Anima, the answer
  is the Phase-3 hybrid, not running calibration per prompt at inference time.
- **No training-loop involvement, no config-default changes.** `window` remains
  the default schedule; `dp` is opt-in behind the existing
  `--spectrum_schedule` flag.
- The ComfyUI node mirror is deliberately deferred until the library mode
  survives Phase 2 (the SEA precedent: library first, node after the gate).

## Falsifiers (cheap exits)

1. **Phase 0**: DP at matched `K` doesn't beat the window's realized schedule on
   true counterfactual cost → global planning adds nothing over a monotone
   cadence at Anima's step counts → stop, archive the bench, one finding doc.
2. **Phase 1**: open-loop and closed-loop PACT rank schedules differently *and*
   the closed-loop planner loses its Phase-0 margin → the paper's calibration
   premise doesn't survive feedback drift on Anima → stop (this would be the
   third prior-inversion on record).
3. **Phase 1**: planned schedules diverge across calibration sets → content-
   agnosticism falsified on Anima → skip Phase 2 for standalone `dp`, re-scope
   to the hybrid or stop.
4. **Phase 2**: quality regression vs `sea` at matched compute → SEA's
   per-prompt adaptivity beats global placement here → keep `sea`, archive `dp`
   with the A/B data.
