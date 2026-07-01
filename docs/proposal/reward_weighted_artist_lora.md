# Reward-weighted artist LoRA — learn the *style manifold*, not the training images

Status: **PROPOSED — not started.** Design-only; no code, no bench yet.

- Planned script: `scripts/rwr/` (new; ReST grow/improve loop over the existing
  CFM inner loop — reuses `library/runtime/harness.py::build_anima`, does **not**
  fork `train.py`).
- Planned bench: `bench/rwr/verifier_probe.py` (new; Phase-0 verifier validation —
  can a frozen feature-space judge separate "this artist's style" from "this exact
  training image"?).
- Premise sources: `docs/methods/turbo.md` (the <1 s 4-step generator this reuses,
  and its DP-DMD diversity-collapse tendency this must not inherit),
  `docs/proposal/turbo_caption_ranking.md` (the "relative FM-ranking at shared
  `(x_t, ε, t)` is a valid compass" reward-validation discipline),
  `docs/proposal/dave_mod_bestofn.md` (best-of-N order-statistics framing for the
  candidate pool), CLAUDE.md § "Text encoder padding" / "The DiT operates on 5D
  latents" (boundary invariants any generation loop must obey).
- Metric-trap priors this design must obey: CMMD is distribution-level and **not
  per-image**; pose/diversity effects **only show in sample grids** (the PE-pooled
  metric is blind to pose — see `turbo.md` § anchor-fidelity caveat); validate any
  learned score against eyeball before trusting it
  (`docs/findings/spectral_fraction_metric_inverts.md` lesson); real captions
  mandatory.

## The claim being made precise

Standard LoRA training minimizes flow-matching loss toward the **training images**.
Its only fixed point is *put mass on those images*. On a handful of images with a
low-rank adapter, the global optimum **is memorization** — caption dropout and
augmentation only regularize *away* from the images; nothing in the objective
rewards a *new* image that is genuinely in the artist's style. There is no term
that can express "this held-out sample is this artist."

A **distributional reward** can. Replace "match this `x0`" with "is this a
plausible draw from the artist's style distribution, at high fidelity?" — a reward
that is *maximized by novel in-style samples* and *indifferent* to reconstructing
any training image. That single sign flip is the entire proposal. It reframes the
adapter as a **policy** `πθ(x | c)` and the training target as the KL-regularized
policy-improvement fixed point:

```
target   πθ(x | c) ∝ π_ref(x | c) · exp( verifier(x) / λ )
                     └─ caption-dropout base ─┘   └─ explicit reward ─┘
```

This is the Boltzmann-tilted improved policy from max-entropy RL. Two facts make it
concrete rather than analogy:

1. **`π_ref` already exists.** Classifier-free-guidance training (caption dropout)
   builds the unconditional branch `p(x)`. CFG at inference is *already* a
   policy-improvement operator, `p̃ ∝ p(x)·p(c|x)^α`, with the implicit classifier
   `log p(c|x)` as reward and guidance scale `α = 1/λ`. This proposal **swaps the
   implicit reward for an explicit verifier** — the only structural change.
2. **"Learn the artist, not the image" ⇔ the reward is distributional, not
   per-sample.** MLE cannot write this objective; a frozen feature-space judge can.

## Estimator: reward-weighted regression (RWR / ReST), not policy gradient

We use the **reward-weighted self-training** member of the diffusion-RL family
(RAFT / ReST), *not* DDPO/AlignProp (backprop-through-sampling) and *not* a GAN.
The load-bearing property:

> **The gradient never flows through the rollouts.** The reward touches only
> *which targets you regress onto*, never the backward path.

Per outer round:

1. **Grow (no_grad).** Generate `N` candidates per prompt under **loose / dropout
   prompts** (exploration over the style manifold, not the caption-specific image).
2. **Score (no_grad).** One frozen-verifier forward per candidate on the final
   image (not per denoise step).
3. **Select / weight.** Keep top-k, or weight each retained sample by
   `w = softmax(reward / λ)`. These weights **are** the self-normalized importance
   weights `exp(Q/λ)` of the RFM/SNIS estimator — we are running RFM by hand,
   without needing the Stein control-variate machinery, *because we can sample
   candidates*.
4. **Improve (grad).** Ordinary flow-matching loss on the selected/weighted samples
   — one forward+backward at a random `t`, **bit-identical cost and conditioning to
   the direct FM loss** we already trust.

Consequences:

- **Safe gradient.** Worst case (bad reward) = training on mediocre self-generated
  targets → *graceful degradation*, not divergence. No policy-gradient variance
  blow-up, no adversarial min-max.
- **Strong gradient.** The pull comes from *target quality* (in-style, non-memorized
  `x0`), not from a sharp reward gradient — stable magnitude, correct direction.
  This is why RWR/ReST are the reliable members of the diffusion-RL zoo.
- **No head on the DiT.** The verifier is an offline judge; it never hooks a block,
  never modifies the graph, never contributes a gradient. Anima's blocks are
  untouched (contrast the GAN lever in `turbo_dmd.py`, which taps teacher features).

## Why turbo generation is safe here — FM training is sampler-agnostic

The candidate-generation cost is the only marginal cost over normal training, and
the 4-step turbo student (`docs/methods/turbo.md`, <1 s @ 1024²) collapses it. The
obvious worry — "will a LoRA trained on 4-step turbo output be healthy under normal
28-step SDE sampling?" — **dissolves under one observation:**

> We train on the turbo *output image* (`x0`), not on turbo's trajectory or step
> count. The FM loss re-noises that `x0` at a fresh random `t`. It cannot tell — and
> does not care — whether the target came from a photo, a 28-step SDE sample, or a
> 4-step rollout.

So **no few-step property transfers**; the artist LoRA is trained as a normal
continuous velocity field and is healthy at 28-step SDE, any CFG. The step-count
mismatch is a genuine non-issue. Turbo composes with the in-progress artist LoRA
via linear LoRA composition (ranks add — `turbo.md` § inference), so the grow phase
stacks `turbo + current-artist-snapshot` and rolls candidates at K=4.

### What *does* transfer — and the mitigations

What transfers is turbo's **output distribution**, not its step count: reduced
diversity (the DP-DMD mode-collapse the diversity anchor exists to fight), the
CFG=4-baked look, and the distilled texture/over-bake signature. The verifier can
*prune* off-style samples but cannot *create* diversity turbo already collapsed —
selection is a ceiling, not a lift. Mitigations, all cheap:

1. **Turbo for volume, full-step for anchor.** ~80% turbo candidates (breadth) +
   ~20% full 28-step candidates (restore diversity *and* explicit CFG control).
2. **Accumulate, don't replace.** Keep the **real artist images** in every training
   round as high-weight positives — never let the pool go 100% synthetic (the
   standing antidote to generative-model autophagy / "curse of recursion", which the
   ReST outer loop otherwise courts once the artist LoRA is in the generator).
3. **Novelty term / dedupe.** Penalize candidates too close to each other or to the
   training images in verifier feature space (re-solving turbo's diversity-anchor
   problem one level up, at the data level).
4. **Validate at the deployment config, not the training loss.** Periodically sample
   the artist LoRA at real **28-step SDE + target CFG**, score + eyeball a grid for
   saturation/pose collapse; rebalance toward full-step candidates if it looks
   turbo-ish. This empirical check *is* the answer — training loss won't show it.

## The load-bearing risk: the verifier is the whole experiment

With `N` artist images you **cannot learn style from scratch** — it must be
*generalized from a pretrained prior*. Non-negotiables:

- **Feature-space, frozen, pretrained.** Score cosine to the artist **centroid** in
  a frozen style/perceptual embedding. Anima already ships **PE-Core / PE-Spatial**
  (loaded in preprocessing) and a tagger — a style-embedding verifier is nearly free
  and never touches the DiT. Feature-space + frozen ⇒ far harder to hack than a
  co-trained pixel discriminator, and it *cannot* overfit the tiny positive set into
  rewarding memorization the way a trained discriminator does.
- **Separate the two objectives.** `reward = style_adherence (centroid cosine)` **+**
  `λ_q · generic_quality (artist-agnostic aesthetic)` **−** `λ_n · novelty_penalty`.
  "Adherent", "high-fidelity", and "not-a-copy" become three tunable knobs — exactly
  the "confusable but genuinely high fidelity" signal in plain terms.
- **Anchor to the reference or it reward-hacks.** Keep the caption-dropout entropy +
  a base-LoRA anchor (the KL to `π_ref`), or RWR drifts toward the style-embedding
  maximizer, which is not art. Every diffusion-RL result depends on this.

**Phase 0 exists precisely to falsify the verifier before any training.** If a
frozen judge cannot separate held-out same-artist images from the training set (and
from generic/other-artist negatives) in feature space, RWR is *worse* than MLE and
the program stops here.

## Phased plan (each phase gated; do not skip Phase 0)

- **Phase 0 — verifier validation (bench only, no training).**
  `bench/rwr/verifier_probe.py`: given the artist set, build the centroid; score
  (a) held-out same-artist images, (b) other-artist / generic negatives, (c) the
  training images themselves. **Gate:** held-out same-artist must rank clearly above
  negatives (AUROC well over chance) *and* the score must not be dominated by
  near-duplicate distance to training images (else it is a memorization detector, not
  a style detector). Report against eyeball on a labelled grid — do not trust the
  scalar alone. Gate FIRES → Phase 1.
- **Phase 1 — RWR loop, turbo grow, single outer round.** `scripts/rwr/`: grow with
  `turbo + base` (no artist LoRA yet) under dropout prompts → score → top-k → CFM
  inner loop (existing loss, `build_anima` harness). **Gate:** 28-step SDE samples
  beat a matched MLE-on-images LoRA on the Phase-0 verifier *and* on a blind grid for
  style adherence without visible memorization of training compositions.
- **Phase 2 — multi-round ReST + collapse guards.** Regenerate with
  `turbo + current-artist-snapshot`; enable the full-step candidate mix, the
  accumulate-don't-replace real-image anchor, and the novelty term. **Gate:**
  diversity (pose/composition on grids; the metric is blind here — eyeball) must not
  degrade across rounds; watch for autophagy narrowing. Tune the **one real knob:
  regeneration frequency** (inner FM steps per outer grow) — more inner steps =
  cheaper but more off-policy / staler SNIS weights.
- **Phase 3 (optional) — heavier estimators only if RWR plateaus.** GRPO-style
  group-relative preference (verifier votes among the dropout-prompt candidates), or
  the adversarial lever reusing `turbo_dmd.py::PooledTokenDiscriminator` repointed to
  the artist set. Both add instability; enter only on a measured RWR ceiling.

## Cost accounting (the efficiency question, answered)

Not "N×K rollouts of gradient vs cheap FM." It is **cheap FM gradient, on better
targets, plus a periodic no-grad generation phase**:

| phase | cost | grad? |
|---|---|---|
| grow | `N` × K forwards; K=4 with turbo (<1 s @ 1024²) | no |
| score | `N` verifier encoder-forwards (final image only) | no |
| improve | 1 fwd+bwd per selected sample, random `t` (= normal training) | **yes** |

Amortize grow over the whole inner loop (regenerate every few hundred steps) and the
per-gradient-step overhead → near zero. With N=4–8, K=4, the marginal cost over a
normal LoRA run is a small fraction. The expensive-looking part never sees autograd.

## Open questions / decision gates

- **λ (reward temperature) and top-k vs soft-weight.** Hard top-k (RAFT) is simplest;
  soft `exp(reward/λ)` weighting avoids the selection cliff. A/B at Phase 1.
- **Seed-from-image (SDEdit) candidates.** Partial-noise img2img from a training image
  yields style-anchored variations → higher hit-rate, smaller `N`, still not copies.
  Cheap win to test in Phase 1.
- **Latent vs pixel scoring.** PE-Core wants pixels (VAE-decode candidates for
  scoring; keep the latent as the FM target — mind the 5D↔4D boundary, CLAUDE.md).
- **Does the artist manifold generalize from `N` images at all?** The verifier leans
  entirely on pretrained priors for this; Phase 0 measures it. If it cannot, no
  estimator downstream can rescue it.

## References

- `docs/methods/turbo.md` — the 4-step DP-DMD generator this reuses; its
  diversity-collapse tendency (the distribution risk); linear LoRA composition.
- `docs/proposal/turbo_caption_ranking.md` — relative-FM-ranking reward discipline
  ("measure at the distribution the loss sees"; validated compass).
- `docs/proposal/dave_mod_bestofn.md` — best-of-N order-statistics framing for the
  candidate pool (µ/σ/ρ readout under correlated draws).
- `library/runtime/harness.py::build_anima` — the load→apply→compile harness the grow
  and improve phases both build on.
- RWR/ReST/RAFT and DDPO lineage (external): reward-weighted regression and
  self-training-with-verifier for generative policies; DDPO/AlignProp as the
  backprop-through-sampling contrast this design deliberately avoids.
