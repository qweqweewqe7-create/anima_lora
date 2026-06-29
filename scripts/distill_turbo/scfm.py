"""SCFM — velocity-space self-distillation loop (turbo ``base_loss="scfm"``).

Implements the Phase-1 minimal port of *Shortcutting Pre-trained Flow Matching
Diffusion Models is Almost Free Lunch* (Cai et al., NeurIPS 2025) as a selectable
turbo objective. Two terms, mixed per-sample by ``k_ratio`` (Bernoulli per step
at the turbo default ``batch_size=1``):

- **Term A — teacher rectification.** A renoised real latent's coarse-step
  velocity is pinned to the CFG-guided teacher field (the quality ceiling).
- **Term B — velocity self-consistency.** One coarse Euler step on a stop-grad
  EMA copy of the student (θ⁻) must equal two finer sub-steps → the trajectory
  straightens, which is what makes few-step Euler work.

The fake/critic stack is repurposed as θ⁻ (``set_view("fake")``); it carries no
optimizer and is updated by parameter EMA (Eq. 14) with a cyclic restart. No
critic, no GAN, no diversity anchor — the saved artifact is a plain velocity
LoRA, identical inference to the DP-DMD student.

Design, decision gates, and the Phase-0/Phase-1 progress log all live in
``docs/proposal/turbo_scfm.md`` (§9). Driven from
``scripts/distill_turbo/distill.py::main`` under ``base_loss="scfm"``; reuses that
file's loaded DiT, compile, dataloader, ``_forward`` closure, and σ grids.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from tqdm import tqdm

from library.anima.uncond import uncond_for_batch
from library.inference.sampling import get_timesteps_sigmas

from .diversity import run_diversity_validation
from .primitives import make_scheduler, renoise, sample_t

logger = logging.getLogger(__name__)


def _step_tag(step: int) -> str:
    """Human checkpoint suffix: 1000 -> ``1k``, else raw count (mirror distill.py)."""
    return f"{step // 1000}k" if step % 1000 == 0 else str(step)


def _velocity_mse(
    v_pred: torch.Tensor, v_target: torch.Tensor, mask: torch.Tensor | None
) -> torch.Tensor:
    """Mean-squared velocity error, optionally foreground-masked.

    ``mask`` is ``[B, 1, H, W]`` (broadcast over the ``[B, C, H, W]`` channels);
    the masked variant is a per-element mean over the kept region so its scale
    matches the unmasked ``.mean()`` (``use_masked_loss`` parity with the DMD
    path). Both operands are upcast to fp32 — the loss is assembled in fp32 like
    every other turbo objective.
    """
    sq = (v_pred.float() - v_target.float()) ** 2
    if mask is None:
        return sq.mean()
    # denom = kept spatial cells × B × channels (mask has 1 channel).
    denom = mask.sum() * sq.shape[1] + 1e-8
    return (sq * mask).sum() / denom


def run_scfm(
    *,
    cfg,
    turbo,
    model,
    forward_fn,
    teacher_cfg_velocity_fn,
    dataloader,
    student_sigmas: list[float],
    uncond_base: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    writer,
    val_cond,
    val_latent_shape,
    val_clean,
) -> None:
    """Run the SCFM distillation loop to completion (saves like the DMD path).

    All heavy setup (DiT load, block-swap, compile, dataloader, the ``_forward``
    closure, σ grids, TB writer, config snapshot) is done by the caller and
    passed in — this owns only the SCFM-specific optimizer, EMA bookkeeping,
    per-step loss, metrics, validation, and checkpoint saves.
    """
    # Student optimizer (the EMA stack has none). Mirrors the DMD student opt.
    student_opt = torch.optim.AdamW(
        turbo.student_params(),
        lr=cfg.student_lr,
        weight_decay=cfg.weight_decay,
        fused=torch.cuda.is_available(),
    )
    student_sched = make_scheduler(student_opt, cfg.iterations, cfg.student_lr)

    # EMA student θ⁻ := the fake stack. Detach it from autograd (no optimizer)
    # and seed θ⁻ ← θ exactly. Both stacks are zero-init (LoRA up = 0) so this is
    # 0 ← 0 at start, but the explicit reset keeps the invariant honest.
    turbo.freeze_ema()
    turbo.reset_ema()

    n_student = sum(p.numel() for p in turbo.student_params())
    logger.info(
        f"SCFM: student={n_student:,} trainable params; EMA stack frozen "
        f"(μ={cfg.scfm_ema_mu}, restart every {cfg.scfm_ema_restart or 'never'})"
    )

    # Finer consistency grid Term B samples adjacent (t_i, t_i+1, t_i+2) triples
    # from. n+1 points, σ: 1 → 0; valid triple-start indices are 0 … n-2.
    grid = get_timesteps_sigmas(cfg.scfm_n_consistency_grid, cfg.flow_shift, "cpu")[
        1
    ]  # [n+1]
    grid_dev = grid.to(device=device, dtype=dtype)
    n_grid = cfg.scfm_n_consistency_grid
    logger.info(
        f"SCFM consistency grid ({n_grid} sub-steps, flow_shift={cfg.flow_shift}): "
        f"σ={['%.3f' % s for s in grid.tolist()]}; student grid "
        f"σ={['%.3f' % s for s in student_sigmas]}"
    )

    # CPU RNG for the per-step role draw + grid-index draw — keeps both off the
    # GPU sync path (mirrors the DMD loop's grad_step='random' g draw), seeded
    # for reproducibility alongside the global torch.manual_seed(cfg.seed).
    cpu_gen = torch.Generator()
    # Offset from the global seed so the role/grid draws don't alias the data
    # shuffle or noise streams that also key off cfg.seed.
    cpu_gen.manual_seed(cfg.seed + 9973)

    # Logging accumulators (GPU tensors flushed every log_interval → no per-step
    # .item() sync). Counts are CPU ints (role drawn on CPU), also sync-free.
    loss_sum = torch.zeros((), device=device)
    lossA_sum = torch.zeros((), device=device)
    lossB_sum = torch.zeros((), device=device)
    residB_sum = torch.zeros((), device=device)  # Term-B straightening headroom
    nA = nB = nlog = 0

    data_iter = iter(dataloader)
    use_masked = cfg.use_masked_loss
    progress = tqdm(range(cfg.iterations), desc="turbo-scfm")

    for step in progress:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        latents = batch["latents"].to(device, dtype=dtype, non_blocking=True)
        crossattn_emb = batch["crossattn_emb"].to(
            device, dtype=dtype, non_blocking=True
        )
        B = latents.shape[0]
        if use_masked:
            mask = batch["mask"].to(device, dtype=torch.float32, non_blocking=True)
        else:
            mask = None

        c_null = uncond_for_batch(uncond_base, crossattn_emb)

        torch.compiler.cudagraph_mark_step_begin()

        # --- per-sample role: Bernoulli(k_ratio). At B=1 this is a per-STEP coin
        # flip (deterministic threshold int(round(k_ratio*B)) would be 0 at B=1
        # → never Term A; Bernoulli generalizes the "k/N of a batch" ratio to any
        # B, expected Term-A fraction = k_ratio). Drawn on CPU (no GPU sync).
        is_termA_cpu = torch.rand(B, generator=cpu_gen) < cfg.scfm_k_ratio  # [B] bool
        is_termA = is_termA_cpu.to(device)
        any_A = bool(is_termA_cpu.any())
        any_B = bool((~is_termA_cpu).any())
        is_termA_b = is_termA.view(B, *([1] * (latents.dim() - 1)))  # broadcast

        # --- per-sample evaluation level t_samp ---------------------------------
        # Term A: a random renoise level t_a (renoised real latent → teacher
        #         rectification target).
        # Term B: the grid level t_i of a sampled adjacent triple.
        t_a = sample_t(
            B,
            distribution=cfg.t_distribution,
            sigmoid_scale=cfg.sigmoid_scale,
            device=device,
            dtype=dtype,
        )
        # Triple-start indices on CPU; Term-A samples get a dummy index 0 (their
        # Term-B target is computed-then-discarded only when any_B forces the EMA
        # forward — at B=1 the inactive branch is skipped entirely).
        idx = torch.randint(0, max(1, n_grid - 1), (B,), generator=cpu_gen)
        t_i = grid_dev[idx].to(device)
        t_ip1 = grid_dev[idx + 1].to(device)
        t_ip2 = grid_dev[idx + 2].to(device)
        d_i = (t_i - t_ip1).view(B, *([1] * (latents.dim() - 1)))
        d_ip1 = (t_ip1 - t_ip2).view(B, *([1] * (latents.dim() - 1)))

        t_samp = torch.where(is_termA, t_a, t_i)  # [B]

        # Single renoise per sample at t_samp (shared by teacher / EMA / student).
        eps = torch.randn_like(latents)
        x_t = renoise(latents, t_samp, eps)

        # --- Term A target: CFG-guided teacher velocity (no grad) ---------------
        v_tea = None
        if any_A:
            v_tea = teacher_cfg_velocity_fn(x_t, t_samp, crossattn_emb, c_null).to(
                dtype
            )

        # --- Term B target: two EMA sub-steps on the stop-grad student θ⁻ -------
        v_target_B = None
        resid_B = None
        if any_B:
            v1 = forward_fn("fake", x_t, t_samp, crossattn_emb, no_grad=True).squeeze(2)
            x_next = x_t - d_i * v1  # Euler sub-step i → i+1
            v2 = forward_fn("fake", x_next, t_ip1, crossattn_emb, no_grad=True).squeeze(
                2
            )
            denom = (d_i + d_ip1).clamp_min(1e-8)
            v_target_B = ((d_i * v1 + d_ip1 * v2) / denom).detach()
            # Straightening headroom: how far the coarse one-step velocity v1 is
            # from the two-sub-step composition (relative). Near 0 ⇒ field already
            # straight at this scale (Term B near-saturated); large ⇒ room to fix.
            resid_B = (
                (v1 - v_target_B).float().norm() / (v_target_B.float().norm() + 1e-8)
            ).detach()

        # Combine per-sample targets. Where only one branch ran, fill the other
        # side with that same tensor so torch.where has a valid operand (the
        # filled side is masked out by is_termA_b anyway).
        if v_tea is None:
            v_tea = v_target_B
        if v_target_B is None:
            v_target_B = v_tea
        target = torch.where(is_termA_b, v_tea, v_target_B).detach()

        # --- student grad forward at the SAME (x_t, t_samp) ---------------------
        x_t.requires_grad_()  # grad-ckpt safety; harmless when ckpt off
        turbo.set_student_step(0)  # no-op (single-head student)
        v_stu = forward_fn(
            "student", x_t, t_samp, crossattn_emb, no_grad=False
        ).squeeze(2)
        loss = _velocity_mse(v_stu, target, mask)

        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                turbo.student_params(), max_norm=cfg.grad_clip
            )
        student_opt.step()
        student_opt.zero_grad(set_to_none=True)
        student_sched.step()

        # --- EMA update θ⁻ ← μθ⁻ + (1−μ)θ (Eq. 14) + cyclic restart -------------
        turbo.update_ema(cfg.scfm_ema_mu)
        if cfg.scfm_ema_restart and (step + 1) % cfg.scfm_ema_restart == 0:
            turbo.reset_ema()

        # --- metrics (GPU-side accumulation; flush at log_interval) -------------
        loss_sum = loss_sum + loss.detach()
        nlog += 1
        nA_step = int(is_termA_cpu.sum())
        nB_step = B - nA_step
        if nA_step:
            # Term-A-only contribution to the loss is the masked MSE over those
            # samples; at B=1 the whole step is one role, so loss IS that role's.
            if nB_step == 0:
                lossA_sum = lossA_sum + loss.detach()
                nA += 1
        if nB_step and nA_step == 0:
            # Pure Term-B step (the only case at the default B=1): loss IS the
            # Term-B loss, and resid_B stays in lockstep with the nB counter.
            lossB_sum = lossB_sum + loss.detach()
            if resid_B is not None:
                residB_sum = residB_sum + resid_B
            nB += 1

        if (step + 1) % cfg.log_interval == 0:
            scalars = {
                "train/scfm_loss": (loss_sum / max(1, nlog)).item(),
                "train/scfm_termA_frac": nA / max(1, nlog),
                "train/student_lr": student_sched.get_last_lr()[0],
            }
            if nA:
                scalars["train/scfm_loss_a"] = (lossA_sum / nA).item()
            if nB:
                scalars["train/scfm_loss_b"] = (lossB_sum / nB).item()
                scalars["train/scfm_consistency_residual"] = (residB_sum / nB).item()
            if writer is not None:
                for k, v in scalars.items():
                    writer.add_scalar(k, v, step + 1)
            progress.set_postfix(
                loss=f"{scalars['train/scfm_loss']:.4f}",
                a=f"{scalars.get('train/scfm_loss_a', float('nan')):.4f}",
                b=f"{scalars.get('train/scfm_loss_b', float('nan')):.4f}",
            )
            loss_sum = torch.zeros((), device=device)
            lossA_sum = torch.zeros((), device=device)
            lossB_sum = torch.zeros((), device=device)
            residB_sum = torch.zeros((), device=device)
            nA = nB = nlog = 0

        # --- diversity validation (DAVE same-prompt probe; structure-sensitive
        # checkpoint signal — CMMD is blind to pose, §5.5) ----------------------
        if val_cond is not None and (step + 1) % cfg.validate_every_n_steps == 0:
            dm = run_diversity_validation(
                model=model,
                forward_fn=forward_fn,
                set_student_step=turbo.set_student_step,
                student_sigmas=student_sigmas,
                crossattn_emb=val_cond,
                latent_shape=val_latent_shape,
                num_seeds=cfg.val_diversity_seeds,
                seed0=cfg.seed,
                device=device,
                dtype=dtype,
                clean_latent=val_clean,
            )
            if writer is not None:
                writer.add_scalar("val/div_ac_sim", dm.ac_sim, step + 1)
                writer.add_scalar("val/div_dc_sim", dm.dc_sim, step + 1)
                writer.add_scalar("val/div_gap", dm.gap, step + 1)
                writer.add_scalar("val/div_xpred_ac_sim", dm.xpred_ac_sim, step + 1)
                writer.add_scalar("val/fm_mse", dm.fm_mse, step + 1)
            logger.info(
                f"[val@{step + 1}] diversity: AC sim={dm.ac_sim:.4f} "
                f"(lower=more diverse) | DC sim={dm.dc_sim:.4f} | gap={dm.gap:+.4f} "
                f"| x_pred AC sim={dm.xpred_ac_sim:.4f} | FM MSE={dm.fm_mse:.4f}"
            )

        # --- checkpoint (mirror the DMD save: step-tagged subdir + final bare) --
        if (step + 1) % cfg.save_every == 0 or (step + 1) == cfg.iterations:
            n = step + 1
            is_final = n == cfg.iterations
            metadata = {
                "ss_turbo_objective": cfg.base_loss,  # "scfm"
                "ss_turbo_student_rank": str(cfg.student_rank),
                "ss_turbo_student_alpha": str(cfg.student_alpha),
                "ss_turbo_student_steps": str(cfg.student_steps),
                "ss_turbo_teacher_cfg": str(cfg.teacher_cfg),
                "ss_turbo_step": str(n),
                "ss_scfm_k_ratio": str(cfg.scfm_k_ratio),
                "ss_scfm_ema_mu": str(cfg.scfm_ema_mu),
                "ss_scfm_n_consistency_grid": str(cfg.scfm_n_consistency_grid),
            }
            ckpt_subdir = Path(cfg.output_dir) / cfg.output_name
            ckpt_subdir.mkdir(parents=True, exist_ok=True)
            save_paths = [
                str(ckpt_subdir / f"{cfg.output_name}_{_step_tag(n)}.safetensors")
            ]
            if is_final:
                save_paths.append(
                    str(Path(cfg.output_dir) / f"{cfg.output_name}.safetensors")
                )
            for save_path in save_paths:
                turbo.save_student(save_path, dtype=torch.bfloat16, metadata=metadata)
                logger.info(f"saved checkpoint: {save_path}")

    if writer is not None:
        writer.close()
    logger.info("SCFM distillation complete.")
