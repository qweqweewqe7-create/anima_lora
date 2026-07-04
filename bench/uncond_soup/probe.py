#!/usr/bin/env python3
"""Does soup-with-uncond really help? (Self-Soupervision Phase 0)

Design (see README.md): a 3-artist bounded pool provides the *unlabeled* data
for an unconditional inter-train (caption_dropout_rate=1.0); supervision is a
single artist. Two ingredient families share the same fine-tuning seeds:

    base_s<k>  = plain fine-tune on the target artist          (seed k)
    self_s<k>  = same fine-tune, initialized from the uncond   (seed k)
                 pool checkpoint via --network_weights

This probe CMMD-scores, against the target artist's held-out split:

    every ingredient, each family's uniform DW soup, each family's
    lambda=0.5 pair midpoints (the LMC gate), and optionally the raw
    uncond pool checkpoint (sanity: should be clearly worse).

Gates (CMMD: lower is better; sigma_within = sampling-noise floor, the
FID-lottery correction — deltas below it are inconclusive):

    LMC        mid(a,b) <= mean(cmmd_a, cmmd_b) + sigma_within, all pairs,
               both families. If this fails, souping is off the table and
               every other number here is moot.
    SOUP       family soup <= best ingredient + sigma_within
               (soup at least doesn't lose to seed-fishing).
    UNCOND     headline: base_soup_mean - self_soup_mean > gate_mult *
               sigma_within  ->  UNCOND_HELPS. Symmetrically UNCOND_HURTS;
               otherwise INCONCLUSIVE.

Reuses the seed-lottery probe's trainer-faithful split / CMMD / compile
machinery wholesale — this bench adds only the soup arms and the reduction.

Usage
-----
    python bench/uncond_soup/probe.py \
        --method lora --preset default --target_artist artistA \
        --self_ckpts output/ckpt/uncondsoup_self_s1001.safetensors ... \
        --base_ckpts output/ckpt/uncondsoup_base_s1001.safetensors ... \
        --pool_ckpt  output/ckpt/uncondsoup_pool.safetensors \
        --validation_split_num 16 --validation_seed 42 --k_val 3

``--validation_split_num`` / ``--validation_seed`` MUST match the training
runs (launch.py defaults) so the held-out set is the one no run ever saw.
"""

from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path
from statistics import median

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from anima_lora import load_vae  # noqa: E402
from bench._common import make_run_dir, write_result  # noqa: E402
from bench.seed_lottery.probe import (  # noqa: E402
    _token_budget,
    build_training_args,
    build_val_items,
    cmmd_for_checkpoint,
    save_contact_sheet,
)
from bench.uncond_soup.soup import build_soup_file  # noqa: E402
from library.datasets.artist_shard import _pattern_for  # noqa: E402
from library.runtime.accelerator import prepare_accelerator  # noqa: E402
from library.runtime.device import str_to_dtype  # noqa: E402
from library.training.cmmd import load_reference_features  # noqa: E402
from library.vision.encoder import load_pe_encoder  # noqa: E402


def build_arms(args_cli, run_dir: Path) -> list[tuple[str, str]]:
    """[(arm_name, ckpt_path)] — ingredients + soups + LMC midpoints (+ pool).

    Soup/midpoint files are built into ``run_dir`` so the run is
    self-contained and re-scoreable.
    """
    arms: list[tuple[str, str]] = []
    for family, ckpts in (
        ("base", args_cli.base_ckpts),
        ("self", args_cli.self_ckpts),
    ):
        for c in ckpts:
            arms.append((f"{family}:{Path(c).stem}", c))
        soup_path = build_soup_file(ckpts, str(run_dir / f"{family}_soup.safetensors"))
        arms.append((f"{family}:soup", str(soup_path)))
        for i, j in combinations(range(len(ckpts)), 2):
            mid = build_soup_file(
                [ckpts[i], ckpts[j]],
                str(run_dir / f"{family}_mid{i}{j}.safetensors"),
                weights=[0.5, 0.5],
            )
            arms.append((f"{family}:mid{i}{j}", str(mid)))
    if args_cli.pool_ckpt:
        arms.append(("pool", args_cli.pool_ckpt))
    return arms


def reduce_family(
    family: str,
    arm_stats: dict[str, dict],
    n_ingredients: int,
    sigma_within: float,
) -> dict:
    ing = {
        k: v
        for k, v in arm_stats.items()
        if k.startswith(f"{family}:") and ":mid" not in k and k != f"{family}:soup"
    }
    soup_mean = arm_stats[f"{family}:soup"]["mean"]
    best_name, best = min(ing.items(), key=lambda kv: kv[1]["mean"])
    mids = {k: v for k, v in arm_stats.items() if k.startswith(f"{family}:mid")}

    # Endpoint order = --ckpts order = arm insertion order (dicts preserve it).
    endpoints = list(ing)

    # LMC per pair: midpoint no worse than the endpoint average, within noise.
    lmc_pairs = []
    for mid_name, mid in mids.items():
        i, j = int(mid_name[-2]), int(mid_name[-1])  # N <= 9 by construction
        e_i, e_j = endpoints[i], endpoints[j]
        endpoint_avg = 0.5 * (arm_stats[e_i]["mean"] + arm_stats[e_j]["mean"])
        holds = mid["mean"] <= endpoint_avg + sigma_within
        lmc_pairs.append(
            {
                "pair": [e_i, e_j],
                "mid": mid_name,
                "mid_mean": mid["mean"],
                "endpoint_avg": endpoint_avg,
                "holds": bool(holds),
            }
        )
    return {
        "n_ingredients": n_ingredients,
        "ingredient_means": {k: v["mean"] for k, v in ing.items()},
        "best_ingredient": best_name,
        "best_mean": best["mean"],
        "soup_mean": soup_mean,
        "soup_minus_best": soup_mean - best["mean"],
        "soup_beats_or_ties_best": bool(soup_mean <= best["mean"] + sigma_within),
        "lmc_pairs": lmc_pairs,
        "lmc_holds": bool(all(p["holds"] for p in lmc_pairs)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--method", default="lora")
    ap.add_argument("--preset", default="default")
    ap.add_argument(
        "--target_artist",
        required=True,
        help="Artist subdirectory name the fine-tunes trained on — the probe "
        "scores against this artist's held-out split.",
    )
    ap.add_argument("--self_ckpts", nargs="+", required=True)
    ap.add_argument("--base_ckpts", nargs="+", required=True)
    ap.add_argument(
        "--pool_ckpt",
        default=None,
        help="The uncond inter-trained checkpoint (sanity arm; optional).",
    )
    ap.add_argument("--k_val", type=int, default=3)
    ap.add_argument("--val_seed_base", type=int, default=1)
    ap.add_argument("--validation_split_num", type=int, default=16)
    ap.add_argument("--validation_seed", type=int, default=42)
    ap.add_argument("--max_val_items", type=int, default=None)
    ap.add_argument("--sample_steps", type=int, default=20)
    ap.add_argument("--cfg", type=float, default=1.0)
    ap.add_argument("--dtype", type=str, default="bf16")
    ap.add_argument(
        "--gate_mult",
        type=float,
        default=1.5,
        help="UNCOND verdict fires when |base_soup - self_soup| exceeds this "
        "multiple of sigma_within.",
    )
    ap.add_argument("--compile", dest="compile", action="store_true", default=True)
    ap.add_argument("--no-compile", dest="compile", action="store_false")
    ap.add_argument("--label", default=None)
    args_cli = ap.parse_args()

    if len(args_cli.self_ckpts) != len(args_cli.base_ckpts):
        ap.error("--self_ckpts and --base_ckpts must be paired (same length/seeds).")
    if len(args_cli.self_ckpts) < 2:
        ap.error("Need >= 2 ingredients per family (soup + LMC need pairs).")
    if len(args_cli.self_ckpts) > 9:
        ap.error("Max 9 ingredients per family (single-digit midpoint naming).")

    dtype = str_to_dtype(args_cli.dtype)
    run_dir = make_run_dir("uncond_soup", label=args_cli.label)

    # Arms first (soup construction is CPU-only and fails fast on bad inputs).
    arms = build_arms(args_cli, run_dir)
    print(f"Arms ({len(arms)}): {[n for n, _ in arms]}", flush=True)

    # Trainer-faithful args + the target artist's held-out split.
    targs = build_training_args(
        args_cli.method,
        args_cli.preset,
        {
            "validation_split_num": args_cli.validation_split_num,
            "validation_seed": args_cli.validation_seed,
            "validation_sample_steps": args_cli.sample_steps,
            "validation_cfg_scale": args_cli.cfg,
            "path_pattern": _pattern_for([args_cli.target_artist]),
        },
    )
    acc = prepare_accelerator(targs)
    items = build_val_items(targs)
    if args_cli.max_val_items is not None:
        items = items[: args_cli.max_val_items]
    if not items:
        raise RuntimeError(
            "No held-out items with cached TE + PE sidecars for "
            f"{args_cli.target_artist!r} — check --validation_split_num/"
            "--validation_seed match the training runs and `make preprocess-pe` ran."
        )
    print(f"Held-out items ({args_cli.target_artist}): {len(items)}", flush=True)

    ref_pool = load_reference_features([pe for (_i, _t, pe) in items]).to(acc.device)
    pe_bundle = load_pe_encoder(acc.device)
    pe_bundle.encoder.inner.to("cpu")
    vae = load_vae(targs.vae, dtype=dtype, eval=True, vae_2d=True, device="cpu")
    flow_shift = float(getattr(targs, "discrete_flow_shift", 3.0))
    vseeds = list(
        range(args_cli.val_seed_base, args_cli.val_seed_base + args_cli.k_val)
    )
    compile_budget = _token_budget(items) if args_cli.compile else None

    arm_stats: dict[str, dict] = {}
    contact_rows: list[list[torch.Tensor]] = []
    for name, ckpt in arms:
        row, contact = cmmd_for_checkpoint(
            ckpt=ckpt,
            args=targs,
            acc=acc,
            vae=vae,
            pe_bundle=pe_bundle,
            items=items,
            ref_pool=ref_pool,
            vseeds=vseeds,
            sample_steps=args_cli.sample_steps,
            cfg_scale=args_cli.cfg,
            flow_shift=flow_shift,
            compile_budget=compile_budget,
        )
        t = torch.tensor(row)
        arm_stats[name] = {
            "cmmd": row,
            "mean": float(t.mean()),
            "std": float(t.std(unbiased=True)) if len(row) > 1 else 0.0,
        }
        contact_rows.append(contact)

    sigma_within = median(v["std"] for v in arm_stats.values())

    fam = {
        family: reduce_family(family, arm_stats, len(args_cli.base_ckpts), sigma_within)
        for family in ("base", "self")
    }
    lmc_ok = fam["base"]["lmc_holds"] and fam["self"]["lmc_holds"]
    uncond_gain = fam["base"]["soup_mean"] - fam["self"]["soup_mean"]
    floor = args_cli.gate_mult * sigma_within
    if not lmc_ok:
        verdict = "LMC_FAILS (souping premise broken — fix init sharing first)"
    elif uncond_gain > floor:
        verdict = "UNCOND_HELPS"
    elif uncond_gain < -floor:
        verdict = "UNCOND_HURTS"
    else:
        verdict = "INCONCLUSIVE (below noise floor)"

    print("\n=== uncond-soup Phase 0 ===")
    for name in arm_stats:
        s = arm_stats[name]
        print(f"  {name:24s} CMMD {s['mean']:.3f} +- {s['std']:.3f}")
    print(f"sigma_within  = {sigma_within:.4f}")
    for family in ("base", "self"):
        f = fam[family]
        print(
            f"{family}: soup {f['soup_mean']:.3f} vs best {f['best_mean']:.3f} "
            f"({f['best_ingredient']})  LMC={'OK' if f['lmc_holds'] else 'FAIL'}"
        )
    print(f"uncond_gain   = {uncond_gain:+.4f}  (gate at +-{floor:.4f})")
    print(f"VERDICT       = {verdict}")

    has_sheet = save_contact_sheet(contact_rows, run_dir / "contact_sheet.png")
    artifacts = [f"{f}_soup.safetensors" for f in ("base", "self")]
    if has_sheet:
        artifacts.append("contact_sheet.png")
    write_result(
        run_dir,
        script=__file__,
        args=args_cli,
        metrics={
            "arms": arm_stats,
            "sigma_within": sigma_within,
            "families": fam,
            "uncond_gain": uncond_gain,
            "gate_mult": args_cli.gate_mult,
            "lmc_ok": lmc_ok,
            "verdict": verdict,
            "vseeds": vseeds,
            "n_val_items": len(items),
            "target_artist": args_cli.target_artist,
        },
        label=args_cli.label,
        artifacts=artifacts,
        device=acc.device,
    )
    print(f"\nWrote {run_dir / 'result.json'}")


if __name__ == "__main__":
    main()
