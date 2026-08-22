#!/usr/bin/env python3
"""Mod-guidance vs prompt-side quality tags: the quality/content trade-off.

Question (docs/findings/mod_guidance_quality_tag_axis.md left it open at the
n=12 eyeball level): can modulation guidance deliver the "subtle quality tag"
— the finishing lift users reach for with `masterpiece, absurdres, score_9` —
WITHOUT the content bias / seed-lock that putting those tags into the
cross-attn prompt causes? The paper's Fig. 3(a) frame, adapted to Anima:
sweep mod `w` and place the prompt-tag arm as a single point on a
(content preservation, quality) plane; the claim holds if the mod curve
Pareto-dominates the prompt-tag point.

Arms (same prompts x same seeds everywhere; paired within this run only):

  base     content prompt, pooled-proj gate OFF        <- pure production path
  qtag     prompt + quality tags after the rating band, gate OFF
  proj0    content prompt, gate ON, w=0                <- isolates base-proj inject
  mod_w{W} content prompt, gate ON, shipped step_i8_skip27 steering at w=W
  both_w3  qtag prompt + w=3                           <- double-drive reference

Automatic axes (quality itself is judged on the review page — index.html —
per the repo's grid-read convention; MUSIQ/CLIP-family scorers are recorded
as untrustworthy here):

  content_score  mean Anima Tagger prob over the prompt's general-band tags
  content_recall fraction of those tags the tagger keeps (per-tag F1 thresholds)
  pe_cos_base    PE-Core embedding cosine to the base arm, same (prompt, seed)
  diversity      mean pairwise PE cosine *distance* across seeds within an arm
                 (drops when a tag "locks" composition across seeds)

Run (GPU work -> submit through the daemon, not a bare background shell):

  uv run python bench/mod_quality_tradeoff/run_bench.py --smoke --label smoke
  uv run python bench/mod_quality_tradeoff/run_bench.py --label full

Metrics-only re-run on an existing image set (no GPU generation):

  uv run python bench/mod_quality_tradeoff/run_bench.py \
      --metrics_only --run_dir bench/mod_quality_tradeoff/results/<dir>
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

import torch  # noqa: E402

from bench._common import REPO_ROOT, make_run_dir, write_result  # noqa: E402

# --------------------------------------------------------------------------- #
# Quality-tag handling (mirrors _archive/bench/mod_guidance/channel_attribution:
# quality/meta tags are not a caption slot — training captions place them right
# after the leading rating literal, so that's where the qtag arm splices them).
# --------------------------------------------------------------------------- #
_QUALITY_RE = re.compile(
    r"^(score_\d(_up)?|masterpiece|(best|high|normal|low|worst) quality"
    r"|absurdres|highres|lowres|newest|oldest|recent|old|year \d{4})$",
    re.IGNORECASE,
)


def _is_quality_tag(tag: str) -> bool:
    return bool(_QUALITY_RE.match(tag.strip()))


# Captions that *ask for* defects make a quality bench unreadable (the steering
# and the prompt pull opposite ways) — reject them at sampling time.
_DEFECT_TAGS = frozenset(
    {
        "bad anatomy",
        "bad hands",
        "bad feet",
        "bad proportions",
        "jpeg artifacts",
        "blurry",
    }
)


def _splice_quality(flat_tags: Tuple[str, ...], quality_tags: List[str]) -> List[str]:
    """Insert quality tags right after a leading rating literal (else prepend)."""
    from library.captioning.taxonomy import CAPTION_RATINGS

    toks = list(flat_tags)
    if toks and toks[0].lower() in CAPTION_RATINGS:
        return [toks[0], *quality_tags, *toks[1:]]
    return [*quality_tags, *toks]


# --------------------------------------------------------------------------- #
# Prompt sampling from the derived-caption corpus
# --------------------------------------------------------------------------- #
@dataclass
class PromptSpec:
    stem: str  # provenance (resized/<artist>/<id>)
    prompt: str  # quality-stripped caption (clauses preserved)
    prompt_qtag: str  # same, with quality tags spliced after the rating band
    flat_tags: List[str]  # quality-stripped flat bag (recall candidates)


def sample_prompts(
    caption_root: Path, n: int, quality_tags: List[str], rng_seed: int
) -> List[PromptSpec]:
    from library.captioning.position_clauses import compose_caption, parse_caption

    files = sorted(
        p for p in caption_root.rglob("*.txt") if not p.name.endswith(".variants.txt")
    )
    if not files:
        raise SystemExit(f"no captions under {caption_root}")

    rng = random.Random(rng_seed)
    rng.shuffle(files)

    specs: List[PromptSpec] = []
    seen: set[str] = set()
    for path in files:
        caption = path.read_text(encoding="utf-8").strip()
        if not caption:
            continue
        parsed = parse_caption(caption)
        if parsed.tag_keys & _DEFECT_TAGS:
            continue
        flat = [t for t in parsed.flat_tags if not _is_quality_tag(t)]
        if not (12 <= len(flat) <= 45):
            continue
        prompt = compose_caption(tuple(flat), parsed.clauses)
        if prompt in seen:
            continue
        seen.add(prompt)
        qtag_flat = _splice_quality(tuple(flat), quality_tags)
        specs.append(
            PromptSpec(
                stem=str(path.relative_to(caption_root)).removesuffix(".txt"),
                prompt=prompt,
                prompt_qtag=compose_caption(tuple(qtag_flat), parsed.clauses),
                flat_tags=flat,
            )
        )
        if len(specs) >= n:
            break
    if len(specs) < n:
        raise SystemExit(f"only {len(specs)} usable captions (wanted {n})")
    return specs


# --------------------------------------------------------------------------- #
# Arms
# --------------------------------------------------------------------------- #
@dataclass
class Arm:
    name: str
    use_qtag_prompt: bool  # quality tags in the cross-attn prompt
    gate: bool  # model.enable_pooled_text_modulation
    mod_w: float  # 0.0 -> no steering delta


def build_arms(w_list: List[float], include_both: bool) -> List[Arm]:
    arms = [
        Arm("base", use_qtag_prompt=False, gate=False, mod_w=0.0),
        Arm("qtag", use_qtag_prompt=True, gate=False, mod_w=0.0),
        Arm("proj0", use_qtag_prompt=False, gate=True, mod_w=0.0),
    ]
    for w in w_list:
        arms.append(Arm(f"mod_w{w:g}", use_qtag_prompt=False, gate=True, mod_w=w))
    if include_both:
        arms.append(Arm("both_w3", use_qtag_prompt=True, gate=True, mod_w=3.0))
    return arms


def image_path(images_dir: Path, arm: Arm, pi: int, seed: int) -> Path:
    return images_dir / arm.name / f"p{pi:02d}_s{seed}.png"


# --------------------------------------------------------------------------- #
# Generation phase (one resident DiT/TE/VAE; shipped inference path throughout)
# --------------------------------------------------------------------------- #
def run_generation(
    args: argparse.Namespace,
    arms: List[Arm],
    prompts: List[PromptSpec],
    seeds: List[int],
    images_dir: Path,
) -> None:
    from anima_lora import (
        GenerationRequest,
        default_checkpoints,
        generate,
        get_generation_settings,
    )
    from library.inference.models import load_dit_model, load_shared_models
    from library.inference.output import decode_to_pil
    from library.models import qwen_vae
    from library.runtime.device import clean_memory_on_device

    ckpt = default_checkpoints()

    def build_args(prompt: str, seed: int, mod_w: float) -> argparse.Namespace:
        extra: List[str] = []
        if mod_w > 0:
            extra += [
                "--mod_pos_prompt",
                args.mod_pos_prompt,
                "--mod_neg_prompt",
                args.mod_neg_prompt,
                "--mod_start_layer",
                str(args.mod_start_layer),
                "--mod_end_layer",
                str(args.mod_end_layer),
            ]
        if args.compile_blocks:
            extra += ["--compile_blocks"]
        req = GenerationRequest(
            dit=ckpt.dit,
            vae=ckpt.vae,
            text_encoder=ckpt.text_encoder,
            prompt=prompt,
            negative_prompt=args.negative_prompt,
            image_size=(args.size, args.size),
            infer_steps=args.steps,
            guidance_scale=args.cfg,
            flow_shift=args.flow_shift,
            sampler="euler",
            seed=seed,
            attn_mode=args.attn_mode,
            vae_chunk_size=64,
            vae_disable_cache=True,
            # pooled_text_proj is passed unconditionally: weights load once at
            # model load (silently-inert-otherwise gotcha), the per-arm gate is
            # toggled on the module directly; w=0 resets the steering buffers.
            pooled_text_proj=args.pooled_text_proj,
            save_path=str(images_dir),  # placeholder; we save via decode_to_pil
            extra_argv=[*extra, "--mod_w", str(mod_w)],
        )
        return req.to_args()

    first_args = build_args(prompts[0].prompt, seeds[0], 0.0)
    gen_settings = get_generation_settings(first_args)
    device = gen_settings.device

    print("loading shared models (TE on CPU) ...", flush=True)
    shared = load_shared_models(first_args)
    shared["conds_cache"] = {}
    print("loading DiT (+ pooled_text_proj) ...", flush=True)
    shared["model"] = load_dit_model(first_args, device, torch.bfloat16)

    vae = qwen_vae.load_vae(
        first_args.vae,
        device="cpu",
        disable_mmap=True,
        spatial_chunk_size=first_args.vae_chunk_size,
        disable_cache=first_args.vae_disable_cache,
        vae_2d=first_args.vae_2d,
    )
    vae.to(torch.bfloat16)
    vae.eval()

    total = sum(
        1
        for arm in arms
        for pi in range(len(prompts))
        for s in seeds
        if not image_path(images_dir, arm, pi, s).exists()
    )
    done = 0
    t0 = time.time()
    for arm in arms:
        (images_dir / arm.name).mkdir(parents=True, exist_ok=True)
        for pi, spec in enumerate(prompts):
            prompt = spec.prompt_qtag if arm.use_qtag_prompt else spec.prompt
            for seed in seeds:
                out_path = image_path(images_dir, arm, pi, seed)
                if out_path.exists():  # resumable
                    continue
                render_args = build_args(prompt, seed, arm.mod_w)
                shared["model"].enable_pooled_text_modulation = arm.gate
                latent = generate(render_args, gen_settings, shared)
                img = decode_to_pil(vae, latent, device)
                img.save(out_path)
                done += 1
                dt = time.time() - t0
                eta = dt / done * (total - done)
                print(
                    f"[{done}/{total}] {arm.name} p{pi:02d} s{seed} "
                    f"({dt / done:.1f}s/img, eta {eta / 60:.0f}m)",
                    flush=True,
                )

    del shared, vae
    clean_memory_on_device(device)


# --------------------------------------------------------------------------- #
# Metrics phase (tagger = scorer AND PE-embedding source; DiT already freed)
# --------------------------------------------------------------------------- #
def run_metrics(
    args: argparse.Namespace,
    arms: List[Arm],
    prompts: List[PromptSpec],
    seeds: List[int],
    run_dir: Path,
    images_dir: Path,
) -> Dict:
    from PIL import Image

    from library.captioning.anima_tagger import (
        DEFAULT_TAGGER_DIR,
        AnimaTagger,
        ensure_tagger_checkpoint,
    )
    from library.env import resolve_under_home

    ckpt_dir = ensure_tagger_checkpoint(resolve_under_home(DEFAULT_TAGGER_DIR))
    print(f"loading Anima Tagger from {ckpt_dir} ...", flush=True)
    tagger = AnimaTagger(ckpt_dir)

    vocab_general = {e.name for e in tagger.tag_entries if e.category == "general"}
    vocab_all = {e.name for e in tagger.tag_entries}
    quality_in_vocab = [t for t in args.quality_tags_list if t in vocab_all]

    # Per-prompt recall targets: general-band tags the tagger can actually score.
    content_tags: List[List[str]] = [
        [t for t in spec.flat_tags if t in vocab_general] for spec in prompts
    ]

    def embed(feat: torch.Tensor) -> torch.Tensor:
        v = feat.mean(dim=0) if feat.dim() == 2 else feat
        v = v.to(torch.float32).cpu()
        return v / (v.norm() + 1e-8)

    rows: List[dict] = []
    emb: Dict[Tuple[str, int, int], torch.Tensor] = {}
    for arm in arms:
        for pi in range(len(prompts)):
            for seed in seeds:
                path = image_path(images_dir, arm, pi, seed)
                if not path.exists():
                    print(f"MISSING {path} — skipped", flush=True)
                    continue
                img = Image.open(path).convert("RGB")
                out = tagger.predict(img)
                scores: Dict[str, float] = out["scores"]  # type: ignore[assignment]
                kept: Dict[str, float] = out["kept"]  # type: ignore[assignment]
                tags = content_tags[pi]
                c_scores = [scores[t] for t in tags]
                row = {
                    "arm": arm.name,
                    "prompt_idx": pi,
                    "seed": seed,
                    "n_content_tags": len(tags),
                    "content_score": sum(c_scores) / max(len(c_scores), 1),
                    "content_recall": (
                        sum(1 for t in tags if t in kept) / max(len(tags), 1)
                    ),
                }
                for qt in quality_in_vocab:
                    row[f"tagscore_{qt.replace(' ', '_')}"] = scores[qt]
                rows.append(row)
                emb[(arm.name, pi, seed)] = embed(tagger._encode_image(img))

    # PE cosine to base (same prompt, seed) + intra-arm seed diversity.
    for row in rows:
        key = (row["arm"], row["prompt_idx"], row["seed"])
        base_key = ("base", row["prompt_idx"], row["seed"])
        row["pe_cos_base"] = (
            float(emb[key] @ emb[base_key]) if base_key in emb else float("nan")
        )

    def arm_diversity(arm_name: str) -> float:
        vals = []
        for pi in range(len(prompts)):
            es = [emb[(arm_name, pi, s)] for s in seeds if (arm_name, pi, s) in emb]
            if len(es) < 2:
                continue
            d = [
                1.0 - float(es[i] @ es[j])
                for i in range(len(es))
                for j in range(i + 1, len(es))
            ]
            vals.append(sum(d) / len(d))
        return sum(vals) / len(vals) if vals else float("nan")

    def arm_mean(arm_name: str, field: str) -> float:
        vals = [
            r[field]
            for r in rows
            if r["arm"] == arm_name and r[field] == r[field]  # skip NaN
        ]
        return sum(vals) / len(vals) if vals else float("nan")

    agg_fields = ["content_score", "content_recall", "pe_cos_base"] + [
        f"tagscore_{qt.replace(' ', '_')}" for qt in quality_in_vocab
    ]
    aggregates = {}
    for arm in arms:
        aggregates[arm.name] = {f: arm_mean(arm.name, f) for f in agg_fields}
        aggregates[arm.name]["seed_diversity"] = arm_diversity(arm.name)

    with open(run_dir / "per_image.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(run_dir / "aggregates.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["arm", *agg_fields, "seed_diversity"])
        writer.writeheader()
        for arm_name, agg in aggregates.items():
            writer.writerow({"arm": arm_name, **agg})

    return {
        "aggregates": aggregates,
        "quality_tags_in_tagger_vocab": quality_in_vocab,
        "n_images_scored": len(rows),
    }


# --------------------------------------------------------------------------- #
# Report: trade-off plot + SbS review page
# --------------------------------------------------------------------------- #
def write_plot(metrics: Dict, arms: List[Arm], run_dir: Path) -> Optional[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    agg = metrics["aggregates"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    mod_arms = [a.name for a in arms if a.name.startswith("mod_w")]
    xs = [agg[a]["pe_cos_base"] for a in mod_arms]
    ys = [agg[a]["content_score"] for a in mod_arms]
    ax.plot(xs, ys, "o-", label="mod guidance (w sweep)")
    for a in mod_arms:
        ax.annotate(
            a.removeprefix("mod_"), (agg[a]["pe_cos_base"], agg[a]["content_score"])
        )
    for name, marker in [("qtag", "s"), ("proj0", "^"), ("both_w3", "D")]:
        if name in agg:
            ax.plot(
                agg[name]["pe_cos_base"],
                agg[name]["content_score"],
                marker,
                label=name,
            )
    ax.set_xlabel("content preservation (PE cos to base) →")
    ax.set_ylabel("prompt-content tagger score →")
    ax.legend(fontsize=8)
    ax.set_title("content cost per arm (base = cos 1.0 by construction)")

    ax = axes[1]
    names = [a.name for a in arms]
    ax.bar(names, [agg[n]["seed_diversity"] for n in names])
    ax.set_ylabel("intra-arm seed diversity (PE cos distance)")
    ax.set_title("seed lock check (lower bar = more locked)")
    ax.tick_params(axis="x", rotation=45)

    fig.tight_layout()
    out = run_dir / "tradeoff.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out.name


def write_review_page(
    arms: List[Arm],
    prompts: List[PromptSpec],
    seeds: List[int],
    run_dir: Path,
    images_dir: Path,
) -> str:
    rel = images_dir.name
    parts = [
        "<html><head><meta charset='utf-8'><style>",
        "body{font-family:sans-serif;background:#111;color:#ddd}",
        "img{width:220px;display:block}",
        "td{padding:2px;vertical-align:top}th{position:sticky;top:0;background:#111}",
        ".p{color:#8ab;font-size:11px;max-width:1200px;word-wrap:break-word}",
        "</style></head><body>",
        "<h2>mod_quality_tradeoff — side-by-side (quality is judged HERE, "
        "not by the automatic metrics)</h2>",
    ]
    for pi, spec in enumerate(prompts):
        parts.append(f"<h3>p{pi:02d} — {spec.stem}</h3>")
        parts.append(f"<div class='p'>{spec.prompt}</div>")
        parts.append("<table><tr>")
        parts += [f"<th>{arm.name}</th>" for arm in arms]
        parts.append("</tr>")
        for seed in seeds:
            parts.append("<tr>")
            for arm in arms:
                p = image_path(images_dir, arm, pi, seed)
                if p.exists():
                    src = f"{rel}/{arm.name}/{p.name}"
                    parts.append(f"<td><a href='{src}'><img src='{src}'></a></td>")
                else:
                    parts.append("<td>missing</td>")
            parts.append("</tr>")
        parts.append("</table>")
    parts.append("</body></html>")
    (run_dir / "index.html").write_text("\n".join(parts), encoding="utf-8")
    return "index.html"


# --------------------------------------------------------------------------- #
def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--label", default=None)
    p.add_argument("--n_prompts", type=int, default=8)
    p.add_argument("--seeds", default="0,1,2,3")
    p.add_argument("--w_list", default="1,2,3,4,6")
    p.add_argument("--no_both_arm", action="store_true")
    p.add_argument("--steps", type=int, default=28)
    p.add_argument("--cfg", type=float, default=4.0)
    p.add_argument("--flow_shift", type=float, default=3.0)
    p.add_argument("--size", type=int, default=1024)
    p.add_argument("--attn_mode", default="flash")
    p.add_argument("--compile_blocks", action="store_true")
    p.add_argument(
        "--negative_prompt",
        default=(
            "worst quality, low quality, score_1, score_2, score_3, blurry, "
            "jpeg artifacts, sepia"
        ),
        help="CFG negative — production default, held constant across arms",
    )
    p.add_argument(
        "--pooled_text_proj",
        default="output/ckpt/pooled_text_proj-0611.safetensors",
    )
    p.add_argument("--quality_tags", default="masterpiece, absurdres, score_9")
    p.add_argument("--mod_pos_prompt", default="absurdres, masterpiece, score_9")
    p.add_argument(
        "--mod_neg_prompt",
        default="worst quality, low quality, score_1",
        help="decoupled steering negative (NOT the CFG negative — cos -0.38 trap)",
    )
    p.add_argument("--mod_start_layer", type=int, default=8)
    p.add_argument("--mod_end_layer", type=int, default=27)
    p.add_argument(
        "--caption_root",
        default="post_image_dataset/resized",
        help="derived-caption corpus to sample prompts from",
    )
    p.add_argument("--prompt_rng_seed", type=int, default=1234)
    p.add_argument(
        "--arms",
        default=None,
        help="comma list of arm names to run/score (default: all); "
        "'base' should stay in the list — pe_cos_base needs it",
    )
    p.add_argument(
        "--smoke", action="store_true", help="2 prompts, 2 seeds, w=3, 12 steps"
    )
    p.add_argument("--metrics_only", action="store_true")
    p.add_argument(
        "--run_dir", default=None, help="existing run dir for --metrics_only"
    )
    return p.parse_args()


def main() -> None:
    args = parse_cli()
    if args.smoke:
        args.n_prompts = 2
        args.seeds = "0,1"
        args.w_list = "3"
        args.steps = 12

    args.quality_tags_list = [
        t.strip() for t in args.quality_tags.split(",") if t.strip()
    ]
    seeds = [int(s) for s in args.seeds.split(",")]
    w_list = [float(w) for w in args.w_list.split(",")]
    arms = build_arms(w_list, include_both=not args.no_both_arm)
    if args.arms:
        keep = {a.strip() for a in args.arms.split(",") if a.strip()}
        unknown = keep - {a.name for a in arms}
        if unknown:
            raise SystemExit(
                f"unknown arm(s) {sorted(unknown)} — have {[a.name for a in arms]}"
            )
        arms = [a for a in arms if a.name in keep]

    if args.metrics_only:
        if not args.run_dir:
            raise SystemExit("--metrics_only needs --run_dir")
        run_dir = Path(args.run_dir)
        prompts = [
            PromptSpec(**d) for d in json.loads((run_dir / "prompts.json").read_text())
        ]
    else:
        run_dir = make_run_dir("mod_quality_tradeoff", label=args.label)
        prompts = sample_prompts(
            REPO_ROOT / args.caption_root,
            args.n_prompts,
            args.quality_tags_list,
            args.prompt_rng_seed,
        )
        (run_dir / "prompts.json").write_text(
            json.dumps([vars(s) for s in prompts], indent=2), encoding="utf-8"
        )
    images_dir = run_dir / "images"
    print(f"run dir: {run_dir}", flush=True)
    print(f"arms: {[a.name for a in arms]}", flush=True)

    if not args.metrics_only:
        run_generation(args, arms, prompts, seeds, images_dir)

    metrics = run_metrics(args, arms, prompts, seeds, run_dir, images_dir)
    artifacts = ["per_image.csv", "aggregates.csv", "prompts.json"]
    if plot := write_plot(metrics, arms, run_dir):
        artifacts.append(plot)
    artifacts.append(write_review_page(arms, prompts, seeds, run_dir, images_dir))

    ns = argparse.Namespace(**{k: v for k, v in vars(args).items()})
    write_result(
        run_dir,
        script="bench/mod_quality_tradeoff/run_bench.py",
        args=ns,
        metrics=metrics,
        artifacts=artifacts,
    )
    print(json.dumps(metrics["aggregates"], indent=2))
    print(f"review page: {run_dir / 'index.html'}")


if __name__ == "__main__":
    main()
