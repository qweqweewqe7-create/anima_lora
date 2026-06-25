"""Tag-level headroom & commitment-σ probe — Phase-0 (detector-free).

Implements the Phase-0 gate of ``docs/proposal/tag_headroom_commitment_sigma.md``.
We do **not** detect whether a tag rendered (no local tagger — too noisy on this
anime+NSFW domain). We *intervene* on each tag and read the model's own causal
response, three detector-free signals from generations we already know how to run:

  * **influence(T)** — localized image distance between the full caption C and
    C∖T (tag dropped), same seed. How much the model *uses* T. Near-zero on a
    high-frequency tag ⇒ the tag is being ignored.
  * **instability(T)** — seed-variance of T's affected region (the region the
    influence diff localizes), across seeds with the full caption. High variance
    = the model is *trying but rendering T inconsistently* = a failure proxy;
    near-zero = solved or memorized. This is the move that replaces the tagger:
    "renders correctly" is unmeasurable detector-free, but "renders
    consistently" is fully detector-free.
  * **commitment-σ(T)** — present T only above a σ cutoff (swap C→C∖T below the
    cutoff via ``generate_body(context_alt=…, tag_drop_sigma=…)``); the cutoff at
    which the result rejoins the full-caption baseline is when T locks.

Phase-0 layout (proposal §0):
  * **0a** — influence + instability on an a-priori suspect set. *Kill* if there
    is no high-influence + high-instability + high-frequency population (every
    suspect tag is either solved or ignored). *Pass* surfaces a flagged set.
  * **0b** — commitment-σ on the 0a-flagged tags only. *Pass* if they commit late
    (need the σ∈[0.6,0.8] band) — the headroom signature (used ∧ late ∧ unstable
    ∧ common) and the green light for Phase-1.

The one genuinely subjective question — *is the rendered feature good* — stays
human: the bench saves a ranked shortlist montage (full | drop | diff) for the
flagged tags; the eye adjudicates, the automation only ranks.

Cost note: tag-drop is pure prompt manipulation over the existing generate;
baselines are cached per (caption, seed) and reused across every tag that shares
a caption. The σ-gated 0b pass reuses the generation hook, not a custom loop.

Example::

    python bench/cross_attn_drive/tag_influence.py \\
        --phase both --seeds 4 --captions-per-tag 6 --infer-steps 28
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from anima_lora import GenerationRequest, load_vae  # noqa: E402
from bench._anima import add_common_args, add_model_args  # noqa: E402
from bench._common import make_run_dir, write_result  # noqa: E402
from library.env import anima_home  # noqa: E402
from library.inference.generation import generate_body  # noqa: E402
from library.inference.models import load_dit_model, load_text_encoder  # noqa: E402
from library.inference.output import decode_latent, pixels_to_pil  # noqa: E402
from library.inference.text import ensure_text_strategies, prepare_text_inputs  # noqa: E402
from library.runtime.device import clean_memory_on_device  # noqa: E402

# ---------------------------------------------------------------------------
# A-priori suspect set — the known-hard categories from the proposal (rendered
# text / logos, hands / fingers, counts, small accessories). Filtered down at
# runtime to the tags that actually appear in the loaded caption corpus with
# enough captions to measure. Override with --suspect-tags.
# ---------------------------------------------------------------------------
DEFAULT_SUSPECT_TAGS = [
    # rendered text / logos — localized glyph rendering, classically hard
    "logo",
    "artist logo",
    "english text",
    "speech bubble",
    "signature",
    "watermark",
    # hands / fingers — the canonical failure mode
    "fingernails",
    "nail polish",
    "holding",
    "interlocked fingers",
    "claw pose",
    # counts — global structure (Ngirls / Nboys)
    "2girls",
    "multiple girls",
    "1boy",
    # small accessories — localized fine detail
    "earrings",
    "hair ornament",
    "choker",
    "glasses",
    "piercing",
    "hairclip",
]

EPS = 1e-8


# ---------------------------------------------------------------------------
# Corpus + tag bookkeeping.
# ---------------------------------------------------------------------------
def _split_tags(caption: str) -> list[str]:
    return [t.strip() for t in caption.split(",") if t.strip()]


def drop_tag(caption: str, tag: str) -> str:
    """Return ``caption`` with every exact (case-insensitive) ``tag`` token removed."""
    tl = tag.strip().lower()
    kept = [t for t in _split_tags(caption) if t.lower() != tl]
    return ", ".join(kept)


def load_corpus(args) -> list[str]:
    """Load the caption corpus.

    Preference order: an explicit ``--prompts`` file → the dataset caption index
    (post_image_dataset/captions/caption_index.json, the 3k-caption master) →
    the cross-attn-drive real_prompts.txt fallback. Returns raw caption strings
    (one per image), with any ``--d/--w/--h`` inference suffixes stripped.
    """
    import json

    def _clean(line: str) -> str:
        line = line.strip()
        # strip trailing `--d 1001 --w 1024 --h 1024`-style inference args
        if " --" in line:
            line = line.split(" --", 1)[0].strip()
        return line

    if args.prompts:
        path = Path(args.prompts)
        lines = [_clean(x) for x in path.read_text().splitlines()]
        return [x for x in lines if x]

    index_path = Path(args.caption_index)
    if not index_path.is_absolute():
        index_path = anima_home() / index_path
    if index_path.exists():
        meta = json.loads(index_path.read_text())
        src_root = anima_home() / meta.get("meta", {}).get("src", "image_dataset")
        captions: list[str] = []
        for rec in meta.get("image_meta", {}).values():
            txt = src_root / rec["path"]
            try:
                captions.append(_clean(txt.read_text()))
            except OSError:
                continue
        captions = [c for c in captions if c]
        if captions:
            return captions

    fallback = Path(__file__).parent / "real_prompts.txt"
    return [_clean(x) for x in fallback.read_text().splitlines() if x.strip()]


def _danbooru_key(tag: str) -> str:
    """Danbooru tag names use underscores; captions use spaces. Normalize."""
    return tag.strip().lower().replace(" ", "_")


def load_danbooru_categories(args) -> dict[str, tuple[int, int]]:
    """danbooru-normalized tag name → (category int, post_count). Empty on miss.

    Keys are underscored/lowercased; look them up via :func:`_danbooru_key` so a
    space-separated caption tag (``hair ornament``) joins to ``hair_ornament``.
    """
    path = Path(args.danbooru_csv)
    if not path.is_absolute():
        path = anima_home() / path
    out: dict[str, tuple[int, int]] = {}
    if not path.exists():
        return out
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            try:
                out[_danbooru_key(row["name"])] = (
                    int(row["category"]),
                    int(row["post_count"]),
                )
            except (KeyError, ValueError):
                continue
    return out


# ---------------------------------------------------------------------------
# Image metrics.
# ---------------------------------------------------------------------------
def diff_map(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per-pixel L2 across channels of two [-1,1] CHW pixel tensors → [H,W]."""
    return (a.float() - b.float()).pow(2).sum(dim=0).sqrt()


def region_mask(mean_diff: torch.Tensor, frac: float) -> torch.Tensor:
    """Boolean [H,W] mask of the top-``frac`` pixels of ``mean_diff`` — the
    region the tag-drop localizes to (T's affected region)."""
    flat = mean_diff.flatten()
    k = max(1, int(round(frac * flat.numel())))
    thresh = torch.topk(flat, k).values.min()
    return mean_diff >= thresh


def caption_signals(
    full: list[torch.Tensor],
    drop: list[torch.Tensor],
    region_frac: float,
) -> dict:
    """Influence + instability for one (caption, tag) from per-seed image lists.

    ``full``/``drop`` are length-S lists of [-1,1] CHW tensors (same seeds).
    Returns the per-caption row plus the mean diff map (for montage saving).
    """
    diffs = torch.stack([diff_map(f, d) for f, d in zip(full, drop)], dim=0)  # [S,H,W]
    mean_diff = diffs.mean(dim=0)  # [H,W]
    mask = region_mask(mean_diff, region_frac)

    influence_bulk = float(diffs.mean())
    influence_local = float(diffs[:, mask].mean())
    # how peaked the effect is: region mean / whole-image mean
    concentration = float(mean_diff[mask].mean() / (mean_diff.mean() + EPS))

    # instability = seed-variance of the full-caption image, in T's region.
    stack = torch.stack(full, dim=0).float()  # [S,3,H,W]
    var_px = stack.var(dim=0, unbiased=False).mean(dim=0)  # [H,W] (mean over channel)
    instability_bulk = float(var_px.mean())
    instability_local = float(var_px[mask].mean())
    # excess instability: T's region vs the image at large (guards generic
    # sampling variance — we want EXCESS wobble, not "everything wobbles").
    instability_rel = float(instability_local / (instability_bulk + EPS))

    return {
        "row": {
            "influence_bulk": influence_bulk,
            "influence_local": influence_local,
            "concentration": concentration,
            "instability_bulk": instability_bulk,
            "instability_local": instability_local,
            "instability_rel": instability_rel,
        },
        "mean_diff": mean_diff,
        "mask": mask,
    }


# ---------------------------------------------------------------------------
# Generation harness.
# ---------------------------------------------------------------------------
class Generator:
    """Holds the loaded DiT + VAE + (resident) text encoder and drives
    ``generate_body``. Baseline images are cached per (prompt, seed)."""

    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        req = GenerationRequest(
            prompt="placeholder",
            dit=args.dit,
            vae=args.vae,
            text_encoder=args.text_encoder,
            negative_prompt=args.negative_prompt,
            image_size=(args.size, args.size),
            infer_steps=args.infer_steps,
            guidance_scale=args.cfg,
            flow_shift=args.flow_shift,
            attn_mode=args.attn_mode,
            device=args.device,
            seed=0,
        )
        self.gen_args = req.to_args()
        if args.lora_weight:
            self.gen_args.lora_weight = [args.lora_weight]
            self.gen_args.lora_multiplier = [args.lora_multiplier]
        # Block-compile (bit-exact, blessed path) — load_dit_model reads these and
        # compiles *after* any adapter apply. At fixed --size it's one token
        # family, so the first gen pays compile then every gen runs ~30% faster.
        self.gen_args.compile_blocks = bool(getattr(args, "compile", False))
        self.gen_args.compile_inductor_mode = getattr(args, "compile_mode", None)

        ensure_text_strategies(args.text_encoder)
        te_device = torch.device("cpu") if args.te_cpu else self.device
        text_encoder = load_text_encoder(
            self.gen_args, dtype=torch.bfloat16, device=te_device
        )
        text_encoder.eval()
        # Resident TE + shared encode cache so each unique prompt encodes once.
        self.shared = {"text_encoder": text_encoder, "conds_cache": {}}

        self.anima = load_dit_model(self.gen_args, self.device, torch.bfloat16)
        self.shared["model"] = self.anima

        self.vae = load_vae(args.vae, device="cpu", vae_2d=True)
        self.vae.to(torch.bfloat16).eval()

        # Bounded baseline cache: full-caption images are reused across suspect
        # tags that co-occur in a caption, so caching them saves real gens — but
        # a 1024px float32 image is ~12 MB on CPU, so an *unbounded* cache OOM-
        # kills a full run. Cap it (FIFO) and never cache drop images (they are
        # unique per (tag, caption) and never reused).
        self._cache: dict[tuple[str, int], torch.Tensor] = {}
        # ~12 MB/img at 1024px → cap ≈ a couple tags' worth of baselines (<1 GB).
        self._cache_cap = max(32, 2 * getattr(args, "seeds", 4) * getattr(args, "captions_per_tag", 6))

    def _context(self, prompt: str):
        return prepare_text_inputs(
            self.gen_args, self.device, self.anima, self.shared, prompt=prompt
        )

    @torch.no_grad()
    def image(self, prompt: str, seed: int, cache: bool = False) -> torch.Tensor:
        """Decoded [-1,1] CHW image for (prompt, seed).

        ``cache=True`` only for reusable baselines (full captions); bounded FIFO.
        """
        key = (prompt, seed)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        ctx, ctx_null = self._context(prompt)
        latent = generate_body(self.gen_args, self.anima, ctx, ctx_null, self.device, seed)
        img = decode_latent(self.vae, latent, self.device)
        clean_memory_on_device(self.device)
        if cache:
            if len(self._cache) >= self._cache_cap:
                self._cache.pop(next(iter(self._cache)))  # FIFO evict oldest
            self._cache[key] = img
        return img

    @torch.no_grad()
    def image_gated(
        self, prompt_full: str, prompt_drop: str, seed: int, cutoff: float
    ) -> torch.Tensor:
        """Commitment-σ image: caption C above ``cutoff``, C∖T below it."""
        ctx, ctx_null = self._context(prompt_full)
        ctx_alt, _ = self._context(prompt_drop)
        latent = generate_body(
            self.gen_args,
            self.anima,
            ctx,
            ctx_null,
            self.device,
            seed,
            context_alt=ctx_alt,
            tag_drop_sigma=cutoff,
        )
        img = decode_latent(self.vae, latent, self.device)
        clean_memory_on_device(self.device)
        return img


# ---------------------------------------------------------------------------
# Phase 0a.
# ---------------------------------------------------------------------------
def _median(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return 0.0
    mid = n // 2
    return s[mid] if n % 2 else 0.5 * (s[mid - 1] + s[mid])


def run_phase0a(gen: Generator, corpus, suspect_tags, danbooru, args, run_dir):
    seeds = list(range(args.seed, args.seed + args.seeds))

    # map each suspect tag → captions that contain it
    tag_caps: dict[str, list[str]] = {}
    for tag in suspect_tags:
        tl = tag.lower()
        caps = [c for c in corpus if tl in {t.lower() for t in _split_tags(c)}]
        if len(caps) >= args.min_captions:
            tag_caps[tag] = caps[: args.captions_per_tag]

    if not tag_caps:
        raise SystemExit(
            "No suspect tag has >= --min-captions captions in the corpus. "
            "Check --suspect-tags / --prompts."
        )

    # training frequency = #captions in corpus carrying the tag
    train_freq = {
        tag: sum(1 for c in corpus if tag.lower() in {t.lower() for t in _split_tags(c)})
        for tag in tag_caps
    }

    per_caption_rows: list[dict] = []
    # montages are written to disk inline (not held in RAM — a 1024px float32
    # montage is ~36 MB; holding all (tag, caption) pairs OOM-kills the run).
    montage_dir = run_dir / "montages"
    montage_dir.mkdir(parents=True, exist_ok=True)
    montage_paths: list[tuple[str, Path]] = []  # (tag, png path)

    for tag, caps in tag_caps.items():
        print(f"[0a] {tag!r}: {len(caps)} captions x {len(seeds)} seeds")
        safe = tag.replace(" ", "_").replace("/", "_")
        for ci, cap in enumerate(caps):
            cap_drop = drop_tag(cap, tag)
            if cap_drop == cap:  # nothing removed (shouldn't happen)
                continue
            # Baselines (full caption) cache + are reused across co-occurring
            # suspect tags; drops are unique per (tag, caption) — never cached.
            full = [gen.image(cap, s, cache=True) for s in seeds]
            drop = [gen.image(cap_drop, s, cache=False) for s in seeds]
            sig = caption_signals(full, drop, args.region_frac)
            per_caption_rows.append({"tag": tag, "caption_idx": ci, **sig["row"]})
            png = montage_dir / f"{safe}__cap{ci}.png"
            _write_montage(png, full, drop[0], sig["mean_diff"])
            montage_paths.append((tag, png))
            del full, drop, sig

    # aggregate per tag
    by_tag: dict[str, list[dict]] = defaultdict(list)
    for r in per_caption_rows:
        by_tag[r["tag"]].append(r)

    per_tag: list[dict] = []
    for tag, rows in by_tag.items():
        cat, post = danbooru.get(_danbooru_key(tag), (None, None))
        per_tag.append(
            {
                "tag": tag,
                "n_captions": len(rows),
                "influence_local": sum(r["influence_local"] for r in rows) / len(rows),
                "influence_bulk": sum(r["influence_bulk"] for r in rows) / len(rows),
                "concentration": sum(r["concentration"] for r in rows) / len(rows),
                "instability_local": sum(r["instability_local"] for r in rows) / len(rows),
                "instability_rel": sum(r["instability_rel"] for r in rows) / len(rows),
                "train_freq": train_freq[tag],
                "danbooru_category": cat,
                "danbooru_post_count": post,
            }
        )

    # relative classification — "high/low" against this measured set's medians.
    med_inf = _median([t["influence_local"] for t in per_tag])
    med_unstab = _median([t["instability_rel"] for t in per_tag])
    med_freq = _median([float(t["train_freq"]) for t in per_tag])

    for t in per_tag:
        inf_hi = t["influence_local"] >= med_inf
        unstab_hi = t["instability_rel"] >= med_unstab
        freq_hi = t["train_freq"] >= med_freq
        # cell assignment (proposal interpretation table)
        if not inf_hi and t["influence_local"] < med_inf * 0.5:
            cell = "ignored"  # high-freq + ~zero influence = not even attempted
        elif inf_hi and unstab_hi:
            cell = "headroom" if freq_hi else "headroom_lowfreq"
        elif inf_hi and not unstab_hi:
            cell = "solved_or_memorized"  # → memorization probe to disambiguate
        else:
            cell = "weak"
        t["inf_hi"] = inf_hi
        t["unstab_hi"] = unstab_hi
        t["freq_hi"] = freq_hi
        t["cell"] = cell
        t["flagged"] = bool(inf_hi and unstab_hi and freq_hi)

    # rank: high influence × high instability × high freq (rank-normalized product)
    def _rank_norm(key):
        order = sorted(per_tag, key=lambda x: x[key])
        return {id(t): i / max(1, len(per_tag) - 1) for i, t in enumerate(order)}

    r_inf = _rank_norm("influence_local")
    r_unstab = _rank_norm("instability_rel")
    r_freq = _rank_norm("train_freq")
    for t in per_tag:
        t["headroom_score"] = r_inf[id(t)] * r_unstab[id(t)] * r_freq[id(t)]
    per_tag.sort(key=lambda x: x["headroom_score"], reverse=True)

    # kill criterion: a trying-but-failing population must exist (some tag with
    # both high influence AND high instability).
    trying_failing = [t for t in per_tag if t["inf_hi"] and t["unstab_hi"]]
    flagged = [t for t in per_tag if t["flagged"]]
    verdict = "PASS" if trying_failing else "KILL"

    # write artifacts
    _write_csv(run_dir / "per_caption.csv", per_caption_rows)
    _write_csv(run_dir / "per_tag.csv", per_tag)
    # promote flagged (or top-ranked) tags' montages into shortlist/ for the eye
    shortlist_tags = {t["tag"] for t in (flagged or per_tag[: args.shortlist_n])}
    _promote_shortlist(run_dir / "shortlist", montage_paths, shortlist_tags)

    summary = {
        "verdict": verdict,
        "n_tags_measured": len(per_tag),
        "n_trying_failing": len(trying_failing),
        "flagged_tags": [t["tag"] for t in flagged],
        "trying_failing_tags": [t["tag"] for t in trying_failing],
        "medians": {
            "influence_local": med_inf,
            "instability_rel": med_unstab,
            "train_freq": med_freq,
        },
        "top_by_headroom_score": [
            {
                "tag": t["tag"],
                "cell": t["cell"],
                "influence_local": round(t["influence_local"], 5),
                "instability_rel": round(t["instability_rel"], 4),
                "train_freq": t["train_freq"],
                "headroom_score": round(t["headroom_score"], 4),
            }
            for t in per_tag[:10]
        ],
    }
    return summary, per_tag


# ---------------------------------------------------------------------------
# Phase 0b — commitment-σ on the flagged tags.
# ---------------------------------------------------------------------------
def run_phase0b(gen: Generator, corpus, flagged_tags, args, run_dir):
    seeds = list(range(args.seed, args.seed + args.seeds_0b))
    cutoffs = sorted(args.cutoffs)

    rows: list[dict] = []
    per_tag_commit: list[dict] = []

    for tag in flagged_tags:
        tl = tag.lower()
        caps = [c for c in corpus if tl in {t.lower() for t in _split_tags(c)}]
        caps = caps[: args.captions_0b]
        if not caps:
            continue
        print(f"[0b] {tag!r}: {len(caps)} captions x {len(cutoffs)} cutoffs")

        rel_by_cutoff: dict[float, list[float]] = defaultdict(list)
        for ci, cap in enumerate(caps):
            cap_drop = drop_tag(cap, tag)
            for s in seeds:
                full = gen.image(cap, s)
                drop = gen.image(cap_drop, s)
                # full-vs-drop distance = the normalizer (T fully present vs fully absent)
                d_full_drop = float(diff_map(full, drop).mean())
                for x in cutoffs:
                    gated = gen.image_gated(cap, cap_drop, s, x)
                    d_to_full = float(diff_map(full, gated).mean())
                    rel = d_to_full / (d_full_drop + EPS)
                    rows.append(
                        {
                            "tag": tag,
                            "caption_idx": ci,
                            "seed": s,
                            "cutoff": x,
                            "dist_to_full": d_to_full,
                            "dist_full_drop": d_full_drop,
                            "rel": rel,
                        }
                    )
                    rel_by_cutoff[x].append(rel)

        # mean rel curve over (caption, seed); commitment-σ = largest cutoff
        # whose result still rejoins the full baseline (rel <= --commit-thresh).
        curve = {x: sum(v) / len(v) for x, v in rel_by_cutoff.items()}
        commit_sigma = _commitment_sigma(curve, args.commit_thresh)
        late = commit_sigma is not None and 0.6 <= commit_sigma <= 0.8
        per_tag_commit.append(
            {
                "tag": tag,
                "commitment_sigma": commit_sigma,
                "late_band": late,
                "rel_curve": {str(k): round(v, 4) for k, v in sorted(curve.items())},
            }
        )

    _write_csv(run_dir / "commitment.csv", rows)

    late_tags = [t["tag"] for t in per_tag_commit if t["late_band"]]
    summary = {
        "verdict": "PASS" if late_tags else "KILL",
        "late_committing_tags": late_tags,
        "per_tag": per_tag_commit,
        "commit_thresh": args.commit_thresh,
    }
    return summary


def _commitment_sigma(curve: dict[float, float], thresh: float) -> Optional[float]:
    """Largest cutoff x with rel(x) <= thresh — the σ at which the tag locks
    (presenting it only above x already reproduces the full caption). ``None``
    if even the smallest cutoff hasn't rejoined the baseline."""
    below = [x for x, r in curve.items() if r <= thresh]
    return max(below) if below else None


# ---------------------------------------------------------------------------
# Artifacts.
# ---------------------------------------------------------------------------
def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})


def _write_montage(
    path: Path,
    full: list[torch.Tensor],
    drop0: torch.Tensor,
    mean_diff: torch.Tensor,
    thumb: int = 320,
) -> None:
    """Write one montage PNG for a (tag, caption), thumbnailed to keep it small.

    Layout (left→right): the **full caption at every seed** (the instability
    strip — read across these: do they agree on the tag's feature, or does it
    wobble?), a divider, then **drop@seed0** and the **diff heatmap** (influence
    — where dropping the tag changed the image). Called inline during 0a so the
    seed images don't accumulate in RAM. See ``how_to_observe.md``.
    """
    from PIL import Image

    def _thumb(pil: Image.Image) -> Image.Image:
        return pil.resize((thumb, thumb), Image.BILINEAR)

    seed_thumbs = [_thumb(pixels_to_pil(f)) for f in full]
    drop_thumb = _thumb(pixels_to_pil(drop0))
    d = (mean_diff - mean_diff.min()) / (mean_diff.max() - mean_diff.min() + EPS)
    diff_thumb = _thumb(
        Image.fromarray((d * 255).to(torch.uint8).cpu().numpy()).convert("RGB")
    )

    sep = 6  # divider between the instability strip and the influence panels
    panels = seed_thumbs + [drop_thumb, diff_thumb]
    width = thumb * len(panels) + sep
    canvas = Image.new("RGB", (width, thumb), (40, 40, 40))
    x = 0
    for i, p in enumerate(seed_thumbs):
        canvas.paste(p, (x, 0))
        x += thumb
    x += sep  # gap marks "full seeds | drop, diff"
    canvas.paste(drop_thumb, (x, 0))
    canvas.paste(diff_thumb, (x + thumb, 0))
    canvas.save(path)


def _promote_shortlist(
    out_dir: Path, montage_paths: list[tuple[str, Path]], shortlist_tags: set
) -> None:
    """Copy the flagged/top tags' montages (already on disk under montages/) into
    shortlist/ so the human eye has the headroom candidates in one place."""
    import shutil

    out_dir.mkdir(parents=True, exist_ok=True)
    for tag, png in montage_paths:
        if tag in shortlist_tags and png.exists():
            shutil.copy2(png, out_dir / png.name)


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    add_model_args(p)  # --dit / --vae / --text_encoder
    # --label/--seed/--device/--dtype/--attn_mode + --compile/--compile_mode.
    # --compile maps to the blessed block-compile path (see Generator); at our
    # fixed square --size every gen is one token family, so it's one graph.
    add_common_args(p)
    p.add_argument("--lora-weight", default=None, help="Optional adapter (default: base DiT).")
    p.add_argument("--lora-multiplier", type=float, default=1.0)

    p.add_argument("--phase", choices=["0a", "0b", "both"], default="both")
    p.add_argument("--prompts", default=None, help="Caption file (overrides caption index).")
    p.add_argument(
        "--caption-index",
        default="post_image_dataset/captions/caption_index.json",
        help="Dataset caption index (the 3k-caption master).",
    )
    p.add_argument("--danbooru-csv", default="models/danbooru_tags_classified.csv")
    p.add_argument(
        "--suspect-tags",
        nargs="*",
        default=None,
        help="Override the a-priori suspect set (default: built-in known-hard set).",
    )

    # generation knobs
    p.add_argument("--size", type=int, default=1024, help="Square gen size (px).")
    p.add_argument("--infer-steps", type=int, default=28)
    p.add_argument("--cfg", type=float, default=4.0)
    p.add_argument("--flow-shift", type=float, default=3.0)
    p.add_argument("--negative-prompt", default="")
    p.add_argument("--te-cpu", action="store_true", help="Keep text encoder on CPU.")

    # 0a knobs
    p.add_argument("--seeds", type=int, default=4, help="Seeds per caption (0a).")
    p.add_argument("--captions-per-tag", type=int, default=6)
    p.add_argument("--min-captions", type=int, default=3)
    p.add_argument("--region-frac", type=float, default=0.05, help="Top-frac region mask.")
    p.add_argument("--shortlist-n", type=int, default=8, help="Tags to save montages for.")

    # 0b knobs
    p.add_argument("--seeds-0b", type=int, default=2)
    p.add_argument("--captions-0b", type=int, default=3)
    p.add_argument("--cutoffs", type=float, nargs="*", default=[0.95, 0.8, 0.6, 0.4])
    p.add_argument(
        "--commit-thresh",
        type=float,
        default=0.2,
        help="rel(x) <= thresh counts as 'rejoined the full baseline'.",
    )
    p.add_argument(
        "--tags",
        nargs="*",
        default=None,
        help="0b: explicit tag list (default: 0a-flagged set, or suspect set if --phase 0b).",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    suspect = args.suspect_tags or DEFAULT_SUSPECT_TAGS

    run_dir = make_run_dir("cross_attn_drive", label=args.label or "tag_influence")
    corpus = load_corpus(args)
    danbooru = load_danbooru_categories(args)
    print(f"corpus: {len(corpus)} captions | suspect tags: {len(suspect)}")

    gen = Generator(args)

    metrics: dict = {"corpus_size": len(corpus), "suspect_tags": suspect}
    artifacts = ["per_caption.csv", "per_tag.csv", "commitment.csv"]

    flagged_for_0b: list[str] = args.tags or []
    if args.phase in ("0a", "both"):
        summary0a, per_tag = run_phase0a(gen, corpus, suspect, danbooru, args, run_dir)
        metrics["phase0a"] = summary0a
        print(f"\n[0a] verdict={summary0a['verdict']}  flagged={summary0a['flagged_tags']}")
        if not flagged_for_0b:
            # prefer the strictly-flagged set; fall back to trying-failing tags
            flagged_for_0b = (
                summary0a["flagged_tags"] or summary0a["trying_failing_tags"]
            )

    if args.phase in ("0b", "both"):
        if not flagged_for_0b:
            flagged_for_0b = suspect[: args.shortlist_n]
            print(f"[0b] no 0a flags — falling back to suspect set: {flagged_for_0b}")
        summary0b = run_phase0b(gen, corpus, flagged_for_0b, args, run_dir)
        metrics["phase0b"] = summary0b
        print(
            f"\n[0b] verdict={summary0b['verdict']}  "
            f"late-committing={summary0b['late_committing_tags']}"
        )

    write_result(
        run_dir,
        script=__file__,
        args=vars(args),
        metrics=metrics,
        label=args.label,
        artifacts=artifacts,
        device=args.device,
    )
    print(f"\nwrote {run_dir}/result.json")


if __name__ == "__main__":
    main()
