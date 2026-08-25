"""Phase 0 bench — does text steering emerge on PE at this dataset's scale?

Trains a :class:`networks.methods.steer_pe.SteerPE` adapter (gated cross-attn
on frozen PE-Spatial-B16-512, Qwen3-TE keys, per-patch BCE) on the SAM3 masks
the region prep already produced, then scores held-out artists on:

* ``pr_auc[prompt]``      — per-patch PR-AUC vs the SAM3 mask, steered
* ``pr_auc_gate0``        — same head with every gate scaled to 0 (unsteered tower)
* ``pr_auc_swapped``      — scored with the *wrong* prompt (paper Tab. 6 control)
* ``pair_steer``          — on images with girl AND boy masks: share of predicted
                            mass inside the girl mask under "the girl" vs "the boy"
* qualitative sheets      — hair-colour prompts on ``2girls`` captions and the
                            audit's zero-box multi-view images

Run: ``make daemon-run ARGS="--label p0 bench/steer_pe/run_bench.py"``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bench._common import make_run_dir, start_heartbeat, write_result  # noqa: E402
from library.env import resolve_under_home  # noqa: E402
from networks.methods.steer_pe import (  # noqa: E402
    SteerPE,
    encode_prompts,
    patch_targets,
    pr_auc,
    soft_bce,
)

KINDS = {
    "masks": "the girl",
    "masks_boy": "the boy",
    "masks_head": "the face",
    "masks_person": "a person",
}
HAIR_RE = re.compile(r"\b([a-z]+) hair\b")
HAIR_COLOURS = {
    "black",
    "brown",
    "blonde",
    "blue",
    "white",
    "silver",
    "grey",
    "gray",
    "red",
    "pink",
    "purple",
    "green",
    "orange",
    "aqua",
    "light brown",
    "dark",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--label", default=None)
    p.add_argument("--region_base", default="post_image_dataset/easycontrol/region")
    p.add_argument("--resized_dir", default="post_image_dataset/resized")
    p.add_argument(
        "--qwen3", default="models/text_encoders/qwen_3_06b_base.safetensors"
    )
    p.add_argument(
        "--anime_seg",
        default="/media/sorryhyun/새 볼륨/dataset/anime_segmentation",
        help="skytnt/anime-segmentation checkout (imgs/ masks/ fg/) — Tier B",
    )
    p.add_argument(
        "--anime_seg_mode",
        default="off",
        choices=["off", "eval", "train"],
        help="off: ignore; eval: score held-out imgs/masks only (control); "
        "train: also add imgs/masks + fg composites to the training mix",
    )
    p.add_argument(
        "--anime_seg_prompt", default="a person", help="prompt bound to Tier B masks"
    )
    p.add_argument("--pe", default="pe_spatial", choices=["pe_spatial", "pe_core"])
    p.add_argument("--res", type=int, default=512)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr_final", type=float, default=3e-5)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--holdout_frac", type=float, default=0.15)
    p.add_argument("--max_eval", type=int, default=120, help="per prompt")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--audit_images",
        default="ama_mitsuki/5828774,ama_mitsuki/12971564,ama_mitsuki/7597435,ama_mitsuki/12971572",
    )
    return p.parse_args()


# ── data ─────────────────────────────────────────────────────────────────────


def collect_pairs(region_base: Path, resized: Path) -> list[dict]:
    rows = []
    for kind, prompt in KINDS.items():
        for m in sorted((region_base / kind).rglob("*_mask.png")):
            rel = m.relative_to(region_base / kind)
            stem = rel.name[: -len("_mask.png")]
            img = resized / rel.parent / f"{stem}.png"
            if not img.exists():
                continue
            rows.append(
                {
                    "image": str(img),
                    "mask": str(m),
                    "prompt": prompt,
                    "kind": kind,
                    "artist": rel.parts[0],
                    "key": f"{rel.parent}/{stem}",
                }
            )
    return rows


def collect_anime_seg(root: Path, prompt: str) -> list[dict]:
    """skytnt/anime-segmentation: ``imgs/*.jpg`` + ``masks/*.jpg`` (real GT, 1 111)
    and ``fg/**/*.png`` (RGBA foregrounds; alpha is the mask, composited on a
    random flat background at load time). Each file is its own split unit."""
    rows = []
    for img in sorted((root / "imgs").glob("*.jpg")):
        m = root / "masks" / img.name
        if m.exists():
            rows.append(
                {
                    "image": str(img),
                    "mask": str(m),
                    "prompt": prompt,
                    "kind": "aseg_imgs",
                    "artist": f"aseg_imgs/{img.stem}",
                    "key": f"aseg_imgs/{img.stem}",
                }
            )
    for fg in sorted((root / "fg").rglob("*.png")):
        rows.append(
            {
                "image": str(fg),
                "mask": None,
                "prompt": prompt,
                "kind": "aseg_fg",
                "artist": f"aseg_fg/{fg.stem}",
                "key": f"aseg_fg/{fg.stem}",
            }
        )
    return rows


def split_by_artist(rows: list[dict], frac: float, seed: int):
    def h(a: str) -> float:
        return (
            int(hashlib.sha1(f"{seed}:{a}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        )

    train = [r for r in rows if h(r["artist"]) >= frac]
    test = [r for r in rows if h(r["artist"]) < frac]
    return train, test


class PairDataset(Dataset):
    def __init__(self, rows: list[dict], res: int, train: bool):
        self.rows, self.res, self.train = rows, res, train

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        if r["mask"] is None:  # RGBA foreground: alpha is the mask
            rgba = Image.open(r["image"]).convert("RGBA")
            msk = rgba.getchannel("A").resize((self.res, self.res), Image.BILINEAR)
            bg = Image.new(
                "RGB", rgba.size, tuple(random.randrange(256) for _ in range(3))
            )
            bg.paste(rgba, mask=rgba.getchannel("A"))
            img = bg.resize((self.res, self.res), Image.BICUBIC)
        else:
            img = (
                Image.open(r["image"])
                .convert("RGB")
                .resize((self.res, self.res), Image.BICUBIC)
            )
            msk = (
                Image.open(r["mask"])
                .convert("L")
                .resize((self.res, self.res), Image.BILINEAR)
            )
        x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 127.5 - 1.0).permute(
            2, 0, 1
        )
        m = (torch.from_numpy(np.asarray(msk, dtype=np.float32)) > 127.5).float()[None]
        if self.train and random.random() < 0.5:
            x, m = x.flip(-1), m.flip(-1)
        return x, m, r["prompt"], i


def load_image(path: Path, res: int) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((res, res), Image.BICUBIC)
    return torch.from_numpy(np.asarray(img, dtype=np.float32) / 127.5 - 1.0).permute(
        2, 0, 1
    )


# ── model ────────────────────────────────────────────────────────────────────


def build(args):
    from library.anima.weights import load_qwen3_text_encoder
    from library.vision.encoders import _load_pe_encoder, _load_pe_spatial_encoder

    dev = torch.device(args.device)
    if args.pe == "pe_spatial":
        pe = _load_pe_spatial_encoder(
            dev, str(REPO_ROOT / "models/pe/PE-Spatial-B16-512.pt"), dtype=torch.float32
        ).inner
    else:
        pe = _load_pe_encoder(
            dev, str(REPO_ROOT / "models/pe/PE-Core-L14-336.pt"), dtype=torch.float32
        ).inner
    te, tok = load_qwen3_text_encoder(args.qwen3, dtype=torch.bfloat16, device=str(dev))
    te = te.to(dev).eval()
    text_dim = te.config.hidden_size
    model = SteerPE(pe, text_dim=text_dim, heads=args.heads).to(dev)
    return model, te, tok


class PromptBank:
    """Encode each distinct prompt once (Qwen3 is frozen)."""

    def __init__(self, te, tok, device):
        self.te, self.tok, self.device = te, tok, device
        self.cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    def __call__(self, prompts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        need = [p for p in dict.fromkeys(prompts) if p not in self.cache]
        if need:
            h, m = encode_prompts(self.te, self.tok, need, device=self.device)
            for j, p in enumerate(need):
                self.cache[p] = (h[j], m[j])
        hs = torch.stack([self.cache[p][0] for p in prompts])
        ms = torch.stack([self.cache[p][1] for p in prompts])
        return hs, ms


# ── eval helpers ─────────────────────────────────────────────────────────────


@torch.no_grad()
def heat(
    model: SteerPE,
    bank: PromptBank,
    x: torch.Tensor,
    prompts: list[str] | None,
    gate: float = 1.0,
):
    model.set_gate_scale(gate)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        if prompts is None:
            tokens = model(x)
        else:
            h, m = bank(prompts)
            tokens = model(x, h, m)
        logits = model.heat_logits(tokens, model.grid(x))
    model.set_gate_scale(1.0)
    return logits.float()


def overlay(
    img: torch.Tensor, prob: torch.Tensor, label: str, mask: torch.Tensor | None = None
) -> Image.Image:
    res = img.shape[-1]
    base = ((img.permute(1, 2, 0).cpu().numpy() + 1) * 127.5).clip(0, 255)
    p = (
        F.interpolate(
            prob[None, None], size=(res, res), mode="bilinear", align_corners=False
        )[0, 0]
        .cpu()
        .numpy()
    )
    ov = base.copy()
    ov[..., 0] = np.clip(base[..., 0] * (1 - p) + 255 * p, 0, 255)
    ov[..., 1] = base[..., 1] * (1 - 0.8 * p)
    ov[..., 2] = base[..., 2] * (1 - 0.8 * p)
    im = Image.fromarray(ov.astype(np.uint8))
    d = ImageDraw.Draw(im)
    if mask is not None:
        mk = mask[0].cpu().numpy() > 0.5
        edge = mk ^ np.roll(mk, 1, 0) | mk ^ np.roll(mk, 1, 1)
        ys, xs = np.nonzero(edge)
        for y, x in zip(ys[::3], xs[::3]):
            d.point((int(x), int(y)), fill=(0, 255, 0))
    d.rectangle([0, 0, res, 14], fill=(0, 0, 0))
    d.text((2, 1), label[:90], fill=(255, 255, 255))
    return im


def sheet(tiles: list[Image.Image], ncol: int) -> Image.Image:
    if not tiles:
        return Image.new("RGB", (8, 8))
    w, h = tiles[0].size
    nrow = math.ceil(len(tiles) / ncol)
    out = Image.new("RGB", (w * ncol, h * nrow), (40, 40, 40))
    for i, t in enumerate(tiles):
        out.paste(t, ((i % ncol) * w, (i // ncol) * h))
    return out


# ── main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    run_dir = make_run_dir("steer_pe", label=args.label)
    start_heartbeat(label="steer_pe")
    region_base = resolve_under_home(args.region_base)
    resized = resolve_under_home(args.resized_dir)

    rows = collect_pairs(region_base, resized)
    train_rows, test_rows = split_by_artist(rows, args.holdout_frac, args.seed)
    aseg_test: list[dict] = []
    if args.anime_seg_mode != "off":
        aseg = collect_anime_seg(Path(args.anime_seg), args.anime_seg_prompt)
        a_train, a_test = split_by_artist(aseg, args.holdout_frac, args.seed)
        aseg_test = [r for r in a_test if r["kind"] == "aseg_imgs"]  # real GT only
        if args.anime_seg_mode == "train":
            train_rows += a_train
        print(
            f"anime_seg[{args.anime_seg_mode}]: imgs={sum(r['kind'] == 'aseg_imgs' for r in aseg)} "
            f"fg={sum(r['kind'] == 'aseg_fg' for r in aseg)} train_added={len(a_train) if args.anime_seg_mode == 'train' else 0} "
            f"eval={len(aseg_test)}",
            flush=True,
        )
    all_kinds = list(KINDS) + ["aseg_imgs", "aseg_fg"]
    counts = {
        k: sum(r["kind"] == k for r in rows + aseg_test + train_rows) for k in all_kinds
    }
    counts = {k: v for k, v in counts.items() if v}
    print(
        f"pairs={len(rows)} train={len(train_rows)} test={len(test_rows)} {counts}",
        flush=True,
    )

    model, te, tok = build(args)
    dev = torch.device(args.device)
    bank = PromptBank(te, tok, dev)
    n_params = sum(p.numel() for p in model.adapter_parameters())
    print(
        f"adapter params: {n_params / 1e6:.2f}M  cross_attn_layers={model.cross_attn_layers}",
        flush=True,
    )

    # ── train ────────────────────────────────────────────────────────────
    loader = DataLoader(
        PairDataset(train_rows, args.res, train=True),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        drop_last=True,
        persistent_workers=args.workers > 0,
    )
    opt = torch.optim.AdamW(
        list(model.adapter_parameters()), lr=args.lr, weight_decay=0.05
    )

    def lr_at(step: int) -> float:
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        t = (step - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr_final + 0.5 * (args.lr - args.lr_final) * (
            1 + math.cos(math.pi * t)
        )

    log = []
    step, t0 = 0, time.time()
    model.train()
    model.pe.eval()
    while step < args.steps:
        for x, m, prompts, _ in loader:
            if step >= args.steps:
                break
            x, m = x.to(dev, non_blocking=True), m.to(dev, non_blocking=True)
            h, mk = bank(list(prompts))
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                tokens = model(x, h, mk)
                logits = model.heat_logits(tokens, model.grid(x))
            target = patch_targets(m, model.grid(x))
            loss = soft_bce(logits, target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.adapter_parameters()), 1.0)
            opt.step()
            if step % 25 == 0:
                gates = [
                    round(float(torch.tanh(ca.gate)), 3)
                    for ca in model.cross_attn.values()
                ]
                log.append(
                    {
                        "step": step,
                        "loss": float(loss),
                        "lr": lr_at(step),
                        "gates": gates,
                    }
                )
                print(
                    f"[{step}/{args.steps}] loss={float(loss):.4f} gates={gates} {time.time() - t0:.0f}s",
                    flush=True,
                )
            step += 1
    model.eval()
    (run_dir / "train_log.json").write_text(json.dumps(log, indent=1))
    from safetensors.torch import save_file

    save_file(
        {k: v.contiguous() for k, v in model.adapter_state_dict().items()},
        str(run_dir / "steer_pe_adapter.safetensors"),
    )

    # ── eval: held-out PR-AUC, gate-0, swapped prompt ────────────────────
    metrics: dict = {
        "pairs": len(rows),
        "train": len(train_rows),
        "test": len(test_rows),
        "kind_counts": counts,
        "adapter_params": n_params,
        "final_gates": [float(torch.tanh(ca.gate)) for ca in model.cross_attn.values()],
    }
    swap = {
        "the girl": "the boy",
        "the boy": "the girl",
        "the face": "a person",
        "a person": "the face",
    }
    per_prompt: dict[str, dict[str, list[float]]] = {
        p: {"steered": [], "gate0": [], "swapped": []} for p in KINDS.values()
    }
    tiles = []
    by_kind: dict[str, list[dict]] = {}
    for r in test_rows:
        by_kind.setdefault(r["kind"], []).append(r)
    eval_rows = [r for k in KINDS for r in by_kind.get(k, [])[: args.max_eval]]
    ds = PairDataset(eval_rows, args.res, train=False)
    for x, m, prompts, idx in DataLoader(
        ds, batch_size=args.batch_size, num_workers=args.workers
    ):
        x, m = x.to(dev), m.to(dev)
        target = patch_targets(m, model.grid(x)) > 0.5
        prompts = list(prompts)
        steered = heat(model, bank, x, prompts)
        gate0 = heat(model, bank, x, prompts, gate=0.0)
        swapped = heat(model, bank, x, [swap[p] for p in prompts])
        for j, p in enumerate(prompts):
            if target[j].sum() == 0:
                continue
            per_prompt[p]["steered"].append(pr_auc(steered[j], target[j]))
            per_prompt[p]["gate0"].append(pr_auc(gate0[j], target[j]))
            per_prompt[p]["swapped"].append(pr_auc(swapped[j], target[j]))
            if len(tiles) < 48 and int(idx[j]) % 7 == 0:
                tiles.append(
                    overlay(
                        x[j],
                        steered[j].sigmoid(),
                        f"{p}  auc={per_prompt[p]['steered'][-1]:.2f}",
                        m[j],
                    )
                )
                tiles.append(
                    overlay(
                        x[j],
                        swapped[j].sigmoid(),
                        f"swap:{swap[p]}  auc={per_prompt[p]['swapped'][-1]:.2f}",
                        m[j],
                    )
                )
    metrics["pr_auc"] = {
        p: {k: (float(np.nanmean(v)) if v else None) for k, v in d.items()}
        | {"n": len(d["steered"])}
        for p, d in per_prompt.items()
    }
    sheet(tiles, 6).save(run_dir / "heldout_sheet.png")
    print("PR-AUC:", json.dumps(metrics["pr_auc"], indent=1), flush=True)

    # ── Tier B: held-out anime-segmentation real GT (teacher-independent) ─
    if aseg_test:
        a_swap = "the face"
        acc = {"steered": [], "gate0": [], "swapped": []}
        for x, m, prompts, _ in DataLoader(
            PairDataset(aseg_test[: args.max_eval], args.res, train=False),
            batch_size=args.batch_size,
            num_workers=args.workers,
        ):
            x, m = x.to(dev), m.to(dev)
            prompts = list(prompts)
            target = patch_targets(m, model.grid(x)) > 0.5
            st = heat(model, bank, x, prompts)
            g0 = heat(model, bank, x, prompts, gate=0.0)
            sw = heat(model, bank, x, [a_swap] * len(prompts))
            for j in range(len(prompts)):
                acc["steered"].append(pr_auc(st[j], target[j]))
                acc["gate0"].append(pr_auc(g0[j], target[j]))
                acc["swapped"].append(pr_auc(sw[j], target[j]))
        metrics["pr_auc_anime_seg"] = {
            "prompt": args.anime_seg_prompt,
            "swap": a_swap,
            "n": len(acc["steered"]),
            **{k: float(np.nanmean(v)) for k, v in acc.items()},
        }
        print("anime_seg PR-AUC:", json.dumps(metrics["pr_auc_anime_seg"]), flush=True)

    # ── pair steer: girl vs boy on images that carry both masks ───────────
    girl = {r["key"]: r for r in test_rows if r["kind"] == "masks"}
    boy = {r["key"]: r for r in test_rows if r["kind"] == "masks_boy"}
    both = sorted(set(girl) & set(boy))[: args.max_eval]
    pair = {
        "n": len(both),
        "girl_share_under_girl": [],
        "girl_share_under_boy": [],
        "boy_share_under_boy": [],
        "boy_share_under_girl": [],
    }
    ptiles = []
    for key in both:
        x = load_image(Path(girl[key]["image"]), args.res)[None].to(dev)
        gm = PairDataset([girl[key]], args.res, False)[0][1].to(dev)
        bm = PairDataset([boy[key]], args.res, False)[0][1].to(dev)
        gt = patch_targets(gm[None], model.grid(x))[0] > 0.5
        bt = patch_targets(bm[None], model.grid(x))[0] > 0.5
        pg = heat(model, bank, x, ["the girl"])[0].sigmoid()
        pb = heat(model, bank, x, ["the boy"])[0].sigmoid()
        pair["girl_share_under_girl"].append(float(pg[gt].sum() / pg.sum()))
        pair["girl_share_under_boy"].append(float(pb[gt].sum() / pb.sum()))
        pair["boy_share_under_boy"].append(float(pb[bt].sum() / pb.sum()))
        pair["boy_share_under_girl"].append(float(pg[bt].sum() / pg.sum()))
        if len(ptiles) < 24:
            ptiles.append(
                overlay(
                    x[0],
                    pg,
                    f"the girl  g={pair['girl_share_under_girl'][-1]:.2f} b={pair['boy_share_under_girl'][-1]:.2f}",
                    gm,
                )
            )
            ptiles.append(
                overlay(
                    x[0],
                    pb,
                    f"the boy  g={pair['girl_share_under_boy'][-1]:.2f} b={pair['boy_share_under_boy'][-1]:.2f}",
                    bm,
                )
            )
    metrics["pair_steer"] = {
        k: (float(np.mean(v)) if isinstance(v, list) and v else v)
        for k, v in pair.items()
    }
    sheet(ptiles, 4).save(run_dir / "pair_sheet.png")
    print("pair_steer:", metrics["pair_steer"], flush=True)

    # ── attribute probe: 2girls captions with two distinct hair colours ──
    atiles = []
    attr = []
    for txt in sorted(resized.rglob("*.txt")):
        if len(atiles) >= 40:
            break
        if txt.name.endswith(".variants.txt"):
            continue
        cap = txt.read_text(encoding="utf-8")
        if "2girls" not in cap or "multiple views" in cap:
            continue
        cols = [c for c in HAIR_RE.findall(cap) if c in HAIR_COLOURS]
        cols = list(dict.fromkeys(cols))
        if len(cols) != 2:
            continue
        img = txt.with_suffix(".png")
        if not img.exists():
            continue
        x = load_image(img, args.res)[None].to(dev)
        probs = [
            heat(model, bank, x, [f"the girl with {c} hair"])[0].sigmoid() for c in cols
        ]
        pg = heat(model, bank, x, ["the girl"])[0].sigmoid()
        # steerability proxy without masks: how different are the two attribute maps?
        a, b = probs[0].flatten(), probs[1].flatten()
        cos = float(F.cosine_similarity(a, b, dim=0))
        attr.append(
            {
                "image": str(img.relative_to(resized)),
                "colours": cols,
                "cos_between_attr_maps": cos,
            }
        )
        atiles.append(overlay(x[0], pg, "the girl"))
        atiles.append(overlay(x[0], probs[0], f"{cols[0]} hair  cos={cos:.2f}"))
        atiles.append(overlay(x[0], probs[1], f"{cols[1]} hair"))
    metrics["attribute_probe"] = {
        "n": len(attr),
        "mean_cos_between_attr_maps": (
            float(np.mean([a["cos_between_attr_maps"] for a in attr])) if attr else None
        ),
    }
    (run_dir / "attribute_probe.json").write_text(json.dumps(attr, indent=1))
    sheet(atiles, 6).save(run_dir / "attribute_sheet.png")
    print("attribute_probe:", metrics["attribute_probe"], flush=True)

    # ── multi-view audit images (SAM3 found zero boxes) ──────────────────
    mtiles = []
    for rel in [s for s in args.audit_images.split(",") if s]:
        img = resized / f"{rel}.png"
        if not img.exists():
            continue
        x = load_image(img, args.res)[None].to(dev)
        for p in ("the girl", "the boy", "the face"):
            pr = heat(model, bank, x, [p])[0].sigmoid()
            mtiles.append(
                overlay(x[0], pr, f"{rel.split('/')[-1]} {p} max={float(pr.max()):.2f}")
            )
    sheet(mtiles, 3).save(run_dir / "multiview_sheet.png")

    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics=metrics,
        label=args.label,
        artifacts=[
            "train_log.json",
            "steer_pe_adapter.safetensors",
            "heldout_sheet.png",
            "pair_sheet.png",
            "attribute_sheet.png",
            "attribute_probe.json",
            "multiview_sheet.png",
        ],
    )
    print(f"result → {run_dir}", flush=True)


if __name__ == "__main__":
    main()
