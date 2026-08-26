#!/usr/bin/env python
"""External tagger vs anima-tagger on OUR held-out split, intersection vocab.

Scores an off-the-shelf timm anime tagger (default: animetimm
``convnextv2_huge.dbv4-full``, 693M params @512px, 12,476 tags) and a trained
anima-tagger checkpoint on the SAME val images — the checkpoint's own
``dataset.json`` split, ground truth = the caption sidecars the manifest was
built from — restricted to the tags BOTH models can emit:

  * ours can't emit tags below its ``min_freq`` cut; dbv4 can't emit our
    ``@artist`` roster / renamed tags. Either side scored on the other's
    blind spots would just measure vocab size, not the tagger.
  * headline = mean AP (threshold-free). Our per-tag thresholds were
    F1-calibrated ON THIS val split, so thresholded F1 is optimistic for us;
    dbv4's ``best_threshold`` was tuned on its own danbooru split. Both are
    reported, plus a flat 0.40 for both, but read mAP first.

Stratified by our vocab category (general / character / count) and by our
train-frequency tier; 4-way rating accuracy is scored separately (dbv4's
general/questionable mapped onto our safe/nsfw).

READOUT: bench/tagger_external/results/<ts>[-label]/{summary.md, result.json,
         per_tag.csv}

Run through the daemon (agent GPU work) from anima_lora/::

    make daemon-run ARGS="bench/tagger_external/run_bench.py --label dbv4-vs-v5"
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
from PIL import Image  # noqa: E402

from bench._common import make_run_dir, write_result  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

RATING_MAP = {  # external danbooru rating name -> our vocab rating name
    "general": "safe",
    "sensitive": "sensitive",
    "questionable": "nsfw",
    "explicit": "explicit",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--model_dir", default="models/captioners/anima-tagger-v5")
    p.add_argument(
        "--external_repo",
        default="animetimm/convnextv2_huge.dbv4-full",
        help="HF repo of a timm anime tagger (model.safetensors + "
        "selected_tags.csv + preprocess.json).",
    )
    p.add_argument(
        "--external_arch",
        default="convnextv2_huge",
        help="timm architecture name used to instantiate the external model.",
    )
    p.add_argument("--external_img_size", type=int, default=512)
    p.add_argument("--external_flat_thr", type=float, default=0.40)
    p.add_argument("--split", choices=["val", "train"], default="val")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--external_batch_size", type=int, default=8)
    p.add_argument("--feature_cache_dir", default=None)
    p.add_argument("--feature_cache_workers", type=int, default=4)
    p.add_argument("--freq_head_min", type=int, default=1000)
    p.add_argument("--freq_mid_min", type=int, default=200)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--label", default=None)
    p.add_argument(
        "--limit", type=int, default=0, help="Debug: score only the first N images."
    )
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Ours — same loader path as the (archived) tagger_eval bench, plus ratings.
# --------------------------------------------------------------------------- #


def collect_ours(args, model_dir: Path, cfg_d: dict, split_stems: List[str]):
    from torch.utils.data import DataLoader

    from library.captioning.anima_tagger_data import (
        BucketBatchSampler,
        CachedDualDataset,
        TaggerManifest,
        collate_dual_token_batch,
    )
    from library.captioning.anima_tagger_model import AnimaTaggerConfig, AnimaTaggerHead
    from library.vision.encoders import get_encoder_info
    from safetensors.torch import load_file as st_load
    from scripts.anima_tagger.caches import cache_dir_for, feature_cache_root

    cfg = AnimaTaggerConfig.from_dict(cfg_d["model"])
    model = AnimaTaggerHead(cfg)
    model.load_state_dict(st_load(str(model_dir / "model.safetensors")))
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model.to(device).eval()

    pool_kind = str(cfg_d.get("pool_kind", cfg.pool_kind))
    pool_kind_aux = str(cfg_d.get("pool_kind_aux", cfg.pool_kind_aux))
    encoder = cfg_d.get("encoder", "pe")
    aux_encoder = cfg_d.get("aux_encoder", "pe_spatial")
    feature_root = feature_cache_root(args)
    cache_dir = cache_dir_for(feature_root, pool_kind, encoder)
    cache_dir_aux = cache_dir_for(feature_root, pool_kind_aux, aux_encoder)
    for d in (cache_dir, cache_dir_aux):
        if not d.exists():
            raise SystemExit(f"missing feature cache {d}")
    spec = get_encoder_info(encoder).bucket_spec if pool_kind == "map" else None
    spec_aux = (
        get_encoder_info(aux_encoder).bucket_spec if pool_kind_aux == "map" else None
    )
    manifest = TaggerManifest.from_path(model_dir / "dataset.json")
    ds = CachedDualDataset(
        manifest,
        cache_dir,
        pool_kind,
        spec,
        cache_dir_aux,
        pool_kind_aux,
        spec_aux,
        stems_subset=split_stems,
    )
    sampler = BucketBatchSampler(
        ds.buckets, batch_size=args.batch_size, seed=args.seed, shuffle=False
    )
    loader = DataLoader(
        ds,
        batch_sampler=sampler,
        num_workers=args.feature_cache_workers,
        collate_fn=collate_dual_token_batch,
        pin_memory=True,
    )
    tl_chunks, rl_chunks, mh_chunks, rate_chunks = [], [], [], []
    autocast = (
        torch.amp.autocast("cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else torch.autocast("cpu", enabled=False)
    )
    with torch.no_grad(), autocast:
        for batch in loader:
            tokens, tokens_aux, mh, rate, _people, _bucket = batch
            tl, rl, _pl = model(
                tokens.to(device, non_blocking=True),
                tokens_aux.to(device, non_blocking=True),
            )
            tl_chunks.append(tl.float().cpu())
            rl_chunks.append(rl.float().cpu())
            mh_chunks.append(mh)
            rate_chunks.append(rate)
    model.cpu()
    return (
        torch.cat(tl_chunks),
        torch.cat(rl_chunks),
        torch.cat(mh_chunks),
        torch.cat(rate_chunks),
        device,
    )


# --------------------------------------------------------------------------- #
# External timm tagger.
# --------------------------------------------------------------------------- #


def _pad_square_white(im: Image.Image) -> Image.Image:
    w, h = im.size
    s = max(w, h)
    if w == h:
        return im
    canvas = Image.new("RGB", (s, s), (255, 255, 255))
    canvas.paste(im, ((s - w) // 2, (s - h) // 2))
    return canvas


def load_external(args, device: torch.device):
    import timm
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file as st_load

    repo = args.external_repo
    weights = hf_hub_download(repo, "model.safetensors")
    tags_csv = hf_hub_download(repo, "selected_tags.csv")
    meta_p = hf_hub_download(repo, "meta.json")
    with open(meta_p) as f:
        meta = json.load(f)
    margs = meta.get("model_args", {})
    kwargs = {}
    if "act_layer" in margs:
        kwargs["act_layer"] = margs["act_layer"]
    rows = list(csv.DictReader(open(tags_csv, newline="")))
    model = timm.create_model(
        args.external_arch, pretrained=False, num_classes=len(rows), **kwargs
    )
    sd = st_load(weights)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise SystemExit(
            f"external state_dict mismatch: missing={missing[:5]} unexpected={unexpected[:5]}"
        )
    model.to(device).eval()
    pcfg = model.pretrained_cfg
    mean = torch.tensor(pcfg.get("mean", (0.485, 0.456, 0.406))).view(1, 3, 1, 1)
    std = torch.tensor(pcfg.get("std", (0.229, 0.224, 0.225))).view(1, 3, 1, 1)
    return model, rows, mean, std


def collect_external(
    args, model, mean, std, image_paths: List[str], device: torch.device
) -> torch.Tensor:
    import numpy as np

    S = args.external_img_size
    probs: List[torch.Tensor] = []
    autocast = (
        torch.amp.autocast("cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else torch.autocast("cpu", enabled=False)
    )
    for i in range(0, len(image_paths), args.external_batch_size):
        chunk = image_paths[i : i + args.external_batch_size]
        arrs = []
        for p in chunk:
            im = Image.open(p).convert("RGB")
            im = _pad_square_white(im).resize((S, S), Image.BICUBIC)
            arrs.append(
                torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).permute(
                    2, 0, 1
                )
            )
        x = (torch.stack(arrs) - mean) / std
        with torch.no_grad(), autocast:
            out = model(x.to(device, non_blocking=True))
        probs.append(out.float().sigmoid().cpu())
        if (i // args.external_batch_size) % 20 == 0:
            log.info("external: %d/%d", i + len(chunk), len(image_paths))
    return torch.cat(probs)


# --------------------------------------------------------------------------- #
# Vocab alignment.
# --------------------------------------------------------------------------- #


def align_vocab(vocab: dict, ext_rows: List[dict], rename_recovery: Dict[str, str]):
    """Return ``(ours_idx, ext_idx, ext_rating_cols, unmatched_by_cat)``.

    External names are danbooru ``snake_case``; ours are space-separated
    (rules.yaml may have renamed some — try the recovered original too).
    """
    ext_by_name = {}
    ext_rating_cols: Dict[str, int] = {}
    for j, r in enumerate(ext_rows):
        cat = int(r["category"])
        name = r["name"].replace("_", " ")
        if cat == 9:
            our = RATING_MAP.get(r["name"])
            if our:
                ext_rating_cols[our] = j
            continue
        ext_by_name[name] = j
    ours_idx, ext_idx = [], []
    unmatched: Dict[str, int] = {}
    for t in vocab["tags"]:
        name = t["name"]
        j = ext_by_name.get(name)
        if j is None and name in rename_recovery:
            j = ext_by_name.get(rename_recovery[name])
        if j is None:
            unmatched[t["category"]] = unmatched.get(t["category"], 0) + 1
            continue
        ours_idx.append(t["index"])
        ext_idx.append(j)
    return torch.tensor(ours_idx), torch.tensor(ext_idx), ext_rating_cols, unmatched


# --------------------------------------------------------------------------- #
# Scoring.
# --------------------------------------------------------------------------- #


def _macro(x: torch.Tensor, mask: torch.Tensor) -> float:
    x = x[mask]
    x = x[~torch.isnan(x)]
    return float(x.mean()) if x.numel() else float("nan")


def score_block(
    pred: torch.Tensor, scores: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> dict:
    """micro/macro F1 + mAP over the tag columns selected by ``mask``."""
    from scripts.anima_tagger.eval_metrics import (
        micro_f1,
        per_tag_average_precision,
        per_tag_prf,
    )

    pred_m, scores_m, target_m = pred[:, mask], scores[:, mask], target[:, mask]
    _p, _r, f1, support = per_tag_prf(pred_m, target_m)
    ap = per_tag_average_precision(scores_m, target_m)
    sup = support > 0
    return {
        "n_tags": int(sup.sum()),
        "n_pos": int(target_m.sum()),
        "micro_f1": micro_f1(pred_m[:, sup], target_m[:, sup])
        if sup.any()
        else float("nan"),
        "macro_f1": _macro(f1, sup),
        "mean_ap": _macro(ap, sup),
    }


def _fmt(x) -> str:
    return "nan" if (isinstance(x, float) and math.isnan(x)) else f"{x:.4f}"


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    cfg_d = json.load(open(model_dir / "config.json"))
    vocab = json.load(open(model_dir / "vocab.json"))
    dataset = json.load(open(model_dir / "dataset.json"))
    split_stems: List[str] = dataset["split"][args.split]
    if args.limit:
        split_stems = split_stems[: args.limit]
    stem_set = set(split_stems)
    n_tags = int(dataset["n_tags"])

    # Ground truth in stem order for the external pass.
    rows = [
        (s, p, ti, ri)
        for s, p, ti, ri in zip(
            dataset["stems"],
            dataset["image_paths"],
            dataset["tag_indices"],
            dataset["rating_indices"],
        )
        if s in stem_set
    ]
    image_paths = [p for _, p, _, _ in rows]
    gt = torch.zeros(len(rows), n_tags)
    for n, (_, _, ti, _) in enumerate(rows):
        gt[n, ti] = 1.0
    gt_rating = torch.tensor([ri for _, _, _, ri in rows])
    log.info("split=%s N=%d n_tags=%d", args.split, len(rows), n_tags)

    from library.captioning import tag_rules as tr
    from scripts.anima_tagger.eval_metrics import predict_with_inference_rule
    from scripts.anima_tagger.train_common import GroupRouter

    rules_path = model_dir / "rules.yaml"
    rename_recovery: Dict[str, str] = {}
    if rules_path.exists():
        rename_recovery = {
            tgt: src for src, tgt in tr.load_rules(rules_path).replacements
        }

    # ---- ours ----
    tl, rl, mh, rate, device = collect_ours(args, model_dir, cfg_d, split_stems)
    if not torch.equal(mh.sum(0), gt.sum(0)):
        raise SystemExit(
            "loader multi_hot disagrees with dataset.json — split mismatch"
        )
    thr_path = model_dir / "thresholds.safetensors"
    if thr_path.exists():
        from safetensors.torch import load_file as st_load

        ours_thr = st_load(str(thr_path))["thresholds"].float()
        ours_thr_src = "calibrated-on-val"
    else:
        ours_thr = torch.full((n_tags,), 0.5)
        ours_thr_src = "flat-0.5"
    router = GroupRouter.from_vocab(vocab, mh, device=torch.device("cpu"))
    ours_pred = predict_with_inference_rule(tl, ours_thr, router)
    ours_pred_flat = tl.sigmoid() > args.external_flat_thr
    ours_scores = tl.sigmoid()
    ours_rating_acc = float((rl.argmax(1) == rate).float().mean())

    # ---- external ----
    ext_model, ext_rows, mean, std = load_external(args, device)
    ext_probs = collect_external(args, ext_model, mean, std, image_paths, device)
    ext_model.cpu()
    del ext_model
    ours_idx, ext_idx, ext_rating_cols, unmatched = align_vocab(
        vocab, ext_rows, rename_recovery
    )
    log.info(
        "intersection: %d / %d of our tags; unmatched by category: %s",
        len(ours_idx),
        n_tags,
        unmatched,
    )

    # Project external into OUR tag index space (zeros elsewhere).
    ext_scores = torch.zeros_like(gt)
    ext_scores[:, ours_idx] = ext_probs[:, ext_idx]
    ext_best_thr = torch.full((n_tags,), 2.0)  # never fires outside the intersection
    ext_best_thr[ours_idx] = torch.tensor(
        [float(ext_rows[j]["best_threshold"]) for j in ext_idx.tolist()]
    )
    ext_pred_best = ext_scores >= ext_best_thr
    ext_pred_flat = ext_scores >= args.external_flat_thr
    ext_pred_flat[:, ~torch.isin(torch.arange(n_tags), ours_idx)] = False
    ext_rating_names = vocab["ratings"]
    ext_rating_logits = torch.stack(
        [
            ext_probs[:, ext_rating_cols[r]]
            if r in ext_rating_cols
            else torch.zeros(len(rows))
            for r in ext_rating_names
        ],
        1,
    )
    ext_rating_acc = float((ext_rating_logits.argmax(1) == gt_rating).float().mean())

    # ---- slices over the intersection ----
    in_inter = torch.zeros(n_tags, dtype=torch.bool)
    in_inter[ours_idx] = True
    cats = [t["category"] for t in vocab["tags"]]
    freqs = torch.tensor([t["freq"] for t in vocab["tags"]])
    tiers = [
        "head"
        if f >= args.freq_head_min
        else "mid"
        if f >= args.freq_mid_min
        else "tail"
        for f in freqs.tolist()
    ]
    slices: Dict[str, torch.Tensor] = {"all": in_inter.clone()}
    for c in sorted(set(cats)):
        m = torch.tensor([x == c for x in cats]) & in_inter
        if m.any():
            slices[f"cat:{c}"] = m
    for t in ("head", "mid", "tail"):
        slices[f"freq:{t}"] = torch.tensor([x == t for x in tiers]) & in_inter

    # Coverage: how much of the val ground truth lives inside the intersection?
    pos_total = gt.sum()
    pos_inter = gt[:, in_inter].sum()

    metrics: Dict[str, object] = {
        "n_images": len(rows),
        "n_tags_ours": n_tags,
        "n_tags_external": len(ext_rows),
        "n_tags_intersection": int(in_inter.sum()),
        "unmatched_ours_by_category": unmatched,
        "gt_positive_coverage_intersection": float(pos_inter / pos_total),
        "ours_threshold_source": ours_thr_src,
        "rating_acc": {"ours": ours_rating_acc, "external": ext_rating_acc},
        "slices": {},
    }
    per_tag_rows = []
    from scripts.anima_tagger.eval_metrics import per_tag_average_precision, per_tag_prf

    # NB: ours is in loader-emission order, external in stem order — per-tag
    # metrics must each pair with the multi_hot from the SAME ordering.
    _, _, f1_o, sup = per_tag_prf(ours_pred, mh)
    _, _, f1_e, _ = per_tag_prf(ext_pred_best, gt)
    ap_o = per_tag_average_precision(ours_scores, mh)
    ap_e = per_tag_average_precision(ext_scores, gt)
    for name, m in slices.items():
        metrics["slices"][name] = {
            "ours@calib": score_block(ours_pred, ours_scores, mh, m),
            f"ours@{args.external_flat_thr}": score_block(
                ours_pred_flat, ours_scores, mh, m
            ),
            "external@best_thr": score_block(ext_pred_best, ext_scores, gt, m),
            f"external@{args.external_flat_thr}": score_block(
                ext_pred_flat, ext_scores, gt, m
            ),
        }
    for i in ours_idx.tolist():
        if sup[i] <= 0:
            continue
        per_tag_rows.append(
            {
                "tag": vocab["tags"][i]["name"],
                "category": cats[i],
                "freq_tier": tiers[i],
                "train_freq": int(freqs[i]),
                "val_support": int(sup[i]),
                "ap_ours": _fmt(float(ap_o[i])),
                "ap_external": _fmt(float(ap_e[i])),
                "f1_ours_calib": _fmt(float(f1_o[i])),
                "f1_external_best": _fmt(float(f1_e[i])),
            }
        )

    run_dir = make_run_dir("tagger_external", args.label)
    with open(run_dir / "per_tag.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_tag_rows[0].keys()))
        w.writeheader()
        w.writerows(per_tag_rows)

    # summary.md
    lines = [
        f"# {args.external_repo} vs {model_dir.name} — split={args.split}, N={len(rows)}",
        "",
        f"- intersection vocab: {int(in_inter.sum())} / {n_tags} of our tags "
        f"(unmatched by category: {unmatched}); covers {100 * float(pos_inter / pos_total):.1f}% of val GT positives",
        f"- ours thresholds: {ours_thr_src} (F1 optimistic for us); external: per-tag best_threshold from its own split",
        f"- rating acc (4-way): ours {ours_rating_acc:.3f} / external {ext_rating_acc:.3f}",
        "",
        "| slice | n_tags | n_pos | arm | micro-F1 | macro-F1 | mAP |",
        "|---|---:|---:|---|---:|---:|---:|",
    ]
    for name, arms in metrics["slices"].items():
        for arm, s in arms.items():
            lines.append(
                f"| {name} | {s['n_tags']} | {s['n_pos']} | {arm} | {_fmt(s['micro_f1'])} | {_fmt(s['macro_f1'])} | {_fmt(s['mean_ap'])} |"
            )
    # Biggest per-tag swings (AP), support >= 5.
    swing = [
        (r, float(r["ap_external"]) - float(r["ap_ours"]))
        for r in per_tag_rows
        if r["val_support"] >= 5 and r["ap_ours"] != "nan"
    ]
    swing.sort(key=lambda x: x[1])
    lines += [
        "",
        "## Largest per-tag AP swings (val support >= 5)",
        "",
        "external >> ours:",
    ]
    lines += [
        f"- {r['tag']} ({r['category']}, n={r['val_support']}): ext {r['ap_external']} vs ours {r['ap_ours']}"
        for r, _ in swing[-15:][::-1]
    ]
    lines += ["", "ours >> external:"]
    lines += [
        f"- {r['tag']} ({r['category']}, n={r['val_support']}): ext {r['ap_external']} vs ours {r['ap_ours']}"
        for r, _ in swing[:15]
    ]
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics=metrics,
        label=args.label,
        artifacts=["per_tag.csv", "summary.md"],
        device=device,
    )
    log.info("wrote %s", run_dir)


if __name__ == "__main__":
    main()
