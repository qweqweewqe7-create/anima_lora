"""Where does SAM3 fail on anime-segmentation's real GT — and does SteerPE cover it?

For every ``imgs/*.jpg`` + ``masks/*.jpg`` pair of skytnt/anime-segmentation:

* SAM3 (region-prep defaults: threshold 0.4) with each of ``--prompts``; per prompt
  the union mask's IoU vs GT and the box count. ``sam_best`` = max IoU over prompts.
* SteerPE (a trained adapter, default the Phase-3 control) under ``--steer_prompt``:
  patch PR-AUC vs GT, plus IoU of the sigmoid map thresholded at 0.5 (upsampled).

Failures = ``sam_best < --fail_iou`` (zero-box counts as IoU 0). The result
records the failure rate per prompt, SteerPE's score on the failure set vs the
rest, and a sheet of the worst SAM3 cases (GT | SAM3 best | SteerPE).

    make daemon-run ARGS="bench/steer_pe/sam3_gap.py --label gap"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

if not hasattr(np, "bool"):
    np.bool = np.bool_  # sam3 pins numpy<2

import torch
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bench._common import make_run_dir, start_heartbeat, write_result  # noqa: E402
from bench.steer_pe.run_bench import PromptBank, build, heat  # noqa: E402
from library.env import resolve_under_home  # noqa: E402
from networks.methods.steer_pe import patch_targets, pr_auc  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--label", default=None)
    p.add_argument(
        "--anime_seg", default="/media/sorryhyun/새 볼륨/dataset/anime_segmentation"
    )
    p.add_argument("--limit", type=int, default=0, help="0 = all 1 111")
    p.add_argument("--prompts", default="girl,person,anime character")
    p.add_argument("--threshold", type=float, default=0.4)
    p.add_argument("--checkpoint", default="models/sam3/sam3.pt")
    p.add_argument("--fail_iou", type=float, default=0.5)
    p.add_argument(
        "--adapter",
        default="bench/steer_pe/results/20260825-1752-p3-ctrl/steer_pe_adapter.safetensors",
    )
    p.add_argument("--steer_prompt", default="a person")
    p.add_argument("--pe", default="pe_spatial", choices=["pe_spatial", "pe_core"])
    p.add_argument(
        "--qwen3", default="models/text_encoders/qwen_3_06b_base.safetensors"
    )
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--res", type=int, default=512)
    p.add_argument("--sheet_n", type=int, default=24)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else float("nan")


def to_tensor(img: Image.Image, res: int) -> torch.Tensor:
    x = np.asarray(img.resize((res, res), Image.BICUBIC), dtype=np.float32)
    return torch.from_numpy(x / 127.5 - 1.0).permute(2, 0, 1)


def tile(img: Image.Image, mask: np.ndarray, title: str, size: int) -> Image.Image:
    im = img.resize((size, size), Image.BICUBIC).convert("RGB")
    arr = np.asarray(im).astype(np.float32)
    m = (
        np.asarray(
            Image.fromarray((mask * 255).astype(np.uint8)).resize(
                (size, size), Image.BILINEAR
            )
        )
        > 127
    )
    arr[m] = arr[m] * 0.45 + np.array([255, 40, 40], np.float32) * 0.55
    out = Image.fromarray(arr.clip(0, 255).astype(np.uint8))
    ImageDraw.Draw(out).text((4, 4), title, fill=(255, 255, 0))
    return out


def main() -> None:
    args = parse_args()
    run_dir = make_run_dir("steer_pe", label=args.label or "sam3_gap")
    start_heartbeat(label="sam3_gap")
    dev = torch.device(args.device)
    root = Path(args.anime_seg)
    prompts = [p.strip() for p in args.prompts.split(",") if p.strip()]
    imgs = sorted((root / "imgs").glob("*.jpg"))
    if args.limit:
        imgs = imgs[: args.limit]

    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    sam = build_sam3_image_model(
        device=args.device,
        eval_mode=True,
        checkpoint_path=str(resolve_under_home(args.checkpoint)),
        load_from_HF=False,
    )
    proc = Sam3Processor(sam)

    model, te, tok = build(args)
    from safetensors.torch import load_file

    model.load_adapter_state_dict(load_file(str(resolve_under_home(args.adapter))))
    model.eval()
    bank = PromptBank(te, tok, dev)

    rows: list[dict] = []
    cache: dict[str, dict] = {}
    for i, path in enumerate(imgs):
        img = Image.open(path).convert("RGB")
        gt = np.asarray(Image.open(root / "masks" / path.name).convert("L")) > 127
        if gt.shape != (img.height, img.width):
            gt = (
                np.asarray(
                    Image.fromarray(gt.astype(np.uint8) * 255).resize(
                        img.size, Image.NEAREST
                    )
                )
                > 127
            )
        rec: dict = {"name": path.stem, "gt_frac": float(gt.mean()), "sam": {}}
        best, best_mask, best_prompt = -1.0, np.zeros_like(gt), None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            state = proc.set_image(img)
            for pr in prompts:
                out = proc.set_text_prompt(state=state, prompt=pr)
                union = np.zeros_like(gt)
                n = 0
                for m, s in zip(out["masks"], out["scores"]):
                    if float(s) < args.threshold:
                        continue
                    n += 1
                    mn = (
                        m.float().cpu().numpy() if torch.is_tensor(m) else np.asarray(m)
                    )
                    mn = mn[0] if mn.ndim == 3 else mn
                    if mn.shape != gt.shape:
                        mn = (
                            np.asarray(
                                Image.fromarray(
                                    (mn > 0.5).astype(np.uint8) * 255
                                ).resize(img.size, Image.NEAREST)
                            )
                            / 255.0
                        )
                    union |= mn > 0.5
                v = iou(union, gt) if n else 0.0
                rec["sam"][pr] = {"n": n, "iou": v}
                if v > best:
                    best, best_mask, best_prompt = v, union, pr
        rec["sam_best"] = best
        rec["sam_best_prompt"] = best_prompt

        x = to_tensor(img, args.res)[None].to(dev)
        gt_t = torch.from_numpy(gt.astype(np.float32))[None, None].to(dev)
        gt_t = torch.nn.functional.interpolate(
            gt_t, size=(args.res, args.res), mode="bilinear"
        )
        target = patch_targets(gt_t, model.grid(x))[0] > 0.5
        logits = heat(model, bank, x, [args.steer_prompt])[0]
        rec["steer_pr_auc"] = pr_auc(logits, target)
        prob = (
            torch.nn.functional.interpolate(
                logits.sigmoid()[None, None], size=gt.shape, mode="bilinear"
            )[0, 0]
            .cpu()
            .numpy()
        )
        rec["steer_iou"] = iou(prob > 0.5, gt)
        rows.append(rec)
        small = lambda m: (
            np.asarray(  # noqa: E731 — 256² thumbnails only, not full-res
                Image.fromarray(m.astype(np.uint8) * 255).resize(
                    (256, 256), Image.BILINEAR
                )
            )
            > 127
        )
        cache[path.stem] = {"best": small(best_mask), "steer": small(prob > 0.5)}
        if i % 50 == 0:
            print(
                f"[{i}/{len(imgs)}] {path.stem} sam_best={best:.2f} ({best_prompt}) "
                f"steer_auc={rec['steer_pr_auc']:.2f} steer_iou={rec['steer_iou']:.2f}",
                flush=True,
            )

    fail = [r for r in rows if r["sam_best"] < args.fail_iou]
    ok = [r for r in rows if r["sam_best"] >= args.fail_iou]

    def mean(rs, k):
        return float(np.nanmean([r[k] for r in rs])) if rs else None

    metrics = {
        "n": len(rows),
        "prompts": prompts,
        "threshold": args.threshold,
        "fail_iou": args.fail_iou,
        "per_prompt": {
            pr: {
                "zero_box": sum(r["sam"][pr]["n"] == 0 for r in rows),
                "fail": sum(r["sam"][pr]["iou"] < args.fail_iou for r in rows),
                "mean_iou": float(np.mean([r["sam"][pr]["iou"] for r in rows])),
            }
            for pr in prompts
        },
        "sam_best_mean_iou": mean(rows, "sam_best"),
        "sam_best_zero_box": sum(
            all(r["sam"][pr]["n"] == 0 for pr in prompts) for r in rows
        ),
        "fail_n": len(fail),
        "fail_rate": len(fail) / max(1, len(rows)),
        "steer_on_fail": {
            "pr_auc": mean(fail, "steer_pr_auc"),
            "iou": mean(fail, "steer_iou"),
            "iou_over_sam": mean(
                [
                    {"d": r["steer_iou"] - r["sam_best"]}
                    for r in fail
                    if not np.isnan(r["steer_iou"])
                ],
                "d",
            ),
            "steer_beats_sam": sum(r["steer_iou"] > r["sam_best"] for r in fail),
        },
        "steer_on_ok": {
            "pr_auc": mean(ok, "steer_pr_auc"),
            "iou": mean(ok, "steer_iou"),
        },
        "steer_all": {
            "pr_auc": mean(rows, "steer_pr_auc"),
            "iou": mean(rows, "steer_iou"),
        },
        "worst": [
            {
                k: r[k]
                for k in ("name", "sam_best", "sam_best_prompt", "steer_iou", "gt_frac")
            }
            for r in sorted(rows, key=lambda r: r["sam_best"])[: args.sheet_n]
        ],
    }
    print(json.dumps({k: v for k, v in metrics.items() if k != "worst"}, indent=1))
    (run_dir / "rows.json").write_text(json.dumps(rows, indent=1))

    worst = sorted(rows, key=lambda r: r["sam_best"])[: args.sheet_n]
    S = 256
    sheet = Image.new("RGB", (3 * S, len(worst) * S), "black")
    for j, r in enumerate(worst):
        img = Image.open(root / "imgs" / f"{r['name']}.jpg").convert("RGB")
        gt = (
            np.asarray(Image.open(root / "masks" / f"{r['name']}.jpg").convert("L"))
            > 127
        )
        c = cache[r["name"]]
        sheet.paste(tile(img, gt, f"{r['name']} GT", S), (0, j * S))
        sheet.paste(
            tile(
                img,
                c["best"],
                f"SAM3 {r['sam_best_prompt']} iou={r['sam_best']:.2f}",
                S,
            ),
            (S, j * S),
        )
        sheet.paste(
            tile(img, c["steer"], f"SteerPE iou={r['steer_iou']:.2f}", S),
            (2 * S, j * S),
        )
    sheet.save(run_dir / "worst_sheet.png")
    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics=metrics,
        label=args.label or "sam3_gap",
        artifacts=["rows.json", "worst_sheet.png"],
    )
    print(f"run_dir: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
