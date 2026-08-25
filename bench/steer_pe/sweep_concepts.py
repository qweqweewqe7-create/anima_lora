#!/usr/bin/env python3
"""Tier A.2 — caption-gated SAM3 concept sweep → (prompt, mask) manifest.

Loads SAM3 once and, per image, grounds every concept from
``bench/steer_pe/concepts.yaml`` whose booru tag is in the caption (plus the
``always`` prompts). Every instance SAM3 returns is OR-ed into one mask — one
prompt may light every instance — and saved as
``<out>/masks/<artist>/<stem>__<tag>_mask.png``. A concept in the caption that
SAM3 cannot ground becomes a ``negative`` row (kind ``c_nodet``) — SAM3's own
abstain; a sampled ``absent_ok`` concept missing from the caption becomes a
``negative`` row without any GPU work (kind ``c_neg``).

Prompt style mirrors ``dump_instance_pairs.py``: booru (``skirt``) and natural
(``the skirt``) rows for the same mask (``--style``).

    make daemon-run ARGS="bench/steer_pe/sweep_concepts.py"
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

if not hasattr(np, "bool"):  # sam3 upstream pins numpy<2
    np.bool = np.bool_

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from library.captioning.position_clauses import parse_caption  # noqa: E402
from library.env import resolve_under_home  # noqa: E402


def _rel(path: Path) -> str:
    """Repo-relative when inside the repo, else absolute (manifest portability)."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


from library.preprocess._dataset import walk_images  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="post_image_dataset/steer_pe/concepts")
    p.add_argument("--concepts", default="bench/steer_pe/concepts.yaml")
    p.add_argument("--resized_dir", default="post_image_dataset/resized")
    p.add_argument("--src", default="image_dataset", help="caption fallback")
    p.add_argument(
        "--path_pattern", "--path-pattern", dest="path_pattern", default=None
    )
    p.add_argument("--checkpoint", default="models/sam3/sam3.pt")
    p.add_argument("--device", default="cuda")
    p.add_argument("--style", default="both", choices=["both", "booru", "natural"])
    p.add_argument("--neg_per_image", type=int, default=2)
    p.add_argument("--max_concepts_per_image", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--force", action="store_true", help="re-run SAM3 on existing masks")
    return p.parse_args()


def prompt_rows(tag: str, phrase: str, style: str) -> list[str]:
    out = []
    if style in ("both", "booru"):
        out.append(tag)
    if style in ("both", "natural"):
        art = "the "
        out.append(f"{art}{phrase}")
    return out


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    cfg = yaml.safe_load(resolve_under_home(args.concepts).read_text())
    concepts: dict[str, str] = cfg["concepts"]
    always: dict[str, str] = cfg.get("always", {})
    absent_ok = [c for c in cfg.get("absent_ok", []) if c in concepts]
    threshold = float(cfg.get("threshold", 0.4))
    out_dir = resolve_under_home(args.out)
    masks_dir = out_dir / "masks"
    resized = resolve_under_home(args.resized_dir)
    src = resolve_under_home(args.src)

    import torch
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    print("Loading SAM3...", flush=True)
    model = build_sam3_image_model(
        device=args.device,
        eval_mode=True,
        checkpoint_path=str(resolve_under_home(args.checkpoint)),
        load_from_HF=False,
    )
    processor = Sam3Processor(model, confidence_threshold=threshold)

    images = walk_images(resized, recursive=True, pattern=args.path_pattern)
    if args.limit:
        images = images[: args.limit]
    rows: list[dict] = []
    manifest = out_dir / "manifest.jsonl"
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n_calls = n_det = n_nodet = 0
    for index, image_path in enumerate(images, 1):
        rel = image_path.relative_to(resized)
        cap_path = image_path.with_suffix(".txt")
        if not cap_path.exists():
            cap_path = src / rel.with_suffix(".txt")
        if not cap_path.exists():
            continue
        caption = cap_path.read_text(encoding="utf-8").strip()
        bag = set(parse_caption(caption).tag_keys)
        present = [t for t in concepts if t in bag]
        rng.shuffle(present)
        present = present[: args.max_concepts_per_image]
        todo = {t: concepts[t] for t in present} | dict(always)
        artist = rel.parts[0]
        stem = rel.stem
        tdir = masks_dir / rel.parent
        tdir.mkdir(parents=True, exist_ok=True)
        base = {"image": _rel(image_path), "artist": artist}
        state = None
        image = None
        for tag, phrase in todo.items():
            slug = tag.replace(" ", "_")
            mp = tdir / f"{stem}__{slug}_mask.png"
            flag = tdir / f"{stem}__{slug}_nodet"
            if not args.force and (mp.exists() or flag.exists()):
                found = mp.exists()
            else:
                if state is None:
                    with Image.open(image_path) as h:
                        image = h.convert("RGB")
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        state = processor.set_image(image)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = processor.set_text_prompt(prompt=phrase, state=state)
                n_calls += 1
                w, h = image.size
                union = np.zeros((h, w), dtype=bool)
                for m, s in zip(out.get("masks", []), out["scores"]):
                    if float(s) < threshold:
                        continue
                    mm = m.cpu().numpy() if torch.is_tensor(m) else np.asarray(m)
                    while mm.ndim > 2:
                        mm = mm[0]
                    union |= mm > 0.5
                found = bool(union.any())
                if found:
                    Image.fromarray((union * 255).astype(np.uint8)).save(mp)
                else:
                    flag.touch()
            kind = f"c:{tag}"
            for prompt in prompt_rows(tag, phrase, args.style):
                if found:
                    n_det += 1
                    rows.append(
                        {
                            **base,
                            "mask": _rel(mp),
                            "prompt": prompt,
                            "kind": kind,
                            "key": f"{rel.parent}/{stem}#{slug}",
                        }
                    )
                else:
                    n_nodet += 1
                    rows.append(
                        {
                            **base,
                            "mask": None,
                            "negative": True,
                            "prompt": prompt,
                            "kind": "c_nodet",
                            "key": f"{rel.parent}/{stem}#{slug}",
                        }
                    )
        absent = [c for c in absent_ok if c not in bag]
        for tag in rng.sample(absent, min(args.neg_per_image, len(absent))):
            for prompt in prompt_rows(tag, concepts[tag], args.style):
                rows.append(
                    {
                        **base,
                        "mask": None,
                        "negative": True,
                        "prompt": prompt,
                        "kind": "c_neg",
                        "key": f"{rel.parent}/{stem}#neg:{tag}",
                    }
                )
        if index % 50 == 0 or index == len(images):
            print(
                f"  [{index}/{len(images)}] sam_calls={n_calls} det={n_det} "
                f"nodet={n_nodet} rows={len(rows)} {time.time() - t0:.0f}s",
                flush=True,
            )
            with open(manifest, "w", encoding="utf-8") as fh:
                for r in rows:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(manifest, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    kinds: dict[str, int] = {}
    for r in rows:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    print(f"wrote {len(rows)} rows → {manifest}", flush=True)
    print(json.dumps(dict(sorted(kinds.items(), key=lambda kv: -kv[1])), indent=1))


if __name__ == "__main__":
    main()
