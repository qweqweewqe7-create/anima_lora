#!/usr/bin/env python3
"""Stage the multi-girl slice of the tagger crawl pool as a resized PNG + txt
mirror (``<out>/<artist>/<stem>.png|.txt``) so ``dump_instance_pairs.py
--mode multi --dst <out>`` can walk it like ``post_image_dataset/resized``.

CPU only; idempotent (skips existing PNGs)."""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from library.env import resolve_under_home  # noqa: E402

MULTI = ("2girls", "3girls", "4girls", "5girls", "6+girls", "multiple girls")
EXTS = (".webp", ".jpg", ".jpeg", ".png")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", default="/media/sorryhyun/새 볼륨/dataset/retrieved")
    p.add_argument("--out", default="post_image_dataset/steer_pe/retrieved_multi")
    p.add_argument("--max_edge", type=int, default=1024)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    src, out = Path(args.src), resolve_under_home(args.out)
    jobs = []
    for txt in sorted(src.glob("*/*.txt")):
        cap = txt.read_text(encoding="utf-8")
        if not any(k in cap for k in MULTI) or "multiple views" in cap:
            continue
        img = next(
            (txt.with_suffix(e) for e in EXTS if txt.with_suffix(e).exists()), None
        )
        if img is None:
            continue
        jobs.append((img, cap, out / txt.parent.name / f"{txt.stem}.png"))

    def one(job):
        img, cap, dst = job
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.with_suffix(".txt").write_text(cap.strip(), encoding="utf-8")
        if dst.exists():
            return
        with Image.open(img) as h:
            im = h.convert("RGB")
        s = args.max_edge / max(im.size)
        if s < 1:
            im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
        im.save(dst)

    with ThreadPoolExecutor(args.workers) as ex:
        for i, _ in enumerate(ex.map(one, jobs), 1):
            if i % 200 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)
    print(f"staged {len(jobs)} → {out}", flush=True)


if __name__ == "__main__":
    main()
