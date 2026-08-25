#!/usr/bin/env python3
"""Tier A.1 / A.3 — attribute- and name-bound (prompt, instance mask) pairs.

Two sources, one JSONL manifest for ``run_bench.py --pairs_manifest``:

* **multi** (GPU: SAM3 + Anima Tagger) — runs the position-caption pipeline's
  ``propose_for_image`` on every multi-subject image and dumps, per bound
  instance, its SAM3 mask together with a prompt composed from the identity
  tags the crop tagger attributed to it ("the girl with black hair and red
  eyes"). The union of the *other* instances' masks rides along as
  ``rival_mask`` so the bench can score the wrong-instance share. Only images
  whose proposal passed every gate (``status == "proposed"``) are used, so the
  known SAM3 group-mask failure never becomes a target.
* **solo** (text only, no GPU) — for ``solo`` images that already carry a
  Phase-0 girl mask, binds the caption's own hair / eye colour and character
  name to that mask. This is where character names (A.3) mostly come from.

Both add **abstain negatives**: a hair / eye colour (or character name) the
caption never mentions → ``negative: true`` (empty target).

Prompt style: ``booru`` ("1girl, black hair, red eyes"), ``natural`` ("the girl
with black hair and red eyes") or ``both`` (default — same mask, two rows).

    make daemon-run ARGS="bench/steer_pe/dump_instance_pairs.py --mode all"

Any flag this script doesn't know is forwarded to
``scripts/preprocess/position_captions.py``'s parser (detection / clause knobs).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from library.captioning.position_clauses import (  # noqa: E402
    flatten_caption,
    parse_caption,
)
from library.env import resolve_under_home  # noqa: E402


def _rel(path: Path) -> str:
    """Repo-relative when inside the repo, else absolute (manifest portability)."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


HAIR_COLOURS = [
    "black",
    "brown",
    "blonde",
    "blue",
    "white",
    "silver",
    "grey",
    "red",
    "pink",
    "purple",
    "green",
    "orange",
    "aqua",
]
EYE_COLOURS = [
    "blue",
    "brown",
    "red",
    "purple",
    "yellow",
    "green",
    "pink",
    "grey",
    "orange",
    "aqua",
    "black",
]
HAIR_RE = re.compile(r"^([a-z]+) hair$")
EYE_RE = re.compile(r"^([a-z]+) eyes$")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="post_image_dataset/steer_pe/instances")
    p.add_argument("--mode", default="all", choices=["all", "multi", "solo"])
    p.add_argument("--style", default="both", choices=["both", "booru", "natural"])
    p.add_argument("--region_base", default="post_image_dataset/easycontrol/region")
    p.add_argument(
        "--caption_index", default="post_image_dataset/captions/caption_index.json"
    )
    p.add_argument("--neg_per_image", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=0, help="debug: stop after N images")
    own, rest = p.parse_known_args()
    return own, rest


def _load_pc_cli():
    """The position-caption CLI module (its SAM3 / tagger factories)."""
    path = REPO_ROOT / "scripts/preprocess/position_captions.py"
    spec = importlib.util.spec_from_file_location("pc_cli", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


# ── prompt composition ──────────────────────────────────────────────────────


_NAME_SET: set[str] | None = None


def _names() -> set[str]:
    """Character tags known to the caption index (no-paren names included)."""
    global _NAME_SET
    if _NAME_SET is None:
        try:
            idx = json.loads(
                resolve_under_home(
                    "post_image_dataset/captions/caption_index.json"
                ).read_text()
            )
            _NAME_SET = set(idx["groups"]["character"])
        except (OSError, KeyError, ValueError):
            _NAME_SET = set()
    return _NAME_SET


def prompts_for(tags: list[str], style: str, subject: str = "girl") -> list[str]:
    tags = [t.strip().lower() for t in tags if t.strip()]
    if not tags:
        return []
    out = []
    if style in ("both", "booru"):
        out.append(f"1{subject}, " + ", ".join(tags))
    if style in ("both", "natural"):
        known = _names()
        names = [t for t in tags if "(" in t or t in known]
        attrs = [t for t in tags if t not in names]
        text = f"the {subject}"
        if names:
            text += " " + names[0]
        if attrs:
            text += " with " + " and ".join(attrs)
        out.append(text)
    return out


def caption_identity(caption: str) -> tuple[list[str], list[str]]:
    """(hair colour tags, eye colour tags) found in the flat bag of a caption."""
    bag = parse_caption(caption).tag_keys
    hair = [t for t in bag if (m := HAIR_RE.match(t)) and m.group(1) in HAIR_COLOURS]
    eyes = [t for t in bag if (m := EYE_RE.match(t)) and m.group(1) in EYE_COLOURS]
    return hair, eyes


def absent_identity(caption: str, rng: random.Random) -> list[str]:
    """A hair and an eye colour the caption does not mention (abstain probes)."""
    text = caption.lower()
    hair = [c for c in HAIR_COLOURS if f"{c} hair" not in text]
    eyes = [c for c in EYE_COLOURS if f"{c} eyes" not in text]
    out = []
    if hair:
        out.append(f"{rng.choice(hair)} hair")
    if eyes:
        out.append(f"{rng.choice(eyes)} eyes")
    return out


# ── multi-subject (GPU) ──────────────────────────────────────────────────────


def run_multi(own, rest, out_dir: Path, rows: list[dict], rng: random.Random):
    import torch

    from library.preprocess.position_captions import (
        PositionCaptionStats,
        _iter_captions,
        is_candidate,
        load_clause_vocabulary,
        propose_for_image,
    )

    pc = _load_pc_cli()
    sys.argv = [sys.argv[0], *rest]
    pc_args = pc.parse_args()
    detect_fn, part_detect_fn, sam_model, sam_processor = pc.build_detect_fn(pc_args)

    # Stash every detection the pipeline sees for the current image so the
    # bound instances (matched by their int box) can be paired with masks.
    stash: dict = {"image": None, "dets": []}

    def _wrap(fn, with_prompt: bool):
        def inner(image, *a):
            dets = fn(image, *a)
            if stash["image"] is not image:
                stash["image"], stash["dets"] = image, []
            stash["dets"].extend(dets)
            return dets

        return inner

    detect_w = _wrap(detect_fn, False)
    part_w = _wrap(part_detect_fn, True)

    from library.captioning.anima_tagger import (
        DEFAULT_TAGGER_DIR,
        AnimaTagger,
        ensure_tagger_checkpoint,
    )

    ckpt_dir = ensure_tagger_checkpoint(
        resolve_under_home(pc_args.tagger_dir or DEFAULT_TAGGER_DIR)
    )
    tagger = AnimaTagger(ckpt_dir, device=pc_args.device)
    vocabulary = load_clause_vocabulary(ckpt_dir)
    options = pc.build_options_from_args(pc_args)

    resized = resolve_under_home(pc_args.dst)
    src = resolve_under_home(pc_args.src)
    stats = PositionCaptionStats()
    masks_dir = out_dir / "masks"
    n_img = n_inst = 0
    t0 = time.time()
    for index, (image_path, rel, _dst, caption) in enumerate(
        _iter_captions(resized, src, pc_args.path_pattern, stats), 1
    ):
        if own.limit and n_img >= own.limit:
            break
        if index % 50 == 0:
            print(
                f"  [multi {index}] imgs={n_img} inst={n_inst} {time.time() - t0:.0f}s",
                flush=True,
            )
        # The corpus already carries applied position clauses (2026-08);
        # is_candidate skips those, and the pipeline expects a flat bag.
        caption = flatten_caption(caption)
        ok, _ = is_candidate(caption)
        if not ok:
            continue
        with Image.open(image_path) as h:
            image = h.convert("RGB")
        stash["image"], stash["dets"] = None, []
        prop = propose_for_image(
            image,
            caption,
            detect_fn=detect_w,
            tag_fn=tagger.predict,
            vocabulary=vocabulary,
            options=options,
            part_detect_fn=part_w,
        )
        if not prop.ok:
            continue
        by_box = {}
        for d in stash["dets"]:
            by_box.setdefault(tuple(int(v) for v in d.box), d)
        inst_masks = []
        for inst in prop.instances:
            d = by_box.get(tuple(inst.box))
            if d is None or d.mask is None:
                inst_masks.append(None)
                continue
            m = np.asarray(d.mask)
            while m.ndim > 2:
                m = m[0]
            inst_masks.append(m > 0.5)
        if sum(m is not None for m in inst_masks) < 2:
            continue
        artist = rel.parts[0]
        stem = rel.stem
        tdir = masks_dir / rel.parent
        tdir.mkdir(parents=True, exist_ok=True)
        n_img += 1
        wrote_any = False
        for i, (inst, m) in enumerate(zip(prop.instances, inst_masks)):
            if m is None or not inst.tags:
                continue
            rival = np.zeros_like(m)
            for j, mm in enumerate(inst_masks):
                if j != i and mm is not None:
                    rival |= mm
            mp = tdir / f"{stem}_{i}_mask.png"
            rp = tdir / f"{stem}_{i}_rival_mask.png"
            Image.fromarray((m * 255).astype(np.uint8)).save(mp)
            Image.fromarray((rival * 255).astype(np.uint8)).save(rp)
            for prompt in prompts_for(inst.tags, own.style):
                rows.append(
                    {
                        "image": _rel(image_path),
                        "mask": _rel(mp),
                        "rival_mask": _rel(rp),
                        "prompt": prompt,
                        "kind": "attr_multi",
                        "artist": artist,
                        "key": f"{rel.parent}/{stem}#{i}",
                        "tags": inst.tags,
                        "position": inst.position,
                    }
                )
            n_inst += 1
            wrote_any = True
        if wrote_any:
            for _ in range(own.neg_per_image):
                absent = absent_identity(caption, rng)
                for prompt in prompts_for(absent[:1], own.style):
                    rows.append(
                        {
                            "image": _rel(image_path),
                            "mask": None,
                            "negative": True,
                            "prompt": prompt,
                            "kind": "attr_neg",
                            "artist": artist,
                            "key": f"{rel.parent}/{stem}#neg",
                        }
                    )
    del sam_processor, sam_model, tagger
    torch.cuda.empty_cache()
    print(f"multi: images={n_img} instances={n_inst}", flush=True)


# ── solo (text only) ─────────────────────────────────────────────────────────


def run_solo(own, out_dir: Path, rows: list[dict], rng: random.Random):
    region = resolve_under_home(own.region_base)
    resized = resolve_under_home("post_image_dataset/resized")
    idx = json.loads(resolve_under_home(own.caption_index).read_text())
    meta = idx["image_meta"]
    all_names = sorted(idx["groups"]["character"])
    n = n_name = 0
    for key, m in sorted(meta.items()):
        if own.limit and n >= own.limit:
            break
        txt = resized / m["path"]
        img = txt.with_suffix(".png")
        mask = region / "masks" / f"{key}_mask.png"
        if not (txt.exists() and img.exists() and mask.exists()):
            continue
        caption = txt.read_text(encoding="utf-8").strip()
        if "solo" not in parse_caption(caption).tag_keys:
            continue
        if "1boy" in caption or "multiple views" in caption:
            continue
        hair, eyes = caption_identity(caption)
        names = [c for c in m.get("character", []) if c]
        artist = key.split("/")[0]
        base = {
            "image": _rel(img),
            "mask": _rel(mask),
            "artist": artist,
            "key": key,
        }
        attrs = hair[:1] + eyes[:1]
        if attrs:
            for prompt in prompts_for(attrs, own.style):
                rows.append(
                    {**base, "prompt": prompt, "kind": "attr_solo", "tags": attrs}
                )
            n += 1
        if len(names) == 1:
            for prompt in prompts_for(names, own.style):
                rows.append({**base, "prompt": prompt, "kind": "name", "tags": names})
            n_name += 1
            other = rng.choice([c for c in all_names if c not in names])
            for prompt in prompts_for([other], own.style):
                rows.append(
                    {
                        **base,
                        "mask": None,
                        "negative": True,
                        "prompt": prompt,
                        "kind": "name_neg",
                    }
                )
        if attrs:
            absent = absent_identity(caption, rng)
            for prompt in prompts_for(absent[:1], own.style):
                rows.append(
                    {
                        **base,
                        "mask": None,
                        "negative": True,
                        "prompt": prompt,
                        "kind": "attr_neg",
                    }
                )
    print(f"solo: attr images={n} named={n_name}", flush=True)


def main():
    own, rest = parse_args()
    rng = random.Random(own.seed)
    out_dir = resolve_under_home(own.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    if own.mode in ("all", "solo"):
        run_solo(own, out_dir, rows, rng)
    if own.mode in ("all", "multi"):
        run_multi(own, rest, out_dir, rows, rng)
    manifest = out_dir / "manifest.jsonl"
    with open(manifest, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    kinds: dict[str, int] = {}
    for r in rows:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    print(f"wrote {len(rows)} rows → {manifest}  {kinds}", flush=True)


if __name__ == "__main__":
    main()
