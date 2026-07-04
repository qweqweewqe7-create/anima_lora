#!/usr/bin/env python3
"""Emit the training commands for the uncond-soup bench (see README.md).

Seven runs total with the defaults (1 pool pre-train + 2 families x N=3
paired-seed fine-tunes). This script does NOT launch anything — it prints the
exact ``python train.py`` command lines (and optionally writes them to a shell
script) so they can be run inline, or queued on the daemon one at a time.

The two families share the same fine-tuning seeds (paired draws: identical
data order + FM training noise per seed), so the only difference between
``self_s<k>`` and ``base_s<k>`` is the unconditional inter-trained
initialization. That pairing is what lets the probe attribute soup-level
deltas to the uncond phase rather than to seed luck.

Usage
-----
    python bench/uncond_soup/launch.py \
        --pool_artists artistA artistB artistC \
        [--target_artist artistA] [--n_seeds 3] [--seed_base 1000] \
        [--write runs.sh]
"""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from library.datasets.artist_shard import _pattern_for, list_artists  # noqa: E402

PREFIX = "uncondsoup"


def build_commands(args) -> list[tuple[str, list[str]]]:
    """[(label, argv), ...] for every training run in the plan."""
    common = [
        "python",
        "train.py",
        "--method",
        args.method,
        "--preset",
        args.preset,
        "--validation_split_num",
        str(args.validation_split_num),
        "--validation_seed",
        str(args.validation_seed),
    ]
    pool_pattern = _pattern_for(args.pool_artists)
    target_pattern = _pattern_for([args.target_artist])

    cmds: list[tuple[str, list[str]]] = []

    # Phase A — unconditional inter-train on the 3-artist pool. Captions are
    # dropped 100% of the time; dropped rows get the T5("") sidecar (staged
    # automatically by train.py when caption dropout is on), so the adapter
    # sees the same uncond embedding CFG-uncond inference feeds.
    pool = common + [
        "--path_pattern",
        pool_pattern,
        "--caption_dropout_rate",
        "1.0",
        "--seed",
        str(args.seed_base),
        "--output_name",
        f"{PREFIX}_pool",
    ]
    if args.pool_epochs is not None:
        pool += ["--max_train_epochs", str(args.pool_epochs)]
    cmds.append(("pool (uncond inter-train)", pool))

    pool_ckpt = f"output/ckpt/{PREFIX}_pool.safetensors"
    for k in range(1, args.n_seeds + 1):
        seed = args.seed_base + k
        ft_common = common + ["--path_pattern", target_pattern, "--seed", str(seed)]
        if args.ft_epochs is not None:
            ft_common += ["--max_train_epochs", str(args.ft_epochs)]
        cmds.append(
            (
                f"self ingredient s{seed}",
                ft_common
                + [
                    "--network_weights",
                    pool_ckpt,
                    "--output_name",
                    f"{PREFIX}_self_s{seed}",
                ],
            )
        )
        cmds.append(
            (
                f"base ingredient s{seed}",
                ft_common + ["--output_name", f"{PREFIX}_base_s{seed}"],
            )
        )
    return cmds


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--pool_artists",
        nargs=3,
        required=True,
        metavar="ARTIST",
        help="Exactly 3 artist subdirectory names — the bounded unlabeled pool.",
    )
    ap.add_argument(
        "--target_artist",
        default=None,
        help="Supervised fine-tune artist (default: first of --pool_artists).",
    )
    ap.add_argument("--n_seeds", type=int, default=3, help="Ingredients per family.")
    ap.add_argument("--seed_base", type=int, default=1000)
    ap.add_argument("--method", default="lora")
    ap.add_argument("--preset", default="default")
    ap.add_argument(
        "--validation_split_num",
        type=int,
        default=16,
        help="Held-out split size — the probe MUST be run with the same value.",
    )
    ap.add_argument("--validation_seed", type=int, default=42)
    ap.add_argument(
        "--pool_epochs",
        type=int,
        default=None,
        help="Epoch override for the uncond pool run (default: method config).",
    )
    ap.add_argument(
        "--ft_epochs",
        type=int,
        default=None,
        help="Epoch override for every fine-tune (default: method config).",
    )
    ap.add_argument(
        "--image_dir",
        default="post_image_dataset/resized",
        help="Where to verify the artist subdirectories exist.",
    )
    ap.add_argument("--write", default=None, help="Also write the commands here (.sh).")
    args = ap.parse_args()

    if args.target_artist is None:
        args.target_artist = args.pool_artists[0]

    known = set(list_artists(str(REPO_ROOT / args.image_dir)))
    for a in set(args.pool_artists) | {args.target_artist}:
        if known and a not in known:
            print(f"WARNING: artist dir {a!r} not found under {args.image_dir}")

    cmds = build_commands(args)
    lines = [" ".join(shlex.quote(t) for t in argv) for _label, argv in cmds]

    print(f"# uncond-soup bench: {len(cmds)} training runs")
    print(f"# pool = {args.pool_artists}  target = {args.target_artist}")
    print("# Run sequentially (or submit each to the daemon queue).\n")
    for (label, _argv), line in zip(cmds, lines):
        print(f"# {label}")
        print(line, "\n")

    self_ckpts = " ".join(
        f"output/ckpt/{PREFIX}_self_s{args.seed_base + k}.safetensors"
        for k in range(1, args.n_seeds + 1)
    )
    base_ckpts = " ".join(
        f"output/ckpt/{PREFIX}_base_s{args.seed_base + k}.safetensors"
        for k in range(1, args.n_seeds + 1)
    )
    print("# Then score everything:")
    print(
        f"python bench/uncond_soup/probe.py --method {args.method} "
        f"--preset {args.preset} --target_artist {shlex.quote(args.target_artist)} "
        f"--self_ckpts {self_ckpts} --base_ckpts {base_ckpts} "
        f"--pool_ckpt output/ckpt/{PREFIX}_pool.safetensors "
        f"--validation_split_num {args.validation_split_num} "
        f"--validation_seed {args.validation_seed}"
    )

    if args.write:
        out = Path(args.write)
        body = "\n".join(f"# {label}\n{line}" for (label, _a), line in zip(cmds, lines))
        out.write_text(f"#!/usr/bin/env bash\nset -euo pipefail\n\n{body}\n")
        print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
