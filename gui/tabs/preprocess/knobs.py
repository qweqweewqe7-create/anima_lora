"""The Preprocessing tab's knob table — single source of truth, Qt-free.

Every knob the tab surfaces is one ``Knob`` row here. The pure functions below
replace the per-knob ladders that used to live in ``preprocess_tab.py``
(``set_variant`` fallback chain, ``preprocess_env`` / ``preprocess_overrides``
serialisation, and the ``_save_variant_preprocess_meta`` pop-or-set chain),
so adding a knob is one row + the widget, and the three "where does the
default come from" policies (hardcoded / ``preprocess.toml`` /
``gui_settings.json``) are declared instead of re-derived at each site.

The contract is pinned byte-for-byte by
``tests/test_gui_preprocess_characterization.py`` (fixture under
``tests/fixtures/``). The quirks it records — e.g. a knob that *loads* from
``preprocess.toml`` but is *elided* against the hardcoded default — are
reproduced on purpose; collapsing the policies is a separate decision.

Like ``gui/config_io.py`` this module must stay importable without PySide6
and without torch (``tests/test_gui_launch_speed.py``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from library.datasets.buckets import DEFAULT_TARGET_RES as _LIB_DEFAULT_TARGET_RES
from library.preprocess.resize_preview import (
    DEFAULT_FREEFIT_MAX_RATIO,
    DEFAULT_RESIZE_CROP_ANCHOR,
    normalize_crop_margins,
)

# Defaults match the historical hardcoded values in scripts/tasks/preprocess.py
# and generate_masks_mit.py, so a fresh GUI runs the same pipeline as the bare CLI.
DEFAULT_SOURCE_IMAGE_DIR = "image_dataset"
DEFAULT_PREPROCESS_PATH_PATTERN = "*"
DEFAULT_DROP_LOWRES_IMAGES = True
DEFAULT_MIN_PIXELS = 500000
DEFAULT_TARGET_RES = list(_LIB_DEFAULT_TARGET_RES)
DEFAULT_RESIZE_BUCKET_RESOS: list[str] = []
DEFAULT_RESIZE_CROP_MARGINS = {"top": 0.0, "right": 0.0, "bottom": 0.0, "left": 0.0}
DEFAULT_TE_SHUFFLE_VARIANTS = 4
DEFAULT_TE_TAG_DROPOUT = 0.1
DEFAULT_CAPTION_CORRECT_ORDER = False
DEFAULT_CAPTION_INSERT_NO_ARTIST = False
DEFAULT_CAPTION_TRIGGER_WORD = ""
DEFAULT_CAPTION_TRIGGER_AT_FRONT = False
DEFAULT_CAPTION_POSITION_CLAUSES = False
DEFAULT_CAPTION_AUTOTAG = False
# Mirrors library.preprocess.autotag.MODES — kept as a literal so the GUI
# doesn't drag in the PIL/torch import chain (see gui/CLAUDE.md).
CAPTION_AUTOTAG_MODES = ("missing", "merge", "overwrite")
DEFAULT_CAPTION_AUTOTAG_MODE = "missing"
DEFAULT_CAPTION_AUTOTAG_MIN_CONFIDENCE = 0.0
DEFAULT_SAM_PROMPTS = ("speech bubble", "text bubble")
DEFAULT_SAM_THRESHOLD = 0.5
DEFAULT_SAM_DILATE = 5
DEFAULT_MASK_PATH_PATTERN = "*"
DEFAULT_MIT_TEXT_THRESHOLD = 0.8
DEFAULT_MIT_DILATE = 5
DEFAULT_RUN_SAM_MASK = True
DEFAULT_RUN_MIT_MASK = True

Kind = Literal[
    "bool",
    "int",
    "float",
    "str",
    "choice",
    "tier_list",
    "str_list",
    "crop_anchor",
    "margins",
    "rules",
]
DefaultFrom = Literal["const", "preprocess_toml", "gui_settings", "sam_yaml"]
# How the knob reaches the variant's ``[variant]`` meta on save:
#   if_changed        — written only when it differs from the *hardcoded* default
#   if_changed_resolved — written only when it differs from the *resolved*
#                       default (preprocess.toml-backed knobs; a checkbox left at
#                       the hardcoded default would otherwise not stick, because
#                       the tab always exports the env var and env beats the TOML)
#   if_truthy         — written when non-empty (path_scope, bucket list, margins>0)
#   always            — always written (target_res sizes the train-time compile cache)
#   mask              — always written, but only when the mask section is being saved
Persist = Literal["if_changed", "if_changed_resolved", "if_truthy", "always", "mask"]


@dataclass(frozen=True)
class Knob:
    key: str
    section: str  # "image" | "text" | "captions" | "autotag" | "mask"
    kind: Kind
    default: object
    default_from: DefaultFrom = "const"
    env: str | None = None
    persist: Persist = "if_changed"
    snapshot: bool = False  # goes into preprocess_overrides()
    choices: tuple[str, ...] = ()
    enabled_by: str | None = None
    # Empty text means "the default": "const" → the hardcoded one, "resolved"
    # → the scenario's resolved one. Applied before env export and elision.
    empty_fallback: Literal["const", "resolved"] | None = None


# Row order == on-disk key order in a freshly written [variant] table.
KNOBS: tuple[Knob, ...] = (
    Knob(
        "source_image_dir",
        "image",
        "str",
        DEFAULT_SOURCE_IMAGE_DIR,
        default_from="preprocess_toml",
        persist="if_changed_resolved",
        empty_fallback="resolved",
    ),
    # Layered on top at submit time by ConfigTab._gui_scoped_paths; the field
    # edits the *unscoped* root. Normalised (and validated) by the tab before save.
    Knob("path_scope", "image", "str", "", persist="if_truthy"),
    Knob(
        "preprocess_path_pattern",
        "image",
        "str",
        DEFAULT_PREPROCESS_PATH_PATTERN,
        env="PREPROCESS_PATH_PATTERN",
        empty_fallback="const",
    ),
    Knob(
        "drop_lowres_images",
        "image",
        "bool",
        DEFAULT_DROP_LOWRES_IMAGES,
        default_from="preprocess_toml",
        env="DROP_LOWRES_IMAGES",
        snapshot=True,
    ),
    Knob(
        "min_pixels",
        "image",
        "int",
        DEFAULT_MIN_PIXELS,
        default_from="preprocess_toml",
        env="MIN_PIXELS",
        snapshot=True,
        enabled_by="drop_lowres_images",
    ),
    Knob(
        "target_res",
        "image",
        "tier_list",
        DEFAULT_TARGET_RES,
        default_from="preprocess_toml",
        env="TARGET_RES",
        persist="always",
        snapshot=True,
    ),
    Knob(
        "resize_bucket_resos",
        "image",
        "str_list",
        DEFAULT_RESIZE_BUCKET_RESOS,
        default_from="preprocess_toml",
        persist="if_truthy",
        snapshot=True,
    ),
    Knob(
        "resize_crop_anchor",
        "image",
        "crop_anchor",
        DEFAULT_RESIZE_CROP_ANCHOR,
        default_from="preprocess_toml",
        snapshot=True,
    ),
    Knob(
        "resize_crop_margins",
        "image",
        "margins",
        DEFAULT_RESIZE_CROP_MARGINS,
        default_from="preprocess_toml",
        persist="if_truthy",
        snapshot=True,
    ),
    Knob(
        "freefit_max_ratio",
        "image",
        "float",
        DEFAULT_FREEFIT_MAX_RATIO,
        default_from="preprocess_toml",
        env="FREEFIT_MAX_RATIO",
        snapshot=True,
    ),
    Knob(
        "caption_shuffle_variants",
        "text",
        "int",
        DEFAULT_TE_SHUFFLE_VARIANTS,
        default_from="gui_settings",
        env="CAPTION_SHUFFLE_VARIANTS",
    ),
    Knob(
        "caption_tag_dropout_rate",
        "text",
        "float",
        DEFAULT_TE_TAG_DROPOUT,
        default_from="gui_settings",
        env="CAPTION_TAG_DROPOUT_RATE",
    ),
    Knob(
        "caption_correct_order",
        "captions",
        "bool",
        DEFAULT_CAPTION_CORRECT_ORDER,
        env="CAPTION_CORRECT_ORDER",
        snapshot=True,
    ),
    Knob(
        "caption_insert_no_artist",
        "captions",
        "bool",
        DEFAULT_CAPTION_INSERT_NO_ARTIST,
        env="CAPTION_INSERT_NO_ARTIST",
        snapshot=True,
    ),
    Knob(
        "caption_trigger_word",
        "captions",
        "str",
        DEFAULT_CAPTION_TRIGGER_WORD,
        env="CAPTION_TRIGGER_WORD",
        snapshot=True,
    ),
    Knob(
        "caption_trigger_at_front",
        "captions",
        "bool",
        DEFAULT_CAPTION_TRIGGER_AT_FRONT,
        env="CAPTION_TRIGGER_AT_FRONT",
        snapshot=True,
    ),
    # Caption-master stages: gate SAM3+tagger (position clauses) before TE and
    # the Anima Tagger stage right after resize (autotag *creates* the master).
    Knob(
        "caption_position_clauses",
        "captions",
        "bool",
        DEFAULT_CAPTION_POSITION_CLAUSES,
        default_from="preprocess_toml",
        env="CAPTION_POSITION_CLAUSES",
        persist="if_changed_resolved",
        snapshot=True,
    ),
    Knob(
        "caption_autotag",
        "autotag",
        "bool",
        DEFAULT_CAPTION_AUTOTAG,
        default_from="preprocess_toml",
        env="CAPTION_AUTOTAG",
        persist="if_changed_resolved",
        snapshot=True,
    ),
    Knob(
        "caption_autotag_mode",
        "autotag",
        "choice",
        DEFAULT_CAPTION_AUTOTAG_MODE,
        default_from="preprocess_toml",
        env="CAPTION_AUTOTAG_MODE",
        persist="if_changed_resolved",
        snapshot=True,
        choices=CAPTION_AUTOTAG_MODES,
        enabled_by="caption_autotag",
    ),
    Knob(
        "caption_autotag_min_confidence",
        "autotag",
        "float",
        DEFAULT_CAPTION_AUTOTAG_MIN_CONFIDENCE,
        default_from="preprocess_toml",
        env="CAPTION_AUTOTAG_MIN_CONFIDENCE",
        persist="if_changed_resolved",
        snapshot=True,
        enabled_by="caption_autotag",
    ),
    Knob(
        "run_sam_mask",
        "mask",
        "bool",
        DEFAULT_RUN_SAM_MASK,
        default_from="gui_settings",
        persist="mask",
    ),
    Knob(
        "run_mit_mask",
        "mask",
        "bool",
        DEFAULT_RUN_MIT_MASK,
        default_from="gui_settings",
        persist="mask",
    ),
    Knob(
        "mask_path_pattern",
        "mask",
        "str",
        DEFAULT_MASK_PATH_PATTERN,
        default_from="sam_yaml",
        persist="mask",
        empty_fallback="const",
    ),
    Knob("mask_rules", "mask", "rules", [], default_from="sam_yaml", persist="mask"),
    Knob(
        "mit_text_threshold",
        "mask",
        "float",
        DEFAULT_MIT_TEXT_THRESHOLD,
        default_from="gui_settings",
        persist="mask",
    ),
    Knob(
        "mit_dilate",
        "mask",
        "int",
        DEFAULT_MIT_DILATE,
        default_from="gui_settings",
        persist="mask",
    ),
)

KNOBS_BY_KEY: dict[str, Knob] = {k.key: k for k in KNOBS}

# Every key the Preprocessing tab owns. ConfigTab strips these from the
# training snapshot (they must not ride into train.py — e.g.
# ``caption_tag_dropout_rate`` collides with a real *live* dataloader arg).
PREPROCESS_ONLY_KEYS: frozenset[str] = frozenset(k.key for k in KNOBS)
ENV_KNOBS: tuple[Knob, ...] = tuple(k for k in KNOBS if k.env)
SNAPSHOT_KNOBS: tuple[Knob, ...] = tuple(k for k in KNOBS if k.snapshot)
MASK_KEYS: frozenset[str] = frozenset(k.key for k in KNOBS if k.persist == "mask")


def load_rules(sam_yaml: dict) -> list[dict]:
    """Normalize either ``sam_mask.yaml`` schema into per-card rule dicts: a
    ``rules:`` array returns card-for-card (missing threshold/dilate fall back
    to top-level); a flat config collapses to one catch-all card."""
    default_threshold = float(sam_yaml.get("threshold", DEFAULT_SAM_THRESHOLD))
    default_dilate = int(sam_yaml.get("dilate", DEFAULT_SAM_DILATE))
    raw = sam_yaml.get("rules")
    if raw is None:
        return [
            {
                "path_pattern": "",
                "prompts": sam_yaml.get("prompts") or list(DEFAULT_SAM_PROMPTS),
                "focus_prompts": sam_yaml.get("focus_prompts") or [],
                "threshold": default_threshold,
                "dilate": default_dilate,
            }
        ]
    return [
        {
            "path_pattern": r.get("path_pattern") or "",
            "prompts": r.get("prompts") or [],
            "focus_prompts": r.get("focus_prompts") or [],
            "threshold": float(r.get("threshold", default_threshold)),
            "dilate": int(r.get("dilate", default_dilate)),
        }
        for r in raw
    ]


def resolve_default(knob: Knob, pp_cfg: dict, settings: dict, sam_yaml: dict):
    """The effective default for one knob under its ``default_from`` policy.

    A ``preprocess.toml``-backed knob resolves *through* the user-owned TOML:
    load-bearing for the caption-master stages, whose env var the tab always
    exports (env beats the TOML in ``tasks.py``), so the widget must start
    from the TOML's answer or a `true` set there would be silently overridden."""
    if knob.default_from == "const":
        return knob.default
    if knob.default_from == "preprocess_toml":
        value = pp_cfg.get(knob.key)
    elif knob.default_from == "gui_settings":
        value = settings.get(knob.key)
    elif knob.key == "mask_rules":
        return load_rules(sam_yaml)
    elif knob.key == "mask_path_pattern":
        return sam_yaml.get("path_pattern") or knob.default
    else:  # pragma: no cover — unreachable by construction
        raise ValueError(f"{knob.key}: unsupported default_from for sam_yaml")
    return knob.default if value is None else value


def resolved_defaults(pp_cfg: dict, settings: dict, sam_yaml: dict) -> dict:
    """``{key: effective default}`` for every knob — the one dict the other
    helpers take, so the tab reads its three default sources exactly once."""
    return {k.key: resolve_default(k, pp_cfg, settings, sam_yaml) for k in KNOBS}


def load_values(meta: dict, defaults: dict) -> dict:
    """Widget values for a variant: its ``[variant]`` meta over the resolved
    defaults (replaces the per-knob fallback ladder in ``set_variant``)."""
    values = {}
    for knob in KNOBS:
        if knob.key == "mask_rules":
            rules = meta.get("mask_rules")
            values[knob.key] = rules if isinstance(rules, list) else defaults[knob.key]
        else:
            values[knob.key] = meta.get(knob.key, defaults[knob.key])
    return values


def _coerce(knob: Knob, value):
    """Normalise a raw widget value to the type the TOML/snapshot carries.
    Strings for float knobs are parsed (the tab hands the dropout rate over as
    the line-edit text so the env export stays byte-exact)."""
    kind = knob.kind
    if kind == "bool":
        return bool(value)
    if kind == "int":
        return int(value)
    if kind == "float":
        return float(value)
    if kind in ("str", "choice", "crop_anchor"):
        return str(value)
    if kind == "tier_list":
        return list(value)
    if kind == "str_list":
        return list(value)
    if kind == "margins":
        return normalize_crop_margins(value)
    if kind == "rules":
        return list(value or [])
    raise ValueError(f"{knob.key}: unknown kind {kind}")  # pragma: no cover


def _with_empty_fallback(knob: Knob, value, defaults: dict):
    if knob.empty_fallback and (value is None or value == ""):
        return knob.default if knob.empty_fallback == "const" else defaults[knob.key]
    return value


def to_env(values: dict, defaults: dict) -> dict[str, str]:
    """Environment consumed by ``tasks.py preprocess``.

    The geometry/filter knobs ride as env, not just the config snapshot,
    because the Train auto-chain hands preprocess a snapshot with the
    preprocess-only keys stripped; env wins over the snapshot in ``tasks.py``.
    A ``str`` for a float/int knob is exported verbatim (already user text)."""
    env: dict[str, str] = {}
    for knob in ENV_KNOBS:
        value = _with_empty_fallback(knob, values[knob.key], defaults)
        if knob.kind == "bool":
            text = "1" if value else "0"
        elif knob.kind == "tier_list":
            text = " ".join(str(e) for e in value)
        elif isinstance(value, str):
            text = value
        elif knob.kind == "int":
            text = str(int(value))
        elif knob.kind == "float":
            text = f"{float(value):g}"
        else:
            text = str(value)
        env[knob.env] = text  # type: ignore[index]
    return env


def to_overrides(values: dict) -> dict[str, object]:
    """Flat config overrides captured in preprocess snapshots
    (``preprocess_overrides``). Margins are passed through as the tab reads
    them (float dict), not normalised."""
    out: dict[str, object] = {}
    for knob in SNAPSHOT_KNOBS:
        value = values[knob.key]
        if knob.kind == "margins":
            out[knob.key] = dict(value)
        else:
            out[knob.key] = _coerce(knob, value)
    return out


def _is_truthy(knob: Knob, value) -> bool:
    if knob.kind == "margins":
        return any(v > 0 for v in value.values())
    return bool(value)


def merge_into_meta(
    meta: dict, values: dict, defaults: dict, *, include_mask: bool
) -> dict:
    """Apply the elision rules to a variant's ``[variant]`` table in place:
    each knob is written or popped per its ``persist`` policy, so a plain
    checkout keeps an empty meta. Mask knobs are only touched when
    ``include_mask`` (so an invalid mask threshold can't block a cache build).
    Returns ``meta``."""
    for knob in KNOBS:
        if knob.persist == "mask":
            if not include_mask:
                continue
            meta[knob.key] = _coerce(knob, values[knob.key])
            continue
        value = _coerce(knob, _with_empty_fallback(knob, values[knob.key], defaults))
        if knob.persist == "always":
            keep = True
        elif knob.persist == "if_truthy":
            keep = _is_truthy(knob, value)
        elif knob.persist == "if_changed":
            keep = value != _coerce(knob, knob.default)
        elif knob.persist == "if_changed_resolved":
            keep = value != _coerce(knob, defaults[knob.key])
        else:  # pragma: no cover
            raise ValueError(f"{knob.key}: unknown persist {knob.persist}")
        if keep:
            meta[knob.key] = value
        else:
            meta.pop(knob.key, None)
    return meta
