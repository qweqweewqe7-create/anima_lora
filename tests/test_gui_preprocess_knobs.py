"""Unit tests for the Qt-free knob table (``gui/tabs/preprocess/knobs.py``).

Feeds the pure functions hand-built value dicts and checks them against the
Phase 0 characterization fixture — so the table reproduces the tab's contract
without constructing a single widget.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gui.tabs.preprocess import knobs as K
from tests.test_gui_preprocess_characterization import FIXTURE, SCENARIOS

EXPECTED = json.loads(Path(FIXTURE).read_text(encoding="utf-8"))

# Widget-side normalisations the pure layer never sees: the tier row returns
# its tiers sorted, and the bucket-reso row drops entries outside the active
# tiers (so the populated scenario's "1024x1024" comes back empty).
_WIDGET_NORMALISED = {"target_res": sorted, "resize_bucket_resos": lambda _v: []}

# The widget values `_flip_every_knob` in the characterization test produces.
FLIPPED_VALUES = {
    "source_image_dir": "flipped_images",
    "path_scope": "artist_a",
    "preprocess_path_pattern": "artist_a/**",
    "min_pixels": 123456,
    "target_res": [768, 1280],
    "resize_bucket_resos": [],
    "resize_crop_anchor": "bottom_right",
    "resize_crop_margins": {"top": 1.5, "right": 2.5, "bottom": 3.5, "left": 4.5},
    "freefit_max_ratio": 2.25,
    "caption_shuffle_variants": 11,
    "caption_tag_dropout_rate": "0.45",
    "caption_correct_order": True,
    "caption_insert_no_artist": True,
    "caption_trigger_word": "@flipped",
    "caption_trigger_at_front": True,
    "caption_autotag_mode": "overwrite",
    "caption_autotag_min_confidence": 0.55,
    "mask_path_pattern": "artist_b/*",
    "mask_rules": [
        {
            "path_pattern": "artist_b/*",
            "prompts": ["bubble", "sfx"],
            "focus_prompts": ["face"],
            "threshold": 0.35,
            "dilate": 7,
        },
        {"prompts": ["watermark"], "threshold": 0.6, "dilate": 2},
    ],
    "mit_text_threshold": "0.65",
    "mit_dilate": 13,
}
_TOGGLED = (
    "drop_lowres_images",
    "caption_position_clauses",
    "caption_autotag",
    "run_sam_mask",
    "run_mit_mask",
)


def _defaults(scenario: str) -> dict:
    s = SCENARIOS[scenario]
    return K.resolved_defaults(s["preprocess_toml"], s["gui_settings"], s["sam_yaml"])


def _default_widget_values(scenario: str) -> dict:
    """What the widgets hold right after ``set_variant`` on an empty variant."""
    values = K.load_values({}, _defaults(scenario))
    for key, fn in _WIDGET_NORMALISED.items():
        values[key] = fn(values[key])
    # Free-text numerics are shown with :g and read back as text.
    for key in ("caption_tag_dropout_rate", "mit_text_threshold"):
        values[key] = f"{float(values[key]):g}"
    return values


def _flipped_widget_values(scenario: str) -> dict:
    values = _default_widget_values(scenario)
    values.update(FLIPPED_VALUES)
    for key in _TOGGLED:
        values[key] = not values[key]
    return values


def _roundtrip(data):
    return json.loads(json.dumps(data, sort_keys=True))


@pytest.mark.parametrize("scenario", list(SCENARIOS))
@pytest.mark.parametrize("state", ["defaults", "flipped"])
def test_env_and_overrides_match_fixture(scenario, state):
    values = (
        _default_widget_values(scenario)
        if state == "defaults"
        else _flipped_widget_values(scenario)
    )
    expected = EXPECTED[scenario][state]
    assert K.to_env(values, _defaults(scenario)) == expected["env"]
    assert _roundtrip(K.to_overrides(values)) == expected["overrides"]


def _card_roundtrip(rule: dict) -> dict:
    """What `_RuleCard.to_dict` hands back for a rule it was built from:
    empty / "*" path_pattern and empty focus_prompts are dropped."""
    out = {}
    if rule.get("path_pattern") and rule["path_pattern"] != "*":
        out["path_pattern"] = rule["path_pattern"]
    if rule.get("prompts"):
        out["prompts"] = list(rule["prompts"])
    if rule.get("focus_prompts"):
        out["focus_prompts"] = list(rule["focus_prompts"])
    out["threshold"] = float(rule["threshold"])
    out["dilate"] = int(rule["dilate"])
    return out


def _persistable(values: dict) -> dict:
    """The tab validates the free-text numerics and reads the rule cards
    before persisting."""
    values = dict(values)
    values["caption_tag_dropout_rate"] = float(values["caption_tag_dropout_rate"])
    values["mit_text_threshold"] = float(values["mit_text_threshold"])
    values["mask_rules"] = [_card_roundtrip(r) for r in values["mask_rules"]]
    return values


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_merge_into_meta_matches_fixture(scenario):
    """Same save sequence as the characterization run, on one meta table:
    inputs-only save, full save, flip, inputs-only save, full save — so a
    mask-less save leaving earlier mask keys untouched is part of the contract."""
    defaults = _defaults(scenario)
    expected = EXPECTED[scenario]
    meta = {"family": "lora"}
    for state, values in (
        ("defaults", _persistable(_default_widget_values(scenario))),
        ("flipped", _persistable(_flipped_widget_values(scenario))),
    ):
        K.merge_into_meta(meta, values, defaults, include_mask=False)
        assert _roundtrip(meta) == expected[state]["meta_inputs_only"], state
        K.merge_into_meta(meta, values, defaults, include_mask=True)
        assert _roundtrip(meta) == expected[state]["meta_full"], state


def test_load_values_is_a_fixed_point_of_merge_on_bare_checkout():
    """Save → load reproduces the flipped values (the tab's reload invariant)."""
    defaults = _defaults("bare")
    values = _persistable(_flipped_widget_values("bare"))
    meta = K.merge_into_meta({}, values, defaults, include_mask=True)
    loaded = K.load_values(meta, defaults)
    for knob in K.KNOBS:
        assert K._coerce(knob, loaded[knob.key]) == K._coerce(knob, values[knob.key]), (
            knob.key
        )


def test_const_elision_under_populated_toml_is_the_recorded_quirk():
    """`drop_lowres_images` *loads* from preprocess.toml but is *elided*
    against the hardcoded default: with the TOML at false, ticking the box back
    to true (== hardcoded default) is popped and reloads as false. Recorded by
    the Phase 0 fixture; collapsing the policies is a separate decision — a
    failure here means that decision was made implicitly."""
    defaults = _defaults("populated")
    assert defaults["drop_lowres_images"] is False
    values = _persistable(_flipped_widget_values("populated"))
    assert values["drop_lowres_images"] is True
    meta = K.merge_into_meta({}, values, defaults, include_mask=False)
    assert "drop_lowres_images" not in meta
    assert K.load_values(meta, defaults)["drop_lowres_images"] is False


def test_elision_keeps_a_plain_checkout_empty():
    """All-defaults on a bare checkout writes only the always-persisted tiers."""
    defaults = _defaults("bare")
    meta = K.merge_into_meta(
        {}, _default_widget_values("bare"), defaults, include_mask=False
    )
    assert meta == {"target_res": K.DEFAULT_TARGET_RES}


def test_preprocess_toml_default_sticks_when_unchecked():
    """The `_pp_default` trap, now declared: a caption-master stage set true
    in preprocess.toml must persist an explicit false when unchecked."""
    pp = {"caption_position_clauses": True, "caption_autotag": True}
    defaults = K.resolved_defaults(pp, {}, {})
    values = K.load_values({}, defaults)
    assert values["caption_position_clauses"] is True
    values["caption_position_clauses"] = False
    values["caption_autotag"] = False
    meta = K.merge_into_meta({}, values, defaults, include_mask=False)
    assert meta["caption_position_clauses"] is False
    assert meta["caption_autotag"] is False


def test_table_invariants():
    keys = [k.key for k in K.KNOBS]
    assert len(keys) == len(set(keys))
    assert K.PREPROCESS_ONLY_KEYS == set(keys)
    for knob in K.KNOBS:
        assert knob.enabled_by is None or knob.enabled_by in K.KNOBS_BY_KEY, knob.key
        if knob.kind == "choice":
            assert knob.default in knob.choices, knob.key
        if knob.persist == "mask":
            assert knob.section == "mask" and not knob.snapshot and not knob.env
    env_names = [k.env for k in K.ENV_KNOBS]
    assert len(env_names) == len(set(env_names))
