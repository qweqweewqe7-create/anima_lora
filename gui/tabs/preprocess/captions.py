"""Caption sections: the auto-tag box (creates a caption from nothing, runs
right after resize) and the caption-editing box (rewrites text that already
exists: order correction, @no-artist, trigger word, position clauses)."""

from __future__ import annotations

from PySide6.QtWidgets import QComboBox

from gui.i18n import t
from gui.tabs.preprocess._section import KnobSection, checkbox, dspin, line, no_wheel
from gui.tabs.preprocess.knobs import (
    CAPTION_AUTOTAG_MODES,
    DEFAULT_CAPTION_AUTOTAG,
    DEFAULT_CAPTION_AUTOTAG_MIN_CONFIDENCE,
    DEFAULT_CAPTION_AUTOTAG_MODE,
    DEFAULT_CAPTION_CORRECT_ORDER,
    DEFAULT_CAPTION_INSERT_NO_ARTIST,
    DEFAULT_CAPTION_POSITION_CLAUSES,
    DEFAULT_CAPTION_TRIGGER_AT_FRONT,
    DEFAULT_CAPTION_TRIGGER_WORD,
)


class AutotagSection(KnobSection):
    def __init__(self, help_cb, *, pp_cfg: dict):
        self._pp = pp_cfg
        super().__init__(t("preprocess_caption_autotag_box"), help_cb)

    def _build(self) -> None:
        pp = self._pp
        on = checkbox(t("preprocess_caption_autotag"))
        on.setChecked(bool(pp.get("caption_autotag", DEFAULT_CAPTION_AUTOTAG)))
        self.add_knob(
            "caption_autotag",
            on,
            t("preprocess_caption_autotag"),
            tooltip=t("preprocess_caption_autotag_tip"),
        )
        # Item data carries the untranslated mode name so a language switch
        # can't change what runs.
        mode = QComboBox()
        for m in CAPTION_AUTOTAG_MODES:
            mode.addItem(t(f"preprocess_caption_autotag_mode_{m}"), m)
        mode.setCurrentIndex(
            max(
                mode.findData(
                    str(pp.get("caption_autotag_mode", DEFAULT_CAPTION_AUTOTAG_MODE))
                ),
                0,
            )
        )
        self.add_knob(
            "caption_autotag_mode",
            no_wheel(mode),
            t("preprocess_caption_autotag_mode"),
            tooltip=t("preprocess_caption_autotag_mode_tip"),
        )
        self.add_knob(
            "caption_autotag_min_confidence",
            dspin(
                0.0,
                1.0,
                float(
                    pp.get(
                        "caption_autotag_min_confidence",
                        DEFAULT_CAPTION_AUTOTAG_MIN_CONFIDENCE,
                    )
                ),
                step=0.05,
                decimals=2,
            ),
            t("preprocess_caption_autotag_min_confidence"),
            tooltip=t("preprocess_caption_autotag_min_confidence_tip"),
        )

    def mode(self) -> str:
        return str(
            self.widgets["caption_autotag_mode"].currentData()
            or DEFAULT_CAPTION_AUTOTAG_MODE
        )


class CaptionEditingSection(KnobSection):
    def __init__(self, help_cb, *, pp_cfg: dict):
        self._pp = pp_cfg
        super().__init__(t("preprocess_caption_editing"), help_cb)

    def _build(self) -> None:
        pp = self._pp
        for key, default in (
            ("caption_correct_order", DEFAULT_CAPTION_CORRECT_ORDER),
            ("caption_insert_no_artist", DEFAULT_CAPTION_INSERT_NO_ARTIST),
        ):
            chk = checkbox(t(f"preprocess_{key}"))
            chk.setChecked(default)
            self.add_knob(
                key, chk, t(f"preprocess_{key}"), tooltip=t(f"preprocess_{key}_tip")
            )
        self.add_knob(
            "caption_trigger_word",
            line(DEFAULT_CAPTION_TRIGGER_WORD, placeholder="@trigger"),
            t("preprocess_caption_trigger_word"),
            tooltip=t("preprocess_caption_trigger_word_tip"),
        )
        front = checkbox(t("preprocess_caption_trigger_at_front"))
        front.setChecked(DEFAULT_CAPTION_TRIGGER_AT_FRONT)
        self.add_knob(
            "caption_trigger_at_front",
            front,
            t("preprocess_caption_trigger_at_front"),
            tooltip=t("preprocess_caption_trigger_at_front_tip"),
        )
        # Unlike its neighbours this is a GPU stage (SAM3 + tagger) that rewrites
        # the caption master in place, so it's off by default and its default
        # resolves through preprocess.toml (Knob.default_from).
        pos = checkbox(t("preprocess_caption_position_clauses"))
        pos.setChecked(
            bool(pp.get("caption_position_clauses", DEFAULT_CAPTION_POSITION_CLAUSES))
        )
        self.add_knob(
            "caption_position_clauses",
            pos,
            t("preprocess_caption_position_clauses"),
            tooltip=t("preprocess_caption_position_clauses_tip"),
        )
