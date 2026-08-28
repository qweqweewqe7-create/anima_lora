"""Masking sections: SAM3 rule cards (+ the shared mask path pattern) and the
MIT text detector. Both persist to the variant's ``[variant]`` table under the
``persist="mask"`` policy; ``sam_mask.yaml`` is only the CLI fallback."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from gui.explanations import preprocess_field_help
from gui.i18n import t
from gui.tabs.preprocess._section import KnobSection, checkbox, line, spin
from gui.tabs.preprocess.knobs import (
    DEFAULT_MASK_PATH_PATTERN,
    DEFAULT_MIT_DILATE,
    DEFAULT_MIT_TEXT_THRESHOLD,
    DEFAULT_RUN_MIT_MASK,
    DEFAULT_RUN_SAM_MASK,
    DEFAULT_SAM_DILATE,
    DEFAULT_SAM_THRESHOLD,
)
from gui.theme import tok
from gui.widgets import ClickableLabel, make_field_label


class _RuleCard(QGroupBox):
    """One SAM mask rule editor: path_pattern + prompts/focus + threshold/dilate.

    ``prompts`` mask OUT (ignored in the loss); ``focus_prompts`` keep ONLY
    that subject (reversed polarity). Empty/``*`` path_pattern is a catch-all.
    """

    removed = Signal(object)
    changed = Signal()

    def __init__(self, rule: dict, help_cb):
        super().__init__(t("preprocess_sam_rule"))
        self._help_cb = help_cb
        form = QFormLayout(self)

        self.path_pattern_edit = QLineEdit(rule.get("path_pattern", ""))
        self.path_pattern_edit.setPlaceholderText("*")
        self.path_pattern_edit.setToolTip(t("preprocess_sam_rule_path_pattern_tip"))
        self.path_pattern_edit.textChanged.connect(lambda *_: self.changed.emit())
        form.addRow(
            self._label("sam_rule_path_pattern", t("preprocess_sam_rule_path_pattern")),
            self.path_pattern_edit,
        )

        self.prompts_edit = self._prompt_box(
            rule.get("prompts"), t("preprocess_sam_prompts_tip")
        )
        form.addRow(
            self._label("sam_prompts", t("preprocess_sam_prompts")), self.prompts_edit
        )
        self.focus_prompts_edit = self._prompt_box(
            rule.get("focus_prompts"), t("preprocess_sam_focus_prompts_tip")
        )
        form.addRow(
            self._label("sam_focus_prompts", t("preprocess_sam_focus_prompts")),
            self.focus_prompts_edit,
        )

        self.threshold_edit = QLineEdit(
            f"{float(rule.get('threshold', DEFAULT_SAM_THRESHOLD)):g}"
        )
        self.threshold_edit.setToolTip(t("preprocess_sam_threshold_tip"))
        self.threshold_edit.textChanged.connect(lambda *_: self.changed.emit())
        form.addRow(
            self._label("sam_threshold", t("preprocess_sam_threshold")),
            self.threshold_edit,
        )

        self.dilate_spin = QSpinBox()
        self.dilate_spin.setRange(0, 64)
        self.dilate_spin.setValue(int(rule.get("dilate", DEFAULT_SAM_DILATE)))
        self.dilate_spin.wheelEvent = lambda e: e.ignore()
        self.dilate_spin.valueChanged.connect(lambda *_: self.changed.emit())
        form.addRow(self._label("sam_dilate", t("preprocess_dilate")), self.dilate_spin)

        self.remove_btn = QPushButton(t("preprocess_sam_remove_rule"))
        self.remove_btn.clicked.connect(lambda: self.removed.emit(self))
        form.addRow("", self.remove_btn)

    def _prompt_box(self, lines, tooltip: str) -> QPlainTextEdit:
        box = QPlainTextEdit("\n".join(lines or []))
        box.setMaximumHeight(70)
        box.setStyleSheet("font-family:monospace;")
        box.setToolTip(tooltip)
        box.textChanged.connect(lambda: self.changed.emit())
        return box

    def _label(self, key: str, text: str) -> ClickableLabel:
        help_text = preprocess_field_help(key)
        return make_field_label(
            text,
            style=f"color:{tok('text')}; text-decoration: underline dotted;",
            on_click=lambda _t=text, _h=help_text: self._help_cb(_t, _h),
        )

    def to_dict(self) -> dict:
        """Raises ValueError on an unparseable threshold."""
        text = self.threshold_edit.text().strip()
        try:
            threshold = float(text)
        except ValueError as exc:
            raise ValueError(text) from exc
        rule: dict = {}
        pattern = self.path_pattern_edit.text().strip()
        if pattern and pattern != "*":
            rule["path_pattern"] = pattern
        prompts = [
            ln.strip()
            for ln in self.prompts_edit.toPlainText().splitlines()
            if ln.strip()
        ]
        if prompts:
            rule["prompts"] = prompts
        focus = [
            ln.strip()
            for ln in self.focus_prompts_edit.toPlainText().splitlines()
            if ln.strip()
        ]
        if focus:
            rule["focus_prompts"] = focus
        rule["threshold"] = threshold
        rule["dilate"] = int(self.dilate_spin.value())
        return rule


class SamMaskSection(KnobSection):
    """Run-SAM toggle + mask path pattern (scopes BOTH backends) + one
    ``_RuleCard`` per rule; matching rules compose."""

    def __init__(
        self,
        help_cb,
        *,
        settings: dict,
        sam_yaml_rules: list[dict],
        mask_path_pattern: str,
    ):
        self._settings = settings
        self._initial_rules = sam_yaml_rules
        self._initial_pattern = mask_path_pattern
        super().__init__(t("preprocess_masking_sam"), help_cb)

    def _build(self) -> None:
        on = checkbox(t("preprocess_run_sam_mask"))
        on.setChecked(bool(self._settings.get("run_sam_mask", DEFAULT_RUN_SAM_MASK)))
        self.add_knob(
            "run_sam_mask",
            on,
            t("preprocess_run_sam_mask"),
            tooltip=t("preprocess_run_sam_mask_tip"),
        )
        # Stored in sam_mask.yaml but scopes BOTH backends (SAM and MIT alike).
        self.add_knob(
            "mask_path_pattern",
            line(self._initial_pattern, placeholder="*"),
            t("preprocess_mask_path_pattern"),
            tooltip=t("preprocess_mask_path_pattern_tip"),
        )
        # Rule cards sit below the form in a vertical stack.
        outer = QVBoxLayout()
        outer.setContentsMargins(0, 0, 0, 0)
        self._rule_cards: list[_RuleCard] = []
        self._rules_layout = QVBoxLayout()
        self._rules_layout.setContentsMargins(0, 0, 0, 0)
        outer.addLayout(self._rules_layout)
        self.add_rule_btn = QPushButton(t("preprocess_sam_add_rule"))
        self.add_rule_btn.setToolTip(t("preprocess_sam_add_rule_tip"))
        self.add_rule_btn.clicked.connect(lambda: self.add_rule_card())
        outer.addWidget(self.add_rule_btn)
        self.form.addRow(outer)
        for rule in self._initial_rules:
            self.add_rule_card(rule)

    # -- rule cards ---------------------------------------------------------

    @property
    def rule_cards(self) -> list[_RuleCard]:
        return self._rule_cards

    def add_rule_card(self, rule: dict | None = None) -> None:
        card = _RuleCard(rule or {}, self._help_cb)
        card.changed.connect(self.changed.emit)
        card.removed.connect(self.remove_rule_card)
        self._rule_cards.append(card)
        self._rules_layout.addWidget(card)
        self._update_remove_buttons()
        self.changed.emit()

    def remove_rule_card(self, card: _RuleCard) -> None:
        if len(self._rule_cards) <= 1:
            return  # keep at least one rule
        self._rule_cards.remove(card)
        self._rules_layout.removeWidget(card)
        card.deleteLater()
        self._update_remove_buttons()
        self.changed.emit()

    def set_rule_cards(self, rules: list[dict]) -> None:
        for card in list(self._rule_cards):
            self._rules_layout.removeWidget(card)
            card.deleteLater()
        self._rule_cards.clear()
        for rule in rules or [{}]:
            self.add_rule_card(rule)
        self._update_remove_buttons()

    def _update_remove_buttons(self) -> None:
        # A lone rule can't be removed (would leave an empty config).
        sole = len(self._rule_cards) <= 1
        for card in self._rule_cards:
            card.remove_btn.setEnabled(not sole)

    def collect_rules(self) -> list[dict] | None:
        """Serialize every rule card, or None (after a warning) if one fails
        validation."""
        rules: list[dict] = []
        for card in self._rule_cards:
            try:
                rules.append(card.to_dict())
            except ValueError as bad_threshold:
                QMessageBox.warning(
                    self,
                    t("error"),
                    t(
                        "preprocess_invalid_float",
                        field=t("preprocess_sam_threshold"),
                        value=str(bad_threshold),
                    ),
                )
                return None
        return rules

    # -- values -------------------------------------------------------------

    def values(self) -> dict[str, object]:
        out = super().values()
        # Rules validate on read, so they're collected explicitly (collect_rules)
        # by the save path rather than through the generic read.
        out["mask_rules"] = None
        return out

    def set_values(self, values: dict) -> None:
        super().set_values(values)
        if "mask_rules" in values:
            self.set_rule_cards(values["mask_rules"])

    def mask_path_pattern(self) -> str:
        return (
            self.widgets["mask_path_pattern"].text().strip()
            or DEFAULT_MASK_PATH_PATTERN
        )


class MitMaskSection(KnobSection):
    def __init__(self, help_cb, *, settings: dict):
        self._settings = settings
        super().__init__(t("preprocess_masking_mit"), help_cb)

    def _build(self) -> None:
        s = self._settings
        on = checkbox(t("preprocess_run_mit_mask"))
        on.setChecked(bool(s.get("run_mit_mask", DEFAULT_RUN_MIT_MASK)))
        self.add_knob(
            "run_mit_mask",
            on,
            t("preprocess_run_mit_mask"),
            tooltip=t("preprocess_run_mit_mask_tip"),
        )
        self.add_knob(
            "mit_text_threshold",
            line(f"{float(s.get('mit_text_threshold', DEFAULT_MIT_TEXT_THRESHOLD)):g}"),
            t("preprocess_mit_threshold"),
            tooltip="",
        )
        self.add_knob(
            "mit_dilate",
            spin(0, 64, int(s.get("mit_dilate", DEFAULT_MIT_DILATE))),
            t("preprocess_dilate"),
            tooltip="",
        )
