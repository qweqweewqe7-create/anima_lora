"""Image-prep section: source dir / scope / pattern / low-res filter /
target-res tiers / crop anchor + margins / free-fit clamp."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QWidget,
)

from gui.i18n import t
from gui.tabs.preprocess._section import (
    KnobSection,
    checkbox,
    dspin,
    line,
    no_wheel,
    spin,
)
from gui.tabs.preprocess.knobs import (
    DEFAULT_MIN_PIXELS,
    DEFAULT_PREPROCESS_PATH_PATTERN,
    DEFAULT_SOURCE_IMAGE_DIR,
    DEFAULT_TARGET_RES,
)
from gui.theme import tok
from gui.widgets import _TargetResWidget
from library.preprocess.resize_preview import (
    DEFAULT_FREEFIT_MAX_RATIO,
    DEFAULT_RESIZE_CROP_ANCHOR,
    normalize_crop_margins,
)


class _ResizeCropAnchorWidget(QWidget):
    """3×3 anchor picker (which side survives when a resize has to crop)."""

    changed = Signal()

    _LAYOUT = (
        ("top_left", "↖", 0, 0),
        ("top", "↑", 0, 1),
        ("top_right", "↗", 0, 2),
        ("left", "←", 1, 0),
        ("center", "●", 1, 1),
        ("right", "→", 1, 2),
        ("bottom_left", "↙", 2, 0),
        ("bottom", "↓", 2, 1),
        ("bottom_right", "↘", 2, 2),
    )

    def __init__(self) -> None:
        super().__init__()
        grid = QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(2)
        grid.setVerticalSpacing(2)
        self._buttons: dict[str, QPushButton] = {}
        for key, text, row, col in self._LAYOUT:
            btn = QPushButton(text)
            btn.setCheckable(True)
            btn.setFixedSize(32, 28)
            btn.setStyleSheet(
                "QPushButton { padding:0; } "
                f"QPushButton:checked {{ background:{tok('accent')}; color:#ffffff; "
                "font-weight:bold; }"
            )
            btn.setToolTip(t(f"resize_crop_anchor_{key}"))
            btn.clicked.connect(lambda _checked, value=key: self.set_value(value))
            grid.addWidget(btn, row, col)
            self._buttons[key] = btn
        self.set_value(DEFAULT_RESIZE_CROP_ANCHOR, emit=False)
        self.setFixedSize(32 * 3 + 2 * 2, 28 * 3 + 2 * 2)

    def value(self) -> str:
        for key, btn in self._buttons.items():
            if btn.isChecked():
                return key
        return DEFAULT_RESIZE_CROP_ANCHOR

    def set_value(self, value, *, emit: bool = True) -> None:
        anchor = str(value or DEFAULT_RESIZE_CROP_ANCHOR)
        if anchor not in self._buttons:
            anchor = DEFAULT_RESIZE_CROP_ANCHOR
        for key, btn in self._buttons.items():
            btn.blockSignals(True)
            btn.setChecked(key == anchor)
            btn.blockSignals(False)
        if emit:
            self.changed.emit()


class _CropMarginsWidget(QWidget):
    """Four percent spins (top/right/bottom/left) — the crop-exclusion margins."""

    changed = Signal()
    _SIDES = ("top", "right", "bottom", "left")

    def __init__(self) -> None:
        super().__init__()
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        self.spins: dict[str, QDoubleSpinBox] = {}
        for side in self._SIDES:
            lbl = QLabel(t(f"resize_crop_margin_{side}"))
            lbl.setMinimumWidth(24)
            lbl.setStyleSheet(f"QLabel {{ color:{tok('text')}; }}")
            row.addWidget(lbl)
            s = self._margin_spin()
            row.addWidget(s)
            self.spins[side] = s
        row.addStretch(1)

    def _margin_spin(self) -> QDoubleSpinBox:
        s = QDoubleSpinBox()
        s.setRange(0.0, 95.0)
        s.setDecimals(1)
        s.setSingleStep(1.0)
        s.setSuffix("%")
        s.setStyleSheet(
            "QDoubleSpinBox { padding-right: 4px; }"
            "QDoubleSpinBox::up-button { width: 16px; }"
            "QDoubleSpinBox::down-button { width: 16px; margin-right: 16px; }"
        )
        # After the stylesheet: setAlignment on the spinbox doesn't survive the styled rebuild.
        s.lineEdit().setTextMargins(0, 0, 0, 0)
        s.setFixedWidth(84)
        no_wheel(s)
        s.valueChanged.connect(lambda _v: self.changed.emit())
        return s

    def value(self) -> dict[str, float]:
        return {side: float(s.value()) for side, s in self.spins.items()}

    def set_value(self, value) -> None:
        margins = normalize_crop_margins(value)
        for side, s in self.spins.items():
            s.blockSignals(True)
            s.setValue(float(margins[side]))
            s.blockSignals(False)


class ImagePrepSection(KnobSection):
    def __init__(self, help_cb, *, pp_cfg: dict):
        self._pp = pp_cfg
        super().__init__(t("preprocess_image_prep"), help_cb)

    def _build(self) -> None:
        pp = self._pp
        self.add_knob(
            "source_image_dir",
            line(
                str(pp.get("source_image_dir", DEFAULT_SOURCE_IMAGE_DIR)),
                placeholder=DEFAULT_SOURCE_IMAGE_DIR,
            ),
            t("preprocess_source_image_dir"),
            tooltip=t("preprocess_source_image_dir_tip"),
        )
        self.add_knob("path_scope", line(placeholder="data_group1"), t("path_scope"))
        self.add_knob(
            "preprocess_path_pattern",
            line(
                str(pp.get("preprocess_path_pattern", DEFAULT_PREPROCESS_PATH_PATTERN)),
                placeholder="*",
            ),
            t("preprocess_path_pattern"),
            tooltip=t("preprocess_path_pattern_tip"),
        )
        drop = checkbox(t("preprocess_drop_lowres"))
        drop.setChecked(bool(pp.get("drop_lowres_images", True)))
        self.add_knob(
            "drop_lowres_images",
            drop,
            t("preprocess_drop_lowres"),
            tooltip=t("preprocess_drop_lowres_tip"),
        )
        # min_pixels only applies when the filter is on (mirrors the CLI:
        # drop_lowres=false → --min_pixels 0) — gated via Knob.enabled_by.
        min_px = spin(
            0, 100_000_000, int(pp.get("min_pixels", DEFAULT_MIN_PIXELS)), step=50_000
        )
        min_px.setGroupSeparatorShown(True)
        self.add_knob("min_pixels", min_px, t("preprocess_min_pixels"))

        # Dual-use: preprocess resizes to these tiers, and train.py reads the same
        # value back to size the compile cache — this widget is the source of truth.
        self.target_res = _TargetResWidget(pp.get("target_res", DEFAULT_TARGET_RES))
        self.add_knob("target_res", self.target_res, t("preprocess_target_res"))
        self.add_knob(
            "resize_crop_anchor",
            _ResizeCropAnchorWidget(),
            t("resize_crop_anchor"),
            tooltip=t("resize_crop_anchor_tip"),
        )
        self.add_knob(
            "resize_crop_margins", _CropMarginsWidget(), t("resize_crop_margins")
        )
        # Free-fit is the only resize mode; only the max-ratio clamp is user-tunable.
        self.add_knob(
            "freefit_max_ratio",
            dspin(
                1.0,
                4.0,
                float(pp.get("freefit_max_ratio", DEFAULT_FREEFIT_MAX_RATIO)),
                step=0.25,
                decimals=2,
            ),
            t("preprocess_freefit_max_ratio"),
            tooltip=t("preprocess_freefit_max_ratio_tip"),
        )

    # ``resize_bucket_resos`` is a second value carried by the target-res
    # widget (its explicit WxH list), so it's read/written alongside the tiers
    # rather than through its own editor.

    def values(self) -> dict[str, object]:
        out = super().values()
        out["resize_bucket_resos"] = self.target_res.bucket_resos()
        return out

    def set_values(self, values: dict) -> None:
        if "target_res" in values:
            self.set_target_res(values["target_res"])
        rest = {k: v for k, v in values.items() if k != "target_res"}
        super().set_values(rest)
        if "resize_bucket_resos" in values:
            self.target_res.set_bucket_resos(values["resize_bucket_resos"])

    def set_target_res(self, values) -> None:
        """Set the tier checkboxes without emitting per-box change signals."""
        if values is None:
            selected = {1024}
        elif isinstance(values, (list, tuple, set)):
            selected = {int(v) for v in values}
        else:
            selected = {int(values)}
        for edge, box in self.target_res._boxes.items():
            box.blockSignals(True)
            box.setChecked(edge in selected)
            box.blockSignals(False)
        self.target_res.refresh_bucket_enabled()
