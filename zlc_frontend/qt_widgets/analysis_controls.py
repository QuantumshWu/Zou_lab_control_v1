"""The per-panel Analysis controls: one fit + ROI control set, built once, embedded twice.

The Setting popup and the Edit tab both show "what does a drag-selection on this panel do?".
They are the same controls, and the reason they cannot drift is that neither owns any state:
every widget here DERIVES from the card's ``config.params['fit_request']`` (the fit) plus
``['selection_action']`` (the ROI), through one shared derive.  That is why this is a single
module rather than two similar ones.

Pure Qt, per the placement axiom: this file may import PyQt5 and may not import matplotlib.

It does ask the plot-kind registry two capability questions -- which curve-fit models a kind
offers, and whether a kind offers ROI -- and that registry currently lives beside the render
layer, so those two imports are LAZY, inside the functions that need them.  The module-level
import graph therefore stays free of the render layer, which is what the top-level purity rule
protects.  The residual coupling is real and named: panel-kind capability declarations want a
toolkit-free home of their own.

There is a concrete way out for the fit half, found while classifying this shell: the lookup
only needs a kind's ``render_family``, and reading that from the kind SPECS rather than from
``PLOT_KIND_BY_KEY`` (whose values carry the plot CLASSES, which is what drags the render layer
in) would leave a function with no frontend dependency at all -- a ``zlc_data`` citizen.  That
is a behaviour-preserving rewrite, not a move, so it belongs to its own round rather than being
smuggled into a pure relocation.
"""

from __future__ import annotations

from typing import Mapping

from PyQt5 import QtCore, QtWidgets

from .fluent import (
    FluentButton,
    FluentComboBox,
    FluentLabel,
    FluentLineEdit,
    FluentSettingRow,
    scaled_px,
    signals_blocked as _signals_blocked,
)
from .style import ACCENT, GREY

__all__ = [
    "AnalysisControls",
    "_FitFixSeedEditor",
    "_apply_analysis_state_to_widgets",
    "_general_fit_models_for_kind",
]


def _general_fit_models_for_kind(kind: str) -> list:
    """The general curve-fit models offered for a panel of ``kind``: resolve kind -> render_family ->
    the ONE :func:`live.general_fit_models` capability table (keyed off ``render_family``, the single
    source).  A kind's OWN built-in fit no longer suppresses the general one -- a histogram
    (render_family ``1d``) is offered the 1-D family here ALONGSIDE its bimodal ``fit`` knob, while a
    site map (render_family ``auto``) is offered nothing (its occupancy rings are not a fittable
    curve).  Both Setting and Edit read this ONE adapter; the fit engine stays the sole owner of model
    keys, families, and parameter names."""
    # Named at its real owner, not at the legacy forwarding module the shell used to reach it
    # through: naming a shim as the source is how a second apparent source is born.
    from ..live_plot.live import PLOT_KIND_BY_KEY, general_fit_models
    pk = PLOT_KIND_BY_KEY.get(str(kind))
    return general_fit_models(pk.render_family) if pk is not None else []


class _FitFixSeedEditor(QtWidgets.QWidget):
    """Compact typed per-parameter fix/seed editor for a fit model -- the TYPED replacement for the
    reference GUI's free-text ``Fit params:`` box.  For each parameter of the selected model it shows a
    narrow ``fix`` field (clamp the parameter at this value) and a ``seed`` field (initial guess); the
    values are reported as :attr:`FitRequest.fixed` / :attr:`FitRequest.initial` through the ONE
    :func:`build_fit_request`, so "fixing a parameter" works again with no ``eval``.  ``initial`` is
    only reported when EVERY parameter carries a seed (the core takes a full seed vector or none); a
    fixed parameter always overrides its seed.  Reused verbatim on both the Setting popup and the Edit
    tab, so the two surfaces can never diverge."""

    changed = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._grid = QtWidgets.QGridLayout(self)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setHorizontalSpacing(scaled_px(6, minimum=4))
        self._grid.setVerticalSpacing(scaled_px(4, minimum=2))
        self._rows: dict[str, tuple] = {}
        self._model = ""

    def set_model(self, model: str) -> None:
        """Rebuild the rows for ``model``'s parameters (a no-op if the model is unchanged)."""
        model = str(model or "")
        if model == self._model and self._rows:
            return
        self._model = model
        while self._grid.count():
            item = self._grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._rows = {}
        if not model:
            return
        from zlc_data.curve_fitting import fit_model
        try:
            names = fit_model(model).names
        except Exception:
            names = ()
        for row, name in enumerate(names):
            label = FluentLabel(name)
            label.setStyleSheet(f"color: {GREY}; background: transparent; border: none;")
            fix = FluentLineEdit("")
            fix.setPlaceholderText("fix")
            fix.setFixedWidth(scaled_px(64, minimum=48))
            fix.setToolTip(f"Clamp {name} at this value (leave blank to fit it freely)")
            seed = FluentLineEdit("")
            seed.setPlaceholderText("seed")
            seed.setFixedWidth(scaled_px(64, minimum=48))
            seed.setToolTip(f"Initial guess for {name} (used only when every parameter is seeded)")
            fix.editingFinished.connect(self.changed)
            seed.editingFinished.connect(self.changed)
            self._grid.addWidget(label, row, 0)
            self._grid.addWidget(fix, row, 1)
            self._grid.addWidget(seed, row, 2)
            self._rows[name] = (fix, seed)

    def values(self):
        """Return ``(fixed: dict[name -> float], initial: tuple | None)`` parsed from the fields."""
        fixed: dict[str, float] = {}
        seeds: list[float] = []
        all_seeded = bool(self._rows)
        for name, (fix, seed) in self._rows.items():
            fix_text = fix.text().strip()
            if fix_text:
                try:
                    fixed[name] = float(fix_text)
                except ValueError:
                    pass
            seed_text = seed.text().strip()
            try:
                seeds.append(float(seed_text))
            except ValueError:
                all_seeded = False
                seeds.append(0.0)
        initial = tuple(seeds) if (all_seeded and seeds) else None
        return fixed, initial

    def seed_from_request(self, request) -> None:
        """Re-seed the fields from a stored request's ``fixed`` / ``initial`` (blocking signals)."""
        from zlc_data.curve_fitting import FitRequest
        req = (FitRequest.from_dict(request) if isinstance(request, Mapping)
               else request if isinstance(request, FitRequest) else None)
        initial = None if req is None else req.initial
        fixed = {} if req is None else dict(req.fixed)
        for index, (name, (fix, seed)) in enumerate(self._rows.items()):
            with _signals_blocked(fix):
                fix.setText("" if name not in fixed else f"{float(fixed[name]):g}")
            with _signals_blocked(seed):
                seed.setText("" if (initial is None or index >= len(initial))
                             else f"{float(initial[index]):g}")


def _apply_analysis_state_to_widgets(card, *, action_combo=None, model_combo=None,
                                     fix_seed=None, result_label=None) -> None:
    """Derive ONE surface's Analysis controls from the card's state -- ``config.params['fit_request']``
    presence (the fit) + ``selection_action`` (the ROI).  Shared VERBATIM by the Setting popup and the
    Edit tab so the two surfaces are pure views of the same single source and can never disagree (#8).
    Signals are blocked while re-seeding so a derive never re-fires the action/model handlers."""
    params = card.config.params
    request = params.get("fit_request")
    active = "fit" if request else str(params.get("selection_action") or "none")
    model = str(request.get("model")) if isinstance(request, Mapping) and request.get("model") else ""
    if action_combo is not None:
        index = action_combo.findData(active)
        with _signals_blocked(action_combo):
            action_combo.setCurrentIndex(index if index >= 0 else 0)
    if model_combo is not None:
        if model:
            model_index = model_combo.findData(model)
            if model_index >= 0:
                with _signals_blocked(model_combo):
                    model_combo.setCurrentIndex(model_index)
        # ONE rule for BOTH surfaces: the model picker + fix/seed stay ENABLED regardless of the action, so
        # a seed can be pre-entered BEFORE turning the fit on (the #6 pre-seed workflow -- previously the
        # Setting popup gated them on ``active=='fit'`` and opened the fit with an empty seed).
        model_combo.setEnabled(True)
    if fix_seed is not None:
        current_model = model_combo.currentData() if model_combo is not None else model
        fix_seed.set_model(str(current_model or ""))
        fix_seed.seed_from_request(request)
        fix_seed.setEnabled(True)
    if result_label is not None:
        text = card._fit_result_text()
        if result_label.text() != text:                # only setText on a real change (no per-tick thrash)
            result_label.setText(text)


class AnalysisControls(QtWidgets.QWidget):
    """The ONE fit + ROI 'Analysis' control set -- built ONCE here and embedded by BOTH the Setting popup
    and the Edit tab.  ``surface='setting'|'edit'`` only tweaks layout (whether the explicit Fit/Clear
    buttons show); every control DERIVES from the card's ``config.params['fit_request']`` (the fit) plus
    ``['selection_action']`` (the ROI) through :func:`_apply_analysis_state_to_widgets`, so no widget owns
    a private 'is fitting' and the two surfaces can never disagree (#8).

    Exposes ``action_combo`` / ``model_combo`` / ``fix_seed`` / ``result_label`` (each ``None`` when the
    panel family offers no fit / ROI); the host aliases the test-keyed attribute names onto these
    (``analysis_combo`` / ``fit_model_combo`` / ``fit_fix_seed`` / ``fit_result_label`` on the card,
    ``fit_combo`` / ``ed_fix_seed`` on the editor)."""

    def __init__(self, card, *, surface: str, label_w: int, parent=None):
        super().__init__(parent)
        # Lazy for the same reason as the fit-model adapter above: the ROI capability table sits with
        # the plot kinds, and this file must not pull the render layer in at import time.
        from ..live_plot.live import kind_supports_roi
        self.setStyleSheet("background: transparent;")
        self.card = card
        self.surface = str(surface)
        self.action_combo = self.model_combo = self.fix_seed = None
        self.result_label = self.fit_button = self.clear_button = None
        models = _general_fit_models_for_kind(card._param_kind())
        offers_roi = kind_supports_roi(card._param_kind())
        col = QtWidgets.QVBoxLayout(self)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(scaled_px(6, minimum=4))
        self.empty = not (models or offers_roi)
        if self.empty:
            return
        # action combo (both surfaces) -- the action VOCABULARY is the AnalysisProcessor's own
        # ANALYSIS_ACTIONS (single source), each entry gated by this panel kind's capability.
        from zlc_data.vocabulary import ANALYSIS_ACTIONS
        self.action_combo = FluentComboBox()
        self.action_combo.addItem("none", "none")
        labels = {"fit": "curve fit", "roi": "ROI"}
        offered = {"fit": bool(models), "roi": offers_roi}
        for action in ANALYSIS_ACTIONS:
            if offered.get(action):
                self.action_combo.addItem(labels.get(action, action), action)
        self.action_combo.setToolTip(
            "What a drag-selection on this panel does:\n"
            "  none      = just report the selected points"
            + ("\n  curve fit = fit the chosen model to the selection (the result overlays the plot;\n"
               "              a 2-D image centre fit ALSO publishes fit_x0/fit_y0/... as hub signals)"
               if models else "")
            + ("\n  ROI       = reduce the selected region to one scalar (publishes roi_value; also\n"
               "              roi_frame -- an image crop, or a 1-D / distribution / site sub-view)"
               if offers_roi else ""))
        self.action_combo.currentIndexChanged.connect(self._on_action)
        col.addWidget(FluentSettingRow("action", self.action_combo, label_width=label_w))
        if models:
            self.model_combo = FluentComboBox()
            for model in models:
                self.model_combo.addItem(model.formula, model.key)
            self.model_combo.setToolTip(
                "Curve-fit model for this panel's plot family (a 2d image offers the 2D-Gaussian\n"
                "'2D center'; a 1d / monitor / distribution offers the peak/decay models).")
            self.model_combo.currentIndexChanged.connect(self._on_model)
            if self.surface == "edit":
                # Edit adds explicit Fit / Clear buttons beside the picker (its historical entry); the
                # action combo above is the SAME entry the Setting popup uses.
                self.fit_button = FluentButton("Fit", color=ACCENT)
                self.fit_button.clicked.connect(self.do_fit)
                self.clear_button = FluentButton("Clear", color=GREY)
                self.clear_button.clicked.connect(self.clear_fit)
                host = QtWidgets.QWidget()
                hl = QtWidgets.QHBoxLayout(host)
                hl.setContentsMargins(0, 0, 0, 0)
                hl.setSpacing(scaled_px(6, minimum=4))
                hl.addWidget(self.model_combo, 0)
                hl.addStretch(1)
                hl.addWidget(self.fit_button, 0)
                hl.addWidget(self.clear_button, 0)
                col.addWidget(FluentSettingRow("model", host, label_width=label_w))
            else:
                col.addWidget(FluentSettingRow("model", self.model_combo, label_width=label_w))
            self.fix_seed = _FitFixSeedEditor()
            self.fix_seed.changed.connect(self._on_fix_seed)
            col.addWidget(FluentSettingRow("fix / seed", self.fix_seed, label_width=label_w))
            self.result_label = FluentLabel("not fitted")
            self.result_label.setWordWrap(True)
            self.result_label.setStyleSheet(f"color: {GREY}; background: transparent; border: none;")
            col.addWidget(FluentSettingRow("result", self.result_label, label_width=label_w))
        self.derive()

    def derive(self) -> None:
        """Re-seed every control from the card's state (fit_request presence + selection_action) -- the
        ONE derive both surfaces share, so they stay pure views of the single source (#8)."""
        _apply_analysis_state_to_widgets(
            self.card, action_combo=self.action_combo, model_combo=self.model_combo,
            fix_seed=self.fix_seed, result_label=self.result_label)

    def set_result(self, text: str) -> None:
        """Update the result line -- only on a real text change, so a per-tick push never thrashes the
        label (and never re-lays-out an open Setting popup for an unchanged string)."""
        if self.result_label is not None and self.result_label.text() != text:
            self.result_label.setText(text)

    # --------------------------------------------------------------- handlers (ONE copy, both surfaces)
    def _on_action(self, _index: int) -> None:
        if self.action_combo is not None:
            self.card._select_analysis_action(
                self.action_combo.currentData(), model_combo=self.model_combo, fix_seed=self.fix_seed)

    def _on_model(self, _index: int) -> None:
        if self.model_combo is None:
            return
        if self.fix_seed is not None:
            self.fix_seed.set_model(str(self.model_combo.currentData() or ""))
        if self.card.config.params.get("fit_request"):        # re-apply the active fit with the new model
            self.card.set_fit_request(self.card._build_fit_request_from_widgets(
                self.model_combo, self.fix_seed, self.card.current_selection()))

    def _on_fix_seed(self) -> None:
        if self.card.config.params.get("fit_request"):        # re-apply the active fit with the new fix/seed
            self.card.set_fit_request(self.card._build_fit_request_from_widgets(
                self.model_combo, self.fix_seed, self.card.current_selection()))

    def do_fit(self) -> None:
        """The Edit-tab explicit Fit action: build a request from THIS surface's model + fix/seed + the
        current selection and land it through the ONE card mutator."""
        if self.model_combo is None or not self.model_combo.currentData():
            self.card.set_status("pick a fit model", error=True)
            return
        self.card.set_fit_request(self.card._build_fit_request_from_widgets(
            self.model_combo, self.fix_seed, self.card.current_selection()))

    def clear_fit(self) -> None:
        self.card.set_fit_request(None)
