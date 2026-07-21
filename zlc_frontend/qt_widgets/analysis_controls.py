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

from zlc_data.vocabulary import ANALYSIS_ACTIONS

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




#: How each action of the shared vocabulary reads in the menu.  The vocabulary
#: itself lives in :data:`zlc_data.vocabulary.ANALYSIS_ACTIONS` -- which names this
#: menu as its consumer -- so this maps wording only and can never offer a verb the
#: dispatch side does not know.
_ACTION_LABELS = {"fit": "curve fit", "roi": "ROI crop"}

#: Always offered, and the only action every panel can honour on its own: a drag
#: marks a region and nothing consumes it.
_NO_ACTION = ("none", "none")


def _wired_analysis_actions(card) -> tuple[tuple[str, str], ...]:
    """The analysis actions this panel can actually carry out, as (value, label).

    The card is ASKED (``analysis_actions_provider``) rather than assumed: the host
    owns the catalog, so only the host knows whether the seam an action needs is
    present.  A menu entry that fails on click is worse than an absent one -- the
    operator would discover the capability is missing exactly when relying on it.

    Order comes from the vocabulary, not from the host's answer, so the menu reads
    the same everywhere no matter what a given host supports.
    """

    provider = getattr(card, "analysis_actions_provider", None)
    available = set(provider() or ()) if callable(provider) else set()
    return (_NO_ACTION,) + tuple(
        (action, _ACTION_LABELS.get(action, action))
        for action in ANALYSIS_ACTIONS if action in available)


def _general_fit_models_for_kind(kind: str):
    """The fit models a plot kind admits, as ``(model_id, display_name)`` pairs.

    Both halves are declared data, so this only PAIRS them -- it holds no list of
    model names of its own.  A kind states its ``render_family``
    (:data:`zlc_data.plot_kind.PLOT_KIND_SPEC_BY_KEY`); a model states how many
    independent axes it needs (:attr:`FitModelDefinition.axis_requirements`, which
    its own validator restricts to one or two).  One axis is a curve, two is an
    image -- that is the whole rule, and a model added to the catalogue appears
    here without this function changing.

    ``render_family == "auto"`` (the site map, which is image-family only when a
    background frame is supplied) offers BOTH: which one it turns out to be is a
    property of the figure, not of the kind, and the fit itself re-checks the
    requirement against the real axes.  Offering too much and letting the fit
    refuse is honest; hiding a model the data would have accepted is not.

    Imports are LAZY on purpose -- see this module's docstring: the top-level
    import graph must stay free of anything but Qt.
    """

    from zlc_data.fit_model import fit_model_catalog
    from zlc_data.plot_kind import PLOT_KIND_SPEC_BY_KEY

    spec = PLOT_KIND_SPEC_BY_KEY.get(str(kind or ""))
    family = "auto" if spec is None else str(spec.render_family)
    wanted = {1} if family == "1D" else {2} if family == "2D" else {1, 2}
    return tuple(
        (definition.model_id, definition.display_name)
        for definition in fit_model_catalog()
        if len(definition.axis_requirements) in wanted
    )


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

    def __init__(self, card, *, surface: str = "setting", label_w: int | None = None,
                 parent=None) -> None:
        super().__init__(parent)
        self.card = card
        self.surface = str(surface)
        self.action_combo = None
        self.model_combo = None
        self.fix_seed = None
        self.result_label = None
        self.fit_button = None
        self.clear_button = None

        label_w = scaled_px(96, minimum=72) if label_w is None else int(label_w)
        column = QtWidgets.QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(scaled_px(4, minimum=2))

        kind = str(getattr(getattr(card, "config", None), "kind", "") or "")
        models = _general_fit_models_for_kind(kind)

        # What a drag on this panel MEANS.  Every entry carries the action string the
        # card's one mutator acts on, so the menu cannot offer a verb the handler does
        # not know.
        self.action_combo = FluentComboBox()
        for value, label in _wired_analysis_actions(card):
            if value == "fit" and not models:
                continue            # a kind with no admissible model cannot offer "fit"
            self.action_combo.addItem(label, value)
        self.action_combo.setToolTip(
            "What a drag-selection on this panel does:\n"
            "  none = select only (the selection is still recorded)\n"
            "  curve fit = fit the model below over the selection\n"
            "  ROI crop = crop the source to the selection")
        self.action_combo.currentIndexChanged.connect(self._on_action)
        column.addWidget(FluentSettingRow("analysis", self.action_combo, label_width=label_w))

        if models:
            self.model_combo = FluentComboBox()
            for model_id, display_name in models:
                self.model_combo.addItem(display_name, model_id)
            self.model_combo.setToolTip(
                "The fit model.  Stays editable whether or not the fit is on, so a seed\n"
                "can be entered BEFORE turning it on.")
            self.model_combo.currentIndexChanged.connect(self._on_model)
            column.addWidget(FluentSettingRow("fit model", self.model_combo, label_width=label_w))

            self.fix_seed = _FitFixSeedEditor()
            self.fix_seed.changed.connect(self._on_fix_seed)
            column.addWidget(FluentSettingRow("fix / seed", self.fix_seed, label_width=label_w))

            # The Edit tab gets explicit buttons; the Setting popup drives the fit from the
            # action chooser alone.  This is the ONLY thing surface changes -- the controls,
            # their state and their handlers are identical, which is what keeps the two
            # surfaces views of one fit rather than two.
            if self.surface == "edit":
                self.fit_button = FluentButton("Fit", color=ACCENT)
                self.fit_button.clicked.connect(self.do_fit)
                self.clear_button = FluentButton("Clear", color=GREY)
                self.clear_button.clicked.connect(self.clear_fit)
                buttons = QtWidgets.QWidget()
                row = QtWidgets.QHBoxLayout(buttons)
                row.setContentsMargins(0, 0, 0, 0)
                row.setSpacing(scaled_px(6, minimum=4))
                row.addWidget(self.fit_button, 0)
                row.addWidget(self.clear_button, 0)
                row.addStretch(1)
                column.addWidget(FluentSettingRow("", buttons, label_width=label_w))

            self.result_label = FluentLabel("")
            self.result_label.setStyleSheet(
                "color: %s; background: transparent; border: none;" % GREY)
            self.result_label.setSizePolicy(QtWidgets.QSizePolicy.Ignored,
                                            QtWidgets.QSizePolicy.Preferred)
            column.addWidget(FluentSettingRow("result", self.result_label, label_width=label_w))

        self.derive()

    @property
    def empty(self) -> bool:
        """True when this panel kind offers no analysis at all, so the host can drop us.

        With only ``none`` on offer there is nothing to choose, so the section is
        not worth a row: the host drops it rather than showing a chooser whose
        single entry means "do nothing".
        """

        return self.action_combo is None or self.action_combo.count() <= 1


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
