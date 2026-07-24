"""The logic-node parameter editor: a MeasurementPanel wrapped with node identity chrome.

Pure Qt, per the placement axiom.
"""

from __future__ import annotations

from PyQt5 import QtWidgets

from zlc_data.param_decl import ParamDecl

from .fluent import (
    FluentLabel,
    FluentScrollArea,
    FluentSectionLabel,
    GREY,
    scaled_px,
)
from .measurement_panel import MeasurementPanel

__all__ = ["LogicNodeEditor"]


class LogicNodeEditor(QtWidgets.QWidget):
    """One logic node's Edit tab (closable): its auto-generated PARAM FORM + Start /
    Stop + a status line.  NO curve fit -- fitting a curve is a plotter concern (add
    a Plot panel on the Monitor board pointed at the signals this node publishes).

    The param form reuses :class:`MeasurementPanel` (single-spec): a camera /
    measurement / processor / task all expose ``.name`` + ``.params`` (ParamDecls),
    so the same form engine + Start / Stop signals drive every logic kind.  The
    Camera Measurement uses the same path for camera role, frames-per-cycle and
    repeat; hardware configuration stays in DeviceManager instead of being
    duplicated in a Measurement form."""

    def __init__(
        self,
        row: "LogicNodeRow",
        console: "TaskConsole",
        spec,
        parent=None,
        *,
        signal_names_providers=None,
    ):
        super().__init__(parent)
        self.row = row
        self.console = console
        self.spec = spec
        self.setStyleSheet("background: transparent;")

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = FluentScrollArea()
        scroll.setWidgetResizable(True)
        outer.addWidget(scroll)
        page = QtWidgets.QWidget()
        page.setStyleSheet("background: transparent;")
        scroll.setWidget(page)
        col = QtWidgets.QVBoxLayout(page)
        m = scaled_px(10, minimum=6)
        col.setContentsMargins(m, m, m, m)
        col.setSpacing(scaled_px(6, minimum=4))

        col.addWidget(FluentSectionLabel(row.node.title))
        # The auto-generated parameter form + Start / Stop (reused MeasurementPanel,
        # which already carries start_requested(self) / stop_requested + the typed,
        # no-eval form).  Every node, including Camera, is driven by its real
        # ParamDecl form; no camera-only controls are injected by this editor.
        # Acquisition knobs are NOT injected here: a definition declares its own
        # physical parameters (for example a monitor's history depth), and the
        # editor renders exactly that form.  Deadlines remain internal Port/Run
        # mechanics, never generic Measurement inputs invented by the UI.
        acquisition = ()
        names_provider = getattr(console, "_signal_names", None)
        field_providers = dict(signal_names_providers or {})
        if row.node.kind == "processor" and callable(names_provider):
            # A reactive processor's source picker must not offer the node's OWN outputs -- picking
            # one is the self-feedback loop Processor.__init__ rejects loud at Start; hide it here
            # so the misclick cannot happen (declared keys, #prebind, so it holds before the first
            # run too).  This filter applies only to PROCESSOR rows; acquisition measurements use
            # their own explicitly bounded input contract (see ``_reactive_ring``).
            def names_provider(_base=names_provider, _console=console, _row=row):
                own = {str(k) for k in _console._declared_signal_keys(_row)}
                return [n for n in _base() if str(n) not in own]
            for key, provider in tuple(field_providers.items()):
                if not callable(provider):
                    continue

                def without_own(
                    _base=provider,
                    _console=console,
                    _row=row,
                ):
                    own = {str(k) for k in _console._declared_signal_keys(_row)}
                    return [name for name in _base() if str(name) not in own]

                field_providers[key] = without_own
        self.form = MeasurementPanel([spec] if spec is not None else [], single=True,
                                     signals_provider=names_provider,
                                     signal_providers=field_providers,
                                     sources_provider=getattr(console, "_signal_providers", None),
                                     formats_provider=getattr(console, "_signal_formats", None),
                                     short_names_provider=getattr(console, "_signal_short_names", None),
                                     acquisition_params=acquisition)
        self.form.seed_values(row.node.values or {})
        self.form.start_requested.connect(lambda *_: self.console._start_logic_node(self.row))
        self.form.stop_requested.connect(lambda: self.console._stop_logic_node(self.row))
        col.addWidget(self.form)
        # (The node's published-signals + shapes are shown on its Logic-tab ROW card,
        # the single place for that legend -- not duplicated here.)
        col.addStretch(1)

    def collect_values(self) -> dict:
        return self.form.collect_values()                # repeat (0 = ∞) comes back like any param

    def set_running(self, running: bool) -> None:
        self.form.set_running(running)

    def set_status(self, text: str, *, error: bool) -> None:
        self.form.set_status(text, error=error)

    def teardown(self) -> None:
        # No matplotlib resources here (a logic node never plots), so teardown is a
        # no-op -- present so the console can treat it like a PanelEditor.
        pass

    def refresh_on_show(self) -> None:
        """When switching back to this Edit tab, refresh the form's dynamic combos so a
        signal that was not yet published when this tab was last open now shows up.
        Delegates to the form's own ``refresh_on_show`` (the one hook every form honours)."""
        hook = getattr(self.form, "refresh_on_show", None)
        if callable(hook):
            hook()
