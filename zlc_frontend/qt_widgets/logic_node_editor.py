"""The logic-node parameter editor: a MeasurementPanel wrapped with node identity chrome.

Pure Qt, per the placement axiom.
"""

from __future__ import annotations

from PyQt5 import QtWidgets

from zlc_data.param_decl import ParamDecl

from .fluent import FluentLabel, GREY, scaled_px
from .measurement_panel import MeasurementPanel

__all__ = ["LogicNodeEditor"]


class LogicNodeEditor(QtWidgets.QWidget):
    """One logic node's Edit tab (closable): its auto-generated PARAM FORM + Start /
    Stop + a status line.  NO curve fit -- fitting a curve is a plotter concern (add
    a Plot panel on the Monitor board pointed at the signals this node publishes).

    The param form reuses :class:`MeasurementPanel` (single-spec): a camera /
    measurement / processor / task all expose ``.name`` + ``.params`` (ParamDecls),
    so the same form engine + Start / Stop signals drive every logic kind.  The
    camera live Measurement's spec is ``readout.camera_spec()`` (its ParamDecls are
    the camera's exposure / frames-per-cycle)."""

    def __init__(self, row: "LogicNodeRow", console: "TaskConsole", spec, parent=None):
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
        # no-eval form).  A spec drives a real ParamDecl form; the camera (spec is
        # None) shows nothing here but Start/Stop still build/run the camera node.
        # ``repeat`` (0 = ∞) is the ONE MEASUREMENT-layer acquisition knob -- the plot can NEVER tell a
        # measurement how many times to run (#H3l).  It is a DECLARED ParamDecl auto-injected into the
        # SAME auto-form as every other param (never a hand-placed widget; 0 is the ∞ sentinel, the same
        # semantics as the scan-repeat count -- no separate Free-run toggle).  An acquisition node
        # (measurement / camera) gets it; a processor / task does not.  How the repeats are DISPLAYED is
        # the PLOT's "repeat mode" Setting.  A camera defaults to ∞ (repeat=0, a live monitor); a scan
        # defaults to a single finite sweep (repeat=1).
        acquisition = (_acquisition_param_decls(repeat_default=(0 if row.node.kind == "camera" else 1))
                       if row.node.kind in ("measurement", "camera") else ())
        names_provider = getattr(console, "_signal_names", None)
        if row.node.kind == "processor" and callable(names_provider):
            # A reactive processor's source picker must not offer the node's OWN outputs -- picking
            # one is the self-feedback loop Processor.__init__ rejects loud at Start; hide it here
            # so the misclick cannot happen (declared keys, #prebind, so it holds before the first
            # run too).  This filter applies only to PROCESSOR rows; acquisition measurements use
            # their own explicitly bounded input contract (see ``_reactive_ring``).
            def names_provider(_base=names_provider, _console=console, _row=row):
                own = {str(k) for k in _console._declared_signal_keys(_row)}
                return [n for n in _base() if str(n) not in own]
        self.form = MeasurementPanel([spec] if spec is not None else [], single=True,
                                     signals_provider=names_provider,
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
