"""The logic-node parameter editor wrapped with node identity chrome.

Pure Qt, per the placement axiom.
"""

from __future__ import annotations

from PyQt5 import QtCore, QtWidgets

from zlc_frontend.qt_widgets import (
    FluentLabel,
    FluentScrollArea,
    FluentSectionLabel,
    GREY,
    scaled_px,
)
from .logic_node_parameter_panel import LogicNodeParameterPanel
from zlc_frontend.qt_widgets import FormRuntimeContext

__all__ = ["LogicNodeEditor"]


class LogicNodeEditor(QtWidgets.QWidget):
    """One logic node's Edit tab (closable): its auto-generated PARAM FORM + Start /
    Stop + a status line.  NO curve fit -- fitting a curve is a plotter concern (add
    a Plot panel on the Monitor board pointed at the signals this node publishes).

    The param form reuses :class:`LogicNodeParameterPanel` (single-spec).  Each stable
    DefinitionKey resolves to one catalog spec and form, so the same form engine
    and Start/Stop signals drive every logic kind.  The
    Camera Measurement uses the same path for camera role, frames-per-cycle and
    repeat; hardware configuration stays in DeviceManager instead of being
    duplicated in a Measurement form."""

    start_requested = QtCore.pyqtSignal()
    stop_requested = QtCore.pyqtSignal()
    draft_changed = QtCore.pyqtSignal()

    def __init__(
        self,
        *,
        title: str,
        spec,
        initial_values,
        runtime: FormRuntimeContext,
        parent: QtWidgets.QWidget,
    ):
        if parent is None:
            raise TypeError("LogicNodeEditor requires its tab-stack parent")
        if not isinstance(runtime, FormRuntimeContext):
            raise TypeError("runtime must be FormRuntimeContext")
        super().__init__(parent)
        self.setStyleSheet("background: transparent;")

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = FluentScrollArea()
        outer.addWidget(scroll)
        page = QtWidgets.QWidget()
        page.setStyleSheet("background: transparent;")
        scroll.set_width_bounded_widget(page)
        col = QtWidgets.QVBoxLayout(page)
        m = scaled_px(10, minimum=6)
        col.setContentsMargins(m, m, m, m)
        col.setSpacing(scaled_px(6, minimum=4))

        col.addWidget(FluentSectionLabel(str(title)))
        # The auto-generated parameter form + Start / Stop (reused parameter panel,
        # which already carries start_requested(self) / stop_requested + the typed,
        # no-eval form).  Every node, including Camera, is driven by its real
        # FormSpec; no camera-only controls are injected by this editor.
        # Acquisition knobs are NOT injected here: a definition declares its own
        # physical parameters (for example a monitor's history depth), and the
        # editor renders exactly that form.  Deadlines remain internal Port/Run
        # mechanics, never generic Measurement inputs invented by the UI.
        self.form = LogicNodeParameterPanel(
            [spec] if spec is not None else [],
            parent=page,
            single=True,
            runtime=runtime,
        )
        self.form.seed_values(initial_values or {})
        self.form.start_requested.connect(lambda *_: self.start_requested.emit())
        self.form.stop_requested.connect(self.stop_requested.emit)
        self.form.draft_changed.connect(self.draft_changed.emit)
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
