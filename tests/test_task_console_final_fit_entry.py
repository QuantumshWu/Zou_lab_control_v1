"""TaskConsole's visible Fit action resolves one exact FINAL task result."""

from __future__ import annotations

import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class _Catalog:
    def __init__(self, spec) -> None:
        self._spec = spec
        self.experiment = None

    def specs(self, kind=None):
        if kind is None or kind == self._spec.kind:
            return (self._spec,)
        return ()

    def spec_named(self, name):
        return self._spec if name == self._spec.name else None


class _FitWindow:
    closed = False

    def __init__(self) -> None:
        self.focuses = 0

    def restore_window(self) -> None:
        self.focuses += 1


def test_panel_analyze_button_opens_and_focuses_the_exact_final_ref() -> None:
    from PyQt5 import QtCore, QtTest

    from zlc_data.console_records import LogicNodeConfig, PanelConfig
    from zlc_frontend.console_state import TaskConsoleState
    from zlc_frontend.form import FormFieldProps, FormSpec
    from zlc_frontend.qt_widgets import ensure_qt_app
    from zlc_neutral_atom.scan import PULSE_SCAN_TASK_KEY, ScanArtifactRef
    from zlc_workbench.task_console.catalog_bridge import (
        ConsoleNodeSpec,
        ConsoleSignalDecl,
    )
    from zlc_workbench.task_console.plot_bridge_console import TaskConsole

    application = ensure_qt_app()
    spec = ConsoleNodeSpec(
        key=PULSE_SCAN_TASK_KEY,
        kind="task",
        title="Pulse scan",
        description="one current pulse scan",
        form_spec=FormSpec(
            (FormFieldProps("enabled", "bool", "Enabled", default=True),)
        ),
        declared_outputs=(ConsoleSignalDecl("scan", "scan", "scan", ""),),
        build_request=lambda values: values,
    )
    reference = ScanArtifactRef("test-scan-repository", "a" * 64)
    opened = []

    def open_fit(source):
        window = _FitWindow()
        opened.append((source, window))
        return window

    state = TaskConsoleState(
        panels=(
            PanelConfig(
                kind="1d",
                title="scan result",
                signal="scan",
            ),
        ),
        logic=(LogicNodeConfig(kind="task", name=spec.name, title=spec.name),),
    )
    console = TaskConsole(
        state=state,
        catalog_view=_Catalog(spec),
        fit_window_factory=open_fit,
        window_px=(1000, 700),
    )
    try:
        console.show()
        application.processEvents()
        row = console.logic_nodes[0]
        card = console.cards[0]
        run = SimpleNamespace(final_result=reference)
        console._last_node[id(row)] = run
        console._sync_fit_analysis_entries()
        assert card.fit_analysis_button.isEnabled()

        QtTest.QTest.mouseClick(card.fit_analysis_button, QtCore.Qt.LeftButton)
        application.processEvents()
        assert [source for source, _window in opened] == [reference]

        QtTest.QTest.mouseClick(card.fit_analysis_button, QtCore.Qt.LeftButton)
        application.processEvents()
        assert len(opened) == 1
        assert opened[0][1].focuses == 1

        run.final_result = None
        console._sync_fit_analysis_entries()
        assert not card.fit_analysis_button.isEnabled()
    finally:
        assert console.shutdown()
        console.close()
        application.processEvents()
