"""Current TaskConsole status-strip ownership and priority contract."""

from __future__ import annotations

import inspect

from zlc_frontend.qt_widgets import FluentStatusStrip, ensure_qt_app


def test_console_has_no_external_running_node_injection_seam() -> None:
    from zlc_workbench.task_console.window import TaskConsole, show_task_console

    assert "running_nodes" not in inspect.signature(TaskConsole).parameters
    assert "running_nodes" not in inspect.signature(show_task_console).parameters
    assert (
        "_replacement_running_nodes"
        not in inspect.signature(TaskConsole.load_state).parameters
    )


def test_status_strip_is_fixed_height_and_rejects_unknown_severity() -> None:
    import pytest

    ensure_qt_app()
    strip = FluentStatusStrip()
    before = strip.height()
    strip.show_message("x" * 4000, severity="warning")

    assert strip.height() == before
    assert strip.text() == "x" * 4000
    assert strip.severity == "warning"
    with pytest.raises(ValueError, match="unknown severity"):
        strip.show_message("bad", severity="foreign")
