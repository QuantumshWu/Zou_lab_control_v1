"""Current TaskConsole status-strip ownership and priority contract."""

from __future__ import annotations

import inspect
from types import SimpleNamespace
import time

from zlc_frontend.qt_widgets import FluentStatusStrip, ensure_qt_app
from zlc_neutral_atom.catalog import DefinitionKey
from zlc_neutral_atom.runtime.hosted_run import HostedRun


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


def test_console_owns_one_explicit_priority_ladder() -> None:
    """Error > task > display advisory > idle, without a frontend domain helper."""

    from zlc_workbench.task_console.window import TaskConsole

    ensure_qt_app()
    def fail_prepare(_request):
        raise RuntimeError("device refused")

    fault = HostedRun(
        definition_key=DefinitionKey("test", "status-failure"),
        request={},
        instance_id="status-failure-instance",
        dataset_output_declarations=(),
        prepare=fail_prepare,
        qualify_output=lambda name: f"@logic/status-failure-instance/{name}",
        request_owner_wake=lambda: None,
    )
    fault.bind_starter(lambda command: command)
    surface = SimpleNamespace(
        cards=[],
        logic_nodes=[],
        _tick_data=SimpleNamespace(names=lambda: ()),
        _logic_nodes={-1: fault},
        _last_node={},
        _task_status_text="scanning",
        _note_display_drops=lambda: 3,
        summary=SimpleNamespace(setText=lambda _text: None),
        status_strip=FluentStatusStrip(),
    )
    surface._node_label = lambda node: TaskConsole._node_label(surface, node)
    try:
        fault.start()
        deadline = time.monotonic() + 2.0
        while fault.last_error is None and time.monotonic() < deadline:
            fault.poll()
            time.sleep(0.005)
        assert fault.last_error == "RuntimeError: device refused"
        TaskConsole._update_summary(surface)
        assert surface.status_strip.severity == "error"
        assert "device refused" in surface.status_strip.text()

        surface._logic_nodes.clear()
        TaskConsole._update_summary(surface)
        assert surface.status_strip.severity == "task"
        assert surface.status_strip.text() == "scanning"

        surface._task_status_text = None
        TaskConsole._update_summary(surface)
        assert surface.status_strip.severity == "warning"
        assert "dropped 3 event" in surface.status_strip.text()

        surface._note_display_drops = lambda: 0
        TaskConsole._update_summary(surface)
        assert surface.status_strip.severity == "info"
        assert surface.status_strip.text() == ""
    finally:
        fault.shutdown()
