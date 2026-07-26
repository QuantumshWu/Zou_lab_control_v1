"""Current TaskConsole status-strip ownership and priority contract."""

from __future__ import annotations

import inspect
from types import SimpleNamespace
import time

from zlc_frontend.qt_widgets import FluentStatusStrip, ensure_qt_app
from zlc_workbench.task_console.run_bridge import ConsoleRunNode


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
    spec = SimpleNamespace(
        key=SimpleNamespace(stable_definition_id="status-failure"),
        name="Camera",
        kind="measurement",
        artifact_outputs=(),
        build_request=lambda values: values,
        outputs_for=lambda request: (),
    )

    def fail_prepare(_request):
        raise RuntimeError("device refused")

    fault = ConsoleRunNode(
        spec,
        {},
        instance_id="status-failure-instance",
        instance_label="Camera",
        prepare=fail_prepare,
        request_owner_wake=lambda: None,
    )
    fault.bind_starter(lambda command: command)
    surface = SimpleNamespace(
        cards=[],
        _tick_data=SimpleNamespace(names=lambda: ()),
        _logic_nodes={-1: fault},
        _task_status_text="scanning",
        _note_display_drops=lambda: 3,
        _node_label=TaskConsole._node_label,
        summary=SimpleNamespace(setText=lambda _text: None),
        status_strip=FluentStatusStrip(),
    )
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
