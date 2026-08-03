"""Current public Experiment lifetime and catalog-driven Node API contract."""

from __future__ import annotations

from pathlib import Path
import threading
import time

import pytest

import Zou_lab_control.api as zlc
import Zou_lab_control.api.facade as facade_impl


def _workspace(project_root: Path) -> zlc.WorkspacePaths:
    return zlc.WorkspacePaths.for_workspace(project_root.resolve())


def _connect(project_root: Path):
    return zlc.connect("virtual", workspace=_workspace(project_root))


class _ControlledWorkbenchHandle:
    """Structural WorkbenchHandle used to prove single-owner close ordering."""

    def __init__(self) -> None:
        self._acknowledged = threading.Event()
        self.wait_entered = threading.Event()
        self.request_count = 0

    @property
    def permanently_closed(self) -> bool:
        return self._acknowledged.is_set()

    def restore_window(self) -> None:
        raise AssertionError("close test must not restore its Workbench")

    def request_owner_close(self) -> None:
        self.request_count += 1

    def wait_owner_closed(self, timeout: float) -> bool:
        self.wait_entered.set()
        return self._acknowledged.wait(timeout)

    def acknowledge_close(self) -> None:
        self._acknowledged.set()


def test_public_root_projects_one_generic_api_per_discovered_node(
    tmp_path: Path,
) -> None:
    with _connect(tmp_path / "project") as experiment:
        assert experiment.nodes.names
        assert tuple(sorted(experiment.nodes.names)) == experiment.nodes.names
        for name in experiment.nodes.names:
            assert isinstance(getattr(experiment.nodes, name), zlc.NodeApi)
        for forbidden in ("devices", "readout", "camera", "sequencer"):
            assert not hasattr(experiment, forbidden)
        assert not hasattr(experiment.pulse, "sequencer")

        roots = experiment._workspace_paths()
        assert all(
            path.is_dir()
            for path in (
                roots.project_root,
                roots.pulses_root,
                roots.tasks_root,
                roots.runs_root,
                roots.figures_root,
            )
        )
        assert not (roots.project_root / "_output").exists()


def test_concurrent_close_has_one_teardown_owner(tmp_path: Path) -> None:
    experiment = _connect(tmp_path / "project")
    services = experiment._services
    handle = _ControlledWorkbenchHandle()
    assert experiment._open_workbench_handle(None, lambda: handle) is handle
    failures: list[BaseException] = []

    def close() -> None:
        try:
            experiment.close()
        except BaseException as error:
            failures.append(error)

    first = threading.Thread(target=close, daemon=False)
    second = threading.Thread(target=close, daemon=False)
    first.start()
    assert handle.wait_entered.wait(2.0)
    second.start()
    time.sleep(0.05)
    assert second.is_alive()
    with services.operation_lock:
        assert services.state == "CLOSING"

    handle.acknowledge_close()
    first.join(2.0)
    second.join(2.0)
    assert not first.is_alive() and not second.is_alive()
    assert failures == []
    assert handle.request_count == 2
    assert services.state == "CLOSED"


def test_close_retry_keeps_unacknowledged_workbench_owned(tmp_path: Path) -> None:
    experiment = _connect(tmp_path / "project")
    services = experiment._services
    handle = _ControlledWorkbenchHandle()
    with services.operation_lock:
        services.gui_handles["controlled"] = handle

    original_wait = handle.wait_owner_closed
    handle.wait_owner_closed = lambda _timeout: False  # type: ignore[method-assign]
    with pytest.raises(facade_impl._ResourceCleanupError, match="Workbench close"):
        experiment.close()
    assert services.state == "CLOSING"
    assert services.closing_gui_handles == (handle,)

    handle.wait_owner_closed = original_wait  # type: ignore[method-assign]
    handle.acknowledge_close()
    experiment.close()
    assert services.state == "CLOSED"


def test_connect_requires_explicit_workspace_and_current_config_type(
    tmp_path: Path,
) -> None:
    with pytest.raises(TypeError):
        zlc.connect("virtual")  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="config must be"):
        zlc.connect(  # type: ignore[arg-type]
            {},
            workspace=_workspace(tmp_path / "project"),
        )
