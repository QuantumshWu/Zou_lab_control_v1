"""Current TaskConsole start ownership at the prepare -> hardware boundary."""

from __future__ import annotations

from types import SimpleNamespace
import threading
import time

from zlc_workbench.task_console.run_bridge import ConsoleRunNode


def test_stop_during_prepare_never_calls_the_hardware_starter() -> None:
    entered = threading.Event()
    release = threading.Event()
    starter_calls: list[object] = []
    build_calls: list[dict[str, object]] = []
    frozen = object()

    spec = SimpleNamespace(
        key=SimpleNamespace(stable_definition_id="prepare-stop"),
        name="Pulse scan",
        title="Pulse scan",
        kind="measurement",
        declared_outputs=(),
        build_request=lambda values: build_calls.append(dict(values)),
    )

    def prepare(request):
        assert request is frozen
        entered.set()
        assert release.wait(2.0)
        return object()

    node = ConsoleRunNode(
        spec,
        {},
        prepare=prepare,
        request_owner_wake=lambda: None,
        frozen_request=frozen,
    )
    node.bind_starter(lambda command: starter_calls.append(command))
    try:
        node.start()
        assert entered.wait(2.0)
        node.cancel("operator stopped before hardware start")
        release.set()

        deadline = time.monotonic() + 2.0
        while not node.cancelled_before_start and time.monotonic() < deadline:
            node.poll()
            time.sleep(0.005)

        assert node.cancelled_before_start
        assert node.last_error is None
        assert not node.running
        assert not starter_calls
        assert not build_calls
        assert node.worker_idle
    finally:
        release.set()
        node.poll()
        node.shutdown()
