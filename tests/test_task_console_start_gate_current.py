"""Current TaskConsole start ownership at the prepare -> hardware boundary."""

from __future__ import annotations

import threading
import time

from zlc_neutral_atom.catalog import DefinitionKey
from zlc_neutral_atom.runtime.hosted_run import HostedRun


def test_stop_during_prepare_never_calls_the_hardware_starter() -> None:
    entered = threading.Event()
    release = threading.Event()
    starter_calls: list[object] = []
    frozen = object()

    def prepare(request):
        assert request is frozen
        entered.set()
        assert release.wait(2.0)
        return object()

    node = HostedRun(
        definition_key=DefinitionKey("test", "prepare-stop"),
        request=frozen,
        instance_id="prepare-stop-instance",
        dataset_output_declarations=(),
        prepare=prepare,
        qualify_output=lambda name: f"@logic/prepare-stop-instance/{name}",
        request_owner_wake=lambda: None,
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
        assert node.worker_idle
    finally:
        release.set()
        node.poll()
        node.shutdown()
