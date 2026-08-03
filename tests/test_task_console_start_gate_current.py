"""Current shared Logic-node host stop-before-hardware boundary."""

from __future__ import annotations

from types import SimpleNamespace
import threading
import time

from zlc_neutral_atom.authoring import AuthoringSchema
from zlc_neutral_atom.catalog import DefinitionKey, LogicNodeDefinition
from zlc_neutral_atom.logic_node import LogicNodeDescriptor
from zlc_neutral_atom.processing.signal_plane import SignalDataPlane
from zlc_neutral_atom.runtime.hosted_run import LogicNodeHost


def test_stop_before_start_and_wait_never_calls_the_hardware_starter() -> None:
    entered = threading.Event()
    release = threading.Event()
    starter_calls: list[object] = []
    frozen = object()

    def bind_execute(request, _application_context):
        assert request is frozen

        def execute(context):
            entered.set()
            assert release.wait(2.0)

            def start_hardware():
                starter_calls.append(object())
                raise AssertionError("cancelled Logic node reached hardware start")

            return context.start_and_wait(start_hardware)

        return execute

    descriptor = LogicNodeDescriptor(
        api_name="prepare_stop",
        definition=LogicNodeDefinition(
            DefinitionKey("test", "prepare-stop"),
            "Prepare stop",
            "measurement",
        ),
        description="stop-before-start contract fixture",
        authoring_schema=AuthoringSchema(()),
        input_specs=(),
        outputs=(),
        build_request=lambda _values: frozen,
        bind_execute=bind_execute,
    )
    node = LogicNodeHost.create(
        descriptor,
        frozen,
        SimpleNamespace(signal_plane=SignalDataPlane()),
        "prepare-stop-instance",
        lambda: None,
    )
    try:
        node.start()
        assert entered.wait(2.0)
        node.cancel("operator stopped before hardware start")
        release.set()

        deadline = time.monotonic() + 2.0
        observation = node.poll()
        while not observation.terminal and time.monotonic() < deadline:
            time.sleep(0.005)
            observation = node.poll()

        assert observation.terminal
        assert observation.phase == "cancelled"
        assert observation.error is None
        assert not node.running
        assert not starter_calls
        assert node.worker_idle
    finally:
        release.set()
        node.poll()
        node.shutdown()
