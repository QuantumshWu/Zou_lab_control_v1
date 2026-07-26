"""Node-neutral public Readout notebook spine.

Only cross-capability capture/scan operations and readout binding live here.
Concrete installation authority is supplied by the static application
composition through :class:`ReadoutCoreHost`; this module imports no concrete
Logic-node adapter.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from zlc_data import OwnedSnapshot
from zlc_neutral_atom.capture.application import CaptureRequest, PreparedFiniteCapture
from zlc_neutral_atom.capture.artifact import CaptureArtifact
from zlc_neutral_atom.capture.reference import CaptureArtifactRef
from zlc_neutral_atom.devices.camera.contract import ReadoutBindingKey
from zlc_neutral_atom.installation import DeviceRef
from zlc_neutral_atom.runtime.run import RunHandle
from zlc_pulse import PulseDocument, PulseExecutionForm


class ReadoutCoreHost(Protocol):
    @property
    def readout_binding(self) -> ReadoutBindingKey | None: ...

    def bind_readout(self, binding: ReadoutBindingKey) -> "ReadoutCoreFacade": ...

    def load_readout_pulse(
        self,
        value: PulseDocument | str | Path,
    ) -> PulseDocument: ...

    def resolve_readout_camera_ref(self, requested: str | None) -> DeviceRef: ...

    def resolve_readout_sequencer_ref(self, requested: str | None) -> DeviceRef: ...

    def bind_finite_capture(self, request: CaptureRequest) -> PreparedFiniteCapture: ...

    def wait_readout_run(self, handle: RunHandle): ...

    def load_capture_artifact(self, reference: CaptureArtifactRef) -> CaptureArtifact: ...

    def materialize_capture_artifact(
        self,
        reference: CaptureArtifactRef,
    ) -> OwnedSnapshot: ...


class ReadoutCoreFacade:
    """Stable flat methods that remain meaningful without any one Logic node."""

    __slots__ = ()

    @property
    def _readout_core_host(self) -> ReadoutCoreHost:
        raise NotImplementedError

    def for_binding(
        self,
        binding: ReadoutBindingKey | str,
    ) -> "ReadoutCoreFacade":
        key = binding if isinstance(binding, ReadoutBindingKey) else ReadoutBindingKey(binding)
        current = self._readout_core_host.readout_binding
        if current is not None and key != current:
            raise ValueError("a bound readout facade cannot switch bindings")
        return self._readout_core_host.bind_readout(key)

    def _require_binding(self, actual: ReadoutBindingKey) -> None:
        if not isinstance(actual, ReadoutBindingKey):
            raise TypeError("actual readout binding must be ReadoutBindingKey")
        expected = self._readout_core_host.readout_binding
        if expected is not None and actual != expected:
            raise ValueError(
                f"bound readout facade requires {expected.value!r}, "
                f"not {actual.value!r}"
            )

    def prepare_capture(self, request: CaptureRequest) -> PreparedFiniteCapture:
        if not isinstance(request, CaptureRequest):
            raise TypeError("request must be CaptureRequest")
        return self._readout_core_host.bind_finite_capture(request)

    def capture_request(
        self,
        pulse: PulseDocument | str | Path,
        *,
        execution_form: PulseExecutionForm = PulseExecutionForm.STATIC_ONCE,
        camera_role: str | None = None,
        sequencer_role: str | None = None,
        trigger_channel: str | None = None,
        repeat_count: int = 1,
        readout_events_per_repeat: int | None = None,
        within_point_grouping: tuple[tuple[int, int], ...] | None = None,
    ) -> CaptureRequest:
        host = self._readout_core_host
        return CaptureRequest(
            host.load_readout_pulse(pulse),
            execution_form,
            host.resolve_readout_camera_ref(camera_role),
            host.resolve_readout_sequencer_ref(sequencer_role),
            trigger_channel,
            repeat_count,
            readout_events_per_repeat,
            within_point_grouping,
        )

    def capture(self, pulse: PulseDocument | str | Path, **kwargs) -> CaptureArtifactRef:
        prepared = self.prepare_capture(self.capture_request(pulse, **kwargs))
        return self._readout_core_host.wait_readout_run(prepared.start())

    def start_capture(self, pulse: PulseDocument | str | Path, **kwargs) -> RunHandle:
        return self.prepare_capture(self.capture_request(pulse, **kwargs)).start()

    def load_capture(self, reference: CaptureArtifactRef) -> CaptureArtifact:
        return self._readout_core_host.load_capture_artifact(reference)

    def materialize_capture(self, reference: CaptureArtifactRef) -> OwnedSnapshot:
        if not isinstance(reference, CaptureArtifactRef):
            raise TypeError("reference must be CaptureArtifactRef")
        return self._readout_core_host.materialize_capture_artifact(reference)

__all__ = ["ReadoutCoreFacade", "ReadoutCoreHost"]
