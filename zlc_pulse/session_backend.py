"""Current pulse execution backend over the deployed streamer host sessions."""

from __future__ import annotations

from typing import Protocol

from fpga.pulse_streamer.host.image import StreamerParams

from .artifact import CompiledPulseArtifact
from .deployment import validate_artifact_for_deployment
from .target import PulseTarget


class CompiledPulseSession(Protocol):
    def prepare_compiled_artifact(self, artifact: CompiledPulseArtifact) -> None: ...

    def fire_compiled_artifact(self, artifact: CompiledPulseArtifact) -> None: ...

    def wait_done_compiled_artifact(
        self,
        artifact: CompiledPulseArtifact,
        timeout: float | None = None,
    ) -> bool: ...

    def safe_state(self) -> None: ...

    def current_snapshot(self) -> dict[str, object]: ...


class PulseStreamerSessionBackend:
    """Bind the current server contract to one concrete hardware-session owner."""

    def __init__(
        self,
        session: CompiledPulseSession,
        target: PulseTarget,
        params: StreamerParams,
        clock_hz: float,
    ) -> None:
        for method in (
            "prepare_compiled_artifact",
            "fire_compiled_artifact",
            "wait_done_compiled_artifact",
            "safe_state",
            "current_snapshot",
        ):
            if not callable(getattr(session, method, None)):
                raise TypeError(f"compiled pulse session is missing {method}()")
        self._session = session
        self._target = target
        self._params = params
        self._clock_hz = float(clock_hz)
        self._prepared_artifact: CompiledPulseArtifact | None = None

    def prepare(self, artifact: CompiledPulseArtifact) -> None:
        self._prepared_artifact = None
        validate_artifact_for_deployment(
            artifact,
            self._target,
            self._params,
            self._clock_hz,
        )
        self._session.prepare_compiled_artifact(artifact)
        self._prepared_artifact = artifact

    def fire(self, artifact: CompiledPulseArtifact) -> None:
        if artifact is not self._prepared_artifact:
            raise RuntimeError(
                "FIRE artifact is not the exact immutable artifact prepared by this backend"
            )
        self._session.fire_compiled_artifact(artifact)

    def wait_done(self, artifact: CompiledPulseArtifact, timeout: float | None) -> bool:
        if artifact is not self._prepared_artifact:
            raise RuntimeError(
                "completion artifact is not the exact immutable prepared artifact"
            )
        return bool(self._session.wait_done_compiled_artifact(artifact, timeout))

    def safe_state(self) -> None:
        try:
            self._session.safe_state()
        finally:
            self._prepared_artifact = None

    def snapshot(self) -> dict[str, object]:
        return dict(self._session.current_snapshot())


__all__ = ["CompiledPulseSession", "PulseStreamerSessionBackend"]
