"""Current pulse execution backend over the deployed streamer host sessions."""

from __future__ import annotations

from typing import Protocol

from .artifact import CompiledPulseArtifact


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

    def __init__(self, session: CompiledPulseSession) -> None:
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

    def prepare(self, artifact: CompiledPulseArtifact) -> None:
        self._session.prepare_compiled_artifact(artifact)

    def fire(self, artifact: CompiledPulseArtifact) -> None:
        self._session.fire_compiled_artifact(artifact)

    def wait_done(self, artifact: CompiledPulseArtifact, timeout: float | None) -> bool:
        return bool(self._session.wait_done_compiled_artifact(artifact, timeout))

    def safe_state(self) -> None:
        self._session.safe_state()

    def snapshot(self) -> dict[str, object]:
        return dict(self._session.current_snapshot())


__all__ = ["CompiledPulseSession", "PulseStreamerSessionBackend"]
