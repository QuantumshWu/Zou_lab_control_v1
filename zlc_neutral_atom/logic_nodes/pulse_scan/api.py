"""Public Experiment API owned by the Pulse-scan Measurement."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from zlc_neutral_atom.artifact_dataset_source import ArtifactDatasetSource

from zlc_neutral_atom.runtime.signal_source import SignalEventSource

from .application import PreparedExactScan
from .reference import ScanArtifactRef
from .repository import MaterializedScanData, ScanArtifact, ScanRepository
from .source_binding import PulseScanBoundRequest


class PulseScanApi:
    __slots__ = ("_prepare", "_pulses_root", "_repository")

    def __init__(
        self,
        repository: ScanRepository,
        *,
        pulses_root: Path,
        prepare: Callable,
    ) -> None:
        if not isinstance(repository, ScanRepository):
            raise TypeError("repository must be ScanRepository")
        if not callable(prepare):
            raise TypeError("prepare must be callable")
        root = Path(pulses_root).expanduser()
        if not root.is_absolute():
            raise ValueError("PulseScan pulses_root must be absolute")
        self._repository = repository
        self._pulses_root = root.resolve()
        self._prepare = prepare

    def close(self) -> tuple[Exception, ...]:
        try:
            self._repository.close()
        except Exception as error:
            return (error,)
        return ()

    def _project_dataset_source(
        self,
        reference: ScanArtifactRef,
        *,
        materialize: bool,
        abort_check: Callable[[], None] | None = None,
    ) -> ArtifactDatasetSource:
        """Project a scan for the application-owned generic Figure/Fit surface."""

        return self._repository.project_dataset_source(
            reference,
            materialize=materialize,
            abort_check=abort_check,
        )

    def prepare_scan_source(
        self,
        request: PulseScanBoundRequest,
        source: SignalEventSource,
        *,
        sequencer_role: str | None = None,
    ) -> PreparedExactScan:
        if not isinstance(request, PulseScanBoundRequest):
            raise TypeError("request must be PulseScanBoundRequest")
        if not isinstance(source, SignalEventSource):
            raise TypeError("source must implement SignalEventSource")
        return self._prepare(request, source, sequencer_role)

    def load_scan(self, reference: ScanArtifactRef) -> ScanArtifact:
        return self._repository.admit(reference)

    def materialize_scan(
        self,
        reference: ScanArtifactRef,
    ) -> MaterializedScanData:
        return self._repository.materialize(reference)


__all__ = ["PulseScanApi"]
