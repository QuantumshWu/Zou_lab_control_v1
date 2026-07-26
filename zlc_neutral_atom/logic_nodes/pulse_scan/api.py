"""Notebook surface owned by the Pulse-scan Measurement."""

from __future__ import annotations

from typing import Protocol

from zlc_neutral_atom.runtime.signal_source import SignalEventSource

from .application import PreparedExactScan
from .reference import ScanArtifactRef
from .repository import MaterializedScanData, ScanArtifact
from .source_binding import PulseScanBoundRequest


class PulseScanNotebookHost(Protocol):
    def bind_pulse_scan_source(
        self,
        request: PulseScanBoundRequest,
        source: SignalEventSource,
        *,
        sequencer_role: str | None,
    ) -> PreparedExactScan: ...

    def load_pulse_scan(self, reference: ScanArtifactRef) -> ScanArtifact: ...

    def materialize_pulse_scan(
        self,
        reference: ScanArtifactRef,
    ) -> MaterializedScanData: ...


class PulseScanNotebookAdapter:
    __slots__ = ()

    @property
    def _pulse_scan_notebook_host(self) -> PulseScanNotebookHost:
        raise NotImplementedError

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
        return self._pulse_scan_notebook_host.bind_pulse_scan_source(
            request,
            source,
            sequencer_role=sequencer_role,
        )

    def load_scan(self, reference: ScanArtifactRef) -> ScanArtifact:
        return self._pulse_scan_notebook_host.load_pulse_scan(reference)

    def materialize_scan(
        self,
        reference: ScanArtifactRef,
    ) -> MaterializedScanData:
        return self._pulse_scan_notebook_host.materialize_pulse_scan(reference)


__all__ = ["PulseScanNotebookAdapter", "PulseScanNotebookHost"]
