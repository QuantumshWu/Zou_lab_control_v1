"""Public Experiment API owned by the Pulse-scan Measurement."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path

from zlc_data import DataTransformSpec
from zlc_neutral_atom.artifact_dataset_source import ArtifactDatasetSource
from zlc_neutral_atom.catalog import DefinitionKey
from zlc_neutral_atom.dataset_output import DatasetOutputDeclaration
from zlc_neutral_atom.runtime.signal_source import SignalEventSource
from zlc_neutral_atom.runtime.run import RunHandle
from zlc_neutral_atom.timing.pulse_parameter_scan import (
    ApiSegmentTable,
    ApiSlotSegmentedProgram,
    AutonomousScanSlotProgram,
    PulseParameterScanProgram,
)
from zlc_pulse import (
    PulseDocument,
    commit_scan_table,
    freeze_scan_table,
)

from .application import PreparedExactScan
from .artifact import (
    MaterializedScanData,
    ScanArtifact,
    load_scan_artifact,
    materialize_scan_data,
    project_scan_dataset,
)
from .reference import ScanArtifactRef
from .source_binding import PulseScanBoundRequest, ScanSignalBinding


class PulseScanApi:
    __slots__ = ("_load_pulse", "_prepare", "_scans_root", "_wait_run")

    def __init__(
        self,
        scans_root: Path,
        *,
        load_pulse: Callable,
        prepare: Callable,
        wait_run: Callable,
    ) -> None:
        if not isinstance(scans_root, Path):
            raise TypeError("scans_root must be pathlib.Path")
        if any(
            not callable(operation)
            for operation in (load_pulse, prepare, wait_run)
        ):
            raise TypeError("PulseScan API operations must be callable")
        self._scans_root = scans_root.resolve()
        self._load_pulse = load_pulse
        self._prepare = prepare
        self._wait_run = wait_run

    def scan_slot_program(
        self,
        pulse: PulseDocument | str | Path,
        *,
        rows: Sequence[Sequence[int | float]] | None = None,
        scan_sweep_count: int | None = None,
        api_values: Mapping[str, int | float] | None = None,
    ) -> AutonomousScanSlotProgram:
        """Freeze one autonomous SCAN_SLOT program in declared column order."""

        document = self._load_pulse(pulse)
        if not isinstance(document, PulseDocument):
            raise TypeError("PulseScan pulse loader returned another value")
        if scan_sweep_count is not None:
            document = replace(document, scan_sweep_count=scan_sweep_count)
        if rows is not None:
            columns = tuple(
                parameter.parameter_id for parameter in document.scan_parameters
            )
            table, _normalization = freeze_scan_table(document, columns, rows)
            document = commit_scan_table(document, table)
        return AutonomousScanSlotProgram.from_api_values(document, api_values)

    def api_slot_program(
        self,
        pulse: PulseDocument | str | Path,
        *,
        rows: Sequence[Sequence[int | float]],
        scan_sweep_count: int | None = None,
    ) -> ApiSlotSegmentedProgram:
        """Freeze the existing segmented API_SLOT execution exception."""

        document = self._load_pulse(pulse)
        if not isinstance(document, PulseDocument):
            raise TypeError("PulseScan pulse loader returned another value")
        if scan_sweep_count is not None:
            document = replace(document, scan_sweep_count=scan_sweep_count)
        columns = tuple(
            parameter.parameter_id for parameter in document.api_parameters
        )
        return ApiSlotSegmentedProgram(
            document,
            ApiSegmentTable(columns, tuple(tuple(row) for row in rows)),
            "Explicit API-slot sweep authored through PulseScan API",
        )

    def bind_scan(
        self,
        program: PulseParameterScanProgram,
        source: SignalEventSource,
        *,
        output_name: str,
        transform: DataTransformSpec | None = None,
    ) -> PulseScanBoundRequest:
        """Bind one declared output without inspecting or claiming its device."""

        if not isinstance(source, SignalEventSource):
            raise TypeError("source must implement SignalEventSource")
        definition_key = getattr(source, "definition_key", None)
        if not isinstance(definition_key, DefinitionKey):
            raise TypeError("source must expose its DefinitionKey")
        declarations = getattr(source, "dataset_output_declarations", None)
        if declarations is None:
            raise TypeError("source must expose Dataset output declarations")
        declarations = tuple(declarations)
        if any(
            not isinstance(declaration, DatasetOutputDeclaration)
            for declaration in declarations
        ):
            raise TypeError(
                "source Dataset outputs must be DatasetOutputDeclaration values"
            )
        matches = tuple(
            declaration
            for declaration in declarations
            if declaration.name == output_name
        )
        if len(matches) != 1:
            raise KeyError(f"source has no unique Dataset output {output_name!r}")
        return PulseScanBoundRequest(
            program,
            ScanSignalBinding(definition_key, matches[0], transform),
        )

    def _project_dataset_source(
        self,
        reference: ScanArtifactRef,
        *,
        materialize: bool,
        abort_check: Callable[[], None] | None = None,
    ) -> ArtifactDatasetSource:
        """Project a scan for the application-owned generic Figure/Fit surface."""

        return project_scan_dataset(
            self._scans_root,
            reference,
            materialize=materialize,
            abort_check=abort_check,
        )

    def prepare_scan(
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

    def start_scan(
        self,
        request: PulseScanBoundRequest,
        source: SignalEventSource,
        *,
        sequencer_role: str | None = None,
    ) -> RunHandle:
        return self.prepare_scan(
            request,
            source,
            sequencer_role=sequencer_role,
        ).start()

    def run_scan(
        self,
        request: PulseScanBoundRequest,
        source: SignalEventSource,
        *,
        sequencer_role: str | None = None,
    ) -> ScanArtifactRef:
        return self._wait_run(
            self.start_scan(
                request,
                source,
                sequencer_role=sequencer_role,
            )
        )

    def load_scan(self, reference: ScanArtifactRef) -> ScanArtifact:
        return load_scan_artifact(self._scans_root, reference)

    def materialize_scan(
        self,
        reference: ScanArtifactRef,
    ) -> MaterializedScanData:
        return materialize_scan_data(self._scans_root, reference)


__all__ = ["PulseScanApi"]
