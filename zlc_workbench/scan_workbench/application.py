"""Headless application projection for one frozen pulse-scan request."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from zlc_data import Selection
from zlc_frontend.figure import ViewPreferences
from zlc_frontend.scan_preview import (
    ScanDisplayIntent,
    build_occupancy_scan_curve,
)
from zlc_neutral_atom.scan import OccupancyScanRequest, ScanRequest
from zlc_neutral_atom.scan.application import PreparedExactScan
from zlc_neutral_atom.scan.contracts import AutonomousScanSlotProgram
from zlc_neutral_atom.scan.reference import ScanArtifactRef
from zlc_storage import canonical_digest
from zlc_workbench.progressive_scan import ProgressiveScanSpec
from zlc_workbench.scan import FinalScanPresentation, PreparedScanPanelRun


@dataclass(frozen=True, slots=True)
class ScanWorkbenchActions:
    """Installation-bound operations consumed by the scan Workbench."""

    prepare: Callable[[ScanRequest | OccupancyScanRequest], PreparedExactScan]
    project_final: Callable[
        [ScanArtifactRef, Selection | None, ViewPreferences | None],
        FinalScanPresentation,
    ]

    def __post_init__(self) -> None:
        if not callable(self.prepare) or not callable(self.project_final):
            raise TypeError("scan Workbench actions must be callable")


class _FrozenScanApplication:
    """Bridge from a frozen domain request and explicit actions to the controller."""

    __slots__ = (
        "_actions",
        "_request",
        "_display_intent",
        "_final_selection",
        "_final_preferences",
    )

    def __init__(
        self,
        actions: ScanWorkbenchActions,
        request: ScanRequest | OccupancyScanRequest,
        display_intent: ScanDisplayIntent = ScanDisplayIntent(),
    ) -> None:
        if not isinstance(actions, ScanWorkbenchActions):
            raise TypeError("actions must be ScanWorkbenchActions")
        if not isinstance(request, (ScanRequest, OccupancyScanRequest)):
            raise TypeError("request must be a current scan request")
        if not isinstance(display_intent, ScanDisplayIntent):
            raise TypeError("display_intent must be ScanDisplayIntent")
        if isinstance(request, ScanRequest) and display_intent != ScanDisplayIntent():
            raise ValueError("direct-camera scan has no site display setting")
        self._actions = actions
        self._request = request
        self._display_intent = display_intent
        self._final_selection = None
        self._final_preferences = None

    def prepare(self):
        command = self._actions.prepare(self._request)
        if not isinstance(command, PreparedExactScan):
            raise TypeError("scan prepare action must return PreparedExactScan")
        if isinstance(self._request, ScanRequest):
            return PreparedScanPanelRun(None, command.start)
        identity = canonical_digest(
            {
                "owner": "zlc_workbench.occupancy-scan",
                "program": self._request.program.fingerprint,
                "source_schema": command.source_schema.fingerprint,
                "output_contract": command.output_contract.fingerprint,
            }
        )[:20]
        presentation = build_occupancy_scan_curve(
            command.output_contract.output_dataset_schema,
            identity=identity,
            display_intent=self._display_intent,
        )
        progressive = ProgressiveScanSpec(
            command,
            presentation,
        )
        self._final_selection = presentation.display_selection
        self._final_preferences = presentation.display_preferences

        def start_occupancy(preview):
            if preview is not None and preview.spec != command.preview_spec:
                raise ValueError(
                    "prepared progressive preview changed before start"
                )
            return command.start(preview)

        progressive_enabled = isinstance(
            self._request.program,
            AutonomousScanSlotProgram,
        )
        return PreparedScanPanelRun(
            progressive if progressive_enabled else None,
            start_occupancy,
        )

    def project_final(
        self,
        source_ref: ScanArtifactRef,
    ) -> FinalScanPresentation:
        selection = None
        preferences = None
        if isinstance(self._request, OccupancyScanRequest):
            if self._final_preferences is None:
                raise RuntimeError("occupancy display was not prepared")
            selection = self._final_selection
            preferences = self._final_preferences
        result = self._actions.project_final(source_ref, selection, preferences)
        if not isinstance(result, FinalScanPresentation):
            raise TypeError("project_final action must return FinalScanPresentation")
        return result


__all__ = ["ScanWorkbenchActions"]
