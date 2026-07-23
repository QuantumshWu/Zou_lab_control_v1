"""Headless application projection for one frozen pulse-scan request."""

from __future__ import annotations

from Zou_lab_control.notebook.facade import (
    Experiment,
    OccupancyScanRequest,
    ScanRequest,
    _prepare_occupancy_scan_for_workbench,
)
from zlc_frontend.figure import ViewIntent
from zlc_neutral_atom.scan.contracts import AutonomousScanSlotProgram
from zlc_neutral_atom.scan.reference import ScanArtifactRef
from zlc_storage import canonical_digest
from zlc_workbench.progressive_scan import (
    ScanDisplayIntent,
    build_occupancy_progressive_spec,
)
from zlc_workbench.scan import FinalScanPresentation, PreparedScanPanelRun


class _FrozenScanApplication:
    """Composition-owned bridge from a frozen public request to the controller."""

    __slots__ = (
        "_experiment",
        "_request",
        "_display_intent",
        "_final_selection",
        "_final_preferences",
    )

    def __init__(
        self,
        experiment: Experiment,
        request: ScanRequest | OccupancyScanRequest,
        display_intent: ScanDisplayIntent = ScanDisplayIntent(),
    ) -> None:
        if not isinstance(display_intent, ScanDisplayIntent):
            raise TypeError("display_intent must be ScanDisplayIntent")
        if isinstance(request, ScanRequest) and display_intent != ScanDisplayIntent():
            raise ValueError("direct-camera scan has no site display setting")
        self._experiment = experiment
        self._request = request
        self._display_intent = display_intent
        self._final_selection = None
        self._final_preferences = None

    def prepare(self):
        if isinstance(self._request, ScanRequest):

            def start_direct(preview):
                if preview is not None:
                    raise ValueError(
                        "direct camera scan has no progressive counts port"
                    )
                return self._experiment.start_scan(self._request)

            return PreparedScanPanelRun(None, start_direct)
        command = _prepare_occupancy_scan_for_workbench(
            self._experiment,
            self._request,
        )
        identity = canonical_digest(
            {
                "owner": "Zou_lab_control.workbench.occupancy-scan",
                "program": self._request.program.fingerprint,
                "source_schema": command.source_schema.fingerprint,
                "output_contract": command.output_contract.fingerprint,
            }
        )[:20]
        progressive = build_occupancy_progressive_spec(
            command.source_schema,
            command.output_contract,
            identity=identity,
            display_intent=self._display_intent,
        )
        self._final_selection = progressive.display_selection
        self._final_preferences = progressive.display_preferences

        def start_occupancy(preview):
            if preview is not None and preview.spec != progressive.preview_spec:
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
        figure_options = {}
        if isinstance(self._request, OccupancyScanRequest):
            if self._final_preferences is None:
                raise RuntimeError("occupancy display was not prepared")
            figure_options.update(
                intent=ViewIntent.CURVE,
                selection=self._final_selection,
                preferences=self._final_preferences,
            )
        figure = self._experiment.figure(source_ref, **figure_options)
        layer = figure.document.layers[0]
        bindings = " · ".join(
            f"{binding.axis_id.value}={binding.role.value.lower()}"
            for binding in layer.view.axis_bindings
        )
        summary = layer.view.intent.value.lower()
        if bindings:
            summary = f"{summary} · {bindings}"
        if layer.view.display_selections:
            summary += f" · selections={len(layer.view.display_selections)}"
        return FinalScanPresentation(
            source_ref,
            figure.to_png_bytes(),
            summary,
        )
