"""The task console's composition root -- the one place that opens the console window.

Every entry goes through :func:`open_task_console`: the double-clickable
``task_console.bat``, the root ``task_console.py`` launcher, and
``Experiment.task_console()`` from a notebook.

The window preserves the main Monitor/Logic tabbed board, panel cards, and
Fluent chrome in the named presentation modules under this package.  Its data plane is
rewired onto the CURRENT architecture per the design document's section 10; the
four contracted seams this root assembles, in rewiring order:

1. CATALOG -- the ``zlc_neutral_atom`` DefinitionCatalog (measurement /
   stream-processor / task definitions), mapped through a local CatalogView
   adapter into the skeleton's Add-Panel / Logic-tab vocabulary.  No global
   registry: plain imports; duplicate keys fail at startup.
2. RUN -- panel/logic Start compiles an immutable PipelineSpec ->
   ``compile_pipeline`` -> one flat RunPlan under a single RunController; the
   skeleton never starts nested runs or owns terminal state.
3. MONITOR -- live panels consume admitted ``MonitorTap -> MonitorDataset ->
   LiveDatasetSlot``: the tick reads coalesced revision notifications and takes
   atomic MonitorDatasetSnapshots; no mutable signal hub returns.
4. RENDER -- panels draw through the worker-raster pipeline (``zlc_frontend``
   encoded-raster / render DTOs onto the qt_widgets raster
   boards); no transitional matplotlib live stack.

Until a seam lands the corresponding skeleton members stay disconnected -- the
window may not fully operate yet, which is the accepted state of the rewiring
phase (the purge deliberately preceded the reconnect).
"""

from __future__ import annotations

__all__ = ["open_task_console"]


def open_task_console(experiment, *, state=None, task=None, **kwargs):
    """Open the console UI for ``experiment`` and return the console body.

    ``experiment`` is the current ``Zou_lab_control.notebook`` Experiment.  The
    seams are derived from it HERE and nowhere else: the skeleton is handed a
    catalog view, a run factory and a data plane, and never imports the domain.
    """

    from Zou_lab_control.notebook.facade import (
        _prepare_camera_monitor_for_workbench,
        _prepare_capture_for_workbench,
        _prepare_finite_occupancy_for_workbench,
        _prepare_temperature_release_recapture_for_workbench,
    )
    from zlc_neutral_atom.acquisition import CAMERA_MEASUREMENT_KEY
    from zlc_neutral_atom.capture_application import CaptureRequest
    from zlc_neutral_atom.monitor_application import (
        CAMERA_MONITOR_MEASUREMENT_KEY,
    )
    from zlc_neutral_atom.mot_field import MOT_FIELD_TASK_KEY
    from zlc_neutral_atom.readout.calibration_reference import (
        CalibrationArtifactRef,
    )
    from zlc_neutral_atom.readout.occupancy import (
        OCCUPANCY_STREAM_PROCESSOR_KEY,
    )
    from zlc_neutral_atom.readout.coupled_measurements import (
        AutonomousMeasurementUnavailable,
        GREY_MOLASSES_CAPABILITY_GAP,
        GREY_MOLASSES_DETUNING_KEY,
        READOUT_DURATION_CAPABILITY_GAP,
        READOUT_DURATION_FIDELITY_KEY,
        TEMPERATURE_RELEASE_RECAPTURE_KEY,
    )
    from zlc_neutral_atom.readout.sitemap import SITEMAP_CALIBRATION_TASK_KEY
    from zlc_neutral_atom.scan import PULSE_SCAN_TASK_KEY

    from .calibration_task import CalibrationTaskHandle
    from .catalog_bridge import ConsoleCatalogView
    from .data_plane import ConsoleDataPlane
    from .mot_field_task import MotFieldTaskHandle
    from .occupancy_binding import OccupancyBindingIntent
    from .coupled_measurement_presenter import (
        GreyMolassesDetuningIntent,
        ReadoutDurationFidelityIntent,
        TemperatureReleaseRecaptureIntent,
        freeze_temperature_release_recapture_request,
    )
    from .result_projection import project_final_signals
    from .window import show_task_console
    from zlc_workbench.data_figure.app import open_local_data_figure_analysis
    from .run_bridge import ConsoleRunNode

    catalog_view = ConsoleCatalogView(experiment)
    data_plane = ConsoleDataPlane()
    console: list = []            # filled below; the wake closure needs the body

    def request_owner_wake() -> None:
        """A worker finished something -- let the next tick pick it up.

        The console already polls every tick, so waking is a no-op here; the
        callback exists because the mailbox promises never to block a worker on
        the GUI thread, and a surface that DID need an immediate repaint would
        hook it here rather than inside the run bridge.
        """

    def submit_rectangle_selection(node, selection):
        """Promote a displayed rectangle only for its exact live monitor owner."""

        if getattr(getattr(node, "spec", None), "key", None) != (
            CAMERA_MONITOR_MEASUREMENT_KEY
        ):
            raise ValueError(
                "this rectangle is display-only; ROI signals are produced only "
                "by a running Camera monitor"
            )
        handle = getattr(node, "handle", None)
        if handle is None or handle.snapshot().state.terminal:
            raise RuntimeError("Camera monitor is not running")
        request = node.request
        state = data_plane.current_camera_roi_state(node)
        applied = getattr(state, "binding", None)
        # Repeating the exact applied rectangle is the one unambiguous delete
        # gesture.  It mirrors the standalone monitor without adding a second
        # ROI toggle or another piece of saved GUI state.
        candidate = (
            None
            if applied is not None and applied.selection == selection
            else selection
        )
        return data_plane.submit_camera_roi_control(
            node,
            candidate,
            request.roi_reduction,
        )

    def run_factory(spec, values):
        if spec.key == CAMERA_MONITOR_MEASUREMENT_KEY:
            node = ConsoleRunNode(
                spec,
                values,
                prepare=lambda request: _prepare_camera_monitor_for_workbench(
                    experiment, request
                ),
                request_owner_wake=request_owner_wake,
            )
            _bind_monitor_view(node, data_plane)
            return node
        if spec.key == CAMERA_MEASUREMENT_KEY:
            node = ConsoleRunNode(
                spec,
                values,
                prepare=lambda request: _prepare_capture_for_workbench(
                    experiment,
                    request,
                ),
                request_owner_wake=request_owner_wake,
            )
            _bind_capture_execution(node, data_plane)
            node.bind_final_projector(
                lambda result, current=node: project_final_signals(
                    experiment,
                    current,
                    result,
                )
            )
            return node
        if spec.key == TEMPERATURE_RELEASE_RECAPTURE_KEY:
            intent = spec.build_request(values)
            if not isinstance(intent, TemperatureReleaseRecaptureIntent):
                raise TypeError(
                    "temperature form did not produce its typed intent"
                )
            if not console:
                raise RuntimeError(
                    "TaskConsole composition is not ready for calibration binding"
                )
            calibration = console[0].resolve_console_producer(
                intent.calibration_signal
            )
            if (
                calibration.definition_key != SITEMAP_CALIBRATION_TASK_KEY
                or calibration.output_name != "calibration"
            ):
                raise ValueError(
                    "Temperature Calibration must select the calibration "
                    "output of a Calibrate readout Task row"
                )
            if calibration.running:
                raise RuntimeError(
                    "the selected Calibrate readout Task is still running"
                )
            if (
                not calibration.final_result_resolved
                or not isinstance(
                    calibration.final_result,
                    CalibrationArtifactRef,
                )
            ):
                raise RuntimeError(
                    "the selected Calibrate readout row has no successful "
                    "current FINAL CalibrationArtifactRef"
                )
            calibration_ref = calibration.final_result

            def prepare_temperature(current_intent):
                if current_intent != intent:
                    raise RuntimeError(
                        "temperature binding changed after producer resolution"
                    )
                request = freeze_temperature_release_recapture_request(
                    experiment,
                    current_intent,
                    calibration_ref=calibration_ref,
                )
                return _prepare_temperature_release_recapture_for_workbench(
                    experiment,
                    request,
                )

            node = ConsoleRunNode(
                spec,
                values,
                prepare=prepare_temperature,
                request_owner_wake=request_owner_wake,
            )
            node.bind_starter(lambda prepared: prepared.start())
            node.bind_final_projector(
                lambda result, current=node: project_final_signals(
                    experiment,
                    current,
                    result,
                )
            )
            return node
        if spec.key == READOUT_DURATION_FIDELITY_KEY:
            intent = spec.build_request(values)
            if not isinstance(intent, ReadoutDurationFidelityIntent):
                raise TypeError(
                    "readout-duration form did not produce its typed intent"
                )

            def reject_duration(current_intent):
                if current_intent != intent:
                    raise RuntimeError(
                        "readout-duration intent changed after construction"
                    )
                raise AutonomousMeasurementUnavailable(
                    READOUT_DURATION_CAPABILITY_GAP
                )

            node = ConsoleRunNode(
                spec,
                values,
                prepare=reject_duration,
                request_owner_wake=request_owner_wake,
            )
            node.bind_starter(lambda prepared: prepared.start())
            return node
        if spec.key == GREY_MOLASSES_DETUNING_KEY:
            intent = spec.build_request(values)
            if not isinstance(intent, GreyMolassesDetuningIntent):
                raise TypeError(
                    "grey-molasses form did not produce its typed intent"
                )

            def reject_grey(current_intent):
                if current_intent != intent:
                    raise RuntimeError(
                        "grey-molasses intent changed after construction"
                    )
                raise AutonomousMeasurementUnavailable(
                    GREY_MOLASSES_CAPABILITY_GAP
                )

            node = ConsoleRunNode(
                spec,
                values,
                prepare=reject_grey,
                request_owner_wake=request_owner_wake,
            )
            node.bind_starter(lambda prepared: prepared.start())
            return node
        if spec.key == OCCUPANCY_STREAM_PROCESSOR_KEY:
            if not console:
                raise RuntimeError(
                    "TaskConsole composition is not ready for processor binding"
                )
            intent = spec.build_request(values)
            if not isinstance(intent, OccupancyBindingIntent):
                raise TypeError(
                    "occupancy form did not produce OccupancyBindingIntent"
                )
            source = console[0].resolve_console_producer(
                intent.camera_frame_signal
            )
            if (
                source.definition_key != CAMERA_MEASUREMENT_KEY
                or source.output_name != "frame"
                or not isinstance(source.request, CaptureRequest)
            ):
                raise ValueError(
                    "occupancy Camera capture must select the frame output of "
                    "a Camera capture Measurement row in this TaskConsole"
                )
            if source.running:
                raise RuntimeError(
                    "the selected Camera capture row is already running; stop "
                    "it before starting occupancy"
                )
            calibration = console[0].resolve_console_producer(
                intent.calibration_signal
            )
            if (
                calibration.definition_key != SITEMAP_CALIBRATION_TASK_KEY
                or calibration.output_name != "calibration"
            ):
                raise ValueError(
                    "occupancy Calibration must select the calibration output "
                    "of a Calibrate readout Task row in this TaskConsole"
                )
            if calibration.running:
                raise RuntimeError(
                    "the selected Calibrate readout Task is still running"
                )
            if (
                not calibration.final_result_resolved
                or not isinstance(
                    calibration.final_result,
                    CalibrationArtifactRef,
                )
            ):
                raise RuntimeError(
                    "the selected Calibrate readout row has no successful "
                    "current FINAL CalibrationArtifactRef; run it successfully "
                    "before starting occupancy"
                )
            capture_request = source.request
            calibration_ref = calibration.final_result

            def prepare_occupancy(current_intent):
                if current_intent != intent:
                    raise RuntimeError(
                        "occupancy binding changed after producer resolution"
                    )
                return _prepare_finite_occupancy_for_workbench(
                    experiment,
                    capture_request,
                    calibration_ref,
                )

            node = ConsoleRunNode(
                spec,
                values,
                prepare=prepare_occupancy,
                request_owner_wake=request_owner_wake,
            )
            _bind_occupancy_preview(node, data_plane)
            node.bind_final_projector(
                lambda result, current=node: project_final_signals(
                    experiment,
                    current,
                    result,
                )
            )
            return node
        if spec.key == SITEMAP_CALIBRATION_TASK_KEY:
            node = ConsoleRunNode(
                spec,
                values,
                prepare=lambda request: request,
                request_owner_wake=request_owner_wake,
            )

            def start_calibration_sequence(request):
                return CalibrationTaskHandle(
                    request,
                    start_capture=experiment.start,
                    build_calibration_request=(
                        lambda source, analysis, timeout: (
                            experiment.readout.calibration_request(
                                source,
                                analysis,
                                timeout_seconds=timeout,
                            )
                        )
                    ),
                    start_calibration=experiment.readout.start_calibration,
                )

            node.bind_starter(start_calibration_sequence)
            node.bind_final_projector(
                lambda result, current=node: project_final_signals(
                    experiment,
                    current,
                    result,
                )
            )
            return node
        if spec.key == MOT_FIELD_TASK_KEY:
            node = ConsoleRunNode(
                spec,
                values,
                prepare=lambda request: request,
                request_owner_wake=request_owner_wake,
            )
            node.bind_starter(
                lambda request: MotFieldTaskHandle(
                    request,
                    start_scan=experiment.readout._start_mot_field_scan,
                    materialize_scan=experiment.readout.materialize_scan,
                )
            )
            node.bind_final_projector(
                lambda result, current=node: project_final_signals(
                    experiment,
                    current,
                    result,
                )
            )
            return node
        if spec.key == PULSE_SCAN_TASK_KEY:
            node = ConsoleRunNode(
                spec,
                values,
                prepare=lambda request: request,
                request_owner_wake=request_owner_wake,
            )
            node.bind_starter(experiment.start_scan)
            node.bind_final_projector(
                lambda result, current=node: project_final_signals(
                    experiment,
                    current,
                    result,
                )
            )
            return node
        raise NotImplementedError(
            f"TaskConsole has no current runtime binding for {spec.key}"
        )

    body = show_task_console(
        state=state, task=task,
        catalog_view=catalog_view,
        run_factory=run_factory,
        data_plane=data_plane,
        fit_window_factory=experiment.fit_gui,
        local_fit_window_factory=open_local_data_figure_analysis,
        rectangle_selection_sink=submit_rectangle_selection,
        **kwargs,
    )
    console.append(body)
    return body


def _bind_monitor_view(node, data_plane) -> None:
    """Teach one run node how to start WITH a live view, and register that view.

    A camera monitor starts only through ``start_with_view``: the factory it
    receives runs on the worker, builds this node's LiveDatasetSlot, and hands it
    to the data plane, which is what makes the node's frames reach the board.
    """

    from zlc_frontend.figure import DatasetId
    from zlc_workbench.live import LiveDatasetSlot

    dataset_id = DatasetId(f"console-{node.spec.key.stable_definition_id}-{id(node):x}")

    def start(command):
        def factory(view_spec):
            slot = LiveDatasetSlot(
                view_spec,
                dataset_id=dataset_id,
                retain_on_terminal=True,
            )
            data_plane.attach(node, slot)

            def source_changed() -> None:
                data_plane.mark_changed(node)

            slot.set_change_listener(source_changed)
            return slot

        return command.start_with_view(factory=factory)

    node.bind_starter(start)


def _bind_capture_execution(node, data_plane) -> None:
    """Start one finite capture, attaching its optional single-event preview.

    The preview is a presentation convenience, not part of the capture
    contract.  Any layout outside the preview's one-cell contract therefore
    starts without it and publishes its complete FINAL dataset after success.
    """

    import uuid

    from zlc_data import BlockId
    from zlc_frontend.figure import DatasetId
    from zlc_workbench.live import LiveDatasetSlot

    def start(command):
        try:
            command.preview_schema
        except ValueError:
            return command.start()

        token = uuid.uuid4().hex
        dataset_id = DatasetId(f"console-capture-{token}")
        block_id = BlockId(f"console-capture-preview-{token}")

        def factory(preview_spec):
            slot = LiveDatasetSlot(
                preview_spec,
                dataset_id=dataset_id,
                retain_on_terminal=True,
            )
            data_plane.attach(node, slot)
            slot.set_change_listener(lambda: data_plane.mark_changed(node))
            return slot

        return command.start_with_preview(
            block_id=block_id,
            factory=factory,
        )

    node.bind_starter(start)


def _bind_occupancy_preview(node, data_plane) -> None:
    """Attach the exact counts preview of one flat occupancy Run."""

    from zlc_workbench.progressive_scan import ExactDatasetLiveSlot

    def start(command):
        def factory(preview_spec):
            slot = ExactDatasetLiveSlot(preview_spec)
            data_plane.attach_exact(node, slot)
            return slot

        return command.start_with_preview(factory=factory)

    node.bind_starter(start)
