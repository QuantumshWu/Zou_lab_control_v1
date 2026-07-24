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

All four seams are required by this composition root.  A missing catalog,
runtime, live-data, or render binding is a startup/runtime defect; the product
does not expose a transitional partial TaskConsole.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["open_task_console"]


def open_task_console(experiment, *, state=None, task=None, **kwargs):
    """Open the console UI for ``experiment`` and return the console body.

    ``experiment`` is the current ``Zou_lab_control.notebook`` Experiment.  The
    seams are derived from it HERE and nowhere else: the skeleton is handed a
    catalog view, a run factory and a data plane, and never imports the domain.
    """

    from Zou_lab_control.notebook.facade import (
        _prepare_capture_for_workbench,
        _prepare_camera_measurement_for_workbench,
        _prepare_grey_molasses_detuning_for_workbench,
        _prepare_temperature_release_recapture_for_workbench,
    )
    from zlc_neutral_atom.acquisition import CAMERA_MEASUREMENT_KEY
    from zlc_neutral_atom.camera_measurement import CameraMeasurementRequest
    from zlc_neutral_atom.mot_field import MOT_FIELD_TASK_KEY
    from zlc_neutral_atom.readout.calibration_reference import (
        CalibrationArtifactRef,
    )
    from zlc_neutral_atom.readout.occupancy import (
        OCCUPANCY_STREAM_PROCESSOR_KEY,
    )
    from zlc_neutral_atom.readout.contracts import ReadoutBindingKey
    from zlc_neutral_atom.readout.coupled_measurements import (
        AutonomousMeasurementUnavailable,
        GREY_MOLASSES_DETUNING_KEY,
        READOUT_DURATION_CAPABILITY_GAP,
        READOUT_DURATION_FIDELITY_KEY,
        TEMPERATURE_RELEASE_RECAPTURE_KEY,
    )
    from zlc_neutral_atom.readout.sitemap import SITEMAP_CALIBRATION_TASK_KEY
    from zlc_neutral_atom.scan import PULSE_SCAN_MEASUREMENT_KEY
    from zlc_neutral_atom.scan import (
        ApiSlotSegmentedProgram,
        AutonomousScanSlotProgram,
    )

    from .calibration_task import (
        CalibrationTaskExecution,
        CalibrationTaskHandle,
        CalibrationTaskIntent,
    )
    from .catalog_bridge import ConsoleCatalogView
    from .data_plane import ConsoleDataPlane
    from .mot_field_task import MotFieldTaskHandle, MotFieldTaskIntent
    from zlc_workbench.mot_field_live import MotFieldGridLiveSlot
    from .occupancy_binding import (
        OccupancyBindingIntent,
        ReactiveOccupancyNode,
    )
    from .pulse_scan_binding import PulseScanBindingIntent
    from .coupled_measurement_presenter import (
        GreyMolassesDetuningIntent,
        ReadoutDurationFidelityIntent,
        TemperatureReleaseRecaptureIntent,
        freeze_grey_molasses_detuning_request,
        freeze_temperature_release_recapture_request,
    )
    from .result_projection import project_final_signals
    from .window import show_task_console
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

    def resolve_calibration_reference(
        signal_key: str,
        *,
        measurement_name: str,
    ) -> CalibrationArtifactRef:
        if not console:
            raise RuntimeError(
                "TaskConsole composition is not ready for calibration binding"
            )
        calibration = console[0].resolve_console_producer(signal_key)
        if (
            calibration.definition_key != SITEMAP_CALIBRATION_TASK_KEY
            or calibration.output_name != "calibration"
        ):
            raise ValueError(
                f"{measurement_name} Calibration must select the calibration "
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
        return calibration.final_result

    def retire_for_exact_scan(node) -> None:
        """Join one already-cancelled hardware producer without polling its GUI."""

        if node is None:
            return
        wait = getattr(node, "wait_until_terminal", None)
        if callable(wait):
            wait(
                reason="Pulse scan is taking exact source ownership",
            )

    def freeze_pulse_scan_request(intent: PulseScanBindingIntent):
        """Resolve Signal(y) into a dedicated current exact source pipeline."""

        if not console:
            raise RuntimeError(
                "TaskConsole composition is not ready for Pulse scan binding"
            )
        source = console[0].resolve_pulse_scan_source(intent.y_signal)
        producer = source.producer
        transform = source.transform_spec
        program = intent.program

        def direct_request(*, camera_role: str):
            common = {
                "camera_role": camera_role,
                "output_transform_spec": transform,
            }
            if isinstance(program, AutonomousScanSlotProgram):
                return experiment.readout.scan_request(
                    program.document,
                    api_values=dict(program.api_values),
                    **common,
                )
            if isinstance(program, ApiSlotSegmentedProgram):
                return experiment.readout.api_scan_request(
                    program.document,
                    api_table=program.table,
                    segmentation_rationale=program.segmentation_rationale,
                    **common,
                )
            raise TypeError("Pulse scan intent contains an unknown program")

        if (
            producer.definition_key == CAMERA_MEASUREMENT_KEY
            and producer.output_name == "frame"
            and isinstance(producer.request, CameraMeasurementRequest)
        ):
            if producer.request.frames_per_cycle != 1:
                raise ValueError(
                    "Pulse scan Camera y requires exactly one frame per scan cell"
                )
            return direct_request(
                camera_role=producer.request.camera_ref.role
            ), (producer.run_node,)

        if (
            producer.definition_key == OCCUPANCY_STREAM_PROCESSOR_KEY
            and producer.output_name in ("counts", "occupied")
            and isinstance(producer.request, OccupancyBindingIntent)
        ):
            camera = console[0].resolve_console_producer(
                producer.request.camera_frame_signal
            )
            if (
                camera.definition_key != CAMERA_MEASUREMENT_KEY
                or camera.output_name != "frame"
                or not isinstance(camera.request, CameraMeasurementRequest)
            ):
                raise ValueError(
                    "selected Occupancy y no longer resolves to one Camera frame"
                )
            if camera.request.frames_per_cycle != 1:
                raise ValueError(
                    "Pulse scan Occupancy y requires one Camera frame per cell"
                )
            calibration_ref = resolve_calibration_reference(
                producer.request.calibration_signal,
                measurement_name="Pulse scan Occupancy y",
            )
            common = {
                "camera_role": camera.request.camera_ref.role,
                "calibration_ref": calibration_ref,
                "output_name": producer.output_name,
                "output_transform_spec": transform,
            }
            if isinstance(program, AutonomousScanSlotProgram):
                request = experiment.readout.occupancy_scan_request(
                    program.document,
                    api_values=dict(program.api_values),
                    **common,
                )
            elif isinstance(program, ApiSlotSegmentedProgram):
                request = experiment.readout.api_occupancy_scan_request(
                    program.document,
                    api_table=program.table,
                    segmentation_rationale=program.segmentation_rationale,
                    **common,
                )
            else:
                raise TypeError("Pulse scan intent contains an unknown program")
            return request, (producer.run_node, camera.run_node)

        raise ValueError(
            "selected Pulse scan y has no exact source builder; current exact "
            "sources are Camera frame, Occupancy counts/occupied, and their "
            "Figure Area data"
        )

    def run_factory(spec, values):
        if spec.key == CAMERA_MEASUREMENT_KEY:
            def prepare_camera(request):
                if isinstance(request, CameraMeasurementRequest):
                    return _prepare_camera_measurement_for_workbench(
                        experiment,
                        request,
                    )
                raise TypeError("Camera form must produce CameraMeasurementRequest")

            node = ConsoleRunNode(
                spec,
                values,
                prepare=prepare_camera,
                request_owner_wake=request_owner_wake,
            )
            _bind_camera_execution(node, data_plane)
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
            calibration_ref = resolve_calibration_reference(
                intent.calibration_signal,
                measurement_name="Temperature",
            )

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

            calibration_ref = resolve_calibration_reference(
                intent.calibration_signal,
                measurement_name="Grey molasses detuning",
            )

            def prepare_grey(current_intent):
                if current_intent != intent:
                    raise RuntimeError(
                        "grey-molasses intent changed after construction"
                    )
                request = freeze_grey_molasses_detuning_request(
                    experiment,
                    current_intent,
                    calibration_ref=calibration_ref,
                )
                return _prepare_grey_molasses_detuning_for_workbench(
                    experiment,
                    request,
                )

            node = ConsoleRunNode(
                spec,
                values,
                prepare=prepare_grey,
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
                or not isinstance(source.request, CameraMeasurementRequest)
                or source.request.repeat != 0
            ):
                raise ValueError(
                    "occupancy Camera source must select the frame output of "
                    "a live Camera Measurement row (repeat = 0) in this "
                    "TaskConsole"
                )
            if not source.running:
                raise RuntimeError(
                    "start the selected Camera Measurement before starting "
                    "the reactive Occupancy Processor"
                )
            source_value = data_plane.freeze().value(
                intent.camera_frame_signal
            )
            if source_value is None:
                raise RuntimeError(
                    "the selected running Camera Measurement has not published "
                    "a frame yet"
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
            calibration_ref = calibration.final_result

            def resolve_calibration():
                resolved = experiment.readout.load_calibration(calibration_ref)
                expected = ReadoutBindingKey(source.request.camera_ref.role)
                if resolved.artifact.frame_contract.binding != expected:
                    raise ValueError(
                        "selected Camera role differs from the admitted "
                        "calibration readout binding"
                    )
                return resolved

            return ReactiveOccupancyNode(
                spec,
                values,
                intent=intent,
                source_node=source.run_node,
                initial_source=source_value,
                resolve_calibration=resolve_calibration,
                data_plane=data_plane,
                request_owner_wake=request_owner_wake,
            )
        if spec.key == SITEMAP_CALIBRATION_TASK_KEY:
            frozen_intent = spec.build_request(values)
            if not isinstance(frozen_intent, CalibrationTaskIntent):
                raise TypeError("calibration catalog returned invalid intent")

            def prepare_calibration_task(current_intent):
                if current_intent != frozen_intent:
                    raise RuntimeError(
                        "calibration binding changed after request freeze"
                    )
                if current_intent.source_mode == "live":
                    sequence = experiment.readout.sitemap_request(
                        frames=current_intent.threshold_frames,
                        camera_role=current_intent.camera_role,
                        pulse=current_intent.pulse,
                        reference_exposure_s=current_intent.reference_exposure_s,
                        readout_exposure_s=current_intent.readout_exposure_s,
                        threshold_method=current_intent.threshold_method,
                        roi_radius=current_intent.roi_radius,
                    )
                    return CalibrationTaskExecution(
                        current_intent,
                        sequence.analysis,
                        sequence=sequence,
                    )
                source = experiment.readout._resolve_saved_calibration_capture(
                    Path(current_intent.folder) / "frames",
                    expected_camera_role=current_intent.camera_role,
                )
                analysis = experiment.readout.sitemap_analysis_request(
                    camera_role=current_intent.camera_role,
                    threshold_method=current_intent.threshold_method,
                    roi_radius=current_intent.roi_radius,
                )
                return CalibrationTaskExecution(
                    current_intent,
                    analysis,
                    source_capture_ref=source,
                )

            node = ConsoleRunNode(
                spec,
                values,
                prepare=prepare_calibration_task,
                request_owner_wake=request_owner_wake,
            )

            def start_calibration_sequence(request):
                if request.sequence is None:
                    start_capture = experiment.start
                else:
                    grouping = (
                        request.sequence.capture_request.within_point_grouping
                    )
                    if grouping is None:
                        raise RuntimeError(
                            "live calibration capture lost its frozen event grouping"
                        )
                    reference_event = (
                        request.analysis.layout.reference_event_indices[0]
                    )
                    reference_ordinals = tuple(
                        ordinal
                        for ordinal, (_repeat, event) in enumerate(grouping)
                        if event == reference_event
                    )

                    start_capture = (
                        lambda capture_request: _start_capture_preview(
                            _prepare_capture_for_workbench(
                                experiment,
                                capture_request,
                            ),
                            node,
                            data_plane,
                            output_name="frame",
                            source_ordinals=reference_ordinals,
                        )
                    )

                return CalibrationTaskHandle(
                    request,
                    start_capture=start_capture,
                    build_calibration_request=(
                        lambda source, analysis: (
                            experiment.readout.calibration_request(
                                source,
                                analysis,
                            )
                        )
                    ),
                    start_calibration=experiment.readout.start_calibration,
                    write_outputs=(
                        lambda source, calibration, intent: (
                            experiment.readout._write_calibration_task_outputs(
                                source,
                                calibration,
                                folder=intent.folder,
                                save_frames=(
                                    intent.save_frames
                                    if intent.source_mode == "live"
                                    else False
                                ),
                            )
                        )
                    ),
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
            intent = spec.build_request(values)
            if not isinstance(intent, MotFieldTaskIntent):
                raise TypeError("MOT form did not produce MotFieldTaskIntent")

            def bind_mot_field(current):
                if current != intent:
                    raise RuntimeError(
                        "MOT-field intent changed after construction"
                    )
                return experiment.readout.mot_field_request(
                    current.pulse,
                    center_x=current.center_x,
                    center_y=current.center_y,
                    center_z=current.center_z,
                    span=current.span,
                    points=current.points,
                    roi_cx=None if current.roi_cx == 0.0 else current.roi_cx,
                    roi_cy=None if current.roi_cy == 0.0 else current.roi_cy,
                    roi_radius=current.roi_radius,
                    camera_role=current.camera_role,
                )

            node = ConsoleRunNode(
                spec,
                values,
                prepare=lambda request: request,
                request_owner_wake=request_owner_wake,
            )

            def start_mot_field(current):
                request = bind_mot_field(current)
                prepared = (
                    experiment.readout._prepare_mot_field_scan_for_workbench(
                        request
                    )
                )
                slot = MotFieldGridLiveSlot(
                    request,
                    prepared.source_schema,
                    prepared.output_contract,
                )
                attached = False
                try:
                    data_plane.attach(node, slot, output_name="grid")
                    attached = True
                    slot.set_change_listener(
                        lambda: data_plane.mark_changed(node)
                    )

                    def start_exact_scan(bound_request):
                        if bound_request != request:
                            raise RuntimeError(
                                "MOT scan request changed after preparation"
                            )
                        return prepared.start(slot.preview_port)

                    return MotFieldTaskHandle(
                        request,
                        report_folder=current.folder,
                        start_scan=start_exact_scan,
                        materialize_scan=experiment.readout.materialize_scan,
                    )
                except BaseException:
                    if attached:
                        data_plane.detach_live(node)
                    else:
                        slot.close()
                    raise

            node.bind_starter(start_mot_field)
            node.bind_final_projector(
                lambda result, current=node: project_final_signals(
                    experiment,
                    current,
                    result,
                )
            )
            return node
        if spec.key == PULSE_SCAN_MEASUREMENT_KEY:
            intent = spec.build_request(values)
            if not isinstance(intent, PulseScanBindingIntent):
                raise TypeError("Pulse scan form did not produce its typed intent")
            request, serving_nodes = freeze_pulse_scan_request(intent)
            serving_nodes = tuple(
                {id(node): node for node in serving_nodes if node is not None}.values()
            )
            for serving in serving_nodes:
                serving.cancel("Pulse scan is taking exact source ownership")

            def prepare_scan(current):
                if current != intent:
                    raise RuntimeError(
                        "Pulse scan binding changed after request freeze"
                    )
                for serving in serving_nodes:
                    retire_for_exact_scan(serving)
                return request

            node = ConsoleRunNode(
                spec,
                values,
                prepare=prepare_scan,
                request_owner_wake=request_owner_wake,
                frozen_request=intent,
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
        **kwargs,
    )
    console.append(body)
    return body


def _bind_camera_execution(node, data_plane) -> None:
    """Start the one Camera definition as live or finite from its typed request."""

    from zlc_frontend.figure import DatasetId
    from zlc_neutral_atom.capture_application import PreparedFiniteCameraMeasurement
    from zlc_neutral_atom.monitor_application import PreparedLiveCameraMeasurement
    from zlc_workbench.live_slot import LiveDatasetSlot

    def start(command):
        if isinstance(command, PreparedLiveCameraMeasurement):
            dataset_id = DatasetId(
                f"console-{node.spec.key.stable_definition_id}-{id(node):x}"
            )

            def live_factory(view_spec):
                slot = LiveDatasetSlot(
                    view_spec,
                    dataset_id=dataset_id,
                    retain_on_terminal=True,
                )
                data_plane.attach(node, slot, output_name="frame")
                slot.set_change_listener(lambda: data_plane.mark_changed(node))
                return slot

            return command.start_with_view(factory=live_factory)
        if not isinstance(command, PreparedFiniteCameraMeasurement):
            raise TypeError(
                "Camera execution requires a prepared live or finite Camera "
                "measurement"
            )
        return _start_capture_preview(command, node, data_plane)

    node.bind_starter(start)


def _start_capture_preview(
    command,
    node,
    data_plane,
    *,
    output_name: str = "frame",
    source_ordinals: tuple[int, ...] | None = None,
):
    """Start one finite exact capture with the ordinary TaskConsole frame view."""

    import uuid

    from zlc_data import BlockId
    from zlc_frontend.figure import DatasetId
    from zlc_workbench.live_slot import LiveDatasetSlot

    try:
        command.preview_schema
    except ValueError:
        return command.start()

    token = uuid.uuid4().hex

    def factory(preview_spec):
        slot = LiveDatasetSlot(
            preview_spec,
            dataset_id=DatasetId(f"console-capture-{token}"),
            retain_on_terminal=True,
        )
        data_plane.attach(
            node,
            slot,
            output_name=output_name,
        )
        slot.set_change_listener(lambda: data_plane.mark_changed(node))
        return slot

    return command.start_with_preview(
        block_id=BlockId(f"console-capture-preview-{token}"),
        factory=factory,
        source_ordinals=source_ordinals,
    )
