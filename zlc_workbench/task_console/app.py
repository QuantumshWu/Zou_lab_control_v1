"""Compose and open the TaskConsole from explicit application capabilities.

The outer application supplies one :class:`TaskConsoleApplicationPorts` value.
This module projects the installed Definition catalog into UI forms, freezes
those forms into typed application intents, binds their prepared run commands to
the console lifecycle, and connects typed live/final outputs to the data plane.
The window owns interaction and presentation only; domain request construction,
acquisition, processing, and dataset schemas remain with their package owners.
"""

from __future__ import annotations

from .application_ports import TaskConsoleApplicationPorts

__all__ = ["TaskConsoleApplicationPorts", "open_task_console"]


def open_task_console(
    ports: TaskConsoleApplicationPorts,
    *,
    state=None,
    task=None,
    **kwargs,
):
    """Open the console from one closed application-port value.

    The shell is handed immutable installation facts plus the exact operations
    it can perform.  It cannot discover another service or retain an owning
    application object.
    """

    if not isinstance(ports, TaskConsoleApplicationPorts):
        raise TypeError("ports must be TaskConsoleApplicationPorts")

    from zlc_neutral_atom.acquisition import CAMERA_MEASUREMENT_KEY
    from zlc_neutral_atom.camera_measurement import CameraMeasurementRequest
    from zlc_neutral_atom.mot_field import MOT_FIELD_TASK_KEY
    from zlc_neutral_atom.mot_field_task import (
        MotFieldTaskIntent,
        PreparedMotFieldTask,
    )
    from zlc_neutral_atom.readout.calibration_task import (
        CalibrationTaskIntent,
        PreparedCalibrationTask,
    )
    from zlc_neutral_atom.readout.calibration_reference import (
        CalibrationArtifactRef,
    )
    from zlc_neutral_atom.readout.calibration_projection import (
        CALIBRATION_FINAL_OUTPUT_NAMES,
    )
    from zlc_neutral_atom.readout.occupancy import (
        OCCUPANCY_STREAM_PROCESSOR_KEY,
    )
    from zlc_neutral_atom.readout.coupled_measurements import (
        GREY_MOLASSES_DETUNING_KEY,
        READOUT_DURATION_FIDELITY_KEY,
        TEMPERATURE_RELEASE_RECAPTURE_KEY,
        GreyMolassesDetuningIntent,
        ReadoutDurationFidelityIntent,
        TemperatureReleaseRecaptureIntent,
    )
    from zlc_neutral_atom.readout.coupled_application import (
        CoupledMeasurementApplicationCommand,
    )
    from zlc_neutral_atom.readout.sitemap import SITEMAP_CALIBRATION_TASK_KEY
    from zlc_neutral_atom.scan import PULSE_SCAN_MEASUREMENT_KEY
    from zlc_neutral_atom.scan.application import PreparedExactScan
    from .catalog_bridge import ConsoleCatalogView
    from .calibration_presentation import CalibrationFinalPresentationAdapter
    from .data_plane import ConsoleDataPlane
    from .occupancy_binding import (
        OccupancyBindingIntent,
        ReactiveOccupancyNode,
    )
    from .pulse_scan_binding import (
        PulseScanBindingIntent,
        resolve_typed_scan_source,
    )
    from .coupled_measurement_forms import (
        CoupledMeasurementBinding,
    )
    from .window import show_task_console
    from .run_bridge import ConsoleRunNode

    catalog_view = ConsoleCatalogView(
        installed_camera_roles=ports.installed_camera_roles,
        sitemap_camera_roles=ports.sitemap_camera_roles,
        installed_rf_roles=ports.installed_rf_roles,
        camera_request_builder=ports.build_camera_measurement_request,
    )
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
            or calibration.output_name != CALIBRATION_FINAL_OUTPUT_NAMES[0]
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

    def resolve_scan_calibration(
        occupancy: OccupancyBindingIntent,
    ) -> CalibrationArtifactRef:
        """Resolve the operator's explicit calibration choice to one ref."""

        if occupancy.calibration_signal is not None:
            return resolve_calibration_reference(
                occupancy.calibration_signal,
                measurement_name="Pulse scan Occupancy y",
            )
        path = occupancy.calibration_ref_path
        if path is None:
            raise RuntimeError("Pulse scan Occupancy source lost its calibration")
        return ports.load_saved_calibration_reference(path)

    def coupled_measurement_node(
        spec,
        values,
        *,
        instance_id: str,
        instance_label: str,
        intent_type,
        measurement_name: str,
        prepare_application,
    ):
        """Bind one of the three closed coupled Measurement commands."""

        binding = spec.build_request(values)
        if (
            not isinstance(binding, CoupledMeasurementBinding)
            or not isinstance(binding.intent, intent_type)
        ):
            raise TypeError(
                f"{measurement_name} form did not produce its typed binding"
            )
        calibration_ref = resolve_calibration_reference(
            binding.calibration_signal,
            measurement_name=measurement_name,
        )

        def prepare(current_binding):
            if current_binding != binding:
                raise RuntimeError(
                    f"{measurement_name} binding changed after request freeze"
                )
            return prepare_application(
                current_binding.intent,
                calibration_ref,
            )

        node = ConsoleRunNode(
            spec,
            values,
            instance_id=instance_id,
            instance_label=instance_label,
            prepare=prepare,
            request_owner_wake=request_owner_wake,
            frozen_request=binding,
        )

        def start(command):
            if not isinstance(command, CoupledMeasurementApplicationCommand):
                raise TypeError(
                    f"{measurement_name} prepare returned another command"
                )
            return command.start()

        node.bind_starter(start)
        return node

    def run_factory(
        spec,
        values,
        *,
        instance_id: str,
        instance_label: str,
    ):
        if spec.key == CAMERA_MEASUREMENT_KEY:
            def prepare_camera(request):
                if isinstance(request, CameraMeasurementRequest):
                    return ports.prepare_camera_measurement(request)
                raise TypeError("Camera form must produce CameraMeasurementRequest")

            node = ConsoleRunNode(
                spec,
                values,
                instance_id=instance_id,
                instance_label=instance_label,
                prepare=prepare_camera,
                request_owner_wake=request_owner_wake,
            )
            _bind_camera_execution(node, data_plane)
            return node
        if spec.key == TEMPERATURE_RELEASE_RECAPTURE_KEY:
            return coupled_measurement_node(
                spec,
                values,
                instance_id=instance_id,
                instance_label=instance_label,
                intent_type=TemperatureReleaseRecaptureIntent,
                measurement_name="Temperature",
                prepare_application=ports.prepare_temperature_release_recapture,
            )
        if spec.key == READOUT_DURATION_FIDELITY_KEY:
            return coupled_measurement_node(
                spec,
                values,
                instance_id=instance_id,
                instance_label=instance_label,
                intent_type=ReadoutDurationFidelityIntent,
                measurement_name="Fidelity vs duration",
                prepare_application=ports.prepare_readout_duration_fidelity,
            )
        if spec.key == GREY_MOLASSES_DETUNING_KEY:
            return coupled_measurement_node(
                spec,
                values,
                instance_id=instance_id,
                instance_label=instance_label,
                intent_type=GreyMolassesDetuningIntent,
                measurement_name="Grey molasses detuning",
                prepare_application=ports.prepare_grey_molasses_detuning,
            )
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
                or not isinstance(source.request, CameraMeasurementRequest)
                or source.output_name not in source.request.output_names
            ):
                raise ValueError(
                    "occupancy Camera source must select one frame_i output of "
                    "a Camera Measurement row in this TaskConsole"
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
            if intent.calibration_signal is not None:
                calibration_ref = resolve_calibration_reference(
                    intent.calibration_signal,
                    measurement_name="Occupancy",
                )
            else:
                calibration_path = intent.calibration_ref_path
                if calibration_path is None:
                    raise RuntimeError("occupancy lost its saved calibration path")
                calibration_ref = ports.load_saved_calibration_reference(
                    calibration_path
                )

            def prepare_occupancy_application():
                return ports.prepare_reactive_occupancy(
                    source.request,
                    source.output_name,
                    calibration_ref=calibration_ref,
                    model_kind=intent.model_kind,
                )

            return ReactiveOccupancyNode(
                spec,
                values,
                instance_id=instance_id,
                instance_label=instance_label,
                intent=intent,
                source_node=source.run_node,
                initial_source=source_value,
                prepare_application=prepare_occupancy_application,
                data_plane=data_plane,
                request_owner_wake=request_owner_wake,
            )
        if spec.key == SITEMAP_CALIBRATION_TASK_KEY:
            intent = spec.build_request(values)
            if not isinstance(intent, CalibrationTaskIntent):
                raise TypeError("calibration catalog returned invalid intent")

            node = ConsoleRunNode(
                spec,
                values,
                instance_id=instance_id,
                instance_label=instance_label,
                prepare=ports.prepare_calibration_task,
                request_owner_wake=request_owner_wake,
                frozen_request=intent,
                final_presentation_owner=CalibrationFinalPresentationAdapter(),
            )

            def start_calibration_task(command):
                if not isinstance(command, PreparedCalibrationTask):
                    raise TypeError(
                        "Calibration application prepare returned another command"
                    )
                if not command.has_live_output:
                    return command.start()
                return command.start(
                    _CalibrationLivePreviewHost(node, data_plane)
                )

            node.bind_starter(start_calibration_task)
            return node
        if spec.key == MOT_FIELD_TASK_KEY:
            intent = spec.build_request(values)
            if not isinstance(intent, MotFieldTaskIntent):
                raise TypeError("MOT form did not produce MotFieldTaskIntent")

            node = ConsoleRunNode(
                spec,
                values,
                instance_id=instance_id,
                instance_label=instance_label,
                prepare=ports.prepare_mot_field_task,
                request_owner_wake=request_owner_wake,
                frozen_request=intent,
            )

            def start_mot_field(prepared):
                if not isinstance(prepared, PreparedMotFieldTask):
                    raise TypeError(
                        "MOT application prepare returned another command"
                    )
                live_output = prepared.live_output
                attached = False
                try:
                    data_plane.attach(node, live_output)
                    attached = True
                    live_output.set_change_listener(
                        lambda: data_plane.mark_changed(node)
                    )
                    return prepared.start()
                except BaseException:
                    if attached:
                        data_plane.detach_live(node)
                    else:
                        live_output.close()
                    raise

            node.bind_starter(start_mot_field)
            return node
        if spec.key == PULSE_SCAN_MEASUREMENT_KEY:
            intent = spec.build_request(values)
            if not isinstance(intent, PulseScanBindingIntent):
                raise TypeError("Pulse scan form did not produce its typed intent")
            if not console:
                raise RuntimeError(
                    "TaskConsole composition is not ready for Pulse scan binding"
                )
            selected = console[0].resolve_pulse_scan_source(intent.y_signal)
            source, serving_nodes = resolve_typed_scan_source(
                selected,
                resolve_producer=console[0].resolve_console_producer,
                resolve_calibration=resolve_scan_calibration,
            )

            def prepare_scan(current):
                if current != intent:
                    raise RuntimeError(
                        "Pulse scan binding changed after request freeze"
                    )
                for serving in serving_nodes:
                    serving.cancel(
                        "Pulse scan is taking exact source ownership"
                    )
                for serving in serving_nodes:
                    retire_for_exact_scan(serving)
                return ports.prepare_scan_source(
                    current.program,
                    source,
                )

            def start_scan(command):
                if not isinstance(command, PreparedExactScan):
                    raise TypeError(
                        "Pulse scan prepare returned another command"
                    )
                return command.start()

            node = ConsoleRunNode(
                spec,
                values,
                instance_id=instance_id,
                instance_label=instance_label,
                prepare=prepare_scan,
                request_owner_wake=request_owner_wake,
                frozen_request=intent,
            )
            node.bind_starter(start_scan)
            return node
        raise RuntimeError(
            "TaskConsole catalog/runtime binding invariant violated for "
            f"{spec.key}"
        )

    body = show_task_console(
        state=state, task=task,
        catalog_view=catalog_view,
        run_factory=run_factory,
        data_plane=data_plane,
        pulse_template_reader=ports.read_pulse_template,
        **kwargs,
    )
    console.append(body)
    return body


class _CalibrationLivePreviewHost:
    """Host one task-owned preview slot without interpreting calibration data."""

    __slots__ = ("_data_plane", "_node", "_slot")

    def __init__(self, node, data_plane) -> None:
        self._node = node
        self._data_plane = data_plane
        self._slot = None

    def open_calibration_preview(self, spec, *, output_owner):
        from zlc_frontend.figure import DatasetId
        from zlc_workbench.live_slot import LiveDatasetSlot

        if self._slot is not None:
            raise RuntimeError("calibration task already owns a live preview")
        slot = LiveDatasetSlot(
            spec,
            dataset_id=DatasetId(f"console-calibration-{id(self._node):x}"),
            retain_on_terminal=True,
            output_owner=output_owner,
        )
        try:
            self._data_plane.attach(self._node, slot)
            slot.set_change_listener(
                lambda: self._data_plane.mark_changed(self._node)
            )
        except BaseException:
            slot.close()
            raise
        self._slot = slot
        return slot


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
            attached = False

            def live_factory(view_spec):
                nonlocal attached
                slot = LiveDatasetSlot(
                    view_spec,
                    dataset_id=dataset_id,
                    retain_on_terminal=True,
                    output_owner=command,
                )
                try:
                    data_plane.attach(node, slot)
                except BaseException:
                    slot.close()
                    raise
                attached = True
                slot.set_change_listener(lambda: data_plane.mark_changed(node))
                return slot

            try:
                return command.start_with_view(factory=live_factory)
            except BaseException:
                # The domain asks for its view before runtime admission.  A
                # rejected resource claim therefore owns no RunHandle but may
                # already own this provisional slot.  Roll that one boundary
                # back so the same frozen request can be retried cleanly.
                if attached:
                    data_plane.detach_live(node)
                raise
        if not isinstance(command, PreparedFiniteCameraMeasurement):
            raise TypeError(
                "Camera execution requires a prepared live or finite Camera "
                "measurement"
            )
        if command.live_preview_output_name is None:
            # The capacity-one exact preview has no READOUT_EVENT identity and
            # therefore cannot truthfully publish a multi-frame cycle.  The
            # committed FINAL artifact publishes every named frame_i instead.
            return command.start()
        return _start_capture_preview(
            command,
            node,
            data_plane,
            output_owner=command,
        )

    node.bind_starter(start)


def _start_capture_preview(
    command,
    node,
    data_plane,
    *,
    output_owner,
):
    """Start one finite exact capture with its application output owner."""

    import uuid

    from zlc_frontend.figure import DatasetId
    from zlc_workbench.live_slot import LiveDatasetSlot

    try:
        command.preview_schema
    except ValueError:
        return command.start()

    token = uuid.uuid4().hex
    attached = False

    def factory(preview_spec):
        nonlocal attached
        slot = LiveDatasetSlot(
            preview_spec,
            dataset_id=DatasetId(f"console-capture-{token}"),
            retain_on_terminal=True,
            output_owner=output_owner,
        )
        try:
            data_plane.attach(node, slot)
        except BaseException:
            slot.close()
            raise
        attached = True
        slot.set_change_listener(lambda: data_plane.mark_changed(node))
        return slot

    try:
        return command.start_with_preview(
            factory=factory,
        )
    except BaseException:
        if attached:
            data_plane.detach_live(node)
        raise
