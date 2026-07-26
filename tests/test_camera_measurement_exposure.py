"""Camera Measurement owns exposure through one run-scoped physical lease."""

from __future__ import annotations

import threading
from pathlib import Path
import time

import Zou_lab_control.api as zlc

from zlc_neutral_atom.logic_nodes.camera_measurement import (
    CameraMonitorViewSpec,
    PreparedFiniteCameraMeasurement,
    PreparedLiveCameraMeasurement,
)
from zlc_pulse import PulseExecutionForm, load_pulse_document


ROOT = Path(__file__).resolve().parents[1]


class _LiveView:
    def __init__(self, spec: CameraMonitorViewSpec) -> None:
        self.spec = spec
        self.terminal = False
        self.dataset = None
        self.updated_event = threading.Event()
        self.failure = None

    def bind(self, dataset, *, run_id: str, causation_domain_id: str) -> None:
        assert run_id and causation_domain_id
        self.dataset = dataset

    def updated(self) -> None:
        self.updated_event.set()

    def notification_failed(self, message: str) -> None:
        self.failure = message

    def fail(self, message: str) -> None:
        self.failure = message
        self.terminal = True
        self.updated_event.set()

    def source_terminal(self) -> None:
        self.terminal = True


def _start_one_live_frame(exp, *, exposure):
    request = exp.nodes.camera_measurement.camera_measurement_request(
        camera_role="mot_camera",
        exposure=exposure,
    )
    prepared = exp.nodes.camera_measurement.prepare_camera_measurement(request)
    assert isinstance(prepared, PreparedLiveCameraMeasurement)
    views: list[_LiveView] = []

    def factory(spec):
        view = _LiveView(spec)
        views.append(view)
        return view

    handle = prepared.start_with_view(factory=factory)
    view = views[0]
    assert view.updated_event.wait(5.0), handle.snapshot()
    assert view.failure is None
    assert view.dataset is not None
    frozen = view.dataset.freeze_current()
    assert frozen.coverage.written_cells >= 1
    outputs = prepared.live_dataset_outputs(frozen)
    frame = outputs["frame_0"].snapshot
    assert frame.block.schema.physical_shape[:2] == (1, 1)
    assert frame.block.schema.cell_schema == frozen.snapshot.block.schema.cell_schema
    assert frame.block.values.dtype == frozen.snapshot.block.values.dtype
    handle.cancel("exposure test complete")
    terminal = handle.wait(5.0)
    assert terminal.state.terminal, terminal
    assert view.failure is None


def test_live_camera_applies_and_restores_requested_exposure(tmp_path) -> None:
    with zlc.connect("virtual", repository=tmp_path / "workspace") as exp:
        _start_one_live_frame(exp, exposure=0.013)
        # A second baseline run executes the endpoint's unchanged-working-point
        # check.  It can succeed only if cleanup restored the installed exposure.
        _start_one_live_frame(exp, exposure=None)


def _run_finite_camera(exp, *, exposure, pulse_request):
    request = exp.nodes.camera_measurement.camera_measurement_request(
        camera_role="camera",
        repeat=1,
        frames_per_cycle=3,
        exposure=exposure,
    )
    prepared = exp.nodes.camera_measurement.prepare_camera_measurement(request)
    assert isinstance(prepared, PreparedFiniteCameraMeasurement)
    handle = prepared.start()
    deadline = time.monotonic() + 2.0
    while handle.snapshot().phase != "execute":
        if time.monotonic() >= deadline:
            raise AssertionError(handle.snapshot())
        time.sleep(0.005)
    exp.pulse.run(pulse_request)
    reference = handle.result(5.0)
    outputs = prepared.final_dataset_outputs(reference)
    assert tuple(outputs) == ("frame_0", "frame_1", "frame_2")


def test_finite_camera_applies_and_restores_requested_exposure(tmp_path) -> None:
    with zlc.connect("virtual", repository=tmp_path / "workspace") as exp:
        document = load_pulse_document(ROOT / "pulses" / "imaging_template.json")
        pulse_request = exp.pulse.request(
            document,
            PulseExecutionForm.STATIC_ONCE,
            api_values={
                parameter.parameter_id: document.field_value(
                    parameter.field
                )[0]
                for parameter in document.api_parameters
            },
        )
        _run_finite_camera(
            exp,
            exposure=0.013,
            pulse_request=pulse_request,
        )
        _run_finite_camera(
            exp,
            exposure=None,
            pulse_request=pulse_request,
        )
