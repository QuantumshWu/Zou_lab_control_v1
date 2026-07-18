"""M2c coherent ROI curve, histogram, and latest-meter display oracles."""

from __future__ import annotations

import os
import time
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt5 import QtCore, QtTest, QtWidgets
import pytest

import Zou_lab_control.notebook as zlc
from Zou_lab_control.workbench._camera_monitor import CameraMonitorWorkbenchWindow
from zlc_data import (
    MONITOR_HISTORY,
    REPEAT,
    AxisId,
    AxisSpec,
    BlockId,
    CellValidity,
    DataBlock,
    DatasetRevision,
    DatasetSchema,
    OwnedSnapshot,
    PointLayout,
    ReductionMethod,
    Selection,
    StreamGenerationId,
    ValidityContract,
    ValueSchema,
    expand_dataset_validity,
)
from zlc_frontend.figure import (
    AxisViewBinding,
    AxisViewRole,
    DatasetDescriptor,
    DatasetId,
    EvaluatedHistogram,
    EvaluatedMeter,
    FigureDocument,
    FigureEvaluationPolicy,
    FigureEvaluator,
    FigureLayer,
    FixedIndex,
    ResolvedDataset,
    ResolvedDatasetMap,
    ViewIntent,
    ViewSpec,
)
from zlc_frontend.matplotlib_render import (
    SinglePanelAggRenderer,
    histogram_bin_counts,
)
from zlc_frontend.qt_widgets import QtRasterBoard
from zlc_neutral_atom.runtime.dataset import MonitorCoverage
from zlc_workbench.live import LiveFrontStatus


@pytest.fixture(scope="module")
def application():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture(scope="module")
def experiment(tmp_path_factory):
    exp = zlc.connect(
        "virtual",
        repository=tmp_path_factory.mktemp("m2c-camera-monitor"),
    )
    yield exp
    exp.close()


def _until(application, predicate, *, timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        application.processEvents(QtCore.QEventLoop.AllEvents, 20)
        time.sleep(0.005)
    assert predicate()


def _close_window(application, window) -> None:
    window.close()
    _until(application, lambda: not window.isVisible(), timeout=8.0)
    window.deleteLater()
    application.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
    application.processEvents()


def _typed_roi(experiment):
    schema = experiment.readout.inspect_camera_monitor(
        experiment.readout.camera_monitor_request(history_capacity=2)
    ).output_schema
    y_axis, x_axis = schema.cell_schema.data_axes
    return Selection.rectangle(
        x_axis.axis_id,
        y_axis.axis_id,
        x_axis.coordinates[0],
        x_axis.coordinates[min(15, x_axis.size - 1)],
        y_axis.coordinates[0],
        y_axis.coordinates[min(11, y_axis.size - 1)],
        coordinate_frame=x_axis.coordinate_frame,
    )


def test_four_panel_board_uses_one_scalar_revision_and_typed_view_semantics(
    experiment,
    application,
):
    request = experiment.readout.camera_monitor_request(
        history_capacity=3,
        roi=_typed_roi(experiment),
        roi_reduction=ReductionMethod.MEAN,
        scalar_history_capacity=12,
    )
    window = experiment.readout.camera_monitor_gui(request)
    try:
        start = window.findChild(QtWidgets.QPushButton, "startButton")
        board = window.findChild(QtRasterBoard, "cameraMonitorImageBoard")
        view_status = window.findChild(QtWidgets.QLabel, "monitorViewStatus")
        _until(application, start.isEnabled)
        QtTest.QTest.mouseClick(start, QtCore.Qt.LeftButton)
        _until(
            application,
            lambda: (
                board.has_front
                and window._live is not None
                and window._live.front_status is not None
                and window._live.front_status.sequence
                == board.front_frame.sequence
                and window._live.front_status.scalar_coverage is not None
                and window._live.front_status.scalar_coverage.written_cells >= 4
                and window._live.front_status.histogram_valid_samples is not None
            ),
        )

        frame = board.front_frame
        assert frame is not None
        assert tuple(panel.panel_id for panel in frame.panels) == (
            "camera-monitor-image",
            "camera-monitor-roi-curve",
            "camera-monitor-roi-histogram",
            "camera-monitor-roi-meter",
        )
        assert all(
            panel.coherence_stamp == frame.panels[0].coherence_stamp
            for panel in frame.panels
        )
        assert frame.panels[0].source_identity != frame.panels[1].source_identity
        assert all(
            panel.source_identity == frame.panels[1].source_identity
            for panel in frame.panels[1:]
        )
        stamp = frame.panels[0].coherence_stamp
        assert len(stamp.inputs) == 2
        assert len(stamp.presentations) == 4
        assert tuple(
            document.layers[0].view.intent
            for document in window._live._scalar_documents
        ) == (ViewIntent.CURVE, ViewIntent.HISTOGRAM, ViewIntent.METER)

        status = window._live.front_status
        assert status is not None and status.sequence == frame.sequence
        scalar_coverage = status.scalar_coverage
        valid_samples = status.histogram_valid_samples
        dropped_samples = status.histogram_dropped_samples
        assert scalar_coverage is not None
        assert valid_samples is not None and dropped_samples is not None
        assert valid_samples <= scalar_coverage.written_cells
        assert valid_samples + dropped_samples == scalar_coverage.total_cells
        assert isinstance(status.latest_scalar_valid, bool)
        assert status.scalar_source_missed == 0
        assert "valid/retained/capacity=" in view_status.text()

        _run, _epoch, joined = window._slot.freeze_camera_current()
        scalar = joined.scalar
        assert scalar is not None
        resolved = ResolvedDatasetMap(
            (
                ResolvedDataset(
                    window._slot.scalar_dataset_id,
                    scalar.snapshot,
                ),
            )
        )
        evaluated = tuple(
            FigureEvaluator(window._slot.evaluation_policy).evaluate(document, resolved)
            for document in window._live._scalar_documents
        )
        histogram = evaluated[1].layers[0].cells[0].series[0].data
        meter = evaluated[2].layers[0].cells[0].series[0].data
        assert isinstance(histogram, EvaluatedHistogram)
        assert isinstance(meter, EvaluatedMeter)
        validity = expand_dataset_validity(
            scalar.block.validity,
            scalar.block.schema,
        )[0]
        np.testing.assert_array_equal(
            histogram.samples,
            scalar.block.values[0][validity],
        )
        assert histogram.dropped_count == request.scalar_history_capacity - int(
            np.count_nonzero(validity)
        )
        assert meter.value == scalar.block.values[0, 0]
        assert meter.valid is bool(validity[0])
    finally:
        if window.isVisible():
            _close_window(application, window)


def _scalar_documents(values, validity):
    repeat = AxisSpec(AxisId("m2c.repeat"), "repeat", REPEAT, 1)
    history = AxisSpec(
        AxisId("m2c.history"),
        "history",
        MONITOR_HISTORY,
        3,
        (0, 1, 2),
    )
    values = np.asarray(values, dtype=np.float64).reshape(1, 3)
    validity = np.asarray(validity, dtype=bool).reshape(1, 3)
    schema = DatasetSchema(
        repeat,
        (history,),
        PointLayout.rect_c((3,)),
        ValueSchema((), ValidityContract.value(), values.dtype),
    )
    block = DataBlock(
        BlockId("m2c-scalar-block"),
        DatasetRevision(1),
        values,
        CellValidity(validity),
        schema,
    )
    dataset_id = DatasetId("m2c-scalar")
    resolved = ResolvedDatasetMap(
        (
            ResolvedDataset(
                dataset_id,
                OwnedSnapshot(
                    block.ref(StreamGenerationId("m2c-generation")),
                    block,
                ),
            ),
        )
    )
    repeat_selected = AxisViewBinding(
        repeat.axis_id,
        AxisViewRole.SELECTED,
        selector=FixedIndex(0),
    )
    views = (
        ViewSpec(
            schema.fingerprint,
            ViewIntent.CURVE,
            (
                AxisViewBinding(history.axis_id, AxisViewRole.X),
                repeat_selected,
            ),
        ),
        ViewSpec(
            schema.fingerprint,
            ViewIntent.HISTOGRAM,
            (
                AxisViewBinding(history.axis_id, AxisViewRole.SAMPLE),
                repeat_selected,
            ),
        ),
        ViewSpec(
            schema.fingerprint,
            ViewIntent.METER,
            (
                AxisViewBinding(
                    history.axis_id,
                    AxisViewRole.SELECTED,
                    selector=FixedIndex(0),
                ),
                repeat_selected,
            ),
        ),
    )
    documents = tuple(
        FigureDocument(
            f"m2c-{view.intent.value.lower()}",
            0,
            (DatasetDescriptor(dataset_id, "scalar", schema.fingerprint),),
            (FigureLayer(f"m2c-{view.intent.value.lower()}", dataset_id, view),),
        )
        for view in views
    )
    return documents, resolved


def test_invalid_latest_is_not_replaced_by_an_older_valid_scalar():
    documents, resolved = _scalar_documents(
        [99.0, 2.0, 1.0],
        [False, True, True],
    )
    histogram_figure, meter_figure = (
        FigureEvaluator().evaluate(document, resolved) for document in documents[1:]
    )
    histogram = histogram_figure.layers[0].cells[0].series[0].data
    meter = meter_figure.layers[0].cells[0].series[0].data
    assert isinstance(histogram, EvaluatedHistogram)
    np.testing.assert_array_equal(histogram.samples, [2.0, 1.0])
    assert histogram.dropped_count == 1
    assert isinstance(meter, EvaluatedMeter)
    assert meter.value == 99.0
    assert meter.valid is False

    renderer = SinglePanelAggRenderer(documents[2], width=320, height=240)
    try:
        raster = renderer.render(meter_figure)
        assert raster.pixels
        assert renderer._artists[0].get_text() == "invalid"
    finally:
        renderer.close()


def test_valid_nonfinite_curve_and_meter_fail_instead_of_changing_validity():
    for nonfinite in (np.nan, np.inf, -np.inf):
        documents, resolved = _scalar_documents(
            [nonfinite, 2.0, 1.0],
            [True, True, True],
        )
        for document in (documents[0], documents[2]):
            evaluated = FigureEvaluator().evaluate(document, resolved)
            renderer = SinglePanelAggRenderer(document, width=320, height=240)
            try:
                with pytest.raises(ValueError, match="must .* finite"):
                    renderer.render(evaluated)
            finally:
                renderer.close()


def test_histogram_binning_is_deterministic_and_never_fakes_empty_data():
    counts, edges = histogram_bin_counts(np.array([], dtype=np.float64))
    assert counts.shape == (60,) and not np.any(counts)
    assert edges.shape == (61,)

    counts, edges = histogram_bin_counts(np.array([0.0, 1.0, 2.0]), bins=2)
    np.testing.assert_array_equal(counts, [1, 2])
    np.testing.assert_array_equal(edges, [0.0, 1.0, 2.0])

    counts, edges = histogram_bin_counts(np.array([False, True, True]), bins=500)
    np.testing.assert_array_equal(counts, [1, 2])
    np.testing.assert_array_equal(edges, [-0.5, 0.5, 1.5])

    counts, _edges = histogram_bin_counts(np.full(5, 3.0))
    assert int(counts.sum()) == 5
    with pytest.raises(ValueError, match="finite"):
        histogram_bin_counts(np.array([1.0, np.nan]))


def test_scalar_display_budget_rejects_before_monitor_arm(experiment, application):
    roi = _typed_roi(experiment)
    roomy = experiment.readout.camera_monitor_request(
        history_capacity=2,
        roi=roi,
        scalar_history_capacity=8,
        memory_limit_bytes=1 << 30,
    )
    base_peak = experiment.readout.inspect_camera_monitor(roomy).base_peak_bytes
    request = experiment.readout.camera_monitor_request(
        history_capacity=2,
        roi=roi,
        scalar_history_capacity=8,
        memory_limit_bytes=base_peak + 1,
    )
    window = experiment.readout.camera_monitor_gui(request)
    try:
        start = window.findChild(QtWidgets.QPushButton, "startButton")
        _until(application, lambda: "peak budget" in window._local_diagnostic)
        assert not start.isEnabled()
        assert window._handle is None
        assert window._slot is None
    finally:
        if window.isVisible():
            _close_window(application, window)

    oversized = experiment.readout.camera_monitor_request(
        history_capacity=2,
        roi=roi,
        scalar_history_capacity=FigureEvaluationPolicy().max_histogram_samples + 1,
        memory_limit_bytes=1 << 30,
    )
    window = experiment.readout.camera_monitor_gui(oversized)
    try:
        start = window.findChild(QtWidgets.QPushButton, "startButton")
        _until(
            application,
            lambda: "histogram sample limit" in window._local_diagnostic,
        )
        assert not start.isEnabled()
        assert window._handle is None
        assert window._slot is None
    finally:
        if window.isVisible():
            _close_window(application, window)


def test_owner_presents_before_next_admission_and_status_is_sequence_gated():
    events = []

    class Board:
        fault = None
        pending_sequence = 7
        front_sequence = None

        def present_pending(self):
            events.append("present")
            self.front_sequence = self.pending_sequence
            self.pending_sequence = None

    board = Board()
    status = LiveFrontStatus(
        7,
        MonitorCoverage(1, 3, 0, False),
    )

    class Live:
        fault = None
        front_status = status

        def reconcile_faults(self):
            return False

        def admit_pending(self):
            events.append("admit")
            assert board.front_sequence == status.sequence
            board.pending_sequence = 8

    harness = SimpleNamespace(
        _drain_results=lambda: None,
        _drain_attachments=lambda: None,
        _drain_roi_controls=lambda: None,
        _live=Live(),
        _board=board,
        _selector_interacting=False,
        _apply_phase=None,
        _slot=None,
        _run_owner=SimpleNamespace(take_pending_snapshot=lambda: None),
        _refresh_view_status=lambda: events.append("status"),
        _record_local_failure=lambda message: events.append(("failure", message)),
        _maybe_finish_close=lambda: None,
    )
    CameraMonitorWorkbenchWindow._owner_cycle(harness)
    assert events[:3] == ["present", "admit", "status"]

    class Label:
        text = ""

        def setText(self, value):
            self.text = value

    label = Label()
    status_harness = SimpleNamespace(
        _slot=None,
        _live=SimpleNamespace(fault=None, front_status=status),
        _board=SimpleNamespace(fault=None),
        _board_widget=SimpleNamespace(
            has_front=True,
            front_frame=SimpleNamespace(sequence=6),
        ),
        _projection_text="projection",
        _roi_presentation_failure=None,
        _view_status=label,
        _handle=None,
        _reconcile_visible_roi=lambda _status: True,
        _update_roi_controls=lambda: None,
    )
    CameraMonitorWorkbenchWindow._refresh_view_status(status_harness)
    assert label.text == "View: WAITING · previous front retained · diagnostics pending"
    status_harness._board_widget.front_frame = SimpleNamespace(sequence=7)
    CameraMonitorWorkbenchWindow._refresh_view_status(status_harness)
    assert "raw missed=0" in label.text


def test_close_failure_after_front_clear_immediately_drops_visible_label():
    board_widget = SimpleNamespace(has_front=True)
    projection_label = SimpleNamespace(text=None)
    projection_label.setText = lambda value: setattr(
        projection_label,
        "text",
        value,
    )

    class Live:
        def close(self):
            board_widget.has_front = False
            raise RuntimeError("synthetic cleanup failure after front clear")

    harness = SimpleNamespace(
        _live=Live(),
        _fold_roi_state_for_detach=lambda: None,
        _board_widget=board_widget,
        _visible_binding_fingerprint="binding",
        _visible_projection_text="old projection",
        _projection_text="target projection",
        _projection_status=projection_label,
    )
    harness._show_projection_target = lambda: (
        CameraMonitorWorkbenchWindow._show_projection_target(harness)
    )

    with pytest.raises(RuntimeError, match="cleanup failure after front clear"):
        CameraMonitorWorkbenchWindow._close_live(harness)

    assert harness._visible_binding_fingerprint is None
    assert harness._visible_projection_text is None
    assert projection_label.text == "Display: no coherent front · target target projection"


def test_selector_end_cannot_resume_a_sticky_presentation_failure():
    calls = []
    live = SimpleNamespace(
        fault=None,
        pause=lambda: calls.append("pause"),
        resume=lambda: calls.append("resume"),
    )
    harness = SimpleNamespace(
        _live=live,
        _slot=SimpleNamespace(
            failure=None,
            notification_failure=None,
            terminal=False,
            withdrawn=False,
        ),
        _board=SimpleNamespace(fault=None),
        _roi_presentation_failure="sticky presentation failure",
        _selector_interacting=False,
    )

    CameraMonitorWorkbenchWindow._sync_live_pause(harness)

    assert calls == ["pause"]


def test_every_presentation_failure_path_freezes_before_recording_status():
    events = []
    status = SimpleNamespace(text=None)
    status.setText = lambda value: setattr(status, "text", value)
    request = object()
    harness = SimpleNamespace(
        _live=SimpleNamespace(
            freeze_presentation=lambda: events.append("freeze")
        ),
        _applied_request=None,
        _request=None,
        _draft_selection=None,
        _roi_presentation_failure=None,
        _roi_terminal_notice=None,
        _roi_status=status,
        _record_local_failure=lambda _message: events.append("record"),
        _update_roi_controls=lambda: events.append("controls"),
    )

    CameraMonitorWorkbenchWindow._record_roi_presentation_failure(
        harness,
        "pre-adopt authority mismatch",
        RuntimeError("synthetic pre-adopt failure"),
        request=request,
        state=None,
        draft_selection=None,
    )

    assert events == ["freeze", "record", "controls"]
    assert harness._applied_request is request and harness._request is request
    assert "synthetic pre-adopt failure" in harness._roi_presentation_failure
