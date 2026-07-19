"""Product contracts for the typed generic HISTOGRAM DataFigure window."""

from __future__ import annotations

from dataclasses import replace
import gc
import os
from pathlib import Path
import threading
import time
import weakref

import numpy as np
import pytest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtCore, QtGui, QtWidgets  # noqa: E402

from zlc_data import (  # noqa: E402
    COMPONENT,
    REPEAT,
    SCAN_POINT,
    SITE,
    AxisId,
    AxisSpec,
    BlockId,
    ComponentValidity,
    DataBlock,
    DatasetRevision,
    DatasetSchema,
    OwnedSnapshot,
    PointLayout,
    StreamGenerationId,
    ValidityContract,
    ValueSchema,
)
from zlc_frontend import DataFigure, HistogramPanelPayload  # noqa: E402
from zlc_frontend.display_range import RelimMode  # noqa: E402
from zlc_frontend.figure import (  # noqa: E402
    AxisViewBinding,
    AxisViewRole,
    DatasetDescriptor,
    DatasetId,
    FigureDocument,
    FigureLayer,
    FixedIndex,
    ResolvedDataset,
    ResolvedDatasetMap,
    ViewIntent,
    ViewSpec,
)
from zlc_frontend.histogram_display import (  # noqa: E402
    HistogramCountScale,
    HistogramDisplayState,
    histogram_display_form_values,
)
from zlc_frontend.qt_widgets import (  # noqa: E402
    FluentRevisionedFormEditor,
    QtImageBoard,
    QtRasterBoard,
    ensure_qt_app,
)
from zlc_frontend.selector import HistogramRangeGesture  # noqa: E402
from Zou_lab_control.workbench import open_data_figure_workbench  # noqa: E402


@pytest.fixture
def application():
    return ensure_qt_app()


def _until(application, predicate, *, timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        application.processEvents()
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("Qt condition did not become true")


def _close(application, window) -> None:
    window.shutdown()
    _until(application, lambda: window.closed)
    application.processEvents()


def _histogram_figure(
    *,
    site_role: AxisViewRole = AxisViewRole.BATCH,
    render_memory_limit_bytes: int = 64 << 20,
) -> DataFigure:
    repeat = AxisSpec(AxisId("u03a.repeat"), "Repeat", REPEAT, 3)
    point = AxisSpec(AxisId("u03a.point"), "Point", SCAN_POINT, 1)
    site = AxisSpec(
        AxisId("u03a.site"),
        "Site",
        SITE,
        2,
        coordinates=("left", "right"),
    )
    channel = AxisSpec(
        AxisId("u03a.channel"),
        "Channel",
        COMPONENT,
        2,
        coordinates=("signal", "reference"),
    )
    values = np.asarray(
        [
            [[[1.0, 2.0], [10.0, 11.0]]],
            [[[3.0, 4.0], [12.0, 13.0]]],
            [[[5.0, 6.0], [14.0, 15.0]]],
        ]
    )
    valid = np.ones_like(values, dtype=bool)
    valid[1, 0, 1, 0] = False
    schema = DatasetSchema(
        repeat,
        (point,),
        PointLayout.rect_c((1,)),
        ValueSchema(
            (site, channel),
            ValidityContract.components(site.axis_id, channel.axis_id),
            values.dtype,
            value_unit="photoelectron",
        ),
    )
    block = DataBlock(
        BlockId("u03a-histogram-block"),
        DatasetRevision(7),
        values,
        ComponentValidity((site.axis_id, channel.axis_id), valid),
        schema,
    )
    dataset_id = DatasetId("u03a-histogram-dataset")
    view = ViewSpec(
        schema.fingerprint,
        ViewIntent.HISTOGRAM,
        (
            AxisViewBinding(repeat.axis_id, AxisViewRole.SAMPLE),
            AxisViewBinding(
                point.axis_id,
                AxisViewRole.SELECTED,
                selector=FixedIndex(0),
            ),
            AxisViewBinding(site.axis_id, site_role),
            AxisViewBinding(channel.axis_id, AxisViewRole.SAMPLE),
        ),
    )
    document = FigureDocument(
        "u03a-histogram-document",
        2,
        (DatasetDescriptor(dataset_id, "site counts", schema.fingerprint),),
        (FigureLayer("histogram", dataset_id, view),),
    )
    snapshot = OwnedSnapshot(
        block.ref(StreamGenerationId("u03a-histogram-generation")),
        block,
    )
    return DataFigure(
        document,
        ResolvedDatasetMap((ResolvedDataset(dataset_id, snapshot),)),
        render_memory_limit_bytes=render_memory_limit_bytes,
    )


def _typed_front(window):
    board = window.findChild(QtRasterBoard, "figureViewerTypedBoard")
    assert board is not None and board.front_frame is not None
    frame = board.front_frame
    payload = frame.panels[0].display_payload
    assert isinstance(payload, HistogramPanelPayload)
    return board, frame, payload


def _initial_session_peak(figure: DataFigure, state: HistogramDisplayState) -> int:
    import Zou_lab_control.workbench._figure as figure_module

    front = figure_module._render_typed_front(
        figure,
        state,
        current_value_limits=None,
        previous_relim_mode=None,
        previous_count_scale=None,
        sequence=0,
        memory_limit_bytes=1 << 30,
        cancelled=threading.Event(),
    )
    return front.session_peak_bytes


def _wheel(board: QtRasterBoard, delta: int):
    binding = board._numeric_binding_for_kind(
        "histogram",
        panel_id="generic-typed",
    )
    assert binding is not None
    target = board._numeric_target(binding)
    assert target is not None
    position = QtCore.QPoint(
        int(round(target.plot.left() + 0.5 * target.plot.width())),
        int(round(target.plot.top() + 0.5 * target.plot.height())),
    )
    event = QtGui.QWheelEvent(
        QtCore.QPointF(position),
        QtCore.QPointF(board.mapToGlobal(position)),
        QtCore.QPoint(),
        QtCore.QPoint(0, delta),
        QtCore.Qt.NoButton,
        QtCore.Qt.NoModifier,
        QtCore.Qt.ScrollUpdate,
        False,
    )
    board.wheelEvent(event)
    return event


def test_generic_histogram_uses_typed_front_edits_and_exact_export(
    application,
    monkeypatch,
    tmp_path: Path,
) -> None:
    figure = _histogram_figure()
    evaluated = figure.evaluated
    owner_thread = threading.get_ident()
    render_threads: list[int] = []
    rerender_started = threading.Event()
    release_rerender = threading.Event()

    from zlc_frontend.matplotlib_render import SinglePanelAggRenderer

    original_render = SinglePanelAggRenderer.render_interactive_histogram

    def traced_render(self, *args, **kwargs):
        render_threads.append(threading.get_ident())
        if len(render_threads) == 2:
            rerender_started.set()
            if not release_rerender.wait(5.0):
                raise TimeoutError("test did not release histogram rerender")
        return original_render(self, *args, **kwargs)

    monkeypatch.setattr(
        SinglePanelAggRenderer,
        "render_interactive_histogram",
        traced_render,
    )
    monkeypatch.setattr(
        DataFigure,
        "to_png_bytes",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("typed HISTOGRAM must not render through a board PNG")
        ),
    )

    window = open_data_figure_workbench(figure)
    try:
        _until(application, lambda: window.worker_idle and window.raster_ready)
        board, first_frame, first_payload = _typed_front(window)
        assert len(render_threads) == 1
        assert render_threads[0] != owner_thread
        assert first_frame.panels[0].coherence_stamp.presentations[
            0
        ].selection_revision == 0
        assert first_payload.value_unit == "photoelectron"
        assert first_payload.series_labels == (
            "u03a.site=left",
            "u03a.site=right",
        )
        assert tuple(item.data.dropped_count for item in first_payload.series) == (0, 1)
        assert tuple(
            tuple(
                (coordinates.axis_id, coordinates.coordinates)
                for coordinates in item.data.sample_coordinates
            )
            for item in first_payload.series
        ) == (
            (
                (
                    AxisId("u03a.channel"),
                    ("signal", "reference", "signal", "reference", "signal", "reference"),
                ),
                (AxisId("u03a.repeat"), (0, 0, 1, 1, 2, 2)),
            ),
            (
                (
                    AxisId("u03a.channel"),
                    ("signal", "reference", "reference", "signal", "reference"),
                ),
                (AxisId("u03a.repeat"), (0, 0, 1, 2, 2)),
            ),
        )
        assert all(
            actual is expected
            for actual, expected in zip(
                first_payload.series,
                evaluated.layers[0].cells[0].series,
                strict=True,
            )
        )
        assert window.findChild(QtWidgets.QLabel, "figureViewerMode").text() == (
            "EXACT HISTOGRAM · INTERACTIVE · DISPLAY ONLY"
        )
        assert tuple(
            window._tabs.tabText(index) for index in range(window._tabs.count())
        ) == ("Histogram", "Edit")
        assert not window._interaction_switch.isHidden()
        assert not window._settings_button.isHidden()
        assert not window._export_button.isHidden()

        origin = board.visible_histogram_origin("generic-typed")
        assert origin is not None
        window._accept_numeric_interaction(
            HistogramRangeGesture(origin, (2.5, 12.5))
        )

        values = histogram_display_form_values(window._display)
        values["bin_count"] = 17
        values["count_scale"] = HistogramCountScale.LOG
        window._apply_display_form(window._edit_display, 0, values)
        _until(application, rerender_started.is_set)
        assert not window.raster_ready
        assert not window._interaction_switch.isEnabled()
        assert not board._numeric_bindings["generic-typed"].interaction_ready
        assert not _wheel(board, -120).isAccepted()
        assert board.histogram_selector_fault is None
        assert board.front_frame is first_frame
        assert len(render_threads) == 2
        release_rerender.set()
        _until(application, lambda: window.worker_idle and window.raster_ready)

        board, second_frame, second_payload = _typed_front(window)
        assert second_frame.sequence > first_frame.sequence
        assert second_frame.panels[0].coherence_stamp.presentations[
            0
        ].selection_revision == 0
        assert second_payload.viewport.display_revision == 1
        assert second_payload.viewport.bin_count == 17
        assert second_payload.viewport.count_scale is HistogramCountScale.LOG
        assert all(
            actual is expected
            for actual, expected in zip(
                second_payload.series,
                evaluated.layers[0].cells[0].series,
                strict=True,
            )
        )
        assert board._numeric_bindings["generic-typed"].applied_span == (
            2.5,
            12.5,
        )
        editors = window.findChildren(FluentRevisionedFormEditor)
        assert len(editors) == 2
        assert {editor.base_revision for editor in editors} == {1}
        assert board._numeric_bindings["generic-typed"].interaction_ready
        assert _wheel(board, -120).isAccepted()
        assert not window.raster_ready
        _until(application, lambda: window.worker_idle and window.raster_ready)
        assert board.histogram_selector_fault is None
        assert len(render_threads) == 3
        assert set(render_threads) == {render_threads[0]}
        assert window._display.revision == 2
        board, export_frame, export_payload = _typed_front(window)
        assert all(
            actual is expected
            for actual, expected in zip(
                export_payload.series,
                evaluated.layers[0].cells[0].series,
                strict=True,
            )
        )
        assert {editor.base_revision for editor in editors} == {2}

        destination = tmp_path / "typed-histogram.png"
        raster = export_frame.panels[0].raster
        window._start_export(destination)
        _until(application, lambda: window.worker_idle and destination.exists())
        from PIL import Image

        with Image.open(destination) as image:
            assert image.mode == "RGBA"
            assert image.size == (raster.width, raster.height)
            assert image.tobytes() == raster.pixels
        assert figure.evaluated is evaluated
    finally:
        release_rerender.set()
        _close(application, window)


def test_failed_histogram_rerender_preserves_exact_old_front(
    application,
    monkeypatch,
) -> None:
    figure = _histogram_figure()

    from zlc_frontend.matplotlib_render import SinglePanelAggRenderer

    original_render = SinglePanelAggRenderer.render_interactive_histogram
    calls = 0

    def fail_second(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected histogram rerender failure")
        return original_render(self, *args, **kwargs)

    monkeypatch.setattr(
        SinglePanelAggRenderer,
        "render_interactive_histogram",
        fail_second,
    )
    window = open_data_figure_workbench(figure)
    try:
        _until(application, lambda: window.worker_idle and window.raster_ready)
        board, old_frame, old_payload = _typed_front(window)
        values = histogram_display_form_values(window._display)
        values["bin_count"] = 19
        window._apply_display_form(window._setting_display, 0, values)
        _until(application, lambda: window.worker_idle)
        assert board.front_frame is old_frame
        assert board.visible_histogram_payload("generic-typed") is old_payload
        assert window._display.revision == 0
        assert window.raster_ready
        assert window._interaction_switch.isEnabled()
        assert "injected histogram rerender failure" in window.findChild(
            QtWidgets.QLabel,
            "figureViewerDiagnostic",
        ).text()
    finally:
        _close(application, window)


def test_histogram_budget_rejects_before_agg_or_qt_front(
    application,
    monkeypatch,
) -> None:
    figure = _histogram_figure()

    from zlc_frontend.matplotlib_render import SinglePanelAggRenderer

    monkeypatch.setattr(
        SinglePanelAggRenderer,
        "__init__",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("rejected histogram reached Agg construction")
        ),
    )
    window = open_data_figure_workbench(figure, memory_limit_bytes=1)
    try:
        _until(application, lambda: window.worker_idle)
        board = window.findChild(QtRasterBoard, "figureViewerTypedBoard")
        assert board is not None and board.front_frame is None
        assert not window.raster_ready
        assert window.findChild(QtWidgets.QLabel, "figureViewerStatus").text() == (
            "FIGURE FAILED"
        )
        assert "MemoryError" in window.findChild(
            QtWidgets.QLabel,
            "figureViewerDiagnostic",
        ).text()
        assert tuple(
            window._tabs.tabText(index) for index in range(window._tabs.count())
        ) == ("Loading",)
        assert window._interaction_switch.isHidden()
        assert window._settings_button.isHidden()
        assert window._export_button.isHidden()
    finally:
        _close(application, window)


def _assert_encoded_fallback(application, window) -> None:
    _until(application, lambda: window.worker_idle and window.raster_ready)
    assert window._view_family == "encoded"
    assert len(window._boards) == 1
    assert isinstance(window._boards[0], QtImageBoard)
    assert window._boards[0].has_front
    assert window.findChild(QtWidgets.QLabel, "figureViewerMode").text() == (
        "FROZEN DATA FIGURE · DISPLAY ONLY"
    )
    assert tuple(
        window._tabs.tabText(index) for index in range(window._tabs.count())
    ) == ("Figure",)
    assert window._interaction_switch.isHidden()
    assert window._settings_button.isHidden()
    assert window._export_button.isHidden()


def test_multi_cell_histogram_uses_encoded_fallback(application) -> None:
    figure = _histogram_figure(site_role=AxisViewRole.FACET)
    window = open_data_figure_workbench(
        figure,
        memory_limit_bytes=64 << 20,
    )
    try:
        _assert_encoded_fallback(application, window)
    finally:
        _close(application, window)


def test_histogram_typed_budget_has_exact_derived_boundary(application) -> None:
    import Zou_lab_control.workbench._figure as figure_module
    from zlc_frontend.matplotlib_render import evaluated_figure_array_nbytes

    figure = _histogram_figure()
    assert evaluated_figure_array_nbytes(figure.evaluated) == 88
    render_required = figure_module._typed_front_required_peak_bytes(
        figure,
        HistogramDisplayState(),
    )
    required = _initial_session_peak(figure, HistogramDisplayState())
    assert required > render_required

    rejected = open_data_figure_workbench(
        figure,
        memory_limit_bytes=required - 1,
    )
    try:
        _assert_encoded_fallback(application, rejected)
    finally:
        _close(application, rejected)

    admitted = open_data_figure_workbench(
        figure,
        memory_limit_bytes=required,
    )
    try:
        _until(application, lambda: admitted.worker_idle and admitted.raster_ready)
        assert admitted._view_family == "histogram"
        _typed_front(admitted)
    finally:
        _close(application, admitted)


def test_frozen_data_figure_budget_cannot_be_widened_by_window(application) -> None:
    import Zou_lab_control.workbench._figure as figure_module

    probe = _histogram_figure()
    render_required = figure_module._typed_front_required_peak_bytes(
        probe,
        HistogramDisplayState(),
    )
    aggregate_required = _initial_session_peak(probe, HistogramDisplayState())
    figure = _histogram_figure(render_memory_limit_bytes=render_required - 1)
    assert figure_module._typed_front_required_peak_bytes(
        figure,
        HistogramDisplayState(),
    ) == render_required
    window = open_data_figure_workbench(
        figure,
        memory_limit_bytes=aggregate_required + (8 << 20),
    )
    try:
        _assert_encoded_fallback(application, window)
    finally:
        _close(application, window)


@pytest.mark.parametrize("limit_owner", ("window", "data-figure"))
def test_larger_histogram_rerender_is_rejected_before_agg_and_keeps_old_front(
    application,
    monkeypatch,
    limit_owner,
) -> None:
    import Zou_lab_control.workbench._figure as figure_module
    from zlc_frontend.matplotlib_render import SinglePanelAggRenderer

    probe = _histogram_figure()
    initial_state = HistogramDisplayState()
    render_required = figure_module._typed_front_required_peak_bytes(
        probe,
        initial_state,
    )
    aggregate_required = _initial_session_peak(probe, initial_state)
    larger_state = replace(initial_state, revision=1, bin_count=500)
    assert figure_module._typed_front_required_peak_bytes(
        probe,
        larger_state,
    ) > render_required
    figure = _histogram_figure(
        render_memory_limit_bytes=(
            render_required if limit_owner == "data-figure" else 64 << 20
        )
    )
    window = open_data_figure_workbench(
        figure,
        memory_limit_bytes=(
            aggregate_required + (8 << 20)
            if limit_owner == "data-figure"
            else aggregate_required
        ),
    )
    try:
        _until(application, lambda: window.worker_idle and window.raster_ready)
        board, old_frame, _old_payload = _typed_front(window)
        monkeypatch.setattr(
            SinglePanelAggRenderer,
            "__init__",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("over-budget rerender reached Agg construction")
            ),
        )
        values = histogram_display_form_values(window._display)
        values["bin_count"] = 500
        window._apply_display_form(window._edit_display, 0, values)
        _until(application, lambda: window.worker_idle)
        assert board.front_frame is old_frame
        assert window._display.revision == 0
        assert window.raster_ready
        diagnostic = window.findChild(
            QtWidgets.QLabel,
            "figureViewerDiagnostic",
        ).text()
        assert "MemoryError" in diagnostic
        assert "interactive histogram requires" in diagnostic
    finally:
        _close(application, window)


def test_closed_histogram_window_releases_frozen_evaluated_arrays(application) -> None:
    figure = _histogram_figure()
    samples = figure.evaluated.layers[0].cells[0].series[0].data.samples
    samples_ref = weakref.ref(samples)
    window = open_data_figure_workbench(figure)
    _until(application, lambda: window.worker_idle and window.raster_ready)
    del samples
    del figure
    _close(application, window)
    gc.collect()
    assert samples_ref() is None


def test_histogram_front_request_and_authored_state_are_compare_and_swap(
    application,
) -> None:
    window = open_data_figure_workbench(_histogram_figure())
    try:
        _until(application, lambda: window.worker_idle and window.raster_ready)
        board, old_frame, _old_payload = _typed_front(window)
        original_renderer = window._typed_renderer

        def wrong_sequence(*args, **kwargs):
            front = original_renderer(*args, **kwargs)
            return replace(
                front,
                frame=replace(front.frame, sequence=front.frame.sequence + 1),
            )

        window._typed_renderer = wrong_sequence
        values = histogram_display_form_values(window._display)
        values["bin_count"] = 23
        window._apply_display_form(window._edit_display, 0, values)
        _until(application, lambda: window.worker_idle)
        assert board.front_frame is old_frame
        assert window._display.revision == 0
        assert window.raster_ready
        assert "another request sequence" in window.findChild(
            QtWidgets.QLabel,
            "figureViewerDiagnostic",
        ).text()

        def wrong_authored_state(*args, **kwargs):
            front = original_renderer(*args, **kwargs)
            panel = front.frame.panels[0]
            payload = panel.display_payload
            assert isinstance(payload, HistogramPanelPayload)
            wrong_payload = replace(
                payload,
                viewport=replace(payload.viewport, relim_mode=RelimMode.NORMAL),
            )
            return replace(
                front,
                frame=replace(
                    front.frame,
                    panels=(replace(panel, display_payload=wrong_payload),),
                ),
            )

        window._typed_renderer = wrong_authored_state
        window._apply_display_form(window._setting_display, 0, values)
        _until(application, lambda: window.worker_idle)
        assert board.front_frame is old_frame
        assert window._display.revision == 0
        assert window.raster_ready
        assert "conflicting authored state" in window.findChild(
            QtWidgets.QLabel,
            "figureViewerDiagnostic",
        ).text()

        def wrong_provenance(*args, **kwargs):
            front = original_renderer(*args, **kwargs)
            panel = front.frame.panels[0]
            return replace(
                front,
                frame=replace(
                    front.frame,
                    panels=(
                        replace(
                            panel,
                            coherence_stamp=replace(
                                panel.coherence_stamp,
                                join_key_digest="0" * 64,
                            ),
                        ),
                    ),
                ),
            )

        window._typed_renderer = wrong_provenance
        window._apply_display_form(window._edit_display, 0, values)
        _until(application, lambda: window.worker_idle)
        assert board.front_frame is old_frame
        assert window._display.revision == 0
        assert window.raster_ready
        assert "changed frozen source provenance" in window.findChild(
            QtWidgets.QLabel,
            "figureViewerDiagnostic",
        ).text()

        def wrong_exact_series(*args, **kwargs):
            front = original_renderer(*args, **kwargs)
            panel = front.frame.panels[0]
            payload = panel.display_payload
            assert isinstance(payload, HistogramPanelPayload)
            substituted = (replace(payload.series[0]), *payload.series[1:])
            return replace(
                front,
                frame=replace(
                    front.frame,
                    panels=(
                        replace(
                            panel,
                            display_payload=replace(
                                payload,
                                series=substituted,
                            ),
                        ),
                    ),
                ),
            )

        window._typed_renderer = wrong_exact_series
        window._apply_display_form(window._setting_display, 0, values)
        _until(application, lambda: window.worker_idle)
        assert board.front_frame is old_frame
        assert window._display.revision == 0
        assert window.raster_ready
        assert "changed frozen evaluated data" in window.findChild(
            QtWidgets.QLabel,
            "figureViewerDiagnostic",
        ).text()

        def wrong_raster_geometry(*args, **kwargs):
            front = original_renderer(*args, **kwargs)
            panel = front.frame.panels[0]
            raster = panel.raster
            stride = raster.stride_bytes + 4
            wrong_raster = replace(
                raster,
                width=raster.width + 1,
                stride_bytes=stride,
                pixels=raster.pixels + b"\0" * (4 * raster.height),
            )
            return replace(
                front,
                frame=replace(
                    front.frame,
                    panels=(replace(panel, raster=wrong_raster),),
                ),
            )

        window._typed_renderer = wrong_raster_geometry
        window._apply_display_form(window._edit_display, 0, values)
        _until(application, lambda: window.worker_idle)
        assert board.front_frame is old_frame
        assert window._display.revision == 0
        assert window.raster_ready
        assert "another raster geometry" in window.findChild(
            QtWidgets.QLabel,
            "figureViewerDiagnostic",
        ).text()
    finally:
        _close(application, window)


def test_close_during_histogram_rerender_is_nonblocking_and_never_late_presents(
    application,
    monkeypatch,
) -> None:
    from zlc_frontend.matplotlib_render import SinglePanelAggRenderer

    entered = threading.Event()
    release = threading.Event()
    calls = 0
    original_render = SinglePanelAggRenderer.render_interactive_histogram

    def blocked_second(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            entered.set()
            if not release.wait(10.0):
                raise TimeoutError("test did not release histogram rerender")
        return original_render(self, *args, **kwargs)

    monkeypatch.setattr(
        SinglePanelAggRenderer,
        "render_interactive_histogram",
        blocked_second,
    )
    figure = _histogram_figure()
    samples = figure.evaluated.layers[0].cells[0].series[0].data.samples
    samples_ref = weakref.ref(samples)
    window = open_data_figure_workbench(figure)
    try:
        _until(application, lambda: window.worker_idle and window.raster_ready)
        board, _frame, _payload = _typed_front(window)
        del _frame, _payload
        values = histogram_display_form_values(window._display)
        values["bin_count"] = 17
        window._apply_display_form(window._edit_display, 0, values)
        _until(application, entered.is_set)
        del samples
        del figure
        began = time.monotonic()
        window.close()
        assert time.monotonic() - began < 0.1
        assert not window.closed
        assert board.front_frame is None
        application.processEvents()
        assert board.front_frame is None
        release.set()
        _until(
            application,
            lambda: window.closed and not window.isVisible(),
        )
        assert window.worker_idle
        assert board.front_frame is None
        gc.collect()
        assert samples_ref() is None
    finally:
        release.set()
        if not window.closed:
            _close(application, window)


def test_close_during_histogram_export_preserves_existing_target_atomically(
    application,
    monkeypatch,
    tmp_path: Path,
) -> None:
    from PIL import Image

    window = open_data_figure_workbench(_histogram_figure())
    _until(application, lambda: window.worker_idle and window.raster_ready)
    destination = tmp_path / "typed-histogram.png"
    original_bytes = b"pre-existing-authoritative-destination"
    destination.write_bytes(original_bytes)
    entered = threading.Event()
    release = threading.Event()
    original_save = Image.Image.save

    def blocked_save(self, *args, **kwargs):
        result = original_save(self, *args, **kwargs)
        entered.set()
        if not release.wait(10.0):
            raise TimeoutError("test did not release staged histogram export")
        return result

    monkeypatch.setattr(Image.Image, "save", blocked_save)
    try:
        window._start_export(destination)
        _until(application, entered.is_set)
        assert tuple(tmp_path.glob(f".{destination.name}.*"))
        began = time.monotonic()
        window.close()
        assert time.monotonic() - began < 0.1
        assert not window.closed
        assert destination.read_bytes() == original_bytes
        release.set()
        _until(
            application,
            lambda: window.closed and not window.isVisible(),
        )
        assert destination.read_bytes() == original_bytes
        assert not tuple(tmp_path.glob(f".{destination.name}.*"))
    finally:
        release.set()
        if not window.closed:
            _close(application, window)
