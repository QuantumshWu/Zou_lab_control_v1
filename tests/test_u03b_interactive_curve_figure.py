"""Product contracts for the typed generic CURVE DataFigure window."""

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
    FitNumericPolicy,
    OwnedSnapshot,
    PointLayout,
    StreamGenerationId,
    ValidityContract,
    ValueSchema,
    bind_fit,
    fit_spec_for,
)
from zlc_frontend import CurvePanelPayload, DataFigure  # noqa: E402
from zlc_frontend.curve_display import (  # noqa: E402
    CurveDisplayState,
    curve_display_form_values,
)
from zlc_frontend.display_range import RelimMode  # noqa: E402
from zlc_frontend.figure import (  # noqa: E402
    AxisViewBinding,
    AxisViewRole,
    DatasetDescriptor,
    DatasetId,
    DisplayReduction,
    DisplayReductionMethod,
    FigureDocument,
    FigureLayer,
    ResolvedDataset,
    ResolvedDatasetMap,
    ViewIntent,
    ViewSpec,
    suggest_fit_view,
)
from zlc_frontend.qt_widgets import (  # noqa: E402
    FrozenRasterView,
    QtRasterBoard,
    ensure_qt_app,
)
from zlc_frontend.selector import CurveRangeGesture  # noqa: E402
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


def _curve_figure(
    *,
    coordinates: tuple[object, ...] | None = None,
    site_role: AxisViewRole = AxisViewRole.BATCH,
    with_fit: bool = False,
) -> DataFigure:
    if coordinates is None:
        coordinates = tuple(float(value) for value in np.linspace(-2.0, 2.0, 21))
    repeat = AxisSpec(AxisId("u03b.repeat"), "Repeat", REPEAT, 2, (0, 1))
    scan = AxisSpec(
        AxisId("u03b.detuning"),
        "Detuning",
        SCAN_POINT,
        len(coordinates),
        coordinates,
        "MHz",
    )
    site = AxisSpec(
        AxisId("u03b.site"),
        "Site",
        SITE,
        3,
        ("left", "middle", "right"),
    )
    component = AxisSpec(
        AxisId("u03b.component"),
        "Component",
        COMPONENT,
        2,
        ("signal", "reference"),
    )
    numeric_x = np.arange(len(coordinates), dtype=np.float64)
    site_values = np.stack(
        (
            1.0 + 4.0 * np.exp(-((numeric_x - 9.0) ** 2) / 18.0),
            2.0 + 3.0 * np.exp(-((numeric_x - 11.0) ** 2) / 14.0),
            0.5 + 2.0 * np.exp(-((numeric_x - 7.0) ** 2) / 12.0),
        ),
        axis=-1,
    )
    values = np.stack(
        (site_values, 0.5 * site_values + 0.3),
        axis=-1,
    )
    values = np.stack((values, values + 0.2), axis=0)
    valid = np.ones(values.shape, dtype=np.bool_)
    valid[0, 4, 1, 0] = False
    valid[1, 4, 1, 0] = False
    schema = DatasetSchema(
        repeat,
        (scan,),
        PointLayout.rect_c((scan.size,)),
        ValueSchema(
            (site, component),
            ValidityContract.components(site.axis_id, component.axis_id),
            values.dtype,
            value_unit="photoelectron",
        ),
    )
    block = DataBlock(
        BlockId("u03b-curve-block"),
        DatasetRevision(5),
        values,
        ComponentValidity((site.axis_id, component.axis_id), valid),
        schema,
    )
    snapshot = OwnedSnapshot(
        block.ref(StreamGenerationId("u03b-curve-generation")),
        block,
    )
    dataset_id = DatasetId("u03b-curve-dataset")
    fit_results = None
    if with_fit:
        result = bind_fit(
            fit_spec_for(
                schema,
                "gaussian_offset",
                fit_axis_ids=(scan.axis_id,),
                numeric_policy=FitNumericPolicy(max_evaluations=300),
            ),
            schema,
        ).run(snapshot)
        suggestion = suggest_fit_view(schema, result)
        assert suggestion.spec is not None
        view = suggestion.spec
        fit_results = {"curve": result}
    else:
        view = ViewSpec(
            schema.fingerprint,
            ViewIntent.CURVE,
            (
                AxisViewBinding(
                    repeat.axis_id,
                    AxisViewRole.REDUCED,
                    reduction=DisplayReduction(DisplayReductionMethod.MEAN),
                ),
                AxisViewBinding(scan.axis_id, AxisViewRole.X),
                AxisViewBinding(site.axis_id, site_role),
                AxisViewBinding(component.axis_id, AxisViewRole.BATCH),
            ),
        )
    document = FigureDocument(
        "u03b-curve-document",
        3,
        (DatasetDescriptor(dataset_id, "site curves", schema.fingerprint),),
        (FigureLayer("curve", dataset_id, view),),
    )
    return DataFigure(
        document,
        ResolvedDatasetMap((ResolvedDataset(dataset_id, snapshot),)),
        fit_results=fit_results,
    )


def _typed_front(window):
    board = window.findChild(QtRasterBoard, "figureViewerTypedBoard")
    assert board is not None and board.front_frame is not None
    frame = board.front_frame
    payload = frame.panels[0].display_payload
    assert isinstance(payload, CurvePanelPayload)
    return board, frame, payload


def _wheel(board: QtRasterBoard, delta: int):
    binding = board._numeric_binding_for_kind(
        "curve",
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


def test_curve_front_preserves_all_series_axes_validity_and_interacts(
    application,
    tmp_path: Path,
) -> None:
    figure = _curve_figure()
    expected_series = figure.evaluated.layers[0].cells[0].series
    window = open_data_figure_workbench(figure)
    try:
        _until(application, lambda: window.raster_ready)
        board, first_frame, first_payload = _typed_front(window)
        assert window._view_family == "curve"
        assert isinstance(window._display, CurveDisplayState)
        assert window.findChild(QtWidgets.QLabel, "figureViewerMode").text() == (
            "EXACT CURVE · INTERACTIVE · DISPLAY ONLY"
        )
        assert tuple(
            window._tabs.tabText(index) for index in range(window._tabs.count())
        ) == ("Curve", "Edit")
        assert len(first_payload.series) == 6
        assert all(
            actual is expected
            for actual, expected in zip(
                first_payload.series,
                expected_series,
                strict=True,
            )
        )
        assert tuple(item.batch_address for item in first_payload.series) == tuple(
            item.batch_address for item in expected_series
        )
        assert tuple(item.reductions for item in first_payload.series) == tuple(
            item.reductions for item in expected_series
        )
        assert any(
            not bool(series.data.validity[4])
            for series in first_payload.series
        )
        assert first_payload.viewport.x_axis.unit == "MHz"
        assert first_payload.value_unit == "photoelectron"

        origin = board.visible_curve_origin("generic-typed")
        assert origin is not None
        window._accept_numeric_interaction(
            CurveRangeGesture(origin, (-0.75, 0.5))
        )
        assert board._numeric_bindings["generic-typed"].applied_span == (
            -0.75,
            0.5,
        )

        wheel = _wheel(board, 120)
        assert wheel.isAccepted()
        _until(
            application,
            lambda: window.raster_ready and window._display.revision == 1,
        )
        assert window._display.x_view is not None
        _board, second_frame, _payload = _typed_front(window)
        assert second_frame is not first_frame

        values = curve_display_form_values(window._display)
        values["relim_mode"] = RelimMode.FIXED
        window._apply_display_form(
            window._setting_display,
            window._display.revision,
            values,
        )
        _until(
            application,
            lambda: window.raster_ready and window._display.revision == 2,
        )
        assert window._display.relim_mode is RelimMode.FIXED
        assert window._display.fixed_y_limits is not None

        destination = tmp_path / "typed-curve.png"
        window._start_export(destination)
        _until(application, lambda: window.worker_idle)
        assert destination.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    finally:
        _close(application, window)


@pytest.mark.parametrize(
    "figure, reason_fragment",
    (
        pytest.param(
            _curve_figure(coordinates=tuple(f"p{index}" for index in range(21))),
            "numeric scalar",
            id="categorical-x",
        ),
        pytest.param(
            _curve_figure(
                coordinates=(
                    *tuple(float(value) for value in range(10)),
                    8.5,
                    *tuple(float(value) for value in range(11, 21)),
                )
            ),
            "strictly monotonic",
            id="nonmonotonic-x",
        ),
        pytest.param(
            _curve_figure(with_fit=True),
            "exact caller-supplied result identity",
            id="fit-overlay",
        ),
    ),
)
def test_unsupported_or_authoritative_curve_content_stays_encoded(
    application,
    figure: DataFigure,
    reason_fragment: str,
) -> None:
    window = open_data_figure_workbench(figure)
    try:
        _until(application, lambda: window.raster_ready)
        assert window._view_family == "encoded"
        assert window.findChild(FrozenRasterView, "figureViewerBoard") is not None
        assert window.findChild(QtRasterBoard, "figureViewerTypedBoard").front_frame is None
        assert "interaction unavailable:" in window._summary.text()
        assert reason_fragment in window._summary.text()
    finally:
        _close(application, window)


def test_curve_front_sequence_and_authored_state_are_compare_and_swap(
    application,
) -> None:
    window = open_data_figure_workbench(_curve_figure())
    try:
        _until(application, lambda: window.raster_ready)
        board, old_frame, _old_payload = _typed_front(window)
        original_renderer = window._typed_renderer
        values = curve_display_form_values(window._display)
        values["x_min"] = -1.0
        values["x_max"] = 1.0

        def wrong_sequence(*args, **kwargs):
            front = original_renderer(*args, **kwargs)
            return replace(
                front,
                frame=replace(front.frame, sequence=front.frame.sequence + 1),
            )

        window._typed_renderer = wrong_sequence
        window._apply_display_form(window._edit_display, 0, values)
        _until(application, lambda: window.worker_idle)
        assert board.front_frame is old_frame
        assert window._display.revision == 0
        assert window.raster_ready
        assert "another request sequence" in window._diagnostic.text()

        def wrong_authored_state(*args, **kwargs):
            front = original_renderer(*args, **kwargs)
            return replace(front, state=CurveDisplayState())

        window._typed_renderer = wrong_authored_state
        window._apply_display_form(window._setting_display, 0, values)
        _until(application, lambda: window.worker_idle)
        assert board.front_frame is old_frame
        assert window._display.revision == 0
        assert window.raster_ready
        assert "conflicting authored state" in window._diagnostic.text()

        def wrong_home_range(*args, **kwargs):
            front = original_renderer(*args, **kwargs)
            panel = front.frame.panels[0]
            payload = panel.display_payload
            assert isinstance(payload, CurvePanelPayload)
            low, high = payload.viewport.home_x_limits
            wrong_payload = replace(
                payload,
                viewport=replace(
                    payload.viewport,
                    home_x_limits=(low - 1.0, high + 1.0),
                ),
            )
            return replace(
                front,
                frame=replace(
                    front.frame,
                    panels=(replace(panel, display_payload=wrong_payload),),
                ),
            )

        window._typed_renderer = wrong_home_range
        window._apply_display_form(window._edit_display, 0, values)
        _until(application, lambda: window.worker_idle)
        assert board.front_frame is old_frame
        assert window._display.revision == 0
        assert window.raster_ready
        assert "conflicting authored state" in window._diagnostic.text()
    finally:
        _close(application, window)


def test_initial_curve_state_is_checked_against_gui_owned_default(
    application,
    monkeypatch,
) -> None:
    from Zou_lab_control.workbench import _figure as figure_module

    original_render = figure_module._render_typed_front

    def forged_initial_state(*args, **kwargs):
        front = original_render(*args, **kwargs)
        return replace(
            front,
            state=CurveDisplayState(revision=1, x_view=(-1.0, 1.0)),
        )

    monkeypatch.setattr(
        figure_module,
        "_render_typed_front",
        forged_initial_state,
    )
    window = open_data_figure_workbench(_curve_figure())
    try:
        _until(application, lambda: window.worker_idle)
        board = window.findChild(QtRasterBoard, "figureViewerTypedBoard")
        assert board.front_frame is None
        assert window._view_family is None
        assert not window.raster_ready
        assert "conflicting authored state" in window._diagnostic.text()
    finally:
        _close(application, window)


def test_control_construction_fault_keeps_the_admitted_curve_visible(
    application,
    monkeypatch,
) -> None:
    from Zou_lab_control.workbench import _figure as figure_module

    def rejected_editor(*_args, **_kwargs):
        raise RuntimeError("injected numeric editor construction fault")

    monkeypatch.setattr(
        figure_module,
        "FluentRevisionedFormEditor",
        rejected_editor,
    )
    window = open_data_figure_workbench(_curve_figure())
    try:
        _until(application, lambda: window.worker_idle)
        board, _frame, payload = _typed_front(window)
        assert window.raster_ready
        assert board.front_frame.panels[0].display_payload is payload
        assert window._display == CurveDisplayState()
        assert tuple(
            window._tabs.tabText(index) for index in range(window._tabs.count())
        ) == ("Curve",)
        assert window._edit_display is None
        assert not window._settings_button.isEnabled()
        assert not window._export_button.isEnabled()
        assert window._status.text() == "TYPED CONTROLS FAILED"
        assert "injected numeric editor construction fault" in window._diagnostic.text()
    finally:
        _close(application, window)


def test_close_during_curve_rerender_never_late_presents_and_releases_arrays(
    application,
    monkeypatch,
) -> None:
    from zlc_frontend.matplotlib_render import SinglePanelAggRenderer

    entered = threading.Event()
    release = threading.Event()
    calls = 0
    original_render = SinglePanelAggRenderer.render_interactive_curve

    def blocked_second(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            entered.set()
            if not release.wait(10.0):
                raise TimeoutError("test did not release curve rerender")
        return original_render(self, *args, **kwargs)

    monkeypatch.setattr(
        SinglePanelAggRenderer,
        "render_interactive_curve",
        blocked_second,
    )
    figure = _curve_figure()
    values_array = figure.evaluated.layers[0].cells[0].series[0].data.values
    values_ref = weakref.ref(values_array)
    window = open_data_figure_workbench(figure)
    try:
        _until(application, lambda: window.raster_ready)
        board, _frame, _payload = _typed_front(window)
        del _frame, _payload
        values = curve_display_form_values(window._display)
        values["x_min"] = -1.0
        values["x_max"] = 1.0
        window._apply_display_form(window._edit_display, 0, values)
        _until(application, entered.is_set)
        del values_array
        del figure
        began = time.monotonic()
        window.close()
        assert time.monotonic() - began < 0.1
        assert not window.closed
        assert board.front_frame is None
        application.processEvents()
        assert board.front_frame is None
        release.set()
        _until(application, lambda: window.closed and not window.isVisible())
        assert window.worker_idle
        assert board.front_frame is None
        gc.collect()
        assert values_ref() is None
    finally:
        release.set()
        if not window.closed:
            _close(application, window)
