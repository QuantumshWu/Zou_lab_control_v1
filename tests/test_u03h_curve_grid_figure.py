"""U0.3h exact faceted CURVE Grid product contracts."""

from __future__ import annotations

from dataclasses import replace
import inspect
import os
from pathlib import Path
import threading
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PIL import Image
from PyQt5 import QtCore, QtGui, QtTest, QtWidgets

from zlc_frontend.qt_widgets import ensure_qt_app
import pytest

import Zou_lab_control.notebook as zlc
from zlc_data import (
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
from zlc_neutral_atom.readout.sitemap import load_packaged_sitemap_pulse
from zlc_neutral_atom.scan import AutonomousScanExecution
from zlc_pulse import FrozenScanTable, RepeatRegion, ScanParameter
from zlc_frontend import CurvePanelPayload, DataFigure
from zlc_frontend.curve_display import (
    CurveDisplayState,
    curve_display_form_values,
    curve_home_x_limits,
)
from zlc_frontend.display_range import RelimMode, target_display_range
from zlc_frontend.figure import (
    AxisViewRole,
    DatasetDescriptor,
    DatasetId,
    FigureDocument,
    FigureLayer,
    ResolvedDataset,
    ResolvedDatasetMap,
    SuggestionStatus,
    ViewIntent,
    ViewPreferences,
    suggest_view,
)
from zlc_frontend.matplotlib_render import (
    SinglePanelAggRenderer,
    release_agg_figure,
)
from zlc_frontend.selector import CurveRangeGesture
from Zou_lab_control.workbench import _figure as figure_workbench


@pytest.fixture(scope="module")
def application():
    return ensure_qt_app()


def _axis(name: str, role, size: int, coordinates, unit=None) -> AxisSpec:
    return AxisSpec(AxisId(name), name, role, size, tuple(coordinates), unit)


def _occupancy_scan_document():
    document = load_packaged_sitemap_pulse()
    camera_port = next(port for port in document.target.ports if port.label == "emCCD")
    trigger_index = document.target.raw_lanes.index(camera_port.lanes[0])
    segment = -1
    previous = 0
    periods = []
    for period in document.periods:
        states = list(period.states)
        current = int(states[trigger_index])
        if current and not previous:
            segment += 1
        states[trigger_index] = int(bool(current and segment == 1))
        periods.append(replace(period, states=tuple(states)))
        previous = current
    scanned_api = document.api_parameters[0]
    scanned_period = next(
        period for period in periods if period.period_id == scanned_api.field.period_id
    )
    parameter = ScanParameter(
        "reference_settle",
        scanned_api.field,
        "reference settle",
        scanned_api.unit,
    )
    start = scanned_period.duration
    step = 1 if isinstance(start, int) else 1e-6
    return replace(
        document,
        name="u03h-occupancy-scan",
        periods=tuple(periods),
        api_parameters=tuple(
            item for item in document.api_parameters if item is not scanned_api
        ),
        scan_parameters=(parameter,),
        scan_table=FrozenScanTable(
            (parameter.parameter_id,),
            ((start,), (start + step,)),
        ),
        repeat=RepeatRegion(periods[0].period_id, periods[-1].period_id, 2),
    )


def _fixed_api_values(document):
    return {
        parameter.parameter_id: document.field_value(parameter.field)[0]
        for parameter in document.api_parameters
    }


def _curve_grid(
    *,
    layers: int = 1,
    revision: int = 11,
    all_invalid: bool = False,
    invalid_endpoints: bool = False,
    scan_coordinates=(-2.0, -1.0, 0.0, 1.0, 2.0),
) -> DataFigure:
    repeat = _axis("u03h.repeat", REPEAT, 2, (0, 1))
    scan = _axis(
        "u03h.detuning",
        SCAN_POINT,
        5,
        scan_coordinates,
        "MHz",
    )
    site = _axis("u03h.site", SITE, 3, ("left", "middle", "right"))
    component = _axis(
        "u03h.component",
        COMPONENT,
        2,
        ("signal", "reference"),
    )
    per_site = np.asarray(
        (
            (0.0, 100.0, -10.0),
            (1.0, 106.0, -8.0),
            (2.0, 112.0, -6.0),
            (3.0, 118.0, -4.0),
            (4.0, 124.0, -2.0),
        ),
        dtype=np.float64,
    )
    components = np.stack((per_site, 0.5 * per_site + 0.25), axis=-1)
    values = np.stack((components, components + 0.5), axis=0)
    valid = np.zeros(values.shape, dtype=bool) if all_invalid else np.ones(
        values.shape,
        dtype=bool,
    )
    if not all_invalid:
        valid[:, 2, 1, 0] = False
        if invalid_endpoints:
            valid[:, (0, -1), 0, :] = False
    schema = DatasetSchema(
        repeat,
        (scan,),
        PointLayout.rect_c((scan.size,)),
        ValueSchema(
            (site, component),
            ValidityContract.components(site.axis_id, component.axis_id),
            values.dtype,
            "photoelectron",
        ),
    )
    block = DataBlock(
        BlockId("u03h-curve-block"),
        DatasetRevision(revision),
        values,
        ComponentValidity((site.axis_id, component.axis_id), valid),
        schema,
    )
    dataset_id = DatasetId("u03h-curve-dataset")
    suggestion = suggest_view(
        schema,
        ViewIntent.CURVE,
        preferences=ViewPreferences(facet_axis_ids=(site.axis_id,)),
    )
    assert suggestion.status is SuggestionStatus.RESOLVED
    assert suggestion.spec.binding(site.axis_id).role is AxisViewRole.FACET
    snapshot = OwnedSnapshot(
        block.ref(StreamGenerationId("u03h-curve-generation")),
        block,
    )
    return DataFigure(
        FigureDocument(
            "u03h-curve-grid",
            3,
            (DatasetDescriptor(dataset_id, "Occupancy scan", schema.fingerprint),),
            tuple(
                FigureLayer(f"u03h-layer-{index}", dataset_id, suggestion.spec)
                for index in range(layers)
            ),
        ),
        ResolvedDatasetMap((ResolvedDataset(dataset_id, snapshot),)),
        render_memory_limit_bytes=128 << 20,
    )


def _until(application, predicate, *, timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        application.processEvents(QtCore.QEventLoop.AllEvents, 20)
        time.sleep(0.005)
    assert predicate()


def _close(application, window) -> None:
    window.shutdown()
    _until(application, lambda: window.closed)


def _center(region) -> tuple[float, float]:
    return (
        (region.left + region.right) / 2.0,
        (region.top + region.bottom) / 2.0,
    )


def _blank_point(regions):
    for y in np.linspace(0.01, 0.99, 40):
        for x in np.linspace(0.01, 0.99, 40):
            if not any(region.contains(float(x), float(y)) for region in regions):
                return float(x), float(y)
    raise AssertionError("curve overview unexpectedly has no blank margin")


def _curve_payload(window) -> CurvePanelPayload:
    payload = window._board_widget.visible_curve_payload("generic-typed")
    assert isinstance(payload, CurvePanelPayload)
    return payload


def _wheel_curve(board, delta: int):
    binding = board._numeric_binding_for_kind("curve", panel_id="generic-typed")
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


def _valid_value_limits(figure: DataFigure) -> tuple[float, float]:
    selected = []
    for cell in figure.evaluated.layers[0].cells:
        for series in cell.series:
            values = np.asarray(series.data.values)
            valid = np.asarray(series.data.validity, dtype=bool)
            selected.extend(float(value) for value in values[valid])
    if not selected:
        return (0.0, 1.0)
    return target_display_range(RelimMode.TIGHT, min(selected), max(selected))


def test_curve_grid_shares_overview_y_but_focus_keeps_local_relim() -> None:
    figure = _curve_grid(invalid_endpoints=True)
    expected_shared = _valid_value_limits(figure)
    x_axis = figure.evaluated.layers[0].cells[0].series[0].data.x_axis
    expected_x = curve_home_x_limits(x_axis)
    rendered = figure.render()
    try:
        limits = tuple(tuple(float(value) for value in axis.get_ylim()) for axis in rendered.axes)
        assert limits == (expected_shared,) * 3
        assert tuple(
            tuple(float(value) for value in axis.get_xlim())
            for axis in rendered.axes
        ) == (expected_x,) * 3
    finally:
        release_agg_figure(rendered)

    _png, regions = figure.to_png_bytes_with_panel_regions()
    expected_series = figure.evaluated.layers[0].cells[1].series
    focused = figure.focused_typed_panel(
        1,
        expected_selection=regions[1].selection,
        expected_intent=ViewIntent.CURVE,
    )
    assert focused.evaluated.inputs == figure.evaluated.inputs
    assert all(
        actual is expected
        for actual, expected in zip(
            focused.evaluated.layers[0].cells[0].series,
            expected_series,
            strict=True,
        )
    )
    binding = focused.document.layers[0].view.binding(AxisId("u03h.site"))
    assert binding.role is AxisViewRole.SELECTED
    assert binding.selector.index == 1
    assert (
        focused.retained_upper_bound_nbytes
        <= figure.focused_typed_panel_retained_upper_bound_nbytes(
            1,
            expected_intent=ViewIntent.CURVE,
        )
    )
    renderer = SinglePanelAggRenderer(focused.document, width=800, height=520)
    try:
        _raster, payload = renderer.render_interactive_curve(
            focused.evaluated,
            CurveDisplayState(),
            current_y_limits=None,
            previous_relim_mode=None,
        )
        local_values = []
        for series in expected_series:
            values = np.asarray(series.data.values)
            valid = np.asarray(series.data.validity, dtype=bool)
            local_values.extend(float(value) for value in values[valid])
        expected_local = target_display_range(
            RelimMode.TIGHT,
            min(local_values),
            max(local_values),
        )
        assert payload.viewport.y_limits == pytest.approx(expected_local)
        assert payload.viewport.y_limits != pytest.approx(expected_shared)
    finally:
        renderer.close()


def test_all_invalid_curve_grid_has_one_deterministic_overview_range() -> None:
    figure = _curve_grid(all_invalid=True)
    rendered = figure.render()
    try:
        assert tuple(axis.get_ylim() for axis in rendered.axes) == ((0.0, 1.0),) * 3
    finally:
        release_agg_figure(rendered)


def test_curve_grid_focus_interaction_back_and_atomic_exports(
    application,
    tmp_path: Path,
) -> None:
    figure = _curve_grid()
    expected = figure.evaluated.layers[0].cells[1].series
    window = figure_workbench.open_data_figure_workbench(figure)
    try:
        _until(application, lambda: window.raster_ready and window.worker_idle)
        overview = window._grid_overview
        assert overview is not None and overview.intent is ViewIntent.CURVE
        assert window._view_family == "curve-overview"
        assert len(overview.regions) == 3
        original_png = window._bundle.pages[0].png_bytes

        window._focus_grid_region(*_blank_point(overview.regions))
        assert window._view_family == "curve-overview" and window._future is None
        window._focus_grid_region(*_center(overview.regions[1]))
        _until(
            application,
            lambda: window.worker_idle and window._view_family == "curve",
        )
        payload = _curve_payload(window)
        assert payload.series == expected
        assert payload.evaluated_input.ref == figure.evaluated.inputs[0].ref
        assert payload.viewport.x_axis.unit == "MHz"
        assert payload.value_unit == "photoelectron"
        assert not bool(payload.series[0].data.validity[2])
        assert window._overview_button.isVisible()
        assert window._overview_button.isEnabled()
        assert tuple(
            window._tabs.tabText(index) for index in range(window._tabs.count())
        ) == ("Curve", "Edit")

        origin = window._board_widget.visible_curve_origin("generic-typed")
        assert origin is not None
        window._accept_numeric_interaction(CurveRangeGesture(origin, (-0.5, 0.75)))
        assert window._board_widget._numeric_bindings[
            "generic-typed"
        ].applied_span == (-0.5, 0.75)
        assert _wheel_curve(window._board_widget, 120).isAccepted()
        _until(application, lambda: window.worker_idle and window._display.revision == 1)
        assert window._display.x_view is not None
        values = curve_display_form_values(window._display)
        values["relim_mode"] = RelimMode.FIXED
        window._apply_display_form(
            window._edit_display,
            window._display.revision,
            values,
        )
        _until(application, lambda: window.worker_idle and window._display.revision == 2)
        assert window._display.fixed_y_limits is not None
        assert window._grid_focus_cache_charge_bytes > 0

        focused_frame = window._board_widget.front_frame
        focused_path = tmp_path / "curve-focus.png"
        window._start_export(focused_path)
        _until(application, lambda: window.worker_idle and focused_path.exists())
        with Image.open(focused_path) as image:
            rgba = image.convert("RGBA")
            assert rgba.size == (
                focused_frame.panels[0].raster.width,
                focused_frame.panels[0].raster.height,
            )
            assert rgba.tobytes() == focused_frame.panels[0].raster.pixels

        window._overview_button.click()
        application.processEvents()
        assert window._view_family == "curve-overview"
        assert window._bundle.pages[0].png_bytes is original_png
        assert (
            window._current_front_peak_bytes
            == window._grid_focus_cache_charge_bytes
            > 0
        )
        overview_path = tmp_path / "curve-overview.png"
        window._start_export(overview_path)
        _until(application, lambda: window.worker_idle and overview_path.exists())
        assert overview_path.read_bytes() == original_png

        window._focus_grid_region(*_center(overview.regions[2]))
        _until(application, lambda: window.worker_idle and window._view_family == "curve")
        QtTest.QTest.keyClick(window, QtCore.Qt.Key_Escape)
        application.processEvents()
        assert window._view_family == "curve-overview"
        assert window._bundle.pages[0].png_bytes is original_png
    finally:
        _close(application, window)


def test_curve_focus_budget_rejects_before_panel_derivation(
    application,
    monkeypatch,
) -> None:
    figure = _curve_grid()
    window = figure_workbench.open_data_figure_workbench(figure)
    try:
        _until(application, lambda: window.raster_ready and window.worker_idle)
        overview = window._grid_overview
        assert overview is not None
        external = (
            overview.external_retained_upper_bound_bytes
            + window._grid_overview_presentation_bytes
        )
        _focused, _render, aggregate = figure_workbench._typed_focus_preflight_nbytes(
            figure,
            0,
            expected_intent=ViewIntent.CURVE,
            display=CurveDisplayState(),
            external_session_retained_bytes=external,
        )
        calls = 0

        def forbidden(*_args, **_kwargs):
            nonlocal calls
            calls += 1
            raise AssertionError("focused curve was derived before admission")

        monkeypatch.setattr(DataFigure, "focused_typed_panel", forbidden)
        window._memory_limit_bytes = aggregate - 1
        window._focus_grid_region(*_center(overview.regions[0]))
        _until(application, lambda: window.worker_idle)
        assert calls == 0
        assert window._view_family == "curve-overview"
        assert window._status.text() == "CURVE FOCUS FAILED"
        assert "aggregate peak" in window._diagnostic.text()
    finally:
        _close(application, window)


def test_failed_curve_focus_does_not_publish_unadmitted_worker_cache(
    application,
    monkeypatch,
) -> None:
    figure = _curve_grid()

    def rejected_render(*_args, **_kwargs):
        raise RuntimeError("forced focused Agg rejection")

    monkeypatch.setattr(
        SinglePanelAggRenderer,
        "render_interactive_curve",
        rejected_render,
    )
    window = figure_workbench.open_data_figure_workbench(figure)
    try:
        _until(application, lambda: window.raster_ready and window.worker_idle)
        overview = window._grid_overview
        assert overview is not None
        window._focus_grid_region(*_center(overview.regions[0]))
        _until(application, lambda: window.worker_idle)
        assert window._status.text() == "CURVE FOCUS FAILED"
        assert window._view_family == "curve-overview"
        assert window._grid_focus_cache_charge_bytes == 0
        assert window._current_front_peak_bytes == 0
        retained = inspect.getclosurevars(window._typed_renderer).nonlocals
        assert retained["cached_typed"] is None
        assert retained["cached_base"] is None
    finally:
        _close(application, window)


def test_rejected_curve_present_keeps_worker_cache_charged(
    application,
    monkeypatch,
) -> None:
    window = figure_workbench.open_data_figure_workbench(_curve_grid())
    try:
        _until(application, lambda: window.raster_ready and window.worker_idle)
        overview = window._grid_overview
        assert overview is not None

        def rejected_present(*_args, **_kwargs):
            raise RuntimeError("forced Qt front rejection")

        monkeypatch.setattr(type(window._board_widget), "present", rejected_present)
        window._focus_grid_region(*_center(overview.regions[0]))
        _until(application, lambda: window.worker_idle)
        assert window._status.text() == "GRID FOCUS FAILED"
        assert window._view_family == "curve-overview"
        retained = inspect.getclosurevars(window._typed_renderer).nonlocals
        assert isinstance(retained["cached_typed"], DataFigure)
        assert retained["cached_base"] is retained["cached_typed"]
        assert (
            window._current_front_peak_bytes
            == window._grid_focus_cache_charge_bytes
            > 0
        )
    finally:
        _close(application, window)


def test_escape_during_curve_rerender_cannot_late_present(
    application,
    monkeypatch,
) -> None:
    figure = _curve_grid()
    original = SinglePanelAggRenderer.render_interactive_curve
    entered = threading.Event()
    release = threading.Event()
    calls = 0

    def blocked(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            entered.set()
            if not release.wait(10.0):
                raise TimeoutError("test did not release curve rerender")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(SinglePanelAggRenderer, "render_interactive_curve", blocked)
    window = figure_workbench.open_data_figure_workbench(figure)
    try:
        _until(application, lambda: window.raster_ready and window.worker_idle)
        overview = window._grid_overview
        assert overview is not None
        original_png = window._bundle.pages[0].png_bytes
        window._focus_grid_region(*_center(overview.regions[0]))
        _until(application, lambda: window.worker_idle and window._view_family == "curve")
        assert _wheel_curve(window._board_widget, 120).isAccepted()
        _until(application, entered.is_set)
        QtTest.QTest.keyClick(window, QtCore.Qt.Key_Escape)
        release.set()
        _until(
            application,
            lambda: window.worker_idle and window._view_family == "curve-overview",
        )
        assert window._board_widget.front_frame is None
        assert window._bundle.pages[0].png_bytes is original_png
    finally:
        release.set()
        _close(application, window)


def test_close_during_curve_focus_cannot_present_a_late_front(
    application,
    monkeypatch,
) -> None:
    figure = _curve_grid()
    entered = threading.Event()
    release = threading.Event()
    original = DataFigure.focused_typed_panel

    def blocked(self, *args, **kwargs):
        entered.set()
        if not release.wait(10.0):
            raise TimeoutError("test did not release curve focus")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(DataFigure, "focused_typed_panel", blocked)
    window = figure_workbench.open_data_figure_workbench(figure)
    _until(application, lambda: window.raster_ready and window.worker_idle)
    overview = window._grid_overview
    assert overview is not None
    window._focus_grid_region(*_center(overview.regions[1]))
    _until(application, entered.is_set)
    window.shutdown()
    release.set()
    _until(application, lambda: window.closed)
    _until(application, lambda: window.worker_idle)
    assert window._board_widget.front_frame is None


def test_multi_layer_curve_grid_stays_on_complete_encoded_fallback(application) -> None:
    window = figure_workbench.open_data_figure_workbench(_curve_grid(layers=2))
    try:
        _until(application, lambda: window.raster_ready and window.worker_idle)
        assert window._view_family == "encoded"
        assert window._grid_overview is None
        assert "requires exactly one layer" in window._summary.text()
    finally:
        _close(application, window)


@pytest.mark.parametrize(
    "coordinates",
    (
        pytest.param(("a", "b", "c", "d", "e"), id="categorical"),
        pytest.param((0.0, 1.0, 1.0, 2.0, 3.0), id="repeated"),
        pytest.param((0.0, 1.0, -1.0, 2.0, 3.0), id="nonmonotonic"),
    ),
)
def test_noninteractive_curve_axes_keep_complete_encoded_fallback(
    application,
    coordinates,
) -> None:
    window = figure_workbench.open_data_figure_workbench(
        _curve_grid(scan_coordinates=coordinates)
    )
    try:
        _until(application, lambda: window.raster_ready and window.worker_idle)
        assert window._view_family == "encoded"
        assert window._grid_overview is None
        assert len(window._bundle.pages) == 1
        assert window._bundle.pages[0].png_bytes.startswith(b"\x89PNG\r\n\x1a\n")
    finally:
        _close(application, window)


def test_public_autonomous_occupancy_scan_opens_exact_curve_grid(
    application,
    monkeypatch,
    tmp_path: Path,
) -> None:
    owner_thread = threading.get_ident()
    calls = []
    with zlc.connect("virtual", repository=tmp_path / "u03h-public") as experiment:
        calibration = experiment.readout.sitemap(frames=6)
        document = _occupancy_scan_document()
        reference = experiment.readout.occupancy_scan(
            document,
            calibration_ref=calibration,
            api_values=_fixed_api_values(document),
            timeout_seconds=20.0,
        )
        assert isinstance(reference, zlc.ScanArtifactRef)
        artifact = experiment.readout.load_scan(reference)
        assert isinstance(artifact.execution, AutonomousScanExecution)
        materialized = experiment.readout.materialize_scan(reference)
        assert materialized.values.shape == (2, 2, 35)
        expected_figure = experiment.figure(reference)
        layer = expected_figure.document.layers[0]
        site_axis = materialized.schema.cell_schema.data_axes[0]
        assert layer.view.intent is ViewIntent.CURVE
        assert layer.view.binding(site_axis.axis_id).role is AxisViewRole.FACET
        expected_cells = expected_figure.evaluated.layers[0].cells
        assert len(expected_cells) == 35

        original = type(experiment).figure

        def traced(self, source, *args, **options):
            calls.append((threading.get_ident(), source, args, options))
            return original(self, source, *args, **options)

        monkeypatch.setattr(type(experiment), "figure", traced)
        window = experiment.figure_gui(reference)
        try:
            _until(
                application,
                lambda: window.worker_idle and window.raster_ready,
                timeout=45.0,
            )
            overview = window._grid_overview
            assert overview is not None and len(overview.regions) == 35
            assert window._view_family == "curve-overview"
            assert len(calls) == 1
            thread_id, source, args, options = calls[0]
            assert thread_id != owner_thread
            assert source == reference and args == ()
            assert options["intent"] is None
            assert options["selection"] is None

            window._focus_grid_region(*_center(overview.regions[1]))
            _until(
                application,
                lambda: window.worker_idle and window._view_family == "curve",
            )
            payload = _curve_payload(window)
            expected = expected_cells[1].series
            for actual, source_series in zip(payload.series, expected, strict=True):
                assert actual.batch_address == source_series.batch_address
                assert actual.reductions == source_series.reductions
                assert actual.data.x_axis == source_series.data.x_axis
                assert actual.data.value_unit == source_series.data.value_unit
                np.testing.assert_array_equal(
                    actual.data.values,
                    source_series.data.values,
                )
                np.testing.assert_array_equal(
                    actual.data.validity,
                    source_series.data.validity,
                )
            assert payload.evaluated_input.ref == expected_figure.evaluated.inputs[0].ref
            assert not window._analyze_button.isVisible()
            assert "display projection" in window._diagnostic.text()
            assert "axis-complete source view" in window._diagnostic.text()
        finally:
            _close(application, window)
