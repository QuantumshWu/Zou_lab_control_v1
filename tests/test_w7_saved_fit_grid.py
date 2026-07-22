"""W7 exact saved-fit GridPlot, sparse topology, and reopen oracles."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt5 import QtCore, QtGui, QtTest, QtWidgets
import pytest

import Zou_lab_control.notebook as zlc
from zlc_data import (
    REPEAT,
    SCAN_POINT,
    SPATIAL_X,
    SPATIAL_Y,
    AxisId,
    AxisLayout,
    AxisSpec,
    BlockId,
    CoordinateFrameId,
    DataBlock,
    DatasetRevision,
    DatasetSchema,
    FitBatchStatus,
    FitNumericPolicy,
    IndexSelection,
    OwnedSnapshot,
    PointLayout,
    Selection,
    StreamGenerationId,
    VALID,
    ValidityContract,
    ValueSchema,
    bind_fit,
    fit_model_catalog,
    fit_spec_for,
    resolve_selection_indices,
)
from zlc_data.fit_model import evaluate_fit_model
from zlc_frontend import (
    DataFigure,
    FitGridModel,
    ImageDisplayState,
    ImagePanelPayload,
    PixelFormat,
)
from zlc_frontend.image_display import ImageColormap
from zlc_frontend.figure import (
    AxisViewRole,
    DatasetDescriptor,
    DatasetId,
    FigureDocument,
    FigureLayer,
    RepeatViewMode,
    ResolvedDataset,
    ResolvedDatasetMap,
    SuggestionStatus,
    suggest_fit_view,
)
from zlc_frontend.qt_widgets import ensure_qt_app  # noqa: F401
from zlc_frontend.qt_widgets import AxisLayoutNavigator, FrozenRasterView, QtRasterBoard
from zlc_neutral_atom.artifacts import AdmittedCapture, FitResultRepository
from zlc_neutral_atom.fit_reference import FitResultArtifactRef

from Zou_lab_control.workbench._fit_grid import (
    _build_image_grid_frame,
    _reframe_existing_image_panels,
    _rerasterize_grid_view,
)


ROOT = Path(__file__).resolve().parents[1]
PULSE = ROOT / "zlc_neutral_atom" / "assets" / "imaging_template.json"
CURVE_PULSE = ROOT / "pulses" / "probe_template.json"
ONE_DIMENSIONAL_MODELS = tuple(
    model.model_id for model in fit_model_catalog() if model.independent_arity == 1
)


@pytest.fixture(scope="module")
def application():
    return ensure_qt_app()


def _until(application, predicate, *, timeout: float = 45.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        application.processEvents(QtCore.QEventLoop.AllEvents, 20)
        QtCore.QCoreApplication.sendPostedEvents(
            None,
            QtCore.QEvent.DeferredDelete,
        )
        time.sleep(0.005)
    assert predicate()


def _close(application, window) -> None:
    window.close()
    _until(
        application,
        lambda: window.closed and not window.isVisible(),
        timeout=10.0,
    )


def _choose_image_display_value(editor, key: str, value: object) -> None:
    widget = editor._form.widget_for(key)
    assert isinstance(widget, QtWidgets.QComboBox)
    index = widget.findData(value)
    assert index >= 0
    widget.setCurrentIndex(index)
    editor._form.changed.emit(key)


def _axis(
    name,
    role,
    size,
    coordinates=None,
    *,
    unit=None,
    coordinate_frame=None,
) -> AxisSpec:
    return AxisSpec(
        AxisId(name),
        name,
        role,
        size,
        tuple(range(size)) if coordinates is None else tuple(coordinates),
        unit,
        coordinate_frame,
    )


@pytest.fixture(scope="module")
def sparse_fit_grid():
    repeat = _axis("repeat", REPEAT, 2)
    event = _axis("event", SCAN_POINT, 3)
    frame = CoordinateFrameId("w7-sparse-camera")
    y_axis = _axis(
        "camera.y",
        SPATIAL_Y,
        6,
        unit="pixel",
        coordinate_frame=frame,
    )
    x_axis = _axis(
        "camera.x",
        SPATIAL_X,
        8,
        unit="pixel",
        coordinate_frame=frame,
    )
    point_layout = PointLayout.explicit((3,), ((2,), (0,)))
    y_values, x_values = np.meshgrid(
        np.arange(y_axis.size),
        np.arange(x_axis.size),
        indexing="ij",
    )
    image = evaluate_fit_model(
        "radial_gaussian_center",
        (x_values, y_values),
        (10.0, 1.0, 2.0, 3.0, 2.0),
    )
    values = np.stack(
        tuple(image * (1.0 + 0.05 * index) for index in range(4))
    ).reshape(2, 2, y_axis.size, x_axis.size)
    schema = DatasetSchema(
        repeat,
        (event,),
        point_layout,
        ValueSchema(
            (y_axis, x_axis),
            ValidityContract.value(),
            np.dtype("<f8"),
            "count",
        ),
    )
    block = DataBlock(
        BlockId("w7-sparse-fit"),
        DatasetRevision(1),
        values,
        VALID,
        schema,
    )
    snapshot = OwnedSnapshot(
        block.ref(StreamGenerationId("w7-sparse-generation")),
        block,
    )
    result = bind_fit(
        fit_spec_for(schema, "radial_gaussian_center"),
        schema,
    ).run(snapshot)
    model = FitGridModel.from_result("fit-result/" + "f" * 64, result)
    page = model.page()
    suggestion = suggest_fit_view(
        schema,
        result,
        page.selection,
        page.preferences,
    )
    assert suggestion.status is SuggestionStatus.RESOLVED
    dataset_id = DatasetId("source")
    document = FigureDocument(
        "w7-sparse-grid",
        0,
        (DatasetDescriptor(dataset_id, "saved fit", schema.fingerprint),),
        (FigureLayer("data", dataset_id, suggestion.spec),),
    )
    figure = DataFigure(
        document,
        ResolvedDatasetMap((ResolvedDataset(dataset_id, snapshot),)),
        fit_results={"data": result},
    )
    return result, model, page, suggestion.spec, figure


@pytest.fixture(scope="module")
def sparse_typed_page(sparse_fit_grid):
    result, model, page, view, figure = sparse_fit_grid
    projected = figure.radial_gaussian_image_fit_panels(
        "data",
        artifact_identity=model.artifact_identity,
    )
    frame, color_limits = _build_image_grid_frame(
        projected,
        ImageDisplayState(),
        current_color_limits=None,
        previous_relim_mode=None,
        layout_generation=0,
        sequence=1,
        cancelled=threading.Event(),
    )
    return result, model, page, view, projected, frame, color_limits


@pytest.fixture(scope="module")
def saved_fit_products(tmp_path_factory):
    workspace = tmp_path_factory.mktemp("w7-saved-fit-workspace")
    with zlc.connect("virtual", repository=workspace) as experiment:
        capture = experiment.readout.capture(PULSE)
        radial_reference = experiment.fit(
            capture,
            model="radial_gaussian_center",
        ).save()
        curve_capture = experiment.readout.capture(CURVE_PULSE)
        schema = experiment.readout.load_capture(curve_capture).frame_source.schema
        x_axis = next(
            axis for axis in schema.cell_schema.data_axes if axis.role == SPATIAL_X
        )
        curve_references = {
            model_id: experiment.fit(
                curve_capture,
                model=model_id,
                fit_axis_ids=(x_axis.axis_id,),
                numeric_policy=FitNumericPolicy(
                    max_evaluations=80,
                ),
            ).save()
            for model_id in ONE_DIMENSIONAL_MODELS
        }
        yield experiment, radial_reference, curve_references, workspace


@pytest.fixture(scope="module")
def saved_fit_product(saved_fit_products):
    experiment, reference, _curve_references, workspace = saved_fit_products
    return experiment, reference, workspace


@pytest.fixture(scope="module")
def saved_1d_fit_products(saved_fit_products):
    experiment, _radial_reference, references, workspace = saved_fit_products
    return experiment, references, workspace


def _manifest_count(workspace: Path) -> int:
    root = workspace / "fits" / "content" / "manifests" / "fit-result"
    return 0 if not root.exists() else len(tuple(root.iterdir()))


def _fit_store_state(workspace: Path) -> tuple[tuple[str, str], ...]:
    root = workspace / "fits" / "content"
    return tuple(
        (
            path.relative_to(root).as_posix(),
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    )


def test_saved_fit_grid_public_imports_stay_headless_and_ref_has_exact_identity():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import zlc_frontend; import Zou_lab_control.notebook; "
                "import Zou_lab_control.workbench; "
                "from zlc_neutral_atom.fit_reference import "
                "FitResultArtifactRef; "
                "r=FitResultArtifactRef('repo','f'*64); "
                "assert r.target_ref == 'fit-result/' + 'f'*64; "
                "assert not any(n == 'PyQt5' or n.startswith('PyQt5.') "
                "for n in sys.modules); "
                "assert not any(n == 'matplotlib' or n.startswith('matplotlib.') "
                "for n in sys.modules); "
                "assert not any(n == 'scipy' or n.startswith('scipy.') "
                "for n in sys.modules)"
            ),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=20.0,
    )
    assert result.returncode == 0, result.stderr


def test_typed_image_page_preserves_repeat_sparse_holes_overlay_and_shared_clim(
    sparse_typed_page,
):
    (
        result,
        model,
        page,
        view,
        projected,
        frame,
        color_limits,
    ) = sparse_typed_page
    repeat_axis = next(axis for axis in model.axes if axis.role == REPEAT)
    assert page.preferences.repeat_mode is RepeatViewMode.FACET
    assert view.binding(repeat_axis.axis_id).role is AxisViewRole.FACET
    assert all(
        view.binding(axis.axis_id).role is AxisViewRole.FACET
        for axis in model.axes
    )
    assert not hasattr(model, "result")
    assert model.layout.storage_size == result.batch_layout.storage_size == 4
    assert 0 < len(frame.panels) == len(projected) == 6 <= 36
    assert all(panel.raster.pixel_format is PixelFormat.INDEXED8 for panel in frame.panels)
    payloads = tuple(panel.display_payload for panel in frame.panels)
    assert all(isinstance(payload, ImagePanelPayload) for payload in payloads)
    assert {payload.color_limits for payload in payloads} == {color_limits}
    assert color_limits[0] < color_limits[1]

    present = tuple(
        panel for panel in projected if panel.fit_storage_index is not None
    )
    holes = tuple(panel for panel in projected if panel.fit_storage_index is None)
    assert len(present) == 4
    assert len(holes) == 2
    assert {panel.fit_storage_index for panel in present} == set(range(4))
    for projected_panel, rendered_panel in zip(
        projected,
        frame.panels,
        strict=True,
    ):
        payload = rendered_panel.display_payload
        assert isinstance(payload, ImagePanelPayload)
        assert payload.image is projected_panel.image
        assert payload.fit_overlay == projected_panel.fit_overlay
        assert payload.fit_overlay.artifact_identity == model.artifact_identity
        assert payload.fit_overlay.source_ref == result.source_ref
        assert (
            model.storage_index_or_none(projected_panel.selection)
            == projected_panel.fit_storage_index
        )
        if projected_panel.fit_storage_index is None:
            assert payload.fit_overlay.status is None
            assert payload.fit_overlay.center_xy is None
            assert payload.fit_overlay.one_over_e_radius is None
            assert payload.fit_overlay.diagnostic == "NOT_PRESENT"
            assert "no neighbouring row was substituted" in projected_panel.summary
        else:
            assert (
                payload.fit_overlay.status
                is result.statuses[projected_panel.fit_storage_index]
            )


def test_focus_and_overview_reframe_cached_typed_panels_without_copying_samples(
    sparse_typed_page,
):
    _result, _model, _page, _view, projected, frame, _limits = (
        sparse_typed_page
    )
    present_index = next(
        index
        for index, panel in enumerate(projected)
        if panel.fit_storage_index is not None
    )
    focus = _reframe_existing_image_panels(
        frame,
        (projected[present_index],),
        layout_generation=1,
        sequence=2,
    )
    restored = _reframe_existing_image_panels(
        frame,
        projected,
        layout_generation=2,
        sequence=3,
    )

    assert focus.board_id == restored.board_id == frame.board_id
    assert len(focus.panels) == 1
    original = frame.panels[present_index]
    assert focus.panels[0].raster is original.raster
    assert focus.panels[0].display_payload is original.display_payload
    assert (
        focus.panels[0].coherence_stamp.join_key_digest
        != frame.panels[0].coherence_stamp.join_key_digest
    )
    assert (
        restored.panels[0].coherence_stamp.join_key_digest
        == frame.panels[0].coherence_stamp.join_key_digest
    )
    assert tuple(panel.panel_id for panel in restored.panels) == tuple(
        panel.panel_id for panel in frame.panels
    )
    assert all(
        replacement.raster is source.raster
        and replacement.display_payload is source.display_payload
        for replacement, source in zip(restored.panels, frame.panels, strict=True)
    )


def test_display_reraster_advances_only_the_presentation_and_keeps_shared_clim(
    sparse_typed_page,
):
    _result, _model, _page, _view, projected, initial, initial_limits = (
        sparse_typed_page
    )
    initial_display = ImageDisplayState()
    changed = ImageDisplayState(revision=1, colormap=ImageColormap.MAGMA)
    (
        request_revision,
        returned_panels,
        returned_display,
        rerasterized,
        rerasterized_limits,
    ) = _rerasterize_grid_view(
        projected,
        changed,
        initial_limits,
        initial_display.relim_mode,
        0,
        2,
        threading.Event(),
    )

    assert request_revision == 2
    assert returned_panels == projected
    assert returned_display is changed
    assert rerasterized.layout_generation == initial.layout_generation
    assert rerasterized.sequence > initial.sequence
    assert rerasterized_limits == initial_limits
    for old, new, source in zip(
        initial.panels,
        rerasterized.panels,
        projected,
        strict=True,
    ):
        old_payload = old.display_payload
        new_payload = new.display_payload
        assert isinstance(old_payload, ImagePanelPayload)
        assert isinstance(new_payload, ImagePanelPayload)
        assert new_payload.image is source.image is old_payload.image
        assert new_payload.evaluated_input is source.evaluated_input
        assert new_payload.viewport.viewport_revision == changed.revision
        assert new_payload.color_limits == rerasterized_limits
        assert new_payload.base_palette != old_payload.base_palette
        presentation = next(
            item
            for item in new.coherence_stamp.presentations
            if item.panel_id == new.panel_id
        )
        assert presentation.panel_revision == changed.revision


def test_grid_pages_are_bounded_and_axis_navigator_skips_sparse_holes(
    application,
    sparse_fit_grid,
):
    _result, model, _page, _view, _figure = sparse_fit_grid
    navigator = AxisLayoutNavigator(
        model.axes,
        model.layout,
        object_prefix="w7Oracle",
        action_text="Focus",
    )
    try:
        assert all(
            isinstance(control, QtWidgets.QSpinBox)
            for _axis, control, _coordinate in navigator._controls
        )
        navigator.set_storage_index(0)
        assert navigator.indices == model.layout.multi_index(0)
        navigator.next_button.click()
        assert navigator.indices == model.layout.multi_index(1)
        assert navigator.storage_index == 1
        hole = (0, 1)
        for (_axis_value, spin, _coordinate), index in zip(
            navigator._controls,
            hole,
            strict=True,
        ):
            spin.setValue(index)
        assert navigator.storage_index is None
        assert not navigator.action_button.isEnabled()
    finally:
        navigator.deleteLater()
        application.processEvents()


def test_axis_navigator_preserves_bigint_sparse_indices_and_physical_order(
    application,
) -> None:
    first = (1 << 40) + 100
    second = (1 << 31) + 7
    size = first + 2
    axis = AxisSpec(
        AxisId("big.logical"),
        "big logical index",
        REPEAT,
        size,
        None,
    )
    layout = AxisLayout.explicit((size,), ((first,), (second,)))
    navigator = AxisLayoutNavigator(
        (axis,),
        layout,
        object_prefix="w7BigintOracle",
        action_text="Focus",
    )
    try:
        control = navigator._controls[0][1]
        assert isinstance(control, QtWidgets.QLineEdit)
        assert not isinstance(control, QtWidgets.QSpinBox)
        control.setFocus()
        QtTest.QTest.keyClicks(control, str(second))
        application.processEvents()
        assert navigator.indices == (second,)
        assert navigator.storage_index == 1

        navigator.set_storage_index(0)
        assert navigator.indices == (first,)
        navigator.next_button.click()
        assert navigator.indices == (second,)
        assert navigator.storage_index == 1
        navigator.previous_button.click()
        assert navigator.indices == (first,)
        assert navigator.storage_index == 0
    finally:
        navigator.deleteLater()
        QtCore.QCoreApplication.sendPostedEvents(
            None,
            QtCore.QEvent.DeferredDelete,
        )


def test_axis_navigator_preserves_full_labels_and_exact_invalid_indices(
    application,
) -> None:
    long_coordinate = "x" * 100_000
    text_axis = AxisSpec(
        AxisId("long.coordinate"),
        "long coordinate",
        REPEAT,
        1,
        (long_coordinate,),
    )
    text_navigator = AxisLayoutNavigator(
        (text_axis,),
        AxisLayout.rect_c((1,)),
        object_prefix="w7LongCoordinateOracle",
        action_text="Focus",
    )
    huge_index = 1 << 20_000
    huge_axis = AxisSpec(
        AxisId("huge.logical"),
        "huge logical index",
        REPEAT,
        huge_index + 1,
        None,
    )
    huge_navigator = AxisLayoutNavigator(
        (huge_axis,),
        AxisLayout.explicit((huge_axis.size,), ((huge_index,),)),
        object_prefix="w7HugeCoordinateOracle",
        action_text="Focus",
    )
    try:
        text_label = text_navigator._controls[0][2].text()
        assert text_label == f"{long_coordinate} · index 0"

        huge_navigator.set_storage_index(0)
        huge_label = huge_navigator._controls[0][2].text()
        assert huge_navigator.indices == (huge_index,)
        assert huge_label == f"{hex(huge_index)} · index {hex(huge_index)}"
        exact_edit = huge_navigator._controls[0][1]
        assert exact_edit.maxLength() == (1 << 31) - 1
        assert exact_edit.text() == hex(huge_index)

        hostile = 1 << 25_000
        with pytest.raises(IndexError) as failure:
            resolve_selection_indices(
                huge_axis,
                IndexSelection(huge_axis.axis_id, hostile),
            )
        assert hex(hostile) in str(failure.value)
    finally:
        text_navigator.deleteLater()
        huge_navigator.deleteLater()
        QtCore.QCoreApplication.sendPostedEvents(
            None,
            QtCore.QEvent.DeferredDelete,
        )


def test_large_grid_is_tiled_without_flattening_or_defaulting_repeat() -> None:
    repeat = _axis("large.repeat", REPEAT, 80)
    y_axis = _axis("large.y", SPATIAL_Y, 3)
    x_axis = _axis("large.x", SPATIAL_X, 3)
    model = FitGridModel(
        "fit-result/" + "e" * 64,
        "radial_gaussian_center",
        (x_axis, y_axis),
        (repeat,),
        AxisLayout.rect_c((repeat.size,)),
        ((FitBatchStatus.CONVERGED, repeat.size),),
    )
    first = model.page()
    second = model.page(first.next_address)
    third = model.page(second.next_address)
    assert model.page_spans == (36,)
    assert first.label == "large.repeat[0:36]"
    assert second.label == "large.repeat[36:72]"
    assert third.label == "large.repeat[72:80]"
    assert first.preferences.repeat_mode is RepeatViewMode.FACET
    assert first.previous_address is None
    assert third.next_address is None
    assert all(
        term.stop - term.start <= 36
        for page in (first, second, third)
        for term in page.selection.terms
    )


def test_saved_ref_uses_one_typed_board_cached_focus_display_and_exact_export(
    application,
    saved_fit_product,
    monkeypatch,
    tmp_path,
):
    experiment, reference, workspace = saved_fit_product
    owner_thread = threading.get_ident()
    load_threads = []
    materialize_threads = []
    original_load = FitResultRepository.load
    original_materialize = AdmittedCapture.materialize_snapshot

    def observed_load(self, *args, **kwargs):
        load_threads.append(threading.get_ident())
        return original_load(self, *args, **kwargs)

    def forbidden_execute(*_args, **_kwargs):
        raise AssertionError("saved-fit reopen/export must never run the solver")

    def observed_materialize(self, *args, **kwargs):
        materialize_threads.append(threading.get_ident())
        return original_materialize(self, *args, **kwargs)

    monkeypatch.setattr(FitResultRepository, "load", observed_load)
    monkeypatch.setattr(FitResultRepository, "execute_capture", forbidden_execute)
    monkeypatch.setattr(AdmittedCapture, "materialize_snapshot", observed_materialize)
    manifests_before = _manifest_count(workspace)
    window = experiment.figure_gui(reference)
    try:
        _until(application, lambda: window.worker_idle)
        assert window.raster_ready, window._diagnostic.text()
        assert type(window).__name__ == "SavedFitGridWindow"
        assert len(load_threads) == 1
        assert load_threads[0] != owner_thread
        assert materialize_threads == [load_threads[0]]
        assert not hasattr(window._model, "result")
        assert window._page is not None
        board = window._board_widget
        assert isinstance(board, QtRasterBoard)
        assert board is window.findChild(QtRasterBoard, "savedFitGridBoard")
        page_front = board.front_frame
        assert page_front is not None
        assert 0 < len(page_front.panels) <= 36
        assert window._page_frame is page_front
        page_payloads = tuple(panel.display_payload for panel in page_front.panels)
        assert all(isinstance(payload, ImagePanelPayload) for payload in page_payloads)
        assert {payload.color_limits for payload in page_payloads} == {
            page_payloads[0].color_limits
        }
        assert all(
            payload.fit_overlay is not None
            and payload.fit_overlay.artifact_identity == reference.target_ref
            for payload in page_payloads
        )
        assert window._bound_panel_ids == set(board.panel_ids)
        assert all(
            board.visible_image_payload(panel_id) is not None
            and board.image_selector_fault(panel_id) is None
            for panel_id in board.panel_ids
        )

        assert window._selector_switch.isEnabled(), window._diagnostic.text()
        window._selector_switch.setChecked(True)
        assert board.selectors_enabled, window._diagnostic.text()
        window._selector_switch.setChecked(False)
        assert not board.selectors_enabled
        assert _manifest_count(workspace) == manifests_before

        before_display = board.front_frame
        assert before_display is not None
        old_images = {
            panel.panel_id: panel.display_payload.image
            for panel in before_display.panels
            if isinstance(panel.display_payload, ImagePanelPayload)
        }
        old_palettes = {
            panel.panel_id: panel.display_payload.base_palette
            for panel in before_display.panels
            if isinstance(panel.display_payload, ImagePanelPayload)
        }
        _choose_image_display_value(
            window._edit_image_display,
            "colormap",
            ImageColormap.MAGMA,
        )
        window._edit_image_display._apply_button.click()
        assert window._display.revision == 1
        _until(
            application,
            lambda: window.worker_idle
            and window.raster_ready
            and board.front_frame is not before_display,
        )
        after_display = board.front_frame
        assert after_display is not None
        assert after_display.layout_generation == before_display.layout_generation
        assert after_display.sequence > before_display.sequence
        assert len(load_threads) == len(materialize_threads) == 1
        for panel in after_display.panels:
            payload = panel.display_payload
            assert isinstance(payload, ImagePanelPayload)
            assert payload.image is old_images[panel.panel_id]
            assert payload.viewport.viewport_revision == window._display.revision
            assert payload.base_palette != old_palettes[panel.panel_id]

        focused_panel = next(
            panel
            for panel in after_display.panels
            if isinstance(panel.display_payload, ImagePanelPayload)
            and panel.display_payload.fit_overlay is not None
            and panel.display_payload.fit_overlay.batch_storage_index is not None
        )
        projected = next(
            panel
            for panel, rendered in zip(
                window._page_panels,
                window._page_frame.panels,
                strict=True,
            )
            if rendered.panel_id == focused_panel.panel_id
        )
        assert projected.selection is not None
        board.imagePanelLeftDoubleClicked.emit(focused_panel.panel_id)
        _until(
            application,
            lambda: window.worker_idle
            and window._current_selection == projected.selection
            and board.front_frame is not None
            and len(board.front_frame.panels) == 1,
        )
        assert window._board_widget is board
        assert len(load_threads) == 1
        assert len(materialize_threads) == 1
        focused_front = board.front_frame
        assert focused_front is not None
        assert focused_front.panels[0].panel_id == focused_panel.panel_id
        assert focused_front.panels[0].raster is focused_panel.raster
        assert (
            focused_front.panels[0].display_payload
            is focused_panel.display_payload
        )
        storage, multi, _label = window._model.resolve_selection(projected.selection)
        assert storage == projected.fit_storage_index
        assert window._navigator.indices == multi
        assert f"storage row {storage}" in window._cell_detail.text()
        assert "status " in window._cell_detail.text()

        escape = QtGui.QKeyEvent(
            QtCore.QEvent.KeyPress,
            QtCore.Qt.Key_Escape,
            QtCore.Qt.NoModifier,
        )
        QtWidgets.QApplication.sendEvent(window, escape)
        application.processEvents()
        assert escape.isAccepted()
        assert window._current_selection is None
        assert window._board_widget is board
        assert board.front_frame is not None
        assert len(board.front_frame.panels) == len(after_display.panels)
        assert tuple(panel.panel_id for panel in board.front_frame.panels) == tuple(
            panel.panel_id for panel in after_display.panels
        )
        assert all(
            restored.raster is source.raster
            and restored.display_payload is source.display_payload
            for restored, source in zip(
                board.front_frame.panels,
                after_display.panels,
                strict=True,
            )
        )
        assert len(load_threads) == len(materialize_threads) == 1

        destination = tmp_path / "saved-fit-grid.png"
        window._start_export(destination)
        _until(
            application,
            lambda: window.worker_idle and destination.exists(),
        )
        assert destination.stat().st_size > 0
        assert len(load_threads) == 1
        assert len(materialize_threads) == 1
        assert all(thread != owner_thread for thread in load_threads)
        assert _manifest_count(workspace) == manifests_before
    finally:
        _close(application, window)


@pytest.mark.parametrize("model_id", ONE_DIMENSIONAL_MODELS)
def test_every_saved_1d_catalog_model_reopens_focuses_and_exports_without_refit(
    application,
    saved_1d_fit_products,
    monkeypatch,
    tmp_path,
    model_id,
):
    assert ONE_DIMENSIONAL_MODELS == (
        "lorentzian",
        "gaussian_offset",
        "symmetric_lorentzian_doublet",
        "damped_sine",
        "exponential_decay",
    )
    experiment, references, workspace = saved_1d_fit_products
    reference = references[model_id]
    owner_thread = threading.get_ident()
    load_threads = []
    materialize_threads = []
    original_load = FitResultRepository.load
    original_materialize = AdmittedCapture.materialize_snapshot

    def observed_load(self, *args, **kwargs):
        load_threads.append(threading.get_ident())
        return original_load(self, *args, **kwargs)

    def forbidden_execute(*_args, **_kwargs):
        raise AssertionError("saved 1D reopen/export must never run the solver")

    def observed_materialize(self, *args, **kwargs):
        materialize_threads.append(threading.get_ident())
        return original_materialize(self, *args, **kwargs)

    monkeypatch.setattr(FitResultRepository, "load", observed_load)
    monkeypatch.setattr(FitResultRepository, "execute_capture", forbidden_execute)
    monkeypatch.setattr(AdmittedCapture, "materialize_snapshot", observed_materialize)
    immutable_store_before = _fit_store_state(workspace)
    manifests_before = _manifest_count(workspace)
    window = experiment.figure_gui(reference)
    try:
        _until(application, lambda: window.worker_idle)
        assert window.raster_ready, window._diagnostic.text()
        assert type(window).__name__ == "SavedFitGridWindow"
        assert window._model.model_id == model_id
        assert window._view_family == "encoded"
        assert isinstance(window._encoded_board, FrozenRasterView)
        assert window._encoded_board.has_front
        assert window._current_encoded_bundle is not None
        assert window._page_encoded_bundle is window._current_encoded_bundle
        assert 0 < len(window._regions) <= 36
        assert len(load_threads) == 1
        assert load_threads[0] != owner_thread
        assert materialize_threads == [load_threads[0]]
        assert not window._selector_switch.isEnabled()
        assert not window._setting_button.isEnabled()

        first_index = window._model.layout.multi_index(0)
        selection = window._model.selection_for_indices(first_index)
        window._start_focus(selection)
        _until(
            application,
            lambda: window.worker_idle
            and window.raster_ready
            and window._current_selection == selection,
        )
        assert window._view_family == "encoded"
        assert window._encoded_board.has_front
        assert len(window._regions) == 1
        storage, multi, _label = window._model.resolve_selection(selection)
        assert multi == first_index
        assert f"storage row {storage}" in window._cell_detail.text()
        assert len(load_threads) == len(materialize_threads) == 1

        window._show_page()
        assert window._current_selection is None
        assert window._encoded_board.has_front
        destination = tmp_path / f"saved-{model_id}.png"
        window._start_export(destination)
        _until(
            application,
            lambda: window.worker_idle and destination.exists(),
        )
        assert destination.stat().st_size > 0
        assert len(load_threads) == len(materialize_threads) == 1
        assert _manifest_count(workspace) == manifests_before
        assert _fit_store_state(workspace) == immutable_store_before
    finally:
        _close(application, window)


def test_saved_fit_grid_close_during_load_is_nonblocking_and_releases_state(
    application,
    saved_fit_product,
    monkeypatch,
):
    experiment, reference, _workspace = saved_fit_product

    entered = threading.Event()
    release = threading.Event()
    original_load = FitResultRepository.load

    def blocked_load(self, *args, **kwargs):
        entered.set()
        if not release.wait(10.0):
            raise TimeoutError("test did not release saved-fit load")
        return original_load(self, *args, **kwargs)

    monkeypatch.setattr(FitResultRepository, "load", blocked_load)
    window = experiment.figure_gui(reference)
    try:
        _until(application, entered.is_set)
        started = time.monotonic()
        window.close()
        assert time.monotonic() - started < 0.1
        assert not window.closed
        release.set()
        _until(application, lambda: window.closed and not window.isVisible())
        assert window._model is None
        assert window._page_frame is None
        assert window._page_panels == ()
        assert window._current_frame is None
        assert window._current_panels == ()
        assert window._bound_panel_ids == set()
        assert not window._board_widget.has_front
    finally:
        release.set()
        if not window.closed:
            _close(application, window)


def test_close_during_export_preserves_existing_destination_atomically(
    application,
    saved_fit_product,
    monkeypatch,
    tmp_path,
):
    experiment, reference, _workspace = saved_fit_product
    window = experiment.figure_gui(reference)
    _until(application, lambda: window.worker_idle and window.raster_ready)

    destination = tmp_path / "saved-fit-grid.png"
    original_bytes = b"pre-existing-authoritative-destination"
    destination.write_bytes(original_bytes)
    entered = threading.Event()
    release = threading.Event()
    import zlc_frontend.matplotlib_render as matplotlib_render

    original_export = matplotlib_render.save_radial_gaussian_image_fit_panels

    def blocked_export(*args, **kwargs):
        exported = original_export(*args, **kwargs)
        entered.set()
        if not release.wait(10.0):
            raise TimeoutError("test did not release staged fit export")
        return exported

    monkeypatch.setattr(
        matplotlib_render,
        "save_radial_gaussian_image_fit_panels",
        blocked_export,
    )
    try:
        window._start_export(destination)
        _until(application, entered.is_set)
        started = time.monotonic()
        window.close()
        assert time.monotonic() - started < 0.1
        assert not window.closed
        release.set()
        _until(application, lambda: window.closed and not window.isVisible())
        assert destination.read_bytes() == original_bytes
        assert not tuple(tmp_path.glob(f".{destination.name}.*"))
    finally:
        release.set()
        if not window.closed:
            _close(application, window)
