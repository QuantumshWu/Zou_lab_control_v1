"""A0 notebook DataFigure product path with independent presentation oracles."""

from __future__ import annotations

from dataclasses import replace
import gc
from pathlib import Path
import subprocess
import sys
import weakref

import numpy as np
import pytest

import Zou_lab_control.notebook as zlc
from zlc_data import (
    REPEAT,
    SCAN_POINT,
    SITE,
    AxisId,
    AxisSpec,
    BlockId,
    CellValidity,
    ComponentValidity,
    DataBlock,
    DataTransformSpec,
    DatasetRevision,
    DatasetSchema,
    FitBatchStatus,
    FitNumericPolicy,
    OwnedSnapshot,
    PointLayout,
    Selection,
    StreamGenerationId,
    ValidityContract,
    ValueSchema,
    commit_transform,
    bind_fit,
    fit_result_retained_upper_bound_nbytes,
    fit_spec_for,
)
from zlc_frontend import DataFigure
from zlc_frontend.figure import (
    AxisViewRole,
    DatasetDescriptor,
    DatasetId,
    FigureDocument,
    FigureLayer,
    ResolvedDataset,
    ResolvedDatasetMap,
    ViewIntent,
    suggest_view,
    suggest_fit_view,
)


ROOT = Path(__file__).resolve().parents[1]


def _axis(name, role, size, coordinates):
    return AxisSpec(
        AxisId(name),
        name,
        role,
        size,
        tuple(coordinates),
        None,
        None,
    )


def _curve_figure():
    repeat = _axis("repeat", REPEAT, 2, (0, 1))
    scan = _axis("detuning", SCAN_POINT, 4, (-2.0, -0.5, 1.0, 3.0))
    schema = DatasetSchema(
        repeat,
        (scan,),
        PointLayout.rect_c((4,)),
        ValueSchema((), ValidityContract.value(), np.dtype("<f8")),
    )
    block = DataBlock(
        BlockId("curve-block"),
        DatasetRevision(2),
        np.asarray(
            (
                (1.0, 2.0, 99.0, 4.0),
                (3.0, 4.0, 101.0, 6.0),
            )
        ),
        CellValidity(
            np.asarray(
                (
                    (True, True, False, True),
                    (True, True, False, True),
                )
            )
        ),
        schema,
    )
    dataset_id = DatasetId("source")
    view = suggest_view(schema, ViewIntent.CURVE).spec
    assert view is not None
    document = FigureDocument(
        "curve-document",
        0,
        (DatasetDescriptor(dataset_id, "curve", schema.fingerprint),),
        (FigureLayer("data", dataset_id, view),),
    )
    snapshot = OwnedSnapshot(block.ref(StreamGenerationId("curve-generation")), block)
    return DataFigure(
        document,
        ResolvedDatasetMap((ResolvedDataset(dataset_id, snapshot),)),
    )


def _resolved_capture(exp, capture_ref, document):
    artifact = exp.readout.load_capture(capture_ref)
    block = artifact.frame_source.materialize(memory_limit_bytes=512 << 20)
    snapshot = OwnedSnapshot(block.ref(artifact.provenance.generation), block)
    dataset_id = document.datasets[0].dataset_id
    return block, ResolvedDatasetMap((ResolvedDataset(dataset_id, snapshot),))


def _site_curve_fit_figure():
    repeat = _axis("repeat", REPEAT, 1, (0,))
    scan = _axis("detuning", SCAN_POINT, 21, np.linspace(-2.0, 2.0, 21))
    site = _axis("site", SITE, 3, ("left", "middle", "dead"))
    x = np.asarray(scan.coordinates)
    values = np.stack(
        (
            1.0 + 4.0 * np.exp(-(x**2)),
            2.0 + 3.0 * np.exp(-((x - 0.4) ** 2) / 0.7),
            np.zeros_like(x),
        ),
        axis=-1,
    )[None, ...]
    valid = np.ones(values.shape, dtype=np.bool_)
    valid[:, :, 2] = False
    schema = DatasetSchema(
        repeat,
        (scan,),
        PointLayout.rect_c((scan.size,)),
        ValueSchema(
            (site,),
            ValidityContract.components(site.axis_id),
            values.dtype,
        ),
    )
    block = DataBlock(
        BlockId("site-curve-block"),
        DatasetRevision(1),
        values,
        ComponentValidity((site.axis_id,), valid),
        schema,
    )
    snapshot = OwnedSnapshot(block.ref(StreamGenerationId("site-curve-generation")), block)
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
    dataset_id = DatasetId("source")
    document = FigureDocument(
        "site-curve-document",
        0,
        (DatasetDescriptor(dataset_id, "site curves", schema.fingerprint),),
        (FigureLayer("data", dataset_id, suggestion.spec),),
    )
    return (
        DataFigure(
            document,
            ResolvedDatasetMap((ResolvedDataset(dataset_id, snapshot),)),
            fit_results={"data": result},
        ),
        result,
        site,
    )


def _sparse_curve_fit_figure():
    repeat = _axis("repeat", REPEAT, 1, (0,))
    grid_x = _axis("grid_x", SITE, 2, ("left", "right"))
    grid_y = _axis("grid_y", SITE, 2, ("lower", "upper"))
    scan = _axis("detuning", SCAN_POINT, 21, np.linspace(-2.0, 2.0, 21))
    present_pairs = ((0, 0), (1, 0), (1, 1))
    mapping = tuple(
        (x_index, y_index, scan_index)
        for x_index, y_index in present_pairs
        for scan_index in range(scan.size)
    )
    point_layout = PointLayout.explicit(
        (grid_x.size, grid_y.size, scan.size),
        mapping,
    )
    coordinates = np.asarray(scan.coordinates)
    curve = 1.0 + 4.0 * np.exp(-(coordinates**2))
    values = np.asarray(
        [
            curve[scan_index] + x_index + 0.25 * y_index
            for x_index, y_index, scan_index in mapping
        ]
    )[None, :]
    schema = DatasetSchema(
        repeat,
        (grid_x, grid_y, scan),
        point_layout,
        ValueSchema((), ValidityContract.value(), values.dtype),
    )
    block = DataBlock(
        BlockId("sparse-curve-block"),
        DatasetRevision(1),
        values,
        CellValidity(np.ones(values.shape, dtype=np.bool_)),
        schema,
    )
    snapshot = OwnedSnapshot(
        block.ref(StreamGenerationId("sparse-curve-generation")),
        block,
    )
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
    dataset_id = DatasetId("source")
    document = FigureDocument(
        "sparse-curve-document",
        0,
        (DatasetDescriptor(dataset_id, "sparse curves", schema.fingerprint),),
        (FigureLayer("data", dataset_id, suggestion.spec),),
    )
    return DataFigure(
        document,
        ResolvedDatasetMap((ResolvedDataset(dataset_id, snapshot),)),
        fit_results={"data": result},
    ), result


def _displayed_batch_indices(data_figure, result):
    indices = []
    layer = data_figure.evaluated.layers[0]
    for cell in layer.cells:
        for series in cell.series:
            addresses = (*cell.facet_address, *series.batch_address)
            by_axis = {item.axis_id: item.index for item in addresses}
            by_axis.update(
                {resolution.axis_id: resolution.index for resolution in layer.resolutions}
            )
            multi = tuple(by_axis[axis.axis_id] for axis in result.batch_axis_specs)
            indices.append(result.batch_layout.storage_index(multi))
    return indices


def test_import_is_lazy_and_invalid_curve_is_masked_in_png_and_svg(tmp_path):
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import zlc_frontend; import Zou_lab_control.notebook; "
                "assert not any(n == 'matplotlib' or n.startswith('matplotlib.') "
                "for n in sys.modules)"
            ),
        ],
        cwd=ROOT,
        check=True,
    )

    data_figure = _curve_figure()
    rendered = data_figure.render()
    y = rendered.axes[0].lines[0].get_ydata()
    assert np.ma.getmaskarray(y).tolist() == [False, False, True, False]
    assert "reduce: mean(repeat, n=0..2)" in rendered.axes[0].get_title()
    png = data_figure.to_png_bytes()
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    svg = data_figure.export(tmp_path / "curve.svg")
    assert svg.read_text(encoding="utf-8").lstrip().startswith("<?xml")


def test_product_render_dpi_and_save_draw_use_the_render_owner_context(
    monkeypatch,
    tmp_path,
):
    import matplotlib
    from matplotlib.figure import Figure

    from zlc_frontend.render_style import RENDER_TEXT

    data_figure = _curve_figure()
    rendered = data_figure.render(dpi=80.0)
    assert rendered.canvas.get_width_height() == (400, 320)
    rendered.clear()

    observed = []
    released_figures = []
    original_savefig = Figure.savefig

    def checked_savefig(self, *args, **kwargs):
        observed.append(matplotlib.rcParams["axes.edgecolor"])
        released_figures.append(weakref.ref(self))
        return original_savefig(self, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", checked_savefig)
    original_edge = matplotlib.rcParams["axes.edgecolor"]
    matplotlib.rcParams["axes.edgecolor"] = "magenta"
    collection_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(3):
            assert data_figure.to_png_bytes(dpi=80.0).startswith(b"\x89PNG")
        data_figure.export(tmp_path / "owned.svg", dpi=80.0)
        assert observed == [RENDER_TEXT] * 4
        assert released_figures and all(ref() is None for ref in released_figures)
        assert matplotlib.rcParams["axes.edgecolor"] == "magenta"
    finally:
        if collection_was_enabled:
            gc.enable()
        matplotlib.rcParams["axes.edgecolor"] = original_edge

    import zlc_frontend.matplotlib_render as render_module

    partial_canvases = []

    def failed_curve(axis, *_args, **_kwargs):
        partial_canvases.append(weakref.ref(axis.figure.canvas))
        raise RuntimeError("injected artist compose failure")

    collection_was_enabled = gc.isenabled()
    gc.disable()
    try:
        with monkeypatch.context() as failure_patch:
            failure_patch.setattr(render_module, "_curve", failed_curve)
            with pytest.raises(RuntimeError, match="injected artist compose failure"):
                data_figure.to_png_bytes(dpi=80.0)
        assert partial_canvases and all(ref() is None for ref in partial_canvases)
    finally:
        if collection_was_enabled:
            gc.enable()


def test_notebook_fit_figure_maps_each_visible_batch_and_skips_failure(monkeypatch, tmp_path):
    with zlc.connect("virtual", repository=tmp_path / "workspace") as exp:
        capture_ref = exp.readout.capture(
            ROOT / "zlc_neutral_atom" / "assets" / "imaging_template.json"
        )
        execution = exp.fit(
            capture_ref,
            model="radial_gaussian_center",
            numeric_policy=FitNumericPolicy(
                max_evaluations=500,
                sample_budget_per_batch=512,
                max_packed_observations=4_096,
            ),
        )
        document = exp.figure_document(execution)
        view = document.layers[0].view
        assert view.intent is ViewIntent.IMAGE
        assert view.binding(execution.result.fit_axis_specs[0].axis_id).role is AxisViewRole.IMAGE_X
        assert view.binding(execution.result.fit_axis_specs[1].axis_id).role is AxisViewRole.IMAGE_Y
        repeat_batch_axis = next(
            axis for axis in execution.result.batch_axis_specs if axis.role == REPEAT
        )
        assert view.binding(repeat_batch_axis.axis_id).role in {
            AxisViewRole.SELECTED,
            AxisViewRole.SLIDER,
        }

        render_limit = 512 << 20
        data_figure = exp.figure(execution, memory_limit_bytes=render_limit)
        assert data_figure.render_memory_limit_bytes == (
            render_limit
            - fit_result_retained_upper_bound_nbytes(execution.result)
        )
        import zlc_frontend.matplotlib_render as render_module

        with monkeypatch.context() as budget_patch:
            budget_patch.setattr(
                render_module,
                "estimate_render_peak_nbytes",
                lambda _evaluated, *, dpi: render_limit + 1,
            )
            with pytest.raises(MemoryError, match="render peak"):
                data_figure._repr_png_()
            blocked_export = tmp_path / "frozen-budget-blocked.png"
            with pytest.raises(MemoryError, match="render peak"):
                data_figure.export(blocked_export)
            assert not blocked_export.exists()
            with pytest.raises(ValueError, match="cannot weaken"):
                data_figure.export(
                    tmp_path / "must-not-export.png",
                    memory_limit_bytes=render_limit + 1,
                )
        assert data_figure.to_png_bytes(
            memory_limit_bytes=render_limit // 2
        ).startswith(b"\x89PNG")
        expected = _displayed_batch_indices(data_figure, execution.result)
        calls = []
        original = type(execution.result).evaluate_batch

        def traced(self, index, coordinates):
            calls.append((index, tuple(array.shape for array in coordinates)))
            return original(self, index, coordinates)

        monkeypatch.setattr(type(execution.result), "evaluate_batch", traced)
        data_figure.render()
        converged = [
            index
            for index in expected
            if execution.result.statuses[index] is FitBatchStatus.CONVERGED
        ]
        assert [index for index, _shape in calls] == converged
        assert all(len(shape) == 2 and shape[0] == shape[1] for _index, shape in calls)

        result = execution.result
        block, datasets = _resolved_capture(exp, capture_ref, document)
        transformed_input = commit_transform(
            block.schema,
            DataTransformSpec(
                (Selection.index(block.schema.repeat_axis.axis_id, 0),)
            ),
        )
        transformed = replace(
            result,
            spec=replace(result.spec, committed_transform=transformed_input),
        )
        with pytest.raises(ValueError, match="transformed fit"):
            DataFigure(document, datasets, fit_results={"data": transformed})

        spoofed_batch_axes = (
            replace(result.batch_axis_specs[0], name="spoofed batch axis"),
            *result.batch_axis_specs[1:],
        )
        spoofed = replace(result, batch_axis_specs=spoofed_batch_axes)
        with pytest.raises(ValueError, match="batch axes differ from source schema"):
            DataFigure(document, datasets, fit_results={"data": spoofed})


def test_curve_grid_fit_uses_named_site_batches_and_component_validity(monkeypatch):
    data_figure, result, site = _site_curve_fit_figure()
    calls = []
    original = type(result).evaluate_batch

    def traced(self, index, coordinates):
        calls.append(index)
        return original(self, index, coordinates)

    monkeypatch.setattr(type(result), "evaluate_batch", traced)
    rendered = data_figure.render()
    displayed = _displayed_batch_indices(data_figure, result)
    dead_multi = tuple(
        2 if axis.axis_id == site.axis_id else 0 for axis in result.batch_axis_specs
    )
    dead_index = result.batch_layout.storage_index(dead_multi)
    assert result.statuses[dead_index] is FitBatchStatus.NO_VALID_DATA
    assert calls == [
        index for index in displayed if result.statuses[index] is FitBatchStatus.CONVERGED
    ]
    assert dead_index not in calls
    assert "site=left" not in rendered.axes[0].get_title()
    assert any(
        "NO_VALID_DATA" in text.get_text()
        for axis in rendered.axes
        for text in axis.texts
    )


def test_sparse_batch_hole_is_labelled_without_shifting_later_fit_rows(monkeypatch):
    data_figure, result = _sparse_curve_fit_figure()
    calls = []
    original = type(result).evaluate_batch

    def traced(self, index, coordinates):
        calls.append(index)
        return original(self, index, coordinates)

    monkeypatch.setattr(type(result), "evaluate_batch", traced)
    rendered = data_figure.render()
    present = []
    missing = 0
    layer = data_figure.evaluated.layers[0]
    for cell in layer.cells:
        for series in cell.series:
            addresses = (*cell.facet_address, *series.batch_address)
            by_axis = {item.axis_id: item.index for item in addresses}
            by_axis.update(
                {resolution.axis_id: resolution.index for resolution in layer.resolutions}
            )
            multi = tuple(by_axis[axis.axis_id] for axis in result.batch_axis_specs)
            try:
                present.append(result.batch_layout.storage_index(multi))
            except KeyError:
                missing += 1
    assert missing == 1
    assert calls == [
        index for index in present if result.statuses[index] is FitBatchStatus.CONVERGED
    ]
    assert any(
        "NOT_PRESENT" in text.get_text()
        for axis in rendered.axes
        for text in axis.texts
    )
