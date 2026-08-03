"""Current public boundary between Experiment composition and ``zlc_plot``."""

from __future__ import annotations

import ast
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import Zou_lab_control.api as public_api
import Zou_lab_control.api.facade as facade_module
import Zou_lab_control.workbench as workbench_api
from zlc_data import (
    REPEAT,
    SCAN_POINT,
    VALID,
    AxisId,
    AxisSpec,
    BlockId,
    DataBlock,
    DatasetRevision,
    DatasetSchema,
    OwnedSnapshot,
    PointColumn,
    PointTable,
    StreamGenerationId,
    ValueSchema,
)
from zlc_plot import AxisRef, CurvePlot


ROOT = Path(__file__).resolve().parents[1]


def _axis(name: str, role, size: int) -> AxisSpec:
    return AxisSpec(
        AxisId(name),
        name,
        role,
        size,
        tuple(range(size)),
        None,
        None,
    )


def _snapshot() -> OwnedSnapshot:
    scan = _axis("scan", SCAN_POINT, 3)
    schema = DatasetSchema(
        _axis("repeat", REPEAT, 1),
        PointTable(
            3,
            (
                PointColumn(
                    scan.axis_id,
                    scan.name,
                    scan.role,
                    PointColumn.NUMERIC,
                    scan.coordinates,
                ),
            ),
        ),
        None,
        ValueSchema.scalar(np.dtype("<f8"), "count"),
    )
    block = DataBlock(
        BlockId("public-figure-boundary"),
        DatasetRevision(0),
        np.asarray([[[1.0], [2.0], [3.0]]], dtype="<f8"),
        VALID,
        schema,
    )
    return OwnedSnapshot(
        block.ref(StreamGenerationId("public-figure-boundary")),
        block,
    )


def _bare_experiment(monkeypatch, output_root: Path):
    experiment = object.__new__(public_api.Experiment)
    experiment._services = object()

    @contextmanager
    def service_guard(_services):
        yield SimpleNamespace(
            workspace_paths=SimpleNamespace(output_root=output_root),
        )

    monkeypatch.setattr(facade_module, "_service_guard", service_guard)
    return experiment


def test_lazy_workbench_entry_forwards_only_the_current_contract(
    monkeypatch,
    tmp_path: Path,
) -> None:
    snapshot = _snapshot()
    spec = CurvePlot(AxisRef.point_rows())
    calls = []
    sentinel = object()

    import zlc_workbench.data_figure.app as data_figure_app

    def open_current(current_snapshot, current_spec, **options):
        calls.append((current_snapshot, current_spec, options))
        return sentinel

    monkeypatch.setattr(data_figure_app, "open_figure_workbench", open_current)
    assert workbench_api.open_figure_workbench(
        snapshot,
        spec,
        output_root=tmp_path,
        size="medium",
        parameters={"title": "Current"},
        archive_path="current.zlcfig",
        metadata={"operator": "test"},
        open_fit=True,
    ) is sentinel
    assert calls == [
        (
            snapshot,
            spec,
            {
                "output_root": tmp_path,
                "size": "medium",
                "parameters": {"title": "Current"},
                "archive_path": "current.zlcfig",
                "metadata": {"operator": "test"},
                "open_fit": True,
            },
        )
    ]


def test_experiment_figure_gui_requires_explicit_snapshot_and_plot_spec(
    monkeypatch,
    tmp_path: Path,
) -> None:
    snapshot = _snapshot()
    spec = CurvePlot(AxisRef.point_rows())
    experiment = _bare_experiment(monkeypatch, tmp_path)
    calls = []
    sentinel = object()

    def open_current(current_snapshot, current_spec, **options):
        calls.append((current_snapshot, current_spec, options))
        return sentinel

    monkeypatch.setattr(workbench_api, "open_figure_workbench", open_current)
    assert experiment.figure_gui(
        snapshot,
        spec=spec,
        size="small",
        parameters={"title": "Explicit"},
        archive_path="explicit.zlcfig",
        metadata={"source": "test"},
        open_fit=True,
    ) is sentinel
    assert calls[0][0] is snapshot
    assert calls[0][1] is spec
    assert calls[0][2] == {
        "output_root": tmp_path,
        "size": "small",
        "parameters": {"title": "Explicit"},
        "archive_path": "explicit.zlcfig",
        "metadata": {"source": "test"},
        "open_fit": True,
    }

    with pytest.raises(ValueError, match="explicit PlotSpec"):
        experiment.figure_gui(snapshot)
    with pytest.raises(TypeError, match="OwnedSnapshot"):
        experiment.figure_gui(object(), spec=spec)
    with pytest.raises(TypeError, match="open_fit"):
        experiment.figure_gui(snapshot, spec=spec, open_fit=1)  # type: ignore[arg-type]


def test_experiment_figure_gui_routes_none_or_path_only_to_viewer(
    monkeypatch,
    tmp_path: Path,
) -> None:
    experiment = _bare_experiment(monkeypatch, tmp_path)
    calls = []
    sentinel = object()

    import zlc_workbench.figure_viewer.app as viewer_app

    def open_viewer(*, path, output_root):
        calls.append((path, output_root))
        return sentinel

    monkeypatch.setattr(viewer_app, "open_figure_viewer", open_viewer)
    assert experiment.figure_gui() is sentinel
    assert experiment.figure_gui("saved.zlcfig") is sentinel
    assert calls == [(None, tmp_path), ("saved.zlcfig", tmp_path)]
    with pytest.raises(ValueError, match="does not accept snapshot options"):
        experiment.figure_gui("saved.zlcfig", spec=CurvePlot(AxisRef.point_rows()))


def test_public_application_has_no_old_fit_or_figure_policy_owner() -> None:
    assert not (ROOT / "Zou_lab_control/api/_figure_projection.py").exists()
    for name in (
        "fit",
        "fit_gui",
        "save_fit",
        "load_fit",
        "figure",
        "figure_document",
        "_open_fit_capable_figure_gui",
    ):
        assert not hasattr(public_api.Experiment, name)
    for name in ("FitResultArtifactRef", "SavedFitResult"):
        assert not hasattr(public_api, name)

    paths = (
        *sorted((ROOT / "Zou_lab_control/api").glob("*.py")),
        ROOT / "Zou_lab_control/workbench/__init__.py",
        ROOT / "Zou_lab_control/workbench/_composition.py",
    )
    forbidden_calls = {
        "CurvePlot",
        "DataFigure",
        "FacetGridPlot",
        "FigureIntent",
        "FigureSource",
        "HistogramPlot",
        "ImagePlot",
        "PlotSession",
        "RollingPlot",
    }
    violations = []
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name):
                name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                name = node.func.attr
            else:
                continue
            if name in forbidden_calls:
                violations.append(f"{path.relative_to(ROOT)}:{node.lineno} calls {name}")
    assert not violations, (
        "public composition may forward PlotSpec, never recreate plot policy:\n"
        + "\n".join(violations)
    )
