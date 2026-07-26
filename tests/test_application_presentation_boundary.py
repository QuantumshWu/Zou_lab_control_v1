"""Narrow ratchets for application composition versus frontend ownership."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

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
    PointLayout,
    StreamGenerationId,
    ValueSchema,
)
from zlc_frontend import (
    FrozenFigureSource,
    ViewIntent,
    build_frozen_data_figure,
    build_frozen_figure_document,
)


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


def _source() -> FrozenFigureSource:
    schema = DatasetSchema(
        _axis("repeat", REPEAT, 1),
        (_axis("scan", SCAN_POINT, 3),),
        PointLayout.rect_c((3,)),
        ValueSchema.scalar(np.dtype("<f8"), "count"),
    )
    block = DataBlock(
        BlockId("presentation-boundary"),
        DatasetRevision(0),
        np.asarray([[[1.0], [2.0], [3.0]]], dtype="<f8"),
        VALID,
        schema,
    )
    snapshot = OwnedSnapshot(
        block.ref(StreamGenerationId("presentation-boundary")),
        block,
    )
    return FrozenFigureSource("signal", schema, snapshot.ref, snapshot)


def test_frontend_builds_the_frozen_document_and_data_figure() -> None:
    source = _source()
    document = build_frozen_figure_document(source)
    figure = build_frozen_data_figure(source)

    assert document.layers[0].view.intent is ViewIntent.CURVE
    assert figure.document.layers[0].view.intent is ViewIntent.CURVE
    assert figure.datasets.resolve(figure.document.datasets[0].dataset_id) == (
        source.snapshot
    )


def test_notebook_facade_does_not_reimplement_frontend_figure_policy() -> None:
    path = ROOT / "Zou_lab_control/notebook/facade.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    forbidden_calls = {
        "DataFigure",
        "DatasetDescriptor",
        "DatasetId",
        "FigureDocument",
        "FigureLayer",
        "ResolvedDataset",
        "ResolvedDatasetMap",
        "suggest_fit_view",
        "suggest_view",
    }
    violations = []
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
        "application composition resolves artifacts; zlc_frontend owns Figure "
        "suggestion/document/evaluation:\n" + "\n".join(violations)
    )


def test_application_boundary_does_not_construct_domain_or_view_owners() -> None:
    """Composition may connect owner exports, never recreate their values."""

    paths = (
        ROOT / "Zou_lab_control/notebook/facade.py",
        ROOT / "Zou_lab_control/workbench/_composition.py",
    )
    forbidden_calls = {
        # Data/schema and Logic-node declaration owners.
        "AxisSpec",
        "AuthoringField",
        "DataBlock",
        "DatasetSchema",
        "DefaultOutputView",
        "LogicNodeDeclaration",
        "OutputPresentation",
        "PointLayout",
        "ValueSchema",
        # Frontend document/view/render owners.
        "ConsoleNodeSpec",
        "DatasetDescriptor",
        "FigureDocument",
        "FigureLayer",
        "PanelComposer",
        "SiteMapComposer",
        "SiteMapView",
        "ViewSpec",
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
                violations.append(
                    f"{path.relative_to(ROOT)}:{node.lineno} constructs {name}"
                )
    assert not violations, (
        "Zou_lab_control is public facade and explicit composition only; domain "
        "and presentation values must be constructed by their owner exports:\n"
        + "\n".join(violations)
    )
