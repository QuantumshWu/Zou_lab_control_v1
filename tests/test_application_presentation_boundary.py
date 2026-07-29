"""Narrow ratchets for application composition versus frontend ownership."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

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
from zlc_frontend.figure import ViewIntent
from zlc_frontend.frozen_figure import (
    FrozenFigureSource,
    build_frozen_data_figure,
    build_frozen_figure_document,
    resolve_frozen_figure_intent,
)
from zlc_neutral_atom.artifact_dataset_source import ArtifactDatasetSource


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
    intent = resolve_frozen_figure_intent(source)
    document = build_frozen_figure_document(source, intent)
    figure = build_frozen_data_figure(source, intent)

    assert document.layers[0].view.intent is ViewIntent.CURVE
    assert figure.document.layers[0].view.intent is ViewIntent.CURVE
    assert figure.datasets.resolve(figure.document.datasets[0].dataset_id) == (
        source.snapshot
    )


def test_artifact_dataset_source_binds_metadata_and_owned_snapshot() -> None:
    source = _source()
    metadata = ArtifactDatasetSource(source.schema, source.ref)
    assert metadata.snapshot is None
    with pytest.raises(RuntimeError, match="not materialised"):
        metadata.require_owned_snapshot()

    materialized = ArtifactDatasetSource(source.schema, source.ref, source.snapshot)
    assert materialized.require_owned_snapshot() is source.snapshot


def test_public_api_does_not_reimplement_frontend_figure_policy() -> None:
    paths = sorted((ROOT / "Zou_lab_control" / "api").glob("*.py"))
    assert paths
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
                    f"{path.relative_to(ROOT)}:{node.lineno} calls {name}"
                )
    assert not violations, (
        "application composition resolves artifacts; zlc_frontend owns Figure "
        "suggestion/document/evaluation:\n" + "\n".join(violations)
    )


def test_application_boundary_does_not_construct_domain_or_view_owners() -> None:
    """Composition may connect owner exports, never recreate their values."""

    paths = (
        *sorted((ROOT / "Zou_lab_control" / "api").glob("*.py")),
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
        "PointTable",
        "GridTopology",
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


def test_application_tree_delegates_artifact_dataset_interpretation() -> None:
    """Split helpers cannot hide artifact storage knowledge from the ratchet."""

    application_root = ROOT / "Zou_lab_control"
    forbidden_attributes = {
        "frame_source",
        "output_schema",
        "output_dataset_ref",
    }
    allowed_materialize_hosts = {
        ("api/_readout_core.py", "materialize_capture"),
    }
    violations = []
    for path in sorted(application_root.rglob("*.py")):
        relative = path.relative_to(application_root).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        parents = {
            child: parent
            for parent in ast.walk(tree)
            for child in ast.iter_child_nodes(parent)
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                if node.attr in forbidden_attributes:
                    violations.append(f"{relative}:{node.lineno} reads .{node.attr}")
                if (
                    node.attr in {"occupied", "counts"}
                    and isinstance(node.value, ast.Attribute)
                    and node.value.attr == "artifact"
                ):
                    violations.append(
                        f"{relative}:{node.lineno} interprets Occupancy artifact output"
                    )
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in {"materialize", "materialize_final"}
            ):
                continue
            owner = parents.get(node)
            while owner is not None and not isinstance(
                owner,
                (ast.FunctionDef, ast.AsyncFunctionDef),
            ):
                owner = parents.get(owner)
            owner_name = None if owner is None else owner.name
            if (relative, owner_name) not in allowed_materialize_hosts:
                violations.append(
                    f"{relative}:{node.lineno} directly calls {node.func.attr}"
                )
    assert not violations, (
        "Zou_lab_control may select typed owner adapters, but artifact fields and "
        "repository materialisation stay with those owners:\n" + "\n".join(violations)
    )
