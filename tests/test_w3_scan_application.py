"""W3b scan-owned snapshot and notebook figure product oracles."""

from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

import Zou_lab_control.notebook as zlc
from zlc_data import (
    READOUT_EVENT,
    REPEAT,
    ReductionMethod,
    ReductionSpec,
    SCAN_POINT,
    SITE,
    AxisId,
    AxisSpec,
    BlockId,
    ComponentValidity,
    DataBlock,
    DataTransformSpec,
    DatasetRevision,
    DatasetRevisionRef,
    DatasetSchema,
    OwnedSnapshot,
    PointLayout,
    Selection,
    StreamGenerationId,
    VALID,
    Valid,
    ValidityContract,
    ValidityPolicy,
    ValueSchema,
    commit_transform,
    materialize_transformed_snapshot,
)
from zlc_frontend.figure import ViewIntent
from zlc_neutral_atom.scan import ScanPointTable
from zlc_neutral_atom.scan.repository import ScanRepository
from zlc_pulse import FrozenScanTable, RepeatRegion, load_pulse_document


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


def _component_snapshot_case():
    repeat = _axis("repeat", REPEAT, 2, (0, 1))
    x = _axis("scan.x", SCAN_POINT, 2, (-1.0, 1.0))
    y = _axis("scan.y", SCAN_POINT, 2, (10.0, 20.0))
    event = _axis("readout.event", READOUT_EVENT, 1, ("image",))
    site = _axis("site", SITE, 3, ("left", "middle", "right"))
    layout = PointLayout.from_mapping((2, 2), ((0, 0), (1, 0), (1, 1)))
    raw_schema = DatasetSchema(
        repeat,
        (x, y),
        layout,
        ValueSchema(
            (event, site),
            ValidityContract.components(site.axis_id),
            np.dtype("<i2"),
        ),
    )
    values = np.arange(np.prod(raw_schema.physical_shape), dtype="<i2").reshape(
        raw_schema.physical_shape
    )
    valid = np.asarray(
        (
            ((True, False, True), (True, True, False), (False, True, True)),
            ((True, True, True), (False, False, True), (True, False, True)),
        )
    )
    raw_block = DataBlock(
        BlockId("raw-component-scan"),
        DatasetRevision(7),
        values,
        ComponentValidity((site.axis_id,), valid),
        raw_schema,
    )
    source = OwnedSnapshot(
        raw_block.ref(StreamGenerationId("component-generation")),
        raw_block,
    )
    transform = commit_transform(
        raw_schema,
        DataTransformSpec((Selection.index(event.axis_id, 0),)),
    )
    output_schema = DatasetSchema(
        repeat,
        (x, y),
        layout,
        ValueSchema(
            (site,),
            ValidityContract.components(site.axis_id),
            np.dtype("<i2"),
        ),
    )
    output_ref = DatasetRevisionRef(
        BlockId("derived-component-scan"),
        source.ref.stream_generation,
        output_schema.fingerprint,
        source.ref.revision,
    )
    return source, transform, output_schema, output_ref, values, valid


def _sparse_scan_document():
    document = load_pulse_document(ROOT / "pulses" / "mot_field_template.json")
    columns = tuple(item.parameter_id for item in document.scan_parameters)
    return replace(
        document,
        scan_table=FrozenScanTable(
            columns,
            ((0, 0, 0), (1, 0, 1), (0, 1, 1)),
        ),
        repeat=RepeatRegion(
            document.periods[0].period_id,
            document.periods[-1].period_id,
            2,
        ),
    )


def test_transform_owner_freezes_once_and_preserves_component_validity(monkeypatch):
    source, transform, schema, output_ref, values, valid = _component_snapshot_case()
    output = materialize_transformed_snapshot(
        source,
        transform,
        output_ref=output_ref,
        output_schema=schema,
        memory_limit_bytes=64 << 20,
    )
    assert output.ref == output_ref
    assert output.block.values.shape == (2, 3, 3)
    np.testing.assert_array_equal(output.block.values, values[:, :, 0, :])
    assert isinstance(output.block.validity, ComponentValidity)
    assert output.block.validity.axis_ids == (AxisId("site"),)
    np.testing.assert_array_equal(output.block.validity.mask, valid)
    assert not output.block.values.flags.writeable

    import zlc_data.transform as transform_module

    executed = False

    def forbidden_execute(*_args, **_kwargs):
        nonlocal executed
        executed = True
        raise AssertionError("transform executed below its admitted peak")

    monkeypatch.setattr(transform_module, "_execute_transform", forbidden_execute)
    with pytest.raises(MemoryError, match="transformed snapshot peak"):
        materialize_transformed_snapshot(
            source,
            transform,
            output_ref=output_ref,
            output_schema=schema,
            memory_limit_bytes=1,
        )
    assert not executed


def test_bounded_snapshot_rejects_cell_reduction():
    repeat = _axis("repeat", REPEAT, 1, (0,))
    point = _axis("scan.point", SCAN_POINT, 2, (0, 1))
    source_schema = DatasetSchema(
        repeat,
        (point,),
        PointLayout.rect_c((2,)),
        ValueSchema((), ValidityContract.value(), np.dtype("<i2")),
    )
    block = DataBlock(
        BlockId("cell-reduction-source"),
        DatasetRevision(0),
        np.asarray(((1, 2),), dtype="<i2"),
        VALID,
        source_schema,
    )
    source = OwnedSnapshot(
        block.ref(StreamGenerationId("cell-reduction-generation")), block
    )
    cell_reduction = commit_transform(
        source_schema,
        DataTransformSpec(
            (ReductionSpec((point.axis_id,), ReductionMethod.MEAN),)
        ),
    )
    output_schema = DatasetSchema(
        repeat,
        (),
        PointLayout.rect_c(()),
        ValueSchema((), ValidityContract.value(), np.dtype("<f8")),
    )
    output_ref = DatasetRevisionRef(
        BlockId("cell-reduction-output"),
        source.ref.stream_generation,
        output_schema.fingerprint,
        source.ref.revision,
    )
    with pytest.raises(ValueError, match="do not reduce repeat/point axes"):
        materialize_transformed_snapshot(
            source,
            cell_reduction,
            output_ref=output_ref,
            output_schema=output_schema,
            memory_limit_bytes=64 << 20,
        )


def test_bounded_snapshot_reduces_only_the_named_trailing_axis():
    source, _transform, _schema, _output_ref, _values, _valid = (
        _component_snapshot_case()
    )
    source_schema = source.block.schema
    transform = commit_transform(
        source_schema,
        DataTransformSpec(
            (
                Selection.index(AxisId("readout.event"), 0),
                ReductionSpec(
                    (AxisId("site"),),
                    ReductionMethod.MEAN,
                    validity_policy=ValidityPolicy.OMIT_INVALID,
                ),
            )
        ),
    )
    output_schema = DatasetSchema(
        source_schema.repeat_axis,
        source_schema.point_axes,
        source_schema.point_layout,
        ValueSchema((), ValidityContract.value(), np.dtype("<f8")),
    )
    output_ref = DatasetRevisionRef(
        BlockId("derived-scalar-scan"),
        source.ref.stream_generation,
        output_schema.fingerprint,
        source.ref.revision,
    )
    output = materialize_transformed_snapshot(
        source,
        transform,
        output_ref=output_ref,
        output_schema=output_schema,
        memory_limit_bytes=64 << 20,
    )
    assert output.block.values.shape == (2, 3)
    np.testing.assert_allclose(
        output.block.values,
        ((1.0, 3.5, 7.5), (10.0, 14.0, 16.0)),
    )
    assert isinstance(output.block.validity, Valid)


def test_public_sparse_scan_reopens_with_stable_identity_and_data_figure(
    tmp_path,
    monkeypatch,
):
    workspace = tmp_path / "workspace"
    document = _sparse_scan_document()
    expected_points = ScanPointTable.from_pulse_document(document)

    with zlc.connect("virtual", repository=workspace) as exp:
        scan_ref = exp.scan(
            exp.readout.scan_request(document, timeout_seconds=15.0)
        )
        with pytest.raises(MemoryError):
            exp.readout.materialize_scan(scan_ref, memory_limit_bytes=1)
        data = exp.readout.materialize_scan(scan_ref)
        artifact = exp.readout.load_scan(scan_ref)
        raw = exp.readout.load_capture(artifact.source_capture_ref)
        raw_block = raw.frame_source.materialize(memory_limit_bytes=512 << 20)
        raw_ref = raw_block.ref(raw.provenance.generation)

        assert data.artifact_ref == artifact.ref
        assert data.source_dataset_ref == raw_ref
        assert data.snapshot.ref != raw_ref
        assert data.snapshot.ref.block_id.value == (
            f"scan-output-{scan_ref.manifest_digest}"
        )
        assert data.values.shape == (2, 3, 96, 128)
        assert data.schema.repeat_axis.size == 2
        assert data.schema.point_axes == expected_points.point_axes
        assert data.schema.point_layout == expected_points.point_layout
        assert data.schema.cell_schema.data_axes == raw_block.schema.cell_schema.data_axes
        assert any(axis.role == READOUT_EVENT for axis in raw_block.schema.point_axes)
        assert all(axis.role != READOUT_EVENT for axis in data.schema.point_axes)
        np.testing.assert_array_equal(data.values, raw_block.values)
        assert exp.readout.materialize_scan(scan_ref).snapshot.ref == data.snapshot.ref

        def forbidden_materialize(*_args, **_kwargs):
            raise AssertionError("figure_document must remain metadata-only")

        with monkeypatch.context() as patch:
            patch.setattr(ScanRepository, "materialize", forbidden_materialize)
            figure_document = exp.figure_document(scan_ref)
        assert figure_document.datasets[0].schema_fingerprint == data.schema.fingerprint
        assert figure_document.layers[0].view.intent is ViewIntent.IMAGE

        figure = exp.figure(scan_ref)
        assert figure.document.datasets == figure_document.datasets
        assert figure.to_png_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        with pytest.raises(TypeError, match="CaptureArtifactRef"):
            exp.fit(scan_ref, model="gaussian_offset")

    digest = hashlib.sha256(data.values.tobytes()).hexdigest()
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import hashlib,sys; import Zou_lab_control.notebook as zlc; "
                "ref=zlc.ScanArtifactRef(sys.argv[2],sys.argv[3]); "
                "exp=zlc.connect('virtual',repository=sys.argv[1]); "
                "data=exp.readout.materialize_scan(ref); "
                "assert data.snapshot.ref.block_id.value==sys.argv[4]; "
                "assert data.snapshot.ref.schema_fingerprint==sys.argv[5]; "
                "assert data.snapshot.ref.stream_generation.value==sys.argv[6]; "
                "assert str(data.snapshot.ref.revision.value)==sys.argv[7]; "
                "assert hashlib.sha256(data.values.tobytes()).hexdigest()==sys.argv[8]; "
                "exp.close()"
            ),
            str(workspace),
            scan_ref.repository_id,
            scan_ref.manifest_digest,
            data.snapshot.ref.block_id.value,
            data.snapshot.ref.schema_fingerprint,
            data.snapshot.ref.stream_generation.value,
            str(data.snapshot.ref.revision.value),
            digest,
        ],
        cwd=ROOT,
        check=True,
        timeout=30,
    )
