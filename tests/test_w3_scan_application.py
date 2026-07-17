"""W3b scan-owned snapshot and notebook figure product oracles."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import copy
import gc
import hashlib
from pathlib import Path
import subprocess
import sys
import threading
import time
import weakref

import numpy as np
import pytest

import Zou_lab_control.notebook as zlc
from Zou_lab_control.notebook.facade import _prepare_occupancy_scan_for_workbench
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
from zlc_frontend.matplotlib_render import SinglePanelAggRenderer
from zlc_frontend.figure import (
    AxisViewRole,
    FigureEvaluationPolicy,
    FigureEvaluator,
    ResolvedDataset,
    ResolvedDatasetMap,
    ViewIntent,
)
from zlc_neutral_atom.runtime.pipeline import ExactDatasetPreviewSpec
from zlc_neutral_atom.runtime.run import RunFailed
from zlc_neutral_atom.scan import (
    AutonomousScanExecution,
    AutonomousScanSlotProgram,
    ScanOutputContract,
    ScanPointTable,
)
from zlc_neutral_atom.scan.repository import ScanRepository
from zlc_neutral_atom.readout.calibration_reference import (
    calibration_artifact_input_ref,
)
from zlc_neutral_atom.readout.sitemap import load_packaged_sitemap_pulse
from zlc_pulse import (
    FrozenScanTable,
    RepeatRegion,
    ScanParameter,
    load_pulse_document,
)
from zlc_workbench.progressive_scan import (
    ExactDatasetLiveSlot,
    build_occupancy_progressive_spec,
)


ROOT = Path(__file__).resolve().parents[1]


class _CountingExactDatasetLiveSlot(ExactDatasetLiveSlot):
    def __init__(self, spec: ExactDatasetPreviewSpec) -> None:
        super().__init__(spec)
        self.source_terminal_calls = 0

    def source_terminal(self) -> None:
        self.source_terminal_calls += 1
        super().source_terminal()


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


def _occupancy_scan_document():
    """Turn the proven sitemap readout event into a two-point SCAN_SLOT."""

    document = load_packaged_sitemap_pulse()
    camera_port = next(
        port for port in document.target.ports if port.label == "emCCD"
    )
    assert len(camera_port.lanes) == 1
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
        period
        for period in periods
        if period.period_id == scanned_api.field.period_id
    )
    scan_parameter = ScanParameter(
        "reference_settle",
        scanned_api.field,
        "reference settle",
        scanned_api.unit,
    )
    start = scanned_period.duration
    step = 1 if isinstance(start, int) else 1e-6
    return replace(
        document,
        name="occupancy-scan-slot",
        periods=tuple(periods),
        api_parameters=tuple(
            parameter
            for parameter in document.api_parameters
            if parameter is not scanned_api
        ),
        scan_parameters=(scan_parameter,),
        scan_table=FrozenScanTable(
            (scan_parameter.parameter_id,),
            ((start,), (start + step,)),
        ),
        repeat=RepeatRegion(
            periods[0].period_id,
            periods[-1].period_id,
            2,
        ),
    )


def _fixed_api_values(document):
    return {
        parameter.parameter_id: document.field_value(parameter.field)[0]
        for parameter in document.api_parameters
    }


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


def test_progressive_renderer_reuses_artists_and_updates_component_validity(monkeypatch):
    source, transform, schema, output_ref, _values, valid = (
        _component_snapshot_case()
    )
    output = materialize_transformed_snapshot(
        source,
        transform,
        output_ref=output_ref,
        output_schema=schema,
        memory_limit_bytes=64 << 20,
    )
    contract = ScanOutputContract(transform, schema)
    progressive = build_occupancy_progressive_spec(
        source.block.schema,
        contract,
        identity="renderer-update",
    )

    def revision(number, mask):
        block = DataBlock(
            output.block.block_id,
            DatasetRevision(number),
            output.block.values,
            ComponentValidity((AxisId("site"),), mask),
            output.block.schema,
        )
        return OwnedSnapshot(block.ref(output.ref.stream_generation), block)

    partial_valid = valid.copy()
    partial_valid[1, :, :] = False
    snapshots = (
        revision(1, partial_valid),
        revision(2, valid),
    )
    evaluator = FigureEvaluator(
        FigureEvaluationPolicy(max_live_nbytes=progressive.evaluation_peak_bytes)
    )

    def evaluate(snapshot):
        return evaluator.evaluate(
            progressive.document,
            ResolvedDatasetMap(
                (ResolvedDataset(progressive.dataset_id, snapshot),)
            ),
        )

    renderer = SinglePanelAggRenderer(
        progressive.document,
        width=360,
        height=240,
    )
    first = renderer.render(evaluate(snapshots[0]))
    figure_id = id(renderer._figure)
    axis_id = id(renderer._axis)
    line_ids = tuple(map(id, renderer._artists))
    first_legend = tuple(
        text.get_text() for text in renderer._axis.get_legend().get_texts()
    )

    second = renderer.render(evaluate(snapshots[1]))
    assert id(renderer._figure) == figure_id
    assert id(renderer._axis) == axis_id
    assert tuple(map(id, renderer._artists)) == line_ids
    second_legend = tuple(
        text.get_text() for text in renderer._axis.get_legend().get_texts()
    )
    assert second_legend != first_legend
    assert second.pixels != first.pixels

    figure_ref = weakref.ref(renderer._figure)
    canvas_ref = weakref.ref(renderer._figure.canvas)
    collection_was_enabled = gc.isenabled()
    gc.disable()
    try:
        renderer.close()
        assert figure_ref() is None
        assert canvas_ref() is None
        renderer.close()
    finally:
        if collection_was_enabled:
            gc.enable()
    with pytest.raises(RuntimeError, match="closed"):
        renderer.render(evaluate(snapshots[1]))

    from matplotlib.figure import Figure

    partial_canvases = []

    def failed_subplots(self, *_args, **_kwargs):
        partial_canvases.append(weakref.ref(self.canvas))
        raise RuntimeError("injected renderer construction failure")

    collection_was_enabled = gc.isenabled()
    gc.disable()
    try:
        with monkeypatch.context() as failure_patch:
            failure_patch.setattr(Figure, "subplots", failed_subplots)
            with pytest.raises(
                RuntimeError,
                match="injected renderer construction failure",
            ):
                SinglePanelAggRenderer(
                    progressive.document,
                    width=360,
                    height=240,
                )
        assert partial_canvases and all(ref() is None for ref in partial_canvases)
    finally:
        if collection_was_enabled:
            gc.enable()


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
        request = exp.readout.scan_request(document, timeout_seconds=15.0)

        def forbidden_stage(*_args, **_kwargs):
            raise AssertionError("inspect_scan must not stage repository blobs")

        with monkeypatch.context() as patch:
            patch.setattr(
                ScanRepository,
                "_stage_static_lineage",
                forbidden_stage,
            )
            descriptor = exp.inspect_scan(request)
        assert descriptor.expected_frames == 6
        with pytest.raises(MemoryError, match="scan final data-plane peak"):
            exp.scan(
                exp.readout.scan_request(
                    document,
                    memory_limit_bytes=1,
                    timeout_seconds=15.0,
                )
            )
        import zlc_neutral_atom.scan.application as scan_application

        base_compiled = False

        def forbidden_base_compile(*_args, **_kwargs):
            nonlocal base_compiled
            base_compiled = True
            raise AssertionError("hardware plan compiled below static-lineage admission")

        with monkeypatch.context() as patch:
            patch.setattr(
                scan_application,
                "compile_triggered_pipeline",
                forbidden_base_compile,
            )
            with pytest.raises(MemoryError, match="scan static-lineage peak"):
                exp.scan(
                    exp.readout.scan_request(
                        document,
                        memory_limit_bytes=1 << 20,
                        timeout_seconds=15.0,
                    )
                )
        assert not base_compiled
        scan_ref = exp.scan(
            exp.readout.scan_request(document, timeout_seconds=15.0)
        )
        with pytest.raises(MemoryError):
            exp.readout.materialize_scan(scan_ref, memory_limit_bytes=1)
        data = exp.readout.materialize_scan(scan_ref)
        artifact = exp.readout.load_scan(scan_ref)

        assert data.artifact_ref == artifact.ref
        assert data.source_dataset_ref == artifact.source_dataset_ref
        assert data.snapshot.ref == artifact.output_dataset_ref
        assert data.snapshot.ref != artifact.source_dataset_ref
        assert data.snapshot.ref.block_id.value.startswith("scan-output-")
        assert data.values.shape == (2, 3, 96, 128)
        assert data.schema.repeat_axis.size == 2
        assert data.schema.point_axes == expected_points.point_axes
        assert data.schema.point_layout == expected_points.point_layout
        assert (
            data.schema.cell_schema.data_axes
            == artifact.source_dataset_schema.cell_schema.data_axes
        )
        assert any(
            axis.role == READOUT_EVENT
            for axis in artifact.source_dataset_schema.point_axes
        )
        assert all(axis.role != READOUT_EVENT for axis in data.schema.point_axes)
        assert artifact.provenance.derivation is None
        assert isinstance(artifact.execution, AutonomousScanExecution)
        assert artifact.execution.evidence.expected_trigger_count == 6
        camera = artifact.execution.camera
        assert camera.event_count == 6
        assert camera.terminal.session_id
        assert camera.terminal.produced_count == 6
        assert camera.terminal.drained_count == 6
        assert camera.arm_spec.digest == camera.terminal.capture_spec_fingerprint
        assert camera.capability.fingerprint == camera.terminal.capability_fingerprint
        assert (
            camera.source_schema_fingerprint
            == artifact.source_dataset_schema.fingerprint
        )
        camera.validate_dataset_provenance(artifact.provenance)

        # Exercise the durable reload boundary, not just the live PipelineResult:
        # a well-typed forged aggregate digest must still be rejected against
        # the independently persisted DatasetSealProvenance.
        import zlc_neutral_atom.scan.repository as scan_repository

        decode_index = scan_repository._decode_metadata_index

        def forged_camera_metadata(payload):
            index = decode_index(payload)
            execution_tree = copy.deepcopy(index.execution_tree)
            execution_tree["camera"]["terminal"][
                "ordered_metadata_digest"
            ] = "0" * 64
            return replace(index, execution_tree=execution_tree)

        with monkeypatch.context() as patch:
            patch.setattr(
                scan_repository,
                "_decode_metadata_index",
                forged_camera_metadata,
            )
            with pytest.raises(
                ValueError,
                match="raw dataset provenance differs from camera aggregate evidence",
            ):
                exp.readout.load_scan(scan_ref)
        assert np.all(np.isfinite(data.values))
        assert not data.values.flags.writeable
        assert exp.readout.materialize_scan(scan_ref).snapshot.ref == data.snapshot.ref

        def forbidden_heavy_read(*_args, **_kwargs):
            raise AssertionError("metadata-only inspection decoded heavy lineage/data")

        with monkeypatch.context() as patch:
            patch.setattr(ScanRepository, "materialize", forbidden_heavy_read)
            patch.setattr(
                "zlc_neutral_atom.scan.repository.decode_compiled_pulse_artifact",
                forbidden_heavy_read,
            )
            patch.setattr(
                "zlc_neutral_atom.scan.repository._decode_program",
                forbidden_heavy_read,
            )
            figure_document = exp.figure_document(scan_ref)
        assert figure_document.datasets[0].schema_fingerprint == data.schema.fingerprint
        assert figure_document.layers[0].view.intent is ViewIntent.IMAGE

        figure = exp.figure(scan_ref)
        assert figure.document.datasets == figure_document.datasets
        with pytest.raises(MemoryError, match="figure render peak"):
            figure.to_png_bytes(memory_limit_bytes=1)
        assert figure.to_png_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        with pytest.raises(TypeError, match="CaptureArtifactRef"):
            exp.fit(scan_ref, model="gaussian_offset")

        _assert_public_occupancy_scan(exp, monkeypatch)
        _assert_scan_window(exp, document, monkeypatch)

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
                "artifact=exp.readout.load_scan(ref); "
                "assert artifact.execution.camera.terminal.produced_count==6; "
                "assert artifact.execution.camera.terminal.drained_count==6; "
                "artifact.execution.camera.validate_dataset_provenance(artifact.provenance); "
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


def _assert_public_occupancy_scan(exp, monkeypatch):
    document = _occupancy_scan_document()
    points = ScanPointTable.from_pulse_document(document)
    values = _fixed_api_values(document)
    original_parameters = document.api_parameters
    original_table = document.scan_table
    with pytest.raises(ValueError, match="missing"):
        exp.readout.scan_request(document)
    with pytest.raises(ValueError, match="missing=.*extra=.*not-an-api"):
        exp.readout.scan_request(document, api_values={"not-an-api": 1})
    direct_request = exp.readout.scan_request(document, api_values=values)
    assert isinstance(direct_request.program, AutonomousScanSlotProgram)
    assert direct_request.program.document == document
    assert direct_request.program.execution_document.api_parameters == ()
    assert (
        direct_request.program.execution_document.scan_parameters
        == document.scan_parameters
    )
    assert direct_request.program.execution_document.scan_table == original_table
    assert document.api_parameters == original_parameters
    with pytest.raises(ValueError, match="missing=.*extra=.*da_x"):
        exp.readout.scan_request(
            _sparse_scan_document(),
            api_values={"da_x": 0},
        )
    calibration_ref = exp.readout.sitemap(frames=6)
    request = exp.readout.occupancy_scan_request(
        document,
        calibration_ref=calibration_ref,
        api_values=values,
        timeout_seconds=20.0,
    )
    assert isinstance(request.program, AutonomousScanSlotProgram)
    assert request.program.execution_document.api_parameters == ()
    guarded = _prepare_occupancy_scan_for_workbench(exp, request)

    @contextmanager
    def closed_guard(_token):
        raise RuntimeError("Experiment is closed")
        yield

    with monkeypatch.context() as patch:
        patch.setattr(
            "Zou_lab_control.notebook.facade._service_guard",
            closed_guard,
        )
        with pytest.raises(RuntimeError, match="Experiment is closed"):
            guarded.start()

    rejected = _prepare_occupancy_scan_for_workbench(exp, request)
    undersized = ExactDatasetLiveSlot(
        ExactDatasetPreviewSpec(rejected.source_schema.fingerprint, 1)
    )
    with pytest.raises(MemoryError, match="frozen source snapshot"):
        rejected.start(undersized)
    assert undersized.terminal
    assert "frozen source snapshot" in (undersized.failure or "")
    second_start = ExactDatasetLiveSlot(undersized.spec)
    with pytest.raises(RuntimeError, match="one-shot"):
        rejected.start(second_start)
    assert second_start.terminal
    assert "one-shot" in (second_start.failure or "")

    import zlc_neutral_atom.timing.occupancy as timing_occupancy

    failed_prepared = _prepare_occupancy_scan_for_workbench(exp, request)
    failed_progressive = build_occupancy_progressive_spec(
        failed_prepared.source_schema,
        failed_prepared.output_contract,
        identity="w3-post-safety-failure",
    )
    failed_slot = ExactDatasetLiveSlot(failed_progressive.preview_spec)

    def reject_post_safety(*_args, **_kwargs):
        raise RuntimeError("post-safety occupancy finalization rejected")

    with monkeypatch.context() as patch:
        patch.setattr(
            timing_occupancy,
            "finalize_occupancy_result",
            reject_post_safety,
        )
        failed_handle = failed_prepared.start(failed_slot)
        with pytest.raises(RunFailed, match="post-safety occupancy finalization"):
            failed_handle.result(timeout=30.0)
    assert failed_slot.terminal
    assert "post-safety occupancy finalization" in (failed_slot.failure or "")

    prepared = _prepare_occupancy_scan_for_workbench(exp, request)
    progressive = build_occupancy_progressive_spec(
        prepared.source_schema,
        prepared.output_contract,
        identity="w3-occupancy",
    )
    site_axis = prepared.output_contract.output_dataset_schema.cell_schema.data_axes[0]
    site_binding = next(
        binding
        for binding in progressive.document.layers[0].view.axis_bindings
        if binding.axis_id == site_axis.axis_id
    )
    if 1 < site_axis.size <= 32:
        assert site_binding.role is AxisViewRole.BATCH
        assert f"batch/{site_axis.size}" in progressive.projection_summary
    else:
        assert site_binding.role is AxisViewRole.SELECTED
        assert f"{site_axis.name}={site_axis.coordinate_at(0)}" in (
            progressive.projection_summary
        )
    slot = _CountingExactDatasetLiveSlot(progressive.preview_spec)
    handle = prepared.start(slot)
    scan_ref = handle.result(timeout=30.0)
    assert slot.terminal
    assert slot.failure is None
    assert slot.source_terminal_calls == 1
    provisional = slot.wait_and_freeze(DatasetRevision(0), timeout=0)
    assert provisional is not None
    _run_id, _causation, preview = provisional
    assert preview.coverage.complete
    assert preview.block.schema.fingerprint == prepared.source_schema.fingerprint
    slot.close()
    artifact = exp.readout.load_scan(scan_ref)
    data = exp.readout.materialize_scan(scan_ref)

    assert data.schema.repeat_axis.size == 2
    assert data.schema.point_axes == points.point_axes
    assert data.schema.point_layout == points.point_layout
    assert data.values.shape[:2] == (2, 2)
    assert len(data.values.shape) == 3
    assert len(data.schema.cell_schema.data_axes) == 1
    site_axis = data.schema.cell_schema.data_axes[0]
    assert site_axis.role == SITE
    assert data.values.shape[2] == site_axis.size
    assert isinstance(data.validity, ComponentValidity)
    assert data.validity.axis_ids == (site_axis.axis_id,)
    assert data.validity.mask.shape == data.values.shape

    derivation = artifact.provenance.derivation
    assert derivation is not None
    assert len(derivation.stages) == 1
    assert calibration_artifact_input_ref(calibration_ref) in (
        derivation.artifact_inputs
    )
    assert artifact.source_dataset_ref == data.source_dataset_ref
    assert artifact.output_dataset_ref == data.snapshot.ref
    assert any(
        axis.role == READOUT_EVENT
        for axis in artifact.source_dataset_schema.point_axes
    )
    assert all(axis.role != READOUT_EVENT for axis in data.schema.point_axes)
    assert isinstance(artifact.execution, AutonomousScanExecution)
    assert artifact.execution.evidence.expected_trigger_count == 4
    _assert_occupancy_scan_window(exp, request, monkeypatch)


def _assert_occupancy_scan_window(exp, request, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PyQt5 import QtWidgets
    from zlc_frontend.qt_widgets import QtImageBoard
    import zlc_workbench.progressive_scan as progressive_scan

    owner_thread = threading.get_ident()
    renderer_construct_threads = []
    raster_threads = []
    renderer_ids = []
    renderer_close_threads = []
    present_threads = []
    original_init = progressive_scan.SinglePanelAggRenderer.__init__
    original_render = progressive_scan.SinglePanelAggRenderer.render
    original_close = progressive_scan.SinglePanelAggRenderer.close
    original_present = QtImageBoard.present

    def record_init(renderer, *args, **kwargs):
        renderer_construct_threads.append(threading.get_ident())
        return original_init(renderer, *args, **kwargs)

    def record_render(renderer, evaluated):
        raster_threads.append(threading.get_ident())
        renderer_ids.append(id(renderer))
        return original_render(renderer, evaluated)

    def record_close(renderer):
        renderer_close_threads.append(threading.get_ident())
        return original_close(renderer)

    def record_present(board, frame):
        present_threads.append(threading.get_ident())
        return original_present(board, frame)

    with monkeypatch.context() as patch:
        patch.setattr(
            progressive_scan.SinglePanelAggRenderer,
            "__init__",
            record_init,
        )
        patch.setattr(
            progressive_scan.SinglePanelAggRenderer,
            "render",
            record_render,
        )
        patch.setattr(
            progressive_scan.SinglePanelAggRenderer,
            "close",
            record_close,
        )
        patch.setattr(QtImageBoard, "present", record_present)
        window = exp.scan_gui(request)
        application = QtWidgets.QApplication.instance()
        assert application is not None
        assert "PROVISIONAL OCCUPANCY" in window.findChild(
            QtWidgets.QLabel,
            "scanMode",
        ).text()
        start = window.findChild(QtWidgets.QPushButton, "startScanButton")
        assert start is not None and start.isEnabled()
        start.click()

        deadline = time.monotonic() + 20.0
        while (
            (window.final_reference is None or not raster_threads or not present_threads)
            and time.monotonic() < deadline
        ):
            application.processEvents()
            time.sleep(0.01)
        assert window.final_reference is not None
        assert raster_threads
        assert renderer_construct_threads
        assert len(set(renderer_ids)) == 1
        assert renderer_close_threads
        assert present_threads
        assert all(thread != owner_thread for thread in raster_threads)
        assert set(renderer_construct_threads) == set(raster_threads)
        assert set(renderer_close_threads) == set(raster_threads)
        assert set(present_threads) == {owner_thread}

        window.close()
        deadline = time.monotonic() + 5.0
        while window.isVisible() and time.monotonic() < deadline:
            application.processEvents()
            time.sleep(0.01)
        assert not window.isVisible()
        assert window.worker_idle
        assert window not in getattr(application, "_zlc_retained_windows", ())


def _assert_scan_window(exp, document, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PyQt5 import QtWidgets
    from zlc_frontend.qt_widgets import QtImageBoard

    request = exp.readout.scan_request(document, timeout_seconds=15.0)
    window = exp.scan_gui(request)
    application = QtWidgets.QApplication.instance()
    assert application is not None
    assert application.primaryScreen().availableGeometry().contains(
        window.frameGeometry()
    )
    assert "FINAL-ONLY" in window.findChild(
        QtWidgets.QLabel,
        "scanMode",
    ).text()
    start = window.findChild(QtWidgets.QPushButton, "startScanButton")
    assert start is not None and start.isEnabled()
    start.click()

    raster = window.findChild(QtImageBoard, "scanRaster")
    assert raster is not None
    deadline = time.monotonic() + 15.0
    while (
        (
            window.final_reference is None
            or not raster.has_front
        )
        and time.monotonic() < deadline
    ):
        application.processEvents()
        time.sleep(0.01)
    assert window.final_reference is not None
    assert raster.has_front
    assert application.primaryScreen().availableGeometry().contains(
        window.frameGeometry()
    )

    assert start.isEnabled()
    start.click()
    assert not raster.has_front

    window.close()
    deadline = time.monotonic() + 5.0
    while window.isVisible() and time.monotonic() < deadline:
        application.processEvents()
        time.sleep(0.01)
    assert not window.isVisible()
    assert window.worker_idle
    assert window not in getattr(application, "_zlc_retained_windows", ())
