"""Exact camera datasets become durable current-schema CaptureArtifacts."""

from __future__ import annotations

import math
import threading
import time
import tracemalloc
from dataclasses import dataclass, replace

import numpy as np
import pytest

from Zou_lab_control.neutral_atom.devices.registry import DeviceSet
from Zou_lab_control.neutral_atom.devices.virtual import VirtualCamera, VirtualTrapArray
from zlc_data import (
    INVALID,
    VALID,
    AxisId,
    AxisSpec,
    BlockId,
    CellValidity,
    ComponentValidity,
    DataBlock,
    DatasetRevision,
    DatasetSchema,
    PointLayout,
    REPEAT,
    SCAN_POINT,
    SPATIAL_X,
    SPATIAL_Y,
    ValidityContract,
    ValueSchema,
)
from zlc_neutral_atom.acquisition import (
    CAMERA_CAPTURE_SPEC_OWNER_FINGERPRINT,
    CameraAcquisitionMode,
    CameraCaptureSpec,
    CameraFrameMetadata,
    freeze_camera_capture_spec,
)
from zlc_neutral_atom.artifacts import (
    CaptureArtifactRef,
    CaptureRepository,
    CaptureRepositoryResourcePolicy,
    compile_capture_artifact_pipeline,
)
from zlc_neutral_atom.capture_reference import (
    capture_artifact_input_ref,
    capture_artifact_ref_from_tree,
    decode_capture_artifact_ref,
    encode_capture_artifact_ref,
)
from zlc_neutral_atom.runtime import (
    CancellationToken,
    compile_pipeline,
    DatasetCellAddress,
    DatasetMaterializerSpec,
    MinimalPipelineSpec,
    PipelineMemoryProfile,
    PostSafetyContext,
    RunFailed,
    RunId,
)
from zlc_neutral_atom.runtime.capture import camera_physical_facts_from_tree
from zlc_neutral_atom.runtime.capture import FrozenCaptureSpec
from zlc_neutral_atom.readout.codec import (
    camera_capture_descriptor_to_tree,
    readout_binding_key_to_tree,
)
from zlc_storage import (
    ContentCorruptionError,
    content_ref_from_tree,
    content_ref_to_tree,
    decode,
    encode,
)
from zlc_workbench.camera_capture import CameraCaptureBindingRequest
from zlc_workbench.legacy_neutral_atom import LegacyNeutralAtomRuntime


def _axis(name, role, size):
    return AxisSpec(AxisId(name), name, role, size, tuple(range(size)))


def _runtime_and_spec(
    *,
    repeat_size=1,
    point_size=2,
    point_layout=None,
    source_cell_schedule=None,
):
    camera = VirtualCamera(
        VirtualTrapArray(grid_shape=(2, 2), image_shape=(6, 8), seed=4),
        exposure=1e-3,
    )
    runtime = LegacyNeutralAtomRuntime(
        DeviceSet(
            {"readout": camera},
            {"readout": {"type": "VirtualCamera", "params": {}}},
        )
    )
    point_layout = point_layout or PointLayout.rect_c((point_size,))
    source_cell_schedule = source_cell_schedule or tuple(
        DatasetCellAddress(repeat, point)
        for repeat in range(repeat_size)
        for point in range(point_layout.storage_size)
    )
    measurement = runtime.bind_camera_measurement(
        CameraCaptureBindingRequest(
            "readout",
            _axis("repeat", REPEAT, repeat_size),
            (_axis("point", SCAN_POINT, point_size),),
            point_layout,
            source_cell_schedule,
            CameraAcquisitionMode.EXTERNAL_TRIGGERED,
            0,
            4 << 20,
        )
    )
    spec = MinimalPipelineSpec(
        "persist exact camera capture",
        measurement,
        DatasetMaterializerSpec(
            BlockId("capture-artifact-test"),
            PipelineMemoryProfile(8 << 20),
        ),
    )
    return camera, runtime, spec


def _deliver_when_armed(camera, images):
    failure = []

    def source():
        try:
            deadline = time.monotonic() + 2.0
            state = camera._recent_state()
            with state["cond"]:
                while not state["armed"]:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("camera was not armed")
                    state["cond"].wait(remaining)
            camera._deliver(images)
        except BaseException as error:  # surfaced on the test owner below
            failure.append(error)

    thread = threading.Thread(target=source, daemon=False)
    thread.start()
    return thread, failure


def test_exact_pipeline_commits_and_reloads_capture_artifact(tmp_path):
    camera, runtime, spec = _runtime_and_spec()
    repository = CaptureRepository(tmp_path / "captures")
    plan = compile_capture_artifact_pipeline(spec, repository)
    thread, source_failure = _deliver_when_armed(
        camera,
        [
            np.full((6, 8), 17, dtype=np.uint16),
            np.full((6, 8), 29, dtype=np.uint16),
        ],
    )
    try:
        handle = runtime.controller.start(plan)
        reference = handle.result(3.0)
        thread.join(2.0)
        assert not thread.is_alive() and source_failure == []
        assert isinstance(reference, CaptureArtifactRef)
        assert handle.snapshot().final_committed

        artifact = repository.load(reference)
        assert artifact.ref == reference
        source = artifact.frame_source
        block = source.materialize(memory_limit_bytes=8 << 20)
        assert block.values.shape == (1, 2, 6, 8)
        assert np.all(block.values[0, 0] == 17)
        assert np.all(block.values[0, 1] == 29)
        assert [item.source_ordinal for item in source.metadata_in_event_order] == [0, 1]
        assert artifact.coverage.complete
        assert artifact.provenance.derivation is None
        assert artifact.terminal.produced_count == 2
        assert artifact.terminal.drained_count == 2
        assert artifact.pulse_lineage is None
        assert artifact.provenance.trace_binding.run_id == handle.snapshot().run_id.value
        assert artifact.run_id == handle.snapshot().run_id.value
        assert artifact.safety_bundle_id == handle.snapshot().safety_bundle_id
        assert (
            artifact.camera_provenance
            == spec.measurement.capture_contract.camera_provenance
        )
        assert source.cell_schedule == (
            DatasetCellAddress(0, 0),
            DatasetCellAddress(0, 1),
        )
        assert len(artifact.chain_contract_digest) == 64
        assert artifact.aggregate_peak_bytes > block.values.nbytes
        assert not hasattr(artifact, "block")
        assert not hasattr(artifact, "event_metadata")
        assert not hasattr(artifact, "source_cell_schedule")
        assert not hasattr(artifact, "camera")
        assert not hasattr(reference, "repository")

        # Current-only codec: a content-addressed manifest with even one legacy
        # or unknown field is not treated as a compatible CaptureArtifact.
        manifest = decode(
            repository._store.read_manifest("capture", reference.manifest_digest)
        )
        assert manifest["camera_provenance"]["descriptor"] == (
            camera_capture_descriptor_to_tree(artifact.camera_provenance.descriptor)
        )
        assert manifest["camera_provenance"]["binding"] == (
            readout_binding_key_to_tree(artifact.camera_provenance.binding)
        )
        assert "readout_event_index" not in manifest["camera_provenance"]
        assert "frame_contract" not in manifest["camera_provenance"]
        assert "source_dataset_schema_blob" not in manifest
        assert manifest["provenance"]["derivation"] is None
        assert (
            manifest["camera_capability_evidence"]["physical_facts_fingerprint"]
            == artifact.camera_capability_evidence.physical_facts.fingerprint
        )
        assert (
            artifact.camera_capability_evidence.fingerprint
            == artifact.terminal.capability_fingerprint
        )
        assert artifact.camera_arm_spec.digest == artifact.terminal.capture_spec_fingerprint
        assert artifact.camera_arm_spec == spec.measurement.capture_spec

        unsupported_manifest = decode(encode(manifest))
        unsupported_manifest["schema"] = "unsupported-capture-artifact"
        unsupported = repository._store.publish_manifest(
            "capture",
            encode(unsupported_manifest),
        )
        with pytest.raises(ValueError, match="CaptureArtifact"):
            repository.load(
                CaptureArtifactRef(
                    repository.repository_id,
                    unsupported.content.digest,
                )
            )

        uncommitted_tree = decode(encode(manifest))
        uncommitted_tree["provenance"]["ordered_event_digest"] = "f" * 64
        uncommitted = repository._store.publish_manifest(
            "capture",
            encode(uncommitted_tree),
        )
        uncommitted_ref = CaptureArtifactRef(
            repository.repository_id,
            uncommitted.content.digest,
        )
        assert repository.load(uncommitted_ref).provenance.ordered_event_digest == "f" * 64
        with pytest.raises(PermissionError, match="no committed journal authority"):
            repository.admit(uncommitted_ref)

        opaque_digest = "f" * 64
        opaque_provenance = replace(
            artifact.camera_provenance,
            descriptor=replace(
                artifact.camera_provenance.descriptor,
                camera_arm_spec_fingerprint=opaque_digest,
            ),
        )
        with pytest.raises(ValueError, match="canonical camera arm spec"):
            replace(
                artifact,
                camera_provenance=opaque_provenance,
                terminal=replace(
                    artifact.terminal,
                    capture_spec_fingerprint=opaque_digest,
                ),
            )

        opaque_manifest = decode(encode(manifest))
        opaque_manifest["camera_provenance"][
            "camera_arm_spec_fingerprint"
        ] = opaque_digest
        opaque_manifest["camera_provenance"]["descriptor"][
            "camera_arm_spec_fingerprint"
        ] = opaque_digest
        opaque_manifest["terminal"]["capture_spec_fingerprint"] = opaque_digest
        invalid_opaque = repository._store.publish_manifest(
            "capture",
            encode(opaque_manifest),
        )
        with pytest.raises(ValueError, match="canonical camera arm spec"):
            repository.load(
                CaptureArtifactRef(
                    repository.repository_id,
                    invalid_opaque.content.digest,
                )
            )

        def rebound_fields(frozen):
            return {
                "camera_arm_spec": frozen,
                "camera_provenance": replace(
                    artifact.camera_provenance,
                    descriptor=replace(
                        artifact.camera_provenance.descriptor,
                        camera_arm_spec_fingerprint=frozen.digest,
                    ),
                ),
                "terminal": replace(
                    artifact.terminal,
                    capture_spec_fingerprint=frozen.digest,
                ),
            }

        def arm_tree(frozen):
            tree = decode(encode(manifest))
            tree["camera_arm_spec"] = {
                "owner_fingerprint": frozen.owner_fingerprint,
                "payload": frozen.payload,
                "digest": frozen.digest,
            }
            tree["camera_provenance"][
                "camera_arm_spec_fingerprint"
            ] = frozen.digest
            tree["camera_provenance"]["descriptor"][
                "camera_arm_spec_fingerprint"
            ] = frozen.digest
            tree["terminal"]["capture_spec_fingerprint"] = frozen.digest
            return tree

        invalid_arm_specs = (
            (
                FrozenCaptureSpec(
                    CAMERA_CAPTURE_SPEC_OWNER_FINGERPRINT,
                    b"not-a-canonical-camera-capture-spec",
                ),
                "canonical",
            ),
            (
                replace(artifact.camera_arm_spec, owner_fingerprint="e" * 64),
                "unknown owner",
            ),
            (
                freeze_camera_capture_spec(
                    CameraCaptureSpec(
                        CameraAcquisitionMode.FREE_RUNNING,
                        len(source.metadata_in_event_order),
                        artifact.terminal.settings_fingerprint,
                    )
                ),
                "EXTERNAL_TRIGGERED",
            ),
            (
                freeze_camera_capture_spec(
                    CameraCaptureSpec(
                        CameraAcquisitionMode.EXTERNAL_TRIGGERED,
                        len(source.metadata_in_event_order) + 1,
                        artifact.terminal.settings_fingerprint,
                    )
                ),
                "expected_frames",
            ),
            (
                freeze_camera_capture_spec(
                    CameraCaptureSpec(
                        CameraAcquisitionMode.EXTERNAL_TRIGGERED,
                        len(source.metadata_in_event_order),
                        "d" * 64,
                    )
                ),
                "arm settings",
            ),
        )
        for invalid_arm, message in invalid_arm_specs:
            with pytest.raises(ValueError, match=message):
                replace(artifact, **rebound_fields(invalid_arm))
            invalid_arm_manifest = repository._store.publish_manifest(
                "capture",
                encode(arm_tree(invalid_arm)),
            )
            with pytest.raises(ValueError, match=message):
                repository.load(
                    CaptureArtifactRef(
                        repository.repository_id,
                        invalid_arm_manifest.content.digest,
                    )
                )

        forged_setting = replace(
            artifact.camera_provenance.descriptor.event_settings[0],
            exposure_seconds=0.017,
        )
        forged_provenance = replace(
            artifact.camera_provenance,
            descriptor=replace(
                artifact.camera_provenance.descriptor,
                event_settings=(forged_setting,),
            ),
        )
        with pytest.raises(ValueError, match="attested physical facts"):
            replace(artifact, camera_provenance=forged_provenance)

        descriptor_only_forge = decode(encode(manifest))
        descriptor_only_forge["camera_provenance"]["descriptor"][
            "event_settings"
        ][0]["exposure_seconds"] = 0.017
        invalid_descriptor = repository._store.publish_manifest(
            "capture",
            encode(descriptor_only_forge),
        )
        with pytest.raises(ValueError, match="attested physical facts"):
            repository.load(
                CaptureArtifactRef(
                    repository.repository_id,
                    invalid_descriptor.content.digest,
                )
            )

        descriptor_and_facts_forge = decode(encode(descriptor_only_forge))
        physical_tree = descriptor_and_facts_forge["camera_capability_evidence"][
            "physical_facts"
        ]
        physical_tree["exposure_seconds"] = 0.017
        forged_facts = camera_physical_facts_from_tree(physical_tree)
        descriptor_and_facts_forge["camera_capability_evidence"][
            "physical_facts_fingerprint"
        ] = forged_facts.fingerprint
        invalid_evidence = repository._store.publish_manifest(
            "capture",
            encode(descriptor_and_facts_forge),
        )
        with pytest.raises(ValueError, match="canonical camera capability evidence"):
            repository.load(
                CaptureArtifactRef(
                    repository.repository_id,
                    invalid_evidence.content.digest,
                )
            )

        wrong_schedule = decode(encode(manifest))
        frame_index_ref = content_ref_from_tree(wrong_schedule["frame_index_blob"])
        frame_index = decode(repository._store.read_blob(frame_index_ref))
        frame_index["cell_schedule"] = list(reversed(frame_index["cell_schedule"]))
        wrong_schedule["frame_index_blob"] = content_ref_to_tree(
            repository._store.put_blob(encode(frame_index))
        )
        invalid_schedule = repository._store.publish_manifest(
            "capture",
            encode(wrong_schedule),
        )
        with pytest.raises(ValueError, match="sealed join plan"):
            repository.load(
                CaptureArtifactRef(
                    repository.repository_id,
                    invalid_schedule.content.digest,
                )
            )

        processor_like = dict(manifest)
        processor_like["chain_contract_digest"] = "f" * 64
        invalid_chain = repository._store.publish_manifest(
            "capture",
            encode(processor_like),
        )
        with pytest.raises(ValueError, match="direct source-to-DatasetBuilder"):
            repository.load(
                CaptureArtifactRef(
                    repository.repository_id,
                    invalid_chain.content.digest,
                )
            )

        derived_raw = dict(manifest)
        derived_raw["provenance"] = dict(manifest["provenance"])
        derived_raw["provenance"]["derivation"] = {"forged": True}
        invalid_derivation = repository._store.publish_manifest(
            "capture",
            encode(derived_raw),
        )
        with pytest.raises(ValueError, match="cannot contain processor derivation"):
            repository.load(
                CaptureArtifactRef(
                    repository.repository_id,
                    invalid_derivation.content.digest,
                )
            )

        manifest["legacy_camera_settings"] = {"exposure": 1e-3}
        invalid = repository._store.publish_manifest("capture", encode(manifest))
        with pytest.raises(ValueError, match="must contain exactly"):
            repository.load(
                CaptureArtifactRef(repository.repository_id, invalid.content.digest)
            )

        repository.close()
        with pytest.raises(RuntimeError, match="closed"):
            source.read(DatasetCellAddress(0, 0))
        reopened = CaptureRepository(tmp_path / "captures")
        assert reopened.startup_reconciliations == ()
        reloaded = reopened.load(reference)
        reloaded_block = reloaded.frame_source.materialize(memory_limit_bytes=8 << 20)
        assert np.array_equal(reloaded_block.values, block.values)
        assert (
            reloaded.frame_source.metadata_in_event_order
            == source.metadata_in_event_order
        )
        reopened.close()
    finally:
        if thread.is_alive():
            camera.finish_record_capture()
            thread.join(2.0)
        assert runtime.shutdown(timeout=2.0)


def test_capture_payload_digest_replays_repeat_and_explicit_point_schedule(tmp_path):
    layout = PointLayout.explicit((3,), ((2,), (0,), (1,)))
    schedule = (
        DatasetCellAddress(1, 2),
        DatasetCellAddress(0, 1),
        DatasetCellAddress(1, 0),
        DatasetCellAddress(0, 2),
        DatasetCellAddress(1, 1),
        DatasetCellAddress(0, 0),
    )
    camera, runtime, spec = _runtime_and_spec(
        repeat_size=2,
        point_size=3,
        point_layout=layout,
        source_cell_schedule=schedule,
    )
    repository = CaptureRepository(tmp_path / "captures")
    plan = compile_capture_artifact_pipeline(spec, repository)
    images = tuple(
        np.full((6, 8), ordinal + 1, dtype=np.uint16)
        for ordinal in range(len(schedule))
    )
    thread, failures = _deliver_when_armed(camera, images)
    try:
        reference = runtime.controller.run(plan)
        thread.join(2.0)
        assert not thread.is_alive() and failures == []
        artifact = repository.load(reference)
        source = artifact.frame_source
        block = source.materialize(memory_limit_bytes=8 << 20)
        assert source.cell_schedule == schedule
        assert block.values.shape == (2, 3, 6, 8)
        for ordinal, address in enumerate(schedule):
            assert np.all(
                block.values[
                    address.repeat_index,
                    address.point_storage_index,
                ]
                == ordinal + 1
            )
    finally:
        if thread.is_alive():
            camera.finish_record_capture()
            thread.join(2.0)
        repository.close()
        assert runtime.shutdown(timeout=2.0)


def test_lazy_iteration_crosses_canonical_frame_chunk_boundaries(tmp_path):
    camera, runtime, spec = _runtime_and_spec(point_size=3)
    frame_nbytes = 6 * 8 * np.dtype(np.uint16).itemsize
    record_nbytes = frame_nbytes + 1  # DatasetBuilder commits CellValidity.
    repository = CaptureRepository(
        tmp_path / "captures",
        resource_policy=CaptureRepositoryResourcePolicy(
            max_frame_chunk_blob_bytes=record_nbytes,
        ),
    )
    images = tuple(
        np.full((6, 8), value, dtype=np.uint16) for value in (11, 22, 33)
    )
    thread, failures = _deliver_when_armed(camera, images)
    try:
        reference = runtime.controller.run(
            compile_capture_artifact_pipeline(spec, repository)
        )
        thread.join(2.0)
        assert not thread.is_alive() and failures == []
        source = repository.load(reference).frame_source
        requested = (
            DatasetCellAddress(0, 0),
            DatasetCellAddress(0, 1),
            DatasetCellAddress(0, 0),
            DatasetCellAddress(0, 2),
        )
        observed = tuple(source.iter_cells(requested))
        assert tuple(cell for cell, _sample in observed) == requested
        assert tuple(int(sample.image.values[0, 0]) for _cell, sample in observed) == (
            11,
            22,
            11,
            33,
        )
        assert source.max_read_scratch_bytes == 3 * record_nbytes
    finally:
        if thread.is_alive():
            camera.finish_record_capture()
            thread.join(2.0)
        repository.close()
        assert runtime.shutdown(timeout=2.0)


def test_frame_record_geometry_drives_stage_reader_and_round_trip(tmp_path):
    from zlc_neutral_atom.artifacts.capture_frames import (
        _FrameResourceExceeded,
        _capture_frame_record_geometry,
        _stage_capture_frame_source,
    )

    y_axis = _axis("geometry-y", SPATIAL_Y, 2)
    x_axis = _axis("geometry-x", SPATIAL_X, 3)
    schema = DatasetSchema(
        _axis("geometry-repeat", REPEAT, 1),
        (_axis("geometry-point", SCAN_POINT, 3),),
        PointLayout.rect_c((3,)),
        ValueSchema(
            (y_axis, x_axis),
            ValidityContract.components(y_axis.axis_id, x_axis.axis_id),
            np.dtype("<u2"),
            "count",
        ),
    )
    values = np.arange(math.prod(schema.physical_shape), dtype="<u2").reshape(
        schema.physical_shape
    )
    validity_mask = np.ones((1, 3, 2), dtype=bool)
    validity_mask[0, 1, 0] = False
    block = DataBlock(
        BlockId("frame-record-geometry"),
        DatasetRevision(1),
        values,
        ComponentValidity((y_axis.axis_id,), validity_mask),
        schema,
    )
    schedule = tuple(DatasetCellAddress(0, point) for point in range(3))
    metadata = tuple(
        CameraFrameMetadata(
            source_ordinal=ordinal,
            produced_count=ordinal + 1,
            frame_stamp=ordinal,
            camera_stamp=ordinal,
            timestamp_seconds=None,
            timestamp_microseconds=None,
            host_received_at_ns=ordinal + 1,
            driver_buffer_index=ordinal,
            correlation_id=f"geometry:{ordinal}",
        )
        for ordinal in range(3)
    )
    frame_nbytes = 2 * 3 * np.dtype("<u2").itemsize
    validity_nbytes = 2
    record_nbytes = frame_nbytes + validity_nbytes
    chunk_capacity = 2 * record_nbytes - 1
    geometry = _capture_frame_record_geometry(
        schema,
        "component",
        (y_axis.axis_id,),
        len(schedule),
        chunk_capacity,
    )
    assert geometry.frame_nbytes == frame_nbytes
    assert geometry.validity_nbytes == validity_nbytes
    assert geometry.record_nbytes == record_nbytes
    assert geometry.frames_per_chunk == 1
    assert geometry.expected_chunks == len(schedule)
    assert geometry.largest_chunk_nbytes == record_nbytes
    # Preserve the historical conservative compile-time admission bound even
    # though the capacity remainder cannot contain another complete record.
    assert geometry.scratch_chunk_upper_nbytes == chunk_capacity
    assert _capture_frame_record_geometry(
        schema,
        "component",
        (y_axis.axis_id,),
        len(schedule),
        record_nbytes,
    ).record_nbytes == record_nbytes
    with pytest.raises(
        _FrameResourceExceeded,
        match="one capture frame record exceeds chunk policy",
    ):
        _capture_frame_record_geometry(
            schema,
            "component",
            (y_axis.axis_id,),
            len(schedule),
            record_nbytes - 1,
        )

    repository = CaptureRepository(tmp_path / "frame-record-geometry")
    try:
        source, _index_ref = _stage_capture_frame_source(
            block=block,
            event_metadata=metadata,
            cell_schedule=schedule,
            store_authority=repository._store_authority,
            root_lease=repository._root_lease,
            max_cells=len(schedule),
            max_total_frame_bytes=len(schedule) * record_nbytes,
            max_chunk_blob_bytes=chunk_capacity,
            max_frame_index_blob_bytes=1 << 20,
            max_canonical_nodes=10_000,
            max_canonical_container_entries=10_000,
        )
        # The returned source was reconstructed through the persisted index;
        # equality proves the reader derived the same canonical geometry.
        assert source._geometry == geometry
        assert tuple(reference.size for reference in source._chunk_refs) == (
            record_nbytes,
            record_nbytes,
            record_nbytes,
        )
        observed = tuple(source.iter_event_order())
        assert tuple(cell for cell, _sample in observed) == schedule
        for ordinal, (_cell, sample) in enumerate(observed):
            assert isinstance(sample.image.validity, ComponentValidity)
            assert np.array_equal(
                sample.image.validity.mask,
                validity_mask[0, ordinal],
            )
            expected = values[0, ordinal].copy()
            expected[~validity_mask[0, ordinal], :] = 0
            assert np.array_equal(sample.image.values, expected)
    finally:
        repository.close()


def test_component_frame_iteration_stays_within_declared_read_scratch(tmp_path):
    from zlc_neutral_atom.artifacts.capture_frames import _stage_capture_frame_source

    height = width = 512
    y_axis = _axis("component-frame-y", SPATIAL_Y, height)
    x_axis = _axis("component-frame-x", SPATIAL_X, width)
    schema = DatasetSchema(
        _axis("component-repeat", REPEAT, 1),
        (_axis("component-point", SCAN_POINT, 2),),
        PointLayout.rect_c((2,)),
        ValueSchema(
            (y_axis, x_axis),
            ValidityContract.components(y_axis.axis_id, x_axis.axis_id),
            np.dtype("<f8"),
            "count",
        ),
    )
    values = np.full(schema.physical_shape, 19.0, dtype="<f8")
    values[:, :, 1, 1] = np.nan
    validity_mask = np.ones(schema.physical_shape, dtype=bool)
    validity_mask[:, :, 0, 0] = False
    block = DataBlock(
        BlockId("component-frame-scratch"),
        DatasetRevision(1),
        values,
        ComponentValidity((y_axis.axis_id, x_axis.axis_id), validity_mask),
        schema,
    )
    schedule = (DatasetCellAddress(0, 0), DatasetCellAddress(0, 1))
    metadata = tuple(
        CameraFrameMetadata(
            source_ordinal=ordinal,
            produced_count=ordinal + 1,
            frame_stamp=ordinal,
            camera_stamp=ordinal,
            timestamp_seconds=None,
            timestamp_microseconds=None,
            host_received_at_ns=ordinal + 1,
            driver_buffer_index=ordinal,
            correlation_id=f"component-frame:{ordinal}",
        )
        for ordinal in range(2)
    )
    frame_nbytes = height * width * np.dtype("<f8").itemsize
    validity_nbytes = height * width
    record_nbytes = frame_nbytes + validity_nbytes
    repository = CaptureRepository(tmp_path / "component-capture")
    try:
        source, _index_ref = _stage_capture_frame_source(
            block=block,
            event_metadata=metadata,
            cell_schedule=schedule,
            store_authority=repository._store_authority,
            root_lease=repository._root_lease,
            max_cells=2,
            max_total_frame_bytes=2 * record_nbytes,
            max_chunk_blob_bytes=record_nbytes,
            max_frame_index_blob_bytes=1 << 20,
            max_canonical_nodes=10_000,
            max_canonical_container_entries=10_000,
        )

        was_tracing = tracemalloc.is_tracing()
        if not was_tracing:
            tracemalloc.start()
        tracemalloc.clear_traces()
        iterator = source.iter_event_order()
        first = next(iterator)
        second = next(iterator)
        _current, peak = tracemalloc.get_traced_memory()
        iterator.close()
        if not was_tracing:
            tracemalloc.stop()

        assert first[1].image.values[0, 0] == 0
        assert second[1].image.values[0, 0] == 0
        assert np.isnan(first[1].image.values[1, 1])
        assert np.isnan(second[1].image.values[1, 1])
        # Retaining the prior immutable sample while crossing a chunk boundary
        # is the source's worst object lifetime.  The margin is Python object
        # overhead only; another frame-sized value or validity mask must fail.
        assert peak <= source.max_read_scratch_bytes + (128 << 10)
    finally:
        repository.close()


def test_scratch_bound_covers_global_and_partial_component_validity(tmp_path):
    from zlc_neutral_atom.artifacts.capture_frames import (
        _capture_frame_source_scratch_upper_bound,
        _stage_capture_frame_source,
    )

    y_axis = _axis("scratch-y", SPATIAL_Y, 2)
    x_axis = _axis("scratch-x", SPATIAL_X, 3)
    schema = DatasetSchema(
        _axis("scratch-repeat", REPEAT, 2),
        (_axis("scratch-point", SCAN_POINT, 3),),
        PointLayout.rect_c((3,)),
        ValueSchema(
            (y_axis, x_axis),
            ValidityContract.components(y_axis.axis_id, x_axis.axis_id),
            np.dtype("<f8"),
            "count",
        ),
    )
    cells = schema.repeat_axis.size * schema.point_layout.storage_size
    schedule = tuple(
        DatasetCellAddress(repeat, point)
        for repeat in range(schema.repeat_axis.size)
        for point in range(schema.point_layout.storage_size)
    )
    metadata = tuple(
        CameraFrameMetadata(
            source_ordinal=ordinal,
            produced_count=ordinal + 1,
            frame_stamp=ordinal,
            camera_stamp=ordinal,
            timestamp_seconds=None,
            timestamp_microseconds=None,
            host_received_at_ns=ordinal + 1,
            driver_buffer_index=ordinal,
            correlation_id=f"scratch:{ordinal}",
        )
        for ordinal in range(cells)
    )
    values = np.arange(math.prod(schema.physical_shape), dtype="<f8").reshape(
        schema.physical_shape
    )
    values[0, 0, 0, 0] = np.nan
    variants = (
        ("valid", VALID, 0),
        ("invalid", INVALID, 0),
        ("cell", CellValidity(np.ones((2, 3), dtype=bool)), 1),
        (
            "partial-y",
            ComponentValidity(
                (y_axis.axis_id,), np.ones((2, 3, 2), dtype=bool)
            ),
            2,
        ),
        (
            "partial-x",
            ComponentValidity(
                (x_axis.axis_id,), np.ones((2, 3, 3), dtype=bool)
            ),
            3,
        ),
        (
            "full",
            ComponentValidity(
                (y_axis.axis_id, x_axis.axis_id),
                np.ones((2, 3, 2, 3), dtype=bool),
            ),
            6,
        ),
    )
    frame_nbytes = math.prod(schema.cell_schema.data_shape) * np.dtype("<f8").itemsize
    max_record_nbytes = frame_nbytes + 6
    repository = CaptureRepository(tmp_path / "scratch-representations")
    try:
        for capacity in (
            max_record_nbytes,
            max_record_nbytes + 1,
            2 * max_record_nbytes - 1,
            4 * max_record_nbytes + 3,
        ):
            compile_upper = _capture_frame_source_scratch_upper_bound(
                schema, cells, capacity
            )
            for name, validity, validity_nbytes in variants:
                block = DataBlock(
                    BlockId(f"scratch-{name}-{capacity}"),
                    DatasetRevision(1),
                    values,
                    validity,
                    schema,
                )
                source, _index_ref = _stage_capture_frame_source(
                    block=block,
                    event_metadata=metadata,
                    cell_schedule=schedule,
                    store_authority=repository._store_authority,
                    root_lease=repository._root_lease,
                    max_cells=cells,
                    max_total_frame_bytes=cells * max_record_nbytes,
                    max_chunk_blob_bytes=capacity,
                    max_frame_index_blob_bytes=1 << 20,
                    max_canonical_nodes=10_000,
                    max_canonical_container_entries=10_000,
                )
                record_nbytes = frame_nbytes + validity_nbytes
                per_chunk = max(1, capacity // record_nbytes)
                largest_chunk = min(cells, per_chunk) * record_nbytes
                exact = largest_chunk + 2 * frame_nbytes + 2 * validity_nbytes + 6
                assert source.max_read_scratch_bytes == exact
                assert source.max_read_scratch_bytes <= compile_upper
    finally:
        repository.close()


def test_lazy_frame_read_fails_closed_on_chunk_corruption(tmp_path):
    camera, runtime, spec = _runtime_and_spec()
    repository = CaptureRepository(tmp_path / "captures")
    images = (
        np.full((6, 8), 7, dtype=np.uint16),
        np.full((6, 8), 9, dtype=np.uint16),
    )
    thread, failures = _deliver_when_armed(camera, images)
    try:
        reference = runtime.controller.run(
            compile_capture_artifact_pipeline(spec, repository)
        )
        thread.join(2.0)
        assert not thread.is_alive() and failures == []
        manifest = decode(
            repository._store.read_manifest("capture", reference.manifest_digest)
        )
        index_ref = content_ref_from_tree(manifest["frame_index_blob"])
        index = decode(repository._store.read_blob(index_ref))
        chunk_ref = content_ref_from_tree(index["frame_chunks"][0])
        repository._store._blob_path(chunk_ref.digest).write_bytes(b"tampered")

        # Ordinary admission intentionally stays bounded and validates chunks
        # lazily; the first attempted data read owns the content-integrity check.
        source = repository.load(reference).frame_source
        with pytest.raises(ContentCorruptionError, match="immutable reference"):
            source.read(DatasetCellAddress(0, 0))
    finally:
        if thread.is_alive():
            camera.finish_record_capture()
            thread.join(2.0)
        repository.close()
        assert runtime.shutdown(timeout=2.0)


@pytest.mark.parametrize(
    "structure_limit",
    (
        {"max_canonical_nodes": 1},
        {"max_canonical_container_entries": 1},
    ),
    ids=("nodes", "container-entries"),
)
def test_frame_index_must_pass_reader_structure_limits_before_publish(
    tmp_path,
    structure_limit,
):
    camera, runtime, spec = _runtime_and_spec()
    repository_root = tmp_path / "captures"
    repository = CaptureRepository(
        repository_root,
        resource_policy=CaptureRepositoryResourcePolicy(
            # Deliberately smaller than even the root frame-index tree.  The
            # byte budget still fits, so only the reader's structural boundary
            # can reject this artifact.
            **structure_limit,
        ),
    )
    thread, failures = _deliver_when_armed(
        camera,
        (
            np.full((6, 8), 7, dtype=np.uint16),
            np.full((6, 8), 9, dtype=np.uint16),
        ),
    )
    try:
        with pytest.raises(RunFailed) as failure:
            runtime.controller.run(
                compile_capture_artifact_pipeline(spec, repository)
            )
        thread.join(2.0)
        assert not thread.is_alive() and failures == []
        assert "CaptureResourceExceeded" in str(failure.value)
        manifest_root = repository_root / "content" / "manifests" / "capture"
        assert not manifest_root.exists() or tuple(manifest_root.iterdir()) == ()
    finally:
        if thread.is_alive():
            camera.finish_record_capture()
            thread.join(2.0)
        repository.close()
        assert runtime.shutdown(timeout=2.0)


def test_failed_capture_never_publishes_a_manifest(tmp_path):
    camera, runtime, spec = _runtime_and_spec()
    repository_root = tmp_path / "captures"
    repository = CaptureRepository(repository_root)
    plan = compile_capture_artifact_pipeline(spec, repository)
    # Capability was frozen while binding.  Drift before preflight must fail
    # before arm and therefore before any artifact commit authority is consumed.
    camera.exposure = 2e-3
    try:
        with pytest.raises(RunFailed):
            runtime.controller.start(plan).result(3.0)
        manifest_root = repository_root / "content" / "manifests" / "capture"
        assert not manifest_root.exists() or tuple(manifest_root.iterdir()) == ()
    finally:
        assert runtime.shutdown(timeout=2.0)


def test_closed_capture_repository_rejects_before_camera_arm(tmp_path, monkeypatch):
    camera, runtime, spec = _runtime_and_spec()
    repository = CaptureRepository(tmp_path / "captures")
    plan = compile_capture_artifact_pipeline(spec, repository)
    original_arm = camera._arm
    arm_calls = 0

    def counted_arm(frames, *, max_inflight_frames=None):
        nonlocal arm_calls
        arm_calls += 1
        return original_arm(
            frames,
            max_inflight_frames=max_inflight_frames,
        )

    monkeypatch.setattr(camera, "_arm", counted_arm)
    repository.close()
    try:
        with pytest.raises(RunFailed):
            runtime.controller.run(plan)
        assert arm_calls == 0
        state = camera._recent_state()
        with state["cond"]:
            assert not state["armed"]
    finally:
        assert runtime.shutdown(timeout=2.0)


def test_capture_run_borrow_blocks_repository_close_during_arm(
    tmp_path,
    monkeypatch,
):
    camera, runtime, spec = _runtime_and_spec()
    repository = CaptureRepository(tmp_path / "captures")
    plan = compile_capture_artifact_pipeline(spec, repository)
    original_arm = camera._arm
    close_attempts = 0

    def guarded_arm(frames, *, max_inflight_frames=None):
        nonlocal close_attempts
        close_attempts += 1
        with pytest.raises(
            RuntimeError,
            match="outstanding operations",
        ):
            repository.close()
        return original_arm(
            frames,
            max_inflight_frames=max_inflight_frames,
        )

    monkeypatch.setattr(camera, "_arm", guarded_arm)
    thread, source_failure = _deliver_when_armed(
        camera,
        (
            np.full((6, 8), 7, dtype=np.uint16),
            np.full((6, 8), 9, dtype=np.uint16),
        ),
    )
    try:
        reference = runtime.controller.run(plan)
        thread.join(2.0)
        assert not thread.is_alive() and source_failure == []
        assert close_attempts == 1
        assert repository.load(reference).ref == reference
    finally:
        if thread.is_alive():
            camera.finish_record_capture()
            thread.join(2.0)
        repository.close()
        assert runtime.shutdown(timeout=2.0)


def test_post_safety_context_cannot_be_forged_for_capture_staging(tmp_path):
    camera, runtime, spec = _runtime_and_spec()
    repository = CaptureRepository(tmp_path / "captures")
    thread, source_failure = _deliver_when_armed(
        camera,
        [
            np.full((6, 8), 3, dtype=np.uint16),
            np.full((6, 8), 5, dtype=np.uint16),
        ],
    )
    try:
        result = runtime.controller.run(compile_pipeline(spec))
        thread.join(2.0)
        assert not thread.is_alive() and source_failure == []
        with pytest.raises(PermissionError, match="minted by RunController"):
            PostSafetyContext(
                run_id=RunId("different-run"),
                cancellation=CancellationToken(),
                deadline=None,
                safety_bundle_id="different-safety-bundle",
                handle=object(),
            )
        manifest_root = (
            tmp_path / "captures" / "content" / "manifests" / "capture"
        )
        assert not manifest_root.exists()
    finally:
        if thread.is_alive():
            camera.finish_record_capture()
            thread.join(2.0)
        assert runtime.shutdown(timeout=2.0)


def test_camera_provenance_is_bound_to_frozen_capability_and_capture_spec():
    _camera, runtime, spec = _runtime_and_spec()
    try:
        measurement = spec.measurement
        contract = measurement.capture_contract
        provenance = contract.camera_provenance
        facts = measurement.capture_port.capability.camera_physical_facts
        assert provenance is not None and facts is not None
        assert provenance.descriptor.camera_identity == facts.camera_identity
        assert provenance.descriptor.event_settings == (facts.event_setting(0),)
        assert provenance.active_settings_fingerprint == (
            measurement.capture_port.capability.settings_fingerprint
        )
        assert provenance.camera_arm_spec_fingerprint == measurement.capture_spec.digest

        forged_setting = replace(
            provenance.descriptor.event_settings[0],
            exposure_seconds=0.017,
        )
        forged_descriptor = replace(
            provenance.descriptor,
            event_settings=(forged_setting,),
        )
        with pytest.raises(ValueError, match="attested physical facts"):
            replace(
                contract,
                camera_provenance=replace(
                    provenance,
                    descriptor=forged_descriptor,
                ),
            )
    finally:
        assert runtime.shutdown(timeout=2.0)


def test_custom_value_projection_cannot_masquerade_as_raw_camera_capture(tmp_path):
    @dataclass(frozen=True)
    class OffsetAdapter:
        payload_contract: object
        operator_fingerprint: str = "e" * 64

        @property
        def value_schema(self):
            return self.payload_contract.value_schema

        @property
        def metadata_contract(self):
            return self.payload_contract.metadata_contract

        def value(self, payload):
            self.payload_contract.validate(payload)
            return replace(
                payload.image,
                values=payload.image.values + np.uint16(100),
            )

    camera, runtime, spec = _runtime_and_spec()
    raw_contract = spec.measurement.capture_contract
    offset = OffsetAdapter(raw_contract.payload_contract)
    with pytest.raises(ValueError, match="owner identity event adapter"):
        replace(raw_contract, event_adapter=offset)

    derived_contract = replace(
        raw_contract,
        event_adapter=offset,
        camera_provenance=None,
    )
    derived_spec = replace(
        spec,
        measurement=replace(
            spec.measurement,
            capture_contract=derived_contract,
        ),
    )
    thread, source_failure = _deliver_when_armed(
        camera,
        [
            np.full((6, 8), 1, dtype=np.uint16),
            np.full((6, 8), 2, dtype=np.uint16),
        ],
    )
    try:
        result = runtime.controller.run(compile_pipeline(derived_spec))
        thread.join(2.0)
        assert not thread.is_alive() and source_failure == []
        assert np.all(result.dataset.block.values[0, 0] == 101)
        assert np.all(result.dataset.block.values[0, 1] == 102)
        assert not result.is_direct_raw_capture
        repository = CaptureRepository(tmp_path / "captures")
        with pytest.raises(ValueError, match="raw camera provenance"):
            compile_capture_artifact_pipeline(derived_spec, repository)
    finally:
        if thread.is_alive():
            camera.finish_record_capture()
            thread.join(2.0)
        assert runtime.shutdown(timeout=2.0)


def test_capture_ref_cannot_be_loaded_from_another_repository(tmp_path):
    first = CaptureRepository(tmp_path / "first", repository_id="first")
    second = CaptureRepository(tmp_path / "second", repository_id="second")
    reference = CaptureArtifactRef("first", "1" * 64)
    with pytest.raises(ValueError, match="another repository"):
        second.load(reference)
    assert first.startup_reconciliations == ()


def test_capture_ref_has_one_leaf_owner_and_a_strict_current_codec():
    reference = CaptureArtifactRef("capture-repository", "a" * 64)
    assert decode_capture_artifact_ref(encode_capture_artifact_ref(reference)) == reference
    dependency = capture_artifact_input_ref(reference)
    assert dependency.reference_schema_id.endswith("capture-artifact-ref")
    assert dependency.content_digest == reference.manifest_digest
    assert decode_capture_artifact_ref(dependency.canonical_reference) == reference
    with pytest.raises(ValueError, match="unknown field set"):
        capture_artifact_ref_from_tree(
            {
                "schema": "zlc_neutral_atom.capture-artifact-ref",
                "repository_id": reference.repository_id,
                "manifest_digest": reference.manifest_digest,
                "source_capture_ref": reference.manifest_digest,
            }
        )

    from zlc_neutral_atom.artifacts.capture import CaptureArtifactRef as repository_ref

    assert repository_ref is CaptureArtifactRef
