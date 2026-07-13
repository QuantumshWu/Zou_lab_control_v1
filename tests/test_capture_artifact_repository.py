"""Exact camera datasets become durable current-schema CaptureArtifacts."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, replace

import numpy as np
import pytest

from Zou_lab_control.neutral_atom.devices.registry import DeviceSet
from Zou_lab_control.neutral_atom.devices.virtual import VirtualCamera, VirtualTrapArray
from zlc_data import (
    AxisId,
    AxisSpec,
    BlockId,
    PointLayout,
    REPEAT,
    SCAN_POINT,
    encode_data_block,
)
from zlc_neutral_atom.acquisition import (
    CAMERA_CAPTURE_SPEC_OWNER_FINGERPRINT,
    CameraAcquisitionMode,
    CameraCaptureSpec,
    freeze_camera_capture_spec,
)
from zlc_neutral_atom.artifacts import (
    CaptureArtifactRef,
    CaptureRepository,
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
from zlc_storage import decode, encode
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
            PipelineMemoryProfile.for_current_runtime(8 << 20),
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
        assert artifact.block.values.shape == (1, 2, 6, 8)
        assert np.all(artifact.block.values[0, 0] == 17)
        assert np.all(artifact.block.values[0, 1] == 29)
        assert [item.source_ordinal for item in artifact.event_metadata] == [0, 1]
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
        assert artifact.source_cell_schedule == (
            DatasetCellAddress(0, 0),
            DatasetCellAddress(0, 1),
        )
        assert len(artifact.chain_contract_digest) == 64
        assert artifact.chain_contract_digest != artifact.memory_profile_fingerprint
        assert not hasattr(artifact, "camera")
        assert not hasattr(reference, "repository")

        # Current-only codec: a content-addressed manifest with even one legacy
        # or unknown field is not treated as a compatible CaptureArtifact.
        manifest = decode(
            repository._store.read_manifest("capture", reference.manifest_digest)
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

        legacy_manifest = decode(encode(manifest))
        legacy_manifest["schema"] = "zlc_neutral_atom.CaptureArtifact/v9"
        legacy = repository._store.publish_manifest(
            "capture",
            encode(legacy_manifest),
        )
        with pytest.raises(ValueError, match="CaptureArtifact/v10"):
            repository.load(
                CaptureArtifactRef(
                    repository.repository_id,
                    legacy.content.digest,
                )
            )

        changed_pixels = np.array(artifact.block.values, copy=True)
        changed_pixels[0, 0, 0, 0] += 1
        changed_block = replace(artifact.block, values=changed_pixels)
        with pytest.raises(ValueError, match="payload event digest"):
            replace(
                artifact,
                block=changed_block,
            )

        with pytest.raises(ValueError, match="payload event digest"):
            replace(
                artifact,
                provenance=replace(
                    artifact.provenance,
                    ordered_event_digest="f" * 64,
                ),
            )

        forged_event_digest = decode(encode(manifest))
        forged_event_digest["provenance"]["ordered_event_digest"] = "f" * 64
        invalid_event_digest = repository._store.publish_manifest(
            "capture",
            encode(forged_event_digest),
        )
        with pytest.raises(ValueError, match="payload event digest"):
            repository.load(
                CaptureArtifactRef(
                    repository.repository_id,
                    invalid_event_digest.content.digest,
                )
            )

        changed_block_ref = repository._store.put_blob(
            encode_data_block(changed_block)
        )
        forged_block_manifest = decode(encode(manifest))
        forged_block_manifest["data_block_blob"] = {
            "digest": changed_block_ref.digest,
            "size": changed_block_ref.size,
        }
        invalid_block = repository._store.publish_manifest(
            "capture",
            encode(forged_block_manifest),
        )
        with pytest.raises(ValueError, match="payload event digest"):
            repository.load(
                CaptureArtifactRef(
                    repository.repository_id,
                    invalid_block.content.digest,
                )
            )

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
                        len(artifact.event_metadata),
                        artifact.terminal.settings_fingerprint,
                    )
                ),
                "EXTERNAL_TRIGGERED",
            ),
            (
                freeze_camera_capture_spec(
                    CameraCaptureSpec(
                        CameraAcquisitionMode.EXTERNAL_TRIGGERED,
                        len(artifact.event_metadata) + 1,
                        artifact.terminal.settings_fingerprint,
                    )
                ),
                "expected_frames",
            ),
            (
                freeze_camera_capture_spec(
                    CameraCaptureSpec(
                        CameraAcquisitionMode.EXTERNAL_TRIGGERED,
                        len(artifact.event_metadata),
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

        wrong_schedule = dict(manifest)
        wrong_schedule["source_cell_schedule"] = list(
            reversed(manifest["source_cell_schedule"])
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
        with pytest.raises(ValueError, match="unknown field set"):
            repository.load(
                CaptureArtifactRef(repository.repository_id, invalid.content.digest)
            )

        repository.close()
        reopened = CaptureRepository(tmp_path / "captures")
        assert reopened.startup_reconciliations == ()
        reloaded = reopened.load(reference)
        assert np.array_equal(reloaded.block.values, artifact.block.values)
        assert reloaded.event_metadata == artifact.event_metadata
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
        assert artifact.source_cell_schedule == schedule
        assert artifact.block.values.shape == (2, 3, 6, 8)
        for ordinal, address in enumerate(schedule):
            assert np.all(
                artifact.block.values[
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
    assert dependency.reference_schema_id.endswith("capture-artifact-ref.v1")
    assert dependency.content_digest == reference.manifest_digest
    assert decode_capture_artifact_ref(dependency.canonical_reference) == reference
    with pytest.raises(ValueError, match="unknown field set"):
        capture_artifact_ref_from_tree(
            {
                "schema": "zlc_neutral_atom.capture-artifact-ref.v1",
                "repository_id": reference.repository_id,
                "manifest_digest": reference.manifest_digest,
                "source_capture_ref": reference.manifest_digest,
            }
        )

    from zlc_neutral_atom.artifacts.capture import CaptureArtifactRef as repository_ref

    assert repository_ref is CaptureArtifactRef
