"""Fit-result persistence over one direct-output raw capture."""

from __future__ import annotations

from pathlib import Path

import pytest

from zlc_data import AxisId, AxisSourceRef, AxisSpec, BlockId, REPEAT
from zlc_data.fit import fit_spec_for
from zlc_data.fit_codec import encode_fit_result_batch
from zlc_neutral_atom.artifact_dataset_source import ArtifactDatasetSource
from zlc_neutral_atom.artifact_dispatch import ArtifactCapability, ArtifactDispatch
from zlc_neutral_atom.artifacts import (
    FitResultArtifactRef,
    SavedFitResult,
    execute_fit,
    load_fit_result,
    write_fit_result,
)
from zlc_neutral_atom.capture.artifact import (
    compile_capture_artifact_pipeline,
    load_capture_artifact,
)
from zlc_neutral_atom.capture.binding import (
    TriggeredCameraLayout,
    bind_triggered_camera_acquisition,
)
from zlc_neutral_atom.capture.frames import CaptureFrameSource
from zlc_neutral_atom.capture.pipeline import MinimalPipelineSpec
from zlc_neutral_atom.capture.reference import (
    CAPTURE_ARTIFACT_REF_SCHEMA,
    CaptureArtifactRef,
    capture_artifact_ref_from_tree,
    capture_artifact_ref_to_tree,
)
from zlc_neutral_atom.capture.triggered import TriggeredCaptureSpec
from zlc_neutral_atom.devices.simulation.installation import create_virtual_installation
from zlc_pulse import PulseExecutionForm, load_pulse_document


_ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize(
    "alias",
    (
        "fit-run//fit.json",
        "fit-run/./fit.json",
        "fit\\alias/fit.json",
    ),
)
def test_fit_artifact_ref_rejects_noncanonical_path_aliases(alias: str) -> None:
    with pytest.raises(ValueError):
        FitResultArtifactRef(alias)


class _CaptureCase:
    def __init__(self, tmp_path: Path) -> None:
        self.captures_root = tmp_path / "captures"
        self.fits_root = tmp_path / "fits"
        self.installation = create_virtual_installation(seed=17)
        runtime = self.installation.runtime
        catalog = runtime.device_catalog
        camera_ref = catalog.require("camera").ref
        sequencer_ref = catalog.require("sequencer").ref
        binding = bind_triggered_camera_acquisition(
            runtime.pulse_port(sequencer_ref),
            runtime.camera_port(camera_ref),
            pulse_document=load_pulse_document(
                _ROOT / "pulses" / "imaging_template.json"
            ),
            execution_form=PulseExecutionForm.STATIC_ONCE,
            trigger_channel="ch11",
            layout=TriggeredCameraLayout(
                AxisSpec(AxisId("repeat"), "repeat", REPEAT, 1, (0,)),
                AxisId("readout-event"),
                AxisId("scan-ordinal"),
                readout_events_per_repeat=3,
            ),
        )
        pipeline = MinimalPipelineSpec(
            "capture fit source",
            binding.capture,
            BlockId("capture-fit-source"),
        )
        triggered = TriggeredCaptureSpec(
            pipeline,
            binding.pulse_port,
            binding.pulse_request,
            binding.trigger_channel,
            binding.cell_plan,
        )
        self.capture_reference = runtime.start(
            compile_capture_artifact_pipeline(triggered, self.captures_root)
        ).result(10.0)

    def close(self) -> None:
        assert self.installation.runtime.shutdown(timeout=2.0)

    def project_capture(
        self,
        reference: CaptureArtifactRef,
        *,
        materialize: bool,
        abort_check=None,
    ) -> ArtifactDatasetSource:
        artifact = load_capture_artifact(
            self.captures_root,
            reference,
            materialize=materialize,
        )
        if abort_check is not None:
            abort_check()
        snapshot = (
            artifact.materialize_snapshot(abort_check=abort_check)
            if materialize
            else None
        )
        return ArtifactDatasetSource(
            artifact.frame_source.schema,
            artifact.frame_source.ref(artifact.provenance.generation),
            snapshot,
        )


@pytest.fixture
def capture_case(tmp_path):
    case = _CaptureCase(tmp_path)
    try:
        yield case
    finally:
        case.close()


def _artifact_dispatch(case: _CaptureCase) -> ArtifactDispatch:
    return ArtifactDispatch((
        ArtifactCapability(
            CAPTURE_ARTIFACT_REF_SCHEMA,
            "capture",
            CaptureArtifactRef,
            project_dataset=case.project_capture,
            reference_to_tree=capture_artifact_ref_to_tree,
            reference_from_tree=capture_artifact_ref_from_tree,
        ),
    ))


def _fit(case: _CaptureCase):
    artifacts = _artifact_dispatch(case)
    capture = load_capture_artifact(case.captures_root, case.capture_reference)
    spec = fit_spec_for(
        capture.frame_source.schema,
        "exponential_decay",
        independent_sources=(AxisSourceRef.tensor(AxisId("camera.x")),),
        batch_sources=(
            AxisSourceRef.point_rows(),
            AxisSourceRef.tensor(AxisId("camera.y")),
        ),
    )
    return artifacts, execute_fit(artifacts, case.capture_reference, spec)


def test_execute_write_and_cold_load_use_one_direct_fit_record(
    capture_case: _CaptureCase,
    monkeypatch,
) -> None:
    artifacts, result = _fit(capture_case)
    reference = write_fit_result(
        capture_case.fits_root,
        artifacts,
        capture_case.capture_reference,
        result,
        label="capture-fit",
    )
    record = capture_case.fits_root / reference.record_path
    assert record.name == "fit.json"
    monkeypatch.setattr(
        CaptureFrameSource,
        "materialize",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("cold Fit load must not materialize source frames")
        ),
    )
    loaded = load_fit_result(
        capture_case.fits_root,
        reference,
        artifacts=artifacts,
    )
    assert isinstance(loaded, SavedFitResult)
    assert loaded.source_artifact_ref == capture_case.capture_reference
    assert encode_fit_result_batch(loaded.result) == encode_fit_result_batch(result)


def test_fit_record_rejects_corrupt_payload(capture_case: _CaptureCase) -> None:
    artifacts, result = _fit(capture_case)
    reference = write_fit_result(
        capture_case.fits_root,
        artifacts,
        capture_case.capture_reference,
        result,
    )
    (capture_case.fits_root / reference.record_path).write_bytes(b"broken")
    with pytest.raises((TypeError, ValueError)):
        load_fit_result(
            capture_case.fits_root,
            reference,
            artifacts=artifacts,
        )
