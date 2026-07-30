"""Direct record-last persistence for readout Calibration artifacts."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any
import uuid

import numpy as np

from zlc_storage import (
    canonical_text,
    decode,
    encode,
    exact_mapping,
    positive_real,
)
from zlc_storage.durability import (
    atomic_write_bytes,
    atomic_write_file,
    durable_makedirs,
    durable_mkdir,
)
from zlc_storage.paths import resolve_under

from zlc_neutral_atom.capture.artifact import CaptureArtifact
from zlc_neutral_atom.capture.reference import (
    CaptureArtifactRef,
    capture_artifact_ref_from_tree,
    capture_artifact_ref_to_tree,
)
from zlc_neutral_atom.devices.camera.contract import ReadoutBindingKey
from zlc_neutral_atom.runtime.cleanup import CleanupReport
from zlc_neutral_atom.runtime.run import PostSafetyContext, RunContext, RunPlan

from .calibration import (
    CalibrationAnalysisRequest,
    ResolvedCalibration,
    _ResolvedCalibrationSource,
    _resolve_calibration_source,
    _validate_calibration_artifact_source_compatibility,
)
from .codec import (
    calibration_artifact_from_tree,
    calibration_artifact_to_tree,
    calibration_report_from_tree,
    calibration_report_to_tree,
)
from .reference import CalibrationArtifactRef


CALIBRATION_RECORD_FORMAT = (
    "zlc_neutral_atom.logic_nodes.readout.calibration.record"
)
_RECORD_FIELDS = {
    "schema",
    "run_id",
    "source_capture_ref",
    "artifact",
    "report",
}
_ARRAY_PATH_FIELD = "array_path"
_PreparedCalibrationAnalysis = tuple[CaptureArtifact, _ResolvedCalibrationSource]


def _load_capture(
    captures_root: Path,
    reference: CaptureArtifactRef,
) -> CaptureArtifact:
    from zlc_neutral_atom.capture.artifact import load_capture_artifact

    capture = load_capture_artifact(captures_root, reference, materialize=False)
    if not isinstance(capture, CaptureArtifact):
        raise TypeError("load_capture_artifact returned a non-CaptureArtifact")
    return capture


def _run_name() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{uuid.uuid4().hex[:8]}"


def _externalize_arrays(
    value: Any,
    prefix: str,
    arrays: list[tuple[str, np.ndarray]],
) -> Any:
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise ValueError("Calibration records cannot persist object arrays")
        ordinal = sum(path.startswith(f"arrays/{prefix}-") for path, _ in arrays)
        relative = f"arrays/{prefix}-{ordinal:04d}.npy"
        arrays.append((relative, value))
        return {_ARRAY_PATH_FIELD: relative}
    if isinstance(value, dict):
        return {
            key: _externalize_arrays(item, prefix, arrays)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_externalize_arrays(item, prefix, arrays) for item in value]
    return value


def _array_path(value: object, prefix: str) -> str:
    canonical = canonical_text(value, "array_path")
    path = PurePosixPath(canonical)
    ordinal = path.stem.removeprefix(f"{prefix}-")
    if (
        path.is_absolute()
        or path.as_posix() != canonical
        or len(path.parts) != 2
        or path.parts[0] != "arrays"
        or path.suffix != ".npy"
        or not path.name.startswith(f"{prefix}-")
        or len(ordinal) < 4
        or not ordinal.isdigit()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError("Calibration array path is not canonical for its owner")
    return path.as_posix()


def _materialize_arrays(
    value: Any,
    run_directory: Path,
    prefix: str,
    seen: set[str],
) -> Any:
    if isinstance(value, np.ndarray):
        raise ValueError("calibration.json must not embed ndarray payloads")
    if isinstance(value, dict):
        if _ARRAY_PATH_FIELD in value:
            if set(value) != {_ARRAY_PATH_FIELD}:
                raise ValueError("Calibration array reference has unknown fields")
            relative = _array_path(value[_ARRAY_PATH_FIELD], prefix)
            if relative in seen:
                raise ValueError("Calibration record reuses one array path")
            seen.add(relative)
            path = resolve_under(run_directory, relative)
            with path.open("rb") as stream:
                loaded = np.load(stream, allow_pickle=False)
            if not isinstance(loaded, np.ndarray):
                close = getattr(loaded, "close", None)
                if callable(close):
                    close()
                raise ValueError("Calibration array path must contain one .npy array")
            if loaded.dtype.hasobject:
                raise ValueError("Calibration record contains an object array")
            loaded.setflags(write=False)
            return loaded
        return {
            key: _materialize_arrays(item, run_directory, prefix, seen)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _materialize_arrays(item, run_directory, prefix, seen)
            for item in value
        ]
    return value


def _write_npy(path: Path, value: np.ndarray) -> None:
    def write(stream) -> None:
        np.save(stream, value, allow_pickle=False)

    atomic_write_file(path, write)


def _read_record(
    calibrations_root: str | Path,
    reference: CalibrationArtifactRef,
):
    from .analysis import CalibrationComputation

    if not isinstance(reference, CalibrationArtifactRef):
        raise TypeError("reference must be CalibrationArtifactRef")
    root = Path(calibrations_root).expanduser().resolve()
    record_path = resolve_under(root, reference.record_path)
    payload = record_path.read_bytes()
    record = exact_mapping(
        decode(payload),
        _RECORD_FIELDS,
        CALIBRATION_RECORD_FORMAT,
    )
    if encode(record) != payload:
        raise ValueError("calibration.json is not canonical current format")
    run_id = canonical_text(record["run_id"], "run_id")
    source_ref = capture_artifact_ref_from_tree(record["source_capture_ref"])
    seen: set[str] = set()
    artifact = calibration_artifact_from_tree(
        _materialize_arrays(
            record["artifact"],
            record_path.parent,
            "artifact",
            seen,
        )
    )
    report = calibration_report_from_tree(
        _materialize_arrays(
            record["report"],
            record_path.parent,
            "report",
            seen,
        )
    )
    if artifact.source_binding.source_capture_ref != source_ref:
        raise ValueError("calibration.json source differs from its artifact")
    return run_id, source_ref, CalibrationComputation(artifact, report)


def _load_validated(
    calibrations_root: str | Path,
    captures_root: str | Path,
    reference: CalibrationArtifactRef,
    *,
    checkpoint: Callable[[], None] | None = None,
):
    if checkpoint is not None and not callable(checkpoint):
        raise TypeError("checkpoint must be callable or None")
    run_id, source_ref, computation = _read_record(calibrations_root, reference)
    if checkpoint is not None:
        checkpoint()
    capture = _load_capture(
        Path(captures_root).expanduser().resolve(),
        source_ref,
    )
    _validate_calibration_artifact_source_compatibility(
        computation.artifact,
        capture,
        checkpoint=checkpoint,
    )
    if checkpoint is not None:
        checkpoint()
    return run_id, computation


def load_calibration_artifact(
    calibrations_root: str | Path,
    captures_root: str | Path,
    reference: CalibrationArtifactRef,
    *,
    checkpoint: Callable[[], None] | None = None,
) -> ResolvedCalibration:
    """Cold-open one record and validate its exact Capture source."""

    run_id, computation = _load_validated(
        calibrations_root,
        captures_root,
        reference,
        checkpoint=checkpoint,
    )
    return ResolvedCalibration(reference, computation.artifact, run_id)


def load_calibration_computation(
    calibrations_root: str | Path,
    captures_root: str | Path,
    reference: CalibrationArtifactRef,
    *,
    checkpoint: Callable[[], None] | None = None,
):
    """Cold-open one validated CalibrationArtifact/CalibrationReport pair."""

    _run_id, computation = _load_validated(
        calibrations_root,
        captures_root,
        reference,
        checkpoint=checkpoint,
    )
    return computation


def write_calibration_artifact(
    calibrations_root: str | Path,
    result,
    *,
    run_id: str,
) -> CalibrationArtifactRef:
    """Write original-dtype arrays first and publish ``calibration.json`` last."""

    from .analysis import CalibrationAnalysisResult, CalibrationComputation

    if not isinstance(result, CalibrationAnalysisResult):
        raise TypeError("result must be CalibrationAnalysisResult")
    run_id = canonical_text(run_id, "run_id")
    resolved = result._source_resolution
    if not resolved.join.matches_contexts(result.report.group_contexts):
        raise ValueError(
            "calibration report group contexts differ from its CaptureArtifact"
        )
    if result.artifact.source_binding.source_capture_ref != result.source.ref:
        raise ValueError("calibration result source changed before persistence")

    root = Path(calibrations_root).expanduser().resolve()
    durable_makedirs(root)
    run_directory = durable_mkdir(resolve_under(root, _run_name()))
    arrays_directory = durable_mkdir(run_directory / "arrays")
    record_path = run_directory / "calibration.json"
    reference = CalibrationArtifactRef(record_path.relative_to(root).as_posix())

    arrays: list[tuple[str, np.ndarray]] = []
    artifact_tree = _externalize_arrays(
        calibration_artifact_to_tree(result.artifact),
        "artifact",
        arrays,
    )
    report_tree = _externalize_arrays(
        calibration_report_to_tree(result.report),
        "report",
        arrays,
    )
    for relative, array in arrays:
        target = resolve_under(run_directory, relative)
        if target.parent != arrays_directory:
            raise RuntimeError("Calibration array escaped its owned directory")
        _write_npy(target, array)

    seen: set[str] = set()
    round_trip_artifact = calibration_artifact_from_tree(
        _materialize_arrays(artifact_tree, run_directory, "artifact", seen)
    )
    round_trip_report = calibration_report_from_tree(
        _materialize_arrays(report_tree, run_directory, "report", seen)
    )
    CalibrationComputation(round_trip_artifact, round_trip_report)
    if round_trip_artifact.source_binding.source_capture_ref != result.source.ref:
        raise ValueError("Calibration array round-trip changed its source")

    record = {
        "schema": CALIBRATION_RECORD_FORMAT,
        "run_id": run_id,
        "source_capture_ref": capture_artifact_ref_to_tree(result.source.ref),
        "artifact": artifact_tree,
        "report": report_tree,
    }
    payload = encode(record)
    if encode(
        exact_mapping(
            decode(payload),
            _RECORD_FIELDS,
            CALIBRATION_RECORD_FORMAT,
        )
    ) != payload:
        raise ValueError("calibration.json failed its canonical round-trip")
    atomic_write_bytes(record_path, payload)
    return reference


def compile_calibration_artifact_plan(
    source_capture_ref: CaptureArtifactRef,
    captures_root: Path,
    calibrations_root: Path,
    request: CalibrationAnalysisRequest,
    *,
    expected_readout_binding: ReadoutBindingKey,
    timeout_seconds: float,
    on_committed: Callable[[CalibrationArtifactRef], None] | None = None,
) -> RunPlan:
    """Adapt one synchronous Calibration calculation to the generic RunPlan."""

    from .analysis import _analyze_calibration_resolved

    if not isinstance(source_capture_ref, CaptureArtifactRef):
        raise TypeError("source_capture_ref must be CaptureArtifactRef")
    if not isinstance(captures_root, Path) or not captures_root.is_absolute():
        raise ValueError("captures_root must be an absolute Path")
    if not isinstance(calibrations_root, Path) or not calibrations_root.is_absolute():
        raise ValueError("calibrations_root must be an absolute Path")
    if not isinstance(request, CalibrationAnalysisRequest):
        raise TypeError("request must be CalibrationAnalysisRequest")
    if request.expected_centers_xy is None:
        raise ValueError(
            "authoritative calibration requires independent expected_centers_xy "
            "and maximum_site_residual_px"
        )
    if not isinstance(expected_readout_binding, ReadoutBindingKey):
        raise TypeError("expected_readout_binding must be ReadoutBindingKey")
    if on_committed is not None and not callable(on_committed):
        raise TypeError("on_committed must be callable or None")
    timeout = positive_real(timeout_seconds, "timeout_seconds")
    capture_root = captures_root.resolve()
    calibration_root = calibrations_root.resolve()

    def preflight(context: RunContext) -> _PreparedCalibrationAnalysis:
        context.checkpoint()
        source = _load_capture(capture_root, source_capture_ref)
        if source.camera_provenance.binding != expected_readout_binding:
            raise ValueError(
                "source capture readout binding differs from the frozen request"
            )
        resolved = _resolve_calibration_source(
            source,
            request.layout,
            checkpoint=context.checkpoint,
        )
        context.checkpoint()
        return source, resolved

    def execute(
        context: RunContext,
        prepared: _PreparedCalibrationAnalysis,
    ):
        source, resolved = prepared
        context.checkpoint()
        result = _analyze_calibration_resolved(source, request, resolved)
        context.checkpoint()
        return result

    def cleanup(
        _context: RunContext,
        _prepared: _PreparedCalibrationAnalysis | None,
        _primary: BaseException | None,
    ) -> CleanupReport:
        return CleanupReport()

    def finalize(context: PostSafetyContext, result) -> CalibrationArtifactRef:
        if not isinstance(context, PostSafetyContext):
            raise TypeError("calibration finalize requires PostSafetyContext")
        reference = write_calibration_artifact(
            calibration_root,
            result,
            run_id=context.run_id.value,
        )
        if on_committed is not None:
            on_committed(reference)
        return reference

    return RunPlan(
        name="calibrate committed camera capture",
        resource_claims=(),
        bound_devices=(),
        preflight=preflight,
        execute=execute,
        cleanup=cleanup,
        finalize=finalize,
        timeout_seconds=timeout,
    )


__all__ = [
    "CALIBRATION_RECORD_FORMAT",
    "compile_calibration_artifact_plan",
    "load_calibration_artifact",
    "load_calibration_computation",
    "write_calibration_artifact",
]
