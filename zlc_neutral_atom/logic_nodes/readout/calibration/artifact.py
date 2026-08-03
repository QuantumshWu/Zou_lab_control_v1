"""Readable, record-last persistence for runtime readout calibrations.

The JSON record owns ordinary metadata.  Large scientific arrays keep their
native NumPy dtype in a small, explicit set of ``.npy`` files, and loading uses
only that committed calibration record.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import json
import math
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from zlc_data import AxisId, AxisSpec, ComponentValidity, CoordinateFrameId
from zlc_data.codec import (
    axis_from_tree,
    axis_to_tree,
    value_schema_from_tree,
    value_schema_to_tree,
)
from zlc_neutral_atom.capture.artifact import CaptureArtifact
from zlc_neutral_atom.capture.reference import CaptureArtifactRef
from zlc_neutral_atom.devices.camera.contract import ReadoutBindingKey
from zlc_neutral_atom.logic_nodes.readout.contracts import (
    CalibrationCaptureLayout,
    FrameContract,
)
from zlc_neutral_atom.logic_nodes.readout.model_contract import ReadoutModelKind
from zlc_neutral_atom.logic_nodes.readout.physical_context import (
    readout_physical_context_from_tree,
    readout_physical_context_to_tree,
)
from zlc_neutral_atom.runtime.cleanup import CleanupReport
from zlc_neutral_atom.runtime.run import PostSafetyContext, RunContext, RunPlan
from zlc_storage import canonical_text, positive_real
from zlc_storage.durability import (
    atomic_write_bytes,
    atomic_write_file,
    durable_makedirs,
    durable_mkdir,
)
from zlc_storage.paths import resolve_under

from .analysis import (
    CalibrationAnalysisResult,
    CalibrationComputation,
)
from .calibration import (
    BackgroundMode,
    BoxFeature,
    BoxReducer,
    CalibrationAnalysisRequest,
    CalibrationArtifact,
    CalibrationSourceBinding,
    GridOrder,
    PerSitePsfFeature,
    ReadoutModel,
    ResolvedCalibration,
    SiteMap,
    UniformPsfFeature,
    _ResolvedCalibrationSource,
    _resolve_calibration_source,
)
from .reference import CalibrationArtifactRef


CALIBRATION_RECORD_VERSION = 1
_CALIBRATION_DIRECTORY = "tasks/calibration"
_CalibrationAnalysisInputs = tuple[CaptureArtifact, _ResolvedCalibrationSource]


@dataclass(frozen=True, slots=True)
class CommittedCalibration:
    """One Analysis Run result retained by its enclosing Task operation."""

    reference: CalibrationArtifactRef
    result: CalibrationAnalysisResult

    def __post_init__(self) -> None:
        if not isinstance(self.reference, CalibrationArtifactRef):
            raise TypeError("reference must be CalibrationArtifactRef")
        if not isinstance(self.result, CalibrationAnalysisResult):
            raise TypeError("result must be CalibrationAnalysisResult")

def _mapping(value: object, fields: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"{name} has an unknown field set")
    return value


def _json_number(value: object) -> float | str:
    number = float(value)
    if math.isnan(number):
        return "nan"
    if math.isinf(number):
        return "+inf" if number > 0 else "-inf"
    return number


def _number(value: object, field: str) -> float:
    if isinstance(value, str):
        values = {"nan": math.nan, "+inf": math.inf, "-inf": -math.inf}
        try:
            return values[value]
        except KeyError as exc:
            raise ValueError(f"{field} is not a recorded number") from exc
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be a number")
    return float(value)


def _json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"{type(value).__name__} is not JSON metadata")


def _write_json(path: Path, value: dict[str, object]) -> None:
    payload = (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            default=_json_default,
        )
        + "\n"
    ).encode("utf-8")
    atomic_write_bytes(path, payload)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path.name} is not readable JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain one JSON object")
    return value


def _write_npy(path: Path, value: np.ndarray) -> None:
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise TypeError("Calibration arrays cannot contain Python objects")
    atomic_write_file(
        path,
        lambda stream: np.save(stream, array, allow_pickle=False),
    )


def _load_npy(path: Path, *, field: str) -> np.ndarray:
    value = np.load(path, allow_pickle=False)
    if not isinstance(value, np.ndarray) or value.dtype.hasobject:
        raise ValueError(f"{field} must be one non-object NumPy array")
    value.setflags(write=False)
    return value


def _run_directory(project_root: Path, run_id: str) -> Path:
    run = canonical_text(run_id, "run_id")
    path = PurePosixPath(run)
    if (
        path.is_absolute()
        or path.as_posix() != run
        or len(path.parts) != 1
        or path.name in {".", ".."}
    ):
        raise ValueError("run_id must be one safe path segment")
    output_root = resolve_under(project_root, _CALIBRATION_DIRECTORY)
    durable_makedirs(output_root)
    target = resolve_under(output_root, run)
    if target.exists():
        raise FileExistsError(f"Calibration run directory already exists: {target}")
    return durable_mkdir(target)


def _layout_to_json(value: CalibrationCaptureLayout) -> dict[str, object]:
    return {
        "readout_event_axis": value.readout_event_axis_id.value,
        "reference_events": list(value.reference_event_indices),
        "readout_event": value.readout_event_index,
    }


def _layout_from_json(value: object) -> CalibrationCaptureLayout:
    data = _mapping(
        value,
        {"readout_event_axis", "reference_events", "readout_event"},
        "calibration capture layout",
    )
    return CalibrationCaptureLayout(
        AxisId(data["readout_event_axis"]),
        tuple(data["reference_events"]),
        data["readout_event"],
    )


def _frame_to_json(value: FrameContract) -> dict[str, object]:
    return {
        "binding": value.binding.value,
        "camera_identity": value.camera_identity,
        "sensor_identity": value.sensor_identity,
        "optical_path": value.optical_path,
        "sensor_shape_yx": list(value.sensor_shape_yx),
        "roi_origin_yx": list(value.roi_origin_yx),
        "roi_shape_yx": list(value.roi_shape_yx),
        "binning_yx": list(value.binning_yx),
        "spatial_y_axis": value.spatial_y_axis_id.value,
        "spatial_x_axis": value.spatial_x_axis_id.value,
        "coordinate_frame": value.coordinate_frame.value,
        "dtype": value.dtype.str,
        "count_unit": value.count_unit,
        "exposure_seconds": value.exposure_seconds,
        "gain": value.gain,
        "readout_mode": value.readout_mode,
        "frame_schema": value_schema_to_tree(value.frame_schema),
    }


def _frame_from_json(value: object) -> FrameContract:
    fields = {
        "binding",
        "camera_identity",
        "sensor_identity",
        "optical_path",
        "sensor_shape_yx",
        "roi_origin_yx",
        "roi_shape_yx",
        "binning_yx",
        "spatial_y_axis",
        "spatial_x_axis",
        "coordinate_frame",
        "dtype",
        "count_unit",
        "exposure_seconds",
        "gain",
        "readout_mode",
        "frame_schema",
    }
    data = _mapping(value, fields, "calibration frame")
    return FrameContract(
        binding=ReadoutBindingKey(data["binding"]),
        camera_identity=data["camera_identity"],
        sensor_identity=data["sensor_identity"],
        optical_path=data["optical_path"],
        sensor_shape_yx=tuple(data["sensor_shape_yx"]),
        roi_origin_yx=tuple(data["roi_origin_yx"]),
        roi_shape_yx=tuple(data["roi_shape_yx"]),
        binning_yx=tuple(data["binning_yx"]),
        spatial_y_axis_id=AxisId(data["spatial_y_axis"]),
        spatial_x_axis_id=AxisId(data["spatial_x_axis"]),
        coordinate_frame=CoordinateFrameId(data["coordinate_frame"]),
        dtype=np.dtype(data["dtype"]),
        count_unit=data["count_unit"],
        exposure_seconds=data["exposure_seconds"],
        gain=data["gain"],
        readout_mode=data["readout_mode"],
        frame_schema=value_schema_from_tree(data["frame_schema"]),
    )


def _feature_to_json(feature, run_directory: Path) -> dict[str, object]:
    common: dict[str, object] = {
        "kind": feature.kind.value,
        "boxes_xywh": feature.boxes_xywh.tolist(),
        "valid_sites": feature.valid_sites.mask.tolist(),
    }
    if isinstance(feature, BoxFeature):
        common["reducer"] = feature.reducer.value
    elif isinstance(feature, PerSitePsfFeature):
        filename = "psf_kernels.npy"
        _write_npy(run_directory / filename, feature.kernels)
        common.update(
            {
                "kernels_file": filename,
                "background": feature.background.value,
                "background_padding": feature.background_padding,
            }
        )
    elif isinstance(feature, UniformPsfFeature):
        common.update(
            {
                "kernel": feature.kernel.tolist(),
                "background": feature.background.value,
                "background_padding": feature.background_padding,
            }
        )
    else:
        raise TypeError("unsupported calibration feature")
    return common


def _feature_from_json(
    value: object,
    site_axis: AxisSpec,
    run_directory: Path,
):
    if not isinstance(value, dict):
        raise ValueError("calibration feature must be a JSON object")
    kind = ReadoutModelKind(value.get("kind"))
    validity = ComponentValidity(
        (site_axis.axis_id,),
        np.asarray(value.get("valid_sites"), dtype=bool),
    )
    boxes = np.asarray(value.get("boxes_xywh"))
    if kind is ReadoutModelKind.BOX:
        _mapping(value, {"kind", "boxes_xywh", "valid_sites", "reducer"}, "box feature")
        return BoxFeature(site_axis, boxes, BoxReducer(value["reducer"]), validity)
    common = {
        "kind", "boxes_xywh", "valid_sites", "background", "background_padding"
    }
    if kind is ReadoutModelKind.PER_SITE_PSF:
        _mapping(value, common | {"kernels_file"}, "per-site PSF feature")
        if value["kernels_file"] != "psf_kernels.npy":
            raise ValueError("per-site PSF kernels use an unknown file")
        return PerSitePsfFeature(
            site_axis,
            boxes,
            _load_npy(run_directory / "psf_kernels.npy", field="PSF kernels"),
            BackgroundMode(value["background"]),
            value["background_padding"],
            validity,
        )
    _mapping(value, common | {"kernel"}, "uniform PSF feature")
    return UniformPsfFeature(
        site_axis,
        boxes,
        np.asarray(value["kernel"]),
        BackgroundMode(value["background"]),
        value["background_padding"],
        validity,
    )


def _artifact_to_json(
    value: CalibrationArtifact,
    run_directory: Path,
) -> dict[str, object]:
    site_map = value.site_map
    models = []
    for model in value.models:
        models.append(
            {
                "feature": _feature_to_json(model.feature, run_directory),
                "thresholds": [_json_number(item) for item in model.thresholds],
                "usable_sites": model.usable_sites.mask.tolist(),
            }
        )
    return {
        "source_capture": value.source_binding.source_capture_ref.record_path,
        "layout": _layout_to_json(value.source_binding.layout),
        "frame": _frame_to_json(value.frame_contract),
        "readout_physical_context": readout_physical_context_to_tree(
            value.readout_physical_context
        ),
        "site_map": {
            "site_axis": axis_to_tree(site_map.site_axis),
            "coordinates_xy": site_map.coordinates_xy.tolist(),
            "grid_shape_yx": list(site_map.grid_shape_yx),
            "ordering": site_map.ordering.value,
            "coordinate_frame": site_map.coordinate_frame.value,
            "validity": site_map.validity.mask.tolist(),
        },
        "models": models,
        "default_model_kind": value.default_model_kind.value,
    }


def _artifact_from_json(value: object, run_directory: Path) -> CalibrationArtifact:
    data = _mapping(
        value,
        {
            "source_capture", "layout", "frame", "readout_physical_context",
            "site_map", "models", "default_model_kind",
        },
        "calibration artifact",
    )
    site_data = _mapping(
        data["site_map"],
        {
            "site_axis", "coordinates_xy", "grid_shape_yx", "ordering",
            "coordinate_frame", "validity",
        },
        "site map",
    )
    site_axis = axis_from_tree(site_data["site_axis"])
    site_map = SiteMap(
        site_axis,
        np.asarray(site_data["coordinates_xy"]),
        tuple(site_data["grid_shape_yx"]),
        GridOrder(site_data["ordering"]),
        CoordinateFrameId(site_data["coordinate_frame"]),
        ComponentValidity(
            (site_axis.axis_id,),
            np.asarray(site_data["validity"], dtype=bool),
        ),
    )
    models = []
    if not isinstance(data["models"], list):
        raise TypeError("calibration models must be a list")
    for item in data["models"]:
        model_data = _mapping(
            item,
            {"feature", "thresholds", "usable_sites"},
            "readout model",
        )
        models.append(
            ReadoutModel(
                _feature_from_json(model_data["feature"], site_axis, run_directory),
                np.asarray(
                    [_number(number, "threshold") for number in model_data["thresholds"]]
                ),
                ComponentValidity(
                    (site_axis.axis_id,),
                    np.asarray(model_data["usable_sites"], dtype=bool),
                ),
            )
        )
    return CalibrationArtifact(
        CalibrationSourceBinding(
            CaptureArtifactRef(data["source_capture"]),
            _layout_from_json(data["layout"]),
        ),
        _frame_from_json(data["frame"]),
        readout_physical_context_from_tree(data["readout_physical_context"]),
        site_map,
        tuple(models),
        ReadoutModelKind(data["default_model_kind"]),
    )


def _read_record(
    project_root: str | Path,
    reference: CalibrationArtifactRef,
) -> tuple[str, CalibrationArtifact]:
    if not isinstance(reference, CalibrationArtifactRef):
        raise TypeError("reference must be CalibrationArtifactRef")
    record_path = resolve_under(project_root, reference.record_path)
    record = _mapping(
        _read_json(record_path),
        {"format_version", "run_id", "artifact"},
        "calibration record",
    )
    if record["format_version"] != CALIBRATION_RECORD_VERSION:
        raise ValueError("calibration.json format_version is not current")
    run_id = canonical_text(record["run_id"], "run_id")
    if PurePosixPath(reference.record_path).parts[2] != run_id:
        raise ValueError("Calibration reference path differs from its run_id")
    artifact = _artifact_from_json(record["artifact"], record_path.parent)
    return run_id, artifact


def load_calibration_artifact(
    project_root: str | Path,
    reference: CalibrationArtifactRef,
    *,
    checkpoint: Callable[[], None] | None = None,
) -> ResolvedCalibration:
    """Load one self-contained runtime calibration without reopening raw frames."""

    if checkpoint is not None:
        checkpoint()
    run_id, artifact = _read_record(project_root, reference)
    if checkpoint is not None:
        checkpoint()
    return ResolvedCalibration(reference, artifact, run_id)


def write_calibration_artifact(
    project_root: str | Path,
    result: CalibrationAnalysisResult,
    *,
    run_id: str,
) -> CalibrationArtifactRef:
    """Write native arrays first and publish readable ``calibration.json`` last."""

    if not isinstance(result, CalibrationAnalysisResult):
        raise TypeError("result must be CalibrationAnalysisResult")
    CalibrationComputation(result.artifact, result.report)
    if result.artifact.source_binding.source_capture_ref != result.source.ref:
        raise ValueError("calibration result source changed before persistence")
    project = Path(project_root).expanduser()
    if not project.is_absolute():
        raise ValueError("project_root must be absolute")
    project = project.resolve()
    run_directory = _run_directory(project, run_id)
    artifact = _artifact_to_json(result.artifact, run_directory)
    record_path = run_directory / "calibration.json"
    _write_json(
        record_path,
        {
            "format_version": CALIBRATION_RECORD_VERSION,
            "run_id": canonical_text(run_id, "run_id"),
            "artifact": artifact,
        },
    )
    return CalibrationArtifactRef(record_path.relative_to(project).as_posix())


def _load_capture(project_root: Path, reference: CaptureArtifactRef) -> CaptureArtifact:
    from zlc_neutral_atom.capture.artifact import load_capture_artifact

    return load_capture_artifact(project_root, reference, materialize=False)


def compile_calibration_analysis_plan(
    source_capture_ref: CaptureArtifactRef,
    project_root: Path,
    request: CalibrationAnalysisRequest,
    *,
    expected_readout_binding: ReadoutBindingKey,
    timeout_seconds: float,
) -> RunPlan:
    """Compile the second flat Run and retain its report for Task outputs."""

    from .analysis import _analyze_calibration_resolved

    timeout = positive_real(timeout_seconds, "timeout_seconds")

    def preflight(context: RunContext) -> _CalibrationAnalysisInputs:
        context.checkpoint()
        source = _load_capture(project_root, source_capture_ref)
        if source.camera_binding != expected_readout_binding:
            raise ValueError("source capture readout binding differs from the request")
        resolved = _resolve_calibration_source(
            source,
            request.layout,
            checkpoint=context.checkpoint,
        )
        return source, resolved

    def execute(context: RunContext, prepared: _CalibrationAnalysisInputs):
        source, resolved = prepared
        context.checkpoint()
        return _analyze_calibration_resolved(source, request, resolved)

    def cleanup(
        _context: RunContext,
        _prepared: _CalibrationAnalysisInputs | None,
        _primary: BaseException | None,
    ) -> CleanupReport:
        return CleanupReport()

    def finalize(context: PostSafetyContext, result) -> CommittedCalibration:
        reference = write_calibration_artifact(
            project_root,
            result,
            run_id=context.run_id.value,
        )
        return CommittedCalibration(reference, result)

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
    "CALIBRATION_RECORD_VERSION",
    "CommittedCalibration",
    "compile_calibration_analysis_plan",
    "load_calibration_artifact",
    "write_calibration_artifact",
]
