"""Canonical durable formats for readout calibration artifacts and reports.

Only the two repository values cross this boundary.  Nested values remain
private implementation details and delegate foreign values to their owner
serializers.  Scientific validation belongs to the domain constructors; this
module enforces exact field sets, scalar types, and canonical bytes.
"""

from __future__ import annotations

from numbers import Integral, Real
from typing import TYPE_CHECKING, Any

import numpy as np

from zlc_data import AxisId, AxisSpec, ComponentValidity, CoordinateFrameId
from zlc_data.codec import axis_from_tree, axis_to_tree, validity_from_tree, validity_to_tree
from zlc_neutral_atom.capture.reference import (
    capture_artifact_ref_from_tree,
    capture_artifact_ref_to_tree,
)
from zlc_storage import (
    ContentRef,
    canonical_text as _text,
    content_ref_from_tree,
    content_ref_to_tree,
    decode,
    encode,
    exact_mapping as _exact_map,
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
    ReadoutFeature,
    ReadoutModel,
    ReadoutModelKind,
    ThresholdMethod,
    SiteMap,
    UniformPsfFeature,
)

if TYPE_CHECKING:
    from .analysis import (
        AblationPoint,
        BimodalFit,
        CalibrationReport,
        ModelCalibrationReport,
        PsfFitDiagnostic,
        ReferenceLabels,
        SiteFidelity,
        TrainTestSplit,
    )
from zlc_neutral_atom.logic_nodes.readout.codec import (
    calibration_capture_layout_from_tree,
    calibration_capture_layout_to_tree,
    frame_contract_from_tree,
    frame_contract_to_tree,
)
from zlc_neutral_atom.logic_nodes.readout.physical_context import (
    ReadoutPhysicalContext,
    readout_physical_context_from_tree,
    readout_physical_context_to_tree,
)


CALIBRATION_ARTIFACT_FORMAT = (
    "zlc_neutral_atom.logic_nodes.readout.calibration.artifact"
)
CALIBRATION_REPORT_FORMAT = "zlc_neutral_atom.logic_nodes.readout.calibration.report"

def _analysis_types():
    """Load report-only scientific values only on a report code path."""

    from . import analysis

    return analysis


def _fields(tree: Any, fields: set[str], name: str) -> dict[str, Any]:
    return _exact_map(tree, fields, name, discriminator=None)


def _outer_fields(tree: Any, fields: set[str], format_name: str) -> dict[str, Any]:
    return _exact_map(tree, fields, format_name, discriminator="format")


def _list(value: Any, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    return value


def _array(value: Any, field: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise ValueError(f"{field} must be a canonical ndarray")
    return value


def _bool(value: Any, field: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{field} must be a boolean")
    return value


def _int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{field} must be an integer")
    return int(value)


def _optional_int(value: Any, field: str) -> int | None:
    return None if value is None else _int(value, field)


def _real(value: Any, field: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{field} must be a real number")
    return float(value)


def _optional_real(value: Any, field: str) -> float | None:
    return None if value is None else _real(value, field)


def _enum(enum_type, value: Any, field: str):
    try:
        return enum_type(_text(value, field))
    except ValueError as exc:
        raise ValueError(f"{field} has an unknown value {value!r}") from exc


def _integer_pair(value: Any, field: str) -> tuple[int, int]:
    items = _list(value, field)
    if len(items) != 2:
        raise ValueError(f"{field} must contain two integers")
    return _int(items[0], f"{field}[0]"), _int(items[1], f"{field}[1]")


def _real_pair(value: Any, field: str) -> tuple[float, float]:
    items = _list(value, field)
    if len(items) != 2:
        raise ValueError(f"{field} must contain two real numbers")
    return _real(items[0], f"{field}[0]"), _real(items[1], f"{field}[1]")


def _component_validity(tree: Any, field: str) -> ComponentValidity:
    value = validity_from_tree(tree)
    if not isinstance(value, ComponentValidity):
        raise ValueError(f"{field} must be ComponentValidity")
    return value


def _decode_typed(
    payload: bytes | bytearray | memoryview,
    parser,
    projector,
    name: str,
):
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise TypeError(f"{name} payload must be bytes-like")
    raw = bytes(payload)
    value = parser(decode(raw))
    if encode(projector(value)) != raw:
        raise ValueError(f"{name} payload is typed but non-canonical")
    return value


def _source_to_tree(value: CalibrationSourceBinding) -> dict[str, Any]:
    if not isinstance(value, CalibrationSourceBinding):
        raise TypeError("value must be CalibrationSourceBinding")
    return {
        "source_capture_ref": capture_artifact_ref_to_tree(value.source_capture_ref),
        "layout": calibration_capture_layout_to_tree(value.layout),
    }


def _source_from_tree(tree: Any) -> CalibrationSourceBinding:
    data = _fields(tree, {"source_capture_ref", "layout"}, "source binding")
    return CalibrationSourceBinding(
        source_capture_ref=capture_artifact_ref_from_tree(data["source_capture_ref"]),
        layout=calibration_capture_layout_from_tree(data["layout"]),
    )


def _site_map_to_tree(value: SiteMap) -> dict[str, Any]:
    if not isinstance(value, SiteMap):
        raise TypeError("value must be SiteMap")
    return {
        "site_axis": axis_to_tree(value.site_axis),
        "coordinates_xy": value.coordinates_xy,
        "grid_shape_yx": list(value.grid_shape_yx),
        "ordering": value.ordering.value,
        "coordinate_frame": value.coordinate_frame.value,
        "validity": validity_to_tree(value.validity),
    }


def _site_map_from_tree(tree: Any) -> SiteMap:
    data = _fields(
        tree,
        {
            "site_axis",
            "coordinates_xy",
            "grid_shape_yx",
            "ordering",
            "coordinate_frame",
            "validity",
        },
        "site map",
    )
    axis = axis_from_tree(data["site_axis"])
    if not isinstance(axis, AxisSpec):
        raise ValueError("site_axis must decode to AxisSpec")
    return SiteMap(
        site_axis=axis,
        coordinates_xy=_array(data["coordinates_xy"], "coordinates_xy"),
        grid_shape_yx=_integer_pair(data["grid_shape_yx"], "grid_shape_yx"),
        ordering=_enum(GridOrder, data["ordering"], "ordering"),
        coordinate_frame=CoordinateFrameId(
            _text(data["coordinate_frame"], "coordinate_frame")
        ),
        validity=_component_validity(data["validity"], "validity"),
    )


def _feature_to_tree(value: ReadoutFeature) -> dict[str, Any]:
    if isinstance(value, BoxFeature):
        return {
            "kind": value.kind.value,
            "boxes_xywh": value.boxes_xywh,
            "reducer": value.reducer.value,
            "valid_sites": validity_to_tree(value.valid_sites),
        }
    if isinstance(value, PerSitePsfFeature):
        return {
            "kind": value.kind.value,
            "boxes_xywh": value.boxes_xywh,
            "kernels": value.kernels,
            "background": value.background.value,
            "background_padding": value.background_padding,
            "valid_sites": validity_to_tree(value.valid_sites),
        }
    if isinstance(value, UniformPsfFeature):
        return {
            "kind": value.kind.value,
            "boxes_xywh": value.boxes_xywh,
            "kernel": value.kernel,
            "background": value.background.value,
            "background_padding": value.background_padding,
            "valid_sites": validity_to_tree(value.valid_sites),
        }
    raise TypeError("value must be a closed ReadoutFeature")


def _feature_from_tree(tree: Any, site_axis: AxisSpec) -> ReadoutFeature:
    if not isinstance(tree, dict) or "kind" not in tree:
        raise ValueError("readout feature must be a tagged mapping")
    kind = _enum(ReadoutModelKind, tree["kind"], "feature.kind")
    if kind is ReadoutModelKind.BOX:
        data = _fields(
            tree,
            {"kind", "boxes_xywh", "reducer", "valid_sites"},
            "box feature",
        )
        return BoxFeature(
            site_axis=site_axis,
            boxes_xywh=_array(data["boxes_xywh"], "boxes_xywh"),
            reducer=_enum(BoxReducer, data["reducer"], "reducer"),
            valid_sites=_component_validity(data["valid_sites"], "valid_sites"),
        )
    common = {
        "kind",
        "boxes_xywh",
        "background",
        "background_padding",
        "valid_sites",
    }
    if kind is ReadoutModelKind.PER_SITE_PSF:
        data = _fields(tree, common | {"kernels"}, "per-site PSF feature")
        return PerSitePsfFeature(
            site_axis=site_axis,
            boxes_xywh=_array(data["boxes_xywh"], "boxes_xywh"),
            kernels=_array(data["kernels"], "kernels"),
            background=_enum(BackgroundMode, data["background"], "background"),
            background_padding=_int(data["background_padding"], "background_padding"),
            valid_sites=_component_validity(data["valid_sites"], "valid_sites"),
        )
    data = _fields(tree, common | {"kernel"}, "uniform PSF feature")
    return UniformPsfFeature(
        site_axis=site_axis,
        boxes_xywh=_array(data["boxes_xywh"], "boxes_xywh"),
        kernel=_array(data["kernel"], "kernel"),
        background=_enum(BackgroundMode, data["background"], "background"),
        background_padding=_int(data["background_padding"], "background_padding"),
        valid_sites=_component_validity(data["valid_sites"], "valid_sites"),
    )


def _model_to_tree(value: ReadoutModel) -> dict[str, Any]:
    if not isinstance(value, ReadoutModel):
        raise TypeError("value must be ReadoutModel")
    return {
        "feature": _feature_to_tree(value.feature),
        "thresholds": value.thresholds,
        "usable_sites": validity_to_tree(value.usable_sites),
    }


def _model_from_tree(tree: Any, site_axis: AxisSpec) -> ReadoutModel:
    data = _fields(tree, {"feature", "thresholds", "usable_sites"}, "readout model")
    return ReadoutModel(
        feature=_feature_from_tree(data["feature"], site_axis),
        thresholds=_array(data["thresholds"], "thresholds"),
        usable_sites=_component_validity(data["usable_sites"], "usable_sites"),
    )


def _artifact_to_tree(value: CalibrationArtifact) -> dict[str, Any]:
    if not isinstance(value, CalibrationArtifact):
        raise TypeError("value must be CalibrationArtifact")
    return {
        "format": CALIBRATION_ARTIFACT_FORMAT,
        "source": _source_to_tree(value.source_binding),
        "frame_contract": frame_contract_to_tree(value.frame_contract),
        "readout_physical_context": readout_physical_context_to_tree(
            value.readout_physical_context
        ),
        "site_map": _site_map_to_tree(value.site_map),
        "models": [_model_to_tree(model) for model in value.models],
        "default_model_kind": value.default_model_kind.value,
    }


def _artifact_from_tree(tree: Any) -> CalibrationArtifact:
    data = _outer_fields(
        tree,
        {
            "format",
            "source",
            "frame_contract",
            "readout_physical_context",
            "site_map",
            "models",
            "default_model_kind",
        },
        CALIBRATION_ARTIFACT_FORMAT,
    )
    site_map = _site_map_from_tree(data["site_map"])
    return CalibrationArtifact(
        source_binding=_source_from_tree(data["source"]),
        frame_contract=frame_contract_from_tree(data["frame_contract"]),
        readout_physical_context=readout_physical_context_from_tree(
            data["readout_physical_context"]
        ),
        site_map=site_map,
        models=tuple(
            _model_from_tree(item, site_map.site_axis)
            for item in _list(data["models"], "models")
        ),
        default_model_kind=_enum(
            ReadoutModelKind,
            data["default_model_kind"],
            "default_model_kind",
        ),
    )


def _request_to_tree(value: CalibrationAnalysisRequest) -> dict[str, Any]:
    if not isinstance(value, CalibrationAnalysisRequest):
        raise TypeError("value must be CalibrationAnalysisRequest")
    return {
        "layout": calibration_capture_layout_to_tree(value.layout),
        "grid_shape_yx": list(value.grid_shape_yx),
        "ordering": value.ordering.value,
        "box_radius": value.box_radius,
        "box_reducer": value.box_reducer.value,
        "psf_half_width": value.psf_half_width,
        "psf_background": value.psf_background.value,
        "psf_background_padding": value.psf_background_padding,
        "model_kinds": [kind.value for kind in value.model_kinds],
        "default_model_kind": value.default_model_kind.value,
        "threshold_method": value.threshold_method.value,
        "train_fraction": value.train_fraction,
        "split_seed": value.split_seed,
        "histogram_bins": value.histogram_bins,
        "minimum_site_fidelity": value.minimum_site_fidelity,
        "max_drop": value.max_drop,
        "detector_min_distance": value.detector_min_distance,
        "detector_threshold_rel": value.detector_threshold_rel,
        "detector_refine_half": value.detector_refine_half,
        "expected_centers_xy": value.expected_centers_xy,
        "maximum_site_residual_px": value.maximum_site_residual_px,
    }


def _request_from_tree(tree: Any) -> CalibrationAnalysisRequest:
    fields = {
        "layout",
        "grid_shape_yx",
        "ordering",
        "box_radius",
        "box_reducer",
        "psf_half_width",
        "psf_background",
        "psf_background_padding",
        "model_kinds",
        "default_model_kind",
        "threshold_method",
        "train_fraction",
        "split_seed",
        "histogram_bins",
        "minimum_site_fidelity",
        "max_drop",
        "detector_min_distance",
        "detector_threshold_rel",
        "detector_refine_half",
        "expected_centers_xy",
        "maximum_site_residual_px",
    }
    data = _fields(tree, fields, "calibration analysis request")
    return CalibrationAnalysisRequest(
        layout=calibration_capture_layout_from_tree(data["layout"]),
        grid_shape_yx=_integer_pair(data["grid_shape_yx"], "grid_shape_yx"),
        ordering=_enum(GridOrder, data["ordering"], "ordering"),
        box_radius=_int(data["box_radius"], "box_radius"),
        box_reducer=_enum(BoxReducer, data["box_reducer"], "box_reducer"),
        psf_half_width=_int(data["psf_half_width"], "psf_half_width"),
        psf_background=_enum(
            BackgroundMode,
            data["psf_background"],
            "psf_background",
        ),
        psf_background_padding=_int(
            data["psf_background_padding"],
            "psf_background_padding",
        ),
        model_kinds=tuple(
            _enum(ReadoutModelKind, item, "model_kinds entry")
            for item in _list(data["model_kinds"], "model_kinds")
        ),
        default_model_kind=_enum(
            ReadoutModelKind,
            data["default_model_kind"],
            "default_model_kind",
        ),
        threshold_method=_enum(
            ThresholdMethod,
            data["threshold_method"],
            "threshold_method",
        ),
        train_fraction=_real(data["train_fraction"], "train_fraction"),
        split_seed=_int(data["split_seed"], "split_seed"),
        histogram_bins=_int(data["histogram_bins"], "histogram_bins"),
        minimum_site_fidelity=_real(
            data["minimum_site_fidelity"],
            "minimum_site_fidelity",
        ),
        max_drop=_int(data["max_drop"], "max_drop"),
        detector_min_distance=_optional_int(
            data["detector_min_distance"],
            "detector_min_distance",
        ),
        detector_threshold_rel=_real(
            data["detector_threshold_rel"],
            "detector_threshold_rel",
        ),
        detector_refine_half=_int(
            data["detector_refine_half"],
            "detector_refine_half",
        ),
        expected_centers_xy=(
            None
            if data["expected_centers_xy"] is None
            else _array(data["expected_centers_xy"], "expected_centers_xy")
        ),
        maximum_site_residual_px=_optional_real(
            data["maximum_site_residual_px"],
            "maximum_site_residual_px",
        ),
    )


_BIMODAL_FLOATS = (
    "threshold",
    "fidelity",
    "dark_mean",
    "dark_sigma",
    "bright_mean",
    "bright_sigma",
    "bright_fraction",
    "dark_fidelity",
    "bright_fidelity",
)


def _bimodal_to_tree(value: BimodalFit) -> dict[str, Any]:
    analysis = _analysis_types()
    if not isinstance(value, analysis.BimodalFit):
        raise TypeError("value must be BimodalFit")
    tree = {field: _real(getattr(value, field), field) for field in _BIMODAL_FLOATS}
    tree.update(
        {
            "bright_above": _bool(value.bright_above, "bright_above"),
            "ok": _bool(value.ok, "ok"),
        }
    )
    return tree


def _bimodal_from_tree(tree: Any) -> BimodalFit:
    analysis = _analysis_types()
    data = _fields(
        tree,
        set(_BIMODAL_FLOATS) | {"bright_above", "ok"},
        "bimodal fit",
    )
    values = {field: _real(data[field], field) for field in _BIMODAL_FLOATS}
    return analysis.BimodalFit(
        **values,
        bright_above=_bool(data["bright_above"], "bright_above"),
        ok=_bool(data["ok"], "ok"),
    )


def _labels_to_tree(value: ReferenceLabels) -> dict[str, Any]:
    analysis = _analysis_types()
    if not isinstance(value, analysis.ReferenceLabels):
        raise TypeError("value must be ReferenceLabels")
    return {
        "occupied": value.occupied,
        "dark": value.dark,
        "valid": value.valid,
        "fits": [_bimodal_to_tree(item) for item in value.fits],
        "n_reference_shots": value.n_reference_shots,
    }


def _labels_from_tree(tree: Any) -> ReferenceLabels:
    analysis = _analysis_types()
    data = _fields(
        tree,
        {"occupied", "dark", "valid", "fits", "n_reference_shots"},
        "reference labels",
    )
    return analysis.ReferenceLabels(
        occupied=_array(data["occupied"], "occupied"),
        dark=_array(data["dark"], "dark"),
        valid=_array(data["valid"], "valid"),
        fits=tuple(_bimodal_from_tree(item) for item in _list(data["fits"], "fits")),
        n_reference_shots=_int(data["n_reference_shots"], "n_reference_shots"),
    )


def _split_to_tree(value: TrainTestSplit) -> dict[str, Any]:
    analysis = _analysis_types()
    if not isinstance(value, analysis.TrainTestSplit):
        raise TypeError("value must be TrainTestSplit")
    return {
        "train": value.train,
        "test": value.test,
        "seed": value.seed,
        "train_fraction": value.train_fraction,
    }


def _split_from_tree(tree: Any) -> TrainTestSplit:
    analysis = _analysis_types()
    data = _fields(tree, {"train", "test", "seed", "train_fraction"}, "train/test split")
    return analysis.TrainTestSplit(
        train=_array(data["train"], "train"),
        test=_array(data["test"], "test"),
        seed=_int(data["seed"], "seed"),
        train_fraction=_real(data["train_fraction"], "train_fraction"),
    )


_SITE_FIDELITY_FLOATS = (
    "threshold",
    "fidelity",
    "fidelity_dark",
    "fidelity_bright",
    "model_fidelity",
    "dark_mean",
    "dark_sigma",
    "bright_mean",
    "bright_sigma",
)
_SITE_FIDELITY_INTS = ("site", "n_test", "n_train_dark", "n_train_bright")


def _site_fidelity_to_tree(value: SiteFidelity) -> dict[str, Any]:
    analysis = _analysis_types()
    if not isinstance(value, analysis.SiteFidelity):
        raise TypeError("value must be SiteFidelity")
    tree = {field: _real(getattr(value, field), field) for field in _SITE_FIDELITY_FLOATS}
    tree.update({field: _int(getattr(value, field), field) for field in _SITE_FIDELITY_INTS})
    tree["bright_above"] = _bool(value.bright_above, "bright_above")
    return tree


def _site_fidelity_from_tree(tree: Any) -> SiteFidelity:
    analysis = _analysis_types()
    fields = set(_SITE_FIDELITY_FLOATS) | set(_SITE_FIDELITY_INTS) | {"bright_above"}
    data = _fields(tree, fields, "site fidelity")
    values = {field: _real(data[field], field) for field in _SITE_FIDELITY_FLOATS}
    values.update({field: _int(data[field], field) for field in _SITE_FIDELITY_INTS})
    return analysis.SiteFidelity(
        **values,
        bright_above=_bool(data["bright_above"], "bright_above"),
    )


def _ablation_to_tree(value: AblationPoint) -> dict[str, Any]:
    analysis = _analysis_types()
    if not isinstance(value, analysis.AblationPoint):
        raise TypeError("value must be AblationPoint")
    return {
        "drop_worst_k": value.drop_worst_k,
        "excluded_sites": value.excluded_sites,
        "fidelity": _real(value.fidelity, "fidelity"),
        "errors": value.errors,
        "n_valid": value.n_valid,
    }


def _ablation_from_tree(tree: Any) -> AblationPoint:
    analysis = _analysis_types()
    data = _fields(
        tree,
        {"drop_worst_k", "excluded_sites", "fidelity", "errors", "n_valid"},
        "ablation point",
    )
    return analysis.AblationPoint(
        drop_worst_k=_int(data["drop_worst_k"], "drop_worst_k"),
        excluded_sites=_array(data["excluded_sites"], "excluded_sites"),
        fidelity=_real(data["fidelity"], "fidelity"),
        errors=_int(data["errors"], "errors"),
        n_valid=_int(data["n_valid"], "n_valid"),
    )


def _model_report_to_tree(value: ModelCalibrationReport) -> dict[str, Any]:
    analysis = _analysis_types()
    if not isinstance(value, analysis.ModelCalibrationReport):
        raise TypeError("value must be ModelCalibrationReport")
    return {
        "kind": value.kind.value,
        "quick_thresholds": value.quick_thresholds,
        "short_signals": value.short_signals,
        "short_validity": value.short_validity,
        "bin_edges": value.bin_edges,
        "predictions": value.predictions,
        "site_fidelity": [_site_fidelity_to_tree(item) for item in value.site_fidelity],
        "aggregate_fidelity": _real(value.aggregate_fidelity, "aggregate_fidelity"),
        "global_threshold": _real(value.global_threshold, "global_threshold"),
        "global_bright_above": _bool(
            value.global_bright_above,
            "global_bright_above",
        ),
        "global_fidelity": _real(value.global_fidelity, "global_fidelity"),
        "ablation": [_ablation_to_tree(item) for item in value.ablation],
    }


def _model_report_from_tree(tree: Any) -> ModelCalibrationReport:
    analysis = _analysis_types()
    fields = {
        "kind",
        "quick_thresholds",
        "short_signals",
        "short_validity",
        "bin_edges",
        "predictions",
        "site_fidelity",
        "aggregate_fidelity",
        "global_threshold",
        "global_bright_above",
        "global_fidelity",
        "ablation",
    }
    data = _fields(tree, fields, "model calibration report")
    return analysis.ModelCalibrationReport(
        kind=_enum(ReadoutModelKind, data["kind"], "kind"),
        quick_thresholds=_array(data["quick_thresholds"], "quick_thresholds"),
        short_signals=_array(data["short_signals"], "short_signals"),
        short_validity=_array(data["short_validity"], "short_validity"),
        bin_edges=_array(data["bin_edges"], "bin_edges"),
        predictions=_array(data["predictions"], "predictions"),
        site_fidelity=tuple(
            _site_fidelity_from_tree(item)
            for item in _list(data["site_fidelity"], "site_fidelity")
        ),
        aggregate_fidelity=_real(data["aggregate_fidelity"], "aggregate_fidelity"),
        global_threshold=_real(data["global_threshold"], "global_threshold"),
        global_bright_above=_bool(
            data["global_bright_above"],
            "global_bright_above",
        ),
        global_fidelity=_real(data["global_fidelity"], "global_fidelity"),
        ablation=tuple(
            _ablation_from_tree(item) for item in _list(data["ablation"], "ablation")
        ),
    )


def _psf_fit_to_tree(value: PsfFitDiagnostic) -> dict[str, Any]:
    analysis = _analysis_types()
    if not isinstance(value, analysis.PsfFitDiagnostic):
        raise TypeError("value must be PsfFitDiagnostic")
    return {
        "site": _int(value.site, "site"),
        "center_xy": [_real(item, "center_xy entry") for item in value.center_xy],
        "sigma_xy": [_real(item, "sigma_xy entry") for item in value.sigma_xy],
        "fit_ok": _bool(value.fit_ok, "fit_ok"),
    }


def _psf_fit_from_tree(tree: Any) -> PsfFitDiagnostic:
    analysis = _analysis_types()
    data = _fields(tree, {"site", "center_xy", "sigma_xy", "fit_ok"}, "PSF fit diagnostic")
    return analysis.PsfFitDiagnostic(
        site=_int(data["site"], "site"),
        center_xy=_real_pair(data["center_xy"], "center_xy"),
        sigma_xy=_real_pair(data["sigma_xy"], "sigma_xy"),
        fit_ok=_bool(data["fit_ok"], "fit_ok"),
    )


def _lineage_to_tree(value: tuple[tuple[str, str], ...]) -> list[list[str]]:
    return [
        [_text(name, "software lineage name"), _text(version, "software lineage version")]
        for name, version in value
    ]


def _lineage_from_tree(tree: Any) -> tuple[tuple[str, str], ...]:
    pairs = []
    for index, item in enumerate(_list(tree, "software_lineage")):
        values = _list(item, f"software_lineage[{index}]")
        if len(values) != 2:
            raise ValueError(f"software_lineage[{index}] must contain name and version")
        pairs.append(
            (
                _text(values[0], f"software_lineage[{index}].name"),
                _text(values[1], f"software_lineage[{index}].version"),
            )
        )
    return tuple(pairs)


def _group_contexts_to_tree(
    contexts: tuple[tuple[tuple[AxisId, int], ...], ...],
) -> list[list[list[Any]]]:
    return [
        [[axis_id.value, index] for axis_id, index in context]
        for context in contexts
    ]


def _group_contexts_from_tree(
    tree: Any,
) -> tuple[tuple[tuple[AxisId, int], ...], ...]:
    contexts = []
    for group, raw_context in enumerate(_list(tree, "group_contexts")):
        context = []
        for item_index, raw_item in enumerate(
            _list(raw_context, f"group_contexts[{group}]")
        ):
            item = _list(raw_item, f"group_contexts[{group}][{item_index}]")
            if len(item) != 2:
                raise ValueError("group context entries must contain axis id and index")
            context.append(
                (
                    AxisId(_text(item[0], "group context axis id")),
                    _int(item[1], "group context index"),
                )
            )
        contexts.append(tuple(context))
    return tuple(contexts)


_REPORT_FIELDS = {
    "format",
    "request",
    "software_lineage",
    "group_contexts",
    "reference_average_blob",
    "reference_average_validity_blob",
    "reference_box_signals",
    "labels",
    "split",
    "psf_fits",
    "models",
}


def _report_to_tree(
    value: CalibrationReport,
    reference_average_blob: ContentRef,
    reference_average_validity_blob: ContentRef,
) -> dict[str, Any]:
    analysis = _analysis_types()
    if not isinstance(value, analysis.CalibrationReport):
        raise TypeError("value must be CalibrationReport")
    if not isinstance(reference_average_blob, ContentRef):
        raise TypeError("reference_average_blob must be ContentRef")
    if not isinstance(reference_average_validity_blob, ContentRef):
        raise TypeError("reference_average_validity_blob must be ContentRef")
    if reference_average_blob.size != value.reference_average.nbytes:
        raise ValueError("reference_average_blob size differs from the report image")
    if reference_average_validity_blob.size != value.reference_average_validity.nbytes:
        raise ValueError(
            "reference_average_validity_blob size differs from the report mask"
        )
    return {
        "format": CALIBRATION_REPORT_FORMAT,
        "request": _request_to_tree(value.request),
        "software_lineage": _lineage_to_tree(value.software_lineage),
        "group_contexts": _group_contexts_to_tree(value.group_contexts),
        "reference_average_blob": content_ref_to_tree(reference_average_blob),
        "reference_average_validity_blob": content_ref_to_tree(
            reference_average_validity_blob
        ),
        "reference_box_signals": value.reference_box_signals,
        "labels": _labels_to_tree(value.labels),
        "split": _split_to_tree(value.split),
        "psf_fits": [_psf_fit_to_tree(item) for item in value.psf_fits],
        "models": [_model_report_to_tree(item) for item in value.models],
    }


def _report_data(tree: Any) -> dict[str, Any]:
    return _outer_fields(tree, _REPORT_FIELDS, CALIBRATION_REPORT_FORMAT)


def _report_from_tree(
    tree: Any,
    *,
    reference_average: np.ndarray,
    reference_average_validity: np.ndarray,
) -> CalibrationReport:
    analysis = _analysis_types()
    data = _report_data(tree)
    content_ref_from_tree(data["reference_average_blob"])
    content_ref_from_tree(data["reference_average_validity_blob"])
    return analysis.CalibrationReport(
        request=_request_from_tree(data["request"]),
        software_lineage=_lineage_from_tree(data["software_lineage"]),
        group_contexts=_group_contexts_from_tree(data["group_contexts"]),
        reference_average=reference_average,
        reference_average_validity=reference_average_validity,
        reference_box_signals=_array(
            data["reference_box_signals"],
            "reference_box_signals",
        ),
        labels=_labels_from_tree(data["labels"]),
        split=_split_from_tree(data["split"]),
        psf_fits=tuple(
            _psf_fit_from_tree(item) for item in _list(data["psf_fits"], "psf_fits")
        ),
        models=tuple(
            _model_report_from_tree(item) for item in _list(data["models"], "models")
        ),
    )


def encode_calibration_artifact(value: CalibrationArtifact) -> bytes:
    return encode(_artifact_to_tree(value))


def decode_calibration_artifact(
    payload: bytes | bytearray | memoryview,
) -> CalibrationArtifact:
    return _decode_typed(
        payload,
        _artifact_from_tree,
        _artifact_to_tree,
        CALIBRATION_ARTIFACT_FORMAT,
    )


def encode_calibration_report_metadata(
    value: CalibrationReport,
    *,
    reference_average_blob: ContentRef,
    reference_average_validity_blob: ContentRef,
) -> bytes:
    return encode(
        _report_to_tree(
            value,
            reference_average_blob,
            reference_average_validity_blob,
        )
    )


def calibration_report_blob_refs(
    payload: bytes | bytearray | memoryview,
) -> tuple[ContentRef, ContentRef]:
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise TypeError("calibration report payload must be bytes-like")
    data = _report_data(decode(payload))
    return (
        content_ref_from_tree(data["reference_average_blob"]),
        content_ref_from_tree(data["reference_average_validity_blob"]),
    )


def _report_image_bytes(
    value: np.ndarray,
    dtype: str,
    field: str,
    finite: bool,
) -> bytes:
    array = np.asarray(value)
    if (
        array.dtype != np.dtype(dtype)
        or array.ndim != 2
        or not array.flags.c_contiguous
        or (finite and not np.all(np.isfinite(array)))
    ):
        qualifier = "finite " if finite else ""
        raise ValueError(f"{field} must be a {qualifier}C-contiguous {dtype} image")
    return array.tobytes(order="C")


def encode_calibration_reference_average(value: np.ndarray) -> bytes:
    return _report_image_bytes(value, "<f8", "reference_average", True)


def encode_calibration_reference_average_validity(value: np.ndarray) -> bytes:
    return _report_image_bytes(value, "bool", "reference_average_validity", False)


def decode_calibration_report_arrays(
    reference_average_payload: bytes | bytearray | memoryview,
    reference_average_validity_payload: bytes | bytearray | memoryview,
    *,
    image_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    if (
        not isinstance(reference_average_payload, (bytes, bytearray, memoryview))
        or not isinstance(
            reference_average_validity_payload,
            (bytes, bytearray, memoryview),
        )
    ):
        raise TypeError("calibration report array payloads must be bytes-like")
    try:
        raw_shape = tuple(image_shape)
    except TypeError as exc:
        raise ValueError("image_shape must contain two positive integers") from exc
    if len(raw_shape) != 2 or any(
        isinstance(size, bool) or not isinstance(size, Integral) or size <= 0
        for size in raw_shape
    ):
        raise ValueError("image_shape must contain two positive integers")
    shape = tuple(int(size) for size in raw_shape)
    pixel_count = shape[0] * shape[1]
    average_payload = bytes(reference_average_payload)
    validity_payload = bytes(reference_average_validity_payload)
    if len(average_payload) != pixel_count * np.dtype("<f8").itemsize:
        raise ValueError("reference_average payload size differs from image_shape")
    if len(validity_payload) != pixel_count:
        raise ValueError(
            "reference_average_validity payload size differs from image_shape"
        )
    average = np.frombuffer(average_payload, dtype="<f8").reshape(shape)
    validity_bytes = np.frombuffer(validity_payload, dtype="uint8")
    if np.any(validity_bytes > 1):
        raise ValueError("reference_average_validity payload is not canonical boolean")
    validity = validity_bytes.view("bool").reshape(shape)
    if not np.all(np.isfinite(average)):
        raise ValueError("reference_average payload contains non-finite values")
    average.setflags(write=False)
    validity.setflags(write=False)
    return average, validity


def decode_calibration_report(
    payload: bytes | bytearray | memoryview,
    *,
    reference_average: np.ndarray,
    reference_average_validity: np.ndarray,
) -> CalibrationReport:
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise TypeError("calibration report payload must be bytes-like")
    raw = bytes(payload)
    tree = decode(raw)
    data = _report_data(tree)
    average_blob = content_ref_from_tree(data["reference_average_blob"])
    validity_blob = content_ref_from_tree(data["reference_average_validity_blob"])
    report = _report_from_tree(
        tree,
        reference_average=reference_average,
        reference_average_validity=reference_average_validity,
    )
    if encode(_report_to_tree(report, average_blob, validity_blob)) != raw:
        raise ValueError("calibration report payload is typed but non-canonical")
    return report


__all__ = [
    "CALIBRATION_ARTIFACT_FORMAT",
    "CALIBRATION_REPORT_FORMAT",
    "decode_calibration_artifact",
    "decode_calibration_report",
    "decode_calibration_report_arrays",
    "encode_calibration_artifact",
    "encode_calibration_reference_average",
    "encode_calibration_reference_average_validity",
    "encode_calibration_report_metadata",
    "calibration_report_blob_refs",
]
