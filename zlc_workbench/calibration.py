"""Headless presentation seam for formal readout-calibration authoring.

The calibration domain remains the sole owner of request semantics.  This
module only projects its editable scalar leaves into the shared form contract
and rebuilds a complete request through the domain constructor.  Spatial
authority is deliberately frozen: detector output and display selections can
never rewrite the independent expected-center evidence.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

from zlc_frontend.form import FormChoice, FormFieldProps, FormSpec
from zlc_neutral_atom.capture_reference import CaptureArtifactRef
from zlc_neutral_atom.readout.calibration import (
    BackgroundMode,
    BoxReducer,
    CalibrationAnalysisRequest,
    ReadoutModelKind,
)
from zlc_neutral_atom.readout.calibration_reference import CalibrationArtifactRef
from zlc_neutral_atom.readout.contracts import ReadoutBindingKey
from zlc_storage import positive_integer, positive_real


@dataclass(frozen=True, slots=True, eq=False)
class CalibrationEditorSeed:
    """Fixed source authority plus the initial editable analysis intent."""

    source_capture_ref: CaptureArtifactRef
    readout_binding: ReadoutBindingKey
    analysis: CalibrationAnalysisRequest
    memory_limit_bytes: int
    timeout_seconds: float
    previous_reference: CalibrationArtifactRef | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.source_capture_ref, CaptureArtifactRef):
            raise TypeError("source_capture_ref must be CaptureArtifactRef")
        if not isinstance(self.readout_binding, ReadoutBindingKey):
            raise TypeError("readout_binding must be ReadoutBindingKey")
        if not isinstance(self.analysis, CalibrationAnalysisRequest):
            raise TypeError("analysis must be CalibrationAnalysisRequest")
        object.__setattr__(
            self,
            "memory_limit_bytes",
            positive_integer(self.memory_limit_bytes, "memory_limit_bytes"),
        )
        object.__setattr__(
            self,
            "timeout_seconds",
            positive_real(self.timeout_seconds, "timeout_seconds"),
        )
        if self.previous_reference is not None and not isinstance(
            self.previous_reference,
            CalibrationArtifactRef,
        ):
            raise TypeError(
                "previous_reference must be CalibrationArtifactRef or None"
            )


def calibration_seed_from_computation(
    computation: object,
    reference: CalibrationArtifactRef,
    *,
    memory_limit_bytes: int,
    timeout_seconds: float,
) -> CalibrationEditorSeed:
    """Project one paired, admitted computation into an immutable editor seed."""

    from zlc_neutral_atom.readout.analysis import CalibrationComputation

    if not isinstance(computation, CalibrationComputation):
        raise TypeError("calibration loader must return CalibrationComputation")
    if not isinstance(reference, CalibrationArtifactRef):
        raise TypeError("reference must be CalibrationArtifactRef")
    artifact = computation.artifact
    return CalibrationEditorSeed(
        artifact.source_binding.source_capture_ref,
        artifact.frame_contract.binding,
        computation.report.request,
        memory_limit_bytes,
        timeout_seconds,
        reference,
    )


def calibration_analysis_form(request: CalibrationAnalysisRequest) -> FormSpec:
    """Return the single shared scalar editor for one complete request."""

    if not isinstance(request, CalibrationAnalysisRequest):
        raise TypeError("request must be CalibrationAnalysisRequest")
    model_fields = tuple(
        FormFieldProps(
            f"model.{kind.value}.enabled",
            "bool",
            f"Enable {kind.value}",
            default=kind in request.model_kinds,
            description=(
                "All enabled models are calibrated and committed atomically."
            ),
        )
        for kind in ReadoutModelKind
    )
    return FormSpec(
        (
            *model_fields,
            FormFieldProps(
                "default_model_kind",
                "choice",
                "Default model",
                default=request.default_model_kind,
                choices=tuple(
                    FormChoice(kind.value, kind) for kind in ReadoutModelKind
                ),
            ),
            FormFieldProps(
                "box_radius",
                "int",
                "Box radius",
                default=request.box_radius,
                required=True,
                unit="px",
                minimum=0,
            ),
            FormFieldProps(
                "box_reducer",
                "choice",
                "Box reducer",
                default=request.box_reducer,
                choices=tuple(FormChoice(item.value, item) for item in BoxReducer),
            ),
            FormFieldProps(
                "psf_half_width",
                "int",
                "PSF half width",
                default=request.psf_half_width,
                required=True,
                unit="px",
                minimum=0,
            ),
            FormFieldProps(
                "psf_background",
                "choice",
                "PSF background",
                default=request.psf_background,
                choices=tuple(
                    FormChoice(item.value, item) for item in BackgroundMode
                ),
            ),
            FormFieldProps(
                "psf_background_padding",
                "int",
                "PSF background padding",
                default=request.psf_background_padding,
                required=True,
                unit="px",
                minimum=1,
            ),
            FormFieldProps(
                "train_fraction",
                "float",
                "Train fraction",
                default=request.train_fraction,
                required=True,
                description="Must remain strictly between zero and one.",
            ),
            FormFieldProps(
                "split_seed",
                "int",
                "Split seed",
                default=request.split_seed,
                required=True,
                minimum=0,
            ),
            FormFieldProps(
                "histogram_bins",
                "int",
                "Histogram bins",
                default=request.histogram_bins,
                required=True,
                minimum=2,
            ),
            FormFieldProps(
                "minimum_site_fidelity",
                "float",
                "Minimum site fidelity",
                default=request.minimum_site_fidelity,
                required=True,
                minimum=0.5,
                maximum=1.0,
            ),
            FormFieldProps(
                "max_drop",
                "int",
                "Maximum dropped sites",
                default=request.max_drop,
                required=True,
                minimum=0,
                maximum=request.site_count,
            ),
            FormFieldProps(
                "detector_min_distance",
                "int",
                "Detector minimum distance",
                default=request.detector_min_distance,
                unit="px",
                minimum=1,
            ),
            FormFieldProps(
                "detector_threshold_rel",
                "float",
                "Detector relative threshold",
                default=request.detector_threshold_rel,
                required=True,
                minimum=0.0,
                maximum=1.0,
            ),
            FormFieldProps(
                "detector_refine_half",
                "int",
                "Detector refine half width",
                default=request.detector_refine_half,
                required=True,
                unit="px",
                minimum=0,
            ),
        )
    )


def calibration_analysis_from_form(
    request: CalibrationAnalysisRequest,
    values: Mapping[str, object],
) -> CalibrationAnalysisRequest:
    """Rebuild a request while preserving every frozen spatial authority fact."""

    spec = calibration_analysis_form(request)
    if not isinstance(values, Mapping):
        raise TypeError("calibration form values must be a mapping")
    supplied = set(values)
    expected = set(spec.keys)
    if supplied != expected:
        raise ValueError(
            "calibration form values must use the exact form keys; "
            f"missing={sorted(expected - supplied)!r}, "
            f"extra={sorted(supplied - expected)!r}"
        )
    model_kinds = tuple(
        kind
        for kind in ReadoutModelKind
        if values[f"model.{kind.value}.enabled"] is True
    )
    if any(
        type(values[f"model.{kind.value}.enabled"]) is not bool
        for kind in ReadoutModelKind
    ):
        raise TypeError("calibration model enabled fields must be bool")
    default_model = values["default_model_kind"]
    if not isinstance(default_model, ReadoutModelKind):
        raise TypeError("default_model_kind must be ReadoutModelKind")
    if default_model not in model_kinds:
        raise ValueError("default model must remain enabled")
    box_reducer = values["box_reducer"]
    background = values["psf_background"]
    if not isinstance(box_reducer, BoxReducer):
        raise TypeError("box_reducer must be BoxReducer")
    if not isinstance(background, BackgroundMode):
        raise TypeError("psf_background must be BackgroundMode")
    return replace(
        request,
        model_kinds=model_kinds,
        default_model_kind=default_model,
        box_radius=values["box_radius"],
        box_reducer=box_reducer,
        psf_half_width=values["psf_half_width"],
        psf_background=background,
        psf_background_padding=values["psf_background_padding"],
        train_fraction=values["train_fraction"],
        split_seed=values["split_seed"],
        histogram_bins=values["histogram_bins"],
        minimum_site_fidelity=values["minimum_site_fidelity"],
        max_drop=values["max_drop"],
        detector_min_distance=values["detector_min_distance"],
        detector_threshold_rel=values["detector_threshold_rel"],
        detector_refine_half=values["detector_refine_half"],
    )


def calibration_authority_summary(seed: CalibrationEditorSeed) -> str:
    """Describe all fixed authority without serializing arrays into the GUI."""

    if not isinstance(seed, CalibrationEditorSeed):
        raise TypeError("seed must be CalibrationEditorSeed")
    request = seed.analysis
    layout = request.layout
    centers = request.expected_centers_xy
    center_text = (
        "missing (formal Run will reject)"
        if centers is None
        else f"{centers.shape[0]} independent centers"
    )
    residual_text = (
        "none"
        if request.maximum_site_residual_px is None
        else f"{request.maximum_site_residual_px:.6g} px"
    )
    previous = (
        "new calibration"
        if seed.previous_reference is None
        else f"editing {seed.previous_reference.target_ref} into a new artifact"
    )
    return (
        f"source={seed.source_capture_ref.target_ref} · "
        f"binding={seed.readout_binding.value} · {previous}\n"
        f"READOUT_EVENT={layout.readout_event_axis_id.value} · "
        f"reference={layout.reference_event_indices} · "
        f"readout={layout.readout_event_index}\n"
        f"grid={request.grid_shape_yx} {request.ordering.value} · "
        f"sites={request.site_count} · {center_text} · max residual={residual_text}\n"
        "Spatial authority is frozen; detector/display output cannot rewrite it."
    )


__all__ = [
    "CalibrationEditorSeed",
    "calibration_analysis_form",
    "calibration_analysis_from_form",
    "calibration_authority_summary",
    "calibration_seed_from_computation",
]
