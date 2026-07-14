"""Calibration preserves named axes while consuming raw captures frame by frame."""

from __future__ import annotations

import ast
from dataclasses import dataclass, replace
import inspect
from itertools import product
from pathlib import Path
import textwrap
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from Zou_lab_control.neutral_atom.devices.registry import DeviceSet
from Zou_lab_control.neutral_atom.devices.virtual import (
    VirtualCamera,
    VirtualSequencer,
    VirtualTrapArray,
)
from Zou_lab_control.neutral_atom.ports import PortCatalog, PortSpec
from zlc_data import (
    READOUT_EVENT,
    REPEAT,
    SCAN_POINT,
    AxisId,
    AxisSpec,
    BlockId,
    PointLayout,
)
from zlc_neutral_atom.acquisition import CameraAcquisitionMode
from zlc_neutral_atom.artifacts.capture import (
    AdmittedCapture,
    CaptureArtifact,
    CaptureRepository,
    compile_capture_artifact_pipeline,
)
import zlc_neutral_atom.readout.analysis as analysis_impl
from zlc_neutral_atom.readout.analysis import (
    CalibrationAnalysisRequest,
    CalibrationAnalysisResult,
    analyze_calibration,
    compute_calibration,
    estimate_calibration_analysis_peak_bytes,
)
from zlc_neutral_atom.readout.calibration import (
    BoxReducer,
    ReadoutModelKind,
    derive_calibration_readout_physical_context,
)
import zlc_neutral_atom.readout.calibration as calibration_impl
from zlc_neutral_atom.readout.calibration_codec import (
    calibration_report_blob_refs,
    decode_calibration_report,
    decode_calibration_report_arrays,
    encode_calibration_reference_average,
    encode_calibration_reference_average_validity,
    encode_calibration_report_metadata,
)
from zlc_neutral_atom.readout.contracts import CalibrationCaptureLayout
from zlc_neutral_atom.runtime import (
    DatasetCellAddress,
    DatasetMaterializerSpec,
    MinimalPipelineSpec,
    PipelineMemoryProfile,
)
from zlc_neutral_atom.timing.capture import TriggeredCaptureSpec
from zlc_neutral_atom.timing.capture_plan import compile_capture_cell_plan
from zlc_neutral_atom.timing.pulse import FinitePulseExecutionRequest
from zlc_pulse import (
    FIELD_DURATION,
    FrozenScanTable,
    PulseExecutionForm,
    PulseFieldRef,
    RepeatRegion,
    ScanParameter,
    compile_pulse_artifact,
    load_pulse_document,
)
from zlc_storage import ContentRef, decode, encode, sha256_digest
from zlc_workbench.camera_capture import CameraCaptureBindingRequest
from zlc_workbench.legacy_neutral_atom import LegacyNeutralAtomRuntime


_CENTERS_XY = ((7, 7), (24, 7), (7, 24), (24, 24))
_SPOT = np.array(
    ((0.42, 0.60, 0.42), (0.60, 1.00, 0.60), (0.42, 0.60, 0.42)),
    dtype=np.float64,
)
_ROOT = Path(__file__).parents[1]


def _pulse_catalog(document) -> PortCatalog:
    return PortCatalog(
        document.target.raw_lanes,
        tuple(
            PortSpec(
                port.key,
                port.kind,
                port.lanes,
                port.label,
                port.bus_index,
                port.width,
                port.encoding,
                port.safe_value,
                port.latch_clock,
            )
            for port in document.target.ports
        ),
    )


def test_full_resolution_report_images_use_exact_raw_binary_payloads():
    average = np.arange(35, dtype="<f8").reshape(5, 7)
    validity = np.ones((5, 7), dtype=bool)
    validity[1, 3] = False
    average_payload = encode_calibration_reference_average(average)
    validity_payload = encode_calibration_reference_average_validity(validity)

    assert len(average_payload) == average.nbytes
    assert len(validity_payload) == validity.nbytes
    decoded_average, decoded_validity = decode_calibration_report_arrays(
        average_payload,
        validity_payload,
        image_shape=(5, 7),
    )
    np.testing.assert_array_equal(decoded_average, average)
    np.testing.assert_array_equal(decoded_validity, validity)
    assert not decoded_average.flags.writeable
    assert not decoded_validity.flags.writeable
    with pytest.raises(ValueError, match="payload size"):
        decode_calibration_report_arrays(
            average_payload[:-1],
            validity_payload,
            image_shape=(5, 7),
        )
    malformed_validity = bytearray(validity_payload)
    malformed_validity[0] = 2
    with pytest.raises(ValueError, match="canonical boolean"):
        decode_calibration_report_arrays(
            average_payload,
            malformed_validity,
            image_shape=(5, 7),
        )


def _axis(name: str, role, size: int) -> AxisSpec:
    return AxisSpec(AxisId(name), name, role, size, tuple(range(size)))


def test_site_center_intent_is_copied_canonical_and_read_only():
    source = np.asarray(_CENTERS_XY, dtype=">f4")
    request = CalibrationAnalysisRequest(
        CalibrationCaptureLayout(AxisId("intent-event"), (0,), 1),
        (2, 2),
        expected_centers_xy=source,
        maximum_site_residual_px=1.5,
    )

    source[0] = (-1.0, -1.0)
    assert request.expected_centers_xy is not None
    assert request.expected_centers_xy.dtype == np.dtype("<f8")
    assert not request.expected_centers_xy.flags.writeable
    np.testing.assert_array_equal(request.expected_centers_xy, _CENTERS_XY)


@pytest.mark.parametrize(
    ("expected", "maximum", "message"),
    (
        (np.zeros((4, 2)), None, "provided together"),
        (None, 1.0, "provided together"),
        (np.zeros((3, 2)), 1.0, "must have shape"),
        (np.array(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (np.nan, 1.0))), 1.0, "must be finite"),
        (np.zeros((4, 2)), np.nan, "must be finite"),
        (np.zeros((4, 2)), 0.0, "must be positive"),
    ),
)
def test_site_center_intent_rejects_partial_shape_and_nonfinite_values(
    expected,
    maximum,
    message,
):
    with pytest.raises(ValueError, match=message):
        CalibrationAnalysisRequest(
            CalibrationCaptureLayout(AxisId("invalid-intent-event"), (0,), 1),
            (2, 2),
            expected_centers_xy=expected,
            maximum_site_residual_px=maximum,
        )


def _frame(repeat: int, event: int, detuning: int, phase: int) -> np.ndarray:
    image = np.zeros((32, 32), dtype=np.uint16)
    for site, (x, y) in enumerate(_CENTERS_XY):
        occupied = (repeat + detuning + phase + site) % 2 == 0
        if event in (0, 2):
            level = 2200.0 if occupied else 180.0
        else:
            level = 1050.0 if occupied else 90.0
        image[y - 1 : y + 2, x - 1 : x + 2] = np.rint(
            level * _SPOT
        ).astype(np.uint16)
    return image


def _deliver_when_armed(camera: VirtualCamera, images: list[np.ndarray]):
    failures: list[BaseException] = []

    def deliver() -> None:
        try:
            deadline = time.monotonic() + 5.0
            state = camera._recent_state()
            with state["cond"]:
                while not state["armed"]:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("camera was not armed")
                    state["cond"].wait(remaining)
            camera._deliver(images)
        except BaseException as error:  # pragma: no cover - surfaced by fixture
            failures.append(error)

    thread = threading.Thread(target=deliver, daemon=False)
    thread.start()
    return thread, failures


@dataclass(frozen=True)
class _MultiaxisCalibrationCase:
    admitted: AdmittedCapture
    capture: CaptureArtifact
    request: CalibrationAnalysisRequest
    result: CalibrationAnalysisResult


@pytest.fixture(scope="module")
def multiaxis_calibration_case(tmp_path_factory) -> _MultiaxisCalibrationCase:
    document = load_pulse_document(_ROOT / "pulses" / "imaging_template.json")
    document = replace(
        document,
        repeat=RepeatRegion(
            document.periods[0].period_id,
            document.periods[-1].period_id,
            6,
        ),
        scan_parameters=(
            ScanParameter(
                "fixture_point_duration",
                PulseFieldRef(
                    FIELD_DURATION,
                    document.periods[0].period_id,
                    None,
                ),
                "fixture point duration",
                "s",
            ),
        ),
        scan_table=FrozenScanTable(
            ("fixture_point_duration",),
            ((document.periods[0].duration,),) * 4,
        ),
    )
    sequencer = VirtualSequencer(
        sleep_scale=0,
        port_catalog=_pulse_catalog(document),
    )
    camera = VirtualCamera(
        VirtualTrapArray(grid_shape=(2, 2), image_shape=(32, 32), seed=29),
        exposure=1e-3,
        capture_trigger_channels=("ch11",),
    )
    camera.recent_capacity = 128
    runtime = LegacyNeutralAtomRuntime(
        DeviceSet(
            {"readout": camera, "sequencer": sequencer},
            {
                "readout": {"type": "VirtualCamera", "params": {}},
                "sequencer": {"type": "VirtualSequencer", "params": {}},
            },
        )
    )
    repeat_axis = _axis("cal-repeat", REPEAT, 6)
    event_axis = _axis("cal-event", READOUT_EVENT, 3)
    detuning_axis = _axis("cal-detuning", SCAN_POINT, 2)
    phase_axis = _axis("cal-phase", SCAN_POINT, 2)
    logical_rows = tuple(product(range(3), range(2), range(2)))
    # Storage order deliberately disagrees with logical C order.  Calibration
    # must join by AxisId/logical index rather than filtered storage position.
    point_layout = PointLayout.explicit((3, 2, 2), tuple(reversed(logical_rows)))
    scan_layout = PointLayout.rect_c((detuning_axis.size, phase_axis.size))
    cells = tuple(
        DatasetCellAddress(
            repeat,
            point_layout.storage_index((event, detuning, phase)),
        )
        for pulse_point in range(scan_layout.storage_size)
        for repeat in range(repeat_axis.size)
        for event in range(event_axis.size)
        for detuning, phase in (scan_layout.multi_index(pulse_point),)
    )
    description = runtime.describe_camera("readout")
    measurement = runtime.bind_camera_measurement(
        CameraCaptureBindingRequest(
            "readout",
            repeat_axis,
            (event_axis, detuning_axis, phase_axis),
            point_layout,
            cells,
            CameraAcquisitionMode.EXTERNAL_TRIGGERED,
            len(cells),
            64 << 20,
            tuple(description.event_setting(index) for index in range(3)),
        )
    )
    capture = MinimalPipelineSpec(
        "multiaxis calibration source",
        measurement,
        DatasetMaterializerSpec(
            BlockId("multiaxis-calibration-source"),
            PipelineMemoryProfile(96 << 20),
        ),
    )
    pulse_artifact = compile_pulse_artifact(
        document,
        clock_hz=sequencer.clock_hz,
        execution_form=PulseExecutionForm.AUTONOMOUS_SCAN_ONCE,
        trigger_channels=("ch11",),
        live_target=document.target,
    )
    cell_plan = compile_capture_cell_plan(
        pulse_artifact,
        "ch11",
        measurement.capture_contract.dataset_schema,
        readout_event_axis_id=event_axis.axis_id,
        scan_point_layout=scan_layout,
        within_point_grouping=tuple(
            (repeat, event)
            for repeat in range(repeat_axis.size)
            for event in range(event_axis.size)
        ),
    )
    spec = TriggeredCaptureSpec(
        capture,
        runtime.bind_sequencer_port(),
        FinitePulseExecutionRequest(document, pulse_artifact),
        "ch11",
        cell_plan,
    )
    images = [
        _frame(cell.repeat_index, *point_layout.multi_index(cell.point_storage_index))
        for cell in cells
    ]
    repository = CaptureRepository(
        tmp_path_factory.mktemp("multiaxis-calibration-capture"),
        repository_id="multiaxis-calibration-captures",
    )
    thread, failures = _deliver_when_armed(camera, images)
    try:
        reference = runtime.controller.start(
            compile_capture_artifact_pipeline(spec, repository)
        ).result(20.0)
        thread.join(5.0)
        assert not thread.is_alive() and failures == []
        admitted = repository.admit(reference)
        capture = admitted.artifact
        request = CalibrationAnalysisRequest(
            layout=CalibrationCaptureLayout(event_axis.axis_id, (0, 2), 1),
            grid_shape_yx=(2, 2),
            box_radius=1,
            box_reducer=BoxReducer.SUM,
            model_kinds=(ReadoutModelKind.BOX,),
            default_model_kind=ReadoutModelKind.BOX,
            train_fraction=0.5,
            split_seed=13,
            histogram_bins=32,
            max_drop=2,
            detector_min_distance=8,
            detector_threshold_rel=0.2,
            detector_refine_half=1,
            expected_centers_xy=np.asarray(_CENTERS_XY, dtype="<f8"),
            maximum_site_residual_px=2.0,
        )
        result = analyze_calibration(admitted, request)
        yield _MultiaxisCalibrationCase(admitted, capture, request, result)
    finally:
        if thread.is_alive():
            camera.finish_record_capture()
            thread.join(2.0)
        runtime.shutdown(timeout=3.0)
        repository.close()


def test_capture_report_preserves_named_multiaxis_context_and_frame_axes(
    multiaxis_calibration_case,
):
    case = multiaxis_calibration_case
    source = case.capture.frame_source
    expected_contexts = tuple(
        (
            (AxisId("cal-repeat"), repeat),
            (AxisId("cal-detuning"), detuning),
            (AxisId("cal-phase"), phase),
        )
        for repeat in range(6)
        for detuning in range(2)
        for phase in range(2)
    )

    assert source.schema.point_layout.logical_shape == (3, 2, 2)
    assert source.schema.physical_shape == (6, 12, 32, 32)
    assert source.schema.cell_schema.data_shape == (32, 32)
    assert case.result.report.group_contexts == expected_contexts
    assert case.result.report.reference_box_signals.shape == (24, 2, 4)
    assert case.result.report.reference_average.shape == (32, 32)
    assert {
        axis_id for context in case.result.report.group_contexts for axis_id, _ in context
    } == {
        AxisId("cal-repeat"),
        AxisId("cal-detuning"),
        AxisId("cal-phase"),
    }


def test_authoritative_analysis_rejects_a_bare_capture_artifact(
    multiaxis_calibration_case,
):
    case = multiaxis_calibration_case
    with pytest.raises(TypeError, match="exact AdmittedCapture"):
        analyze_calibration(case.capture, case.request)


def test_camera_only_calibration_source_without_pulse_lineage_is_rejected(
    multiaxis_calibration_case,
):
    case = multiaxis_calibration_case
    camera_only = replace(case.capture, pulse_lineage=None)

    with pytest.raises(ValueError, match="requires pulse-trigger lineage"):
        compute_calibration(camera_only, case.request)


def test_virtual_camera_offset_and_current_compiled_lineage_derive_context(
    multiaxis_calibration_case,
):
    case = multiaxis_calibration_case
    facts = case.capture.camera_capability_evidence.physical_facts
    assert facts.external_trigger_integration_start_offset_seconds == 0.0

    derived = derive_calibration_readout_physical_context(
        case.capture,
        case.request.layout,
        case.result.artifact.frame_contract,
    )
    assert derived == case.result.artifact.readout_physical_context
    assert derived.integration_start_offset_seconds == 0.0


def test_unqualified_qcmos_integration_offset_is_not_guessed_as_zero(
    multiaxis_calibration_case,
):
    case = multiaxis_calibration_case
    evidence = case.capture.camera_capability_evidence
    unqualified = replace(
        evidence,
        physical_facts=replace(
            evidence.physical_facts,
            external_trigger_integration_start_offset_seconds=None,
        ),
    )
    capture_view = SimpleNamespace(
        pulse_lineage=case.capture.pulse_lineage,
        camera_capability_evidence=unqualified,
    )

    with pytest.raises(ValueError, match="qualified integration start offset"):
        derive_calibration_readout_physical_context(
            capture_view,
            case.request.layout,
            case.result.artifact.frame_contract,
        )


def test_missing_site_intent_is_preview_only_and_cannot_mint_authority(
    multiaxis_calibration_case,
):
    case = multiaxis_calibration_case
    preview_request = replace(
        case.request,
        expected_centers_xy=None,
        maximum_site_residual_px=None,
    )

    preview = compute_calibration(case.capture, preview_request)
    assert preview.report.request.expected_centers_xy is None
    with pytest.raises(
        analysis_impl.CalibrationAnalysisError,
        match="authoritative calibration requires expected_centers_xy",
    ):
        analyze_calibration(case.admitted, preview_request)


def test_max_drop_is_bounded_by_sites_and_changes_memory_admission(
    multiaxis_calibration_case,
):
    case = multiaxis_calibration_case
    maximum = case.request.site_count
    accepted = replace(case.request, max_drop=maximum)
    with pytest.raises(ValueError, match="must not exceed.*declared sites"):
        replace(case.request, max_drop=maximum + 1)

    baseline = estimate_calibration_analysis_peak_bytes(
        case.capture.frame_source.schema,
        replace(case.request, max_drop=0),
    )
    expanded = estimate_calibration_analysis_peak_bytes(
        case.capture.frame_source.schema,
        accepted,
    )
    minimum_added_mask_bytes = (
        len(accepted.model_kinds) * maximum * accepted.site_count
    )
    assert expanded - baseline >= minimum_added_mask_bytes

    without_site_intent = estimate_calibration_analysis_peak_bytes(
        case.capture.frame_source.schema,
        replace(
            case.request,
            expected_centers_xy=None,
            maximum_site_residual_px=None,
        ),
    )
    with_site_intent = estimate_calibration_analysis_peak_bytes(
        case.capture.frame_source.schema,
        case.request,
    )
    assert case.request.expected_centers_xy is not None
    assert with_site_intent - without_site_intent == (
        case.request.expected_centers_xy.nbytes
    )


def _numpy_attribute_calls(function, names: set[str]) -> tuple[str, ...]:
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    return tuple(
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "np"
        and node.func.attr in names
    )


class _StreamingNumpyGuard:
    """Reject frame-count-scaled image materialization inside analysis.py."""

    def __init__(self, image_shape: tuple[int, int], raw_dtype: np.dtype):
        self._image_shape = image_shape
        self._raw_dtype = raw_dtype

    def _raw_frame(self, value) -> bool:
        return (
            isinstance(value, np.ndarray)
            and value.shape == self._image_shape
            and value.dtype == self._raw_dtype
        )

    def __getattr__(self, name: str):
        if name in {"stack", "vstack", "dstack", "concatenate"}:
            real = getattr(np, name)

            def guarded_stack(values, *args, **kwargs):
                arrays = tuple(value for value in values if isinstance(value, np.ndarray))
                assert not any(self._raw_frame(value) for value in arrays), (
                    "raw capture analysis must not stack frame arrays"
                )
                return real(values, *args, **kwargs)

            return guarded_stack
        if name == "asarray":
            def guarded_asarray(value, *args, **kwargs):
                result = np.asarray(value, *args, **kwargs)
                if self._raw_frame(value):
                    assert np.shares_memory(result, value)
                    assert result.dtype == value.dtype
                if (
                    result.ndim > 2
                    and result.shape[-2:] == self._image_shape
                ):
                    assert isinstance(value, np.ndarray)
                    assert np.shares_memory(result, value)
                    assert result.dtype == value.dtype
                return result

            return guarded_asarray
        if name == "array":
            real_array = np.array

            def guarded_array(value, *args, **kwargs):
                assert not self._raw_frame(
                    value
                ), "analysis copied a raw capture frame"
                result = real_array(value, *args, **kwargs)
                assert not (
                    result.ndim > 2
                    and result.shape[-2:] == self._image_shape
                ), "analysis materialized a raw image stack"
                return result

            return guarded_array
        if name in {"empty", "zeros", "ones", "full"}:
            real = getattr(np, name)

            def guarded_allocation(shape, *args, **kwargs):
                normalized = (shape,) if isinstance(shape, int) else tuple(shape)
                assert not (
                    len(normalized) > 2
                    and tuple(normalized[-2:]) == self._image_shape
                ), "analysis allocated a frame-count-scaled image tensor"
                return real(shape, *args, **kwargs)

            return guarded_allocation
        return getattr(np, name)


def _whole_frame_float_casts(function) -> tuple[str, ...]:
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    violations = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if (
            node.func.attr == "astype"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in {"image", "frame"}
        ):
            violations.append(ast.unparse(node))
            continue
        if not (
            isinstance(node.func.value, ast.Name)
            and node.func.value.id == "np"
            and node.func.attr in {"array", "asarray", "ascontiguousarray"}
            and node.args
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id in {"image", "frame"}
        ):
            continue
        dtype = next((item.value for item in node.keywords if item.arg == "dtype"), None)
        if dtype is not None:
            violations.append(ast.unparse(node))
    return tuple(violations)


def test_capture_analysis_borrows_uint16_frames_and_never_materializes_a_stack(
    multiaxis_calibration_case,
    monkeypatch,
):
    case = multiaxis_calibration_case
    prepared: list[tuple[np.dtype, tuple[int, ...], bool]] = []
    extracted: list[tuple[np.dtype, tuple[int, ...], bool, np.dtype, tuple[int, ...]]] = []
    real_prepare = analysis_impl._prepare_frame
    real_extract = analysis_impl._extract_readout_arrays

    def observe_prepare(values, validity, frame_contract):
        prepared.append((values.dtype, values.shape, values.flags.writeable))
        return real_prepare(values, validity, frame_contract)

    def observe_extract(feature, image, pixel_validity):
        result, validity = real_extract(feature, image, pixel_validity)
        extracted.append(
            (
                image.dtype,
                image.shape,
                image.flags.writeable,
                result.dtype,
                result.shape,
            )
        )
        return result, validity

    guard = _StreamingNumpyGuard((32, 32), np.dtype("uint16"))
    monkeypatch.setattr(analysis_impl, "np", guard)
    monkeypatch.setattr(analysis_impl, "_prepare_frame", observe_prepare)
    monkeypatch.setattr(analysis_impl, "_extract_readout_arrays", observe_extract)
    rerun = compute_calibration(case.capture, case.request)

    assert rerun.report.group_contexts == case.result.report.group_contexts
    assert prepared and extracted
    assert all(item == (np.dtype("uint16"), (32, 32), False) for item in prepared)
    assert all(
        item == (np.dtype("uint16"), (32, 32), False, np.dtype("float64"), (4,))
        for item in extracted
    )
    forbidden = {"stack", "vstack", "dstack", "concatenate"}
    assert _numpy_attribute_calls(analysis_impl._capture_frame_source, forbidden) == ()
    assert _numpy_attribute_calls(analysis_impl._calibrate_readout_source, forbidden) == ()
    assert _numpy_attribute_calls(analysis_impl._extract_source_stack, forbidden) == ()
    assert _whole_frame_float_casts(calibration_impl._extract_readout_arrays) == ()


def test_multiple_models_read_short_frames_once_and_match_single_feature_oracle(
    multiaxis_calibration_case,
    monkeypatch,
):
    case = multiaxis_calibration_case
    request = replace(
        case.request,
        model_kinds=tuple(ReadoutModelKind),
        default_model_kind=ReadoutModelKind.BOX,
    )
    real_capture_source = analysis_impl._capture_frame_source
    short_factory_calls = 0
    short_frames_yielded = 0

    def counted_capture_source(source, layout):
        group_count, references, shorts, contexts = real_capture_source(
            source,
            layout,
        )

        def counted_shorts():
            nonlocal short_factory_calls, short_frames_yielded
            short_factory_calls += 1
            for frame in shorts():
                short_frames_yielded += 1
                yield frame

        return group_count, references, counted_shorts, contexts

    monkeypatch.setattr(
        analysis_impl,
        "_capture_frame_source",
        counted_capture_source,
    )
    result = analyze_calibration(case.admitted, request)

    assert short_factory_calls == 1
    assert short_frames_yielded == len(result.report.group_contexts)

    group_count, _references, oracle_shorts, _contexts = real_capture_source(
        case.capture.frame_source,
        request.layout,
    )
    for model in result.artifact.models:
        expected_signals, expected_validity = analysis_impl._extract_source_stack(
            model.feature,
            (group_count,),
            oracle_shorts,
            result.artifact.frame_contract,
        )
        report = result.report.model(model.kind)
        np.testing.assert_array_equal(report.short_signals, expected_signals)
        np.testing.assert_array_equal(report.short_validity, expected_validity)


def test_report_codec_preserves_contexts_and_reference_pixel_validity(
    multiaxis_calibration_case,
):
    original = multiaxis_calibration_case.result.report
    pixel_validity = np.array(original.reference_average_validity, copy=True)
    pixel_validity[0, 0] = False
    pixel_validity[5, 11] = False
    report = replace(original, reference_average_validity=pixel_validity)
    average_payload = encode_calibration_reference_average(report.reference_average)
    validity_payload = encode_calibration_reference_average_validity(
        report.reference_average_validity
    )
    average_ref = ContentRef(sha256_digest(average_payload), len(average_payload))
    validity_ref = ContentRef(sha256_digest(validity_payload), len(validity_payload))
    payload = encode_calibration_report_metadata(
        report,
        reference_average_blob=average_ref,
        reference_average_validity_blob=validity_ref,
    )
    average, validity = decode_calibration_report_arrays(
        average_payload,
        validity_payload,
        image_shape=report.reference_average.shape,
    )
    decoded = decode_calibration_report(
        payload,
        reference_average=average,
        reference_average_validity=validity,
    )

    assert decoded.group_contexts == report.group_contexts
    assert decoded.request.expected_centers_xy is not None
    np.testing.assert_array_equal(
        decoded.request.expected_centers_xy,
        report.request.expected_centers_xy,
    )
    assert not decoded.request.expected_centers_xy.flags.writeable
    assert (
        decoded.request.maximum_site_residual_px
        == report.request.maximum_site_residual_px
    )
    np.testing.assert_array_equal(
        decoded.reference_average_validity,
        report.reference_average_validity,
    )
    assert decoded.reference_average_validity.dtype == np.dtype("bool")
    assert decoded.reference_average_validity.shape == (32, 32)
    assert calibration_report_blob_refs(payload) == (average_ref, validity_ref)
    assert encode_calibration_report_metadata(
        decoded,
        reference_average_blob=average_ref,
        reference_average_validity_blob=validity_ref,
    ) == payload

    wrong_shape = decode(payload)
    wrong_shape["request"]["expected_centers_xy"] = np.zeros((3, 2), dtype="<f8")
    with pytest.raises(ValueError, match="must have shape"):
        decode_calibration_report(
            encode(wrong_shape),
            reference_average=average,
            reference_average_validity=validity,
        )

    nonfinite = decode(payload)
    nonfinite_centers = np.array(
        nonfinite["request"]["expected_centers_xy"],
        copy=True,
    )
    nonfinite_centers[0, 0] = np.nan
    nonfinite["request"]["expected_centers_xy"] = nonfinite_centers
    with pytest.raises(ValueError, match="must be finite"):
        decode_calibration_report(
            encode(nonfinite),
            reference_average=average,
            reference_average_validity=validity,
        )
