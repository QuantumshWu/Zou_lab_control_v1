from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from zlc_data.axis import (
    READOUT_EVENT,
    REPEAT,
    SCAN_POINT,
    SPATIAL_X,
    SPATIAL_Y,
    AxisId,
    AxisSpec,
    CoordinateFrameId,
)
from zlc_data.schema import (
    DatasetSchema,
    GridTopology,
    PointColumn,
    PointTable,
    ValueSchema,
)
from zlc_data.validity import ValidityContract
from zlc_neutral_atom.logic_nodes.readout.contracts import (
    CalibrationCaptureLayout,
    FrameContract,
)
from zlc_neutral_atom.devices.camera.contract import (
    CameraCaptureDescriptor,
    CameraEventReadoutSetting,
    ReadoutBindingKey,
)
from zlc_neutral_atom.devices.camera.contract import (
    camera_capture_descriptor_from_tree,
    camera_capture_descriptor_to_tree,
    readout_binding_key_from_tree,
    readout_binding_key_to_tree,
)
from zlc_neutral_atom.devices.camera.contract import CameraPhysicalFacts


Y = AxisId("camera-y")
X = AxisId("camera-x")
EVENT = AxisId("readout-event")
SCAN = AxisId("detuning")
FRAME = CoordinateFrameId("qcm-camera-pixels")
BINDING = ReadoutBindingKey("primary-readout")


def _event_settings() -> tuple[CameraEventReadoutSetting, ...]:
    return (
        CameraEventReadoutSetting(0, 0.001, 1.0, "fast"),
        CameraEventReadoutSetting(1, 0.002, 2.0, "low-noise"),
        CameraEventReadoutSetting(2, 0.003, 3.0, "low-noise"),
    )


def _descriptor(**changes: object) -> CameraCaptureDescriptor:
    value = CameraCaptureDescriptor(
        camera_identity="qcm-camera:serial-42",
        sensor_identity="qcm-sensor:serial-42",
        optical_path="science-imaging-v1",
        sensor_shape_yx=(100, 120),
        roi_origin_yx=(10, 20),
        roi_shape_yx=(40, 60),
        binning_yx=(2, 3),
        spatial_y_axis_id=Y,
        spatial_x_axis_id=X,
        coordinate_frame=FRAME,
        dtype=np.dtype("uint16"),
        count_unit="camera-count",
        readout_event_axis_id=EVENT,
        event_settings=_event_settings(),
    )
    return replace(value, **changes)


def _default_point_domain() -> tuple[PointTable, GridTopology]:
    cells = tuple(
        (event_index, scan_index)
        for event_index in range(3)
        for scan_index in range(2)
    )
    table = PointTable(
        len(cells),
        (
            PointColumn(
                EVENT,
                "readout event",
                READOUT_EVENT,
                PointColumn.NUMERIC,
                tuple(event_index for event_index, _scan_index in cells),
            ),
            PointColumn(
                SCAN,
                "detuning",
                SCAN_POINT,
                PointColumn.NUMERIC,
                tuple(scan_index for _event_index, scan_index in cells),
                unit="MHz",
            ),
        ),
    )
    return table, GridTopology(
        (EVENT, SCAN),
        (tuple(range(3)), tuple(range(2))),
        cells,
    )


def _schema(
    *,
    descriptor: CameraCaptureDescriptor | None = None,
    point_table: PointTable | None = None,
    grid_topology: GridTopology | None = None,
    data_axes: tuple[AxisSpec, ...] | None = None,
    validity: ValidityContract | None = None,
    dtype: object | None = None,
    unit: str = "camera-count",
) -> DatasetSchema:
    descriptor = _descriptor() if descriptor is None else descriptor
    if point_table is None:
        point_table, default_grid_topology = _default_point_domain()
        if grid_topology is None:
            grid_topology = default_grid_topology
    if data_axes is None:
        height, width = descriptor.output_shape_yx
        data_axes = (
            AxisSpec(
                Y,
                "ROI-local y",
                SPATIAL_Y,
                height,
                coordinates=tuple(range(height)),
                unit="pixel",
                coordinate_frame=FRAME,
            ),
            AxisSpec(
                X,
                "ROI-local x",
                SPATIAL_X,
                width,
                coordinates=tuple(range(width)),
                unit="pixel",
                coordinate_frame=FRAME,
            ),
        )
    return DatasetSchema(
        repeat_axis=AxisSpec(AxisId("repeat"), "repeat", REPEAT, 4),
        point_table=point_table,
        grid_topology=grid_topology,
        cell_schema=ValueSchema(
            data_axes,
            ValidityContract.value() if validity is None else validity,
            descriptor.dtype if dtype is None else dtype,
            unit,
        ),
    )


def _calibration_layout() -> CalibrationCaptureLayout:
    return CalibrationCaptureLayout(EVENT, (2, 0), 1)


def _contract() -> FrameContract:
    descriptor = _descriptor()
    setting = descriptor.setting(_calibration_layout().readout_event_index)
    physical_facts = replace(
        _physical_facts(),
        exposure_seconds=setting.exposure_seconds,
        gain=setting.gain,
        readout_mode=setting.readout_mode,
    )
    return FrameContract.from_camera_working_point(
        BINDING,
        physical_facts,
        _schema(descriptor=descriptor).cell_schema,
    )


def _physical_facts() -> CameraPhysicalFacts:
    descriptor = _descriptor()
    setting = descriptor.setting(0)
    return CameraPhysicalFacts(
        camera_identity=descriptor.camera_identity,
        sensor_identity=descriptor.sensor_identity,
        optical_path=descriptor.optical_path,
        sensor_shape_yx=descriptor.sensor_shape_yx,
        roi_origin_yx=descriptor.roi_origin_yx,
        roi_shape_yx=descriptor.roi_shape_yx,
        binning_yx=descriptor.binning_yx,
        spatial_y_axis_id=descriptor.spatial_y_axis_id,
        spatial_x_axis_id=descriptor.spatial_x_axis_id,
        coordinate_frame=descriptor.coordinate_frame,
        dtype=descriptor.dtype,
        count_unit=descriptor.count_unit,
        exposure_seconds=setting.exposure_seconds,
        required_external_trigger_interval_seconds=0.0,
        external_trigger_integration_start_offset_seconds=0.0,
        gain=setting.gain,
        readout_mode=setting.readout_mode,
    )


def test_camera_frame_values_share_geometry_and_count_dtype_normalization() -> None:
    changes = {
        "sensor_shape_yx": [np.int64(100), np.int64(120)],
        "roi_origin_yx": [np.int64(10), np.int64(20)],
        "roi_shape_yx": [np.int64(40), np.int64(60)],
        "binning_yx": [np.int64(2), np.int64(3)],
        "dtype": np.dtype(">u2"),
    }
    descriptor = replace(_descriptor(), **changes)
    contract = replace(_contract(), **changes)
    physical_facts = replace(_physical_facts(), **changes)

    expected_geometry = ((100, 120), (10, 20), (40, 60), (2, 3))
    for value in (descriptor, contract, physical_facts):
        assert (
            value.sensor_shape_yx,
            value.roi_origin_yx,
            value.roi_shape_yx,
            value.binning_yx,
        ) == expected_geometry
        assert value.dtype == np.dtype("<u2")
    assert descriptor.output_shape_yx == physical_facts.output_shape_yx == (20, 20)


@pytest.mark.parametrize(
    ("changes", "error_type", "message"),
    [
        (
            {"roi_origin_yx": (70, 20)},
            ValueError,
            "camera ROI lies outside the declared sensor geometry",
        ),
        (
            {"roi_shape_yx": (41, 60)},
            ValueError,
            "roi_shape_yx must be exactly divisible by binning_yx",
        ),
        (
            {"binning_yx": (0, 3)},
            ValueError,
            "binning_yx\\[0\\] must be at least 1",
        ),
        (
            {"dtype": np.dtype("complex64")},
            TypeError,
            "dtype must be a real integer or floating dtype",
        ),
    ],
)
def test_camera_frame_values_reject_the_same_invalid_primitives(
    changes: dict[str, object],
    error_type: type[Exception],
    message: str,
) -> None:
    for value in (_descriptor(), _contract(), _physical_facts()):
        with pytest.raises(error_type, match=message):
            replace(value, **changes)


def test_frame_axes_are_explicit_roi_local_output_pixel_coordinates() -> None:
    descriptor = _descriptor()
    height, width = descriptor.output_shape_yx
    wrong_unit = _schema(
        descriptor=descriptor,
        data_axes=(
            AxisSpec(
                Y,
                "ROI-local y",
                SPATIAL_Y,
                height,
                coordinates=tuple(range(height)),
                unit="px",
                coordinate_frame=FRAME,
            ),
            AxisSpec(
                X,
                "ROI-local x",
                SPATIAL_X,
                width,
                coordinates=tuple(range(width)),
                unit="pixel",
                coordinate_frame=FRAME,
            ),
        ),
    )
    with pytest.raises(ValueError, match="canonical 'pixel' unit"):
        descriptor.validate_schema(wrong_unit)
    shifted = _schema(
        descriptor=descriptor,
        data_axes=(
            AxisSpec(
                Y,
                "sensor-global y",
                SPATIAL_Y,
                height,
                coordinates=tuple(range(10, 10 + height)),
                unit="pixel",
                coordinate_frame=FRAME,
            ),
            AxisSpec(
                X,
                "ROI-local x",
                SPATIAL_X,
                width,
                coordinates=tuple(range(width)),
                unit="pixel",
                coordinate_frame=FRAME,
            ),
        ),
    )
    with pytest.raises(ValueError, match="ROI-local"):
        descriptor.validate_schema(shifted)
    float_spelling = _schema(
        descriptor=descriptor,
        data_axes=(
            AxisSpec(
                Y,
                "ROI-local y",
                SPATIAL_Y,
                height,
                coordinates=tuple(float(index) for index in range(height)),
                unit="pixel",
                coordinate_frame=FRAME,
            ),
            AxisSpec(
                X,
                "ROI-local x",
                SPATIAL_X,
                width,
                coordinates=tuple(range(width)),
                unit="pixel",
                coordinate_frame=FRAME,
            ),
        ),
    )
    descriptor.validate_schema(float_spelling)
    assert all(
        all(type(coordinate) is int for coordinate in axis.coordinates)
        for axis in float_spelling.cell_schema.data_axes
    )


def _single_event_descriptor(
    *,
    exposure: float = 0.002,
    gain: float = 2.0,
    mode: str = "low-noise",
    **changes: object,
) -> CameraCaptureDescriptor:
    return _descriptor(
        readout_event_axis_id=None,
        event_settings=(
            CameraEventReadoutSetting(
                0,
                exposure,
                gain,
                mode,
            ),
        ),
        **changes,
    )


def _single_event_schema(
    descriptor: CameraCaptureDescriptor,
    **changes: object,
) -> DatasetSchema:
    point_table = PointTable(
        2,
        (
            PointColumn(
                SCAN,
                "detuning",
                SCAN_POINT,
                PointColumn.NUMERIC,
                (0, 1),
            ),
        ),
    )
    return _schema(
        descriptor=descriptor,
        point_table=point_table,
        grid_topology=GridTopology((SCAN,), ((0, 1),), ((0,), (1,))),
        **changes,
    )


def test_multievent_calibration_contract_matches_same_single_event_occupancy() -> None:
    calibrated = _contract()
    occupancy = _single_event_descriptor()
    occupancy_schema = _single_event_schema(occupancy)
    calibrated.assert_compatible(
        BINDING,
        occupancy,
        occupancy_schema,
        readout_event_index=0,
    )
    assert calibrated.exposure_seconds == 0.002
    assert calibrated.gain == 2.0
    assert calibrated.readout_mode == "low-noise"


def test_live_camera_working_point_requires_explicit_applicability_facts() -> None:
    calibrated = _contract()
    live_facts = replace(
        _physical_facts(),
        exposure_seconds=0.002,
        gain=2.0,
        readout_mode="low-noise",
    )
    frame_schema = _schema().cell_schema
    calibrated.assert_compatible_working_point(
        BINDING,
        live_facts,
        frame_schema,
    )

    for change in (
        {"optical_path": "alternate-imaging-v1"},
        {"roi_origin_yx": (11, 20)},
        {"gain": 2.1},
        {"readout_mode": "alternate-mode"},
    ):
        with pytest.raises(ValueError, match="readout frame contract mismatch"):
            calibrated.assert_compatible_working_point(
                BINDING,
                replace(live_facts, **change),
                frame_schema,
            )

    # Raw camera exposure is not the pulse-defined atom illumination window.
    calibrated.assert_compatible_working_point(
        BINDING,
        replace(live_facts, exposure_seconds=0.0021),
        frame_schema,
    )


@pytest.mark.parametrize(
    ("gain", "mode"),
    [
        (2.1, "low-noise"),
        (2.0, "alternate-mode"),
    ],
)
def test_selected_gain_and_mode_changes_are_rejected(
    gain: float,
    mode: str,
) -> None:
    descriptor = _single_event_descriptor(exposure=0.002, gain=gain, mode=mode)
    with pytest.raises(ValueError, match="frame contract mismatch"):
        _contract().assert_compatible(
            BINDING,
            descriptor,
            _single_event_schema(descriptor),
            readout_event_index=0,
        )


def test_selected_raw_exposure_is_provenance_not_an_applicability_blocker() -> None:
    descriptor = _single_event_descriptor(exposure=0.0021)
    _contract().assert_compatible(
        BINDING,
        descriptor,
        _single_event_schema(descriptor),
        readout_event_index=0,
    )


@pytest.mark.parametrize(
    "descriptor_change",
    [
        {"camera_identity": "qcm-camera:serial-99"},
        {"sensor_identity": "qcm-sensor:serial-99"},
        {"optical_path": "alternate-imaging-v1"},
        {"sensor_shape_yx": (101, 120)},
        {"roi_origin_yx": (11, 20)},
        {"roi_shape_yx": (42, 60)},
        {"binning_yx": (1, 3)},
        {"coordinate_frame": CoordinateFrameId("other-pixels")},
        {"dtype": np.dtype("uint32")},
        {"count_unit": "electron"},
    ],
)
def test_frame_contract_rejects_changed_physical_fact(
    descriptor_change: dict[str, object],
) -> None:
    descriptor = _single_event_descriptor(**descriptor_change)
    with pytest.raises(ValueError):
        _contract().assert_compatible(
            BINDING,
            descriptor,
            _single_event_schema(descriptor),
            readout_event_index=0,
        )


def test_binding_validity_and_selected_event_metadata_mismatch_fail_closed() -> None:
    descriptor = _single_event_descriptor()
    schema = _single_event_schema(descriptor)
    with pytest.raises(ValueError, match="binding"):
        _contract().assert_compatible(
            ReadoutBindingKey("secondary-readout"),
            descriptor,
            schema,
            readout_event_index=0,
        )
    component_schema = _single_event_schema(
        descriptor,
        validity=ValidityContract.components(Y),
    )
    with pytest.raises(ValueError, match="frame_schema"):
        _contract().assert_compatible(
            BINDING,
            descriptor,
            component_schema,
            readout_event_index=0,
        )
    wrong_event = _single_event_descriptor(mode="event-metadata-changed")
    with pytest.raises(ValueError, match="readout_mode"):
        _contract().assert_compatible(
            BINDING,
            wrong_event,
            _single_event_schema(wrong_event),
            readout_event_index=0,
        )


@pytest.mark.parametrize("dtype", [np.dtype("uint8"), np.dtype("int32"), np.dtype("float32")])
def test_real_integer_and_float_count_dtypes_are_supported(dtype: np.dtype) -> None:
    descriptor = _descriptor(dtype=dtype)
    descriptor.validate_schema(_schema(descriptor=descriptor))


@pytest.mark.parametrize("dtype", [np.dtype(bool), np.dtype("complex64")])
def test_bool_and_complex_count_dtypes_are_forbidden(dtype: np.dtype) -> None:
    with pytest.raises(TypeError, match="real integer or floating"):
        _descriptor(dtype=dtype)


def test_frame_contract_is_immutable() -> None:
    contract = _contract()
    with pytest.raises(FrozenInstanceError):
        contract.binding = ReadoutBindingKey("other")  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        contract.roi_shape_yx = (1, 1)  # type: ignore[misc]


def test_event_schedule_coverage_and_event_axis_metadata_are_strict() -> None:
    descriptor = _descriptor(event_settings=_event_settings()[:2])
    with pytest.raises(ValueError, match="cover every"):
        descriptor.validate_schema(_schema(descriptor=descriptor))
    point_table, _grid_topology = _default_point_domain()
    wrong_role = replace(
        point_table,
        columns=(
            replace(point_table.columns[0], role=SCAN_POINT),
            point_table.columns[1],
        ),
    )
    with pytest.raises(ValueError, match="wrong role"):
        _descriptor().validate_schema(_schema(point_table=wrong_role))


def test_event_setting_permutation_has_one_canonical_capture_descriptor() -> None:
    forward = _descriptor()
    permuted = _descriptor(
        event_settings=(_event_settings()[2], _event_settings()[0], _event_settings()[1])
    )
    assert permuted == forward
    assert camera_capture_descriptor_to_tree(
        permuted
    ) == camera_capture_descriptor_to_tree(
        forward
    )


def test_camera_descriptor_tree_matches_an_independent_field_oracle() -> None:
    common = {
        "camera_identity": "qcm-camera:serial-42",
        "sensor_identity": "qcm-sensor:serial-42",
        "optical_path": "science-imaging-v1",
        "sensor_shape_yx": [100, 120],
        "roi_origin_yx": [10, 20],
        "roi_shape_yx": [40, 60],
        "binning_yx": [2, 3],
        "spatial_y_axis_id": "camera-y",
        "spatial_x_axis_id": "camera-x",
        "coordinate_frame": "qcm-camera-pixels",
        "dtype": "<u2",
        "count_unit": "camera-count",
    }
    descriptor_tree = {
        **common,
        "readout_event_axis_id": "readout-event",
        "event_settings": [
            {
                "event_index": 0,
                "exposure_seconds": 0.001,
                "gain": 1.0,
                "readout_mode": "fast",
            },
            {
                "event_index": 1,
                "exposure_seconds": 0.002,
                "gain": 2.0,
                "readout_mode": "low-noise",
            },
            {
                "event_index": 2,
                "exposure_seconds": 0.003,
                "gain": 3.0,
                "readout_mode": "low-noise",
            },
        ],
    }
    assert camera_capture_descriptor_to_tree(_descriptor()) == descriptor_tree
    assert camera_capture_descriptor_from_tree(descriptor_tree) == _descriptor()
    assert readout_binding_key_to_tree(BINDING) == {"value": "primary-readout"}
    assert readout_binding_key_from_tree({"value": "primary-readout"}) == BINDING

    for parser, tree in ((camera_capture_descriptor_from_tree, descriptor_tree),):
        unknown = dict(tree)
        unknown["unknown_future_field"] = "forbidden"
        with pytest.raises(ValueError, match="exactly"):
            parser(unknown)


def test_roi_binning_dtype_unit_frame_and_axis_order_are_cross_validated() -> None:
    descriptor = _descriptor()
    height, width = descriptor.output_shape_yx
    invalid_schemas = (
        _schema(descriptor=descriptor, dtype=np.dtype("uint32")),
        _schema(descriptor=descriptor, unit="electron"),
        _schema(
            descriptor=descriptor,
            data_axes=(
                AxisSpec(Y, "y", SPATIAL_Y, height, coordinate_frame=FRAME),
                AxisSpec(X, "x", SPATIAL_X, width - 1, coordinate_frame=FRAME),
            ),
        ),
        _schema(
            descriptor=descriptor,
            data_axes=(
                AxisSpec(
                    Y,
                    "y",
                    SPATIAL_Y,
                    height,
                    coordinate_frame=CoordinateFrameId("wrong-frame"),
                ),
                AxisSpec(X, "x", SPATIAL_X, width, coordinate_frame=FRAME),
            ),
        ),
        _schema(
            descriptor=descriptor,
            data_axes=(
                AxisSpec(X, "x", SPATIAL_X, width, coordinate_frame=FRAME),
                AxisSpec(Y, "y", SPATIAL_Y, height, coordinate_frame=FRAME),
            ),
        ),
    )
    for schema in invalid_schemas:
        with pytest.raises(ValueError):
            descriptor.validate_schema(schema)


def test_no_event_axis_requires_exactly_one_index_zero_setting() -> None:
    with pytest.raises(ValueError, match="event index 0"):
        _descriptor(
            readout_event_axis_id=None,
            event_settings=(CameraEventReadoutSetting(1, 0.001, 1.0, "fast"),),
        )
