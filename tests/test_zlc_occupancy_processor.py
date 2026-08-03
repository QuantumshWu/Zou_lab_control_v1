"""Current committed-calibration occupancy contracts."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap
import time

import numpy as np
import pytest

from zlc_data.axis import SITE, AxisId, AxisSpec
from zlc_data.schema import ValueSchema
from zlc_data.validity import ComponentValidity, ValidityContract
from zlc_data.value import Value


ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize(
    "alias",
    (
        "occupancy-run//occupancy.json",
        "occupancy-run/./occupancy.json",
        "occupancy\\alias/occupancy.json",
    ),
)
def test_occupancy_artifact_ref_rejects_noncanonical_path_aliases(
    alias: str,
) -> None:
    from zlc_neutral_atom.logic_nodes.readout.occupancy.reference import (
        OccupancyArtifactRef,
    )

    with pytest.raises(ValueError):
        OccupancyArtifactRef(alias)


def _axis(name: str, role, size: int) -> AxisSpec:
    return AxisSpec(AxisId(name), name, role, size, tuple(range(size)))


def _workspace_with_pulse(root: Path, name: str) -> Path:
    pulses = root / "pulses"
    pulses.mkdir(parents=True)
    shutil.copy2(ROOT / "pulses" / name, pulses / name)
    return root


def _run_isolated(script: str, workspace: Path) -> dict[str, object]:
    completed = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script), str(workspace)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        timeout=60,
        check=False,
    )
    if completed.returncode:
        pytest.fail(
            "isolated occupancy probe failed\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    marker = "RESULT_JSON="
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith(marker):
            return json.loads(line[len(marker) :])
    pytest.fail(f"isolated probe returned no result marker: {completed.stdout}")


def test_committed_detection_preserves_r_p_site_and_binds_both_artifacts(tmp_path):
    workspace = _workspace_with_pulse(
        tmp_path / "occupancy-current",
        "imaging_template.json",
    )
    result = _run_isolated(
        """
        from dataclasses import replace
        from io import BytesIO
        import json
        from pathlib import Path
        import sys
        import time

        import numpy as np
        from PyQt5 import QtWidgets
        from Zou_lab_control.api import WorkspacePaths, connect
        from zlc_frontend.qt_widgets import ensure_qt_app
        from zlc_neutral_atom.logic_nodes.readout.occupancy.artifact import (
            load_occupancy_artifact,
        )
        from zlc_pulse import RepeatRegion, load_pulse_document
        from zlc_storage import decode
        from zlc_neutral_atom.runtime.dataset import DatasetCellAddress

        workspace = Path(sys.argv[1])
        experiment = connect(
            "virtual",
            workspace=WorkspacePaths.for_workspace(workspace),
            seed=7,
        )
        try:
            calibration_ref = experiment.nodes.calibration.sitemap(frames=4)
            current_calibration_matches = (
                experiment.nodes.calibration.current_calibration_ref
                == calibration_ref
            )
            experiment.nodes.calibration.current_calibration_ref = None
            document = load_pulse_document(
                Path("pulses/imaging_template.json")
            )
            trigger_index = document.target.raw_lanes.index("ch11")
            periods = []
            for index, period in enumerate(document.periods):
                states = list(period.states)
                if index in (1, 5):
                    states[trigger_index] = 0
                periods.append(replace(period, states=tuple(states)))
            document = replace(
                document,
                periods=tuple(periods),
                repeat=RepeatRegion(
                    document.periods[0].period_id,
                    document.periods[-1].period_id,
                    2,
                ),
            )
            capture_request = experiment.readout.capture_request(
                document,
                repeat_count=2,
                readout_events_per_repeat=1,
            )
            descriptor = experiment.inspect(capture_request)
            capture_ref = experiment.run(capture_request)
            explicit_detection_request = (
                experiment.nodes.occupancy.detection_request(
                    capture_ref,
                    calibration_ref,
                )
            )
            try:
                experiment.nodes.occupancy.detection_request(capture_ref)
            except RuntimeError as error:
                missing_current_error = "no current Calibration" in str(error)
            else:
                missing_current_error = False
            experiment.nodes.calibration.current_calibration_ref = calibration_ref
            detection_request = experiment.nodes.occupancy.detection_request(
                capture_ref,
            )
            occupancy_ref = experiment.nodes.occupancy.detect(detection_request)
            resolved = experiment.nodes.occupancy.load_occupancy(occupancy_ref)
            artifact = resolved.artifact
            cell_session = experiment.nodes.occupancy.occupancy_cell_view(
                occupancy_ref,
                address=DatasetCellAddress(0, 0),
            )
            try:
                image_stream = BytesIO()
                cell_session.save(image_stream, format="png")
                cell_overlay = cell_session.image_overlay
                cell_plot = {
                    "kind": cell_session.surface_plan.kind,
                    "overlay_count": cell_overlay.count,
                    "status_values": sorted(
                        {status.value for status in cell_overlay.statuses}
                    ),
                    "png_bytes": len(image_stream.getvalue()),
                }
            finally:
                cell_session.close()
            application = ensure_qt_app()
            cell_window = experiment.nodes.occupancy.occupancy_cell_gui(
                occupancy_ref,
                address=DatasetCellAddress(0, 0),
            )
            deadline = time.monotonic() + 15.0
            while not cell_window.raster_ready and time.monotonic() < deadline:
                application.processEvents()
                time.sleep(0.005)
            cell_window.show()
            application.processEvents()
            grabbed = cell_window.grab()
            cell_gui = {
                "raster_ready": cell_window.raster_ready,
                "grab_valid": not grabbed.isNull(),
                "plot_host_count": int(
                    cell_window.findChild(
                        QtWidgets.QWidget,
                        "occupancyCellPlot",
                    )
                    is not None
                ),
            }
            cell_window.shutdown()
            deadline = time.monotonic() + 15.0
            while not cell_window.permanently_closed and time.monotonic() < deadline:
                application.processEvents()
                time.sleep(0.005)
            cell_gui["closed"] = cell_window.permanently_closed
            validity = artifact.counts.validity
            invalid = ~validity.mask
            result = {
                "capture_descriptor_shape": list(
                    descriptor.output_schema.physical_shape
                ),
                "counts_shape": list(artifact.counts.values.shape),
                "occupied_shape": list(artifact.occupied.values.shape),
                "validity_shape": list(validity.mask.shape),
                "validity_axes": [axis.value for axis in validity.axis_ids],
                "same_validity_owner": (
                    artifact.counts.validity is artifact.occupied.validity
                ),
                "repeat_role": artifact.counts.schema.repeat_axis.role.value,
                "point_roles": [
                    column.role.value
                    for column in artifact.counts.schema.point_table.columns
                ],
                "data_roles": [
                    axis.role.value
                    for axis in artifact.counts.schema.cell_schema.data_axes
                ],
                "model_kind": artifact.model_kind.value,
                "capture_ref_matches": artifact.source_capture_ref == capture_ref,
                "calibration_ref_matches": (
                    artifact.calibration_reference == calibration_ref
                ),
                "current_calibration_matches": current_calibration_matches,
                "explicit_and_default_requests_match": (
                    explicit_detection_request == detection_request
                ),
                "missing_current_error": missing_current_error,
                "invalid_count_fillers_are_zero": bool(
                    np.all(artifact.counts.values[invalid] == 0.0)
                    and not np.any(np.signbit(artifact.counts.values[invalid]))
                ),
                "invalid_occupied_fillers_are_false": bool(
                    not np.any(artifact.occupied.values[invalid])
                ),
                "cell_plot": cell_plot,
                "cell_gui": cell_gui,
            }
        finally:
            experiment.close()

        output = workspace / "_output"
        occupancy_root = output / "occupancy"
        reopened = load_occupancy_artifact(
            occupancy_root,
            output / "captures",
            output / "calibrations",
            occupancy_ref,
        )
        record_path = occupancy_root / occupancy_ref.record_path
        record = decode(record_path.read_bytes())
        result.update(
            {
                "reopened_capture_ref_matches": (
                    reopened.artifact.source_capture_ref == capture_ref
                ),
                "reopened_calibration_ref_matches": (
                    reopened.artifact.calibration_reference == calibration_ref
                ),
                "canonical_run_id": record["run_id"],
                "record_path": occupancy_ref.record_path,
                "record_fields": sorted(record),
                "counts_dtype": str(
                    np.load(record_path.parent / "counts.npy", allow_pickle=False).dtype
                ),
                "occupied_dtype": str(
                    np.load(record_path.parent / "occupied.npy", allow_pickle=False).dtype
                ),
                "validity_dtype": str(
                    np.load(record_path.parent / "validity.npy", allow_pickle=False).dtype
                ),
            }
        )
        print("RESULT_JSON=" + json.dumps(result, sort_keys=True))
        """,
        workspace,
    )
    assert result["capture_descriptor_shape"] == [2, 1, 96, 128]
    assert result["counts_shape"] == [2, 1, 35]
    assert result["occupied_shape"] == [2, 1, 35]
    assert result["validity_shape"] == [2, 1, 35]
    assert result["validity_axes"] == ["readout-site"]
    assert result["same_validity_owner"] is True
    assert result["repeat_role"] == "repeat"
    assert result["point_roles"] == ["readout-event"]
    assert result["data_roles"] == ["site"]
    assert result["model_kind"] == "box"
    assert result["capture_ref_matches"] is True
    assert result["calibration_ref_matches"] is True
    assert result["current_calibration_matches"] is True
    assert result["explicit_and_default_requests_match"] is True
    assert result["missing_current_error"] is True
    assert result["reopened_capture_ref_matches"] is True
    assert result["reopened_calibration_ref_matches"] is True
    assert isinstance(result["canonical_run_id"], str)
    assert result["canonical_run_id"]
    assert result["record_path"].endswith("/occupancy.json")
    assert result["record_fields"] == [
        "calibration_ref",
        "counts_schema",
        "model_kind",
        "occupied_schema",
        "readout_binding",
        "readout_event_axis_id",
        "run_id",
        "schema",
        "source_capture_ref",
    ]
    assert result["counts_dtype"] == "float64"
    assert result["occupied_dtype"] == "bool"
    assert result["validity_dtype"] == "bool"
    assert result["invalid_count_fillers_are_zero"] is True
    assert result["invalid_occupied_fillers_are_false"] is True
    assert result["cell_plot"]["kind"] == "image"
    assert result["cell_plot"]["overlay_count"] == 35
    assert set(result["cell_plot"]["status_values"]) <= {
        "empty",
        "occupied",
        "invalid",
    }
    assert result["cell_plot"]["png_bytes"] > 1000
    assert result["cell_gui"] == {
        "closed": True,
        "grab_valid": True,
        "plot_host_count": 1,
        "raster_ready": True,
    }


def test_occupancy_publishes_only_signals_without_leaf_presentation_sidecars():
    from zlc_neutral_atom.logic_nodes.readout.occupancy.package import (
        LOGIC_NODE_PACKAGE,
    )

    assert LOGIC_NODE_PACKAGE.declaration.task_previews == ()
    assert not hasattr(LOGIC_NODE_PACKAGE, "project_signal_presentation")
    assert getattr(LOGIC_NODE_PACKAGE, "bind_artifact_capabilities", None) is None


def _direct_output_artifact():
    from zlc_data import (
        REPEAT,
        SITE,
        AxisId,
        AxisSpec,
        DataBlock,
        DatasetComponentValidity,
        DatasetRevision,
        DatasetSchema,
        PointTable,
        ValidityContract,
        ValueSchema,
    )
    from zlc_neutral_atom.capture.reference import CaptureArtifactRef
    from zlc_neutral_atom.logic_nodes.readout.calibration.reference import (
        CalibrationArtifactRef,
    )
    from zlc_neutral_atom.logic_nodes.readout.model_contract import ReadoutModelKind
    from zlc_neutral_atom.logic_nodes.readout.occupancy.processor import (
        OCCUPANCY_COUNTS_BLOCK_ID,
        OCCUPANCY_OCCUPIED_BLOCK_ID,
        OccupancyArtifact,
    )

    site = AxisSpec(
        AxisId("site"),
        "site",
        SITE,
        2,
        (0, 1),
    )
    repeat = AxisSpec(
        AxisId("repeat"),
        "repeat",
        REPEAT,
        1,
        (0,),
    )
    validity_contract = ValidityContract.components(site.axis_id)
    counts_schema = DatasetSchema(
        repeat,
        PointTable(1),
        None,
        ValueSchema((site,), validity_contract, np.dtype("<f8"), "count"),
    )
    occupied_schema = DatasetSchema(
        repeat,
        PointTable(1),
        None,
        ValueSchema((site,), validity_contract, np.dtype(bool), "occupation"),
    )
    validity = DatasetComponentValidity(
        (site.axis_id,),
        np.array([[[True, False]]], dtype=bool),
    )
    revision = DatasetRevision(3)
    return OccupancyArtifact(
        CaptureArtifactRef("source/capture.json"),
        CalibrationArtifactRef("calibration/calibration.json"),
        AxisId("readout-event"),
        ReadoutModelKind.BOX,
        "occupancy-run",
        DataBlock(
            OCCUPANCY_COUNTS_BLOCK_ID,
            revision,
            np.array([[[7.5, 0.0]]], dtype="<f8"),
            validity,
            counts_schema,
        ),
        DataBlock(
            OCCUPANCY_OCCUPIED_BLOCK_ID,
            revision,
            np.array([[[True, False]]], dtype=bool),
            validity,
            occupied_schema,
        ),
    )


def test_direct_output_writes_original_arrays_before_record(tmp_path):
    from zlc_neutral_atom.devices.camera.contract import ReadoutBindingKey
    from zlc_neutral_atom.logic_nodes.readout.occupancy.artifact import (
        write_occupancy_artifact,
    )
    from zlc_storage import decode

    root = (tmp_path / "occupancy").resolve()
    artifact = _direct_output_artifact()
    reference = write_occupancy_artifact(
        root,
        artifact,
        readout_binding=ReadoutBindingKey("camera"),
        run_id=artifact.run_id,
    )
    record_path = root / reference.record_path
    record = decode(record_path.read_bytes())
    assert reference.record_path == "occupancy-run/occupancy.json"
    assert record["run_id"] == artifact.run_id
    assert np.load(record_path.parent / "counts.npy", allow_pickle=False).dtype == np.dtype(
        "<f8"
    )
    assert np.load(
        record_path.parent / "occupied.npy",
        allow_pickle=False,
    ).dtype == np.dtype(bool)
    assert np.load(
        record_path.parent / "validity.npy",
        allow_pickle=False,
    ).dtype == np.dtype(bool)


def test_record_failure_leaves_no_visible_occupancy_artifact(tmp_path, monkeypatch):
    import zlc_neutral_atom.logic_nodes.readout.occupancy.artifact as artifact_io
    from zlc_neutral_atom.devices.camera.contract import ReadoutBindingKey

    root = (tmp_path / "occupancy").resolve()
    artifact = _direct_output_artifact()

    def fail_record(_target, _payload):
        raise OSError("synthetic record publication failure")

    monkeypatch.setattr(artifact_io, "atomic_write_bytes", fail_record)
    with pytest.raises(OSError, match="record publication failure"):
        artifact_io.write_occupancy_artifact(
            root,
            artifact,
            readout_binding=ReadoutBindingKey("camera"),
            run_id=artifact.run_id,
        )

    run_directory = root / artifact.run_id
    assert (run_directory / "counts.npy").is_file()
    assert (run_directory / "occupied.npy").is_file()
    assert (run_directory / "validity.npy").is_file()
    assert not (run_directory / "occupancy.json").exists()
