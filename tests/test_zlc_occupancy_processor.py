"""Current committed-calibration occupancy contracts."""

from __future__ import annotations

import json
from pathlib import Path
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


def _axis(name: str, role, size: int) -> AxisSpec:
    return AxisSpec(AxisId(name), name, role, size, tuple(range(size)))


def _run_isolated(script: str, workspace: Path) -> dict[str, object]:
    completed = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script), str(workspace)],
        cwd=ROOT,
        text=True,
        capture_output=True,
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
    result = _run_isolated(
        """
        from dataclasses import replace
        import json
        from pathlib import Path
        import sys

        import numpy as np
        from Zou_lab_control.api import WorkspacePaths, connect
        from zlc_neutral_atom.capture.artifact import CaptureRepository
        from zlc_neutral_atom.logic_nodes.readout.calibration.repository import (
            CalibrationRepository,
        )
        from zlc_neutral_atom.logic_nodes.readout.occupancy.repository import (
            OccupancyRepository,
        )
        from zlc_pulse import RepeatRegion, load_pulse_document
        from zlc_storage import content_ref_from_tree, decode

        workspace = Path(sys.argv[1])
        experiment = connect(
            "virtual",
            workspace=WorkspacePaths.for_workspace(
                Path.cwd(),
                repository_root=workspace,
            ),
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
            }
        finally:
            experiment.close()

        captures = CaptureRepository(workspace / "captures")
        calibrations = CalibrationRepository(workspace / "calibrations")
        occupancies = OccupancyRepository(workspace / "occupancy")
        try:
            reopened = occupancies.admit(
                occupancy_ref,
                captures,
                calibrations,
            )
            manifest = decode(
                occupancies._store_authority.read_manifest(
                    "occupancy",
                    occupancy_ref.manifest_digest,
                )
            )
            metadata_ref = content_ref_from_tree(manifest["metadata_blob"])
            metadata = decode(
                occupancies._store_authority.read_blob(metadata_ref)
            )
            result.update(
                {
                    "reopened_capture_ref_matches": (
                        reopened.artifact.source_capture_ref == capture_ref
                    ),
                    "reopened_calibration_ref_matches": (
                        reopened.artifact.calibration_reference == calibration_ref
                    ),
                    "canonical_run_id": metadata["run_id"],
                }
            )
        finally:
            occupancies.close()
            calibrations.close()
            captures.close()
        print("RESULT_JSON=" + json.dumps(result, sort_keys=True))
        """,
        tmp_path / "occupancy-current",
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
    assert result["invalid_count_fillers_are_zero"] is True
    assert result["invalid_occupied_fillers_are_false"] is True
