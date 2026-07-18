"""Current committed-calibration occupancy contracts."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from zlc_data import (
    READOUT_EVENT,
    REPEAT,
    SITE,
    AxisId,
    AxisSpec,
    ComponentValidity,
    DatasetSchema,
    PointLayout,
    ValidityContract,
    Value,
    ValueSchema,
)
from zlc_neutral_atom.readout.occupancy import (
    _require_occupancy_output_schemas,
    _validate_sample_fields,
)


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


def test_output_schema_keeps_repeat_point_and_site_as_distinct_axes():
    repeat = _axis("repeat", REPEAT, 2)
    event = _axis("event", READOUT_EVENT, 1)
    site = _axis("site", SITE, 3)
    validity_contract = ValidityContract.components(site.axis_id)
    counts_schema = DatasetSchema(
        repeat,
        (event,),
        PointLayout.rect_c((1,)),
        ValueSchema(
            (site,),
            validity_contract,
            np.dtype("<f8"),
            value_unit="count",
        ),
    )
    occupied_schema = DatasetSchema(
        repeat,
        (event,),
        PointLayout.rect_c((1,)),
        ValueSchema(
            (site,),
            validity_contract,
            np.dtype(bool),
            value_unit="occupation",
        ),
    )
    assert _require_occupancy_output_schemas(counts_schema, occupied_schema) == site
    assert counts_schema.physical_shape == (2, 1, 3)
    assert counts_schema.cell_schema.data_shape == (3,)


def test_component_validity_is_per_site_and_invalid_fillers_are_canonical():
    site = _axis("site", SITE, 3)
    contract = ValidityContract.components(site.axis_id)
    validity = ComponentValidity((site.axis_id,), np.asarray((True, False, True)))
    counts_schema = ValueSchema(
        (site,), contract, np.dtype("<f8"), value_unit="count"
    )
    occupied_schema = ValueSchema(
        (site,), contract, np.dtype(bool), value_unit="occupation"
    )
    counts = Value(np.asarray((1.0, 0.0, 2.0)), validity, counts_schema)
    occupied = Value(np.asarray((True, False, True)), validity, occupied_schema)
    _validate_sample_fields(counts, occupied)

    negative_zero = Value(
        np.asarray((1.0, -0.0, 2.0)), validity, counts_schema
    )
    with pytest.raises(ValueError, match="positive-zero"):
        _validate_sample_fields(negative_zero, occupied)
    invalid_true = Value(
        np.asarray((True, True, True)), validity, occupied_schema
    )
    with pytest.raises(ValueError, match="canonical False"):
        _validate_sample_fields(counts, invalid_true)


def test_committed_detection_preserves_r_p_site_and_binds_both_artifacts(tmp_path):
    result = _run_isolated(
        """
        from dataclasses import replace
        import json
        from pathlib import Path
        import sys

        import numpy as np
        from Zou_lab_control.notebook import connect
        from zlc_pulse import RepeatRegion, load_pulse_document

        workspace = Path(sys.argv[1])
        experiment = connect("virtual", repository=workspace, seed=7)
        try:
            calibration_ref = experiment.readout.sitemap(frames=4)
            document = load_pulse_document(
                Path("zlc_neutral_atom/assets/imaging_template.json")
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
            detection_request = experiment.readout.detection_request(
                capture_ref,
                calibration_ref,
            )
            occupancy_ref = experiment.readout.detect(detection_request)
            resolved = experiment.readout.load_occupancy(occupancy_ref)
            artifact = resolved.artifact
            validity = artifact.counts.validity
            invalid = ~validity.mask
            result = {
                "capture_descriptor_shape": list(descriptor.output_shape),
                "counts_shape": list(artifact.counts.values.shape),
                "occupied_shape": list(artifact.occupied.values.shape),
                "validity_shape": list(validity.mask.shape),
                "validity_axes": [axis.value for axis in validity.axis_ids],
                "same_validity_owner": (
                    artifact.counts.validity is artifact.occupied.validity
                ),
                "repeat_role": artifact.counts.schema.repeat_axis.role.value,
                "point_roles": [
                    axis.role.value for axis in artifact.counts.schema.point_axes
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
                "invalid_count_fillers_are_zero": bool(
                    np.all(artifact.counts.values[invalid] == 0.0)
                    and not np.any(np.signbit(artifact.counts.values[invalid]))
                ),
                "invalid_occupied_fillers_are_false": bool(
                    not np.any(artifact.occupied.values[invalid])
                ),
            }
            print("RESULT_JSON=" + json.dumps(result, sort_keys=True))
        finally:
            experiment.close()
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
    assert result["invalid_count_fillers_are_zero"] is True
    assert result["invalid_occupied_fillers_are_false"] is True


def test_normal_committed_path_has_one_repository_compiler_owner():
    from zlc_neutral_atom.readout.occupancy_repository import (
        compile_occupancy_artifact_plan,
    )
    import zlc_neutral_atom.readout.occupancy_repository as repository_module

    assert compile_occupancy_artifact_plan.__module__.endswith(
        "readout.occupancy_repository"
    )
    assert not hasattr(repository_module, "OccupancyCheckpoint")
    assert not hasattr(repository_module, "LegacyOccupancyProcessor")
