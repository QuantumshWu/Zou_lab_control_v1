"""Calibration axis, validity, and pure-computation contracts."""

from __future__ import annotations

import numpy as np
import pytest

from zlc_data.axis import AxisId
from zlc_neutral_atom.logic_nodes.readout.calibration.calibration import (
    CalibrationAnalysisRequest,
)
from zlc_neutral_atom.logic_nodes.readout.contracts import CalibrationCaptureLayout
from zlc_neutral_atom.logic_nodes.readout.model_contract import (
    ReadoutModelKind,
    readout_model_authoring_schema,
    readout_model_kind_from_authoring,
)


def test_request_owns_independent_expected_center_evidence():
    caller_centers = np.asarray(
        ((5.0, 5.0), (15.0, 5.0), (5.0, 15.0), (15.0, 15.0)),
        dtype="<f8",
    )
    request = CalibrationAnalysisRequest(
        layout=CalibrationCaptureLayout(AxisId("event"), (0, 2), 1),
        grid_shape_yx=(2, 2),
        expected_centers_xy=caller_centers,
        maximum_site_residual_px=2.0,
    )
    caller_centers[:] = -1.0
    assert np.array_equal(
        request.expected_centers_xy,
        np.asarray(((5, 5), (15, 5), (5, 15), (15, 15)), dtype="<f8"),
    )
    assert not request.expected_centers_xy.flags.writeable
    with pytest.raises(ValueError, match="less than half"):
        CalibrationAnalysisRequest(
            layout=request.layout,
            grid_shape_yx=(2, 2),
            expected_centers_xy=np.asarray(
                ((0, 0), (2, 0), (0, 2), (2, 2)), dtype="<f8"
            ),
            maximum_site_residual_px=1.0,
        )


def test_model_batch_is_real_current_domain_not_a_scalar_only_placeholder():
    assert tuple(kind.value for kind in ReadoutModelKind) == (
        "box",
        "psf",
        "uniform_psf",
    )


def test_readout_model_selection_contract_is_owned_by_the_readout_family():
    authored_default = readout_model_authoring_schema().freeze({})["model_kind"]
    assert readout_model_kind_from_authoring(authored_default) is None
    assert readout_model_kind_from_authoring("psf") is ReadoutModelKind.PER_SITE_PSF
    with pytest.raises(ValueError, match="unknown readout model choice"):
        readout_model_kind_from_authoring("missing")
