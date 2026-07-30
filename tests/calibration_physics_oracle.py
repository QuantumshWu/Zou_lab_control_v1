"""Array-input harness for current calibration physics oracles.

Production calibration accepts a loaded capture.  The deterministic tests
below start from frozen arrays, so this module adapts only their input shape to
the same product science core; it owns no algorithm, artifact authority, or
public compatibility surface.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from zlc_data import AxisId
from zlc_neutral_atom.logic_nodes.readout.calibration.analysis import (
    CalibrationComputation,
    _calibrate_readout_source,
)
from zlc_neutral_atom.logic_nodes.readout.calibration.calibration import (
    CalibrationAnalysisRequest,
    CalibrationSourceBinding,
)
from zlc_neutral_atom.logic_nodes.readout.contracts import FrameContract
from zlc_neutral_atom.logic_nodes.readout.physical_context import (
    ReadoutPhysicalContext,
)


def calibrate_readout_arrays_for_test(
    reference_frames: np.ndarray,
    short_frames: np.ndarray,
    *,
    source_binding: CalibrationSourceBinding,
    frame_contract: FrameContract,
    readout_physical_context: ReadoutPhysicalContext,
    request: CalibrationAnalysisRequest,
    reference_validity: np.ndarray | None = None,
    short_validity: np.ndarray | None = None,
) -> CalibrationComputation:
    """Feed frozen ``(G,K,H,W)``/``(G,H,W)`` facts to the product core."""

    references = np.asarray(reference_frames)
    short = np.asarray(short_frames)
    if references.ndim != 4 or short.ndim != 3:
        raise ValueError("calibration oracle arrays have invalid ranks")
    if references.shape[0] != short.shape[0] or (
        references.shape[-2:] != short.shape[-2:]
    ):
        raise ValueError("calibration oracle arrays differ in group/image shape")
    reference_valid = (
        np.broadcast_to(True, references.shape)
        if reference_validity is None
        else np.asarray(reference_validity, dtype=bool)
    )
    short_valid = (
        np.broadcast_to(True, short.shape)
        if short_validity is None
        else np.asarray(short_validity, dtype=bool)
    )
    if reference_valid.shape != references.shape or short_valid.shape != short.shape:
        raise ValueError("calibration oracle validity has another shape")

    def reference_sequence() -> Iterator[tuple[np.ndarray, np.ndarray]]:
        for group, shot in np.ndindex(references.shape[:2]):
            yield references[group, shot], reference_valid[group, shot]

    def short_sequence() -> Iterator[tuple[np.ndarray, np.ndarray]]:
        for group in range(short.shape[0]):
            yield short[group], short_valid[group]

    group_axis = AxisId("calibration-test-group")
    return _calibrate_readout_source(
        group_count=references.shape[0],
        reference_shot_count=references.shape[1],
        reference_frames=reference_sequence,
        short_frames=short_sequence,
        group_contexts=tuple(
            ((group_axis, group),) for group in range(references.shape[0])
        ),
        source_binding=source_binding,
        frame_contract=frame_contract,
        readout_physical_context=readout_physical_context,
        request=request,
    )


__all__ = ["calibrate_readout_arrays_for_test"]
