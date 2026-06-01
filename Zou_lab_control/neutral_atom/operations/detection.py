"""Standalone atom-detection operation."""

from __future__ import annotations

import numpy as np

from ..core.calibration import TrapCalibration
from ..core.results import DetectionResult
from ..timing import PulseSequence


def detect_image(
    image,
    calibration: TrapCalibration,
    *,
    sequence: PulseSequence | None = None,
    display: bool = True,
    what: str = "occupancy",
) -> DetectionResult:
    """Classify one image with a calibration, usable outside a session."""

    sequence = sequence or PulseSequence(name="standalone_detect")
    detection = calibration.detect(image)
    result = DetectionResult(np.asarray(image), detection, calibration, sequence)
    if str(what).lower() in {"counts", "count"}:
        result.plot_counts(display=display)
    else:
        result.plot_occupancy(display=display)
    return result


__all__ = ["detect_image"]
