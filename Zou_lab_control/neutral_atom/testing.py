"""White-box simulation surfaces for tests and adapter contract kits.

Production notebooks use :func:`neutral_atom.connect`.  Tests that deliberately
need a raw fake import it from this visibly non-production namespace instead of
recovering hardware from an Experiment or the ordinary umbrella.
"""

from __future__ import annotations

from .devices.sequencer import ManualSequencer, bind_pulse as bind_test_pulse
from .devices.virtual import (
    DEFAULT_CHANNELS,
    VirtualCamera,
    VirtualLaser,
    VirtualRF,
    VirtualSequencer,
    VirtualTrapArray,
)

__all__ = [
    "DEFAULT_CHANNELS",
    "ManualSequencer",
    "VirtualCamera",
    "VirtualLaser",
    "VirtualRF",
    "VirtualSequencer",
    "VirtualTrapArray",
    "bind_test_pulse",
]
