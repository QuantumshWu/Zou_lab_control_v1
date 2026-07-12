"""Explicit contracts for authors of installation adapters.

This namespace is intentionally separate from the ordinary notebook facade.  It
contains device-domain base contracts and discovery value types, but no concrete
qCMOS/FPGA adapter, open connection, global registry mutation, or drive-capable
session object.
"""

from __future__ import annotations

from .devices.base import (
    BaseDevice,
    CameraBufferOverrun,
    CameraFrameRecord,
    CameraDevice,
    LaserDevice,
    RFSourceDevice,
    RuntimeControl,
    SequencerDevice,
    TrapArrayDevice,
)
from .devices.discovery import DiscoveredDevice, discovery_note
from .devices.registry import validate_device_contract

__all__ = [
    "BaseDevice",
    "CameraBufferOverrun",
    "CameraDevice",
    "CameraFrameRecord",
    "DiscoveredDevice",
    "LaserDevice",
    "RFSourceDevice",
    "RuntimeControl",
    "SequencerDevice",
    "TrapArrayDevice",
    "discovery_note",
    "validate_device_contract",
]
