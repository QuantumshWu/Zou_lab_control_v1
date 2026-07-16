"""Contracts implemented by composition-owned camera adapters.

This namespace is for adapter authors and composition roots.  Ordinary
Experiment objects do not expose adapters or their physical records.
"""

from .camera import (
    CameraAdapter,
    CameraCaptureTerminalRecord,
    CameraFrameRecord,
    CameraWorkingPoint,
)

__all__ = [
    "CameraAdapter",
    "CameraCaptureTerminalRecord",
    "CameraFrameRecord",
    "CameraWorkingPoint",
]
