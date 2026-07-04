"""Contract: every camera backend's ``configure`` REJECTS an unknown keyword the
SAME way (virtual == real).

The base ``CameraDevice.configure`` signature carries ``**kwargs``; each backend
declares the options it accepts (always ``exposure``) and routes leftover keys
through the shared :meth:`CameraDevice._reject_unknown_configure_keys`.  So a
mistyped or backend-specific option fails LOUDLY and identically on both the
virtual and the real camera -- never a silent no-op on one and a ``TypeError`` on
the other (which would be a virtual != real break, masking a typo offline that
crashes on the real qCMOS).
"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if sys.path[0] != str(REPO_ROOT):
    sys.path.insert(0, str(REPO_ROOT))


def _virtual_camera():
    from Zou_lab_control.neutral_atom.devices.virtual import VirtualCamera, VirtualTrapArray
    return VirtualCamera(VirtualTrapArray(grid_shape=(2, 2), image_shape=(16, 16), seed=1), exposure=0.01)


def _qcmos_camera():
    # No DCAM needed: configure only touches hardware once the device is open;
    # the unknown-key guard runs at the method entry, before any hardware write.
    from Zou_lab_control.neutral_atom.devices.qcmos import QCMOSCamera
    return QCMOSCamera()


@pytest.mark.parametrize("make_cam", [_virtual_camera, _qcmos_camera])
def test_configure_unknown_key_raises_valueerror(make_cam):
    cam = make_cam()
    with pytest.raises(ValueError) as exc:
        cam.configure(nonsense=1)
    # The error names the configurable options (so the operator knows the valid keys),
    # and it is a ValueError on EVERY backend -- not a TypeError on one, a no-op on another.
    assert "exposure" in str(exc.value)


@pytest.mark.parametrize("make_cam,allowed", [
    (_virtual_camera, {"exposure", "roi"}),
    (_qcmos_camera, {"exposure", "readout_speed", "roi"}),
])
def test_configure_accepts_its_declared_options(make_cam, allowed):
    cam = make_cam()
    # Each declared option is accepted (exposure on every backend; the rest per backend).
    cam.configure(exposure=0.02)
    if "roi" in allowed:
        cam.configure(roi="None")          # the clear-sentinel -> full frame, no hardware needed
    if "readout_speed" in allowed:
        cam.configure(readout_speed=1)


def test_both_backends_share_the_base_reject_helper():
    # The rejection logic is defined ONCE on the base, so the contract cannot drift
    # between backends.
    from Zou_lab_control.neutral_atom.devices.base import CameraDevice
    from Zou_lab_control.neutral_atom.devices.qcmos import QCMOSCamera
    from Zou_lab_control.neutral_atom.devices.virtual import VirtualCamera

    assert hasattr(CameraDevice, "_reject_unknown_configure_keys")
    for cls in (VirtualCamera, QCMOSCamera):
        assert cls._reject_unknown_configure_keys is CameraDevice._reject_unknown_configure_keys, (
            f"{cls.__name__} overrides the shared unknown-key guard; it must stay single-source on the base"
        )
