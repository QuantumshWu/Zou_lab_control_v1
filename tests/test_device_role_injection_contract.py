"""Device selection is wired in ONE place (the base), never hand-rolled per spec.

A measurement / task / processor that needs a device DECLARES its role on the decorator
(``@measurement(devices=["camera"])`` / imperative ``spec.with_devices_bound(...)``) and the
base (:meth:`CatalogSpec.with_devices_bound`, called at the registry funnel) turns each role
into a ``choice`` param AND injects the RESOLVED device into the build closure.  These contracts
mechanically pin that so no future spec regresses to ``choices=camera_names()`` /
``s.devices[name]`` in its own factory:

* Pin A -- no spec-factory file hand-rolls a camera dropdown (AST: no call to
  ``camera_names`` / ``default_camera_name`` anywhere in the built-in factory packages).
* Pin B -- every catalog spec that carries a ``camera`` choice param has REAL device-name
  choices + default (the base built it from the live DeviceSet).
* Pin C -- the injected value reaching a build closure is a resolved ``CameraDevice``
  instance, never the choice STRING.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if sys.path[0] != str(REPO_ROOT):
    sys.path.insert(0, str(REPO_ROOT))

import pytest

OPS = REPO_ROOT / "Zou_lab_control" / "neutral_atom" / "operations"


def _factory_files():
    """Every built-in spec-factory module (measurements / tasks / processors)."""
    for pkg in ("measurements", "tasks", "processors"):
        yield from sorted((OPS / pkg).glob("*.py"))


@pytest.mark.parametrize("path", list(_factory_files()), ids=lambda p: f"{p.parent.name}/{p.name}")
def test_no_hand_rolled_camera_dropdown(path):
    """A spec factory must NOT build its own camera choice: the ``choices``/``default`` for a
    device dropdown come from the base's device-role injection, not a call to the device-name
    helpers.  AST (not grep) so a docstring that MENTIONS the helper is not a false positive."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    banned = {"camera_names", "default_camera_name"}
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", None)
            if name in banned:
                hits.append((name, getattr(node, "lineno", "?")))
    assert not hits, (
        f"{path.parent.name}/{path.name} hand-rolls a camera dropdown {hits}; declare "
        "devices=['camera'] on the decorator and let the base append + resolve it.")


# ------------------------------------------------------------------ live-catalog contracts
@pytest.fixture(scope="module")
def readout():
    import Zou_lab_control.neutral_atom as na
    exp = na.connect("virtual")
    return exp.readout


def _camera_param(spec):
    for p in spec.params:
        if getattr(p, "key", None) == "camera":
            return p
    return None


def test_declared_camera_roles_become_real_choice_params(readout):
    """Every catalog spec that exposes a ``camera`` role has a ``choice`` param whose choices are
    the live DeviceSet's cameras and whose default is one of them -- proof the base built the
    dropdown from the resolved devices, not a hardcoded list."""
    from Zou_lab_control.neutral_atom.devices.base import CameraDevice

    cams = readout.session.devices.device_names(CameraDevice)
    assert cams, "virtual config must load at least one camera"
    specs = [*readout.measurement_specs(), *readout.task_specs(), readout.camera_spec()]
    saw_any = False
    for spec in specs:
        p = _camera_param(spec)
        if p is None:
            continue
        saw_any = True
        assert p.kind == "choice", f"{spec.name}: camera param must be a choice, got {p.kind}"
        assert tuple(p.choices) == tuple(cams), f"{spec.name}: choices != live cameras"
        assert p.default in cams, f"{spec.name}: default {p.default!r} not a real camera"
    assert saw_any, "no spec exposed a camera role -- the injection is not wired"


def test_injection_passes_a_device_not_a_string(readout):
    """The role key reaching a build closure is a resolved ``CameraDevice``, never the choice
    string -- so a factory's build receives the device and never indexes ``s.devices[name]``."""
    from Zou_lab_control.neutral_atom.devices.base import CameraDevice
    from Zou_lab_control.neutral_atom.core.signals import SignalHub

    hub = SignalHub()

    # measurement path: make_node -> build(**values), camera name in -> device out on the plan
    ps = next(s for s in readout.measurement_specs() if s.key == "pulse_scan")
    plan = ps.make_node(hub, prefix="pulse_scan_", camera="camera").plan
    assert isinstance(plan.camera, CameraDevice), f"pulse_scan got {type(plan.camera)}, not a device"

    # camera_spec path: build(hub, camera=name) -> the node holds the resolved device
    node = readout.camera_spec().build(hub, camera="camera")
    assert isinstance(node.camera, CameraDevice), f"camera_spec got {type(node.camera)}"

    # task path: spec.build(hub, camera=name) -> CalibrateReadoutTask.camera is the device
    cal = next(s for s in readout.task_specs() if s.name == "Calibrate readout")
    task = cal.build(hub, camera="camera", threshold_frames=2)
    assert isinstance(task.camera, CameraDevice), f"calibrate task got {type(task.camera)}"

    # blank / missing selection falls back to the role default (still a device, never None-crash)
    node2 = readout.camera_spec().build(hub, camera=None)
    assert isinstance(node2.camera, CameraDevice)
