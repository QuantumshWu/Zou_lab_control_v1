"""Device config-schema contracts: a device class DESCRIBES ITS OWN config form.

Pins the three-hook contract (``config_params`` / ``config_to_form`` / ``form_to_config``,
``devices.base``) the device-manager editor renders -- the editor never names a concrete
class, so these contracts are what keep a new device's form usable:

1. ``ParamDecl`` has ONE home (``core.params``, the dependency-free leaf below devices) --
   an import that still pulls it through ``operations.measurement`` is a re-grown parallel
   path and fails here.
2. Every REGISTERED device class's declared form COVERS its constructor's required
   parameters (else the editor could produce a config that cannot construct).
3. Device-graph constructor args (``$device:`` cross-references) are declared kind
   ``device``; enum-ish hardware knobs are typed ``choice`` (pylon); the qCMOS's nested
   config dataclass flattens/regroups losslessly.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if sys.path[0] != str(REPO_ROOT):
    sys.path.insert(0, str(REPO_ROOT))


def test_paramdecl_lives_only_in_core_params():
    from Zou_lab_control.neutral_atom.core.params import ParamDecl

    assert ParamDecl.__module__ == "Zou_lab_control.neutral_atom.core.params"
    # No import statement anywhere still pulls ParamDecl from a NON-core.params module
    # (operations.measurement was the old home; a processor/task re-export would be a
    # second path that drifts).  Source-level: import blocks are the architectural seam.
    offenders: list[str] = []
    import_stmt = re.compile(r"from\s+([.\w]+)\s+import\s+(\([^)]*\)|[^\n]*)", re.S)
    files = list((REPO_ROOT / "Zou_lab_control").rglob("*.py")) + list((REPO_ROOT / "tests").glob("*.py"))
    for path in files:
        text = path.read_text(encoding="utf-8", errors="ignore")
        for match in import_stmt.finditer(text):
            module, names = match.group(1), match.group(2)
            if "ParamDecl" in names and not module.endswith("core.params"):
                offenders.append(f"{path.relative_to(REPO_ROOT)}: from {module} import ... ParamDecl")
    assert not offenders, (
        "ParamDecl must be imported ONLY from neutral_atom.core.params (its single home); "
        f"stale import paths: {offenders}")


def _registered_classes():
    """Every registry class that imports on this machine (a missing driver library must
    skip the class, never fail the contract -- the discovery posture)."""
    from Zou_lab_control.neutral_atom.devices.registry import device_class_registry, resolve_class

    out = []
    for name in device_class_registry():
        try:
            out.append((name, resolve_class(name)))
        except Exception:
            continue
    return out


def test_every_registered_class_covers_its_required_init_params():
    classes = _registered_classes()
    assert classes, "no registry class imported at all -- the scan itself is broken"
    for name, cls in classes:
        decls = cls.config_params()
        keys = [d.key for d in decls]
        assert len(keys) == len(set(keys)), f"{name}: duplicate config_params keys {keys}"
        required = {
            pname for pname, p in inspect.signature(cls.__init__).parameters.items()
            if pname != "self"
            and p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
            and p.default is inspect.Parameter.empty
        }
        # the form's values, mapped back through the class's own regrouping, must be able
        # to supply every required constructor parameter
        mapped = set(cls.form_to_config({k: "x" for k in keys}))
        missing = required - mapped
        assert not missing, (
            f"{name}.config_params() cannot supply required __init__ params {sorted(missing)} "
            f"(declared: {keys}) -- the editor would emit a config that cannot construct.")


def test_device_graph_args_are_declared_device_kind():
    from Zou_lab_control.neutral_atom.devices.virtual import VirtualCamera, VirtualMotCamera

    cam = {d.key: d for d in VirtualCamera.config_params()}
    assert cam["trap_array"].kind == "device" and cam["trap_array"].required
    assert cam["sequencer"].kind == "device" and not cam["sequencer"].required
    mot = {d.key: d for d in VirtualMotCamera.config_params()}
    assert mot["sequencer"].kind == "device"


def test_pylon_enum_knobs_are_typed_choices():
    from Zou_lab_control.neutral_atom.devices.base import SOFTWARE_TRIGGER
    from Zou_lab_control.neutral_atom.devices.pylon import PylonCamera

    decls = {d.key: d for d in PylonCamera.config_params()}
    assert decls["trigger_source"].kind == "choice"
    assert SOFTWARE_TRIGGER in decls["trigger_source"].choices
    assert decls["pixel_format"].kind == "choice" and "Mono8" in decls["pixel_format"].choices
    # the reflected remainder keeps its typed kinds
    assert decls["exposure"].kind == "float"
    assert decls["serial"].kind == "text"
    assert decls["subarray_step"].kind == "int"
    assert decls["capture_trigger_channels"].kind == "json"
    assert isinstance(decls["capture_trigger_channels"].default, list)


def test_qcmos_nested_config_flattens_and_regroups_losslessly():
    import json

    from Zou_lab_control.neutral_atom.devices.qcmos import QCMOSCamera, QCMOSConfig
    from Zou_lab_control.neutral_atom.devices.registry import device_config_dir
    from dataclasses import fields

    decls = {d.key: d for d in QCMOSCamera.config_params()}
    for f in fields(QCMOSConfig):
        assert f.name in decls, f"QCMOSConfig.{f.name} has no form row"
    assert "dcam_module" in decls

    # the bundled remote template's camera entry round-trips through the flatten/regroup
    template = json.loads((device_config_dir() / "remote_template.json").read_text(encoding="utf-8"))
    params = template["camera"]["params"]
    form = QCMOSCamera.config_to_form(params)
    assert form["exposure"] == params["config"]["exposure"]         # flattened for the form
    back = QCMOSCamera.form_to_config(form)
    assert back["config"] == params["config"]                       # regrouped identically
    assert "dcam_module" not in back                                # default never written out


def test_remote_sequencer_prefills_derive_from_their_single_sources():
    from Zou_lab_control.neutral_atom.devices.fpga_pulse_streamer import hardware_channel_names
    from Zou_lab_control.neutral_atom.devices.sequencer import RemoteSequencer, serve_runtime_sequencer

    decls = {d.key: d for d in RemoteSequencer.config_params()}
    assert decls["host"].required and decls["host"].kind == "text"
    serve_port = inspect.signature(serve_runtime_sequencer).parameters["port"].default
    assert decls["port"].default == serve_port                      # the server's own default
    assert decls["channels"].kind == "json"
    assert decls["channels"].default == list(hardware_channel_names())
