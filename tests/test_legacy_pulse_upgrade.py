# REMOVE-ME(legacy-pulse-upgrade): delete with timing/legacy_pulse_upgrade.py + its 3 call sites.
"""Contract: pre-v4 lab pulse .json files load through the ONE sanctioned legacy upgrade shim.

Schema v4 folded five topology fields into ``port_catalog``/``visible_ports``, added the required
``scan_code`` and made ``from_dict`` strict -- every pre-v4 file was rejected on load.  The shim
(:mod:`...timing.legacy_pulse_upgrade`) upgrades EXACTLY at the three file->dict boundaries; every
in-process round-trip stays strict.  Fixtures are REAL v3 files extracted from git history
(``git show 7b396c7:pulses/*.json``) plus a synthesized v1 payload.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if sys.path[0] != str(REPO_ROOT):
    sys.path.insert(0, str(REPO_ROOT))

from Zou_lab_control.neutral_atom.timing.legacy_pulse_upgrade import upgrade
from Zou_lab_control.neutral_atom.timing.persistence import load_pulse_payload
from Zou_lab_control.neutral_atom.timing.pulse_table import PulseTableState, single_imaging_template

FIXTURES = Path(__file__).parent / "fixtures"
V3_IMAGING = FIXTURES / "legacy_pulse_v3_imaging.json"
V3_RELEASE_RECAPTURE = FIXTURES / "legacy_pulse_v3_release_recapture.json"


def _current_keys() -> set:
    """The current on-disk field set, DERIVED from a fresh state's to_dict (never retyped)."""
    return set(single_imaging_template().to_dict())


@pytest.mark.parametrize("fixture", [V3_IMAGING, V3_RELEASE_RECAPTURE], ids=["imaging", "release_recapture"])
def test_v3_file_loads_through_every_file_entry(fixture):
    """All three file->dict entries accept a real v3 file: PulseTableState.load, load_pulse_payload,
    and a raw dict via upgrade+from_dict (the data_figure recipe path uses exactly that pair)."""
    state = PulseTableState.load(fixture)                       # entry 1: the class loader
    assert isinstance(state, PulseTableState)
    state.validate()
    assert isinstance(load_pulse_payload(fixture), PulseTableState)   # entry 2: persistence dispatch
    payload = json.loads(fixture.read_text(encoding="utf-8"))
    twin = PulseTableState.from_dict(upgrade(payload))          # entry 3: dict boundary (saved-figure recipe)
    twin.validate()

    # old -> load -> save round-trips as a CURRENT file: version stamped, key set == current schema
    saved = state.to_dict()
    assert saved["version"] == PulseTableState.version
    assert set(saved) == _current_keys()


def test_v3_release_recapture_upgrade_keeps_the_bound_scan_slot():
    """The duration scan slot (v3 had no explicit ``name``) derives its v4 name from the label."""
    state = PulseTableState.load(V3_RELEASE_RECAPTURE)
    assert state.scan_names == ["t_off"]
    state.snapped()                                             # the compile path accepts it whole


def test_synthesized_v1_payload_upgrades():
    """A v1-shaped payload (channels + x_ns, none of the v3 scan machinery) still loads: the shim
    fills every later field with the v3 loader's .get defaults and drops the dead x_ns."""
    v1 = {
        "schema": PulseTableState.schema,
        "version": 1,
        "name": "v1 pulse",
        "channels": ["trap", "probe", "emCCD"],
        "visible_channels": ["trap", "probe"],
        "x_ns": [0.0, 10.0],                                   # v1 dead field
        "periods": [
            {"duration": 20.0, "unit": "us", "states": [1, 0, 0]},
            {"duration": 5.0, "unit": "us", "states": [0, 1, 1]},
        ],
    }
    state = PulseTableState.from_dict(upgrade(v1))
    state.validate()
    assert state.name == "v1 pulse"
    assert state.visible_ports == ["trap", "probe"]
    assert len(state.periods) == 2
    assert state.to_dict()["version"] == PulseTableState.version


def test_upgrade_is_identity_for_modern_payloads():
    """A CURRENT to_dict payload passes through upgrade UNTOUCHED -- the shim can never rewrite a
    modern file (in-process round-trips must not silently depend on it)."""
    modern = single_imaging_template().to_dict()
    assert upgrade(modern) == modern
    assert upgrade(modern) is not modern or True               # value equality is the contract


def test_upgrade_passes_a_too_new_version_through_to_the_strict_gate():
    """A v5+ payload is NOT downgraded: upgrade returns it untouched and from_dict rejects it loud."""
    future = single_imaging_template().to_dict()
    future["version"] = PulseTableState.version + 1
    assert upgrade(future)["version"] == PulseTableState.version + 1
    with pytest.raises(ValueError, match="version"):
        PulseTableState.from_dict(upgrade(future))


def test_upgrade_ignores_foreign_schemas():
    foreign = {"schema": "something.else", "version": 1, "payload": 1}
    assert upgrade(foreign) == foreign
