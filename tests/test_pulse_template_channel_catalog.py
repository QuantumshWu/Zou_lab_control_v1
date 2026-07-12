"""The pulse-template CHANNEL CATALOG contract (#8: "Show All can't show emCCD"):

The port catalog is a DEVICE property (the real rig's board.xdc topology; the virtual rig's
catalog) and is embedded intact in each saved pulse. Loading a template to fire/edit aligns one
catalog onto the device catalog (``aligned_to_catalog``: physical order, missing raw lanes as off),
so the GUI's "Show All"
really shows every device channel -- with the shipped templates regenerated onto the full virtual
catalog the same way (never hand-typed).  Four mechanical pins:

  1. the virtual catalog and the monitor camera's coil wiring derive from ONE source
     (``MOT_COIL_BUSES``) -- they used to be two hand-typed mirror copies;
  2. every shipped pulse-table template carries the FULL virtual catalog while its saved
     default view (``visible_ports``) stays the original driving subset;
  3. ``resolve_fireable_template(..., sequencer=...)`` expands a SUBSET template onto the
     device catalog with an IDENTICAL compiled product, and leaves a non-subset template
     untouched (the prepare layer / coupled resolver own that case);
  4. the GUI ``load_from_file`` walks the same rule: subset -> expanded (Show All then lists
     the whole catalog), non-subset -> an error message, never a crash or a silent swap of
     the editor's channel universe.

Plus the align-preservation pin (5): ``aligned_to_catalog`` carries ``api_slots`` and
``scan_repeats``.
"""

from __future__ import annotations
from Zou_lab_control.neutral_atom.ports import PortCatalog

import json
from pathlib import Path
import sys

import pytest
from conftest import pulse_editor_for_test

REPO_ROOT = Path(__file__).resolve().parents[1]
if sys.path[0] != str(REPO_ROOT):
    sys.path.insert(0, str(REPO_ROOT))

from Zou_lab_control.neutral_atom.devices.virtual import (
    DEFAULT_CHANNELS, MOT_COIL_BUSES, VirtualMotCamera, VirtualSequencer)
from Zou_lab_control.neutral_atom.timing import (
    PulseTableState, resolve_fireable_template, single_imaging_template)

CATALOG = list(DEFAULT_CHANNELS)
TEMPLATE_FILES = (
    "mot_field_template.json",
    "imaging_template.json",
    "probe_template.json",
    "release_recapture.json",
)


def _pulse_key(state: PulseTableState, *, channels=None):
    """Comparable compiled pulses, optionally restricted to logical source lanes.

    Added offset-binary DAC lanes sit at their catalog ``safe_value`` (signed zero),
    which can legitimately hold the most-significant raw bit high.  They are not an
    extra logical pulse and therefore must not be compared with a digital-only subset.
    """

    selected = None if channels is None else set(channels)
    return sorted(
        (p.channel, float(p.start), float(p.duration), getattr(p, "value", None))
        for p in state.to_sequence().base_pulses()
        if selected is None or p.channel in selected
    )


def _assert_added_dacs_are_safe(state: PulseTableState, source_lanes) -> None:
    source_lanes = set(source_lanes)
    lane_index = {lane: index for index, lane in enumerate(state.port_catalog.raw_lanes)}
    for port in state.port_catalog.dac_ports:
        if set(port.lanes) <= source_lanes:
            continue
        for period in state.periods:
            code = sum(
                int(period.states[lane_index[lane]]) << bit
                for bit, lane in enumerate(port.lanes)
            )
            assert code == port.safe_value


# --------------------------------------------------- (1) catalog and coil wiring share one source
def test_virtual_catalog_and_mot_camera_coil_buses_share_one_source():
    cam = VirtualMotCamera()
    assert cam.coil_buses == {k: tuple(v) for k, v in MOT_COIL_BUSES.items()}, \
        "VirtualMotCamera's default coil wiring must BE MOT_COIL_BUSES (no second hand-typed copy)"
    members = [c for bus in MOT_COIL_BUSES.values() for c in bus]
    assert len(set(members)) == len(members)
    for channel in members:
        assert CATALOG.count(channel) == 1, f"coil bit {channel!r} must appear once in the catalog"
    assert "mot_trigger" in CATALOG, "the monitor camera's hardware trigger line is a device channel"
    assert len(CATALOG) == len(set(CATALOG))


# ------------------------------------------- (2) shipped templates carry the full virtual catalog
@pytest.mark.parametrize("name", TEMPLATE_FILES)
def test_shipped_template_channels_are_the_full_virtual_catalog(name):
    state = PulseTableState.load(REPO_ROOT / "pulses" / name)
    assert list(state.port_catalog.raw_lanes) == CATALOG, f"{name}: raw lanes must match the device catalog"
    # the saved default VIEW stays the original driving subset -- load looks unchanged,
    # only Show All spreads to the whole catalog
    programmable = [
        port.key for port in state.port_catalog.ports if port.kind != "clock"
    ]
    assert set(state.visible_ports) <= set(programmable)
    assert state.visible_ports == [key for key in programmable if key in set(state.visible_ports)]
    # the expansion added OFF rows only: the template does not drive the whole catalog
    assert len(state.period_active_channels()) < len(state.port_catalog.raw_lanes)


def test_regenerated_mot_template_kept_its_semantic_dac_scan_slots_and_buses():
    """The MOT task streams one named hardware-scan slot per coil bus."""
    mot = PulseTableState.load(REPO_ROOT / "pulses" / "mot_field_template.json")
    assert mot.scan_names == ["da_x", "da_y", "da_z"]
    assert mot.api_slots == []
    assert mot.port_catalog.analog_buses == {k: list(v) for k, v in MOT_COIL_BUSES.items()}


# --------------------------------- (3) resolve_fireable_template expands a subset, equivalently
def test_resolve_fireable_template_expands_a_subset_template_to_the_device_catalog(tmp_path):
    subset = single_imaging_template()                       # trap/cooling/probe/emCCD -- a subset
    assert set(subset.port_catalog.raw_lanes) < set(CATALOG)
    path = tmp_path / "subset_probe.json"
    subset.save(path)

    seqr = VirtualSequencer()
    state = resolve_fireable_template(str(path), default_name=str(path),
                                      default_factory=single_imaging_template, sequencer=seqr)
    assert list(state.port_catalog.raw_lanes) == list(seqr.port_catalog.raw_lanes) == CATALOG
    assert _pulse_key(state, channels=subset.port_catalog.raw_lanes) == _pulse_key(subset)
    _assert_added_dacs_are_safe(state, subset.port_catalog.raw_lanes)
    assert state.api_names() == subset.api_names(), "api handles survive the load-time align"


def test_resolve_fireable_template_rejects_a_foreign_catalog(tmp_path):
    foreign = PulseTableState(
        port_catalog=PortCatalog.from_channels(["trap", "alien_line"]),
        visible_ports=["alien_line"],
    )
    path = tmp_path / "foreign.json"
    foreign.save(path)
    with pytest.raises(ValueError, match="has no matching hardware port"):
        resolve_fireable_template(str(path), default_name=str(path),
                                  default_factory=single_imaging_template,
                                  sequencer=VirtualSequencer())


def test_resolve_fireable_template_without_a_sequencer_keeps_the_saved_channels(tmp_path):
    subset = single_imaging_template()
    path = tmp_path / "subset_probe.json"
    subset.save(path)
    state = resolve_fireable_template(str(path), default_name=str(path),
                                      default_factory=single_imaging_template)
    assert state.port_catalog == subset.port_catalog


# ----------------------------------------------- (5) aligned_to_catalog carries api_slots et al.
def test_aligned_to_catalog_preserves_api_slots_and_scan_repeats():
    st = PulseTableState(port_catalog=PortCatalog.from_channels(["a", "b"]))
    st.bind_api_field("duration", "0", name="a1", unit="us")
    st.bind_api_field("delay", "b", name="a2", unit="us")
    st.scan_repeats = 4
    out = st.aligned_to_catalog(PortCatalog.from_channels(["a", "b", "c"]))
    assert [(s.name, s.kind, s.target) for s in out.api_slots] == \
        [("a1", "duration", "0"), ("a2", "delay", "b")]
    assert int(out.scan_repeats) == 4


# --------------------------------------------------------- (4) the GUI load walks the same rule
@pytest.fixture
def _app(monkeypatch):
    pytest.importorskip("PyQt5")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PyQt5 import QtWidgets
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _catalog_editor():
    return pulse_editor_for_test(
        sequencer=VirtualSequencer(sleep_scale=0)
    )


def _point_load_dialog_at(monkeypatch, path):
    from PyQt5 import QtWidgets
    monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (str(path), "ZLC pulse (*.json)")))


def test_gui_load_expands_a_subset_file_and_show_all_lists_the_whole_catalog(_app, monkeypatch, tmp_path):
    subset = single_imaging_template()                       # 4 channels, all in the catalog
    path = tmp_path / "subset_probe.json"
    subset.save(path)

    ed = _catalog_editor()
    _point_load_dialog_at(monkeypatch, path)
    ed.load_from_file()

    assert list(ed.state.port_catalog.raw_lanes) == CATALOG
    assert list(ed.state.visible_ports) == list(subset.visible_ports), \
        "the default view stays the file's own driving rows"
    assert _pulse_key(
        ed.read_state(), channels=subset.port_catalog.raw_lanes
    ) == _pulse_key(subset)
    _assert_added_dacs_are_safe(ed.read_state(), subset.port_catalog.raw_lanes)

    ed.show_all_ports()
    shown = ed.read_state().visible_ports
    expected = [
        port.key for port in ed.state.port_catalog.ports if port.kind != "clock"
    ]
    assert list(shown) == expected


def test_gui_load_rejects_a_non_subset_file_without_crashing(_app, monkeypatch, tmp_path):
    foreign = PulseTableState(
        port_catalog=PortCatalog.from_channels(["trap", "alien_line"]),
        visible_ports=["alien_line"],
    )
    path = tmp_path / "foreign.json"
    foreign.save(path)

    ed = _catalog_editor()
    before = ed.state.port_catalog
    _point_load_dialog_at(monkeypatch, path)
    ed.load_from_file()                                      # must not raise

    assert ed.state.port_catalog == before
    assert "has no matching hardware port" in ed.summary.text()
