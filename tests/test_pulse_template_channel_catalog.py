"""The pulse-template CHANNEL CATALOG contract (#8: "Show All can't show emCCD"):

The channel catalog is a DEVICE property (the real rig's board.xdc pin list; the virtual rig's
DEFAULT_CHANNELS) -- a saved pulse template is a SUBSET of it, never its definition.  Loading a
template to fire/edit therefore ALIGNS it onto the device catalog (``aligned_to_channels``:
catalog order, missing channels as off rows, compiled program identical), so the GUI's "Show All"
really shows every device channel -- with the shipped templates regenerated onto the full virtual
catalog the same way (never hand-typed).  Four mechanical pins:

  1. the virtual catalog and the monitor camera's coil wiring derive from ONE source
     (``MOT_COIL_BUSES``) -- they used to be two hand-typed mirror copies;
  2. every shipped pulse-table template carries the FULL virtual catalog while its saved
     default view (``visible_channels``) stays the original driving subset;
  3. ``resolve_fireable_template(..., sequencer=...)`` expands a SUBSET template onto the
     device catalog with an IDENTICAL compiled product, and leaves a non-subset template
     untouched (the prepare layer / coupled resolver own that case);
  4. the GUI ``load_from_file`` walks the same rule: subset -> expanded (Show All then lists
     the whole catalog), non-subset -> an error message, never a crash or a silent swap of
     the editor's channel universe.

Plus the align-preservation pin (5): ``aligned_to_channels`` carries ``api_slots`` and
``scan_repeats`` (the same built-but-not-passed bug class as the clk_channels regression --
dropping them would strip the MOT template's a1/a2/a3 DAC handles on every load).
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

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


def _pulse_key(state: PulseTableState):
    """The compiled product as comparable data: off rows added by an align contribute NO pulses,
    so two states that compile to the same key fire the identical program."""
    return sorted((p.channel, float(p.start), float(p.duration), getattr(p, "value", None))
                  for p in state.to_sequence().base_pulses())


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
    assert list(state.channels) == CATALOG, f"{name}: channels must be the device catalog"
    # the saved default VIEW stays the original driving subset -- load looks unchanged,
    # only Show All spreads to the whole catalog
    assert set(state.visible_channels) < set(state.channels), f"{name}: visible must stay a proper subset"
    assert state.visible_channels == [c for c in state.channels if c in set(state.visible_channels)]
    # the expansion added OFF rows only: the template does not drive the whole catalog
    assert len(state.period_active_channels()) < len(state.channels)


def test_regenerated_mot_template_kept_its_dac_api_slots_and_buses():
    """The regen must not strip what the file carried: the a1/a2/a3 DAC handles the MOT task sets
    and the three coil buses (which must equal the ONE source the virtual camera wires)."""
    mot = PulseTableState.load(REPO_ROOT / "pulses" / "mot_field_template.json")
    assert {s.name for s in mot.api_slots if s.kind == "dac"} == {"a1", "a2", "a3"}
    assert dict(mot.analog_buses) == {k: list(v) for k, v in MOT_COIL_BUSES.items()}


# --------------------------------- (3) resolve_fireable_template expands a subset, equivalently
def test_resolve_fireable_template_expands_a_subset_template_to_the_device_catalog(tmp_path):
    subset = single_imaging_template()                       # trap/cooling/probe/emCCD -- a subset
    assert set(subset.channels) < set(CATALOG)
    path = tmp_path / "subset_probe.json"
    subset.save(path)

    seqr = VirtualSequencer()
    state = resolve_fireable_template(str(path), default_name=str(path),
                                      default_factory=single_imaging_template, sequencer=seqr)
    assert list(state.channels) == list(seqr.channels) == CATALOG
    assert _pulse_key(state) == _pulse_key(subset), "the expansion must not change the compiled program"
    assert state.api_names() == subset.api_names(), "api handles survive the load-time align"


def test_resolve_fireable_template_leaves_a_non_subset_template_untouched(tmp_path):
    foreign = PulseTableState(channels=["trap", "alien_line"])
    path = tmp_path / "foreign.json"
    foreign.save(path)
    state = resolve_fireable_template(str(path), default_name=str(path),
                                      default_factory=single_imaging_template,
                                      sequencer=VirtualSequencer())
    assert list(state.channels) == ["trap", "alien_line"], \
        "a non-subset template is NOT remapped here (prepare / the coupled resolver own that)"


def test_resolve_fireable_template_without_a_sequencer_keeps_the_saved_channels(tmp_path):
    subset = single_imaging_template()
    path = tmp_path / "subset_probe.json"
    subset.save(path)
    state = resolve_fireable_template(str(path), default_name=str(path),
                                      default_factory=single_imaging_template)
    assert list(state.channels) == list(subset.channels)     # no device in scope -> no catalog


# ----------------------------------------------- (5) aligned_to_channels carries api_slots et al.
def test_aligned_to_channels_preserves_api_slots_and_scan_repeats():
    st = PulseTableState(channels=["a", "b"])
    st.bind_api_field("duration", "0", name="a1", unit="us")
    st.bind_api_field("delay", "b", name="a2", unit="us")
    st.scan_repeats = 4
    out = st.aligned_to_channels(["a", "b", "c"])
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
    from Zou_lab_control.frontend.pulse_gui import PulseSequenceEditor
    return PulseSequenceEditor(channels=CATALOG)


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

    assert list(ed.state.channels) == CATALOG, "load must expand a subset file onto the catalog"
    assert list(ed.state.visible_channels) == list(subset.visible_channels), \
        "the default view stays the file's own driving rows"
    assert _pulse_key(ed.read_state()) == _pulse_key(subset)

    ed.show_all_channels()
    shown = ed.read_state().visible_channels
    assert list(shown) == CATALOG, "Show All must list every device channel (emCCD, trap, ...)"


def test_gui_load_rejects_a_non_subset_file_without_crashing(_app, monkeypatch, tmp_path):
    foreign = PulseTableState(channels=["trap", "alien_line"])
    path = tmp_path / "foreign.json"
    foreign.save(path)

    ed = _catalog_editor()
    before = list(ed.state.channels)
    _point_load_dialog_at(monkeypatch, path)
    ed.load_from_file()                                      # must not raise

    assert list(ed.state.channels) == before, "a rejected load must not swap the channel universe"
    assert "do not match the sequencer channels" in ed.summary.text()
