"""The Scan-tab code editor's SOURCE text round-trips through Save/Load.

User report: "you saved the scan program, but loading doesn't bring the program back (or did save
never actually save the code in the editor box?)".  Root cause: the editor code (``scan_code``) --
the Python that GENERATES ``scan_table`` -- was never persisted; only the derived ``scan_table`` was.
So a Load showed the DEFAULT template, not the user's program, and a subsequent Run would regenerate
from that default and clobber the loaded custom scan.

Fix: ``scan_code`` is a persisted ``PulseTableState`` field (source next to the frozen result), read
from the editor on ``read_state`` and restored into it on ``load_state``.  These guard both seams:
the data-model round-trip (headless) and the pulse_gui editor round-trip (offscreen Qt).
"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if sys.path[0] != str(REPO_ROOT):
    sys.path.insert(0, str(REPO_ROOT))

from Zou_lab_control.neutral_atom.timing.pulse_table import PulseTableState

_CUSTOM = "# my custom scan program\nimport numpy as np\nscan_table = np.column_stack([np.linspace(20, 200, 7)])"


# --------------------------------------------------------------- data model (no Qt)
def test_scan_code_round_trips_through_to_dict_from_dict_and_save_load(tmp_path):
    s = PulseTableState(channels=["ch00", "ch01"], scan_code=_CUSTOM)
    assert s.to_dict()["scan_code"] == _CUSTOM
    assert PulseTableState.from_dict(s.to_dict()).scan_code == _CUSTOM
    p = tmp_path / "pulse.json"
    s.save(p)
    assert PulseTableState.load(p).scan_code == _CUSTOM          # persisted in the .json bundle


def test_scan_code_survives_aligned_to_channels_and_snapped():
    """A subset-channel file load goes through ``aligned_to_channels``; ``snapped`` (compile/display
    snap) goes through ``from_dict(to_dict())`` -- both must keep the editor code."""
    s = PulseTableState(channels=["ch00", "ch01"], scan_code=_CUSTOM)
    assert s.aligned_to_channels(["ch00", "ch01", "ch02", "ch03"]).scan_code == _CUSTOM
    assert s.snapped().scan_code == _CUSTOM


def test_old_pulse_without_scan_code_loads_with_empty_default():
    payload = PulseTableState(channels=["ch00", "ch01"]).to_dict()
    payload.pop("scan_code")                                     # an old save predating the field
    assert PulseTableState.from_dict(payload).scan_code == ""    # default "", never a crash


# --------------------------------------------------------------- pulse_gui editor (offscreen Qt)
@pytest.fixture
def _app(monkeypatch):
    pytest.importorskip("PyQt5")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PyQt5 import QtWidgets
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _pulse_editor():
    from Zou_lab_control.frontend.pulse_gui import PulseSequenceEditor
    st = PulseTableState(channels=["probe", "trig"])
    st.bind_field("duration", "0", unit="us")
    st.set_scan_table([[10.0], [20.0], [30.0]])
    return PulseSequenceEditor(st)


def test_read_state_captures_the_editor_code(_app):
    ed = _pulse_editor()
    ed.scan_code.setPlainText(_CUSTOM)
    assert ed.read_state().scan_code == _CUSTOM                  # the editor text reaches the state


def test_editor_code_round_trips_save_then_load_into_a_fresh_editor(_app, tmp_path):
    """The whole user story: type a program, Save, then Load into a new editor -- the SAME code comes
    back in the box, and Run is NOT dirty (code + table came from the same saved bundle)."""
    ed = _pulse_editor()
    ed.scan_code.setPlainText(_CUSTOM)
    state = ed.read_state()
    p = tmp_path / "pulse.json"
    state.save(p)

    ed2 = _pulse_editor()
    ed2.load_state(PulseTableState.load(p))
    assert ed2.scan_code.toPlainText() == _CUSTOM               # the program is restored verbatim
    assert ed2.scan_run_button.is_dirty() is False             # not stale -> no spurious '*'


def test_in_session_load_state_does_not_wipe_edited_code(_app):
    """An in-session mutation (add/remove period, clk toggle) round-trips read_state -> load_state; the
    editor carries its own text, so state.scan_code already equals it and the restore is a no-op -- the
    user's in-progress edit must NOT be reset."""
    ed = _pulse_editor()
    ed.scan_code.setPlainText(_CUSTOM)
    ed.load_state(ed.read_state())                              # the add/remove-period cycle
    assert ed.scan_code.toPlainText() == _CUSTOM


def test_loading_a_codeless_pulse_clears_stale_editor_code(_app):
    """Loading a pulse with no saved code (a brand-new / notebook-built one) must NOT leave the previous
    pulse's program in the box -- else the editor lies about what generated the loaded table."""
    ed = _pulse_editor()
    ed.scan_code.setPlainText("scan_table = [[999.0]]")        # stale from a previous pulse
    codeless = PulseTableState(channels=["probe", "trig"])     # scan_code == ""
    codeless.bind_field("duration", "0", unit="us")
    codeless.set_scan_table([[10.0]])
    ed.load_state(codeless)
    assert "999.0" not in ed.scan_code.toPlainText()          # the stale program is gone
