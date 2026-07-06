"""Contract: the virtual sequencer carries display LABELS so its DAC coil bits fold into
``da_x``/``da_y``/``da_z`` bus rows -- the same way the real rig folds them.

A device names its coil bits ``dx0..dz5`` (real hardware: ``chNN``) and LABELS each in
``base[bit]`` syntax (``dx0 -> "da_x[0]"``; real: ``da_dipole[0]`` off the board XDC).
``infer_bus_channels`` folds on the label regex, so a fresh pulse editor started from a stock
``VirtualSequencer`` -- no template loaded -- already shows three bus rows, not 18 single
channel rows.  The labels are DERIVED from ``MOT_COIL_BUSES`` (one source), so name<->label
never drift.
"""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if sys.path[0] != str(REPO_ROOT):
    sys.path.insert(0, str(REPO_ROOT))

from Zou_lab_control.neutral_atom.devices.virtual import (
    MOT_COIL_BUSES,
    MOT_COIL_LABELS,
    VirtualSequencer,
)
from Zou_lab_control.neutral_atom.timing.pulse_table import PulseTableState


def test_coil_labels_derive_from_the_one_bus_table():
    """Every coil bit is labelled ``base[bit]`` and the table is DERIVED from MOT_COIL_BUSES
    (no second hand-typed copy that could drift from the channel names)."""
    expected = {ch: f"{bus}[{bit}]"
                for bus, members in MOT_COIL_BUSES.items()
                for bit, ch in enumerate(members)}
    assert MOT_COIL_LABELS == expected
    assert MOT_COIL_LABELS["dx0"] == "da_x[0]" and MOT_COIL_LABELS["dz5"] == "da_z[5]"


def test_virtual_sequencer_advertises_coil_labels():
    seq = VirtualSequencer()
    # only the coil bits carry labels (the plain TTL lines stay unlabelled)
    assert seq.channel_labels["dx0"] == "da_x[0]"
    assert "trap" not in seq.channel_labels
    assert set(seq.channel_labels) == {ch for members in MOT_COIL_BUSES.values() for ch in members}


def test_state_from_virtual_sequencer_folds_the_three_buses():
    """A PulseTableState built from a stock virtual sequencer's channels + labels folds the 18
    coil bits into exactly the three coil buses (members == the single-source table), so the GUI
    draws three bus rows -- no template needed."""
    seq = VirtualSequencer()
    state = PulseTableState(channels=list(seq.channels), channel_labels=dict(seq.channel_labels))
    buses = state.bus_channels()
    for bus, members in MOT_COIL_BUSES.items():
        assert bus in buses, f"{bus} should fold from labels"
        assert tuple(buses[bus]) == tuple(members)


def test_custom_channel_list_drops_unused_coil_labels():
    """A sequencer built with a channel list that omits the coil bits advertises no coil labels
    (never a bus whose members are absent)."""
    seq = VirtualSequencer(channels=("trap", "cooling", "probe", "emCCD"))
    assert seq.channel_labels == {}
