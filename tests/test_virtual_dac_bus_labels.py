"""Contract: the virtual sequencer carries display LABELS so its DAC coil bits fold into
``da_x``/``da_y``/``da_z`` bus rows -- the same way the real rig folds them.

A device names its coil bits ``dx0..dz5`` (real hardware: ``chNN``) and LABELS each in
``base[bit]`` syntax (``dx0 -> "da_x[0]"``; real: ``da_dipole[0]`` off the board XDC).
``infer_bus_channels`` folds on the label regex -- from the DEVICE's own labels, with NO template
loaded -- so WHEN the coil bits are shown (they are not in a fresh editor's first-four default
visible set) the editor draws three bus rows, not 18 single channel rows.  The labels are DERIVED
from ``MOT_COIL_BUSES`` (one source), so name<->label never drift.

The folding is checked on the REAL display path (``pulse_gui._display_rows``, the function the
editor renders from), not just the ``bus_channels()`` data -- so a claim about what the editor draws
is proven against what it actually draws.
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
    coil bits into exactly the three coil buses (members == the single-source table) -- no template
    needed."""
    seq = VirtualSequencer()
    state = PulseTableState(channels=list(seq.channels), channel_labels=dict(seq.channel_labels))
    buses = state.bus_channels()
    for bus, members in MOT_COIL_BUSES.items():
        assert bus in buses, f"{bus} should fold from labels"
        assert tuple(buses[bus]) == tuple(members)


def test_display_rows_fold_the_coils_only_when_shown():
    """What the editor DRAWS (``pulse_gui._display_rows``, the real render source), not just the
    ``bus_channels()`` data: a FRESH editor's default visible set is the first four channels (no coil
    bits), so it draws ZERO bus rows; once the coils are shown they fold into EXACTLY the three
    ``da_*`` bus rows (and none of the 18 single coil rows) -- the device labels alone drive the fold,
    with no template loaded."""
    from Zou_lab_control.frontend.pulse_gui import _display_rows

    seq = VirtualSequencer()
    channels = list(seq.channels)
    labels = dict(seq.channel_labels)
    coil_bits = {ch for members in MOT_COIL_BUSES.values() for ch in members}

    # Fresh editor default: only the first four channels are visible (pulse_gui seeds
    # visible_channels=channels[:4]); none are coil bits, so NO bus row is drawn.
    fresh = PulseTableState(channels=channels, channel_labels=labels,
                            visible_channels=channels[: min(4, len(channels))])
    assert not [r for r in _display_rows(fresh) if r["kind"] == "bus"]

    # Show-all: every coil bit visible -> exactly the three da_* bus rows, zero single coil rows.
    shown = PulseTableState(channels=channels, channel_labels=labels, visible_channels=channels)
    rows = _display_rows(shown)
    bus_rows = {str(r["name"]): tuple(r["channels"]) for r in rows if r["kind"] == "bus"}
    assert bus_rows == {bus: tuple(members) for bus, members in MOT_COIL_BUSES.items()}
    drawn_channels = {str(r["name"]) for r in rows if r["kind"] == "channel"}
    assert not (drawn_channels & coil_bits), "a folded coil bit must never also draw as a single row"


def test_custom_channel_list_drops_unused_coil_labels():
    """A sequencer built with a channel list that omits the coil bits advertises no coil labels
    (never a bus whose members are absent)."""
    seq = VirtualSequencer(channels=("trap", "cooling", "probe", "emCCD"))
    assert seq.channel_labels == {}


def test_channels_as_buses_folds_the_dac_coils():
    """``channels_as_buses`` (the ONE fold a device snapshot / viewer uses) collapses the 18 coil bit
    lines into the three ``da_*`` buses and keeps every plain channel -- so the device presents its 3
    physical DAC buses, not 18 separate lines."""
    from Zou_lab_control.neutral_atom.timing import channels_as_buses
    seq = VirtualSequencer()
    folded = channels_as_buses(seq.channels, seq.channel_labels)
    assert {"da_x", "da_y", "da_z"} <= set(folded)
    coils = {ch for members in MOT_COIL_BUSES.values() for ch in members}
    assert not (set(folded) & coils), "a folded bus must not also list its raw bit lines"
    for ch in seq.channels:                      # every non-coil channel survives, in order
        if ch not in coils:
            assert ch in folded


def test_sequencer_snapshot_and_readback_report_folded_channels():
    """The sequencer's OBSERVE surface (snapshot + the ``channels`` runtime-control read-back) presents
    the DAC as buses, NOT 18 separately-counted bit lines -- the "counted separately" report.  Both read
    the ONE ``display_channels`` source, and the raw catalog stays on ``self.channels`` for the compiler."""
    seq = VirtualSequencer()
    snap = seq.snapshot()
    assert "da_x" in snap["channels"] and "dx0" not in snap["channels"]
    assert seq.display_channels() == snap["channels"]            # ONE source
    assert list(seq.channels) != snap["channels"]                # raw 25 kept for the compiler, folded for display
    readback = {c.decl.key: c.getter(seq) for c in seq.runtime_controls()}["channels"]
    assert "da_x" in readback and "dx0" not in readback
