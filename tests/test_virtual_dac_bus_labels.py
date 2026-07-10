"""Contract: the virtual sequencer publishes one PortCatalog whose DAC ports are
``da_x``/``da_y``/``da_z`` -- semantic controls projected to the real bias buses.

A device names its coil bits ``dx0..dz5`` (real hardware: ``chNN``) and LABELS each in
``base[bit]`` syntax (``dx0 -> "da_x[0]"``).
The device-boundary catalog builder folds labels once, with NO template loaded, so WHEN the coil
bits are shown (they are not in a fresh editor's first-four default
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
    assert MOT_COIL_LABELS["dx0"] == "da_x[0]"
    assert MOT_COIL_LABELS["dz5"] == "da_z[5]"


def test_virtual_sequencer_advertises_coil_labels():
    seq = VirtualSequencer()
    assert seq.port_catalog.channel_labels["dx0"] == "da_x[0]"
    assert not hasattr(seq, "channel_labels")
    assert [port.key for port in seq.port_catalog.dac_ports] == ["da_x", "da_y", "da_z"]


def test_state_from_virtual_sequencer_folds_the_three_buses():
    """A PulseTableState built from a stock virtual sequencer's channels + labels folds the 18
    coil bits into exactly the three coil buses (members == the single-source table) -- no template
    needed."""
    seq = VirtualSequencer()
    state = PulseTableState(port_catalog=seq.port_catalog)
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
    programmable = [
        port.key for port in seq.port_catalog.ports if port.kind != "clock"
    ]
    coil_bits = {ch for members in MOT_COIL_BUSES.values() for ch in members}

    # Fresh editor default: only the first four logical ports are visible.
    fresh = PulseTableState(port_catalog=seq.port_catalog,
                            visible_ports=programmable[:4])
    assert not [r for r in _display_rows(fresh) if r["kind"] == "bus"]

    # Show-all: each DAC is one visible port, never its member lanes.
    shown = PulseTableState(port_catalog=seq.port_catalog, visible_ports=programmable)
    rows = _display_rows(shown)
    bus_rows = {str(r["name"]): tuple(r["channels"]) for r in rows if r["kind"] == "bus"}
    assert bus_rows == {bus: tuple(members) for bus, members in MOT_COIL_BUSES.items()}
    drawn_channels = {str(r["name"]) for r in rows if r["kind"] == "channel"}
    assert not (drawn_channels & coil_bits), "a folded coil bit must never also draw as a single row"


def test_custom_channel_list_drops_unused_coil_labels():
    """A sequencer built with a channel list that omits the coil bits advertises no coil labels
    (never a bus whose members are absent)."""
    seq = VirtualSequencer(channels=("trap", "cooling", "probe", "emCCD"))
    assert seq.port_catalog.dac_ports == ()
    assert set(seq.port_catalog.channel_labels) == {"trap", "cooling", "probe", "emCCD"}


def test_port_catalog_owns_the_three_dac_ports():
    seq = VirtualSequencer()
    assert seq.port_catalog.analog_buses == {
        bus: list(members) for bus, members in MOT_COIL_BUSES.items()}
    assert all(seq.port_catalog.port_for_lane(lane).kind == "dac"
               for members in MOT_COIL_BUSES.values() for lane in members)


def test_sequencer_snapshot_separates_ports_and_raw_lanes():
    seq = VirtualSequencer()
    snap = seq.snapshot()
    assert "channels" not in snap
    assert snap["raw_channels"] == list(seq.port_catalog.raw_lanes)
    assert snap["port_catalog_fingerprint"] == seq.port_catalog.fingerprint
    readback = {c.decl.key: c.getter(seq) for c in seq.runtime_controls()}["ports"]
    assert "da_x" in readback and "dx0" not in readback
