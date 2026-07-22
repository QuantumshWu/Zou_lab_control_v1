"""The board is where the pulse topology comes from -- everything else carries a copy.

Three artefacts describe the same lanes: the platform's XDC constraints, the
``PulseTarget`` inside every shipped pulse document, and the authoring
``PortCatalog`` the editor shows.  Nothing forced them to agree, and two of the
three are only read at runtime, so a drift would surface as a hardware fault
(a reversed DAC bus, a lane addressed by the wrong name) rather than as a failure
here.  These tests make the board the origin and the other two its projections.

The DAC bus ordering is the sharp edge: ``da_bias_y`` is pinned MSB-first in the
XDC, so a target built in file order would drive that bus backwards while looking
completely reasonable in the GUI.
"""

from __future__ import annotations

from pathlib import Path
import json

from zlc_neutral_atom.timing.board_config import load_board_config
from zlc_neutral_atom.timing.ports import PORT_DAC
from zlc_pulse.target import load_deployed_pulse_target, pulse_target_to_tree

ROOT = Path(__file__).resolve().parents[1]
#: Documents shipped WITH the board config; a user's own saved pulse may legitimately
#: come from another board -- that mismatch is the geometry fingerprint's job to reject
#: at connect time, not this test's.
SHIPPED_DOCUMENTS = sorted((ROOT / "zlc_neutral_atom" / "assets").glob("*.json"))


def test_the_deployed_target_is_the_board_projection() -> None:
    """The server's default target must not drift from the XDC lane topology."""

    assert pulse_target_to_tree(load_deployed_pulse_target()) == pulse_target_to_tree(
        load_board_config().pulse_target()
    )


def test_the_board_builds_the_target_shipped_documents_carry() -> None:
    assert SHIPPED_DOCUMENTS, "no shipped pulse documents found to check against the board"
    built = pulse_target_to_tree(load_board_config().pulse_target())

    for path in SHIPPED_DOCUMENTS:
        saved = json.loads(path.read_text(encoding="utf-8")).get("target")
        if saved is None:
            continue
        assert saved["raw_lanes"] == built["raw_lanes"], f"{path.name}: lane order differs"
        assert saved["ports"] == built["ports"], f"{path.name}: port specs differ"
        # The fingerprint folds every field above; equal fingerprints mean a host packing
        # for this board will not be refused by a bitstream built from the same config.
        assert saved["abi_fingerprint"] == built["abi_fingerprint"], f"{path.name}: ABI differs"


def test_the_authoring_catalog_is_a_projection_of_the_target() -> None:
    board = load_board_config()
    catalog = board.port_catalog()
    target = board.pulse_target()

    assert catalog.raw_lanes == target.raw_lanes
    assert [(p.key, p.kind, p.lanes, p.label) for p in catalog.ports] == \
           [(p.key, p.kind, p.lanes, p.label) for p in target.ports], (
        "the authoring catalog must be the target's topology with the hardware fields "
        "dropped -- not a second, independently inferred, port list")


def test_a_dac_bus_is_ordered_by_bit_index_not_by_pin_order() -> None:
    board = load_board_config()
    target = board.pulse_target()
    buses = [port for port in target.ports if port.kind == PORT_DAC]
    assert buses, "the board defines no DAC bus"

    for bus in buses:
        # Reconstruct each lane's declared bit index from the board labels and require
        # the port to list them 0..n-1.  Pinning order is NOT this order: at least one
        # bus on this board is constrained MSB-first.
        indices = [int(board.labels[lane].split("[", 1)[1].rstrip("]")) for lane in bus.lanes]
        assert indices == list(range(len(bus.lanes))), (
            f"{bus.key} lanes are ordered {indices}, expected LSB-first 0..{len(bus.lanes) - 1}")
        assert bus.safe_value == 1 << (bus.width - 1), (
            f"{bus.key} idles at {bus.safe_value}, not the mid code (0 V)")
