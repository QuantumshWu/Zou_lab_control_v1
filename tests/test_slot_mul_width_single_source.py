"""``SLOT_MUL_WIDTH`` (the affine-MAC slot operand width) is ONE value across every layer.

It appears, by necessity, as a synthesis-time ``localparam`` baked into the bitstream, as a host
value that REJECTS scan points that would not fit that width, and in the Python engine model that
mirrors the RTL narrowing.  If any copy drifts from the RTL, the host accepts (or rejects) scan
values the FPGA cannot (or can) actually play -- silently, on real hardware.  There is no runtime
handshake for a localparam, so this contract is the guard: the RTL ``localparam`` must equal the
board config, the engine model, and the host default (same pattern as
``test_fpga_pulse_streamer_repo_vivado_entrypoint_contract`` grepping the RTL).
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if sys.path[0] != str(REPO_ROOT):
    sys.path.insert(0, str(REPO_ROOT))


def _rtl_slot_mul_width() -> int:
    rtl = (REPO_ROOT / "fpga" / "pulse_streamer" / "zlc_edge_streamer.v").read_text(encoding="utf-8")
    m = re.search(r"localparam\s+integer\s+SLOT_MUL_WIDTH\s*=\s*(\d+)\s*;", rtl)
    assert m, "zlc_edge_streamer.v must declare `localparam integer SLOT_MUL_WIDTH = <N>;`"
    return int(m.group(1))


def test_slot_mul_width_matches_across_rtl_config_model_and_host():
    from fpga.pulse_streamer.host.engine_model import SLOT_MUL_WIDTH as MODEL
    from fpga.pulse_streamer.host.image import DEFAULT_SLOT_MUL_WIDTH as HOST_DEFAULT

    cfg = json.loads((REPO_ROOT / "fpga" / "board_config" / "streamer_config.json").read_text(encoding="utf-8"))
    config_value = int(cfg["params"]["slot_mul_width"] if "params" in cfg else cfg["slot_mul_width"])

    rtl = _rtl_slot_mul_width()
    assert rtl == config_value == MODEL == HOST_DEFAULT, (
        "SLOT_MUL_WIDTH drifted across layers -- "
        f"RTL={rtl}, config={config_value}, engine_model={MODEL}, host_default={HOST_DEFAULT}. "
        "The RTL localparam is synthesised into the bitstream; every host copy must equal it or the "
        "host validates scan values against the wrong slot-operand bound.")


def test_device_layer_reads_slot_mul_width_from_the_board_config():
    """The device layer's ``DEFAULT_SLOT_MUL_WIDTH`` is DERIVED from the board config (single source
    it can edit), not a second hand-typed literal that could drift from the JSON."""
    from Zou_lab_control.neutral_atom.devices.fpga_pulse_streamer import DEFAULT_SLOT_MUL_WIDTH

    cfg = json.loads((REPO_ROOT / "fpga" / "board_config" / "streamer_config.json").read_text(encoding="utf-8"))
    config_value = int(cfg["params"]["slot_mul_width"] if "params" in cfg else cfg["slot_mul_width"])
    assert int(DEFAULT_SLOT_MUL_WIDTH) == config_value
