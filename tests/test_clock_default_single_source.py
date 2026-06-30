"""Contract test: the 50 MHz runtime/compile clock default has ONE authority.

The FPGA tick rate is configured in exactly one place --
``fpga/board_config/streamer_config.json`` (``clock_hz``), read by
``fpga.pulse_streamer.host.image.default_clock_hz`` and re-exposed through the
dependency-free ``Zou_lab_control._clock`` seam.  Every default that once spelled
``50_000_000.0`` as a bare literal -- the timing compilers, the runtime sequencer +
AXI session, the session-facade fallback, the virtual sequencer, the pulse GUI --
must now resolve to THAT value, or the compile clock and the AXI runtime clock can
silently diverge the moment someone edits the config.

These tests pin that single source:
  * the seam value == the config reader (no second definition);
  * sequencer + axi_session expose ONE ``DEFAULT_RUNTIME_CLOCK_HZ`` object, both
    equal to the config clock;
  * every previously-hardcoded default (timing edges/verilog/from_sequence,
    session fallbacks, virtual sequencer ctor, pulse GUI) equals the config clock;
  * a repo-wide grep finds ``50_000_000`` only in the config-fallback constant in
    ``host.image`` (and config JSON / docs), never as a live default elsewhere.
"""

from __future__ import annotations

import inspect
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if sys.path[0] != str(REPO_ROOT):
    sys.path.insert(0, str(REPO_ROOT))

from fpga.pulse_streamer.host.image import default_clock_hz
from Zou_lab_control import _clock
from Zou_lab_control.neutral_atom.devices import sequencer as sq
from Zou_lab_control.neutral_atom.devices import axi_session as axi
from Zou_lab_control.neutral_atom.devices import virtual as virt
from Zou_lab_control.neutral_atom.timing import sequence as seq_mod
from Zou_lab_control.neutral_atom.timing import verilog as verilog_mod
from Zou_lab_control.neutral_atom.timing import pulse_table as pt_mod


# --------------------------------------------------------------------------- the one authority
def test_clock_seam_equals_config_reader():
    """The seam adds no second definition -- it is the config reader's value."""
    assert _clock.default_clock_hz is default_clock_hz
    assert _clock.DEFAULT_CLOCK_HZ == default_clock_hz()


def test_sequencer_and_axi_share_one_runtime_clock_object():
    """Both device modules expose the SAME constant object (axi imports it from
    sequencer), so there is exactly one ``DEFAULT_RUNTIME_CLOCK_HZ`` in the package."""
    assert sq.DEFAULT_RUNTIME_CLOCK_HZ == default_clock_hz()
    assert axi.DEFAULT_RUNTIME_CLOCK_HZ == sq.DEFAULT_RUNTIME_CLOCK_HZ


def _default(callable_or_func, name: str):
    return inspect.signature(callable_or_func).parameters[name].default


def test_every_clock_default_resolves_to_the_config_clock():
    """Every previously-hardcoded ``50_000_000.0`` default now equals the config clock."""
    cfg = default_clock_hz()
    # timing layer (dependency-free): its module-level constant + the three default args
    assert seq_mod.DEFAULT_CLOCK_HZ == cfg
    assert _default(seq_mod.PulseSequence.edges, "clock_hz") == cfg
    assert _default(seq_mod.plot_sequence, "clock_hz") == cfg
    assert _default(verilog_mod.generate_verilog, "clock_hz") == cfg
    assert _default(pt_mod.PulseTableState.from_sequence, "clock_hz") == cfg
    # device layer
    assert _default(virt.VirtualSequencer.__init__, "clock_hz") == cfg
    # the AXI session ctor default is the shared runtime constant
    assert _default(axi.VivadoAxiStreamerSession.__init__, "clock_hz") == cfg


def test_pulse_gui_hardware_clock_default_is_the_config_clock():
    from Zou_lab_control.frontend import pulse_gui

    assert pulse_gui.DEFAULT_HARDWARE_CLOCK_HZ == default_clock_hz()
    # and its derived tick step is consistent with that one source
    assert pulse_gui.DEFAULT_TIME_STEP_NS == 1_000_000_000.0 / default_clock_hz()


def test_no_stray_hardcoded_clock_literal_in_the_package():
    """The only live ``50_000_000`` in the Python package is the config-fallback
    constant in ``host.image`` (config JSON + docs are not code defaults)."""
    pkg = REPO_ROOT / "Zou_lab_control"
    offenders: list[str] = []
    for path in pkg.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if "50_000_000" in line:
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{i}: {line.strip()}")
    # host.image.DEFAULT_CLOCK_HZ is the documented config fallback -- allowed.
    offenders = [o for o in offenders if "fpga/pulse_streamer/host/image.py" not in o.replace("\\", "/")]
    assert offenders == [], (
        "a 50_000_000 clock literal survived outside the single config source: " + "; ".join(offenders)
    )
