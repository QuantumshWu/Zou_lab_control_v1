"""Contract (#H3s-F2): the GUI slot label names the PERIOD/CHANNEL, never the raw 0-based target.

The user complained "a1  duration @ 1" is information-free ("what is 1?").  `slot_label(kind, target)`
is the ONE formatter (pulse editor + task-console pulse-scan form share it); it humanizes the internal
target handle to a 1-based period / named channel / bus.  This guard fails if a call site reverts to
dumping the raw "@ target".
"""

from __future__ import annotations

import os
import re
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from Zou_lab_control.frontend.pulse_gui import slot_label

_TC = Path(__file__).resolve().parents[1] / "Zou_lab_control" / "frontend" / "task_console.py"


def test_duration_label_is_one_based_period():
    # target is the 0-based period index -> "Period 2" matches the card's "Period 2/5"
    assert slot_label("duration", "1") == "Period 2 duration"
    assert slot_label("duration", "0") == "Period 1 duration"


def test_dac_label_names_bus_and_period():
    out = slot_label("dac", "da@1")
    assert "da" in out and "Period 2" in out          # bus name + 1-based period, no bare "@ 1"
    assert "@" not in out


def test_delay_label_names_the_channel():
    assert slot_label("delay", "ch07") == "ch07 delay"
    assert slot_label("delay", "da") == "da delay"     # a bus-delay handle names the bus


def test_no_raw_at_target_template_in_task_console():
    # mechanical single-source guard: the meaningless f"{kind} @ {target}" template is gone
    src = _TC.read_text(encoding="utf-8")
    assert "@ {target}" not in src, "task_console.py still dumps a raw '@ {target}' slot label"
    assert not re.search(r'\{kind\}\s*@\s*\{target\}', src)
