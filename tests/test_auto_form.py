"""Contract: the auto-generated form renders a producer's ParamField list and reads
values back, with no per-producer hand-written UI (the P5 auto-UI engine end to end).

Offscreen Qt; the form is built from ``params_from_signature`` (the same single source
the API call uses), so a measurement/processor/task gets its parameter UI for free.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if sys.path[0] != str(REPO_ROOT):
    sys.path.insert(0, str(REPO_ROOT))

# Module-level so eval_str can resolve the annotations in _sample's globals (the
# real measurement modules import the spec types the same way).
from Zou_lab_control.neutral_atom.operations.params import (
    Choice, Param, ScanArray, SignalRef, params_from_signature)


@pytest.fixture(autouse=True)
def _offscreen(monkeypatch):
    pytest.importorskip("PyQt5")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from Zou_lab_control.frontend.qt_fluent import ensure_qt_app
    ensure_qt_app()


def _sample(camera, *, exposure: Param(float, unit="s") = 0.1, n: Param(int) = 4,
            avg: Param(bool) = False, mode: Choice(("roll", "average")) = "roll",
            sweep: ScanArray() = None, src: SignalRef() = "frame"):
    ...


def test_auto_form_builds_from_signature_and_round_trips():
    from Zou_lab_control.frontend.auto_form import AutoForm

    fields = params_from_signature(_sample)
    form = AutoForm(fields, signal_names=["frame", "rate"])

    v = form.values()
    assert v["exposure"] == 0.1 and v["n"] == 4 and v["avg"] is False
    assert v["mode"] == "roll" and v["src"] == "frame" and v["sweep"] is None

    form.set_values({"exposure": 0.5, "n": 10, "avg": True, "mode": "average",
                     "sweep": np.arange(0.0, 1.01, 0.5), "src": "rate"})
    v2 = form.values()
    assert v2["exposure"] == 0.5 and v2["n"] == 10 and v2["avg"] is True
    assert v2["mode"] == "average" and v2["src"] == "rate"
    np.testing.assert_allclose(v2["sweep"], [0.0, 0.5, 1.0])


def test_parse_scan_text():
    from Zou_lab_control.frontend.auto_form import parse_scan_text
    np.testing.assert_allclose(parse_scan_text("0:1:0.5"), [0.0, 0.5, 1.0])
    np.testing.assert_allclose(parse_scan_text("1, 2, 3"), [1.0, 2.0, 3.0])
    assert parse_scan_text("") is None
