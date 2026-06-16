"""Contract: a producer's parameters are derived from its signature (single source).

``params_from_signature`` turns a callable's keyword parameters + typed annotations
into ``ParamField`` records the GUI form renders and the API call shares -- injected
positional args (devices/hub/out) are skipped, types/units/choices are carried, and
order is preserved.
"""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if sys.path[0] != str(REPO_ROOT):
    sys.path.insert(0, str(REPO_ROOT))

from Zou_lab_control.neutral_atom.operations.params import (
    Choice, Param, ParamField, PulseScan, ScanArray, SignalRef, params_from_signature)


def _sample(camera, sequencer, *, exposure: Param(float, unit="s") = 0.1,
            roi_radius: Param(int, lo=0) = 1, average: Param(bool) = False,
            update_mode: Choice(("roll", "average")) = "roll",
            sweep: ScanArray(unit="s") = None, source: SignalRef() = "frame",
            duration: PulseScan("Imaging", "duration") = None, note: str = "hi"):
    ...


def test_signature_to_param_fields():
    fields = params_from_signature(_sample)
    by = {f.name: f for f in fields}

    # injected positional args are NOT UI fields
    assert "camera" not in by and "sequencer" not in by

    assert by["exposure"].kind == "float" and by["exposure"].default == 0.1 and by["exposure"].unit == "s"
    assert by["roi_radius"].kind == "int" and by["roi_radius"].lo == 0 and by["roi_radius"].default == 1
    assert by["average"].kind == "bool" and by["average"].default is False
    assert by["update_mode"].kind == "choice" and by["update_mode"].choices == ("roll", "average")
    assert by["update_mode"].default == "roll"
    assert by["sweep"].kind == "array" and by["sweep"].unit == "s"
    assert by["source"].kind == "signal" and by["source"].default == "frame"
    assert by["duration"].kind == "pulse_scan" and by["duration"].target == "Imaging.duration"
    assert by["note"].kind == "str" and by["note"].default == "hi"

    # signature order preserved (a form lays fields out top-to-bottom in this order)
    assert [f.name for f in fields] == [
        "exposure", "roi_radius", "average", "update_mode", "sweep", "source", "duration", "note"]
    assert all(isinstance(f, ParamField) for f in fields)


def test_injected_and_varargs_skipped():
    def producer(hub, camera, *, exposure: Param(float) = 0.02, **extra):
        ...
    names = [f.name for f in params_from_signature(producer)]
    assert names == ["exposure"]   # hub/camera injected, **extra dropped
