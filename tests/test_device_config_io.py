"""Contract for the experiment-config load/save round-trip (the device-manager GUI's Load / Save /
Open buttons + the notebook ``exp.save_config`` / ``exp.load_config`` / ``exp.open_devices`` API).

The guarantee: ``exp.save_config(path)`` writes a config that ``na.connect(path)`` (or ``load_config``)
reproduces the SAME device set from -- so a discovered / tuned hardware config is reusable, not retyped.
``to_config`` is the ONE round-trippable serializer (vs ``snapshot``, a per-device state dump)."""

from __future__ import annotations

from conftest import raw_device_set

import json
from pathlib import Path

import Zou_lab_control.neutral_atom as na


def _names(exp):
    return sorted(raw_device_set(exp).devices.keys())


def test_to_config_round_trips_through_connect(tmp_path):
    exp = na.connect("virtual", sitemap={"grid_shape": (3, 4)})
    try:
        cfg = raw_device_set(exp).to_config()
        # to_config is a {role: {"type", "params"}} dict -- the SAME shape connect() takes as a dict.
        assert set(cfg) == set(raw_device_set(exp).devices)
        for entry in cfg.values():
            assert "type" in entry
        # deep copy: mutating the returned config must not touch the live one.
        cfg["camera"]["params"]["__scratch__"] = 1
        assert "__scratch__" not in raw_device_set(exp).to_config().get("camera", {}).get("params", {})
        # round-trips through a fresh connect().
        rebuilt = na.connect(raw_device_set(exp).to_config())
        try:
            assert _names(rebuilt) == _names(exp)
        finally:
            rebuilt.close()
    finally:
        exp.close()


def test_save_config_writes_json_and_reconnects(tmp_path):
    exp = na.connect("virtual", sitemap={"grid_shape": (3, 4)})
    try:
        written = exp.save_config(tmp_path / "myexp")     # missing .json suffix is added
        assert written.name == "myexp.json" and written.exists()
        json.loads(written.read_text(encoding="utf-8"))   # valid JSON
        reconnected = na.connect(str(written))
        try:
            assert _names(reconnected) == _names(exp)
        finally:
            reconnected.close()
    finally:
        exp.close()


def test_load_config_swaps_devices_and_rederives_state(tmp_path):
    exp = na.connect("virtual", sitemap={"grid_shape": (3, 4)})
    try:
        path = exp.save_config(tmp_path / "cfg.json")
        exp.readout.sitemap(frames=4, display=False)      # produce a calibration to prove it is reset
        assert exp.calibration_data is not None
        before = id(raw_device_set(exp))
        returned = exp.load_config(str(path))
        assert returned is exp                            # returns self
        assert id(raw_device_set(exp)) != before                 # a NEW device set
        assert _names(exp) == ["camera", "laser", "monitor_camera", "rf", "sequencer", "trap_array"]
        assert exp.sequence is not None                   # imaging sequence re-derived
        assert exp.calibration_data is None               # stale calibration dropped
    finally:
        exp.close()


def test_open_devices_is_idempotent_on_virtual():
    exp = na.connect("virtual")
    try:
        assert exp.open_devices() is exp                  # returns self, no error on the virtual backend
        assert exp.open_devices() is exp                  # safe to call again
    finally:
        exp.close()
