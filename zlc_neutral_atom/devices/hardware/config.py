"""Leaf-owned authoring and canonical codec for the real apparatus."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass

from zlc_neutral_atom.authoring import AuthoringField, AuthoringSchema
from zlc_neutral_atom.devices.sequencer.config import (
    DEFAULT_REMOTE_PORT,
    DEFAULT_TRANSPORT_TIMEOUT_SECONDS,
)
from zlc_storage import canonical_text, integer, positive_real


_MIN_POSITIVE_FLOAT = math.nextafter(0.0, math.inf)


def _optional_roi(values: tuple[object, object, object, object], field: str):
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError(f"{field} requires all four x/y/width/height values or none")
    return tuple(
        integer(value, f"{field}[{index}]", minimum=0 if index < 2 else 1)
        for index, value in enumerate(values)
    )


def _centers_from_json(value: object) -> tuple[tuple[float, float], ...]:
    text = canonical_text(value, "readout_site_centers_json")
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("readout_site_centers_json is not valid JSON") from exc
    if not isinstance(decoded, list) or not decoded:
        raise ValueError("readout_site_centers_json must be a non-empty list")
    centers: list[tuple[float, float]] = []
    for index, raw in enumerate(decoded):
        if not isinstance(raw, list) or len(raw) != 2:
            raise ValueError(f"readout site center {index} must be [x, y]")
        if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in raw):
            raise TypeError(f"readout site center {index} must contain two numbers")
        x, y = float(raw[0]), float(raw[1])
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("readout site centers must be finite")
        centers.append((x, y))
    if len(set(centers)) != len(centers):
        raise ValueError("readout site centers must be unique")
    return tuple(centers)


def _centers_json(value: tuple[tuple[float, float], ...]) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


@dataclass(frozen=True, slots=True)
class HardwareInstallationConfig:
    pulse_host: str
    pylon_serial: str
    readout_site_centers_xy: tuple[tuple[float, float], ...]
    readout_grid_shape_yx: tuple[int, int]
    pulse_port: int = DEFAULT_REMOTE_PORT
    pulse_transport_timeout_seconds: float = DEFAULT_TRANSPORT_TIMEOUT_SECONDS
    dcam_device_index: int = 0
    dcam_exposure_seconds: float = 0.02
    dcam_readout_speed: int = 1
    dcam_binning: int = 1
    dcam_roi_xywh: tuple[int, int, int, int] | None = None
    readout_trigger_lane: str = "ch11"
    pylon_exposure_seconds: float = 0.005
    pylon_trigger_source: str = "Line1"
    pylon_roi_xywh: tuple[int, int, int, int] | None = None
    pylon_timeout_seconds: float = 2.0
    mot_trigger_lane: str = "ch06"

    def __post_init__(self) -> None:
        for name in (
            "pulse_host",
            "pylon_serial",
            "readout_trigger_lane",
            "pylon_trigger_source",
            "mot_trigger_lane",
        ):
            object.__setattr__(self, name, canonical_text(getattr(self, name), name))
        port = integer(self.pulse_port, "pulse_port", minimum=1)
        assert port is not None
        if port > 65535:
            raise ValueError("pulse_port must be at most 65535")
        object.__setattr__(self, "pulse_port", port)
        for name in (
            "pulse_transport_timeout_seconds",
            "dcam_exposure_seconds",
            "pylon_exposure_seconds",
            "pylon_timeout_seconds",
        ):
            object.__setattr__(self, name, positive_real(getattr(self, name), name))
        for name, minimum in (
            ("dcam_device_index", 0),
            ("dcam_readout_speed", 1),
            ("dcam_binning", 1),
        ):
            value = integer(getattr(self, name), name, minimum=minimum)
            assert value is not None
            object.__setattr__(self, name, value)
        if self.dcam_binning not in (1, 2, 4, 8, 16):
            raise ValueError("dcam_binning must be one of 1, 2, 4, 8, or 16")
        for name in ("dcam_roi_xywh", "pylon_roi_xywh"):
            roi = getattr(self, name)
            if roi is not None:
                object.__setattr__(self, name, _optional_roi(tuple(roi), name))
        shape = tuple(
            integer(value, f"readout_grid_shape_yx[{index}]", minimum=1)
            for index, value in enumerate(self.readout_grid_shape_yx)
        )
        if len(shape) != 2:
            raise ValueError("readout_grid_shape_yx must contain rows and columns")
        object.__setattr__(self, "readout_grid_shape_yx", shape)
        centers = tuple((float(x), float(y)) for x, y in self.readout_site_centers_xy)
        if len(centers) != shape[0] * shape[1]:
            raise ValueError("readout site center count must equal rows * columns")
        if any(not math.isfinite(value) for pair in centers for value in pair):
            raise ValueError("readout site centers must be finite")
        if len(set(centers)) != len(centers):
            raise ValueError("readout site centers must be unique")
        object.__setattr__(self, "readout_site_centers_xy", centers)


def _value(config: HardwareInstallationConfig | None, name: str, default: object) -> object:
    return default if config is None else getattr(config, name)


def hardware_authoring_schema(config: object | None) -> AuthoringSchema:
    if config is not None and not isinstance(config, HardwareInstallationConfig):
        raise TypeError("config must be HardwareInstallationConfig or None")
    c = config
    dcam_roi = (None, None, None, None) if c is None or c.dcam_roi_xywh is None else c.dcam_roi_xywh
    pylon_roi = (None, None, None, None) if c is None or c.pylon_roi_xywh is None else c.pylon_roi_xywh
    fields = (
        AuthoringField("pulse_host", "text", "Pulse server host", _value(c, "pulse_host", ""), True),
        AuthoringField("pulse_port", "int", "Pulse server port", _value(c, "pulse_port", DEFAULT_REMOTE_PORT), True, minimum=1, maximum=65535),
        AuthoringField("pulse_transport_timeout_seconds", "float", "Pulse transport timeout", _value(c, "pulse_transport_timeout_seconds", DEFAULT_TRANSPORT_TIMEOUT_SECONDS), True, unit="s", minimum=_MIN_POSITIVE_FLOAT),
        AuthoringField("dcam_device_index", "int", "qCMOS device index", _value(c, "dcam_device_index", 0), True, minimum=0),
        AuthoringField("dcam_exposure_seconds", "float", "qCMOS exposure", _value(c, "dcam_exposure_seconds", 0.02), True, unit="s", minimum=_MIN_POSITIVE_FLOAT),
        AuthoringField("dcam_readout_speed", "int", "qCMOS readout speed", _value(c, "dcam_readout_speed", 1), True, minimum=1),
        AuthoringField("dcam_binning", "int", "qCMOS binning", _value(c, "dcam_binning", 1), True, minimum=1, maximum=16),
        AuthoringField("dcam_roi_x", "int", "qCMOS ROI x", dcam_roi[0], False, minimum=0, allow_blank=True),
        AuthoringField("dcam_roi_y", "int", "qCMOS ROI y", dcam_roi[1], False, minimum=0, allow_blank=True),
        AuthoringField("dcam_roi_width", "int", "qCMOS ROI width", dcam_roi[2], False, minimum=1, allow_blank=True),
        AuthoringField("dcam_roi_height", "int", "qCMOS ROI height", dcam_roi[3], False, minimum=1, allow_blank=True),
        AuthoringField("readout_trigger_lane", "text", "Readout trigger lane", _value(c, "readout_trigger_lane", "ch11"), True),
        AuthoringField("readout_grid_rows", "int", "Readout grid rows", 1 if c is None else c.readout_grid_shape_yx[0], True, minimum=1),
        AuthoringField("readout_grid_columns", "int", "Readout grid columns", 1 if c is None else c.readout_grid_shape_yx[1], True, minimum=1),
        AuthoringField("readout_site_centers_json", "text", "Readout site centers [x,y]", "" if c is None else _centers_json(c.readout_site_centers_xy), True),
        AuthoringField("pylon_serial", "text", "Basler serial", _value(c, "pylon_serial", ""), True),
        AuthoringField("pylon_exposure_seconds", "float", "Basler exposure", _value(c, "pylon_exposure_seconds", 0.005), True, unit="s", minimum=_MIN_POSITIVE_FLOAT),
        AuthoringField("pylon_trigger_source", "text", "Basler trigger source", _value(c, "pylon_trigger_source", "Line1"), True),
        AuthoringField("pylon_roi_x", "int", "Basler ROI x", pylon_roi[0], False, minimum=0, allow_blank=True),
        AuthoringField("pylon_roi_y", "int", "Basler ROI y", pylon_roi[1], False, minimum=0, allow_blank=True),
        AuthoringField("pylon_roi_width", "int", "Basler ROI width", pylon_roi[2], False, minimum=1, allow_blank=True),
        AuthoringField("pylon_roi_height", "int", "Basler ROI height", pylon_roi[3], False, minimum=1, allow_blank=True),
        AuthoringField("pylon_timeout_seconds", "float", "Basler frame timeout", _value(c, "pylon_timeout_seconds", 2.0), True, unit="s", minimum=_MIN_POSITIVE_FLOAT),
        AuthoringField("mot_trigger_lane", "text", "MOT trigger lane", _value(c, "mot_trigger_lane", "ch06"), True),
    )
    return AuthoringSchema(fields)


def hardware_config_from_parameters(values: Mapping[str, object]) -> HardwareInstallationConfig:
    frozen = hardware_authoring_schema(None).freeze(values)
    return HardwareInstallationConfig(
        pulse_host=frozen["pulse_host"],
        pulse_port=frozen["pulse_port"],
        pulse_transport_timeout_seconds=frozen["pulse_transport_timeout_seconds"],
        dcam_device_index=frozen["dcam_device_index"],
        dcam_exposure_seconds=frozen["dcam_exposure_seconds"],
        dcam_readout_speed=frozen["dcam_readout_speed"],
        dcam_binning=frozen["dcam_binning"],
        dcam_roi_xywh=_optional_roi((frozen["dcam_roi_x"], frozen["dcam_roi_y"], frozen["dcam_roi_width"], frozen["dcam_roi_height"]), "dcam_roi_xywh"),
        readout_trigger_lane=frozen["readout_trigger_lane"],
        readout_grid_shape_yx=(frozen["readout_grid_rows"], frozen["readout_grid_columns"]),
        readout_site_centers_xy=_centers_from_json(frozen["readout_site_centers_json"]),
        pylon_serial=frozen["pylon_serial"],
        pylon_exposure_seconds=frozen["pylon_exposure_seconds"],
        pylon_trigger_source=frozen["pylon_trigger_source"],
        pylon_roi_xywh=_optional_roi((frozen["pylon_roi_x"], frozen["pylon_roi_y"], frozen["pylon_roi_width"], frozen["pylon_roi_height"]), "pylon_roi_xywh"),
        pylon_timeout_seconds=frozen["pylon_timeout_seconds"],
        mot_trigger_lane=frozen["mot_trigger_lane"],
    )


def hardware_config_to_parameters(config: object) -> dict[str, object]:
    if not isinstance(config, HardwareInstallationConfig):
        raise TypeError("config must be HardwareInstallationConfig")
    dcam_roi = (None, None, None, None) if config.dcam_roi_xywh is None else config.dcam_roi_xywh
    pylon_roi = (None, None, None, None) if config.pylon_roi_xywh is None else config.pylon_roi_xywh
    return {
        "pulse_host": config.pulse_host,
        "pulse_port": config.pulse_port,
        "pulse_transport_timeout_seconds": config.pulse_transport_timeout_seconds,
        "dcam_device_index": config.dcam_device_index,
        "dcam_exposure_seconds": config.dcam_exposure_seconds,
        "dcam_readout_speed": config.dcam_readout_speed,
        "dcam_binning": config.dcam_binning,
        "dcam_roi_x": dcam_roi[0], "dcam_roi_y": dcam_roi[1],
        "dcam_roi_width": dcam_roi[2], "dcam_roi_height": dcam_roi[3],
        "readout_trigger_lane": config.readout_trigger_lane,
        "readout_grid_rows": config.readout_grid_shape_yx[0],
        "readout_grid_columns": config.readout_grid_shape_yx[1],
        "readout_site_centers_json": _centers_json(config.readout_site_centers_xy),
        "pylon_serial": config.pylon_serial,
        "pylon_exposure_seconds": config.pylon_exposure_seconds,
        "pylon_trigger_source": config.pylon_trigger_source,
        "pylon_roi_x": pylon_roi[0], "pylon_roi_y": pylon_roi[1],
        "pylon_roi_width": pylon_roi[2], "pylon_roi_height": pylon_roi[3],
        "pylon_timeout_seconds": config.pylon_timeout_seconds,
        "mot_trigger_lane": config.mot_trigger_lane,
    }


__all__ = [
    "HardwareInstallationConfig",
    "hardware_authoring_schema",
    "hardware_config_from_parameters",
    "hardware_config_to_parameters",
]
