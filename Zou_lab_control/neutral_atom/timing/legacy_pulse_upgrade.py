# REMOVE-ME(legacy-pulse-upgrade): the ONE sanctioned legacy allowance in this repo.
# Delete this module + its three one-line call sites (grep "REMOVE-ME(legacy-pulse-upgrade)":
# timing/pulse_table.py PulseTableState.load, timing/persistence.py load_pulse_payload,
# frontend/data_figure.py pulse_state) + tests/test_legacy_pulse_upgrade.py once the lab's
# pre-v4 pulse .json files are gone -- target: before pulse-table schema version 6.
"""Upgrade pre-v4 pulse-table JSON payloads at the FILE boundary.

Schema v4 folded five topology fields (``channels`` / ``visible_channels`` / ``channel_labels`` /
``analog_buses`` / ``clk_channels``) into ``port_catalog`` + ``visible_ports``, added the required
``scan_code``, and made ``from_dict`` STRICT (exact fields + version equality).  Every lab .json
saved before that refactor is therefore rejected on load.  :func:`upgrade` rebuilds such a payload
into the current shape using the SAME construction primitives the modern code uses
(:meth:`PortCatalog.from_channels`, ``port_for_lane``, ``default_visible_ports``) plus the old v3
loader's ``.get`` defaults -- then stamps the current version so the strict ``from_dict`` accepts it.

Scope is deliberately minimal: ONLY the three file->dict entry points route through here.
``from_dict`` / ``to_dict`` / ``save`` and every in-process round-trip stay strict; a payload whose
schema does not match, whose version is not an int, or whose version is the current one or NEWER is
returned untouched (a too-new file still fails loudly in ``from_dict``).
"""

from __future__ import annotations

from typing import Mapping

from ..ports import PortCatalog, coerce_port_catalog


def upgrade(raw: Mapping[str, object]) -> dict:
    """Upgrade a pre-v4 pulse-table payload to the current schema; anything else passes through."""
    from .pulse_table import PulseTableState, default_visible_ports   # lazy: pulse_table calls back here

    payload = dict(raw or {})
    version = payload.get("version")
    if (payload.get("schema") != PulseTableState.schema
            or type(version) is not int
            or version >= PulseTableState.version):
        return payload                                     # modern / foreign / too-new: untouched

    # ---- port topology: five folded v1..v3 fields -> port_catalog + visible_ports -------------
    if "port_catalog" in payload:
        catalog_payload = payload["port_catalog"]
        catalog = coerce_port_catalog(catalog_payload)
    else:
        catalog = PortCatalog.from_channels(
            [str(c) for c in payload.get("channels") or ()],
            channel_labels=dict(payload.get("channel_labels") or {}),
            analog_buses=dict(payload.get("analog_buses") or {}),
            clk_channels=[str(c) for c in payload.get("clk_channels") or ()],
        )
        catalog_payload = catalog.to_dict()
    if "visible_ports" in payload:
        visible = [str(k) for k in payload["visible_ports"] or ()]
    else:
        # v1..v3 stored VISIBLE RAW LANES; each maps to its owning port's key (a DAC bus's lanes
        # all map to the one bus key -- ordered dedup keeps the first occurrence).
        visible = []
        for lane in payload.get("visible_channels") or ():
            try:
                key = catalog.port_for_lane(str(lane)).key
            except KeyError:
                continue                                   # a lane the catalog no longer knows
            if key not in visible:
                visible.append(key)
    if not visible:
        visible = list(default_visible_ports(catalog))     # old files without visible_channels

    def _slot(entry: Mapping[str, object]) -> dict:
        out = dict(entry)
        out.setdefault("name", "")                          # v4 added the explicit name field
        out.setdefault("label", "")
        out.setdefault("unit", "ns")
        out.setdefault("nominal", 0.0)
        return out

    def _api(entry: Mapping[str, object]) -> dict:
        out = dict(entry)
        out.setdefault("unit", "ns")
        return out

    def _period(entry: Mapping[str, object]) -> dict:
        out = dict(entry)
        out.setdefault("name", "")
        out.setdefault("unit", "ns")
        return out

    def _int_or(value, default: int) -> int:
        return int(default) if value is None else int(value)

    # ---- the v3 loader's .get defaults, re-shaped to the EXACT v4 field set --------------------
    # (Building a fresh dict drops every dead field -- x_ns, the five folded topology fields --
    # in one stroke, exactly what the strict from_dict's unexpected-fields gate demands.)
    return {
        "schema": PulseTableState.schema,
        "version": PulseTableState.version,
        "name": str(payload.get("name") or "pulse"),
        "port_catalog": catalog_payload,
        "scan_slots": [_slot(s) for s in payload.get("scan_slots") or ()],
        "scan_table": list(payload.get("scan_table") or ()),
        "scan_code": str(payload.get("scan_code") or ""),
        "api_slots": [_api(s) for s in payload.get("api_slots") or ()],
        "time_step_ns": float(payload.get("time_step_ns") or 1.0),
        "periods": [_period(p) for p in payload.get("periods") or ()],
        "visible_ports": visible,
        "analog_bus_modes": dict(payload.get("analog_bus_modes") or {}),
        "delays": dict(payload.get("delays") or {}),
        "delay_units": dict(payload.get("delay_units") or {}),
        "repeat_start": None if payload.get("repeat_start") is None else int(payload["repeat_start"]),
        "repeat_end": None if payload.get("repeat_end") is None else int(payload["repeat_end"]),
        "repeat_count": _int_or(payload.get("repeat_count"), 1),
        "repeat_forever": bool(payload.get("repeat_forever", True)),
        "scan_repeats": _int_or(payload.get("scan_repeats"), 0),
    }


__all__ = ["upgrade"]
