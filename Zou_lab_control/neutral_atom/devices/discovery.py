"""Port/bus scanning: find attached instruments and suggest ready-to-use device configs.

The confocal-GUI pattern (its ``_VisaScanWorker``): enumerate every bus a lab instrument could
live on, annotate each hit with an identity string, and NEVER fail because a driver library is
missing -- a missing library is itself a reported row ("pip install ..."), not an exception.

Decoupling rule: **a device class owns its own discovery.**  Any class in the device registry
may implement a ``discover() -> list[DiscoveredDevice]`` classmethod that enumerates its bus and
returns rows whose ``config`` is a ready ``{"type", "params"}`` entry for itself (see
:meth:`~.pylon.PylonCamera.discover`).  This module only AGGREGATES: it walks the registry,
asks each discoverable class, and appends the one bus that belongs to no class (bare VISA
instruments, listed informationally with their ``*IDN?`` identity so the operator can wire the
address into a custom class via ``register_device_class``).  Adding a new discoverable device
touches ONLY that device's class -- never this file.

    >>> import Zou_lab_control.neutral_atom as na
    >>> found = na.discover_devices()          # prints a table, returns the rows
    >>> cams = [d for d in found if d.config]  # rows with a ready config
    >>> devices = na.load_devices({"monitor_camera": cams[0].config})
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DiscoveredDevice:
    """One row of a discovery scan.

    ``kind``    the bus/driver family (``"basler"`` / ``"visa"`` / ``"note"``);
    ``ident``   the stable address to reach it (camera serial, VISA resource name);
    ``label``   the human identity (model name, ``*IDN?`` reply);
    ``config``  a ready ``{"type": ..., "params": ...}`` device-config entry, or ``None``
                when no registered class answers this row (informational listing).
    """

    kind: str
    ident: str
    label: str
    config: dict | None = field(default=None)

    def __str__(self) -> str:  # the table row a notebook prints
        ready = "ready" if self.config is not None else "-"
        return f"[{self.kind:>6}] {self.ident:<28} {self.label}  ({ready})"


def discovery_note(kind: str, text: str) -> DiscoveredDevice:
    """A status row (missing library / empty bus / probe failure) -- the confocal contract:
    report, never raise.  Shared by every device class's ``discover()``."""
    return DiscoveredDevice(kind="note", ident=str(kind), label=str(text), config=None)


# --------------------------------------------------------------------------- registry aggregation
def _registered_discoverers() -> tuple[list[tuple[str, object]], list[DiscoveredDevice]]:
    """Every registry class that self-describes its discovery (has a ``discover`` classmethod),
    plus a note row per class that FAILED to import -- the confocal contract applies to the
    aggregator itself: a broken driver is a visible row, never a silent hole in the scan."""
    from .registry import DEVICE_CLASSES, resolve_class

    found: list[tuple[str, object]] = []
    notes: list[DiscoveredDevice] = []
    for name in sorted(DEVICE_CLASSES):
        try:
            cls = resolve_class(name)
        except Exception as exc:
            notes.append(discovery_note(name, f"device class failed to import: {exc}"))
            continue
        if callable(getattr(cls, "discover", None)):
            found.append((name, cls))
    return found, notes


# --------------------------------------------------------------------------- the class-less bus
def visa_rows(resources, idn_of) -> list[DiscoveredDevice]:
    """VISA enumeration -> rows.  ``resources`` is the ``list_resources()`` tuple, ``idn_of``
    maps a resource name to its ``*IDN?`` reply (or a placeholder) -- pure apart from the
    injected query.  No registered class answers a bare VISA instrument, so the rows are
    informational (``config=None``): the operator wires the printed address into a custom
    device class (``register_device_class``)."""

    return [DiscoveredDevice(kind="visa", ident=str(r), label=str(idn_of(r)), config=None)
            for r in resources]


def _probe_visa(idn_timeout_ms: int = 300) -> list[DiscoveredDevice]:
    try:
        import pyvisa
    except ImportError:
        return [discovery_note("visa", "pyvisa not installed -- pip install pyvisa")]
    try:
        rm = pyvisa.ResourceManager()
        resources = rm.list_resources()
    except Exception as exc:
        return [discovery_note("visa", f"list_resources failed: {exc}")]
    if not resources:
        return [discovery_note("visa", "no VISA resources")]

    def idn_of(resource: str) -> str:
        # the confocal probe: open, ask *IDN? with a short timeout, never raise
        try:
            inst = rm.open_resource(resource)
            try:
                inst.timeout = int(idn_timeout_ms)
                try:
                    return inst.query("*IDN?").strip()
                except Exception:
                    return "<no *IDN? reply>"
            finally:
                try:
                    inst.close()
                except Exception:
                    pass
        except Exception:
            return "<open failed>"

    return visa_rows(resources, idn_of)


def discover_devices(*, visa: bool = True, idn_timeout_ms: int = 300,
                     display: bool = True) -> list[DiscoveredDevice]:
    """Scan the buses lab instruments live on and return :class:`DiscoveredDevice` rows.

    Every REGISTERED device class that implements ``discover()`` scans its own bus (the class
    owns the knowledge of how it is found and what its config looks like); ``visa=True`` also
    lists bare VISA resources (each queried with ``*IDN?`` under ``idn_timeout_ms``).  A missing
    driver library or an empty bus is reported as a row, never raised.  ``display=True`` also
    prints the table."""

    discoverers, rows = _registered_discoverers()
    rows = list(rows)                           # import-failure notes lead the table
    seen: set[tuple[str, str]] = set()
    for _name, cls in discoverers:
        try:
            class_rows = list(cls.discover())
        except Exception as exc:                # a discover() must not break the whole scan
            class_rows = [discovery_note(cls.__name__, f"discover failed: {exc}")]
        for row in class_rows:
            key = (row.kind, row.ident)         # two registry names for one class scan once
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
    if visa:
        rows += _probe_visa(idn_timeout_ms)
    if display:
        for row in rows:
            print(row)
    return rows
