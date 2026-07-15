"""Port/bus scanning: find attached instruments and suggest ready-to-use device configs.

The confocal-GUI pattern (its ``_VisaScanWorker``): enumerate every bus a lab instrument could
live on, annotate each hit with an identity string, and NEVER fail because a driver library is
missing -- a missing library is itself a reported row ("pip install ..."), not an exception.

Decoupling rule: **a device class owns its own discovery.**  Any class in the device registry
may implement a ``discover() -> list[DiscoveredDevice]`` classmethod that enumerates its bus and
returns rows whose ``config`` is a ready ``{"type", "params"}`` entry for itself (see
:meth:`~.pylon.PylonCamera.discover`).  A bus that belongs to NO device class is scanned by a
DISCOVERY PROVIDER -- a plain zero-arg callable returning rows, registered under a name in
``DISCOVERY_PROVIDERS`` via :func:`register_discovery_provider`.  VISA is the built-in provider:
it lists every resource with its ``*IDN?`` identity so the operator can wire the address into a
custom device class (``register_device_class``).  This module only AGGREGATES the two
registries -- adding a discoverable device or a new bus never touches this file.

    >>> import Zou_lab_control.neutral_atom as na
    >>> found = na.discover_devices()          # prints a table, returns the rows
    >>> cams = [d for d in found if d.config]  # rows with a ready config
    >>> devices = na.load_devices({"monitor_camera": cams[0].config})
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


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


# --------------------------------------------------------------------------- class-less buses
#: Named bus scanners for instruments NO device class claims (name -> zero-arg callable
#: returning :class:`DiscoveredDevice` rows).  Every provider obeys the same contract as a
#: class's ``discover()``: report rows/notes, never raise.  VISA is the built-in entry;
#: operators add their own via :func:`register_discovery_provider`.
DISCOVERY_PROVIDERS: dict[str, Callable[[], "list[DiscoveredDevice]"]] = {}


def register_discovery_provider(name: str, provider: Callable[[], "list[DiscoveredDevice]"]) -> None:
    """Register a bus scanner under ``name`` so ``discover_devices()`` runs it.

    A provider is for a bus that belongs to no single device class (a serial port zoo, a
    site-specific chassis...).  It takes no arguments and returns :class:`DiscoveredDevice`
    rows -- use :func:`discovery_note` for a missing library / empty bus, and NEVER raise
    (a crash is caught and reported as a note row anyway).  Re-registering a name replaces
    the previous provider."""

    if not str(name).strip():
        raise ValueError("discovery provider name must not be empty.")
    if not callable(provider):
        raise TypeError("discovery provider must be a callable returning DiscoveredDevice rows.")
    DISCOVERY_PROVIDERS[str(name)] = provider


def visa_rows(resources, idn_of) -> list[DiscoveredDevice]:
    """VISA enumeration -> rows.  ``resources`` is the ``list_resources()`` tuple, ``idn_of``
    maps a resource name to its ``*IDN?`` reply (or a placeholder) -- pure apart from the
    injected query.  No registered class answers a bare VISA instrument, so the rows are
    informational (``config=None``): the operator wires the printed address into a custom
    device class (``register_device_class``)."""

    return [DiscoveredDevice(kind="visa", ident=str(r), label=str(idn_of(r)), config=None)
            for r in resources]


def probe_visa(idn_timeout_ms: int = 300) -> list[DiscoveredDevice]:
    """The built-in VISA provider: every resource listed with its ``*IDN?`` identity.

    Missing pyvisa / empty bus / probe failures are note rows (report, never raise).
    Registered under ``"visa"``; an operator needing a longer identity timeout replaces it:
    ``register_discovery_provider("visa", lambda: probe_visa(idn_timeout_ms=1000))``."""
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


DISCOVERY_PROVIDERS["visa"] = probe_visa


def discover_devices(*, display: bool = True) -> list[DiscoveredDevice]:
    """Scan the buses lab instruments live on and return :class:`DiscoveredDevice` rows.

    Every REGISTERED device class that implements ``discover()`` scans its own bus (the class
    owns the knowledge of how it is found and what its config looks like), then every named
    DISCOVERY PROVIDER scans its class-less bus (VISA built in, each resource queried with
    ``*IDN?``).  A missing driver library, an empty bus, or a crashing scanner is reported
    as a row, never raised.  ``display=True`` also prints the table."""

    discoverers, rows = _registered_discoverers()
    rows = list(rows)                           # import-failure notes lead the table
    seen: set[tuple[str, str]] = set()

    def scan(source: str, fn) -> None:
        """One scanner's rows, deduped -- classes and providers share this exact path."""
        try:
            new_rows = list(fn())
        except Exception as exc:                # one broken scanner must not break the scan
            new_rows = [discovery_note(source, f"discover failed: {exc}")]
        for row in new_rows:
            key = (row.kind, row.ident)         # two registry names for one class scan once
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)

    for _name, cls in discoverers:
        scan(cls.__name__, cls.discover)
    for name, provider in sorted(DISCOVERY_PROVIDERS.items()):
        scan(name, provider)
    if display:
        for row in rows:
            print(row)
    return rows
