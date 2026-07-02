"""Port/bus scanning: find attached instruments and suggest ready-to-use device configs.

The confocal-GUI pattern (its ``_VisaScanWorker``): enumerate every bus a lab instrument could
live on, annotate each hit with an identity string, and NEVER fail because a driver library is
missing -- a missing library is itself a reported row ("pip install ..."), not an exception.
Here the same idea is a plain notebook function instead of a GUI worker:

    >>> import Zou_lab_control.neutral_atom as na
    >>> found = na.discover_devices()          # prints a table, returns the rows
    >>> found[0].config                        # ready config entry for the first Basler camera
    {'type': 'PylonCamera', 'params': {'serial': '24123456', 'trigger_source': 'Software'}}

Two probe tiers per bus (mechanically testable without hardware):

* ``_*_rows(raw)`` -- a PURE function turning the driver's enumeration into
  :class:`DiscoveredDevice` rows (unit-tested with fake enumerations);
* ``_probe_*()`` -- the thin IO shell that imports the driver lazily, calls it, and degrades to
  a note row when the library / bus is absent.

A row's ``config`` is a ready ``{"type": ..., "params": ...}`` entry for ``load_devices`` /
a config JSON -- for device kinds this package has a driver class for (Basler -> PylonCamera).
Buses we can see but have no driver class for (bare VISA instruments) still get listed with
their ``*IDN?`` identity so the operator knows the address to wire into a custom device class
(``register_device_class``), with ``config=None``.
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
                when this package has no driver class for the row (informational listing).
    """

    kind: str
    ident: str
    label: str
    config: dict | None = field(default=None)

    def __str__(self) -> str:  # the table row a notebook prints
        ready = "ready" if self.config is not None else "-"
        return f"[{self.kind:>6}] {self.ident:<28} {self.label}  ({ready})"


# --------------------------------------------------------------------------- pure row builders
def basler_rows(infos) -> list[DiscoveredDevice]:
    """Basler enumeration -> rows.  ``infos`` is any iterable of objects with the pylon
    ``DeviceInfo`` face (``GetModelName`` / ``GetSerialNumber``) -- pure, so tests feed fakes.
    Every hit gets a READY config: a :class:`~.pylon.PylonCamera` pinned to that serial,
    free-running (``Software`` trigger) so the first ``capture()`` shows an image without any
    FPGA wiring; switch to ``Line1`` when the hardware trigger line is connected."""

    rows = []
    for info in infos:
        model = str(info.GetModelName())
        serial = str(info.GetSerialNumber())
        rows.append(DiscoveredDevice(
            kind="basler", ident=serial, label=model,
            config={"type": "PylonCamera",
                    "params": {"serial": serial, "trigger_source": "Software"}}))
    return rows


def visa_rows(resources, idn_of) -> list[DiscoveredDevice]:
    """VISA enumeration -> rows.  ``resources`` is the ``list_resources()`` tuple, ``idn_of``
    maps a resource name to its ``*IDN?`` reply (or a placeholder) -- pure apart from the
    injected query.  No driver class in this package answers a bare VISA instrument, so the
    rows are informational (``config=None``): the operator wires the printed address into a
    custom device class (``register_device_class``)."""

    return [DiscoveredDevice(kind="visa", ident=str(r), label=str(idn_of(r)), config=None)
            for r in resources]


def _note(kind: str, text: str) -> DiscoveredDevice:
    return DiscoveredDevice(kind="note", ident=kind, label=text, config=None)


# --------------------------------------------------------------------------- IO probes
def _probe_basler() -> list[DiscoveredDevice]:
    try:
        from pypylon import pylon
    except ImportError:
        return [_note("basler", "pypylon not installed -- pip install pypylon")]
    try:
        infos = pylon.TlFactory.GetInstance().EnumerateDevices()
    except Exception as exc:                      # runtime present but transport layer unhappy
        return [_note("basler", f"enumerate failed: {exc}")]
    if not infos:
        return [_note("basler", "no Basler camera attached")]
    return basler_rows(infos)


def _probe_visa(idn_timeout_ms: int = 300) -> list[DiscoveredDevice]:
    try:
        import pyvisa
    except ImportError:
        return [_note("visa", "pyvisa not installed -- pip install pyvisa")]
    try:
        rm = pyvisa.ResourceManager()
        resources = rm.list_resources()
    except Exception as exc:
        return [_note("visa", f"list_resources failed: {exc}")]
    if not resources:
        return [_note("visa", "no VISA resources")]

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

    Scans Basler (pypylon) cameras and, when ``visa=True``, VISA resources (each queried with
    ``*IDN?`` under ``idn_timeout_ms``).  A missing driver library or an empty bus is reported
    as a row, never raised.  ``display=True`` also prints the table -- the notebook one-liner:

        >>> found = na.discover_devices()
        >>> cams = [d for d in found if d.config]                 # rows with a ready config
        >>> devices = na.load_devices({"monitor_camera": cams[0].config})
    """

    rows = _probe_basler()
    if visa:
        rows += _probe_visa(idn_timeout_ms)
    if display:
        for row in rows:
            print(row)
    return rows
