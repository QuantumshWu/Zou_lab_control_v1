"""Open registry + auto-discovery for measurements (the catalog single source).

A measurement is contributed as a FACTORY ``build(readout) -> MeasurementSpec``:
it receives the readout subsystem so its ``build`` closure can capture the
session (devices/calibration) and reuse the subsystem's scan builders.  Two ways
to make one appear in ``exp.readout.measurement_specs()`` (and therefore in the
task console's Add-Panel list):

* drop a module into the ``operations/measurements/`` package with a factory
  decorated ``@measurement`` -- it is AUTO-DISCOVERED (no hardcoded catalog list
  to edit); the built-ins live there too, so a new measurement is exactly as
  first-class as ``temperature``;
* or call :func:`register_measurement` from a notebook for an ad-hoc one.

``discovered_specs(readout)`` imports the package once, then calls every
registered factory with ``readout`` and returns the freshly-built specs ordered
by ``order`` (built-ins first) then registration, de-duplicated by ``name``
(later wins, so a notebook can override a built-in by re-using its name).

The mechanics (register/decorator/unregister/ordering/auto-discovery/dedup) are
the SHARED :class:`._open_registry.OpenRegistry`, identical to the processor and
task catalogs; this module just binds the measurement-named public API to one
instance and injects the measurement-specific collision check (no two specs may
publish the same ``x_key``/``y_key`` on the shared SignalHub).  No concrete
backend is imported and the frontend is never touched.
"""

from __future__ import annotations

from ._open_registry import DEFAULT_ORDER, OpenRegistry

_REGISTRY = OpenRegistry(noun="measurement", package=f"{__package__}.measurements")

register_measurement = _REGISTRY.register
measurement = _REGISTRY.decorator
unregister_measurement = _REGISTRY.unregister
registered_measurements = _REGISTRY.registered
discovered_specs = _REGISTRY.discovered_specs


__all__ = [
    "DEFAULT_ORDER",
    "discovered_specs",
    "measurement",
    "register_measurement",
    "registered_measurements",
    "unregister_measurement",
]
