"""Auto-discovered measurement catalog.

Every module here defines a ``@measurement``-decorated factory
``build(readout) -> MeasurementSpec``.  The package is imported module-by-module
by :func:`operations.measurement_registry.discovered_specs` (via
``exp.readout.measurement_specs()``), so DROPPING A NEW FILE here makes a new
measurement appear in the task console's Add-Panel list automatically -- there is
no hardcoded catalog list to edit.  Built-ins (``temperature``,
``readout_duration``) live here too, so a new measurement is exactly as
first-class as they are.
"""
