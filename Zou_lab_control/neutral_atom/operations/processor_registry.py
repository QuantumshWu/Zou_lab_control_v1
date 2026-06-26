"""Open registry + auto-discovery for data-processing actions (the catalog source).

Mirrors :mod:`measurement_registry` one tier over.  A processor is contributed as
a FACTORY ``build(readout) -> ProcessorSpec``: it receives the readout subsystem so
its ``run`` closure can capture the session (devices/calibration) and DRIVE the
subsystem's analysis.  Two ways to make one appear in
``exp.readout.processor_specs()`` (and so in the task console's Add-Panel
"Data processing" group):

* drop a module into the ``operations/processors/`` package with a factory
  decorated ``@processor`` -- it is AUTO-DISCOVERED (no hardcoded catalog to edit);
* or call :func:`register_processor` from a notebook for an ad-hoc one.

``discovered_processor_specs(readout)`` imports the package once, builds every
registered spec, de-duplicates by ``name`` (later wins), and FAILS LOUD if two
processors publish the same ``result_keys`` signal -- they would clobber each other
on the shared SignalHub.

The mechanics are the SHARED :class:`._open_registry.OpenRegistry` (identical to the
measurement and task catalogs); this module just binds the processor-named public
API to one instance and injects the ``result_keys`` collision check.  Imports no
concrete backend and never touches the frontend.
"""

from __future__ import annotations

from ._open_registry import DEFAULT_ORDER, OpenRegistry

_REGISTRY = OpenRegistry(noun="processor", package=f"{__package__}.processors")

register_processor = _REGISTRY.register
processor = _REGISTRY.decorator
unregister_processor = _REGISTRY.unregister
registered_processors = _REGISTRY.registered
discovered_processor_specs = _REGISTRY.discovered_specs


__all__ = [
    "DEFAULT_ORDER",
    "discovered_processor_specs",
    "processor",
    "register_processor",
    "registered_processors",
    "unregister_processor",
]
