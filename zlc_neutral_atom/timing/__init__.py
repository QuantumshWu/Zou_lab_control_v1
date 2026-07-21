"""Neutral-atom timing domain.

Import values from their owning leaf module.  Keeping this package boundary
empty prevents an ordinary capture or pulse import from loading triggered
occupancy and its readout pipeline as an unrelated side effect.

TWO pulse representations live here ON PURPOSE, and their zero cross-references
are a NAMED BOUNDARY, not an accident (adjudicated 2026-07-21, ledger row
"timing 双源裁决"):

* ``pulse.py`` -- the EXECUTION-SESSION PROTOCOL of the target pipeline:
  Prepare/Fire/Complete commands, terminal acks and evidence, device binding.
  It consumes the compiled artifact of the ``zlc_pulse`` IR and never holds an
  authoring table.
* ``pulse_table.py`` + ``sequence_model.py`` -- the AUTHORING model and legacy
  compiler of the CURRENT production pipeline (PulseTableState ->
  RuntimeSequenceProgram -> devices/sequencer), the machine-verified behaviour
  authority (C22) behind the pulse GUI, the task console and every notebook.

The two PIPELINES are the actual dual source, bridged by NOTHING today: no
projection from ``PulseTableState`` to ``PulseDocument`` exists.  The legacy
pipeline is transitional (C25): it dies only when the target pipeline gains
(i) an authoring bridge or replacement covering the table model's facts and
(ii) real-machine verification.  Until then BOTH live, NEITHER may import the
other, and no third pulse representation may enter this package
(guard: ``tests/test_timing_pulse_boundary.py``).
"""

__all__: tuple[str, ...] = ()
