"""Neutral-atom timing domain.

Import values from their owning leaf module.  Keeping this package boundary
empty prevents an ordinary capture or pulse import from loading triggered
occupancy and its readout pipeline as an unrelated side effect.

TWO pulse representations live here ON PURPOSE, and their zero cross-references
are a NAMED BOUNDARY, not an accident (adjudicated 2026-07-21, ledger row
"timing 双源裁决"; the transitional death condition was superseded by the same
day's one-shot purge directive -- both sides are PERMANENT residents now):

* ``pulse.py`` -- the EXECUTION-SESSION PROTOCOL of the target pipeline:
  Prepare/Fire/Complete commands, terminal acks and evidence, device binding.
  It consumes the compiled artifact of the ``zlc_pulse`` IR and never holds an
  authoring table.
* ``pulse_table.py`` + ``sequence_model.py`` + ``runtime_compiler.py`` -- the
  AUTHORING model and PRODUCTION compiler (PulseTableState ->
  RuntimeSequenceProgram edge/DAC/affine-scan tables + wire codec), the
  machine-verified behaviour authority (C22) behind the pulse GUI, the task
  console and every notebook.

``pulse_table.py`` is a LEGACY authoring model, not a second sanctioned one.  The
design keeps exactly one authoring contract -- ``schema="zlc_pulse.PulseDocument"``
(SYSTEM_ARCHITECTURE_DESIGN_zh §15.1) -- and names its retirement condition: each
remaining consumer migrates to that document in its own dependency-closed slice,
and the legacy reader dies in the same commit (C25).  Its last consumers today are
``zlc_workbench/pulse_editor/`` and ``zlc_workbench/_domain_wiring.py``.

Nothing bridges the two, and nothing may: the design forbids a
``PulseDocument <-> PulseTableState`` converter by name, because a converter gives
each model a reason to keep existing and turns a scheduled deletion into a
permanent seam.  The way out is migration, not translation.  Meanwhile neither
side may import the other and no third pulse representation may enter this package
(guard: ``tests/test_timing_pulse_boundary.py``).

Retiring ``pulse_table.py`` is gated on one thing beyond the consumer cuts: the
``PulseTableState -> RuntimeSequenceProgram`` chain still produces the wire bytes
that drive the real machine, so the target compiler must be shown byte-equivalent
before the legacy chain is removed.
"""

__all__: tuple[str, ...] = ()
