"""Neutral-atom timing execution contracts.

``zlc_pulse`` owns pulse authoring, target topology, compilation, and transport
values.  This package owns only neutral-atom execution ports/evidence that
consume those compiled values.  Import concrete contracts from their leaf
modules; the package root remains inert so an ordinary acquisition import does
not load unrelated hardware or readout composition.
"""

__all__: tuple[str, ...] = ()
