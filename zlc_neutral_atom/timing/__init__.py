"""Neutral-atom timing domain.

Import values from their owning leaf module.  Keeping this package boundary
empty prevents an ordinary capture or pulse import from loading triggered
occupancy and its readout pipeline as an unrelated side effect.
"""

__all__: tuple[str, ...] = ()
