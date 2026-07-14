"""Neutral-atom readout domain.

Import values from their owning submodule.  The package root deliberately does
not import calibration analysis or SciPy when a caller only needs camera/readout
contracts.
"""

__all__: tuple[str, ...] = ()
