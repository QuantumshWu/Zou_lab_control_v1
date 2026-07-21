"""Desktop composition package.

Import contracts from their owning leaf modules.  The package facade exports
nothing until a production composition root gives a contract a real consumer.

Importing the package wires the render layer's domain ports (the composition
call inherited from the deleted legacy frontend root -- see ``_domain_wiring``):
any window composed from here can replay a saved pulse figure and read pulse
templates, while ``zlc_frontend`` itself never imports the pulse compiler.
"""

from . import _domain_wiring as _domain_wiring  # registers the domain ports (lazy bodies)
