"""Desktop composition package.

Package import is deliberately inert.  Concrete application roots inject every
domain capability into the window they compose; importing ``zlc_workbench``
never mutates a process-global frontend registry.
"""
