"""DeviceManager desktop product package.

Import the explicit :mod:`.app`, :mod:`.controller`, :mod:`.editor_session`,
or :mod:`.window` leaf that owns the required responsibility.  The package
root is deliberately inert so a headless import of ``zlc_workbench`` products
never loads Qt as a side effect.
"""
