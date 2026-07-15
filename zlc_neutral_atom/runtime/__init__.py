"""Runtime package.

Import contracts from their owning leaf modules.  The package root deliberately
exports no capability or authority object, so importing an unrelated runtime
domain cannot eagerly construct the whole execution dependency graph.
"""
