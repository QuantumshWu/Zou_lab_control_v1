"""The ONE base for every catalog spec (measurement / processor / task).

A "catalog spec" is the dependency-free declarative record the open-registries
discover and the (decoupled) GUI consumes: a `name`, a tuple of `ParamDecl`
`params` (the SINGLE param mechanism every form reuses), free-form `metadata`, and
a `collision_key()` -- the hub keys two specs must not share, so a second spec
cannot silently clobber another's signals on the shared SignalHub.

`MeasurementSpec`, `ProcessorSpec` and `TaskSpec` were three dataclasses that each
re-typed `name`/`params`/`metadata` and re-spelled byte-identical `param()` /
`defaults()`, and each registry carried its OWN collision lambda.  This base owns
all of that ONCE; a subclass declares only its specifics and its `collision_key`.

ENFORCEMENT (so a future 4th spec cannot go wrong):
- `__init_subclass__` raises at import if a subclass forgets to override
  `collision_key` (the de-dup rule is mandatory, not optional);
- `collision_key` is abstract (`NotImplementedError`) -- a spec with no de-dup rule
  cannot run through discovery;
- `__post_init__` validates a non-empty `name`, that `params` is a tuple of
  ParamDecl-shaped objects, and that param keys are UNIQUE (duplicate keys silently
  shadowed each other in every form loop before this);
- `tests/test_spec_base.py` is the mechanical guard.

Kept dependency-free on purpose (no import of `ParamDecl` or any operations module):
`params` are validated by DUCK TYPE (`.key`), so `measurement.py` can import
`CatalogSpec` without an import cycle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar


@dataclass(frozen=True, kw_only=True)
class CatalogSpec:
    """Shared base for the three catalog specs.  Subclasses add their own fields and
    MUST override :meth:`collision_key`.  All construction is keyword-only."""

    name: str
    params: tuple = ()
    metadata: dict = field(default_factory=dict)

    # The sentence appended to a collision error, overridden per subclass.
    collision_advice: ClassVar[str] = "give each entry a unique signal key."

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # A new catalog tier that forgets a de-dup rule fails LOUD at import, not at
        # the first silent signal clobber.
        if cls.collision_key is CatalogSpec.collision_key:
            raise TypeError(
                f"{cls.__name__} must override collision_key() -> tuple "
                "(the hub keys two specs must not share).")

    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.name, str):
            raise ValueError(f"{type(self).__name__} must have a non-empty string name.")
        if not isinstance(self.params, tuple):
            raise TypeError(f"{type(self).__name__} {self.name!r}: params must be a tuple of ParamDecl.")
        keys = [getattr(decl, "key", None) for decl in self.params]
        if any(k is None for k in keys):
            raise TypeError(f"{type(self).__name__} {self.name!r}: every param must be a ParamDecl (with a .key).")
        dups = sorted({k for k in keys if keys.count(k) > 1})
        if dups:
            raise ValueError(
                f"{type(self).__name__} {self.name!r}: duplicate param keys {dups} -- "
                "param keys must be unique (they silently shadow in every form loop).")

    def collision_key(self) -> tuple:
        """The hub keys this spec publishes that must be unique across the catalog.

        ABSTRACT -- every subclass returns its de-dup tuple (measurement: x_key/y_key,
        processor: result_keys, task: prefix)."""

        raise NotImplementedError(f"{type(self).__name__} must implement collision_key().")

    def param(self, key: str):
        """Return the declaration for ``key`` (raises ``KeyError`` if absent)."""

        for decl in self.params:
            if decl.key == key:
                return decl
        raise KeyError(key)

    def defaults(self) -> dict[str, Any]:
        """The declared default value for every parameter, keyed by ``key``."""

        return {decl.key: decl.default for decl in self.params}


__all__ = ["CatalogSpec"]
