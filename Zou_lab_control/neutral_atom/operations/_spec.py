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

from dataclasses import dataclass, field, fields as _dc_fields, replace as _dc_replace
from typing import Any, ClassVar

#: Sentinel default for a REQUIRED field on a catalog spec.  ``@dataclass(kw_only=True)`` made
#: these dataclasses keyword-only so a subclass could add required fields AFTER the base's
#: defaulted ``params``/``metadata`` -- but ``kw_only`` needs Python >= 3.10, and the FPGA server
#: runs on an older Python.  Instead every "required" field defaults to ``REQUIRED`` (so field
#: ordering is always valid -- 3.9-compatible) and :meth:`CatalogSpec.__post_init__` raises if one
#: was left unset -- the SAME "you must pass it" contract, no version dependency.
REQUIRED: Any = object()


def _bind_device_args(fn, keys, decls, device_set):
    """Wrap a build-callable so each declared device-role key is popped from the caller's
    ``**values``, resolved to a device INSTANCE via ``device_set[name]``, and passed to
    ``fn`` as that same keyword (``camera=<CameraDevice>``) -- so a spec's ``build`` receives
    the resolved device and never hand-writes ``device_set[name]``.  A blank / missing
    selection falls back to the role param's own default (the conventional device).  Leading
    POSITIONAL args (``hub`` / ``prefix``) pass through untouched -- only the role KWARGS are
    rewritten, so the one wrapper fits every call convention (``build(**values)`` for a
    measurement, ``build(hub, **values)`` for a task/camera_spec)."""

    defaults = {d.key: d.default for d in decls}

    def bound(*args, **values):
        for key in keys:
            name = values.pop(key, None)
            if name in (None, ""):
                name = defaults.get(key)
            values[key] = device_set[name] if name is not None else None
        return fn(*args, **values)

    bound.__wrapped__ = fn          # keep introspection / tests able to reach the original
    return bound


@dataclass(frozen=True)
class CatalogSpec:
    """Shared base for the three catalog specs.  Subclasses add their own fields and
    MUST override :meth:`collision_key`.  Required subclass fields default to :data:`REQUIRED`
    and are enforced in :meth:`__post_init__` (keeps construction keyword-required without the
    Python-3.10-only ``kw_only``)."""

    name: str
    params: tuple = ()
    metadata: dict = field(default_factory=dict)

    # The sentence appended to a collision error, overridden per subclass.
    collision_advice: ClassVar[str] = "give each entry a unique signal key."

    #: The spec ATTRIBUTES whose value is a build-callable taking ``**param_values``.  The
    #: device wrapper is applied to each that is an actual dataclass FIELD (so ``replace`` can
    #: swap it) -- ``MeasurementSpec.build`` (its ``make_node`` method then calls the wrapped
    #: build), ``TaskSpec.build``, ``ProcessorSpec.make_node`` (a field closure).  A one-shot
    #: processor's ``run(ctx)`` is intentionally NOT here: it receives devices via
    #: ``ctx.camera`` / ``ctx.sequencer``, not ``**values``.
    _DEVICE_INJECTABLE: ClassVar[tuple[str, ...]] = ("build", "make_node")

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # A new catalog tier that forgets a de-dup rule fails LOUD at import, not at
        # the first silent signal clobber.
        if cls.collision_key is CatalogSpec.collision_key:
            raise TypeError(
                f"{cls.__name__} must override collision_key() -> tuple "
                "(the hub keys two specs must not share).")

    def __post_init__(self) -> None:
        # A field left at the REQUIRED sentinel was not passed -> raise, exactly like a no-default
        # keyword argument would (the kw_only replacement; runs before any subclass __post_init__).
        missing = [f.name for f in _dc_fields(self) if getattr(self, f.name, None) is REQUIRED]
        if missing:
            raise TypeError(
                f"{type(self).__name__} missing required argument(s): {', '.join(missing)} "
                "(pass them by keyword).")
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

    def with_devices_bound(self, device_set, roles):
        """Return a COPY of this spec with each declared device ``role`` wired ONCE:
        (a) a ``choice`` param appended (its choices = the ``device_set`` devices of that
        role's type, default = the conventional device), and (b) the resolved device
        injected into every build-callable, so the factory's ``build`` receives
        ``camera=<CameraDevice>`` instead of hand-writing ``device_set[name]``.

        This is the SINGLE place device selection is turned into UI + resolution -- the
        registry funnel calls it (where ``device_set`` is in scope) for any spec that declared
        ``devices=[...]``, and an imperative builder (``readout.camera_spec``) calls it directly.
        A spec that declares no roles is returned unchanged.  Only build-callables that are
        real dataclass FIELDS are wrapped (``replace`` can swap those); a method like
        ``MeasurementSpec.make_node`` is left alone -- it calls the wrapped ``build`` itself."""

        from .measurement import device_params_for, normalize_device_roles

        norm = normalize_device_roles(roles)
        if not norm:
            return self
        keys = [key for key, _domain, _opts in norm]
        appended = device_params_for(roles, device_set)
        field_names = {f.name for f in _dc_fields(self)}
        new_fields: dict = {"params": (*self.params, *appended)}
        for attr in self._DEVICE_INJECTABLE:
            if attr in field_names:
                fn = getattr(self, attr, None)
                if callable(fn):
                    new_fields[attr] = _bind_device_args(fn, keys, appended, device_set)
        return _dc_replace(self, **new_fields)


__all__ = ["CatalogSpec", "REQUIRED"]
