"""Contract: every catalog spec is a `CatalogSpec` and the base ENFORCES the shared rules.

`MeasurementSpec` / `ProcessorSpec` / `TaskSpec` used to be three independent dataclasses that
re-typed `name`/`params`/`metadata`, re-spelled `param()`/`defaults()`, and each carried its OWN
collision lambda in its registry.  `CatalogSpec` (#H3r-F2) owns that ONCE and enforces:

- a non-empty name, a tuple of ParamDecl params, and UNIQUE param keys (duplicates silently
  shadowed each other in every form loop before);
- a mandatory `collision_key()` -- a 4th spec tier that forgets it fails at import.

This guard is the mechanical enforcement so a future spec cannot regress the framework.
"""

from __future__ import annotations

import pytest

import Zou_lab_control.neutral_atom as na
from Zou_lab_control.neutral_atom.operations._spec import CatalogSpec

SPECS = (na.MeasurementSpec, na.ProcessorSpec, na.TaskSpec)


def _measurement(**kw):
    base = dict(name="M", params=(), result_labels=("x", "y"), x_key="mx", y_key="my",
                build=lambda **k: None)
    base.update(kw)
    return na.MeasurementSpec(**base)


def _processor(**kw):
    base = dict(name="P", params=(), result_keys=("pa",), run=lambda ctx: {})
    base.update(kw)
    return na.ProcessorSpec(**base)


def _task(**kw):
    base = dict(name="T", build=lambda hub, **k: None, prefix="t_")
    base.update(kw)
    return na.TaskSpec(**base)


@pytest.mark.parametrize("spec_cls", SPECS)
def test_every_spec_subclasses_the_base(spec_cls):
    assert issubclass(spec_cls, CatalogSpec)


@pytest.mark.parametrize("spec_cls", SPECS)
def test_base_methods_and_collision_key_present(spec_cls):
    # param/defaults inherited from the base, collision_key overridden by the subclass
    assert spec_cls.param is CatalogSpec.param
    assert spec_cls.defaults is CatalogSpec.defaults
    assert spec_cls.collision_key is not CatalogSpec.collision_key


@pytest.mark.parametrize("factory", [_measurement, _processor, _task])
def test_collision_key_and_defaults_are_usable(factory):
    decl = na.ParamDecl("g", "G", "int", default=3)
    spec = factory(params=(decl,))
    assert isinstance(spec.collision_key(), tuple) and spec.collision_key()
    assert spec.defaults() == {"g": 3}
    assert spec.param("g") is decl
    with pytest.raises(KeyError):
        spec.param("nope")


@pytest.mark.parametrize("factory", [_measurement, _processor, _task])
def test_empty_name_raises(factory):
    with pytest.raises(ValueError):
        factory(name="")


@pytest.mark.parametrize("factory", [_measurement, _processor, _task])
def test_duplicate_param_keys_raise(factory):
    dup = (na.ParamDecl("k", "K", "int", default=1), na.ParamDecl("k", "K2", "int", default=2))
    with pytest.raises(ValueError):
        factory(params=dup)


def test_task_prefix_is_required():
    # prefix is REQUIRED now (no 'cal_' default) -> a task that omits it fails at construction
    with pytest.raises(TypeError):
        na.TaskSpec(name="T", build=lambda hub, **k: None)
    with pytest.raises(ValueError):
        _task(prefix="")


def test_a_subclass_that_forgets_collision_key_fails_at_import():
    with pytest.raises(TypeError):
        class _BadSpec(CatalogSpec):  # no collision_key override
            pass


def test_params_must_be_paramdecls():
    with pytest.raises(TypeError):
        _measurement(params=("not-a-decl",))
