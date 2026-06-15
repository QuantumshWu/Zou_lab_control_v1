"""Auto-discovery + open registry for the measurement catalog.

The catalog (``exp.readout.measurement_specs()``) is no longer a hardcoded list:
it is assembled by ``operations.measurement_registry`` from every
``@measurement`` factory in ``operations/measurements/`` plus anything a notebook
registered.  These contract tests guard the user-facing promise: the built-ins
are auto-discovered, a runtime registration appears, names are unique (last
wins), and -- the headline -- DROPPING A NEW MODULE into the package makes a new
measurement appear with no edit to any catalog list.
"""

from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
if sys.path[0] != str(REPO_ROOT):
    sys.path.insert(0, str(REPO_ROOT))

import Zou_lab_control.neutral_atom as na
from Zou_lab_control.neutral_atom.operations import measurement_registry as mr


@pytest.fixture
def exp():
    e = na.connect("virtual", sitemap={"grid_shape": (5, 7)})
    try:
        yield e
    finally:
        e.close()


def test_builtins_are_autodiscovered_in_order(exp):
    names = [s.name for s in exp.readout.measurement_specs()]
    # Both built-ins come from operations/measurements/*.py (NOT a hardcoded list
    # in readout.py); order is deterministic (built-ins use small @measurement
    # order values, temperature before readout-duration).
    assert names[:2] == ["Temperature (release-recapture)", "Readout duration -> fidelity"]


def test_register_makes_a_new_measurement_appear_then_unregister_removes(exp):
    @na.measurement
    def _demo_runtime(readout):
        return na.MeasurementSpec(
            name="Demo runtime scan",
            params=(na.ParamDecl("n", "N", "int", default=5),),
            result_labels=("x", "y"), x_key="demo_x", y_key="demo_y",
            build=lambda **kw: None,
        )

    try:
        names = [s.name for s in exp.readout.measurement_specs()]
        assert "Demo runtime scan" in names
        # ad-hoc measurements sort AFTER the built-ins (default order)
        assert names.index("Demo runtime scan") > names.index("Temperature (release-recapture)")
    finally:
        assert mr.unregister_measurement(_demo_runtime) is True
    assert "Demo runtime scan" not in [s.name for s in exp.readout.measurement_specs()]


def test_duplicate_name_last_registration_wins(exp):
    def first(readout):
        return na.MeasurementSpec(name="Clash", params=(), result_labels=("x", "y"),
                                  x_key="a_x", y_key="a_y", build=lambda **kw: None)

    def second(readout):
        return na.MeasurementSpec(name="Clash", params=(), result_labels=("x", "y"),
                                  x_key="b_x", y_key="b_y", build=lambda **kw: None)

    na.register_measurement(first)
    na.register_measurement(second)
    try:
        specs = exp.readout.measurement_specs()
        clash = [s for s in specs if s.name == "Clash"]
        assert len(clash) == 1               # one slot, not two
        assert clash[0].x_key == "b_x"       # later registration wins
    finally:
        na.unregister_measurement(first)
        na.unregister_measurement(second)


def test_dropping_a_module_in_the_package_is_autodiscovered(exp):
    """The headline promise: a NEW file in operations/measurements/ is found
    automatically -- no catalog list to edit."""

    # NOTE: not "_dropped..." -- the discovery deliberately skips underscore
    # modules (so a package-local _helpers.py is never treated as a measurement).
    pkg_dir = REPO_ROOT / "Zou_lab_control" / "neutral_atom" / "operations" / "measurements"
    new_file = pkg_dir / "zz_dropped_for_test.py"
    mod_name = "Zou_lab_control.neutral_atom.operations.measurements.zz_dropped_for_test"
    new_file.write_text(
        "from ..measurement import MeasurementSpec, ParamDecl\n"
        "from ..measurement_registry import measurement\n"
        "@measurement\n"
        "def dropped_measurement(readout):\n"
        "    return MeasurementSpec(name='Dropped-in scan', params=(ParamDecl('k','K','int',default=1),),\n"
        "                           result_labels=('x','y'), x_key='drop_x', y_key='drop_y', build=lambda **kw: None)\n",
        encoding="utf-8",
    )
    dropped_factory = None
    try:
        # force a fresh discovery pass so the just-written file is imported.
        # invalidate_caches() is a TEST artifact: the import system caches a
        # package directory's listing per-process, but a real user drops a file
        # and RESTARTS task_console (a fresh process), so production never needs it.
        import importlib
        importlib.invalidate_caches()
        mr._DISCOVERED = False
        names = [s.name for s in exp.readout.measurement_specs()]
        assert "Dropped-in scan" in names
        import importlib
        dropped_factory = importlib.import_module(mod_name).dropped_measurement
    finally:
        if dropped_factory is not None:
            mr.unregister_measurement(dropped_factory)
        sys.modules.pop(mod_name, None)
        new_file.unlink(missing_ok=True)
        mr._DISCOVERED = False  # re-import only the surviving (real) modules next call

    assert "Dropped-in scan" not in [s.name for s in exp.readout.measurement_specs()]
