"""Contract: the one-shot DATA-PROCESSING action core (na-side, no GUI).

A processor is the discrete sibling of a measurement: a declared ProcessorSpec
(params in, named data out) whose run() DRIVES existing analysis and returns a
hub-ready dict.  This pins:

1. the built-in per-site fidelity processor is auto-discovered via
   exp.readout.processor_specs(), declares the existing atom 'sites' kind as its
   default view, and its run() DRIVES ReadoutSubsystem.characterize_from_dir
   (trained thresholds land back in the session calibration) returning the declared
   result_keys with the right shapes;
2. running the spec once and publishing the result dict to the hub is a one-line
   pattern in user code (no monolithic node);
3. the registry rejects two processors that publish the same result_keys signal
   (they would clobber each other on the shared SignalHub).

Virtual == real: the only data source is a saved frames folder
(na.write_virtual_run output); switching to hardware changes only who wrote the
frames, not this path.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if sys.path[0] != str(REPO_ROOT):
    sys.path.insert(0, str(REPO_ROOT))

import Zou_lab_control.neutral_atom as na
from Zou_lab_control.neutral_atom.operations.processor import ProcessorContext, ProcessorSpec
from Zou_lab_control.neutral_atom.operations import processor_registry as R


def test_readout_fidelity_processor_drives_characterize(tmp_path):
    data_dir = tmp_path / "run01"
    na.write_virtual_run(str(data_dir), prefix="img", groups=80, shots_per_group=4,
                         short_shot=3, ref_shots=(1, 2, 4), short_exposure=5e-3,
                         reference_exposure=20e-3, grid_shape=(5, 7),
                         loading_probability=0.55, seed=9)

    exp = na.connect("virtual", sitemap={"grid_shape": (5, 7)})
    exp.readout.sitemap_from_dir(str(data_dir), prefix="img", method="psf")

    spec = next(s for s in exp.readout.processor_specs() if s.name == "Readout fidelity")
    # the default view is the EXISTING atom 'sites' kind (no new plot kind)
    assert spec.default_kind == "sites"
    assert spec.default_value_key == "fidelity_site"
    assert "fidelity_site" in spec.result_keys

    params = {**spec.defaults(), "data_dir": str(data_dir), "seed": 2,
              "store_thresholds": True}
    out = spec.run(ProcessorContext(readout=exp.readout, params=params))

    assert set(spec.result_keys) <= set(out)            # every declared key produced
    assert np.asarray(out["fidelity_site"]).shape == (1, 1, 35)
    assert np.asarray(out["fidelity_centers"]).shape == (1, 1, 35, 2)
    assert float(np.asarray(out["aggregate_fidelity"])[0, 0, 0]) > 0.9
    # it DROVE characterize -> trained thresholds written back through the subsystem
    assert exp.readout.current.metadata.get("threshold_method") == "per_site_reference"
    assert np.all(np.isfinite(exp.readout.current.thresholds))


def test_processor_run_once_and_publishes_to_hub(tmp_path):
    """Running a ProcessorSpec is a one-line pattern: build a ProcessorContext, call
    run(ctx) to get the result dict, hub.publish() it.  No monolithic runner --
    the user composes the action exactly like the loading readout (camera +
    detect processor) is composed."""
    from Zou_lab_control.neutral_atom.core.signals import SignalHub

    data_dir = tmp_path / "run02"
    na.write_virtual_run(str(data_dir), prefix="img", groups=60, shots_per_group=4,
                         short_shot=3, ref_shots=(1, 2, 4), grid_shape=(4, 5),
                         loading_probability=0.5, seed=3)
    exp = na.connect("virtual", sitemap={"grid_shape": (4, 5)})
    exp.readout.sitemap_from_dir(str(data_dir), prefix="img", method="psf")
    spec = next(s for s in exp.readout.processor_specs() if s.name == "Readout fidelity")

    hub = SignalHub()
    params = {**spec.defaults(), "data_dir": str(data_dir), "seed": 1,
              "store_thresholds": True}
    ctx = ProcessorContext(readout=exp.readout, params=params,
                           camera=None, sequencer=None, stop=None)
    result = spec.run(ctx)
    hub.publish(result)

    assert np.asarray(hub.latest("fidelity_site")).shape == (1, 1, 20)  # R=1, P=1, N=4x5 sites
    assert hub.schema("fidelity_site").point_shape == (1,)
    assert hub.schema("fidelity_site").data_shape == (20,)
    assert set(spec.result_keys) <= set(hub.names())                     # all declared landed
    assert exp.readout.current.metadata.get("threshold_method") == "per_site_reference"


def test_readout_fidelity_summary_keys_are_single_sourced():
    """SINGLE SOURCE: ``FidelityReport.summary()`` and its declared ``SUMMARY_KEYS`` must agree, and
    the ``readout_fidelity`` processor declares its scalar keys STRAIGHT from that tuple (no hand-typed
    copy that could drift -- a drift would make the processor publish a key it never declared and crash
    at ``_assert_declared`` in the console).  Pin both edges of the single source here."""
    from Zou_lab_control.neutral_atom.operations.fidelity import FidelityReport, characterize_readout
    from Zou_lab_control.neutral_atom.operations.processors.readout_fidelity import readout_fidelity

    # A real FidelityReport via the pure characterization path (cleanly separable bright/dark, no IO):
    rng = np.random.default_rng(0)
    n_groups, n_sites, n_ref = 40, 3, 3
    occupied = rng.integers(0, 2, size=(n_groups, n_sites)).astype(bool)
    short = np.where(occupied, 100.0, 10.0) + rng.normal(0, 1.0, size=(n_groups, n_sites))
    ref = (np.where(occupied[:, None, :], 100.0, 10.0)
           + rng.normal(0, 1.0, size=(n_groups, n_ref, n_sites)))
    report = characterize_readout(short, ref, seed=0)

    # 1) the report's own summary() returns EXACTLY the keys it advertises in SUMMARY_KEYS
    assert set(report.summary()) == set(FidelityReport.SUMMARY_KEYS)
    # 2) the processor's declared summary_keys ARE that tuple (read straight from the report, not retyped)
    spec = readout_fidelity(readout=None)
    assert spec.summary_keys == FidelityReport.SUMMARY_KEYS
    assert set(FidelityReport.SUMMARY_KEYS) <= set(spec.result_keys)


def test_short_shot_form_bounds_sit_inside_index_run_domain():
    """CONTRACT (#C5): the readout-fidelity form's ``short_shot`` bounds derive from the ONE
    imageio contract -- ``index_run`` coerces shot indices via ``positive_int`` (so the minimum
    is ``SHOT_INDEX_MIN``, imported, never re-typed) and requires short_shot <= shots_per_group
    (so the form's two upper bounds share one constant).  A form that accepted 0 was a
    guaranteed runtime ValueError."""
    from Zou_lab_control.neutral_atom.core.analysis import positive_int
    from Zou_lab_control.neutral_atom.operations.imageio import SHOT_INDEX_MIN
    from Zou_lab_control.neutral_atom.operations.processors.readout_fidelity import readout_fidelity

    spec = readout_fidelity(readout=None)
    decls = {p.key: p for p in spec.params}
    short, group = decls["short_shot"], decls["shots_per_group"]
    assert short.lo == SHOT_INDEX_MIN                      # the imported contract minimum...
    positive_int(int(short.lo), "short_shot")              # ...is accepted by index_run's coercion
    with pytest.raises(ValueError):
        positive_int(int(short.lo) - 1, "short_shot")      # ...and is the SMALLEST accepted value
    assert short.hi == group.hi                            # short_shot <= shots_per_group: one shared hi
    assert short.lo <= short.default <= short.hi
    assert group.lo <= group.default <= group.hi


def test_processor_registry_rejects_duplicate_result_keys():
    a = lambda readout: ProcessorSpec(name="A", params=(), run=lambda c: {}, result_keys=("dup",))
    b = lambda readout: ProcessorSpec(name="B", params=(), run=lambda c: {}, result_keys=("dup",))
    R.register_processor(a)
    R.register_processor(b)
    try:
        with pytest.raises(ValueError):
            R.discovered_processor_specs(object())
    finally:
        R.unregister_processor(a)
        R.unregister_processor(b)
