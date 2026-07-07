"""MECHANICAL guard for a core design principle (AGENTS.md §2).

Virtual (fake-data) testing must run the EXACT same code path as real hardware;
the ONLY thing a virtual backend may fake is the lowest-level data source
(camera frames). That is the whole point of preparing on virtual data: switching
to real hardware must be a one-line `na.connect("virtual", ...)` ->
`na.connect("qcmos", ...)` change, with every analysis line unchanged.

So the analysis / orchestration layer (core/, operations/, subsystems/) must:
  1. depend ONLY on the device CONTRACT (devices/base.py) -- never import a
     concrete camera backend (devices/virtual.py, devices/qcmos.py);
  2. never read simulation ground-truth (the sim's occupancy, the known site
     centers `_site_centers`, the all-sites `render_image`/`force_all_sites`
     flag) -- site centers are DETECTED from images, thresholds LEARNED from
     data, labels derived from reference frames.

If either is violated, swapping virtual->real would NOT be a no-op, and the
tutorial / tests would be validating a path real hardware never runs. This test
fails the build in that case.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

import Zou_lab_control.neutral_atom as na


_NA = Path(na.__file__).resolve().parent
_ANALYSIS_DIRS = [_NA / "core", _NA / "operations", _NA / "subsystems"]
# session.py is the orchestration facade: it picks a backend by NAME via the
# registry but must not import a concrete backend or read its internal fields.
_ANALYSIS_FILES = [_NA / "session.py"]

# Backends / rendering the analysis must NOT import.  Frames arrive via the
# CameraDevice contract, so a concrete camera backend (``virtual`` / ``qcmos`` /
# the ``pylon`` Basler SDK) is forbidden; devices/base.py (the abstract contract)
# is allowed.  The analysis layer must ALSO never import the GUI backend
# (``frontend`` / ``PyQt5``) or ``matplotlib`` -- all plotting is routed through
# the dependency-free ``_viewer_registry`` seam (``neutral_atom/views`` is the ONE
# adapter layer where matplotlib lives, and it is not scanned here).  A concrete
# backend named ``virtual``/``qcmos`` or a ``pylon``/``frontend``/``PyQt5``/
# ``matplotlib`` import anywhere in the analysis tree means virtual<->real is no
# longer a drop-in swap (or the analysis grew a rendering dependency it must not
# own).  The sequencer concrete modules are not named any of these so are
# unaffected.
_BACKEND_IMPORT = re.compile(
    r"^\s*(?:from|import)\b.*\b(?:virtual|qcmos|pylon|frontend|PyQt5|matplotlib)\b", re.M)
# Simulation ground-truth the analysis must NOT read: the sim's known occupancy /
# per-atom brightness+lifetime knobs / the all-sites render + known site centers /
# the manual-occupancy mutator.  Attribute-access / call forms only (a leading
# ``.`` or a bare ``force_all_sites`` flag), so a docstring word like
# "occupancy/counts" never false-positives AND the legitimate DETECTION function
# ``find_site_centers`` (a bare name, no leading ``.``) is never caught -- centers
# are DETECTED from images, not read from the sim's ``._site_centers``/``site_centers``.
_GROUND_TRUTH = re.compile(
    r"\.(?:occupancy|render_image|_site_centers|site_centers"
    r"|detection_lifetime|atom_rate|set_occupancy)\b"
    r"|\bforce_all_sites\b")


def _analysis_files():
    for d in _ANALYSIS_DIRS:
        # rglob (RECURSIVE): the fast-growing operations/tasks, operations/processors
        # and operations/measurements subpackages are analysis too -- a non-recursive
        # glob would silently skip them and let a backend/matplotlib import slip in.
        yield from sorted(d.rglob("*.py"))
    for f in _ANALYSIS_FILES:
        if f.exists():
            yield f


def test_analysis_layer_never_imports_a_concrete_camera_backend():
    offenders = []
    for f in _analysis_files():
        src = f.read_text(encoding="utf-8")
        for m in _BACKEND_IMPORT.finditer(src):
            offenders.append(f"{f.relative_to(_NA)}: {m.group(0).strip()}")
    assert offenders == [], (
        "analysis layer imports a concrete camera backend -- it must use ONLY the "
        f"devices/base.py contract so virtual<->real is a drop-in swap: {offenders}"
    )


def test_analysis_layer_never_reads_simulation_ground_truth():
    offenders = []
    for f in _analysis_files():
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if _GROUND_TRUTH.search(line):
                offenders.append(f"{f.relative_to(_NA)}:{i}: {line.strip()}")
    assert offenders == [], (
        "analysis layer reads simulation ground-truth (sim occupancy / known site "
        f"centers / all-sites render flag) instead of computing from images: {offenders}"
    )


def test_virtual_readout_runs_the_real_contract_path_end_to_end():
    """On the virtual backend the real flow runs verbatim: sites DETECTED from the
    image, thresholds LEARNED from data, occupancy detected -- all via the camera
    contract, no ground truth. Swapping the backend would change none of this."""
    exp = na.connect("virtual", sitemap={"grid_shape": (3, 4), "spacing_px": 12.0})

    sm = exp.readout.sitemap(method="psf", frames=6, display=False)
    # centers were DETECTED from the rendered all-sites image (count == grid),
    # and a PSF weight was fit from the data -- not handed in by the simulator.
    assert sm.calibration.centers.shape == (12, 2)
    assert sm.calibration.psf_weights is not None and sm.calibration.psf_weights.shape[0] == 12

    th = exp.readout.thresholds(frames=40, display=False)          # learned from data
    assert th.calibration.metadata.get("thresholds_calibrated") is True

    shot = exp.readout.detect(display=False)                       # uses learned calibration
    assert shot.occupied.shape == (12,)
    assert np.asarray(shot.counts).shape == (12,)


def test_virtual_sequencer_rejects_out_of_catalog_channels_like_real():
    """virtual == real on the CHANNEL contract: ``VirtualSequencer.prepare`` compiles against its
    FIXED device catalog and REJECTS a template using channels outside it -- exactly the rejection
    the real ``SequencerService`` gives -- instead of the old accommodation that silently widened
    the catalog to whatever the program used (which let a virtual run "succeed" on a template real
    hardware would refuse).  A subset (an in-catalog channel) still prepares normally, and neither
    prepare mutates the catalog."""
    from Zou_lab_control.neutral_atom.devices.virtual import VirtualSequencer
    from Zou_lab_control.neutral_atom.timing import PulseSequence

    seqr = VirtualSequencer(channels=["emCCD", "probe"])
    seqr.prepare(PulseSequence(name="ok").pulse("emCCD", 0.0, 1e-3))   # subset -> prepares, no raise
    with pytest.raises(ValueError, match="outside this sequencer's catalog"):
        seqr.prepare(PulseSequence(name="bad").pulse("not_a_channel", 0.0, 1e-3))
    # The catalog stayed FIXED through both prepares -- never widened to fit the program.
    assert tuple(seqr.channels) == ("emCCD", "probe")
    assert tuple(seqr.service.channels) == ("emCCD", "probe")
