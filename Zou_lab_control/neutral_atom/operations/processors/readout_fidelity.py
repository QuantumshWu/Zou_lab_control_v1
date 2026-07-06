"""Per-site readout-fidelity characterization as a one-shot data-processing action.

This is the user's Jupyter flow -- grab/read grouped frames, characterize per-site
readout fidelity, get a pile of numbers + per-site arrays -- surfaced as a
:class:`ProcessorSpec` so the task console can run it from a panel and publish its
results to the shared SignalHub.  ``run`` DRIVES
``ReadoutSubsystem.characterize_from_dir`` (which itself extracts per-site signals
through the current calibration's box/PSF method and runs the held-out
characterization in ``operations.fidelity``); it re-implements NO math and reads no
simulation ground truth -- the only data source is a saved frames folder, so a
virtual run (``na.write_virtual_run`` output) and a real run traverse the identical
path, differing only in who wrote the frames.
"""

from __future__ import annotations

import numpy as np

from ..fidelity import FidelityReport
from ..imageio import DEFAULT_SHORT_SHOT, DEFAULT_SHOTS_PER_GROUP, SHOT_INDEX_MIN
from ..processor import ParamDecl, ProcessorContext, ProcessorSpec
from ..processor_registry import processor

# The form's upper bound on the group size.  ``short_shot`` must land in
# SHOT_INDEX_MIN..shots_per_group (imageio.index_run's contract), so its hi is THIS
# same bound -- one constant for both decls, never a re-typed near-miss like 63/64.
_MAX_SHOTS_PER_GROUP = 64


@processor(order=10)
def readout_fidelity(readout) -> ProcessorSpec:
    """Characterize per-site readout fidelity from a saved frames folder.

    Publishes per-site arrays (``fidelity_site``, ``fidelity_threshold``) for a
    ``sites`` map and the scalar summary (aggregate / mean / min fidelity, ...) for
    the panel's numeric pane.  Writes the trained thresholds back into the session
    calibration when ``store_thresholds`` is set."""

    params = (
        ParamDecl("data_dir", "Frames folder", "path", default="", path_mode="dir",
                  required=True, tooltip="Folder of saved frames (na.write_virtual_run output, or a real run)."),
        ParamDecl("prefix", "Frame prefix", "text", default="img"),
        ParamDecl("shots_per_group", "Shots/group", "int", default=DEFAULT_SHOTS_PER_GROUP, lo=2,
                  hi=_MAX_SHOTS_PER_GROUP),
        ParamDecl("short_shot", "Short-shot index", "int", default=DEFAULT_SHORT_SHOT,
                  lo=SHOT_INDEX_MIN, hi=_MAX_SHOTS_PER_GROUP,
                  tooltip="1-based index of the short readout within each group (1..shots_per_group)."),
        ParamDecl("train_fraction", "Train fraction", "float", default=0.9, lo=0.5, hi=0.99),
        ParamDecl("seed", "Seed", "int", default=0, lo=0, hi=1_000_000),
        ParamDecl("store_thresholds", "Write thresholds back", "bool", default=True),
    )

    def run(ctx: ProcessorContext) -> dict:
        p = ctx.params
        # ``readout`` is captured from the factory (like a measurement's build
        # closure captures the session), so the console can stay decoupled -- it
        # drives the action through the spec without holding the subsystem.
        report = readout.characterize_from_dir(
            str(p["data_dir"]),
            prefix=str(p.get("prefix", "img")),
            shots_per_group=int(p["shots_per_group"]),
            short_shot=int(p["short_shot"]),
            train_fraction=float(p["train_fraction"]),
            seed=int(p["seed"]),
            store_thresholds=bool(p["store_thresholds"]),
            save=False,
        )
        out: dict = {
            "fidelity_site": np.asarray(report.site_fidelities, dtype=float),
            "fidelity_threshold": np.asarray(report.thresholds, dtype=float),
        }
        # The site centers (N, 2) so the default 'sites' atom map can place its
        # circles standalone (no live logic node needed): read from the calibration the
        # characterization just used/updated -- not recomputed here.  Published under a
        # processor-UNIQUE key (the live 'Judge occupancy' processor owns plain
        # ``centers``; two processors may not publish the same hub signal).
        cal = readout.current
        if cal is not None:
            out["fidelity_centers"] = np.asarray(cal.centers, dtype=float)
        # report.summary() is all scalars -> the numeric pane (single source: the
        # report owns these names, we just republish them).
        out.update({str(k): float(v) for k, v in report.summary().items()})
        return out

    # SINGLE SOURCE: the report owns its scalar key names; declare them straight from
    # FidelityReport.SUMMARY_KEYS so the spec's declaration can never drift from what
    # report.summary() actually publishes (run() above republishes summary() verbatim).
    summary_keys = FidelityReport.SUMMARY_KEYS
    return ProcessorSpec(
        name="Readout fidelity",
        params=params,
        run=run,
        result_keys=("fidelity_site", "fidelity_threshold", "fidelity_centers") + summary_keys,
        summary_keys=summary_keys,
        default_kind="sites",            # per-site fidelity map (the existing atom kind)
        default_value_key="fidelity_site",
        metadata={"reads_frames": "saved_dir"},
    )
