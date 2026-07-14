"""Per-site readout-fidelity characterization as a one-shot data-processing action.

This is the user's Jupyter flow -- grab/read grouped frames, characterize per-site
readout fidelity, get a pile of numbers + per-site arrays -- surfaced as a
:class:`ProcessorSpec` so the task console can run it from a panel and publish its
results to the shared SignalHub.  ``run`` DRIVES
``ReadoutSubsystem.characterize_from_dir`` (which itself extracts per-site signals
through the current calibration's box/PSF method and runs the held-out
characterization in ``operations.fidelity``); it re-implements NO math and reads no
simulation ground truth -- the only data source is a saved frames folder, so a
virtual run (``na.simulation.write_virtual_run`` output) and a real run traverse the identical
path, differing only in who wrote the frames.
"""

from __future__ import annotations

import numpy as np

from ..fidelity import FidelityReport
from ...core.params import ParamDecl
from ...core.signal_tensor import SignalSchema, SignalTensor
from ..imageio import DEFAULT_SHORT_SHOT, DEFAULT_SHOTS_PER_GROUP, SHOT_INDEX_MIN
from ..processor import ProcessorContext, ProcessorSpec
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
    the panel's numeric pane.  This panel is a READ-ONLY CHECK by default: it must not
    silently retrain the live readout just because you opened it to look at the numbers
    (the readout pipeline uses that same admitted calibration).  Turning
    ``store_thresholds`` ON overwrites the session calibration's thresholds from this
    folder; changing the live model remains an explicit calibration operation."""

    params = (
        ParamDecl("data_dir", "Frames folder", "path", default="", path_mode="dir",
                  required=True, tooltip="Folder of saved frames (na.simulation.write_virtual_run output, or a real run)."),
        ParamDecl("prefix", "Frame prefix", "text", default="img"),
        ParamDecl("shots_per_group", "Shots/group", "int", default=DEFAULT_SHOTS_PER_GROUP, lo=2,
                  hi=_MAX_SHOTS_PER_GROUP),
        ParamDecl("short_shot", "Short-shot index", "int", default=DEFAULT_SHORT_SHOT,
                  lo=SHOT_INDEX_MIN, hi=_MAX_SHOTS_PER_GROUP,
                  tooltip="1-based index of the short readout within each group (1..shots_per_group)."),
        ParamDecl("train_fraction", "Train fraction", "float", default=0.9, lo=0.5, hi=0.99),
        ParamDecl("seed", "Seed", "int", default=0, lo=0, hi=1_000_000),
        ParamDecl("store_thresholds", "Write thresholds back", "bool", default=False,
                  tooltip="OFF by default: this panel is a read-only fidelity check.  Turn on ONLY to retrain "
                          "and OVERWRITE the session calibration's thresholds from this folder -- it changes "
                          "what the live readout uses."),
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
        site_fidelity = np.asarray(report.site_fidelities, dtype=float).reshape(-1)
        thresholds = np.asarray(report.thresholds, dtype=float).reshape(-1)
        if site_fidelity.shape != thresholds.shape or site_fidelity.size < 1:
            raise ValueError(
                "readout fidelity must return one non-empty threshold per site; "
                f"got fidelity={site_fidelity.shape}, thresholds={thresholds.shape}.")
        n_sites = int(site_fidelity.size)

        def tensor(data, *, point_shape, data_shape, label, unit="") -> SignalTensor:
            array = np.asarray(data, dtype=float)
            schema = SignalSchema(
                point_shape=point_shape,
                data_shape=data_shape,
                dtype=np.float64,
                repeat_capacity=1,
                label=label,
                unit=unit,
            )
            return SignalTensor(array, schema)

        out: dict = {
            "fidelity_site": tensor(
                site_fidelity.reshape(1, 1, n_sites),
                point_shape=(1,), data_shape=(n_sites,), label="site fidelity"),
            "fidelity_threshold": tensor(
                thresholds.reshape(1, 1, n_sites),
                point_shape=(1,), data_shape=(n_sites,),
                label="readout threshold", unit="counts"),
        }
        # The site centers (N, 2) so the default 'sites' atom map can place its
        # circles standalone (no live logic node needed): read from the calibration the
        # characterization just used/updated -- not recomputed here.  Published under a
        # processor-UNIQUE key (an occupancy stream may own plain ``centers``;
        # two processors may not publish the same hub signal).
        cal = readout.current
        if cal is not None:
            centers = np.asarray(cal.centers, dtype=float)
            if centers.shape != (n_sites, 2):
                raise ValueError(
                    f"readout calibration centers must have shape {(n_sites, 2)}, got {centers.shape}.")
            out["fidelity_centers"] = tensor(
                centers.reshape(1, 1, n_sites, 2),
                point_shape=(1,), data_shape=(n_sites, 2),
                label="site centre", unit="px")
        # report.summary() is all scalars -> the numeric pane (single source: the
        # report owns these names, we just republish them).
        out.update({
            str(key): tensor(
                np.asarray(value, dtype=float).reshape(1, 1, 1),
                point_shape=(1,), data_shape=(1,), label=str(key))
            for key, value in report.summary().items()
        })
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
        # Its ONLY data source is the saved frames folder (data_dir): it drives no ctx
        # hardware, so it declares no device roles -- the console hands it None for
        # camera/sequencer, it occupies nothing, and starting it never stops a live node.
        devices=(),
        metadata={"reads_frames": "saved_dir"},
    )
