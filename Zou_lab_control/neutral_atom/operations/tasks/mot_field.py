"""Optimize the MOT coil field -- sweep three coil DACs, watch the monitor camera, report the
optimum.

The one-click flow the notebook / console runs: for every (Bx, By, Bz) point of a 3-D coil grid
the task resolves the pulse template's three DAC api slots, fires the sequence through the one
arm-before-fire helper (``triggered_frames``) and measures the MOT fluorescence on the MONITOR camera through the
SAME :func:`~..processors.mot_intensity.mot_roi_intensity` primitive the live "MOT intensity"
processor uses -- so the optimiser and the live view can never disagree.  The optimum is the
argmax refined by a local centre-of-mass; the report (a facet grid of the Bz planes + the raw
intensity block) lands in ``folder`` through the registered frontend plotter (this layer never
imports frontend.*).

For a MANUAL sweep with live plots, run the generic *Pulse-template scan* measurement instead
with ``camera = monitor_camera`` and y = the "MOT intensity" processor's ``mot_intensity``; the
grid facet display then shows the same block live.  This task is the batch/one-click sibling.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from Zou_lab_control._paths import resolve_under_project
from Zou_lab_control._viewer_registry import active_plotter

from ...devices.base import CameraDevice
from ..logic import Task, TaskOutput
from ..measurement import ParamDecl, triggered_frames
from ..task import TaskSpec
from ..task_registry import task
from ..measurements.pulse_scan import _resolve_probe_template
from ..processors.mot_intensity import mot_roi_intensity

DEFAULT_MOT_TEMPLATE = "pulses/mot_field_template.json"


class OptimizeMotFieldTask(Task):
    """Scan the three coil-field DAC api slots over a 3-D grid, measure the MOT spot intensity per
    point on the monitor camera, and report the optimum field (argmax + local centre-of-mass)."""

    # The mid-run signals the console shows LIVE while the task runs: ``grid`` (the accumulating 3-D
    # intensity block, one (Bx,By) map per Bz plane, filling in point-by-point) is the primary panel;
    # ``frame`` / ``progress`` / ``stage`` back the banner.  ``grid`` is FIRST so a console that shows a
    # single mid-run panel shows the live 3-D scan grid, not just the current camera frame.
    mid_run = ("grid", "frame", "progress", "stage")

    def __init__(self, hub, camera: CameraDevice, sequencer, *,
                 template: str = DEFAULT_MOT_TEMPLATE,
                 center_x: float = 0.0, center_y: float = 0.0, center_z: float = 0.0,
                 span: float = 12.0, points: int = 7,
                 roi_cx: float = 0.0, roi_cy: float = 0.0, roi_radius: float = 8.0,
                 folder: str = "mot_field", prefix: str = "mot_"):
        super().__init__(hub, prefix=prefix)
        self.camera = camera
        self.sequencer = sequencer
        self.template = str(template)
        self.centers = (float(center_x), float(center_y), float(center_z))
        self.span = float(span)
        self.points = max(2, int(points))
        self.roi_cx, self.roi_cy, self.roi_radius = float(roi_cx), float(roi_cy), float(roi_radius)
        self.folder = str(folder)
        # The 3-D scan shape (nx, ny, nz) is DETERMINISTIC from the sweep params (each axis' distinct
        # integer DAC codes), so the console can lay out the live facet grid the moment the task starts,
        # BEFORE run() fires a point.  Same rule run() uses (``_axis_codes``), so panel shape == data shape.
        self.grid_shape = tuple(len(self._axis_codes(c)) for c in self.centers)
        # Self-describe the scan axes (the SAME contract PulseScanMeasurement exposes: ``scan_names`` +
        # ``scan_arrays``) so the console's facet grid titles each cell with its real coil coordinate
        # (``Bz=<code>``) instead of ``pt k``.  The axis NAME + per-cell COORDINATE are metadata of this
        # scan node -- independent of the panel's value expression -- and are DETERMINISTIC from the sweep
        # params, so they are available BEFORE ``run()`` fires a point (when the console lays out the grid).
        self.scan_names = ["Bx", "By", "Bz"]
        self.scan_arrays = [self._axis_codes(c) for c in self.centers]

    # ------------------------------------------------------------------ pieces
    def _coil_slots(self, state) -> list[str]:
        """The template's DAC api slots, in declaration order -- the three coil handles."""
        names = [s.name for s in state.api_slots if s.kind == "dac"]
        if len(names) != 3:
            raise ValueError(
                f"the MOT template must expose exactly THREE dac api slots (the x/y/z coils); "
                f"{self.template!r} exposes {names or 'none'}.  Tag the three coil buses as api "
                "slots in the pulse GUI (a-state on the DAC cell) and save the template.")
        return names

    def _axis_codes(self, center: float) -> np.ndarray:
        """One coil axis' scan values: ``points`` integer DAC codes across center +- span."""
        vals = np.round(np.linspace(center - self.span, center + self.span, self.points))
        return np.unique(vals.astype(int))

    @staticmethod
    def refine_optimum(block: np.ndarray, axes: Sequence[np.ndarray]) -> tuple[tuple[float, ...], float]:
        """Argmax refined by the centre-of-mass of the 3^n neighbourhood (above the local floor) --
        sub-step accuracy without fitting, robust to shot noise on a coarse grid."""
        block = np.asarray(block, dtype=float)
        idx = np.unravel_index(int(np.nanargmax(block)), block.shape)
        lo = [max(0, i - 1) for i in idx]
        hi = [min(n, i + 2) for i, n in zip(idx, block.shape)]
        region = block[tuple(slice(a, b) for a, b in zip(lo, hi))]
        w = np.clip(region - float(np.nanmin(region)), 0.0, None)
        total = float(np.nansum(w))
        best = []
        for d, ax in enumerate(axes):
            coords = np.asarray(ax, dtype=float)[lo[d]:hi[d]]
            shape = [1] * region.ndim
            shape[d] = region.shape[d]
            if total > 0:
                best.append(float(np.nansum(w * coords.reshape(shape)) / total))
            else:
                best.append(float(np.asarray(ax)[idx[d]]))
        return tuple(best), float(block[idx])

    # ------------------------------------------------------------------ the flow
    def run(self, out: TaskOutput) -> dict:
        state = _resolve_probe_template(self.template, sequencer=self.sequencer)
        slots = self._coil_slots(state)
        axes = [self._axis_codes(c) for c in self.centers]
        nx, ny, nz = (len(a) for a in axes)
        n_total = nx * ny * nz
        block = np.full((nx, ny, nz), np.nan)

        # The LIVE 3-D grid: the console binds a facet grid to this (repeat=1, n_points, 1) block +
        # points_shape=(nx,ny,nz) (see ``grid_shape``), facets the Bz axis -> one (Bx,By) map per plane
        # that fills in point-by-point.  Publish the all-NaN block ONCE up front so the empty grid shows
        # immediately, then re-publish after every point (a copy -- the reader must not see it mutate).
        out.publish(progress=0.0, stage=f"sweeping {nx}x{ny}x{nz} coil grid on {slots}",
                    grid=block.reshape(1, -1, 1).copy())
        done = 0
        for i, bx in enumerate(axes[0]):
            for j, by in enumerate(axes[1]):
                for k, bz in enumerate(axes[2]):
                    if self._stop.is_set():
                        raise RuntimeError("MOT field scan stopped.")
                    resolved = state.with_api_resolved(
                        {slots[0]: int(bx), slots[1]: int(by), slots[2]: int(bz)})
                    sequence = resolved.to_sequence()
                    frames = triggered_frames(self.camera, self.sequencer, sequence, 1, stop=self._stop)
                    if not frames:
                        continue                      # stopped mid-wait / no trigger
                    frame = frames[0]
                    h, w = frame.shape[-2:]
                    cx = self.roi_cx if self.roi_cx > 0 else w / 2.0
                    cy = self.roi_cy if self.roi_cy > 0 else h / 2.0
                    block[i, j, k] = mot_roi_intensity(frame, cx, cy, self.roi_radius)
                    done += 1
                    out.publish(frame=np.asarray(frame, dtype=float),
                                progress=done / n_total,
                                stage=f"({bx}, {by}, {bz}) -> {block[i, j, k]:.1f}",
                                grid=block.reshape(1, -1, 1).copy())     # the live 3-D grid, one point fuller

        if not np.isfinite(block).any():
            raise RuntimeError("the coil sweep produced no frames -- is the sequencer firing?")
        (best_x, best_y, best_z), best_i = self.refine_optimum(block, axes)
        out.publish(progress=1.0,
                    stage=f"optimum B = ({best_x:.1f}, {best_y:.1f}, {best_z:.1f}), I = {best_i:.1f}")
        report_dir = self._write_report(block, axes, (best_x, best_y, best_z))
        return {
            "best_x": best_x, "best_y": best_y, "best_z": best_z,
            "best_intensity": best_i,
            "n_points": float(n_total),
            "report_dir": str(report_dir),
        }

    def _write_report(self, block: np.ndarray, axes: Sequence[np.ndarray], best) -> Path:
        """Save the raw block (npz) + a facet-grid figure (one Bz plane per cell) through the
        registered frontend plotter -- this layer draws nothing itself."""
        folder = resolve_under_project(self.folder)
        folder.mkdir(parents=True, exist_ok=True)
        np.savez(folder / "mot_field_scan.npz",
                 intensity=block, bx=axes[0], by=axes[1], bz=axes[2],
                 best=np.asarray(best, dtype=float))
        plotter = active_plotter()
        if plotter is not None and hasattr(plotter, "grid"):
            try:
                # facet the Bz axis: one (Bx, By) intensity map per plane -- the SAME grid facet
                # (frontend.grid) a console panel shows live during the scan (block is (bx, by, bz)).
                g = plotter.grid([block[:, :, k].T for k in range(block.shape[2])],
                                 sub_plot_kind="2d", display=False,
                                 title="MOT intensity vs coil field (one Bz plane per cell)")
                # Save through the grid's OWN sealed save (frontend-owned savefig + matching .npz) --
                # this layer imports no matplotlib and touches no figure/canvas itself.  Passing the
                # .png path lets resolve_save_base keep the exact stem, so mot_field_planes.png (and a
                # faithful-reload mot_field_planes.npz) land beside the raw block.
                g.save(str(folder / "mot_field_planes.png"), dpi=200)
            except Exception as exc:              # a plotter-less run keeps the npz, but a REAL figure
                import warnings                    # break must SURFACE (it was silently swallowed before,
                warnings.warn(                     # which hid a wrong plotter.plot(kind="grid") call).
                    f"MOT field report figure not written: {exc!r}", RuntimeWarning, stacklevel=2)
        return folder


MOT_FIELD_PARAMS = (
    ParamDecl("template", "Pulse template", "path", default=DEFAULT_MOT_TEMPLATE,
              path_mode="file", base_dir="pulses",
              file_filter="Pulse program (*.json);;All files (*)",
              tooltip="The pulse fired per point.  It must expose exactly THREE dac api slots "
                      "(the x/y/z coil buses) -- the bundled mot_field_template.json does."),
    ParamDecl("center_x", "Bx centre (code)", "float", default=0.0,
              tooltip="Coil-x sweep centre, signed DAC code (0 = 0 V)."),
    ParamDecl("center_y", "By centre (code)", "float", default=0.0,
              tooltip="Coil-y sweep centre, signed DAC code."),
    ParamDecl("center_z", "Bz centre (code)", "float", default=0.0,
              tooltip="Coil-z sweep centre, signed DAC code."),
    ParamDecl("span", "Span (+- code)", "float", default=12.0,
              tooltip="Half-width of each axis' sweep, in DAC codes."),
    ParamDecl("points", "Points per axis", "int", default=7, lo=2, hi=15,
              tooltip="Grid points per axis (total shots = points^3)."),
    ParamDecl("roi_cx", "ROI centre x (px)", "float", default=0.0,
              tooltip="MOT spot centre on the monitor frame (0 = frame centre)."),
    ParamDecl("roi_cy", "ROI centre y (px)", "float", default=0.0,
              tooltip="MOT spot centre (0 = frame centre)."),
    ParamDecl("roi_radius", "ROI radius (px)", "float", default=8.0,
              tooltip="Circular ROI; the 1x..2x annulus is the subtracted local background."),
    ParamDecl("folder", "Report folder", "path", default="mot_field", path_mode="dir",
              tooltip="Where the intensity block (npz) + the Bz-plane figure land."),
)


@task(order=20, devices=[("camera", {"default": "monitor_camera",
                                     "tooltip": "Which camera watches the MOT fluorescence."})])
def optimize_mot_field(readout) -> TaskSpec:
    """Sweep the x/y/z coil DACs, measure the MOT on the monitor camera, report the optimum.

    The camera is a declared device ROLE (``devices=[("camera", {"default": "monitor_camera"})]``):
    the base appends its dropdown (defaulting to the MOT viewer when the config has one, else the
    conventional camera) and injects the RESOLVED device into ``build`` -- the SAME declarative
    selection every device-using node uses, so a new device domain needs no edit here."""

    s = readout.session

    def build(hub, *, prefix: str = "mot_", camera=None, **values):
        return OptimizeMotFieldTask(hub, camera, s.devices.sequencer, prefix=prefix, **values)

    return TaskSpec(name="Optimize MOT field", build=build, params=MOT_FIELD_PARAMS,
                    mid_run_key="grid", default_kind="grid", prefix="mot_")
