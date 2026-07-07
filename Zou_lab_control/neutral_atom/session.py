"""Jupyter-first neutral-atom experiment session.

This file is the public shape of the lightweight first milestone.  The lower
layers still exist as device/timing/analysis/verilog boundaries; the notebook
user mostly talks to ``NeutralAtomSession`` and result objects.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence
import json

import numpy as np

from .core.analysis import grid_shape_tuple
from .core.calibration import TrapCalibration
from .core.results import (
    CaptureResult,
    DetectionResult,
    DetectionTimeScanResult,
    MeasurementTaskResult,
    PreflightReport,
    ResultObject,
    SitemapResult,
    ThresholdResult,
)
from .core.utils import html_summary, json_ready
from .devices import CameraDevice, DeviceSet, SequencerDevice, load_devices, resolve_connect_config
from .operations import calibrate_sitemap_from_images, calibrate_threshold_from_images, detect_image
from .timing import PulseSequence, imaging_channel_kwargs, imaging_sequence
from .subsystems import ExperimentSubsystem, ReadoutSubsystem, TimingSubsystem


class NeutralAtomSession:
    """Notebook-facing experiment session.

    The session is deliberately stateful: it owns the currently connected
    devices, the current pulse sequence, the current calibration, and recent
    results.  This keeps notebook cells short while preserving a clean path to
    real hardware and future GUI frontends.
    """

    def __init__(self, devices: DeviceSet, *, name: str = "neutral_atom", defaults: dict[str, Any] | None = None):
        self.devices = devices
        self.name = str(name)
        self.defaults = dict(defaults or {})
        self.sequence = self.build_imaging_sequence(exposure=self.camera_exposure(), load=True)
        self._calibration: TrapCalibration | None = None
        self.history: list[Any] = []
        # Device CONSUMER teardown hooks: callables invoked BEFORE this session's devices are
        # closed (:meth:`close`) or swapped (:meth:`load_config`), so long-lived consumers --
        # the task console's running logic nodes, whose worker threads block inside
        # ``camera.acquire`` and hold camera / RPyC handles -- are stopped FIRST and can never
        # keep a closed/replaced device alive.  Duck-typed callables so the frontend registers
        # ``console.stop_all_nodes`` here without the session ever importing the frontend
        # (the one-way na -> frontend boundary stays intact).
        self._device_teardown_hooks: list[Any] = []
        # A FINER seam than the full teardown above: hooks here are told exactly which device
        # INSTANCES are being replaced/closed (their ``id()`` set), so a consumer stops only the
        # work that rides the affected hardware -- swapping the camera leaves a scan on the
        # untouched sequencer running.  ``load_config`` (a per-role instance swap) notifies here;
        # ``close`` still runs the full-teardown hooks (everything goes).
        self._device_change_hooks: list[Any] = []
        # Devices stay session-blind: a camera is a pure grabber and a sequencer a pure
        # streamer; ALL cross-device orchestration (capture / readout / scans) lives on the
        # session + subsystems, which reach devices only through their contracts.
        self._readout_subsystem = ReadoutSubsystem(self)
        self._timing_subsystem = TimingSubsystem(self)

    @property
    def camera(self) -> CameraDevice:
        return self.devices.camera

    @property
    def sequencer(self) -> SequencerDevice:
        return self.devices.sequencer

    @property
    def readout(self) -> ReadoutSubsystem:
        if not hasattr(self, "_readout_subsystem"):
            self._readout_subsystem = ReadoutSubsystem(self)
        return self._readout_subsystem

    @property
    def calibration_data(self) -> TrapCalibration | None:
        return self._calibration

    @calibration_data.setter
    def calibration_data(self, calibration: TrapCalibration | None) -> None:
        """The PUBLIC write seam for the current calibration state.

        The readout subsystem OWNS calibration operations (sitemap / thresholds /
        characterize) but the calibration is SESSION state (shared with detect, the
        GUI, ``load_config`` reset), so it lives here and the subsystem stores its
        result through this setter instead of reaching into ``session._calibration``.
        The one place ``None`` clears it back to "no calibration"."""
        self._calibration = None if calibration is None else calibration

    @property
    def timing(self) -> TimingSubsystem:
        if not hasattr(self, "_timing_subsystem"):
            self._timing_subsystem = TimingSubsystem(self)
        return self._timing_subsystem

    # ---- GUI launchers (confocal-style ``exp.task_console()`` / ``exp.pulse_gui()``) --------
    # The windows live in the frontend; these are thin sugar reached LAZILY through the
    # GUI-action module ``_gui`` (which imports the frontend only when a window is opened), so
    # the analysis path (connect / sitemap / thresholds / detect) never pulls the frontend and
    # virtual==real stays headless.  ``session.py`` references only ``_gui`` -- never the
    # frontend itself -- keeping the one-directional neutral_atom -> frontend seal.
    def task_console(self, *, task: str | None = None, **kwargs):
        """Open the Task console GUI bound to this session.

        Sugar over ``frontend.show_task_console``: fills the hub + the auto-discovered
        measurement / processor / task catalogs from this session.  ``task`` loads a saved
        layout (``tasks/<name>.json``)."""
        from ._gui import open_task_console
        return open_task_console(self, task=task, **kwargs)

    def pulse_gui(self, *, state=None, **kwargs):
        """Open the pulse-sequence editor GUI bound to this session, so a measurement can read
        the edited program back.  To run the editor WITHOUT a session (it picks its own server
        connection, needing no experiment) call ``Zou_lab_control.frontend.show_pulse_gui()``
        directly."""
        from ._gui import open_pulse_gui
        return open_pulse_gui(self, state=state, **kwargs)

    def figure_viewer(self, path=None, **kwargs):
        """Open the saved-figure viewer GUI (reopen a ``.npz`` written by a panel / notebook Save).

        A PURE VIEWER -- no hardware, no acquisition: pick a saved figure (or a folder of them, e.g. a
        calibration run) and re-view / relim / fit / re-save it through the same DataFigure stack.  A
        ONE-per-session singleton (a later call reshows the same window).  To open a viewer WITHOUT a
        session call ``Zou_lab_control.frontend.show_figure_viewer()`` directly."""
        from ._gui import open_figure_viewer
        return open_figure_viewer(self, path=path, **kwargs)

    def device_manager(self, **kwargs):
        """Open the device-manager GUI bound to this session: see every device the config loaded,
        grouped by device DOMAIN (Camera / Sequencer / Trap array / a future RF source -- the SAME
        registry the per-measurement device dropdowns read), and a "Scan hardware" button that probes
        the buses.  The GUI face of ``na.load_devices`` / ``na.discover_devices``; a ONE-per-session
        window."""
        from ._gui import open_device_manager
        return open_device_manager(self, **kwargs)

    def device_viewer(self, **kwargs):
        """Open the READ-ONLY device viewer bound to this session -- one tab per loaded device
        showing its snapshot + live runtime read-backs, with NO editing / add / remove.  The safe
        "look at a device while an experiment runs" window (the task console's Devices button opens
        this); the full config EDITOR is the separate :meth:`device_manager` entry.  A
        ONE-per-session window."""
        from ._gui import open_device_viewer
        return open_device_viewer(self, **kwargs)

    def save_config(self, path: str | Path) -> Path:
        """Write this session's device CONFIG to ``path`` as JSON so it can be reloaded later.

        The config is ``self.devices.to_config()`` -- the round-trippable ``{role: {"type", "params"}}``
        dict the device set was built from -- so ``na.connect(path)`` (or the device manager's "Load
        config" button, or :meth:`load_config`) reproduces the SAME hardware set on the next session.  A
        missing ``.json`` suffix is added; parent dirs are created.  Returns the written path."""
        import json

        path = Path(path)
        if path.suffix.lower() != ".json":
            path = path.with_suffix(".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.devices.to_config(), indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def load_config(self, config: str | Path | dict[str, Any], *, open_devices: bool = False):
        """Rebuild this session's devices from a NEW experiment config -- a SETUP-time swap of what
        hardware the session drives (do it before running measurements, not mid-acquisition).

        ``config`` takes the SAME forms as :func:`connect` (``"virtual"``, a bundled config name, a JSON
        path, or a ``{role: {type, params}}`` dict).  The NEW set is built FIRST (so a bad config leaves
        the current session untouched), then swapped in and every device-derived piece of state -- the
        imaging sequence, the calibration, the readout / timing subsystems -- is re-derived so the
        session is consistent with the new hardware; the OLD devices are then closed.  ``open_devices``
        opens the real hardware immediately (else it stays lazy, opened on first use / :meth:`open_devices`).
        The device manager's "Load config" button routes here.  Returns ``self``."""
        from .devices import load_devices, read_config

        new_devices = load_devices(read_config(config), open_devices=open_devices)
        # The new set built OK -- NOW stop the consumers that ride the devices ABOUT TO BE
        # REPLACED, before the swap, so nothing keeps driving hardware that is about to be closed.
        # Ordered after the build on purpose: a bad config leaves the whole session -- including
        # its running nodes -- untouched.  ``affected`` = every OLD device whose role now maps to a
        # DIFFERENT instance (``load_devices`` builds fresh, so a full-config swap replaces every
        # role; a future single-role reinit would flag only that one) -- so a swap of just the
        # camera leaves a scan running on the untouched sequencer.
        old_devices = getattr(self, "devices", None)
        affected: set[int] = set()
        if old_devices is not None:
            new_by_role = dict(getattr(new_devices, "devices", {}) or {})
            for role, old_dev in dict(getattr(old_devices, "devices", {}) or {}).items():
                if old_dev is not None and new_by_role.get(role) is not old_dev:
                    affected.add(id(old_dev))
        self._notify_device_change(affected)
        self.devices = new_devices
        self.sequence = self.build_imaging_sequence(exposure=self.camera_exposure(), load=True)
        self._calibration = None                          # the old calibration was for the old camera
        self._readout_subsystem = ReadoutSubsystem(self)  # subsystems read devices through the session
        self._timing_subsystem = TimingSubsystem(self)
        if old_devices is not None and old_devices is not new_devices:
            try:
                old_devices.close()
            except Exception:
                # A failed close (dead RPyC / camera handle -- common on real hardware) must be
                # SURFACED, not silently swallowed: an un-closed device lingers on the hardware.
                # Log it like every other teardown path here, never a bare ``pass``.
                import logging
                logging.getLogger(__name__).warning(
                    "closing the previous device set after load_config failed", exc_info=True)
        return self

    def open_devices(self):
        """Initialize / connect the loaded hardware -- open every device in dependency order (cameras
        LAST, so they bind to an already-open sequencer / trigger source).  A no-op-safe convenience over
        ``self.devices.open()`` for the notebook and the device manager's "Open devices" button; on the
        virtual backend the opens are trivial.  Returns ``self``."""
        self.devices.open()
        return self

    def capture(self, frames: int = 1, *, camera: str | None = None, exposure: float | None = None,
                display: bool = True) -> CaptureResult:
        """Grab raw frames from a named camera and return a notebook-friendly ``CaptureResult``.

        SESSION-level orchestration of the one-off snapshot: the session picks the sensor
        (``camera`` names any camera in the device config; None = the conventional readout
        role), optionally writes ``exposure`` to THAT camera and refreshes the imaging
        sequence, then runs the standard arm-before-fire shot (``triggered_frames``: arm the
        camera, fire the session sequencer, read the frames back).  The camera itself stays a
        pure grabber -- it knows nothing about the session.  ``capture`` always shows raw
        camera data; site overlays belong to calibrated readout/detection, not to capture."""
        from .operations.measurement import triggered_frames
        from .views.plots import plot_image

        name = str(camera) if camera else self.devices.default_camera_name()
        cam = self.devices[name]
        if exposure is not None:
            # The ONE configure-imaging path (owned by the timing subsystem): write the
            # exposure to THIS camera and rebuild the imaging sequence -- never hand-rolled.
            self.timing.configure_imaging(exposure=float(exposure), camera=cam)
        sequence = self.sequence
        images = triggered_frames(cam, getattr(self.devices, "sequencer", None), sequence, int(frames))
        if not images:
            # No edge on this camera's trigger line -> no frame (the pure-grabber contract).  The
            # readout imaging sequence gates the READOUT camera's trigger; a camera wired to another
            # line (e.g. the MOT monitor on ``mot_trigger``) is snapshotted through ITS own template,
            # not this convenience.  Fail with an actionable message, never an IndexError on ``[-1]``.
            from .devices.camera_trigger import resolve_camera_trigger_channels
            trig = resolve_camera_trigger_channels(cam)
            raise RuntimeError(
                f"camera {name!r} captured no frames: the imaging sequence gates the readout "
                f"camera's trigger, but {name!r} is triggered on {trig or '(unknown line)'}.  "
                f"Snapshot a non-readout camera (e.g. the MOT monitor) through its OWN pulse "
                f"template -- run a Pulse-scan measurement or the Optimize-MOT-field task with "
                f"camera={name!r} (they fire the coil template that pulses its trigger).")
        plot = plot_image(images[-1], display=display)      # display=False builds the figure, doesn't show it
        result = CaptureResult(images=images, sequence=sequence, plot=plot)
        self.history.append(result)
        return result

    def build_imaging_sequence(self, **kwargs) -> PulseSequence:
        """Build a readout imaging pulse sequence for the current devices.

        The PUBLIC seam the readout / timing subsystems use to compose an imaging
        sequence (``exposure`` / ``load`` / ``name`` / ``trigger_width`` /
        ``pre_trigger``): the channel wiring (which line gates the frame) is filled
        in from the ONE source ``_imaging_channel_kwargs`` so every imaging path --
        sitemap, thresholds, detect, ``capture`` -- gates identically."""
        return imaging_sequence(**kwargs, **self._imaging_channel_kwargs())

    def _imaging_channel_kwargs(self) -> dict[str, str]:
        # Single source of truth lives in timing.imaging_channel_kwargs so the
        # session and the loading readout map channels identically (see M4 / logic nodes).
        # The CAMERA owns which line gates a frame, so the imaging pulse triggers THAT channel --
        # read through the one derived fact (CameraDevice.primary_trigger_channel: the first
        # active counting line, None for a free-running sensor / no camera in the config).
        cam = getattr(self.devices, "camera", None)
        return imaging_channel_kwargs(getattr(self.devices, "sequencer", None),
                                      trigger_channel=getattr(cam, "primary_trigger_channel", None))

    def load_calibration(self, path: str | Path) -> TrapCalibration:
        self._calibration = TrapCalibration.load(path)
        return self._calibration

    def save_calibration(self, path: str | Path) -> Path:
        return self.require_calibration(require_thresholds=False).save(path)

    def require_calibration(self, *, require_thresholds: bool = True) -> TrapCalibration:
        if self._calibration is None:
            raise RuntimeError("No calibration is loaded. Run exp.readout.sitemap() first.")
        if require_thresholds and not self._calibration.metadata.get("thresholds_calibrated", False):
            raise RuntimeError("No threshold calibration is loaded. Run exp.readout.thresholds() first.")
        return self._calibration

    def status(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "devices": self.devices.snapshot(),
            "sequence": self.sequence.table(),
            "calibration": None if self._calibration is None else self._calibration.to_dict(),
            "history_length": len(self.history),
        }

    def _repr_html_(self) -> str:
        calibration = "none" if self._calibration is None else f"{self._calibration.n_sites} sites"
        devices = ", ".join(sorted(self.devices.devices))
        return html_summary(
            "NeutralAtomSession",
            {
                "name": self.name,
                "devices": devices,
                "sequence": self.sequence.name,
                "calibration": calibration,
                "history": len(self.history),
            },
        )

    def save_status(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(json_ready(self.status()), indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def add_device_teardown_hook(self, hook) -> None:
        """Register a callable run BEFORE this session's devices are closed (:meth:`close`) or
        swapped (:meth:`load_config`) -- the seam a long-lived device CONSUMER uses to be stopped
        first.  The task console registers its ``stop_all_nodes`` here when it opens, so a
        notebook's ``exp.close()`` / device-manager "Load config" can never pull devices out from
        under running acquisition threads (which would otherwise keep old camera / RPyC handles
        alive and keep driving closed hardware).  Idempotent: re-registering the same callable is
        a no-op (the singleton console re-opens without stacking duplicates)."""
        if hook not in self._device_teardown_hooks:
            self._device_teardown_hooks.append(hook)

    def _release_device_consumers(self) -> None:
        """Run every registered teardown hook (stop running consumers), never letting one broken
        hook block the device close/swap itself -- teardown must always complete."""
        import logging

        for hook in tuple(self._device_teardown_hooks):
            try:
                hook()
            except Exception:
                logging.getLogger(__name__).warning(
                    "device teardown hook %r failed; continuing teardown", hook, exc_info=True)

    def add_device_change_hook(self, hook) -> None:
        """Register a callable ``hook(affected_ids: set[int])`` run BEFORE specific device
        INSTANCES are replaced (:meth:`load_config`) -- the fine-grained sibling of
        :meth:`add_device_teardown_hook`.  ``affected_ids`` is the ``id()`` set of the devices
        being swapped out, so a consumer stops only the work riding THOSE devices (the console
        registers ``stop_nodes_using`` here: swapping the camera stops camera nodes, but a scan
        on the untouched sequencer keeps running).  Idempotent."""
        if hook not in self._device_change_hooks:
            self._device_change_hooks.append(hook)

    def _notify_device_change(self, affected_ids: set) -> None:
        """Tell every change hook which device instances are going away -- a broken hook never
        blocks the swap."""
        import logging

        for hook in tuple(self._device_change_hooks):
            try:
                hook(set(affected_ids))
            except Exception:
                logging.getLogger(__name__).warning(
                    "device change hook %r failed; continuing swap", hook, exc_info=True)

    def close(self) -> None:
        self._release_device_consumers()   # stop consumers (running GUI nodes) BEFORE the devices go
        self.devices.close()

    def camera_exposure(self) -> float:
        """The readout camera's current exposure (seconds) -- the PUBLIC seam the
        readout / timing subsystems read to gate a frame.  A camera-less config still
        composes imaging sequences with the stock default (no fabricated device needed)."""
        camera = getattr(self.devices, "camera", None)
        if camera is None:
            return 20e-3
        return float(getattr(camera, "exposure", getattr(getattr(camera, "config", None), "exposure", 20e-3)))

    def resolve_grid_shape(self, grid_shape: Sequence[int] | None) -> tuple[int, int]:
        """Resolve a tweezer grid shape -- the explicit ``grid_shape`` if given, else the
        loaded ``trap_array``'s.  The PUBLIC seam the readout subsystem / occupancy processor
        use so grid-shape resolution has ONE source (raises when neither is available)."""
        if grid_shape is not None:
            return grid_shape_tuple(grid_shape)
        trap_array = getattr(self.devices, "trap_array", None)
        if trap_array is not None and hasattr(trap_array, "grid_shape"):
            return grid_shape_tuple(trap_array.grid_shape)
        raise ValueError("grid_shape is required when the device config has no trap_array.")

def connect(
    config: str | Path | dict[str, Any] = "virtual",
    *,
    name: str = "neutral_atom",
    trap_array: dict[str, Any] | None = None,
    sitemap: dict[str, Any] | None = None,
    camera: dict[str, Any] | None = None,
    sequencer: dict[str, Any] | None = None,
    defaults: dict[str, Any] | None = None,
    open_devices: bool = False,
    **virtual_params,
) -> NeutralAtomSession:
    """Load devices and return a notebook-facing neutral-atom session."""

    default_values = dict(defaults or {})
    # The device layer (registry) owns backend dispatch + any backend-specific
    # config shortcuts; the session never imports a concrete backend or reads its
    # internal fields, so virtual <-> real is a one-line `config` change.
    device_config, device_overrides, inferred_defaults = resolve_connect_config(
        config,
        trap_array=trap_array,
        sitemap=sitemap,
        camera=camera,
        sequencer=sequencer,
        params=virtual_params,
    )
    default_values.update(inferred_defaults)
    return NeutralAtomSession(
        load_devices(device_config, overrides=device_overrides, open_devices=open_devices),
        name=name,
        defaults=default_values,
    )


__all__ = [
    "CaptureResult",
    "DetectionResult",
    "DetectionTimeScanResult",
    "ExperimentSubsystem",
    "MeasurementTaskResult",
    "NeutralAtomSession",
    "PreflightReport",
    "ReadoutSubsystem",
    "ResultObject",
    "SitemapResult",
    "ThresholdResult",
    "TimingSubsystem",
    "calibrate_sitemap_from_images",
    "calibrate_threshold_from_images",
    "connect",
    "detect_image",
]
