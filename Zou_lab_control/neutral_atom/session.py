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
from .timing import DEFAULT_EXPOSURE_S, PulseSequence, imaging_channel_kwargs, imaging_sequence
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
        """Open the device viewer bound to this session -- one tab per loaded device showing its
        snapshot + live runtime read-backs, and (``editable=True`` by default) editing each device's
        basic runtime params LIVE like the API (exposure, ROI, RF detuning, ...): every write routes
        through the device's OWN validated setter.  It is NOT the config editor -- no add / remove /
        config-swap; the full config EDITOR is the separate :meth:`device_manager` entry.  Pass
        ``editable=False`` for a pure read-only peek.  A ONE-per-session window (the task console's
        Devices button opens this)."""
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
        path, or a ``{role: {type, params}}`` dict).  The NEW set and every purely-derived replacement
        object are built FIRST (so a bad config leaves the current session untouched); consumers then
        stop, the OLD devices confirm close, and the complete new state is published in one assignment
        boundary.  ``open_devices``
        opens the real hardware immediately (else it stays lazy, opened on first use / :meth:`open_devices`).
        The device manager's "Load config" button routes here.  Returns ``self``."""
        from .devices import load_devices, read_config

        # Build/validate replacement objects first, but never OPEN the new physical set while
        # the old owner is still live.  Hardware opening happens only after consumers stop and
        # the old DeviceSet has confirmed close.
        new_devices = load_devices(read_config(config), open_devices=False)
        try:
            staged_sequence = self._build_imaging_sequence_for_devices(
                new_devices,
                exposure=self._camera_exposure_for_devices(new_devices),
                load=True,
            )
            staged_readout = ReadoutSubsystem(self)
            staged_timing = TimingSubsystem(self)
        except BaseException:
            new_devices.close()
            raise
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
        try:
            self._notify_device_change(affected)
        except BaseException as consumer_error:
            try:
                new_devices.close()
            except BaseException as new_close_error:
                raise RuntimeError(
                    "consumer stop failed and the staged replacement devices also failed to close: "
                    f"{type(new_close_error).__name__}: {new_close_error}"
                ) from consumer_error
            raise
        if old_devices is not None and old_devices is not new_devices:
            try:
                old_devices.close()
            except BaseException as old_close_error:
                try:
                    new_devices.close()
                except BaseException as new_close_error:
                    raise RuntimeError(
                        "previous devices failed to close and staged replacement cleanup also failed: "
                        f"{type(new_close_error).__name__}: {new_close_error}"
                    ) from old_close_error
                raise RuntimeError(
                    "device config swap refused because the previous device set did not close"
                ) from old_close_error
        # Publish all session-visible derived state together.  No constructor or sequence compilation
        # remains on the far side of this boundary, so an exception cannot expose new devices with an
        # old sequence/calibration/subsystem graph.
        (
            self.devices,
            self.sequence,
            self._calibration,
            self._readout_subsystem,
            self._timing_subsystem,
        ) = (new_devices, staged_sequence, None, staged_readout, staged_timing)
        if open_devices:
            self.open_devices()
        return self

    def open_devices(self):
        """Initialize / connect the loaded hardware -- open every device in dependency order (cameras
        LAST, so they bind to an already-open sequencer / trigger source).  A no-op-safe convenience over
        ``self.devices.open()`` for the notebook and the device manager's "Open devices" button; on the
        virtual backend the opens are trivial.  Returns ``self``."""
        self.devices.open()
        # A remote sequencer binds its sole PortCatalog while opening.  Rebuild
        # the session's convenience imaging sequence only after that boundary,
        # so an offline semantic placeholder can never survive as the program
        # later sent to real physical lanes.
        self.sequence = self.build_imaging_sequence(
            exposure=self.camera_exposure(), load=True)
        return self

    def capture(self, frames: int = 1, *, camera: str | None = None, exposure: float | None = None,
                display: bool = True) -> CaptureResult:
        """Grab raw frames from a named camera and return a notebook-friendly ``CaptureResult``.

        The selected camera's declared capture-trigger wiring determines the only legal path:

        * a free-running camera (no active capture-trigger channels) is acquired directly;
          ``exposure`` configures only that sensor and no pulse is rebuilt or fired;
        * the conventional ``"camera"`` readout role uses the session imaging sequence and
          the standard arm-before-fire shot;
        * any other externally-triggered camera is rejected before touching hardware because
          only the measurement/task owning both that camera and its pulse template can fire it.

        The camera stays a pure grabber and ``capture`` always shows raw pixels; site overlays
        belong to calibrated readout/detection, not to capture."""
        from .operations.measurement import triggered_frames
        from .devices.camera_trigger import resolve_camera_trigger_channels
        from .views.plots import plot_image

        name = str(camera) if camera else self.devices.default_camera_name()
        cam = self.devices[name]
        trigger_channels = resolve_camera_trigger_channels(cam)
        readout_owned = self.devices.devices.get("camera") is cam

        if not trigger_channels:
            # A free-running sensor owns its exposure clock.  Configuring it must not mutate the
            # readout sequence (which may belong to a different camera), and there is no trigger
            # edge for the session to fire.
            if exposure is not None:
                cam.configure(exposure=float(exposure))
            sequence = None
            images = cam.acquire(int(frames))
        elif not readout_owned:
            # Reject BEFORE configure/arm/prepare/fire: the session readout pulse does not own this
            # camera's trigger wire, so trying it and inspecting an empty buffer afterwards is too
            # late -- an unrelated hardware program has already run by then.
            raise RuntimeError(
                f"camera {name!r} is externally triggered on {trigger_channels}; "
                f"session.capture() only owns the conventional 'camera' readout pulse.  "
                f"Use the measurement/task that owns both {name!r} and its pulse template; "
                f"for a MOT camera, run Optimize-MOT-field with camera={name!r}.")
        else:
            sequencer = self.devices.devices.get("sequencer")
            if sequencer is None:
                raise RuntimeError(
                    f"camera {name!r} is externally triggered on {trigger_channels}, but this "
                    f"device config has no sequencer to drive the readout pulse.")
            if exposure is not None:
                self.timing.configure_imaging(exposure=float(exposure))
            sequence = self.sequence
            images = triggered_frames(cam, sequencer, sequence, int(frames))

        if not images:
            raise RuntimeError(
                f"camera {name!r} returned no frames via its "
                f"{'free-running acquisition' if sequence is None else 'readout pulse'}.")
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
        return self._build_imaging_sequence_for_devices(self.devices, **kwargs)

    @staticmethod
    def _build_imaging_sequence_for_devices(devices: DeviceSet, **kwargs) -> PulseSequence:
        return imaging_sequence(
            **kwargs,
            **NeutralAtomSession._imaging_channel_kwargs_for_devices(devices),
        )

    def _imaging_channel_kwargs(self) -> dict[str, str]:
        # Single source of truth lives in timing.imaging_channel_kwargs so the
        # session and the loading readout map channels identically (see M4 / logic nodes).
        # The CAMERA owns which line gates a frame, so the imaging pulse triggers THAT channel --
        # read through the one derived fact (CameraDevice.primary_trigger_channel: the first
        # active counting line, None for a free-running sensor / no camera in the config).
        return self._imaging_channel_kwargs_for_devices(self.devices)

    @staticmethod
    def _imaging_channel_kwargs_for_devices(devices: DeviceSet) -> dict[str, str]:
        cam = getattr(devices, "camera", None)
        return imaging_channel_kwargs(
            getattr(devices, "sequencer", None),
            trigger_channel=getattr(cam, "primary_trigger_channel", None),
        )

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
        """Prove every registered consumer stopped before closing its devices."""

        failures = []
        for hook in tuple(self._device_teardown_hooks):
            try:
                result = hook()
                if result is False:
                    failures.append(f"{hook!r} reported unresolved ownership")
            except BaseException as exc:
                failures.append(f"{hook!r} failed: {type(exc).__name__}: {exc}")
        if failures:
            raise RuntimeError(
                "device close refused because consumers did not terminate: "
                + "; ".join(failures)
            )

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
        """Stop affected consumers; an unconfirmed stop vetoes the device swap."""

        failures = []
        for hook in tuple(self._device_change_hooks):
            try:
                result = hook(set(affected_ids))
                if result is False:
                    failures.append(f"{hook!r} reported unresolved ownership")
            except BaseException as exc:
                failures.append(f"{hook!r} failed: {type(exc).__name__}: {exc}")
        if failures:
            raise RuntimeError(
                "device swap refused because consumers did not terminate: "
                + "; ".join(failures)
            )

    def close(self) -> None:
        self._release_device_consumers()   # stop consumers (running GUI nodes) BEFORE the devices go
        self.devices.close()

    def camera_exposure(self) -> float:
        """The readout camera's current exposure (seconds) -- the PUBLIC seam the
        readout / timing subsystems read to gate a frame.  A camera-less config still
        composes imaging sequences with the stock default (no fabricated device needed)."""
        return self._camera_exposure_for_devices(self.devices)

    @staticmethod
    def _camera_exposure_for_devices(devices: DeviceSet) -> float:
        camera = getattr(devices, "camera", None)
        if camera is None:
            return DEFAULT_EXPOSURE_S
        return float(getattr(camera, "exposure", getattr(getattr(camera, "config", None), "exposure", DEFAULT_EXPOSURE_S)))

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
