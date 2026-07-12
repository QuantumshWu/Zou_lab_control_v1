"""Jupyter-first neutral-atom experiment session.

This file is the public shape of the lightweight first milestone.  The lower
layers still exist as device/timing/analysis/verilog boundaries; the notebook
user mostly talks to ``NeutralAtomSession`` and result objects.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence
import json
import threading

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
from .device_catalog import DeviceCatalogView
from .devices import DeviceSet, load_devices, resolve_connect_config
from .installation import InstallationSupervisor, RecoveryStatusRef
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

    def __init__(
        self,
        devices: DeviceSet,
        *,
        name: str = "neutral_atom",
        defaults: dict[str, Any] | None = None,
        _runtime_services: object | None = None,
    ):
        self._installation_supervisor = InstallationSupervisor(
            devices, _runtime_services
        )
        self._hardware_authority_local = threading.local()
        self.name = str(name)
        self.defaults = dict(defaults or {})
        self.sequence = self.build_imaging_sequence(exposure=self.camera_exposure(), load=True)
        self._calibration: TrapCalibration | None = None
        self.history: list[Any] = []
        # Devices stay session-blind: a camera is a pure grabber and a sequencer a pure
        # streamer; ALL cross-device orchestration (capture / readout / scans) lives on the
        # session + subsystems, which reach devices only through their contracts.
        self._readout_subsystem = ReadoutSubsystem(self)
        self._timing_subsystem = TimingSubsystem(self)

    @property
    def devices(self) -> DeviceCatalogView:
        """Read-only configured-device metadata; never a hardware drive capability."""

        return self._installation_supervisor.catalog

    @property
    def device_catalog(self) -> DeviceCatalogView:
        """Immutable public installation observation."""

        return self._installation_supervisor.catalog

    @property
    def _device_set(self):
        return self._installation_supervisor._available_device_set()

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
        """Open the pulse editor through this installation's managed command port.

        ``Zou_lab_control.frontend.show_pulse_gui(state=...)`` without a session is an
        offline editor and never creates or discovers hardware.
        """
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

        The config is the private device set's round-trippable ``{role: {"type", "params"}}``
        dict the device set was built from -- so ``na.connect(path)`` (or the device manager's "Load
        config" button, or :meth:`load_config`) reproduces the SAME hardware set on the next session.  A
        missing ``.json`` suffix is added; parent dirs are created.  Returns the written path."""
        import json

        path = Path(path)
        if path.suffix.lower() != ".json":
            path = path.with_suffix(".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self._device_set.to_config(), indent=2, ensure_ascii=False), encoding="utf-8")
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

        unavailable = getattr(self, "_hardware_unavailable_reason", None)
        if unavailable is not None:
            raise RuntimeError(
                f"{unavailable}; create a new NeutralAtomSession to re-establish hardware"
            )

        # Build/validate replacement objects first, but never OPEN the new physical set while
        # the old owner is still live.  Hardware opening happens only after consumers stop and
        # the old DeviceSet has confirmed close.
        supervisor = self._installation_supervisor
        current_catalog = supervisor.catalog
        old_devices = supervisor._available_device_set()
        new_devices = load_devices(read_config(config), open_devices=False)
        runtime_services = supervisor._runtime()
        if runtime_services is None or runtime_services.closed:
            raise RuntimeError("this session has no live installation hardware authority")
        prepared_runtime_devices = None
        try:
            staged_sequence = self._build_imaging_sequence_for_devices(
                new_devices,
                exposure=self._camera_exposure_for_devices(new_devices),
                load=True,
            )
            staged_readout = ReadoutSubsystem(self)
            staged_timing = TimingSubsystem(self)
            prepared_runtime_devices = runtime_services.prepare_device_set(new_devices)
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
        affected: set[int] = set()
        if old_devices is not None:
            new_by_role = dict(getattr(new_devices, "devices", {}) or {})
            for role, old_dev in dict(getattr(old_devices, "devices", {}) or {}).items():
                if old_dev is not None and new_by_role.get(role) is not old_dev:
                    affected.add(id(old_dev))
        transition_token = None
        irreversible = False
        try:
            transition_token = runtime_services.begin_device_transition()

            # Hardware quiescence belongs solely to the runtime authority.  GUI models
            # observe terminal handles later; no QWidget hook participates in this proof.
            affected_devices = tuple(
                device
                for device in dict(getattr(old_devices, "devices", {}) or {}).values()
                if id(device) in affected
            )
            receipts = runtime_services.fence.stop_nodes_using(
                affected_devices,
                timeout=2.0,
            )
            if any(not receipt.terminated for receipt in receipts):
                raise RuntimeError(
                    "device config swap refused because runtime ownership is still cancelling"
                )

            # This is the public linearization point for the irreversible phase.  From
            # here onward no caller can observe the old generation as AVAILABLE, even
            # though closing its connections happens immediately afterwards.
            supervisor._publish_swapping()
            irreversible = True

            if old_devices is not None and old_devices is not new_devices:
                old_devices.close()

            if open_devices:
                runtime_services.ensure_prepared_device_set_connections(
                    prepared_runtime_devices
                )
                staged_sequence = self._build_imaging_sequence_for_devices(
                    new_devices,
                    exposure=self._camera_exposure_for_devices(new_devices),
                    load=True,
                )

            def publish_session() -> None:
                (
                    self.sequence,
                    self._calibration,
                    self._readout_subsystem,
                    self._timing_subsystem,
                ) = (
                    staged_sequence,
                    None,
                    staged_readout,
                    staged_timing,
                )
                supervisor._publish_available(new_devices)

            runtime_services.commit_device_transition(
                transition_token,
                new_devices,
                prepared_runtime_devices,
                publish=publish_session,
            )
            transition_token = None
        except BaseException as swap_error:
            if not irreversible:
                if transition_token is not None:
                    runtime_services.abort_device_transition(transition_token)
                new_devices.close()
            else:
                reason = (
                    "device config swap crossed the old-device close boundary and failed: "
                    f"{type(swap_error).__name__}: {swap_error}"
                )
                recovery_ref = RecoveryStatusRef(
                    f"recovery/status/config-swap-{current_catalog.installation_state_revision}"
                )
                supervisor._publish_recovery_required(recovery_ref)
                supervisor._retain_swap_recovery(
                    transition_token=transition_token,
                    candidate_device_set=new_devices,
                    prepared_binding_state=prepared_runtime_devices,
                    reason=reason,
                )
                self._hardware_unavailable_reason = reason
            raise
        return self

    def open_devices(self):
        """Initialize / connect the loaded hardware -- open every device in dependency order (cameras
        LAST, so they bind to an already-open sequencer / trigger source).  A no-op-safe convenience over
        the device manager's "Open devices" button; on the
        virtual backend the opens are trivial.  Returns ``self``."""
        unavailable = getattr(self, "_hardware_unavailable_reason", None)
        if unavailable is not None:
            raise RuntimeError(
                f"{unavailable}; create a new NeutralAtomSession to re-establish hardware"
            )
        runtime_services = self._require_runtime_services()
        runtime_services.ensure_device_set_connections(self._device_set)
        # A remote sequencer binds its sole PortCatalog while opening.  Rebuild
        # the session's convenience imaging sequence only after that boundary,
        # so an offline semantic placeholder can never survive as the program
        # later sent to real physical lanes.
        self.sequence = self.build_imaging_sequence(
            exposure=self.camera_exposure(), load=True)
        return self

    def _require_runtime_services(self):
        runtime_services = self._installation_supervisor._runtime()
        if runtime_services is None or runtime_services.closed:
            raise RuntimeError(
                "this session has no live installation hardware authority; construct it through connect()"
            )
        return runtime_services

    def _run_hardware_call(self, devices, callback, *, name: str):
        """Run one legacy device operation under the installation authority.

        Nested orchestration on the same owner thread (for example capture -> timing
        configure -> triggered_frames) reuses the already-held claim, but it may not
        smuggle in a device absent from the outer declaration.
        """

        requested = tuple(dict.fromkeys(id(device) for device in devices if device is not None))
        active = getattr(self._hardware_authority_local, "device_ids", None)
        if active is not None:
            if not set(requested).issubset(active):
                raise RuntimeError("nested hardware operation requested an undeclared device")
            return callback()

        def owned_call():
            self._hardware_authority_local.device_ids = frozenset(requested)
            try:
                return callback()
            finally:
                del self._hardware_authority_local.device_ids

        concrete = tuple(device for device in devices if device is not None)
        return self._require_runtime_services()._run_hardware_call(
            concrete,
            owned_call,
            name=name,
        )

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

        name = str(camera) if camera else self._device_set.default_camera_name()
        cam = self._device_set[name]
        trigger_channels = resolve_camera_trigger_channels(cam)
        readout_owned = self._device_set.devices.get("camera") is cam

        if not trigger_channels:
            # A free-running sensor owns its exposure clock.  Configuring it must not mutate the
            # readout sequence (which may belong to a different camera), and there is no trigger
            # edge for the session to fire.
            def acquire_free_running():
                if exposure is not None:
                    cam.configure(exposure=float(exposure))
                return cam.acquire(int(frames)), None

            call_devices = (cam,)
            hardware_call = acquire_free_running
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
            sequencer = self._device_set.devices.get("sequencer")
            if sequencer is None:
                raise RuntimeError(
                    f"camera {name!r} is externally triggered on {trigger_channels}, but this "
                    f"device config has no sequencer to drive the readout pulse.")
            def acquire_triggered():
                if exposure is not None:
                    self.timing.configure_imaging(exposure=float(exposure))
                sequence = self.sequence
                return triggered_frames(cam, sequencer, sequence, int(frames)), sequence

            call_devices = (cam, sequencer)
            hardware_call = acquire_triggered

        images, sequence = self._run_hardware_call(
            call_devices,
            hardware_call,
            name=f"capture-{name}",
        )

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
        return self._build_imaging_sequence_for_devices(self._device_set, **kwargs)

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
        return self._imaging_channel_kwargs_for_devices(self._device_set)

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
            "devices": self.devices.to_dict(),
            "sequence": self.sequence.table(),
            "calibration": None if self._calibration is None else self._calibration.to_dict(),
            "history_length": len(self.history),
        }

    def _repr_html_(self) -> str:
        calibration = "none" if self._calibration is None else f"{self._calibration.n_sites} sites"
        devices = ", ".join(self.devices)
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

    def close(self) -> None:
        supervisor = self._installation_supervisor
        runtime_services = supervisor._runtime()
        if runtime_services is not None and not runtime_services.shutdown(timeout=2.0):
            raise RuntimeError(
                "device close refused because runtime ownership is still cancelling"
            )
        errors: list[BaseException] = []
        for device_set in supervisor._device_sets_for_shutdown():
            try:
                device_set.close()
            except BaseException as exc:
                errors.append(exc)
        if errors:
            raise RuntimeError(
                f"{len(errors)} installation device set(s) failed to close"
            ) from errors[0]

    def camera_exposure(self) -> float:
        """The readout camera's current exposure (seconds) -- the PUBLIC seam the
        readout / timing subsystems read to gate a frame.  A camera-less config still
        composes imaging sequences with the stock default (no fabricated device needed)."""
        return self._camera_exposure_for_devices(self._device_set)

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
        trap_array = getattr(self._device_set, "trap_array", None)
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
    devices = load_devices(device_config, overrides=device_overrides, open_devices=False)
    # Temporary S0.5 composition bridge.  The final notebook facade owns this import and the
    # neutral domain receives only typed ports; until that cut, every old notebook/GUI entry gets
    # the same installation authority instead of lazily creating a second one in each window.
    from zlc_workbench.legacy_neutral_atom import LegacyNeutralAtomRuntime

    runtime_services = None
    try:
        runtime_services = LegacyNeutralAtomRuntime(devices)
        session = NeutralAtomSession(
            devices,
            name=name,
            defaults=default_values,
            _runtime_services=runtime_services,
        )
        if open_devices:
            session.open_devices()
        return session
    except BaseException:
        if runtime_services is not None:
            runtime_services.shutdown(timeout=0.0)
        devices.close()
        raise


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
