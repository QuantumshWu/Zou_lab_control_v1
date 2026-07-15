"""Camera boundary for the lightweight neutral-atom session."""

from __future__ import annotations

from dataclasses import dataclass, fields
import importlib
import math
import threading
import time
from typing import Any, Sequence

import numpy as np

from ..core.analysis import nonnegative_int, positive_float, positive_int
from ..timing import DEFAULT_EXPOSURE_S
from .base import (
    ROI_CLEAR_SENTINELS,
    AcquisitionCancelled,
    CameraCaptureTerminalRecord,
    CameraDevice,
    CameraFrameRecord,
    config_param_decl,
    snap_subarray,
)
from .camera_trigger import DEFAULT_CAMERA_TRIGGER_CHANNELS


# --------------------------------------------------------------------------- DCAM API ownership
# The Hamamatsu ``Dcamapi`` runtime is a PROCESS-WIDE singleton: ``init()`` flips one class bool
# and ``uninit()`` clears it with NO reference count of its own.  So two qCMOS cameras -- or a
# camera plus a ``discover()`` enumeration running while that camera is open -- must NOT each call
# ``uninit()`` blindly: the first ``uninit`` would tear the driver out from under the still-open
# camera, whose next wait / prop / buf call then crashes.  This module owns the single reference
# count the driver lacks: every ``open`` / ``discover`` that needs the runtime ACQUIRES it, every
# ``close`` / end-of-enumeration RELEASES it, and only the LAST release actually calls ``uninit``.
_DCAM_API_LOCK = threading.Lock()
# id(api) -> [refcount, api, owned].  ``owned`` records WHO started the runtime: True when our
# init() brought it up (we own the matching uninit), False when we merely ADOPTED an
# ALREADYINITIALIZED runtime someone outside this counter started (an external script / a prior
# session) -- releasing to zero must then just drop the entry, never uninit a runtime we do not own.
_DCAM_API_REFCOUNT: dict[int, list] = {}


def _dcam_acquire(api) -> None:
    """Take a reference on the process-wide DCAM runtime, initialising it on the FIRST holder.

    A driver that reports ``ALREADYINITIALIZED`` is already live (another holder, or a prior
    session that outlived its owner): that is SUCCESS for us -- we still record the reference so
    the matching :func:`_dcam_release` participates in the shared count -- never a failure.  Any
    other init error is a real fault and raised."""
    with _DCAM_API_LOCK:
        entry = _DCAM_API_REFCOUNT.get(id(api))
        if entry is not None:
            entry[0] += 1                       # already ours: another holder joins, no re-init
            return
        if api.init() is False:
            # Import the error enum lazily -- only when we must classify an init failure -- so the
            # SUCCESS path never touches the DCAM DLL-backed module (a machine without the runtime
            # still constructs/tests the counter with a fake api).
            from .drivers.dcam.dcamapi4 import DCAMERR
            if int(api.lasterr()) != int(DCAMERR.ALREADYINITIALIZED):
                raise RuntimeError(f"DCAM init failed: {api.lasterr()}")
            # Runtime already up (initialised outside this counter): ADOPT it -- take the first
            # reference but mark it not-owned, so release-to-zero never uninits a runtime an
            # external owner is still using.
            _DCAM_API_REFCOUNT[id(api)] = [1, api, False]
            return
        _DCAM_API_REFCOUNT[id(api)] = [1, api, True]


def _dcam_release(api) -> None:
    """Drop a reference taken by :func:`_dcam_acquire`; the LAST holder calls ``uninit`` ONLY
    when this counter's own ``init`` started the runtime (``owned``) -- an adopted runtime
    (ALREADYINITIALIZED, started by an external owner) just loses our bookkeeping entry.

    Releasing an untracked api (double release / never acquired) is a defensive no-op -- it must
    never reach through to ``uninit`` and tear the runtime out from under another live camera."""
    with _DCAM_API_LOCK:
        entry = _DCAM_API_REFCOUNT.get(id(api))
        if entry is None:
            return
        entry[0] -= 1
        if entry[0] <= 0:
            _DCAM_API_REFCOUNT.pop(id(api), None)
            if entry[2]:
                try:
                    api.uninit()
                except Exception:
                    pass


@dataclass
class QCMOSConfig:
    """Configuration for the thin real qCMOS adapter."""

    exposure: float = DEFAULT_EXPOSURE_S
    readout_speed: int = 1
    roi: tuple[int, int, int, int] | None = None
    device_index: int = 0
    timeout_ms: int = 10_000
    # Shutter/exposure scheme.  Left None = the camera's power-on default
    # (ORCA-qCMOS is rolling-shutter AREA), which staggers row exposure relative
    # to the single FPGA trigger.  Synchronous atom imaging usually wants a
    # consciously chosen mode -- set these (raw DCAMPROP enum ints) in the camera
    # config to pin it; they are written + read back only when not None.
    sensor_mode: int | None = None
    trigger_global_exposure: int | None = None
    # Which sequencer line the camera's external-trigger input is physically wired to.  The real
    # hardware gates on that edge via ``TRIGGERSOURCE.EXTERNAL`` regardless, so for the real camera
    # this is METADATA the upper layers (measurement / cali) read to build the right imaging pulse --
    # the camera never uses it to drive the sequencer.
    capture_trigger_channels: tuple[str, ...] = DEFAULT_CAMERA_TRIGGER_CHANNELS

    def __post_init__(self) -> None:
        self.capture_trigger_channels = tuple(str(c) for c in self.capture_trigger_channels)
        # The ONE exposure validation every camera write path shares (devices.base).
        self.exposure = CameraDevice._validated_exposure(self.exposure)
        self.readout_speed = positive_int(self.readout_speed, "readout_speed")
        self.device_index = nonnegative_int(self.device_index, "device_index")
        self.timeout_ms = positive_int(self.timeout_ms, "timeout_ms")
        if self.roi is not None:
            self.roi = normalize_roi(self.roi)
        if self.sensor_mode is not None:
            self.sensor_mode = positive_int(self.sensor_mode, "sensor_mode")
        if self.trigger_global_exposure is not None:
            self.trigger_global_exposure = positive_int(
                self.trigger_global_exposure,
                "trigger_global_exposure",
            )


DEFAULT_DCAM_MODULE = "Zou_lab_control.neutral_atom.devices.drivers.dcam.dcam"


class QCMOSCamera(CameraDevice):
    """Thin external-trigger Hamamatsu qCMOS adapter.

    The offline session uses ``VirtualCamera``. This class is intentionally
    small: it only owns DCAM open/configure/acquire/close. FPGA trigger timing
    still belongs to the sequencer.
    """

    def __init__(self, config: QCMOSConfig | dict[str, Any] | None = None, *, dcam_module: str = DEFAULT_DCAM_MODULE):
        self.config = config if isinstance(config, QCMOSConfig) else QCMOSConfig(**dict(config or {}))
        # Passive wiring fact exposed upward (which sequencer line gates this camera); the camera
        # never uses it to drive the sequencer -- a real qCMOS is gated by the hardware edge.
        self.capture_trigger_channels = tuple(self.config.capture_trigger_channels)
        self.dcam_module_name = str(dcam_module)
        self._module = None
        self._api = None
        self._dcam = None
        # The sub-array the camera ACTUALLY applied (read back from DCAM after the
        # hardware snaps the request to its valid grid).  None until an open camera
        # has applied a ROI; the `roi` property reports this -- not the raw request
        # -- so a consumer (the Edit 'now:' reference, the 2D panel's axes) reflects
        # the region truly being imaged.
        self._applied_roi: tuple[int, int, int, int] | None = None
        # Full sensor geometry is queried while SUBARRAYMODE is deliberately
        # OFF during the settings transaction, then cached.  Merely observing
        # sensor_shape later must never change the live ROI.
        self._sensor_shape_yx: tuple[int, int] | None = None
        # Physical timing read back after every complete settings write.  The
        # requested config value is not evidence that DCAM accepted that exact
        # exposure, and TIMING_MINTRIGGERINTERVAL depends on the applied
        # exposure/ROI/readout working point.  Capability probes consume these
        # readbacks; None honestly means no live hardware snapshot exists yet.
        self._applied_exposure_seconds: float | None = None
        self._required_external_trigger_interval_seconds: float | None = None
        self._applied_readout_speed: int | None = None
        self._applied_sensor_mode: int | None = None
        self._applied_trigger_global_exposure: int | None = None
        self._applied_trigger_source: int | None = None
        self._applied_trigger_active: int | None = None
        self._applied_trigger_polarity: int | None = None
        # Armed-session bookkeeping for the acquisition hooks (_arm/_grab/_disarm):
        # the DCAM buffer depth of the current session + the next frame index to transfer.
        self._buf_alloc = 0
        self._next_frame = 0
        self._last_transfer_count = 0
        self._last_transfer_newest_index: int | None = None
        self._capture_io_condition = threading.Condition()
        self._active_grabs = 0
        self._capture_terminalizing = False

    # ------------------------------------------------------------------ config schema (self-describing)
    @classmethod
    def config_params(cls):
        """The qCMOS config form: the :class:`QCMOSConfig` dataclass FIELDS flattened into
        rows (the constructor nests them under ``params["config"]``; the editor should not
        show one opaque JSON blob) plus the ``dcam_module`` driver path.  The
        :meth:`config_to_form` / :meth:`form_to_config` pair does the flatten/regroup, so
        the entry on disk keeps the exact nested shape ``QCMOSCamera(**params)`` consumes."""
        rows = [config_param_decl(f.name, default=f.default, annotation=f.type,
                                  owner_module=None)
                for f in fields(QCMOSConfig)]
        rows.append(config_param_decl("dcam_module", default=DEFAULT_DCAM_MODULE,
                                      tooltip="DCAM driver module path (leave as shipped)."))
        return tuple(rows)

    @classmethod
    def config_to_form(cls, params) -> dict:
        params = dict(params or {})
        form = dict(params.get("config") or {})
        if "dcam_module" in params:
            form["dcam_module"] = params["dcam_module"]
        return form

    @classmethod
    def form_to_config(cls, values) -> dict:
        values = dict(values or {})
        dcam = values.pop("dcam_module", None)
        out: dict[str, Any] = {"config": values}
        if dcam not in (None, "", DEFAULT_DCAM_MODULE):
            out["dcam_module"] = dcam
        return out

    # ------------------------------------------------------------------ discovery (self-describing)
    @classmethod
    def discover(cls):
        """Enumerate attached Hamamatsu DCAM cameras -- each hit carries a READY config entry
        for THIS class (device_index-pinned).  The device class owns its own discovery:
        :func:`~.discovery.discover_devices` only aggregates.  A missing DCAM runtime / empty
        bus is a reported row, never an exception (the confocal contract)."""
        from .discovery import discovery_note

        try:
            mod = importlib.import_module(DEFAULT_DCAM_MODULE)
        except Exception as exc:
            return [discovery_note("qcmos", f"DCAM driver not importable: {exc}")]
        api = mod.Dcamapi
        # Take a reference on the shared runtime rather than a raw init/uninit: if a camera is
        # ALREADY OPEN, the runtime is up and this reference just joins the count -- enumeration
        # reads the bus without ever tearing the API out from under that live camera.  An
        # ALREADYINITIALIZED report is therefore normal (enumerate anyway), never a failure note.
        try:
            _dcam_acquire(api)
        except Exception as exc:
            return [discovery_note("qcmos", f"DCAM runtime init failed -- {exc}")]
        try:
            count = int(api.get_devicecount() or 0)

            def model_of(index: int) -> str:
                try:
                    from .drivers.dcam.dcamapi4 import DCAM_IDSTR
                    text = mod.Dcam(index).dev_getstring(DCAM_IDSTR.MODEL)
                    return str(text) if text else "Hamamatsu DCAM camera"
                except Exception:
                    return "Hamamatsu DCAM camera"

            return (cls.discovery_rows(count, model_of)
                    or [discovery_note("qcmos", "no Hamamatsu camera attached")])
        except Exception as exc:                # runtime present but transport layer unhappy
            return [discovery_note("qcmos", f"enumerate failed: {exc}")]
        finally:
            _dcam_release(api)                   # drop only OUR reference; a live camera keeps the API up

    @classmethod
    def discovery_rows(cls, count: int, model_of):
        """Pure enumeration->rows mapping (tests feed fakes): each attached camera becomes a
        row whose config is a ready entry for THIS class, pinned to its device index."""
        from .discovery import DiscoveredDevice

        return [DiscoveredDevice(
            kind="qcmos", ident=str(i), label=str(model_of(i)),
            config={"type": cls.__name__, "params": {"config": {"device_index": i}}})
            for i in range(max(0, int(count)))]

    @property
    def exposure(self) -> float:
        return self.config.exposure

    @exposure.setter
    def exposure(self, value: float) -> None:
        # write-through to configure (the single DCAM-write path) -- so `cam.exposure = 3e-3`
        # works the same as on the virtual backend; configure only touches hardware when open.
        self.configure(exposure=float(value))

    @property
    def roi(self) -> tuple[int, int, int, int] | None:
        # Report what the camera is ACTUALLY reading out (post hardware snap) when
        # open; the requested config ROI only when not yet applied.
        if self._applied_roi is not None:
            return self._applied_roi
        return self.config.roi

    @property
    def sensor_shape(self) -> tuple[int, int] | None:
        # The FULL sensor (height, width) -- the same contract VirtualCamera exposes, so a
        # raw-frame Edit shows the ROI as the full-frame window before any sub-array is set.
        # Queried from DCAM's sub-array max once the camera is open; None before that (the
        # honest "unknown until opened" the base default also returns).
        if self._dcam is None:
            return None
        return self._sensor_shape_yx

    def _require_configuration_epoch(self) -> None:
        state = self._recent_state()
        with state["cond"]:
            if state["arming"] or state["armed"]:
                raise RuntimeError(
                    "qCMOS settings cannot change during an armed acquisition epoch"
                )

    def configure(
        self,
        *,
        exposure: float | None = None,
        readout_speed: int | None = None,
        roi: Sequence[int] | None | object = None,
        **kwargs,
    ) -> None:
        self._reject_unknown_configure_keys({"exposure", "readout_speed", "roi"}, kwargs)
        # The first check makes a same-thread call fail despite the acquisition
        # RLock being re-entrant.  The second check, after taking that lock,
        # closes the race with arm(): either configuration linearizes first or
        # it cannot mutate the device until the acquisition epoch is terminal.
        self._require_configuration_epoch()
        with self._acquire_lock():
            self._require_configuration_epoch()
            if exposure is not None:
                self.config.exposure = self._validated_exposure(exposure)
            if readout_speed is not None:
                self.config.readout_speed = positive_int(readout_speed, "readout_speed")
            if roi is not None:
                # FULL_FRAME (and its blank-box equivalents) clears back to the full sensor -- the
                # ONE sentinel set in devices.base (roi is None means "leave unchanged"), so
                # configure(roi=...) is backend-agnostic.
                self.config.roi = None if roi in ROI_CLEAR_SENTINELS else normalize_roi(roi)
            if self._dcam is not None:
                self._write_settings()

    def runtime_controls(self):
        """The qCMOS live controls: the shared camera set (exposure / ROI / geometry) PLUS the WRITABLE
        ``readout_speed`` (this backend's extra live knob) -- a backend that has more knobs than the
        contract extends ``super().runtime_controls()`` here.  The write routes through the ONE
        ``configure(readout_speed=...)`` (validated by ``positive_int``, pushed to open hardware via
        ``_write_settings``), so the GUI never re-implements the device's own validation."""
        from .base import RuntimeControl
        from ..core.params import ParamDecl
        return super().runtime_controls() + (
            RuntimeControl(
                ParamDecl(key="readout_speed", label="readout speed", kind="choice",
                          choices=("1", "2"), default="1",
                          tooltip="qCMOS readout speed: 1 = ultra-quiet (low read noise), 2 = standard (fast)."),
                getter=lambda dev: str(dev.config.readout_speed),
                setter=lambda dev, value: dev.configure(readout_speed=int(value))),
        )

    def open(self) -> "QCMOSCamera":
        # Preserve BaseDevice.open's idempotence even inside an acquisition
        # epoch.  An already-live handle needs no configuration transaction.
        if self._dcam is not None:
            return self
        self._require_configuration_epoch()
        with self._acquire_lock():
            self._require_configuration_epoch()
            return self._open_locked()

    def _open_locked(self) -> "QCMOSCamera":
        """Open and configure while the caller owns the acquisition lock."""

        if self._dcam is not None:
            return self
        mod = importlib.import_module(self.dcam_module_name)
        api = mod.Dcamapi
        # Reference-counted acquire: initialises the runtime on the FIRST camera, joins the count
        # on the next -- so opening a second qCMOS never re-inits, and closing one never uninits
        # the other (the ownership seam the raw init/uninit lacked).
        _dcam_acquire(api)
        try:
            dcam = mod.Dcam(self.config.device_index)
            if dcam.dev_open() is False:
                raise RuntimeError(f"failed to open qCMOS device {self.config.device_index}: {dcam.lasterr()}")
        except BaseException:
            _dcam_release(api)                   # give back the reference we took; last holder uninits
            raise
        self._module = mod
        self._api = api
        self._dcam = dcam
        try:
            self._write_settings()
        except BaseException:
            try:
                dcam.dev_close()
            finally:
                self._dcam = None
                self._module = None
                self._api = None
                _dcam_release(api)
            raise
        return self

    @property
    def is_open(self) -> bool:
        """Live once :meth:`open` holds a DCAM handle -- the predicate the base ``ensure_open`` reads
        to lazily open on first ``arm`` (single-sourced with the pylon / virtual camera invariant)."""
        dcam = self._dcam
        if dcam is None:
            return False
        check = getattr(dcam, "is_opened", None)
        if callable(check):
            try:
                return bool(check())
            except BaseException:
                return False
        return True

    # ------------------------------------------------------------------ acquisition hooks
    # The public acquisition surface (arm / read_frames / disarm / acquire) lives on the
    # CameraDevice base; this adapter only implements the DCAM-facing hooks.  A measurement
    # that needs an external fire between arm and read goes through the single helper
    # ``operations.measurement.triggered_frames``.

    def _arm(
        self,
        frames: int | None,
        *,
        max_inflight_frames: int | None = None,
    ) -> None:
        """Allocate the DCAM frame buffer and start capture -- the hardware is then armed and
        waiting for FPGA trigger edges.  ``frames=None`` (continuous) allocates a
        ``recent_capacity``-deep ring the sequence capture cycles through."""
        dcam = self._dcam                        # arm() ensured the DCAM handle is open (base ensure_open)
        with self._capture_io_condition:
            if self._active_grabs:
                raise RuntimeError("qCMOS cannot arm while a frame read is still active")
            self._capture_terminalizing = False
        alloc = (
            int(max_inflight_frames)
            if max_inflight_frames is not None
            else (
                int(frames)
                if frames is not None
                else max(1, int(self.recent_capacity))
            )
        )
        if dcam.buf_alloc(alloc) is False:
            raise RuntimeError(f"qCMOS buf_alloc({alloc}) failed: {dcam.lasterr()}")
        started = False
        self._clear_applied_working_point()
        try:
            if dcam.cap_start(bSequence=True) is False:
                raise RuntimeError(f"qCMOS cap_start failed: {dcam.lasterr()}")
            started = True
            # cap_start is part of the physical arm boundary.  Refresh every
            # hardware-applied acquisition fact after it returns so the endpoint's
            # armed-start fingerprint compares real readback, not the cache minted
            # by the earlier configuration transaction.
            self._read_applied_working_point()
        except BaseException as primary:
            if started:
                try:
                    if dcam.cap_stop() is False:
                        raise RuntimeError(
                            f"qCMOS cap_stop after failed arm validation: {dcam.lasterr()}"
                        )
                except BaseException as cleanup_error:
                    try:
                        primary.add_note(
                            "qCMOS cap_stop after failed arm validation also failed: "
                            f"{type(cleanup_error).__name__}: {cleanup_error}"
                        )
                    except BaseException:
                        pass
            try:
                if dcam.buf_release() is False:
                    raise RuntimeError(
                        f"qCMOS buf_release after failed arm validation: {dcam.lasterr()}"
                    )
            except BaseException as cleanup_error:
                try:
                    primary.add_note(
                        "qCMOS buf_release after failed arm validation also failed: "
                        f"{type(cleanup_error).__name__}: {cleanup_error}"
                    )
                except BaseException:
                    pass
            raise
        self._buf_alloc = alloc
        self._next_frame = 0
        self._last_transfer_count = 0
        self._last_transfer_newest_index = None

    def _grab(
        self,
        n: int,
        *,
        timeout: float | None = None,
        stop=None,
        exact: bool = False,
    ) -> bool:
        """Run one DCAM read under the adapter's terminalization join gate."""

        requested = positive_int(n, "n")
        with self._capture_io_condition:
            if self._capture_terminalizing:
                raise AcquisitionCancelled("qCMOS capture is terminalizing")
            self._active_grabs += 1
        try:
            return self._grab_owned(
                requested,
                timeout=timeout,
                stop=stop,
                exact=exact,
            )
        finally:
            with self._capture_io_condition:
                self._active_grabs -= 1
                self._capture_io_condition.notify_all()

    def _grab_owned(
        self,
        n: int,
        *,
        timeout: float | None = None,
        stop=None,
        exact: bool = False,
    ) -> bool:
        """Wait for externally triggered frames, then drain every frame already reported by
        DCAM into the bounded base-class queue.  The caller may request only one frame, but a
        single ready event can represent a larger backlog; leaving that backlog in DCAM would
        incorrectly depend on a second ready event for frames that already exist.

        The qCMOS fault model is LOUD: a trigger that never
        comes raises ``TimeoutError`` after ``timeout`` (seconds; the config ``timeout_ms``
        default), and a Stop raises :class:`AcquisitionCancelled` within one poll slice.

        ``timeout`` bounds the complete call, including every ready wait and every frame copied
        from an already reported backlog.  A Stop event shortens the wait poll slice and is also
        checked between copies; neither path may extend the advertised blocking-call bound."""
        dcam = self._dcam
        timeout_ms = self.config.timeout_ms if timeout is None else max(1, int(float(timeout) * 1000))
        # When a live logic node passes a stop event, wait in short slices and check it between
        # slices so Stop interrupts a wedged trigger within ~one slice instead of blocking the
        # complete call budget.  Without a stop event, one wait may use the remaining budget.
        poll_ms = min(timeout_ms, 200) if stop is not None else timeout_ms
        deadline = time.monotonic() + timeout_ms / 1000.0
        last_observed_available = max(
            self._next_frame,
            self._last_transfer_count,
        )
        last_observed_newest = self._last_transfer_newest_index
        ready_without_count = False

        def transfer_window() -> tuple[int, int | None]:
            nonlocal last_observed_available, last_observed_newest, ready_without_count
            info = dcam.cap_transferinfo()
            if info is False:
                if exact:
                    raise RuntimeError(
                        "qCMOS cap_transferinfo failed during exact capture; "
                        f"produced-count/overrun state is unknown: {dcam.lasterr()}"
                    )
                if ready_without_count:
                    ready_without_count = False
                    return self._next_frame + 1, None
                return self._next_frame, None
            available = int(info.nFrameCount)
            newest = int(info.nNewestFrameIndex)
            if available < 0:
                raise RuntimeError("qCMOS nFrameCount is negative")
            if available < last_observed_available:
                raise RuntimeError(
                    "qCMOS nFrameCount moved backwards during capture: "
                    f"{available} < prior observation {last_observed_available}"
                )
            if available > 0 and not 0 <= newest < int(self._buf_alloc):
                raise RuntimeError(
                    "qCMOS newest frame index is outside the allocated ring: "
                    f"{newest} not in [0, {int(self._buf_alloc)})"
                )
            if (
                available > 0
                and last_observed_available > 0
                and last_observed_newest is not None
            ):
                expected_newest = (
                    last_observed_newest
                    + available
                    - last_observed_available
                ) % int(self._buf_alloc)
                if newest != expected_newest:
                    raise RuntimeError(
                        "qCMOS transfer count/newest-index pair is inconsistent: "
                        f"count advanced {available - last_observed_available} but "
                        f"newest index is {newest}, expected {expected_newest}"
                    )
            last_observed_available = available
            last_observed_newest = newest if available > 0 else None
            self._last_transfer_count = available
            self._last_transfer_newest_index = last_observed_newest
            ready_without_count = False
            return available, newest

        while True:
            with self._capture_io_condition:
                if self._capture_terminalizing:
                    raise AcquisitionCancelled("qCMOS capture is terminalizing")
            if stop is not None and stop.is_set():
                raise AcquisitionCancelled(f"qCMOS cancelled while waiting for frame {self._next_frame}.")
            available, newest = transfer_window()
            if available == self._next_frame:
                remaining_ms = int((deadline - time.monotonic()) * 1000)
                if remaining_ms <= 0:
                    raise TimeoutError(
                        f"qCMOS timed out after {timeout_ms} ms waiting for frame "
                        f"{self._next_frame}."
                    )
                slice_ms = max(1, min(poll_ms, remaining_ms))
                if dcam.wait_capevent_frameready(slice_ms) is False:
                    if stop is not None and stop.is_set():
                        raise AcquisitionCancelled(
                            f"qCMOS cancelled while waiting for frame {self._next_frame}."
                        )
                    if time.monotonic() < deadline:
                        continue
                    raise TimeoutError(
                        f"qCMOS timed out after {timeout_ms} ms waiting for frame "
                        f"{self._next_frame}."
                    )
                ready_without_count = True
                continue
            # Ring-overrun guard: the DCAM buffer holds ``_buf_alloc`` frames; if the camera has
            # produced more than that beyond our read cursor, the hardware has already overwritten
            # the slot we are about to read.
            # Fail LOUD with the dropped-frame count rather than silently deliver mis-ordered frames.
            if available - self._next_frame > int(self._buf_alloc):
                dropped = available - self._next_frame - int(self._buf_alloc)
                raise RuntimeError(
                    f"qCMOS ring overrun: {dropped} frame(s) overwritten -- the camera produced "
                    f"{available} frames but the {self._buf_alloc}-deep buffer was read only up to "
                    f"frame {self._next_frame}. Read frames faster or allocate a deeper ring.")
            snapshot_available = available
            snapshot_newest = newest
            while self._next_frame < snapshot_available:
                if stop is not None and stop.is_set():
                    raise AcquisitionCancelled(
                        f"qCMOS cancelled while draining frame {self._next_frame}."
                    )
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"qCMOS exceeded its {timeout_ms} ms read budget while draining "
                        f"frame {self._next_frame}."
                    )
                if snapshot_newest is None:
                    # Only a non-authoritative preview whose count read failed may
                    # use the historical next-slot fallback after a ready event.
                    ring_index = self._next_frame % int(self._buf_alloc)
                else:
                    distance_from_newest = (
                        snapshot_available - 1 - self._next_frame
                    )
                    ring_index = (
                        snapshot_newest - distance_from_newest
                    ) % int(self._buf_alloc)
                data = dcam.buf_getframedata(ring_index)
                if data is False:
                    raise RuntimeError(
                        f"qCMOS buf_getframedata({ring_index}) failed for source frame "
                        f"{self._next_frame}: {dcam.lasterr()}"
                    )
                frame_info, pixels = data
                timestamp = getattr(frame_info, "timestamp", None)
                if stop is not None and stop.is_set():
                    raise AcquisitionCancelled(
                        f"qCMOS cancelled after copying frame {self._next_frame}."
                    )
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"qCMOS exceeded its {timeout_ms} ms read budget while copying "
                        f"frame {self._next_frame}."
                    )

                if exact:
                    post_copy_available, _post_copy_newest = transfer_window()
                    if post_copy_available - self._next_frame > int(self._buf_alloc):
                        raise RuntimeError(
                            "qCMOS ring advanced far enough to overwrite the frame "
                            f"being copied at source ordinal {self._next_frame}"
                        )
                def metadata_int(owner, name):
                    value = getattr(owner, name, None) if owner is not None else None
                    return None if value is None else int(value)

                # CameraFrameRecord takes the final owner-owned ndarray copy.  It may be
                # materially slower than the SDK copy for a full sensor frame, so construct
                # it before the final deadline/Stop decision.  Commit and terminalization
                # then linearize under one short adapter-owned condition.
                record = CameraFrameRecord(
                    image=np.asarray(pixels),
                    source_ordinal=self._next_frame,
                    produced_count=available,
                    frame_stamp=metadata_int(frame_info, "framestamp"),
                    camera_stamp=metadata_int(frame_info, "camerastamp"),
                    timestamp_seconds=metadata_int(timestamp, "sec"),
                    timestamp_microseconds=metadata_int(timestamp, "microsec"),
                    host_received_at_ns=time.time_ns(),
                    driver_buffer_index=ring_index,
                )
                with self._capture_io_condition:
                    if self._capture_terminalizing:
                        raise AcquisitionCancelled(
                            f"qCMOS terminalized before frame {self._next_frame} commit"
                        )
                    if stop is not None and stop.is_set():
                        raise AcquisitionCancelled(
                            f"qCMOS cancelled before committing frame {self._next_frame}."
                        )
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            f"qCMOS exceeded its {timeout_ms} ms read budget before committing "
                            f"frame {self._next_frame}."
                        )
                    self._deliver_records([record])
                    self._next_frame += 1
            # The hook promises at least one newly queued frame, not the caller's
            # complete cardinality.  Return after this bounded driver snapshot so
            # the base owner can consume pending before another burst arrives.
            return True

    def _disarm(self) -> None:
        self._finish_record_capture()

    def _finish_record_capture(self) -> CameraCaptureTerminalRecord:
        """Stop DCAM, join every reader, freeze final count, then release the ring."""

        dcam = self._dcam
        if dcam is None:
            return CameraCaptureTerminalRecord(0, True, True, True)
        with self._capture_io_condition:
            self._capture_terminalizing = True
        try:
            stop_ok = dcam.cap_stop() is True
        except BaseException as exc:
            raise RuntimeError(
                "qCMOS cap_stop raised; active driver ring is retained"
            ) from exc
        if not stop_ok:
            raise RuntimeError(
                "qCMOS cap_stop was not acknowledged; active driver ring is retained"
            )

        # A cross-thread interrupt may issue cap_stop to unblock the SDK, but it
        # must never release storage while dcambuf_copyframe/transferinfo is still
        # executing.  The terminal owner waits for the sole read boundary to join.
        with self._capture_io_condition:
            while self._active_grabs:
                self._capture_io_condition.wait()

        def final_snapshot() -> tuple[int, int]:
            info = dcam.cap_transferinfo()
            if info is False:
                raise RuntimeError(
                    f"qCMOS final cap_transferinfo failed: {dcam.lasterr()}"
                )
            count = int(info.nFrameCount)
            newest = int(info.nNewestFrameIndex)
            if count < 0:
                raise RuntimeError("qCMOS final nFrameCount is negative")
            if count > 0 and not 0 <= newest < int(self._buf_alloc):
                raise RuntimeError(
                    "qCMOS final newest frame index is outside the allocated ring"
                )
            return count, newest

        transfer_error: BaseException | None = None
        produced_count: int | None = None
        newest_index: int | None = None
        try:
            first = final_snapshot()
            second = final_snapshot()
            if second != first:
                raise RuntimeError(
                    "qCMOS transfer count/newest index did not stabilize after cap_stop"
                )
            produced_count, newest_index = second
            if produced_count < max(
                self._next_frame,
                self._last_transfer_count,
            ):
                raise RuntimeError(
                    "qCMOS final nFrameCount moved backwards from the armed epoch"
                )
            if (
                produced_count > 0
                and self._last_transfer_count > 0
                and self._last_transfer_newest_index is not None
            ):
                expected_newest = (
                    self._last_transfer_newest_index
                    + produced_count
                    - self._last_transfer_count
                ) % int(self._buf_alloc)
                if newest_index != expected_newest:
                    raise RuntimeError(
                        "qCMOS final count/newest-index pair is inconsistent"
                    )
        except BaseException as exc:
            transfer_error = exc

        try:
            release_ok = dcam.buf_release() is True
        except BaseException as exc:
            raise RuntimeError("qCMOS buf_release raised after cap_stop") from exc
        if not release_ok:
            raise RuntimeError("qCMOS buf_release was not acknowledged after cap_stop")
        if transfer_error is not None:
            raise RuntimeError(
                "qCMOS final transfer state was not stable after cap_stop"
            ) from transfer_error
        assert produced_count is not None
        self._last_transfer_count = produced_count
        self._last_transfer_newest_index = (
            newest_index if produced_count > 0 else None
        )
        return CameraCaptureTerminalRecord(produced_count, True, True, True)

    def _set_prop(self, idprop, value, name: str, *, verify: bool = False) -> None:
        """Write a DCAM property and FAIL LOUDLY if the camera rejects it.

        ``prop_setvalue`` returns False (and sets ``lasterr``) on a rejected
        write, so an unchecked write leaves the camera silently in its prior
        state -- e.g. a rejected external-trigger write keeps internal trigger,
        and the next acquire just times out with a confusing message.  ``verify``
        additionally reads the value back (critical trigger props) so a clamp /
        silent substitution also fails here, not at first light."""
        dcam = self._dcam
        if dcam.prop_setvalue(idprop, value) is False:
            raise RuntimeError(f"qCMOS rejected {name} = {value}: {dcam.lasterr()}")
        if verify:
            read = dcam.prop_getvalue(idprop)
            if read is False:
                raise RuntimeError(f"qCMOS could not read back {name}: {dcam.lasterr()}")
            # DCAM round-trips properties as doubles; compare numerically with a
            # tolerance, falling back to equality for non-numeric enum sentinels.
            try:
                mismatch = abs(float(read) - float(value)) > 1e-9
            except (TypeError, ValueError):
                mismatch = read != value
            if mismatch:
                raise RuntimeError(f"qCMOS {name} read back {read!r}, expected {value!r} "
                                   "(property clamped/substituted by the camera).")

    def _clear_applied_working_point(self) -> None:
        """Invalidate the one cached hardware working-point observation."""

        self._applied_roi = None
        self._applied_exposure_seconds = None
        self._required_external_trigger_interval_seconds = None
        self._applied_readout_speed = None
        self._applied_sensor_mode = None
        self._applied_trigger_global_exposure = None
        self._applied_trigger_source = None
        self._applied_trigger_active = None
        self._applied_trigger_polarity = None

    def _write_settings(self) -> None:
        mod = self._module
        # Invalidate the previous proof before touching any setting.  A failed
        # partial reconfiguration must not leave a stale working-point timing
        # snapshot available to a later capability probe.
        self._clear_applied_working_point()
        self._sensor_shape_yx = None
        self._set_prop(mod.DCAM_IDPROP.EXPOSURETIME, self.config.exposure, "exposure")
        # External rising-edge trigger is the imaging scheme -- verify the camera
        # actually accepted it (these are the writes that fail silently on a
        # mis-set / unsupported camera and then hang acquire).
        self._set_prop(mod.DCAM_IDPROP.TRIGGERSOURCE, mod.DCAMPROP.TRIGGERSOURCE.EXTERNAL, "trigger_source", verify=True)
        self._set_prop(mod.DCAM_IDPROP.TRIGGERACTIVE, mod.DCAMPROP.TRIGGERACTIVE.EDGE, "trigger_active", verify=True)
        self._set_prop(mod.DCAM_IDPROP.TRIGGERPOLARITY, mod.DCAMPROP.TRIGGERPOLARITY.POSITIVE, "trigger_polarity", verify=True)
        self._set_prop(mod.DCAM_IDPROP.READOUTSPEED, self.config.readout_speed, "readout_speed")
        if self.config.sensor_mode is not None:
            self._set_prop(mod.DCAM_IDPROP.SENSORMODE, self.config.sensor_mode, "sensor_mode", verify=True)
        if self.config.trigger_global_exposure is not None:
            self._set_prop(mod.DCAM_IDPROP.TRIGGER_GLOBALEXPOSURE, self.config.trigger_global_exposure, "trigger_global_exposure", verify=True)
        subarray_grid = self._subarray_grid()
        _step, max_w, max_h = subarray_grid
        if max_w > 0 and max_h > 0:
            self._sensor_shape_yx = (int(max_h), int(max_w))
        self._apply_subarray(subarray_grid)
        try:
            self._read_applied_working_point()
        except BaseException:
            # `_apply_subarray` records its immediate set/get result for UI use,
            # but no part of a failed final working-point observation is valid
            # capability evidence.
            self._clear_applied_working_point()
            raise

    def _read_applied_working_point(self) -> None:
        """Freeze one atomic readback of the fully applied DCAM working point."""

        mod, dcam = self._module, self._dcam
        values: dict[str, float] = {}
        for name, idprop in (
            ("applied exposure", mod.DCAM_IDPROP.EXPOSURETIME),
            (
                "minimum external trigger interval",
                mod.DCAM_IDPROP.TIMING_MINTRIGGERINTERVAL,
            ),
            ("applied readout speed", mod.DCAM_IDPROP.READOUTSPEED),
            ("applied sensor mode", mod.DCAM_IDPROP.SENSORMODE),
            (
                "applied trigger global exposure",
                mod.DCAM_IDPROP.TRIGGER_GLOBALEXPOSURE,
            ),
            ("applied trigger source", mod.DCAM_IDPROP.TRIGGERSOURCE),
            ("applied trigger active", mod.DCAM_IDPROP.TRIGGERACTIVE),
            ("applied trigger polarity", mod.DCAM_IDPROP.TRIGGERPOLARITY),
        ):
            raw = dcam.prop_getvalue(idprop)
            if raw is False:
                raise RuntimeError(f"qCMOS could not read back {name}: {dcam.lasterr()}")
            try:
                value = float(raw)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(f"qCMOS returned a non-numeric {name}: {raw!r}") from exc
            if not math.isfinite(value) or value < 0:
                raise RuntimeError(f"qCMOS returned an invalid {name}: {raw!r}")
            values[name] = value
        if values["applied exposure"] <= 0:
            raise RuntimeError("qCMOS returned a non-positive applied exposure")
        if values["minimum external trigger interval"] <= 0:
            raise RuntimeError(
                "qCMOS returned a non-positive minimum external trigger interval"
            )
        applied_modes = {}
        for name in (
            "applied readout speed",
            "applied sensor mode",
            "applied trigger global exposure",
        ):
            value = values[name]
            if value <= 0:
                raise RuntimeError(f"qCMOS returned a non-positive {name}")
            if not value.is_integer():
                raise RuntimeError(
                    f"qCMOS returned a non-integral {name}: {value!r}"
                )
            applied_modes[name] = int(value)

        expected_trigger_modes = {
            "applied trigger source": mod.DCAMPROP.TRIGGERSOURCE.EXTERNAL,
            "applied trigger active": mod.DCAMPROP.TRIGGERACTIVE.EDGE,
            "applied trigger polarity": mod.DCAMPROP.TRIGGERPOLARITY.POSITIVE,
        }
        applied_trigger_modes: dict[str, int] = {}
        for name, expected in expected_trigger_modes.items():
            value = values[name]
            if not value.is_integer():
                raise RuntimeError(f"qCMOS returned a non-integral {name}: {value!r}")
            applied = int(value)
            if applied != int(expected):
                raise RuntimeError(
                    f"qCMOS {name} read back {applied!r}, expected {int(expected)!r}"
                )
            applied_trigger_modes[name] = applied

        subarray_mode_raw = dcam.prop_getvalue(mod.DCAM_IDPROP.SUBARRAYMODE)
        if subarray_mode_raw is False:
            raise RuntimeError(
                f"qCMOS could not read back applied subarray mode: {dcam.lasterr()}"
            )
        try:
            subarray_mode_value = float(subarray_mode_raw)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"qCMOS returned a non-numeric applied subarray mode: {subarray_mode_raw!r}"
            ) from exc
        if not math.isfinite(subarray_mode_value) or not subarray_mode_value.is_integer():
            raise RuntimeError(
                f"qCMOS returned an invalid applied subarray mode: {subarray_mode_raw!r}"
            )
        subarray_mode = int(subarray_mode_value)
        if subarray_mode == int(mod.DCAMPROP.MODE.OFF):
            applied_roi = None
        elif subarray_mode == int(mod.DCAMPROP.MODE.ON):
            roi_values: dict[str, int] = {}
            for name, idprop in (
                ("subarray_hpos", mod.DCAM_IDPROP.SUBARRAYHPOS),
                ("subarray_hsize", mod.DCAM_IDPROP.SUBARRAYHSIZE),
                ("subarray_vpos", mod.DCAM_IDPROP.SUBARRAYVPOS),
                ("subarray_vsize", mod.DCAM_IDPROP.SUBARRAYVSIZE),
            ):
                raw = dcam.prop_getvalue(idprop)
                if raw is False:
                    raise RuntimeError(f"qCMOS could not read back {name}: {dcam.lasterr()}")
                try:
                    numeric = float(raw)
                except (TypeError, ValueError) as exc:
                    raise RuntimeError(
                        f"qCMOS returned a non-numeric {name}: {raw!r}"
                    ) from exc
                if not math.isfinite(numeric) or numeric < 0 or not numeric.is_integer():
                    raise RuntimeError(f"qCMOS returned an invalid {name}: {raw!r}")
                roi_values[name] = int(numeric)
            applied_roi = (
                roi_values["subarray_hpos"],
                roi_values["subarray_hsize"],
                roi_values["subarray_vpos"],
                roi_values["subarray_vsize"],
            )
            if applied_roi[1] <= 0 or applied_roi[3] <= 0:
                raise RuntimeError("qCMOS returned a non-positive applied subarray size")
            sensor_shape = self._sensor_shape_yx
            if sensor_shape is None:
                raise RuntimeError("qCMOS sensor geometry is unavailable during ROI readback")
            if (
                applied_roi[0] + applied_roi[1] > sensor_shape[1]
                or applied_roi[2] + applied_roi[3] > sensor_shape[0]
            ):
                raise RuntimeError("qCMOS applied subarray lies outside the sensor")
        else:
            raise RuntimeError(
                f"qCMOS applied subarray mode {subarray_mode!r} is neither OFF nor ON"
            )

        self._applied_exposure_seconds = values["applied exposure"]
        self._required_external_trigger_interval_seconds = values[
            "minimum external trigger interval"
        ]
        self._applied_readout_speed = applied_modes["applied readout speed"]
        self._applied_sensor_mode = applied_modes["applied sensor mode"]
        self._applied_trigger_global_exposure = applied_modes[
            "applied trigger global exposure"
        ]
        self._applied_trigger_source = applied_trigger_modes[
            "applied trigger source"
        ]
        self._applied_trigger_active = applied_trigger_modes[
            "applied trigger active"
        ]
        self._applied_trigger_polarity = applied_trigger_modes[
            "applied trigger polarity"
        ]
        self._applied_roi = applied_roi

    def _subarray_grid(self) -> tuple[int, int, int]:
        """The camera's valid sub-array grid (step, max_w, max_h) queried with
        SUBARRAYMODE OFF, where HSIZE/VSIZE max report the FULL sensor.  Falls back
        to a step of 4 (the qCMOS hardware requirement) if the attribute query
        is unavailable, so a request is always snapped to a legal window."""
        mod, dcam = self._module, self._dcam
        self._set_prop(
            mod.DCAM_IDPROP.SUBARRAYMODE,
            mod.DCAMPROP.MODE.OFF,
            "subarray_mode",
            verify=True,
        )

        def _attr(idprop, field, default):
            attr = dcam.prop_getattr(idprop)
            if attr is False:
                return default
            value = getattr(attr, field, 0.0)
            return int(value) if value else default

        step = _attr(mod.DCAM_IDPROP.SUBARRAYHSIZE, "valuestep", 4) or 4
        max_w = _attr(mod.DCAM_IDPROP.SUBARRAYHSIZE, "valuemax", 0)
        max_h = _attr(mod.DCAM_IDPROP.SUBARRAYVSIZE, "valuemax", 0)
        return step, max_w, max_h

    def _set_get_prop(self, idprop, value, name: str) -> int:
        """Write a sub-array property and return the value the camera ACTUALLY
        applied (DCAM ``prop_setgetvalue`` snaps to the hardware grid and reports
        it back), failing loudly on rejection -- so a clamped ROI is observed here
        rather than silently imaging the wrong region."""
        actual = self._dcam.prop_setgetvalue(idprop, value)
        if actual is False:
            raise RuntimeError(f"qCMOS rejected {name} = {value}: {self._dcam.lasterr()}")
        return int(round(float(actual)))

    def _apply_subarray(
        self,
        grid: tuple[int, int, int] | None = None,
    ) -> None:
        """Apply the requested ROI as a valid sub-array and record what the camera
        actually reads out.  The plot/acquisition layer hands a window in raw
        sensor-pixel coordinates; here it is SNAPPED to the camera's grid
        (``snap_subarray``) and written in a SAFE ORDER (positions to 0 first, then
        sizes, then positions) so an intermediate ``pos+size`` never exceeds the
        sensor, with SUBARRAYMODE ON last.  ``prop_setgetvalue`` gives the camera
        the final say; the read-back tuple becomes ``self._applied_roi``."""
        mod = self._module
        if self.config.roi is None:
            self._set_prop(
                mod.DCAM_IDPROP.SUBARRAYMODE,
                mod.DCAMPROP.MODE.OFF,
                "subarray_mode",
                verify=True,
            )
            self._applied_roi = None
            return
        step, max_w, max_h = self._subarray_grid() if grid is None else grid
        x, width, y, height = snap_subarray(
            self.config.roi, step=step,
            max_w=max_w or self.config.roi[0] + self.config.roi[1],
            max_h=max_h or self.config.roi[2] + self.config.roi[3],
        )
        self._set_get_prop(mod.DCAM_IDPROP.SUBARRAYHPOS, 0, "subarray_hpos")
        self._set_get_prop(mod.DCAM_IDPROP.SUBARRAYVPOS, 0, "subarray_vpos")
        applied_w = self._set_get_prop(mod.DCAM_IDPROP.SUBARRAYHSIZE, width, "subarray_hsize")
        applied_h = self._set_get_prop(mod.DCAM_IDPROP.SUBARRAYVSIZE, height, "subarray_vsize")
        applied_x = self._set_get_prop(mod.DCAM_IDPROP.SUBARRAYHPOS, x, "subarray_hpos")
        applied_y = self._set_get_prop(mod.DCAM_IDPROP.SUBARRAYVPOS, y, "subarray_vpos")
        self._set_prop(
            mod.DCAM_IDPROP.SUBARRAYMODE,
            mod.DCAMPROP.MODE.ON,
            "subarray_mode",
            verify=True,
        )
        self._applied_roi = (applied_x, applied_w, applied_y, applied_h)

    def close(self) -> None:
        if self._dcam is not None:
            try:
                self._dcam.dev_close()
            finally:
                self._dcam = None
                self._sensor_shape_yx = None
                self._clear_applied_working_point()
        if self._api is not None:
            # Drop THIS camera's reference; the runtime is uninited only once the LAST holder
            # closes -- so closing one of two qCMOS cameras leaves the other's DCAM handle live.
            try:
                _dcam_release(self._api)
            finally:
                self._api = None
        self._module = None

    def stop(self) -> None:
        if self._dcam is not None:
            try:
                self._dcam.cap_stop()
            except Exception:
                pass

    def snapshot(self) -> dict[str, object]:
        out = super().snapshot()          # the ``type`` key has ONE producer: BaseDevice.snapshot
        out.update({
            "exposure": self.config.exposure,
            "applied_exposure_seconds": self._applied_exposure_seconds,
            "required_external_trigger_interval_seconds": (
                self._required_external_trigger_interval_seconds
            ),
            "applied_readout_speed": self._applied_readout_speed,
            "applied_sensor_mode": self._applied_sensor_mode,
            "applied_trigger_global_exposure": (
                self._applied_trigger_global_exposure
            ),
            "applied_trigger_source": self._applied_trigger_source,
            "applied_trigger_active": self._applied_trigger_active,
            "applied_trigger_polarity": self._applied_trigger_polarity,
            "readout_speed": self.config.readout_speed,
            "roi": self.roi,          # the region truly being imaged (hardware-snapped when open)
            "device_index": self.config.device_index,
            "timeout_ms": self.config.timeout_ms,
            "open": self._dcam is not None,
        })
        return out


def normalize_roi(roi: Sequence[int]) -> tuple[int, int, int, int]:
    try:
        raw = tuple(roi)
    except TypeError as exc:
        raise ValueError("roi must be (x, width, y, height).") from exc
    if len(raw) != 4:
        raise ValueError("roi must be (x, width, y, height).")
    out = tuple(nonnegative_int(v, f"roi[{i}]") for i, v in enumerate(raw))
    if out[1] <= 0 or out[3] <= 0:
        raise ValueError("roi width and height must be positive.")
    return out


__all__ = ["DEFAULT_DCAM_MODULE", "QCMOSCamera", "QCMOSConfig", "normalize_roi"]
