"""Camera boundary for the lightweight neutral-atom session."""

from __future__ import annotations

from dataclasses import dataclass, fields
import importlib
import threading
import time
from typing import Any, Sequence

import numpy as np

from ..core.analysis import nonnegative_int, positive_float, positive_int
from ..timing import DEFAULT_EXPOSURE_S
from .base import (
    ROI_CLEAR_SENTINELS,
    AcquisitionCancelled,
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
        self.readout_speed = nonnegative_int(self.readout_speed, "readout_speed")
        self.device_index = nonnegative_int(self.device_index, "device_index")
        self.timeout_ms = positive_int(self.timeout_ms, "timeout_ms")
        if self.roi is not None:
            self.roi = normalize_roi(self.roi)
        if self.sensor_mode is not None:
            self.sensor_mode = nonnegative_int(self.sensor_mode, "sensor_mode")
        if self.trigger_global_exposure is not None:
            self.trigger_global_exposure = nonnegative_int(self.trigger_global_exposure, "trigger_global_exposure")


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
        # Armed-session bookkeeping for the acquisition hooks (_arm/_grab/_disarm):
        # the DCAM buffer depth of the current session + the next frame index to transfer.
        self._buf_alloc = 0
        self._next_frame = 0

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
        try:
            _step, max_w, max_h = self._subarray_grid()
        except Exception:
            return None
        return (int(max_h), int(max_w)) if max_w and max_h else None

    def configure(self, *, exposure: float | None = None, readout_speed: int | None = None, roi: Sequence[int] | None | object = None, **kwargs) -> None:
        self._reject_unknown_configure_keys({"exposure", "readout_speed", "roi"}, kwargs)
        if exposure is not None:
            self.config.exposure = self._validated_exposure(exposure)
        if readout_speed is not None:
            self.config.readout_speed = nonnegative_int(readout_speed, "readout_speed")
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
        ``configure(readout_speed=...)`` (validated by ``nonnegative_int``, pushed to open hardware via
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

    def _arm(self, frames: int | None) -> None:
        """Allocate the DCAM frame buffer and start capture -- the hardware is then armed and
        waiting for FPGA trigger edges.  ``frames=None`` (continuous) allocates a
        ``recent_capacity``-deep ring the sequence capture cycles through."""
        dcam = self._dcam                        # arm() ensured the DCAM handle is open (base ensure_open)
        alloc = int(frames) if frames is not None else max(1, int(self.recent_capacity))
        if dcam.buf_alloc(alloc) is False:
            raise RuntimeError(f"qCMOS buf_alloc({alloc}) failed: {dcam.lasterr()}")
        try:
            if dcam.cap_start(bSequence=True) is False:
                raise RuntimeError(f"qCMOS cap_start failed: {dcam.lasterr()}")
        except BaseException:
            try:
                dcam.buf_release()
            except Exception:
                pass
            raise
        self._buf_alloc = alloc
        self._next_frame = 0

    def _grab(self, n: int, *, timeout: float | None = None, stop=None) -> bool:
        """Wait for and transfer up to ``n`` externally triggered frames from the DCAM buffer
        into the base-class frame queue.  The qCMOS fault model is LOUD: a trigger that never
        comes raises ``TimeoutError`` after ``timeout`` (seconds; the config ``timeout_ms``
        default), and a Stop raises :class:`AcquisitionCancelled` within one poll slice.

        ``timeout`` is the budget PER AWAITED TRIGGER, not for the whole ``n``-frame read: it
        resets each time a frame arrives.  So an FPGA firing ``n`` triggers spaced under the
        timeout succeeds the same whether or not a Stop event is passed -- the stop event only
        shortens the POLL SLICE (so a Stop is honoured within ~one slice), never the trigger
        budget itself."""
        dcam = self._dcam
        n = positive_int(n, "n")
        timeout_ms = self.config.timeout_ms if timeout is None else max(1, int(float(timeout) * 1000))
        # When a live logic node passes a stop event, wait in short slices and check it between
        # slices so Stop interrupts a wedged trigger within ~one slice instead of blocking the
        # full per-trigger timeout.  Without a stop event, one wait of the full timeout.
        poll_ms = min(timeout_ms, 200) if stop is not None else timeout_ms
        got = 0
        deadline = time.monotonic() + timeout_ms / 1000.0
        while got < n:
            if stop is not None and stop.is_set():
                raise AcquisitionCancelled(f"qCMOS cancelled while waiting for frame {self._next_frame}.")
            slice_ms = poll_ms
            if stop is not None:
                slice_ms = max(1, min(poll_ms, int((deadline - time.monotonic()) * 1000)))
            if dcam.wait_capevent_frameready(slice_ms) is False:
                if stop is not None and not stop.is_set() and time.monotonic() < deadline:
                    continue  # slice expired but neither stopped nor timed out -- keep polling
                if stop is not None and stop.is_set():
                    raise AcquisitionCancelled(f"qCMOS cancelled while waiting for frame {self._next_frame}.")
                raise TimeoutError(f"qCMOS timed out after {timeout_ms} ms waiting for frame {self._next_frame}.")
            info = dcam.cap_transferinfo()
            available = int(getattr(info, "nFrameCount", self._next_frame + 1)) if info is not False else self._next_frame + 1
            # Ring-overrun guard: the DCAM buffer holds ``_buf_alloc`` frames; if the camera has
            # produced more than that beyond our read cursor, the hardware has already overwritten
            # the slot we are about to read (``next % buf_alloc`` now points at a NEWER frame).
            # Fail LOUD with the dropped-frame count rather than silently deliver mis-ordered frames.
            if available - self._next_frame > int(self._buf_alloc):
                dropped = available - self._next_frame - int(self._buf_alloc)
                raise RuntimeError(
                    f"qCMOS ring overrun: {dropped} frame(s) overwritten -- the camera produced "
                    f"{available} frames but the {self._buf_alloc}-deep buffer was read only up to "
                    f"frame {self._next_frame}. Read frames faster or allocate a deeper ring.")
            while self._next_frame < available and got < n:
                # A continuous session cycles the allocated ring; a finite one indexes 0..N-1.
                data = dcam.buf_getframedata(self._next_frame % max(1, int(self._buf_alloc)))
                if data is False:
                    raise RuntimeError(f"qCMOS buf_getframedata({self._next_frame}) failed: {dcam.lasterr()}")
                frame_info, pixels = data
                timestamp = getattr(frame_info, "timestamp", None)

                def metadata_int(owner, name):
                    value = getattr(owner, name, None) if owner is not None else None
                    return None if value is None else int(value)

                ring_index = self._next_frame % max(1, int(self._buf_alloc))
                self._deliver_records(
                    [
                        CameraFrameRecord(
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
                    ]
                )
                self._next_frame += 1
                got += 1
                deadline = time.monotonic() + timeout_ms / 1000.0   # per-trigger budget resets on delivery
        return True

    def _disarm(self) -> None:
        dcam = self._dcam
        if dcam is None:
            return
        stop_ok = False
        release_ok = False
        try:
            stop_ok = dcam.cap_stop() is True
        finally:
            release_ok = dcam.buf_release() is True
        if not stop_ok or not release_ok:
            failed = []
            if not stop_ok:
                failed.append("cap_stop")
            if not release_ok:
                failed.append("buf_release")
            raise RuntimeError(
                "qCMOS disarm did not acknowledge " + " and ".join(failed)
            )

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

    def _write_settings(self) -> None:
        mod = self._module
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
        self._apply_subarray()

    def _subarray_grid(self) -> tuple[int, int, int, int]:
        """The camera's valid sub-array grid (step, max_w, max_h) queried with
        SUBARRAYMODE OFF, where HSIZE/VSIZE max report the FULL sensor.  Falls back
        to a step of 4 (the qCMOS hardware requirement) if the attribute query
        is unavailable, so a request is always snapped to a legal window."""
        mod, dcam = self._module, self._dcam
        self._set_prop(mod.DCAM_IDPROP.SUBARRAYMODE, mod.DCAMPROP.MODE.OFF, "subarray_mode")

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

    def _apply_subarray(self) -> None:
        """Apply the requested ROI as a valid sub-array and record what the camera
        actually reads out.  The plot/acquisition layer hands a window in raw
        sensor-pixel coordinates; here it is SNAPPED to the camera's grid
        (``snap_subarray``) and written in a SAFE ORDER (positions to 0 first, then
        sizes, then positions) so an intermediate ``pos+size`` never exceeds the
        sensor, with SUBARRAYMODE ON last.  ``prop_setgetvalue`` gives the camera
        the final say; the read-back tuple becomes ``self._applied_roi``."""
        mod = self._module
        if self.config.roi is None:
            self._set_prop(mod.DCAM_IDPROP.SUBARRAYMODE, mod.DCAMPROP.MODE.OFF, "subarray_mode")
            self._applied_roi = None
            return
        step, max_w, max_h = self._subarray_grid()
        x, width, y, height = snap_subarray(
            self.config.roi, step=step,
            max_w=max_w or self.config.roi[0] + self.config.roi[1],
            max_h=max_h or self.config.roi[2] + self.config.roi[3],
        )
        self._set_prop(mod.DCAM_IDPROP.SUBARRAYMODE, mod.DCAMPROP.MODE.ON, "subarray_mode")
        self._set_get_prop(mod.DCAM_IDPROP.SUBARRAYHPOS, 0, "subarray_hpos")
        self._set_get_prop(mod.DCAM_IDPROP.SUBARRAYVPOS, 0, "subarray_vpos")
        applied_w = self._set_get_prop(mod.DCAM_IDPROP.SUBARRAYHSIZE, width, "subarray_hsize")
        applied_h = self._set_get_prop(mod.DCAM_IDPROP.SUBARRAYVSIZE, height, "subarray_vsize")
        applied_x = self._set_get_prop(mod.DCAM_IDPROP.SUBARRAYHPOS, x, "subarray_hpos")
        applied_y = self._set_get_prop(mod.DCAM_IDPROP.SUBARRAYVPOS, y, "subarray_vpos")
        self._applied_roi = (applied_x, applied_w, applied_y, applied_h)

    def close(self) -> None:
        if self._dcam is not None:
            try:
                self._dcam.dev_close()
            finally:
                self._dcam = None
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
