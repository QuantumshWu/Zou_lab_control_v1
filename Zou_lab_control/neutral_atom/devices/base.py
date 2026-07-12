"""Base device contracts for hardware and virtual neutral-atom devices."""

from __future__ import annotations

import inspect
import re
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, replace
from typing import Any, Callable, Optional

import numpy as np

from ..core.analysis import positive_float, positive_int
from .camera_trigger import DEFAULT_CAMERA_TRIGGER_CHANNELS


@dataclass(frozen=True)
class RuntimeControl:
    """ONE live, tunable-at-runtime device attribute (see :meth:`BaseDevice.runtime_controls`).

    ``decl`` is the :class:`~..core.params.ParamDecl` the shared ``PARAM_WIDGETS`` registry
    builds the control widget from (so a runtime control renders / validates through the
    exact same typed-widget path as a config form -- never an ``eval`` of free text).
    ``getter`` reads the live value off the device (``(device) -> value``); it is polled to
    show the current value.  ``setter`` writes an edited value back (``(device, value) ->
    None``) with the value validated in the DEVICE'S OWN setter, or is ``None`` for a
    read-only read-back (the panel disables the control's editor)."""

    decl: Any                                        # a core.params.ParamDecl
    getter: Callable[[Any], Any]
    setter: Optional[Callable[[Any, Any], None]] = None

    def __post_init__(self) -> None:
        # A live device knob ALWAYS has a value (the getter seeds it) -- it is never the blank
        # "leave empty = use the default" state a measurement's optional arg offers, nor is it ever
        # the "missing required" a form marks with ``*``.  So normalise the decl HERE, at the base,
        # so a device author can never forget it and need not re-state it on each ParamDecl:
        #   * a numeric control renders as a scrollable spin box (``optional=False``, never a line edit);
        #   * ``required`` is cleared (no ``*`` on a control that is always populated).
        # An explicit ``optional`` on the decl is honoured (only the ``None`` = infer case is pinned).
        decl = self.decl
        patch = {}
        if getattr(decl, "kind", None) in ("float", "int") and getattr(decl, "optional", None) is None:
            patch["optional"] = False
        if getattr(decl, "required", False):
            patch["required"] = False
        if patch:
            object.__setattr__(self, "decl", replace(decl, **patch))

    @property
    def writable(self) -> bool:
        """True when this control can be WRITTEN (has a setter) -- the read-only-vs-editable
        fact a panel reads once, instead of re-testing ``setter is None`` at each call site."""
        return self.setter is not None


class DeviceProperty:
    """ONE declaration of a live, tunable device attribute -- our confocal ``ManagedProperty``.

    The descriptor IS the Python property (validated get / set) AND auto-registers into the device's
    runtime-control catalog, so the device viewer builds the widget with NO hand-written
    :class:`RuntimeControl`.  A device author declares the metadata ONCE -- the ``kind`` plus
    label / unit / bounds / choices -- and either lets the value AUTO-STORE in ``_<name>`` (seeded to
    ``default``) or attaches a getter / setter for a derived or routed value.  The base
    :meth:`BaseDevice.runtime_controls` DERIVES the whole catalog from every DeviceProperty on the
    class (MRO-merged, nearest class wins, declaration order), so ``runtime_controls`` is never
    hand-typed and the property + its GUI can never drift apart.

    ``kind`` selects the shared ``PARAM_WIDGETS`` widget: ``float`` / ``int`` -> a scrollable spin box
    (bounded by ``lo`` / ``hi``), ``bool`` -> a switch, ``choice`` -> a combo (of ``choices``),
    ``text`` -> a read-only display.  A property with NO setter -- a getter-only derived read-back
    (``on_d1``, ``frame_shape``) or an explicit ``read_only=True`` -- is a read-only control (the panel
    shows its Current line and no editor).  Validation lives here (``float`` / ``int`` clamp to the
    bounds like confocal; ``bool`` coerces; ``choice`` rejects a value outside ``choices``); a custom
    setter body may add stricter checks (a camera's ``exposure`` rejects a non-positive value).

    Usage (confocal ``@ManagedProperty`` style)::

        exposure = DeviceProperty("float", unit="s", lo=0.0, hi=1e6, tooltip="Frame exposure (s).")
        @exposure.getter
        def exposure(self): return float(self._exposure)
        @exposure.setter
        def exposure(self, value): self.configure(exposure=self._validated_exposure(value))

    or, for a plain stored knob, just a default and no getter / setter (auto-stored in ``_<name>``)::

        saturation = DeviceProperty("float", label="saturation I/Isat", lo=0.0, hi=100.0, default=3.0)
    """

    def __init__(self, kind, *, label=None, unit="", lo=0.0, hi=1e12, choices=(), default=None,
                 tooltip="", segmented=False, read_only=False):
        self._decl_kw = dict(label=label, kind=str(kind), unit=unit, lo=lo, hi=hi,
                             choices=tuple(choices), tooltip=tooltip, segmented=segmented)
        self.kind = str(kind)
        self.lo = lo
        self.hi = hi
        self.choices = tuple(choices)
        self.default = default
        self.read_only = bool(read_only)
        self.name: str | None = None
        self.fget: Optional[Callable[[Any], Any]] = None
        self.fset: Optional[Callable[[Any, Any], None]] = None

    # -- authoring: decorate the getter (confocal), or attach getter / setter explicitly --
    def __call__(self, fget):
        return self.getter(fget)

    def getter(self, fget):
        self.fget = fget
        return self

    def setter(self, fset):
        self.fset = fset
        self.read_only = False
        return self

    def __set_name__(self, owner, name):
        self.name = name
        if self._decl_kw["label"] is None:
            self._decl_kw["label"] = name
        own = owner.__dict__.get("_own_device_properties")
        if own is None:
            own = {}
            owner._own_device_properties = own
        own[name] = self          # dict preserves declaration order; runtime_controls MRO-merges

    # -- the property itself --
    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        if self.fget is not None:
            return self.fget(instance)
        return getattr(instance, f"_{self.name}", self.default)   # auto-storage, seeded to default

    def __set__(self, instance, value):
        if not self.writable:
            raise AttributeError(f"{self.name} is read-only")
        value = self._coerce(value)
        if self.fset is not None:
            self.fset(instance, value)
        else:
            object.__setattr__(instance, f"_{self.name}", value)

    def _coerce(self, value):
        if self.kind == "float":
            return min(max(float(value), float(self.lo)), float(self.hi))   # clamp to bounds (confocal)
        if self.kind == "int":
            return int(min(max(int(value), int(self.lo)), int(self.hi)))
        if self.kind == "bool":
            return bool(value)
        if self.kind == "choice":
            s = str(value)
            allowed = [str(c) for c in self.choices]
            if allowed and s not in allowed:
                raise ValueError(f"{self.name} must be one of {allowed}")
            return s
        return value

    @property
    def writable(self) -> bool:
        """A control is writable unless it is explicitly ``read_only`` or a getter-only derived
        read-back (a getter but no setter)."""
        return not self.read_only and not (self.fget is not None and self.fset is None)

    def build_decl(self):
        """The :class:`~..core.params.ParamDecl` the shared registry builds the widget from."""
        from ..core.params import ParamDecl
        return ParamDecl(key=self.name, **self._decl_kw)


def collect_device_properties(cls) -> "dict[str, DeviceProperty]":
    """Every :class:`DeviceProperty` on ``cls``, MRO-merged base-first so a subclass's declaration
    overrides a base's of the same name (keeping the base's display position) and new subclass knobs
    append after the inherited ones -- the single derivation :meth:`BaseDevice.runtime_controls` uses."""
    merged: "dict[str, DeviceProperty]" = {}
    for base in reversed(cls.__mro__):
        own = base.__dict__.get("_own_device_properties")
        if own:
            merged.update(own)
    return merged


class AcquisitionCancelled(Exception):
    """Raised by ``acquire`` when its optional ``stop`` event fires mid-wait.

    Distinct from ``TimeoutError`` (a real fault): cancellation is an
    intentional Stop, so the logic node loop treats it as a clean exit rather than a
    recorded error / banner."""


class CameraBufferOverrun(RuntimeError):
    """A captured frame could not be retained without overwriting authority data."""


@dataclass(frozen=True)
class CameraCaptureTerminalRecord:
    """Adapter observation after capture stop and before session ownership release."""

    produced_count: int
    source_stopped: bool
    no_more_frames: bool
    joined: bool

    def __post_init__(self) -> None:
        if isinstance(self.produced_count, bool) or not isinstance(
            self.produced_count, (int, np.integer)
        ):
            raise TypeError("produced_count must be a non-negative integer")
        produced = int(self.produced_count)
        if produced < 0:
            raise ValueError("produced_count must be a non-negative integer")
        object.__setattr__(self, "produced_count", produced)
        for field in ("source_stopped", "no_more_frames", "joined"):
            if type(getattr(self, field)) is not bool:
                raise TypeError(f"{field} must be bool")


@dataclass(frozen=True, eq=False)
class CameraFrameRecord:
    """One immutable frame plus the source metadata observed with it.

    ``source_ordinal`` is zero-based within the current arm epoch.  Hardware
    stamps are optional because not every camera exposes them; absence is explicit
    and never fabricated from array position.  ``produced_count`` is the source's
    cumulative count snapshot when the frame was drained, when available.
    """

    image: np.ndarray
    source_ordinal: int
    produced_count: int | None
    frame_stamp: int | None
    camera_stamp: int | None
    timestamp_seconds: int | None
    timestamp_microseconds: int | None
    host_received_at_ns: int
    driver_buffer_index: int | None = None
    __hash__ = None

    def __post_init__(self) -> None:
        if isinstance(self.source_ordinal, bool) or not isinstance(
            self.source_ordinal, (int, np.integer)
        ):
            raise TypeError("source_ordinal must be a non-negative integer")
        ordinal = int(self.source_ordinal)
        if ordinal < 0:
            raise ValueError("source_ordinal must be a non-negative integer")
        object.__setattr__(self, "source_ordinal", ordinal)
        for name in (
            "produced_count",
            "frame_stamp",
            "camera_stamp",
            "timestamp_seconds",
            "timestamp_microseconds",
            "driver_buffer_index",
        ):
            value = getattr(self, name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise TypeError(f"{name} must be an integer or None")
            value = int(value)
            if name in {
                "produced_count",
                "timestamp_seconds",
                "timestamp_microseconds",
                "driver_buffer_index",
            } and value < 0:
                raise ValueError(f"{name} must be non-negative")
            if name == "timestamp_microseconds" and value >= 1_000_000:
                raise ValueError("timestamp_microseconds must be less than 1_000_000")
            object.__setattr__(self, name, value)
        if isinstance(self.host_received_at_ns, bool) or not isinstance(
            self.host_received_at_ns, (int, np.integer)
        ):
            raise TypeError("host_received_at_ns must be a positive integer")
        received = int(self.host_received_at_ns)
        if received <= 0:
            raise ValueError("host_received_at_ns must be positive")
        object.__setattr__(self, "host_received_at_ns", received)
        image = np.array(self.image, copy=True, order="C")
        image.setflags(write=False)
        object.__setattr__(self, "image", image)

    @property
    def captured_at(self) -> float:
        if self.timestamp_seconds is not None and self.timestamp_microseconds is not None:
            return float(self.timestamp_seconds) + float(self.timestamp_microseconds) * 1e-6
        return float(self.host_received_at_ns) * 1e-9


class BaseDevice(ABC):
    """Common device lifecycle.

    Concrete hardware adapters must inherit this class or one of its typed
    subclasses.  This is intentionally stricter than duck typing: missing
    methods should be caught when the class is written, not halfway through a
    long experiment.
    """

    #: The device's SIDE-EFFECT-FREE surface -- the names an OBSERVE-mode consumer (a logic
    #: node that reads but never drives, see :func:`read_only`) may touch.  Each contract
    #: class extends this with its own read-only facts; drive verbs (open/close/configure/
    #: arm/prepare/fire/...) are never listed.  The device OWNS this fact: consumers narrow
    #: through it instead of each maintaining a private notion of "what is safe to read".
    OBSERVE_API: frozenset = frozenset({"snapshot"})

    def open(self):
        return self

    @property
    def is_open(self) -> bool:
        """True once the device's hardware handle is live.  A backend whose ``open()`` acquires a
        handle (a camera's grabber, an RPyC link) OVERRIDES this with that predicate; a device with a
        no-op ``open()`` is always ready.  This is the ONE source of the "opened on first use"
        invariant -- :meth:`ensure_open` reads it, so no backend re-decides the policy by RAISING on
        an unopened use (the historical ``PylonCamera.arm before open()`` outlier)."""
        return True

    def ensure_open(self):
        """Idempotently open the device on first use.  The session contract is that devices open
        LAZILY on first use (``na.connect`` builds the set with ``open_devices=False`` -- nothing is
        eagerly opened), so a use-method calls this before touching the hardware and NEVER has to
        raise "used before open"; it just gets opened.  A no-op once :attr:`is_open`, so calling it
        before every arm/read costs nothing.  ``open()`` must therefore be idempotent."""
        if not self.is_open:
            self.open()
        return self

    def close(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def snapshot(self) -> dict[str, Any]:
        """A frozen read-only dump of the device's identity + every LIVE knob it declares.  The
        ``type`` key leads (its single producer, see ``test_snapshot_type_key...``); then every
        :class:`DeviceProperty` value is auto-included -- the knob is DECLARED once (the descriptor)
        and DUMPED from that one declaration, so a backend never re-lists its knobs in a snapshot
        override (the laser / RF used to).  A backend overrides this only to add NON-property facts
        (a camera's exposure / roi, a sequencer's channel labels) via ``super().snapshot()``."""
        out: dict[str, Any] = {"type": type(self).__name__}
        for name, prop in collect_device_properties(type(self)).items():
            out[name] = prop.__get__(self, type(self))
        return out

    # ------------------------------------------------------------- config schema
    # A device class DESCRIBES ITS OWN config form: the three hooks below are what a
    # config editor (the device manager) renders and round-trips -- it never names a
    # concrete class.  A new device gets a usable form for free (reflection over its
    # ``__init__``); a class wanting better controls overrides ``config_params`` (e.g. a
    # ``choice`` for a trigger line), and one whose constructor NESTS its options (the
    # qCMOS config dataclass) also overrides the ``config_to_form`` / ``form_to_config``
    # pair.  Values flow ``{"type", "params"}`` -> ``config_to_form`` -> form widgets ->
    # ``form_to_config`` -> ``{"type", "params"}``; nothing here is ever ``eval``'d.

    @classmethod
    def config_params(cls) -> tuple:
        """The :class:`~..core.params.ParamDecl` rows describing this class's config-entry
        ``params`` -- the SINGLE declaration a device-manager form renders.  Default:
        reflect ``cls.__init__`` (:func:`config_params_from_signature`): the keyword
        parameters become typed rows (float/int/bool -> spin/switch, str -> text,
        tuple/list/dict -> a JSON literal, a device-typed / domain-named arg -> a
        ``$device:`` cross-reference choice)."""
        return config_params_from_signature(cls)

    @classmethod
    def config_to_form(cls, params) -> dict:
        """Map a config entry's ``params`` dict to the FLAT ``{key: value}`` the form
        seeds its widgets from.  Identity for most classes; a class whose constructor
        nests options (the qCMOS ``config`` dataclass) flattens them here."""
        return dict(params or {})

    @classmethod
    def form_to_config(cls, values) -> dict:
        """Map the flat form values back to the constructor ``params`` dict written into
        the config entry -- the exact inverse of :meth:`config_to_form`."""
        return dict(values or {})

    # ------------------------------------------------------------- runtime controls
    def runtime_controls(self) -> tuple["RuntimeControl", ...]:
        """The device's LIVE, tunable-at-runtime attributes -- a distinct layer from the
        construction-time ``config_params`` (what the config editor writes into a JSON
        entry), the read-only ``OBSERVE_API`` whitelist (what an observe-mode node may
        touch), and the ``snapshot`` blob (a frozen read-only dump).  This is the catalog a
        device CONTROL panel renders: each :class:`RuntimeControl` pairs a
        :class:`~..core.params.ParamDecl` (so the SAME ``PARAM_WIDGETS`` registry builds /
        validates the widget -- never an ``eval`` of typed text) with a ``getter``
        (``(device) -> value``, polled to show the current value) and a ``setter``
        (``(device, value) -> None``, or ``None`` for a read-only read-back).  Value
        validation lives in the device's OWN setter (e.g. a camera's ``exposure`` setter
        rejects a non-positive value), never in the GUI.

        DERIVED, never hand-typed: the catalog is built from every :class:`DeviceProperty` the
        device declares (MRO-merged), so declaring a knob once -- the property + its metadata --
        auto-injects its control (confocal ``ManagedProperty`` -> ``gui_dict``).  A device with no
        DeviceProperty has an empty catalog; a backend that needs a dynamic control can still
        override this and extend ``super().runtime_controls()``."""
        controls = []
        for prop in collect_device_properties(type(self)).values():
            getter = (lambda p: (lambda dev: p.__get__(dev, type(dev))))(prop)
            setter = (lambda p: (lambda dev, value: p.__set__(dev, value)))(prop) if prop.writable else None
            controls.append(RuntimeControl(decl=prop.build_decl(), getter=getter, setter=setter))
        return tuple(controls)


#: Device ACCESS MODES a logic node declares per device (its ``_devices`` mapping).
#: ``EXCLUSIVE`` -- the node DRIVES the hardware (arms the camera / prepares+fires the
#: streamer): at most one exclusive holder per device instance at a time, the fact the
#: console's mutual exclusion intersects.  ``OBSERVE`` -- the node only READS state: it is
#: handed a capability-narrowed :class:`ReadOnlyDevice` view (the device's own
#: ``OBSERVE_API`` whitelist), so a later edit that tries to drive an observed device fails
#: loudly instead of silently fighting the exclusive owner -- the declaration IS the
#: capability, never just metadata that code can drift from.
EXCLUSIVE = "exclusive"
OBSERVE = "observe"


class ReadOnlyDevice:
    """A capability-narrowed view of a device: forwards ONLY the names in the device's own
    ``OBSERVE_API`` (each contract class declares its side-effect-free surface); anything
    else -- prepare/fire/arm/configure/... -- raises ``PermissionError`` naming the
    violation.  Handed to a logic node for every device it declared ``OBSERVE``, so
    "recorded but never driven" is machine-enforced rather than a comment."""

    __slots__ = ("_device", "_allowed")

    def __init__(self, device, allowed):
        object.__setattr__(self, "_device", device)
        object.__setattr__(self, "_allowed", frozenset(str(n) for n in allowed))

    def __getattr__(self, name):
        if name in self._allowed:
            return getattr(self._device, name)
        raise PermissionError(
            f"{type(self._device).__name__}.{name} drives the device, but this node declared "
            f"the device OBSERVE (read-only).  Declare it EXCLUSIVE in the node's _devices "
            f"mapping if it really must drive it; observable names: {sorted(self._allowed)}.")

    def __setattr__(self, name, value):
        raise PermissionError(
            f"cannot set {type(self._device).__name__}.{name}: this node observes the device "
            "read-only (declared OBSERVE in its _devices mapping).")

    def __repr__(self):
        return f"<read-only {type(self._device).__name__}>"


def read_only(device):
    """The observe-mode view of ``device``: narrowed to its side-effect-free surface -- the ONE
    place the observe whitelist is assembled.  Two sources, both derived so a knob is declared once:
    the explicit ``OBSERVE_API`` (walked over the MRO so a contract subclass extends, never re-lists,
    its parent's non-property read-backs -- ``snapshot`` / a camera's ``latest`` / ``drain``), PLUS
    every :class:`DeviceProperty` NAME the device declares (a live knob's getter is side-effect-free,
    and the proxy blocks writes anyway).  So the laser / RF no longer hand-list their knob names in
    OBSERVE_API -- adding a DeviceProperty auto-exposes it to observe mode."""
    allowed: set = set()
    for cls in type(device).__mro__:
        allowed.update(getattr(cls, "OBSERVE_API", ()))
    allowed.update(collect_device_properties(type(device)))     # every declared live knob is a safe read
    return ReadOnlyDevice(device, allowed)


def underlying_device(device):
    """The REAL device instance behind a possibly read-only view -- THE single source for
    recovering a device's identity through the OBSERVE proxy.  ``ReadOnlyDevice`` narrows a
    device's callable surface but is a DIFFERENT object, so ``id(proxy) != id(device)``; any
    code comparing "does this node reference that hardware instance" (the device-swap / close
    invalidation, #2) must unwrap through here first, never reach into ``_device`` by hand.
    A plain device passes through unchanged."""
    if isinstance(device, ReadOnlyDevice):
        return object.__getattribute__(device, "_device")
    return device


def _is_settable_attr(obj, name: str) -> bool:
    """True when ``obj.name`` can be WRITTEN: a plain instance attribute, OR a class
    descriptor with a setter (a ``property`` whose ``fset`` is not None, or any descriptor
    exposing ``__set__``).  A ``property`` whose getter alone is defined is READ-ONLY -- so
    a runtime-control catalog only offers a setter where a write would actually take, never
    one that raises ``AttributeError: can't set attribute``.  The same "can I write this?"
    probe the device manager makes before wiring an editable control."""
    attr = inspect.getattr_static(type(obj), name, None)
    if isinstance(attr, property):
        return attr.fset is not None
    if attr is not None and hasattr(attr, "__set__"):
        return getattr(attr, "fset", True) is not None
    # not a class descriptor: a plain instance attribute is settable (its presence is the
    # writability), but a bound method / classmethod is not a value to overwrite.
    return name in getattr(obj, "__dict__", {})


# --------------------------------------------------------------------- config-form reflection
# The DEFAULT ``BaseDevice.config_params()``: turn a constructor signature into typed
# ParamDecl rows.  Shared module-level helpers (not hidden inside the classmethod) so a
# class overriding ``config_params`` can still reflect MOST of its signature and replace
# just the rows it wants richer controls for (see ``PylonCamera.config_params``).


def _json_ready_value(value):
    """A signature default made JSON-round-trippable (tuples -> lists, keys -> str) so a
    ``json``-kind row seeds with exactly what ``json.loads`` of its text would return."""
    if isinstance(value, (tuple, list)):
        return [_json_ready_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_ready_value(v) for k, v in value.items()}
    return value


def _annotation_param_kind(annotation, owner_module) -> str | None:
    """Best-effort ParamDecl kind for an ``__init__`` annotation.  Under ``from __future__
    import annotations`` the annotation is a STRING -- it is TOKEN-matched (never eval'd,
    the trust posture of every form value): container tokens -> ``json``, scalar tokens ->
    their kind, and a bare class name resolving (via the owner module's namespace, a plain
    ``getattr``) to a :class:`BaseDevice` subclass -> a ``$device:`` reference."""
    if annotation is inspect.Parameter.empty or annotation is None:
        return None
    if isinstance(annotation, type):
        if issubclass(annotation, BaseDevice):
            return "device"
        return {bool: "bool", int: "int", float: "float", str: "text",
                tuple: "json", list: "json", dict: "json"}.get(annotation)
    text = str(annotation)
    if re.search(r"\b(Sequence|Mapping|List|Tuple|Dict|list|tuple|dict)\b", text):
        return "json"
    for token, kind in (("bool", "bool"), ("int", "int"), ("float", "float"), ("str", "text")):
        if re.search(rf"\b{token}\b", text):
            return kind
    bare = text.strip()
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", bare) and owner_module is not None:
        target = getattr(owner_module, bare, None)
        if isinstance(target, type) and issubclass(target, BaseDevice):
            return "device"
    return None


def config_param_decl(name, *, default=inspect.Parameter.empty,
                      annotation=inspect.Parameter.empty, owner_module=None, tooltip=""):
    """ONE constructor parameter -> its typed :class:`~..core.params.ParamDecl` row.

    Kind resolution order (first hit wins):

    1. the param is NAMED after a registered device DOMAIN (``camera`` / ``sequencer`` /
       ``trap_array`` / a lab-registered role) -> a ``device`` cross-reference (its value
       is ``"$device:<entry>"``; the editor fills the choices from the working config);
    2. a concrete default value declares its own kind (bool/int/float -> spin/switch,
       str -> text, tuple/list/dict -> a JSON literal);
    3. the annotation (:func:`_annotation_param_kind`);
    4. free text.

    A parameter with NO default is ``required`` (the form highlights it empty)."""
    from ..core.params import ParamDecl

    required = default is inspect.Parameter.empty
    value = None if required else default
    kind = None
    from .registry import DEVICE_DOMAINS   # lazy: registry imports this module
    if str(name) in DEVICE_DOMAINS:
        kind = "device"
    if kind is None and value is not None:
        if isinstance(value, bool):
            kind = "bool"
        elif isinstance(value, int):
            kind = "int"
        elif isinstance(value, float):
            kind = "float"
        elif isinstance(value, str):
            kind = "text"
        elif isinstance(value, (tuple, list, dict)):
            kind = "json"
    if kind is None:
        kind = _annotation_param_kind(annotation, owner_module) or "text"
    if kind == "json":
        value = _json_ready_value(value)
    return ParamDecl(key=str(name), label=str(name).replace("_", " "), kind=kind,
                     default=value, required=required, lo=-1e12, hi=1e12, tooltip=str(tooltip))


def config_params_from_signature(cls) -> tuple:
    """Reflect ``cls.__init__`` into the default config form (the base
    :meth:`BaseDevice.config_params`): every keyword-assignable parameter becomes one
    :func:`config_param_decl` row, in signature order (``self`` / ``*args`` / ``**kwargs``
    skipped)."""
    import sys

    module = sys.modules.get(getattr(cls, "__module__", ""), None)
    decls = []
    for name, param in inspect.signature(cls.__init__).parameters.items():
        if name == "self" or param.kind in (inspect.Parameter.VAR_POSITIONAL,
                                            inspect.Parameter.VAR_KEYWORD):
            continue
        decls.append(config_param_decl(name, default=param.default,
                                       annotation=param.annotation, owner_module=module))
    return tuple(decls)


#: The ONE "clear the ROI back to the full sensor" sentinel every camera ``configure(roi=...)``
#: accepts.  ``roi=None`` means "leave unchanged" (so a multi-field configure can skip it), which
#: is why clearing needs an explicit value: the measurement layer sends this canonical spelling.
FULL_FRAME = "full"

#: Every spelling ``configure(roi=...)`` treats as "clear to full frame": the canonical
#: ``FULL_FRAME`` plus ``""`` / ``"None"`` (what a GUI's blank text box naturally produces).
#: Single source -- each backend tests membership HERE, so the accepted spellings can never
#: drift between cameras (the old per-backend ("", "None") tuples were an unreachable chain:
#: the measurement layer mapped a blank region to None, which means "leave unchanged").
ROI_CLEAR_SENTINELS = (FULL_FRAME, "", "None")

#: The human-readable "no sub-array" label the device viewer's ROI read-back SHOWS (``roi is None``).
#: Declared once here so the getter that displays it and the free-text setter that accepts it back
#: read the SAME string -- a distinct concern from the canonical ``FULL_FRAME`` the API layer sends.
FULL_FRAME_DISPLAY = "full frame"


def is_roi_clear(value) -> bool:
    """True when ``value`` -- FREE-FORM text typed into the device viewer's ROI box -- means "clear to
    the full frame".  THE case-insensitive predicate the ``roi`` runtime-control setter uses: it accepts
    every canonical :data:`ROI_CLEAR_SENTINELS` spelling PLUS the :data:`FULL_FRAME_DISPLAY` read-back
    the getter shows, so what the box displays round-trips and the accepted spellings can never drift
    from the sentinels.  (Backends comparing an API-supplied canonical value still use exact
    ``roi in ROI_CLEAR_SENTINELS`` -- that layer is not free text.)"""
    text = str(value).strip().lower()
    return text in {s.lower() for s in (*ROI_CLEAR_SENTINELS, FULL_FRAME_DISPLAY)}


#: The ONE ``trigger_source`` value meaning "software-trigger / free-run" -- the industry-standard
#: GenICam spelling (Basler: ``TriggerMode Off``): the sensor exposes on its OWN clock and every
#: ``acquire`` yields frames with NO trigger wiring at all.  Any other ``trigger_source`` value is
#: a hardware trigger line: one frame per external edge, the externally-triggered contract.
#: Single vocabulary shared by the real ``PylonCamera`` and the virtual ``VirtualMotCamera`` so the
#: two modes can never be spelled differently between backends (virtual == real).  The user-facing
#: config value stays the plain string ``"Software"`` (compared case-insensitively).
SOFTWARE_TRIGGER = "Software"


def is_software_trigger(value) -> bool:
    """True when ``value`` names the software-trigger / free-run mode -- THE case-insensitive
    comparison against :data:`SOFTWARE_TRIGGER` (declared right above), so the rule can never
    be spelled differently between backends (virtual == real): every ``_free_run`` predicate
    calls this instead of re-typing ``.lower() ==``."""
    return str(value).lower() == SOFTWARE_TRIGGER.lower()


class CameraDevice(BaseDevice):
    """Required contract for a camera used by ``NeutralAtomSession``.

    A camera is a PURE frame-grabber with THREE acquisition primitives:

    * :meth:`arm` -- ready the hardware for externally triggered frames;
    * :meth:`read_frames` -- blockingly consume frames from the device's own buffer;
    * :meth:`disarm` -- stand the hardware down.

    (:meth:`acquire` is the free-run convenience composing the three.)  The camera NEVER
    drives a sequencer, NEVER counts triggers in a pulse, NEVER infers exposure, and knows
    NOTHING about the experiment session -- once armed it simply waits for triggers.  The
    MEASUREMENT layer orchestrates the shot (arm the camera, fire the sequencer, read the
    frames back): see ``operations.measurement.triggered_frames``, the single arm-before-fire
    helper.  The one thing a camera owns about the wider system is
    ``capture_trigger_channels`` -- which sequencer line its hardware trigger input is wired
    to -- a PASSIVE fact it exposes upward, never used to control the sequencer.

    Frame-loss protection is a DEVICE-OWNED buffer: every frame that arrives while the
    camera is armed is queued in a predeclared bounded ``pending`` queue until
    :meth:`read_frames` consumes it, so a fire that lands before the consumer starts
    reading ("late consumption") drops nothing.  Independently, a small RECENT-FRAMES ring
    (``recent_capacity``) keeps the newest frames for live viewers (:meth:`drain` /
    :meth:`latest`) -- lossy by design, a convenience view, never the acquisition path.
    Both live on this base class so the virtual and real cameras buffer identically
    (virtual == real); subclasses feed frames through :meth:`_deliver`.
    """

    recent_capacity: int = 16
    #: Which sequencer channel(s) the camera's external-trigger input is wired to.  A passive
    #: device property (the camera is the source of this wiring fact), default the conventional
    #: ``emCCD`` line; a real camera is configured with its actual chNN at construction.
    capture_trigger_channels: tuple[str, ...] = DEFAULT_CAMERA_TRIGGER_CHANNELS

    #: Read-only facts an OBSERVE-mode consumer may touch (see :class:`ReadOnlyDevice`):
    #: the imaging geometry/exposure read-backs and the lossy live-view taps -- never
    #: arm/read_frames/acquire/configure (those drive the exclusive acquisition path).
    OBSERVE_API = BaseDevice.OBSERVE_API | frozenset(
        {"exposure", "roi", "sensor_shape", "frame_shape", "capture_trigger_channels",
         "effective_trigger_channels", "primary_trigger_channel", "latest", "drain"})

    @property
    def _free_run(self) -> bool:
        """True when this camera exposes on its OWN clock (software-trigger / free-run) and so counts
        capture edges on NO sequencer line.  THE single predicate both backends share (was an identical
        copy on PylonCamera and VirtualMotCamera): a camera whose ``trigger_source`` knob names the
        software-trigger mode (:func:`is_software_trigger`) is free-running; a camera with NO
        ``trigger_source`` knob (the qCMOS, the virtual readout camera) is always hardware-triggered."""
        source = getattr(self, "trigger_source", None)
        return source is not None and is_software_trigger(source)

    @property
    def effective_trigger_channels(self) -> tuple[str, ...]:
        """The line(s) this camera ACTIVELY counts capture edges on -- the machine-readable
        "which channel gates my frames" fact every consumer reads (via
        ``camera_trigger.resolve_camera_trigger_channels``), derived ONCE here from :attr:`_free_run`:

        * a FREE-RUNNING sensor exposes on its own clock and never consults its trigger input, so it
          counts on NO line -> ``()`` (edge counts are honestly zero) -- identical for the real Basler
          and the virtual monitor camera (virtual == real);
        * otherwise the declared :attr:`capture_trigger_channels` wiring.  A camera with no
          ``trigger_source`` knob (the qCMOS, the virtual readout camera) is always
          hardware-triggered, so its wiring IS its counting line."""
        if self._free_run:
            return ()
        return tuple(str(c) for c in self.capture_trigger_channels)

    @property
    def primary_trigger_channel(self) -> "str | None":
        """The FIRST active counting line, or None when there is none (a free-running sensor) --
        THE spelling every "which single channel should the imaging pulse trigger" consumer
        reads (the session's imaging kwargs, the coupled measurement templates), so the
        ``channels[0] if channels else None`` fallback is never re-typed at call sites."""
        channels = self.effective_trigger_channels
        return channels[0] if channels else None

    @staticmethod
    def _validated_exposure(value) -> float:
        """THE one exposure validation every write path shares (this base setter, each
        backend's re-declared setter/constructor/configure): a camera exposure is a positive
        finite float -- identical on every backend (virtual == real), so an illegal value
        fails loudly at the API boundary instead of reaching the hardware layer."""
        return positive_float(value, "exposure")

    @property
    @abstractmethod
    def exposure(self) -> float:
        """Current default exposure in seconds."""

    @exposure.setter
    def exposure(self, value: float) -> None:
        """Write-through to :meth:`configure` -- so ``cam.exposure = 3e-3`` works uniformly on
        every backend.  Exposure is the ONE intrinsic scalar where ``=`` reads naturally; it
        routes through ``configure`` (the single hardware-write path) and never hides more than
        setting the exposure.  Multi-field / ROI changes still go through ``configure(...)``
        directly (an ROI snap / multi-subsystem write must not hide behind ``=``).  A backend
        that overrides the ``exposure`` getter must re-declare this setter (it would otherwise be
        shadowed); the concrete cameras (VirtualCamera, QCMOSCamera) do -- and every setter
        validates through :meth:`_validated_exposure` (the single validation source)."""
        self.configure(exposure=self._validated_exposure(value))

    @property
    def roi(self) -> tuple[int, int, int, int] | None:
        """Sub-array readout window ``(x, width, y, height)``, or None for full frame.

        Part of the contract so a consumer (e.g. a raw-frame logic node's Edit panel)
        reads ROI without reaching into a backend's private ``config``.  Default
        None: a camera with no sub-array concept (the virtual renderer) honestly
        has no ROI; the real qCMOS overrides this."""
        return None

    @property
    def sensor_shape(self) -> tuple[int, int] | None:
        """The FULL sensor size ``(height, width)`` in pixels, or None if unknown.

        Part of the contract so a consumer (a raw-frame Edit panel) can show the ROI as
        the full-frame window even when no sub-array is set (``roi is None``) -- without
        reaching into a backend's internals.  Default None (size unknown until a frame is
        read); a backend that knows its sensor up front (the virtual renderer, the real
        qCMOS) overrides this."""
        return None

    @property
    def frame_shape(self) -> tuple[int, int] | None:
        """The ``(height, width)`` of the NEXT frame this camera will deliver -- DERIVED
        (never stored) from the two contract facts above: the sub-array window when an
        ROI is set, else the full :attr:`sensor_shape`; ``None`` when the backend knows
        neither yet (shape is then learnt from the first frame).

        This is what lets a consumer DECLARE its output structure at BUILD time instead
        of waiting for a frame: a ``CameraMeasurement`` seeds its ``data_shape`` from
        here, so a panel bound to ``frame_0`` can render the hub's lingering block the
        moment the measurement (re)starts -- even while no trigger is firing yet."""
        r = self.roi
        if r is not None:
            _x, width, _y, height = (int(v) for v in r)   # ROI contract order: (x, width, y, height)
            return (height, width)
        return self.sensor_shape

    def runtime_controls(self) -> tuple["RuntimeControl", ...]:
        """The camera's live control catalog (see :meth:`BaseDevice.runtime_controls`): the
        WRITABLE ``exposure`` (through the contract's own ``exposure`` setter, which validates
        via ``_validated_exposure`` -- an illegal value is rejected at the device boundary, not
        the GUI) plus the read-only geometry / wiring read-backs (``roi`` / ``sensor_shape`` /
        ``frame_shape`` / trigger channels).  Every backend (virtual == real) inherits the SAME
        set, so a camera control panel is identical across cameras; a backend with an extra live
        knob extends this."""
        from ..core.params import ParamDecl

        def _fmt_shape(value) -> str:
            return "" if value is None else str(tuple(int(v) for v in value))

        def _fmt_channels(value) -> str:
            return ", ".join(str(c) for c in (value or ())) or "(none)"

        def _set_roi(dev, value) -> None:
            # Route the WRITE through the one ``configure(roi=...)`` API -- the SAME call the notebook
            # makes, so its normalize/validate (``normalize_roi`` / clear sentinels) is the single
            # boundary.  ``is_roi_clear`` is the ONE free-text predicate (accepts the sentinels + the
            # read-back's own FULL_FRAME_DISPLAY spelling), so what the field shows round-trips: blank /
            # full -> whole sensor, else four ints.
            if is_roi_clear(value):
                dev.configure(roi=FULL_FRAME)   # the CLEAR sentinel (configure(roi=None) means "unchanged")
                return
            text = str(value).strip().lower()
            parts = [int(float(p)) for p in text.replace("(", "").replace(")", "").split(",") if p.strip()]
            dev.configure(roi=tuple(parts))

        return (
            RuntimeControl(
                ParamDecl(key="exposure", label="exposure", kind="float", unit="s",
                          lo=0.0, hi=1e6, required=True,
                          tooltip="Frame exposure time in seconds (a positive value; the "
                                  "device setter rejects anything else)."),
                getter=lambda dev: float(dev.exposure),
                setter=lambda dev, value: setattr(dev, "exposure", float(value))),
            RuntimeControl(
                ParamDecl(key="roi", label="ROI", kind="text",
                          tooltip="Sub-array readout window as x, width, y, height (or 'full' for the "
                                  "whole sensor).  Applied via the device's own configure(roi=...)."),
                getter=lambda dev: FULL_FRAME_DISPLAY if dev.roi is None else str(tuple(int(v) for v in dev.roi)),
                setter=_set_roi),
            RuntimeControl(
                ParamDecl(key="sensor_shape", label="sensor", kind="text",
                          tooltip="Full sensor size (height, width) in pixels."),
                getter=lambda dev: _fmt_shape(dev.sensor_shape)),
            RuntimeControl(
                ParamDecl(key="frame_shape", label="frame", kind="text",
                          tooltip="Shape (height, width) of the next frame delivered."),
                getter=lambda dev: _fmt_shape(dev.frame_shape)),
            RuntimeControl(
                ParamDecl(key="trigger_channels", label="trigger", kind="text",
                          tooltip="Sequencer line(s) this camera counts capture edges on."),
                getter=lambda dev: _fmt_channels(dev.effective_trigger_channels)),
        )

    @abstractmethod
    def configure(self, *, exposure: float | None = None, **kwargs) -> None:
        """Configure camera settings that are stable across an acquisition.

        Backends declare the keyword arguments they accept and reject any unknown
        key via :meth:`_reject_unknown_configure_keys` -- so a mistyped or
        backend-specific option fails LOUDLY and identically on every backend
        (virtual == real), never silently ignored on one and a ``TypeError`` on
        another."""

    @staticmethod
    def _reject_unknown_configure_keys(allowed: set[str], got) -> None:
        """Raise ``ValueError`` listing the configurable options if ``got`` carries an
        unknown key.

        Shared so ``configure`` enforces the SAME contract on every backend: an
        option a backend does not recognise is an explicit error (naming the keys it
        DOES accept), not a silent no-op on one camera and a ``TypeError`` on the
        next.  ``allowed`` is the backend's own configurable set (always includes
        ``exposure``); ``got`` is its ``**kwargs`` of leftover keys."""
        unknown = sorted(set(got) - allowed)
        if unknown:
            raise ValueError(
                f"unknown configure option(s) {unknown}; configurable: {sorted(allowed)}")

    # ------------------------------------------------------------- acquisition
    def arm(
        self,
        frames: int | None = None,
        *,
        max_inflight_frames: int | None = None,
        timeout: float | None = None,
        stop=None,
    ) -> None:
        """Ready the camera for ``frames`` externally triggered frames (None = continuous).

        When this method RETURNS the hardware is armed and waiting for triggers, so a fire
        issued after ``arm()`` can NEVER outrun the camera -- this ordering IS the
        arm-before-fire guarantee every measurement relies on (arm the camera, THEN fire the
        sequencer, then :meth:`read_frames`).  Frames arriving while armed are queued
        losslessly until read.  Arming also takes the camera's acquisition lock, serializing
        concurrent consumers (a live monitor polling :meth:`acquire` waits while a
        measurement holds an armed session); :meth:`disarm` releases it.

        Taking that lock is BOUNDED + CANCELLABLE, exactly like :meth:`read_frames` /
        :meth:`_grab`: ``timeout`` (seconds) caps the wait for the lock and raises
        ``TimeoutError`` if another consumer still holds an armed session; ``stop`` (a
        ``threading.Event``) cancels the wait cooperatively and raises
        :class:`AcquisitionCancelled`.  Passing NEITHER preserves the legacy unbounded blocking
        acquire.  This is the ONE fix that stops a contended arm from wedging a task at 0% with
        no way for Stop to break it -- arm now joins read/disarm's cancellable contract."""
        self.ensure_open()          # lazy-open on first use (session contract) -- BEFORE the lock,
                                    # so a failed open never leaves the acquire lock held; a backend
                                    # never raises "arm before open", it just gets opened.
        if frames is not None:
            frames = positive_int(frames, "frames")
        if max_inflight_frames is not None:
            max_inflight_frames = positive_int(
                max_inflight_frames, "max_inflight_frames"
            )
            if frames is not None and max_inflight_frames > frames:
                raise ValueError("max_inflight_frames cannot exceed finite frame budget")
        self._acquire_session_lock(timeout=timeout, stop=stop)
        state = self._recent_state()
        with state["cond"]:
            state["pending"].clear()
            state["armed"] = True
            state["armed_frames"] = frames
            state["last_terminal"] = None
            state["last_terminal_error"] = None
            state["pending_capacity"] = (
                int(max_inflight_frames)
                if max_inflight_frames is not None
                else (
                    int(frames)
                    if frames is not None
                    else max(1, int(self.recent_capacity))
                )
            )
            state["source_ordinal"] = 0
        try:
            self._arm(frames, max_inflight_frames=max_inflight_frames)
        except BaseException:
            with state["cond"]:
                state["armed"] = False
                state["pending"].clear()
            self._acquire_lock().release()
            raise

    def _read_frame_records(
        self,
        n: int = 1,
        *,
        timeout: float | None = None,
        stop=None,
        **kwargs,
    ) -> list[CameraFrameRecord]:
        """Consume records from the sole armed-session queue."""

        n = positive_int(n, "n")
        state = self._recent_state()
        if not state["armed"]:
            raise RuntimeError(
                "read_frame_records() requires arm() first; use acquire_records() "
                "for a one-shot that arms internally."
            )
        out: list[CameraFrameRecord] = []
        while len(out) < n:
            with state["cond"]:
                while state["pending"] and len(out) < n:
                    out.append(state["pending"].popleft())
            if len(out) >= n:
                break
            if not self._grab(n - len(out), timeout=timeout, stop=stop, **kwargs):
                break
        return out

    def read_frame_records(
        self,
        n: int = 1,
        *,
        timeout: float | None = None,
        stop=None,
        **kwargs,
    ) -> list[CameraFrameRecord]:
        """Return immutable frame records including all adapter metadata."""

        return self._read_frame_records(n, timeout=timeout, stop=stop, **kwargs)

    def read_frames(self, n: int = 1, *, timeout: float | None = None, stop=None, **kwargs) -> list[np.ndarray]:
        """Blockingly consume ``n`` frames from the device's own buffer (one numpy array each).

        Frames queued while armed (pushed by the data source, or fetched by the backend's own
        grab hook) are drained first -- so a fire that happened BEFORE this call loses nothing;
        when the buffer runs dry the backend's ``_grab`` hook produces/awaits more.  ``timeout``
        (seconds, backend default when None) bounds the wait for a trigger that never comes;
        ``stop`` (a ``threading.Event``) cancels a blocking wait cooperatively.  Returns what
        arrived (possibly fewer than ``n``); backends whose fault model is loud (the qCMOS)
        raise ``TimeoutError`` / :class:`AcquisitionCancelled` from their hook instead.

        Requires the camera to be ARMED first: the three primitives are ordered arm -> read ->
        disarm, and only :meth:`arm` fills the lossless ``pending`` queue the data source pushes
        into.  Reading unarmed would consume from an empty queue while a push-fed source's
        ``_grab`` (e.g. the virtual trigger wire during a continuous firing) reports "more will
        arrive" -- an unbounded live-lock.  So an unarmed read is a programming error, raised
        loudly; use :meth:`acquire` for a one-shot (it arms internally)."""
        return [
            record.image
            for record in self._read_frame_records(
                n, timeout=timeout, stop=stop, **kwargs
            )
        ]

    def disarm(self) -> None:
        """Stand the camera down and release the acquisition lock taken by :meth:`arm`."""
        self.finish_record_capture()

    def finish_record_capture(self) -> CameraCaptureTerminalRecord:
        """Stop one armed epoch, freeze terminal facts, then release its owner lock.

        Concrete hardware may override :meth:`_finish_record_capture` to read a
        final production counter after the source has stopped but before its driver
        buffer is released.  This method owns the common state/lock teardown exactly
        once; callers must never reproduce it around backend hooks.
        """

        state = self._recent_state()
        with state["cond"]:
            must_stop_source = bool(state["armed"])
            previous_terminal = state["last_terminal"]
            previous_error = state["last_terminal_error"]
        try:
            if must_stop_source:
                terminal = self._finish_record_capture()
            elif previous_error is not None:
                raise RuntimeError(
                    "camera arm epoch already failed terminal completion"
                ) from previous_error
            elif previous_terminal is not None:
                terminal = previous_terminal
            else:
                terminal = CameraCaptureTerminalRecord(0, True, True, True)
            if not isinstance(terminal, CameraCaptureTerminalRecord):
                raise TypeError(
                    "camera _finish_record_capture must return CameraCaptureTerminalRecord"
                )
            if must_stop_source:
                with state["cond"]:
                    state["last_terminal"] = terminal
            return terminal
        except BaseException as error:
            if must_stop_source:
                with state["cond"]:
                    state["last_terminal_error"] = error
            raise
        finally:
            if must_stop_source:
                with state["cond"]:
                    state["armed"] = False
                    state["pending"].clear()
            try:
                self._acquire_lock().release()
            except RuntimeError:
                # An out-of-band interrupt may stop the source from a different
                # thread.  The original owner calls this method during cleanup
                # and releases its RLock then; a fully released/no-arm call is a
                # defensive no-op.
                pass

    def acquire(self, frames: int = 1, *, stop=None, **kwargs) -> list[np.ndarray]:
        """Convenience one-shot: ``arm(frames)`` + ``read_frames(frames)`` + ``disarm()``.

        For a camera that needs no external fire -- a free-running sensor (Basler
        ``Software`` mode), or a virtual camera watching an already-firing continuous
        pulse -- this is the whole story.  An externally TRIGGERED measurement must fire
        its sequencer between arm and read: use the single measurement-layer helper
        ``operations.measurement.triggered_frames`` (never hand-roll prepare+fire)."""
        frames = positive_int(frames, "frames")
        self.arm(frames, stop=stop)          # cancellable arm too, so a Stop breaks a lock wait
        try:
            return self.read_frames(frames, stop=stop, **kwargs)
        finally:
            self.disarm()

    def acquire_records(
        self, frames: int = 1, *, stop=None, **kwargs
    ) -> list[CameraFrameRecord]:
        """One-shot acquisition preserving source metadata."""

        frames = positive_int(frames, "frames")
        self.arm(frames, stop=stop)
        try:
            return self.read_frame_records(frames, stop=stop, **kwargs)
        finally:
            self.disarm()

    # --------------------------------------------------- backend hooks (subclass seam)
    def _arm(
        self,
        frames: int | None,
        *,
        max_inflight_frames: int | None = None,
    ) -> None:
        """Backend hook: ready the hardware (``buf_alloc`` + ``cap_start`` on a qCMOS,
        ``StartGrabbing`` on a pylon camera).  ``frames`` is total cardinality;
        ``max_inflight_frames`` is the separately proven driver-retention bound.
        Default: nothing to do (a push-fed source)."""

    def _grab(self, n: int, *, timeout: float | None = None, stop=None) -> bool:
        """Backend hook: produce or await at least one more frame for :meth:`read_frames`.

        Pull-based hardware fetches from its driver and feeds :meth:`_deliver`; a push-fed
        source may simply wait on the buffer condition.  Return True when new frames were
        (or will now be found) queued; False when none will arrive within ``timeout``
        (``read_frames`` then returns what it has).  Default: wait for a pushed frame."""
        state = self._recent_state()
        deadline = None if timeout is None else time.monotonic() + max(0.0, float(timeout))
        with state["cond"]:
            while not state["pending"]:
                if stop is not None and stop.is_set():
                    return False
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0.0:
                    return False
                state["cond"].wait(0.05 if remaining is None else min(0.05, remaining))
            return True

    def _disarm(self) -> None:
        """Backend hook: stand the hardware down (``cap_stop`` / ``StopGrabbing``)."""

    def _finish_record_capture(self) -> CameraCaptureTerminalRecord:
        """Backend terminal hook; subclasses with counters override this atomically."""

        self._disarm()
        state = self._recent_state()
        with state["cond"]:
            produced = int(state["source_ordinal"])
        return CameraCaptureTerminalRecord(produced, True, True, True)

    # ----------------------------------------------------------- frame buffers
    def _recent_state(self) -> dict:
        """Lazily create the frame-buffer state (subclasses define their own
        ``__init__`` and need not call ``super().__init__()``).

        ``dict.setdefault`` is the ATOMIC create-or-get: two threads first touching an
        unbuilt camera (a measurement arming while a live monitor drains) both build a
        candidate, but the GIL makes ``setdefault`` a single winner -- both then read back
        the SAME state, never two divergent buffers with acquisitions serialised against
        different locks.  The loser's freshly-built dict is discarded (harmless -- it holds
        no frames yet)."""
        lock = threading.RLock()
        fresh = {
            "frames": deque(maxlen=max(1, int(self.recent_capacity))),
            "lock": lock,
            "cond": threading.Condition(lock),
            "seq": 0,        # total frames ever retained
            "cursor": 0,     # drain watermark
            "pending": deque(),  # bounded CameraFrameRecord queue
            "armed": False,
            "armed_frames": None,
            "pending_capacity": max(1, int(self.recent_capacity)),
            "source_ordinal": 0,
            "last_terminal": None,
            "last_terminal_error": None,
        }
        return self.__dict__.setdefault("_zlc_recent", fresh)

    def _acquire_lock(self) -> "threading.RLock":
        """The per-camera acquisition lock :meth:`arm`/:meth:`disarm` hold across an armed
        session, so two consumers (a measurement + a live monitor) never interleave their
        armed state on one sensor.  Atomically create-or-get via ``setdefault`` (like
        :meth:`_recent_state`) so concurrent first touches share ONE lock -- a check-then-set
        would let two threads build two locks and defeat the mutual exclusion."""
        return self.__dict__.setdefault("_zlc_acquire_lock", threading.RLock())

    def _acquire_session_lock(self, *, timeout: float | None, stop=None) -> None:
        """Take the acquisition lock for an armed session, BOUNDED + CANCELLABLE (the same
        poll-with-deadline idiom :meth:`_grab` uses to await a frame).  With neither ``timeout``
        nor ``stop`` this is the legacy blocking acquire (unbounded, uninterruptible) -- callers
        that don't opt in keep the old behaviour.  With ``stop`` it polls so a cooperative Stop
        breaks a wait on a lock another (possibly abandoned) consumer still holds, raising
        :class:`AcquisitionCancelled`; with ``timeout`` it raises ``TimeoutError`` once the
        budget is spent.  Raising here (before any ``armed`` state is set) leaves the camera
        untouched, so no disarm is owed on the cancel/timeout path."""
        lock = self._acquire_lock()
        if stop is None and timeout is None:
            lock.acquire()
            return
        # A FREE lock is taken IMMEDIATELY and arm proceeds -- ``stop``/``timeout`` only govern a wait on
        # a CONTENDED lock.  This keeps the common (uncontended) path identical to the legacy blocking
        # acquire: a caller passing a ``stop`` that happens to be already set (e.g. a finite node stepped
        # one past its block) still arms and lets the read return empty, rather than raising a spurious
        # cancellation before we ever needed to wait.
        if lock.acquire(blocking=False):
            return
        deadline = None if timeout is None else time.monotonic() + max(0.0, float(timeout))
        while True:
            if stop is not None and stop.is_set():
                raise AcquisitionCancelled(
                    "camera arm cancelled before the acquisition lock was free.")
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0.0:
                raise TimeoutError(
                    f"camera still armed by another consumer after {float(timeout):.1f}s; "
                    "refusing to arm (another armed session has not been released).")
            slice_ = 0.05 if remaining is None else min(0.05, remaining)
            if lock.acquire(timeout=slice_):
                return

    def _deliver(self, images) -> list[np.ndarray]:
        """Queue freshly captured frames: into the lossless armed ``pending`` queue (when
        armed) AND the recent-frames ring, then wake any waiting reader.  The ONE entry
        point every frame source uses (a real backend's grab hook, the virtual trigger
        wire), so buffering behaviour cannot drift between backends."""
        arrs = [np.asarray(image) for image in images]
        state = self._recent_state()
        with state["cond"]:
            start = int(state["source_ordinal"] if state["armed"] else state["seq"])
            records = [
                CameraFrameRecord(
                    image=image,
                    source_ordinal=start + index,
                    produced_count=start + index + 1,
                    frame_stamp=None,
                    camera_stamp=None,
                    timestamp_seconds=None,
                    timestamp_microseconds=None,
                    host_received_at_ns=time.time_ns(),
                )
                for index, image in enumerate(arrs)
            ]
            return [record.image for record in self._deliver_records(records)]

    def _deliver_records(
        self, records
    ) -> list[CameraFrameRecord]:
        """Atomically retain typed records and enforce the arm-epoch budget."""

        normalized = [
            record
            if isinstance(record, CameraFrameRecord)
            else CameraFrameRecord(**dict(record))
            for record in records
        ]
        state = self._recent_state()
        with state["cond"]:
            if state["armed"]:
                expected = int(state["source_ordinal"])
                for record in normalized:
                    if record.source_ordinal != expected:
                        raise CameraBufferOverrun(
                            f"camera source ordinal {record.source_ordinal} differs from "
                            f"expected {expected}"
                        )
                    expected += 1
                budget = state["armed_frames"]
                if budget is not None and expected > int(budget):
                    raise CameraBufferOverrun(
                        f"camera produced {expected} frames for an arm budget of {budget}"
                    )
                if len(state["pending"]) + len(normalized) > int(
                    state["pending_capacity"]
                ):
                    raise CameraBufferOverrun(
                        "camera pending retention capacity exhausted before the consumer drained it"
                    )
                state["pending"].extend(normalized)
                state["source_ordinal"] = expected
            self._retain_locked(state, normalized)
            state["cond"].notify_all()
        return normalized

    @staticmethod
    def _retain_locked(state: dict, records) -> None:
        """Append frames to the recent ring; caller holds ``state['lock']``."""
        for record in records:
            if not isinstance(record, CameraFrameRecord):
                raise TypeError("camera retention accepts CameraFrameRecord values")
            state["frames"].append(record)
            state["seq"] += 1

    def _retain(self, images) -> Any:
        """Append frames to the recent-frames ring only (thread-safe); returns ``images``."""
        arrs = [np.asarray(image) for image in images]
        state = self._recent_state()
        with state["lock"]:
            start = int(state["seq"])
            records = [
                CameraFrameRecord(
                    image=image,
                    source_ordinal=start + index,
                    produced_count=start + index + 1,
                    frame_stamp=None,
                    camera_stamp=None,
                    timestamp_seconds=None,
                    timestamp_microseconds=None,
                    host_received_at_ns=time.time_ns(),
                )
                for index, image in enumerate(arrs)
            ]
            self._retain_locked(state, records)
        return [record.image for record in records]

    def recent_frames(self, n: int | None = None) -> list[np.ndarray]:
        """The most-recent retained frames (newest last); all of them when ``n`` is None."""
        state = self._recent_state()
        with state["lock"]:
            records = list(state["frames"])
        records = records if n is None else records[-int(n):]
        return [record.image for record in records]

    def recent_frame_records(
        self, n: int | None = None
    ) -> list[CameraFrameRecord]:
        state = self._recent_state()
        with state["lock"]:
            records = list(state["frames"])
        return records if n is None else records[-int(n):]

    def latest(self) -> np.ndarray | None:
        """The single newest retained frame, or None if nothing has been acquired."""
        state = self._recent_state()
        with state["lock"]:
            return state["frames"][-1].image if state["frames"] else None

    def drain(self) -> list[np.ndarray]:
        """Every frame retained since the previous ``drain`` (lossless up to capacity).

        A live consumer polling this gets ALL frames captured between its polls --
        so a multi-trigger shot's extra frames are never dropped on the floor."""
        state = self._recent_state()
        with state["lock"]:
            records = list(state["frames"])
            n_new = max(0, min(len(records), state["seq"] - state["cursor"]))
            state["cursor"] = state["seq"]
            return [record.image for record in records[-n_new:]] if n_new else []

    def drain_frame_records(self) -> list[CameraFrameRecord]:
        state = self._recent_state()
        with state["lock"]:
            records = list(state["frames"])
            n_new = max(0, min(len(records), state["seq"] - state["cursor"]))
            state["cursor"] = state["seq"]
            return records[-n_new:] if n_new else []

    def clear_recent(self) -> None:
        """Drop retained frames and reset the drain watermark (e.g. on reconfigure)."""
        state = self._recent_state()
        with state["lock"]:
            state["frames"].clear()
            state["cursor"] = state["seq"]

class SequencerDevice(BaseDevice):
    """Required contract for a timing/sequencer backend."""

    channels: list[str] | tuple[str, ...]
    port_catalog: Any
    clock_hz: float

    #: Read-only facts an OBSERVE-mode consumer may touch (see :class:`ReadOnlyDevice`):
    #: the live firing/progress state and static geometry -- never prepare/fire/
    #: set_safe_state/settle (those drive the one shared streamer).
    OBSERVE_API = BaseDevice.OBSERVE_API | frozenset(
        {"firing", "last_fired", "scan_progress", "raw_channels", "port_catalog",
         "clock_hz", "sleep_scale"})

    @abstractmethod
    def prepare(self, sequence) -> Any:
        """Validate/compile/arm a pulse sequence."""

    @abstractmethod
    def fire(self, sequence=None) -> None:
        """Start a previously prepared sequence."""

    @property
    def firing(self) -> "Any | None":
        """The ``repeat_forever`` program the streamer is continuously playing, or None when
        idle/safe -- the software model of "the FPGA is emitting camera triggers right now".

        Part of the contract (like :meth:`scan_progress`) so a DEVICE-layer consumer (the
        virtual trigger wire inside ``devices.virtual``) reads the live firing state through
        the abstraction, not via a duck-typed ``getattr`` on a concrete backend.  Default
        None: a real/remote streamer keeps no local firing flag -- a real camera learns "is it
        triggering" from the PRESENCE/ABSENCE of hardware trigger edges, not from a host-side
        handle -- so None is the correct real-hardware semantics.  The in-process
        VirtualSequencer overrides this to expose the program it is simulating."""

        return None

    def wait_done(self, timeout: float | None = None) -> bool:
        """Wait until the prepared finite sequence is done, when supported."""

        return True

    def scan_progress(self) -> dict:
        """Where the running scan is now: the SINGLE-source dict {scanning, point, n_points,
        sweep, n_repeats} (see ``sequencer.scan_progress_fields``).  A backend with no live
        scan -- or one that does not track progress -- reports idle.  The GUI polls this to show
        "point K / N · sweep r / R"; virtual and real backends return the same shape."""

        from .sequencer import SCAN_PROGRESS_IDLE
        return dict(SCAN_PROGRESS_IDLE)

    @property
    def raw_channels(self) -> tuple[str, ...]:
        """Physical compiler lanes, explicitly separated from logical output ports."""

        catalog = getattr(self, "port_catalog", None)
        if catalog is None:
            return tuple(str(value) for value in getattr(self, "channels", ()))
        return tuple(catalog.raw_lanes)

    def runtime_controls(self) -> tuple["RuntimeControl", ...]:
        """The sequencer's live control catalog (see :meth:`BaseDevice.runtime_controls`): the
        read-only firing / progress / geometry read-backs a monitor shows, plus the WRITABLE
        ``sleep_scale`` (the virtual inter-shot fast-forward factor) WHEN the backend carries a
        writable ``sleep_scale`` attribute -- a real streamer has no such knob, so the control is
        offered only where the attribute exists (a read-back otherwise, never a write that would
        fail).  The read-back getters use ``getattr`` fallbacks so a backend lacking ``last_fired``
        / ``sleep_scale`` still renders an honest '(none)' rather than raising."""
        from ..core.params import ParamDecl

        def _fmt_progress(dev) -> str:
            prog = dev.scan_progress()
            if not prog.get("scanning"):
                return "idle"
            return (f"point {prog.get('point', 0)}/{prog.get('n_points', 0)} · "
                    f"sweep {prog.get('sweep', 0)}/{prog.get('n_repeats', 0)}")

        controls = [
            RuntimeControl(
                ParamDecl(key="firing", label="firing", kind="text",
                          tooltip="Whether the streamer is continuously emitting a program right now."),
                getter=lambda dev: "yes" if dev.firing is not None else "idle"),
            RuntimeControl(
                ParamDecl(key="last_fired", label="last fired", kind="text",
                          tooltip="The most recently fired program, when the backend tracks it."),
                getter=lambda dev: str(getattr(dev, "last_fired", None) or "(none)")),
            RuntimeControl(
                ParamDecl(key="scan_progress", label="scan", kind="text",
                          tooltip="Where a running scan is now (point K/N · sweep r/R), or idle."),
                getter=_fmt_progress),
            RuntimeControl(
                ParamDecl(key="ports", label="ports", kind="text",
                          tooltip="Logical sequencer outputs; DAC data lanes are represented by one port."),
                getter=lambda dev: ", ".join(
                    port.key for port in getattr(getattr(dev, "port_catalog", None), "ports", ()))
                or "(none)"),
        ]
        # ``sleep_scale`` is a WRITABLE knob only where the backend actually carries a settable
        # attribute (the virtual sequencer's plain instance field); a real/remote streamer has
        # none, so it is offered as a read-back only (or skipped) rather than a setter that would
        # fail on write.  ``_is_settable_attr`` probes for a plain instance attribute OR a property
        # with an fset -- the same "can I write this" test the config editor makes.
        if _is_settable_attr(self, "sleep_scale"):
            controls.append(RuntimeControl(
                ParamDecl(key="sleep_scale", label="sleep scale", kind="float",
                          lo=0.0, hi=1e6, default=0.0,
                          tooltip="Virtual inter-shot delay multiplier (0 = instant, for tests)."),
                getter=lambda dev: float(getattr(dev, "sleep_scale", 0.0)),
                setter=lambda dev, value: setattr(dev, "sleep_scale", float(value))))
        elif hasattr(self, "sleep_scale"):
            controls.append(RuntimeControl(
                ParamDecl(key="sleep_scale", label="sleep scale", kind="text",
                          tooltip="Inter-shot delay multiplier (read-only on this backend)."),
                getter=lambda dev: str(getattr(dev, "sleep_scale", "(none)"))))
        return tuple(controls)

    def settle(self, seconds: float, *, stop: "threading.Event | None" = None) -> None:
        """Idle for ``seconds`` after a finite pulse before the next load+fire.

        The DEVICE owns this inter-shot wait so a software-stepped sweep (e.g. an API-slot
        pulse-scan: load -> on_pulse -> wait the pulse done -> settle -> next) does not hand-roll
        timing in the caller.  The hardware simply sits in its idle/safe state during this window;
        the default is a plain host-side wait (a real backend is idle between fires).  ``stop``
        (when given) makes the wait COOPERATIVELY cancellable -- it returns early once the event is
        set, so a Stop pressed mid-settle does not block teardown for the full delay.  A virtual
        backend scales the delay by ``sleep_scale`` so tests fast-forward."""

        self._sleep_interruptible(float(seconds), stop)

    @staticmethod
    def _sleep_interruptible(seconds: float, stop: "threading.Event | None") -> None:
        """Sleep ``seconds`` in small slices, returning early if ``stop`` is set."""
        if seconds <= 0.0:
            return
        if stop is None:
            time.sleep(seconds)
            return
        deadline = time.monotonic() + seconds
        while not stop.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return
            stop.wait(min(0.05, remaining))

    def abort(self) -> None:
        """Abort the current sequence, when supported."""

        self.stop()

    def set_safe_state(self) -> None:
        """Drive outputs to a safe idle state, when supported."""

        self.stop()


class TrapArrayDevice(BaseDevice):
    """Required contract for a trap-array state source.

    Device implementations intentionally do not expose camera-space site
    centers.  Those are experimental calibration data and must enter the
    readout stack through sitemap calibration, not through simulator or hardware
    internals.
    """

    @property
    @abstractmethod
    def n_sites(self) -> int:
        """Number of trap sites."""


class LaserDevice(BaseDevice):
    """Contract for the grey-molasses / PGC cooling LASER -- a single beam LOCKED blue on the Rb87 D1
    line.  A laser has no scannable "detuning" of its own here: the D1 lock point is fixed, and the
    detuning that matters for grey molasses is the TWO-PHOTON (Raman) detuning between the cooling and
    repump beams, which is set by the :class:`RFSourceDevice` sideband (see it).  So the laser's knobs
    are just whether the WAVELENGTH is on the D1 line at all (794.98 nm -- off it there is NO cooling)
    and the beam SATURATION I/I_sat (which power-broadens the dark resonance).  A real laser wraps a
    wavemeter + AOM servo behind these; the virtual one is a pure set-point.  ``runtime_controls`` is
    the ONE editable / observed surface (virtual == real inherits the same set)."""

    # The laser's live controls AND its observe surface AND its snapshot are ALL derived from the
    # DeviceProperty knobs a concrete backend declares (:class:`~..devices.virtual.VirtualLaser`
    # wavelength / saturation / beam_on / on_d1): ``runtime_controls`` builds the catalog, ``read_only``
    # auto-exposes every knob name, and ``BaseDevice.snapshot`` auto-dumps every knob value.  So this
    # contract needs no hand-written OBSERVE_API / snapshot -- declaring a knob once is enough.


class RFSourceDevice(BaseDevice):
    """Contract for the RF / microwave source that makes the grey-molasses REPUMPER sideband near the
    Rb87 ground-state hyperfine splitting (6.834682 GHz).  It is what PRODUCES the two-photon (Raman)
    detuning δ between the cooling and repump beams -- the detuning a laser lock cannot set.  δ = 0 (RF
    exactly on the splitting) is the coherent-dark-state resonance where grey molasses cools strongest;
    a wrong RF frequency (δ != 0) spoils the dark state and heats (a Fano feature).

    So ``two_photon_detuning_gamma`` (δ in linewidths) is a WRITABLE control -- the grey-molasses knob
    an operator tunes / a detuning scan sweeps -- backed by ``frequency_hz`` (the raw hardware set-point,
    also writable): setting either moves the same drive.  The device owns the δ<->frequency conversion
    (its atomic reference), so a scan just writes δ in Γ and every backend converts identically."""

    # The RF source's live controls AND observe surface AND snapshot are ALL derived from the
    # DeviceProperty knobs the concrete backend declares (:class:`~..devices.virtual.VirtualRF`
    # two-photon δ / frequency / power / drive_on / waveform): ``runtime_controls`` builds the catalog,
    # ``read_only`` auto-exposes every knob name, ``BaseDevice.snapshot`` auto-dumps every value -- so
    # no hand-written OBSERVE_API / snapshot here (declaring each knob once is the single source).


def snap_subarray(roi, *, step: int, max_w: int, max_h: int):
    """Snap a REQUESTED sub-array window to a camera's valid sub-array grid.

    A plot selection (scroll / area-select) is in continuous source-pixel
    coordinates; a real sensor only reads sub-arrays whose origin AND size are
    multiples of a hardware ``step`` (the Hamamatsu qCMOS requires multiples of
    4 -- see the DCAM ``SUBARRAY*`` properties), within the sensor.  This is the
    SINGLE source of truth for that adaptation, so the GUI/measurement layer can
    stay in plain plot coordinates and every camera (real or virtual) snaps the
    SAME way -- a window written raw would otherwise be silently clamped by the
    hardware and the camera would image the wrong region.

    ``roi`` is ``(x, width, y, height)``.  Returns the snapped ``(x, w, y, h)``
    with every field a multiple of ``step``, ``w``/``h`` >= ``step`` and the
    window fully inside ``(max_w, max_h)``.
    """
    step = int(step)
    if step <= 0:
        step = 1
    x, w, y, h = (int(round(float(v))) for v in roi)

    def _snap(v: int) -> int:
        return int(round(v / step)) * step

    max_w = (int(max_w) // step) * step
    max_h = (int(max_h) // step) * step
    w = min(max(step, _snap(w)), max_w)
    h = min(max(step, _snap(h)), max_h)
    x = min(max(0, _snap(x)), max_w - w)
    y = min(max(0, _snap(y)), max_h - h)
    return (x, w, y, h)


__all__ = [
    "BaseDevice",
    "CameraBufferOverrun",
    "CameraCaptureTerminalRecord",
    "CameraDevice",
    "CameraFrameRecord",
    "EXCLUSIVE",
    "FULL_FRAME",
    "FULL_FRAME_DISPLAY",
    "LaserDevice",
    "OBSERVE",
    "RFSourceDevice",
    "ROI_CLEAR_SENTINELS",
    "ReadOnlyDevice",
    "RuntimeControl",
    "SOFTWARE_TRIGGER",
    "SequencerDevice",
    "TrapArrayDevice",
    "is_roi_clear",
    "is_software_trigger",
    "read_only",
    "underlying_device",
    "snap_subarray",
]
