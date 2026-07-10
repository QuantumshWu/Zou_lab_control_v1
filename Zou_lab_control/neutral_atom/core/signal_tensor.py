"""Typed signal tensors and the in-place store used by :mod:`.signals`.

The transport has two mandatory leading axes and preserves every data axis::

    (R, P, *data_shape)

``point_shape`` is logical scan geometry only: its product is the one physical
``P`` axis.  ``data_shape`` is the non-empty physical trailing shape
and is never flattened.  Thus a camera is ``(R, 1, H, W)``, per-site data is
``(R, 1, N)``, a scalar is ``(R, 1, 1)``, and a scalar scan is ``(R, P, 1)``.

There is deliberately no ndarray-rank inference here.  An unregistered raw
value has one deterministic meaning: one repeat, one point, and the entire
original value as the trailing data shape.  A producer that owns repeat/point
semantics must register a :class:`SignalSchema` or publish a
:class:`SignalTensor`.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field, replace
from typing import Any, Mapping

import numpy as np


_NUMERIC_KINDS = frozenset("biufc")


class SignalHistoryGap(RuntimeError):
    """A consumer cursor fell behind the bounded update journal."""


@dataclass(frozen=True, order=True)
class SignalCursor:
    """Position in one schema-versioned signal update stream."""

    schema_version: int
    version: int

    def __post_init__(self) -> None:
        if int(self.schema_version) < 0 or int(self.version) < 0:
            raise ValueError("signal cursor fields must be >= 0.")
        object.__setattr__(self, "schema_version", int(self.schema_version))
        object.__setattr__(self, "version", int(self.version))


@dataclass(frozen=True)
class TensorUpdateRef:
    """Lightweight ordered journal entry; payload is reconstructed on demand."""

    cursor: SignalCursor
    provenance: int


def _positive_shape(value, name: str) -> tuple[int, ...]:
    shape = tuple(int(n) for n in value)
    if not shape or any(n < 1 for n in shape):
        raise ValueError(f"{name} must be a non-empty tuple of positive sizes; got {value!r}.")
    return shape


def _numeric_array(value, *, name: str = "signal") -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in _NUMERIC_KINDS:
        raise TypeError(
            f"{name} data must be a numeric/bool ndarray, got dtype {array.dtype}.  "
            "Control text and domain objects belong on a control/metadata channel, not SignalHub.")
    return array


def _dtype_or_none(value) -> np.dtype | None:
    if value is None:
        return None
    dtype = np.dtype(value)
    if dtype.kind not in _NUMERIC_KINDS:
        raise TypeError(f"signal dtype must be numeric/bool, got {dtype}.")
    return dtype


def _metadata_fingerprint(value):
    """Deterministic equality key for schema metadata (including ndarrays)."""

    if isinstance(value, Mapping):
        return tuple(sorted((str(k), _metadata_fingerprint(v)) for k, v in value.items()))
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return ("ndarray", array.dtype.str, tuple(array.shape), array.tobytes())
    if isinstance(value, (list, tuple)):
        return tuple(_metadata_fingerprint(v) for v in value)
    if isinstance(value, np.generic):
        return value.item()
    return value


@dataclass(frozen=True)
class SignalSchema:
    """Authoritative axes and meaning of one signal.

    ``point_shape`` and ``data_shape`` must both be present, even for a scalar.
    The product of ``point_shape`` is the physical P size; ``data_shape`` is
    preserved verbatim after it.  ``repeat_capacity`` fixes the allocated R
    axis when known; ``None`` binds R from the first full tensor.

    A ``None`` dtype is allowed only until the first value from hardware.  The
    hub binds it on that publish and validates exact dtype equality thereafter.
    """

    point_shape: tuple[int, ...] = (1,)
    data_shape: tuple[int, ...] = (1,)
    dtype: np.dtype | str | type | None = None
    repeat_capacity: int | None = None
    label: str = ""
    unit: str = ""
    description: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "point_shape", _positive_shape(self.point_shape, "point_shape"))
        object.__setattr__(self, "data_shape", _positive_shape(self.data_shape, "data_shape"))
        object.__setattr__(self, "dtype", _dtype_or_none(self.dtype))
        if self.repeat_capacity is not None and int(self.repeat_capacity) < 1:
            raise ValueError(
                f"repeat_capacity must be >= 1 or None, got {self.repeat_capacity!r}.")
        if self.repeat_capacity is not None:
            object.__setattr__(self, "repeat_capacity", int(self.repeat_capacity))
        object.__setattr__(self, "label", str(self.label))
        object.__setattr__(self, "unit", str(self.unit))
        object.__setattr__(self, "description", str(self.description))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @property
    def point_count(self) -> int:
        return int(np.prod(self.point_shape, dtype=np.int64))

    @property
    def physical_ndim(self) -> int:
        return 2 + len(self.data_shape)

    def physical_shape(self, repeat: int | None = None) -> tuple[int, ...]:
        r = self.repeat_capacity if repeat is None else int(repeat)
        if r is None:
            raise ValueError("repeat size is not bound; publish a full tensor first or declare repeat_capacity.")
        if int(r) < 1:
            raise ValueError(f"repeat size must be >= 1, got {r!r}.")
        if self.repeat_capacity is not None and int(r) > self.repeat_capacity:
            raise ValueError(
                f"repeat size {r} exceeds schema repeat_capacity {self.repeat_capacity}.")
        return (int(r), self.point_count, *self.data_shape)

    def logical_shape(self, repeat: int) -> tuple[int, ...]:
        """Axis-aware shape: ``(R, *point_shape, *data_shape)``."""

        return (int(repeat), *self.point_shape, *self.data_shape)

    def bind_dtype(self, dtype) -> "SignalSchema":
        actual = _dtype_or_none(dtype)
        if actual is None:  # pragma: no cover - callers pass an ndarray dtype
            raise TypeError("cannot bind a None signal dtype")
        if self.dtype is not None and self.dtype != actual:
            raise TypeError(f"schema dtype is {self.dtype}, cannot bind {actual}.")
        return self if self.dtype == actual else replace(self, dtype=actual)

    def assert_compatible(self, other: "SignalSchema", *, check_repeat: bool = True) -> None:
        mismatches = []
        for key in ("point_shape", "data_shape", "dtype"):
            left, right = getattr(self, key), getattr(other, key)
            if left is not None and right is not None and left != right:
                mismatches.append(f"{key}: {left!r} != {right!r}")
        if check_repeat and self.repeat_capacity is not None and other.repeat_capacity is not None \
                and self.repeat_capacity != other.repeat_capacity:
            mismatches.append(
                f"repeat_capacity: {self.repeat_capacity!r} != {other.repeat_capacity!r}")
        if mismatches:
            raise ValueError("incompatible signal schema (" + "; ".join(mismatches) + ")")

    def same_definition(self, other: "SignalSchema") -> bool:
        """Exact registry equality, including coordinates/field metadata."""

        if not isinstance(other, SignalSchema):
            return False
        return (
            self.point_shape == other.point_shape
            and self.data_shape == other.data_shape
            and self.dtype == other.dtype
            and self.repeat_capacity == other.repeat_capacity
            and self.label == other.label
            and self.unit == other.unit
            and self.description == other.description
            and _metadata_fingerprint(self.metadata) == _metadata_fingerprint(other.metadata)
        )

    def canonicalize(self, value, *, name: str = "signal") -> tuple[np.ndarray, np.ndarray]:
        """Normalize ``value`` to ``(R,P,*data_shape)`` plus ``(R,P)`` validity.

        With a registered schema there is no guessing.  Accepted forms are the
        physical shape, its declared logical point shape, the logical shape
        without repeat (R=1), and a scalar only for ``P=1,data_shape=(1,)``.
        """

        if isinstance(value, SignalTensor):
            self.assert_compatible(value.schema, check_repeat=False)
            array = value.data
            valid = value.valid
        else:
            array = _numeric_array(value, name=name)
            if array.ndim == 0:
                if self.point_count != 1 or self.data_shape != (1,):
                    raise ValueError(
                        f"{name} scalar cannot satisfy physical tail "
                        f"(P={self.point_count}, data_shape={self.data_shape}).")
                array = array.reshape(1, 1, 1)
            elif array.ndim == self.physical_ndim \
                    and array.shape[1:] == (self.point_count, *self.data_shape):
                pass
            elif array.ndim == 1 + len(self.point_shape) + len(self.data_shape) \
                    and array.shape[1:] == (*self.point_shape, *self.data_shape):
                array = array.reshape(array.shape[0], self.point_count, *self.data_shape)
            elif array.shape == (*self.point_shape, *self.data_shape):
                array = array.reshape(1, self.point_count, *self.data_shape)
            elif self.point_count == 1 and array.shape == self.data_shape:
                # A schema-known single datum may omit both mandatory leading
                # axes at the API boundary.  Their meaning comes from the
                # schema (R=P=1), never from ndarray rank.
                array = array.reshape(1, 1, *self.data_shape)
            else:
                raise ValueError(
                    f"{name} has shape {array.shape}; schema requires physical "
                    f"(R, {self.point_count}, *{self.data_shape}) or logical "
                    f"(R, *{self.point_shape}, *{self.data_shape}).")
            valid = np.ones(array.shape[:2], dtype=bool)
        if array.ndim != self.physical_ndim \
                or array.shape[1:] != (self.point_count, *self.data_shape):
            raise ValueError(
                f"{name} physical shape must be (R,{self.point_count},*{self.data_shape}); "
                f"got {array.shape}.")
        self.physical_shape(array.shape[0])
        if self.dtype is not None and array.dtype != self.dtype:
            raise TypeError(f"{name} dtype {array.dtype} does not match registered dtype {self.dtype}.")
        valid = np.asarray(valid, dtype=bool)
        if valid.shape != array.shape[:2]:
            raise ValueError(
                f"{name} validity mask must have shape {array.shape[:2]}, got {valid.shape}.")
        return array, valid

    @classmethod
    def external(cls, value, *, label: str = "", unit: str = "") -> "SignalSchema":
        """Deterministic schema for an unregistered raw publish.

        The entire input is one datum.  Rank does not imply repeat/points.
        Callers that mean anything else must register or publish SignalTensor.
        """

        array = _numeric_array(value)
        data_shape = tuple(int(n) for n in array.shape) if array.ndim else (1,)
        return cls(point_shape=(1,), data_shape=data_shape, dtype=array.dtype,
                   repeat_capacity=1, label=label, unit=unit,
                   metadata={"origin": "external-single-datum"})


@dataclass(frozen=True)
class SignalTensor:
    """One immutable typed signal value in canonical physical form."""

    data: np.ndarray
    schema: SignalSchema
    valid: np.ndarray | None = None
    version: int = 0
    provenance: int = -1
    schema_version: int = 1

    def __post_init__(self) -> None:
        schema = self.schema
        array = _numeric_array(self.data)
        if schema.dtype is None:
            schema = schema.bind_dtype(array.dtype)
            object.__setattr__(self, "schema", schema)
        array, inferred_valid = schema.canonicalize(array)
        valid = inferred_valid if self.valid is None else np.asarray(self.valid, dtype=bool)
        if valid.shape != array.shape[:2]:
            raise ValueError(f"validity mask must have shape {array.shape[:2]}, got {valid.shape}.")
        owned = np.array(array, copy=True)
        mask = np.array(valid, dtype=bool, copy=True)
        owned.setflags(write=False)
        mask.setflags(write=False)
        object.__setattr__(self, "data", owned)
        object.__setattr__(self, "valid", mask)
        object.__setattr__(self, "version", int(self.version))
        object.__setattr__(self, "provenance", int(self.provenance))
        object.__setattr__(self, "schema_version", int(self.schema_version))

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(n) for n in self.data.shape)

    def logical(self, *, copy: bool = True) -> np.ndarray:
        """Restore ``(R,*point_shape,*data_shape)`` without guessing."""

        out = self.data.reshape(self.schema.logical_shape(self.data.shape[0]))
        return np.array(out, copy=True) if copy else out

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        array = np.asarray(self.data, dtype=dtype)
        return np.array(array, copy=True) if copy else array

    @classmethod
    def from_value(cls, value, schema: SignalSchema, **metadata) -> "SignalTensor":
        array = _numeric_array(value)
        bound = schema.bind_dtype(array.dtype) if schema.dtype is None else schema
        data, valid = bound.canonicalize(value)
        return cls(data, bound, valid=valid, **metadata)

    @classmethod
    def from_external(cls, value, **metadata) -> "SignalTensor":
        schema = SignalSchema.external(value)
        array = _numeric_array(value)
        if array.ndim == 0:
            data = array.reshape(1, 1, 1)
        else:
            data = array.reshape(1, 1, *array.shape)
        return cls(data, schema, **metadata)


@dataclass(frozen=True)
class TensorPatch:
    """An in-place update of a registered tensor store.

    ``index`` addresses at least ``(repeat, point)``.  A two-item index means
    the entire trailing data value; otherwise it must match the tensor's actual
    variable rank.  Patch values match the selected target exactly—implicit
    broadcasting is forbidden.  ``valid`` updates the selected R/P cells.
    """

    index: tuple[Any, ...]
    values: Any
    valid: Any = True
    expected_version: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.index, tuple) or len(self.index) < 2:
            raise ValueError(
                f"TensorPatch.index must start with (repeat, point); got {self.index!r}.")
        if self.expected_version is not None and int(self.expected_version) < 0:
            raise ValueError("expected_version must be >= 0 or None.")

    @classmethod
    def point(cls, repeat: int, point: int, values, *, valid=True,
              expected_version: int | None = None) -> "TensorPatch":
        return cls((int(repeat), int(point)), values,
                   valid=valid, expected_version=expected_version)


@dataclass(frozen=True)
class TensorHistory:
    """Version history with an explicit leading update axis.

    ``data`` is ``(updates,R,P,*data_shape)``.  This update axis is history
    metadata, never confused with a physical signal axis.
    """

    data: np.ndarray
    valid: np.ndarray
    schema: SignalSchema
    versions: tuple[int, ...]
    provenance: tuple[int, ...]

    def logical(self, *, copy: bool = True) -> np.ndarray:
        shape = (self.data.shape[0], self.data.shape[1],
                 *self.schema.point_shape, *self.schema.data_shape)
        out = self.data.reshape(shape)
        return np.array(out, copy=True) if copy else out

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        array = np.asarray(self.data, dtype=dtype)
        return np.array(array, copy=True) if copy else array


@dataclass
class _StoreUpdate:
    index: tuple[Any, ...]
    after: np.ndarray
    valid_index: tuple[Any, Any]
    valid_after: np.ndarray
    version: int
    provenance: int

    @property
    def payload_nbytes(self) -> int:
        return int(self.after.nbytes + self.valid_after.nbytes)


class TensorStore:
    """One schema-bound current tensor plus a forward delta journal.

    A cumulative scan writes one point at a time.  The journal stores only that
    point's trailing data payload.  ``_base_*`` is the exact state immediately
    before the first retained delta; bounded eviction advances that checkpoint.
    History and subscriptions therefore replay one canonical forward stream.
    Storage is O(R*P*prod(data_shape)), not O(P*R*P*prod(data_shape)).
    """

    def __init__(self, schema: SignalSchema, *, history: int, schema_version: int = 1,
                 initialize: bool = False, fill_value=None):
        if int(history) < 1:
            raise ValueError("TensorStore history must be >= 1.")
        self.schema = schema
        self.schema_version = int(schema_version)
        self._history_limit = int(history)
        self._updates: OrderedDict[int, _StoreUpdate] = OrderedDict()
        self._data: np.ndarray | None = None
        self._valid: np.ndarray | None = None
        self._base_data: np.ndarray | None = None
        self._base_valid: np.ndarray | None = None
        self._base_version = 0
        self.version = 0
        self.provenance = -1
        self.patch_updates = 0
        self.full_updates = 0
        self.bytes_copied_in = 0
        if initialize:
            self.initialize(fill_value=fill_value)

    @property
    def initialized(self) -> bool:
        return self._data is not None

    @property
    def history_limit(self) -> int:
        return self._history_limit

    @property
    def cursor(self) -> SignalCursor:
        """Cursor immediately after the latest stored update."""

        return SignalCursor(self.schema_version, self.version)

    def _validated_cursor(self, cursor: SignalCursor) -> SignalCursor:
        if not isinstance(cursor, SignalCursor):
            raise TypeError("cursor must be a SignalCursor.")
        wildcard = cursor == SignalCursor(0, 0)
        if cursor.schema_version == 0 and not wildcard:
            raise SignalHistoryGap("only SignalCursor(0, 0) is a valid wildcard cursor.")
        if not wildcard and cursor.schema_version != self.schema_version:
            raise SignalHistoryGap(
                f"signal schema changed from version {cursor.schema_version} to "
                f"{self.schema_version}; the old cursor cannot cross layouts.")
        if cursor.version > self.version:
            raise SignalHistoryGap(
                f"cursor version {cursor.version} is ahead of current version {self.version}.")
        if cursor.version < self._base_version:
            earliest = self._base_version + 1
            raise SignalHistoryGap(
                f"signal update gap: cursor={cursor.version}, earliest retained={earliest}, "
                f"current={self.version}, journal_limit={self._history_limit}.")
        return SignalCursor(self.schema_version, cursor.version)

    def seed_for_cursor(
        self, cursor: SignalCursor,
    ) -> tuple[SignalCursor, np.ndarray | None, np.ndarray | None]:
        """Copy the exact state at ``cursor`` once for a forward subscription.

        Every later state is obtained by :meth:`apply_update_to`; no subscriber
        re-scans the journal or reconstructs again from the current tensor.
        """

        bound = self._validated_cursor(cursor)
        if not self.initialized:
            return bound, None, None
        assert self._data is not None and self._valid is not None
        if bound.version == self.version:
            return bound, self._data.copy(), self._valid.copy()
        assert self._base_data is not None and self._base_valid is not None
        data = self._base_data.copy()
        valid = self._base_valid.copy()
        for version in range(self._base_version + 1, bound.version + 1):
            update = self._updates.get(version)
            if update is None:  # guarded by _validated_cursor; defensive corruption check
                raise SignalHistoryGap(f"signal update {version} is missing from the retained journal.")
            self._apply(data, valid, update)
        return bound, data, valid

    def validate_cursor(self, cursor: SignalCursor) -> SignalCursor:
        """Validate a subscription position without allocating update metadata."""

        return self._validated_cursor(cursor)

    def next_update_ref(self, cursor: SignalCursor) -> TensorUpdateRef | None:
        """Return exactly the next retained delta after ``cursor`` in O(1)."""

        bound = self._validated_cursor(cursor)
        version = bound.version + 1
        if version > self.version:
            return None
        update = self._updates.get(version)
        if update is None:  # a bounded eviction between cursor and current
            earliest = self._base_version + 1
            raise SignalHistoryGap(
                f"signal update gap: cursor={bound.version}, earliest retained={earliest}, "
                f"current={self.version}, journal_limit={self._history_limit}.")
        return TensorUpdateRef(SignalCursor(self.schema_version, version), int(update.provenance))

    def apply_update_to(
        self, data: np.ndarray, valid: np.ndarray, ref: TensorUpdateRef,
    ) -> None:
        """Apply one exact retained forward delta to subscription-owned state."""

        cursor = self._validated_cursor(ref.cursor)
        update = self._updates.get(cursor.version)
        if update is None:
            earliest = self._base_version + 1
            raise SignalHistoryGap(
                f"signal update {cursor.version} was evicted; earliest retained={earliest}.")
        if int(update.provenance) != int(ref.provenance):
            raise SignalHistoryGap(
                f"signal update {cursor.version} provenance changed inside the journal.")
        self._apply(data, valid, update)

    @staticmethod
    def _apply(data: np.ndarray, valid: np.ndarray, update: _StoreUpdate) -> None:
        data[update.index] = update.after
        valid[update.valid_index] = update.valid_after

    def _evict_oldest(self) -> None:
        _, update = self._updates.popitem(last=False)
        assert self._base_data is not None and self._base_valid is not None
        self._apply(self._base_data, self._base_valid, update)
        self._base_version = int(update.version)

    def set_history_limit(self, history: int) -> None:
        history = max(1, int(history))
        if history == self._history_limit:
            return
        self._history_limit = history
        while len(self._updates) > history:
            self._evict_oldest()

    def initialize(self, *, repeat: int | None = None, fill_value=None) -> None:
        if self.schema.dtype is None:
            raise TypeError("cannot initialize TensorStore before the schema dtype is bound.")
        shape = self.schema.physical_shape(repeat)
        if fill_value is None:
            fill_value = np.nan if self.schema.dtype.kind in "fc" else 0
        self._data = np.full(shape, fill_value, dtype=self.schema.dtype)
        self._valid = np.zeros(shape[:2], dtype=bool)
        self._updates.clear()
        self.version = 0
        self.provenance = -1
        self._base_data = self._data.copy()
        self._base_valid = self._valid.copy()
        self._base_version = 0

    def _ensure_for_tensor(self, tensor: SignalTensor) -> None:
        self.schema.assert_compatible(tensor.schema, check_repeat=False)
        if self.schema.dtype is None:
            self.schema = self.schema.bind_dtype(tensor.data.dtype)
        if not self.initialized:
            repeat = self.schema.repeat_capacity or int(tensor.data.shape[0])
            self.initialize(repeat=repeat)
        assert self._data is not None
        if tensor.data.shape[0] > self._data.shape[0]:
            raise ValueError(
                f"tensor repeat size {tensor.data.shape[0]} exceeds allocated store {self._data.shape[0]}; "
                "declare repeat_capacity before publishing.")

    def _append(self, *, index, after, valid_index, valid_after, provenance: int) -> int:
        self.version += 1
        self.provenance = int(provenance)
        update = _StoreUpdate(
            index=index,
            after=np.array(after, copy=True),
            valid_index=valid_index,
            valid_after=np.array(valid_after, dtype=bool, copy=True),
            version=self.version,
            provenance=self.provenance,
        )
        if len(self._updates) >= self._history_limit:
            self._evict_oldest()
        self._updates[self.version] = update
        return self.version

    def validate_tensor(self, tensor: SignalTensor) -> None:
        self.schema.assert_compatible(tensor.schema, check_repeat=False)
        if self.schema.dtype is not None and tensor.data.dtype != self.schema.dtype:
            raise TypeError(
                f"tensor dtype {tensor.data.dtype} does not match store dtype {self.schema.dtype}.")
        self.schema.physical_shape(tensor.data.shape[0])
        if self._data is not None and tensor.data.shape[0] > self._data.shape[0]:
            raise ValueError(
                f"tensor repeat size {tensor.data.shape[0]} exceeds allocated store {self._data.shape[0]}; "
                "replace the schema explicitly or publish patches into a declared repeat_capacity.")

    def publish(self, tensor: SignalTensor, *, provenance: int) -> int:
        self.validate_tensor(tensor)
        self._ensure_for_tensor(tensor)
        assert self._data is not None and self._valid is not None
        self._valid[:] = False
        r = int(tensor.data.shape[0])
        self._data[:r] = tensor.data
        self._valid[:r] = tensor.valid
        if r < self._data.shape[0]:
            fill = np.nan if self._data.dtype.kind in "fc" else 0
            self._data[r:] = fill
        full_index = tuple(slice(None) for _ in range(self._data.ndim))
        self.full_updates += 1
        self.bytes_copied_in += int(tensor.data.nbytes)
        return self._append(index=full_index, after=self._data,
                            valid_index=(slice(None), slice(None)),
                            valid_after=self._valid, provenance=provenance)

    def _expanded_index(self, patch: TensorPatch) -> tuple[Any, ...]:
        assert self._data is not None
        index = tuple(patch.index)
        if len(index) == 2:
            index = (*index, *(slice(None) for _ in self.schema.data_shape))
        if len(index) != self._data.ndim:
            raise ValueError(
                f"tensor patch index has {len(index)} axes, but signal physical ndim is "
                f"{self._data.ndim} for data_shape={self.schema.data_shape}.")
        return index

    def validate_patch(self, patch: TensorPatch) -> tuple[tuple[Any, ...], np.ndarray, np.ndarray]:
        """Validate a patch without mutating published data (hub atomic preflight)."""

        if not self.initialized:
            self.initialize()
        assert self._data is not None and self._valid is not None
        if patch.expected_version is not None and int(patch.expected_version) != self.version:
            raise RuntimeError(
                f"stale tensor patch: expected version {patch.expected_version}, current {self.version}.")
        index = self._expanded_index(patch)
        target = self._data[index]
        values = _numeric_array(patch.values, name="tensor patch")
        if values.dtype != self._data.dtype:
            raise TypeError(
                f"tensor patch dtype {values.dtype} does not match store dtype {self._data.dtype}.")
        if values.shape != target.shape:
            raise ValueError(
                f"tensor patch values shape {values.shape} does not match target {target.shape}; "
                "implicit broadcasting is forbidden.")
        valid_index = (index[0], index[1])
        valid_target = self._valid[valid_index]
        valid_after = np.asarray(patch.valid, dtype=bool)
        if valid_after.ndim == 0:
            valid_after = np.full(np.shape(valid_target), bool(valid_after), dtype=bool)
        if valid_after.shape != np.shape(valid_target):
            raise ValueError(
                f"tensor patch valid shape {valid_after.shape} does not match selected "
                f"repeat/point cells {np.shape(valid_target)}.")
        return index, values, valid_after

    def patch(self, patch: TensorPatch, *, provenance: int) -> int:
        index, values, valid_after = self.validate_patch(patch)
        assert self._data is not None and self._valid is not None
        valid_index = (index[0], index[1])
        self._data[index] = values
        self._valid[valid_index] = valid_after
        self.patch_updates += 1
        self.bytes_copied_in += int(values.nbytes)
        return self._append(index=index, after=self._data[index],
                            valid_index=valid_index, valid_after=self._valid[valid_index],
                            provenance=provenance)

    def snapshot(self) -> SignalTensor:
        if not self.initialized:
            raise KeyError("signal has a registered schema but no published value")
        assert self._data is not None and self._valid is not None
        return SignalTensor(self._data, self.schema, valid=self._valid,
                            version=self.version, provenance=self.provenance,
                            schema_version=self.schema_version)

    def history(self, n: int | None = None) -> TensorHistory:
        if not self.initialized or not self._updates:
            raise KeyError("signal has no published history")
        assert self._data is not None and self._valid is not None
        updates = list(self._updates.values())
        take = len(updates) if n is None else min(len(updates), max(1, int(n)))
        start = len(updates) - take
        assert self._base_data is not None and self._base_valid is not None
        data = self._base_data.copy()
        valid = self._base_valid.copy()
        # Allocate the returned history once and fill it while walking the
        # forward journal.  Appending per-version copies and then np.stack-ing
        # them retained both representations simultaneously (about 103 MiB peak
        # for a 50 MiB P4096,D=(2,3),N=256 history read).
        states = np.empty((take, *data.shape), dtype=data.dtype)
        masks = np.empty((take, *valid.shape), dtype=bool)
        versions = [0] * take
        provenance = [0] * take
        for update in updates[:start]:
            self._apply(data, valid, update)
        for index in range(start, len(updates)):
            update = updates[index]
            self._apply(data, valid, update)
            output = index - start
            states[output] = data
            masks[output] = valid
            versions[output] = update.version
            provenance[output] = update.provenance
        return TensorHistory(states, masks, self.schema, tuple(versions), tuple(provenance))

    def at_provenance(self, target: int) -> SignalTensor | None:
        if not self.initialized:
            return None
        match = None
        for update in self._updates.values():
            if update.provenance == int(target):
                match = update
        if match is None:
            return None
        _, data, valid = self.seed_for_cursor(
            SignalCursor(self.schema_version, match.version))
        assert data is not None and valid is not None
        return SignalTensor(data, self.schema, valid=valid, version=match.version,
                            provenance=match.provenance, schema_version=self.schema_version)

    @property
    def journal_payload_nbytes(self) -> int:
        return sum(update.payload_nbytes for update in self._updates.values())

    @property
    def storage_nbytes(self) -> int:
        current = 0 if self._data is None else int(self._data.nbytes)
        current_mask = 0 if self._valid is None else int(self._valid.nbytes)
        base = 0 if self._base_data is None else int(self._base_data.nbytes)
        base_mask = 0 if self._base_valid is None else int(self._base_valid.nbytes)
        return current + current_mask + base + base_mask + self.journal_payload_nbytes

    def stats(self) -> dict[str, int]:
        return {
            "version": int(self.version),
            "full_updates": int(self.full_updates),
            "patch_updates": int(self.patch_updates),
            "bytes_copied_in": int(self.bytes_copied_in),
            "journal_payload_nbytes": int(self.journal_payload_nbytes),
            "storage_nbytes": int(self.storage_nbytes),
        }


__all__ = [
    "SignalCursor",
    "SignalHistoryGap",
    "SignalSchema",
    "SignalTensor",
    "TensorHistory",
    "TensorPatch",
    "TensorStore",
    "TensorUpdateRef",
]
