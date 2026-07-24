"""Physical value validation shared by the three calibrated Measurements."""

from __future__ import annotations

from decimal import Decimal, localcontext
from math import isfinite
from numbers import Integral, Real

from zlc_neutral_atom.logic_nodes.calibration.calibration import ReadoutModelKind
from zlc_pulse import PulseDocument
from zlc_storage import canonical_text, finite_real


def numeric_axis(
    values: object,
    name: str,
    *,
    positive: bool,
) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a numeric sequence")
    try:
        raw = tuple(values)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(f"{name} must be a numeric sequence") from exc
    if not raw:
        raise ValueError(f"{name} must contain at least one value")
    result = []
    for value in raw:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{name} values must be real numbers")
        item = float(value)
        if not isfinite(item) or (positive and item <= 0.0) or (
            not positive and item < 0.0
        ):
            qualifier = "positive" if positive else "non-negative"
            raise ValueError(f"{name} values must be finite and {qualifier}")
        result.append(item)
    return tuple(result)


def finite_signed_axis(values: object, name: str) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a numeric sequence")
    try:
        raw = tuple(values)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(f"{name} must be a numeric sequence") from exc
    if not raw:
        raise ValueError(f"{name} must contain at least one value")
    result = []
    for value in raw:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{name} values must be real numbers")
        item = float(value)
        if not isfinite(item):
            raise ValueError(f"{name} values must be finite")
        result.append(item)
    return tuple(result)


def linear_axis_from_range(
    value: object,
    name: str,
    *,
    scale: float,
    positive: bool,
) -> tuple[float, ...]:
    """Resolve one authored ``(start, stop, count)`` into physical values."""

    if isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be (minimum, maximum, points)")
    try:
        start, stop, count = tuple(value)  # type: ignore[misc]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be (minimum, maximum, points)") from exc
    if isinstance(start, bool) or not isinstance(start, Real):
        raise TypeError(f"{name} minimum must be a real number")
    if isinstance(stop, bool) or not isinstance(stop, Real):
        raise TypeError(f"{name} maximum must be a real number")
    if isinstance(count, bool) or not isinstance(count, Integral):
        raise TypeError(f"{name} points must be an integer")
    points = int(count)
    if points < 1:
        raise ValueError(f"{name} points must be positive")
    start_value = finite_real(start, f"{name} minimum")
    stop_value = finite_real(stop, f"{name} maximum")
    factor = finite_real(scale, f"{name} unit scale", positive=True)
    start_decimal = Decimal(str(start_value))
    stop_decimal = Decimal(str(stop_value))
    factor_decimal = Decimal(str(factor))
    if points == 1:
        authored_values = (start_decimal,)
    else:
        with localcontext() as context:
            context.prec = 50
            interval = stop_decimal - start_decimal
            denominator = Decimal(points - 1)
            authored_values = tuple(
                start_decimal
                if index == 0
                else stop_decimal
                if index == points - 1
                else start_decimal + interval * Decimal(index) / denominator
                for index in range(points)
            )
    result = tuple(float(item * factor_decimal) for item in authored_values)
    if positive and any(item <= 0.0 for item in result):
        raise ValueError(f"{name} values must be positive")
    return result


def scale_authored_value(value: object, scale: object, name: str) -> float:
    """Scale an authored decimal without inventing a sub-tick residue."""

    authored = finite_real(value, name)
    factor = finite_real(scale, f"{name} unit scale", positive=True)
    return float(Decimal(str(authored)) * Decimal(str(factor)))


def optional_trigger(value: str | None) -> str | None:
    return None if value is None else canonical_text(value, "trigger_channel")


def duration_axis_for_document(
    values: object,
    name: str,
    document: PulseDocument,
) -> tuple[float, ...]:
    axis = numeric_axis(values, name, positive=True)
    minimum_seconds = document.time_step_ns * 1e-9
    if any(value < minimum_seconds for value in axis):
        raise ValueError(
            f"{name} values must be at least one pulse target clock tick "
            f"({minimum_seconds:.12g} s)"
        )
    return axis


def readout_model_kind(
    value: ReadoutModelKind | None,
) -> ReadoutModelKind | None:
    if value is not None and not isinstance(value, ReadoutModelKind):
        raise TypeError("model_kind must be ReadoutModelKind or None")
    return value


__all__ = [
    "duration_axis_for_document",
    "finite_signed_axis",
    "linear_axis_from_range",
    "numeric_axis",
    "optional_trigger",
    "readout_model_kind",
    "scale_authored_value",
]
