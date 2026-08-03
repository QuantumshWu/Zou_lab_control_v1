"""Smart tick locator and formatter for compact scientific plot axes.

The paired objects keep short tick labels around a large common offset and
place the scale/offset text at fixed plot-relative positions.  This
module owns no Figure lifecycle and can be applied identically by notebook,
GUI, and export renderers.
"""

from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_EVEN, localcontext
from numbers import Integral
import types
from typing import Any

import matplotlib.ticker as ticker
import numpy as np


class SmartOffsetLocator(ticker.Locator):
    """Separate a large common offset from short coordinate tick labels."""

    def __init__(
        self,
        steps: Sequence[int] = (1, 2, 5),
        min_ticks: int = 3,
        max_ticks: int = 8,
        oom: int = 3,
    ) -> None:
        super().__init__()
        selected_steps = tuple(steps)
        if not selected_steps or any(
            isinstance(step, bool) or not isinstance(step, Integral) or step <= 0
            for step in selected_steps
        ):
            raise ValueError("steps must contain positive integers")
        if any(
            isinstance(value, bool) or not isinstance(value, Integral) or value <= 0
            for value in (min_ticks, max_ticks, oom)
        ):
            raise ValueError("min_ticks, max_ticks, and oom must be positive integers")
        if min_ticks > max_ticks:
            raise ValueError("min_ticks must not exceed max_ticks")
        self.steps = tuple(int(step) for step in selected_steps)
        self.min_ticks = int(min_ticks)
        self.max_ticks = int(max_ticks)
        self.oom = int(oom)
        self.k = 0
        self.m = 0
        self.C = 0
        self.C_int = 0
        self.C_exp = 0
        self.step = 1
        self.n_array: list[int] = []
        self.ticks: list[float] = []

    def tick_values(self, vmin: float, vmax: float) -> list[float]:
        lower, upper = sorted((float(vmin), float(vmax)))
        if not np.isfinite((lower, upper)).all() or lower == upper:
            self.k = self.m = self.C_int = self.C_exp = 0
            self.C = 0.0
            self.step = 1
            self.ticks = []
            self.n_array = []
            return self.ticks

        # Decimal arithmetic is used only for the small locator calculation.
        # It keeps both subnormal spans and opposite-sign float extremes out of
        # the ``10**exponent`` overflow/underflow paths of binary floats.
        with localcontext() as context:
            context.prec = 32
            lower_decimal = Decimal.from_float(lower)
            upper_decimal = Decimal.from_float(upper)
            delta = upper_decimal - lower_decimal
            exp_part = delta.adjusted()
            float_part = float(delta.scaleb(-exp_part))
            for step in self.steps:
                if self.min_ticks <= float_part / step <= self.max_ticks:
                    self.step, self.m, self.k = step, exp_part, 0
                    break
                if self.min_ticks <= float_part * 10 / step <= self.max_ticks:
                    self.step, self.m, self.k = step, exp_part - 1, 0
                    break
            else:
                self.step, self.m, self.k = 1, exp_part, 0

            self.C_exp = self.m + self.k + self.oom
            average = (lower_decimal + upper_decimal) / 2
            self.C_int = int(
                average.scaleb(-self.C_exp).to_integral_value(
                    rounding=ROUND_HALF_EVEN
                )
            )
            offset = Decimal(self.C_int).scaleb(self.C_exp)
            unit = Decimal(self.step).scaleb(self.m + self.k)
            self.C = float(offset)
            if not np.isfinite(self.C):
                self.C_int = self.C_exp = 0
                self.C = 0.0
                offset = Decimal(0)
            n_min = int(
                ((lower_decimal - offset) / unit).to_integral_value(
                    rounding=ROUND_CEILING
                )
            )
            n_max = int(
                ((upper_decimal - offset) / unit).to_integral_value(
                    rounding=ROUND_FLOOR
                )
            )
            distinct: dict[float, tuple[Decimal, int]] = {}
            for n in range(n_min, n_max + 1):
                target = Decimal(n) * unit + offset
                tick = float(target)
                error = abs(Decimal.from_float(tick) - target)
                if tick not in distinct or error < distinct[tick][0]:
                    distinct[tick] = (error, n)
            self.ticks = list(distinct)
            self.n_array = [value[1] for value in distinct.values()]
            residual = (
                Decimal(max(map(abs, self.n_array), default=0) * self.step)
                .scaleb(self.m)
            )

        if vmin > vmax:
            self.n_array.reverse()
            self.ticks.reverse()
        if self.n_array:
            if self.m <= -self.oom or residual >= Decimal(1).scaleb(
                self.oom + 1
            ):
                self.k, self.m = self.m, 0
            else:
                self.k = 0
        return self.ticks

    def __call__(self) -> list[float]:
        if self.axis is None:
            return []
        vmin, vmax = self.axis.get_view_interval()
        return [] if vmax == vmin else self.tick_values(vmin, vmax)


class SmartOffsetFormatter(ticker.Formatter):
    """Formatter paired with :class:`SmartOffsetLocator`."""

    def __init__(
        self,
        locator: SmartOffsetLocator,
        axis_type: str = "y",
        offset_xy: tuple[float, float] | None = None,
        offset_coords: str = "axes",
        offset_ha: str | None = None,
        offset_va: str | None = None,
    ) -> None:
        super().__init__()
        if axis_type not in {"x", "y"}:
            raise ValueError("axis_type must be 'x' or 'y'")
        self.locator = locator
        self.axis_type = axis_type
        self._offset_xy = offset_xy
        self._offset_coords = offset_coords
        self._offset_ha = offset_ha
        self._offset_va = offset_va

    def set_axis(self, axis: Any) -> None:
        super().set_axis(axis)

        def apply_offset(offset: Any) -> None:
            if self._offset_xy is None:
                return
            offset.set_transform(
                axis.axes.transAxes
                if self._offset_coords == "axes"
                else axis.axes.transData
            )
            offset.set_position(self._offset_xy)
            if self._offset_ha is not None:
                offset.set_ha(self._offset_ha)
            if self._offset_va is not None:
                offset.set_va(self._offset_va)
            offset.set_clip_on(False)
            offset.set_visible(True)

        if (
            getattr(axis, "_smart_offset_patched_by", None) is not self
            and hasattr(axis, "_update_offset_text_position")
        ):
            if not hasattr(axis, "_smart_offset_original_update_position"):
                axis._smart_offset_original_update_position = (
                    axis._update_offset_text_position
                )

            def patched(target: Any, *args: object, **kwargs: object) -> Any:
                result = target._smart_offset_original_update_position(
                    *args,
                    **kwargs,
                )
                apply_offset(target.get_offset_text())
                return result

            axis._update_offset_text_position = types.MethodType(patched, axis)
            axis._smart_offset_patched_by = self

    @staticmethod
    def _fmt_scaled_int(
        value_int: int,
        exp10: int,
        force_sign: bool = False,
    ) -> str:
        value = int(value_int)
        if value == 0:
            return "+0" if force_sign else "0"
        sign = "-" if value < 0 else ("+" if force_sign else "")
        base = abs(value)
        if exp10 >= 0:
            return sign + str(base * 10**exp10)
        denominator = 10 ** (-exp10)
        quotient, remainder = divmod(base, denominator)
        fraction = f"{remainder:0{-exp10}d}".rstrip("0")
        return f"{sign}{quotient}.{fraction}" if fraction else f"{sign}{quotient}"

    def __call__(self, value: float, pos: int | None = None) -> str:
        del pos
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return ""
        if not np.isfinite(numeric) or not self.locator.ticks:
            return ""
        index = int(
            np.argmin([abs(numeric - tick) for tick in self.locator.ticks])
        )
        return self._fmt_scaled_int(
            int(self.locator.n_array[index] * self.locator.step),
            int(self.locator.m),
        )

    def _format_C(self) -> str:
        plain = self._fmt_scaled_int(
            self.locator.C_int,
            int(self.locator.C_exp),
            force_sign=True,
        )
        if plain in ("", "+0", "-0"):
            return ""
        max_length = 8
        if len(plain) <= max_length:
            return plain
        value = int(self.locator.C_int)
        sign = "-" if value < 0 else "+"
        digits = str(abs(value))
        if digits == "0":
            return ""
        exponent = int(self.locator.C_exp) + len(digits) - 1
        suffix = f"e{exponent:d}"
        keep = max(0, max_length - 2 - len(suffix))
        fraction = digits[1:keep]
        return sign + digits[0] + (("." + fraction) if fraction else "") + suffix

    def get_offset(self) -> str:
        parts = []
        if self.locator.k != 0:
            parts.append(f"×1e{self.locator.k}")
        constant = self._format_C()
        if constant:
            parts.append(constant)
        if not parts:
            return ""
        if self.axis_type == "x" and len(parts) == 2:
            return parts[0] + "\n" + parts[1]
        return "".join(parts)


def apply_smart_ticks(
    axis: Any,
    which: str = "both",
    *,
    max_ticks_x: int | None = None,
    max_ticks_y: int | None = None,
) -> None:
    """Install the shared tick policy on ``axis``.

    The compact offset locator is only meaningful on a linear coordinate.
    Applying it to a logarithmic count axis treats log-space as linear data,
    which produces labels such as ``200, 400, 600`` at visually logarithmic
    positions.  Log y therefore has its own decade-aware locator/formatter;
    x and all linear axes continue to use the existing compact policy.
    """

    if which not in {"x", "y", "both"}:
        raise ValueError("which must be 'x', 'y', or 'both'")
    if which in ("x", "both"):
        locator = SmartOffsetLocator(
            max_ticks=8 if max_ticks_x is None else max_ticks_x
        )
        axis.xaxis.set_major_locator(locator)
        axis.xaxis.set_major_formatter(
            SmartOffsetFormatter(
                locator,
                axis_type="x",
                offset_xy=(0.9, -0.1),
                offset_ha="left",
                offset_va="top",
            )
        )
    if which in ("y", "both"):
        if axis.get_yscale() == "log":
            low, high = sorted(map(float, axis.get_ylim()))
            if low <= 0.0 or not np.isfinite((low, high)).all():
                raise ValueError("logarithmic y limits must be finite and positive")
            decades = np.log10(high) - np.log10(low)
            # For a narrow range, 1/2/5 subdivisions are useful major labels;
            # over several decades only decade ticks are labelled and the
            # 2..9 ticks remain visual minor guides.
            if decades <= 1.5:
                major_subs = (1.0, 2.0, 5.0)
                formatter = ticker.LogFormatter(
                    base=10.0,
                    labelOnlyBase=False,
                )
            else:
                major_subs = (1.0,)
                formatter = ticker.LogFormatterMathtext(
                    base=10.0,
                    labelOnlyBase=True,
                )
            axis.yaxis.set_major_locator(
                ticker.LogLocator(
                    base=10.0,
                    subs=major_subs,
                    numticks=8 if max_ticks_y is None else max_ticks_y,
                )
            )
            axis.yaxis.set_major_formatter(formatter)
            axis.yaxis.set_minor_locator(
                ticker.LogLocator(
                    base=10.0,
                    subs=tuple(float(value) for value in range(2, 10)),
                    numticks=16 if max_ticks_y is None else max(2 * max_ticks_y, 8),
                )
            )
            axis.yaxis.set_minor_formatter(ticker.NullFormatter())
            axis.yaxis.get_offset_text().set_visible(False)
        else:
            locator = SmartOffsetLocator(
                max_ticks=8 if max_ticks_y is None else max_ticks_y
            )
            axis.yaxis.set_major_locator(locator)
            axis.yaxis.set_major_formatter(
                SmartOffsetFormatter(
                    locator,
                    axis_type="y",
                    offset_xy=(0.0, 1.005),
                    offset_ha="left",
                    offset_va="bottom",
                )
            )
            # Do not retain logarithmic minor ticks after switching back to a
            # linear histogram.  The linear style deliberately has no minor
            # labels or extra grid generated by this policy.
            axis.yaxis.set_minor_locator(ticker.NullLocator())
            axis.yaxis.set_minor_formatter(ticker.NullFormatter())
            axis.yaxis.get_offset_text().set_visible(True)


__all__ = [
    "SmartOffsetFormatter",
    "SmartOffsetLocator",
    "apply_smart_ticks",
]
