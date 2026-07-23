"""Main-compatible smart tick locator/formatter for frontend figures."""

from __future__ import annotations

import types
from collections.abc import Sequence

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
        self.steps = list(steps)
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
        self.axis = None

    def set_axis(self, axis):
        self.axis = axis
        return super().set_axis(axis)

    def tick_values(self, vmin, vmax):
        vmin_order, vmax_order = np.sort([vmin, vmax])
        delta = vmax_order - vmin_order
        if not np.isfinite(delta) or delta == 0:
            self.ticks = []
            self.n_array = []
            return self.ticks
        exp_part = int(np.floor(np.log10(delta)))
        float_part = delta / 10**exp_part
        chosen = False
        for step in self.steps:
            if self.min_ticks <= float_part / step <= self.max_ticks:
                self.step, self.m, self.k = step, exp_part, 0
                chosen = True
                break
            if self.min_ticks <= float_part * 10 / step <= self.max_ticks:
                self.step, self.m, self.k = step, exp_part - 1, 0
                chosen = True
                break
        if not chosen:
            self.step, self.m, self.k = 1, exp_part, 0
        average = 0.5 * (vmin_order + vmax_order)
        self.C_int = int(round(average / 10 ** (self.m + self.k + self.oom)))
        self.C_exp = int(round(self.m + self.k + self.oom))
        self.C = self.C_int * 10**self.C_exp
        n_min = int(
            np.ceil(
                (vmin_order - self.C)
                * 10 ** (-self.m - self.k)
                / self.step
            )
        )
        n_max = int(
            np.floor(
                (vmax_order - self.C)
                * 10 ** (-self.m - self.k)
                / self.step
            )
        )
        self.n_array = list(range(n_min, n_max + 1))
        if vmin > vmax:
            self.n_array.reverse()
        unit = self.step * 10 ** (self.k + self.m)
        self.ticks = [n * unit + self.C for n in self.n_array]
        if self.n_array:
            if self.m <= -self.oom or (
                max(np.abs(self.n_array)) * self.step * 10**self.m
                >= 10 ** (self.oom + 1)
            ):
                self.k, self.m = self.m, 0
            else:
                self.k = 0
        return self.ticks

    def __call__(self):
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
        self.locator = locator
        self.axis_type = axis_type
        self._offset_xy = offset_xy
        self._offset_coords = offset_coords
        self._offset_ha = offset_ha
        self._offset_va = offset_va
        self.C_maxlen = 8
        self.abs_step = 1.0

    def set_axis(self, axis):
        super().set_axis(axis)

        def apply_offset(offset):
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
            if not hasattr(axis, "_smart_offset_orig_uotp"):
                axis._smart_offset_orig_uotp = axis._update_offset_text_position

            def patched(target, *args, **kwargs):
                result = target._smart_offset_orig_uotp(*args, **kwargs)
                apply_offset(target.get_offset_text())
                return result

            axis._update_offset_text_position = types.MethodType(patched, axis)
            axis._smart_offset_patched_by = self

    def set_locs(self, locs):
        self.locs = np.asarray(locs, dtype=float)
        try:
            self.abs_step = abs(
                self.locator.step
                * 10 ** (self.locator.k + self.locator.m)
            )
        except Exception:
            self.abs_step = 1.0

    @staticmethod
    def _fmt_scaled_int(value_int: int, exp10: int, force_sign=False) -> str:
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
        return (
            f"{sign}{quotient}.{fraction}"
            if fraction
            else f"{sign}{quotient}"
        )

    def __call__(self, value, pos=None):
        del pos
        try:
            value = float(value)
        except Exception:
            return ""
        if not np.isfinite(value) or not self.locator.ticks:
            return ""
        index = int(
            np.argmin([abs(value - tick) for tick in self.locator.ticks])
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
        if len(plain) <= self.C_maxlen:
            return plain
        value = int(self.locator.C_int)
        sign = "-" if value < 0 else "+"
        digits = str(abs(value))
        if digits == "0":
            return ""
        exponent = int(self.locator.C_exp) + len(digits) - 1
        suffix = f"e{exponent:d}"
        keep = max(0, self.C_maxlen - 2 - len(suffix))
        return sign + digits[0] + "." + digits[1:keep] + suffix

    def get_offset(self):
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
    axis,
    which: str = "both",
    *,
    max_ticks_x: int | None = None,
    max_ticks_y: int | None = None,
) -> None:
    """Install main's paired smart locators and formatters."""

    if which in ("x", "both"):
        locator = SmartOffsetLocator(
            max_ticks=int(max_ticks_x) if max_ticks_x else 8
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
        locator = SmartOffsetLocator(
            max_ticks=int(max_ticks_y) if max_ticks_y else 8
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


__all__ = [
    "SmartOffsetFormatter",
    "SmartOffsetLocator",
    "apply_smart_ticks",
]
