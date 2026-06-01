"""Smart tick locator/formatter used by Zou lab front-end figures."""

from __future__ import annotations

import types
from typing import Sequence

import matplotlib.ticker as ticker
import numpy as np


class SmartOffsetLocator(ticker.Locator):
    """Locator that separates a large common offset from compact tick labels."""

    def __init__(
        self,
        steps: Sequence[int] = (1, 2, 5),
        min_ticks: int = 3,
        max_ticks: int = 8,
        oom: int = 3,
    ):
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
                self.step = step
                self.m = exp_part
                self.k = 0
                chosen = True
                break
            if self.min_ticks <= float_part * 10 / step <= self.max_ticks:
                self.step = step
                self.m = exp_part - 1
                self.k = 0
                chosen = True
                break
        if not chosen:
            self.step = 1
            self.m = exp_part
            self.k = 0

        ave = 0.5 * (vmin_order + vmax_order)
        self.C_int = int(round(ave / 10 ** (self.m + self.k + self.oom)))
        self.C_exp = int(round(self.m + self.k + self.oom))
        self.C = self.C_int * 10**self.C_exp

        n_min = int(np.ceil((vmin_order - self.C) * 10 ** (-self.m - self.k) / self.step))
        n_max = int(np.floor((vmax_order - self.C) * 10 ** (-self.m - self.k) / self.step))
        self.n_array = list(range(n_min, n_max + 1))
        if vmin > vmax:
            self.n_array = self.n_array[::-1]

        unit = self.step * 10 ** (self.k + self.m)
        self.ticks = [n * unit + self.C for n in self.n_array]

        if self.n_array:
            if (self.m <= -self.oom) or max(np.abs(self.n_array)) * self.step * 10**self.m >= 10 ** (
                self.oom + 1
            ):
                self.k = self.m
                self.m = 0
            else:
                self.k = 0
        return self.ticks

    def __call__(self):
        if self.axis is None:
            return []
        vmin, vmax = self.axis.get_view_interval()
        if vmax == vmin:
            return []
        return self.tick_values(vmin, vmax)


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
    ):
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

        def _apply(off):
            if self._offset_xy is None:
                return
            off.set_transform(axis.axes.transAxes if self._offset_coords == "axes" else axis.axes.transData)
            off.set_position(self._offset_xy)
            if self._offset_ha is not None:
                off.set_ha(self._offset_ha)
            if self._offset_va is not None:
                off.set_va(self._offset_va)
            off.set_clip_on(False)
            off.set_visible(True)

        needs_patch = getattr(axis, "_smart_offset_patched_by", None) is not self
        if needs_patch and hasattr(axis, "_update_offset_text_position"):
            if not hasattr(axis, "_smart_offset_orig_uotp"):
                axis._smart_offset_orig_uotp = axis._update_offset_text_position

            def _patched_uotp(_self, *args, **kwargs):
                ret = _self._smart_offset_orig_uotp(*args, **kwargs)
                _apply(_self.get_offset_text())
                return ret

            axis._update_offset_text_position = types.MethodType(_patched_uotp, axis)
            axis._smart_offset_patched_by = self

    def set_locs(self, locs):
        self.locs = np.asarray(locs, dtype=float)
        try:
            self.abs_step = abs(self.locator.step * 10 ** (self.locator.k + self.locator.m))
        except Exception:
            self.abs_step = 1.0

    @staticmethod
    def _fmt_scaled_int(value_int: int, exp10: int, force_sign: bool = False) -> str:
        v = int(value_int)
        if v == 0:
            return "+0" if force_sign else "0"
        sign = "-" if v < 0 else ("+" if force_sign else "")
        base = abs(v)
        if exp10 >= 0:
            return sign + str(base * (10**exp10))
        denom = 10 ** (-exp10)
        q, r = divmod(base, denom)
        frac = f"{r:0{-exp10}d}".rstrip("0")
        return f"{sign}{q}.{frac}" if frac else f"{sign}{q}"

    def __call__(self, x, pos=None):
        try:
            x = float(x)
        except Exception:
            return ""
        if not np.isfinite(x) or not getattr(self.locator, "ticks", None):
            return ""
        idx = int(np.argmin([abs(x - t) for t in self.locator.ticks]))
        n = self.locator.n_array[idx]
        base_int = int(n * self.locator.step)
        return self._fmt_scaled_int(base_int, int(self.locator.m), force_sign=False)

    def _format_C(self) -> str:
        plain = self._fmt_scaled_int(self.locator.C_int, int(self.locator.C_exp), force_sign=True)
        if plain in ("", "+0", "-0"):
            return ""
        if len(plain) <= self.C_maxlen:
            return plain
        cint = int(self.locator.C_int)
        sign = "-" if cint < 0 else "+"
        digits = str(abs(cint))
        if digits == "0":
            return ""
        sci_exp = int(self.locator.C_exp) + len(digits) - 1
        exp_str = f"e{sci_exp:d}"
        keep = max(0, self.C_maxlen - 2 - len(exp_str))
        return sign + digits[0] + "." + digits[1:keep] + exp_str

    def get_offset(self):
        parts = []
        if self.locator.k != 0:
            parts.append(f"×1e{self.locator.k}")
        cstr = self._format_C()
        if cstr:
            parts.append(cstr)
        if not parts:
            return ""
        if self.axis_type == "x" and len(parts) == 2:
            return parts[0] + "\n" + parts[1]
        return "".join(parts)


def apply_smart_ticks(ax, axis: str = "both") -> None:
    """Apply smart offset ticks to one or both axes."""
    if axis in ("x", "both"):
        xloc = SmartOffsetLocator()
        ax.xaxis.set_major_locator(xloc)
        ax.xaxis.set_major_formatter(
            SmartOffsetFormatter(xloc, axis_type="x", offset_xy=(0.9, -0.1), offset_ha="left", offset_va="top")
        )
    if axis in ("y", "both"):
        yloc = SmartOffsetLocator()
        ax.yaxis.set_major_locator(yloc)
        ax.yaxis.set_major_formatter(
            SmartOffsetFormatter(yloc, axis_type="y", offset_xy=(0.0, 1.005), offset_ha="left", offset_va="bottom")
        )


__all__ = ["SmartOffsetFormatter", "SmartOffsetLocator", "apply_smart_ticks"]

