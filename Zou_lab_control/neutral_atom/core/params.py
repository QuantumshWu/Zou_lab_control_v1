"""The ONE declarative parameter record every form-bearing layer shares.

:class:`ParamDecl` is a dependency-free LEAF (this module imports nothing above the
standard library): the operations layer (measurement / processor / task specs), the
devices layer (a device class's :meth:`~..devices.base.BaseDevice.config_params`
config schema) and the frontend's plot-panel params all declare their tunable
parameters with this ONE record, and the ONE param-kind -> widget registry
(``frontend.param_widgets.PARAM_WIDGETS``) renders every kind.  Keeping the record
below the devices layer is what lets a device class describe its own config form
without ``devices -> operations`` (which would be an import cycle).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


#: The config-value prefix that names ANOTHER entry of the same device config
#: (``"$device:sequencer"``): the loader builds that entry first and passes the built
#: device in its place.  Spelled ONCE here -- the loader (``devices.registry``), the
#: reflection layer (``devices.base``) and the device-manager editor all read it.
DEVICE_REF_PREFIX = "$device:"


def is_device_ref(value: Any) -> bool:
    """True when a config param VALUE is a ``$device:<entry>`` cross-reference."""
    return isinstance(value, str) and value.startswith(DEVICE_REF_PREFIX)


@dataclass(frozen=True)
class ParamDecl:
    """Declarative description of ONE tunable/scannable parameter.

    Dependency-free: a measurement / processor / device declares each of its
    parameters once (``key``/``label``/``kind``/``default``/bounds/...) and BOTH the
    Python API default AND the GUI control derive from this single declaration (the
    single source of truth -- explicit declaration, not signature reflection or AST
    inspection).  ``kind`` selects how a caller/GUI interprets the value:

    ``"float"``       a scalar in ``[lo, hi]`` (``unit`` annotates it)
    ``"int"``         an integer in ``[lo, hi]``
    ``"axis_range"``  a swept range; the value is ``(min, max, points)`` (the GUI
                      renders three boxes).  ``default`` is that 3-tuple, in
                      ``unit``.
    ``"bool"``        a flag (checkbox)
    ``"choice"``      one of ``choices`` (a combo box)
    ``"text"``        a free string (a line edit) -- e.g. a label; taken verbatim,
                      NEVER ``eval``'d (lo/hi/choices ignored)
    ``"json"``        a JSON literal in a line edit (a list / mapping / number -- e.g.
                      a device config's channel list ``["ch00", "ch01"]``).  Parsed
                      with ``json.loads`` -- typed, never ``eval``'d; blank = None
                      (the consumer's own default applies).
    ``"device"``      a ``$device:<entry>`` cross-reference to ANOTHER entry of the
                      SAME device config (a constructor arg that takes a built device,
                      e.g. a virtual camera's ``sequencer`` wire).  Rendered as a
                      choice whose options the config editor fills from the working
                      config's other entry names -- so ``choices`` here may be empty.
    ``"path"``        a filesystem path: a line edit + a Browse button (native dialog).
                      ``path_mode='file'`` picks a file (filtered by ``file_filter``),
                      ``path_mode='dir'`` picks a folder.  Taken verbatim, never eval'd.
    ``"signal"``      the NAME of a hub signal to consume (a processor's input): a combo
                      box of the live hub signals, like a plot's input picker.
    ``"signal_expr"`` a MULTI-slot signal picker + a ``value = ...`` expression (the same
                      one a plot panel's source uses): pick one or more hub signals (read as
                      ``signal`` / ``signal[i]``) and combine them.  The value is a
                      ``{"inputs": [name, ...], "source": "value = ..."}`` dict -- so a
                      processor/measurement "source" can subscribe to several running nodes'
                      signals and combine them, never just one bare name.
    ``"pulse_param"`` a parameter of a pulse template to sweep: a combo box whose choices are
                      introspected from the template FILE named in the ``depends_on`` field
                      (its periods / channels / DAC buses), so picking the template repopulates
                      it.  The value is a ``"kind:target"`` token (e.g. ``"duration:2"``).

    ``required`` marks a parameter a GUI must highlight when missing.  ``depends_on`` (kind
    ``pulse_param``) names the sibling ``path`` field whose pulse template is introspected to
    populate this control.  ``display`` is a pure DATA placement flag (not an art/geometry knob)
    that splits a plot panel's params between its two surfaces:
      * ``display=True``  -- a pure DISPLAY knob (how the SAME data is drawn): bins, history length,
                             fit chooser, log axis, colormap, relim.  Rendered in the lightweight
                             Setting popup, alongside size / relim, where an operator reaches for them.
      * ``display=False`` -- an ACQUISITION / measurement-API param (what data is taken).  Rendered in
                             the panel's Edit tab instead, so Setting and Edit never duplicate.
    A measurement param ignores it (defaults True).  No value here is ever ``eval``'d -- the spec
    consumer validates / coerces by ``kind``.
    """

    key: str
    label: str
    kind: str
    default: Any = None
    unit: str = ""
    lo: float = 0.0
    hi: float = 1e12
    required: bool = False
    choices: tuple = ()
    tooltip: str = ""
    path_mode: str = "file"          # kind="path": "file" (open-file dialog) | "dir" (folder dialog)
    file_filter: str = "All files (*)"  # kind="path", path_mode="file": the open-file filter
    base_dir: str = ""               # kind="path": the folder a Browse dialog opens in when the
                                     # field doesn't resolve to an existing path (e.g. "pulses"
                                     # for a pulse template, "calibrations" for a data folder)
    depends_on: str = ""             # kind="pulse_param": the sibling kind="path" field whose
                                     # pulse template is introspected to populate this combo
    display: bool = True             # plot-panel placement flag (DATA, not art): True = a pure DISPLAY
                                     # knob (bins / history / fit / colormap …) in the Setting popup;
                                     # False = an acquisition / measurement-API param in the Edit tab.
                                     # Ignored by measurements.
    segmented: bool = False          # kind="choice" RENDER hint (DATA, not art): True = a capsule
                                     # tri/multi-state toggle (confocal TriStateToggleSwitch) instead of
                                     # a combo box; same value semantics (one of ``choices``).
    optional: Any = None             # kind="float"/"int" render intent (tri-state): True = a blank-able
                                     # line edit ("leave blank = use the library / device default");
                                     # False = ALWAYS a scrollable spin box (never blank); None = infer
                                     # (blank-able exactly when there is no default and it isn't required
                                     # -- an optional API arg).  A device runtime control pins False so a
                                     # live knob is a spin box by construction (see ``blank_allowed``).

    def __post_init__(self) -> None:
        kind = str(self.kind).lower()
        if kind not in ("float", "int", "axis_range", "bool", "choice", "text", "json", "device",
                        "path", "signal", "signal_expr", "pulse_param", "pulse_slots"):
            raise ValueError(
                "ParamDecl.kind must be one of "
                "float/int/axis_range/bool/choice/text/json/device/path/signal/signal_expr/"
                f"pulse_param/pulse_slots, got {self.kind!r}."
            )
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "key", str(self.key))
        object.__setattr__(self, "choices", tuple(self.choices))

    @property
    def is_numeric(self) -> bool:
        """A scalar/range the GUI renders as a spin box (float / int / axis_range) -- so a read-back
        display knows to engineering-scale its value (``float2str_eng``), unlike a text/choice/bool."""
        return self.kind in ("float", "int", "axis_range")

    @property
    def blank_allowed(self) -> bool:
        """kind ``float`` / ``int``: does this control offer a BLANK "use the default" state (a line
        edit) instead of a spin box?  The ONE predicate ``FloatHandler`` / ``IntHandler`` read, so the
        blank-vs-spin decision lives on the declaration (not hard-coded in a widget handler).

        Explicit ``optional`` wins (a device runtime control pins it ``False`` -> always a spin box);
        otherwise infer: an optional API arg (no default, not required) is blank-able, everything else
        is a spin box."""
        if self.optional is not None:
            return bool(self.optional)
        return self.default is None and not self.required

    def row_label(self) -> str:
        """The ONE label a form row / control title shows -- the single source EVERY form reads
        (the config editor, the device viewer, the measurement Edit, the Setting popup, the
        signal-expr title) instead of re-typing the ``"<label> (<unit>) *"`` idiom, so a new form
        can never drift or forget a piece.

        ``label`` (or ``key``) + ``(unit)`` when a unit is declared + a trailing ``*`` when
        ``required`` (the marker a form highlights when the field is missing)."""
        text = self.label or self.key
        if self.unit:
            text += f" ({self.unit})"
        if self.required:
            text += " *"
        return text


__all__ = ["DEVICE_REF_PREFIX", "ParamDecl", "is_device_ref"]
