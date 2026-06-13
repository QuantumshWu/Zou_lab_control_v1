# frontend — sealed public API contract (read before adding any export)

`Zou_lab_control.frontend` exposes a **small, sealed** public surface. An external
caller (a notebook, the lab) calls it and gets correct **art, geometry, dpi and
typography** — and must **not** be able to break the visual design. Geometry,
margins, dpi, colours, shadow and scale are **owned by the frontend**, never
configurable per call.

The authoritative statement of this contract is the module docstring of
`frontend/__init__.py`. This file adds the rationale and the failure history so
the rules don't get re-broken.

## The six rules

1. **Geometry is owned, never passed in.** No public callable accepts `data_px`,
   `margins_px`, `spec` or `dpi`. Sizes come only from `FigureSpec` defaults,
   `panel_plot_spec`, `pulse_plot_spec`, `create_axes_fixed`. `plot()` /
   `panel_plot()` reject those keys (`live._SEALED_PLOT_KWARGS`); internal
   geometry is injected through the private `plot(_spec=...)` argument.
2. **Sizes / colormaps are curated and validated on entry.** Panel size is one
   of `PANEL_SIZES` (`panel_size_cells` raises otherwise). An invalid cmap
   raises rather than silently falling back. (Pattern to copy: `panel_size_cells`.)
3. **One typography system, one dpi.** `style.DEFAULT_STYLE` (300 dpi, fixed font
   sizes) is the only typography source and is exported **read-only**
   (`MappingProxyType` over the private `_DEFAULT_STYLE`). No public font-size,
   colour or dpi override.
4. **One display-scale rule.** On-screen scale comes only from
   `qt_fluent.resolve_fluent_auto_scale` (GUI controls) and
   `qt_canvas.panel_canvas` / `PANEL_DISPLAY_SCALE` (embedded figures). Public
   `show_pulse_gui` / `show_task_console` default to `scale=None` (auto).
5. **Art-bearing fluent widgets stay internal.** `qt_fluent.*` and `qt_canvas.*`
   (including `FluentGroupBox(shadow=)`, `FluentFrame(shadow=)`,
   `add_fluent_shadow`, `EmbeddedFigureCanvas`) are **not** re-exported from
   `frontend/__init__.py`. Shadows / borders / corner radii are construction
   details, never public knobs. Do not promote them to the package surface.
6. **Adding a public parameter requires classifying it.** Before adding a
   parameter to a `__all__` callable, classify it: **DATA** (labels, title,
   bins, thresholds, relim_mode, cmap-from-a-list, channels, roi_radius) is
   allowed on the public surface; **ART / GEOMETRY / TYPOGRAPHY** (margins, dpi,
   colours, sizes, shadow, fonts, bad_color) lives on **internal classes only**.

## The patterns that already seal correctly (copy these)

- `panel_plot_spec` / `panel_size_cells` own panel geometry; size is the only knob.
- `qt_canvas.panel_canvas` owns display scale (`PANEL_DISPLAY_SCALE`);
  `EmbeddedFigureCanvas` keeps the three invariants (fixed inches, retina-only
  dpi, logical = design_px × display_scale).
- `qt_fluent.resolve_fluent_auto_scale` is the single GUI-scale owner.
- `notes.render_tex_pdf` owns the temp-dir/2-pass/aux-skip compile; callers only
  hand it a tex string (or path) and an output pdf path — nothing is left behind.

## Why these rules exist (failure history)

- **Two GUIs disagreed on control size** (commit 57a1d25): `set_fluent_scale(None)`
  silently fell back to 1.0 while the pulse editor fit the screen → mismatched
  lineedits on small screens. → rule 4 (one scale rule).
- **High-DPI figure warp** (commit 82f941a): stock Qt synced figure size from the
  widget, breaking fixed-inch axes. → `EmbeddedFigureCanvas` invariants, rule 4.
- **Panel cards lost their shadow** when a `shadow=False` toggle was reachable
  during construction → the visual-design regression that motivated rule 5/6.
- **PDF builds littered docs/** with `.tex/.sty/.aux/.log/.toc` because the build
  wrote intermediates in place → `render_notes_pdf` now assembles the tex in
  memory and compiles via `render_tex_pdf` in a temp dir (only the `.pdf` lands
  in `docs/`); `docs/**/*.tex` and `*.sty` are gitignored.

## Verifying visual changes

Any UI/plot change must pass `devtools.capture_user_view` at QT_SCALE_FACTOR
1.0 / 1.25 / 1.5 (three-scale screenshots, inspected as 1:1 crops). The
`parity` target opens both GUIs in one process on one screen and fails if their
fluent control sizes disagree. A DPR=1 offscreen pass alone is NOT acceptance.
