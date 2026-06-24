# frontend — sealed public API contract (read before adding any export)

`Zou_lab_control.frontend` exposes a **small, sealed** public surface. An external
caller (a notebook, the lab) calls it and gets correct **art, geometry, dpi and
typography** — and must **not** be able to break the visual design. Geometry,
margins, dpi, colours, shadow and scale are **owned by the frontend**, never
configurable per call.

The authoritative statement of this contract is the module docstring of
`frontend/__init__.py`. This file adds the rationale and the failure history so
the rules don't get re-broken.

## The rules

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
7. **Every plot IS a `BaseLivePlot` — the reusable layer is not optional.** The
   reason this package exists is that every plot reuses ONE layer: the selectors
   (`selectors.py`: zoom/pan, area, cross, draggable lines) and the `DataFigure`
   fitting/post-processing stack. A plot type gets that layer **only** by
   subclassing `live.BaseLivePlot` and rendering through its `show()` lifecycle
   (`_create_axes` → `init_core` → `_attach_interactions` → `to_data_figure`).
   **Never hand-roll a raw-matplotlib figure for a plot** — it silently loses all
   of it. Multi-axes plots override `_create_axes` to build their layout, set
   `self.axes`, and attach per-cell tools; `DataFigure(ax=...)` binds the stack
   to one cell so even a grid cell fits like a standalone plot. Design tokens
   (sizes/colours/title) come from `style.py` (PALETTE, `apply_title`,
   `*_fontsize`), never re-picked per plot. **Enforced** by
   `tests/test_frontend_plot_contract.py`: a plot-shaped class that bypasses
   `BaseLivePlot`, or an entry point whose figure has no selectors / no
   `DataFigure`, fails the build.

8. **One row/label system for EVERY settings / Edit / param form.** A form is built from exactly
   two primitives: `FluentSectionLabel` (bold, dark, own-line GROUP header) and `FluentSettingRow`
   (grey fixed-width label | control). The section-vs-row hierarchy is shown by **weight + colour
   only — never indentation, never a third colour**. The label-column width comes from the single
   `setting_label_width(labels)` rule (fit the widest label, shared minimum) so a form's rows align.
   **Do NOT** use a `QFormLayout` for a param form (its labels render dark + auto-width = a rogue
   style), **do NOT** use a bare bold `FluentLabel` as a section (use `FluentSectionLabel`), and
   **do NOT** indent a composite's sub-rows. Bold lives in the stylesheet (`font-weight: bold`),
   not a QFont-only `setBold` that `setStyleSheet` re-polish would drop. **Enforced** by
   `tests/test_frontend_layout_uniformity.py`. (Failure history: three competing label logics —
   QFormLayout dark labels, grey-bold sections, indented sub-forms — plus section labels that were
   never actually bold, so a section looked like a row.)

## The patterns that already seal correctly (copy these)

- `panel_plot_spec` / `panel_size_cells` own panel geometry; size is the only knob.
- `qt_canvas.panel_canvas` owns display scale (`PANEL_DISPLAY_SCALE`);
  `EmbeddedFigureCanvas` keeps the three invariants (fixed inches, retina-only
  dpi, logical = design_px × display_scale).
- `qt_fluent.resolve_fluent_auto_scale` is the single GUI-scale owner.
- `notes.render_tex_pdf` owns the temp-dir/2-pass/aux-skip compile; callers only
  hand it a tex string (or path) and an output pdf path — nothing is left behind.
- `FluentComboBox` drop-down matches the `FluentPopup` Setting card: a translucent,
  frameless container whose **event filter** (`_RoundedPopupCard`) paints one
  antialiased rounded rect (white fill + 1 px `DIVIDER` border) and returns True to
  suppress the default opaque square; the `QListView` viewport is genuinely
  transparent so the card shows; the selected/hover row is a delegate-drawn
  `::item:selected`/`::item:hover` pill (NOT `selection-background-color`, which a
  translucent viewport drops → white-on-white); the same filter, on the Move event,
  re-applies a few-px OUTER gap so the popup sits off the box (a `showPopup`-time
  `move()` is overwritten by Qt's deferred flush-positioning); the scrollbar is the
  shared `fluent_scrollbar_stylesheet`. Verify with real-window `widget.grab()` of
  the container (offscreen does not render the top-level popup).

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
- **The multi-site histogram grid shipped with no selectors and no `data_figure`**
  (2026-06): it was written as a bare matplotlib figure (`class SiteHistogramGrid:`)
  instead of a `BaseLivePlot`, so the entire reusable layer — the whole point of
  the frontend — was silently lost. The principle existed only as prose, so
  nothing failed. → rule 7, now mechanically enforced by
  `tests/test_frontend_plot_contract.py`. **Lesson: a design principle that CAN be
  a test MUST be one — prose alone gets re-broken.**

## Layout: no overlap, no cutoff, aligned (composite / multi-panel figures)

A figure is only correct if every element is fully visible and orderly. This is
a core art principle, not a nicety:

- **No overlap.** Panels/cells never overlap each other; annotations (per-cell
  labels, threshold lines, stats text) never sit on top of the data bars/points
  — give the axes headroom so labels float in clear space.
- **No cutoff.** Nothing is clipped by the figure edge: titles, tick labels,
  outer axis labels and the last row/column must all fit inside the canvas.
- **Aligned.** Cells in a grid share one fixed grid and (where comparable) one
  shared data range, so they line up exactly.

Build multi-panel figures on `canvas.create_axes_grid` (fixed-pixel cells +
explicit gaps + a figure sized to fit them + margins) so these three hold **by
construction**; `grid_shape_for` picks a general `(rows, cols)` for any N (do not
hard-code a site count). `site_histogram_grid` is the reference implementation.

## Verifying visual changes

Any UI/plot change must pass `devtools.capture_user_view` at QT_SCALE_FACTOR
1.0 / 1.25 / 1.5 (three-scale screenshots, inspected as 1:1 crops). The
`parity` target opens both GUIs in one process on one screen and fails if their
fluent control sizes disagree. A DPR=1 offscreen pass alone is NOT acceptance.

For a static notebook figure (matplotlib, fixed-inch geometry, dpi-invariant
layout) the equivalent acceptance is: **render the actual output and inspect it**
for the no-overlap / no-cutoff / aligned rules above — at a representative size
AND a stress count (e.g. an N-site grid), not just one hand-picked case. Reading
the rendered PNG is the check; an object-level "it ran" is not.
