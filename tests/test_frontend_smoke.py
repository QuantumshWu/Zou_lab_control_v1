import matplotlib

matplotlib.use("Agg")

import re
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import Zou_lab_control.frontend as zf
import Zou_lab_control.neutral_atom as na


def assert_raises_contains(text, func, *args, **kwargs):
    try:
        func(*args, **kwargs)
    except Exception as exc:
        assert text in str(exc)
        return exc
    raise AssertionError(f"expected exception containing {text!r}")


def wait_until_done(session, *, timeout=5.0):
    deadline = time.perf_counter() + timeout
    while not session.done and time.perf_counter() < deadline:
        time.sleep(0.01)
    session.refresh(draw=False)
    if session.error is not None:
        raise session.error
    assert session.done
    return session


def test_frontend_static_plots_and_data_figure():
    x = np.linspace(737.0, 737.2, 121)
    y = 20 * ((0.02 / 2) ** 2) / ((x - 737.1) ** 2 + (0.02 / 2) ** 2) + 3

    plot = zf.plot(x, y, labels=("Wavelength (nm)", "Counts", "Counts"), display=False)
    result, popt = plot.data_figure.lorent(is_display=False)

    assert plot.fig.axes
    assert result.function == "lorent"
    assert popt is not None
    assert abs(popt[0] - 737.1) < 0.01


def test_frontend_2d_and_histogram():
    x = np.linspace(-2, 2, 25)
    y = np.linspace(-1, 1, 15)
    xx, yy = np.meshgrid(x, y)
    z = np.exp(-(xx**2 + yy**2))
    data_x = np.column_stack([xx.ravel(), yy.ravel()])
    data_y = z.ravel()

    plot2 = zf.plot(data_x, data_y, labels=("X", "Y", "Counts"), display=False)
    hist = zf.plot(np.r_[np.zeros(20), np.ones(30) * 10], kind="hist", thresholds=[5], display=False)

    assert plot2.grid.shape == (len(y), len(x))
    assert len(plot2.fig.axes) == 3
    assert hist.fractions() == {0: 0.4, 1: 0.6}
    assert hist.threshold_draggers
    assert hist.fit_threshold is not None
    assert "th=" in hist.stats_text.get_text()
    assert "fit F=" in hist.stats_text.get_text()
    assert "fit cut=" in hist.stats_text.get_text()


def test_frontend_title_pulse_and_public_2d_square_guard():
    line = zf.plot(np.arange(5), np.arange(5), title="Small title", display=False)
    assert line.ax.get_title() == "Small title"
    assert line.ax.title.get_fontsize() <= line.ax.xaxis.label.get_fontsize() + 0.1

    class Seq:
        name = "ten channel sequence"
        channels = [f"ch{i}" for i in range(10)]

        def effective_pulses(self):
            return [
                SimpleNamespace(channel=channel, start=i * 2e-6, duration=1.2e-6, value=1, name=f"p{i}")
                for i, channel in enumerate(self.channels)
            ]

    pulse = zf.plot(Seq(), kind="pulse", display=False)
    assert pulse.plot_type == "pulse"
    assert len(pulse.ax.get_yticklabels()) == 10
    assert pulse.ax.get_yticklabels()[0].get_color() != "black"
    assert len(pulse.off_lines) == 10
    assert len(pulse.pulse_artists) == 10
    first_line = pulse.off_lines[0].get_segments()[0]
    last_line = pulse.off_lines[-1].get_segments()[0]
    first_patch = pulse.pulse_artists[0]
    assert first_line[0, 1] > last_line[0, 1]
    assert abs(first_line[0, 1] - first_patch.get_y()) < 1e-12
    assert np.allclose(pulse.off_lines[0].get_colors()[0], first_patch.get_facecolor())
    assert pulse.off_lines[0].get_zorder() == first_patch.get_zorder()
    assert max(line.get_zorder() for line in pulse.ax.get_xgridlines()) < first_patch.get_zorder()
    pulse.fig.canvas.draw()
    x_tick_labels = [tick.get_text().strip() for tick in pulse.ax.get_xticklabels() if tick.get_visible()]
    assert any(label for label in x_tick_labels)
    assert pulse.ax.get_xlabel() == "Time (us)"
    assert all("e" not in label.lower() for label in x_tick_labels if label)

    ns_pulse = zf.plot([{"channel": "gate", "start": 0.0, "duration": 30e-9, "value": 1}], kind="pulse", display=False)
    ns_pulse.fig.canvas.draw()
    assert ns_pulse.ax.get_xlabel() == "Time (ns)"

    class FortySeq:
        name = "forty channel sequence"
        channels = [f"ch{i:02d}" for i in range(40)]

        def effective_pulses(self):
            return [
                SimpleNamespace(channel="ch00", start=0.0, duration=1e-6, value=1, name="load"),
                SimpleNamespace(channel="ch17", start=2e-6, duration=1e-6, value=1, name="probe"),
            ]

    filtered = zf.plot(FortySeq(), kind="pulse", display=False)
    labeled = zf.plot(FortySeq(), kind="pulse", channel_labels={"ch00": "trap", "ch17": "probe"}, display=False)
    all_rows = zf.plot(FortySeq(), kind="pulse", include_always_off=True, display=False)
    assert [tick.get_text() for tick in filtered.ax.get_yticklabels()] == ["ch00", "ch17"]
    assert [tick.get_text() for tick in labeled.ax.get_yticklabels()] == ["trap", "probe"]
    assert len(all_rows.ax.get_yticklabels()) == 40
    assert all_rows.spec.data_px[1] >= 4 * 360
    assert zf.pulse_repeat_notation() == "repeat ∞"
    assert zf.pulse_repeat_notation(0, 2, 4) == "repeat P1-P3 x4"
    assert zf.pulse_repeat_marker(total_duration_s=3e-6) == (0.0, 3e-6, "∞")
    repeat_state = na.PulseTableState(
        channels=["ch00", "ch01"],
        periods=[
            na.PulsePeriod(100, (1, 0), unit="ns"),
            na.PulsePeriod(200, (0, 1), unit="ns"),
            na.PulsePeriod(300, (0, 0), unit="ns"),
        ],
        time_step_ns=10,
        repeat_start=1,
        repeat_end=1,
        repeat_count=5,
    )
    assert zf.pulse_repeat_notation(repeat_state) == "repeat ∞ + P2-P2 x5"
    repeat_markers = zf.pulse_repeat_markers(repeat_state)
    assert [marker[2] for marker in repeat_markers] == ["∞", "x5"]
    assert np.allclose([(marker[0], marker[1]) for marker in repeat_markers], [(0.0, 600e-9), (100e-9, 300e-9)])
    outer_repeat_state = na.PulseTableState.from_dict({**repeat_state.to_dict(), "repeat_start": 0, "repeat_end": 2, "repeat_count": 3})
    assert zf.pulse_repeat_notation(outer_repeat_state) == "repeat P1-P3 x3"
    outer_markers = zf.pulse_repeat_markers(outer_repeat_state)
    assert [marker[2] for marker in outer_markers] == ["x3"]
    assert np.allclose([(marker[0], marker[1]) for marker in outer_markers], [(0.0, 600e-9)])
    bracketed = zf.plot(
        FortySeq(),
        kind="pulse",
        repeat_notation="repeat ∞",
        repeat_bracket=(0.0, 3e-6, "∞"),
        display=False,
    )
    bracketed.fig.canvas.draw()
    assert len(bracketed.repeat_bracket_artists) == 2
    assert all(abs(bracket.get_linewidth() - bracketed.repeat_bracket_artists[0].get_linewidth()) < 1e-12 for bracket in bracketed.repeat_bracket_artists)
    assert all(abs(bracket.get_alpha() - 0.58) < 1e-12 for bracket in bracketed.repeat_bracket_artists)
    for bracket in bracketed.repeat_bracket_artists:
        bracket_y = bracket.get_ydata()
        assert min(bracket_y) <= -0.3
        assert max(bracket_y) >= len(bracketed.channels) - 0.2
    assert bracketed.repeat_bracket_label.get_text() == "∞"
    assert bracketed.repeat_bracket_label.get_position()[0] > 3e-6
    assert bracketed.repeat_bracket_label.get_ha() == "left"
    assert bracketed.repeat_bracket_label.get_clip_on() is False
    assert abs(bracketed.repeat_bracket_label.get_alpha() - 0.58) < 1e-12
    xlim = bracketed.ax.get_xlim()
    assert xlim[0] < 0
    assert min(bracketed.repeat_bracket_artists[0].get_xdata()) == 0.0
    assert min(bracketed.repeat_bracket_artists[0].get_xdata()) > xlim[0]
    assert max(bracketed.repeat_bracket_artists[1].get_xdata()) < xlim[1]
    assert bracketed.repeat_bracket_label.get_position()[0] < xlim[1]
    x_tick_labels = [tick.get_text().strip() for tick in bracketed.ax.get_xticklabels() if tick.get_visible()]
    assert all(not label.startswith("-") for label in x_tick_labels if label)

    bracketed_far = zf.plot(
        FortySeq(),
        kind="pulse",
        repeat_notation="repeat ∞",
        repeat_bracket=(0.0, 5e-6, "∞"),
        display=False,
    )
    bracketed_far.fig.canvas.draw()
    far_xlim = bracketed_far.ax.get_xlim()
    assert far_xlim[0] < 0
    assert min(bracketed_far.repeat_bracket_artists[0].get_xdata()) == 0.0
    assert min(bracketed_far.repeat_bracket_artists[0].get_xdata()) > far_xlim[0]
    assert max(bracketed_far.repeat_bracket_artists[1].get_xdata()) < far_xlim[1]
    assert bracketed_far.repeat_bracket_label.get_position()[0] < far_xlim[1]
    far_tick_labels = [tick.get_text().strip() for tick in bracketed_far.ax.get_xticklabels() if tick.get_visible()]
    assert all(not label.startswith("-") for label in far_tick_labels if label)

    middle_repeat = zf.plot(
        [
            {"channel": "ch00", "start": 0.0, "duration": 0.8e-6, "value": 1},
            {"channel": "ch00", "start": 2.2e-6, "duration": 0.6e-6, "value": 1},
            {"channel": "ch01", "start": 3.6e-6, "duration": 0.7e-6, "value": 1},
        ],
        kind="pulse",
        channels=["ch00", "ch01"],
        repeat_notation="repeat ∞ + P2-P3 x4",
        repeat_brackets=[(0.0, 4.3e-6, "∞"), (1.0e-6, 3.2e-6, "x4")],
        display=False,
    )
    middle_repeat.fig.canvas.draw()
    assert len(middle_repeat.repeat_bracket_artists) == 4
    assert [label.get_text() for label in middle_repeat.repeat_bracket_labels] == ["∞", "x4"]
    assert min(middle_repeat.repeat_bracket_artists[0].get_xdata()) == 0.0
    assert max(middle_repeat.repeat_bracket_artists[1].get_xdata()) == 4.3e-6
    assert min(middle_repeat.repeat_bracket_artists[2].get_xdata()) == 1.0e-6
    assert max(middle_repeat.repeat_bracket_artists[3].get_xdata()) == 3.2e-6
    outer_y = np.r_[middle_repeat.repeat_bracket_artists[0].get_ydata(), middle_repeat.repeat_bracket_artists[1].get_ydata()]
    inner_y = np.r_[middle_repeat.repeat_bracket_artists[2].get_ydata(), middle_repeat.repeat_bracket_artists[3].get_ydata()]
    assert outer_y.min() < inner_y.min()
    assert outer_y.max() > inner_y.max()
    assert middle_repeat.ax.get_xlim()[0] < 0.0
    assert middle_repeat.ax.get_xlim()[1] > 4.3e-6
    assert middle_repeat.ax.get_xlim()[0] < 1.0e-6
    assert middle_repeat.ax.get_xlim()[1] > 3.2e-6
    assert middle_repeat.repeat_bracket_label.get_position()[0] < middle_repeat.ax.get_xlim()[1]
    middle_tick_labels = [tick.get_text().strip() for tick in middle_repeat.ax.get_xticklabels() if tick.get_visible()]
    assert all(not label.startswith("-") for label in middle_tick_labels if label)

    data_x = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    assert_raises_contains("always square", zf.plot, data_x, np.arange(4), square=False, display=False)


def test_live_2d_distribution_uses_reasonable_bins():
    x = np.linspace(-2, 2, 9)
    y = np.linspace(-1, 1, 7)
    xx, yy = np.meshgrid(x, y)
    data_x = np.column_stack([xx.ravel(), yy.ravel()])
    data_y = np.full((len(data_x), 1), np.nan)

    plot = zf.plot(data_x, data_y, labels=("X", "Y", "Counts"), update="watch", display=False)
    assert plot.n_bins > 3

    data_y[:, 0] = np.arange(len(data_y))
    plot._watcher.refresh(draw=False)
    plot.stop()

    assert plot.points_done == len(data_y)
    assert np.count_nonzero(np.isfinite(plot.grid)) == len(data_y)


def test_live_update_point_and_roll():
    plot = zf.Live1D(np.arange(5), np.full(5, np.nan), labels=("Index", "Signal", "Signal")).show(display=False)
    plot.update_point(2, [4.0], mode="replace", draw=False)
    plot.roll([9.0], draw=False)

    assert plot.data_y[3, 0] == 4.0
    assert plot.data_y[0, 0] == 9.0


def test_crosshair_second_right_click_clears():
    plot = zf.plot(np.arange(5), np.arange(5), display=False)
    event = SimpleNamespace(inaxes=plot.ax, button=3, xdata=2.0, ydata=3.0, dblclick=False)

    plot.cross.on_press(event)
    assert plot.cross.xy == [2.0, 3.0]

    plot.cross.last_click_time = time.time()
    plot.cross.on_press(event)
    assert plot.cross.xy is None


def test_threshold_drag_temporarily_disables_area_selector():
    hist = zf.plot(np.r_[np.zeros(20), np.ones(30) * 10], kind="hist", thresholds=[5], display=False)
    dragger = hist.threshold_draggers[0]
    area_selector = hist.tools.area.selector
    event = SimpleNamespace(inaxes=hist.ax, button=1, xdata=5.0, ydata=1.0, dblclick=False)

    assert area_selector.active
    dragger.on_press(event)
    assert dragger.dragging
    assert not area_selector.active

    dragger.on_release(event)
    assert area_selector.active


def test_zoom_destroy_disconnects_without_drag_state_reference():
    plot = zf.plot(np.arange(5), np.arange(5), display=False)

    plot.zoom.destroy()


def test_unified_watch_refresh_infers_progress():
    x = np.arange(5).reshape(-1, 1)
    y = np.full((5, 1), np.nan)
    plot = zf.plot(x, y, labels=("Index", "Signal", "Signal"), update="watch", display=False)

    y[:3, 0] = [1, 2, 3]
    plot._watcher.refresh(draw=False)
    plot.stop()

    assert plot.points_done == 3
    assert plot.data_y[2, 0] == 3


def test_run_session_scan_and_histogram():
    x = np.arange(5).reshape(-1, 1)
    session = zf.run(x, lambda point: point * 2, labels=("Index", "Signal", "Signal"), display=False)
    wait_until_done(session)

    assert session.points_done == 5
    assert np.allclose(session.data_y[:, 0], np.arange(5) * 2)
    assert session.data_figure is not None

    hist = zf.run(20, lambda: 3.0, kind="hist", thresholds=[2], display=False)
    wait_until_done(hist)

    assert hist.data_y.shape == (20, 1)
    assert hist.fractions() == {1: 1.0}


def test_run_session_prints_stop_hint(capsys):
    x = np.arange(2).reshape(-1, 1)
    session = zf.run(x, lambda point: point, display=True, stop_hint="Call obj.stop()")
    session.stop()

    assert "Call obj.stop()" in capsys.readouterr().out


def test_run_session_rejects_ambiguous_lifecycle_parameters():
    x = np.arange(5).reshape(-1, 1)

    assert_raises_contains(
        "update_time must be finite, not a boolean",
        zf.run,
        x,
        lambda point: point,
        display=False,
        update_time=True,
    )
    assert_raises_contains(
        "update_time must be finite and > 0",
        zf.run,
        x,
        lambda point: point,
        display=False,
        update_time=0,
    )
    assert_raises_contains("display must be a boolean", zf.run, x, lambda point: point, display="False")
    assert_raises_contains(
        "autostart must be a boolean",
        zf.run,
        x,
        lambda point: point,
        display=False,
        autostart="False",
    )
    assert_raises_contains(
        "copy_on_refresh must be a boolean",
        zf.run,
        x,
        lambda point: point,
        display=False,
        copy_on_refresh="False",
    )
    assert_raises_contains(
        "stop_when_full must be a boolean",
        zf.run,
        x,
        lambda point: point,
        display=False,
        stop_when_full="False",
    )
    assert_raises_contains(
        "max_points must be a positive integer, not a boolean",
        zf.run,
        x,
        lambda point: point,
        display=False,
        max_points=True,
    )
    assert_raises_contains(
        "max_points must be a positive integer",
        zf.run,
        x,
        lambda point: point,
        display=False,
        max_points=1.5,
    )
    assert_raises_contains("histogram count must be a positive integer, not a boolean", zf.run, True, lambda: 1, kind="hist", display=False)
    assert_raises_contains("histogram count must be a positive integer", zf.run, 0, lambda: 1, kind="hist", display=False)

    hist = zf.run(np.int64(4), lambda: 1, kind="hist", display=False)
    wait_until_done(hist)
    assert hist.data_y.shape == (4, 1)


def test_array_watcher_rejects_ambiguous_refresh_parameters():
    x = np.arange(5).reshape(-1, 1)
    y = np.full((5, 1), np.nan)

    assert_raises_contains(
        "watch_interval must be finite, not a boolean",
        zf.plot,
        x,
        y,
        update="watch",
        watch_interval=True,
        display=False,
    )
    assert_raises_contains(
        "update_time must be finite, not a boolean",
        zf.plot,
        x,
        y,
        update="watch",
        update_time=True,
        display=False,
    )
    assert_raises_contains(
        "update_time must be finite and > 0",
        zf.plot,
        x,
        y,
        update="watch",
        update_time=0,
        display=False,
    )
    assert_raises_contains(
        "watch_interval must be finite and > 0",
        zf.plot,
        x,
        y,
        update="watch",
        watch_interval=0,
        display=False,
    )
    assert_raises_contains(
        "stop_when_full must be a boolean",
        zf.plot,
        x,
        y,
        update="watch",
        stop_when_full="False",
        display=False,
    )
    assert_raises_contains(
        "copy must be a boolean",
        zf.plot,
        x,
        y,
        update="watch",
        copy="False",
        display=False,
    )
    assert_raises_contains(
        "done must be a boolean",
        zf.plot,
        x,
        y,
        update="watch",
        done="False",
        display=False,
    )

    plot = zf.plot(x, y, update="watch", points_done=lambda: True, display=False)
    try:
        assert_raises_contains("points_done must be a non-negative integer, not a boolean", plot._watcher.refresh, draw=False)
    finally:
        plot.stop()

    plot = zf.plot(x, y, update="watch", done=lambda: "False", display=False)
    try:
        assert_raises_contains("done callback return must be a boolean", plot._watcher.refresh, draw=False)
    finally:
        plot.stop()


def test_run_session_returns_without_blocking():
    x = np.arange(10).reshape(-1, 1)

    def slow_measure(point):
        time.sleep(0.02)
        return point

    start = time.perf_counter()
    session = zf.run(x, slow_measure, display=False)
    elapsed = time.perf_counter() - start

    assert elapsed < 0.1
    assert session.running
    assert not hasattr(session, "wait")

    time.sleep(0.07)
    assert session.points_done > 0
    session.stop()


def test_run_session_stop_stops_measurement_and_plot():
    x = np.arange(20).reshape(-1, 1)

    def slow_measure(point):
        time.sleep(0.02)
        return point

    session = zf.run(x, slow_measure, display=False, update_time=0.01)
    session.stop()

    assert not session.running
    assert session.plot._stopped


def test_notes_tex_writer_copies_template(tmp_path):
    tex_path = zf.write_notes_tex(
        tmp_path,
        title="中文_notes_test",
        subtitle="XeLaTeX",
        description="代码和命令行样式",
        body="\\chapter{测试}\\begin{codeblock}[Python]\nprint('ok')\n\\end{codeblock}",
    )

    assert tex_path.exists()
    assert (tmp_path / "zlc_frontend_notes.sty").exists()
    text = tex_path.read_text(encoding="utf-8")
    assert "中文\\_notes\\_test" in text
    assert "\\begin{codeblock}" in text


def test_frontend_generates_utf8_tutorial_notebooks(tmp_path):
    frontend = zf.write_frontend_tutorial(tmp_path / "frontend.ipynb")
    neutral = zf.write_neutral_atom_tutorial(tmp_path / "neutral.ipynb")
    hardware = zf.write_neutral_atom_hardware_tutorial(tmp_path / "hardware.ipynb")
    fpga_server = zf.write_neutral_atom_fpga_server_tutorial(tmp_path / "fpga_server.ipynb")

    for result in (frontend, neutral, hardware, fpga_server):
        text = result.path.read_text(encoding="utf-8")
        assert "???" not in text
        assert "\ufffd" not in text
    assert "这个 notebook" in neutral.path.read_text(encoding="utf-8")
    hardware_text = hardware.path.read_text(encoding="utf-8")
    assert "open_devices=True" in hardware_text
    assert "zf.require_attrs" not in hardware_text
    assert "isinstance(" not in hardware_text
    assert "BOOTSTRAP_CELL" in zf.__all__
    assert "write_neutral_atom_tutorial" in zf.__all__


def test_pulse_gui_constructs_40_channel_editor(monkeypatch, tmp_path):
    pytest.importorskip("PyQt5")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")

    from PyQt5 import QtCore, QtTest, QtWidgets

    from Zou_lab_control.frontend.pulse_gui import FigureCanvas, PulseSequenceEditor, ensure_qt_app

    app = ensure_qt_app()
    channels = [f"ch{i:02d}" for i in range(40)]

    class DummySequencer:
        clock_hz = 100e6
        trigger_channels = ["ch03"]

        def __init__(self, channels):
            self.channels = channels

    editor = PulseSequenceEditor(sequencer=DummySequencer(channels))
    try:
        editor.resize(1280, 720)
        editor.show()
        app.processEvents()

        assert len(editor.state.channels) == 40
        assert re.fullmatch(r"pulse_\d{8}_\d{6}", editor.state.name)
        assert editor.state.time_step_ns == 10
        assert editor.state.visible_channels == ["ch00", "ch01", "ch02", "ch03"]
        assert editor.names_panel.label_edits["ch00"].text() == ""
        assert "4/40 visible" in editor.summary.text()
        assert "step 10 ns" in editor.summary.text()
        assert editor.add_channel_combo.count() == 36
        first_card = editor.drag_container.pulse_cards()[0]
        assert first_card.checks["ch00"].text() == "ch00"
        assert editor.channel_panel.channel_labels["ch00"].text() == "ch00"
        assert hasattr(first_card, "unit_combo")
        assert not hasattr(first_card, "name_edit")
        assert first_card.duration_edit.geometry().right() <= first_card.width()
        assert first_card.unit_combo.geometry().right() <= first_card.width()
        assert editor.tabs.graphicsEffect() is not None
        assert editor.names_panel.graphicsEffect() is not None
        assert editor.channel_panel.graphicsEffect() is not None
        assert first_card.graphicsEffect() is not None
        assert editor.button_frame.graphicsEffect() is not None
        assert editor.button_frame.metaObject().className() == "FluentFrame"
        assert editor.channel_view.graphicsEffect() is not None
        assert editor.channel_view.parent() is editor.button_frame
        assert editor.names_panel_layout.contentsMargins().top() == editor.drag_container.layout_main.contentsMargins().top()
        assert editor.names_panel_layout.contentsMargins().top() > 0
        margins = editor.edit_tab.layout().contentsMargins()
        assert margins.left() > 0 and margins.top() > 0
        controls = [
            editor.add_channel_combo,
            editor.add_channel_button,
            editor.hide_off_button,
            editor.show_all_button,
        ]
        assert {control.width() for control in controls} == {editor.add_channel_combo.width()}
        assert {control.height() for control in controls} == {editor.add_channel_combo.height()}
        assert editor.add_channel_combo.geometry().x() == editor.hide_off_button.geometry().x()
        assert editor.add_channel_button.geometry().x() == editor.show_all_button.geometry().x()
        assert editor.preview_include_off.metaObject().className() == "FluentSwitch"
        assert not hasattr(editor, "preview_refresh_button")
        assert hasattr(editor, "preview_save_figure_button")
        assert not hasattr(editor, "save_figure_button")
        assert editor.preview_save_figure_button.text() == "Save Figure"
        assert "\n" not in editor.preview_save_figure_button.text()
        editor.hide_left_panels()
        app.processEvents()
        assert editor.names_panel_holder.isHidden()
        assert editor.channel_panel_holder.isHidden()
        assert editor.left_panel_stub.isVisible()
        editor.show_left_panels()
        app.processEvents()
        assert not editor.names_panel_holder.isHidden()
        assert not editor.channel_panel_holder.isHidden()
        assert editor.left_panel_stub.isHidden()
        assert editor.tabs.currentWidget() is editor.edit_tab
        editor.tabs.setCurrentWidget(editor.preview_tab)
        app.processEvents()
        assert editor.tabs.currentWidget() is editor.preview_tab
        if FigureCanvas is not None:
            assert editor._preview_canvas is not None
            assert "plotted" in editor.preview_status.text()
            assert editor.preview_save_figure_button.geometry().left() > editor.preview_status.geometry().right()
            editor.preview_include_off.setChecked(True)
            QtTest.QTest.qWait(250)
            app.processEvents()
            assert "40/40 plotted" in editor.preview_status.text()
            editor.preview_include_off.setChecked(False)
            QtTest.QTest.qWait(250)
            app.processEvents()
        editor.tabs.setCurrentWidget(editor.edit_tab)
        app.processEvents()

        editor.show_all_channels()
        app.processEvents()
        assert len(editor.state.visible_channels) == 40
        assert "40/40 visible" in editor.summary.text()
        assert editor.add_channel_combo.count() == 0
        assert not editor.add_channel_button.isEnabled()
        assert editor.dataset_scroll.verticalScrollBar().width() == editor.timeline_hbar.height()

        editor.hide_off_channels()
        app.processEvents()
        assert editor.state.visible_channels == ["ch00", "ch01", "ch02", "ch03"]

        editor.names_panel.label_edits["ch01"].setText("cooling_laser")
        editor.channel_panel.delay_edits["ch01"].setText("30")
        app.processEvents()
        assert editor.channel_panel.channel_labels["ch01"].text() == "cooling_laser"
        current_first_card = editor.drag_container.pulse_cards()[0]
        assert current_first_card.checks["ch01"].text() == "cooling_laser"
        editor.clear_channel("ch01")
        app.processEvents()
        assert editor.state.visible_channels == ["ch00", "ch01", "ch02", "ch03"]
        assert all(not period.states[editor.state.channel_index("ch01")] for period in editor.state.periods)
        assert editor.state.channel_labels["ch01"] == "cooling_laser"
        assert editor.state.delays["ch01"] == "30"

        editor.hide_off_channels()
        app.processEvents()
        assert editor.state.visible_channels == ["ch00", "ch01", "ch02", "ch03"]
        assert editor.add_channel_combo.findText("ch01", QtCore.Qt.MatchStartsWith) == -1

        editor.show_all_channels()
        app.processEvents()
        editor.names_panel.label_edits["ch04"].setText("aux_04")
        editor.channel_panel.delay_edits["ch04"].setText("40")
        app.processEvents()
        editor.clear_channel("ch04")
        app.processEvents()
        editor.hide_off_channels()
        app.processEvents()
        assert editor.state.visible_channels == ["ch00", "ch01", "ch02", "ch03"]
        ch04_index = editor.add_channel_combo.findText("ch04", QtCore.Qt.MatchStartsWith)
        assert ch04_index >= 0
        editor.add_channel_combo.setCurrentIndex(ch04_index)
        editor.add_selected_channel()
        app.processEvents()
        assert editor.state.visible_channels == ["ch00", "ch01", "ch02", "ch03", "ch04"]
        assert editor.names_panel.label_edits["ch04"].text() == "aux_04"
        assert editor.channel_panel.delay_edits["ch04"].text() == "40"

        editor.show_all_channels()
        app.processEvents()
        editor.tabs.setCurrentWidget(editor.preview_tab)
        app.processEvents()
        if FigureCanvas is not None:
            assert editor._preview_canvas is not None
            assert "plotted" in editor.preview_status.text()
            assert "repeat ∞" in editor.preview_status.text()
            assert editor._preview_plot.repeat_bracket[2] == "∞"
            assert len(editor._preview_plot.repeat_bracket_artists) == 2
            assert all(abs(bracket.get_alpha() - 0.58) < 1e-12 for bracket in editor._preview_plot.repeat_bracket_artists)
            for bracket in editor._preview_plot.repeat_bracket_artists:
                preview_y = bracket.get_ydata()
                assert min(preview_y) <= -0.3
                assert max(preview_y) >= len(editor._preview_plot.channels) - 0.2
            start, stop, _label = editor._preview_plot.repeat_bracket
            assert editor._preview_plot.repeat_bracket_label.get_position()[0] > stop
            assert editor._preview_plot.repeat_bracket_label.get_ha() == "left"
            assert editor._preview_plot.repeat_bracket_label.get_clip_on() is False
            assert editor.preview_body.width() >= editor._preview_canvas.width()
            assert editor.preview_body.height() >= editor._preview_canvas.height()

        editor.name_edit.setText("pulse_custom")
        app.processEvents()
        captured = {}

        def fake_save_dialog(*args, **kwargs):
            captured.setdefault("defaults", []).append(args[2])
            if "figure" in args[1].lower():
                return str(tmp_path / "pulse_custom.png"), "Pulse figure (*.png)"
            return str(tmp_path / "pulse_custom.json"), "ZLC pulse (*.json)"

        monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", fake_save_dialog)
        editor.save_to_file()
        assert "pulse_custom.json" in captured["defaults"][0].replace("\\", "/")
        assert (tmp_path / "pulse_custom.json").exists()
        assert not (tmp_path / "pulse_custom.png").exists()
        editor.save_figure()
        assert (tmp_path / "pulse_custom.png").exists()

        sequence = editor.to_sequence()
        assert sequence.validate(clock_hz=100e6, channels=channels).ok
        screenshot = editor.grab_screenshot(tmp_path / "pulse_gui_40ch.png")
        assert screenshot.exists()
        assert screenshot.stat().st_size > 1000
    finally:
        editor.close()


def test_standalone_pulse_gui_defaults_use_hardware_channel_names():
    import importlib
    from types import SimpleNamespace

    launcher = importlib.import_module("pulse_gui")

    channels = launcher._default_channels(5)
    assert channels == ["ch00", "ch01", "ch02", "ch03", "ch04"]
    assert not hasattr(launcher, "DEFAULT_CHANNEL_LABELS")
    args = SimpleNamespace(trigger_channels=None)
    assert launcher._resolve_trigger_channels(args, channels) == ["ch03"]


def test_pulse_gui_controls_call_attached_40ch_sequencer(monkeypatch):
    pytest.importorskip("PyQt5")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")

    from Zou_lab_control.frontend.pulse_gui import PulseSequenceEditor, ensure_qt_app

    app = ensure_qt_app()
    channels = [f"ch{i:02d}" for i in range(40)]

    class RecordingSequencer:
        clock_hz = 100e6
        trigger_channels = ["ch03"]

        def __init__(self):
            self.channels = channels
            self.prepared = None
            self.events = []

        def prepare(self, sequence):
            self.prepared = sequence
            self.events.append(("prepare", sequence.name, list(sequence.channels)))
            return na.compile_runtime_program(sequence, channels=self.channels, clock_hz=self.clock_hz, trigger_channels=self.trigger_channels)

        def fire(self):
            self.events.append(("fire",))

        def wait_done(self, timeout=None):
            self.events.append(("wait_done", timeout))
            return True

        def set_safe_state(self):
            self.events.append(("safe",))

    sequencer = RecordingSequencer()
    editor = PulseSequenceEditor(
        state=na.PulseTableState.load(Path(__file__).resolve().parents[1] / "pulses" / "camera_imaging_40ch.json"),
        sequencer=sequencer,
    )
    try:
        editor.show()
        app.processEvents()

        editor.prepare()
        editor.fire()
        editor.wait_done()
        editor.safe_state()
        app.processEvents()

        assert sequencer.events[0][0] == "prepare"
        assert [event[0] for event in sequencer.events] == ["prepare", "fire", "wait_done", "safe"]
        assert editor.last_program.channels == channels
        assert editor.last_program.trigger_count == 1
        assert sequencer.prepared.validate(clock_hz=100e6, channels=channels).ok
    finally:
        editor.close()


def test_pulse_gui_repeat_preview_uses_unexpanded_periods(monkeypatch):
    pytest.importorskip("PyQt5")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")

    from PyQt5 import QtWidgets

    from Zou_lab_control.frontend.pulse_gui import FigureCanvas, PulseSequenceEditor, ensure_qt_app

    if FigureCanvas is None:
        pytest.skip("Matplotlib Qt canvas is unavailable")

    app = ensure_qt_app()
    channels = ["ch00", "ch01", "ch02", "ch03"]
    editor = PulseSequenceEditor(channels=channels, scale=0.86)
    try:
        editor.show()
        app.processEvents()
        editor.names_panel.label_edits["ch00"].setText("trap")
        editor.names_panel.label_edits["ch01"].setText("cooling")
        editor.toggle_bracket()
        app.processEvents()
        bracket_end = next(item.widget for item in editor.drag_container.items if item.item_type == "bracket_end")
        assert bracket_end.repeat_spin.metaObject().className() == "FluentDoubleSpinBox"
        assert hasattr(bracket_end.repeat_spin, "_step_btn")
        assert bracket_end.repeat_spin._step_btn.text() == "."
        assert bracket_end.repeat_spin.buttonSymbols() == QtWidgets.QAbstractSpinBox.PlusMinus
        assert "FluentInputDialog" in bracket_end.repeat_spin._on_edit_step.__globals__
        state = editor.read_state()
        assert (state.repeat_start, state.repeat_end, state.repeat_count) == (0, 1, 2)

        editor.add_period()
        app.processEvents()
        state = editor.read_state()
        assert (state.repeat_start, state.repeat_end, state.repeat_count) == (0, 1, 2)
        assert len(state.periods) == 3

        editor.preview_include_off.setChecked(True)
        editor.tabs.setCurrentWidget(editor.preview_tab)
        app.processEvents()
        unexpanded = state.to_sequence(expand_repeat=False).duration
        expanded = state.to_sequence().duration
        assert expanded > unexpanded
        assert editor._preview_plot.sequence.duration == unexpanded
        assert "repeat ∞ + P1-P2 x2" in editor.preview_status.text()
        assert [label.get_text() for label in editor._preview_plot.repeat_bracket_labels] == ["∞", "x2"]
        assert [tick.get_text() for tick in editor._preview_plot.ax.get_yticklabels()] == ["trap", "cooling", "ch02", "ch03"]
        assert len(editor._preview_plot.repeat_bracket_artists) == 4
        outer_y = np.r_[editor._preview_plot.repeat_bracket_artists[0].get_ydata(), editor._preview_plot.repeat_bracket_artists[1].get_ydata()]
        inner_y = np.r_[editor._preview_plot.repeat_bracket_artists[2].get_ydata(), editor._preview_plot.repeat_bracket_artists[3].get_ydata()]
        assert outer_y.min() < inner_y.min()
        assert outer_y.max() > inner_y.max()
        assert editor._preview_plot.repeat_bracket[1] < expanded
        assert editor._preview_plot.ax.get_xlim()[0] < 0
        assert min(editor._preview_plot.repeat_bracket_artists[0].get_xdata()) == 0.0
        assert min(editor._preview_plot.repeat_bracket_artists[2].get_xdata()) == 0.0
        editor._preview_plot.fig.canvas.draw()
        x_tick_labels = [tick.get_text().strip() for tick in editor._preview_plot.ax.get_xticklabels() if tick.get_visible()]
        assert all(not label.startswith("-") for label in x_tick_labels if label)
        assert editor._preview_plot.channels == channels
    finally:
        editor.close()


def test_fluent_combo_box_wheel_ignores_closed_popup(monkeypatch):
    pytest.importorskip("PyQt5")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")

    from Zou_lab_control.frontend.qt_fluent import FluentComboBox, ensure_qt_app

    ensure_qt_app()
    combo = FluentComboBox()
    combo.addItems(["ns", "us", "ms"])
    combo.setCurrentIndex(0)

    class Event:
        ignored = False

        def ignore(self):
            self.ignored = True

    event = Event()
    combo.wheelEvent(event)
    assert combo.currentIndex() == 0
    assert event.ignored


def test_fluent_text_width_supports_old_qfontmetrics():
    from Zou_lab_control.frontend.qt_fluent import fluent_text_width

    class OldMetrics:
        def width(self, text):
            return len(text) * 7

    assert fluent_text_width(OldMetrics(), "pulse") == 35


def test_checked_in_tutorial_notebooks_are_utf8():
    for path in sorted((Path(__file__).resolve().parents[1] / "tutorials").glob("*.ipynb")):
        text = path.read_text(encoding="utf-8")
        assert "???" not in text, path
        assert "\ufffd" not in text, path
