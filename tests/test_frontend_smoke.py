import matplotlib

matplotlib.use("Agg")

import time
from types import SimpleNamespace

import numpy as np

import Zou_lab_control.frontend as zf


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
    first_patch = pulse.pulse_artists[0]
    assert first_line[0, 1] < 0
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

    for result in (frontend, neutral):
        text = result.path.read_text(encoding="utf-8")
        assert "???" not in text
        assert "\ufffd" not in text
    assert "这个 notebook" in neutral.path.read_text(encoding="utf-8")
    assert "BOOTSTRAP_CELL" in zf.__all__
    assert "write_neutral_atom_tutorial" in zf.__all__
