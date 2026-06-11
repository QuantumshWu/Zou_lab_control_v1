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


def test_render_latex_pdf_clean_copies_only_final_pdf(tmp_path, monkeypatch):
    import Zou_lab_control.frontend.notes as notes

    calls = []

    class Result:
        returncode = 0
        stdout = "fake xelatex ok"

    def fake_run(cmd, cwd, text, stdout, stderr):
        calls.append((cmd, Path(cwd)))
        tex_name = Path(cmd[-1])
        (Path(cwd) / tex_name.with_suffix(".pdf").name).write_bytes(b"%PDF fake\n")
        (Path(cwd) / tex_name.with_suffix(".aux").name).write_text("aux", encoding="utf-8")
        return Result()

    monkeypatch.setattr(notes.subprocess, "run", fake_run)
    out = tmp_path / "manual.pdf"

    pdf = zf.render_latex_pdf_clean(
        r"\documentclass{article}\begin{document}hello\end{document}",
        out,
        xelatex="fake-xelatex",
        runs=1,
    )

    assert pdf == out.resolve()
    assert out.read_bytes().startswith(b"%PDF")
    assert calls and calls[0][0][0] == "fake-xelatex"
    assert not (tmp_path / "manual.aux").exists()
    assert not (tmp_path / "manual.build.log").exists()

    out2 = tmp_path / "manual2.pdf"
    pdf2 = zf.render_tex_pdf(
        r"\documentclass{article}\begin{document}hello again\end{document}",
        out2,
        xelatex="fake-xelatex",
        runs=1,
    )
    assert pdf2 == out2.resolve()
    assert out2.read_bytes().startswith(b"%PDF")
    assert not (tmp_path / "manual2.aux").exists()

    result = zf.render_notes_pdf(
        tmp_path / "notes",
        filename="quick_note.tex",
        title="Quick Note",
        body="hello",
        xelatex="fake-xelatex",
        runs=1,
    )
    assert result.pdf_path.read_bytes().startswith(b"%PDF")
    assert result.tex_path.name == "quick_note.tex"
    assert not (result.tex_path.parent / "quick_note.aux").exists()
    assert result.log_path is None

    draft = zf.render_notes_pdf(
        tmp_path / "draft",
        filename="draft_note.tex",
        title="Draft Note",
        body="hello",
        compile_pdf=False,
    )
    assert draft.tex_path.exists()
    assert draft.pdf_path == draft.tex_path.with_suffix(".pdf")
    assert draft.log_path is None


def test_render_tex_pdf_skips_stale_latex_auxiliary_files(tmp_path, monkeypatch):
    import Zou_lab_control.frontend.notes as notes

    copied_files = []

    class Result:
        returncode = 0
        stdout = "fake xelatex ok"

    def fake_run(cmd, cwd, text, stdout, stderr):
        build_dir = Path(cwd)
        copied_files.append({path.name for path in build_dir.rglob("*") if path.is_file()})
        tex_name = Path(cmd[-1])
        (build_dir / tex_name.with_suffix(".pdf").name).write_bytes(b"%PDF fake\n")
        return Result()

    monkeypatch.setattr(notes.subprocess, "run", fake_run)
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "manual.tex").write_text(r"\documentclass{article}\begin{document}hello\end{document}", encoding="utf-8")
    (source_dir / "manual.aux").write_text("old aux", encoding="utf-8")
    (source_dir / "manual.log").write_text("old log", encoding="utf-8")
    (source_dir / "manual.build.log").write_text("old build log", encoding="utf-8")
    (source_dir / "asset.txt").write_text("asset", encoding="utf-8")

    out = tmp_path / "out" / "manual.pdf"
    pdf = zf.render_tex_pdf(source_dir / "manual.tex", out, xelatex="fake-xelatex", runs=1)

    assert pdf == out.resolve()
    assert out.read_bytes().startswith(b"%PDF")
    assert copied_files
    assert "manual.tex" in copied_files[0]
    assert "asset.txt" in copied_files[0]
    assert "manual.aux" not in copied_files[0]
    assert "manual.log" not in copied_files[0]
    assert "manual.build.log" not in copied_files[0]
    assert not (out.parent / "manual.aux").exists()
    assert not (out.parent / "manual.log").exists()
    assert not (out.parent / "manual.build.log").exists()


def test_render_tex_pdf_does_not_copy_stale_source_or_output_pdf(tmp_path, monkeypatch):
    import Zou_lab_control.frontend.notes as notes

    copied_files = []

    class Result:
        returncode = 0
        stdout = "fake xelatex ok"

    def fake_run(cmd, cwd, text, stdout, stderr):
        build_dir = Path(cwd)
        copied_files.append({str(path.relative_to(build_dir)) for path in build_dir.rglob("*") if path.is_file()})
        tex_name = Path(cmd[-1])
        (build_dir / tex_name.with_suffix(".pdf").name).write_bytes(b"%PDF fresh\n")
        return Result()

    monkeypatch.setattr(notes.subprocess, "run", fake_run)
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "manual.tex").write_text(r"\documentclass{article}\begin{document}fresh\end{document}", encoding="utf-8")
    (source_dir / "manual.pdf").write_bytes(b"%PDF stale source\n")
    (source_dir / "assets").mkdir()
    (source_dir / "assets" / "figure.pdf").write_bytes(b"%PDF asset\n")
    out = tmp_path / "out" / "manual.pdf"
    out.parent.mkdir()
    out.write_bytes(b"%PDF stale output\n")
    out.with_suffix(".build.log").write_text("old failure", encoding="utf-8")

    pdf = zf.render_tex_pdf(source_dir / "manual.tex", out, xelatex="fake-xelatex", runs=1)

    assert pdf == out.resolve()
    assert out.read_bytes() == b"%PDF fresh\n"
    assert copied_files
    assert "manual.tex" in copied_files[0]
    assert "manual.pdf" not in copied_files[0]
    assert "assets\\figure.pdf" in copied_files[0] or "assets/figure.pdf" in copied_files[0]
    assert not out.with_suffix(".build.log").exists()


def test_render_tex_pdf_skips_stale_output_pdf_inside_source_tree(tmp_path, monkeypatch):
    import Zou_lab_control.frontend.notes as notes

    copied_files = []

    class Result:
        returncode = 0
        stdout = "fake xelatex ok"

    def fake_run(cmd, cwd, text, stdout, stderr):
        build_dir = Path(cwd)
        copied_files.append({str(path.relative_to(build_dir)) for path in build_dir.rglob("*") if path.is_file()})
        (build_dir / "manual.pdf").write_bytes(b"%PDF fresh\n")
        return Result()

    monkeypatch.setattr(notes.subprocess, "run", fake_run)
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "manual.tex").write_text(r"\documentclass{article}\begin{document}fresh\end{document}", encoding="utf-8")
    (source_dir / "final.pdf").write_bytes(b"%PDF stale output\n")
    (source_dir / "assets").mkdir()
    (source_dir / "assets" / "figure.pdf").write_bytes(b"%PDF asset\n")

    pdf = zf.render_tex_pdf(source_dir / "manual.tex", source_dir / "final.pdf", xelatex="fake-xelatex", runs=1)

    assert pdf == (source_dir / "final.pdf").resolve()
    assert (source_dir / "final.pdf").read_bytes() == b"%PDF fresh\n"
    assert copied_files
    assert "manual.tex" in copied_files[0]
    assert "final.pdf" not in copied_files[0]
    assert "assets\\figure.pdf" in copied_files[0] or "assets/figure.pdf" in copied_files[0]


def test_render_tex_pdf_failure_writes_only_build_log(tmp_path, monkeypatch):
    import Zou_lab_control.frontend.notes as notes

    class Result:
        returncode = 1
        stdout = "fake xelatex failure"

    def fake_run(cmd, cwd, text, stdout, stderr):
        build_dir = Path(cwd)
        tex_name = Path(cmd[-1])
        (build_dir / tex_name.with_suffix(".aux").name).write_text("aux", encoding="utf-8")
        (build_dir / tex_name.with_suffix(".log").name).write_text("log", encoding="utf-8")
        return Result()

    monkeypatch.setattr(notes.subprocess, "run", fake_run)

    out = tmp_path / "failed.pdf"
    out.write_bytes(b"%PDF stale\n")
    with pytest.raises(RuntimeError, match="xelatex failed"):
        zf.render_tex_pdf(
            r"\documentclass{article}\begin{document}bad\end{document}",
            out,
            xelatex="fake-xelatex",
            runs=1,
        )

    assert not out.exists()
    assert (tmp_path / "failed.build.log").read_text(encoding="utf-8") == "fake xelatex failure"
    assert not (tmp_path / "failed.aux").exists()
    assert not (tmp_path / "failed.log").exists()


def test_render_tex_pdf_missing_xelatex_writes_build_log(tmp_path, monkeypatch):
    import Zou_lab_control.frontend.notes as notes

    monkeypatch.setattr(notes.shutil, "which", lambda name: None)
    out = tmp_path / "missing.pdf"
    out.write_bytes(b"%PDF stale\n")

    with pytest.raises(RuntimeError, match="See"):
        zf.render_tex_pdf(r"\documentclass{article}\begin{document}hello\end{document}", out, runs=1)

    assert not out.exists()
    log_text = out.with_suffix(".build.log").read_text(encoding="utf-8")
    assert "xelatex was not found on PATH" in log_text
    assert not (tmp_path / "missing.aux").exists()
    assert not (tmp_path / "missing.log").exists()


def test_render_tex_pdf_bad_xelatex_path_writes_build_log(tmp_path, monkeypatch):
    import Zou_lab_control.frontend.notes as notes

    def fake_run(cmd, cwd, text, stdout, stderr):
        raise FileNotFoundError(cmd[0])

    monkeypatch.setattr(notes.subprocess, "run", fake_run)
    out = tmp_path / "bad_path.pdf"
    out.write_bytes(b"%PDF stale\n")

    with pytest.raises(RuntimeError, match="executable was not found"):
        zf.render_tex_pdf(
            r"\documentclass{article}\begin{document}hello\end{document}",
            out,
            xelatex="missing-xelatex",
            runs=1,
        )

    assert not out.exists()
    log_text = out.with_suffix(".build.log").read_text(encoding="utf-8")
    assert "xelatex executable was not found: missing-xelatex" in log_text
    assert not (tmp_path / "bad_path.aux").exists()
    assert not (tmp_path / "bad_path.log").exists()


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
    # Non-bracket timeline: the display window is a touch wider than the data on
    # BOTH sides, so a first edge at t=0 gets breathing room instead of sitting
    # flush on the spine (the left used to be clamped to 0). Margins are
    # symmetric, and the negative-time headroom is never given a tick label.
    pulse_xlim = pulse.ax.get_xlim()
    pulse_stop = max(row.start + row.duration for row in Seq().effective_pulses())
    left_margin = 0.0 - pulse_xlim[0]
    right_margin = pulse_xlim[1] - pulse_stop
    assert pulse_xlim[0] < 0.0
    assert left_margin > 0.0 and right_margin > 0.0
    assert abs(left_margin - right_margin) < 1e-6 * left_margin
    assert all(not label.startswith("-") for label in x_tick_labels if label)

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
    assert zf.pulse_repeat_marker(total_duration_s=3e-6) == (0.0, 3e-6, "×∞")
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
    assert [marker[2] for marker in repeat_markers] == ["×∞", "×5"]
    assert np.allclose([(marker[0], marker[1]) for marker in repeat_markers], [(0.0, 600e-9), (100e-9, 300e-9)])
    outer_repeat_state = na.PulseTableState.from_dict({**repeat_state.to_dict(), "repeat_start": 0, "repeat_end": 2, "repeat_count": 3})
    assert zf.pulse_repeat_notation(outer_repeat_state) == "repeat P1-P3 x3"
    outer_markers = zf.pulse_repeat_markers(outer_repeat_state)
    assert [marker[2] for marker in outer_markers] == ["×3"]
    assert np.allclose([(marker[0], marker[1]) for marker in outer_markers], [(0.0, 600e-9)])
    bracketed = zf.plot(
        FortySeq(),
        kind="pulse",
        repeat_notation="repeat ∞",
        repeat_bracket=(0.0, 3e-6, "×∞"),
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
    assert bracketed.repeat_bracket_label.get_text() == "×∞"
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
        repeat_bracket=(0.0, 5e-6, "×∞"),
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
        repeat_brackets=[(0.0, 4.3e-6, "×∞"), (1.0e-6, 3.2e-6, "×4")],
        display=False,
    )
    middle_repeat.fig.canvas.draw()
    assert len(middle_repeat.repeat_bracket_artists) == 4
    assert [label.get_text() for label in middle_repeat.repeat_bracket_labels] == ["×∞", "×4"]
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


def test_pulse_gui_layout_geometry_contract(monkeypatch):
    """Geometric layout self-check (the 'GUI test interface' the user asked for).

    Measures actual widget geometry instead of eyeballing screenshots:
    - the scan dot is EMBEDDED in its line edit (parent is the field) and
      vertically CENTERED, sitting on the right edge like a spinbox spin button;
    - the Delay/Scan panel's 'Load Array' button, file-path label and the
      generated/loaded source toggle are present and aligned in the row;
    - no Control/Channels button has its text clipped.
    """

    pytest.importorskip("PyQt5")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PyQt5 import QtGui
    from Zou_lab_control.frontend import devtools as dt

    editor = dt.demo_editor(size=(1480, 900))
    dt.settle(editor, 300)

    # --- scan dot: embedded + centered + right-edge ---
    field = editor.drag_container.pulse_cards()[0].duration_edit
    dot = field.dot
    assert dot.parent() is field  # the dot lives INSIDE the line edit
    dr, fr = dot.geometry(), field.rect()
    assert abs((dr.y() + dr.height() / 2) - fr.height() / 2) <= 1.5  # vertically centred
    assert 0 <= fr.right() - dr.right() <= 8  # hugs the right edge (spinbox-spin style)

    # --- Load Array button + file-path label + source toggle ---
    la = editor.channel_panel.load_button
    fl = editor.channel_panel.scan_file_label
    tg = editor.channel_panel.scan_source_toggle
    assert la.text() == "Load Array"
    # button and the path label share the same row (same top within a pixel)
    assert abs(la.mapTo(editor, la.rect().topLeft()).y() - fl.mapTo(editor, fl.rect().topLeft()).y()) <= 1
    # the source toggle starts OFF (use the generated table) and sits below the row
    assert not tg.isChecked()
    assert tg.mapTo(editor, tg.rect().topLeft()).y() > la.mapTo(editor, la.rect().topLeft()).y()

    # --- no control/channel button text clipped ---
    buttons = [
        editor.safe_button, editor.fire_button, editor.remove_button, editor.add_button,
        editor.bracket_button, editor.collapse_button, editor.save_button, editor.load_button,
        editor.add_channel_button, editor.hide_off_button, editor.show_all_button,
    ]
    for b in buttons:
        metrics = QtGui.QFontMetrics(b.font())
        text_w = metrics.horizontalAdvance(b.text().replace("\n", " "))
        assert text_w <= b.width() - 8, (b.text(), text_w, b.width())


def test_pulse_gui_preview_robust_across_states(monkeypatch):
    """The preview must render (never blank/crash) across many edge-case states,
    and the Show-off-rows toggle must be stable (OFF->ON->OFF is reproducible).

    This guards the user's '#3' reports: 'preview sometimes shows no image' and
    'show-off toggle breaks the display'.
    """

    pytest.importorskip("PyQt5")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from Zou_lab_control.frontend import devtools as dt
    from Zou_lab_control.frontend.pulse_gui import PulseSequenceEditor
    from Zou_lab_control.frontend.qt_fluent import ensure_qt_app

    ensure_qt_app()

    def all_off():
        return na.PulseTableState(
            channels=["ch00", "ch01", "ch02"], visible_channels=["ch00", "ch01", "ch02"],
            periods=[na.PulsePeriod(100, (0, 0, 0), unit="ns"), na.PulsePeriod(200, (0, 0, 0), unit="ns")],
            time_step_ns=20,
        )

    def one_min():
        return na.PulseTableState(channels=["ch00"], visible_channels=["ch00"],
                                  periods=[na.PulsePeriod(100, (1,), unit="ns")], time_step_ns=20)

    def scanned():
        st = dt.demo_state()
        for p, v in enumerate([0, 300, 500, 300, 0]):
            st.set_bus_value(p, "da_dipole", v)
        return st

    def ramped():
        st = dt.demo_state()
        st.set_analog_bus_mode(0, "da_dipole", "edge", value=0)
        st.set_analog_bus_mode(1, "da_dipole", "ramp", value=500)
        st.apply_analog_bus_modes_to_period_states()
        return st

    builders = {"all_off": all_off, "one_min": one_min, "scanned": scanned, "ramped": ramped}
    for name, build in builders.items():
        editor = PulseSequenceEditor(state=build())
        if name == "scanned":
            editor._toggle_duration_scan(editor.drag_container.pulse_cards()[3])
            editor._toggle_dac_scan(editor.drag_container.pulse_cards()[2], "da_dipole")
        # both toggle states must produce a real figure with at least one axis line/patch
        for include_off in (False, True, False):  # OFF -> ON -> OFF reproducible
            plotter, channels, _repeat = editor._create_preview_plot(editor.read_state(), include_always_off=include_off)
            assert plotter is not None and plotter.fig is not None, name
            # the axes must have drawn *something* (a baseline line or a pulse patch),
            # i.e. it is never a truly blank canvas.
            ax = plotter.ax
            assert (len(ax.lines) + len(ax.patches) + len(ax.collections)) > 0, (name, include_off)


def test_pulse_gui_row_alignment_contract(monkeypatch):
    """Each channel's raw label, delay box and first checkbox must share a row.

    This is the visible "对齐契约": the three columns scroll together and a
    given channel's controls line up.  Asserting the vertical centre coincide
    catches layout drift that screenshots would only show subtly.
    """

    pytest.importorskip("PyQt5")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from Zou_lab_control.frontend import devtools as dt

    editor = dt.demo_editor(size=(1480, 900))

    def centre_y(widget):
        return widget.mapTo(editor, widget.rect().center()).y()

    def check_alignment(label):
        names = editor.names_panel
        delays = editor.channel_panel
        first_card = editor.drag_container.pulse_cards()[0]
        checked = 0
        for channel in editor.state.visible_channels[:8]:
            raw = names.raw_label_widgets.get(channel)
            delay = delays.delay_edits.get(channel)
            check = first_card.checks.get(channel)
            ys = [centre_y(w) for w in (raw, delay, check) if w is not None]
            if len(ys) >= 2:
                assert max(ys) - min(ys) <= 1, (label, channel, ys)
                checked += 1
        assert checked >= 4, label
        # The analog-bus (DAC) row must line up too -- its period-card row used to
        # be inflated to 30 px while the Names/Delay panels kept it at the
        # compressed 26 px, knocking every row below it out of alignment.
        for bus in first_card.bus_value_edits:
            raw = names.raw_label_widgets.get(f"bus:{bus}")
            delay = delays.delay_edits.get(f"bus:{bus}")
            period = first_card.bus_value_edits.get(bus)
            ys = [centre_y(w) for w in (raw, delay, period) if w is not None]
            assert len(ys) == 3, (label, bus, "missing bus row widget")
            assert max(ys) - min(ys) <= 1, (label, bus, ys)

    dt.settle(editor, 300)
    check_alignment("default")

    # Now reveal every hardware channel (62 -> 26 px compressed rows) and re-check:
    # this is the regression scenario where the DAC row height diverged.
    editor.show_all_channels()
    dt.settle(editor, 300)
    check_alignment("show-all")


def test_pulse_gui_channel_display_order_is_fixed_hardware_order(monkeypatch):
    """Edit (and preview) channel rows must ALWAYS render in the fixed hardware
    channel order, no matter how show/hide/hide-off scrambled visible_channels.

    This is the user's "通道显示固定排序不被打乱": _display_rows now walks
    state.channels (hardware order) filtered by visibility, so the order cannot
    depend on the order entries happen to sit in visible_channels.
    """

    pytest.importorskip("PyQt5")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    import Zou_lab_control.neutral_atom as na
    from Zou_lab_control.frontend.pulse_gui import _display_rows

    # --- data-model: a deliberately scrambled visible_channels still displays
    #     in hardware order, visible-filtered.
    channels = [f"ch{i:02d}" for i in range(10)]
    state = na.PulseTableState(
        channels=channels,
        periods=[na.PulsePeriod(100, tuple(1 for _ in channels), unit="ns")],
        time_step_ns=20,
        visible_channels=channels,
    )
    state.visible_channels = ["ch05", "ch01", "ch09", "ch00", "ch03"]  # scrambled
    keys = [row["key"] for row in _display_rows(state)]
    assert keys == ["ch00", "ch01", "ch03", "ch05", "ch09"], keys

    # --- editor path: show-all then hide-off must leave visible_channels in
    #     hardware order AND render the panel rows top-to-bottom in that order.
    from Zou_lab_control.frontend import devtools as dt

    editor = dt.demo_editor(size=(1480, 900))
    editor.show_all_channels()
    dt.settle(editor, 200)
    editor.hide_off_channels()
    dt.settle(editor, 200)

    vis = list(editor.state.visible_channels)
    assert vis == sorted(vis, key=editor.state.channel_index), vis

    rows = _display_rows(editor.state)
    names = editor.names_panel
    ys = []
    for row in rows:
        widget = names.raw_label_widgets.get(row["key"])
        if widget is not None:
            ys.append(widget.mapTo(editor, widget.rect().center()).y())
    assert len(ys) >= 3, "names panel should expose the visible row labels"
    assert ys == sorted(ys), ("panel rows must render top-to-bottom in hardware order", ys)


def test_pulse_gui_scan_dot_retoggle_preserves_values(monkeypatch):
    """Toggling a scan dot OFF then ON restores the typed scan column.

    The user's "#3": re-clicking a scan dot must not wipe the values that were
    already entered for that field.
    """

    pytest.importorskip("PyQt5")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from Zou_lab_control.frontend import devtools as dt
    from Zou_lab_control.frontend.pulse_gui import PulseSequenceEditor
    from Zou_lab_control.frontend.qt_fluent import ensure_qt_app

    ensure_qt_app()
    editor = PulseSequenceEditor(state=dt.demo_state())
    editor._toggle_duration_scan(editor.drag_container.pulse_cards()[1])
    editor.state.set_scan_table([[100], [200], [300]])
    editor.load_state(editor.state)
    assert editor.state.scan_table == [[100.0], [200.0], [300.0]]

    editor._toggle_duration_scan(editor.drag_container.pulse_cards()[1])  # unbind
    assert len(editor.state.scan_slots) == 0

    editor._toggle_duration_scan(editor.drag_container.pulse_cards()[1])  # rebind -> values restored, not reset to nominal
    assert editor.state.scan_table == [[100.0], [200.0], [300.0]]


def test_pulse_gui_scan_tab_columns_bottom_aligned(monkeypatch):
    """The Scan tab's code box and table box must share a bottom edge.

    The user's "#5": the right (table) box used to hang ~one row lower than the
    left (code) box -- which has the Run/Load/Save buttons beneath it -- so its
    grey border read as a stray "extra grey edge" protruding past the left box.
    """

    pytest.importorskip("PyQt5")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from Zou_lab_control.frontend import devtools as dt

    editor = dt.demo_editor(size=(1480, 900))
    editor.tabs.setCurrentWidget(editor.scan_tab)
    dt.settle(editor, 300)

    code_bottom = editor.scan_code.mapTo(editor.scan_tab, editor.scan_code.rect().bottomLeft()).y()
    table_bottom = editor.scan_table_view.mapTo(
        editor.scan_tab, editor.scan_table_view.rect().bottomLeft()
    ).y()
    assert abs(code_bottom - table_bottom) <= 1, (code_bottom, table_bottom)
    # And the two boxes' right edges line up with the info banner above them.
    info = editor.scan_slots_label.parentWidget()
    info_right = info.mapTo(editor.scan_tab, info.rect().topRight()).x()
    table_right = editor.scan_table_view.parentWidget().mapTo(
        editor.scan_tab, editor.scan_table_view.parentWidget().rect().topRight()
    ).x()
    assert abs(info_right - table_right) <= 1, (info_right, table_right)


def test_pulse_gui_inplace_scan_refresh_matches_rebuild(monkeypatch):
    """The fast in-place scan refresh must be pixel-faithful to a full rebuild.

    Toggling a scan dot updates the existing widgets in place (the perf fix for
    the ~400 ms 'Show All' lag) instead of recreating them.  After a sequence of
    binds/unbinds, forcing a full load_state must reproduce exactly the same
    widget state -- otherwise the fast path has drifted from the source of truth.
    """

    pytest.importorskip("PyQt5")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from Zou_lab_control.frontend import devtools as dt
    from Zou_lab_control.frontend.pulse_gui import PulseSequenceEditor
    from Zou_lab_control.frontend.qt_fluent import ensure_qt_app

    app = ensure_qt_app()
    editor = PulseSequenceEditor(state=dt.demo_state())
    editor.show()
    editor.show_all_channels()
    app.processEvents()

    def snapshot():
        fields = []
        for i, card in enumerate(editor.drag_container.pulse_cards()):
            d = card.duration_edit
            fields.append(("dur", i, d.text(), d._bound, d.dot.number(), card.unit_combo.currentText()))
            for bus, e in card.bus_value_edits.items():
                fields.append(("dac", i, bus, e.text(), e._bound, e.dot.number(),
                               card.bus_mode_combos[bus].currentText()))
        for k, e in editor.channel_panel.delay_edits.items():
            # delay is a fixed per-channel value -- a plain line edit, never scannable
            fields.append(("del", k, e.text(), editor.channel_panel.delay_units[k].currentText()))
        return editor.read_state().to_dict(), fields

    bus = list(editor.state.bus_channels())[0]
    editor.drag_container.pulse_cards()[0].bus_mode_combos[bus].setCurrentText("Edge")
    app.processEvents()
    # A mix of binds and unbinds across duration + DAC kinds, all via the in-place path.
    editor._toggle_duration_scan(editor.drag_container.pulse_cards()[2]); app.processEvents()
    editor._toggle_duration_scan(editor.drag_container.pulse_cards()[1]); app.processEvents()
    editor.drag_container.pulse_cards()[0].bus_dots[bus].clicked.emit(); app.processEvents()
    editor._toggle_duration_scan(editor.drag_container.pulse_cards()[3]); app.processEvents()
    editor._toggle_duration_scan(editor.drag_container.pulse_cards()[2]); app.processEvents()  # unbind -> renumbers later slots

    state_fast, fields_fast = snapshot()
    # Force the slow path: a full rebuild from the read-back state.
    editor.load_state(editor.read_state())
    app.processEvents()
    state_slow, fields_slow = snapshot()

    assert fields_fast == fields_slow
    assert state_fast == state_slow


def test_pulse_gui_save_button_keeps_width_on_state_change(monkeypatch):
    """The Save button must not collapse when its colour changes (e.g. on load).

    set_color() used to reset the size policy to Fixed, so the bottom-bar Save
    button (opted into Expanding) shrank to its text width the first time its
    colour flipped -- which happens on load (dirty yellow -> clean blue).
    """

    pytest.importorskip("PyQt5")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PyQt5 import QtWidgets
    from Zou_lab_control.frontend import devtools as dt

    editor = dt.demo_editor(size=(1480, 900))
    dt.settle(editor, 200)
    width0 = editor.save_button.width()
    assert editor.save_button.sizePolicy().horizontalPolicy() == QtWidgets.QSizePolicy.Expanding
    FileState = type(editor.stateui_manager).FileState
    editor.stateui_manager.filestate = FileState.UNSAVED  # -> yellow
    dt.settle(editor, 40)
    editor.stateui_manager.filestate = FileState.LOAD  # -> blue (colour change)
    dt.settle(editor, 40)
    assert editor.save_button.sizePolicy().horizontalPolicy() == QtWidgets.QSizePolicy.Expanding
    assert abs(editor.save_button.width() - width0) <= 1, (width0, editor.save_button.width())


def test_pulse_gui_confocal_star_state_semantics(monkeypatch):
    """State indication follows confocal: stars + status dot, NEVER base colours.

    The user's complaints this guards:
    * an orange UNSYNCED On Pulse was indistinguishable from the permanently
      orange Remove/Load/Sync buttons ("你根本就不知道哪个是高亮的");
    * Add Bracket was permanently yellow, colliding with Save's dirty yellow;
    * the confocal '*' suffix was missing, and only Save reflected state while
      On Pulse did not.
    """

    pytest.importorskip("PyQt5")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PyQt5 import QtGui

    from Zou_lab_control.frontend import devtools as dt
    from Zou_lab_control.frontend.qt_fluent import ACCENT, GREEN, ORANGE, RED, YELLOW

    def bg(button):
        return button._current_bg

    editor = dt.demo_editor(size=(1480, 900))
    try:
        dt.settle(editor, 100)
        manager = editor.stateui_manager
        RunState = type(manager).RunState
        FileState = type(manager).FileState
        green = QtGui.QColor(GREEN).name(QtGui.QColor.HexRgb)
        yellow = QtGui.QColor(YELLOW).name(QtGui.QColor.HexRgb)
        accent = QtGui.QColor(ACCENT).name(QtGui.QColor.HexRgb)

        # Base palette must not reuse the state colours: Add Bracket is accent
        # (yellow is reserved for Save-dirty).
        assert bg(editor.bracket_button) == accent

        # INIT: editor not applied anywhere -> star, GREEN (colour fixed).
        assert editor.fire_button.text() == "On Pulse*"
        assert bg(editor.fire_button) == green

        # RUNNING and in sync: the ONLY state without the star.
        manager.runstate = RunState.RUNNING
        assert editor.fire_button.text() == "On Pulse"
        assert bg(editor.fire_button) == green

        # An edit while running -> UNSYNCED: star comes back, colour STAYS green
        # (the orange is on the status dot, not the button).
        editor._mark_dirty()
        assert manager.runstate == RunState.UNSYNCED
        assert editor.fire_button.text() == "On Pulse*"
        assert bg(editor.fire_button) == green
        dot_style = editor.status_dot.styleSheet().lower()
        assert ORANGE.lower() in dot_style

        # Every other run state keeps the star and the green colour.
        for state in (RunState.PREPARED, RunState.STOP, RunState.SAFE, RunState.ERROR):
            manager.runstate = state
            assert editor.fire_button.text() == "On Pulse*", state
            assert bg(editor.fire_button) == green, state

        # Save: confocal FileState semantics -- star+yellow dirty, plain+accent clean.
        manager.filestate = FileState.UNSAVED
        assert editor.save_button.text() == "Save*"
        assert bg(editor.save_button) == yellow
        manager.filestate = FileState.LOAD
        assert editor.save_button.text() == "Save"
        assert bg(editor.save_button) == accent
        manager.filestate = FileState.UNTITLED
        assert editor.save_button.text() == "Save*"
        assert bg(editor.save_button) == yellow

        # The stars must not change button geometry (equal-stretch grid columns).
        manager.runstate = RunState.RUNNING
        dt.settle(editor, 40)
        width_clean = editor.fire_button.width()
        manager.runstate = RunState.UNSYNCED
        dt.settle(editor, 40)
        assert abs(editor.fire_button.width() - width_clean) <= 1
    finally:
        editor.close()


def test_pulse_gui_scan_unbind_restores_field_state(monkeypatch):
    """Unbinding a scan slot must restore the field's ORIGINAL value/mode.

    The user's bug: a DAC that was "edge / 500" came back as "hold" after a
    bind/unbind round-trip.  The same hard-default reset hit duration (-> 1000).
    Verify both return to exactly what they were.
    """

    pytest.importorskip("PyQt5")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from Zou_lab_control.frontend import devtools as dt
    from Zou_lab_control.frontend.pulse_gui import PulseSequenceEditor
    from Zou_lab_control.frontend.qt_fluent import ensure_qt_app

    app = ensure_qt_app()
    editor = PulseSequenceEditor(state=dt.demo_state())
    editor.show()
    app.processEvents()

    # --- DAC: edge / 500 -> bind -> unbind -> edge / 500 ---
    bus = list(editor.state.bus_channels())[0]
    plan = editor.state.analog_bus_plan(bus)
    plan[0] = {"mode": "edge", "value": 500}
    editor.state.analog_bus_modes[bus] = plan
    editor.load_state(editor.state)
    app.processEvents()
    editor.drag_container.pulse_cards()[0].bus_dots[bus].clicked.emit()
    app.processEvents()
    editor.drag_container.pulse_cards()[0].bus_dots[bus].clicked.emit()
    app.processEvents()
    assert editor.state.analog_bus_plan(bus)[0] == {"mode": "edge", "value": 500}

    # --- duration: keep its original ns value across bind/unbind ---
    original = editor.state.periods[1].duration
    editor._toggle_duration_scan(editor.drag_container.pulse_cards()[1])
    app.processEvents()
    editor._toggle_duration_scan(editor.drag_container.pulse_cards()[1])
    app.processEvents()
    assert str(editor.state.periods[1].duration) == str(original)


def test_pulse_gui_scan_dots_clickable_bind_and_unbind(monkeypatch):
    """Clicking a scan dot must bind AND later unbind -- for duration and DAC.

    The user's bug: "DA dot can't be clicked back".  Root cause was the DAC
    value field being *disabled* when bound, which also disabled its embedded
    dot, so the second click never reached the toggle.  This exercises the real
    signal path (dot.clicked -> scanClicked -> toggle) and asserts the dot stays
    enabled the whole time.  (A per-channel delay is a fixed value and has no
    scan dot.)
    """

    pytest.importorskip("PyQt5")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from Zou_lab_control.frontend import devtools as dt
    from Zou_lab_control.frontend.pulse_gui import PulseSequenceEditor
    from Zou_lab_control.frontend.qt_fluent import ensure_qt_app

    app = ensure_qt_app()
    editor = PulseSequenceEditor(state=dt.demo_state())
    editor.show()
    app.processEvents()

    # --- duration dot (period card) ---
    dur_dot = editor.drag_container.pulse_cards()[1].duration_edit.dot
    assert dur_dot.isEnabled()
    dur_dot.clicked.emit()
    app.processEvents()
    assert any(s.kind == "duration" and s.target == "1" for s in editor.state.scan_slots)
    dur_dot = editor.drag_container.pulse_cards()[1].duration_edit.dot  # rebuilt
    assert dur_dot.isEnabled()
    dur_dot.clicked.emit()
    app.processEvents()
    assert not any(s.kind == "duration" and s.target == "1" for s in editor.state.scan_slots)

    # --- DAC value dot (force Edge mode so the value is scannable) ---
    bus = list(editor.state.bus_channels())[0]
    card = editor.drag_container.pulse_cards()[0]
    card.bus_mode_combos[bus].setCurrentText("Edge")
    app.processEvents()
    dac_dot = editor.drag_container.pulse_cards()[0].bus_dots[bus]
    assert dac_dot.isEnabled()
    dac_dot.clicked.emit()
    app.processEvents()
    assert any(s.kind == "dac" for s in editor.state.scan_slots)
    dac_dot = editor.drag_container.pulse_cards()[0].bus_dots[bus]  # rebuilt, now bound
    assert dac_dot.isEnabled(), "bound DAC dot must stay clickable so it can be unbound"
    dac_dot.clicked.emit()
    app.processEvents()
    assert not any(s.kind == "dac" for s in editor.state.scan_slots)


def test_zoompan_scroll_down_zooms_in():
    """Scroll DOWN must zoom in (smaller view range), scroll UP zoom out."""

    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from Zou_lab_control.frontend.selectors import ZoomPan

    fig, ax = plt.subplots()
    ax.set_xlim(0, 10)
    zoom = ZoomPan(ax)

    class _Event:
        inaxes = ax
        xdata = 5.0
        ydata = 5.0
        button = "down"

    zoom.on_scroll(_Event())
    assert (ax.get_xlim()[1] - ax.get_xlim()[0]) < 10, "scroll down should zoom in"
    ax.set_xlim(0, 10)
    _Event.button = "up"
    zoom.on_scroll(_Event())
    assert (ax.get_xlim()[1] - ax.get_xlim()[0]) > 10, "scroll up should zoom out"
    plt.close(fig)


def test_pulse_preview_has_area_selector(monkeypatch):
    """The pulse preview must expose a left-drag area selector (was disabled)."""

    pytest.importorskip("PyQt5")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from Zou_lab_control.frontend import devtools as dt

    editor = dt.demo_editor(size=(1200, 800))
    editor.tabs.setCurrentWidget(editor.preview_tab)
    dt.settle(editor, 200)
    assert getattr(editor._preview_plot, "area", None) is not None


def test_pulse_save_bundles_artifacts(monkeypatch, tmp_path):
    """Saving a scanned pulse must bundle pulse + preview + scan data together."""

    pytest.importorskip("PyQt5")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PyQt5 import QtWidgets
    from Zou_lab_control.frontend import devtools as dt

    editor = dt.demo_editor(size=(1200, 800))
    dt.settle(editor, 150)
    target = tmp_path / "mypulse.json"
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getSaveFileName",
        staticmethod(lambda *a, **k: (str(target), "ZLC pulse (*.json)")),
    )
    editor.save_to_file()
    names = {p.name for p in tmp_path.iterdir()}
    assert "mypulse.json" in names
    assert "mypulse.png" in names  # preview figure
    assert "mypulse_scan.npy" in names  # raw scan data
    # The compiled scan program is attempted too; if it cannot compile the status
    # surfaces it rather than failing silently.
    assert "Saved:" in editor.preview_status.text()


def test_fluent_combo_popup_fits_widest_item(monkeypatch):
    pytest.importorskip("PyQt5")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from Zou_lab_control.frontend.qt_fluent import (
        COMBO_WIDTH,
        EDIT_PADDING_H,
        FluentComboBox,
        ensure_qt_app,
        scaled_px,
    )

    app = ensure_qt_app()
    combo = FluentComboBox()
    combo.addItems(["Edge", "Ramp", "Hold"])
    combo.setFixedWidth(60)  # deliberately narrow, like the DAC mode combo
    combo.show()
    app.processEvents()
    combo.showPopup()
    app.processEvents()
    # The popup is sized to its widest item plus the dropdown + padding so options
    # are never clipped.  Verify that exact relationship (font-independent) rather
    # than a fixed pixel threshold that drifts with the ambient font.
    view = combo.view()
    metrics = view.fontMetrics()
    try:
        widest = max(metrics.horizontalAdvance(combo.itemText(i)) for i in range(combo.count()))
    except AttributeError:  # pragma: no cover - very old Qt
        widest = max(metrics.width(combo.itemText(i)) for i in range(combo.count()))
    assert view.minimumWidth() == widest + scaled_px(COMBO_WIDTH) + scaled_px(EDIT_PADDING_H) * 2
    assert view.minimumWidth() >= widest  # never clips the widest item
    combo.hidePopup()


def test_pulse_gui_constructs_xdc_channel_editor(monkeypatch, tmp_path):
    pytest.importorskip("PyQt5")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")

    from PyQt5 import QtCore, QtTest, QtWidgets

    from Zou_lab_control.frontend.pulse_gui import FigureCanvas, PulseSequenceEditor, ensure_qt_app

    app = ensure_qt_app()
    channels = [f"ch{i:02d}" for i in range(62)]

    class DummySequencer:
        clock_hz = 50e6
        trigger_channels = ["ch11"]

        def __init__(self, channels):
            self.channels = channels

    editor = PulseSequenceEditor(sequencer=DummySequencer(channels))
    try:
        editor.resize(1280, 720)
        editor.show()
        app.processEvents()

        assert len(editor.state.channels) == 62
        assert re.fullmatch(r"pulse_\d{8}_\d{6}", editor.state.name)
        assert editor.state.time_step_ns == 20
        assert editor.state.visible_channels == ["ch00", "ch01", "ch02", "ch03"]
        assert editor.names_panel.label_edits["ch00"].text() == ""
        assert "4/62 visible" in editor.summary.text()
        assert "step 20 ns" in editor.summary.text()
        assert editor.add_channel_combo.count() == 58
        assert not hasattr(editor.channel_panel, "x_edit")
        assert not hasattr(editor.channel_panel, "y_edit")
        assert not hasattr(editor.channel_panel, "scan_y_switch")
        assert not hasattr(editor.channel_panel, "scan_edit")
        # The single named-slot scan summary replaces the old Use-Y / Scan X UI.
        assert editor.channel_panel.scan_summary.metaObject().className() == "FluentLineEdit"
        assert editor.channel_panel.scan_summary.isEnabled() is False
        assert editor.channel_panel.scan_summary.text() == "no scan slots"
        assert editor.channel_panel.load_button.text() == "Load Array"
        first_card = editor.drag_container.pulse_cards()[0]
        assert first_card.checks["ch00"].text() == "ch00"
        assert editor.channel_panel.channel_labels["ch00"].text() == "ch00"
        assert hasattr(first_card, "unit_combo")
        # The period card has an editable NAME field (the index stays in the title).
        assert hasattr(first_card, "name_edit")
        assert first_card.name_edit.placeholderText() != ""
        assert first_card.duration_edit.geometry().right() <= first_card.width()
        assert first_card.unit_combo.geometry().right() <= first_card.width()
        assert editor.tabs.graphicsEffect() is not None
        assert editor.names_panel.graphicsEffect() is not None
        assert editor.channel_panel.graphicsEffect() is not None
        assert first_card.graphicsEffect() is not None
        assert editor.button_frame.metaObject().className() == "FluentFrame"

        def _has_ancestor(widget, frame):
            node = widget
            while node is not None:
                if node is frame:
                    return True
                node = node.parent()
            return False

        def _group_box_ancestor(widget):
            node = widget.parent()
            while node is not None:
                if node.metaObject().className() == "FluentGroupBox":
                    return node
                node = node.parent()
            return None

        # The bottom bar holds the controls in two titled Fluent cards (Control /
        # Channels), consistent with the other panels' group-box-with-title style.
        assert _has_ancestor(editor.safe_button, editor.button_frame)
        assert _has_ancestor(editor.add_channel_combo, editor.button_frame)
        assert _has_ancestor(editor.hide_off_button, editor.button_frame)
        assert _has_ancestor(editor.show_all_button, editor.button_frame)
        control_box = _group_box_ancestor(editor.safe_button)
        channels_box = _group_box_ancestor(editor.add_channel_combo)
        assert control_box is not None and control_box.title() == "Control"
        assert channels_box is not None and channels_box.title() == "Channels"
        # The titled cards carry the Fluent shadow (the container itself is flat).
        assert control_box.graphicsEffect() is not None
        assert channels_box.graphicsEffect() is not None
        assert editor.load_button.text() == "Load"
        # The control bar is compact: every button is single-line (no wrapped
        # "Stop\nPulse" art) so the bar stays short and the editor area expands.
        for _button in (
            editor.safe_button, editor.fire_button, editor.remove_button, editor.add_button,
            editor.bracket_button, editor.collapse_button, editor.save_button, editor.load_button,
        ):
            assert "\n" not in _button.text(), _button.text()
        assert editor.names_panel_layout.contentsMargins().top() == editor.drag_container.layout_main.contentsMargins().top()
        assert editor.names_panel_layout.contentsMargins().top() > 0
        def y_in_editor(widget):
            return widget.mapTo(editor, QtCore.QPoint(0, 0)).y()

        def x_in_editor(widget):
            return widget.mapTo(editor, QtCore.QPoint(0, 0)).x()

        assert editor.channel_panel.top_labels["scan"].alignment() == QtCore.Qt.AlignCenter
        assert editor.channel_panel.top_labels["step"].alignment() == QtCore.Qt.AlignCenter
        assert x_in_editor(editor.channel_panel.top_labels["step"]) == x_in_editor(editor.channel_panel.top_labels["scan"])
        assert x_in_editor(editor.channel_panel.step_display) == x_in_editor(editor.channel_panel.scan_summary)
        assert abs(y_in_editor(editor.channel_panel.top_labels["scan"]) - y_in_editor(editor.channel_panel.scan_summary)) <= 1
        assert editor.channel_panel.top_labels["scan"].height() == editor.channel_panel.scan_summary.height()
        assert abs(y_in_editor(editor.channel_panel.top_labels["step"]) - y_in_editor(editor.channel_panel.step_display)) <= 1
        assert editor.channel_panel.top_labels["step"].height() == editor.channel_panel.step_display.height()
        # The clock step is read-only (fixed by the FPGA clock), shown as MHz · ns/tick.
        assert editor.channel_panel.step_display.isEnabled() is False
        assert "MHz" in editor.channel_panel.step_display.text()
        assert abs(y_in_editor(editor.names_panel.raw_label_widgets["ch00"]) - y_in_editor(editor.channel_panel.delay_edits["ch00"])) <= 1
        assert abs(y_in_editor(editor.channel_panel.delay_edits["ch00"]) - y_in_editor(first_card.checks["ch00"])) <= 1
        margins = editor.edit_tab.layout().contentsMargins()
        assert margins.left() > 0 and margins.top() > 0
        # Compact channel-view layout: a full-width combo above a left-to-right
        # row of three equal-height buttons (Add / Hide Off / Show All).
        row_buttons = [editor.add_channel_button, editor.hide_off_button, editor.show_all_button]
        assert {button.height() for button in row_buttons} == {editor.add_channel_button.height()}
        assert editor.add_channel_combo.height() == editor.add_channel_button.height()
        assert (
            editor.add_channel_button.geometry().x()
            < editor.hide_off_button.geometry().x()
            < editor.show_all_button.geometry().x()
        )
        assert editor.add_channel_combo.width() >= editor.show_all_button.width()
        assert editor.preview_include_off.metaObject().className() == "FluentSwitch"
        assert editor.preview_include_off.text() == "Show off rows"
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
            assert "62/62 plotted" in editor.preview_status.text()
            editor.preview_include_off.setChecked(False)
            QtTest.QTest.qWait(250)
            app.processEvents()
        editor.tabs.setCurrentWidget(editor.edit_tab)
        app.processEvents()

        editor.show_all_channels()
        app.processEvents()
        assert len(editor.state.visible_channels) == 62
        assert "62/62 visible" in editor.summary.text()
        assert editor.add_channel_combo.count() == 0
        assert not editor.add_channel_button.isEnabled()
        assert editor.dataset_scroll.verticalScrollBar().width() == editor.timeline_hbar.height()

        editor.hide_off_channels()
        app.processEvents()
        assert editor.state.visible_channels == ["ch00", "ch01", "ch02", "ch03"]

        editor.names_panel.label_edits["ch01"].setText("cooling_laser")
        editor.channel_panel.delay_edits["ch01"].setText("40")
        app.processEvents()
        assert editor.channel_panel.channel_labels["ch01"].text() == "cooling_laser"
        current_first_card = editor.drag_container.pulse_cards()[0]
        # Period-card checkboxes elide long labels (with the full name in the
        # tooltip + tracked in check_full_labels) so they never spill the card.
        assert current_first_card.check_full_labels["ch01"] == "cooling_laser"
        assert "cooling_laser" in current_first_card.checks["ch01"].toolTip()
        editor.clear_channel("ch01")
        app.processEvents()
        assert editor.state.visible_channels == ["ch00", "ch01", "ch02", "ch03"]
        assert all(not period.states[editor.state.channel_index("ch01")] for period in editor.state.periods)
        assert editor.state.channel_labels["ch01"] == "cooling_laser"
        assert editor.state.delays["ch01"] == "40"

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
            assert editor._preview_plot.repeat_bracket[2] == "×∞"
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
        # Saving a pulse bundles the preview figure next to the JSON; an explicit
        # Save Figure export then re-writes the same PNG.
        assert (tmp_path / "pulse_custom.png").exists()
        (tmp_path / "pulse_custom.png").unlink()
        editor.save_figure()
        assert (tmp_path / "pulse_custom.png").exists()

        sequence = editor.to_sequence()
        assert sequence.validate(clock_hz=50e6, channels=channels).ok
        screenshot = editor.grab_screenshot(tmp_path / "pulse_gui_address_switch.png")
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
    with pytest.raises(ValueError, match="emCCD"):
        launcher._resolve_trigger_channels(args, [f"ch{i:02d}" for i in range(62)])
    assert launcher._resolve_trigger_channels(args, [f"ch{i:02d}" for i in range(62)], {"ch11": "emCCD"}) == ["ch11"]
    assert launcher._build_parser().parse_args([]).clock_hz == 50_000_000


def test_pulse_gui_controls_call_attached_address_switch_sequencer(monkeypatch):
    pytest.importorskip("PyQt5")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")

    from Zou_lab_control.frontend.pulse_gui import PulseSequenceEditor, ensure_qt_app

    app = ensure_qt_app()
    channels = [f"ch{i:02d}" for i in range(62)]

    class RecordingSequencer:
        clock_hz = 50e6
        trigger_channels = ["ch11"]

        def __init__(self):
            self.channels = channels
            self.prepared = None
            self.events = []

        def prepare(self, sequence):
            self.prepared = sequence
            self.events.append(("prepare", sequence.name, list(sequence.channels)))
            return na.compile_runtime_program_for_payload(sequence, channels=self.channels, clock_hz=self.clock_hz, trigger_channels=self.trigger_channels)

        def fire(self):
            self.events.append(("fire",))

        def wait_done(self, timeout=None):
            self.events.append(("wait_done", timeout))
            return True

        def set_safe_state(self):
            self.events.append(("safe",))

    sequencer = RecordingSequencer()
    editor = PulseSequenceEditor(
        state=na.PulseTableState.load(Path(__file__).resolve().parents[1] / "pulses" / "camera_imaging_address_switch.json"),
        sequencer=sequencer,
        channel_pins={"ch00": "F15", "ch11": "M13"},
    )
    try:
        editor.show()
        app.processEvents()

        # INIT: nothing applied yet -> confocal star ("pressing would apply").
        assert editor.fire_button.text() == "On Pulse*"
        assert editor.safe_button.text() == "Stop Pulse"
        assert editor.names_panel.raw_label_widgets["ch00"].text() == "F15"
        assert editor.names_panel.raw_label_widgets["ch11"].text() == "M13"
        assert not hasattr(editor, "prepare_button")
        assert not hasattr(editor, "wait_button")
        assert not hasattr(editor, "repeat_forever_switch")
        assert not hasattr(editor, "wait_done")
        editor.fire()
        editor.safe_state()
        app.processEvents()

        assert sequencer.events[0][0] == "prepare"
        assert [event[0] for event in sequencer.events] == ["prepare", "fire", "safe"]
        assert editor.last_program.channels == channels
        assert editor.last_program.trigger_count == 1
        assert isinstance(sequencer.prepared, na.PulseTableState)
        assert sequencer.prepared.to_sequence(expand_repeat=False).validate(clock_hz=50e6, channels=channels).ok
        assert editor.last_program.repeat_forever is True
    finally:
        editor.close()


def test_pulse_gui_repeat_preview_uses_unexpanded_periods(monkeypatch):
    pytest.importorskip("PyQt5")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")

    from PyQt5 import QtCore, QtWidgets

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
        first_card = editor.drag_container.pulse_cards()[0]
        spin_y = bracket_end.repeat_spin.mapTo(editor, QtCore.QPoint(0, 0)).y()
        duration_y = first_card.duration_edit.mapTo(editor, QtCore.QPoint(0, 0)).y()
        assert abs(spin_y - duration_y) <= 1
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
        assert [label.get_text() for label in editor._preview_plot.repeat_bracket_labels] == ["×∞", "×2"]
        repeat_label_sizes = [label.get_fontsize() for label in editor._preview_plot.repeat_bracket_labels]
        assert max(repeat_label_sizes) == min(repeat_label_sizes)
        assert editor._preview_plot.repeat_bracket_labels[1].get_position()[1] < editor._preview_plot.repeat_bracket_labels[0].get_position()[1]
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


def test_pulse_gui_preview_refresh_skips_rebuild_when_unchanged(monkeypatch):
    """Revisiting the Preview tab with an UNCHANGED state must not rebuild the
    ~130 ms figure+canvas -- but must look exactly like a rebuild: the view
    returns to the home zoom and any selection rectangle is cleared.  A real
    edit or the Show-off-rows toggle still rebuilds."""

    pytest.importorskip("PyQt5")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from Zou_lab_control.frontend import devtools as dt
    from Zou_lab_control.frontend.pulse_gui import FigureCanvas

    if FigureCanvas is None:
        pytest.skip("Matplotlib Qt canvas is unavailable")

    editor = dt.demo_editor(size=(1480, 900))
    try:
        editor.tabs.setCurrentWidget(editor.preview_tab)
        dt.settle(editor, 200)
        canvas = editor._preview_canvas
        assert canvas is not None

        # unchanged state -> the canvas object survives (no rebuild)
        editor.refresh_preview()
        assert editor._preview_canvas is canvas

        # ...but a zoomed view returns to home (identical to the old rebuild)
        zoom = editor._preview_plot.tools.zoom
        home = tuple(zoom.ax.get_xlim())
        zoom.ax.set_xlim(home[0], home[1] * 0.5)
        editor.refresh_preview()
        assert tuple(zoom.ax.get_xlim()) == home
        assert editor._preview_canvas is canvas

        # ...and an auxiliary status message (Save-figure / sync / scan-load) left
        # on the shared status label must NOT linger across an unchanged re-entry;
        # the skip-path restores the "N/M plotted" line a rebuild would have shown.
        plotted = editor.preview_status.text()
        assert "plotted" in plotted
        editor.preview_status.setText("Saved figure: foo.png")
        editor.refresh_preview()
        assert editor._preview_canvas is canvas          # still skipped (no rebuild)
        assert editor.preview_status.text() == plotted    # status restored

        # a real edit rebuilds
        editor.drag_container.pulse_cards()[0].duration_edit.setText("1234567")
        dt.settle(editor, 50)
        editor.refresh_preview()
        assert editor._preview_canvas is not canvas

        # the Show-off-rows toggle rebuilds too
        canvas = editor._preview_canvas
        editor.preview_include_off.setChecked(True)
        dt.settle(editor, 400)
        assert editor._preview_canvas is not canvas
    finally:
        editor.close()


def test_pulse_gui_shadows_pixel_match_stock_effect(monkeypatch):
    """CachedDropShadow must render the editor pixel-identical to the stock
    QGraphicsDropShadowEffect (the look is mandatory; only the implementation
    may differ).

    Guards two real regressions the user caught on screen:
    * drawSource()-based effects silently break NESTED effects (every card
      shadow inside the shadowed tab widget vanished);
    * a rounded-rect silhouette bake painted a white band over the transparent
      strip right of the tabs (the tab widget is not an opaque rounded rect).
    """

    pytest.importorskip("PyQt5")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")

    from PyQt5 import QtGui, QtWidgets

    from Zou_lab_control.frontend import devtools as dt
    from Zou_lab_control.frontend import qt_fluent as qf

    def grab_editor() -> np.ndarray:
        editor = dt.demo_editor(size=(1480, 900))
        dt.settle(editor)
        image = editor.grab().toImage().convertToFormat(QtGui.QImage.Format_ARGB32)
        w, h = image.width(), image.height()
        ptr = image.bits()
        ptr.setsize(h * image.bytesPerLine())
        arr = np.frombuffer(ptr, np.uint8).reshape(h, image.bytesPerLine() // 4, 4)[:, :w, :3].copy()
        editor.close()
        return arr.astype(int)

    cached = grab_editor()

    def add_stock_shadow(widget, *, blur=20, alpha=50, offset=0, **_ignored):
        effect = QtWidgets.QGraphicsDropShadowEffect(widget)
        effect.setBlurRadius(qf.scaled_px(blur))
        effect.setColor(QtGui.QColor(0, 0, 0, alpha))
        effect.setOffset(0, qf.scaled_px(offset, minimum=0))
        widget.setGraphicsEffect(effect)

    monkeypatch.setattr(qf, "add_fluent_shadow", add_stock_shadow)
    stock = grab_editor()

    diff = np.abs(stock - cached).max(axis=2)
    # measured: 3 px differ by <= 12 (a sub-pixel at the tab widget's square NW
    # corner); shadows missing or a white band would differ by thousands of px.
    assert int((diff > 4).sum()) < 200, (int(diff.max()), int((diff > 4).sum()))
    assert int(diff.max()) <= 40

    # ... and the shadows must actually exist (compare against no effect at all):
    monkeypatch.setattr(qf, "add_fluent_shadow", lambda widget, **kw: None)
    none = grab_editor()
    presence = np.abs(stock - none).max(axis=2)
    presence_cached = np.abs(cached - none).max(axis=2)
    assert int((presence > 10).sum()) > 10_000          # stock shadows touch many px
    assert int((presence_cached > 10).sum()) > 10_000   # ours must too


def test_pulse_gui_preview_wheel_over_plot_never_scrolls_page(monkeypatch):
    """A wheel over the pulse plot must zoom the plot ONLY -- the preview page
    must not scroll underneath it, no matter how Qt delivers the wheel.

    Guards the user report: "在preview的pulse plot区滚滚轮，整个preview页面的
    scroll还是会响应".  Accepting on the canvas is NOT enough: Qt can deliver
    the wheel straight to the scroll-area viewport, which is why the editor
    installs a viewport event filter.  This test injects the wheel on the
    VIEWPORT (the previously-broken delivery path) and on the canvas, and also
    checks that off-canvas wheels still scroll the page.
    """

    pytest.importorskip("PyQt5")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")

    from PyQt5 import QtCore, QtGui, QtWidgets

    from Zou_lab_control.frontend import devtools as dt
    from Zou_lab_control.frontend.pulse_gui import FigureCanvas

    if FigureCanvas is None:
        pytest.skip("Matplotlib Qt canvas is unavailable")

    editor = dt.demo_editor(size=(1200, 800))
    try:
        editor.tabs.setCurrentWidget(editor.preview_tab)
        dt.settle(editor)
        canvas = editor._preview_canvas
        assert canvas is not None
        scroll = editor.preview_scroll
        vp = scroll.viewport()
        vbar = scroll.verticalScrollBar()
        # Make the page genuinely scrollable (same situation as a many-channel
        # preview taller than the window).
        editor.preview_body.setFixedHeight(vp.height() + 600)
        dt.settle(editor)
        assert vbar.maximum() > 0

        def wheel(target, pos_local, delta=-120):
            event = QtGui.QWheelEvent(
                QtCore.QPointF(pos_local),
                QtCore.QPointF(target.mapToGlobal(pos_local)),
                QtCore.QPoint(0, 0), QtCore.QPoint(0, delta),
                QtCore.Qt.NoButton, QtCore.Qt.NoModifier,
                QtCore.Qt.NoScrollPhase, False)
            QtWidgets.QApplication.sendEvent(target, event)

        # 1) the broken path: wheel delivered to the viewport at a point over
        #    the canvas -> must be consumed (page does not move)...
        over_canvas = vp.mapFromGlobal(canvas.mapToGlobal(canvas.rect().center()))
        vbar.setValue(0)
        ax = canvas.figure.axes[0]
        xlim_before = ax.get_xlim()
        wheel(vp, over_canvas, delta=120)   # zoom IN (zoom-out is clamped at full view)
        dt.settle(editor)
        assert vbar.value() == 0
        # ...and the plot itself must have responded (zoom changed the x-range).
        assert ax.get_xlim() != xlim_before

        # 2) wheel delivered directly to the canvas -> also no page scroll.
        vbar.setValue(0)
        wheel(canvas, canvas.rect().center())
        dt.settle(editor)
        assert vbar.value() == 0

        # 3) wheel on the viewport OFF the canvas -> normal page scrolling.
        canvas_in_vp = QtCore.QRect(
            vp.mapFromGlobal(canvas.mapToGlobal(canvas.rect().topLeft())), canvas.size())
        off_canvas = QtCore.QPoint(vp.width() - 4, vp.height() - 4)
        if not canvas_in_vp.contains(off_canvas):
            vbar.setValue(0)
            wheel(vp, off_canvas)
            dt.settle(editor)
            assert vbar.value() > 0
    finally:
        editor.close()


def test_pulse_gui_scan_array_toggle_validates_and_marks_symbolic_regions(monkeypatch):
    pytest.importorskip("PyQt5")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")

    from PyQt5 import QtCore

    from Zou_lab_control.frontend.pulse_gui import PulseSequenceEditor, ensure_qt_app

    app = ensure_qt_app()
    state = na.PulseTableState(
        channels=["ch00", "ch03"],
        periods=[
            na.PulsePeriod(20, (1, 0), unit="ns", name="load"),
            na.PulsePeriod(80, (0, 1), unit="ns", name="trigger"),
        ],
        time_step_ns=20,
        visible_channels=["ch00", "ch03"],
        channel_labels={"ch00": "trap", "ch03": "trig"},
    )
    editor = PulseSequenceEditor(state=state, scale=0.86)
    try:
        editor.show()
        app.processEvents()

        # The old x/y scan widgets are gone; scanning is now per-field named slots.
        assert not hasattr(editor.channel_panel, "x_edit")
        assert not hasattr(editor.channel_panel, "y_edit")
        assert not hasattr(editor.channel_panel, "scan_y_switch")
        assert not hasattr(editor.channel_panel, "scan_edit")
        assert editor.channel_panel.scan_summary.text() == "no scan slots"
        assert editor.read_state().scan_slots == []

        # Bind the first period's duration to a scan slot via its dot.
        first_card = editor.drag_container.pulse_cards()[0]
        editor._toggle_duration_scan(first_card)
        app.processEvents()
        assert editor.state.slot_index_for("duration", "0") == 0
        assert editor.drag_container.pulse_cards()[0].duration_dot.isChecked()

        # A one-slot scan table feeds the seamless hardware scan compiler.
        editor.state.set_scan_table([[20.0], [40.0]])
        editor.load_state(editor.state)
        app.processEvents()
        assert editor.channel_panel.scan_summary.text() == "1 slot · 2 pts"
        scan_program = editor.read_state().compile_scan(clock_hz=50_000_000, trigger_channels=["ch03"])
        assert scan_program.scan_enabled is True
        assert scan_program.slot_count == 1
        assert scan_program.scan_points == [[1], [2]]

        # Bind a second slot (the second period's duration) so the preview shades
        # two regions.  (A per-channel delay is a fixed value and is not scannable.)
        editor._toggle_duration_scan(editor.drag_container.pulse_cards()[1])
        app.processEvents()
        assert [slot.kind for slot in editor.state.scan_slots] == ["duration", "duration"]
        assert editor.state.slot_index_for("duration", "1") == 1
        assert editor.drag_container.pulse_cards()[1].duration_dot.isChecked()
        editor.state.set_scan_table([[20.0, 80.0], [40.0, 120.0]])
        editor.load_state(editor.state)
        app.processEvents()
        assert editor.channel_panel.scan_summary.text() == "2 slots · 2 pts"
        restored = editor.read_state()
        assert restored.slot_count == 2
        assert restored.scan_table == [[20.0, 80.0], [40.0, 120.0]]
        assert restored.n_points == 2

        # The preview shades each scanned span with the slot's 1-based number.
        plotter, _channels, _repeat = editor._create_preview_plot(restored, include_always_off=True)
        labels = [text.get_text() for text in plotter.variable_region_labels]
        assert "1" in labels
        assert "2" in labels
        assert len(plotter.variable_region_artists) >= 2
        for patch in plotter.variable_region_artists:
            y_values = [round(float(point[1]), 6) for point in patch.get_path().vertices]
            assert min(y_values) == 0.0
            assert max(y_values) == 1.0
        if hasattr(plotter, "fig") and plotter.fig is not None:
            plotter.fig.canvas.draw()

        # Toggling a dot off unbinds its slot and renumbers the rest.
        editor._toggle_duration_scan(editor.drag_container.pulse_cards()[0])
        app.processEvents()
        assert [slot.kind for slot in editor.state.scan_slots] == ["duration"]
        assert editor.state.slot_index_for("duration", "1") == 0
    finally:
        editor.close()


def test_pulse_gui_analog_bus_uses_line_edit_and_hollow_preview(monkeypatch):
    pytest.importorskip("PyQt5")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")

    from Zou_lab_control.frontend.pulse_gui import PulseSequenceEditor, ensure_qt_app

    app = ensure_qt_app()
    state = na.PulseTableState(
        channels=["ch00", "ch01", "ch02"],
        periods=[
            na.PulsePeriod(20, (0, 0, 0), unit="ns"),
            na.PulsePeriod(20, (0, 0, 0), unit="ns"),
        ],
        visible_channels=["ch00", "ch01", "ch02"],
        channel_labels={"ch00": "da_test[0]", "ch01": "da_test[1]", "ch02": "da_test[2]"},
        analog_buses={"da_test": ["ch00", "ch01", "ch02"]},
        time_step_ns=20,
    )
    state.set_analog_bus_mode(0, "da_test", "edge", value=0)
    state.set_analog_bus_mode(1, "da_test", "ramp", value=3)   # 3-bit bus: signed -4..+3
    state.apply_analog_bus_modes_to_period_states()
    editor = PulseSequenceEditor(state=state, scale=0.86)
    try:
        editor.show()
        app.processEvents()
        first_card, second_card = editor.drag_container.pulse_cards()

        # The DAC value is a (scan-dot) line edit, not a spinbox; FluentScanLineEdit
        # is a FluentLineEdit subclass with an embedded scan toggle.
        assert first_card.bus_value_edits["da_test"].metaObject().className() in ("FluentLineEdit", "FluentScanLineEdit")
        assert not hasattr(first_card, "bus_spins")
        second_card.bus_value_edits["da_test"].setText("2048")
        restored = editor.read_state()
        assert editor.drag_container.pulse_cards()[1].bus_value_edits["da_test"].text() == "3"
        assert restored.analog_bus_modes["da_test"][1] == {"mode": "ramp", "value": 3}
        editor.add_period()
        app.processEvents()
        added = editor.read_state()
        assert len(added.periods) == 3
        assert len(added.analog_bus_modes["da_test"]) == 3
        assert added.analog_bus_modes["da_test"][2] == {"mode": "hold", "value": None}
        editor.remove_period()
        app.processEvents()
        removed = editor.read_state()
        assert len(removed.periods) == 2
        assert len(removed.analog_bus_modes["da_test"]) == 2

        plotter, channels, _repeat = editor._create_preview_plot(restored, include_always_off=True)
        assert channels == []
        assert len(plotter.analog_traces) == 1
        # One solid value-following line per analog row, matching the digital
        # channel lines' weight + opacity (0.65 / 1.0).  A separate dashed,
        # half-transparent "0" reference baseline is drawn too but is NOT counted
        # as a value trace.
        assert len(plotter.analog_trace_artists) == 1
        value_line = plotter.analog_trace_artists[0]
        assert value_line.get_linewidth() == 0.65
        assert value_line.get_alpha() == 1.0
        assert value_line.get_linestyle() in ("solid", "-")
        # The dashed reference baseline shares the trace colour but is dashed +
        # more transparent.
        baseline_dashes = [
            ln for ln in plotter.ax.get_lines()
            if ln is not value_line and ln.get_alpha() == 0.5 and ln.get_linestyle() != "solid"
            and tuple(ln.get_color()) == tuple(value_line.get_color())
        ]
        assert baseline_dashes, "expected a dashed reference baseline for the analog row"
        assert plotter._analog_baseline_y  # plotter exposes analog row geometry for annotations
        assert plotter.analog_trace_labels == []
        labels = plotter.ax.get_yticklabels()
        assert [tick.get_text() for tick in labels] == ["da test"]
        # The analog row label is tinted to its trace colour, like digital rows.
        assert tuple(labels[0].get_color()) == tuple(value_line.get_color())
    finally:
        editor.close()


def test_pulse_gui_summary_warns_about_repeat_forever_table_boundary_high(monkeypatch):
    pytest.importorskip("PyQt5")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")

    from Zou_lab_control.frontend.pulse_gui import PulseSequenceEditor, ensure_qt_app

    app = ensure_qt_app()
    state = na.PulseTableState(
        channels=["ch09", "ch06"],
        periods=[
            na.PulsePeriod(100, (1, 0), unit="ns"),
            na.PulsePeriod(20, (0, 1), unit="ns"),
            na.PulsePeriod(1000, (0, 0), unit="ns"),
        ],
        repeat_start=1,
        repeat_end=1,
        repeat_count=5,
        repeat_forever=True,
        channel_labels={"ch09": "trap", "ch06": "trig"},
        visible_channels=["ch09", "ch06"],
        time_step_ns=20,
    )
    editor = PulseSequenceEditor(state=state, scale=0.86)
    try:
        editor.show()
        app.processEvents()
        editor._update_summary()

        assert "repeat ∞" in editor.summary.text()
        assert "table restart high every 1.2 us: trap" in editor.summary.text()
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
    pytest.importorskip("PyQt5")   # qt_fluent imports PyQt5 at module top
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


def test_preview_hides_idle_dac_when_off_rows_off(monkeypatch):
    """#5: with 'Show off rows' OFF an idle (all-zero) DAC bus is hidden, just like an
    always-off TTL channel; a DAC carrying a real value (or scanned) still shows."""

    pytest.importorskip("PyQt5")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from Zou_lab_control.frontend.qt_fluent import ensure_qt_app
    from Zou_lab_control.frontend import pulse_gui
    from Zou_lab_control.neutral_atom.timing.pulse_table import PulsePeriod, PulseTableState

    ensure_qt_app()
    ch = [f"da[{i}]" for i in range(10)] + ["trig"]
    labels = {f"da[{i}]": f"da[{i}]" for i in range(10)}
    state = PulseTableState(
        channels=ch,
        visible_channels=ch,
        periods=[PulsePeriod(1000, tuple([0] * 11), unit="ns")],  # DAC = 0 in every period (idle)
        channel_labels=labels,
        time_step_ns=20.0,
    )
    off, _ = pulse_gui._analog_bus_traces(state, include_always_off=False)
    on, _ = pulse_gui._analog_bus_traces(state, include_always_off=True)
    assert off == []          # idle DAC hidden when off-rows are hidden
    assert len(on) == 1       # but shown when off-rows are shown

    state.set_bus_value(0, "da", 200)   # now it carries a real value
    off2, _ = pulse_gui._analog_bus_traces(state, include_always_off=False)
    assert len(off2) == 1     # an active DAC is shown even with off-rows hidden


def test_scan_source_toggle_switches_generated_and_loaded(monkeypatch):
    """#4: the Delay/Scan toggle picks the active scan table -- off = generated (Run),
    on = the array loaded from a file -- without re-running anything."""

    pytest.importorskip("PyQt5")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from Zou_lab_control.frontend import devtools as dt

    ed = dt.demo_editor(size=(1400, 880))
    dt.settle(ed, 200)
    ed._scan_tables["generated"] = [[10000.0, 100.0], [20000.0, 200.0]]
    ed._scan_tables["loaded"] = [[30000.0, 300.0]]
    ed._scan_use_loaded = False
    ed._apply_scan_source()
    assert len(ed.state.scan_table) == 2          # generated source active

    ed.channel_panel.scan_source_toggle.setChecked(True)   # -> loaded
    dt.settle(ed, 100)
    assert ed.channel_panel.scan_source_toggle.isChecked()
    assert len(ed.state.scan_table) == 1

    ed.channel_panel.scan_source_toggle.setChecked(False)  # -> generated
    dt.settle(ed, 100)
    assert len(ed.state.scan_table) == 2


def test_scan_default_template_adapts_to_slot_count(monkeypatch):
    """#6: the default Scan-tab code is the column_stack template, regenerated to match
    the bound slot count until the user edits it."""

    pytest.importorskip("PyQt5")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from Zou_lab_control.frontend import devtools as dt

    ed = dt.demo_editor(size=(1400, 880))   # binds 2 scan slots (s0 duration, s1 dac)
    dt.settle(ed, 150)
    ed._refresh_scan_tab()
    code = ed.scan_code.toPlainText()
    assert "column_stack([s0, s1])" in code      # adapted to the 2 bound slots
    assert "2 bound slot(s)" in code


def test_delay_edit_caps_at_delay_depth(monkeypatch):
    """#1: a delay larger than the delay-line depth is clamped on the field."""

    pytest.importorskip("PyQt5")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from Zou_lab_control.frontend import devtools as dt
    from Zou_lab_control.neutral_atom.timing.pulse_table import DELAY_DEPTH_TICKS

    ed = dt.demo_editor(size=(1400, 880))
    dt.settle(ed, 120)
    panel = ed.channel_panel
    channel = next(iter(panel.delay_edits))
    edit = panel.delay_edits[channel]
    panel.delay_units[channel].setCurrentText("us")
    edit.setText("999")                  # 999 us: inside the new ~42.9 s TTL bound -> UNCHANGED
    panel._clamp_delay_edit(channel, edit)
    assert abs(float(edit.text()) - 999.0) < 1e-6
    edit.setText(str(10 ** 9))           # 1e9 us = 1000 s: beyond the 32-bit field -> clamped
    panel._clamp_delay_edit(channel, edit)
    capped_us = DELAY_DEPTH_TICKS * ed.state.time_step_ns / 1000.0
    assert abs(float(edit.text()) - capped_us) <= max(1e-6, capped_us * 1e-9)


def test_delay_unit_combo_offers_ns_us_ms_s(monkeypatch):
    """The per-channel delay unit combo offers ns/us/ms/s -- the TTL delay is now a true
    physical delay bounded by the 32-bit field (~42.9 s), not the old ~41 us ring, so
    ms/s are valid. ('str (ns)' stays excluded: a fixed delay is not scannable.)"""

    pytest.importorskip("PyQt5")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from Zou_lab_control.frontend import devtools as dt

    ed = dt.demo_editor(size=(1400, 880))
    dt.settle(ed, 120)
    for combo in ed.channel_panel.delay_units.values():
        items = [combo.itemText(i) for i in range(combo.count())]
        assert items == ["ns", "us", "ms", "s"], items


def test_dac_value_field_and_dot_stay_inside_card(monkeypatch):
    """The DAC value field (and its embedded scan dot) must never spill past the period
    card's right border -- the '右边缘 cutoff' the user reported.  Sizing the value field
    to the card's remaining width guarantees this regardless of the rendering font."""

    pytest.importorskip("PyQt5")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from Zou_lab_control.frontend import devtools as dt

    ed = dt.demo_editor(size=(1480, 900))
    st = ed.state
    st.set_bus_value(0, "da_dipole", -512)   # widest signed value: the widest content
    ed.load_state(st)
    dt.settle(ed, 200)
    card = ed.drag_container.pulse_cards()[0]
    value_edit = list(card.bus_value_edits.values())[0]
    dot = value_edit.dot
    card_right = card.width()
    # both the value field and the dot end at or before the card's right edge
    assert value_edit.mapTo(card, value_edit.rect().topRight()).x() <= card_right
    assert dot.mapTo(card, dot.rect().topRight()).x() <= card_right


def test_clk_button_marks_channel_disables_delay_and_hides_from_preview(monkeypatch):
    """The per-channel 'clk' toggle: pressing it wires the channel to the FPGA clk -> it
    enters state.clk_channels, its delay/unit fields grey out, and it drops from the
    preview (it is no longer engine-driven).  Pressing again restores it."""

    pytest.importorskip("PyQt5")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from Zou_lab_control.frontend import devtools as dt

    ed = dt.demo_editor(size=(1500, 920))
    dt.settle(ed, 200)
    ch = "ch00"
    assert ch in ed.channel_panel.clk_buttons
    assert ed.channel_panel.delay_edits[ch].isEnabled()
    ed.channel_panel.clk_buttons[ch].click()
    dt.settle(ed, 150)
    assert ed.state.clk_channels == [ch]
    assert not ed.channel_panel.delay_edits[ch].isEnabled()      # delay greyed out
    assert not ed.channel_panel.delay_units[ch].isEnabled()
    _plotter, channels, _repeat = ed._create_preview_plot(ed.read_state(), include_always_off=True)
    assert ch not in channels                                    # excluded from preview
    ed.channel_panel.clk_buttons[ch].click()                     # toggle off
    dt.settle(ed, 150)
    assert ed.state.clk_channels == []
    assert ed.channel_panel.delay_edits[ch].isEnabled()


def test_duration_unit_dropdown_excludes_str_ns_until_scanned(monkeypatch):
    """#4 residue: a normal duration's unit dropdown is ns/us/ms/s only; the internal
    'str (ns)' expression unit appears only for a scan-bound duration (auto, disabled)."""

    pytest.importorskip("PyQt5")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from Zou_lab_control.frontend import devtools as dt

    ed = dt.demo_editor(size=(1480, 900))   # binds period-4 duration to s0
    dt.settle(ed, 150)
    cards = ed.drag_container.pulse_cards()
    unbound = cards[0].unit_combo
    assert [unbound.itemText(i) for i in range(unbound.count())] == ["ns", "us", "ms", "s"]
    bound = cards[3].unit_combo                      # period 4 is bound to s0
    assert bound.currentText() == "str (ns)" and not bound.isEnabled()


def test_clk_channel_disables_period_checkboxes(monkeypatch):
    """#2: marking a channel as clk locks (disables) its checkbox in every period card."""

    pytest.importorskip("PyQt5")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from Zou_lab_control.frontend import devtools as dt

    ed = dt.demo_editor(size=(1480, 900))
    dt.settle(ed, 120)
    ed._toggle_clk_channel("ch01")
    dt.settle(ed, 150)
    assert "ch01" in ed.state.clk_channels
    for card in ed.drag_container.pulse_cards():
        if "ch01" in card.checks:
            assert not card.checks["ch01"].isEnabled()


def test_floatorx_lineedit_residue_removed():
    """#4 residue: the dead x/y FloatOrXLineEdit (and its regexes) are gone."""

    pytest.importorskip("PyQt5")   # qt_fluent imports PyQt5 at module top
    from Zou_lab_control.frontend import qt_fluent as qf

    assert not hasattr(qf, "FloatOrXLineEdit")
    assert not hasattr(qf, "_FLOAT_OR_X_RE")
    assert not hasattr(qf, "_OLD_FLOAT_OR_X_RE")
    assert not hasattr(qf, "_VARIABLE_TOKEN_RE")


def test_bus_mode_combo_fits_ramp(monkeypatch):
    """#3: the Edge/Ramp/Hold combo is wide enough to show 'Ramp' without eliding."""

    pytest.importorskip("PyQt5")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PyQt5 import QtGui
    from Zou_lab_control.frontend import devtools as dt
    from Zou_lab_control.frontend.qt_fluent import COMBO_WIDTH, EDIT_PADDING_H, scaled_px

    ed = dt.demo_editor(size=(1480, 900))
    dt.settle(ed, 120)
    combo = next(iter(ed.drag_container.pulse_cards()[0].bus_mode_combos.values()))
    metrics = QtGui.QFontMetrics(QtGui.QFont(combo.font()))
    ramp_w = metrics.horizontalAdvance("Ramp")
    # paintEvent reserves drop arrow + insets (~drop + 2*pad + 2); the combo must leave at
    # least the "Ramp" text width after that reserve.
    reserve = scaled_px(COMBO_WIDTH) + scaled_px(EDIT_PADDING_H) * 2 + scaled_px(2)
    assert combo.width() - reserve >= ramp_w


def test_hold_field_tracks_upstream_edge_change(monkeypatch):
    """#1 (the user's explicit worry): a HOLD period shows the value carried in from the
    preceding edge/ramp, and that shown value UPDATES reactively when the upstream value
    is edited (via busChanged -> _refresh_bus_displays), with no full rebuild."""

    pytest.importorskip("PyQt5")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from Zou_lab_control.frontend import devtools as dt
    from Zou_lab_control.neutral_atom.timing.pulse_table import PulsePeriod, PulseTableState

    ch = [f"da[{i}]" for i in range(10)] + ["trig"]
    labels = {f"da[{i}]": f"da[{i}]" for i in range(10)}
    state = PulseTableState(
        channels=ch, visible_channels=ch,
        periods=[PulsePeriod(1000, tuple([0] * 11), unit="ns") for _ in range(3)],
        channel_labels=labels, time_step_ns=20.0,
    )
    state.set_analog_bus_mode(0, "da", "edge", value=100)
    state.set_analog_bus_mode(1, "da", "hold")
    state.set_analog_bus_mode(2, "da", "hold")

    ed = dt.demo_editor(size=(1200, 820))
    ed.load_state(state)
    dt.settle(ed, 150)
    cards = ed.drag_container.pulse_cards()
    # both hold periods initially show the carried 100
    assert cards[1].bus_value_edits["da"].text() == "100"
    assert cards[2].bus_value_edits["da"].text() == "100"
    # edit the upstream edge (period 0) to 500 and commit -> the reactive refresh runs
    cards[0].bus_value_edits["da"].setText("500")
    ed._refresh_bus_displays()
    dt.settle(ed, 60)
    # the downstream hold fields now track the new upstream value (no rebuild)
    assert cards[1].bus_value_edits["da"].text() == "500"
    assert cards[2].bus_value_edits["da"].text() == "500"


def test_pulse_gui_clear_bus_clears_analog_bus_modes(monkeypatch):
    """BUG 1.3: clearing a DAC bus only cleared the member TTL bits; analog_bus_modes
    re-projected the stale edge/ramp value back.  Clearing must zero the LOGICAL bus too."""

    pytest.importorskip("PyQt5")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from Zou_lab_control.frontend import devtools as dt
    from Zou_lab_control.frontend.pulse_gui import PulseSequenceEditor
    from Zou_lab_control.frontend.qt_fluent import ensure_qt_app

    app = ensure_qt_app()
    editor = PulseSequenceEditor(state=dt.demo_state())
    editor.show(); editor.show_all_channels(); app.processEvents()

    bus = list(editor.state.bus_channels())[0]
    card0 = editor.drag_container.pulse_cards()[0]
    card0.bus_mode_combos[bus].setCurrentText("Edge")
    card0.bus_value_edits[bus].setText("500")
    app.processEvents()
    assert editor.read_state().analog_bus_modes[bus][0]["mode"] == "edge"

    editor.clear_channel(f"bus:{bus}")
    app.processEvents()
    after = editor.read_state()
    assert all(str(e.get("mode", "hold")).lower() == "hold" for e in after.analog_bus_modes.get(bus, []))
    assert after.bus_value(0, bus) == 0   # the clear actually sticks now


def test_pulse_gui_dac_scan_target_follows_period_reorder(monkeypatch):
    """BUG 1.2: _reconcile_scan_slots only remapped duration targets; a DAC slot's
    bus@period_index went stale when periods were reordered.  It must follow the move."""

    pytest.importorskip("PyQt5")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from Zou_lab_control.frontend import devtools as dt
    from Zou_lab_control.frontend.pulse_gui import PulseSequenceEditor, PeriodCard
    from Zou_lab_control.frontend.qt_fluent import ensure_qt_app

    app = ensure_qt_app()
    editor = PulseSequenceEditor(state=dt.demo_state())
    editor.show(); editor.show_all_channels(); app.processEvents()
    if editor.state.repeat_start is not None:
        pytest.skip("reorder test assumes no repeat bracket")

    bus = list(editor.state.bus_channels())[0]
    cards = editor.drag_container.pulse_cards()
    cards[1].bus_mode_combos[bus].setCurrentText("Edge")
    cards[1].bus_value_edits[bus].setText("300")
    app.processEvents()
    editor._toggle_dac_scan(cards[1], bus); app.processEvents()
    dac_slot = next(s for s in editor.read_state().scan_slots if s.kind == "dac")
    assert dac_slot.target == f"{bus}@1"

    # Move the period-1 card to index 0 (reorder the drag items, then refresh + read).
    dc = editor.drag_container
    moved = next(it for it in dc.items if it.widget is cards[1])
    dc.items.remove(moved)
    dc.items.insert(0, moved)
    dc.refresh_layout()
    app.processEvents()
    dac_slot2 = next(s for s in editor.read_state().scan_slots if s.kind == "dac")
    assert dac_slot2.target == f"{bus}@0"   # target followed the reorder


def test_pulse_gui_period_name_edit_round_trips(monkeypatch):
    """#1 (period names): each card has an editable name field; the typed name lands in
    read_state().periods[i].name, survives load_state, and the index title stays."""

    pytest.importorskip("PyQt5")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from Zou_lab_control.frontend import devtools as dt

    editor = dt.demo_editor(size=(1400, 860), bind_scans=False)
    cards = editor.drag_container.pulse_cards()
    assert cards[0].title().startswith("Period 1/")        # the i/N index display stays
    cards[0].name_edit.setText("load")
    cards[2].name_edit.setText("image")
    state = editor.read_state()
    assert state.periods[0].name == "load"
    assert state.periods[2].name == "image"
    assert state.periods[1].name == ""
    editor.load_state(state)                               # rebuild restores the names
    cards2 = editor.drag_container.pulse_cards()
    assert cards2[0].name_edit.text() == "load"
    assert cards2[2].name_edit.text() == "image"
    assert cards2[0].title().startswith("Period 1/")
