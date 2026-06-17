"""Rebuild the Task-console design review PDF.

Regenerates the three screenshots into ``assets/`` and compiles
``task_console_design_zh.texbody`` through the frontend's sealed PDF interface
(``render_notes_pdf`` -> temp-dir XeLaTeX), leaving ONLY the ``.pdf`` here.

    python docs/task_console_design/build.py
"""

from __future__ import annotations

import os
import sys
from datetime import date
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.fonts=false")

HERE = Path(__file__).resolve().parent
# run-from-anywhere: put the repo root (docs/task_console_design -> docs -> root)
# on the path so ``Zou_lab_control`` imports without an editable install.
sys.path.insert(0, str(HERE.parents[1]))
ASSETS = HERE / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)


def _generate_figures() -> None:
    import warnings
    warnings.filterwarnings("ignore")
    from Zou_lab_control.frontend.devtools import install_screenshot_font, settle, screenshot
    install_screenshot_font()
    from Zou_lab_control.frontend import devtools

    # 1. Dashboard (full window body: header + tabs + grid)
    console = devtools.demo_console(shots=40)
    settle(console, 900)
    console.refresh_once()
    settle(console, 600)
    screenshot(console, str(ASSETS / "dashboard.png"))

    # 2. Setting popup of the Site map panel
    sites = next(c for c in console.cards if c.config.kind == "sites")
    sites._open_settings()
    settle(sites.settings_popup, 500)
    sites.settings_popup.grab().save(str(ASSETS / "setting.png"))
    sites.settings_popup.hide()

    # 3. Panel Editor bound to the 1D panel, with a Gaussian command fit.
    # The fit controls live on the per-panel PanelEditor (registered under
    # id(card)), not on the console -- the Edit redesign moved them there.
    card = next(c for c in console.cards if c.config.kind == "1d")
    console._edit_card(card)
    settle(console, 400)
    editor = console._panel_editors[id(card)]
    editor.fit_combo.setCurrentText("Gaussian")
    editor.fit_cmd.setText("")
    editor.do_fit()
    settle(console, 400)
    screenshot(console, str(ASSETS / "editor.png"))
    console.shutdown()


def main() -> None:
    _generate_figures()
    from Zou_lab_control.frontend.notes import render_notes_pdf

    body = (HERE / "task_console_design_zh.texbody").read_text(encoding="utf-8")
    result = render_notes_pdf(
        HERE,
        filename="task_console_design_zh.tex",
        title="Task 控制台设计审查文档",
        subtitle="实验侧可配置仪表盘 / frontend 组件 / 扩展到真实设备",
        description="五层架构（device/measurement/processor/task/plot）+ SignalHub、面板几何与密封 API、Panel Editor、真机接线",
        body=body,
        doc_date=date.today().isoformat(),
        compile_pdf=True,
    )
    print(f"built {result.pdf_path}  (exists={result.pdf_path.exists()})")


if __name__ == "__main__":
    main()
