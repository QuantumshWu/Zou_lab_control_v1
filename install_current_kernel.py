"""Install this checkout into the currently running Python kernel.

Use this from a VSCode/Jupyter notebook when the selected kernel works, but its
python.exe is not available from PowerShell:

    %run ../install_current_kernel.py

It mirrors the important parts of ``install_requirements.bat``: requirements,
editable package install, Jupyter kernel registration, and ``.zlc_python_path``.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
REQUIREMENTS = ROOT / "requirements.txt"
PYTHON_PATH_RECORD = ROOT / ".zlc_python_path"


def run(*args: str) -> None:
    print("+", sys.executable, *args)
    subprocess.check_call([sys.executable, *args])


try:
    import pip  # noqa: F401
except Exception:
    run("-m", "ensurepip", "--upgrade")

run("-m", "pip", "install", "--upgrade", "pip")
run("-m", "pip", "install", "-r", str(REQUIREMENTS))
run("-m", "pip", "install", "-e", str(ROOT))
PYTHON_PATH_RECORD.write_text(sys.executable, encoding="utf-8")
run(
    "-m",
    "ipykernel",
    "install",
    "--user",
    "--name",
    "zou_lab_control",
    "--display-name",
    "Python (Zou lab control)",
)

print(f"\nRecorded Python path for launchers: {sys.executable}")
print("Done. Restart the notebook kernel, then choose: Python (Zou lab control)")
