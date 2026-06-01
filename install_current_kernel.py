"""Install project requirements into the currently running Python kernel.

Use this from a VSCode/Jupyter notebook when the selected kernel works, but its
python.exe is not available from PowerShell:

    %run ../install_current_kernel.py
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
REQUIREMENTS = ROOT / "requirements.txt"


def run(*args: str) -> None:
    print("+", sys.executable, *args)
    subprocess.check_call([sys.executable, *args])


try:
    import pip  # noqa: F401
except Exception:
    run("-m", "ensurepip", "--upgrade")

run("-m", "pip", "install", "--upgrade", "pip")
run("-m", "pip", "install", "-r", str(REQUIREMENTS))
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

print("\nDone. Restart the notebook kernel, then choose: Python (Zou lab control)")
