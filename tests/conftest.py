from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
root_text = str(REPO_ROOT)
if sys.path[0] != root_text:
    sys.path.insert(0, root_text)


@pytest.fixture(autouse=True)
def close_matplotlib_figures():
    yield
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    plt.close("all")
