"""Optional Jupyter integration, kept out of the Matplotlib render owner."""

from __future__ import annotations


def use_widget_backend() -> None:
    """Switch an active IPython shell to the Jupyter Matplotlib widget backend."""

    try:
        from IPython import get_ipython
    except ImportError as exc:
        raise RuntimeError("IPython is not available.") from exc
    shell = get_ipython()
    if shell is None:
        raise RuntimeError("No active IPython shell is available.")
    shell.run_line_magic("matplotlib", "widget")


def enable_long_output() -> None:
    """Remove notebook output scroll boxes when IPython display is available."""

    try:
        from IPython.display import HTML, display
    except ImportError:
        return
    display(
        HTML(
            """
            <style>
            .output_scroll {
                height: auto !important;
                max-height: none !important;
            }
            </style>
            """
        )
    )


__all__ = ["enable_long_output", "use_widget_backend"]
