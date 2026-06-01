"""Small shared utilities for neutral-atom session objects."""

from __future__ import annotations

from html import escape
from typing import Any

import numpy as np


def site_index(value, n_sites: int) -> int:
    """Validate and normalize one site index."""

    if isinstance(value, (bool, np.bool_)):
        raise ValueError("site must be an integer, not a boolean.")
    numeric = float(value)
    if not np.isfinite(numeric) or int(numeric) != numeric or numeric < 0 or numeric >= n_sites:
        raise ValueError(f"site must be in [0, {n_sites}).")
    return int(numeric)


def json_ready(value):
    """Convert numpy-heavy status payloads into JSON-serializable objects."""

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    return value


def html_summary(title: str, rows: dict[str, Any]) -> str:
    """Compact HTML representation used by notebooks and future GUI logs."""

    items = "".join(
        f"<tr><th style='text-align:left;padding:2px 10px 2px 0'>{escape(str(key))}</th>"
        f"<td style='padding:2px 0'>{escape(str(value))}</td></tr>"
        for key, value in rows.items()
    )
    return (
        "<div style='border-left:3px solid #238b8d;padding:6px 10px;margin:4px 0'>"
        f"<b>{escape(title)}</b><table style='margin-top:4px'>{items}</table></div>"
    )


__all__ = ["html_summary", "json_ready", "site_index"]
