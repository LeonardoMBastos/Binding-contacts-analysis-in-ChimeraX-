"""DockAnalyzer molecular interaction analysis toolkit.

Importing :mod:`dockanalyzer` is intentionally lightweight and does not start
an analysis, load ChimeraX integrations, or create output directories.
"""

from __future__ import annotations

from typing import Any

from ._version import __version__

__all__ = ["__version__", "run_analysis"]


def run_analysis(*args: Any, **kwargs: Any) -> Any:
    """Run the DockAnalyzer pipeline using a lazily imported orchestrator."""

    from .analyze import run_analysis as _run_analysis

    return _run_analysis(*args, **kwargs)
