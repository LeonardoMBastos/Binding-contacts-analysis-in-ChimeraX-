"""ChimeraX ``runscript`` launcher for DockAnalyzer.

Usage in the ChimeraX command line::

    runscript C:\\Users\\leoma\\Desktop\\dockanalyzer_runscript.py

The launcher must remain next to the ``dockanalyzer`` package directory.
ChimeraX supplies the active session as the global variable ``session``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


LAUNCHER_DIRECTORY = Path(__file__).resolve().parent
PACKAGE_DIRECTORY = LAUNCHER_DIRECTORY / "dockanalyzer"


def _prepare_package_import() -> None:
    """Make the adjacent DockAnalyzer package importable and validate its origin."""

    package_initializer = PACKAGE_DIRECTORY / "__init__.py"
    if not package_initializer.is_file():
        raise RuntimeError(
            "DockAnalyzer package not found. Keep dockanalyzer_runscript.py "
            "next to the dockanalyzer directory. Expected file: "
            f"{package_initializer}"
        )

    launcher_path = str(LAUNCHER_DIRECTORY)
    if launcher_path not in sys.path:
        sys.path.insert(0, launcher_path)

    import dockanalyzer

    loaded_initializer = Path(dockanalyzer.__file__).resolve()
    if loaded_initializer != package_initializer.resolve():
        raise RuntimeError(
            "A different DockAnalyzer copy is already loaded in this ChimeraX "
            f"session: {loaded_initializer}. Restart ChimeraX and run this "
            "launcher again."
        )


def main(active_session: Any) -> Any:
    """Run DockAnalyzer against the models open in one ChimeraX session."""

    if active_session is None:
        raise RuntimeError(
            "No active ChimeraX session was supplied. Execute this file with "
            "the ChimeraX 'runscript' command."
        )

    _prepare_package_import()

    from dockanalyzer.analyze import run_chimerax_autorun

    return run_chimerax_autorun(active_session)


_ACTIVE_SESSION = globals().get("session")
if _ACTIVE_SESSION is None:
    raise RuntimeError(
        "This launcher must be executed inside ChimeraX with the 'runscript' "
        "command; the global ChimeraX session is unavailable."
    )

DOCKANALYZER_LAST_RESULT = main(_ACTIVE_SESSION)
