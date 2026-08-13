"""ChimeraX ``runscript`` launcher for DockAnalyzer.

Usage in the ChimeraX command line::

    runscript C:\\Users\\leoma\\Desktop\\dockanalyzer_runscript_v1.py

The launcher must remain next to the ``dockanalyzer`` package directory.
ChimeraX supplies the active session as the global variable ``session``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable


LAUNCHER_DIRECTORY = Path(__file__).resolve().parent
PACKAGE_DIRECTORY = LAUNCHER_DIRECTORY / "dockanalyzer"


def _prepare_package_import() -> None:
    """Make the adjacent DockAnalyzer package importable and validate its origin."""

    package_initializer = PACKAGE_DIRECTORY / "__init__.py"
    if not package_initializer.is_file():
        raise RuntimeError(
            "DockAnalyzer package not found. Keep dockanalyzer_runscript_v1.py "
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


def _ask_yes_no(
    active_session: Any,
    *,
    title: str,
    message: str,
    default: bool = False,
) -> bool:
    """Ask one English yes/no question using GUI when available."""

    ui = getattr(active_session, "ui", None)
    if ui is not None and getattr(ui, "is_gui", False):
        try:
            from Qt.QtWidgets import QMessageBox  # type: ignore

            parent = getattr(ui, "main_window", None)
            buttons = QMessageBox.Yes | QMessageBox.No
            default_button = QMessageBox.Yes if default else QMessageBox.No
            choice = QMessageBox.question(parent, title, message, buttons, default_button)
            return choice == QMessageBox.Yes
        except Exception:
            pass

    try:
        response = input(f"{message} [{'Y/n' if default else 'y/N'}]: ").strip().lower()
    except Exception:
        return bool(default)
    if not response:
        return bool(default)
    return response in {"y", "yes", "true", "1"}


def _install_optional_prompt_hook(active_session: Any) -> None:
    """Ask about performance plots right after the pose snapshot question."""

    import dockanalyzer.analyze as analyze_module
    import dockanalyzer.config as config_module

    original_prompt = getattr(analyze_module, "_ask_generate_pose_snapshots", None)
    if not callable(original_prompt):
        return
    if getattr(original_prompt, "_dockanalyzer_runscript_v1_wrapped", False):
        return

    asked_once = {"done": False}

    def wrapped_prompt(
        session: Any = None,
        *,
        pose_count: int = 0,
        default: bool = False,
    ) -> bool:
        """Delegate the snapshot prompt and then ask about performance plots."""

        result = original_prompt(session=session, pose_count=pose_count, default=default)
        if not asked_once["done"]:
            target_session = session if session is not None else active_session
            performance_default = bool(
                getattr(config_module, "GENERATE_PERFORMANCE_PLOTS", True)
            )
            generate_plots = _ask_yes_no(
                target_session,
                title="DockAnalyzer performance plots",
                message="Generate performance plots for this analysis?",
                default=performance_default,
            )
            setattr(config_module, "GENERATE_PERFORMANCE_PLOTS", bool(generate_plots))
            asked_once["done"] = True
        return result

    setattr(wrapped_prompt, "_dockanalyzer_runscript_v1_wrapped", True)
    analyze_module._ask_generate_pose_snapshots = wrapped_prompt  # type: ignore[attr-defined]


def main(active_session: Any) -> Any:
    """Run DockAnalyzer against the models open in one ChimeraX session."""

    if active_session is None:
        raise RuntimeError(
            "No active ChimeraX session was supplied. Execute this file with "
            "the ChimeraX 'runscript' command."
        )

    _prepare_package_import()

    from dockanalyzer.analyze import run_chimerax_autorun

    _install_optional_prompt_hook(active_session)
    return run_chimerax_autorun(active_session)


_ACTIVE_SESSION = globals().get("session")
if _ACTIVE_SESSION is None:
    raise RuntimeError(
        "This launcher must be executed inside ChimeraX with the 'runscript' "
        "command; the global ChimeraX session is unavailable."
    )

DOCKANALYZER_LAST_RESULT = main(_ACTIVE_SESSION)
