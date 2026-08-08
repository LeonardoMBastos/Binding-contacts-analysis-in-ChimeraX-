"""
===============================================================================
DockAnalyzer
Utilities Module
===============================================================================

Author:
    Leonardo Bastos

Project:
    DockAnalyzer

Description
-----------
General utility functions used throughout the DockAnalyzer package.

This module centralizes reusable tools that are independent of any specific
interaction type (hydrogen bonds, hydrophobic contacts, π-π interactions,
etc.). Every other module in the project imports functions or classes from
here.

Main responsibilities
---------------------
• Logging
• Timing
• File and directory management
• ChimeraX model discovery
• Atom and residue utilities
• Geometry helper functions
• Generic export functions
• Pretty-print utilities
• Exception handling

This module contains NO interaction-detection algorithms.

Importing this module is side-effect free. Log handlers and output files are
initialized lazily, only when logging or export functionality is invoked.

===============================================================================
"""

from __future__ import annotations

from ._version import __version__

# =============================================================================
# Standard Library
# =============================================================================

import asyncio
import contextvars
import copy
import functools
import inspect
import importlib
import json
import logging
import math
import re
import sys
import time
import traceback
import tempfile
import threading
from dataclasses import dataclass, field
from datetime import datetime
from collections import OrderedDict
from collections.abc import Iterable
from pathlib import Path
from typing import (
    Any,
    Callable,
    cast,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

# =============================================================================
# Third-party Libraries
# =============================================================================


class _LazyModuleProxy:
    """Load an optional third-party module on first attribute access."""

    __slots__ = (
        "_module_name",
        "_global_name",
        "_module",
    )

    def __init__(
        self,
        module_name: str,
        global_name: str,
    ) -> None:
        self._module_name = module_name
        self._global_name = global_name
        self._module = None

    def _load(self) -> Any:
        module = self._module

        if module is None:
            module = importlib.import_module(
                self._module_name
            )
            self._module = module
            globals()[
                self._global_name
            ] = module

        return module

    def __getattr__(self, name: str) -> Any:
        return getattr(
            self._load(),
            name,
        )

    def __dir__(self) -> List[str]:
        return sorted(
            set(super().__dir__())
            | set(dir(self._load()))
        )

    def __repr__(self) -> str:
        if self._module is None:
            return (
                f"<lazy module "
                f"{self._module_name!r}>"
            )

        return repr(self._module)


import numpy as np

pd = _LazyModuleProxy("pandas", "pd")

# =============================================================================
# DockAnalyzer Modules
# =============================================================================

from . import config

# =============================================================================
# ChimeraX Imports
# =============================================================================
#
# IMPORTANT
# ---------
# Imports that depend on ChimeraX should preferably occur INSIDE functions
# whenever possible. This allows portions of DockAnalyzer (such as report
# generation and data processing) to be executed outside ChimeraX for testing.
#
# Example:
#
#     from chimerax.atomic import AtomicStructure
#
# should be imported inside the function that requires it.
#
# =============================================================================

# =============================================================================
# Module Information
# =============================================================================

__author__ = "Leonardo Bastos"
__license__ = "MIT"
__status__ = "Development"

# =============================================================================
# Global Constants
# =============================================================================

SECONDS_PER_MINUTE = 60.0

ANGSTROM_SYMBOL = "\u212B"

DEFAULT_FLOAT_PRECISION = 3

DEFAULT_SEPARATOR = "=" * 80

_FILENAME_WHITESPACE_PATTERN = re.compile(r"\s+")
_FILENAME_UNSAFE_PATTERN = re.compile(r"[^\w\-.]+", re.UNICODE)
_FILENAME_UNDERSCORE_PATTERN = re.compile(r"_+")
_NATURAL_SORT_PATTERN = re.compile(r"(\d+)")

# =============================================================================
# Public Objects
# =============================================================================

__all__ = [
    # Classes (to be implemented)
    "DockModel",
    "AnalysisTimer",
    "DockLogger",

    # Model utilities
    "get_receptor",
    "get_pose_models",
    "get_ligand",

    # Geometry
    "distance",
    "centroid",
    "angle",

    # Residues
    "unique_residues",

    # Files
    "ensure_output_directories",
    "build_output_filename",

    # Export
    "save_csv",
    "save_json",
]

# =============================================================================
# End of Section 1
# =============================================================================

# =============================================================================
# Section 2 — Logging
# =============================================================================


class DockLogger:
    """
    Centralized logging utility for DockAnalyzer.

    The logger can write messages to:

    - The Python console.
    - A log file.
    - The ChimeraX logger, when a valid ChimeraX session is provided.

    The class prevents duplicate handlers when modules are reloaded, which is
    particularly important during interactive development inside ChimeraX.

    Parameters
    ----------
    name : str, optional
        Name assigned to the underlying Python logger.
    session : Any, optional
        Active ChimeraX session. When provided, messages are also forwarded to
        ``session.logger``.
    log_directory : str or Path, optional
        Directory in which log files are created. When omitted, the value is
        obtained from ``config.LOG_DIR`` or defaults to ``logs``.
    log_filename : str, optional
        Name of the log file. When omitted, a timestamped filename is created.
    level : int or str, optional
        Logging level. Accepted values include ``logging.INFO``, ``"INFO"``,
        ``"DEBUG"``, ``"WARNING"``, ``"ERROR"`` and ``"CRITICAL"``.
    save_log : bool, optional
        Whether messages should be written to a log file. When omitted, the
        value is obtained from ``config.SAVE_LOG`` or defaults to ``True``.
    verbose : bool, optional
        Whether messages should be printed to the console. When omitted, the
        value is obtained from ``config.VERBOSE`` or defaults to ``True``.

    Examples
    --------
    Create a logger outside ChimeraX:

    >>> logger = DockLogger()
    >>> logger.info("DockAnalyzer started.")

    Create a logger inside ChimeraX:

    >>> logger = DockLogger(session=session)
    >>> logger.warning("No ligand model was detected.")
    """

    _LOGGER_PREFIX = "DockAnalyzer"

    def __init__(
        self,
        name: str = "DockAnalyzer",
        session: Optional[Any] = None,
        log_directory: Optional[Union[str, Path]] = None,
        log_filename: Optional[str] = None,
        level: Union[int, str] = logging.INFO,
        save_log: Optional[bool] = None,
        verbose: Optional[bool] = None,
    ) -> None:
        """
        Initialize the DockAnalyzer logging system.
        """

        self.name = str(name)
        self.session = session

        self.save_log = (
            bool(save_log)
            if save_log is not None
            else bool(getattr(config, "SAVE_LOG", True))
        )

        self.verbose = (
            bool(verbose)
            if verbose is not None
            else bool(getattr(config, "VERBOSE", True))
        )

        self.level = self._normalize_level(level)

        configured_log_directory = getattr(config, "LOG_DIR", "logs")

        self.log_directory = Path(
            log_directory
            if log_directory is not None
            else configured_log_directory
        ).expanduser()

        self.log_filename = (
            str(log_filename)
            if log_filename is not None
            else self._generate_log_filename()
        )

        self.log_path: Optional[Path] = None

        self._logger = logging.getLogger(self.name)
        self._logger.setLevel(self.level)
        self._logger.propagate = False
        self._configured = False

    # -------------------------------------------------------------------------
    # Logger configuration
    # -------------------------------------------------------------------------

    def _ensure_configured(self) -> None:
        """Configure handlers on first use instead of during module import."""

        if not self._configured:
            self._configure_handlers()

    @staticmethod
    def _normalize_level(level: Union[int, str]) -> int:
        """
        Convert a logging level into its integer representation.

        Parameters
        ----------
        level : int or str
            Logging level represented either as an integer or string.

        Returns
        -------
        int
            Valid logging level.

        Raises
        ------
        ValueError
            If the provided logging level is invalid.
        TypeError
            If the provided value is neither an integer nor a string.
        """

        if isinstance(level, int):
            return level

        if isinstance(level, str):
            normalized_level = level.strip().upper()

            level_mapping = {
                "DEBUG": logging.DEBUG,
                "INFO": logging.INFO,
                "WARNING": logging.WARNING,
                "WARN": logging.WARNING,
                "ERROR": logging.ERROR,
                "CRITICAL": logging.CRITICAL,
                "FATAL": logging.CRITICAL,
            }

            if normalized_level not in level_mapping:
                valid_levels = ", ".join(level_mapping.keys())
                raise ValueError(
                    f"Invalid logging level '{level}'. "
                    f"Valid values are: {valid_levels}."
                )

            return level_mapping[normalized_level]

        raise TypeError(
            "Logging level must be represented as an integer or string."
        )

    @staticmethod
    def _generate_log_filename() -> str:
        """
        Generate a timestamped log filename.

        Returns
        -------
        str
            Filename in the format
            ``dockanalyzer_YYYYMMDD_HHMMSS.log``.
        """

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"dockanalyzer_{timestamp}.log"

    @staticmethod
    def _build_formatter() -> logging.Formatter:
        """
        Create the standard DockAnalyzer log formatter.

        Returns
        -------
        logging.Formatter
            Formatter used by console and file handlers.
        """

        return logging.Formatter(
            fmt=(
                "%(asctime)s | "
                "%(levelname)-8s | "
                "%(name)s | "
                "%(message)s"
            ),
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    def _configure_handlers(self) -> None:
        """
        Configure console and file handlers.

        Existing DockAnalyzer handlers are removed before new handlers are
        created. This prevents duplicated output when the module is reloaded.
        """

        self._remove_dockanalyzer_handlers()
        self._configured = True

        formatter = self._build_formatter()

        if self.verbose:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.level)
            console_handler.setFormatter(formatter)
            console_handler._dockanalyzer_handler = True

            self._logger.addHandler(console_handler)

        if self.save_log:
            try:
                self.log_directory.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                self.log_path = self.log_directory / self.log_filename

                file_handler = logging.FileHandler(
                    self.log_path,
                    mode="a",
                    encoding="utf-8",
                )

                file_handler.setLevel(self.level)
                file_handler.setFormatter(formatter)
                file_handler._dockanalyzer_handler = True

                self._logger.addHandler(file_handler)

            except OSError as error:
                self.log_path = None

                fallback_message = (
                    "DockAnalyzer could not create the log file at "
                    f"'{self.log_directory}'. Error: {error}"
                )

                if self.verbose:
                    print(
                        fallback_message,
                        file=sys.stderr,
                    )

                self._send_to_chimerax(
                    fallback_message,
                    level="warning",
                )

    def _remove_dockanalyzer_handlers(self) -> None:
        """
        Remove handlers previously created by DockLogger.

        Only handlers marked internally as DockAnalyzer handlers are removed.
        External handlers attached by other applications are preserved.
        """

        handlers_to_remove = [
            handler
            for handler in self._logger.handlers
            if getattr(
                handler,
                "_dockanalyzer_handler",
                False,
            )
        ]

        for handler in handlers_to_remove:
            self._logger.removeHandler(handler)

            try:
                handler.close()
            except Exception:
                pass

    # -------------------------------------------------------------------------
    # ChimeraX integration
    # -------------------------------------------------------------------------

    def set_session(self, session: Optional[Any]) -> None:
        """
        Assign or replace the active ChimeraX session.

        Parameters
        ----------
        session : Any or None
            ChimeraX session object. Use ``None`` to disable ChimeraX logging.
        """

        self.session = session

    def _send_to_chimerax(
        self,
        message: str,
        level: str,
    ) -> None:
        """
        Forward a message to the ChimeraX logger.

        Parameters
        ----------
        message : str
            Message to be displayed.
        level : str
            ChimeraX logger method to call.

        Notes
        -----
        This method fails silently if a ChimeraX session is unavailable or if
        the session does not expose the requested logger method. Python logging
        remains active independently.
        """

        if self.session is None:
            return

        chimerax_logger = getattr(
            self.session,
            "logger",
            None,
        )

        if chimerax_logger is None:
            return

        method_name = level.lower()

        if method_name == "critical":
            method_name = "error"

        chimerax_method = getattr(
            chimerax_logger,
            method_name,
            None,
        )

        if not callable(chimerax_method):
            return

        try:
            chimerax_method(str(message))
        except Exception:
            pass

    # -------------------------------------------------------------------------
    # Public logging methods
    # -------------------------------------------------------------------------

    def debug(
        self,
        message: Any,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Record a debug message.
        """

        self._ensure_configured()

        formatted_message = self._prepare_message(
            message,
            args,
        )

        self._logger.debug(
            formatted_message,
            **kwargs,
        )

        self._send_to_chimerax(
            formatted_message,
            level="debug",
        )

    def info(
        self,
        message: Any,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Record an informational message.
        """

        self._ensure_configured()

        formatted_message = self._prepare_message(
            message,
            args,
        )

        self._logger.info(
            formatted_message,
            **kwargs,
        )

        self._send_to_chimerax(
            formatted_message,
            level="info",
        )

    def warning(
        self,
        message: Any,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Record a warning message.
        """

        self._ensure_configured()

        formatted_message = self._prepare_message(
            message,
            args,
        )

        self._logger.warning(
            formatted_message,
            **kwargs,
        )

        self._send_to_chimerax(
            formatted_message,
            level="warning",
        )

    def error(
        self,
        message: Any,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Record an error message.
        """

        self._ensure_configured()

        formatted_message = self._prepare_message(
            message,
            args,
        )

        self._logger.error(
            formatted_message,
            **kwargs,
        )

        self._send_to_chimerax(
            formatted_message,
            level="error",
        )

    def critical(
        self,
        message: Any,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Record a critical error message.
        """

        self._ensure_configured()

        formatted_message = self._prepare_message(
            message,
            args,
        )

        self._logger.critical(
            formatted_message,
            **kwargs,
        )

        self._send_to_chimerax(
            formatted_message,
            level="critical",
        )

    def exception(
        self,
        message: Any,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Record an error message including the current traceback.

        This method should normally be called inside an ``except`` block.
        """

        self._ensure_configured()

        formatted_message = self._prepare_message(
            message,
            args,
        )

        self._logger.exception(
            formatted_message,
            **kwargs,
        )

        traceback_text = traceback.format_exc()

        chimerax_message = formatted_message

        if (
            traceback_text
            and traceback_text.strip() != "NoneType: None"
        ):
            chimerax_message = (
                f"{formatted_message}\n"
                f"{traceback_text}"
            )

        self._send_to_chimerax(
            chimerax_message,
            level="error",
        )

    def log(
        self,
        level: Union[int, str],
        message: Any,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Record a message using a dynamically selected level.

        Parameters
        ----------
        level : int or str
            Logging level.
        message : Any
            Message or object to record.
        *args : Any
            Optional formatting arguments.
        **kwargs : Any
            Additional arguments passed to the Python logger.
        """

        self._ensure_configured()

        normalized_level = self._normalize_level(level)

        formatted_message = self._prepare_message(
            message,
            args,
        )

        self._logger.log(
            normalized_level,
            formatted_message,
            **kwargs,
        )

        level_name = logging.getLevelName(
            normalized_level
        ).lower()

        self._send_to_chimerax(
            formatted_message,
            level=level_name,
        )

    # -------------------------------------------------------------------------
    # Convenience methods
    # -------------------------------------------------------------------------

    @staticmethod
    def _prepare_message(
        message: Any,
        args: Sequence[Any],
    ) -> str:
        """
        Convert a message and optional formatting arguments into text.

        Parameters
        ----------
        message : Any
            Message or object to convert.
        args : Sequence[Any]
            Optional values used with percent-style formatting.

        Returns
        -------
        str
            Prepared log message.
        """

        message_text = str(message)

        if not args:
            return message_text

        try:
            return message_text % tuple(args)
        except (TypeError, ValueError):
            joined_arguments = " ".join(
                str(argument)
                for argument in args
            )

            return (
                f"{message_text} "
                f"{joined_arguments}"
            ).strip()

    def separator(
        self,
        character: str = "=",
        length: int = 80,
        level: Union[int, str] = logging.INFO,
    ) -> None:
        """
        Write a separator line to the log.

        Parameters
        ----------
        character : str, optional
            Character used to construct the separator.
        length : int, optional
            Number of repeated characters.
        level : int or str, optional
            Logging level used for the separator.
        """

        if not character:
            character = "="

        separator_text = character[0] * max(
            1,
            int(length),
        )

        self.log(
            level,
            separator_text,
        )

    def section(
        self,
        title: str,
        level: Union[int, str] = logging.INFO,
        character: str = "=",
        length: int = 80,
    ) -> None:
        """
        Write a formatted section heading to the log.

        Parameters
        ----------
        title : str
            Section title.
        level : int or str, optional
            Logging level used for the heading.
        character : str, optional
            Character used to construct the separator.
        length : int, optional
            Separator length.
        """

        self.separator(
            character=character,
            length=length,
            level=level,
        )

        self.log(
            level,
            str(title),
        )

        self.separator(
            character=character,
            length=length,
            level=level,
        )

    def get_log_path(self) -> Optional[Path]:
        """
        Return the active log-file path.

        Returns
        -------
        Path or None
            Log-file path, or ``None`` when file logging is disabled or failed.
        """

        return self.log_path

    def set_level(
        self,
        level: Union[int, str],
    ) -> None:
        """
        Change the active logging level.

        Parameters
        ----------
        level : int or str
            New logging level.
        """

        self.level = self._normalize_level(level)
        self._logger.setLevel(self.level)

        for handler in self._logger.handlers:
            if getattr(
                handler,
                "_dockanalyzer_handler",
                False,
            ):
                handler.setLevel(self.level)

    def close(self) -> None:
        """
        Close all handlers created by this DockLogger instance.
        """

        self._remove_dockanalyzer_handlers()
        self._configured = False

    @property
    def python_logger(self) -> logging.Logger:
        """
        Return the underlying Python logger.

        Returns
        -------
        logging.Logger
            Internal logger instance.
        """

        return self._logger


# =============================================================================
# Default Module Logger
# =============================================================================

logger = DockLogger()


# =============================================================================
# End of Section 2
# =============================================================================


# =============================================================================
# Section 3 — Timer
# =============================================================================


@dataclass
class TimerRecord:
    """
    Store timing information for one analysis step.

    Parameters
    ----------
    name : str
        Name assigned to the measured step.
    start_time : float
        Start time obtained from ``time.perf_counter()``.
    end_time : float
        End time obtained from ``time.perf_counter()``.
    elapsed : float
        Elapsed time in seconds.
    """

    name: str
    start_time: float
    end_time: float
    elapsed: float
   
    def to_dict(
        self,
    ) -> Dict[str, Union[str, float]]:
        """Return the timing record as a dictionary."""

        return {
            "name": self.name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "elapsed_seconds": self.elapsed,
        }

    def to_dicts(
        self,
    ) -> Dict[str, Union[str, float]]:
        """Return the timing record as a dictionary.

        This compatibility alias preserves the original public method name.
        """

        return self.to_dict()


class AnalysisTimer:
    """
    Measure execution times throughout DockAnalyzer.

    The timer supports:

    - Manual start and stop operations.
    - Measurement of multiple sequential steps.
    - Context-manager usage with ``with``.
    - Integration with :class:`DockLogger`.
    - Timing summaries.
    - Conversion of timing records to dictionaries.

    Parameters
    ----------
    name : str, optional
        Default name assigned to the measured operation.
    logger : DockLogger, optional
        Logger used to report timing information.
    auto_log : bool, optional
        Whether completed measurements should automatically be written to the
        logger.
    precision : int, optional
        Number of decimal places used when displaying elapsed times.

    Examples
    --------
    Manual usage:

    >>> timer = AnalysisTimer("Hydrogen bonds", logger=logger)
    >>> timer.start()
    >>> detect_hydrogen_bonds()
    >>> timer.stop()

    Context-manager usage:

    >>> with AnalysisTimer("Hydrophobic contacts", logger=logger):
    ...     detect_hydrophobic_contacts()

    Multiple steps:

    >>> timer = AnalysisTimer(logger=logger)
    >>> timer.start("Load receptor")
    >>> load_receptor()
    >>> timer.stop()
    >>> timer.start("Detect contacts")
    >>> detect_contacts()
    >>> timer.stop()
    >>> timer.print_summary()
    """

    def __init__(
        self,
        name: str = "Analysis",
        logger: Optional[DockLogger] = None,
        auto_log: bool = True,
        precision: int = DEFAULT_FLOAT_PRECISION,
    ) -> None:
        """
        Initialize the analysis timer.
        """

        self.name = str(name)
        self.logger = logger
        self.auto_log = bool(auto_log)
        self.precision = max(0, int(precision))

        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None
        self._elapsed: float = 0.0
        self._running: bool = False
        self._active_name: Optional[str] = None

        self.records: List[TimerRecord] = []

    # -------------------------------------------------------------------------
    # Timer control
    # -------------------------------------------------------------------------

    def start(
        self,
        name: Optional[str] = None,
    ) -> AnalysisTimer:
        """
        Start a new timing measurement.

        Parameters
        ----------
        name : str, optional
            Name assigned to the new measurement. When omitted, the timer's
            default name is used.

        Returns
        -------
        AnalysisTimer
            The current timer instance.

        Raises
        ------
        RuntimeError
            If the timer is already running.
        """

        if self._running:
            active_name = self._active_name or self.name

            raise RuntimeError(
                f"AnalysisTimer is already running for '{active_name}'. "
                "Call stop() before starting another measurement."
            )

        self._active_name = (
            str(name)
            if name is not None
            else self.name
        )

        self._start_time = time.perf_counter()
        self._end_time = None
        self._elapsed = 0.0
        self._running = True

        return self

    def stop(
        self,
        name: Optional[str] = None,
    ) -> float:
        """
        Stop the active timing measurement.

        Parameters
        ----------
        name : str, optional
            Alternative name assigned to the completed timing record. When
            omitted, the active step name is preserved.

        Returns
        -------
        float
            Elapsed time in seconds.

        Raises
        ------
        RuntimeError
            If the timer is not currently running.
        """

        if not self._running or self._start_time is None:
            raise RuntimeError(
                "AnalysisTimer is not running. Call start() before stop()."
            )

        self._end_time = time.perf_counter()
        self._elapsed = self._end_time - self._start_time
        self._running = False

        record_name = (
            str(name)
            if name is not None
            else self._active_name or self.name
        )

        record = TimerRecord(
            name=record_name,
            start_time=self._start_time,
            end_time=self._end_time,
            elapsed=self._elapsed,
        )

        self.records.append(record)

        if self.auto_log:
            self._log_completed_record(record)

        self._active_name = None

        return self._elapsed

    def reset(
        self,
        clear_records: bool = True,
    ) -> None:
        """
        Reset the timer state.

        Parameters
        ----------
        clear_records : bool, optional
            Whether previously stored timing records should also be removed.
        """

        self._start_time = None
        self._end_time = None
        self._elapsed = 0.0
        self._running = False
        self._active_name = None

        if clear_records:
            self.records.clear()

    # -------------------------------------------------------------------------
    # Timing information
    # -------------------------------------------------------------------------

    @property
    def running(self) -> bool:
        """
        Return whether the timer is currently running.
        """

        return self._running

    @property
    def elapsed(self) -> float:
        """
        Return the elapsed time in seconds.

        If the timer is running, the returned value is calculated from the
        current time. Otherwise, the duration of the most recently completed
        measurement is returned.
        """

        if self._running and self._start_time is not None:
            return time.perf_counter() - self._start_time

        return self._elapsed

    @property
    def total_elapsed(self) -> float:
        """
        Return the sum of all completed timing records.

        Returns
        -------
        float
            Total elapsed time in seconds.
        """

        return sum(
            record.elapsed
            for record in self.records
        )

    @property
    def last_record(self) -> Optional[TimerRecord]:
        """
        Return the most recently completed timing record.

        Returns
        -------
        TimerRecord or None
            Last timing record, or ``None`` when no measurement was completed.
        """

        if not self.records:
            return None

        return self.records[-1]

    # -------------------------------------------------------------------------
    # Formatting
    # -------------------------------------------------------------------------

    def format_duration(
        self,
        seconds: float,
    ) -> str:
        """
        Format a duration using seconds or minutes.

        Parameters
        ----------
        seconds : float
            Duration in seconds.

        Returns
        -------
        str
            Human-readable duration.
        """

        seconds = float(seconds)

        if seconds < SECONDS_PER_MINUTE:
            return f"{seconds:.{self.precision}f} s"

        minutes = int(seconds // SECONDS_PER_MINUTE)
        remaining_seconds = seconds % SECONDS_PER_MINUTE

        return (
            f"{minutes} min "
            f"{remaining_seconds:.{self.precision}f} s"
        )

    def summary(
        self,
        title: str = "DockAnalyzer Timing Summary",
    ) -> str:
        """
        Generate a formatted summary of all timing records.

        Parameters
        ----------
        title : str, optional
            Title displayed at the top of the summary.

        Returns
        -------
        str
            Formatted timing summary.
        """

        separator = DEFAULT_SEPARATOR

        lines = [
            separator,
            str(title),
            separator,
        ]

        if not self.records:
            lines.append("No completed timing records.")
            lines.append(separator)

            return "\n".join(lines)

        longest_name = max(
            len(record.name)
            for record in self.records
        )

        for record in self.records:
            duration = self.format_duration(record.elapsed)

            lines.append(
                f"{record.name.ljust(longest_name)}"
                f"  {duration.rjust(15)}"
            )

        lines.append("-" * len(separator))

        total_duration = self.format_duration(
            self.total_elapsed
        )

        lines.append(
            f"{'Total'.ljust(longest_name)}"
            f"  {total_duration.rjust(15)}"
        )

        lines.append(separator)

        return "\n".join(lines)

    def print_summary(
        self,
        title: str = "DockAnalyzer Timing Summary",
        use_logger: bool = True,
    ) -> str:
        """
        Display and return the timing summary.

        Parameters
        ----------
        title : str, optional
            Summary title.
        use_logger : bool, optional
            Whether the summary should be written through ``DockLogger`` when
            a logger is available.

        Returns
        -------
        str
            Formatted timing summary.
        """

        summary_text = self.summary(title=title)

        if use_logger and self.logger is not None:
            self.logger.info("\n%s", summary_text)
        else:
            print(summary_text)

        return summary_text

    # -------------------------------------------------------------------------
    # Data export helpers
    # -------------------------------------------------------------------------

    def to_dicts(
        self,
    ) -> List[Dict[str, Union[str, float]]]:
        """Return all completed timing records as dictionaries."""

        return [
            record.to_dict()
            for record in self.records
        ]

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        """Return the complete timer state as a dictionary."""

        return {
            "name": self.name,
            "running": self.running,
            "elapsed_seconds": self.elapsed,
            "total_elapsed_seconds": self.total_elapsed,
            "records": self.to_dicts(),
        }

    def to_dataframe(
        self,
    ) -> pd.DataFrame:
        """
        Convert all timing records into a pandas DataFrame.

        Returns
        -------
        pandas.DataFrame
            Table containing the stored timing records.
        """

        return pd.DataFrame(self.to_dicts())

    # -------------------------------------------------------------------------
    # Logging integration
    # -------------------------------------------------------------------------

    def _log_completed_record(
        self,
        record: TimerRecord,
    ) -> None:
        """
        Log a completed timing record.

        Parameters
        ----------
        record : TimerRecord
            Completed measurement.
        """

        if self.logger is None:
            return

        self.logger.info(
            "%s completed in %s.",
            record.name,
            self.format_duration(record.elapsed),
        )

    # -------------------------------------------------------------------------
    # Context-manager protocol
    # -------------------------------------------------------------------------

    def __enter__(self) -> AnalysisTimer:
        """
        Start the timer when entering a ``with`` block.
        """

        self.start()
        return self

    def __exit__(
        self,
        exception_type: Optional[type],
        exception_value: Optional[BaseException],
        exception_traceback: Optional[Any],
    ) -> bool:
        """
        Stop the timer when leaving a ``with`` block.

        Exceptions raised inside the block are logged but are not suppressed.

        Returns
        -------
        bool
            Always ``False``, ensuring that exceptions continue propagating.
        """

        if self._running:
            self.stop()

        if exception_type is not None and self.logger is not None:
            self.logger.error(
                "%s failed with %s: %s",
                self.name,
                exception_type.__name__,
                exception_value,
            )

        return False

    # -------------------------------------------------------------------------
    # Special methods
    # -------------------------------------------------------------------------

    def __len__(self) -> int:
        """
        Return the number of completed timing records.
        """

        return len(self.records)

    def __str__(self) -> str:
        """
        Return a concise timer representation.
        """

        if self._running:
            return (
                f"AnalysisTimer("
                f"name='{self._active_name or self.name}', "
                f"running=True, "
                f"elapsed={self.format_duration(self.elapsed)}"
                f")"
            )

        if self.last_record is not None:
            return (
                f"AnalysisTimer("
                f"name='{self.last_record.name}', "
                f"running=False, "
                f"elapsed={self.format_duration(self.last_record.elapsed)}"
                f")"
            )

        return (
            f"AnalysisTimer("
            f"name='{self.name}', "
            f"running=False, "
            f"elapsed={self.format_duration(0.0)}"
            f")"
        )


# Add TimerRecord to the public module interface.
if "TimerRecord" not in __all__:
    __all__.append("TimerRecord")


# =============================================================================
# End of Section 3
# =============================================================================


# =============================================================================
# Internal receptor preparation cache
# =============================================================================

_PREPARED_RECEPTOR_ATTRIBUTE = "_dockanalyzer_prepared_receptor"
_PREPARED_RECEPTOR_CACHE_LIMIT = 8
_PREPARED_RECEPTOR_CACHE_LOCK = threading.RLock()
_PREPARED_RECEPTOR_CACHE: "OrderedDict[int, _PreparedReceptor]" = OrderedDict()


@dataclass
class _PreparedReceptor:
    """Internal immutable-source cache shared across docking poses.

    The cache stores receptor atoms, coordinates, one reusable spatial index,
    and detector-specific derived values. It is deliberately private so the
    public DockAnalyzer API remains unchanged.
    """

    source: Any
    atoms: Tuple[Any, ...]
    coordinates: Any
    spatial_index: Any
    derived: Dict[Any, Any] = field(default_factory=dict, repr=False)
    created_at: float = field(default_factory=time.perf_counter, repr=False)
    _lock: Any = field(default_factory=threading.RLock, repr=False, compare=False)

    @property
    def source_identity(self) -> int:
        return id(self.source)

    @property
    def atom_count(self) -> int:
        return len(self.atoms)

    def matches(self, source: Any) -> bool:
        return source is self.source

    def get_or_create(self, key: Any, builder: Callable[[], Any]) -> Any:
        """Return one derived receptor value, creating it once if needed."""

        with self._lock:
            if key in self.derived:
                return self.derived[key]
        value = builder()
        with self._lock:
            return self.derived.setdefault(key, value)

    def nearby_indices(self, query_coordinates: Any, radius: float) -> Tuple[int, ...]:
        return self.spatial_index.query_unique_indices(query_coordinates, radius)

    def summary(self) -> Dict[str, Any]:
        return {
            "available": True,
            "atom_count": self.atom_count,
            "spatial_backend": getattr(self.spatial_index, "backend", "unknown"),
            "derived_cache_entries": len(self.derived),
        }


def _materialize_receptor_atoms(source: Any) -> Tuple[Any, ...]:
    """Extract a receptor atom tuple without importing interaction modules."""

    if source is None:
        raise ValueError("A receptor source is required for preparation.")
    atom_collection = getattr(source, "atoms", None)
    if atom_collection is None:
        atom_collection = getattr(source, "all_atoms", None)
    if atom_collection is None:
        if isinstance(source, (str, bytes, Mapping)):
            raise TypeError("The receptor source does not expose an atom collection.")
        try:
            atom_collection = tuple(source)
        except TypeError as exc:
            raise TypeError(
                "The receptor source does not expose an iterable atom collection."
            ) from exc
    atoms = tuple(atom_collection)
    if not atoms:
        raise ValueError("The receptor atom collection is empty.")
    return atoms


def _prepare_receptor_cache(source: Any, *, prefer_scipy: bool = True) -> _PreparedReceptor:
    """Return the process-local prepared receptor for one source object."""

    cache_key = id(source)
    with _PREPARED_RECEPTOR_CACHE_LOCK:
        cached = _PREPARED_RECEPTOR_CACHE.get(cache_key)
        if cached is not None and cached.matches(source):
            _PREPARED_RECEPTOR_CACHE.move_to_end(cache_key)
            return cached

    atoms = _materialize_receptor_atoms(source)
    from . import geometry as _geometry

    coordinates = _geometry.get_coordinates(
        atoms,
        scene=True,
        name="prepared receptor atoms",
        allow_empty=False,
        require_finite=True,
        copy=True,
    )
    spatial_index = _geometry._build_spatial_neighbor_index(
        coordinates,
        prefer_scipy=prefer_scipy,
    )
    prepared = _PreparedReceptor(
        source=source,
        atoms=atoms,
        coordinates=coordinates,
        spatial_index=spatial_index,
    )
    with _PREPARED_RECEPTOR_CACHE_LOCK:
        _PREPARED_RECEPTOR_CACHE[cache_key] = prepared
        _PREPARED_RECEPTOR_CACHE.move_to_end(cache_key)
        while len(_PREPARED_RECEPTOR_CACHE) > _PREPARED_RECEPTOR_CACHE_LIMIT:
            _PREPARED_RECEPTOR_CACHE.popitem(last=False)
    return prepared


def _get_prepared_receptor_cache(source: Any) -> Optional[_PreparedReceptor]:
    """Return an existing prepared receptor without constructing one."""

    if source is None:
        return None
    with _PREPARED_RECEPTOR_CACHE_LOCK:
        cached = _PREPARED_RECEPTOR_CACHE.get(id(source))
        if cached is None or not cached.matches(source):
            return None
        _PREPARED_RECEPTOR_CACHE.move_to_end(id(source))
        return cached


def _attach_prepared_receptor_cache(
    dock_models: Iterable[Any],
    prepared: _PreparedReceptor,
) -> None:
    """Attach one private prepared receptor reference to several DockModels."""

    for dock_model in dock_models:
        try:
            setattr(dock_model, _PREPARED_RECEPTOR_ATTRIBUTE, prepared)
        except Exception:
            continue


def _prepared_receptor_from_dock_model(
    dock_model: Any,
    *,
    receptor: Any = None,
) -> Optional[_PreparedReceptor]:
    """Resolve a prepared receptor attached to a DockModel or source cache."""

    prepared = getattr(dock_model, _PREPARED_RECEPTOR_ATTRIBUTE, None)
    if isinstance(prepared, _PreparedReceptor):
        if receptor is None or prepared.matches(receptor):
            return prepared
    return _get_prepared_receptor_cache(receptor)


# =============================================================================
# Section 4 — DockModel
# =============================================================================


@dataclass
class DockModel:
    """
    Central data container for one molecular-docking pose.

    A DockModel instance stores the original ChimeraX pose model, its ligand,
    detected molecular interactions, scores, statistics and generated output
    files.

    The class does not perform interaction detection. Detection algorithms are
    implemented in specialized modules such as ``contacts.py``, ``hbonds.py``,
    ``hydrophobic.py`` and ``pi.py``.

    Parameters
    ----------
    name : str
        Human-readable name assigned to the docking pose.
    pose : Any, optional
        ChimeraX atomic model representing the docking pose.
    ligand : Any, optional
        ChimeraX ligand model, residue, atom collection or ligand descriptor.
    contacts : list, optional
        General receptor-ligand contacts.
    hbonds : list, optional
        Hydrogen-bond interactions.
    hydrophobic : list, optional
        Hydrophobic interactions.
    pi : dict, optional
        Pi-related interactions, such as pi-stacking and pi-cation contacts.
    score : float, optional
        Global DockAnalyzer score assigned to the pose.
    statistics : dict, optional
        Calculated statistical information.
    files : dict, optional
        Paths of files generated for the pose.
    metadata : dict, optional
        Additional information associated with the docking pose.

    Examples
    --------
    Create an empty docking model:

    >>> dock_model = DockModel(name="pose_01")

    Store a hydrogen bond:

    >>> dock_model.add_hbond(hbond_data)

    Store an output file:

    >>> dock_model.set_file("csv", "results/pose_01.csv")

    Generate a dictionary representation:

    >>> data = dock_model.to_dict()
    """

    name: str = "unnamed_pose"

    pose: Optional[Any] = None

    receptor: Optional[Any] = None

    ligand: Optional[Any] = None

    contacts: List[Any] = field(
        default_factory=list
    )

    hbonds: List[Any] = field(
        default_factory=list
    )

    hydrophobic: List[Any] = field(
        default_factory=list
    )

    pi: Dict[str, List[Any]] = field(
        default_factory=lambda: {
            "stacking": [],
            "cation": [],
        }
    )

    score: Optional[float] = None

    statistics: Dict[str, Any] = field(
        default_factory=dict
    )

    files: Dict[str, Optional[Path]] = field(
        default_factory=lambda: {
            "csv": None,
            "excel": None,
            "json": None,
            "image": None,
            "report": None,
            "session": None,
            "log": None,
        }
    )

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------

    def __post_init__(self) -> None:
        """Validate and normalize the DockModel attributes."""

        if self.name is None:
            self.name = "unnamed_pose"

        self.name = str(self.name).strip()

        if not self.name:
            raise ValueError(
                "DockModel name cannot be empty."
            )

        if self.score is not None:
            self.score = float(self.score)

        self.contacts = self._normalize_interaction_collection(
            self.contacts,
            field_name="contacts",
        )
        self.hbonds = self._normalize_interaction_collection(
            self.hbonds,
            field_name="hbonds",
        )
        self.hydrophobic = self._normalize_interaction_collection(
            self.hydrophobic,
            field_name="hydrophobic",
        )
        self.statistics = self._normalize_mapping_field(
            self.statistics,
            field_name="statistics",
        )
        self.metadata = self._normalize_mapping_field(
            self.metadata,
            field_name="metadata",
        )

        self._normalize_pi_dictionary()
        self._normalize_file_dictionary()

    @staticmethod
    def _normalize_interaction_collection(
        value: Any,
        *,
        field_name: str,
    ) -> List[Any]:
        """Normalize one interaction collection into an independent list."""

        if value is None:
            return []

        if isinstance(value, list):
            return list(value)

        if isinstance(value, Mapping):
            return [value]

        if isinstance(
            value,
            (
                str,
                bytes,
                bytearray,
            ),
        ):
            return [value]

        try:
            return list(value)
        except TypeError as error:
            raise TypeError(
                f"DockModel.{field_name} must be an iterable "
                "of interactions or None."
            ) from error

    @staticmethod
    def _normalize_mapping_field(
        value: Any,
        *,
        field_name: str,
    ) -> Dict[str, Any]:
        """Normalize one mapping field into an independent dictionary."""

        if value is None:
            return {}

        if not isinstance(value, Mapping):
            raise TypeError(
                f"DockModel.{field_name} must be a mapping or None."
            )

        return dict(value)

    def _normalize_pi_dictionary(self) -> None:
        """Ensure that pi-interaction groups contain independent lists."""

        if self.pi is None:
            self.pi = {}

        if not isinstance(self.pi, Mapping):
            raise TypeError(
                "DockModel.pi must be a mapping or None."
            )

        normalized_pi: Dict[str, List[Any]] = {}

        for interaction_type, interactions in self.pi.items():
            normalized_pi[str(interaction_type)] = (
                self._normalize_interaction_collection(
                    interactions,
                    field_name=(
                        f"pi[{interaction_type!r}]"
                    ),
                )
            )

        normalized_pi.setdefault(
            "stacking",
            [],
        )
        normalized_pi.setdefault(
            "cation",
            [],
        )

        self.pi = normalized_pi

    def _normalize_file_dictionary(self) -> None:
        """Ensure that output-file entries use normalized Path objects."""

        if self.files is None:
            self.files = {}

        if not isinstance(self.files, Mapping):
            raise TypeError(
                "DockModel.files must be a mapping or None."
            )

        normalized_files: Dict[str, Optional[Path]] = dict(
            self.files
        )

        standard_file_types = (
            "csv",
            "excel",
            "json",
            "image",
            "report",
            "session",
            "log",
        )

        for file_type in standard_file_types:
            normalized_files.setdefault(
                file_type,
                None,
            )

        for file_type, file_path in list(
            normalized_files.items()
        ):
            if file_path is None:
                continue

            try:
                normalized_files[file_type] = Path(
                    file_path
                )
            except TypeError as error:
                raise TypeError(
                    f"DockModel.files[{file_type!r}] must be a path-like "
                    "value or None."
                ) from error

        self.files = normalized_files

    # -------------------------------------------------------------------------
    # Interaction storage
    # -------------------------------------------------------------------------

    def add_contact(
        self,
        interaction: Any,
    ) -> None:
        """
        Add a general receptor-ligand contact.

        Parameters
        ----------
        interaction : Any
            Contact object or dictionary.
        """

        self.contacts.append(interaction)

    def add_hbond(
        self,
        interaction: Any,
    ) -> None:
        """
        Add a hydrogen-bond interaction.

        Parameters
        ----------
        interaction : Any
            Hydrogen-bond object or dictionary.
        """

        self.hbonds.append(interaction)

    def add_hydrophobic(
        self,
        interaction: Any,
    ) -> None:
        """
        Add a hydrophobic interaction.

        Parameters
        ----------
        interaction : Any
            Hydrophobic-contact object or dictionary.
        """

        self.hydrophobic.append(interaction)

    def add_pi_interaction(
        self,
        interaction: Any,
        interaction_type: str = "stacking",
    ) -> None:
        """
        Add a pi-related interaction.

        Parameters
        ----------
        interaction : Any
            Pi-interaction object or dictionary.
        interaction_type : str, optional
            Pi-interaction category. Standard categories are ``stacking`` and
            ``cation``.
        """

        normalized_type = (
            str(interaction_type)
            .strip()
            .lower()
            .replace("-", "_")
            .replace(" ", "_")
        )

        aliases = {
            "pi_stacking": "stacking",
            "pistacking": "stacking",
            "stack": "stacking",
            "pi_cation": "cation",
            "pication": "cation",
            "cation_pi": "cation",
        }

        normalized_type = aliases.get(
            normalized_type,
            normalized_type,
        )

        if normalized_type not in self.pi:
            self.pi[normalized_type] = []

        self.pi[normalized_type].append(
            interaction
        )

    def add_interaction(
        self,
        interaction_type: str,
        interaction: Any,
    ) -> None:
        """
        Add an interaction using a generic interface.

        Parameters
        ----------
        interaction_type : str
            Interaction category.
        interaction : Any
            Interaction object or dictionary.

        Raises
        ------
        ValueError
            If the interaction type is not recognized.
        """

        normalized_type = (
            str(interaction_type)
            .strip()
            .lower()
            .replace("-", "_")
            .replace(" ", "_")
        )

        if normalized_type in {
            "contact",
            "contacts",
            "general_contact",
        }:
            self.add_contact(interaction)
            return

        if normalized_type in {
            "hbond",
            "hbonds",
            "hydrogen_bond",
            "hydrogen_bonds",
        }:
            self.add_hbond(interaction)
            return

        if normalized_type in {
            "hydrophobic",
            "hydrophobic_contact",
            "hydrophobic_contacts",
        }:
            self.add_hydrophobic(interaction)
            return

        if normalized_type in {
            "pi",
            "pi_stacking",
            "stacking",
            "pistacking",
        }:
            self.add_pi_interaction(
                interaction,
                interaction_type="stacking",
            )
            return

        if normalized_type in {
            "pi_cation",
            "cation_pi",
            "cation",
            "pication",
        }:
            self.add_pi_interaction(
                interaction,
                interaction_type="cation",
            )
            return

        raise ValueError(
            f"Unknown interaction type: '{interaction_type}'."
        )

    # -------------------------------------------------------------------------
    # Interaction access
    # -------------------------------------------------------------------------

    def get_pi_interactions(
        self,
        interaction_type: Optional[str] = None,
    ) -> List[Any]:
        """
        Return stored pi interactions.

        Parameters
        ----------
        interaction_type : str, optional
            Specific pi-interaction category. When omitted, interactions from
            all pi categories are returned.

        Returns
        -------
        list
            Pi-interaction objects.
        """

        if interaction_type is None:
            interactions: List[Any] = []

            for pi_interactions in self.pi.values():
                interactions.extend(
                    pi_interactions
                )

            return interactions

        normalized_type = (
            str(interaction_type)
            .strip()
            .lower()
            .replace("-", "_")
            .replace(" ", "_")
        )

        aliases = {
            "pi_stacking": "stacking",
            "pistacking": "stacking",
            "pi_cation": "cation",
            "pication": "cation",
            "cation_pi": "cation",
        }

        normalized_type = aliases.get(
            normalized_type,
            normalized_type,
        )

        return list(
            self.pi.get(
                normalized_type,
                [],
            )
        )

    def get_all_interactions(
        self,
    ) -> List[Any]:
        """
        Return all stored molecular interactions.

        Returns
        -------
        list
            Combined interaction list.
        """

        interactions: List[Any] = []

        interactions.extend(self.contacts)
        interactions.extend(self.hbonds)
        interactions.extend(self.hydrophobic)
        interactions.extend(
            self.get_pi_interactions()
        )

        return interactions

    # -------------------------------------------------------------------------
    # Score and statistics
    # -------------------------------------------------------------------------

    def set_score(
        self,
        score: Optional[float],
    ) -> None:
        """
        Assign the global DockAnalyzer score.

        Parameters
        ----------
        score : float or None
            Global score assigned to the docking pose.
        """

        self.score = (
            None
            if score is None
            else float(score)
        )

    def update_statistics(
        self,
        additional_statistics: Optional[
            Dict[str, Any]
        ] = None,
    ) -> Dict[str, Any]:
        """
        Update interaction-count statistics.

        Parameters
        ----------
        additional_statistics : dict, optional
            Additional statistical values to merge into the standard
            statistics dictionary.

        Returns
        -------
        dict
            Updated statistics dictionary.
        """

        standard_statistics = {
            "contacts": len(self.contacts),
            "hbonds": len(self.hbonds),
            "hydrophobic": len(self.hydrophobic),
            "pi_stacking": len(
                self.pi.get(
                    "stacking",
                    [],
                )
            ),
            "pi_cation": len(
                self.pi.get(
                    "cation",
                    [],
                )
            ),
            "pi_total": len(
                self.get_pi_interactions()
            ),
            "total_interactions": len(
                self.get_all_interactions()
            ),
            "score": self._serialize_value(
                self.score
            ),
        }

        self.statistics.update(
            standard_statistics
        )

        if additional_statistics:
            self.statistics.update(
                additional_statistics
            )

        return self.statistics

    # -------------------------------------------------------------------------
    # Output-file management
    # -------------------------------------------------------------------------

    def set_file(
        self,
        file_type: str,
        file_path: Optional[Union[str, Path]],
    ) -> None:
        """
        Store the path of a generated output file.

        Parameters
        ----------
        file_type : str
            Output-file category, such as ``csv``, ``json`` or ``image``.
        file_path : str, Path or None
            Path of the generated file.
        """

        normalized_type = (
            str(file_type)
            .strip()
            .lower()
            .replace("-", "_")
            .replace(" ", "_")
        )

        self.files[normalized_type] = (
            None
            if file_path is None
            else Path(file_path)
        )

    def get_file(
        self,
        file_type: str,
    ) -> Optional[Path]:
        """
        Return a stored output-file path.

        Parameters
        ----------
        file_type : str
            Output-file category.

        Returns
        -------
        Path or None
            Stored file path.
        """

        normalized_type = (
            str(file_type)
            .strip()
            .lower()
            .replace("-", "_")
            .replace(" ", "_")
        )

        return self.files.get(
            normalized_type
        )

    # -------------------------------------------------------------------------
    # Model information
    # -------------------------------------------------------------------------

    @property
    def interaction_count(self) -> int:
        """
        Return the total number of stored interactions.
        """

        return len(
            self.get_all_interactions()
        )

    @property
    def has_interactions(self) -> bool:
        """
        Return whether at least one interaction was detected.
        """

        return self.interaction_count > 0

    @property
    def model_id(self) -> Optional[Any]:
        """
        Attempt to return the ChimeraX model identifier.

        Returns
        -------
        Any or None
            ChimeraX model identifier, when available.
        """

        if self.pose is None:
            return None

        for attribute_name in (
            "id_string",
            "id",
        ):
            model_identifier = getattr(
                self.pose,
                attribute_name,
                None,
            )

            if model_identifier is not None:
                return model_identifier

        return None

    # -------------------------------------------------------------------------
    # Reset methods
    # -------------------------------------------------------------------------

    def clear_interactions(self) -> None:
        """
        Remove all stored interactions.
        """

        self.contacts.clear()
        self.hbonds.clear()
        self.hydrophobic.clear()

        for pi_interactions in self.pi.values():
            pi_interactions.clear()

        self.statistics.clear()
        self.score = None

    def clear_files(self) -> None:
        """
        Remove all stored output-file paths.
        """

        for file_type in self.files:
            self.files[file_type] = None

    def reset(
        self,
        clear_metadata: bool = False,
    ) -> None:
        """
        Reset calculated results while preserving pose and ligand references.

        Parameters
        ----------
        clear_metadata : bool, optional
            Whether user-defined metadata should also be removed.
        """

        self.clear_interactions()
        self.clear_files()

        if clear_metadata:
            self.metadata.clear()

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    @staticmethod
    def _serialize_value(
        value: Any,
    ) -> Any:
        """Convert a value into a strict JSON-compatible representation."""

        return _make_serializable(value)

    def to_dict(
        self,
        include_pose: bool = False,
        include_ligand: bool = False,
        include_receptor: bool = False,
    ) -> Dict[str, Any]:
        """
        Convert the DockModel into a dictionary.

        Parameters
        ----------
        include_pose : bool, optional
            Whether the pose object should be included.
        include_ligand : bool, optional
            Whether the ligand object should be included.
        include_receptor : bool, optional
            Whether the receptor object should be included.

        Returns
        -------
        dict
            Serializable DockModel representation.
        """

        self.update_statistics()

        data: Dict[str, Any] = {
            "name": self.name,
            "model_id": self._serialize_value(
                self.model_id
            ),
            "contacts": self._serialize_value(
                self.contacts
            ),
            "hbonds": self._serialize_value(
                self.hbonds
            ),
            "hydrophobic": self._serialize_value(
                self.hydrophobic
            ),
            "pi": self._serialize_value(
                self.pi
            ),
            "score": self._serialize_value(
                self.score
            ),
            "statistics": self._serialize_value(
                self.statistics
            ),
            "files": self._serialize_value(
                self.files
            ),
            "metadata": self._serialize_value(
                self.metadata
            ),
        }

        if include_pose:
            data["pose"] = self._serialize_value(
                self.pose
            )
        if include_receptor:
            data["receptor"] = self._serialize_value(
                self.receptor
            )
        if include_ligand:
            data["ligand"] = self._serialize_value(
                self.ligand
            )

        return data

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    def summary(self) -> str:
        """
        Generate a concise text summary of the docking model.

        Returns
        -------
        str
            Formatted docking-model summary.
        """

        self.update_statistics()

        separator = DEFAULT_SEPARATOR

        score_text = (
            "Not calculated"
            if self.score is None
            else f"{self.score:.{DEFAULT_FLOAT_PRECISION}f}"
        )

        lines = [
            separator,
            f"DockModel: {self.name}",
            separator,
            f"Model ID:             {self.model_id}",
            f"Contacts:             {len(self.contacts)}",
            f"Hydrogen bonds:       {len(self.hbonds)}",
            f"Hydrophobic contacts: {len(self.hydrophobic)}",
            (
                "Pi stacking:          "
                f"{len(self.pi.get('stacking', []))}"
            ),
            (
                "Pi-cation:             "
                f"{len(self.pi.get('cation', []))}"
            ),
            f"Total interactions:   {self.interaction_count}",
            f"Score:                {score_text}",
            separator,
        ]

        return "\n".join(lines)

    def print_summary(
        self,
        logger_instance: Optional[DockLogger] = None,
    ) -> str:
        """
        Display and return the docking-model summary.

        Parameters
        ----------
        logger_instance : DockLogger, optional
            Logger used to display the summary. When omitted, the summary is
            printed directly.

        Returns
        -------
        str
            Formatted summary.
        """

        summary_text = self.summary()

        if logger_instance is not None:
            logger_instance.info(
                "\n%s",
                summary_text,
            )
        else:
            print(summary_text)

        return summary_text

    # -------------------------------------------------------------------------
    # Special methods
    # -------------------------------------------------------------------------

    def __len__(self) -> int:
        """
        Return the total number of stored interactions.
        """

        return self.interaction_count

    def __bool__(self) -> bool:
        """
        Return whether the DockModel contains a pose.
        """

        return self.pose is not None

    def __str__(self) -> str:
        """
        Return a concise DockModel representation.
        """

        score_text = (
            "None"
            if self.score is None
            else f"{self.score:.{DEFAULT_FLOAT_PRECISION}f}"
        )

        return (
            f"DockModel("
            f"name='{self.name}', "
            f"model_id={self.model_id}, "
            f"interactions={self.interaction_count}, "
            f"score={score_text}"
            f")"
        )


# =============================================================================
# End of Section 4
# =============================================================================


# =============================================================================
# Section 5 — Model Identification and DockModel Creation
# =============================================================================


# -----------------------------------------------------------------------------
# Molecular classification constants
# -----------------------------------------------------------------------------

STANDARD_AMINO_ACIDS = {
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
}

COMMON_AMINO_ACID_VARIANTS = {
    "ASH",
    "CYM",
    "CYX",
    "GLH",
    "HID",
    "HIE",
    "HIP",
    "LYN",
    "MSE",
    "SEC",
    "PYL",
}

STANDARD_NUCLEIC_ACIDS = {
    "A",
    "C",
    "G",
    "U",
    "T",
    "DA",
    "DC",
    "DG",
    "DT",
    "DU",
    "ADE",
    "CYT",
    "GUA",
    "THY",
    "URA",
}

COMMON_SOLVENT_RESIDUES = {
    "HOH",
    "WAT",
    "H2O",
    "TIP",
    "TIP3",
    "TIP3P",
    "SOL",
}

COMMON_ION_RESIDUES = {
    "NA",
    "K",
    "CL",
    "CA",
    "MG",
    "ZN",
    "FE",
    "MN",
    "CU",
    "CO",
    "NI",
    "BR",
    "IOD",
}

DEFAULT_MIN_RECEPTOR_ATOMS = 200
DEFAULT_MIN_PROTEIN_RESIDUES = 20
DEFAULT_MAX_LIGAND_ATOMS = 250
DEFAULT_MIN_LIGAND_ATOMS = 2
DEFAULT_RECEPTOR_SCORE_THRESHOLD = 3.0


# -----------------------------------------------------------------------------
# Basic model access
# -----------------------------------------------------------------------------

def get_model_name(
    model: Any,
    default: str = "unnamed_model",
) -> str:
    """
    Return a readable model name.

    Parameters
    ----------
    model : Any
        ChimeraX model or model-like object.
    default : str, optional
        Name returned when the model does not provide a valid name.

    Returns
    -------
    str
        Model name.
    """

    if model is None:
        return default

    name = getattr(
        model,
        "name",
        None,
    )

    if name is None:
        return default

    name = str(name).strip()

    return name or default


def get_model_id(
    model: Any,
) -> Optional[str]:
    """
    Return the ChimeraX identifier of a model.

    Parameters
    ----------
    model : Any
        ChimeraX model.

    Returns
    -------
    str or None
        Model identifier without the leading ``#``.
    """

    if model is None:
        return None

    id_string = getattr(
        model,
        "id_string",
        None,
    )

    if id_string not in {
        None,
        "",
    }:
        return str(id_string)

    model_id = getattr(
        model,
        "id",
        None,
    )

    if model_id is None:
        return None

    if isinstance(
        model_id,
        tuple,
    ):
        return ".".join(
            str(value)
            for value in model_id
        )

    return str(model_id)


def get_model_atomspec(
    model: Any,
) -> Optional[str]:
    """
    Return the ChimeraX atom-specifier string for a model.

    Parameters
    ----------
    model : Any
        ChimeraX model.

    Returns
    -------
    str or None
        Atom-specifier string, normally beginning with ``#``.
    """

    if model is None:
        return None

    atomspec = getattr(
        model,
        "atomspec",
        None,
    )

    if atomspec:
        return str(atomspec)

    model_id = get_model_id(model)

    if model_id:
        return f"#{model_id}"

    return None


def _safe_collection_length(
    collection: Any,
) -> int:
    """
    Safely determine the length of a ChimeraX collection.
    """

    if collection is None:
        return 0

    try:
        return len(collection)
    except (TypeError, AttributeError):
        pass

    try:
        return int(collection.num_atoms)
    except (TypeError, ValueError, AttributeError):
        pass

    try:
        return sum(
            1
            for _ in collection
        )
    except TypeError:
        return 0


def get_model_atoms(
    model: Any,
) -> Any:
    """
    Return the atom collection associated with a model.

    Parameters
    ----------
    model : Any
        ChimeraX atomic model.

    Returns
    -------
    Any
        ChimeraX atom collection or an empty tuple.
    """

    if model is None:
        return ()

    atoms = getattr(
        model,
        "atoms",
        None,
    )

    if atoms is None:
        return ()

    return atoms


def get_model_residues(
    model: Any,
) -> Any:
    """
    Return the residue collection associated with a model.

    Parameters
    ----------
    model : Any
        ChimeraX atomic model.

    Returns
    -------
    Any
        ChimeraX residue collection or an empty tuple.
    """

    if model is None:
        return ()

    residues = getattr(
        model,
        "residues",
        None,
    )

    if residues is not None:
        return residues

    atoms = get_model_atoms(model)

    residues = getattr(
        atoms,
        "unique_residues",
        None,
    )

    if residues is not None:
        return residues

    return ()


def get_atom_count(
    model: Any,
) -> int:
    """
    Return the number of atoms in a model.
    """

    return _safe_collection_length(
        get_model_atoms(model)
    )


def get_residue_count(
    model: Any,
) -> int:
    """
    Return the number of residues in a model.
    """

    return _safe_collection_length(
        get_model_residues(model)
    )


def get_residue_name(
    residue: Any,
) -> str:
    """
    Return a normalized residue name.
    """

    if residue is None:
        return ""

    name = getattr(
        residue,
        "name",
        "",
    )

    return str(name).strip().upper()


def get_residue_names(
    model: Any,
) -> List[str]:
    """
    Return the normalized residue names contained in a model.

    Parameters
    ----------
    model : Any
        ChimeraX atomic model.

    Returns
    -------
    list of str
        Residue names.
    """

    residues = get_model_residues(model)

    names: List[str] = []

    try:
        for residue in residues:
            name = get_residue_name(
                residue
            )

            if name:
                names.append(name)

    except TypeError:
        return names

    return names


def is_atomic_model(
    model: Any,
) -> bool:
    """
    Return whether an object appears to be an atomic structure.

    The test uses attribute inspection instead of importing ChimeraX classes,
    allowing ``utils.py`` to remain importable outside ChimeraX.
    """

    if model is None:
        return False

    return (
        hasattr(model, "atoms")
        and get_atom_count(model) > 0
    )


# -----------------------------------------------------------------------------
# Model composition
# -----------------------------------------------------------------------------

def count_protein_residues(
    model: Any,
) -> int:
    """
    Count standard or common amino-acid residues in a model.
    """

    valid_names = (
        STANDARD_AMINO_ACIDS
        | COMMON_AMINO_ACID_VARIANTS
    )

    return sum(
        name in valid_names
        for name in get_residue_names(model)
    )


def count_nucleic_acid_residues(
    model: Any,
) -> int:
    """
    Count nucleic-acid residues in a model.
    """

    return sum(
        name in STANDARD_NUCLEIC_ACIDS
        for name in get_residue_names(model)
    )


def count_solvent_residues(
    model: Any,
) -> int:
    """
    Count common solvent residues in a model.
    """

    return sum(
        name in COMMON_SOLVENT_RESIDUES
        for name in get_residue_names(model)
    )


def count_ion_residues(
    model: Any,
) -> int:
    """
    Count common ion residues in a model.
    """

    return sum(
        name in COMMON_ION_RESIDUES
        for name in get_residue_names(model)
    )


def get_polymer_residue_count(
    model: Any,
) -> int:
    """
    Return the approximate number of polymer residues.

    Protein and nucleic-acid residues are considered polymers.
    """

    return (
        count_protein_residues(model)
        + count_nucleic_acid_residues(model)
    )


def get_protein_fraction(
    model: Any,
) -> float:
    """
    Return the fraction of residues identified as amino acids.
    """

    residue_count = get_residue_count(
        model
    )

    if residue_count == 0:
        return 0.0

    return (
        count_protein_residues(model)
        / residue_count
    )


def model_contains_protein(
    model: Any,
    minimum_residues: int = 1,
) -> bool:
    """
    Return whether a model contains protein residues.
    """

    return (
        count_protein_residues(model)
        >= int(minimum_residues)
    )


# -----------------------------------------------------------------------------
# Classification
# -----------------------------------------------------------------------------

def _collect_model_composition(
    model: Any,
) -> Tuple[int, int, int, int, int, int]:
    """Collect model size and residue categories in one pass."""

    atom_count = get_atom_count(model)
    residue_count = get_residue_count(model)
    residue_names = get_residue_names(model)
    protein_names = (
        STANDARD_AMINO_ACIDS
        | COMMON_AMINO_ACID_VARIANTS
    )

    protein_count = 0
    nucleic_count = 0
    solvent_count = 0
    ion_count = 0

    for residue_name in residue_names:
        if residue_name in protein_names:
            protein_count += 1

        if residue_name in STANDARD_NUCLEIC_ACIDS:
            nucleic_count += 1

        if residue_name in COMMON_SOLVENT_RESIDUES:
            solvent_count += 1

        if residue_name in COMMON_ION_RESIDUES:
            ion_count += 1

    return (
        atom_count,
        residue_count,
        protein_count,
        nucleic_count,
        solvent_count,
        ion_count,
    )


def _calculate_receptor_score_from_composition(
    composition: Tuple[int, int, int, int, int, int],
    *,
    min_receptor_atoms: int,
    min_protein_residues: int,
) -> float:
    """Calculate a receptor score from precomputed composition."""

    (
        atom_count,
        residue_count,
        protein_count,
        nucleic_count,
        _,
        _,
    ) = composition

    if atom_count <= 0:
        return float("-inf")

    polymer_count = protein_count + nucleic_count
    score = 0.0

    if atom_count >= min_receptor_atoms:
        score += 2.0

    if atom_count >= 1000:
        score += 1.0

    if protein_count >= min_protein_residues:
        score += 3.0

    if polymer_count >= min_protein_residues:
        score += 1.0

    if residue_count >= min_protein_residues:
        score += 1.0

    if residue_count > 0:
        polymer_fraction = polymer_count / residue_count

        if polymer_fraction >= 0.50:
            score += 1.0

        if polymer_fraction >= 0.80:
            score += 1.0

    if atom_count <= DEFAULT_MAX_LIGAND_ATOMS:
        score -= 2.0

    if polymer_count == 0:
        score -= 3.0

    return score


def _calculate_ligand_score_from_composition(
    composition: Tuple[int, int, int, int, int, int],
    *,
    min_ligand_atoms: int,
    max_ligand_atoms: int,
) -> float:
    """Calculate a ligand score from precomputed composition."""

    (
        atom_count,
        residue_count,
        protein_count,
        nucleic_count,
        solvent_count,
        ion_count,
    ) = composition

    if atom_count <= 0:
        return float("-inf")

    polymer_count = protein_count + nucleic_count
    score = 0.0

    if min_ligand_atoms <= atom_count <= max_ligand_atoms:
        score += 3.0

    if atom_count <= 100:
        score += 1.0

    if residue_count == 1:
        score += 2.0
    elif 1 < residue_count <= 10:
        score += 1.0

    if polymer_count == 0:
        score += 2.0
    else:
        score -= 3.0

    if residue_count > 0 and solvent_count == residue_count:
        score -= 5.0

    if residue_count > 0 and ion_count == residue_count:
        score -= 4.0

    if atom_count > max_ligand_atoms:
        score -= 2.0

    return score


def calculate_receptor_score(
    model: Any,
    min_receptor_atoms: int = DEFAULT_MIN_RECEPTOR_ATOMS,
    min_protein_residues: int = DEFAULT_MIN_PROTEIN_RESIDUES,
) -> float:
    """
    Calculate a heuristic receptor-classification score.

    Higher scores indicate a greater probability that the model is a
    macromolecular receptor.
    """

    if model is None or not hasattr(model, "atoms"):
        return float("-inf")

    return _calculate_receptor_score_from_composition(
        _collect_model_composition(model),
        min_receptor_atoms=min_receptor_atoms,
        min_protein_residues=min_protein_residues,
    )


def calculate_ligand_score(
    model: Any,
    min_ligand_atoms: int = DEFAULT_MIN_LIGAND_ATOMS,
    max_ligand_atoms: int = DEFAULT_MAX_LIGAND_ATOMS,
) -> float:
    """Calculate a heuristic ligand or pose classification score."""

    if model is None or not hasattr(model, "atoms"):
        return float("-inf")

    return _calculate_ligand_score_from_composition(
        _collect_model_composition(model),
        min_ligand_atoms=min_ligand_atoms,
        max_ligand_atoms=max_ligand_atoms,
    )


def classify_model(
    model: Any,
    min_receptor_atoms: int = DEFAULT_MIN_RECEPTOR_ATOMS,
    min_protein_residues: int = DEFAULT_MIN_PROTEIN_RESIDUES,
    max_ligand_atoms: int = DEFAULT_MAX_LIGAND_ATOMS,
) -> str:
    """Classify a ChimeraX model as receptor, ligand or auxiliary data."""

    if model is None or not hasattr(model, "atoms"):
        return "non_atomic"

    composition = _collect_model_composition(model)
    (
        atom_count,
        residue_count,
        _,
        _,
        solvent_count,
        ion_count,
    ) = composition

    if atom_count <= 0:
        return "non_atomic"

    if residue_count > 0 and solvent_count == residue_count:
        return "solvent"

    if residue_count > 0 and ion_count == residue_count:
        return "ion"

    receptor_score = _calculate_receptor_score_from_composition(
        composition,
        min_receptor_atoms=min_receptor_atoms,
        min_protein_residues=min_protein_residues,
    )
    ligand_score = _calculate_ligand_score_from_composition(
        composition,
        min_ligand_atoms=DEFAULT_MIN_LIGAND_ATOMS,
        max_ligand_atoms=max_ligand_atoms,
    )

    if (
        receptor_score >= DEFAULT_RECEPTOR_SCORE_THRESHOLD
        and receptor_score > ligand_score
    ):
        return "receptor"

    if ligand_score > 0:
        return "ligand"

    return "unknown"


def describe_model(
    model: Any,
) -> Dict[str, Any]:
    """Generate a model-classification report."""

    composition = _collect_model_composition(model)
    (
        atom_count,
        residue_count,
        protein_count,
        nucleic_count,
        solvent_count,
        ion_count,
    ) = composition
    polymer_count = protein_count + nucleic_count

    if model is None or not hasattr(model, "atoms") or atom_count <= 0:
        receptor_score = float("-inf")
        ligand_score = float("-inf")
        classification = "non_atomic"
    else:
        receptor_score = _calculate_receptor_score_from_composition(
            composition,
            min_receptor_atoms=DEFAULT_MIN_RECEPTOR_ATOMS,
            min_protein_residues=DEFAULT_MIN_PROTEIN_RESIDUES,
        )
        ligand_score = _calculate_ligand_score_from_composition(
            composition,
            min_ligand_atoms=DEFAULT_MIN_LIGAND_ATOMS,
            max_ligand_atoms=DEFAULT_MAX_LIGAND_ATOMS,
        )

        if residue_count > 0 and solvent_count == residue_count:
            classification = "solvent"
        elif residue_count > 0 and ion_count == residue_count:
            classification = "ion"
        elif (
            receptor_score >= DEFAULT_RECEPTOR_SCORE_THRESHOLD
            and receptor_score > ligand_score
        ):
            classification = "receptor"
        elif ligand_score > 0:
            classification = "ligand"
        else:
            classification = "unknown"

    return {
        "name": get_model_name(model),
        "model_id": get_model_id(model),
        "atomspec": get_model_atomspec(model),
        "atom_count": atom_count,
        "residue_count": residue_count,
        "protein_residue_count": protein_count,
        "nucleic_acid_residue_count": nucleic_count,
        "polymer_residue_count": polymer_count,
        "receptor_score": receptor_score,
        "ligand_score": ligand_score,
        "classification": classification,
    }


# -----------------------------------------------------------------------------
# ChimeraX session models
# -----------------------------------------------------------------------------

def get_atomic_models(
    session: Any,
) -> List[Any]:
    """
    Return all atomic models currently open in a ChimeraX session.

    Parameters
    ----------
    session : Any
        Active ChimeraX session.

    Returns
    -------
    list
        Open atomic models.

    Raises
    ------
    ValueError
        If no valid ChimeraX session is provided.
    """

    if session is None:
        raise ValueError(
            "A valid ChimeraX session is required."
        )

    model_manager = getattr(
        session,
        "models",
        None,
    )

    if model_manager is None:
        raise ValueError(
            "The provided object does not contain "
            "a ChimeraX model manager."
        )

    models: List[Any] = []

    list_method = getattr(
        model_manager,
        "list",
        None,
    )

    if callable(list_method):
        try:
            models = list(
                list_method()
            )
        except TypeError:
            try:
                from chimerax.atomic import AtomicStructure
            except ModuleNotFoundError as exc:
                if exc.name != "chimerax":
                    raise
                models = []
            else:
                try:
                    models = list(
                        list_method(
                            type=AtomicStructure,
                        )
                    )
                except TypeError:
                    models = []

    if not models:
        try:
            models = list(model_manager)
        except TypeError:
            models = []

    return [
        model
        for model in models
        if is_atomic_model(model)
    ]


def _matches_model_reference(
    model: Any,
    reference: str,
) -> bool:
    """
    Return whether a model matches an ID, atom specifier or name.
    """

    normalized_reference = (
        str(reference)
        .strip()
    )

    normalized_without_hash = (
        normalized_reference
        .lstrip("#")
    )

    model_name = get_model_name(
        model
    )

    model_id = get_model_id(
        model
    )

    atomspec = get_model_atomspec(
        model
    )

    return any(
        (
            normalized_reference == model_name,
            normalized_reference.lower() == model_name.lower(),
            normalized_without_hash == model_id,
            normalized_reference == atomspec,
        )
    )


def resolve_model_reference(
    models: List[Any],
    reference: Any,
) -> Any:
    """
    Resolve a model object, model ID, atom specifier or model name.

    Parameters
    ----------
    models : list
        Available ChimeraX atomic models.
    reference : Any
        Model object, model ID, ``#`` atom specifier or model name.

    Returns
    -------
    Any
        Resolved ChimeraX model.

    Raises
    ------
    ValueError
        If the reference cannot be resolved or is ambiguous.
    """

    if reference is None:
        return None

    for model in models:
        if reference is model:
            return model

    matched_models = [
        model
        for model in models
        if _matches_model_reference(
            model,
            str(reference),
        )
    ]

    if len(matched_models) == 1:
        return matched_models[0]

    if not matched_models:
        raise ValueError(
            f"Model reference '{reference}' was not found."
        )

    matched_text = ", ".join(
        get_model_atomspec(model)
        or get_model_name(model)
        for model in matched_models
    )

    raise ValueError(
        f"Model reference '{reference}' is ambiguous. "
        f"Matches: {matched_text}."
    )


# -----------------------------------------------------------------------------
# Receptor and pose identification
# -----------------------------------------------------------------------------

def identify_receptor(
    models: List[Any],
    receptor: Optional[Any] = None,
    strict: bool = True,
) -> Any:
    """
    Identify the receptor among the available atomic models.

    An explicit receptor can be supplied as:

    - a ChimeraX model object;
    - a model ID such as ``"1"`` or ``"#1"``;
    - an atom specifier;
    - a model name.

    When no explicit receptor is provided, classification scores are used.

    Parameters
    ----------
    models : list
        Available ChimeraX atomic models.
    receptor : Any, optional
        Explicit receptor reference.
    strict : bool, optional
        Whether ambiguous automatic detection should raise an exception.

    Returns
    -------
    Any
        Identified receptor model.

    Raises
    ------
    ValueError
        If no receptor is found or automatic detection is ambiguous.
    """

    if not models:
        raise ValueError(
            "No atomic models are available."
        )

    if receptor is not None:
        resolved_receptor = resolve_model_reference(
            models=models,
            reference=receptor,
        )

        if not is_atomic_model(
            resolved_receptor
        ):
            raise ValueError(
                "The selected receptor is not an atomic model."
            )

        return resolved_receptor

    scored_models = sorted(
        (
            (
                calculate_receptor_score(model),
                get_atom_count(model),
                model,
            )
            for model in models
        ),
        key=lambda item: (
            item[0],
            item[1],
        ),
        reverse=True,
    )

    valid_candidates = [
        item
        for item in scored_models
        if item[0]
        >= DEFAULT_RECEPTOR_SCORE_THRESHOLD
    ]

    if not valid_candidates:
        model_report = "; ".join(
            (
                f"{get_model_atomspec(model) or '?'} "
                f"{get_model_name(model)} "
                f"(score={score:.2f})"
            )
            for score, _, model in scored_models
        )

        raise ValueError(
            "No receptor could be identified automatically. "
            f"Available models: {model_report}"
        )

    best_score, best_atom_count, best_model = (
        valid_candidates[0]
    )

    if (
        strict
        and len(valid_candidates) > 1
    ):
        second_score, second_atom_count, second_model = (
            valid_candidates[1]
        )

        scores_are_similar = (
            abs(
                best_score
                - second_score
            )
            < 1.0
        )

        sizes_are_similar = (
            best_atom_count > 0
            and (
                second_atom_count
                / best_atom_count
            )
            >= 0.70
        )

        if (
            scores_are_similar
            and sizes_are_similar
        ):
            raise ValueError(
                "More than one probable receptor was detected: "
                f"{get_model_atomspec(best_model)} "
                f"{get_model_name(best_model)} and "
                f"{get_model_atomspec(second_model)} "
                f"{get_model_name(second_model)}. "
                "Specify the receptor explicitly."
            )

    return best_model


def identify_pose_models(
    models: List[Any],
    receptor: Any,
    poses: Optional[Any] = None,
    include_unknown: bool = False,
) -> List[Any]:
    """
    Identify docking-pose models.

    Parameters
    ----------
    models : list
        Available ChimeraX atomic models.
    receptor : Any
        Receptor model that must be excluded.
    poses : Any, optional
        Explicit pose reference or iterable of references.
    include_unknown : bool, optional
        Whether small unclassified models should also be accepted.

    Returns
    -------
    list
        Models classified as docking poses.

    Raises
    ------
    ValueError
        If explicit pose references cannot be resolved or no poses are found.
    """

    if poses is not None:
        if isinstance(
            poses,
            (
                str,
                int,
            ),
        ):
            pose_references = [poses]

        else:
            try:
                pose_references = list(
                    poses
                )
            except TypeError:
                pose_references = [poses]

        resolved_poses: List[Any] = []

        for reference in pose_references:
            pose_model = resolve_model_reference(
                models=models,
                reference=reference,
            )

            if pose_model is receptor:
                raise ValueError(
                    "The receptor cannot also be used as a pose."
                )

            if pose_model not in resolved_poses:
                resolved_poses.append(
                    pose_model
                )

        if not resolved_poses:
            raise ValueError(
                "No valid pose models were provided."
            )

        return resolved_poses

    detected_poses: List[Any] = []

    for model in models:
        if model is receptor:
            continue

        classification = classify_model(
            model
        )

        if classification == "ligand":
            detected_poses.append(model)
            continue

        if (
            include_unknown
            and classification == "unknown"
            and (
                DEFAULT_MIN_LIGAND_ATOMS
                <= get_atom_count(model)
                <= DEFAULT_MAX_LIGAND_ATOMS
            )
        ):
            detected_poses.append(model)

    if not detected_poses:
        raise ValueError(
            "No docking poses were identified. "
            "Open the pose models or specify them explicitly."
        )

    return detected_poses


# -----------------------------------------------------------------------------
# Ligand extraction
# -----------------------------------------------------------------------------

def get_ligand_from_pose(
    pose_model: Any,
) -> Any:
    """
    Extract the ligand representation from a docking-pose model.

    For a single-residue pose, the corresponding ChimeraX residue is returned.
    For poses containing multiple residues, the complete pose model is
    returned to avoid discarding atoms.

    Parameters
    ----------
    pose_model : Any
        ChimeraX model representing a docking pose.

    Returns
    -------
    Any
        ChimeraX residue or complete pose model.
    """

    if pose_model is None:
        raise ValueError(
            "A pose model is required."
        )

    residues = get_model_residues(
        pose_model
    )

    residue_count = _safe_collection_length(
        residues
    )

    if residue_count == 1:
        try:
            return residues[0]
        except (
            TypeError,
            IndexError,
            KeyError,
        ):
            try:
                return next(
                    iter(residues)
                )
            except (
                TypeError,
                StopIteration,
            ):
                pass

    return pose_model


def generate_dock_model_name(
    pose_model: Any,
    index: int,
) -> str:
    """
    Generate a stable name for a DockModel.
    """

    pose_name = get_model_name(
        pose_model,
        default="pose",
    )

    pose_id = get_model_id(
        pose_model
    )

    cleaned_name = (
        pose_name
        .strip()
        .replace(" ", "_")
    )

    if pose_id:
        return (
            f"{cleaned_name}_model_{pose_id}"
        )

    return (
        f"{cleaned_name}_{index:03d}"
    )


# -----------------------------------------------------------------------------
# DockModel factory
# -----------------------------------------------------------------------------

def create_dock_models(
    session: Any,
    receptor: Optional[Any] = None,
    poses: Optional[Any] = None,
    logger: Optional[DockLogger] = None,
    strict: bool = True,
    include_unknown_poses: bool = False,
) -> List[DockModel]:
    """
    Identify receptor and docking poses and create DockModel instances.

    This function is the main entry point for automatic model discovery.

    Parameters
    ----------
    session : Any
        Active ChimeraX session.
    receptor : Any, optional
        Explicit receptor object, ID, atom specifier or name. When omitted,
        the receptor is identified automatically.
    poses : Any, optional
        Explicit pose reference or iterable of pose references. When omitted,
        pose models are identified automatically.
    logger : DockLogger, optional
        Logger used to report model identification.
    strict : bool, optional
        Whether ambiguous receptor detection should raise an exception.
    include_unknown_poses : bool, optional
        Whether small models classified as unknown should be accepted as
        docking poses.

    Returns
    -------
    list of DockModel
        One DockModel for each identified pose.

    Raises
    ------
    ValueError
        If receptor or poses cannot be identified.

    Examples
    --------
    Fully automatic discovery:

    >>> dock_models = create_dock_models(session)

    Explicit receptor:

    >>> dock_models = create_dock_models(
    ...     session,
    ...     receptor="#1",
    ... )

    Explicit receptor and poses:

    >>> dock_models = create_dock_models(
    ...     session,
    ...     receptor="#1",
    ...     poses=["#2", "#3", "#4"],
    ... )
    """

    atomic_models = get_atomic_models(
        session
    )

    if not atomic_models:
        raise ValueError(
            "No atomic models are open in the ChimeraX session."
        )

    receptor_model = identify_receptor(
        models=atomic_models,
        receptor=receptor,
        strict=strict,
    )

    pose_models = identify_pose_models(
        models=atomic_models,
        receptor=receptor_model,
        poses=poses,
        include_unknown=include_unknown_poses,
    )

    if logger is not None:
        logger.info(
            "Receptor identified: %s (%s; %d atoms; %d residues).",
            get_model_name(receptor_model),
            get_model_atomspec(receptor_model),
            get_atom_count(receptor_model),
            get_residue_count(receptor_model),
        )

        logger.info(
            "%d docking pose(s) identified.",
            len(pose_models),
        )

    dock_models: List[DockModel] = []

    receptor_description = describe_model(
        receptor_model
    )

    for index, pose_model in enumerate(
        pose_models,
        start=1,
    ):
        ligand = get_ligand_from_pose(
            pose_model
        )

        model_name = generate_dock_model_name(
            pose_model=pose_model,
            index=index,
        )

        pose_description = describe_model(
            pose_model
        )

        dock_model = DockModel(
            name=model_name,
            pose=pose_model,
            receptor=receptor_model,
            ligand=ligand,
            metadata={
                "pose_index": index,
                "pose_name": get_model_name(
                    pose_model
                ),
                "pose_id": get_model_id(
                    pose_model
                ),
                "pose_atomspec": get_model_atomspec(
                    pose_model
                ),
                "pose_atom_count": get_atom_count(
                    pose_model
                ),
                "pose_residue_count": get_residue_count(
                    pose_model
                ),
                "pose_classification": pose_description[
                    "classification"
                ],
                "pose_ligand_score": pose_description[
                    "ligand_score"
                ],
                "receptor_name": get_model_name(
                    receptor_model
                ),
                "receptor_id": get_model_id(
                    receptor_model
                ),
                "receptor_atomspec": get_model_atomspec(
                    receptor_model
                ),
                "receptor_atom_count": get_atom_count(
                    receptor_model
                ),
                "receptor_residue_count": get_residue_count(
                    receptor_model
                ),
                "receptor_classification": (
                    receptor_description[
                        "classification"
                    ]
                ),
                "automatic_model_detection": (
                    receptor is None
                    and poses is None
                ),
            },
        )

        dock_models.append(
            dock_model
        )

        if logger is not None:
            logger.info(
                "DockModel created: %s | pose=%s | atoms=%d.",
                dock_model.name,
                get_model_atomspec(pose_model),
                get_atom_count(pose_model),
            )

    return dock_models


# -----------------------------------------------------------------------------
# Compatibility model-access API
# -----------------------------------------------------------------------------

def _resolve_atomic_model_source(
    source: Any,
) -> List[Any]:
    """Return atomic models from a session or model iterable."""

    if source is None:
        raise ValueError(
            "A ChimeraX session or model collection is required."
        )

    if getattr(
        source,
        "models",
        None,
    ) is not None:
        return get_atomic_models(
            source
        )

    try:
        models = list(
            source
        )

    except TypeError as error:
        raise TypeError(
            "Model source must be a ChimeraX session or iterable of models."
        ) from error

    return [
        model
        for model in models
        if is_atomic_model(
            model
        )
    ]


def get_receptor(
    source: Any,
    receptor: Optional[Any] = None,
    strict: bool = True,
) -> Any:
    """Return the receptor from a session or atomic-model collection.

    This compatibility wrapper delegates to :func:`identify_receptor`.
    """

    models = _resolve_atomic_model_source(
        source
    )

    return identify_receptor(
        models=models,
        receptor=receptor,
        strict=strict,
    )


def get_pose_models(
    source: Any,
    receptor: Optional[Any] = None,
    poses: Optional[Any] = None,
    include_unknown: bool = False,
    strict: bool = True,
) -> List[Any]:
    """Return docking-pose models from a session or model collection.

    When ``receptor`` is omitted, it is identified automatically.
    """

    models = _resolve_atomic_model_source(
        source
    )

    receptor_model = identify_receptor(
        models=models,
        receptor=receptor,
        strict=strict,
    )

    return identify_pose_models(
        models=models,
        receptor=receptor_model,
        poses=poses,
        include_unknown=include_unknown,
    )


def get_ligand(
    pose_model: Any,
) -> Any:
    """Return the ligand representation from a docking-pose model."""

    return get_ligand_from_pose(
        pose_model
    )


# -----------------------------------------------------------------------------
# Model discovery report
# -----------------------------------------------------------------------------

def model_discovery_summary(
    session: Any,
) -> str:
    """
    Generate a summary of all atomic models open in ChimeraX.

    Parameters
    ----------
    session : Any
        Active ChimeraX session.

    Returns
    -------
    str
        Formatted model-classification table.
    """

    models = get_atomic_models(
        session
    )

    separator = DEFAULT_SEPARATOR

    lines = [
        separator,
        "DockAnalyzer Model Discovery",
        separator,
    ]

    if not models:
        lines.append(
            "No atomic models are open."
        )
        lines.append(separator)

        return "\n".join(lines)

    header = (
        f"{'Model':<10}"
        f"{'Name':<28}"
        f"{'Atoms':>10}"
        f"{'Residues':>12}"
        f"{'Protein':>11}"
        f"{'Class':>14}"
    )

    lines.append(header)
    lines.append(
        "-" * len(header)
    )

    for model in models:
        description = describe_model(
            model
        )

        atomspec = (
            description["atomspec"]
            or "?"
        )

        model_name = (
            description["name"][:26]
        )

        lines.append(
            f"{atomspec:<10}"
            f"{model_name:<28}"
            f"{description['atom_count']:>10}"
            f"{description['residue_count']:>12}"
            f"{description['protein_residue_count']:>11}"
            f"{description['classification']:>14}"
        )

    lines.append(separator)

    return "\n".join(lines)


def print_model_discovery(
    session: Any,
    logger: Optional[DockLogger] = None,
) -> str:
    """
    Display and return the model-discovery summary.
    """

    summary_text = model_discovery_summary(
        session
    )

    if logger is not None:
        logger.info(
            "\n%s",
            summary_text,
        )
    else:
        print(summary_text)

    return summary_text


# -----------------------------------------------------------------------------
# Public module interface
# -----------------------------------------------------------------------------

_SECTION_5_PUBLIC_NAMES = [
    "STANDARD_AMINO_ACIDS",
    "COMMON_AMINO_ACID_VARIANTS",
    "STANDARD_NUCLEIC_ACIDS",
    "COMMON_SOLVENT_RESIDUES",
    "COMMON_ION_RESIDUES",
    "get_model_name",
    "get_model_id",
    "get_model_atomspec",
    "get_model_atoms",
    "get_model_residues",
    "get_atom_count",
    "get_residue_count",
    "get_residue_names",
    "get_residue_name",
    "is_atomic_model",
    "count_protein_residues",
    "count_nucleic_acid_residues",
    "count_solvent_residues",
    "count_ion_residues",
    "get_polymer_residue_count",
    "get_protein_fraction",
    "model_contains_protein",
    "calculate_receptor_score",
    "calculate_ligand_score",
    "classify_model",
    "describe_model",
    "get_atomic_models",
    "resolve_model_reference",
    "identify_receptor",
    "identify_pose_models",
    "get_ligand_from_pose",
    "generate_dock_model_name",
    "create_dock_models",
    "get_receptor",
    "get_pose_models",
    "get_ligand",
    "model_discovery_summary",
    "print_model_discovery",
]

for public_name in _SECTION_5_PUBLIC_NAMES:
    if public_name not in __all__:
        __all__.append(public_name)


# =============================================================================
# End of Section 5
# =============================================================================


# =============================================================================
# Section 6 — Basic Geometry
# =============================================================================


# -----------------------------------------------------------------------------
# Coordinate conversion
# -----------------------------------------------------------------------------

def _as_coordinate_array(
    value: Any,
    *,
    name: str = "coordinate",
    dimensions: Optional[int] = 3,
    copy: bool = False,
) -> np.ndarray:
    """
    Convert a coordinate-like object into a NumPy array.

    The function accepts:

    - lists and tuples;
    - NumPy arrays;
    - ChimeraX atoms exposing ``coord`` or ``scene_coord``;
    - objects exposing ``coords`` or ``coordinates``;
    - objects exposing numeric ``x``, ``y`` and ``z`` attributes.

    Parameters
    ----------
    value : Any
        Coordinate-like object.
    name : str, optional
        Human-readable name used in validation messages.
    dimensions : int or None, optional
        Expected number of dimensions. Use ``None`` to accept vectors of any
        dimensionality.
    copy : bool, optional
        Whether a copy of the resulting array should be created.

    Returns
    -------
    numpy.ndarray
        One-dimensional floating-point coordinate array.

    Raises
    ------
    TypeError
        If the supplied value cannot be interpreted as coordinates.
    ValueError
        If the coordinates have an invalid shape or contain non-finite values.
    """

    if value is None:
        raise TypeError(
            f"{name.capitalize()} cannot be None."
        )

    coordinate_value = value

    for attribute_name in (
        "scene_coord",
        "coord",
        "coordinates",
        "coords",
    ):
        attribute_value = getattr(
            value,
            attribute_name,
            None,
        )

        if attribute_value is not None:
            coordinate_value = attribute_value
            break

    else:
        if all(
            hasattr(value, attribute_name)
            for attribute_name in (
                "x",
                "y",
                "z",
            )
        ):
            coordinate_value = [
                getattr(value, "x"),
                getattr(value, "y"),
                getattr(value, "z"),
            ]

    try:
        coordinate_array = np.asarray(
            coordinate_value,
            dtype=float,
        )

    except (
        TypeError,
        ValueError,
    ) as error:
        raise TypeError(
            f"{name.capitalize()} could not be converted "
            "to numeric coordinates."
        ) from error

    coordinate_array = np.squeeze(
        coordinate_array
    )

    if coordinate_array.ndim != 1:
        raise ValueError(
            f"{name.capitalize()} must represent one vector; "
            f"received shape {coordinate_array.shape}."
        )

    if dimensions is not None and (
        coordinate_array.size != dimensions
    ):
        raise ValueError(
            f"{name.capitalize()} must contain exactly "
            f"{dimensions} values; received "
            f"{coordinate_array.size}."
        )

    if coordinate_array.size == 0:
        raise ValueError(
            f"{name.capitalize()} cannot be empty."
        )

    if not np.all(
        np.isfinite(coordinate_array)
    ):
        raise ValueError(
            f"{name.capitalize()} contains NaN or "
            "infinite values."
        )

    if copy:
        coordinate_array = coordinate_array.copy()

    return coordinate_array


def _as_coordinate_matrix(
    values: Any,
    *,
    name: str = "coordinates",
    dimensions: Optional[int] = 3,
) -> np.ndarray:
    """
    Convert multiple coordinate-like objects into a coordinate matrix.

    Parameters
    ----------
    values : Any
        Iterable containing coordinate vectors or ChimeraX atom-like objects.
    name : str, optional
        Human-readable name used in validation messages.
    dimensions : int or None, optional
        Expected number of dimensions per coordinate.

    Returns
    -------
    numpy.ndarray
        Matrix with shape ``(n_coordinates, n_dimensions)``.

    Raises
    ------
    TypeError
        If the supplied object is not iterable.
    ValueError
        If no valid coordinates are supplied.
    """

    if values is None:
        raise TypeError(
            f"{name.capitalize()} cannot be None."
        )

    direct_value = values

    for attribute_name in (
        "scene_coords",
        "coords",
        "coordinates",
    ):
        attribute_value = getattr(
            values,
            attribute_name,
            None,
        )

        if attribute_value is not None:
            direct_value = attribute_value
            break

    try:
        direct_array = np.asarray(
            direct_value,
            dtype=float,
        )

    except (
        TypeError,
        ValueError,
    ):
        direct_array = None

    if (
        direct_array is not None
        and direct_array.ndim == 2
    ):
        coordinate_matrix = direct_array

    else:
        try:
            coordinate_list = [
                _as_coordinate_array(
                    value,
                    name=f"{name} item {index}",
                    dimensions=dimensions,
                )
                for index, value in enumerate(
                    values,
                    start=1,
                )
            ]

        except TypeError as error:
            raise TypeError(
                f"{name.capitalize()} must be an iterable "
                "of coordinate-like objects."
            ) from error

        if not coordinate_list:
            raise ValueError(
                f"{name.capitalize()} cannot be empty."
            )

        coordinate_matrix = np.vstack(
            coordinate_list
        )

    if coordinate_matrix.shape[0] == 0:
        raise ValueError(
            f"{name.capitalize()} cannot be empty."
        )

    if dimensions is not None and (
        coordinate_matrix.shape[1] != dimensions
    ):
        raise ValueError(
            f"Each item in {name} must contain exactly "
            f"{dimensions} values; received matrix shape "
            f"{coordinate_matrix.shape}."
        )

    if not np.all(
        np.isfinite(coordinate_matrix)
    ):
        raise ValueError(
            f"{name.capitalize()} contains NaN or "
            "infinite values."
        )

    return coordinate_matrix


# -----------------------------------------------------------------------------
# Vector normalization
# -----------------------------------------------------------------------------

def normalize(
    vector: Any,
    *,
    zero_tolerance: float = 1e-12,
) -> np.ndarray:
    """
    Normalize a vector to unit length.

    Parameters
    ----------
    vector : Any
        Vector-like object.
    zero_tolerance : float, optional
        Norm values equal to or below this threshold are treated as zero.

    Returns
    -------
    numpy.ndarray
        Unit vector with the same direction as the original vector.

    Raises
    ------
    ValueError
        If the supplied vector has zero or near-zero length.

    Examples
    --------
    >>> normalize([3.0, 0.0, 0.0])
    array([1., 0., 0.])
    """

    vector_array = _as_coordinate_array(
        vector,
        name="vector",
        dimensions=None,
    )

    vector_norm = float(
        np.linalg.norm(vector_array)
    )

    if vector_norm <= float(
        zero_tolerance
    ):
        raise ValueError(
            "A zero-length vector cannot be normalized."
        )

    return vector_array / vector_norm


# -----------------------------------------------------------------------------
# Distance
# -----------------------------------------------------------------------------

def distance(
    point_a: Any,
    point_b: Any,
) -> float:
    """
    Calculate the Euclidean distance between two points.

    The function can receive coordinate vectors or ChimeraX atoms directly.

    Parameters
    ----------
    point_a : Any
        First point or atom-like object.
    point_b : Any
        Second point or atom-like object.

    Returns
    -------
    float
        Euclidean distance between the points.

    Raises
    ------
    ValueError
        If the points have incompatible dimensions.

    Examples
    --------
    >>> distance([0.0, 0.0, 0.0], [3.0, 4.0, 0.0])
    5.0

    ChimeraX atoms may be passed directly:

    >>> atom_distance = distance(atom_1, atom_2)
    """

    coordinate_a = _as_coordinate_array(
        point_a,
        name="point A",
        dimensions=None,
    )

    coordinate_b = _as_coordinate_array(
        point_b,
        name="point B",
        dimensions=None,
    )

    if coordinate_a.shape != coordinate_b.shape:
        raise ValueError(
            "Point A and point B must have the same "
            f"dimensionality; received {coordinate_a.size} "
            f"and {coordinate_b.size}."
        )

    return float(
        np.linalg.norm(
            coordinate_b
            - coordinate_a
        )
    )


# -----------------------------------------------------------------------------
# Centroid
# -----------------------------------------------------------------------------

def centroid(
    coordinates: Any,
    *,
    weights: Optional[Any] = None,
) -> np.ndarray:
    """
    Calculate the centroid of multiple coordinates.

    By default, the arithmetic centroid is calculated. When weights are
    supplied, a weighted centroid is returned.

    Parameters
    ----------
    coordinates : Any
        Coordinate matrix, iterable of coordinate vectors or iterable of
        ChimeraX atom-like objects.
    weights : Any, optional
        Numeric weight assigned to each coordinate. This may later be used to
        calculate centers of mass using atomic masses.

    Returns
    -------
    numpy.ndarray
        Centroid coordinates.

    Raises
    ------
    ValueError
        If the number of weights does not match the number of coordinates or
        if the total weight is zero.

    Examples
    --------
    Arithmetic centroid:

    >>> centroid([
    ...     [0.0, 0.0, 0.0],
    ...     [2.0, 2.0, 2.0],
    ... ])
    array([1., 1., 1.])

    Weighted centroid:

    >>> centroid(
    ...     [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
    ...     weights=[1.0, 3.0],
    ... )
    array([1.5, 0. , 0. ])
    """

    coordinate_matrix = _as_coordinate_matrix(
        coordinates,
        name="coordinates",
        dimensions=3,
    )

    if weights is None:
        return np.mean(
            coordinate_matrix,
            axis=0,
        )

    try:
        weight_array = np.asarray(
            weights,
            dtype=float,
        )

    except (
        TypeError,
        ValueError,
    ) as error:
        raise TypeError(
            "Weights must contain numeric values."
        ) from error

    weight_array = np.squeeze(
        weight_array
    )

    if weight_array.ndim != 1:
        raise ValueError(
            "Weights must be a one-dimensional sequence."
        )

    if (
        weight_array.size
        != coordinate_matrix.shape[0]
    ):
        raise ValueError(
            "The number of weights must match the number "
            f"of coordinates; received {weight_array.size} "
            f"weights for {coordinate_matrix.shape[0]} "
            "coordinates."
        )

    if not np.all(
        np.isfinite(weight_array)
    ):
        raise ValueError(
            "Weights contain NaN or infinite values."
        )

    total_weight = float(
        np.sum(weight_array)
    )

    if np.isclose(
        total_weight,
        0.0,
    ):
        raise ValueError(
            "The sum of the centroid weights cannot be zero."
        )

    return np.average(
        coordinate_matrix,
        axis=0,
        weights=weight_array,
    )


# -----------------------------------------------------------------------------
# Angle
# -----------------------------------------------------------------------------

def angle(
    point_a: Any,
    vertex: Any,
    point_c: Any,
    *,
    degrees: bool = True,
    zero_tolerance: float = 1e-12,
) -> float:
    """
    Calculate the angle formed by three points.

    The central point, ``vertex``, defines the angle:

    ``point_a — vertex — point_c``

    Parameters
    ----------
    point_a : Any
        First endpoint of the angle.
    vertex : Any
        Central point or angle vertex.
    point_c : Any
        Second endpoint of the angle.
    degrees : bool, optional
        Whether the result should be returned in degrees. When ``False``, the
        result is returned in radians.
    zero_tolerance : float, optional
        Norm threshold below which a vector is considered to have zero length.

    Returns
    -------
    float
        Angle in degrees or radians.

    Raises
    ------
    ValueError
        If one of the vectors forming the angle has zero length.

    Examples
    --------
    >>> angle(
    ...     [1.0, 0.0, 0.0],
    ...     [0.0, 0.0, 0.0],
    ...     [0.0, 1.0, 0.0],
    ... )
    90.0

    For a hydrogen bond, the donor is commonly used as the vertex:

    >>> dha_angle = angle(
    ...     hydrogen_atom,
    ...     donor_atom,
    ...     acceptor_atom,
    ... )
    """

    coordinate_a = _as_coordinate_array(
        point_a,
        name="point A",
        dimensions=None,
    )

    vertex_coordinate = _as_coordinate_array(
        vertex,
        name="vertex",
        dimensions=None,
    )

    coordinate_c = _as_coordinate_array(
        point_c,
        name="point C",
        dimensions=None,
    )

    if not (
        coordinate_a.shape
        == vertex_coordinate.shape
        == coordinate_c.shape
    ):
        raise ValueError(
            "All angle coordinates must have the same "
            "dimensionality."
        )

    vector_a = (
        coordinate_a
        - vertex_coordinate
    )

    vector_c = (
        coordinate_c
        - vertex_coordinate
    )

    norm_a = float(
        np.linalg.norm(vector_a)
    )

    norm_c = float(
        np.linalg.norm(vector_c)
    )

    if (
        norm_a <= zero_tolerance
        or norm_c <= zero_tolerance
    ):
        raise ValueError(
            "An angle cannot be calculated when an endpoint "
            "coincides with the vertex."
        )

    cosine_value = float(
        np.dot(
            vector_a,
            vector_c,
        )
        / (
            norm_a
            * norm_c
        )
    )

    # Floating-point operations may produce values such as 1.0000000002,
    # which are invalid inputs for arccos despite representing a valid angle.
    cosine_value = float(
        np.clip(
            cosine_value,
            -1.0,
            1.0,
        )
    )

    angle_radians = float(
        np.arccos(cosine_value)
    )

    if degrees:
        return float(
            np.degrees(angle_radians)
        )

    return angle_radians


# -----------------------------------------------------------------------------
# Public module interface
# -----------------------------------------------------------------------------

_SECTION_6_PUBLIC_NAMES = [
    "distance",
    "centroid",
    "angle",
    "normalize",
]

for public_name in _SECTION_6_PUBLIC_NAMES:
    if public_name not in __all__:
        __all__.append(public_name)


# =============================================================================
# End of Section 6
# =============================================================================


# =============================================================================
# Section 7 — File Management
# =============================================================================


# -----------------------------------------------------------------------------
# File-name normalization
# -----------------------------------------------------------------------------

def _sanitize_filename_component(
    value: Any,
    *,
    default: str = "output",
    lowercase: bool = False,
) -> str:
    """
    Convert a value into a safe file-name component.

    Spaces and unsupported characters are replaced with underscores.

    Parameters
    ----------
    value : Any
        Value used as part of a file name.
    default : str, optional
        Value returned when the supplied content is empty.
    lowercase : bool, optional
        Whether the resulting text should be converted to lowercase.

    Returns
    -------
    str
        Safe file-name component.
    """

    if value is None:
        text = ""

    else:
        text = str(value).strip()

    if lowercase:
        text = text.lower()

    text = _FILENAME_WHITESPACE_PATTERN.sub(
        "_",
        text,
    )

    text = _FILENAME_UNSAFE_PATTERN.sub(
        "_",
        text,
    )

    text = _FILENAME_UNDERSCORE_PATTERN.sub(
        "_",
        text,
    )

    text = text.strip(
        "._-"
    )

    return text or default


def _normalize_extension(
    extension: Optional[str],
) -> str:
    """
    Normalize a file extension.

    Parameters
    ----------
    extension : str or None
        File extension with or without a leading period.

    Returns
    -------
    str
        Normalized extension beginning with ``.`` or an empty string.
    """

    if extension is None:
        return ""

    normalized_extension = (
        str(extension)
        .strip()
        .lower()
    )

    if not normalized_extension:
        return ""

    if not normalized_extension.startswith(
        "."
    ):
        normalized_extension = (
            f".{normalized_extension}"
        )

    return normalized_extension


# -----------------------------------------------------------------------------
# Output-directory creation
# -----------------------------------------------------------------------------

def create_output(
    output_directory: Union[str, Path],
    *,
    subdirectories: Optional[
        Iterable[str]
    ] = None,
    exist_ok: bool = True,
) -> Dict[str, Path]:
    """
    Create the DockAnalyzer output directory structure.

    Parameters
    ----------
    output_directory : str or Path
        Root directory used to store generated files.
    subdirectories : iterable of str, optional
        Subdirectories created inside the root directory. When omitted,
        standard DockAnalyzer directories are created.
    exist_ok : bool, optional
        Whether an existing directory should be accepted.

    Returns
    -------
    dict
        Dictionary containing the root path and created subdirectories.

    Raises
    ------
    TypeError
        If the output path is invalid.
    OSError
        If a directory cannot be created.

    Examples
    --------
    >>> paths = create_output(
    ...     "DockAnalyzer_results"
    ... )

    >>> paths["json"]
    PosixPath('DockAnalyzer_results/json')
    """

    if output_directory is None:
        raise TypeError(
            "Output directory cannot be None."
        )

    root_path = Path(
        output_directory
    ).expanduser()

    root_path.mkdir(
        parents=True,
        exist_ok=exist_ok,
    )

    if not root_path.is_dir():
        raise NotADirectoryError(
            f"Output path is not a directory: "
            f"{root_path}"
        )

    if subdirectories is None:
        subdirectories = (
            "csv",
            "json",
            "images",
            "reports",
            "sessions",
            "logs",
        )

    created_paths: Dict[str, Path] = {
        "root": root_path
    }

    for subdirectory in subdirectories:
        normalized_name = (
            _sanitize_filename_component(
                subdirectory,
                default="files",
                lowercase=True,
            )
        )

        subdirectory_path = (
            root_path
            / normalized_name
        )

        subdirectory_path.mkdir(
            parents=True,
            exist_ok=exist_ok,
        )

        created_paths[
            normalized_name
        ] = subdirectory_path

    return created_paths


# -----------------------------------------------------------------------------
# File-name construction
# -----------------------------------------------------------------------------

def build_filename(
    name: Any,
    *,
    extension: Optional[str] = None,
    prefix: Optional[Any] = None,
    suffix: Optional[Any] = None,
    index: Optional[int] = None,
    timestamp: bool = False,
    lowercase: bool = False,
) -> str:
    """
    Build a standardized and file-system-safe file name.

    Parameters
    ----------
    name : Any
        Main file name.
    extension : str, optional
        File extension with or without a leading period.
    prefix : Any, optional
        Text inserted before the main name.
    suffix : Any, optional
        Text inserted after the main name.
    index : int, optional
        Numeric index appended to the name using three digits.
    timestamp : bool, optional
        Whether a timestamp should be appended.
    lowercase : bool, optional
        Whether text components should be converted to lowercase.

    Returns
    -------
    str
        Standardized file name.

    Examples
    --------
    >>> build_filename(
    ...     "Geraniol pose 1",
    ...     extension="json",
    ... )
    'Geraniol_pose_1.json'

    >>> build_filename(
    ...     "contacts",
    ...     prefix="6X3W",
    ...     index=2,
    ...     extension=".csv",
    ... )
    '6X3W_contacts_002.csv'
    """

    components: List[str] = []

    if prefix is not None:
        components.append(
            _sanitize_filename_component(
                prefix,
                lowercase=lowercase,
            )
        )

    components.append(
        _sanitize_filename_component(
            name,
            lowercase=lowercase,
        )
    )

    if index is not None:
        try:
            numeric_index = int(index)

        except (
            TypeError,
            ValueError,
        ) as error:
            raise TypeError(
                "File index must be an integer."
            ) from error

        components.append(
            f"{numeric_index:03d}"
        )

    if suffix is not None:
        components.append(
            _sanitize_filename_component(
                suffix,
                lowercase=lowercase,
            )
        )

    if timestamp:
        components.append(
            datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )
        )

    filename = "_".join(
        component
        for component in components
        if component
    )

    return (
        filename
        + _normalize_extension(
            extension
        )
    )


# -----------------------------------------------------------------------------
# Generic serialization
# -----------------------------------------------------------------------------

def _serialize_molecular_reference(
    value: Any,
) -> Optional[Dict[str, Any]]:
    """Return a compact representation of a molecular object when possible."""

    residue_number = first_not_none(
        getattr(value, "number", None),
        getattr(value, "residue_number", None),
        getattr(value, "resnum", None),
    )

    if (
        residue_number is not None
        and hasattr(value, "name")
        and (
            hasattr(value, "chain_id")
            or hasattr(value, "chain")
            or hasattr(value, "structure")
        )
    ):
        return {
            "type": type(value).__name__,
            "name": _get_residue_name(value),
            "chain_id": _get_residue_chain_id(value),
            "number": residue_number,
            "insertion_code": _get_residue_insertion_code(value),
            "atomspec": getattr(value, "atomspec", None),
            "structure_id": _get_structure_identifier(
                _get_residue_structure(value)
            ),
        }

    coordinate_value = first_not_none(
        getattr(value, "scene_coord", None),
        getattr(value, "coord", None),
        getattr(value, "coordinates", None),
    )

    if (
        coordinate_value is not None
        and hasattr(value, "name")
        and hasattr(value, "residue")
    ):
        element = getattr(value, "element", None)
        element_name = first_not_none(
            getattr(element, "name", None),
            getattr(value, "element_name", None),
        )

        return {
            "type": type(value).__name__,
            "name": str(getattr(value, "name", "")),
            "element": (
                None
                if element_name is None
                else str(element_name)
            ),
            "atomspec": getattr(value, "atomspec", None),
            "residue": residue_to_string(
                getattr(value, "residue", None),
                include_structure=True,
            ),
            "coordinates": np.asarray(
                coordinate_value,
                dtype=float,
            ).tolist(),
        }

    if hasattr(value, "atoms") and (
        hasattr(value, "id_string")
        or hasattr(value, "atomspec")
        or hasattr(value, "residues")
    ):
        return {
            "type": type(value).__name__,
            "name": get_model_name(
                value,
                default=type(value).__name__,
            ),
            "model_id": get_model_id(value),
            "atomspec": get_model_atomspec(value),
            "atom_count": get_atom_count(value),
            "residue_count": get_residue_count(value),
        }

    return None


def _make_serializable(
    value: Any,
    *,
    _active_ids: Optional[Set[int]] = None,
) -> Any:
    """Convert a value into a strict, cycle-safe JSON representation."""

    if value is None:
        return None

    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        return value

    if isinstance(value, int) and not isinstance(value, bool):
        return value

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    if isinstance(value, np.generic):
        return _make_serializable(
            value.item(),
            _active_ids=_active_ids,
        )

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, bytes):
        return value.decode(
            "utf-8",
            errors="replace",
        )

    if isinstance(value, bytearray):
        return bytes(value).decode(
            "utf-8",
            errors="replace",
        )

    if _active_ids is None:
        _active_ids = set()

    value_id = id(value)

    if value_id in _active_ids:
        return "<recursive>"

    _active_ids.add(value_id)

    try:
        if isinstance(value, np.ndarray):
            return _make_serializable(
                value.tolist(),
                _active_ids=_active_ids,
            )

        if isinstance(value, pd.DataFrame):
            return _make_serializable(
                value.to_dict(
                    orient="records"
                ),
                _active_ids=_active_ids,
            )

        if isinstance(value, pd.Series):
            return _make_serializable(
                value.to_dict(),
                _active_ids=_active_ids,
            )

        if isinstance(value, Mapping):
            return {
                str(key): _make_serializable(
                    item,
                    _active_ids=_active_ids,
                )
                for key, item in value.items()
            }

        if isinstance(
            value,
            (
                list,
                tuple,
                set,
                frozenset,
            ),
        ):
            return [
                _make_serializable(
                    item,
                    _active_ids=_active_ids,
                )
                for item in value
            ]

        molecular_reference = _serialize_molecular_reference(
            value
        )

        if molecular_reference is not None:
            return _make_serializable(
                molecular_reference,
                _active_ids=_active_ids,
            )

        to_dict_method = getattr(
            value,
            "to_dict",
            None,
        )

        if callable(to_dict_method):
            try:
                converted_value = to_dict_method()
            except Exception:
                converted_value = None
            else:
                return _make_serializable(
                    converted_value,
                    _active_ids=_active_ids,
                )

        if hasattr(value, "__dict__"):
            try:
                public_attributes = {
                    str(key): item
                    for key, item in vars(value).items()
                    if not str(key).startswith("_")
                }
            except TypeError:
                public_attributes = None

            if public_attributes is not None:
                return _make_serializable(
                    public_attributes,
                    _active_ids=_active_ids,
                )

        return str(value)

    finally:
        _active_ids.discard(value_id)


def _prepare_output_file(
    file_path: Union[str, Path],
    *,
    extension: str,
    overwrite: bool,
    create_parent: bool,
) -> Path:
    """
    Validate and prepare an output file path.
    """

    if file_path is None:
        raise TypeError(
            "Output file path cannot be None."
        )

    output_path = Path(
        file_path
    ).expanduser()

    normalized_extension = (
        _normalize_extension(
            extension
        )
    )

    if output_path.suffix == "":
        output_path = output_path.with_suffix(
            normalized_extension
        )

    if create_parent:
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    if (
        output_path.exists()
        and not overwrite
    ):
        raise FileExistsError(
            f"Output file already exists: "
            f"{output_path}"
        )

    return output_path


# -----------------------------------------------------------------------------
# JSON export
# -----------------------------------------------------------------------------

def save_json(
    data: Any,
    file_path: Union[str, Path],
    *,
    indent: Optional[int] = 4,
    ensure_ascii: bool = False,
    sort_keys: bool = False,
    overwrite: bool = True,
    create_parent: bool = True,
) -> Path:
    """
    Save data as a JSON file.

    The function supports dictionaries, lists, NumPy values, pandas objects,
    Path objects and objects implementing ``to_dict()``.

    Parameters
    ----------
    data : Any
        Data to serialize.
    file_path : str or Path
        Destination file path.
    indent : int or None, optional
        JSON indentation level.
    ensure_ascii : bool, optional
        Whether non-ASCII characters should be escaped.
    sort_keys : bool, optional
        Whether dictionary keys should be sorted.
    overwrite : bool, optional
        Whether an existing file may be replaced.
    create_parent : bool, optional
        Whether missing parent directories should be created.

    Returns
    -------
    Path
        Path of the generated JSON file.

    Examples
    --------
    >>> save_json(
    ...     dock_model.to_dict(),
    ...     "results/json/pose_01.json",
    ... )
    """

    import json

    output_path = _prepare_output_file(
        file_path=file_path,
        extension=".json",
        overwrite=overwrite,
        create_parent=create_parent,
    )

    serializable_data = _make_serializable(
        data
    )

    try:
        with output_path.open(
            mode="w",
            encoding="utf-8",
        ) as json_file:
            json.dump(
                serializable_data,
                json_file,
                indent=indent,
                ensure_ascii=ensure_ascii,
                sort_keys=sort_keys,
                allow_nan=False,
            )

    except OSError as error:
        raise OSError(
            f"Could not save JSON file: "
            f"{output_path}"
        ) from error

    return output_path


# -----------------------------------------------------------------------------
# CSV export
# -----------------------------------------------------------------------------

def _to_dataframe(
    data: Any,
) -> pd.DataFrame:
    """
    Convert supported data structures into a pandas DataFrame.

    Parameters
    ----------
    data : Any
        DataFrame, Series, dictionary, sequence or object implementing
        ``to_dict()``.

    Returns
    -------
    pandas.DataFrame
        Converted tabular data.

    Raises
    ------
    TypeError
        If the supplied value cannot be converted to tabular data.
    """

    if isinstance(
        data,
        pd.DataFrame,
    ):
        return data.copy()

    if isinstance(
        data,
        pd.Series,
    ):
        return data.to_frame()

    if isinstance(
        data,
        dict,
    ):
        try:
            return pd.DataFrame(
                data
            )

        except ValueError:
            return pd.DataFrame(
                [data]
            )

    if isinstance(
        data,
        (
            list,
            tuple,
        ),
    ):
        converted_data = [
            (
                item.to_dict()
                if callable(
                    getattr(
                        item,
                        "to_dict",
                        None,
                    )
                )
                else item
            )
            for item in data
        ]

        return pd.DataFrame(
            converted_data
        )

    to_dict_method = getattr(
        data,
        "to_dict",
        None,
    )

    if callable(to_dict_method):
        converted_data = (
            to_dict_method()
        )

        if isinstance(
            converted_data,
            dict,
        ):
            return pd.DataFrame(
                [converted_data]
            )

        return pd.DataFrame(
            converted_data
        )

    raise TypeError(
        "Data cannot be converted into a CSV-compatible table."
    )


def save_csv(
    data: Any,
    file_path: Union[str, Path],
    *,
    index: bool = False,
    encoding: str = "utf-8",
    separator: str = ",",
    decimal: str = ".",
    overwrite: bool = True,
    create_parent: bool = True,
) -> Path:
    """
    Save tabular data as a CSV file.

    Parameters
    ----------
    data : Any
        DataFrame, Series, dictionary, list of dictionaries or objects
        implementing ``to_dict()``.
    file_path : str or Path
        Destination CSV file.
    index : bool, optional
        Whether pandas row indexes should be included.
    encoding : str, optional
        Output-file encoding.
    separator : str, optional
        Column separator.
    decimal : str, optional
        Decimal separator.
    overwrite : bool, optional
        Whether an existing file may be replaced.
    create_parent : bool, optional
        Whether missing parent directories should be created.

    Returns
    -------
    Path
        Path of the generated CSV file.

    Examples
    --------
    >>> save_csv(
    ...     dock_models,
    ...     "results/csv/dock_models.csv",
    ... )
    """

    if not isinstance(
        separator,
        str,
    ) or len(separator) != 1:
        raise ValueError(
            "CSV separator must contain exactly one character."
        )

    if not isinstance(
        decimal,
        str,
    ) or len(decimal) != 1:
        raise ValueError(
            "Decimal separator must contain exactly one character."
        )

    output_path = _prepare_output_file(
        file_path=file_path,
        extension=".csv",
        overwrite=overwrite,
        create_parent=create_parent,
    )

    dataframe = _to_dataframe(
        data
    )

    try:
        dataframe.to_csv(
            output_path,
            index=index,
            encoding=encoding,
            sep=separator,
            decimal=decimal,
        )

    except OSError as error:
        raise OSError(
            f"Could not save CSV file: "
            f"{output_path}"
        ) from error

    return output_path


# -----------------------------------------------------------------------------
# Compatibility file-management API
# -----------------------------------------------------------------------------

def ensure_output_directories(
    output_directory: Union[str, Path],
    subdirectories: Optional[
        Iterable[str]
    ] = None,
    exist_ok: bool = True,
) -> Dict[str, Path]:
    """Create and return the DockAnalyzer output directory structure."""

    return create_output(
        output_directory,
        subdirectories=subdirectories,
        exist_ok=exist_ok,
    )


def build_output_filename(
    name: Any,
    extension: Optional[str] = None,
    prefix: Optional[Any] = None,
    suffix: Optional[Any] = None,
    index: Optional[int] = None,
    timestamp: bool = False,
    lowercase: bool = False,
) -> str:
    """Build a standardized output filename.

    This compatibility wrapper accepts the historical positional option order
    and delegates to :func:`build_filename`.
    """

    return build_filename(
        name,
        extension=extension,
        prefix=prefix,
        suffix=suffix,
        index=index,
        timestamp=timestamp,
        lowercase=lowercase,
    )


# -----------------------------------------------------------------------------
# Public module interface
# -----------------------------------------------------------------------------

_SECTION_7_PUBLIC_NAMES = [
    "create_output",
    "ensure_output_directories",
    "build_filename",
    "build_output_filename",
    "save_json",
    "save_csv",
]

for public_name in _SECTION_7_PUBLIC_NAMES:
    if public_name not in __all__:
        __all__.append(public_name)


# =============================================================================
# End of Section 7
# =============================================================================

# =============================================================================
# Section 8 — Residue Utilities
# =============================================================================


# -----------------------------------------------------------------------------
# Basic residue attributes
# -----------------------------------------------------------------------------

def _get_residue_structure(
    residue: Any,
) -> Any:
    """
    Return the molecular structure associated with a residue.

    Parameters
    ----------
    residue : Any
        ChimeraX residue or residue-like object.

    Returns
    -------
    Any
        Parent atomic structure, when available.
    """

    if residue is None:
        return None

    structure = getattr(
        residue,
        "structure",
        None,
    )

    if structure is not None:
        return structure

    return getattr(
        residue,
        "model",
        None,
    )


def _get_residue_chain_id(
    residue: Any,
) -> str:
    """
    Return the chain identifier associated with a residue.
    """

    if residue is None:
        return ""

    chain_id = getattr(
        residue,
        "chain_id",
        None,
    )

    if chain_id is not None:
        return str(chain_id).strip()

    chain = getattr(
        residue,
        "chain",
        None,
    )

    if chain is not None:
        chain_id = getattr(
            chain,
            "chain_id",
            None,
        )

        if chain_id is None:
            chain_id = getattr(
                chain,
                "id",
                None,
            )

        if chain_id is not None:
            return str(chain_id).strip()

    return ""


def _get_residue_number(
    residue: Any,
) -> Any:
    """
    Return the residue number.

    The value is kept in its original form because some model-like objects may
    use non-integer residue identifiers.
    """

    if residue is None:
        return None

    for attribute_name in (
        "number",
        "residue_number",
        "resnum",
    ):
        value = getattr(
            residue,
            attribute_name,
            None,
        )

        if value is not None:
            return value

    return None


def _get_residue_insertion_code(
    residue: Any,
) -> str:
    """
    Return the residue insertion code.
    """

    if residue is None:
        return ""

    for attribute_name in (
        "insertion_code",
        "insert",
        "icode",
    ):
        value = getattr(
            residue,
            attribute_name,
            None,
        )

        if value not in (
            None,
            "",
            " ",
        ):
            return str(value).strip()

    return ""


def _get_residue_name(
    residue: Any,
) -> str:
    """
    Return the normalized residue name.
    """

    if residue is None:
        return ""

    name = getattr(
        residue,
        "name",
        None,
    )

    if name is None:
        return ""

    return str(name).strip().upper()


def _get_structure_identifier(
    structure: Any,
) -> str:
    """
    Return a stable readable identifier for a molecular structure.
    """

    if structure is None:
        return ""

    model_id = get_model_id(
        structure
    )

    if model_id:
        return str(model_id)

    atomspec = get_model_atomspec(
        structure
    )

    if atomspec:
        return str(atomspec).lstrip("#")

    name = get_model_name(
        structure,
        default="",
    )

    return str(name)


def _residue_identity_key(
    residue: Any,
    *,
    include_structure: bool = True,
    include_name: bool = True,
) -> Tuple[Any, ...]:
    """
    Build a hashable molecular identity key for a residue.

    Parameters
    ----------
    residue : Any
        ChimeraX residue or residue-like object.
    include_structure : bool, optional
        Whether the parent structure should be included.
    include_name : bool, optional
        Whether the residue name should be included.

    Returns
    -------
    tuple
        Residue identity key.
    """

    structure_identifier = ""

    if include_structure:
        structure_identifier = (
            _get_structure_identifier(
                _get_residue_structure(
                    residue
                )
            )
        )

    residue_name = ""

    if include_name:
        residue_name = _get_residue_name(
            residue
        )

    return (
        structure_identifier,
        _get_residue_chain_id(
            residue
        ),
        _get_residue_number(
            residue
        ),
        _get_residue_insertion_code(
            residue
        ),
        residue_name,
    )


# -----------------------------------------------------------------------------
# Unique residues
# -----------------------------------------------------------------------------

def unique_residues(
    residues: Any,
    *,
    preserve_order: bool = True,
    by_identity: bool = True,
    include_structure: bool = True,
    include_name: bool = True,
    ignore_none: bool = True,
) -> List[Any]:
    """
    Remove duplicated residues.

    Parameters
    ----------
    residues : Any
        Iterable of ChimeraX residues or residue-like objects.
    preserve_order : bool, optional
        Whether the original order should be preserved.
    by_identity : bool, optional
        When ``True``, residues are compared using molecular identity:
        structure, chain, number, insertion code and residue name. When
        ``False``, Python object identity is used.
    include_structure : bool, optional
        Whether the parent structure should be included in the molecular
        identity key.
    include_name : bool, optional
        Whether the residue name should be included in the molecular identity
        key.
    ignore_none : bool, optional
        Whether ``None`` values should be ignored.

    Returns
    -------
    list
        Unique residues.

    Raises
    ------
    TypeError
        If residues is not iterable.

    Examples
    --------
    >>> residues = unique_residues(
    ...     contact_residues
    ... )
    """

    if residues is None:
        return []

    try:
        residue_list = list(
            residues
        )

    except TypeError as error:
        raise TypeError(
            "Residues must be an iterable of residue-like objects."
        ) from error

    unique_list: List[Any] = []
    seen_keys = set()

    for residue in residue_list:
        if (
            residue is None
            and ignore_none
        ):
            continue

        if by_identity:
            key = _residue_identity_key(
                residue,
                include_structure=include_structure,
                include_name=include_name,
            )

        else:
            key = id(residue)

        if key in seen_keys:
            continue

        seen_keys.add(key)
        unique_list.append(
            residue
        )

    if preserve_order:
        return unique_list

    return sort_residues(
        unique_list
    )


# -----------------------------------------------------------------------------
# Residue string representation
# -----------------------------------------------------------------------------

def residue_to_string(
    residue: Any,
    *,
    include_name: bool = True,
    include_chain: bool = True,
    include_structure: bool = False,
    include_atomspec: bool = False,
    separator: str = ":",
    unknown: str = "?",
) -> str:
    """
    Convert a residue into a readable standardized string.

    The default format is:

    ``CHAIN:RESNAME123``

    Examples:

    ``A:TYR58``

    ``B:GLU155A``

    Parameters
    ----------
    residue : Any
        ChimeraX residue or residue-like object.
    include_name : bool, optional
        Whether the residue name should be included.
    include_chain : bool, optional
        Whether the chain identifier should be included.
    include_structure : bool, optional
        Whether the parent model identifier should be prepended.
    include_atomspec : bool, optional
        Whether a ChimeraX residue atom-specifier should be used when
        available.
    separator : str, optional
        Separator between structure, chain and residue components.
    unknown : str, optional
        Placeholder used for missing attributes.

    Returns
    -------
    str
        Human-readable residue representation.

    Examples
    --------
    >>> residue_to_string(residue)
    'A:TYR58'

    >>> residue_to_string(
    ...     residue,
    ...     include_structure=True,
    ... )
    '1:A:TYR58'
    """

    if residue is None:
        return unknown

    if include_atomspec:
        atomspec = getattr(
            residue,
            "atomspec",
            None,
        )

        if atomspec:
            return str(atomspec)

    components: List[str] = []

    if include_structure:
        structure_identifier = (
            _get_structure_identifier(
                _get_residue_structure(
                    residue
                )
            )
        )

        components.append(
            structure_identifier
            or unknown
        )

    if include_chain:
        chain_id = _get_residue_chain_id(
            residue
        )

        components.append(
            chain_id
            or unknown
        )

    residue_number = _get_residue_number(
        residue
    )

    insertion_code = (
        _get_residue_insertion_code(
            residue
        )
    )

    residue_label = ""

    if include_name:
        residue_name = _get_residue_name(
            residue
        )

        residue_label += (
            residue_name
            or unknown
        )

    if residue_number is not None:
        residue_label += str(
            residue_number
        )

    elif not residue_label:
        residue_label = unknown

    else:
        residue_label += unknown

    if insertion_code:
        residue_label += insertion_code

    components.append(
        residue_label
    )

    return separator.join(
        components
    )


# -----------------------------------------------------------------------------
# Residue sorting
# -----------------------------------------------------------------------------

def _natural_sort_key(
    value: Any,
) -> Tuple[Any, ...]:
    """
    Convert text containing numbers into a natural sorting key.

    Examples
    --------
    ``A2`` is sorted before ``A10``.
    """

    if value is None:
        return ("",)

    text = str(value)

    parts = _NATURAL_SORT_PATTERN.split(
        text
    )

    return tuple(
        int(part)
        if part.isdigit()
        else part.lower()
        for part in parts
        if part != ""
    )


def _residue_number_sort_key(
    residue_number: Any,
) -> Tuple[int, Any]:
    """
    Build a sorting key for numeric and non-numeric residue identifiers.
    """

    if residue_number is None:
        return (
            2,
            float("inf"),
        )

    try:
        return (
            0,
            int(residue_number),
        )

    except (
        TypeError,
        ValueError,
    ):
        return (
            1,
            _natural_sort_key(
                residue_number
            ),
        )


def _residue_sort_key(
    residue: Any,
    *,
    include_structure: bool = True,
) -> Tuple[Any, ...]:
    """
    Build the default sorting key for a residue.
    """

    structure_key = ""

    if include_structure:
        structure_key = (
            _get_structure_identifier(
                _get_residue_structure(
                    residue
                )
            )
        )

    return (
        _natural_sort_key(
            structure_key
        ),
        _natural_sort_key(
            _get_residue_chain_id(
                residue
            )
        ),
        _residue_number_sort_key(
            _get_residue_number(
                residue
            )
        ),
        _natural_sort_key(
            _get_residue_insertion_code(
                residue
            )
        ),
        _natural_sort_key(
            _get_residue_name(
                residue
            )
        ),
    )


def sort_residues(
    residues: Any,
    *,
    reverse: bool = False,
    unique: bool = False,
    include_structure: bool = True,
    none_last: bool = True,
) -> List[Any]:
    """
    Sort residues by structure, chain, number, insertion code and name.

    Parameters
    ----------
    residues : Any
        Iterable of ChimeraX residues or residue-like objects.
    reverse : bool, optional
        Whether the sorting order should be reversed.
    unique : bool, optional
        Whether duplicated residues should be removed before sorting.
    include_structure : bool, optional
        Whether the parent structure should participate in sorting.
    none_last : bool, optional
        Whether ``None`` entries should be placed after valid residues.

    Returns
    -------
    list
        Sorted residue list.

    Raises
    ------
    TypeError
        If residues is not iterable.

    Examples
    --------
    >>> ordered_residues = sort_residues(
    ...     residues
    ... )
    """

    if residues is None:
        return []

    try:
        residue_list = list(
            residues
        )

    except TypeError as error:
        raise TypeError(
            "Residues must be an iterable of residue-like objects."
        ) from error

    if unique:
        residue_list = unique_residues(
            residue_list,
            preserve_order=True,
            include_structure=include_structure,
        )

    def sorting_key(
        residue: Any,
    ) -> Tuple[Any, ...]:
        if residue is None:
            none_position = (
                1
                if none_last
                else 0
            )

            return (
                none_position,
                (),
            )

        valid_position = (
            0
            if none_last
            else 1
        )

        return (
            valid_position,
            _residue_sort_key(
                residue,
                include_structure=include_structure,
            ),
        )

    return sorted(
        residue_list,
        key=sorting_key,
        reverse=reverse,
    )


# -----------------------------------------------------------------------------
# Public module interface
# -----------------------------------------------------------------------------

_SECTION_8_PUBLIC_NAMES = [
    "unique_residues",
    "residue_to_string",
    "sort_residues",
]

for public_name in _SECTION_8_PUBLIC_NAMES:
    if public_name not in __all__:
        __all__.append(public_name)


# =============================================================================
# End of Section 8
# =============================================================================


# =============================================================================
# Section 9 — Pretty Printing
# =============================================================================


# -----------------------------------------------------------------------------
# Text helpers
# -----------------------------------------------------------------------------

def _stringify_terminal_value(
    value: Any,
    *,
    float_precision: int = 3,
    none_value: str = "-",
) -> str:
    """
    Convert a value into a terminal-friendly string.

    Parameters
    ----------
    value : Any
        Value to convert.
    float_precision : int, optional
        Number of decimal places used for floating-point values.
    none_value : str, optional
        Text used when the value is ``None``.

    Returns
    -------
    str
        Terminal-friendly representation.
    """

    if value is None:
        return none_value

    if isinstance(
        value,
        (
            float,
            np.floating,
        ),
    ):
        if np.isnan(value):
            return "NaN"

        if np.isposinf(value):
            return "inf"

        if np.isneginf(value):
            return "-inf"

        return f"{float(value):.{float_precision}f}"

    if isinstance(
        value,
        (
            int,
            np.integer,
        ),
    ):
        return str(int(value))

    if isinstance(
        value,
        (
            bool,
            np.bool_,
        ),
    ):
        return "Yes" if bool(value) else "No"

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, np.ndarray):
        return np.array2string(
            value,
            precision=float_precision,
            separator=", ",
        )

    if isinstance(
        value,
        (
            list,
            tuple,
            set,
        ),
    ):
        return ", ".join(
            _stringify_terminal_value(
                item,
                float_precision=float_precision,
                none_value=none_value,
            )
            for item in value
        )

    return str(value)


def _truncate_text(
    text: str,
    maximum_width: Optional[int],
    *,
    suffix: str = "...",
) -> str:
    """
    Truncate text to a maximum width.

    Parameters
    ----------
    text : str
        Text to truncate.
    maximum_width : int or None
        Maximum allowed width.
    suffix : str, optional
        Suffix appended to truncated text.

    Returns
    -------
    str
        Original or truncated text.
    """

    if maximum_width is None:
        return text

    maximum_width = int(
        maximum_width
    )

    if maximum_width <= 0:
        return ""

    if len(text) <= maximum_width:
        return text

    if maximum_width <= len(suffix):
        return text[:maximum_width]

    return (
        text[
            :maximum_width
            - len(suffix)
        ]
        + suffix
    )


def _normalize_alignment(
    alignment: str,
) -> str:
    """
    Normalize a column-alignment value.

    Accepted values are:

    - ``left``
    - ``center``
    - ``right``
    - ``<``
    - ``^``
    - ``>``
    """

    alignment_map = {
        "left": "<",
        "center": "^",
        "right": ">",
        "<": "<",
        "^": "^",
        ">": ">",
    }

    normalized_alignment = str(
        alignment
    ).strip().lower()

    if normalized_alignment not in alignment_map:
        raise ValueError(
            "Alignment must be 'left', 'center', "
            "'right', '<', '^' or '>'."
        )

    return alignment_map[
        normalized_alignment
    ]


# -----------------------------------------------------------------------------
# Title formatting
# -----------------------------------------------------------------------------

def format_title(
    title: Any,
    *,
    width: Optional[int] = None,
    character: str = "=",
    alignment: str = "center",
    uppercase: bool = True,
    padding: int = 2,
) -> str:
    """
    Format a terminal section title.

    Parameters
    ----------
    title : Any
        Title text.
    width : int or None, optional
        Total width of the separator. When omitted, the width is determined
        automatically from the title.
    character : str, optional
        Character used to build the separator.
    alignment : str, optional
        Title alignment: ``left``, ``center`` or ``right``.
    uppercase : bool, optional
        Whether the title should be converted to uppercase.
    padding : int, optional
        Minimum number of separator characters around the title.

    Returns
    -------
    str
        Formatted multi-line title.

    Examples
    --------
    >>> print(format_title("Hbonds"))
    ======================
    HBONDS
    ======================
    """

    if title is None:
        title_text = ""

    else:
        title_text = str(
            title
        ).strip()

    if uppercase:
        title_text = title_text.upper()

    if not character:
        raise ValueError(
            "Separator character cannot be empty."
        )

    separator_character = str(
        character
    )[0]

    minimum_width = (
        len(title_text)
        + (
            max(
                int(padding),
                0,
            )
            * 2
        )
    )

    if width is None:
        width = max(
            minimum_width,
            22,
        )

    else:
        width = max(
            int(width),
            minimum_width,
        )

    alignment_symbol = _normalize_alignment(
        alignment
    )

    separator = separator_character * width

    formatted_title = (
        f"{title_text:{alignment_symbol}{width}}"
    )

    return "\n".join(
        (
            separator,
            formatted_title,
            separator,
        )
    )


def print_title(
    title: Any,
    *,
    width: Optional[int] = None,
    character: str = "=",
    alignment: str = "center",
    uppercase: bool = True,
    padding: int = 2,
    logger: Optional[DockLogger] = None,
) -> str:
    """
    Print a formatted terminal title.

    Parameters
    ----------
    title : Any
        Title text.
    width : int or None, optional
        Separator width.
    character : str, optional
        Separator character.
    alignment : str, optional
        Title alignment.
    uppercase : bool, optional
        Whether the title should be uppercase.
    padding : int, optional
        Minimum horizontal padding.
    logger : DockLogger, optional
        Logger used instead of the standard ``print`` function.

    Returns
    -------
    str
        Printed title text.
    """

    formatted_title = format_title(
        title=title,
        width=width,
        character=character,
        alignment=alignment,
        uppercase=uppercase,
        padding=padding,
    )

    if logger is not None:
        logger.info(
            "\n%s",
            formatted_title,
        )

    else:
        print(
            formatted_title
        )

    return formatted_title


# -----------------------------------------------------------------------------
# Table data conversion
# -----------------------------------------------------------------------------

def _records_from_table_data(
    data: Any,
) -> Tuple[List[str], List[List[Any]]]:
    """
    Convert common data structures into table headers and rows.

    Supported inputs include:

    - pandas DataFrame;
    - pandas Series;
    - dictionary;
    - list of dictionaries;
    - list of objects implementing ``to_dict()``;
    - list of lists or tuples.

    Parameters
    ----------
    data : Any
        Table-like data.

    Returns
    -------
    tuple
        Column names and table rows.

    Raises
    ------
    TypeError
        If the supplied data cannot be represented as a table.
    """

    if data is None:
        return [], []

    if isinstance(
        data,
        pd.DataFrame,
    ):
        columns = [
            str(column)
            for column in data.columns
        ]

        rows = data.values.tolist()

        return columns, rows

    if isinstance(
        data,
        pd.Series,
    ):
        column_name = (
            str(data.name)
            if data.name is not None
            else "value"
        )

        return (
            ["index", column_name],
            [
                [
                    index,
                    value,
                ]
                for index, value in data.items()
            ],
        )

    if isinstance(
        data,
        dict,
    ):
        columns = [
            str(key)
            for key in data.keys()
        ]

        values = list(
            data.values()
        )

        sequence_lengths = []

        for value in values:
            if isinstance(
                value,
                (
                    list,
                    tuple,
                    np.ndarray,
                    pd.Series,
                ),
            ):
                sequence_lengths.append(
                    len(value)
                )

            else:
                sequence_lengths.append(
                    None
                )

        valid_lengths = [
            length
            for length in sequence_lengths
            if length is not None
        ]

        if (
            valid_lengths
            and len(valid_lengths)
            == len(values)
            and len(set(valid_lengths)) == 1
        ):
            rows = [
                [
                    values[
                        column_index
                    ][row_index]
                    for column_index in range(
                        len(values)
                    )
                ]
                for row_index in range(
                    valid_lengths[0]
                )
            ]

        else:
            rows = [
                values
            ]

        return columns, rows

    try:
        items = list(
            data
        )

    except TypeError as error:
        raise TypeError(
            "Table data must be a DataFrame, dictionary "
            "or iterable."
        ) from error

    if not items:
        return [], []

    converted_items = []

    for item in items:
        to_dict_method = getattr(
            item,
            "to_dict",
            None,
        )

        if callable(
            to_dict_method
        ):
            try:
                item = to_dict_method()

            except TypeError:
                pass

        converted_items.append(
            item
        )

    if all(
        isinstance(
            item,
            dict,
        )
        for item in converted_items
    ):
        columns: List[str] = []

        for record in converted_items:
            for key in record.keys():
                string_key = str(
                    key
                )

                if string_key not in columns:
                    columns.append(
                        string_key
                    )

        rows = [
            [
                record.get(
                    column
                )
                for column in columns
            ]
            for record in converted_items
        ]

        return columns, rows

    if all(
        isinstance(
            item,
            (
                list,
                tuple,
                np.ndarray,
            ),
        )
        for item in converted_items
    ):
        rows = [
            list(item)
            for item in converted_items
        ]

        maximum_columns = max(
            len(row)
            for row in rows
        )

        columns = [
            f"Column {index}"
            for index in range(
                1,
                maximum_columns + 1,
            )
        ]

        normalized_rows = [
            row
            + [
                None
            ] * (
                maximum_columns
                - len(row)
            )
            for row in rows
        ]

        return columns, normalized_rows

    return (
        ["Value"],
        [
            [item]
            for item in converted_items
        ],
    )


# -----------------------------------------------------------------------------
# Table formatting
# -----------------------------------------------------------------------------

def format_table(
    data: Any,
    *,
    headers: Optional[
        Iterable[Any]
    ] = None,
    alignments: Optional[
        Union[
            str,
            Iterable[str],
            Dict[str, str],
        ]
    ] = None,
    float_precision: int = 3,
    none_value: str = "-",
    column_spacing: int = 3,
    maximum_column_width: Optional[int] = 40,
    show_header: bool = True,
    show_separator: bool = True,
    empty_message: str = "No data available.",
) -> str:
    """
    Format tabular data for terminal display.

    Parameters
    ----------
    data : Any
        DataFrame, dictionary, list of dictionaries, list of sequences or
        objects implementing ``to_dict()``.
    headers : iterable, optional
        Custom column names. When omitted, names are inferred from the data.
    alignments : str, iterable or dict, optional
        Column alignments. A single value applies to all columns. A dictionary
        maps column names to alignment values.
    float_precision : int, optional
        Number of decimal places for floating-point values.
    none_value : str, optional
        Text used for missing values.
    column_spacing : int, optional
        Number of spaces between columns.
    maximum_column_width : int or None, optional
        Maximum width of each column.
    show_header : bool, optional
        Whether column names should be displayed.
    show_separator : bool, optional
        Whether a separator should be printed below the header.
    empty_message : str, optional
        Text returned when the table contains no rows.

    Returns
    -------
    str
        Formatted table.

    Examples
    --------
    >>> table = format_table(
    ...     [
    ...         {
    ...             "Donor": "A:SER205",
    ...             "Acceptor": "LIG:O1",
    ...             "Distance": 2.81,
    ...             "Angle": 164.3,
    ...         }
    ...     ]
    ... )
    """

    inferred_headers, raw_rows = (
        _records_from_table_data(
            data
        )
    )

    if headers is None:
        column_names = inferred_headers

    else:
        column_names = [
            str(header)
            for header in headers
        ]

    if not raw_rows:
        return empty_message

    number_of_columns = max(
        len(column_names),
        max(
            len(row)
            for row in raw_rows
        ),
    )

    if not column_names:
        column_names = [
            f"Column {index}"
            for index in range(
                1,
                number_of_columns + 1,
            )
        ]

    elif len(column_names) < number_of_columns:
        column_names.extend(
            f"Column {index}"
            for index in range(
                len(column_names) + 1,
                number_of_columns + 1,
            )
        )

    normalized_rows = [
        list(row)
        + [
            None
        ] * (
            number_of_columns
            - len(row)
        )
        for row in raw_rows
    ]

    string_rows = [
        [
            _truncate_text(
                _stringify_terminal_value(
                    value,
                    float_precision=float_precision,
                    none_value=none_value,
                ),
                maximum_column_width,
            )
            for value in row
        ]
        for row in normalized_rows
    ]

    formatted_headers = [
        _truncate_text(
            str(column_name),
            maximum_column_width,
        )
        for column_name in column_names
    ]

    column_widths = []

    for column_index in range(
        number_of_columns
    ):
        content_width = max(
            len(row[column_index])
            for row in string_rows
        )

        header_width = (
            len(
                formatted_headers[
                    column_index
                ]
            )
            if show_header
            else 0
        )

        column_widths.append(
            max(
                content_width,
                header_width,
            )
        )

    default_alignments = [
        "left"
    ] * number_of_columns

    if alignments is None:
        for column_index in range(
            number_of_columns
        ):
            column_values = [
                row[column_index]
                for row in normalized_rows
                if row[column_index]
                is not None
            ]

            if column_values and all(
                isinstance(
                    value,
                    (
                        int,
                        float,
                        np.integer,
                        np.floating,
                    ),
                )
                and not isinstance(
                    value,
                    (
                        bool,
                        np.bool_,
                    ),
                )
                for value in column_values
            ):
                default_alignments[
                    column_index
                ] = "right"

    elif isinstance(
        alignments,
        str,
    ):
        default_alignments = [
            alignments
        ] * number_of_columns

    elif isinstance(
        alignments,
        dict,
    ):
        default_alignments = [
            alignments.get(
                column_name,
                "left",
            )
            for column_name in column_names
        ]

    else:
        alignment_list = list(
            alignments
        )

        if (
            len(alignment_list)
            != number_of_columns
        ):
            raise ValueError(
                "The number of alignments must match "
                "the number of table columns."
            )

        default_alignments = alignment_list

    alignment_symbols = [
        _normalize_alignment(
            alignment
        )
        for alignment in default_alignments
    ]

    spacing = " " * max(
        int(column_spacing),
        0,
    )

    lines: List[str] = []

    if show_header:
        header_line = spacing.join(
            format(
                formatted_headers[index],
                f"{alignment_symbols[index]}{column_widths[index]}",
            )
            for index in range(
                number_of_columns
            )
        )

        lines.append(
            header_line.rstrip()
        )

        if show_separator:
            separator_line = spacing.join(
                "-" * width
                for width in column_widths
            )

            lines.append(
                separator_line.rstrip()
            )

    for row in string_rows:
        row_line = spacing.join(
            format(
                row[index],
                f"{alignment_symbols[index]}{column_widths[index]}",
            )
            for index in range(
                number_of_columns
            )
        )

        lines.append(
            row_line.rstrip()
        )

    return "\n".join(
        lines
    )


def print_table(
    data: Any,
    *,
    title: Optional[Any] = None,
    headers: Optional[
        Iterable[Any]
    ] = None,
    alignments: Optional[
        Union[
            str,
            Iterable[str],
            Dict[str, str],
        ]
    ] = None,
    float_precision: int = 3,
    none_value: str = "-",
    column_spacing: int = 3,
    maximum_column_width: Optional[int] = 40,
    show_header: bool = True,
    show_separator: bool = True,
    empty_message: str = "No data available.",
    title_character: str = "=",
    logger: Optional[DockLogger] = None,
) -> str:
    """
    Print a formatted table, optionally preceded by a title.

    Parameters
    ----------
    data : Any
        Table-like data.
    title : Any, optional
        Section title displayed above the table.
    headers : iterable, optional
        Custom column names.
    alignments : str, iterable or dict, optional
        Column alignment configuration.
    float_precision : int, optional
        Number of decimal places.
    none_value : str, optional
        Text used for missing values.
    column_spacing : int, optional
        Spaces between columns.
    maximum_column_width : int or None, optional
        Maximum column width.
    show_header : bool, optional
        Whether headers should be shown.
    show_separator : bool, optional
        Whether a separator should be placed below the headers.
    empty_message : str, optional
        Text displayed when no rows are available.
    title_character : str, optional
        Character used for the title separator.
    logger : DockLogger, optional
        Logger used instead of ``print``.

    Returns
    -------
    str
        Complete formatted output.
    """

    table_text = format_table(
        data=data,
        headers=headers,
        alignments=alignments,
        float_precision=float_precision,
        none_value=none_value,
        column_spacing=column_spacing,
        maximum_column_width=maximum_column_width,
        show_header=show_header,
        show_separator=show_separator,
        empty_message=empty_message,
    )

    output_sections: List[str] = []

    if title is not None:
        table_width = max(
            (
                len(line)
                for line in table_text.splitlines()
            ),
            default=22,
        )

        output_sections.append(
            format_title(
                title=title,
                width=max(
                    table_width,
                    22,
                ),
                character=title_character,
            )
        )

    output_sections.append(
        table_text
    )

    complete_output = "\n".join(
        output_sections
    )

    if logger is not None:
        logger.info(
            "\n%s",
            complete_output,
        )

    else:
        print(
            complete_output
        )

    return complete_output


# -----------------------------------------------------------------------------
# Public module interface
# -----------------------------------------------------------------------------

_SECTION_9_PUBLIC_NAMES = [
    "format_title",
    "format_table",
    "print_title",
    "print_table",
]

for public_name in _SECTION_9_PUBLIC_NAMES:
    if public_name not in __all__:
        __all__.append(public_name)


# =============================================================================
# End of Section 9
# =============================================================================


# =============================================================================
# Section 10 — General Helpers
# =============================================================================


# -----------------------------------------------------------------------------
# List conversion
# -----------------------------------------------------------------------------

def ensure_list(
    value: Any,
    *,
    none_as_empty: bool = True,
    preserve_tuple: bool = False,
) -> Union[List[Any], Tuple[Any, ...]]:
    """
    Convert a value into a list-like collection.

    Parameters
    ----------
    value : Any
        Value to convert.
    none_as_empty : bool, optional
        Whether ``None`` should become an empty list. When ``False``,
        ``None`` becomes ``[None]``.
    preserve_tuple : bool, optional
        Whether tuples should be returned unchanged.

    Returns
    -------
    list or tuple
        Normalized collection.

    Notes
    -----
    Strings are treated as single values rather than iterables.

    Examples
    --------
    >>> ensure_list(5)
    [5]

    >>> ensure_list((1, 2))
    [1, 2]

    >>> ensure_list(None)
    []
    """

    if value is None:
        return (
            []
            if none_as_empty
            else [None]
        )

    if isinstance(value, list):
        return value

    if isinstance(value, tuple):
        if preserve_tuple:
            return value

        return list(value)

    if isinstance(
        value,
        (
            str,
            bytes,
            bytearray,
            dict,
            Path,
        ),
    ):
        return [value]

    if isinstance(
        value,
        Iterable,
    ):
        return list(value)

    return [value]


# -----------------------------------------------------------------------------
# Flattening
# -----------------------------------------------------------------------------

def flatten(
    values: Any,
    *,
    recursive: bool = True,
    iterable_types: Tuple[type, ...] = (
        list,
        tuple,
        set,
    ),
    ignore_none: bool = False,
) -> List[Any]:
    """
    Flatten nested collections.

    Parameters
    ----------
    values : Any
        Collection to flatten.
    recursive : bool, optional
        Whether nested collections should be flattened recursively. When
        ``False``, only one nesting level is removed.
    iterable_types : tuple of type, optional
        Collection types considered flattenable.
    ignore_none : bool, optional
        Whether ``None`` values should be omitted.

    Returns
    -------
    list
        Flattened values.

    Notes
    -----
    Strings and dictionaries are preserved as individual values unless their
    types are explicitly included in ``iterable_types``.

    Examples
    --------
    >>> flatten([[1, 2], [3, 4]])
    [1, 2, 3, 4]

    >>> flatten([1, [2, [3, 4]]])
    [1, 2, 3, 4]

    >>> flatten([1, [2, [3, 4]]], recursive=False)
    [1, 2, [3, 4]]
    """

    if values is None:
        return []

    if not isinstance(
        values,
        iterable_types,
    ):
        return (
            []
            if (
                values is None
                and ignore_none
            )
            else [values]
        )

    flattened_values: List[Any] = []

    for value in values:
        if value is None and ignore_none:
            continue

        if isinstance(
            value,
            iterable_types,
        ):
            if recursive:
                flattened_values.extend(
                    flatten(
                        value,
                        recursive=True,
                        iterable_types=iterable_types,
                        ignore_none=ignore_none,
                    )
                )

            else:
                flattened_values.extend(
                    value
                )

        else:
            flattened_values.append(
                value
            )

    return flattened_values


# -----------------------------------------------------------------------------
# Sequence chunking
# -----------------------------------------------------------------------------

def chunks(
    values: Iterable[Any],
    size: int,
    *,
    as_iterator: bool = False,
) -> Union[
    List[List[Any]],
    Iterator[List[Any]],
]:
    """
    Split values into consecutive chunks.

    Parameters
    ----------
    values : iterable
        Values to divide.
    size : int
        Maximum number of values in each chunk.
    as_iterator : bool, optional
        Whether the result should be returned as an iterator instead of a
        complete list.

    Returns
    -------
    list or iterator
        Consecutive chunks.

    Raises
    ------
    TypeError
        If ``size`` is not an integer.
    ValueError
        If ``size`` is less than one.

    Examples
    --------
    >>> chunks([1, 2, 3, 4, 5], 2)
    [[1, 2], [3, 4], [5]]
    """

    if isinstance(
        size,
        bool,
    ) or not isinstance(
        size,
        int,
    ):
        raise TypeError(
            "Chunk size must be an integer."
        )

    if size < 1:
        raise ValueError(
            "Chunk size must be greater than zero."
        )

    if values is None:
        raise TypeError(
            "Values cannot be None."
        )

    def chunk_iterator() -> Iterator[
        List[Any]
    ]:
        current_chunk: List[Any] = []

        for value in values:
            current_chunk.append(
                value
            )

            if len(current_chunk) == size:
                yield current_chunk
                current_chunk = []

        if current_chunk:
            yield current_chunk

    iterator = chunk_iterator()

    if as_iterator:
        return iterator

    return list(iterator)


# -----------------------------------------------------------------------------
# Pairwise iteration
# -----------------------------------------------------------------------------

def pairwise(
    values: Iterable[Any],
    *,
    circular: bool = False,
) -> Iterator[Tuple[Any, Any]]:
    """
    Iterate over consecutive pairs.

    Parameters
    ----------
    values : iterable
        Input values.
    circular : bool, optional
        Whether the final value should be paired with the first value.

    Yields
    ------
    tuple
        Consecutive value pairs.

    Examples
    --------
    >>> list(pairwise([1, 2, 3, 4]))
    [(1, 2), (2, 3), (3, 4)]

    >>> list(pairwise([1, 2, 3], circular=True))
    [(1, 2), (2, 3), (3, 1)]
    """

    if values is None:
        raise TypeError(
            "Values cannot be None."
        )

    iterator = iter(values)

    try:
        first_value = next(
            iterator
        )

    except StopIteration:
        return

    previous_value = first_value

    for current_value in iterator:
        yield (
            previous_value,
            current_value,
        )

        previous_value = current_value

    if (
        circular
        and previous_value is not first_value
    ):
        yield (
            previous_value,
            first_value,
        )


# -----------------------------------------------------------------------------
# Optional-value selection
# -----------------------------------------------------------------------------

def first_not_none(
    *values: Any,
    default: Any = None,
) -> Any:
    """
    Return the first value that is not ``None``.

    Parameters
    ----------
    *values : Any
        Candidate values.
    default : Any, optional
        Value returned when all candidates are ``None``.

    Returns
    -------
    Any
        First non-``None`` value or the supplied default.

    Examples
    --------
    >>> first_not_none(None, None, 5, 10)
    5

    >>> first_not_none(None, None, default="unknown")
    'unknown'
    """

    for value in values:
        if value is not None:
            return value

    return default


# -----------------------------------------------------------------------------
# Empty-value removal
# -----------------------------------------------------------------------------

def compact(
    values: Iterable[Any],
    *,
    remove_none: bool = True,
    remove_empty_strings: bool = False,
    remove_false: bool = False,
) -> List[Any]:
    """
    Remove selected empty values from a collection.

    Parameters
    ----------
    values : iterable
        Input values.
    remove_none : bool, optional
        Whether ``None`` values should be removed.
    remove_empty_strings : bool, optional
        Whether empty or whitespace-only strings should be removed.
    remove_false : bool, optional
        Whether all false-like values should be removed.

    Returns
    -------
    list
        Filtered values.

    Examples
    --------
    >>> compact([1, None, 2, None])
    [1, 2]

    >>> compact(
    ...     ["A", "", " ", "B"],
    ...     remove_empty_strings=True,
    ... )
    ['A', 'B']
    """

    if values is None:
        return []

    compacted_values: List[Any] = []

    for value in values:
        if (
            remove_none
            and value is None
        ):
            continue

        if (
            remove_empty_strings
            and isinstance(
                value,
                str,
            )
            and not value.strip()
        ):
            continue

        if (
            remove_false
            and not value
        ):
            continue

        compacted_values.append(
            value
        )

    return compacted_values


# -----------------------------------------------------------------------------
# Public module interface
# -----------------------------------------------------------------------------

_SECTION_10_PUBLIC_NAMES = [
    "ensure_list",
    "flatten",
    "chunks",
    "pairwise",
    "first_not_none",
    "compact",
]

for public_name in _SECTION_10_PUBLIC_NAMES:
    if public_name not in __all__:
        __all__.append(public_name)


# =============================================================================
# End of Section 10
# =============================================================================

# =============================================================================
# Section 11.1 — Decorator Base Infrastructure
# =============================================================================


# -----------------------------------------------------------------------------
# Internal constants and sentinels
# -----------------------------------------------------------------------------

_DECORATOR_UNSET = object()
_DECORATOR_MISSING = object()

_DECORATOR_LOG_LEVELS = {
    "debug",
    "info",
    "warning",
    "error",
    "critical",
    "exception",
}


# -----------------------------------------------------------------------------
# Internal decorator types
# -----------------------------------------------------------------------------

F = TypeVar(
    "F",
    bound=Callable[..., Any],
)

R = TypeVar(
    "R",
)


DecoratorFunction = Callable[
    [F],
    F,
]


# -----------------------------------------------------------------------------
# Basic callable inspection
# -----------------------------------------------------------------------------

def _is_callable_object(
    value: Any,
) -> bool:
    """
    Return whether a value can be called.

    Parameters
    ----------
    value : Any
        Value to inspect.

    Returns
    -------
    bool
        ``True`` when the value is callable.
    """

    return callable(value)


def _is_decorated_function_candidate(
    value: Any,
) -> bool:
    """
    Return whether a value appears to be a function passed directly to a
    decorator.

    This helper is used to distinguish:

    ``@decorator``

    from:

    ``@decorator(...)``

    Parameters
    ----------
    value : Any
        Candidate value.

    Returns
    -------
    bool
        Whether the value appears to be a callable suitable for direct
        decoration.
    """

    if not callable(value):
        return False

    return (
        inspect.isfunction(value)
        or inspect.ismethod(value)
        or inspect.iscoroutinefunction(value)
        or hasattr(value, "__call__")
    )


def _is_coroutine_callable(
    function: Callable[..., Any],
) -> bool:
    """
    Return whether a callable must be awaited.

    Parameters
    ----------
    function : callable
        Function or callable object.

    Returns
    -------
    bool
        ``True`` for asynchronous callables.
    """

    if inspect.iscoroutinefunction(function):
        return True

    call_method = getattr(
        function,
        "__call__",
        None,
    )

    return bool(
        call_method is not None
        and inspect.iscoroutinefunction(
            call_method
        )
    )


# -----------------------------------------------------------------------------
# Function naming
# -----------------------------------------------------------------------------

def _resolve_function_name(
    function: Callable[..., Any],
    *,
    qualified: bool = True,
    include_module: bool = False,
    fallback: str = "<callable>",
) -> str:
    """
    Return a readable name for a callable.

    Parameters
    ----------
    function : callable
        Callable to describe.
    qualified : bool, optional
        Whether ``__qualname__`` should be preferred over ``__name__``.
    include_module : bool, optional
        Whether the module name should be prepended.
    fallback : str, optional
        Name used when no callable name can be resolved.

    Returns
    -------
    str
        Readable callable name.

    Examples
    --------
    A method may be represented as:

    ``DockAnalyzer.analyze_contacts``

    With ``include_module=True``:

    ``dockanalyzer.analysis.DockAnalyzer.analyze_contacts``
    """

    if function is None:
        return fallback

    function_name = None

    if qualified:
        function_name = getattr(
            function,
            "__qualname__",
            None,
        )

    if not function_name:
        function_name = getattr(
            function,
            "__name__",
            None,
        )

    if not function_name:
        function_class = getattr(
            function,
            "__class__",
            None,
        )

        if function_class is not None:
            function_name = getattr(
                function_class,
                "__qualname__",
                None,
            )

    if not function_name:
        function_name = fallback

    if include_module:
        module_name = getattr(
            function,
            "__module__",
            None,
        )

        if (
            module_name
            and module_name
            not in {
                "__main__",
                "builtins",
            }
        ):
            return (
                f"{module_name}."
                f"{function_name}"
            )

    return str(
        function_name
    )


def _resolve_callable_label(
    function: Callable[..., Any],
    *,
    custom_name: Optional[str] = None,
    include_module: bool = False,
) -> str:
    """
    Resolve a custom or automatically generated label for a callable.

    Parameters
    ----------
    function : callable
        Callable being decorated.
    custom_name : str, optional
        Explicit label.
    include_module : bool, optional
        Whether the module should be included in automatically generated
        labels.

    Returns
    -------
    str
        Callable label.
    """

    if custom_name is not None:
        custom_name = str(
            custom_name
        ).strip()

        if custom_name:
            return custom_name

    return _resolve_function_name(
        function,
        qualified=True,
        include_module=include_module,
    )


# -----------------------------------------------------------------------------
# Object attribute resolution
# -----------------------------------------------------------------------------

def _get_bound_instance(
    args: Tuple[Any, ...],
) -> Any:
    """
    Return the likely bound instance from positional arguments.

    For an instance method, the first positional argument is generally
    ``self``. For a class method, it is generally ``cls``.

    Parameters
    ----------
    args : tuple
        Positional call arguments.

    Returns
    -------
    Any
        First positional argument or ``None``.
    """

    if not args:
        return None

    return args[0]


def _get_attribute_from_candidates(
    obj: Any,
    names: Iterable[str],
    *,
    default: Any = None,
) -> Any:
    """
    Return the first available attribute from a list of candidate names.

    Parameters
    ----------
    obj : Any
        Object to inspect.
    names : iterable of str
        Candidate attribute names.
    default : Any, optional
        Value returned when no attribute is available.

    Returns
    -------
    Any
        First resolved attribute or the default value.
    """

    if obj is None:
        return default

    for name in names:
        try:
            value = getattr(
                obj,
                name,
            )

        except (
            AttributeError,
            TypeError,
            RuntimeError,
        ):
            continue

        if value is not None:
            return value

    return default


def _get_mapping_value_from_candidates(
    mapping: Any,
    names: Iterable[str],
    *,
    default: Any = None,
) -> Any:
    """
    Return the first available item from a mapping-like object.

    Parameters
    ----------
    mapping : Any
        Mapping to inspect.
    names : iterable of str
        Candidate keys.
    default : Any, optional
        Value returned when no key is found.

    Returns
    -------
    Any
        First resolved item or the default value.
    """

    if not isinstance(
        mapping,
        Mapping,
    ):
        return default

    for name in names:
        if name in mapping:
            value = mapping[name]

            if value is not None:
                return value

    return default


# -----------------------------------------------------------------------------
# Logger detection and resolution
# -----------------------------------------------------------------------------

def _is_logger_like(
    value: Any,
) -> bool:
    """
    Return whether an object provides a logger-compatible interface.

    The object does not need to inherit from ``logging.Logger``. This permits
    use with ``DockLogger`` and ChimeraX logger adapters.

    Parameters
    ----------
    value : Any
        Object to inspect.

    Returns
    -------
    bool
        Whether the object behaves like a logger.
    """

    if value is None:
        return False

    common_methods = (
        "debug",
        "info",
        "warning",
        "error",
        "critical",
        "exception",
    )

    available_methods = sum(
        callable(
            getattr(
                value,
                method_name,
                None,
            )
        )
        for method_name in common_methods
    )

    return available_methods >= 2


def _resolve_decorator_logger(
    *,
    explicit_logger: Any = None,
    args: Tuple[Any, ...] = (),
    kwargs: Optional[Mapping[str, Any]] = None,
    fallback: Any = None,
) -> Any:
    """
    Resolve the logger that should be used by a decorator.

    Resolution order
    ----------------
    1. Explicit logger supplied to the decorator.
    2. Logger supplied in the decorated function's keyword arguments.
    3. Logger attached to the bound object.
    4. Logger attached to a session object.
    5. Fallback logger.

    Parameters
    ----------
    explicit_logger : Any, optional
        Logger passed directly to the decorator.
    args : tuple, optional
        Positional function arguments.
    kwargs : mapping, optional
        Keyword function arguments.
    fallback : Any, optional
        Fallback logger.

    Returns
    -------
    Any
        Resolved logger or ``None``.
    """

    if _is_logger_like(
        explicit_logger
    ):
        return explicit_logger

    kwargs = kwargs or {}

    keyword_logger = (
        _get_mapping_value_from_candidates(
            kwargs,
            (
                "logger",
                "log",
                "dock_logger",
            ),
        )
    )

    if _is_logger_like(
        keyword_logger
    ):
        return keyword_logger

    bound_instance = _get_bound_instance(
        args
    )

    instance_logger = (
        _get_attribute_from_candidates(
            bound_instance,
            (
                "logger",
                "log",
                "dock_logger",
                "_logger",
            ),
        )
    )

    if _is_logger_like(
        instance_logger
    ):
        return instance_logger

    session = _get_mapping_value_from_candidates(
        kwargs,
        (
            "session",
            "chimerax_session",
        ),
    )

    if session is None:
        session = _get_attribute_from_candidates(
            bound_instance,
            (
                "session",
                "_session",
                "chimerax_session",
            ),
        )

    session_logger = (
        _get_attribute_from_candidates(
            session,
            (
                "logger",
                "log",
            ),
        )
    )

    if _is_logger_like(
        session_logger
    ):
        return session_logger

    if _is_logger_like(
        fallback
    ):
        return fallback

    return None


# -----------------------------------------------------------------------------
# Timer detection and resolution
# -----------------------------------------------------------------------------

def _is_timer_like(
    value: Any,
) -> bool:
    """
    Return whether an object behaves like an analysis timer.

    The check is intentionally flexible so it supports ``AnalysisTimer``
    without tightly coupling this section to one implementation.

    Parameters
    ----------
    value : Any
        Object to inspect.

    Returns
    -------
    bool
        Whether the value provides a usable timer interface.
    """

    if value is None:
        return False

    if isinstance(
        value,
        AnalysisTimer,
    ):
        return True

    context_methods = (
        "__enter__",
        "__exit__",
    )

    if all(
        callable(
            getattr(
                value,
                method_name,
                None,
            )
        )
        for method_name in context_methods
    ):
        return True

    timer_methods = (
        "start",
        "stop",
    )

    return all(
        callable(
            getattr(
                value,
                method_name,
                None,
            )
        )
        for method_name in timer_methods
    )


def _resolve_decorator_timer(
    *,
    explicit_timer: Any = None,
    args: Tuple[Any, ...] = (),
    kwargs: Optional[Mapping[str, Any]] = None,
    fallback: Any = None,
) -> Any:
    """
    Resolve an ``AnalysisTimer`` or timer-like object.

    Resolution order
    ----------------
    1. Explicit timer supplied to the decorator.
    2. Timer supplied in the decorated function's keyword arguments.
    3. Timer attached to the bound object.
    4. Fallback timer.

    Parameters
    ----------
    explicit_timer : Any, optional
        Timer passed directly to the decorator.
    args : tuple, optional
        Positional function arguments.
    kwargs : mapping, optional
        Keyword function arguments.
    fallback : Any, optional
        Fallback timer.

    Returns
    -------
    Any
        Resolved timer or ``None``.
    """

    if _is_timer_like(
        explicit_timer
    ):
        return explicit_timer

    kwargs = kwargs or {}

    keyword_timer = (
        _get_mapping_value_from_candidates(
            kwargs,
            (
                "timer",
                "analysis_timer",
                "dock_timer",
            ),
        )
    )

    if _is_timer_like(
        keyword_timer
    ):
        return keyword_timer

    bound_instance = _get_bound_instance(
        args
    )

    instance_timer = (
        _get_attribute_from_candidates(
            bound_instance,
            (
                "timer",
                "analysis_timer",
                "dock_timer",
                "_timer",
            ),
        )
    )

    if _is_timer_like(
        instance_timer
    ):
        return instance_timer

    if _is_timer_like(
        fallback
    ):
        return fallback

    return None


# -----------------------------------------------------------------------------
# Logging abstraction
# -----------------------------------------------------------------------------

def _normalize_log_level(
    level: str,
    *,
    default: str = "info",
) -> str:
    """
    Normalize and validate a logging level.

    Parameters
    ----------
    level : str
        Requested logging level.
    default : str, optional
        Default level used for empty values.

    Returns
    -------
    str
        Normalized logging level.

    Raises
    ------
    ValueError
        If the level is unsupported.
    """

    if level is None:
        normalized_level = default

    else:
        normalized_level = str(
            level
        ).strip().lower()

    if not normalized_level:
        normalized_level = default

    if (
        normalized_level
        not in _DECORATOR_LOG_LEVELS
    ):
        valid_levels = ", ".join(
            sorted(
                _DECORATOR_LOG_LEVELS
            )
        )

        raise ValueError(
            f"Unsupported log level "
            f"{level!r}. Expected one of: "
            f"{valid_levels}."
        )

    return normalized_level


def _log_message(
    logger: Any,
    level: str,
    message: str,
    *message_args: Any,
    exc_info: Any = None,
) -> bool:
    """
    Send a message to a logger-like object.

    The function supports:

    - ``DockLogger``;
    - ``logging.Logger``;
    - ChimeraX logger-like objects;
    - minimal custom loggers.

    Parameters
    ----------
    logger : Any
        Logger-like object.
    level : str
        Logging level.
    message : str
        Message template.
    *message_args : Any
        Optional template arguments.
    exc_info : Any, optional
        Exception information forwarded when supported.

    Returns
    -------
    bool
        ``True`` when logging succeeded.
    """

    if not _is_logger_like(
        logger
    ):
        return False

    normalized_level = _normalize_log_level(
        level
    )

    log_method = getattr(
        logger,
        normalized_level,
        None,
    )

    if not callable(
        log_method
    ):
        fallback_methods = (
            "error",
            "warning",
            "info",
            "debug",
        )

        for fallback_method_name in fallback_methods:
            candidate_method = getattr(
                logger,
                fallback_method_name,
                None,
            )

            if callable(
                candidate_method
            ):
                log_method = candidate_method
                break

    if not callable(
        log_method
    ):
        return False

    try:
        if exc_info is not None:
            log_method(
                message,
                *message_args,
                exc_info=exc_info,
            )

        else:
            log_method(
                message,
                *message_args,
            )

        return True

    except TypeError:
        try:
            if message_args:
                formatted_message = (
                    message
                    % message_args
                )

            else:
                formatted_message = message

        except Exception:
            formatted_message = " ".join(
                [
                    str(message),
                    *(
                        str(argument)
                        for argument
                        in message_args
                    ),
                ]
            )

        try:
            log_method(
                formatted_message
            )

            return True

        except Exception:
            return False

    except Exception:
        return False


def _log_or_print(
    message: str,
    *,
    logger: Any = None,
    level: str = "info",
    fallback_to_print: bool = False,
    exc_info: Any = None,
) -> None:
    """
    Log a message and optionally fall back to ``print``.

    Parameters
    ----------
    message : str
        Message to emit.
    logger : Any, optional
        Logger-like object.
    level : str, optional
        Logging level.
    fallback_to_print : bool, optional
        Whether ``print`` should be used when no logger is available.
    exc_info : Any, optional
        Exception information.
    """

    logged = _log_message(
        logger,
        level,
        message,
        exc_info=exc_info,
    )

    if (
        not logged
        and fallback_to_print
    ):
        print(
            message
        )


# -----------------------------------------------------------------------------
# Safe value representation
# -----------------------------------------------------------------------------

def _truncate_decorator_text(
    text: Any,
    *,
    maximum_length: Optional[int] = 200,
    suffix: str = "...",
) -> str:
    """
    Truncate text used in decorator messages.

    Parameters
    ----------
    text : Any
        Text-like value.
    maximum_length : int or None, optional
        Maximum number of characters. ``None`` disables truncation.
    suffix : str, optional
        Suffix appended to truncated values.

    Returns
    -------
    str
        Truncated text.
    """

    string_value = str(
        text
    )

    if maximum_length is None:
        return string_value

    maximum_length = int(
        maximum_length
    )

    if maximum_length <= 0:
        return ""

    if len(string_value) <= maximum_length:
        return string_value

    if maximum_length <= len(suffix):
        return string_value[
            :maximum_length
        ]

    return (
        string_value[
            :maximum_length
            - len(suffix)
        ]
        + suffix
    )


def _safe_length(
    value: Any,
) -> Optional[int]:
    """
    Return the length of a value without propagating errors.

    Parameters
    ----------
    value : Any
        Value to inspect.

    Returns
    -------
    int or None
        Length or ``None`` when unavailable.
    """

    try:
        return len(value)

    except (
        TypeError,
        AttributeError,
        RuntimeError,
    ):
        return None


def _safe_shape(
    value: Any,
) -> Optional[Tuple[Any, ...]]:
    """
    Return a normalized shape tuple when available.

    Parameters
    ----------
    value : Any
        Value to inspect.

    Returns
    -------
    tuple or None
        Shape information.
    """

    shape = getattr(
        value,
        "shape",
        None,
    )

    if shape is None:
        return None

    try:
        return tuple(
            shape
        )

    except TypeError:
        return None


def _safe_object_name(
    value: Any,
) -> Optional[str]:
    """
    Return a readable object name when available.

    Parameters
    ----------
    value : Any
        Object to inspect.

    Returns
    -------
    str or None
        Resolved object name.
    """

    for attribute_name in (
        "name",
        "id_string",
        "atomspec",
    ):
        attribute_value = getattr(
            value,
            attribute_name,
            None,
        )

        if attribute_value:
            return str(
                attribute_value
            )

    return None


def _format_decorator_value(
    value: Any,
    *,
    maximum_length: Optional[int] = 200,
    maximum_items: int = 6,
    depth: int = 0,
    maximum_depth: int = 2,
    include_type: bool = False,
) -> str:
    """
    Build a concise and safe representation of a value.

    This function avoids flooding logs with large arrays, DataFrames,
    structures, atom collections or deeply nested containers.

    Parameters
    ----------
    value : Any
        Value to format.
    maximum_length : int or None, optional
        Maximum output length.
    maximum_items : int, optional
        Maximum displayed items for containers.
    depth : int, optional
        Current recursion depth.
    maximum_depth : int, optional
        Maximum recursion depth.
    include_type : bool, optional
        Whether the Python type name should be included.

    Returns
    -------
    str
        Concise value representation.
    """

    type_name = type(
        value
    ).__name__

    if value is None:
        representation = "None"

    elif isinstance(
        value,
        (
            bool,
            int,
            float,
            complex,
            np.number,
        ),
    ):
        representation = repr(
            value
        )

    elif isinstance(
        value,
        str,
    ):
        representation = repr(
            _truncate_decorator_text(
                value,
                maximum_length=maximum_length,
            )
        )

    elif isinstance(
        value,
        (
            bytes,
            bytearray,
        ),
    ):
        value_length = _safe_length(
            value
        )

        representation = (
            f"<{type_name} "
            f"length={value_length}>"
        )

    elif isinstance(
        value,
        Path,
    ):
        representation = str(
            value
        )

    elif isinstance(
        value,
        np.ndarray,
    ):
        representation = (
            f"<ndarray "
            f"shape={value.shape} "
            f"dtype={value.dtype}>"
        )

    elif isinstance(
        value,
        pd.DataFrame,
    ):
        representation = (
            f"<DataFrame "
            f"shape={value.shape} "
            f"columns={list(value.columns)[:maximum_items]!r}>"
        )

    elif isinstance(
        value,
        pd.Series,
    ):
        representation = (
            f"<Series "
            f"length={len(value)} "
            f"name={value.name!r} "
            f"dtype={value.dtype}>"
        )

    elif isinstance(
        value,
        DockModel,
    ):
        model_name = first_not_none(
            getattr(
                value,
                "name",
                None,
            ),
            getattr(
                value,
                "pose_name",
                None,
            ),
            default=None,
        )

        if model_name:
            representation = (
                f"<DockModel "
                f"name={model_name!r}>"
            )

        else:
            representation = "<DockModel>"

    elif isinstance(
        value,
        Mapping,
    ):
        if depth >= maximum_depth:
            representation = (
                f"<{type_name} "
                f"length={len(value)}>"
            )

        else:
            mapping_items = list(
                value.items()
            )

            displayed_items = []

            for key, item_value in mapping_items[
                :maximum_items
            ]:
                formatted_key = (
                    _format_decorator_value(
                        key,
                        maximum_length=50,
                        maximum_items=maximum_items,
                        depth=depth + 1,
                        maximum_depth=maximum_depth,
                    )
                )

                formatted_value = (
                    _format_decorator_value(
                        item_value,
                        maximum_length=80,
                        maximum_items=maximum_items,
                        depth=depth + 1,
                        maximum_depth=maximum_depth,
                    )
                )

                displayed_items.append(
                    f"{formatted_key}: "
                    f"{formatted_value}"
                )

            if (
                len(mapping_items)
                > maximum_items
            ):
                displayed_items.append(
                    "..."
                )

            representation = (
                "{"
                + ", ".join(
                    displayed_items
                )
                + "}"
            )

    elif isinstance(
        value,
        (
            list,
            tuple,
            set,
            frozenset,
        ),
    ):
        if depth >= maximum_depth:
            representation = (
                f"<{type_name} "
                f"length={len(value)}>"
            )

        else:
            sequence_values = list(
                value
            )

            displayed_values = [
                _format_decorator_value(
                    item,
                    maximum_length=80,
                    maximum_items=maximum_items,
                    depth=depth + 1,
                    maximum_depth=maximum_depth,
                )
                for item in sequence_values[
                    :maximum_items
                ]
            ]

            if (
                len(sequence_values)
                > maximum_items
            ):
                displayed_values.append(
                    "..."
                )

            opening_character = (
                "("
                if isinstance(
                    value,
                    tuple,
                )
                else (
                    "{"
                    if isinstance(
                        value,
                        (
                            set,
                            frozenset,
                        ),
                    )
                    else "["
                )
            )

            closing_character = (
                ")"
                if isinstance(
                    value,
                    tuple,
                )
                else (
                    "}"
                    if isinstance(
                        value,
                        (
                            set,
                            frozenset,
                        ),
                    )
                    else "]"
                )
            )

            representation = (
                opening_character
                + ", ".join(
                    displayed_values
                )
                + closing_character
            )

    else:
        object_name = _safe_object_name(
            value
        )

        shape = _safe_shape(
            value
        )

        value_length = _safe_length(
            value
        )

        if object_name:
            representation = (
                f"<{type_name} "
                f"name={object_name!r}>"
            )

        elif shape is not None:
            representation = (
                f"<{type_name} "
                f"shape={shape}>"
            )

        elif value_length is not None:
            representation = (
                f"<{type_name} "
                f"length={value_length}>"
            )

        else:
            try:
                representation = repr(
                    value
                )

            except Exception:
                representation = (
                    f"<{type_name} "
                    f"at {hex(id(value))}>"
                )

    representation = (
        _truncate_decorator_text(
            representation,
            maximum_length=maximum_length,
        )
    )

    if include_type:
        if not representation.startswith(
            f"<{type_name}"
        ):
            representation = (
                f"{type_name}("
                f"{representation}"
                f")"
            )

    return representation


# -----------------------------------------------------------------------------
# Function signature and argument formatting
# -----------------------------------------------------------------------------

def _safe_signature(
    function: Callable[..., Any],
) -> Optional[inspect.Signature]:
    """
    Return a callable signature without propagating inspection errors.

    Parameters
    ----------
    function : callable
        Callable to inspect.

    Returns
    -------
    inspect.Signature or None
        Resolved signature.
    """

    try:
        return inspect.signature(
            function
        )

    except (
        TypeError,
        ValueError,
    ):
        return None


def _bind_call_arguments(
    function: Callable[..., Any],
    args: Tuple[Any, ...],
    kwargs: Mapping[str, Any],
    *,
    apply_defaults: bool = False,
) -> Optional[
    inspect.BoundArguments
]:
    """
    Bind runtime arguments to a callable signature.

    Parameters
    ----------
    function : callable
        Called function.
    args : tuple
        Positional arguments.
    kwargs : mapping
        Keyword arguments.
    apply_defaults : bool, optional
        Whether default parameter values should be included.

    Returns
    -------
    inspect.BoundArguments or None
        Bound arguments when signature inspection succeeds.
    """

    signature = _safe_signature(
        function
    )

    if signature is None:
        return None

    try:
        bound_arguments = (
            signature.bind_partial(
                *args,
                **kwargs,
            )
        )

        if apply_defaults:
            bound_arguments.apply_defaults()

        return bound_arguments

    except TypeError:
        return None


def _is_sensitive_argument_name(
    name: str,
    *,
    additional_names: Optional[
        Iterable[str]
    ] = None,
) -> bool:
    """
    Return whether an argument name may contain sensitive information.

    Parameters
    ----------
    name : str
        Argument name.
    additional_names : iterable of str, optional
        Additional protected names.

    Returns
    -------
    bool
        Whether the value should be redacted.
    """

    normalized_name = str(
        name
    ).strip().lower()

    sensitive_names = {
        "password",
        "passwd",
        "pwd",
        "secret",
        "token",
        "access_token",
        "refresh_token",
        "api_key",
        "apikey",
        "authorization",
        "auth",
        "credential",
        "credentials",
        "private_key",
    }

    if additional_names is not None:
        sensitive_names.update(
            str(item).strip().lower()
            for item in additional_names
        )

    return any(
        sensitive_name
        in normalized_name
        for sensitive_name
        in sensitive_names
    )


def _format_call_arguments(
    function: Callable[..., Any],
    args: Tuple[Any, ...],
    kwargs: Mapping[str, Any],
    *,
    include_self: bool = False,
    include_defaults: bool = False,
    maximum_value_length: int = 160,
    maximum_items: int = 6,
    redact_sensitive: bool = True,
    sensitive_names: Optional[
        Iterable[str]
    ] = None,
    multiline: bool = False,
) -> str:
    """
    Format runtime function arguments for logging.

    Parameters
    ----------
    function : callable
        Called function.
    args : tuple
        Positional arguments.
    kwargs : mapping
        Keyword arguments.
    include_self : bool, optional
        Whether ``self`` and ``cls`` should be displayed.
    include_defaults : bool, optional
        Whether default argument values should be included.
    maximum_value_length : int, optional
        Maximum representation length per argument.
    maximum_items : int, optional
        Maximum displayed container items.
    redact_sensitive : bool, optional
        Whether sensitive argument values should be replaced.
    sensitive_names : iterable of str, optional
        Additional argument names to redact.
    multiline : bool, optional
        Whether each argument should appear on a separate line.

    Returns
    -------
    str
        Formatted argument string.
    """

    bound_arguments = _bind_call_arguments(
        function,
        args,
        kwargs,
        apply_defaults=include_defaults,
    )

    argument_items: List[
        Tuple[str, Any]
    ] = []

    if bound_arguments is not None:
        argument_items = list(
            bound_arguments.arguments.items()
        )

    else:
        argument_items.extend(
            (
                f"arg_{index}",
                value,
            )
            for index, value in enumerate(
                args,
                start=1,
            )
        )

        argument_items.extend(
            kwargs.items()
        )

    formatted_arguments = []

    for argument_name, argument_value in argument_items:
        if (
            not include_self
            and argument_name
            in {
                "self",
                "cls",
            }
        ):
            continue

        if (
            redact_sensitive
            and _is_sensitive_argument_name(
                argument_name,
                additional_names=sensitive_names,
            )
        ):
            formatted_value = (
                "<redacted>"
            )

        else:
            formatted_value = (
                _format_decorator_value(
                    argument_value,
                    maximum_length=(
                        maximum_value_length
                    ),
                    maximum_items=maximum_items,
                )
            )

        formatted_arguments.append(
            f"{argument_name}="
            f"{formatted_value}"
        )

    if not formatted_arguments:
        return ""

    if multiline:
        return "\n".join(
            f"    {argument}"
            for argument
            in formatted_arguments
        )

    return ", ".join(
        formatted_arguments
    )


def _format_call_signature(
    function: Callable[..., Any],
    args: Tuple[Any, ...],
    kwargs: Mapping[str, Any],
    *,
    function_name: Optional[str] = None,
    include_arguments: bool = True,
    include_self: bool = False,
    maximum_value_length: int = 160,
    maximum_items: int = 6,
    redact_sensitive: bool = True,
) -> str:
    """
    Build a readable runtime call signature.

    Parameters
    ----------
    function : callable
        Called function.
    args : tuple
        Positional arguments.
    kwargs : mapping
        Keyword arguments.
    function_name : str, optional
        Custom function label.
    include_arguments : bool, optional
        Whether call arguments should be included.
    include_self : bool, optional
        Whether ``self`` and ``cls`` should be displayed.
    maximum_value_length : int, optional
        Maximum argument representation length.
    maximum_items : int, optional
        Maximum container items.
    redact_sensitive : bool, optional
        Whether sensitive arguments should be redacted.

    Returns
    -------
    str
        Formatted call signature.
    """

    resolved_name = _resolve_callable_label(
        function,
        custom_name=function_name,
    )

    if not include_arguments:
        return f"{resolved_name}()"

    formatted_arguments = (
        _format_call_arguments(
            function,
            args,
            kwargs,
            include_self=include_self,
            maximum_value_length=(
                maximum_value_length
            ),
            maximum_items=maximum_items,
            redact_sensitive=redact_sensitive,
        )
    )

    return (
        f"{resolved_name}("
        f"{formatted_arguments}"
        f")"
    )


# -----------------------------------------------------------------------------
# Exception formatting
# -----------------------------------------------------------------------------

def _format_exception(
    exception: BaseException,
    *,
    function: Optional[
        Callable[..., Any]
    ] = None,
    function_name: Optional[str] = None,
    include_traceback: bool = True,
    include_exception_type: bool = True,
    maximum_length: Optional[int] = None,
) -> str:
    """
    Format an exception for decorator logs.

    Parameters
    ----------
    exception : BaseException
        Captured exception.
    function : callable, optional
        Function that raised the exception.
    function_name : str, optional
        Custom function label.
    include_traceback : bool, optional
        Whether the traceback should be included.
    include_exception_type : bool, optional
        Whether the exception class name should be included.
    maximum_length : int or None, optional
        Maximum formatted message length.

    Returns
    -------
    str
        Formatted exception report.
    """

    if not isinstance(
        exception,
        BaseException,
    ):
        exception = RuntimeError(
            str(exception)
        )

    exception_type_name = type(
        exception
    ).__name__

    exception_message = str(
        exception
    ).strip()

    if include_exception_type:
        if exception_message:
            error_description = (
                f"{exception_type_name}: "
                f"{exception_message}"
            )

        else:
            error_description = (
                exception_type_name
            )

    else:
        error_description = (
            exception_message
            or exception_type_name
        )

    message_lines = []

    if function is not None:
        resolved_function_name = (
            _resolve_callable_label(
                function,
                custom_name=function_name,
            )
        )

        message_lines.append(
            f"Function: "
            f"{resolved_function_name}"
        )

    message_lines.append(
        f"Error: "
        f"{error_description}"
    )

    if include_traceback:
        traceback_text = "".join(
            traceback.format_exception(
                type(exception),
                exception,
                exception.__traceback__,
            )
        ).rstrip()

        if traceback_text:
            message_lines.extend(
                (
                    "",
                    "Traceback:",
                    traceback_text,
                )
            )

    formatted_message = "\n".join(
        message_lines
    )

    return _truncate_decorator_text(
        formatted_message,
        maximum_length=maximum_length,
    )


# -----------------------------------------------------------------------------
# Exception callback handling
# -----------------------------------------------------------------------------

def _call_error_callback(
    callback: Optional[
        Callable[..., Any]
    ],
    *,
    exception: BaseException,
    function: Callable[..., Any],
    args: Tuple[Any, ...],
    kwargs: Mapping[str, Any],
    logger: Any = None,
) -> Any:
    """
    Execute an exception callback using a flexible calling convention.

    Supported callback signatures include:

    ``callback(exception)``

    ``callback(exception, function)``

    ``callback(exception, function, args, kwargs)``

    or keyword-based forms accepting any subset of:

    - ``exception``;
    - ``function``;
    - ``args``;
    - ``kwargs``;
    - ``logger``.

    Parameters
    ----------
    callback : callable or None
        Error callback.
    exception : BaseException
        Captured exception.
    function : callable
        Decorated function.
    args : tuple
        Original positional arguments.
    kwargs : mapping
        Original keyword arguments.
    logger : Any, optional
        Resolved logger.

    Returns
    -------
    Any
        Callback result.

    Raises
    ------
    TypeError
        If callback is not callable.
    """

    if callback is None:
        return None

    if not callable(
        callback
    ):
        raise TypeError(
            "Error callback must be callable."
        )

    callback_signature = _safe_signature(
        callback
    )

    callback_values = {
        "exception": exception,
        "error": exception,
        "exc": exception,
        "function": function,
        "func": function,
        "args": args,
        "kwargs": kwargs,
        "logger": logger,
    }

    if callback_signature is not None:
        accepted_keyword_arguments = {}

        accepts_variable_keywords = any(
            parameter.kind
            is inspect.Parameter.VAR_KEYWORD
            for parameter in (
                callback_signature
                .parameters
                .values()
            )
        )

        if accepts_variable_keywords:
            accepted_keyword_arguments = {
                "exception": exception,
                "function": function,
                "args": args,
                "kwargs": kwargs,
                "logger": logger,
            }

        else:
            for parameter_name in (
                callback_signature.parameters
            ):
                if (
                    parameter_name
                    in callback_values
                ):
                    accepted_keyword_arguments[
                        parameter_name
                    ] = callback_values[
                        parameter_name
                    ]

        try:
            return callback(
                **accepted_keyword_arguments
            )

        except TypeError:
            pass

    callback_attempts = (
        (
            exception,
            function,
            args,
            kwargs,
        ),
        (
            exception,
            function,
        ),
        (
            exception,
        ),
        (),
    )

    last_error = None

    for callback_arguments in callback_attempts:
        try:
            return callback(
                *callback_arguments
            )

        except TypeError as error:
            last_error = error
            continue

    if last_error is not None:
        raise last_error

    return None


# -----------------------------------------------------------------------------
# Default-value factories
# -----------------------------------------------------------------------------

def _resolve_default_value(
    default: Any = _DECORATOR_UNSET,
    *,
    default_factory: Any = _DECORATOR_UNSET,
) -> Any:
    """
    Resolve a static default value or call a default factory.

    Parameters
    ----------
    default : Any, optional
        Static fallback value.
    default_factory : callable, optional
        Function used to create the fallback value.

    Returns
    -------
    Any
        Resolved default value.

    Raises
    ------
    ValueError
        If both ``default`` and ``default_factory`` are supplied.
    TypeError
        If ``default_factory`` is not callable.
    """

    has_default = (
        default is not _DECORATOR_UNSET
    )

    has_factory = (
        default_factory
        is not _DECORATOR_UNSET
    )

    if has_default and has_factory:
        raise ValueError(
            "Use either default or "
            "default_factory, not both."
        )

    if has_factory:
        if not callable(
            default_factory
        ):
            raise TypeError(
                "default_factory must be callable."
            )

        return default_factory()

    if has_default:
        return default

    return None


# -----------------------------------------------------------------------------
# Exception-type validation
# -----------------------------------------------------------------------------

def _normalize_exception_types(
    exceptions: Any,
    *,
    parameter_name: str = "exceptions",
) -> Tuple[
    Type[BaseException],
    ...,
]:
    """
    Normalize one or more exception classes into a tuple.

    Parameters
    ----------
    exceptions : exception class or iterable
        Exception type or types.
    parameter_name : str, optional
        Parameter name used in validation messages.

    Returns
    -------
    tuple
        Validated exception classes.

    Raises
    ------
    TypeError
        If an item is not an exception class.
    ValueError
        If no exception types are supplied.
    """

    if inspect.isclass(
        exceptions
    ) and issubclass(
        exceptions,
        BaseException,
    ):
        exception_types = (
            exceptions,
        )

    else:
        try:
            exception_types = tuple(
                exceptions
            )

        except TypeError as error:
            raise TypeError(
                f"{parameter_name} must be "
                "an exception class or an "
                "iterable of exception classes."
            ) from error

    if not exception_types:
        raise ValueError(
            f"{parameter_name} cannot be empty."
        )

    for exception_type in exception_types:
        if not (
            inspect.isclass(
                exception_type
            )
            and issubclass(
                exception_type,
                BaseException,
            )
        ):
            raise TypeError(
                f"Every item in "
                f"{parameter_name} must be "
                "an exception class."
            )

    return exception_types


# -----------------------------------------------------------------------------
# Decorator configuration validation
# -----------------------------------------------------------------------------

def _validate_boolean_option(
    value: Any,
    *,
    parameter_name: str,
) -> bool:
    """
    Validate a strict boolean decorator option.

    Parameters
    ----------
    value : Any
        Value to validate.
    parameter_name : str
        Option name.

    Returns
    -------
    bool
        Validated value.

    Raises
    ------
    TypeError
        If the value is not boolean.
    """

    if not isinstance(
        value,
        bool,
    ):
        raise TypeError(
            f"{parameter_name} must be "
            "a boolean value."
        )

    return value


def _validate_optional_positive_integer(
    value: Any,
    *,
    parameter_name: str,
    allow_zero: bool = False,
) -> Optional[int]:
    """
    Validate an optional non-negative or positive integer.

    Parameters
    ----------
    value : Any
        Value to validate.
    parameter_name : str
        Option name.
    allow_zero : bool, optional
        Whether zero is accepted.

    Returns
    -------
    int or None
        Validated value.

    Raises
    ------
    TypeError
        If the value is not an integer.
    ValueError
        If the value is below the allowed minimum.
    """

    if value is None:
        return None

    if (
        isinstance(
            value,
            bool,
        )
        or not isinstance(
            value,
            int,
        )
    ):
        raise TypeError(
            f"{parameter_name} must be "
            "an integer or None."
        )

    minimum_value = (
        0
        if allow_zero
        else 1
    )

    if value < minimum_value:
        comparison_text = (
            "zero or greater"
            if allow_zero
            else "greater than zero"
        )

        raise ValueError(
            f"{parameter_name} must be "
            f"{comparison_text}."
        )

    return value


# -----------------------------------------------------------------------------
# Wrapper metadata helpers
# -----------------------------------------------------------------------------

def _copy_wrapper_metadata(
    wrapper: Callable[..., Any],
    wrapped: Callable[..., Any],
) -> Callable[..., Any]:
    """
    Copy metadata from a wrapped callable.

    This function complements ``functools.wraps`` and safely preserves the
    original signature when possible.

    Parameters
    ----------
    wrapper : callable
        Wrapper function.
    wrapped : callable
        Original function.

    Returns
    -------
    callable
        Updated wrapper.
    """

    updated_wrapper = functools.update_wrapper(
        wrapper,
        wrapped,
    )

    signature = _safe_signature(
        wrapped
    )

    if signature is not None:
        try:
            updated_wrapper.__signature__ = (
                signature
            )

        except (
            AttributeError,
            TypeError,
        ):
            pass

    try:
        updated_wrapper.__wrapped__ = (
            wrapped
        )

    except (
        AttributeError,
        TypeError,
    ):
        pass

    return updated_wrapper


def _set_decorator_metadata(
    wrapper: Callable[..., Any],
    *,
    decorator_name: str,
    configuration: Optional[
        Mapping[str, Any]
    ] = None,
) -> Callable[..., Any]:
    """
    Attach internal DockAnalyzer decorator metadata to a wrapper.

    Parameters
    ----------
    wrapper : callable
        Wrapper function.
    decorator_name : str
        Decorator identifier.
    configuration : mapping, optional
        Decorator configuration.

    Returns
    -------
    callable
        Wrapper with metadata.
    """

    try:
        wrapper.__dockanalyzer_decorator__ = (
            decorator_name
        )

        wrapper.__dockanalyzer_decorator_config__ = (
            dict(
                configuration
                or {}
            )
        )

    except (
        AttributeError,
        TypeError,
    ):
        pass

    return wrapper


def _get_decorator_chain(
    function: Callable[..., Any],
) -> List[str]:
    """
    Return DockAnalyzer decorators detected in a wrapper chain.

    Parameters
    ----------
    function : callable
        Function to inspect.

    Returns
    -------
    list of str
        Detected decorator names.
    """

    decorator_names: List[str] = []
    current_function = function
    visited_ids = set()

    while callable(
        current_function
    ):
        current_id = id(
            current_function
        )

        if current_id in visited_ids:
            break

        visited_ids.add(
            current_id
        )

        decorator_name = getattr(
            current_function,
            "__dockanalyzer_decorator__",
            None,
        )

        if decorator_name:
            decorator_names.append(
                str(
                    decorator_name
                )
            )

        wrapped_function = getattr(
            current_function,
            "__wrapped__",
            None,
        )

        if wrapped_function is None:
            break

        current_function = wrapped_function

    return decorator_names


# -----------------------------------------------------------------------------
# Internal module interface
# -----------------------------------------------------------------------------

_SECTION_11_1_INTERNAL_NAMES = [
    "_DECORATOR_UNSET",
    "_DECORATOR_MISSING",
    "_is_callable_object",
    "_is_decorated_function_candidate",
    "_is_coroutine_callable",
    "_resolve_function_name",
    "_resolve_callable_label",
    "_get_bound_instance",
    "_get_attribute_from_candidates",
    "_get_mapping_value_from_candidates",
    "_is_logger_like",
    "_resolve_decorator_logger",
    "_is_timer_like",
    "_resolve_decorator_timer",
    "_normalize_log_level",
    "_log_message",
    "_log_or_print",
    "_truncate_decorator_text",
    "_safe_length",
    "_safe_shape",
    "_safe_object_name",
    "_format_decorator_value",
    "_safe_signature",
    "_bind_call_arguments",
    "_is_sensitive_argument_name",
    "_format_call_arguments",
    "_format_call_signature",
    "_format_exception",
    "_call_error_callback",
    "_resolve_default_value",
    "_normalize_exception_types",
    "_validate_boolean_option",
    "_validate_optional_positive_integer",
    "_copy_wrapper_metadata",
    "_set_decorator_metadata",
    "_get_decorator_chain",
]


# Internal helpers are intentionally not appended to __all__.


# =============================================================================
# End of Section 11.1
# =============================================================================

# =============================================================================
# Section 11.2 — Timer Decorator
# =============================================================================


# -----------------------------------------------------------------------------
# Timer constants
# -----------------------------------------------------------------------------

_TIMER_ATTRIBUTE_NAMES = (
    "timer",
    "analysis_timer",
    "dock_timer",
    "_timer",
)

_TIMER_START_METHOD_NAMES = (
    "start",
    "start_step",
    "begin",
)

_TIMER_STOP_METHOD_NAMES = (
    "stop",
    "stop_step",
    "end",
)

_TIMER_FAIL_METHOD_NAMES = (
    "fail",
    "fail_step",
    "error",
)

_TIMER_RECORD_METHOD_NAMES = (
    "record",
    "add_record",
    "add",
)


# -----------------------------------------------------------------------------
# Timer result helpers
# -----------------------------------------------------------------------------

def _format_elapsed_time(
    elapsed_seconds: float,
    *,
    precision: int = 3,
) -> str:
    """
    Format an elapsed duration for terminal and log output.

    Parameters
    ----------
    elapsed_seconds : float
        Duration in seconds.
    precision : int, optional
        Decimal precision for durations below one minute.

    Returns
    -------
    str
        Human-readable duration.

    Examples
    --------
    >>> _format_elapsed_time(0.284)
    '0.284 s'

    >>> _format_elapsed_time(75.2)
    '1 min 15.200 s'
    """

    try:
        elapsed_seconds = float(
            elapsed_seconds
        )

    except (
        TypeError,
        ValueError,
    ) as error:
        raise TypeError(
            "Elapsed time must be numeric."
        ) from error

    precision = max(
        int(precision),
        0,
    )

    if elapsed_seconds < 0:
        elapsed_seconds = 0.0

    if elapsed_seconds < 1e-3:
        milliseconds = (
            elapsed_seconds
            * 1000.0
        )

        return (
            f"{milliseconds:.{precision}f} ms"
        )

    if elapsed_seconds < 60.0:
        return (
            f"{elapsed_seconds:.{precision}f} s"
        )

    total_minutes = int(
        elapsed_seconds // 60
    )

    remaining_seconds = (
        elapsed_seconds
        - total_minutes * 60
    )

    if total_minutes < 60:
        return (
            f"{total_minutes} min "
            f"{remaining_seconds:.{precision}f} s"
        )

    hours = total_minutes // 60
    minutes = total_minutes % 60

    return (
        f"{hours} h "
        f"{minutes} min "
        f"{remaining_seconds:.{precision}f} s"
    )


def _timer_status_from_exception(
    exception: Optional[
        BaseException
    ],
) -> str:
    """
    Return the timer status associated with an exception.

    Parameters
    ----------
    exception : BaseException or None
        Exception raised during the timed call.

    Returns
    -------
    str
        ``"success"`` or ``"error"``.
    """

    return (
        "error"
        if exception is not None
        else "success"
    )


# -----------------------------------------------------------------------------
# Timer method invocation
# -----------------------------------------------------------------------------

def _call_timer_method(
    timer_object: Any,
    method_names: Iterable[str],
    *,
    step: str,
    elapsed: Optional[float] = None,
    status: Optional[str] = None,
    exception: Optional[
        BaseException
    ] = None,
    metadata: Optional[
        Mapping[str, Any]
    ] = None,
) -> Tuple[bool, Any]:
    """
    Call the first compatible method found on a timer-like object.

    Different timer implementations may expose slightly different APIs.
    This helper attempts common calling conventions without coupling the
    decorator to one exact implementation.

    Parameters
    ----------
    timer_object : Any
        Timer-like object.
    method_names : iterable of str
        Candidate method names.
    step : str
        Timed step name.
    elapsed : float, optional
        Elapsed duration.
    status : str, optional
        Completion status.
    exception : BaseException, optional
        Exception raised by the timed function.
    metadata : mapping, optional
        Additional timer metadata.

    Returns
    -------
    tuple
        ``(called, result)``.
    """

    if timer_object is None:
        return False, None

    metadata_dictionary = dict(
        metadata
        or {}
    )

    for method_name in method_names:
        method = getattr(
            timer_object,
            method_name,
            None,
        )

        if not callable(method):
            continue

        keyword_values = {
            "step": step,
            "name": step,
            "label": step,
            "elapsed": elapsed,
            "elapsed_seconds": elapsed,
            "duration": elapsed,
            "seconds": elapsed,
            "status": status,
            "exception": exception,
            "error": exception,
            "metadata": metadata_dictionary,
        }

        signature = _safe_signature(
            method
        )

        if signature is not None:
            accepted_kwargs = {}
            accepts_var_kwargs = any(
                parameter.kind
                is inspect.Parameter.VAR_KEYWORD
                for parameter in (
                    signature.parameters.values()
                )
            )

            if accepts_var_kwargs:
                accepted_kwargs = {
                    key: value
                    for key, value
                    in keyword_values.items()
                    if value is not None
                }

            else:
                for parameter_name in (
                    signature.parameters
                ):
                    if (
                        parameter_name
                        in keyword_values
                        and keyword_values[
                            parameter_name
                        ] is not None
                    ):
                        accepted_kwargs[
                            parameter_name
                        ] = keyword_values[
                            parameter_name
                        ]

            try:
                return (
                    True,
                    method(
                        **accepted_kwargs
                    ),
                )

            except TypeError:
                pass

        positional_attempts = []

        if (
            elapsed is not None
            and exception is not None
        ):
            positional_attempts.extend(
                (
                    (
                        step,
                        elapsed,
                        exception,
                    ),
                    (
                        step,
                        elapsed,
                    ),
                    (
                        step,
                        exception,
                    ),
                )
            )

        elif elapsed is not None:
            positional_attempts.extend(
                (
                    (
                        step,
                        elapsed,
                    ),
                    (
                        step,
                    ),
                )
            )

        elif exception is not None:
            positional_attempts.extend(
                (
                    (
                        step,
                        exception,
                    ),
                    (
                        step,
                    ),
                )
            )

        else:
            positional_attempts.append(
                (
                    step,
                )
            )

        positional_attempts.append(
            ()
        )

        for positional_arguments in (
            positional_attempts
        ):
            try:
                return (
                    True,
                    method(
                        *positional_arguments
                    ),
                )

            except TypeError:
                continue

    return False, None


# -----------------------------------------------------------------------------
# AnalysisTimer integration
# -----------------------------------------------------------------------------

def _create_local_analysis_timer(
    *,
    logger: Any = None,
) -> Any:
    """
    Create an ``AnalysisTimer`` using compatible constructor signatures.

    Parameters
    ----------
    logger : Any, optional
        Logger associated with the timer.

    Returns
    -------
    Any
        New timer instance or ``None`` when construction fails.
    """

    constructor_attempts = (
        {
            "logger": logger,
        },
        {},
    )

    for constructor_kwargs in (
        constructor_attempts
    ):
        filtered_kwargs = {
            key: value
            for key, value
            in constructor_kwargs.items()
            if value is not None
        }

        try:
            return AnalysisTimer(
                **filtered_kwargs
            )

        except TypeError:
            continue

        except Exception:
            return None

    return None


def _start_timer_step(
    timer_object: Any,
    *,
    step: str,
    metadata: Optional[
        Mapping[str, Any]
    ] = None,
) -> Any:
    """
    Start a step on a timer-like object.

    Parameters
    ----------
    timer_object : Any
        Timer-like object.
    step : str
        Step name.
    metadata : mapping, optional
        Step metadata.

    Returns
    -------
    Any
        Token or result returned by the timer.
    """

    called, result = _call_timer_method(
        timer_object,
        _TIMER_START_METHOD_NAMES,
        step=step,
        metadata=metadata,
    )

    if called:
        return result

    return None


def _stop_timer_step(
    timer_object: Any,
    *,
    step: str,
    elapsed: float,
    status: str,
    exception: Optional[
        BaseException
    ] = None,
    metadata: Optional[
        Mapping[str, Any]
    ] = None,
) -> Any:
    """
    Complete a timer step.

    Parameters
    ----------
    timer_object : Any
        Timer-like object.
    step : str
        Step name.
    elapsed : float
        Elapsed duration.
    status : str
        Completion status.
    exception : BaseException, optional
        Exception raised during the step.
    metadata : mapping, optional
        Step metadata.

    Returns
    -------
    Any
        Result returned by the timer.
    """

    if timer_object is None:
        return None

    if exception is not None:
        called, result = _call_timer_method(
            timer_object,
            _TIMER_FAIL_METHOD_NAMES,
            step=step,
            elapsed=elapsed,
            status=status,
            exception=exception,
            metadata=metadata,
        )

        if called:
            return result

    called, result = _call_timer_method(
        timer_object,
        _TIMER_STOP_METHOD_NAMES,
        step=step,
        elapsed=elapsed,
        status=status,
        exception=exception,
        metadata=metadata,
    )

    if called:
        return result

    called, result = _call_timer_method(
        timer_object,
        _TIMER_RECORD_METHOD_NAMES,
        step=step,
        elapsed=elapsed,
        status=status,
        exception=exception,
        metadata=metadata,
    )

    if called:
        return result

    return None


# -----------------------------------------------------------------------------
# Timer metadata
# -----------------------------------------------------------------------------

def _build_timer_metadata(
    function: Callable[..., Any],
    args: Tuple[Any, ...],
    kwargs: Mapping[str, Any],
    *,
    include_arguments: bool = False,
    custom_metadata: Optional[
        Mapping[str, Any]
    ] = None,
) -> Dict[str, Any]:
    """
    Build metadata associated with a timed function call.

    Parameters
    ----------
    function : callable
        Decorated callable.
    args : tuple
        Runtime positional arguments.
    kwargs : mapping
        Runtime keyword arguments.
    include_arguments : bool, optional
        Whether a formatted argument representation should be included.
    custom_metadata : mapping, optional
        Additional metadata.

    Returns
    -------
    dict
        Timer metadata.
    """

    metadata: Dict[str, Any] = {
        "function": (
            _resolve_function_name(
                function,
                qualified=True,
                include_module=False,
            )
        ),
        "module": getattr(
            function,
            "__module__",
            None,
        ),
    }

    if include_arguments:
        metadata["arguments"] = (
            _format_call_arguments(
                function,
                args,
                kwargs,
                include_self=False,
                include_defaults=False,
                maximum_value_length=120,
                maximum_items=4,
                redact_sensitive=True,
                multiline=False,
            )
        )

    if custom_metadata:
        metadata.update(
            dict(
                custom_metadata
            )
        )

    return metadata


# -----------------------------------------------------------------------------
# Timer logging
# -----------------------------------------------------------------------------

def _log_timer_start(
    logger: Any,
    *,
    step: str,
    level: str,
    enabled: bool,
) -> None:
    """
    Log the beginning of a timed step.
    """

    if not enabled:
        return

    _log_message(
        logger,
        level,
        "Starting timed step: %s",
        step,
    )


def _log_timer_end(
    logger: Any,
    *,
    step: str,
    elapsed: float,
    level: str,
    precision: int,
    enabled: bool,
) -> None:
    """
    Log the successful completion of a timed step.
    """

    if not enabled:
        return

    formatted_elapsed = (
        _format_elapsed_time(
            elapsed,
            precision=precision,
        )
    )

    _log_message(
        logger,
        level,
        "Finished timed step: %s | Elapsed: %s",
        step,
        formatted_elapsed,
    )


def _log_timer_error(
    logger: Any,
    *,
    step: str,
    elapsed: float,
    exception: BaseException,
    level: str,
    precision: int,
    enabled: bool,
) -> None:
    """
    Log the failed completion of a timed step.
    """

    if not enabled:
        return

    formatted_elapsed = (
        _format_elapsed_time(
            elapsed,
            precision=precision,
        )
    )

    _log_message(
        logger,
        level,
        (
            "Timed step failed: %s | "
            "Elapsed: %s | %s: %s"
        ),
        step,
        formatted_elapsed,
        type(exception).__name__,
        str(exception),
    )


# -----------------------------------------------------------------------------
# Timer decorator factory
# -----------------------------------------------------------------------------

def _build_timer_decorator(
    *,
    step: Optional[str] = None,
    timer_object: Any = None,
    logger: Any = None,
    enabled: bool = True,
    create_timer: bool = True,
    log_start: bool = False,
    log_end: bool = True,
    log_errors: bool = True,
    log_level: str = "info",
    error_level: str = "error",
    precision: int = 3,
    include_arguments: bool = False,
    metadata: Optional[
        Mapping[str, Any]
    ] = None,
) -> Callable[
    [F],
    F,
]:
    """
    Build the actual timer decorator.

    This internal factory is separated from ``timer`` so the public function
    can support both direct and configured decorator syntax.
    """

    enabled = _validate_boolean_option(
        enabled,
        parameter_name="enabled",
    )

    create_timer = _validate_boolean_option(
        create_timer,
        parameter_name="create_timer",
    )

    log_start = _validate_boolean_option(
        log_start,
        parameter_name="log_start",
    )

    log_end = _validate_boolean_option(
        log_end,
        parameter_name="log_end",
    )

    log_errors = _validate_boolean_option(
        log_errors,
        parameter_name="log_errors",
    )

    include_arguments = (
        _validate_boolean_option(
            include_arguments,
            parameter_name=(
                "include_arguments"
            ),
        )
    )

    precision = (
        _validate_optional_positive_integer(
            precision,
            parameter_name="precision",
            allow_zero=True,
        )
    )

    if precision is None:
        precision = 3

    normalized_log_level = (
        _normalize_log_level(
            log_level,
            default="info",
        )
    )

    normalized_error_level = (
        _normalize_log_level(
            error_level,
            default="error",
        )
    )

    if step is not None:
        step = str(
            step
        ).strip()

        if not step:
            step = None

    static_metadata = dict(
        metadata
        or {}
    )

    def decorator(
        function: F,
    ) -> F:
        if not callable(
            function
        ):
            raise TypeError(
                "@timer can only decorate "
                "callable objects."
            )

        step_name = (
            step
            or _resolve_function_name(
                function,
                qualified=True,
                include_module=False,
            )
        )

        configuration = {
            "step": step_name,
            "enabled": enabled,
            "create_timer": create_timer,
            "log_start": log_start,
            "log_end": log_end,
            "log_errors": log_errors,
            "log_level": (
                normalized_log_level
            ),
            "error_level": (
                normalized_error_level
            ),
            "precision": precision,
            "include_arguments": (
                include_arguments
            ),
        }

        if _is_coroutine_callable(
            function
        ):

            async def async_wrapper(
                *args: Any,
                **kwargs: Any,
            ) -> Any:
                if not enabled:
                    return await function(
                        *args,
                        **kwargs,
                    )

                resolved_logger = (
                    _resolve_decorator_logger(
                        explicit_logger=logger,
                        args=args,
                        kwargs=kwargs,
                    )
                )

                resolved_timer = (
                    _resolve_decorator_timer(
                        explicit_timer=(
                            timer_object
                        ),
                        args=args,
                        kwargs=kwargs,
                    )
                )

                if (
                    resolved_timer is None
                    and create_timer
                ):
                    resolved_timer = (
                        _create_local_analysis_timer(
                            logger=resolved_logger,
                        )
                    )

                runtime_metadata = (
                    _build_timer_metadata(
                        function,
                        args,
                        kwargs,
                        include_arguments=(
                            include_arguments
                        ),
                        custom_metadata=(
                            static_metadata
                        ),
                    )
                )

                _start_timer_step(
                    resolved_timer,
                    step=step_name,
                    metadata=runtime_metadata,
                )

                _log_timer_start(
                    resolved_logger,
                    step=step_name,
                    level=(
                        normalized_log_level
                    ),
                    enabled=log_start,
                )

                start_time = (
                    time.perf_counter()
                )

                captured_exception = None

                try:
                    return await function(
                        *args,
                        **kwargs,
                    )

                except BaseException as error:
                    captured_exception = error

                    elapsed = (
                        time.perf_counter()
                        - start_time
                    )

                    _stop_timer_step(
                        resolved_timer,
                        step=step_name,
                        elapsed=elapsed,
                        status="error",
                        exception=error,
                        metadata=runtime_metadata,
                    )

                    _log_timer_error(
                        resolved_logger,
                        step=step_name,
                        elapsed=elapsed,
                        exception=error,
                        level=(
                            normalized_error_level
                        ),
                        precision=precision,
                        enabled=log_errors,
                    )

                    raise

                finally:
                    if captured_exception is None:
                        elapsed = (
                            time.perf_counter()
                            - start_time
                        )

                        _stop_timer_step(
                            resolved_timer,
                            step=step_name,
                            elapsed=elapsed,
                            status="success",
                            metadata=(
                                runtime_metadata
                            ),
                        )

                        _log_timer_end(
                            resolved_logger,
                            step=step_name,
                            elapsed=elapsed,
                            level=(
                                normalized_log_level
                            ),
                            precision=precision,
                            enabled=log_end,
                        )

            wrapped_function = (
                _copy_wrapper_metadata(
                    async_wrapper,
                    function,
                )
            )

        else:

            def sync_wrapper(
                *args: Any,
                **kwargs: Any,
            ) -> Any:
                if not enabled:
                    return function(
                        *args,
                        **kwargs,
                    )

                resolved_logger = (
                    _resolve_decorator_logger(
                        explicit_logger=logger,
                        args=args,
                        kwargs=kwargs,
                    )
                )

                resolved_timer = (
                    _resolve_decorator_timer(
                        explicit_timer=(
                            timer_object
                        ),
                        args=args,
                        kwargs=kwargs,
                    )
                )

                if (
                    resolved_timer is None
                    and create_timer
                ):
                    resolved_timer = (
                        _create_local_analysis_timer(
                            logger=resolved_logger,
                        )
                    )

                runtime_metadata = (
                    _build_timer_metadata(
                        function,
                        args,
                        kwargs,
                        include_arguments=(
                            include_arguments
                        ),
                        custom_metadata=(
                            static_metadata
                        ),
                    )
                )

                _start_timer_step(
                    resolved_timer,
                    step=step_name,
                    metadata=runtime_metadata,
                )

                _log_timer_start(
                    resolved_logger,
                    step=step_name,
                    level=(
                        normalized_log_level
                    ),
                    enabled=log_start,
                )

                start_time = (
                    time.perf_counter()
                )

                captured_exception = None

                try:
                    return function(
                        *args,
                        **kwargs,
                    )

                except BaseException as error:
                    captured_exception = error

                    elapsed = (
                        time.perf_counter()
                        - start_time
                    )

                    _stop_timer_step(
                        resolved_timer,
                        step=step_name,
                        elapsed=elapsed,
                        status="error",
                        exception=error,
                        metadata=runtime_metadata,
                    )

                    _log_timer_error(
                        resolved_logger,
                        step=step_name,
                        elapsed=elapsed,
                        exception=error,
                        level=(
                            normalized_error_level
                        ),
                        precision=precision,
                        enabled=log_errors,
                    )

                    raise

                finally:
                    if captured_exception is None:
                        elapsed = (
                            time.perf_counter()
                            - start_time
                        )

                        _stop_timer_step(
                            resolved_timer,
                            step=step_name,
                            elapsed=elapsed,
                            status="success",
                            metadata=(
                                runtime_metadata
                            ),
                        )

                        _log_timer_end(
                            resolved_logger,
                            step=step_name,
                            elapsed=elapsed,
                            level=(
                                normalized_log_level
                            ),
                            precision=precision,
                            enabled=log_end,
                        )

            wrapped_function = (
                _copy_wrapper_metadata(
                    sync_wrapper,
                    function,
                )
            )

        wrapped_function = (
            _set_decorator_metadata(
                wrapped_function,
                decorator_name="timer",
                configuration=configuration,
            )
        )

        return cast(
            F,
            wrapped_function,
        )

    return decorator


# -----------------------------------------------------------------------------
# Public timer decorator
# -----------------------------------------------------------------------------

def timer(
    function: Any = None,
    step: Optional[str] = None,
    *,
    timer_object: Any = None,
    logger: Any = None,
    enabled: bool = True,
    create_timer: bool = True,
    log_start: bool = False,
    log_end: bool = True,
    log_errors: bool = True,
    log_level: str = "info",
    error_level: str = "error",
    precision: int = 3,
    include_arguments: bool = False,
    metadata: Optional[
        Mapping[str, Any]
    ] = None,
) -> Any:
    """
    Measure the execution time of a callable.

    The decorator supports all of the following forms:

    ``@timer``

    ``@timer()``

    ``@timer("HBONDS")``

    ``@timer(step="HBONDS")``

    Parameters
    ----------
    function : callable or str, optional
        Function being decorated when ``@timer`` is used directly. A string
        passed positionally is interpreted as the step name.
    step : str, optional
        Custom timer step name.
    timer_object : Any, optional
        Explicit ``AnalysisTimer`` or timer-like object.
    logger : Any, optional
        Explicit logger.
    enabled : bool, optional
        Whether timing is active.
    create_timer : bool, optional
        Whether a local ``AnalysisTimer`` should be created when no timer is
        found.
    log_start : bool, optional
        Whether the start of the call should be logged.
    log_end : bool, optional
        Whether successful completion and duration should be logged.
    log_errors : bool, optional
        Whether failed calls and their duration should be logged.
    log_level : str, optional
        Logging level for start and successful completion messages.
    error_level : str, optional
        Logging level for failed calls.
    precision : int, optional
        Decimal precision used when formatting durations.
    include_arguments : bool, optional
        Whether a concise argument representation should be stored in timer
        metadata.
    metadata : mapping, optional
        Static metadata added to each timer record.

    Returns
    -------
    callable
        Decorated callable or configured decorator.

    Notes
    -----
    This decorator does not suppress exceptions. Failed calls are timed and
    registered, and the original exception is raised again.

    Examples
    --------
    >>> @timer
    ... def analyze_contacts():
    ...     ...

    >>> @timer("HBONDS")
    ... def detect_hbonds():
    ...     ...

    >>> @timer(
    ...     step="Hydrophobic contacts",
    ...     log_start=True,
    ...     log_end=True,
    ... )
    ... def detect_hydrophobic():
    ...     ...
    """

    positional_function = function

    if isinstance(
        positional_function,
        str,
    ):
        if step is not None:
            raise TypeError(
                "The timer step was provided "
                "both positionally and by keyword."
            )

        step = positional_function
        positional_function = None

    elif (
        positional_function is not None
        and not _is_decorated_function_candidate(
            positional_function
        )
    ):
        raise TypeError(
            "The positional argument to @timer "
            "must be a callable or a step name."
        )

    configured_decorator = (
        _build_timer_decorator(
            step=step,
            timer_object=timer_object,
            logger=logger,
            enabled=enabled,
            create_timer=create_timer,
            log_start=log_start,
            log_end=log_end,
            log_errors=log_errors,
            log_level=log_level,
            error_level=error_level,
            precision=precision,
            include_arguments=(
                include_arguments
            ),
            metadata=metadata,
        )
    )

    if positional_function is not None:
        return configured_decorator(
            positional_function
        )

    return configured_decorator


# -----------------------------------------------------------------------------
# Public module interface
# -----------------------------------------------------------------------------

_SECTION_11_2_PUBLIC_NAMES = [
    "timer",
]

for public_name in (
    _SECTION_11_2_PUBLIC_NAMES
):
    if public_name not in __all__:
        __all__.append(
            public_name
        )


# =============================================================================
# End of Section 11.2
# =============================================================================

# =============================================================================
# Section 11.3 — Log Call Decorator
# =============================================================================


# -----------------------------------------------------------------------------
# Log-call constants
# -----------------------------------------------------------------------------

_LOG_CALL_DEFAULT_MAXIMUM_VALUE_LENGTH = 180
_LOG_CALL_DEFAULT_MAXIMUM_RETURN_LENGTH = 240
_LOG_CALL_DEFAULT_MAXIMUM_ITEMS = 6

_LOG_CALL_ENTER_SYMBOL = "→"
_LOG_CALL_EXIT_SYMBOL = "←"
_LOG_CALL_ERROR_SYMBOL = "✗"

_LOG_CALL_SENSITIVE_ARGUMENT_NAMES = {
    "password",
    "passwd",
    "pwd",
    "secret",
    "token",
    "access_token",
    "refresh_token",
    "api_key",
    "apikey",
    "authorization",
    "credentials",
    "private_key",
}


# -----------------------------------------------------------------------------
# Nested-call context
# -----------------------------------------------------------------------------

_LOG_CALL_DEPTH: contextvars.ContextVar[int] = (
    contextvars.ContextVar(
        "dockanalyzer_log_call_depth",
        default=0,
    )
)


def _get_log_call_depth() -> int:
    """
    Return the current nested ``@log_call`` depth.

    Returns
    -------
    int
        Current call depth.
    """

    try:
        depth = int(
            _LOG_CALL_DEPTH.get()
        )

    except (
        TypeError,
        ValueError,
        LookupError,
    ):
        return 0

    return max(
        depth,
        0,
    )


def _build_log_call_indent(
    depth: int,
    *,
    indent_size: int = 2,
    indent_character: str = " ",
) -> str:
    """
    Build indentation for nested logged calls.

    Parameters
    ----------
    depth : int
        Nested-call depth.
    indent_size : int, optional
        Number of indentation characters per level.
    indent_character : str, optional
        Character used for indentation.

    Returns
    -------
    str
        Indentation prefix.
    """

    if isinstance(
        depth,
        bool,
    ) or not isinstance(
        depth,
        int,
    ):
        raise TypeError(
            "depth must be an integer."
        )

    if isinstance(
        indent_size,
        bool,
    ) or not isinstance(
        indent_size,
        int,
    ):
        raise TypeError(
            "indent_size must be an integer."
        )

    if depth < 0:
        depth = 0

    if indent_size < 0:
        raise ValueError(
            "indent_size cannot be negative."
        )

    indent_character = str(
        indent_character
    )

    return (
        indent_character
        * indent_size
        * depth
    )


# -----------------------------------------------------------------------------
# Argument-name normalization
# -----------------------------------------------------------------------------

def _normalize_argument_name_collection(
    values: Optional[Iterable[str]],
    *,
    parameter_name: str,
) -> Optional[Set[str]]:
    """
    Normalize an optional collection of argument names.

    Parameters
    ----------
    values : iterable of str or None
        Argument names.
    parameter_name : str
        Configuration parameter name.

    Returns
    -------
    set of str or None
        Normalized names.

    Raises
    ------
    TypeError
        If the supplied value is invalid.
    """

    if values is None:
        return None

    if isinstance(
        values,
        str,
    ):
        values = (
            values,
        )

    try:
        normalized_values = {
            str(value).strip()
            for value in values
            if str(value).strip()
        }

    except TypeError as error:
        raise TypeError(
            f"{parameter_name} must be "
            "an iterable of argument names."
        ) from error

    return normalized_values


def _should_include_logged_argument(
    argument_name: str,
    *,
    include_arguments: Optional[Set[str]],
    exclude_arguments: Optional[Set[str]],
    include_self: bool,
) -> bool:
    """
    Return whether an argument should appear in the log.

    Parameters
    ----------
    argument_name : str
        Runtime argument name.
    include_arguments : set of str or None
        Explicit argument allowlist.
    exclude_arguments : set of str or None
        Explicit argument blocklist.
    include_self : bool
        Whether ``self`` and ``cls`` are eligible.

    Returns
    -------
    bool
        Whether the argument should be included.
    """

    if (
        not include_self
        and argument_name
        in {
            "self",
            "cls",
        }
    ):
        return False

    if (
        include_arguments is not None
        and argument_name
        not in include_arguments
    ):
        return False

    if (
        exclude_arguments is not None
        and argument_name
        in exclude_arguments
    ):
        return False

    return True


# -----------------------------------------------------------------------------
# Logged argument formatting
# -----------------------------------------------------------------------------

def _format_logged_arguments(
    function: Callable[..., Any],
    args: Tuple[Any, ...],
    kwargs: Mapping[str, Any],
    *,
    include_self: bool = False,
    include_defaults: bool = False,
    include_argument_names: Optional[
        Set[str]
    ] = None,
    exclude_argument_names: Optional[
        Set[str]
    ] = None,
    maximum_value_length: int = (
        _LOG_CALL_DEFAULT_MAXIMUM_VALUE_LENGTH
    ),
    maximum_items: int = (
        _LOG_CALL_DEFAULT_MAXIMUM_ITEMS
    ),
    redact_sensitive: bool = True,
    sensitive_names: Optional[
        Iterable[str]
    ] = None,
    multiline: bool = False,
) -> str:
    """
    Format selected runtime arguments for ``@log_call``.

    Parameters
    ----------
    function : callable
        Decorated function.
    args : tuple
        Runtime positional arguments.
    kwargs : mapping
        Runtime keyword arguments.
    include_self : bool, optional
        Whether ``self`` and ``cls`` should be logged.
    include_defaults : bool, optional
        Whether default values should be included.
    include_argument_names : set of str, optional
        Argument allowlist.
    exclude_argument_names : set of str, optional
        Argument blocklist.
    maximum_value_length : int, optional
        Maximum representation length for each value.
    maximum_items : int, optional
        Maximum displayed items for containers.
    redact_sensitive : bool, optional
        Whether sensitive values should be hidden.
    sensitive_names : iterable of str, optional
        Additional sensitive argument names.
    multiline : bool, optional
        Whether arguments should be placed on separate lines.

    Returns
    -------
    str
        Formatted argument representation.
    """

    bound_arguments = _bind_call_arguments(
        function,
        args,
        kwargs,
        apply_defaults=include_defaults,
    )

    if bound_arguments is not None:
        argument_items = list(
            bound_arguments.arguments.items()
        )

    else:
        argument_items = [
            (
                f"arg_{index}",
                value,
            )
            for index, value in enumerate(
                args,
                start=1,
            )
        ]

        argument_items.extend(
            kwargs.items()
        )

    formatted_arguments: List[str] = []

    for (
        argument_name,
        argument_value,
    ) in argument_items:
        if not _should_include_logged_argument(
            argument_name,
            include_arguments=(
                include_argument_names
            ),
            exclude_arguments=(
                exclude_argument_names
            ),
            include_self=include_self,
        ):
            continue

        if (
            redact_sensitive
            and _is_sensitive_argument_name(
                argument_name,
                additional_names=(
                    sensitive_names
                ),
            )
        ):
            formatted_value = "<redacted>"

        else:
            formatted_value = (
                _format_decorator_value(
                    argument_value,
                    maximum_length=(
                        maximum_value_length
                    ),
                    maximum_items=maximum_items,
                    maximum_depth=2,
                )
            )

        formatted_arguments.append(
            f"{argument_name}="
            f"{formatted_value}"
        )

    if not formatted_arguments:
        return ""

    if multiline:
        return "\n".join(
            f"    {argument}"
            for argument
            in formatted_arguments
        )

    return ", ".join(
        formatted_arguments
    )


# -----------------------------------------------------------------------------
# Return-value formatting
# -----------------------------------------------------------------------------

def _format_logged_return_value(
    value: Any,
    *,
    maximum_length: int = (
        _LOG_CALL_DEFAULT_MAXIMUM_RETURN_LENGTH
    ),
    maximum_items: int = (
        _LOG_CALL_DEFAULT_MAXIMUM_ITEMS
    ),
    formatter: Optional[
        Callable[[Any], Any]
    ] = None,
) -> str:
    """
    Format a function return value for logging.

    Parameters
    ----------
    value : Any
        Function return value.
    maximum_length : int, optional
        Maximum formatted length.
    maximum_items : int, optional
        Maximum displayed container items.
    formatter : callable, optional
        Custom return-value formatter.

    Returns
    -------
    str
        Formatted return value.
    """

    if formatter is not None:
        if not callable(
            formatter
        ):
            raise TypeError(
                "return_formatter must be callable."
            )

        try:
            formatted_value = formatter(
                value
            )

        except Exception as error:
            formatted_value = (
                "<return formatter failed: "
                f"{type(error).__name__}: "
                f"{error}>"
            )

        return _truncate_decorator_text(
            formatted_value,
            maximum_length=maximum_length,
        )

    return _format_decorator_value(
        value,
        maximum_length=maximum_length,
        maximum_items=maximum_items,
        maximum_depth=2,
    )


# -----------------------------------------------------------------------------
# Log-call message builders
# -----------------------------------------------------------------------------

def _build_log_call_start_message(
    *,
    call_label: str,
    arguments: str = "",
    prefix: Optional[str] = None,
    symbol: str = _LOG_CALL_ENTER_SYMBOL,
    indentation: str = "",
    multiline_arguments: bool = False,
) -> str:
    """
    Build the message emitted before a function call.

    Parameters
    ----------
    call_label : str
        Function or custom call label.
    arguments : str, optional
        Formatted arguments.
    prefix : str, optional
        Static message prefix.
    symbol : str, optional
        Entry symbol.
    indentation : str, optional
        Nested-call indentation.
    multiline_arguments : bool, optional
        Whether arguments use multiline formatting.

    Returns
    -------
    str
        Start message.
    """

    message_prefix = ""

    if prefix:
        message_prefix = (
            f"{str(prefix).strip()} "
        )

    if arguments:
        if multiline_arguments:
            call_text = (
                f"{call_label}(\n"
                f"{arguments}\n"
                f"{indentation})"
            )

        else:
            call_text = (
                f"{call_label}("
                f"{arguments}"
                f")"
            )

    else:
        call_text = (
            f"{call_label}()"
        )

    return (
        f"{indentation}"
        f"{message_prefix}"
        f"{symbol} "
        f"{call_text}"
    )


def _build_log_call_end_message(
    *,
    call_label: str,
    elapsed: Optional[float] = None,
    return_value: Any = _DECORATOR_UNSET,
    prefix: Optional[str] = None,
    symbol: str = _LOG_CALL_EXIT_SYMBOL,
    indentation: str = "",
    precision: int = 3,
    maximum_return_length: int = (
        _LOG_CALL_DEFAULT_MAXIMUM_RETURN_LENGTH
    ),
    maximum_items: int = (
        _LOG_CALL_DEFAULT_MAXIMUM_ITEMS
    ),
    return_formatter: Optional[
        Callable[[Any], Any]
    ] = None,
) -> str:
    """
    Build the successful-completion message.

    Parameters
    ----------
    call_label : str
        Function or custom call label.
    elapsed : float, optional
        Execution duration.
    return_value : Any, optional
        Function return value. The internal unset sentinel disables it.
    prefix : str, optional
        Static message prefix.
    symbol : str, optional
        Completion symbol.
    indentation : str, optional
        Nested-call indentation.
    precision : int, optional
        Duration precision.
    maximum_return_length : int, optional
        Maximum return representation length.
    maximum_items : int, optional
        Maximum displayed return container items.
    return_formatter : callable, optional
        Custom return formatter.

    Returns
    -------
    str
        Completion message.
    """

    message_prefix = ""

    if prefix:
        message_prefix = (
            f"{str(prefix).strip()} "
        )

    message = (
        f"{indentation}"
        f"{message_prefix}"
        f"{symbol} "
        f"{call_label} completed"
    )

    message_parts = []

    if elapsed is not None:
        message_parts.append(
            "elapsed="
            + _format_elapsed_time(
                elapsed,
                precision=precision,
            )
        )

    if (
        return_value
        is not _DECORATOR_UNSET
    ):
        formatted_return = (
            _format_logged_return_value(
                return_value,
                maximum_length=(
                    maximum_return_length
                ),
                maximum_items=maximum_items,
                formatter=return_formatter,
            )
        )

        message_parts.append(
            f"return={formatted_return}"
        )

    if message_parts:
        message += (
            " | "
            + " | ".join(
                message_parts
            )
        )

    return message


def _build_log_call_error_message(
    *,
    call_label: str,
    exception: BaseException,
    elapsed: Optional[float] = None,
    prefix: Optional[str] = None,
    symbol: str = _LOG_CALL_ERROR_SYMBOL,
    indentation: str = "",
    precision: int = 3,
    include_traceback: bool = False,
    maximum_exception_length: Optional[
        int
    ] = None,
) -> str:
    """
    Build the message emitted when the function raises an exception.

    Parameters
    ----------
    call_label : str
        Function or custom call label.
    exception : BaseException
        Raised exception.
    elapsed : float, optional
        Execution duration before failure.
    prefix : str, optional
        Static message prefix.
    symbol : str, optional
        Failure symbol.
    indentation : str, optional
        Nested-call indentation.
    precision : int, optional
        Duration precision.
    include_traceback : bool, optional
        Whether the traceback should be included.
    maximum_exception_length : int or None, optional
        Maximum exception report length.

    Returns
    -------
    str
        Failure message.
    """

    message_prefix = ""

    if prefix:
        message_prefix = (
            f"{str(prefix).strip()} "
        )

    exception_report = (
        _format_exception(
            exception,
            include_traceback=(
                include_traceback
            ),
            include_exception_type=True,
            maximum_length=(
                maximum_exception_length
            ),
        )
    )

    message = (
        f"{indentation}"
        f"{message_prefix}"
        f"{symbol} "
        f"{call_label} failed"
    )

    if elapsed is not None:
        message += (
            " | elapsed="
            + _format_elapsed_time(
                elapsed,
                precision=precision,
            )
        )

    message += (
        f" | {exception_report}"
    )

    return message


# -----------------------------------------------------------------------------
# Conditional logging
# -----------------------------------------------------------------------------

def _evaluate_log_call_condition(
    condition: Optional[
        Callable[..., Any]
    ],
    *,
    function: Callable[..., Any],
    args: Tuple[Any, ...],
    kwargs: Mapping[str, Any],
) -> bool:
    """
    Evaluate an optional predicate controlling call logging.

    Parameters
    ----------
    condition : callable or None
        Logging predicate.
    function : callable
        Decorated function.
    args : tuple
        Runtime positional arguments.
    kwargs : mapping
        Runtime keyword arguments.

    Returns
    -------
    bool
        Whether logging should be active for the call.

    Notes
    -----
    The predicate may accept any of these signatures:

    ``condition()``

    ``condition(*args, **kwargs)``

    ``condition(function, args, kwargs)``
    """

    if condition is None:
        return True

    if not callable(
        condition
    ):
        raise TypeError(
            "condition must be callable."
        )

    signature = _safe_signature(
        condition
    )

    if signature is not None:
        parameter_names = set(
            signature.parameters
        )

        if parameter_names.intersection(
            {
                "function",
                "func",
                "args",
                "kwargs",
            }
        ):
            predicate_kwargs = {}

            if "function" in parameter_names:
                predicate_kwargs[
                    "function"
                ] = function

            if "func" in parameter_names:
                predicate_kwargs[
                    "func"
                ] = function

            if "args" in parameter_names:
                predicate_kwargs[
                    "args"
                ] = args

            if "kwargs" in parameter_names:
                predicate_kwargs[
                    "kwargs"
                ] = kwargs

            try:
                return bool(
                    condition(
                        **predicate_kwargs
                    )
                )

            except TypeError:
                pass

    try:
        return bool(
            condition(
                *args,
                **kwargs,
            )
        )

    except TypeError:
        return bool(
            condition()
        )


# -----------------------------------------------------------------------------
# Message emission
# -----------------------------------------------------------------------------

def _emit_log_call_message(
    message: str,
    *,
    logger: Any,
    level: str,
    fallback_to_print: bool,
    exc_info: Any = None,
) -> None:
    """
    Emit a decorator message without interrupting the decorated function.

    Parameters
    ----------
    message : str
        Message to emit.
    logger : Any
        Logger-like object.
    level : str
        Logging level.
    fallback_to_print : bool
        Whether to print when no logger can be used.
    exc_info : Any, optional
        Exception information.

    Notes
    -----
    Logging failures are intentionally suppressed. A failure in the logging
    infrastructure must not cause a scientific analysis to fail.
    """

    try:
        _log_or_print(
            message,
            logger=logger,
            level=level,
            fallback_to_print=(
                fallback_to_print
            ),
            exc_info=exc_info,
        )

    except Exception:
        if fallback_to_print:
            try:
                print(
                    message
                )

            except Exception:
                pass


# -----------------------------------------------------------------------------
# Log-call decorator factory
# -----------------------------------------------------------------------------

def _build_log_call_decorator(
    *,
    name: Optional[str] = None,
    logger: Any = None,
    enabled: bool = True,
    level: str = "debug",
    error_level: str = "error",
    log_arguments: bool = True,
    log_return: bool = False,
    log_duration: bool = True,
    log_exceptions: bool = True,
    include_traceback: bool = False,
    include_self: bool = False,
    include_defaults: bool = False,
    include_argument_names: Optional[
        Iterable[str]
    ] = None,
    exclude_argument_names: Optional[
        Iterable[str]
    ] = None,
    redact_sensitive: bool = True,
    sensitive_names: Optional[
        Iterable[str]
    ] = None,
    maximum_value_length: int = (
        _LOG_CALL_DEFAULT_MAXIMUM_VALUE_LENGTH
    ),
    maximum_return_length: int = (
        _LOG_CALL_DEFAULT_MAXIMUM_RETURN_LENGTH
    ),
    maximum_exception_length: Optional[
        int
    ] = None,
    maximum_items: int = (
        _LOG_CALL_DEFAULT_MAXIMUM_ITEMS
    ),
    multiline_arguments: bool = False,
    nested_indentation: bool = True,
    indent_size: int = 2,
    prefix: Optional[str] = None,
    start_symbol: str = (
        _LOG_CALL_ENTER_SYMBOL
    ),
    end_symbol: str = (
        _LOG_CALL_EXIT_SYMBOL
    ),
    error_symbol: str = (
        _LOG_CALL_ERROR_SYMBOL
    ),
    precision: int = 3,
    fallback_to_print: bool = False,
    condition: Optional[
        Callable[..., Any]
    ] = None,
    argument_formatter: Optional[
        Callable[..., Any]
    ] = None,
    return_formatter: Optional[
        Callable[[Any], Any]
    ] = None,
) -> Callable[[F], F]:
    """
    Build the configured ``@log_call`` decorator.

    Parameters are validated once, when the decorator is created, rather than
    during every function call.
    """

    enabled = _validate_boolean_option(
        enabled,
        parameter_name="enabled",
    )

    log_arguments = (
        _validate_boolean_option(
            log_arguments,
            parameter_name="log_arguments",
        )
    )

    log_return = _validate_boolean_option(
        log_return,
        parameter_name="log_return",
    )

    log_duration = (
        _validate_boolean_option(
            log_duration,
            parameter_name="log_duration",
        )
    )

    log_exceptions = (
        _validate_boolean_option(
            log_exceptions,
            parameter_name="log_exceptions",
        )
    )

    include_traceback = (
        _validate_boolean_option(
            include_traceback,
            parameter_name=(
                "include_traceback"
            ),
        )
    )

    include_self = (
        _validate_boolean_option(
            include_self,
            parameter_name="include_self",
        )
    )

    include_defaults = (
        _validate_boolean_option(
            include_defaults,
            parameter_name=(
                "include_defaults"
            ),
        )
    )

    redact_sensitive = (
        _validate_boolean_option(
            redact_sensitive,
            parameter_name=(
                "redact_sensitive"
            ),
        )
    )

    multiline_arguments = (
        _validate_boolean_option(
            multiline_arguments,
            parameter_name=(
                "multiline_arguments"
            ),
        )
    )

    nested_indentation = (
        _validate_boolean_option(
            nested_indentation,
            parameter_name=(
                "nested_indentation"
            ),
        )
    )

    fallback_to_print = (
        _validate_boolean_option(
            fallback_to_print,
            parameter_name=(
                "fallback_to_print"
            ),
        )
    )

    normalized_level = (
        _normalize_log_level(
            level,
            default="debug",
        )
    )

    normalized_error_level = (
        _normalize_log_level(
            error_level,
            default="error",
        )
    )

    precision = (
        _validate_optional_positive_integer(
            precision,
            parameter_name="precision",
            allow_zero=True,
        )
    )

    if precision is None:
        precision = 3

    maximum_value_length = (
        _validate_optional_positive_integer(
            maximum_value_length,
            parameter_name=(
                "maximum_value_length"
            ),
            allow_zero=False,
        )
    )

    maximum_return_length = (
        _validate_optional_positive_integer(
            maximum_return_length,
            parameter_name=(
                "maximum_return_length"
            ),
            allow_zero=False,
        )
    )

    maximum_exception_length = (
        _validate_optional_positive_integer(
            maximum_exception_length,
            parameter_name=(
                "maximum_exception_length"
            ),
            allow_zero=False,
        )
    )

    maximum_items = (
        _validate_optional_positive_integer(
            maximum_items,
            parameter_name="maximum_items",
            allow_zero=False,
        )
    )

    indent_size = (
        _validate_optional_positive_integer(
            indent_size,
            parameter_name="indent_size",
            allow_zero=True,
        )
    )

    if maximum_value_length is None:
        maximum_value_length = (
            _LOG_CALL_DEFAULT_MAXIMUM_VALUE_LENGTH
        )

    if maximum_return_length is None:
        maximum_return_length = (
            _LOG_CALL_DEFAULT_MAXIMUM_RETURN_LENGTH
        )

    if maximum_items is None:
        maximum_items = (
            _LOG_CALL_DEFAULT_MAXIMUM_ITEMS
        )

    if indent_size is None:
        indent_size = 2

    normalized_include_names = (
        _normalize_argument_name_collection(
            include_argument_names,
            parameter_name=(
                "include_argument_names"
            ),
        )
    )

    normalized_exclude_names = (
        _normalize_argument_name_collection(
            exclude_argument_names,
            parameter_name=(
                "exclude_argument_names"
            ),
        )
    )

    if (
        normalized_include_names
        and normalized_exclude_names
    ):
        overlap = (
            normalized_include_names
            & normalized_exclude_names
        )

        if overlap:
            overlap_text = ", ".join(
                sorted(
                    overlap
                )
            )

            raise ValueError(
                "The same argument cannot appear "
                "in include_argument_names and "
                "exclude_argument_names: "
                f"{overlap_text}."
            )

    normalized_sensitive_names = set(
        _LOG_CALL_SENSITIVE_ARGUMENT_NAMES
    )

    if sensitive_names is not None:
        if isinstance(
            sensitive_names,
            str,
        ):
            sensitive_names = (
                sensitive_names,
            )

        normalized_sensitive_names.update(
            str(item).strip().lower()
            for item in sensitive_names
            if str(item).strip()
        )

    if name is not None:
        name = str(
            name
        ).strip()

        if not name:
            name = None

    if prefix is not None:
        prefix = str(
            prefix
        ).strip()

        if not prefix:
            prefix = None

    start_symbol = str(
        start_symbol
    )

    end_symbol = str(
        end_symbol
    )

    error_symbol = str(
        error_symbol
    )

    if (
        argument_formatter is not None
        and not callable(
            argument_formatter
        )
    ):
        raise TypeError(
            "argument_formatter must be "
            "callable or None."
        )

    if (
        return_formatter is not None
        and not callable(
            return_formatter
        )
    ):
        raise TypeError(
            "return_formatter must be "
            "callable or None."
        )

    if (
        condition is not None
        and not callable(
            condition
        )
    ):
        raise TypeError(
            "condition must be callable or None."
        )

    def decorator(
        function: F,
    ) -> F:
        if not callable(
            function
        ):
            raise TypeError(
                "@log_call can only decorate "
                "callable objects."
            )

        call_label = (
            _resolve_callable_label(
                function,
                custom_name=name,
                include_module=False,
            )
        )

        configuration = {
            "name": call_label,
            "enabled": enabled,
            "level": normalized_level,
            "error_level": (
                normalized_error_level
            ),
            "log_arguments": log_arguments,
            "log_return": log_return,
            "log_duration": log_duration,
            "log_exceptions": log_exceptions,
            "include_traceback": (
                include_traceback
            ),
            "include_self": include_self,
            "include_defaults": (
                include_defaults
            ),
            "redact_sensitive": (
                redact_sensitive
            ),
            "multiline_arguments": (
                multiline_arguments
            ),
            "nested_indentation": (
                nested_indentation
            ),
            "fallback_to_print": (
                fallback_to_print
            ),
        }

        def prepare_call_logging(
            args: Tuple[Any, ...],
            kwargs: Mapping[str, Any],
        ) -> Tuple[
            bool,
            Any,
            str,
            int,
        ]:
            """
            Resolve runtime logging configuration.

            Returns
            -------
            tuple
                ``(should_log, logger, indentation, depth)``.
            """

            if not enabled:
                return (
                    False,
                    None,
                    "",
                    0,
                )

            should_log = (
                _evaluate_log_call_condition(
                    condition,
                    function=function,
                    args=args,
                    kwargs=kwargs,
                )
            )

            if not should_log:
                return (
                    False,
                    None,
                    "",
                    0,
                )

            resolved_logger = (
                _resolve_decorator_logger(
                    explicit_logger=logger,
                    args=args,
                    kwargs=kwargs,
                )
            )

            current_depth = (
                _get_log_call_depth()
            )

            indentation = ""

            if nested_indentation:
                indentation = (
                    _build_log_call_indent(
                        current_depth,
                        indent_size=indent_size,
                    )
                )

            return (
                True,
                resolved_logger,
                indentation,
                current_depth,
            )

        def prepare_arguments(
            args: Tuple[Any, ...],
            kwargs: Mapping[str, Any],
        ) -> str:
            """
            Format runtime arguments using the configured formatter.
            """

            if not log_arguments:
                return ""

            if argument_formatter is not None:
                try:
                    formatted = (
                        argument_formatter(
                            function,
                            args,
                            kwargs,
                        )
                    )

                except TypeError:
                    try:
                        formatted = (
                            argument_formatter(
                                args,
                                kwargs,
                            )
                        )

                    except TypeError:
                        formatted = (
                            argument_formatter(
                                *args,
                                **dict(kwargs),
                            )
                        )

                except Exception as error:
                    return (
                        "<argument formatter "
                        f"failed: "
                        f"{type(error).__name__}: "
                        f"{error}>"
                    )

                return _truncate_decorator_text(
                    formatted,
                    maximum_length=(
                        maximum_value_length
                        * max(
                            maximum_items,
                            1,
                        )
                    ),
                )

            return _format_logged_arguments(
                function,
                args,
                kwargs,
                include_self=include_self,
                include_defaults=(
                    include_defaults
                ),
                include_argument_names=(
                    normalized_include_names
                ),
                exclude_argument_names=(
                    normalized_exclude_names
                ),
                maximum_value_length=(
                    maximum_value_length
                ),
                maximum_items=maximum_items,
                redact_sensitive=(
                    redact_sensitive
                ),
                sensitive_names=(
                    normalized_sensitive_names
                ),
                multiline=(
                    multiline_arguments
                ),
            )

        if _is_coroutine_callable(
            function
        ):

            async def async_wrapper(
                *args: Any,
                **kwargs: Any,
            ) -> Any:
                (
                    should_log,
                    resolved_logger,
                    indentation,
                    current_depth,
                ) = prepare_call_logging(
                    args,
                    kwargs,
                )

                if not should_log:
                    return await function(
                        *args,
                        **kwargs,
                    )

                formatted_arguments = (
                    prepare_arguments(
                        args,
                        kwargs,
                    )
                )

                start_message = (
                    _build_log_call_start_message(
                        call_label=call_label,
                        arguments=(
                            formatted_arguments
                        ),
                        prefix=prefix,
                        symbol=start_symbol,
                        indentation=indentation,
                        multiline_arguments=(
                            multiline_arguments
                        ),
                    )
                )

                _emit_log_call_message(
                    start_message,
                    logger=resolved_logger,
                    level=normalized_level,
                    fallback_to_print=(
                        fallback_to_print
                    ),
                )

                depth_token = (
                    _LOG_CALL_DEPTH.set(
                        current_depth + 1
                    )
                )

                start_time = (
                    time.perf_counter()
                )

                try:
                    result = await function(
                        *args,
                        **kwargs,
                    )

                except Exception as error:
                    elapsed = (
                        time.perf_counter()
                        - start_time
                    )

                    if log_exceptions:
                        error_message = (
                            _build_log_call_error_message(
                                call_label=call_label,
                                exception=error,
                                elapsed=(
                                    elapsed
                                    if log_duration
                                    else None
                                ),
                                prefix=prefix,
                                symbol=error_symbol,
                                indentation=indentation,
                                precision=precision,
                                include_traceback=(
                                    include_traceback
                                ),
                                maximum_exception_length=(
                                    maximum_exception_length
                                ),
                            )
                        )

                        _emit_log_call_message(
                            error_message,
                            logger=(
                                resolved_logger
                            ),
                            level=(
                                normalized_error_level
                            ),
                            fallback_to_print=(
                                fallback_to_print
                            ),
                            exc_info=(
                                True
                                if include_traceback
                                else None
                            ),
                        )

                    raise

                else:
                    elapsed = (
                        time.perf_counter()
                        - start_time
                    )

                    end_message = (
                        _build_log_call_end_message(
                            call_label=call_label,
                            elapsed=(
                                elapsed
                                if log_duration
                                else None
                            ),
                            return_value=(
                                result
                                if log_return
                                else _DECORATOR_UNSET
                            ),
                            prefix=prefix,
                            symbol=end_symbol,
                            indentation=indentation,
                            precision=precision,
                            maximum_return_length=(
                                maximum_return_length
                            ),
                            maximum_items=(
                                maximum_items
                            ),
                            return_formatter=(
                                return_formatter
                            ),
                        )
                    )

                    _emit_log_call_message(
                        end_message,
                        logger=resolved_logger,
                        level=normalized_level,
                        fallback_to_print=(
                            fallback_to_print
                        ),
                    )

                    return result

                finally:
                    try:
                        _LOG_CALL_DEPTH.reset(
                            depth_token
                        )

                    except (
                        LookupError,
                        ValueError,
                    ):
                        _LOG_CALL_DEPTH.set(
                            current_depth
                        )

            wrapped_function = (
                _copy_wrapper_metadata(
                    async_wrapper,
                    function,
                )
            )

        else:

            def sync_wrapper(
                *args: Any,
                **kwargs: Any,
            ) -> Any:
                (
                    should_log,
                    resolved_logger,
                    indentation,
                    current_depth,
                ) = prepare_call_logging(
                    args,
                    kwargs,
                )

                if not should_log:
                    return function(
                        *args,
                        **kwargs,
                    )

                formatted_arguments = (
                    prepare_arguments(
                        args,
                        kwargs,
                    )
                )

                start_message = (
                    _build_log_call_start_message(
                        call_label=call_label,
                        arguments=(
                            formatted_arguments
                        ),
                        prefix=prefix,
                        symbol=start_symbol,
                        indentation=indentation,
                        multiline_arguments=(
                            multiline_arguments
                        ),
                    )
                )

                _emit_log_call_message(
                    start_message,
                    logger=resolved_logger,
                    level=normalized_level,
                    fallback_to_print=(
                        fallback_to_print
                    ),
                )

                depth_token = (
                    _LOG_CALL_DEPTH.set(
                        current_depth + 1
                    )
                )

                start_time = (
                    time.perf_counter()
                )

                try:
                    result = function(
                        *args,
                        **kwargs,
                    )

                except Exception as error:
                    elapsed = (
                        time.perf_counter()
                        - start_time
                    )

                    if log_exceptions:
                        error_message = (
                            _build_log_call_error_message(
                                call_label=call_label,
                                exception=error,
                                elapsed=(
                                    elapsed
                                    if log_duration
                                    else None
                                ),
                                prefix=prefix,
                                symbol=error_symbol,
                                indentation=indentation,
                                precision=precision,
                                include_traceback=(
                                    include_traceback
                                ),
                                maximum_exception_length=(
                                    maximum_exception_length
                                ),
                            )
                        )

                        _emit_log_call_message(
                            error_message,
                            logger=(
                                resolved_logger
                            ),
                            level=(
                                normalized_error_level
                            ),
                            fallback_to_print=(
                                fallback_to_print
                            ),
                            exc_info=(
                                True
                                if include_traceback
                                else None
                            ),
                        )

                    raise

                else:
                    elapsed = (
                        time.perf_counter()
                        - start_time
                    )

                    end_message = (
                        _build_log_call_end_message(
                            call_label=call_label,
                            elapsed=(
                                elapsed
                                if log_duration
                                else None
                            ),
                            return_value=(
                                result
                                if log_return
                                else _DECORATOR_UNSET
                            ),
                            prefix=prefix,
                            symbol=end_symbol,
                            indentation=indentation,
                            precision=precision,
                            maximum_return_length=(
                                maximum_return_length
                            ),
                            maximum_items=(
                                maximum_items
                            ),
                            return_formatter=(
                                return_formatter
                            ),
                        )
                    )

                    _emit_log_call_message(
                        end_message,
                        logger=resolved_logger,
                        level=normalized_level,
                        fallback_to_print=(
                            fallback_to_print
                        ),
                    )

                    return result

                finally:
                    try:
                        _LOG_CALL_DEPTH.reset(
                            depth_token
                        )

                    except (
                        LookupError,
                        ValueError,
                    ):
                        _LOG_CALL_DEPTH.set(
                            current_depth
                        )

            wrapped_function = (
                _copy_wrapper_metadata(
                    sync_wrapper,
                    function,
                )
            )

        wrapped_function = (
            _set_decorator_metadata(
                wrapped_function,
                decorator_name="log_call",
                configuration=configuration,
            )
        )

        return cast(
            F,
            wrapped_function,
        )

    return decorator


# -----------------------------------------------------------------------------
# Public log-call decorator
# -----------------------------------------------------------------------------

def log_call(
    function: Any = None,
    name: Optional[str] = None,
    *,
    logger: Any = None,
    enabled: bool = True,
    level: str = "debug",
    error_level: str = "error",
    log_arguments: bool = True,
    log_return: bool = False,
    log_duration: bool = True,
    log_exceptions: bool = True,
    include_traceback: bool = False,
    include_self: bool = False,
    include_defaults: bool = False,
    include_argument_names: Optional[
        Iterable[str]
    ] = None,
    exclude_argument_names: Optional[
        Iterable[str]
    ] = None,
    redact_sensitive: bool = True,
    sensitive_names: Optional[
        Iterable[str]
    ] = None,
    maximum_value_length: int = (
        _LOG_CALL_DEFAULT_MAXIMUM_VALUE_LENGTH
    ),
    maximum_return_length: int = (
        _LOG_CALL_DEFAULT_MAXIMUM_RETURN_LENGTH
    ),
    maximum_exception_length: Optional[
        int
    ] = None,
    maximum_items: int = (
        _LOG_CALL_DEFAULT_MAXIMUM_ITEMS
    ),
    multiline_arguments: bool = False,
    nested_indentation: bool = True,
    indent_size: int = 2,
    prefix: Optional[str] = None,
    start_symbol: str = (
        _LOG_CALL_ENTER_SYMBOL
    ),
    end_symbol: str = (
        _LOG_CALL_EXIT_SYMBOL
    ),
    error_symbol: str = (
        _LOG_CALL_ERROR_SYMBOL
    ),
    precision: int = 3,
    fallback_to_print: bool = False,
    condition: Optional[
        Callable[..., Any]
    ] = None,
    argument_formatter: Optional[
        Callable[..., Any]
    ] = None,
    return_formatter: Optional[
        Callable[[Any], Any]
    ] = None,
) -> Any:
    """
    Log the execution of a callable.

    The decorator records function entry, successful completion, optional
    arguments, optional return values, duration and raised exceptions.

    Supported forms
    ---------------
    ``@log_call``

    ``@log_call()``

    ``@log_call("HBOND analysis")``

    ``@log_call(name="HBOND analysis")``

    Parameters
    ----------
    function : callable or str, optional
        Function being decorated when used directly. A positional string is
        interpreted as a custom call name.
    name : str, optional
        Custom function label used in messages.
    logger : Any, optional
        Explicit logger-like object. When omitted, the decorator searches the
        function arguments, bound instance and ChimeraX session.
    enabled : bool, optional
        Whether logging is active.
    level : str, optional
        Log level for function entry and successful completion.
    error_level : str, optional
        Log level for failed calls.
    log_arguments : bool, optional
        Whether function arguments should be logged.
    log_return : bool, optional
        Whether the return value should be logged.
    log_duration : bool, optional
        Whether execution duration should be included.
    log_exceptions : bool, optional
        Whether raised exceptions should be logged.
    include_traceback : bool, optional
        Whether exception tracebacks should be included.
    include_self : bool, optional
        Whether ``self`` and ``cls`` should be logged.
    include_defaults : bool, optional
        Whether default parameter values should be included.
    include_argument_names : iterable of str, optional
        Allowlist of argument names.
    exclude_argument_names : iterable of str, optional
        Blocklist of argument names.
    redact_sensitive : bool, optional
        Whether sensitive argument values should be hidden.
    sensitive_names : iterable of str, optional
        Additional argument names treated as sensitive.
    maximum_value_length : int, optional
        Maximum representation length for each argument.
    maximum_return_length : int, optional
        Maximum return-value representation length.
    maximum_exception_length : int or None, optional
        Maximum exception report length.
    maximum_items : int, optional
        Maximum displayed items for containers.
    multiline_arguments : bool, optional
        Whether each argument should appear on a separate line.
    nested_indentation : bool, optional
        Whether nested decorated calls should be indented.
    indent_size : int, optional
        Number of spaces per nesting level.
    prefix : str, optional
        Static prefix inserted before messages.
    start_symbol : str, optional
        Symbol used for function entry.
    end_symbol : str, optional
        Symbol used for successful completion.
    error_symbol : str, optional
        Symbol used for failed calls.
    precision : int, optional
        Decimal precision for durations.
    fallback_to_print : bool, optional
        Whether messages should be printed when no logger is available.
    condition : callable, optional
        Predicate deciding whether a specific call should be logged.
    argument_formatter : callable, optional
        Custom runtime argument formatter.
    return_formatter : callable, optional
        Custom return-value formatter.

    Returns
    -------
    callable
        Decorated callable or configured decorator.

    Notes
    -----
    The decorator never suppresses exceptions. It logs the failure and raises
    the original exception again.

    Logging errors are suppressed so they do not interrupt molecular analyses.

    Examples
    --------
    >>> @log_call
    ... def analyze_contacts(model, cutoff=3.5):
    ...     return []

    >>> @log_call(
    ...     level="info",
    ...     log_arguments=True,
    ...     log_return=True,
    ... )
    ... def detect_hbonds(model):
    ...     return []
    """

    positional_function = function

    if isinstance(
        positional_function,
        str,
    ):
        if name is not None:
            raise TypeError(
                "The log-call name was provided "
                "both positionally and by keyword."
            )

        name = positional_function
        positional_function = None

    elif (
        positional_function is not None
        and not _is_decorated_function_candidate(
            positional_function
        )
    ):
        raise TypeError(
            "The positional argument to "
            "@log_call must be a callable "
            "or a custom call name."
        )

    configured_decorator = (
        _build_log_call_decorator(
            name=name,
            logger=logger,
            enabled=enabled,
            level=level,
            error_level=error_level,
            log_arguments=log_arguments,
            log_return=log_return,
            log_duration=log_duration,
            log_exceptions=log_exceptions,
            include_traceback=(
                include_traceback
            ),
            include_self=include_self,
            include_defaults=(
                include_defaults
            ),
            include_argument_names=(
                include_argument_names
            ),
            exclude_argument_names=(
                exclude_argument_names
            ),
            redact_sensitive=(
                redact_sensitive
            ),
            sensitive_names=(
                sensitive_names
            ),
            maximum_value_length=(
                maximum_value_length
            ),
            maximum_return_length=(
                maximum_return_length
            ),
            maximum_exception_length=(
                maximum_exception_length
            ),
            maximum_items=maximum_items,
            multiline_arguments=(
                multiline_arguments
            ),
            nested_indentation=(
                nested_indentation
            ),
            indent_size=indent_size,
            prefix=prefix,
            start_symbol=start_symbol,
            end_symbol=end_symbol,
            error_symbol=error_symbol,
            precision=precision,
            fallback_to_print=(
                fallback_to_print
            ),
            condition=condition,
            argument_formatter=(
                argument_formatter
            ),
            return_formatter=(
                return_formatter
            ),
        )
    )

    if positional_function is not None:
        return configured_decorator(
            positional_function
        )

    return configured_decorator


# -----------------------------------------------------------------------------
# Public module interface
# -----------------------------------------------------------------------------

_SECTION_11_3_PUBLIC_NAMES = [
    "log_call",
]

for public_name in (
    _SECTION_11_3_PUBLIC_NAMES
):
    if public_name not in __all__:
        __all__.append(
            public_name
        )


# =============================================================================
# End of Section 11.3
# =============================================================================

# =============================================================================
# Section 11.4 — Safe Execution Decorator
# =============================================================================


# -----------------------------------------------------------------------------
# Safe-execution constants
# -----------------------------------------------------------------------------

_SAFE_EXECUTION_DEFAULT_EXCEPTIONS = (
    Exception,
)

_SAFE_EXECUTION_NEVER_SUPPRESS = (
    KeyboardInterrupt,
    SystemExit,
    GeneratorExit,
)

_SAFE_EXECUTION_DEFAULT_ERROR_LEVEL = "error"

_SAFE_EXECUTION_ERROR_SYMBOL = "✗"
_SAFE_EXECUTION_RECOVERY_SYMBOL = "↳"
_SAFE_EXECUTION_RERAISE_SYMBOL = "↑"


# -----------------------------------------------------------------------------
# Safe-execution exception validation
# -----------------------------------------------------------------------------

def _normalize_safe_execution_exceptions(
    exceptions: Any,
    *,
    parameter_name: str = "exceptions",
    allow_empty: bool = False,
) -> Tuple[
    Type[BaseException],
    ...,
]:
    """
    Normalize exception classes used by ``@safe_execution``.

    Parameters
    ----------
    exceptions : exception class or iterable of exception classes
        Exception types to normalize.
    parameter_name : str, optional
        Parameter name used in validation messages.
    allow_empty : bool, optional
        Whether an empty collection is accepted.

    Returns
    -------
    tuple of exception classes
        Normalized exception types.

    Raises
    ------
    TypeError
        If the supplied value is not an exception class or iterable of
        exception classes.
    ValueError
        If an empty collection is supplied while ``allow_empty=False``.
    """

    if exceptions is None:
        exception_types: Tuple[
            Type[BaseException],
            ...,
        ] = ()

    elif (
        inspect.isclass(
            exceptions
        )
        and issubclass(
            exceptions,
            BaseException,
        )
    ):
        exception_types = (
            exceptions,
        )

    else:
        if isinstance(
            exceptions,
            str,
        ):
            raise TypeError(
                f"{parameter_name} must contain "
                "exception classes, not strings."
            )

        try:
            exception_types = tuple(
                exceptions
            )

        except TypeError as error:
            raise TypeError(
                f"{parameter_name} must be an "
                "exception class or an iterable "
                "of exception classes."
            ) from error

    if (
        not exception_types
        and not allow_empty
    ):
        raise ValueError(
            f"{parameter_name} cannot be empty."
        )

    normalized_types: List[
        Type[BaseException]
    ] = []

    for exception_type in exception_types:
        if not inspect.isclass(
            exception_type
        ):
            raise TypeError(
                f"Every item in {parameter_name} "
                "must be an exception class."
            )

        if not issubclass(
            exception_type,
            BaseException,
        ):
            raise TypeError(
                f"Every item in {parameter_name} "
                "must inherit from BaseException."
            )

        if exception_type not in normalized_types:
            normalized_types.append(
                exception_type
            )

    return tuple(
        normalized_types
    )


def _validate_safe_execution_exception_configuration(
    caught_exceptions: Tuple[
        Type[BaseException],
        ...,
    ],
    excluded_exceptions: Tuple[
        Type[BaseException],
        ...,
    ],
) -> None:
    """
    Validate the relationship between caught and excluded exceptions.

    Parameters
    ----------
    caught_exceptions : tuple of exception classes
        Exceptions eligible for handling.
    excluded_exceptions : tuple of exception classes
        Exceptions that must always propagate.

    Raises
    ------
    ValueError
        If the configuration is internally contradictory.
    """

    if not caught_exceptions:
        raise ValueError(
            "At least one caught exception type "
            "must be configured."
        )

    for excluded_type in excluded_exceptions:
        if excluded_type in (
            KeyboardInterrupt,
            SystemExit,
            GeneratorExit,
        ):
            continue

        if any(
            caught_type is excluded_type
            for caught_type in caught_exceptions
        ):
            raise ValueError(
                f"{excluded_type.__name__} appears "
                "in both exceptions and "
                "exclude_exceptions."
            )


def _exception_matches_any(
    exception: BaseException,
    exception_types: Tuple[
        Type[BaseException],
        ...,
    ],
) -> bool:
    """
    Return whether an exception matches any configured exception type.

    Parameters
    ----------
    exception : BaseException
        Exception to inspect.
    exception_types : tuple of exception classes
        Candidate exception types.

    Returns
    -------
    bool
        Whether the exception matches.
    """

    if not exception_types:
        return False

    return isinstance(
        exception,
        exception_types,
    )


def _should_handle_exception(
    exception: BaseException,
    *,
    exceptions: Tuple[
        Type[BaseException],
        ...,
    ],
    exclude_exceptions: Tuple[
        Type[BaseException],
        ...,
    ],
) -> bool:
    """
    Return whether ``@safe_execution`` should handle an exception.

    Parameters
    ----------
    exception : BaseException
        Raised exception.
    exceptions : tuple of exception classes
        Exception types eligible for handling.
    exclude_exceptions : tuple of exception classes
        Exception types that must propagate.

    Returns
    -------
    bool
        Whether the exception should be handled.
    """

    if isinstance(
        exception,
        _SAFE_EXECUTION_NEVER_SUPPRESS,
    ):
        return False

    if _exception_matches_any(
        exception,
        exclude_exceptions,
    ):
        return False

    return _exception_matches_any(
        exception,
        exceptions,
    )


# -----------------------------------------------------------------------------
# Fallback-value resolution
# -----------------------------------------------------------------------------

def _copy_safe_default_value(
    value: Any,
    *,
    copy_default: bool,
    deep_copy: bool,
) -> Any:
    """
    Copy a static fallback value when requested.

    Parameters
    ----------
    value : Any
        Static fallback value.
    copy_default : bool
        Whether the value should be copied.
    deep_copy : bool
        Whether ``copy.deepcopy`` should be preferred.

    Returns
    -------
    Any
        Copied or original fallback value.

    Notes
    -----
    Copying avoids returning the same mutable list or dictionary on every
    failed function call.
    """

    if not copy_default:
        return value

    try:
        if deep_copy:
            return copy.deepcopy(
                value
            )

        return copy.copy(
            value
        )

    except Exception:
        return value


def _call_default_factory(
    factory: Callable[..., Any],
    *,
    exception: BaseException,
    function: Callable[..., Any],
    args: Tuple[Any, ...],
    kwargs: Mapping[str, Any],
    logger: Any = None,
) -> Any:
    """
    Call a fallback factory using a flexible signature.

    Supported signatures include:

    ``factory()``

    ``factory(exception)``

    ``factory(exception, function)``

    ``factory(exception, function, args, kwargs)``

    The factory may also accept keyword parameters named:

    - ``exception``;
    - ``error``;
    - ``exc``;
    - ``function``;
    - ``func``;
    - ``args``;
    - ``kwargs``;
    - ``logger``.

    Parameters
    ----------
    factory : callable
        Fallback factory.
    exception : BaseException
        Captured exception.
    function : callable
        Decorated function.
    args : tuple
        Original positional arguments.
    kwargs : mapping
        Original keyword arguments.
    logger : Any, optional
        Resolved logger.

    Returns
    -------
    Any
        Factory result.
    """

    if not callable(
        factory
    ):
        raise TypeError(
            "default_factory must be callable."
        )

    factory_signature = _safe_signature(
        factory
    )

    available_values = {
        "exception": exception,
        "error": exception,
        "exc": exception,
        "function": function,
        "func": function,
        "args": args,
        "kwargs": kwargs,
        "logger": logger,
    }

    if factory_signature is not None:
        accepted_kwargs = {}

        accepts_variable_keywords = any(
            parameter.kind
            is inspect.Parameter.VAR_KEYWORD
            for parameter in (
                factory_signature
                .parameters
                .values()
            )
        )

        if accepts_variable_keywords:
            accepted_kwargs = {
                "exception": exception,
                "function": function,
                "args": args,
                "kwargs": kwargs,
                "logger": logger,
            }

        else:
            for parameter_name in (
                factory_signature.parameters
            ):
                if parameter_name in available_values:
                    accepted_kwargs[
                        parameter_name
                    ] = available_values[
                        parameter_name
                    ]

        try:
            return factory(
                **accepted_kwargs
            )

        except TypeError:
            pass

    positional_attempts = (
        (
            exception,
            function,
            args,
            kwargs,
        ),
        (
            exception,
            function,
        ),
        (
            exception,
        ),
        (),
    )

    last_type_error = None

    for factory_args in positional_attempts:
        try:
            return factory(
                *factory_args
            )

        except TypeError as error:
            last_type_error = error

    if last_type_error is not None:
        raise last_type_error

    return factory()


def _resolve_safe_execution_fallback(
    *,
    default: Any = _DECORATOR_UNSET,
    default_factory: Any = _DECORATOR_UNSET,
    exception: BaseException,
    function: Callable[..., Any],
    args: Tuple[Any, ...],
    kwargs: Mapping[str, Any],
    logger: Any = None,
    copy_default: bool = True,
    deep_copy_default: bool = True,
) -> Any:
    """
    Resolve the value returned after a handled failure.

    Parameters
    ----------
    default : Any, optional
        Static fallback value.
    default_factory : callable, optional
        Callable that creates the fallback value.
    exception : BaseException
        Captured exception.
    function : callable
        Decorated function.
    args : tuple
        Original positional arguments.
    kwargs : mapping
        Original keyword arguments.
    logger : Any, optional
        Resolved logger.
    copy_default : bool, optional
        Whether static defaults should be copied.
    deep_copy_default : bool, optional
        Whether static defaults should be deep-copied.

    Returns
    -------
    Any
        Resolved fallback value.

    Raises
    ------
    ValueError
        If both ``default`` and ``default_factory`` are configured.
    """

    has_default = (
        default is not _DECORATOR_UNSET
    )

    has_default_factory = (
        default_factory
        is not _DECORATOR_UNSET
    )

    if (
        has_default
        and has_default_factory
    ):
        raise ValueError(
            "Use either default or "
            "default_factory, not both."
        )

    if has_default_factory:
        return _call_default_factory(
            default_factory,
            exception=exception,
            function=function,
            args=args,
            kwargs=kwargs,
            logger=logger,
        )

    if has_default:
        return _copy_safe_default_value(
            default,
            copy_default=copy_default,
            deep_copy=deep_copy_default,
        )

    return None


# -----------------------------------------------------------------------------
# Error-callback invocation
# -----------------------------------------------------------------------------

def _invoke_safe_execution_callback(
    callback: Optional[
        Callable[..., Any]
    ],
    *,
    exception: BaseException,
    function: Callable[..., Any],
    args: Tuple[Any, ...],
    kwargs: Mapping[str, Any],
    logger: Any,
    fallback: Any = _DECORATOR_UNSET,
) -> Any:
    """
    Execute an optional safe-execution error callback.

    Parameters
    ----------
    callback : callable or None
        Error callback.
    exception : BaseException
        Captured exception.
    function : callable
        Decorated function.
    args : tuple
        Original positional arguments.
    kwargs : mapping
        Original keyword arguments.
    logger : Any
        Resolved logger.
    fallback : Any, optional
        Resolved fallback value.

    Returns
    -------
    Any
        Callback result.

    Notes
    -----
    In addition to the callback signatures supported by
    ``_call_error_callback``, this helper supports a ``fallback`` keyword.
    """

    if callback is None:
        return None

    if not callable(
        callback
    ):
        raise TypeError(
            "on_error must be callable."
        )

    callback_signature = _safe_signature(
        callback
    )

    available_values = {
        "exception": exception,
        "error": exception,
        "exc": exception,
        "function": function,
        "func": function,
        "args": args,
        "kwargs": kwargs,
        "logger": logger,
        "fallback": (
            None
            if fallback is _DECORATOR_UNSET
            else fallback
        ),
        "default": (
            None
            if fallback is _DECORATOR_UNSET
            else fallback
        ),
    }

    if callback_signature is not None:
        callback_kwargs = {}

        accepts_variable_keywords = any(
            parameter.kind
            is inspect.Parameter.VAR_KEYWORD
            for parameter in (
                callback_signature
                .parameters
                .values()
            )
        )

        if accepts_variable_keywords:
            callback_kwargs = {
                "exception": exception,
                "function": function,
                "args": args,
                "kwargs": kwargs,
                "logger": logger,
                "fallback": (
                    None
                    if fallback
                    is _DECORATOR_UNSET
                    else fallback
                ),
            }

        else:
            for parameter_name in (
                callback_signature.parameters
            ):
                if parameter_name in available_values:
                    callback_kwargs[
                        parameter_name
                    ] = available_values[
                        parameter_name
                    ]

        try:
            return callback(
                **callback_kwargs
            )

        except TypeError:
            pass

    callback_attempts = (
        (
            exception,
            function,
            args,
            kwargs,
        ),
        (
            exception,
            function,
        ),
        (
            exception,
        ),
        (),
    )

    last_type_error = None

    for callback_args in callback_attempts:
        try:
            return callback(
                *callback_args
            )

        except TypeError as error:
            last_type_error = error

    if last_type_error is not None:
        raise last_type_error

    return None


# -----------------------------------------------------------------------------
# Safe-execution result callback
# -----------------------------------------------------------------------------

def _invoke_safe_execution_success_callback(
    callback: Optional[
        Callable[..., Any]
    ],
    *,
    result: Any,
    function: Callable[..., Any],
    args: Tuple[Any, ...],
    kwargs: Mapping[str, Any],
    logger: Any,
) -> Any:
    """
    Execute an optional callback after successful execution.

    Parameters
    ----------
    callback : callable or None
        Success callback.
    result : Any
        Function return value.
    function : callable
        Decorated function.
    args : tuple
        Original positional arguments.
    kwargs : mapping
        Original keyword arguments.
    logger : Any
        Resolved logger.

    Returns
    -------
    Any
        Callback result.
    """

    if callback is None:
        return None

    if not callable(
        callback
    ):
        raise TypeError(
            "on_success must be callable."
        )

    callback_signature = _safe_signature(
        callback
    )

    available_values = {
        "result": result,
        "value": result,
        "function": function,
        "func": function,
        "args": args,
        "kwargs": kwargs,
        "logger": logger,
    }

    if callback_signature is not None:
        callback_kwargs = {}

        accepts_variable_keywords = any(
            parameter.kind
            is inspect.Parameter.VAR_KEYWORD
            for parameter in (
                callback_signature
                .parameters
                .values()
            )
        )

        if accepts_variable_keywords:
            callback_kwargs = {
                "result": result,
                "function": function,
                "args": args,
                "kwargs": kwargs,
                "logger": logger,
            }

        else:
            for parameter_name in (
                callback_signature.parameters
            ):
                if parameter_name in available_values:
                    callback_kwargs[
                        parameter_name
                    ] = available_values[
                        parameter_name
                    ]

        try:
            return callback(
                **callback_kwargs
            )

        except TypeError:
            pass

    callback_attempts = (
        (
            result,
            function,
            args,
            kwargs,
        ),
        (
            result,
            function,
        ),
        (
            result,
        ),
        (),
    )

    last_type_error = None

    for callback_args in callback_attempts:
        try:
            return callback(
                *callback_args
            )

        except TypeError as error:
            last_type_error = error

    if last_type_error is not None:
        raise last_type_error

    return None


# -----------------------------------------------------------------------------
# Callback-failure handling
# -----------------------------------------------------------------------------

def _handle_safe_execution_callback_failure(
    callback_exception: BaseException,
    *,
    callback_name: str,
    original_exception: Optional[
        BaseException
    ] = None,
    logger: Any = None,
    level: str = "error",
    fallback_to_print: bool = False,
    include_traceback: bool = False,
    reraise_callback_errors: bool = False,
) -> None:
    """
    Handle an exception raised by a decorator callback.

    Parameters
    ----------
    callback_exception : BaseException
        Exception raised by the callback.
    callback_name : str
        Callback label.
    original_exception : BaseException, optional
        Original decorated-function exception.
    logger : Any, optional
        Logger-like object.
    level : str, optional
        Log level.
    fallback_to_print : bool, optional
        Whether printing is allowed when no logger is available.
    include_traceback : bool, optional
        Whether callback traceback information should be included.
    reraise_callback_errors : bool, optional
        Whether the callback exception should propagate.
    """

    message = (
        f"Safe-execution callback "
        f"{callback_name!r} failed: "
        f"{type(callback_exception).__name__}: "
        f"{callback_exception}"
    )

    if original_exception is not None:
        message += (
            " | Original error: "
            f"{type(original_exception).__name__}: "
            f"{original_exception}"
        )

    _emit_log_call_message(
        message,
        logger=logger,
        level=level,
        fallback_to_print=(
            fallback_to_print
        ),
        exc_info=(
            True
            if include_traceback
            else None
        ),
    )

    if reraise_callback_errors:
        raise callback_exception


# -----------------------------------------------------------------------------
# Safe-execution message builders
# -----------------------------------------------------------------------------

def _build_safe_execution_error_message(
    *,
    call_label: str,
    exception: BaseException,
    include_traceback: bool,
    maximum_exception_length: Optional[
        int
    ],
    prefix: Optional[str],
    symbol: str,
) -> str:
    """
    Build the primary safe-execution failure message.

    Parameters
    ----------
    call_label : str
        Decorated-function label.
    exception : BaseException
        Captured exception.
    include_traceback : bool
        Whether traceback text should be included.
    maximum_exception_length : int or None
        Maximum exception report length.
    prefix : str or None
        Static message prefix.
    symbol : str
        Failure symbol.

    Returns
    -------
    str
        Failure message.
    """

    message_prefix = ""

    if prefix:
        message_prefix = (
            f"{str(prefix).strip()} "
        )

    exception_report = _format_exception(
        exception,
        include_traceback=(
            include_traceback
        ),
        include_exception_type=True,
        maximum_length=(
            maximum_exception_length
        ),
    )

    return (
        f"{message_prefix}"
        f"{symbol} "
        f"Safe execution failed in "
        f"{call_label} | "
        f"{exception_report}"
    )


def _build_safe_execution_recovery_message(
    *,
    call_label: str,
    fallback: Any,
    maximum_value_length: int,
    maximum_items: int,
    prefix: Optional[str],
    symbol: str,
) -> str:
    """
    Build a message describing the fallback returned after failure.

    Parameters
    ----------
    call_label : str
        Decorated-function label.
    fallback : Any
        Returned fallback value.
    maximum_value_length : int
        Maximum fallback representation length.
    maximum_items : int
        Maximum container items shown.
    prefix : str or None
        Static message prefix.
    symbol : str
        Recovery symbol.

    Returns
    -------
    str
        Recovery message.
    """

    message_prefix = ""

    if prefix:
        message_prefix = (
            f"{str(prefix).strip()} "
        )

    formatted_fallback = (
        _format_decorator_value(
            fallback,
            maximum_length=(
                maximum_value_length
            ),
            maximum_items=maximum_items,
            maximum_depth=2,
        )
    )

    return (
        f"{message_prefix}"
        f"{symbol} "
        f"{call_label} returning fallback: "
        f"{formatted_fallback}"
    )


def _build_safe_execution_reraise_message(
    *,
    call_label: str,
    exception: BaseException,
    prefix: Optional[str],
    symbol: str,
) -> str:
    """
    Build a message indicating that an exception will be reraised.

    Parameters
    ----------
    call_label : str
        Decorated-function label.
    exception : BaseException
        Captured exception.
    prefix : str or None
        Static message prefix.
    symbol : str
        Reraise symbol.

    Returns
    -------
    str
        Reraise message.
    """

    message_prefix = ""

    if prefix:
        message_prefix = (
            f"{str(prefix).strip()} "
        )

    return (
        f"{message_prefix}"
        f"{symbol} "
        f"{call_label} reraising "
        f"{type(exception).__name__}"
    )


# -----------------------------------------------------------------------------
# Safe-execution error logging
# -----------------------------------------------------------------------------

def _log_safe_execution_failure(
    *,
    logger: Any,
    call_label: str,
    exception: BaseException,
    level: str,
    include_traceback: bool,
    maximum_exception_length: Optional[
        int
    ],
    fallback_to_print: bool,
    prefix: Optional[str],
    symbol: str,
) -> None:
    """
    Log a handled safe-execution failure.
    """

    message = (
        _build_safe_execution_error_message(
            call_label=call_label,
            exception=exception,
            include_traceback=(
                include_traceback
            ),
            maximum_exception_length=(
                maximum_exception_length
            ),
            prefix=prefix,
            symbol=symbol,
        )
    )

    _emit_log_call_message(
        message,
        logger=logger,
        level=level,
        fallback_to_print=(
            fallback_to_print
        ),
        exc_info=(
            True
            if include_traceback
            else None
        ),
    )


def _log_safe_execution_recovery(
    *,
    logger: Any,
    call_label: str,
    fallback: Any,
    level: str,
    maximum_value_length: int,
    maximum_items: int,
    fallback_to_print: bool,
    prefix: Optional[str],
    symbol: str,
) -> None:
    """
    Log the fallback returned after failure.
    """

    message = (
        _build_safe_execution_recovery_message(
            call_label=call_label,
            fallback=fallback,
            maximum_value_length=(
                maximum_value_length
            ),
            maximum_items=maximum_items,
            prefix=prefix,
            symbol=symbol,
        )
    )

    _emit_log_call_message(
        message,
        logger=logger,
        level=level,
        fallback_to_print=(
            fallback_to_print
        ),
    )


def _log_safe_execution_reraise(
    *,
    logger: Any,
    call_label: str,
    exception: BaseException,
    level: str,
    fallback_to_print: bool,
    prefix: Optional[str],
    symbol: str,
) -> None:
    """
    Log that the original exception will propagate.
    """

    message = (
        _build_safe_execution_reraise_message(
            call_label=call_label,
            exception=exception,
            prefix=prefix,
            symbol=symbol,
        )
    )

    _emit_log_call_message(
        message,
        logger=logger,
        level=level,
        fallback_to_print=(
            fallback_to_print
        ),
    )


# -----------------------------------------------------------------------------
# Safe-execution decorator factory
# -----------------------------------------------------------------------------

def _build_safe_execution_decorator(
    *,
    name: Optional[str] = None,
    default: Any = _DECORATOR_UNSET,
    default_factory: Any = _DECORATOR_UNSET,
    exceptions: Any = (
        _SAFE_EXECUTION_DEFAULT_EXCEPTIONS
    ),
    exclude_exceptions: Any = (),
    reraise: bool = False,
    logger: Any = None,
    enabled: bool = True,
    log_errors: bool = True,
    log_recovery: bool = True,
    log_reraise: bool = True,
    error_level: str = (
        _SAFE_EXECUTION_DEFAULT_ERROR_LEVEL
    ),
    recovery_level: str = "warning",
    reraise_level: str = "error",
    include_traceback: bool = True,
    maximum_exception_length: Optional[
        int
    ] = None,
    maximum_fallback_length: int = 180,
    maximum_items: int = 6,
    copy_default: bool = True,
    deep_copy_default: bool = True,
    fallback_to_print: bool = False,
    on_error: Optional[
        Callable[..., Any]
    ] = None,
    on_success: Optional[
        Callable[..., Any]
    ] = None,
    callback_result_as_fallback: bool = False,
    suppress_callback_errors: bool = True,
    prefix: Optional[str] = None,
    error_symbol: str = (
        _SAFE_EXECUTION_ERROR_SYMBOL
    ),
    recovery_symbol: str = (
        _SAFE_EXECUTION_RECOVERY_SYMBOL
    ),
    reraise_symbol: str = (
        _SAFE_EXECUTION_RERAISE_SYMBOL
    ),
) -> Callable[[F], F]:
    """
    Build the configured ``@safe_execution`` decorator.

    Parameters are validated once when the decorator is created.
    """

    enabled = _validate_boolean_option(
        enabled,
        parameter_name="enabled",
    )

    reraise = _validate_boolean_option(
        reraise,
        parameter_name="reraise",
    )

    log_errors = _validate_boolean_option(
        log_errors,
        parameter_name="log_errors",
    )

    log_recovery = (
        _validate_boolean_option(
            log_recovery,
            parameter_name="log_recovery",
        )
    )

    log_reraise = (
        _validate_boolean_option(
            log_reraise,
            parameter_name="log_reraise",
        )
    )

    include_traceback = (
        _validate_boolean_option(
            include_traceback,
            parameter_name=(
                "include_traceback"
            ),
        )
    )

    copy_default = (
        _validate_boolean_option(
            copy_default,
            parameter_name="copy_default",
        )
    )

    deep_copy_default = (
        _validate_boolean_option(
            deep_copy_default,
            parameter_name=(
                "deep_copy_default"
            ),
        )
    )

    fallback_to_print = (
        _validate_boolean_option(
            fallback_to_print,
            parameter_name=(
                "fallback_to_print"
            ),
        )
    )

    callback_result_as_fallback = (
        _validate_boolean_option(
            callback_result_as_fallback,
            parameter_name=(
                "callback_result_as_fallback"
            ),
        )
    )

    suppress_callback_errors = (
        _validate_boolean_option(
            suppress_callback_errors,
            parameter_name=(
                "suppress_callback_errors"
            ),
        )
    )

    normalized_error_level = (
        _normalize_log_level(
            error_level,
            default="error",
        )
    )

    normalized_recovery_level = (
        _normalize_log_level(
            recovery_level,
            default="warning",
        )
    )

    normalized_reraise_level = (
        _normalize_log_level(
            reraise_level,
            default="error",
        )
    )

    maximum_exception_length = (
        _validate_optional_positive_integer(
            maximum_exception_length,
            parameter_name=(
                "maximum_exception_length"
            ),
            allow_zero=False,
        )
    )

    maximum_fallback_length = (
        _validate_optional_positive_integer(
            maximum_fallback_length,
            parameter_name=(
                "maximum_fallback_length"
            ),
            allow_zero=False,
        )
    )

    maximum_items = (
        _validate_optional_positive_integer(
            maximum_items,
            parameter_name="maximum_items",
            allow_zero=False,
        )
    )

    if maximum_fallback_length is None:
        maximum_fallback_length = 180

    if maximum_items is None:
        maximum_items = 6

    caught_exceptions = (
        _normalize_safe_execution_exceptions(
            exceptions,
            parameter_name="exceptions",
            allow_empty=False,
        )
    )

    excluded_exceptions = (
        _normalize_safe_execution_exceptions(
            exclude_exceptions,
            parameter_name=(
                "exclude_exceptions"
            ),
            allow_empty=True,
        )
    )

    _validate_safe_execution_exception_configuration(
        caught_exceptions,
        excluded_exceptions,
    )

    has_default = (
        default is not _DECORATOR_UNSET
    )

    has_factory = (
        default_factory
        is not _DECORATOR_UNSET
    )

    if has_default and has_factory:
        raise ValueError(
            "Use either default or "
            "default_factory, not both."
        )

    if (
        has_factory
        and not callable(
            default_factory
        )
    ):
        raise TypeError(
            "default_factory must be callable."
        )

    if (
        on_error is not None
        and not callable(
            on_error
        )
    ):
        raise TypeError(
            "on_error must be callable or None."
        )

    if (
        on_success is not None
        and not callable(
            on_success
        )
    ):
        raise TypeError(
            "on_success must be callable or None."
        )

    if name is not None:
        name = str(
            name
        ).strip()

        if not name:
            name = None

    if prefix is not None:
        prefix = str(
            prefix
        ).strip()

        if not prefix:
            prefix = None

    error_symbol = str(
        error_symbol
    )

    recovery_symbol = str(
        recovery_symbol
    )

    reraise_symbol = str(
        reraise_symbol
    )

    def decorator(
        function: F,
    ) -> F:
        if not callable(
            function
        ):
            raise TypeError(
                "@safe_execution can only "
                "decorate callable objects."
            )

        call_label = (
            _resolve_callable_label(
                function,
                custom_name=name,
                include_module=False,
            )
        )

        configuration = {
            "name": call_label,
            "enabled": enabled,
            "exceptions": tuple(
                exception_type.__name__
                for exception_type
                in caught_exceptions
            ),
            "exclude_exceptions": tuple(
                exception_type.__name__
                for exception_type
                in excluded_exceptions
            ),
            "reraise": reraise,
            "log_errors": log_errors,
            "log_recovery": (
                log_recovery
            ),
            "log_reraise": log_reraise,
            "include_traceback": (
                include_traceback
            ),
            "copy_default": copy_default,
            "deep_copy_default": (
                deep_copy_default
            ),
            "callback_result_as_fallback": (
                callback_result_as_fallback
            ),
            "suppress_callback_errors": (
                suppress_callback_errors
            ),
        }

        def handle_failure(
            exception: BaseException,
            args: Tuple[Any, ...],
            kwargs: Mapping[str, Any],
        ) -> Any:
            """
            Handle one failed function call.

            Parameters
            ----------
            exception : BaseException
                Raised exception.
            args : tuple
                Runtime positional arguments.
            kwargs : mapping
                Runtime keyword arguments.

            Returns
            -------
            Any
                Fallback value.

            Raises
            ------
            BaseException
                The original error when it is excluded, unmatched or
                configured for reraising.
            """

            if not _should_handle_exception(
                exception,
                exceptions=(
                    caught_exceptions
                ),
                exclude_exceptions=(
                    excluded_exceptions
                ),
            ):
                raise exception

            resolved_logger = (
                _resolve_decorator_logger(
                    explicit_logger=logger,
                    args=args,
                    kwargs=kwargs,
                )
            )

            if log_errors:
                _log_safe_execution_failure(
                    logger=resolved_logger,
                    call_label=call_label,
                    exception=exception,
                    level=(
                        normalized_error_level
                    ),
                    include_traceback=(
                        include_traceback
                    ),
                    maximum_exception_length=(
                        maximum_exception_length
                    ),
                    fallback_to_print=(
                        fallback_to_print
                    ),
                    prefix=prefix,
                    symbol=error_symbol,
                )

            if reraise:
                if log_reraise:
                    _log_safe_execution_reraise(
                        logger=resolved_logger,
                        call_label=call_label,
                        exception=exception,
                        level=(
                            normalized_reraise_level
                        ),
                        fallback_to_print=(
                            fallback_to_print
                        ),
                        prefix=prefix,
                        symbol=reraise_symbol,
                    )

                raise exception

            fallback = (
                _resolve_safe_execution_fallback(
                    default=default,
                    default_factory=(
                        default_factory
                    ),
                    exception=exception,
                    function=function,
                    args=args,
                    kwargs=kwargs,
                    logger=resolved_logger,
                    copy_default=copy_default,
                    deep_copy_default=(
                        deep_copy_default
                    ),
                )
            )

            if on_error is not None:
                try:
                    callback_result = (
                        _invoke_safe_execution_callback(
                            on_error,
                            exception=exception,
                            function=function,
                            args=args,
                            kwargs=kwargs,
                            logger=resolved_logger,
                            fallback=fallback,
                        )
                    )

                    if callback_result_as_fallback:
                        fallback = (
                            callback_result
                        )

                except BaseException as callback_error:
                    if isinstance(
                        callback_error,
                        _SAFE_EXECUTION_NEVER_SUPPRESS,
                    ):
                        raise

                    _handle_safe_execution_callback_failure(
                        callback_error,
                        callback_name="on_error",
                        original_exception=(
                            exception
                        ),
                        logger=resolved_logger,
                        level=(
                            normalized_error_level
                        ),
                        fallback_to_print=(
                            fallback_to_print
                        ),
                        include_traceback=(
                            include_traceback
                        ),
                        reraise_callback_errors=(
                            not suppress_callback_errors
                        ),
                    )

            if log_recovery:
                _log_safe_execution_recovery(
                    logger=resolved_logger,
                    call_label=call_label,
                    fallback=fallback,
                    level=(
                        normalized_recovery_level
                    ),
                    maximum_value_length=(
                        maximum_fallback_length
                    ),
                    maximum_items=(
                        maximum_items
                    ),
                    fallback_to_print=(
                        fallback_to_print
                    ),
                    prefix=prefix,
                    symbol=recovery_symbol,
                )

            return fallback

        def handle_success(
            result: Any,
            args: Tuple[Any, ...],
            kwargs: Mapping[str, Any],
        ) -> Any:
            """
            Execute the optional success callback.

            The decorated function's original result is always preserved.
            """

            if on_success is None:
                return result

            resolved_logger = (
                _resolve_decorator_logger(
                    explicit_logger=logger,
                    args=args,
                    kwargs=kwargs,
                )
            )

            try:
                _invoke_safe_execution_success_callback(
                    on_success,
                    result=result,
                    function=function,
                    args=args,
                    kwargs=kwargs,
                    logger=resolved_logger,
                )

            except BaseException as callback_error:
                if isinstance(
                    callback_error,
                    _SAFE_EXECUTION_NEVER_SUPPRESS,
                ):
                    raise

                _handle_safe_execution_callback_failure(
                    callback_error,
                    callback_name="on_success",
                    logger=resolved_logger,
                    level=(
                        normalized_error_level
                    ),
                    fallback_to_print=(
                        fallback_to_print
                    ),
                    include_traceback=(
                        include_traceback
                    ),
                    reraise_callback_errors=(
                        not suppress_callback_errors
                    ),
                )

            return result

        if _is_coroutine_callable(
            function
        ):

            async def async_wrapper(
                *args: Any,
                **kwargs: Any,
            ) -> Any:
                if not enabled:
                    return await function(
                        *args,
                        **kwargs,
                    )

                try:
                    result = await function(
                        *args,
                        **kwargs,
                    )

                except _SAFE_EXECUTION_NEVER_SUPPRESS:
                    raise

                except BaseException as exception:
                    return handle_failure(
                        exception,
                        args,
                        kwargs,
                    )

                return handle_success(
                    result,
                    args,
                    kwargs,
                )

            wrapped_function = (
                _copy_wrapper_metadata(
                    async_wrapper,
                    function,
                )
            )

        else:

            def sync_wrapper(
                *args: Any,
                **kwargs: Any,
            ) -> Any:
                if not enabled:
                    return function(
                        *args,
                        **kwargs,
                    )

                try:
                    result = function(
                        *args,
                        **kwargs,
                    )

                except _SAFE_EXECUTION_NEVER_SUPPRESS:
                    raise

                except BaseException as exception:
                    return handle_failure(
                        exception,
                        args,
                        kwargs,
                    )

                return handle_success(
                    result,
                    args,
                    kwargs,
                )

            wrapped_function = (
                _copy_wrapper_metadata(
                    sync_wrapper,
                    function,
                )
            )

        wrapped_function = (
            _set_decorator_metadata(
                wrapped_function,
                decorator_name=(
                    "safe_execution"
                ),
                configuration=configuration,
            )
        )

        return cast(
            F,
            wrapped_function,
        )

    return decorator


# -----------------------------------------------------------------------------
# Public safe-execution decorator
# -----------------------------------------------------------------------------

def safe_execution(
    function: Any = None,
    name: Optional[str] = None,
    *,
    default: Any = _DECORATOR_UNSET,
    default_factory: Any = _DECORATOR_UNSET,
    exceptions: Any = (
        _SAFE_EXECUTION_DEFAULT_EXCEPTIONS
    ),
    exclude_exceptions: Any = (),
    reraise: bool = False,
    logger: Any = None,
    enabled: bool = True,
    log_errors: bool = True,
    log_recovery: bool = True,
    log_reraise: bool = True,
    error_level: str = (
        _SAFE_EXECUTION_DEFAULT_ERROR_LEVEL
    ),
    recovery_level: str = "warning",
    reraise_level: str = "error",
    include_traceback: bool = True,
    maximum_exception_length: Optional[
        int
    ] = None,
    maximum_fallback_length: int = 180,
    maximum_items: int = 6,
    copy_default: bool = True,
    deep_copy_default: bool = True,
    fallback_to_print: bool = False,
    on_error: Optional[
        Callable[..., Any]
    ] = None,
    on_success: Optional[
        Callable[..., Any]
    ] = None,
    callback_result_as_fallback: bool = False,
    suppress_callback_errors: bool = True,
    prefix: Optional[str] = None,
    error_symbol: str = (
        _SAFE_EXECUTION_ERROR_SYMBOL
    ),
    recovery_symbol: str = (
        _SAFE_EXECUTION_RECOVERY_SYMBOL
    ),
    reraise_symbol: str = (
        _SAFE_EXECUTION_RERAISE_SYMBOL
    ),
) -> Any:
    """
    Execute a callable with standardized exception handling.

    Supported forms
    ---------------
    ``@safe_execution``

    ``@safe_execution()``

    ``@safe_execution("Contact analysis")``

    ``@safe_execution(default=None)``

    ``@safe_execution(default_factory=list)``

    Parameters
    ----------
    function : callable or str, optional
        Function being decorated when the decorator is used directly. A
        positional string is interpreted as a custom function name.
    name : str, optional
        Custom function label used in log messages.
    default : Any, optional
        Static value returned after a handled error.
    default_factory : callable, optional
        Callable used to construct the fallback value. It may optionally
        receive exception and function context.
    exceptions : exception class or iterable, optional
        Exception types that should be handled. The default is ``Exception``.
    exclude_exceptions : exception class or iterable, optional
        Exception types that must propagate even if covered by ``exceptions``.
    reraise : bool, optional
        Whether handled exceptions should be logged and reraised.
    logger : Any, optional
        Explicit logger-like object.
    enabled : bool, optional
        Whether safe execution is active.
    log_errors : bool, optional
        Whether handled exceptions should be logged.
    log_recovery : bool, optional
        Whether returned fallback values should be logged.
    log_reraise : bool, optional
        Whether reraising should be explicitly logged.
    error_level : str, optional
        Log level for handled failures.
    recovery_level : str, optional
        Log level for fallback messages.
    reraise_level : str, optional
        Log level for reraising messages.
    include_traceback : bool, optional
        Whether tracebacks should be included in failure logs.
    maximum_exception_length : int or None, optional
        Maximum exception report length.
    maximum_fallback_length : int, optional
        Maximum fallback representation length.
    maximum_items : int, optional
        Maximum container items shown in fallback representations.
    copy_default : bool, optional
        Whether static fallback values should be copied before being returned.
    deep_copy_default : bool, optional
        Whether ``copy.deepcopy`` should be used for static fallbacks.
    fallback_to_print : bool, optional
        Whether messages should be printed when no logger is available.
    on_error : callable, optional
        Callback executed after a handled exception.
    on_success : callable, optional
        Callback executed after successful function execution.
    callback_result_as_fallback : bool, optional
        Whether the result of ``on_error`` should replace the configured
        fallback.
    suppress_callback_errors : bool, optional
        Whether exceptions raised by callbacks should be logged and suppressed.
    prefix : str, optional
        Static prefix inserted before safe-execution messages.
    error_symbol : str, optional
        Symbol used for failure messages.
    recovery_symbol : str, optional
        Symbol used for fallback messages.
    reraise_symbol : str, optional
        Symbol used for reraising messages.

    Returns
    -------
    callable
        Decorated callable or configured decorator.

    Notes
    -----
    ``KeyboardInterrupt``, ``SystemExit`` and ``GeneratorExit`` are never
    suppressed.

    By default, the decorator handles subclasses of ``Exception``. It does not
    normally suppress low-level interpreter-control exceptions.

    Examples
    --------
    >>> @safe_execution(default_factory=list)
    ... def detect_contacts(model):
    ...     ...

    >>> @safe_execution(
    ...     exceptions=(ValueError, OSError),
    ...     exclude_exceptions=PermissionError,
    ... )
    ... def load_results(path):
    ...     ...

    >>> @safe_execution(reraise=True)
    ... def validate_model(model):
    ...     ...
    """

    positional_function = function

    if isinstance(
        positional_function,
        str,
    ):
        if name is not None:
            raise TypeError(
                "The safe-execution name was "
                "provided both positionally "
                "and by keyword."
            )

        name = positional_function
        positional_function = None

    elif (
        positional_function is not None
        and not _is_decorated_function_candidate(
            positional_function
        )
    ):
        raise TypeError(
            "The positional argument to "
            "@safe_execution must be a "
            "callable or a custom name."
        )

    configured_decorator = (
        _build_safe_execution_decorator(
            name=name,
            default=default,
            default_factory=(
                default_factory
            ),
            exceptions=exceptions,
            exclude_exceptions=(
                exclude_exceptions
            ),
            reraise=reraise,
            logger=logger,
            enabled=enabled,
            log_errors=log_errors,
            log_recovery=log_recovery,
            log_reraise=log_reraise,
            error_level=error_level,
            recovery_level=(
                recovery_level
            ),
            reraise_level=reraise_level,
            include_traceback=(
                include_traceback
            ),
            maximum_exception_length=(
                maximum_exception_length
            ),
            maximum_fallback_length=(
                maximum_fallback_length
            ),
            maximum_items=maximum_items,
            copy_default=copy_default,
            deep_copy_default=(
                deep_copy_default
            ),
            fallback_to_print=(
                fallback_to_print
            ),
            on_error=on_error,
            on_success=on_success,
            callback_result_as_fallback=(
                callback_result_as_fallback
            ),
            suppress_callback_errors=(
                suppress_callback_errors
            ),
            prefix=prefix,
            error_symbol=error_symbol,
            recovery_symbol=(
                recovery_symbol
            ),
            reraise_symbol=(
                reraise_symbol
            ),
        )
    )

    if positional_function is not None:
        return configured_decorator(
            positional_function
        )

    return configured_decorator


# -----------------------------------------------------------------------------
# Public module interface
# -----------------------------------------------------------------------------

_SECTION_11_4_PUBLIC_NAMES = [
    "safe_execution",
]

for public_name in (
    _SECTION_11_4_PUBLIC_NAMES
):
    if public_name not in __all__:
        __all__.append(
            public_name
        )


# =============================================================================
# End of Section 11.4
# =============================================================================


# =============================================================================
# Section 12 — Module Self-Test
# =============================================================================


# -----------------------------------------------------------------------------
# Self-test support classes
# -----------------------------------------------------------------------------

class _SelfTestLogger:
    """
    Minimal in-memory logger used by the module self-test.

    The class provides a logger-like interface compatible with the utility
    decorators without depending on ChimeraX or the standard logging module.
    """

    def __init__(
        self,
    ) -> None:
        self.records: List[
            Tuple[str, str]
        ] = []

    def _record(
        self,
        level: str,
        message: Any,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Store one formatted log record.
        """

        del kwargs

        try:
            if args:
                formatted_message = (
                    str(message)
                    % args
                )

            else:
                formatted_message = str(
                    message
                )

        except Exception:
            formatted_message = " ".join(
                [
                    str(message),
                    *(
                        str(argument)
                        for argument in args
                    ),
                ]
            )

        self.records.append(
            (
                level,
                formatted_message,
            )
        )

    def debug(
        self,
        message: Any,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Store a DEBUG record."""

        self._record(
            "debug",
            message,
            *args,
            **kwargs,
        )

    def info(
        self,
        message: Any,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Store an INFO record."""

        self._record(
            "info",
            message,
            *args,
            **kwargs,
        )

    def warning(
        self,
        message: Any,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Store a WARNING record."""

        self._record(
            "warning",
            message,
            *args,
            **kwargs,
        )

    def error(
        self,
        message: Any,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Store an ERROR record."""

        self._record(
            "error",
            message,
            *args,
            **kwargs,
        )

    def critical(
        self,
        message: Any,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Store a CRITICAL record."""

        self._record(
            "critical",
            message,
            *args,
            **kwargs,
        )

    def exception(
        self,
        message: Any,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Store an EXCEPTION record."""

        self._record(
            "exception",
            message,
            *args,
            **kwargs,
        )

    def contains(
        self,
        text: str,
        *,
        level: Optional[str] = None,
    ) -> bool:
        """
        Return whether a stored record contains text.

        Parameters
        ----------
        text : str
            Text to search for.
        level : str, optional
            Optional log-level filter.

        Returns
        -------
        bool
            Whether a matching record exists.
        """

        normalized_text = str(
            text
        ).lower()

        for (
            record_level,
            record_message,
        ) in self.records:
            if (
                level is not None
                and record_level
                != str(level).lower()
            ):
                continue

            if (
                normalized_text
                in record_message.lower()
            ):
                return True

        return False


class _SelfTestTimer:
    """
    Minimal timer-like object used to test ``@timer``.

    The object intentionally exposes only ``start`` and ``stop`` methods so
    the decorator's duck-typed timer integration can be tested independently
    of the full ``AnalysisTimer`` implementation.
    """

    def __init__(
        self,
    ) -> None:
        self.started_steps: List[str] = []
        self.completed_steps: List[
            Dict[str, Any]
        ] = []

    def start(
        self,
        step: str,
        **kwargs: Any,
    ) -> None:
        """
        Register a started step.
        """

        del kwargs

        self.started_steps.append(
            str(step)
        )

    def stop(
        self,
        step: str,
        elapsed: Optional[float] = None,
        status: Optional[str] = None,
        exception: Optional[
            BaseException
        ] = None,
        **kwargs: Any,
    ) -> None:
        """
        Register a completed step.
        """

        del kwargs

        self.completed_steps.append(
            {
                "step": str(step),
                "elapsed": elapsed,
                "status": status,
                "exception": exception,
            }
        )


class _SelfTestStructure:
    """
    Minimal structure object for residue tests.
    """

    def __init__(
        self,
        name: str,
        identifier: str,
    ) -> None:
        self.name = name
        self.id_string = identifier


class _SelfTestResidue:
    """
    Minimal ChimeraX-like residue object for residue utility tests.
    """

    def __init__(
        self,
        name: str,
        chain_id: str,
        number: int,
        *,
        insertion_code: str = "",
        structure: Any = None,
        atomspec: Optional[str] = None,
    ) -> None:
        self.name = name
        self.chain_id = chain_id
        self.number = number
        self.insertion_code = (
            insertion_code
        )
        self.structure = structure
        self.atomspec = atomspec


class _SelfTestAtom:
    """Minimal ChimeraX-like atom used by integration tests."""

    def __init__(
        self,
        name: str,
        residue: Any,
        coordinates: Sequence[float],
    ) -> None:
        self.name = str(name)
        self.residue = residue
        self.coord = np.asarray(
            coordinates,
            dtype=float,
        )
        self.element_name = "C"
        self.atomspec = (
            f"{getattr(residue, 'atomspec', '')}@{self.name}"
        )


class _SelfTestAtoms(list):
    """List-like atom collection exposing unique residues."""

    @property
    def unique_residues(
        self,
    ) -> List[Any]:
        residues: List[Any] = []

        for atom in self:
            residue = getattr(
                atom,
                "residue",
                None,
            )

            if residue not in residues:
                residues.append(residue)

        return residues


class _SelfTestAtomicStructure:
    """Synthetic atomic structure used for model-discovery tests."""

    def __init__(
        self,
        name: str,
        identifier: str,
        residue_names: Sequence[str],
        *,
        atoms_per_residue: int,
    ) -> None:
        self.name = str(name)
        self.id_string = str(identifier)
        self.id = tuple(
            int(part)
            for part in self.id_string.split(".")
        )
        self.residues: List[_SelfTestResidue] = []
        self.atoms = _SelfTestAtoms()

        for residue_index, residue_name in enumerate(
            residue_names,
            start=1,
        ):
            residue = _SelfTestResidue(
                residue_name,
                "A",
                residue_index,
                structure=self,
                atomspec=(
                    f"#{self.id_string}/A:{residue_index}"
                ),
            )
            residue.atoms = []
            self.residues.append(residue)

            for atom_index in range(
                int(atoms_per_residue)
            ):
                atom = _SelfTestAtom(
                    f"C{atom_index + 1}",
                    residue,
                    (
                        float(atom_index),
                        float(residue_index),
                        0.0,
                    ),
                )
                residue.atoms.append(atom)
                self.atoms.append(atom)

    @property
    def atomspec(
        self,
    ) -> str:
        return f"#{self.id_string}"


class _SelfTestModelManager:
    """Minimal ChimeraX-like model manager."""

    def __init__(
        self,
        models: Iterable[Any],
    ) -> None:
        self._models = list(models)

    def list(
        self,
    ) -> List[Any]:
        return list(self._models)


class _SelfTestSession:
    """Minimal ChimeraX-like session containing a model manager."""

    def __init__(
        self,
        models: Iterable[Any],
    ) -> None:
        self.models = _SelfTestModelManager(
            models
        )


# -----------------------------------------------------------------------------
# Self-test result manager
# -----------------------------------------------------------------------------

_SELF_TEST_CODE_FAILURE = "code_failure"
_SELF_TEST_TEST_FAILURE = "test_failure"
_SELF_TEST_ENVIRONMENTAL_LIMITATION = "environmental_limitation"

_SELF_TEST_FAILURE_CATEGORIES = {
    _SELF_TEST_CODE_FAILURE,
    _SELF_TEST_TEST_FAILURE,
    _SELF_TEST_ENVIRONMENTAL_LIMITATION,
}


def _normalize_self_test_failure_category(
    category: str,
) -> str:
    """Return a validated self-test failure category."""

    normalized_category = (
        str(category)
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )

    if normalized_category not in _SELF_TEST_FAILURE_CATEGORIES:
        valid_categories = ", ".join(
            sorted(_SELF_TEST_FAILURE_CATEGORIES)
        )
        raise ValueError(
            f"Unsupported self-test failure category {category!r}. "
            f"Expected one of: {valid_categories}."
        )

    return normalized_category


class _SelfTestRunner:
    """
    Small assertion-based test runner for ``utils.py``.

    The runner avoids external testing dependencies, classifies failures as
    code failures, test failures or environmental limitations, and provides a
    compact terminal report.
    """

    def __init__(
        self,
        *,
        emit: bool = True,
    ) -> None:
        self.emit = bool(emit)
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.failures: List[str] = []
        self.skips: List[str] = []
        self.failure_counts: Dict[str, int] = {
            _SELF_TEST_CODE_FAILURE: 0,
            _SELF_TEST_TEST_FAILURE: 0,
            _SELF_TEST_ENVIRONMENTAL_LIMITATION: 0,
        }

    def _print(
        self,
        message: str = "",
    ) -> None:
        """Print a runner message when output is enabled."""

        if self.emit:
            print(message)

    def check(
        self,
        condition: Any,
        message: str,
    ) -> None:
        """Assert a boolean condition."""

        if not condition:
            raise AssertionError(message)

    def equal(
        self,
        observed: Any,
        expected: Any,
        message: str,
    ) -> None:
        """Assert equality."""

        if observed != expected:
            raise AssertionError(
                f"{message}\n"
                f"Observed: {observed!r}\n"
                f"Expected: {expected!r}"
            )

    def close(
        self,
        observed: float,
        expected: float,
        *,
        tolerance: float = 1e-9,
        message: str,
    ) -> None:
        """Assert numerical proximity."""

        if not math.isclose(
            float(observed),
            float(expected),
            rel_tol=tolerance,
            abs_tol=tolerance,
        ):
            raise AssertionError(
                f"{message}\n"
                f"Observed: {observed!r}\n"
                f"Expected: {expected!r}"
            )

    def run(
        self,
        name: str,
        test_function: Callable[
            [],
            Any,
        ],
        *,
        failure_category: str = _SELF_TEST_CODE_FAILURE,
    ) -> None:
        """Execute and classify one test function."""

        normalized_category = _normalize_self_test_failure_category(
            failure_category
        )

        try:
            test_function()

        except Exception as error:
            self.failed += 1
            self.failure_counts[normalized_category] += 1

            failure_message = (
                f"{name} [{normalized_category}]: "
                f"{type(error).__name__}: {error}"
            )

            self.failures.append(failure_message)
            self._print(
                f"[FAIL] {name} [{normalized_category}]"
            )

        else:
            self.passed += 1
            self._print(f"[PASS] {name}")

    def skip(
        self,
        name: str,
        reason: str,
        *,
        category: str = _SELF_TEST_ENVIRONMENTAL_LIMITATION,
    ) -> None:
        """Register and classify one skipped test."""

        normalized_category = _normalize_self_test_failure_category(
            category
        )

        self.skipped += 1
        self.failure_counts[normalized_category] += 1

        skip_message = (
            f"{name} [{normalized_category}]: {reason}"
        )
        self.skips.append(skip_message)
        self._print(
            f"[SKIP] {name} [{normalized_category}]: {reason}"
        )

    @property
    def total(
        self,
    ) -> int:
        """Return the total number of registered tests."""

        return self.passed + self.failed + self.skipped

    @property
    def successful(
        self,
    ) -> bool:
        """Return whether every executed test passed."""

        return self.failed == 0

    @property
    def unjustified_failures(
        self,
    ) -> int:
        """Return failures not classified as environmental limitations."""

        return (
            self.failure_counts[_SELF_TEST_CODE_FAILURE]
            + self.failure_counts[_SELF_TEST_TEST_FAILURE]
        )

    def print_summary(
        self,
    ) -> None:
        """Print the final test summary and failure classification."""

        self._print()

        if not self.emit:
            return

        print_title(
            "UTILS.PY SELF-TEST SUMMARY",
            width=72,
        )

        summary_rows = [
            {"Status": "Passed", "Count": self.passed},
            {"Status": "Failed", "Count": self.failed},
            {"Status": "Skipped", "Count": self.skipped},
            {"Status": "Total", "Count": self.total},
            {
                "Status": "Code failures",
                "Count": self.failure_counts[
                    _SELF_TEST_CODE_FAILURE
                ],
            },
            {
                "Status": "Test failures",
                "Count": self.failure_counts[
                    _SELF_TEST_TEST_FAILURE
                ],
            },
            {
                "Status": "Environmental limitations",
                "Count": self.failure_counts[
                    _SELF_TEST_ENVIRONMENTAL_LIMITATION
                ],
            },
        ]

        print_table(summary_rows)

        if self.failures:
            print()
            print_title(
                "FAILURES",
                width=72,
                character="-",
            )

            for failure_index, failure_message in enumerate(
                self.failures,
                start=1,
            ):
                print(f"{failure_index}. {failure_message}")

        if self.skips:
            print()
            print_title(
                "ENVIRONMENTAL LIMITATIONS",
                width=72,
                character="-",
            )

            for skip_index, skip_message in enumerate(
                self.skips,
                start=1,
            ):
                print(f"{skip_index}. {skip_message}")


# -----------------------------------------------------------------------------
# Section 2 — Logging tests
# -----------------------------------------------------------------------------

def _test_logging_utilities(
    runner: _SelfTestRunner,
) -> None:
    """
    Test logger-detection and logging helper behavior.
    """

    logger = _SelfTestLogger()

    runner.check(
        _is_logger_like(
            logger
        ),
        "The in-memory logger was not "
        "recognized as logger-like.",
    )

    logged = _log_message(
        logger,
        "info",
        "Pose %s analyzed",
        "pose_01",
    )

    runner.check(
        logged,
        "_log_message() reported failure.",
    )

    runner.check(
        logger.contains(
            "pose_01 analyzed",
            level="info",
        ),
        "Expected log message was not stored.",
    )

    class Analyzer:
        def __init__(
            self,
        ) -> None:
            self.logger = logger

    analyzer = Analyzer()

    resolved_logger = (
        _resolve_decorator_logger(
            args=(
                analyzer,
            )
        )
    )

    runner.check(
        resolved_logger is logger,
        "Logger resolution from self.logger failed.",
    )


# -----------------------------------------------------------------------------
# Section 3 — Timer tests
# -----------------------------------------------------------------------------

def _test_timer_infrastructure(
    runner: _SelfTestRunner,
) -> None:
    """
    Test timer-like object detection and duration formatting.
    """

    timer_object = _SelfTestTimer()

    runner.check(
        _is_timer_like(
            timer_object
        ),
        "The self-test timer was not "
        "recognized as timer-like.",
    )

    runner.equal(
        _format_elapsed_time(
            0.5,
            precision=2,
        ),
        "0.50 s",
        "Unexpected elapsed-time formatting.",
    )

    runner.equal(
        _format_elapsed_time(
            65.25,
            precision=2,
        ),
        "1 min 5.25 s",
        "Unexpected minute formatting.",
    )


# -----------------------------------------------------------------------------
# Section 4 — DockModel tests
# -----------------------------------------------------------------------------

def _test_dock_model(
    runner: _SelfTestRunner,
) -> None:
    """
    Test basic ``DockModel`` construction and serialization.
    """

    dock_model = DockModel(
        receptor="receptor",
        pose="pose",
        ligand="ligand",
    )

    dock_model.contacts = [
        {
            "distance": 3.2,
        }
    ]

    dock_model.metadata[
        "pose_index"
    ] = 1

    serialized = dock_model.to_dict()

    runner.check(
        isinstance(
            serialized,
            dict,
        ),
        "DockModel.to_dict() did not "
        "return a dictionary.",
    )

    runner.equal(
        serialized.get(
            "metadata",
            {}
        ).get(
            "pose_index"
        ),
        1,
        "DockModel metadata was not serialized.",
    )

    serialized_with_pose = (
        dock_model.to_dict(
            include_pose=True,
            include_receptor=True,
            include_ligand=True,
        )
    )

    runner.check(
        "pose" in serialized_with_pose,
        "Pose was not included when requested.",
    )

    runner.check(
        "receptor"
        in serialized_with_pose,
        "Receptor was not included when requested.",
    )

    runner.check(
        "ligand"
        in serialized_with_pose,
        "Ligand was not included when requested.",
    )


# -----------------------------------------------------------------------------
# Section 4.1 — Deep DockModel integration tests
# -----------------------------------------------------------------------------

def _test_strict_serialization(
    runner: _SelfTestRunner,
) -> None:
    """Test strict JSON output, recursive values and molecular references."""

    dock_model = DockModel(
        name="strict_pose",
        score=float("nan"),
        contacts=[
            {
                "distance": np.float32(
                    3.2
                ),
            }
        ],
        metadata={
            "positive_infinity": float("inf"),
            "negative_infinity": float("-inf"),
            "array": np.array(
                [
                    1,
                    2,
                ]
            ),
        },
    )
    dock_model.metadata[
        "recursive"
    ] = dock_model.metadata

    serialized = dock_model.to_dict()
    encoded = json.dumps(
        serialized,
        allow_nan=False,
        sort_keys=True,
    )

    runner.check(
        bool(encoded),
        "Strict JSON serialization returned empty output.",
    )
    runner.equal(
        serialized["score"],
        None,
        "A non-finite DockModel score was not normalized.",
    )
    runner.equal(
        serialized["metadata"][
            "positive_infinity"
        ],
        None,
        "Positive infinity was not normalized.",
    )
    runner.equal(
        serialized["metadata"][
            "recursive"
        ],
        "<recursive>",
        "A recursive metadata reference was not bounded.",
    )

    with tempfile.TemporaryDirectory() as temporary_directory:
        output_path = save_json(
            dock_model,
            Path(temporary_directory)
            / "strict.json",
        )

        with output_path.open(
            mode="r",
            encoding="utf-8",
        ) as input_file:
            loaded_data = json.load(
                input_file
            )

        runner.equal(
            loaded_data["score"],
            None,
            "save_json() did not write strict JSON null values.",
        )
        runner.equal(
            loaded_data["metadata"][
                "recursive"
            ],
            "<recursive>",
            "save_json() did not preserve recursion protection.",
        )


def _test_dock_model_normalization(
    runner: _SelfTestRunner,
) -> None:
    """Test optional collections, mappings and invalid input handling."""

    dock_model = DockModel(
        name=None,
        contacts=None,
        hbonds=(
            {
                "kind": "hydrogen_bond",
            },
        ),
        hydrophobic=None,
        pi={
            "stacking": None,
        },
        statistics=None,
        metadata=None,
        files={
            "json": "results.json",
        },
    )

    runner.equal(
        dock_model.name,
        "unnamed_pose",
        "A None DockModel name was not normalized.",
    )
    runner.equal(
        dock_model.contacts,
        [],
        "None contacts were not normalized to an empty list.",
    )
    runner.equal(
        len(dock_model.hbonds),
        1,
        "Tuple hydrogen bonds were not normalized to a list.",
    )
    runner.equal(
        dock_model.pi["stacking"],
        [],
        "None pi interactions were not normalized.",
    )
    runner.check(
        isinstance(
            dock_model.files["json"],
            Path,
        ),
        "A path-like DockModel file was not normalized to Path.",
    )

    invalid_configurations = (
        {
            "statistics": [],
        },
        {
            "metadata": [],
        },
        {
            "pi": [],
        },
        {
            "files": {
                "json": 3,
            },
        },
    )

    for configuration in invalid_configurations:
        try:
            DockModel(
                name="invalid",
                **configuration,
            )
        except TypeError:
            continue

        raise AssertionError(
            "DockModel accepted an invalid field configuration: "
            f"{configuration!r}."
        )


def _test_synthetic_model_discovery(
    runner: _SelfTestRunner,
) -> None:
    """Test receptor and pose discovery with synthetic atomic structures."""

    receptor = _SelfTestAtomicStructure(
        "receptor",
        "1",
        [
            "ALA",
        ] * 30,
        atoms_per_residue=10,
    )
    pose = _SelfTestAtomicStructure(
        "pose",
        "2",
        [
            "LIG",
        ],
        atoms_per_residue=20,
    )
    session = _SelfTestSession(
        [
            receptor,
            pose,
        ]
    )

    dock_models = create_dock_models(
        session
    )

    runner.equal(
        len(dock_models),
        1,
        "Synthetic model discovery returned an unexpected pose count.",
    )

    dock_model = dock_models[0]

    runner.check(
        dock_model.receptor is receptor,
        "The synthetic receptor was not attached to DockModel.",
    )
    runner.check(
        dock_model.pose is pose,
        "The synthetic pose was not attached to DockModel.",
    )
    runner.check(
        isinstance(
            dock_model.ligand,
            _SelfTestResidue,
        ),
        "The single-residue ligand was not extracted from the pose.",
    )

    serialized = dock_model.to_dict(
        include_pose=True,
        include_receptor=True,
        include_ligand=True,
    )

    json.dumps(
        serialized,
        allow_nan=False,
    )

    runner.equal(
        serialized["pose"]["type"],
        "_SelfTestAtomicStructure",
        "The pose was not serialized as a compact model reference.",
    )
    runner.equal(
        serialized["ligand"]["name"],
        "LIG",
        "The ligand residue reference was not serialized correctly.",
    )


def _test_chimerax_model_manager_compatibility(
    runner: _SelfTestRunner,
) -> None:
    """Test the ChimeraX list(type=AtomicStructure) access pattern."""

    module_type = type(sys)
    chimerax_module = module_type(
        "chimerax"
    )
    atomic_module = module_type(
        "chimerax.atomic"
    )

    class AtomicStructure(
        _SelfTestAtomicStructure
    ):
        pass

    atomic_module.AtomicStructure = (
        AtomicStructure
    )
    chimerax_module.atomic = atomic_module

    previous_chimerax = sys.modules.get(
        "chimerax"
    )
    previous_atomic = sys.modules.get(
        "chimerax.atomic"
    )

    class TypedModelManager:
        def __init__(
            self,
            models: Iterable[Any],
        ) -> None:
            self._models = list(models)

        def list(
            self,
            *,
            type: type,
        ) -> List[Any]:
            return [
                model
                for model in self._models
                if isinstance(
                    model,
                    type,
                )
            ]

    try:
        sys.modules[
            "chimerax"
        ] = chimerax_module
        sys.modules[
            "chimerax.atomic"
        ] = atomic_module

        atomic_structure = AtomicStructure(
            "typed_pose",
            "3",
            [
                "LIG",
            ],
            atoms_per_residue=3,
        )
        session = type(
            "TypedSession",
            (),
            {
                "models": TypedModelManager(
                    [
                        object(),
                        atomic_structure,
                    ]
                ),
            },
        )()

        atomic_models = get_atomic_models(
            session
        )

        runner.equal(
            atomic_models,
            [
                atomic_structure,
            ],
            "The ChimeraX typed model-manager pattern was not supported.",
        )

    finally:
        if previous_chimerax is None:
            sys.modules.pop(
                "chimerax",
                None,
            )
        else:
            sys.modules[
                "chimerax"
            ] = previous_chimerax

        if previous_atomic is None:
            sys.modules.pop(
                "chimerax.atomic",
                None,
            )
        else:
            sys.modules[
                "chimerax.atomic"
            ] = previous_atomic


# -----------------------------------------------------------------------------
# Section 6 — Geometry tests
# -----------------------------------------------------------------------------

def _test_geometry(
    runner: _SelfTestRunner,
) -> None:
    """
    Test vector and geometric utilities.
    """

    vector = normalize(
        [
            3.0,
            0.0,
            0.0,
        ]
    )

    runner.check(
        np.allclose(
            vector,
            np.array(
                [
                    1.0,
                    0.0,
                    0.0,
                ]
            ),
        ),
        "normalize() returned an "
        "unexpected vector.",
    )

    calculated_distance = distance(
        [
            0.0,
            0.0,
            0.0,
        ],
        [
            3.0,
            4.0,
            0.0,
        ],
    )

    runner.close(
        calculated_distance,
        5.0,
        message=(
            "distance() returned an "
            "unexpected value."
        ),
    )

    calculated_centroid = centroid(
        [
            [
                0.0,
                0.0,
                0.0,
            ],
            [
                2.0,
                2.0,
                2.0,
            ],
        ]
    )

    runner.check(
        np.allclose(
            calculated_centroid,
            np.array(
                [
                    1.0,
                    1.0,
                    1.0,
                ]
            ),
        ),
        "centroid() returned an "
        "unexpected coordinate.",
    )

    calculated_angle = angle(
        [
            1.0,
            0.0,
            0.0,
        ],
        [
            0.0,
            0.0,
            0.0,
        ],
        [
            0.0,
            1.0,
            0.0,
        ],
    )

    runner.close(
        calculated_angle,
        90.0,
        tolerance=1e-7,
        message=(
            "angle() did not return "
            "90 degrees."
        ),
    )


# -----------------------------------------------------------------------------
# Section 7 — File tests
# -----------------------------------------------------------------------------

def _test_file_management(
    runner: _SelfTestRunner,
) -> None:
    """
    Test output-directory and serialization utilities.
    """

    with tempfile.TemporaryDirectory() as (
        temporary_directory
    ):
        root_path = Path(
            temporary_directory
        )

        output_paths = create_output(
            root_path
            / "dockanalyzer_output"
        )

        expected_directories = {
            "root",
            "csv",
            "json",
            "images",
            "reports",
            "sessions",
            "logs",
        }

        runner.check(
            expected_directories.issubset(
                output_paths
            ),
            "create_output() did not return "
            "all expected directories.",
        )

        for directory_name in (
            expected_directories
        ):
            runner.check(
                output_paths[
                    directory_name
                ].exists(),
                (
                    "Expected output directory "
                    f"{directory_name!r} "
                    "was not created."
                ),
            )

        filename = build_filename(
            "Pose Analysis",
            suffix="contacts",
            extension="json",
            index=1,
            lowercase=True,
        )

        runner.check(
            filename.endswith(
                ".json"
            ),
            "build_filename() did not add "
            "the expected extension.",
        )

        runner.check(
            "pose_analysis" in filename,
            "build_filename() did not "
            "sanitize the prefix.",
        )

        json_path = save_json(
            {
                "pose": "pose_01",
                "affinity": -7.5,
            },
            output_paths["json"]
            / "result.json",
        )

        runner.check(
            Path(
                json_path
            ).exists(),
            "save_json() did not create a file.",
        )

        csv_path = save_csv(
            [
                {
                    "pose": "pose_01",
                    "affinity": -7.5,
                },
                {
                    "pose": "pose_02",
                    "affinity": -7.2,
                },
            ],
            output_paths["csv"]
            / "results.csv",
        )

        runner.check(
            Path(
                csv_path
            ).exists(),
            "save_csv() did not create a file.",
        )

        loaded_frame = pd.read_csv(
            csv_path
        )

        runner.equal(
            len(
                loaded_frame
            ),
            2,
            "Saved CSV contained an "
            "unexpected number of rows.",
        )


# -----------------------------------------------------------------------------
# Section 8 — Residue tests
# -----------------------------------------------------------------------------

def _test_residue_utilities(
    runner: _SelfTestRunner,
) -> None:
    """
    Test residue identity, formatting and sorting.
    """

    structure = _SelfTestStructure(
        "receptor",
        "1",
    )

    residue_1 = _SelfTestResidue(
        "TYR",
        "A",
        58,
        structure=structure,
        atomspec="#1/A:58",
    )

    residue_2 = _SelfTestResidue(
        "GLY",
        "A",
        100,
        insertion_code="A",
        structure=structure,
        atomspec="#1/A:100A",
    )

    residue_duplicate = (
        _SelfTestResidue(
            "TYR",
            "A",
            58,
            structure=structure,
            atomspec="#1/A:58",
        )
    )

    unique_values = unique_residues(
        [
            residue_1,
            residue_2,
            residue_duplicate,
        ]
    )

    runner.equal(
        len(
            unique_values
        ),
        2,
        "unique_residues() did not "
        "remove the duplicate.",
    )

    residue_label = residue_to_string(
        residue_1
    )

    runner.equal(
        residue_label,
        "A:TYR58",
        "Unexpected residue string.",
    )

    atomspec_label = residue_to_string(
        residue_1,
        include_atomspec=True,
    )

    runner.equal(
        atomspec_label,
        "#1/A:58",
        "Unexpected residue atomspec.",
    )

    sorted_values = sort_residues(
        [
            residue_2,
            residue_1,
        ]
    )

    runner.check(
        sorted_values[0]
        is residue_1,
        "sort_residues() returned an "
        "unexpected order.",
    )


# -----------------------------------------------------------------------------
# Section 9 — Pretty-printing tests
# -----------------------------------------------------------------------------

def _test_pretty_printing(
    runner: _SelfTestRunner,
) -> None:
    """
    Test title and table formatting.
    """

    title_text = format_title(
        "DockAnalyzer",
        width=30,
        character="=",
    )

    runner.check(
        "DOCKANALYZER"
        in title_text,
        "format_title() did not include "
        "the expected title.",
    )

    table_text = format_table(
        [
            {
                "Pose": "pose_01",
                "Affinity": -7.5,
            },
            {
                "Pose": "pose_02",
                "Affinity": -7.2,
            },
        ]
    )

    runner.check(
        "pose_01" in table_text,
        "format_table() did not include "
        "the expected row.",
    )

    runner.check(
        "Affinity" in table_text,
        "format_table() did not include "
        "the expected column.",
    )


# -----------------------------------------------------------------------------
# Section 10 — Helper tests
# -----------------------------------------------------------------------------

def _test_helpers(
    runner: _SelfTestRunner,
) -> None:
    """
    Test general-purpose helper functions.
    """

    runner.equal(
        ensure_list(
            None
        ),
        [],
        "ensure_list(None) should return [].",
    )

    runner.equal(
        ensure_list(
            "pose"
        ),
        [
            "pose",
        ],
        "ensure_list() should preserve strings "
        "as one item.",
    )

    runner.equal(
        flatten(
            [
                [
                    1,
                    2,
                ],
                [
                    3,
                    [
                        4,
                    ],
                ],
            ]
        ),
        [
            1,
            2,
            3,
            4,
        ],
        "flatten() returned an "
        "unexpected result.",
    )

    runner.equal(
        list(
            chunks(
                [
                    1,
                    2,
                    3,
                    4,
                    5,
                ],
                2,
            )
        ),
        [
            [
                1,
                2,
            ],
            [
                3,
                4,
            ],
            [
                5,
            ],
        ],
        "chunks() returned unexpected blocks.",
    )

    runner.equal(
        list(
            pairwise(
                [
                    "A",
                    "B",
                    "C",
                ]
            )
        ),
        [
            (
                "A",
                "B",
            ),
            (
                "B",
                "C",
            ),
        ],
        "pairwise() returned unexpected pairs.",
    )

    runner.equal(
        first_not_none(
            None,
            None,
            5,
            10,
        ),
        5,
        "first_not_none() returned an "
        "unexpected value.",
    )

    runner.equal(
        compact(
            [
                None,
                "",
                "pose",
                0,
            ]
        ),
        [
            "",
            "pose",
            0,
        ],
        "compact() did not preserve empty strings by default.",
    )

    runner.equal(
        compact(
            [
                None,
                "",
                " ",
                "pose",
                0,
            ],
            remove_empty_strings=True,
        ),
        [
            "pose",
            0,
        ],
        "compact() did not remove empty strings when requested.",
    )


# -----------------------------------------------------------------------------
# Section 11.2 — @timer tests
# -----------------------------------------------------------------------------

def _test_timer_decorator(
    runner: _SelfTestRunner,
) -> None:
    """
    Test the synchronous ``@timer`` decorator.
    """

    logger = _SelfTestLogger()
    timer_object = _SelfTestTimer()

    @timer(
        "SELF TEST TIMER",
        timer_object=timer_object,
        logger=logger,
        log_start=True,
        log_end=True,
    )
    def calculate_sum(
        value_1: int,
        value_2: int,
    ) -> int:
        return value_1 + value_2

    result = calculate_sum(
        2,
        3,
    )

    runner.equal(
        result,
        5,
        "@timer changed the function result.",
    )

    runner.equal(
        timer_object.started_steps,
        [
            "SELF TEST TIMER",
        ],
        "@timer did not start the expected step.",
    )

    runner.equal(
        len(
            timer_object.completed_steps
        ),
        1,
        "@timer did not complete exactly one step.",
    )

    runner.equal(
        timer_object.completed_steps[
            0
        ][
            "status"
        ],
        "success",
        "@timer did not record success.",
    )

    runner.check(
        logger.contains(
            "SELF TEST TIMER"
        ),
        "@timer did not emit the expected log.",
    )

    runner.check(
        "timer"
        in _get_decorator_chain(
            calculate_sum
        ),
        "@timer metadata was not attached.",
    )


def _test_timer_decorator_failure(
    runner: _SelfTestRunner,
) -> None:
    """
    Test that ``@timer`` records and propagates failures.
    """

    timer_object = _SelfTestTimer()

    @timer(
        "FAILING TIMER",
        timer_object=timer_object,
        log_errors=False,
    )
    def fail() -> None:
        raise ValueError(
            "expected failure"
        )

    try:
        fail()

    except ValueError:
        pass

    else:
        raise AssertionError(
            "@timer suppressed an exception."
        )

    runner.equal(
        len(
            timer_object.completed_steps
        ),
        1,
        "Failed timer step was not completed.",
    )

    runner.equal(
        timer_object.completed_steps[
            0
        ][
            "status"
        ],
        "error",
        "Failed timer step did not receive "
        "error status.",
    )


# -----------------------------------------------------------------------------
# Section 11.3 — @log_call tests
# -----------------------------------------------------------------------------

def _test_log_call_decorator(
    runner: _SelfTestRunner,
) -> None:
    """
    Test argument, return and duration logging.
    """

    logger = _SelfTestLogger()

    @log_call(
        logger=logger,
        level="debug",
        log_arguments=True,
        log_return=True,
        log_duration=True,
    )
    def multiply(
        value: int,
        factor: int = 2,
    ) -> int:
        return value * factor

    result = multiply(
        4,
        factor=3,
    )

    runner.equal(
        result,
        12,
        "@log_call changed the return value.",
    )

    runner.check(
        logger.contains(
            "multiply"
        ),
        "@log_call did not record the "
        "function name.",
    )

    runner.check(
        logger.contains(
            "factor=3"
        ),
        "@log_call did not record arguments.",
    )

    runner.check(
        logger.contains(
            "return=12"
        ),
        "@log_call did not record the return value.",
    )

    runner.check(
        "log_call"
        in _get_decorator_chain(
            multiply
        ),
        "@log_call metadata was not attached.",
    )


def _test_log_call_redaction(
    runner: _SelfTestRunner,
) -> None:
    """
    Test sensitive argument redaction.
    """

    logger = _SelfTestLogger()

    @log_call(
        logger=logger,
        log_arguments=True,
    )
    def authenticate(
        username: str,
        api_key: str,
    ) -> bool:
        return bool(
            username
            and api_key
        )

    authenticate(
        "researcher",
        "secret-value",
    )

    runner.check(
        logger.contains(
            "api_key=<redacted>"
        ),
        "Sensitive argument was not redacted.",
    )

    runner.check(
        not logger.contains(
            "secret-value"
        ),
        "Sensitive value leaked into the log.",
    )


# -----------------------------------------------------------------------------
# Section 11.4 — @safe_execution tests
# -----------------------------------------------------------------------------

def _test_safe_execution_default(
    runner: _SelfTestRunner,
) -> None:
    """
    Test fallback-return behavior.
    """

    logger = _SelfTestLogger()

    @safe_execution(
        default_factory=list,
        logger=logger,
        include_traceback=False,
    )
    def fail_analysis() -> List[Any]:
        raise ValueError(
            "analysis failed"
        )

    result_1 = fail_analysis()
    result_2 = fail_analysis()

    runner.equal(
        result_1,
        [],
        "@safe_execution did not return "
        "the expected fallback.",
    )

    runner.check(
        result_1 is not result_2,
        "default_factory did not create "
        "independent values.",
    )

    runner.check(
        logger.contains(
            "analysis failed"
        ),
        "@safe_execution did not log "
        "the handled failure.",
    )

    runner.check(
        "safe_execution"
        in _get_decorator_chain(
            fail_analysis
        ),
        "@safe_execution metadata "
        "was not attached.",
    )


def _test_safe_execution_exception_filter(
    runner: _SelfTestRunner,
) -> None:
    """
    Test exception filtering and propagation.
    """

    @safe_execution(
        default="fallback",
        exceptions=ValueError,
        log_errors=False,
        log_recovery=False,
    )
    def fail_with_type(
        exception_type: Type[Exception],
    ) -> str:
        raise exception_type(
            "expected"
        )

    handled_result = fail_with_type(
        ValueError
    )

    runner.equal(
        handled_result,
        "fallback",
        "Configured ValueError was not handled.",
    )

    try:
        fail_with_type(
            TypeError
        )

    except TypeError:
        pass

    else:
        raise AssertionError(
            "Unconfigured TypeError was suppressed."
        )


def _test_safe_execution_reraise(
    runner: _SelfTestRunner,
) -> None:
    """
    Test explicit reraising.
    """

    @safe_execution(
        reraise=True,
        log_errors=False,
        log_reraise=False,
    )
    def fail() -> None:
        raise RuntimeError(
            "expected"
        )

    try:
        fail()

    except RuntimeError:
        pass

    else:
        raise AssertionError(
            "reraise=True did not propagate "
            "the original exception."
        )


def _test_safe_execution_callback(
    runner: _SelfTestRunner,
) -> None:
    """
    Test callback-generated fallback values.
    """

    def build_fallback(
        exception: BaseException,
    ) -> Dict[str, Any]:
        return {
            "success": False,
            "error": str(
                exception
            ),
        }

    @safe_execution(
        on_error=build_fallback,
        callback_result_as_fallback=True,
        log_errors=False,
        log_recovery=False,
    )
    def fail() -> Dict[str, Any]:
        raise ValueError(
            "callback failure"
        )

    result = fail()

    runner.equal(
        result[
            "success"
        ],
        False,
        "Callback result was not used "
        "as fallback.",
    )

    runner.equal(
        result[
            "error"
        ],
        "callback failure",
        "Callback did not receive the "
        "original exception.",
    )


# -----------------------------------------------------------------------------
# Combined decorator tests
# -----------------------------------------------------------------------------

def _test_decorator_chain(
    runner: _SelfTestRunner,
) -> None:
    """
    Test the recommended decorator ordering.
    """

    logger = _SelfTestLogger()
    timer_object = _SelfTestTimer()

    @safe_execution(
        default_factory=list,
        logger=logger,
        include_traceback=False,
    )
    @timer(
        "COMBINED TEST",
        timer_object=timer_object,
        logger=logger,
        log_end=False,
        log_errors=False,
    )
    @log_call(
        logger=logger,
        log_return=True,
    )
    def analyze(
        fail: bool = False,
    ) -> List[int]:
        if fail:
            raise ValueError(
                "combined failure"
            )

        return [
            1,
            2,
            3,
        ]

    successful_result = analyze()

    failed_result = analyze(
        fail=True
    )

    runner.equal(
        successful_result,
        [
            1,
            2,
            3,
        ],
        "Combined decorators changed "
        "the successful result.",
    )

    runner.equal(
        failed_result,
        [],
        "Combined decorators did not return "
        "the safe fallback.",
    )

    decorator_chain = (
        _get_decorator_chain(
            analyze
        )
    )

    runner.equal(
        decorator_chain,
        [
            "safe_execution",
            "timer",
            "log_call",
        ],
        "Unexpected decorator chain order.",
    )

    runner.equal(
        len(
            timer_object.completed_steps
        ),
        2,
        "Combined timer did not record both calls.",
    )


# -----------------------------------------------------------------------------
# Asynchronous decorator tests
# -----------------------------------------------------------------------------

async def _run_async_decorator_checks(
    runner: _SelfTestRunner,
) -> None:
    """
    Execute asynchronous decorator checks.
    """

    logger = _SelfTestLogger()
    timer_object = _SelfTestTimer()

    @safe_execution(
        default=-1,
        logger=logger,
        include_traceback=False,
    )
    @timer(
        "ASYNC TEST",
        timer_object=timer_object,
        logger=logger,
        log_end=False,
        log_errors=False,
    )
    @log_call(
        logger=logger,
        log_return=True,
    )
    async def async_double(
        value: int,
        *,
        fail: bool = False,
    ) -> int:
        await asyncio.sleep(
            0
        )

        if fail:
            raise ValueError(
                "async failure"
            )

        return value * 2

    successful_result = await async_double(
        5
    )

    failed_result = await async_double(
        5,
        fail=True,
    )

    runner.equal(
        successful_result,
        10,
        "Async decorators changed the "
        "successful result.",
    )

    runner.equal(
        failed_result,
        -1,
        "Async safe execution did not "
        "return the fallback.",
    )

    runner.equal(
        len(
            timer_object.completed_steps
        ),
        2,
        "Async timer did not record both calls.",
    )


def _test_async_decorators(
    runner: _SelfTestRunner,
) -> None:
    """
    Run asynchronous decorator tests.
    """

    try:
        asyncio.run(
            _run_async_decorator_checks(
                runner
            )
        )

    except RuntimeError as error:
        if (
            "asyncio.run() cannot be called"
            in str(error)
        ):
            raise RuntimeError(
                "The self-test was executed inside "
                "an active asyncio event loop."
            ) from error

        raise


# -----------------------------------------------------------------------------
# Self-test classification tests
# -----------------------------------------------------------------------------

def _test_self_test_classification(
    runner: _SelfTestRunner,
) -> None:
    """Test permanent self-test failure classification."""

    nested_runner = _SelfTestRunner(
        emit=False
    )

    def raise_code_failure() -> None:
        raise RuntimeError(
            "deliberate code failure"
        )

    def raise_test_failure() -> None:
        raise AssertionError(
            "deliberate test failure"
        )

    nested_runner.run(
        "Deliberate code failure",
        raise_code_failure,
        failure_category=_SELF_TEST_CODE_FAILURE,
    )
    nested_runner.run(
        "Deliberate test failure",
        raise_test_failure,
        failure_category=_SELF_TEST_TEST_FAILURE,
    )
    nested_runner.skip(
        "Deliberate environmental limitation",
        "No live ChimeraX session.",
    )

    runner.equal(
        nested_runner.failure_counts,
        {
            _SELF_TEST_CODE_FAILURE: 1,
            _SELF_TEST_TEST_FAILURE: 1,
            _SELF_TEST_ENVIRONMENTAL_LIMITATION: 1,
        },
        "Self-test failure categories were not counted correctly.",
    )
    runner.equal(
        nested_runner.unjustified_failures,
        2,
        "Unjustified self-test failures were not counted correctly.",
    )

    try:
        _normalize_self_test_failure_category(
            "unsupported"
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Invalid self-test failure categories were accepted."
        )


# -----------------------------------------------------------------------------
# Public-interface tests
# -----------------------------------------------------------------------------

def _test_public_interface(
    runner: _SelfTestRunner,
) -> None:
    """Test exported names, aliases and downstream signatures."""

    duplicate_names = sorted(
        {
            name
            for name in __all__
            if __all__.count(
                name
            ) > 1
        }
    )

    runner.check(
        not duplicate_names,
        (
            "Duplicate names in __all__: "
            f"{duplicate_names}"
        ),
    )

    missing_names = sorted(
        name
        for name in __all__
        if name not in globals()
    )

    runner.check(
        not missing_names,
        (
            "Exported names are undefined: "
            f"{missing_names}"
        ),
    )

    public_definitions = {
        name
        for name, value in globals().items()
        if (
            not name.startswith(
                "_"
            )
            and (
                inspect.isfunction(
                    value
                )
                or inspect.isclass(
                    value
                )
            )
            and getattr(
                value,
                "__module__",
                None,
            ) == __name__
        )
    }

    unexported_definitions = sorted(
        public_definitions
        - set(
            __all__
        )
    )

    runner.check(
        not unexported_definitions,
        (
            "Public definitions are missing from __all__: "
            f"{unexported_definitions}"
        ),
    )

    expected_parameters = {
        "DockLogger": {
            "name",
            "session",
        },
        "DockModel": {
            "name",
            "pose",
            "receptor",
            "ligand",
        },
        "get_receptor": {
            "source",
            "receptor",
            "strict",
        },
        "get_pose_models": {
            "source",
            "receptor",
            "poses",
            "include_unknown",
            "strict",
        },
        "get_ligand": {
            "pose_model",
        },
        "ensure_output_directories": {
            "output_directory",
            "subdirectories",
            "exist_ok",
        },
        "build_output_filename": {
            "name",
            "extension",
            "prefix",
            "suffix",
            "index",
            "timestamp",
            "lowercase",
        },
    }

    for public_name, parameter_names in (
        expected_parameters.items()
    ):
        signature = inspect.signature(
            globals()[
                public_name
            ]
        )

        runner.check(
            parameter_names.issubset(
                signature.parameters
            ),
            (
                f"{public_name} has an incompatible signature: "
                f"{signature}"
            ),
        )

    dock_model_signature = inspect.signature(
        DockModel.to_dict
    )

    runner.check(
        {
            "include_pose",
            "include_ligand",
            "include_receptor",
        }.issubset(
            dock_model_signature.parameters
        ),
        (
            "DockModel.to_dict() does not preserve all public "
            "serialization options."
        ),
    )

    runner.equal(
        get_ligand(
            "pose"
        ),
        "pose",
        "get_ligand() did not delegate to get_ligand_from_pose().",
    )

    runner.equal(
        build_output_filename(
            "pose",
            "json",
            "dock",
            None,
            2,
        ),
        "dock_pose_002.json",
        "build_output_filename() returned an incompatible filename.",
    )

    with tempfile.TemporaryDirectory() as temporary_directory:
        paths = ensure_output_directories(
            temporary_directory,
            (
                "json",
            ),
        )

        runner.check(
            paths[
                "json"
            ].is_dir(),
            "ensure_output_directories() did not create the requested path.",
        )


# -----------------------------------------------------------------------------
# Main self-test coordinator
# -----------------------------------------------------------------------------

def _run_module_self_test() -> bool:
    """
    Run the complete ``utils.py`` self-test.

    Returns
    -------
    bool
        ``True`` when every executed test passes.
    """

    runner = _SelfTestRunner()

    print_title(
        "DOCKANALYZER UTILS.PY SELF-TEST",
        width=72,
    )

    print(
        "Running isolated tests without "
        "requiring an active ChimeraX session."
    )
    print()

    runner.run(
        "Logging infrastructure",
        lambda: _test_logging_utilities(
            runner
        ),
    )

    runner.run(
        "Timer infrastructure",
        lambda: _test_timer_infrastructure(
            runner
        ),
    )

    runner.run(
        "DockModel",
        lambda: _test_dock_model(
            runner
        ),
    )

    runner.run(
        "DockModel strict serialization",
        lambda: _test_strict_serialization(
            runner
        ),
    )

    runner.run(
        "DockModel input normalization",
        lambda: _test_dock_model_normalization(
            runner
        ),
    )

    runner.run(
        "Synthetic model discovery",
        lambda: _test_synthetic_model_discovery(
            runner
        ),
    )

    runner.run(
        "ChimeraX model-manager compatibility",
        lambda: _test_chimerax_model_manager_compatibility(
            runner
        ),
    )

    runner.skip(
        "Automatic model discovery",
        (
            "Requires a live ChimeraX session "
            "with atomic models."
        ),
    )

    runner.run(
        "Basic geometry",
        lambda: _test_geometry(
            runner
        ),
    )

    runner.run(
        "File management",
        lambda: _test_file_management(
            runner
        ),
    )

    runner.run(
        "Residue utilities",
        lambda: _test_residue_utilities(
            runner
        ),
    )

    runner.run(
        "Pretty printing",
        lambda: _test_pretty_printing(
            runner
        ),
    )

    runner.run(
        "General helpers",
        lambda: _test_helpers(
            runner
        ),
    )

    runner.run(
        "@timer success",
        lambda: _test_timer_decorator(
            runner
        ),
    )

    runner.run(
        "@timer failure",
        lambda: _test_timer_decorator_failure(
            runner
        ),
    )

    runner.run(
        "@log_call",
        lambda: _test_log_call_decorator(
            runner
        ),
    )

    runner.run(
        "@log_call redaction",
        lambda: _test_log_call_redaction(
            runner
        ),
    )

    runner.run(
        "@safe_execution fallback",
        lambda: _test_safe_execution_default(
            runner
        ),
    )

    runner.run(
        "@safe_execution filtering",
        lambda: (
            _test_safe_execution_exception_filter(
                runner
            )
        ),
    )

    runner.run(
        "@safe_execution reraise",
        lambda: _test_safe_execution_reraise(
            runner
        ),
    )

    runner.run(
        "@safe_execution callback",
        lambda: _test_safe_execution_callback(
            runner
        ),
    )

    runner.run(
        "Combined decorator chain",
        lambda: _test_decorator_chain(
            runner
        ),
    )

    runner.run(
        "Asynchronous decorators",
        lambda: _test_async_decorators(
            runner
        ),
    )

    runner.run(
        "Self-test failure classification",
        lambda: _test_self_test_classification(
            runner
        ),
    )

    runner.run(
        "Public module interface",
        lambda: _test_public_interface(
            runner
        ),
    )

    runner.print_summary()

    return runner.successful


# -----------------------------------------------------------------------------
# Script entry point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    _SELF_TEST_SUCCESS = (
        _run_module_self_test()
    )

    if not _SELF_TEST_SUCCESS:
        raise SystemExit(
            1
        )

    raise SystemExit(
        0
    )


# =============================================================================
# End of Section 12
# =============================================================================
