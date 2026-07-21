"""
===============================================================================
DockAnalyzer
Utilities Module
===============================================================================

Author:
    Leonardo Bastos

Project:
    DockAnalyzer

Version:
    0.1.0

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

===============================================================================
"""

from __future__ import annotations

# =============================================================================
# Standard Library
# =============================================================================

import csv
import json
import logging
import math
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

# =============================================================================
# Third-party Libraries
# =============================================================================

import numpy as np
import pandas as pd

# =============================================================================
# DockAnalyzer Modules
# =============================================================================

import config

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
__version__ = "0.1.0"
__license__ = "MIT"
__status__ = "Development"

# =============================================================================
# Global Constants
# =============================================================================

SECONDS_PER_MINUTE = 60.0

ANGSTROM_SYMBOL = "\u212B"

DEFAULT_FLOAT_PRECISION = 3

DEFAULT_SEPARATOR = "=" * 80

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

        self._configure_handlers()

    # -------------------------------------------------------------------------
    # Logger configuration
    # -------------------------------------------------------------------------

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


@dataclass(slots=True)
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
   
    def to_dicts(
        self,
        ) -> List[Dict[str, Union[str, float]]]:
   
        """
        Convert the timing record into a dictionary.

        Returns
        -------
        dict
            Dictionary containing the step name and timing values.
        """

        return {
            "name": self.name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "elapsed_seconds": self.elapsed,
        }


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

   def to_dict(
        self,
        include_pose: bool = False,
        include_receptor: bool = False,
        include_ligand: bool = False,
    ) -> Dict[str, Any]:
        """
        Convert all timing records into dictionaries.

        Returns
        -------
        list of dict
            Timing records suitable for JSON, CSV or DataFrame conversion.
        """

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

    name: str

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
        """
        Validate and normalize the DockModel attributes.
        """

        self.name = str(self.name).strip()

        if not self.name:
            raise ValueError(
                "DockModel name cannot be empty."
            )

        if self.score is not None:
            self.score = float(self.score)

        self._normalize_pi_dictionary()
        self._normalize_file_dictionary()

    def _normalize_pi_dictionary(self) -> None:
        """
        Ensure that the pi-interaction dictionary contains standard keys.
        """

        if self.pi is None:
            self.pi = {}

        if not isinstance(self.pi, dict):
            raise TypeError(
                "DockModel.pi must be a dictionary."
            )

        self.pi.setdefault(
            "stacking",
            [],
        )

        self.pi.setdefault(
            "cation",
            [],
        )

    def _normalize_file_dictionary(self) -> None:
        """
        Ensure that the output-file dictionary contains standard keys.
        """

        if self.files is None:
            self.files = {}

        if not isinstance(self.files, dict):
            raise TypeError(
                "DockModel.files must be a dictionary."
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
            self.files.setdefault(
                file_type,
                None,
            )

        for file_type, file_path in list(
            self.files.items()
        ):
            if file_path is not None:
                self.files[file_type] = Path(
                    file_path
                )

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
            "score": self.score,
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
        """
        Convert a value into a JSON-compatible representation.

        Parameters
        ----------
        value : Any
            Value to serialize.

        Returns
        -------
        Any
            JSON-compatible value.
        """

        if value is None:
            return None

        if isinstance(
            value,
            (
                str,
                int,
                float,
                bool,
            ),
        ):
            return value

        if isinstance(value, Path):
            return str(value)

        if isinstance(value, np.ndarray):
            return value.tolist()

        if isinstance(value, dict):
            return {
                str(key): DockModel._serialize_value(
                    item
                )
                for key, item in value.items()
            }

        if isinstance(
            value,
            (
                list,
                tuple,
                set,
            ),
        ):
            return [
                DockModel._serialize_value(item)
                for item in value
            ]

        if hasattr(value, "to_dict") and callable(
            value.to_dict
        ):
            try:
                return DockModel._serialize_value(
                    value.to_dict()
                )
            except Exception:
                pass

        if hasattr(value, "__dict__"):
            try:
                return {
                    str(key): DockModel._serialize_value(
                        item
                    )
                    for key, item in vars(value).items()
                    if not str(key).startswith("_")
                }
            except Exception:
                pass

        return str(value)

    def to_dict(
        self,
        include_pose: bool = False,
        include_ligand: bool = False,
    ) -> Dict[str, Any]:
        """
        Convert the DockModel into a dictionary.

        Parameters
        ----------
        include_pose : bool, optional
            Whether the pose object should be included.
        include_ligand : bool, optional
            Whether the ligand object should be included.

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
            "score": self.score,
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

def calculate_receptor_score(
    model: Any,
    min_receptor_atoms: int = DEFAULT_MIN_RECEPTOR_ATOMS,
    min_protein_residues: int = DEFAULT_MIN_PROTEIN_RESIDUES,
) -> float:
    """
    Calculate a heuristic receptor-classification score.

    Higher scores indicate a greater probability that the model is a
    macromolecular receptor.

    Parameters
    ----------
    model : Any
        ChimeraX atomic model.
    min_receptor_atoms : int, optional
        Approximate minimum receptor size.
    min_protein_residues : int, optional
        Minimum number of protein residues expected in a receptor.

    Returns
    -------
    float
        Receptor-classification score.
    """

    if not is_atomic_model(model):
        return float("-inf")

    atom_count = get_atom_count(model)
    residue_count = get_residue_count(model)
    protein_count = count_protein_residues(
        model
    )
    nucleic_count = count_nucleic_acid_residues(
        model
    )
    polymer_count = (
        protein_count
        + nucleic_count
    )

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
        polymer_fraction = (
            polymer_count
            / residue_count
        )

        if polymer_fraction >= 0.50:
            score += 1.0

        if polymer_fraction >= 0.80:
            score += 1.0

    if atom_count <= DEFAULT_MAX_LIGAND_ATOMS:
        score -= 2.0

    if polymer_count == 0:
        score -= 3.0

    return score


def calculate_ligand_score(
    model: Any,
    min_ligand_atoms: int = DEFAULT_MIN_LIGAND_ATOMS,
    max_ligand_atoms: int = DEFAULT_MAX_LIGAND_ATOMS,
) -> float:
    """
    Calculate a heuristic ligand or pose classification score.

    Parameters
    ----------
    model : Any
        ChimeraX atomic model.
    min_ligand_atoms : int, optional
        Minimum accepted number of atoms.
    max_ligand_atoms : int, optional
        Maximum typical number of atoms in a ligand pose.

    Returns
    -------
    float
        Ligand-classification score.
    """

    if not is_atomic_model(model):
        return float("-inf")

    atom_count = get_atom_count(model)
    residue_count = get_residue_count(model)
    polymer_count = get_polymer_residue_count(
        model
    )
    solvent_count = count_solvent_residues(
        model
    )
    ion_count = count_ion_residues(
        model
    )

    score = 0.0

    if (
        min_ligand_atoms
        <= atom_count
        <= max_ligand_atoms
    ):
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

    if (
        residue_count > 0
        and solvent_count == residue_count
    ):
        score -= 5.0

    if (
        residue_count > 0
        and ion_count == residue_count
    ):
        score -= 4.0

    if atom_count > max_ligand_atoms:
        score -= 2.0

    return score


def classify_model(
    model: Any,
    min_receptor_atoms: int = DEFAULT_MIN_RECEPTOR_ATOMS,
    min_protein_residues: int = DEFAULT_MIN_PROTEIN_RESIDUES,
    max_ligand_atoms: int = DEFAULT_MAX_LIGAND_ATOMS,
) -> str:
    """
    Classify a ChimeraX model.

    Possible classifications are:

    - ``"receptor"``
    - ``"ligand"``
    - ``"solvent"``
    - ``"ion"``
    - ``"unknown"``
    - ``"non_atomic"``

    Parameters
    ----------
    model : Any
        ChimeraX model.

    Returns
    -------
    str
        Model classification.
    """

    if not is_atomic_model(model):
        return "non_atomic"

    residue_count = get_residue_count(
        model
    )
    solvent_count = count_solvent_residues(
        model
    )
    ion_count = count_ion_residues(
        model
    )

    if (
        residue_count > 0
        and solvent_count == residue_count
    ):
        return "solvent"

    if (
        residue_count > 0
        and ion_count == residue_count
    ):
        return "ion"

    receptor_score = calculate_receptor_score(
        model=model,
        min_receptor_atoms=min_receptor_atoms,
        min_protein_residues=min_protein_residues,
    )

    ligand_score = calculate_ligand_score(
        model=model,
        max_ligand_atoms=max_ligand_atoms,
    )

    if (
        receptor_score
        >= DEFAULT_RECEPTOR_SCORE_THRESHOLD
        and receptor_score > ligand_score
    ):
        return "receptor"

    if ligand_score > 0:
        return "ligand"

    return "unknown"


def describe_model(
    model: Any,
) -> Dict[str, Any]:
    """
    Generate a model-classification report.

    Parameters
    ----------
    model : Any
        ChimeraX atomic model.

    Returns
    -------
    dict
        Model properties and classification scores.
    """

    return {
        "name": get_model_name(model),
        "model_id": get_model_id(model),
        "atomspec": get_model_atomspec(model),
        "atom_count": get_atom_count(model),
        "residue_count": get_residue_count(model),
        "protein_residue_count": count_protein_residues(
            model
        ),
        "nucleic_acid_residue_count": (
            count_nucleic_acid_residues(model)
        ),
        "polymer_residue_count": (
            get_polymer_residue_count(model)
        ),
        "receptor_score": calculate_receptor_score(
            model
        ),
        "ligand_score": calculate_ligand_score(
            model
        ),
        "classification": classify_model(
            model
        ),
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
    "is_atomic_model",
    "count_protein_residues",
    "count_nucleic_acid_residues",
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
    "model_discovery_summary",
    "print_model_discovery",
]

for public_name in _SECTION_5_PUBLIC_NAMES:
    if public_name not in __all__:
        __all__.append(public_name)


# =============================================================================
# End of Section 5
# =============================================================================
