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
