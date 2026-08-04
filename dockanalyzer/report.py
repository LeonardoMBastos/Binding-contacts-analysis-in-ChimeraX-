# =============================================================================
# DockAnalyzer — Report generation
# Section 1 — Header, imports and metadata
# =============================================================================

"""Human-readable and machine-readable reports for DockAnalyzer.

This module consolidates analysis, interaction, scoring and multipose results
into structured reports rendered as text, Markdown, HTML or JSON.

The report layer does not detect interactions, recalculate scores or replace
``export.py``. It consumes existing objects, mappings and serialized data,
builds report sections and delegates broader export workflows through optional
adapters.

The implementation supports ordinary Python objects, DockAnalyzer dataclasses,
dictionary-based data, synthetic self-tests and ChimeraX objects. Specialized
modules are imported locally when required to reduce circular dependencies.
"""

from __future__ import annotations

# 1.1. Standard-library imports
# -----------------------------------------------------------------------------

from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping, MutableMapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, fields, is_dataclass, replace
from datetime import date, datetime, timezone
from enum import Enum
from hashlib import sha256
from html import escape as html_escape
from io import StringIO
from numbers import Real
from pathlib import Path
from statistics import fmean, median
from tempfile import NamedTemporaryFile
from types import MappingProxyType
from typing import (
    IO,
    Any,
    Callable,
    Dict,
    Final,
    FrozenSet,
    Hashable,
    List,
    Literal,
    NamedTuple,
    Optional,
    Protocol,
    Set,
    TextIO,
    Tuple,
    TypeAlias,
    TypeVar,
    Union,
    runtime_checkable,
)
import csv
import inspect
import json
import math
import os
import platform
import re
import sys
import traceback
import warnings

# 1.2. Optional NumPy support
# -----------------------------------------------------------------------------

try:
    import numpy as np

    NUMPY_AVAILABLE: Final[bool] = True
except ImportError:  # pragma: no cover - environment dependent
    np = None  # type: ignore[assignment]
    NUMPY_AVAILABLE = False

# 1.3. Internal DockAnalyzer imports
# -----------------------------------------------------------------------------

try:
    from . import config
    from .utils import DockLogger, DockModel
except ImportError:
    import config
    from utils import DockLogger, DockModel

# ``scoring`` and ``export`` are imported locally by integration functions.

# 1.4. Module metadata
# -----------------------------------------------------------------------------

__author__: Final[str] = "Leonardo Bastos and DockAnalyzer contributors"
__version__: Final[str] = "0.1.0"
__license__: Final[str] = "MIT"
__status__: Final[str] = "Development"

_MODULE_NAME: Final[str] = "report"
_MODULE_DESCRIPTION: Final[str] = (
    "Structured report construction and rendering for DockAnalyzer."
)
_LOGGER: Final[DockLogger] = DockLogger(_MODULE_NAME)

_RUN_IMPORT_VALIDATIONS: Final[bool] = os.getenv(
    "DOCKANALYZER_VALIDATE_IMPORTS",
    "",
).strip().lower() in {"1", "true", "yes", "on"}

# 1.5. Public-name registration
# -----------------------------------------------------------------------------

__all__: List[str] = []


def _register_public_names(names: Iterable[str]) -> None:
    """Register unique public names in declaration order."""

    known = set(__all__)
    for name in names:
        if name not in known:
            __all__.append(name)
            known.add(name)

# 1.6. Generic type variables
# -----------------------------------------------------------------------------

T = TypeVar("T")
K = TypeVar("K", bound=Hashable)
V = TypeVar("V")
ReportT = TypeVar("ReportT")
SectionT = TypeVar("SectionT")
RowT = TypeVar("RowT", bound=Mapping[str, Any])

# 1.7. Core aliases
# -----------------------------------------------------------------------------

PathLike: TypeAlias = Union[str, os.PathLike[str], Path]
Number: TypeAlias = Union[int, float, Real]

JSONPrimitive: TypeAlias = Union[str, int, float, bool, None]
JSONValue: TypeAlias = Union[
    JSONPrimitive,
    List["JSONValue"],
    Dict[str, "JSONValue"],
]
JSONMapping: TypeAlias = Dict[str, JSONValue]

Metadata: TypeAlias = Mapping[str, Any]
MutableMetadata: TypeAlias = MutableMapping[str, Any]

ReportInputLike: TypeAlias = Any
ReportLike: TypeAlias = Any
ReportSectionLike: TypeAlias = Any
ReportBlockLike: TypeAlias = Any
InteractionLike: TypeAlias = Any
ScoringLike: TypeAlias = Any
PoseLike: TypeAlias = Any
DockModelLike: TypeAlias = Union[DockModel, Any]
ChimeraXSessionLike: TypeAlias = Any

ReportRow: TypeAlias = Dict[str, Any]
ReportRows: TypeAlias = List[ReportRow]
ReportTableData: TypeAlias = Sequence[Mapping[str, Any]]
ReportTableMap: TypeAlias = Mapping[str, ReportTableData]

TextWriter: TypeAlias = Callable[[str], Any]
ValueFormatter: TypeAlias = Callable[[Any], str]
ReportRenderer: TypeAlias = Callable[[ReportLike], str]
SectionBuilder: TypeAlias = Callable[..., ReportSectionLike]

# 1.8. Shared immutable values
# -----------------------------------------------------------------------------

_EMPTY_METADATA: Final[Mapping[str, Any]] = MappingProxyType({})
_EMPTY_ROWS: Final[Tuple[Mapping[str, Any], ...]] = ()
_EMPTY_STRINGS: Final[Tuple[str, ...]] = ()
_EMPTY_OBJECTS: Final[Tuple[Any, ...]] = ()

# 1.9. Initial public interface
# -----------------------------------------------------------------------------

_SECTION_1_PUBLIC_NAMES: Final[Tuple[str, ...]] = (
    "NUMPY_AVAILABLE",
    "PathLike",
    "Number",
    "JSONPrimitive",
    "JSONValue",
    "JSONMapping",
    "Metadata",
    "MutableMetadata",
    "ReportInputLike",
    "ReportLike",
    "ReportSectionLike",
    "ReportBlockLike",
    "InteractionLike",
    "ScoringLike",
    "PoseLike",
    "DockModelLike",
    "ChimeraXSessionLike",
    "ReportRow",
    "ReportRows",
    "ReportTableData",
    "ReportTableMap",
    "TextWriter",
    "ValueFormatter",
    "ReportRenderer",
    "SectionBuilder",
)

_register_public_names(_SECTION_1_PUBLIC_NAMES)

# =============================================================================
# End of Section 1
# =============================================================================

# =============================================================================
# Section 2 — Constants
# =============================================================================

# 2.1. Schema and report identity
# -----------------------------------------------------------------------------

REPORT_SCHEMA_NAME: Final[str] = "dockanalyzer.report"
REPORT_SCHEMA_VERSION: Final[str] = "1.0"
REPORT_GENERATOR_NAME: Final[str] = "DockAnalyzer"
REPORT_GENERATOR_MODULE: Final[str] = _MODULE_NAME

DEFAULT_REPORT_TITLE: Final[str] = "DockAnalyzer Report"
DEFAULT_REPORT_SUBTITLE: Final[str] = "Molecular interaction analysis"
DEFAULT_REPORT_BASENAME: Final[str] = "dockanalyzer_report"
DEFAULT_REPORT_LANGUAGE: Final[str] = "en"
DEFAULT_ENCODING: Final[str] = "utf-8"
DEFAULT_NEWLINE: Final[str] = "\n"

# 2.2. Numerical and display defaults
# -----------------------------------------------------------------------------

REPORT_EPSILON: Final[float] = 1.0e-12
REPORT_COMPARISON_TOLERANCE: Final[float] = 1.0e-9

DEFAULT_TEXT_WIDTH: Final[int] = 100
MIN_TEXT_WIDTH: Final[int] = 40
MAX_TEXT_WIDTH: Final[int] = 240
DEFAULT_INDENT: Final[int] = 2

DEFAULT_FLOAT_DIGITS: Final[int] = 3
DEFAULT_SCORE_DIGITS: Final[int] = 4
DEFAULT_DISTANCE_DIGITS: Final[int] = 3
DEFAULT_ANGLE_DIGITS: Final[int] = 1
DEFAULT_PERCENT_DIGITS: Final[int] = 1

DEFAULT_MAX_ROWS: Final[int] = 1000
DEFAULT_MAX_ITEMS: Final[int] = 100
DEFAULT_MAX_CELL_LENGTH: Final[int] = 160
DEFAULT_MAX_TITLE_LENGTH: Final[int] = 120
DEFAULT_MAX_WARNING_LENGTH: Final[int] = 500
DEFAULT_TOP_RESIDUES: Final[int] = 20
DEFAULT_TOP_HOTSPOTS: Final[int] = 10
DEFAULT_TOP_POSES: Final[int] = 20

DEFAULT_EMPTY_TEXT: Final[str] = ""
DEFAULT_MISSING_TEXT: Final[str] = "N/A"
DEFAULT_UNKNOWN_TEXT: Final[str] = "Unknown"
DEFAULT_NOT_APPLICABLE_TEXT: Final[str] = "N/A"
DEFAULT_TRUNCATION_MARKER: Final[str] = "…"

# 2.3. Supported report formats
# -----------------------------------------------------------------------------

REPORT_FORMAT_TEXT: Final[str] = "text"
REPORT_FORMAT_MARKDOWN: Final[str] = "markdown"
REPORT_FORMAT_HTML: Final[str] = "html"
REPORT_FORMAT_JSON: Final[str] = "json"

DEFAULT_REPORT_FORMAT: Final[str] = REPORT_FORMAT_TEXT
SUPPORTED_REPORT_FORMATS: Final[FrozenSet[str]] = frozenset(
    {
        REPORT_FORMAT_TEXT,
        REPORT_FORMAT_MARKDOWN,
        REPORT_FORMAT_HTML,
        REPORT_FORMAT_JSON,
    }
)

REPORT_FORMAT_ALIASES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "txt": REPORT_FORMAT_TEXT,
        "plain": REPORT_FORMAT_TEXT,
        "plaintext": REPORT_FORMAT_TEXT,
        "text": REPORT_FORMAT_TEXT,
        "md": REPORT_FORMAT_MARKDOWN,
        "markdown": REPORT_FORMAT_MARKDOWN,
        "mkd": REPORT_FORMAT_MARKDOWN,
        "htm": REPORT_FORMAT_HTML,
        "html": REPORT_FORMAT_HTML,
        "json": REPORT_FORMAT_JSON,
    }
)

REPORT_FILE_SUFFIXES: Final[Mapping[str, str]] = MappingProxyType(
    {
        REPORT_FORMAT_TEXT: ".txt",
        REPORT_FORMAT_MARKDOWN: ".md",
        REPORT_FORMAT_HTML: ".html",
        REPORT_FORMAT_JSON: ".json",
    }
)

REPORT_MIME_TYPES: Final[Mapping[str, str]] = MappingProxyType(
    {
        REPORT_FORMAT_TEXT: "text/plain",
        REPORT_FORMAT_MARKDOWN: "text/markdown",
        REPORT_FORMAT_HTML: "text/html",
        REPORT_FORMAT_JSON: "application/json",
    }
)

# 2.4. Report detail levels
# -----------------------------------------------------------------------------

REPORT_DETAIL_MINIMAL: Final[str] = "minimal"
REPORT_DETAIL_STANDARD: Final[str] = "standard"
REPORT_DETAIL_DETAILED: Final[str] = "detailed"
REPORT_DETAIL_FULL: Final[str] = "full"

DEFAULT_REPORT_DETAIL: Final[str] = REPORT_DETAIL_STANDARD
SUPPORTED_REPORT_DETAILS: Final[FrozenSet[str]] = frozenset(
    {
        REPORT_DETAIL_MINIMAL,
        REPORT_DETAIL_STANDARD,
        REPORT_DETAIL_DETAILED,
        REPORT_DETAIL_FULL,
    }
)

REPORT_DETAIL_ALIASES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "brief": REPORT_DETAIL_MINIMAL,
        "minimal": REPORT_DETAIL_MINIMAL,
        "short": REPORT_DETAIL_MINIMAL,
        "default": REPORT_DETAIL_STANDARD,
        "normal": REPORT_DETAIL_STANDARD,
        "standard": REPORT_DETAIL_STANDARD,
        "detailed": REPORT_DETAIL_DETAILED,
        "extended": REPORT_DETAIL_DETAILED,
        "complete": REPORT_DETAIL_FULL,
        "full": REPORT_DETAIL_FULL,
        "verbose": REPORT_DETAIL_FULL,
    }
)

# 2.5. Standard section identifiers
# -----------------------------------------------------------------------------

SECTION_OVERVIEW: Final[str] = "overview"
SECTION_INPUTS: Final[str] = "inputs"
SECTION_INTERACTIONS: Final[str] = "interactions"
SECTION_RESIDUES: Final[str] = "residues"
SECTION_HOTSPOTS: Final[str] = "hotspots"
SECTION_SCORING: Final[str] = "scoring"
SECTION_MULTIPOSE: Final[str] = "multipose"
SECTION_PROVENANCE: Final[str] = "provenance"
SECTION_WARNINGS: Final[str] = "warnings"
SECTION_ERRORS: Final[str] = "errors"

STANDARD_SECTION_IDS: Final[Tuple[str, ...]] = (
    SECTION_OVERVIEW,
    SECTION_INPUTS,
    SECTION_INTERACTIONS,
    SECTION_RESIDUES,
    SECTION_HOTSPOTS,
    SECTION_SCORING,
    SECTION_MULTIPOSE,
    SECTION_PROVENANCE,
    SECTION_WARNINGS,
    SECTION_ERRORS,
)

DEFAULT_SECTION_ORDER: Final[Tuple[str, ...]] = STANDARD_SECTION_IDS

SECTION_TITLES: Final[Mapping[str, str]] = MappingProxyType(
    {
        SECTION_OVERVIEW: "Overview",
        SECTION_INPUTS: "Inputs",
        SECTION_INTERACTIONS: "Interactions",
        SECTION_RESIDUES: "Residue Summary",
        SECTION_HOTSPOTS: "Hotspots",
        SECTION_SCORING: "Scoring and Explainability",
        SECTION_MULTIPOSE: "Multipose Ranking",
        SECTION_PROVENANCE: "Provenance",
        SECTION_WARNINGS: "Warnings",
        SECTION_ERRORS: "Errors",
    }
)

SECTION_DESCRIPTIONS: Final[Mapping[str, str]] = MappingProxyType(
    {
        SECTION_OVERVIEW: "General pose and analysis summary.",
        SECTION_INPUTS: "Input models, files and analysis settings.",
        SECTION_INTERACTIONS: "Normalized molecular interactions.",
        SECTION_RESIDUES: "Interaction and score totals by residue.",
        SECTION_HOTSPOTS: "Residues with recurrent or high-value interactions.",
        SECTION_SCORING: "Score totals, components and explanations.",
        SECTION_MULTIPOSE: "Pose comparison, ranking and consensus.",
        SECTION_PROVENANCE: "Generation environment and source metadata.",
        SECTION_WARNINGS: "Non-fatal issues recorded during report creation.",
        SECTION_ERRORS: "Errors retained in permissive mode.",
    }
)

DEFAULT_ENABLED_SECTIONS: Final[FrozenSet[str]] = frozenset(
    {
        SECTION_OVERVIEW,
        SECTION_INPUTS,
        SECTION_INTERACTIONS,
        SECTION_RESIDUES,
        SECTION_HOTSPOTS,
        SECTION_SCORING,
        SECTION_MULTIPOSE,
        SECTION_PROVENANCE,
    }
)

# 2.6. Report block and table kinds
# -----------------------------------------------------------------------------

BLOCK_PARAGRAPH: Final[str] = "paragraph"
BLOCK_KEY_VALUE: Final[str] = "key_value"
BLOCK_TABLE: Final[str] = "table"
BLOCK_LIST: Final[str] = "list"
BLOCK_CODE: Final[str] = "code"
BLOCK_NOTICE: Final[str] = "notice"
BLOCK_SEPARATOR: Final[str] = "separator"

STANDARD_BLOCK_KINDS: Final[FrozenSet[str]] = frozenset(
    {
        BLOCK_PARAGRAPH,
        BLOCK_KEY_VALUE,
        BLOCK_TABLE,
        BLOCK_LIST,
        BLOCK_CODE,
        BLOCK_NOTICE,
        BLOCK_SEPARATOR,
    }
)

TABLE_OVERVIEW: Final[str] = "overview"
TABLE_INPUTS: Final[str] = "inputs"
TABLE_INTERACTIONS: Final[str] = "interactions"
TABLE_CONTACTS: Final[str] = "contacts"
TABLE_HBONDS: Final[str] = "hydrogen_bonds"
TABLE_HYDROPHOBIC: Final[str] = "hydrophobic"
TABLE_PI: Final[str] = "pi_interactions"
TABLE_SALT_BRIDGES: Final[str] = "salt_bridges"
TABLE_CLASHES: Final[str] = "clashes"
TABLE_RESIDUES: Final[str] = "residues"
TABLE_HOTSPOTS: Final[str] = "hotspots"
TABLE_SCORES: Final[str] = "scores"
TABLE_SCORE_COMPONENTS: Final[str] = "score_components"
TABLE_EXPLAINABILITY: Final[str] = "explainability"
TABLE_POSES: Final[str] = "poses"
TABLE_RANKING: Final[str] = "ranking"
TABLE_CONSENSUS: Final[str] = "consensus"
TABLE_PERSISTENCE: Final[str] = "persistence"
TABLE_PROVENANCE: Final[str] = "provenance"
TABLE_WARNINGS: Final[str] = "warnings"
TABLE_ERRORS: Final[str] = "errors"

STANDARD_TABLE_NAMES: Final[Tuple[str, ...]] = (
    TABLE_OVERVIEW,
    TABLE_INPUTS,
    TABLE_INTERACTIONS,
    TABLE_CONTACTS,
    TABLE_HBONDS,
    TABLE_HYDROPHOBIC,
    TABLE_PI,
    TABLE_SALT_BRIDGES,
    TABLE_CLASHES,
    TABLE_RESIDUES,
    TABLE_HOTSPOTS,
    TABLE_SCORES,
    TABLE_SCORE_COMPONENTS,
    TABLE_EXPLAINABILITY,
    TABLE_POSES,
    TABLE_RANKING,
    TABLE_CONSENSUS,
    TABLE_PERSISTENCE,
    TABLE_PROVENANCE,
    TABLE_WARNINGS,
    TABLE_ERRORS,
)

# 2.7. Canonical interaction families
# -----------------------------------------------------------------------------

INTERACTION_CONTACT: Final[str] = "contact"
INTERACTION_HBOND: Final[str] = "hydrogen_bond"
INTERACTION_HYDROPHOBIC: Final[str] = "hydrophobic"
INTERACTION_PI: Final[str] = "pi"
INTERACTION_SALT_BRIDGE: Final[str] = "salt_bridge"
INTERACTION_CLASH: Final[str] = "clash"
INTERACTION_UNKNOWN: Final[str] = "unknown"

INTERACTION_FAMILIES: Final[FrozenSet[str]] = frozenset(
    {
        INTERACTION_CONTACT,
        INTERACTION_HBOND,
        INTERACTION_HYDROPHOBIC,
        INTERACTION_PI,
        INTERACTION_SALT_BRIDGE,
        INTERACTION_CLASH,
        INTERACTION_UNKNOWN,
    }
)

FAVORABLE_INTERACTION_FAMILIES: Final[FrozenSet[str]] = frozenset(
    {
        INTERACTION_CONTACT,
        INTERACTION_HBOND,
        INTERACTION_HYDROPHOBIC,
        INTERACTION_PI,
        INTERACTION_SALT_BRIDGE,
    }
)

PENALTY_INTERACTION_FAMILIES: Final[FrozenSet[str]] = frozenset(
    {
        INTERACTION_CLASH,
    }
)

INTERACTION_FAMILY_ALIASES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "contact": INTERACTION_CONTACT,
        "contacts": INTERACTION_CONTACT,
        "atomic_contact": INTERACTION_CONTACT,
        "atomic_contacts": INTERACTION_CONTACT,
        "hbond": INTERACTION_HBOND,
        "hbonds": INTERACTION_HBOND,
        "hydrogen_bond": INTERACTION_HBOND,
        "hydrogen_bonds": INTERACTION_HBOND,
        "hydrophobic": INTERACTION_HYDROPHOBIC,
        "hydrophobics": INTERACTION_HYDROPHOBIC,
        "hydrophobic_interaction": INTERACTION_HYDROPHOBIC,
        "hydrophobic_interactions": INTERACTION_HYDROPHOBIC,
        "pi": INTERACTION_PI,
        "pi_interaction": INTERACTION_PI,
        "pi_interactions": INTERACTION_PI,
        "stacking": INTERACTION_PI,
        "saltbridge": INTERACTION_SALT_BRIDGE,
        "saltbridges": INTERACTION_SALT_BRIDGE,
        "salt_bridge": INTERACTION_SALT_BRIDGE,
        "salt_bridges": INTERACTION_SALT_BRIDGE,
        "ionic": INTERACTION_SALT_BRIDGE,
        "ionic_interaction": INTERACTION_SALT_BRIDGE,
        "clash": INTERACTION_CLASH,
        "clashes": INTERACTION_CLASH,
        "steric_clash": INTERACTION_CLASH,
        "steric_clashes": INTERACTION_CLASH,
        "unknown": INTERACTION_UNKNOWN,
    }
)

INTERACTION_FAMILY_LABELS: Final[Mapping[str, str]] = MappingProxyType(
    {
        INTERACTION_CONTACT: "Contacts",
        INTERACTION_HBOND: "Hydrogen bonds",
        INTERACTION_HYDROPHOBIC: "Hydrophobic interactions",
        INTERACTION_PI: "Pi interactions",
        INTERACTION_SALT_BRIDGE: "Salt bridges",
        INTERACTION_CLASH: "Steric clashes",
        INTERACTION_UNKNOWN: "Unknown interactions",
    }
)

# 2.8. Interaction containers on DockModel-like objects
# -----------------------------------------------------------------------------

INTERACTION_CONTAINER_ATTRIBUTES: Final[Mapping[str, Tuple[str, ...]]] = (
    MappingProxyType(
        {
            INTERACTION_CONTACT: (
                "contacts",
                "contact",
                "atomic_contacts",
            ),
            INTERACTION_HBOND: (
                "hbonds",
                "hbond",
                "hydrogen_bonds",
            ),
            INTERACTION_HYDROPHOBIC: (
                "hydrophobic",
                "hydrophobics",
                "hydrophobic_interactions",
            ),
            INTERACTION_PI: (
                "pi",
                "pi_interactions",
                "aromatic_interactions",
            ),
            INTERACTION_SALT_BRIDGE: (
                "saltbridge",
                "saltbridges",
                "salt_bridge",
                "salt_bridges",
            ),
            INTERACTION_CLASH: (
                "clashes",
                "clash",
                "steric_clashes",
            ),
        }
    )
)

ALL_INTERACTION_CONTAINER_ATTRIBUTES: Final[Tuple[str, ...]] = tuple(
    dict.fromkeys(
        attribute
        for attributes in INTERACTION_CONTAINER_ATTRIBUTES.values()
        for attribute in attributes
    )
)

# 2.9. Generic object field aliases
# -----------------------------------------------------------------------------

FIELD_ALIASES: Final[Mapping[str, Tuple[str, ...]]] = MappingProxyType(
    {
        "id": ("id", "identifier", "uid", "key"),
        "name": ("name", "title", "label"),
        "model": ("model", "structure", "atomic_structure"),
        "model_id": ("model_id", "structure_id", "model_identifier"),
        "model_name": ("model_name", "structure_name", "model_label"),
        "pose": ("pose", "dock_pose", "conformation"),
        "pose_id": ("pose_id", "pose_index", "pose_number", "rank"),
        "pose_name": ("pose_name", "pose_label", "name"),
        "ligand": ("ligand", "ligand_model", "small_molecule"),
        "receptor": ("receptor", "target", "protein"),
        "source": ("source", "module", "detector", "origin"),
        "interaction_type": (
            "interaction_type",
            "type",
            "kind",
            "family",
            "category",
        ),
        "interaction_subtype": (
            "interaction_subtype",
            "subtype",
            "classification",
            "geometry_class",
        ),
        "distance": (
            "distance",
            "distance_angstrom",
            "distance_a",
            "separation",
        ),
        "angle": ("angle", "angle_degrees", "theta"),
        "strength": ("strength", "interaction_strength", "quality"),
        "score": ("score", "total_score", "interaction_score"),
        "raw_score": ("raw_score", "score_raw", "unnormalized_score"),
        "normalized_score": (
            "normalized_score",
            "score_normalized",
            "norm_score",
        ),
        "affinity": (
            "affinity",
            "binding_affinity",
            "docking_affinity",
            "vina_score",
            "docking_score",
        ),
        "rank": ("rank", "ranking", "position"),
        "residue": ("residue", "receptor_residue", "protein_residue"),
        "residue_id": (
            "residue_id",
            "residue_identifier",
            "residue_key",
        ),
        "residue_name": ("residue_name", "resname", "residue_type"),
        "residue_number": (
            "residue_number",
            "residue_id_number",
            "resnum",
            "number",
        ),
        "chain_id": ("chain_id", "chain", "chain_identifier"),
        "atom": ("atom", "receptor_atom", "protein_atom"),
        "atom_id": ("atom_id", "atom_identifier", "serial_number"),
        "atom_name": ("atom_name", "name"),
        "ligand_atom": ("ligand_atom", "atom1", "first_atom"),
        "receptor_atom": ("receptor_atom", "atom2", "second_atom"),
        "ligand_residue": (
            "ligand_residue",
            "residue1",
            "first_residue",
        ),
        "receptor_residue": (
            "receptor_residue",
            "residue2",
            "second_residue",
        ),
        "metadata": ("metadata", "meta", "details", "extra"),
        "statistics": ("statistics", "stats", "summary"),
        "scoring": ("scoring", "score_result", "scoring_result"),
        "warnings": ("warnings", "warning_messages"),
        "errors": ("errors", "error_messages"),
    }
)

# 2.10. Normalized interaction keys
# -----------------------------------------------------------------------------

KEY_ID: Final[str] = "id"
KEY_SOURCE: Final[str] = "source"
KEY_FAMILY: Final[str] = "family"
KEY_TYPE: Final[str] = "type"
KEY_SUBTYPE: Final[str] = "subtype"
KEY_POSE_ID: Final[str] = "pose_id"
KEY_MODEL_ID: Final[str] = "model_id"
KEY_LIGAND_ATOM: Final[str] = "ligand_atom"
KEY_RECEPTOR_ATOM: Final[str] = "receptor_atom"
KEY_LIGAND_RESIDUE: Final[str] = "ligand_residue"
KEY_RECEPTOR_RESIDUE: Final[str] = "receptor_residue"
KEY_CHAIN_ID: Final[str] = "chain_id"
KEY_DISTANCE: Final[str] = "distance"
KEY_ANGLE: Final[str] = "angle"
KEY_STRENGTH: Final[str] = "strength"
KEY_CLASSIFICATION: Final[str] = "classification"
KEY_SCORE: Final[str] = "score"
KEY_RAW_SCORE: Final[str] = "raw_score"
KEY_NORMALIZED_SCORE: Final[str] = "normalized_score"
KEY_AFFINITY: Final[str] = "affinity"
KEY_RANK: Final[str] = "rank"
KEY_COUNT: Final[str] = "count"
KEY_PERCENT: Final[str] = "percent"
KEY_METADATA: Final[str] = "metadata"

NORMALIZED_INTERACTION_FIELDS: Final[Tuple[str, ...]] = (
    KEY_ID,
    KEY_SOURCE,
    KEY_FAMILY,
    KEY_TYPE,
    KEY_SUBTYPE,
    KEY_POSE_ID,
    KEY_MODEL_ID,
    KEY_LIGAND_ATOM,
    KEY_RECEPTOR_ATOM,
    KEY_LIGAND_RESIDUE,
    KEY_RECEPTOR_RESIDUE,
    KEY_CHAIN_ID,
    KEY_DISTANCE,
    KEY_ANGLE,
    KEY_STRENGTH,
    KEY_CLASSIFICATION,
    KEY_SCORE,
    KEY_METADATA,
)

# 2.11. Overview, scoring and provenance keys
# -----------------------------------------------------------------------------

KEY_SCHEMA_NAME: Final[str] = "schema_name"
KEY_SCHEMA_VERSION: Final[str] = "schema_version"
KEY_GENERATOR: Final[str] = "generator"
KEY_GENERATOR_VERSION: Final[str] = "generator_version"
KEY_GENERATED_AT: Final[str] = "generated_at"
KEY_TITLE: Final[str] = "title"
KEY_SUBTITLE: Final[str] = "subtitle"
KEY_DESCRIPTION: Final[str] = "description"
KEY_SECTIONS: Final[str] = "sections"
KEY_TABLES: Final[str] = "tables"
KEY_INPUTS: Final[str] = "inputs"
KEY_INTERACTIONS: Final[str] = "interactions"
KEY_RESIDUES: Final[str] = "residues"
KEY_HOTSPOTS: Final[str] = "hotspots"
KEY_SCORING: Final[str] = "scoring"
KEY_MULTIPOSE: Final[str] = "multipose"
KEY_PROVENANCE: Final[str] = "provenance"
KEY_WARNINGS: Final[str] = "warnings"
KEY_ERRORS: Final[str] = "errors"

KEY_TOTAL_INTERACTIONS: Final[str] = "total_interactions"
KEY_TOTAL_RESIDUES: Final[str] = "total_residues"
KEY_TOTAL_POSES: Final[str] = "total_poses"
KEY_TOTAL_SCORE: Final[str] = "total_score"
KEY_SCORE_COMPONENTS: Final[str] = "score_components"
KEY_EXPLAINABILITY: Final[str] = "explainability"
KEY_RANKING: Final[str] = "ranking"
KEY_CONSENSUS: Final[str] = "consensus"
KEY_PERSISTENCE: Final[str] = "persistence"

KEY_PLATFORM: Final[str] = "platform"
KEY_PYTHON_VERSION: Final[str] = "python_version"
KEY_CHIMERAX_VERSION: Final[str] = "chimerax_version"
KEY_NUMPY_VERSION: Final[str] = "numpy_version"
KEY_SOURCE_FILES: Final[str] = "source_files"
KEY_PARAMETERS: Final[str] = "parameters"
KEY_TIMESTAMP: Final[str] = "timestamp"
KEY_CHECKSUM: Final[str] = "checksum"

REQUIRED_REPORT_KEYS: Final[Tuple[str, ...]] = (
    KEY_SCHEMA_NAME,
    KEY_SCHEMA_VERSION,
    KEY_GENERATED_AT,
    KEY_SECTIONS,
)

# 2.12. Column labels and preferred table orders
# -----------------------------------------------------------------------------

COLUMN_LABELS: Final[Mapping[str, str]] = MappingProxyType(
    {
        KEY_ID: "ID",
        KEY_SOURCE: "Source",
        KEY_FAMILY: "Family",
        KEY_TYPE: "Type",
        KEY_SUBTYPE: "Subtype",
        KEY_POSE_ID: "Pose",
        KEY_MODEL_ID: "Model",
        KEY_LIGAND_ATOM: "Ligand atom",
        KEY_RECEPTOR_ATOM: "Receptor atom",
        KEY_LIGAND_RESIDUE: "Ligand residue",
        KEY_RECEPTOR_RESIDUE: "Receptor residue",
        KEY_CHAIN_ID: "Chain",
        KEY_DISTANCE: "Distance (Å)",
        KEY_ANGLE: "Angle (°)",
        KEY_STRENGTH: "Strength",
        KEY_CLASSIFICATION: "Classification",
        KEY_SCORE: "Score",
        KEY_RAW_SCORE: "Raw score",
        KEY_NORMALIZED_SCORE: "Normalized score",
        KEY_AFFINITY: "Docking affinity",
        KEY_RANK: "Rank",
        KEY_COUNT: "Count",
        KEY_PERCENT: "Percent",
        KEY_TOTAL_INTERACTIONS: "Total interactions",
        KEY_TOTAL_RESIDUES: "Total residues",
        KEY_TOTAL_POSES: "Total poses",
        KEY_TOTAL_SCORE: "Total score",
        KEY_GENERATED_AT: "Generated at",
        KEY_GENERATOR: "Generator",
        KEY_GENERATOR_VERSION: "Generator version",
        KEY_PLATFORM: "Platform",
        KEY_PYTHON_VERSION: "Python version",
        KEY_CHIMERAX_VERSION: "ChimeraX version",
        KEY_NUMPY_VERSION: "NumPy version",
    }
)

INTERACTION_TABLE_COLUMNS: Final[Tuple[str, ...]] = (
    KEY_POSE_ID,
    KEY_FAMILY,
    KEY_TYPE,
    KEY_SUBTYPE,
    KEY_LIGAND_ATOM,
    KEY_RECEPTOR_ATOM,
    KEY_RECEPTOR_RESIDUE,
    KEY_CHAIN_ID,
    KEY_DISTANCE,
    KEY_ANGLE,
    KEY_STRENGTH,
    KEY_SCORE,
)

RESIDUE_TABLE_COLUMNS: Final[Tuple[str, ...]] = (
    KEY_RANK,
    KEY_RECEPTOR_RESIDUE,
    KEY_CHAIN_ID,
    KEY_COUNT,
    KEY_TOTAL_SCORE,
    KEY_PERCENT,
)

HOTSPOT_TABLE_COLUMNS: Final[Tuple[str, ...]] = (
    KEY_RANK,
    KEY_RECEPTOR_RESIDUE,
    KEY_CHAIN_ID,
    KEY_COUNT,
    KEY_TOTAL_SCORE,
    KEY_PERSISTENCE,
)

RANKING_TABLE_COLUMNS: Final[Tuple[str, ...]] = (
    KEY_RANK,
    KEY_POSE_ID,
    KEY_TOTAL_SCORE,
    KEY_NORMALIZED_SCORE,
    KEY_AFFINITY,
    KEY_TOTAL_INTERACTIONS,
    KEY_TOTAL_RESIDUES,
)

# 2.13. Strength, quality and ranking conventions
# -----------------------------------------------------------------------------

STRENGTH_STRONG: Final[str] = "strong"
STRENGTH_MODERATE: Final[str] = "moderate"
STRENGTH_WEAK: Final[str] = "weak"
STRENGTH_UNCLASSIFIED: Final[str] = "unclassified"
STRENGTH_UNKNOWN: Final[str] = "unknown"

STRENGTH_ORDER: Final[Mapping[str, int]] = MappingProxyType(
    {
        STRENGTH_STRONG: 4,
        STRENGTH_MODERATE: 3,
        STRENGTH_WEAK: 2,
        STRENGTH_UNCLASSIFIED: 1,
        STRENGTH_UNKNOWN: 0,
    }
)

QUALITY_OPTIMAL: Final[str] = "optimal"
QUALITY_FAVORABLE: Final[str] = "favorable"
QUALITY_BORDERLINE: Final[str] = "borderline"
QUALITY_REJECTED: Final[str] = "rejected"
QUALITY_UNKNOWN: Final[str] = "unknown"

QUALITY_ORDER: Final[Mapping[str, int]] = MappingProxyType(
    {
        QUALITY_OPTIMAL: 4,
        QUALITY_FAVORABLE: 3,
        QUALITY_BORDERLINE: 2,
        QUALITY_REJECTED: 1,
        QUALITY_UNKNOWN: 0,
    }
)

RANK_DIRECTION_HIGHER_IS_BETTER: Final[str] = "higher_is_better"
RANK_DIRECTION_LOWER_IS_BETTER: Final[str] = "lower_is_better"
DEFAULT_SCORE_DIRECTION: Final[str] = RANK_DIRECTION_HIGHER_IS_BETTER
DEFAULT_AFFINITY_DIRECTION: Final[str] = RANK_DIRECTION_LOWER_IS_BETTER

# 2.14. Text rendering conventions
# -----------------------------------------------------------------------------

TEXT_HEADING_CHARACTERS: Final[Tuple[str, ...]] = ("=", "-", "~", "^")
TEXT_BULLET: Final[str] = "-"
TEXT_ORDERED_LIST_SUFFIX: Final[str] = "."
TEXT_KEY_VALUE_SEPARATOR: Final[str] = ": "
TEXT_COLUMN_SEPARATOR: Final[str] = "  "
TEXT_SECTION_SPACING: Final[int] = 2
TEXT_TABLE_BORDER: Final[str] = "-"
TEXT_WARNING_PREFIX: Final[str] = "Warning:"
TEXT_ERROR_PREFIX: Final[str] = "Error:"

# 2.15. Markdown rendering conventions
# -----------------------------------------------------------------------------

MARKDOWN_HEADING_PREFIXES: Final[Tuple[str, ...]] = (
    "#",
    "##",
    "###",
    "####",
    "#####",
    "######",
)
MARKDOWN_BULLET: Final[str] = "-"
MARKDOWN_TABLE_SEPARATOR: Final[str] = "---"
MARKDOWN_CODE_FENCE: Final[str] = "```"
MARKDOWN_LINE_BREAK: Final[str] = "  \n"
MARKDOWN_ESCAPE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"([\\`*_{}\[\]<>#+.!|~-])"
)

# 2.16. HTML rendering conventions
# -----------------------------------------------------------------------------

HTML_DOCUMENT_TYPE: Final[str] = "<!DOCTYPE html>"
HTML_DEFAULT_LANG: Final[str] = "en"
HTML_CHARSET: Final[str] = "utf-8"
HTML_VIEWPORT: Final[str] = "width=device-width, initial-scale=1"
HTML_TABLE_CLASS: Final[str] = "dockanalyzer-table"
HTML_SECTION_CLASS: Final[str] = "dockanalyzer-section"
HTML_NOTICE_CLASS: Final[str] = "dockanalyzer-notice"
HTML_WARNING_CLASS: Final[str] = "dockanalyzer-warning"
HTML_ERROR_CLASS: Final[str] = "dockanalyzer-error"

DEFAULT_HTML_CSS: Final[str] = """
:root {
  color-scheme: light dark;
  font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
body {
  margin: 2rem auto;
  max-width: 1100px;
  padding: 0 1rem;
  line-height: 1.5;
}
h1, h2, h3, h4 { line-height: 1.2; }
table {
  border-collapse: collapse;
  display: block;
  max-width: 100%;
  overflow-x: auto;
  width: max-content;
}
th, td {
  border: 1px solid #8886;
  padding: 0.35rem 0.55rem;
  text-align: left;
  vertical-align: top;
}
th { font-weight: 600; }
code, pre { font-family: ui-monospace, SFMono-Regular, Consolas, monospace; }
pre { overflow-x: auto; padding: 0.75rem; }
.dockanalyzer-notice { border-left: 0.25rem solid #888; padding-left: 0.75rem; }
.dockanalyzer-warning { border-left-color: #b7791f; }
.dockanalyzer-error { border-left-color: #c53030; }
""".strip()

# 2.17. JSON conventions
# -----------------------------------------------------------------------------

DEFAULT_JSON_INDENT: Final[int] = 2
DEFAULT_JSON_ENSURE_ASCII: Final[bool] = False
DEFAULT_JSON_SORT_KEYS: Final[bool] = False
DEFAULT_JSON_ALLOW_NAN: Final[bool] = False
DEFAULT_JSON_COMPACT_SEPARATORS: Final[Tuple[str, str]] = (",", ":")
DEFAULT_JSON_PRETTY_SEPARATORS: Final[Tuple[str, str]] = (",", ": ")

JSON_TYPE_KEY: Final[str] = "__type__"
JSON_VALUE_KEY: Final[str] = "value"
JSON_ITEMS_KEY: Final[str] = "items"

# 2.18. Safe writing and file naming
# -----------------------------------------------------------------------------

WRITE_MODE_TEXT: Final[str] = "w"
WRITE_MODE_EXCLUSIVE: Final[str] = "x"
DEFAULT_OVERWRITE: Final[bool] = False
DEFAULT_ATOMIC_WRITE: Final[bool] = True
DEFAULT_CREATE_PARENTS: Final[bool] = True
DEFAULT_TEMP_SUFFIX: Final[str] = ".tmp"
DEFAULT_BACKUP_SUFFIX: Final[str] = ".bak"

INVALID_FILENAME_PATTERN: Final[re.Pattern[str]] = re.compile(
    r'[<>:"/\\|?*\x00-\x1F]'
)
WHITESPACE_PATTERN: Final[re.Pattern[str]] = re.compile(r"\s+")
MULTIPLE_UNDERSCORES_PATTERN: Final[re.Pattern[str]] = re.compile(r"_+")
TRAILING_DOT_SPACE_PATTERN: Final[re.Pattern[str]] = re.compile(r"[. ]+$")

WINDOWS_RESERVED_NAMES: Final[FrozenSet[str]] = frozenset(
    {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{index}" for index in range(1, 10)),
        *(f"LPT{index}" for index in range(1, 10)),
    }
)

# 2.19. Provenance and source-module conventions
# -----------------------------------------------------------------------------

SOURCE_MODULE_CONTACTS: Final[str] = "contacts"
SOURCE_MODULE_HBONDS: Final[str] = "hbonds"
SOURCE_MODULE_HYDROPHOBIC: Final[str] = "hydrophobic"
SOURCE_MODULE_PI: Final[str] = "pi"
SOURCE_MODULE_SALTBRIDGE: Final[str] = "saltbridge"
SOURCE_MODULE_SCORING: Final[str] = "scoring"
SOURCE_MODULE_EXPORT: Final[str] = "export"
SOURCE_MODULE_REPORT: Final[str] = "report"

STANDARD_SOURCE_MODULES: Final[Tuple[str, ...]] = (
    SOURCE_MODULE_CONTACTS,
    SOURCE_MODULE_HBONDS,
    SOURCE_MODULE_HYDROPHOBIC,
    SOURCE_MODULE_PI,
    SOURCE_MODULE_SALTBRIDGE,
    SOURCE_MODULE_SCORING,
    SOURCE_MODULE_EXPORT,
    SOURCE_MODULE_REPORT,
)

CHECKSUM_ALGORITHM: Final[str] = "sha256"
ISO_DATETIME_TIMESPEC: Final[str] = "seconds"
UTC_SUFFIX: Final[str] = "Z"

# 2.20. Error and permissive-mode conventions
# -----------------------------------------------------------------------------

ERROR_MODE_RAISE: Final[str] = "raise"
ERROR_MODE_WARN: Final[str] = "warn"
ERROR_MODE_COLLECT: Final[str] = "collect"
ERROR_MODE_IGNORE: Final[str] = "ignore"

DEFAULT_ERROR_MODE: Final[str] = ERROR_MODE_RAISE
SUPPORTED_ERROR_MODES: Final[FrozenSet[str]] = frozenset(
    {
        ERROR_MODE_RAISE,
        ERROR_MODE_WARN,
        ERROR_MODE_COLLECT,
        ERROR_MODE_IGNORE,
    }
)

SEVERITY_INFO: Final[str] = "info"
SEVERITY_WARNING: Final[str] = "warning"
SEVERITY_ERROR: Final[str] = "error"
SEVERITY_CRITICAL: Final[str] = "critical"

SEVERITY_ORDER: Final[Mapping[str, int]] = MappingProxyType(
    {
        SEVERITY_INFO: 10,
        SEVERITY_WARNING: 20,
        SEVERITY_ERROR: 30,
        SEVERITY_CRITICAL: 40,
    }
)

# 2.21. Public constants
# -----------------------------------------------------------------------------

_SECTION_2_PUBLIC_NAMES: Final[Tuple[str, ...]] = (
    "REPORT_SCHEMA_NAME",
    "REPORT_SCHEMA_VERSION",
    "REPORT_GENERATOR_NAME",
    "REPORT_GENERATOR_MODULE",
    "DEFAULT_REPORT_TITLE",
    "DEFAULT_REPORT_SUBTITLE",
    "DEFAULT_REPORT_BASENAME",
    "DEFAULT_REPORT_LANGUAGE",
    "DEFAULT_ENCODING",
    "DEFAULT_NEWLINE",
    "REPORT_EPSILON",
    "REPORT_COMPARISON_TOLERANCE",
    "DEFAULT_TEXT_WIDTH",
    "MIN_TEXT_WIDTH",
    "MAX_TEXT_WIDTH",
    "DEFAULT_INDENT",
    "DEFAULT_FLOAT_DIGITS",
    "DEFAULT_SCORE_DIGITS",
    "DEFAULT_DISTANCE_DIGITS",
    "DEFAULT_ANGLE_DIGITS",
    "DEFAULT_PERCENT_DIGITS",
    "DEFAULT_MAX_ROWS",
    "DEFAULT_MAX_ITEMS",
    "DEFAULT_MAX_CELL_LENGTH",
    "DEFAULT_MAX_TITLE_LENGTH",
    "DEFAULT_MAX_WARNING_LENGTH",
    "DEFAULT_TOP_RESIDUES",
    "DEFAULT_TOP_HOTSPOTS",
    "DEFAULT_TOP_POSES",
    "DEFAULT_EMPTY_TEXT",
    "DEFAULT_MISSING_TEXT",
    "DEFAULT_UNKNOWN_TEXT",
    "DEFAULT_NOT_APPLICABLE_TEXT",
    "DEFAULT_TRUNCATION_MARKER",
    "REPORT_FORMAT_TEXT",
    "REPORT_FORMAT_MARKDOWN",
    "REPORT_FORMAT_HTML",
    "REPORT_FORMAT_JSON",
    "DEFAULT_REPORT_FORMAT",
    "SUPPORTED_REPORT_FORMATS",
    "REPORT_FORMAT_ALIASES",
    "REPORT_FILE_SUFFIXES",
    "REPORT_MIME_TYPES",
    "REPORT_DETAIL_MINIMAL",
    "REPORT_DETAIL_STANDARD",
    "REPORT_DETAIL_DETAILED",
    "REPORT_DETAIL_FULL",
    "DEFAULT_REPORT_DETAIL",
    "SUPPORTED_REPORT_DETAILS",
    "REPORT_DETAIL_ALIASES",
    "SECTION_OVERVIEW",
    "SECTION_INPUTS",
    "SECTION_INTERACTIONS",
    "SECTION_RESIDUES",
    "SECTION_HOTSPOTS",
    "SECTION_SCORING",
    "SECTION_MULTIPOSE",
    "SECTION_PROVENANCE",
    "SECTION_WARNINGS",
    "SECTION_ERRORS",
    "STANDARD_SECTION_IDS",
    "DEFAULT_SECTION_ORDER",
    "SECTION_TITLES",
    "SECTION_DESCRIPTIONS",
    "DEFAULT_ENABLED_SECTIONS",
    "BLOCK_PARAGRAPH",
    "BLOCK_KEY_VALUE",
    "BLOCK_TABLE",
    "BLOCK_LIST",
    "BLOCK_CODE",
    "BLOCK_NOTICE",
    "BLOCK_SEPARATOR",
    "STANDARD_BLOCK_KINDS",
    "TABLE_OVERVIEW",
    "TABLE_INPUTS",
    "TABLE_INTERACTIONS",
    "TABLE_CONTACTS",
    "TABLE_HBONDS",
    "TABLE_HYDROPHOBIC",
    "TABLE_PI",
    "TABLE_SALT_BRIDGES",
    "TABLE_CLASHES",
    "TABLE_RESIDUES",
    "TABLE_HOTSPOTS",
    "TABLE_SCORES",
    "TABLE_SCORE_COMPONENTS",
    "TABLE_EXPLAINABILITY",
    "TABLE_POSES",
    "TABLE_RANKING",
    "TABLE_CONSENSUS",
    "TABLE_PERSISTENCE",
    "TABLE_PROVENANCE",
    "TABLE_WARNINGS",
    "TABLE_ERRORS",
    "STANDARD_TABLE_NAMES",
    "INTERACTION_CONTACT",
    "INTERACTION_HBOND",
    "INTERACTION_HYDROPHOBIC",
    "INTERACTION_PI",
    "INTERACTION_SALT_BRIDGE",
    "INTERACTION_CLASH",
    "INTERACTION_UNKNOWN",
    "INTERACTION_FAMILIES",
    "FAVORABLE_INTERACTION_FAMILIES",
    "PENALTY_INTERACTION_FAMILIES",
    "INTERACTION_FAMILY_ALIASES",
    "INTERACTION_FAMILY_LABELS",
    "INTERACTION_CONTAINER_ATTRIBUTES",
    "ALL_INTERACTION_CONTAINER_ATTRIBUTES",
    "FIELD_ALIASES",
    "KEY_ID",
    "KEY_SOURCE",
    "KEY_FAMILY",
    "KEY_TYPE",
    "KEY_SUBTYPE",
    "KEY_POSE_ID",
    "KEY_MODEL_ID",
    "KEY_LIGAND_ATOM",
    "KEY_RECEPTOR_ATOM",
    "KEY_LIGAND_RESIDUE",
    "KEY_RECEPTOR_RESIDUE",
    "KEY_CHAIN_ID",
    "KEY_DISTANCE",
    "KEY_ANGLE",
    "KEY_STRENGTH",
    "KEY_CLASSIFICATION",
    "KEY_SCORE",
    "KEY_RAW_SCORE",
    "KEY_NORMALIZED_SCORE",
    "KEY_AFFINITY",
    "KEY_RANK",
    "KEY_COUNT",
    "KEY_PERCENT",
    "KEY_METADATA",
    "NORMALIZED_INTERACTION_FIELDS",
    "KEY_SCHEMA_NAME",
    "KEY_SCHEMA_VERSION",
    "KEY_GENERATOR",
    "KEY_GENERATOR_VERSION",
    "KEY_GENERATED_AT",
    "KEY_TITLE",
    "KEY_SUBTITLE",
    "KEY_DESCRIPTION",
    "KEY_SECTIONS",
    "KEY_TABLES",
    "KEY_INPUTS",
    "KEY_INTERACTIONS",
    "KEY_RESIDUES",
    "KEY_HOTSPOTS",
    "KEY_SCORING",
    "KEY_MULTIPOSE",
    "KEY_PROVENANCE",
    "KEY_WARNINGS",
    "KEY_ERRORS",
    "KEY_TOTAL_INTERACTIONS",
    "KEY_TOTAL_RESIDUES",
    "KEY_TOTAL_POSES",
    "KEY_TOTAL_SCORE",
    "KEY_SCORE_COMPONENTS",
    "KEY_EXPLAINABILITY",
    "KEY_RANKING",
    "KEY_CONSENSUS",
    "KEY_PERSISTENCE",
    "KEY_PLATFORM",
    "KEY_PYTHON_VERSION",
    "KEY_CHIMERAX_VERSION",
    "KEY_NUMPY_VERSION",
    "KEY_SOURCE_FILES",
    "KEY_PARAMETERS",
    "KEY_TIMESTAMP",
    "KEY_CHECKSUM",
    "REQUIRED_REPORT_KEYS",
    "COLUMN_LABELS",
    "INTERACTION_TABLE_COLUMNS",
    "RESIDUE_TABLE_COLUMNS",
    "HOTSPOT_TABLE_COLUMNS",
    "RANKING_TABLE_COLUMNS",
    "STRENGTH_STRONG",
    "STRENGTH_MODERATE",
    "STRENGTH_WEAK",
    "STRENGTH_UNCLASSIFIED",
    "STRENGTH_UNKNOWN",
    "STRENGTH_ORDER",
    "QUALITY_OPTIMAL",
    "QUALITY_FAVORABLE",
    "QUALITY_BORDERLINE",
    "QUALITY_REJECTED",
    "QUALITY_UNKNOWN",
    "QUALITY_ORDER",
    "RANK_DIRECTION_HIGHER_IS_BETTER",
    "RANK_DIRECTION_LOWER_IS_BETTER",
    "DEFAULT_SCORE_DIRECTION",
    "DEFAULT_AFFINITY_DIRECTION",
    "TEXT_HEADING_CHARACTERS",
    "TEXT_BULLET",
    "TEXT_ORDERED_LIST_SUFFIX",
    "TEXT_KEY_VALUE_SEPARATOR",
    "TEXT_COLUMN_SEPARATOR",
    "TEXT_SECTION_SPACING",
    "TEXT_TABLE_BORDER",
    "TEXT_WARNING_PREFIX",
    "TEXT_ERROR_PREFIX",
    "MARKDOWN_HEADING_PREFIXES",
    "MARKDOWN_BULLET",
    "MARKDOWN_TABLE_SEPARATOR",
    "MARKDOWN_CODE_FENCE",
    "MARKDOWN_LINE_BREAK",
    "MARKDOWN_ESCAPE_PATTERN",
    "HTML_DOCUMENT_TYPE",
    "HTML_DEFAULT_LANG",
    "HTML_CHARSET",
    "HTML_VIEWPORT",
    "HTML_TABLE_CLASS",
    "HTML_SECTION_CLASS",
    "HTML_NOTICE_CLASS",
    "HTML_WARNING_CLASS",
    "HTML_ERROR_CLASS",
    "DEFAULT_HTML_CSS",
    "DEFAULT_JSON_INDENT",
    "DEFAULT_JSON_ENSURE_ASCII",
    "DEFAULT_JSON_SORT_KEYS",
    "DEFAULT_JSON_ALLOW_NAN",
    "DEFAULT_JSON_COMPACT_SEPARATORS",
    "DEFAULT_JSON_PRETTY_SEPARATORS",
    "JSON_TYPE_KEY",
    "JSON_VALUE_KEY",
    "JSON_ITEMS_KEY",
    "WRITE_MODE_TEXT",
    "WRITE_MODE_EXCLUSIVE",
    "DEFAULT_OVERWRITE",
    "DEFAULT_ATOMIC_WRITE",
    "DEFAULT_CREATE_PARENTS",
    "DEFAULT_TEMP_SUFFIX",
    "DEFAULT_BACKUP_SUFFIX",
    "INVALID_FILENAME_PATTERN",
    "WHITESPACE_PATTERN",
    "MULTIPLE_UNDERSCORES_PATTERN",
    "TRAILING_DOT_SPACE_PATTERN",
    "WINDOWS_RESERVED_NAMES",
    "SOURCE_MODULE_CONTACTS",
    "SOURCE_MODULE_HBONDS",
    "SOURCE_MODULE_HYDROPHOBIC",
    "SOURCE_MODULE_PI",
    "SOURCE_MODULE_SALTBRIDGE",
    "SOURCE_MODULE_SCORING",
    "SOURCE_MODULE_EXPORT",
    "SOURCE_MODULE_REPORT",
    "STANDARD_SOURCE_MODULES",
    "CHECKSUM_ALGORITHM",
    "ISO_DATETIME_TIMESPEC",
    "UTC_SUFFIX",
    "ERROR_MODE_RAISE",
    "ERROR_MODE_WARN",
    "ERROR_MODE_COLLECT",
    "ERROR_MODE_IGNORE",
    "DEFAULT_ERROR_MODE",
    "SUPPORTED_ERROR_MODES",
    "SEVERITY_INFO",
    "SEVERITY_WARNING",
    "SEVERITY_ERROR",
    "SEVERITY_CRITICAL",
    "SEVERITY_ORDER",
)

_register_public_names(_SECTION_2_PUBLIC_NAMES)

# =============================================================================
# End of Section 2
# =============================================================================

# =============================================================================
# Section 3 — Exceptions
# =============================================================================

# 3.1. Internal exception helpers
# -----------------------------------------------------------------------------

def _normalize_error_message(message: Any, fallback: str) -> str:
    """Return a concise non-empty error message."""

    text = str(message).strip() if message is not None else ""
    return text or fallback


def _normalize_error_path(path: Optional[PathLike]) -> Optional[str]:
    """Convert a path-like value without requiring its existence."""

    if path is None:
        return None
    try:
        return os.fspath(path)
    except TypeError:
        return str(path)


def _normalize_error_context(
    context: Optional[Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Copy exception context into an immutable mapping."""

    if not context:
        return _EMPTY_METADATA
    return MappingProxyType({str(key): value for key, value in context.items()})


def _exception_name(error: BaseException) -> str:
    """Return a stable exception class name."""

    return type(error).__name__


def _exception_message(error: BaseException) -> str:
    """Return an exception message with a class-name fallback."""

    return _normalize_error_message(str(error), _exception_name(error))


# 3.2. Base report exception
# -----------------------------------------------------------------------------

class ReportError(Exception):
    """Base exception for report generation."""

    default_code = "report_error"
    default_message = "Report operation failed."

    def __init__(
        self,
        message: Any = None,
        *,
        code: Optional[str] = None,
        section: Optional[str] = None,
        report_format: Optional[str] = None,
        path: Optional[PathLike] = None,
        context: Optional[Mapping[str, Any]] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        self.message = _normalize_error_message(message, self.default_message)
        self.code = str(code or self.default_code)
        self.section = str(section) if section is not None else None
        self.report_format = (
            str(report_format) if report_format is not None else None
        )
        self.path = _normalize_error_path(path)
        self.context = _normalize_error_context(context)
        self.cause = cause
        super().__init__(self.message)

    def __str__(self) -> str:
        details: List[str] = []
        if self.section:
            details.append(f"section={self.section}")
        if self.report_format:
            details.append(f"format={self.report_format}")
        if self.path:
            details.append(f"path={self.path}")
        suffix = f" ({', '.join(details)})" if details else ""
        return f"{self.message}{suffix}"

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"message={self.message!r}, code={self.code!r}, "
            f"section={self.section!r}, "
            f"report_format={self.report_format!r}, path={self.path!r})"
        )

    def with_context(self, **values: Any) -> "ReportError":
        """Add diagnostic context and return this exception."""

        merged = dict(self.context)
        merged.update(values)
        self.context = MappingProxyType(merged)
        return self

    def to_dict(
        self,
        *,
        include_context: bool = True,
        include_cause: bool = True,
    ) -> Dict[str, Any]:
        """Return a serializable-oriented error record."""

        record: Dict[str, Any] = {
            "type": type(self).__name__,
            "code": self.code,
            "message": self.message,
        }
        if self.section is not None:
            record["section"] = self.section
        if self.report_format is not None:
            record["format"] = self.report_format
        if self.path is not None:
            record["path"] = self.path
        if include_context and self.context:
            record["context"] = dict(self.context)
        if include_cause and self.cause is not None:
            record["cause"] = {
                "type": _exception_name(self.cause),
                "message": _exception_message(self.cause),
            }
        return record

    @classmethod
    def from_exception(
        cls,
        error: BaseException,
        *,
        message: Any = None,
        code: Optional[str] = None,
        section: Optional[str] = None,
        report_format: Optional[str] = None,
        path: Optional[PathLike] = None,
        context: Optional[Mapping[str, Any]] = None,
    ) -> "ReportError":
        """Wrap an arbitrary exception while preserving its cause."""

        if isinstance(error, cls) and message is None:
            return error
        return cls(
            message or _exception_message(error),
            code=code,
            section=section,
            report_format=report_format,
            path=path,
            context=context,
            cause=error,
        )


# 3.3. Configuration and input exceptions
# -----------------------------------------------------------------------------

class ReportConfigurationError(ReportError):
    """Invalid report configuration."""

    default_code = "configuration_error"
    default_message = "Invalid report configuration."


class ReportInputError(ReportError):
    """Invalid or unsupported report input."""

    default_code = "input_error"
    default_message = "Invalid report input."


class ReportAccessError(ReportInputError):
    """Required object data could not be accessed."""

    default_code = "access_error"
    default_message = "Unable to access report input data."


class MissingReportFieldError(ReportAccessError):
    """A required field is absent."""

    default_code = "missing_field"

    def __init__(
        self,
        field_name: str,
        *,
        location: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        context = dict(kwargs.pop("context", {}) or {})
        context["field"] = field_name
        if location is not None:
            context["location"] = location
        message = f"Required report field is missing: {field_name}."
        super().__init__(message, context=context, **kwargs)
        self.field_name = field_name
        self.location = location


class ReportDependencyError(ReportConfigurationError):
    """An optional dependency required by an operation is unavailable."""

    default_code = "dependency_error"

    def __init__(
        self,
        dependency: str,
        *,
        purpose: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        context = dict(kwargs.pop("context", {}) or {})
        context["dependency"] = dependency
        if purpose is not None:
            context["purpose"] = purpose
        message = f"Required dependency is unavailable: {dependency}."
        if purpose:
            message = f"{message[:-1]} for {purpose}."
        super().__init__(message, context=context, **kwargs)
        self.dependency = dependency
        self.purpose = purpose


# 3.4. Format and normalization exceptions
# -----------------------------------------------------------------------------

class ReportFormatError(ReportConfigurationError):
    """Invalid report format or detail level."""

    default_code = "format_error"
    default_message = "Invalid report format."


class UnsupportedReportFormatError(ReportFormatError):
    """Requested report format is unsupported."""

    default_code = "unsupported_format"

    def __init__(
        self,
        report_format: Any,
        *,
        supported: Iterable[str] = SUPPORTED_REPORT_FORMATS,
        **kwargs: Any,
    ) -> None:
        supported_values = tuple(sorted(str(value) for value in supported))
        context = dict(kwargs.pop("context", {}) or {})
        context.update(
            {
                "requested_format": report_format,
                "supported_formats": supported_values,
            }
        )
        message = f"Unsupported report format: {report_format!r}."
        super().__init__(
            message,
            report_format=str(report_format),
            context=context,
            **kwargs,
        )
        self.requested_format = report_format
        self.supported_formats = supported_values


class UnsupportedReportDetailError(ReportFormatError):
    """Requested report detail level is unsupported."""

    default_code = "unsupported_detail"

    def __init__(
        self,
        detail: Any,
        *,
        supported: Iterable[str] = SUPPORTED_REPORT_DETAILS,
        **kwargs: Any,
    ) -> None:
        supported_values = tuple(sorted(str(value) for value in supported))
        context = dict(kwargs.pop("context", {}) or {})
        context.update(
            {
                "requested_detail": detail,
                "supported_details": supported_values,
            }
        )
        super().__init__(
            f"Unsupported report detail level: {detail!r}.",
            context=context,
            **kwargs,
        )
        self.requested_detail = detail
        self.supported_details = supported_values


class ReportNormalizationError(ReportInputError):
    """Input data could not be normalized."""

    default_code = "normalization_error"
    default_message = "Unable to normalize report data."


class InteractionNormalizationError(ReportNormalizationError):
    """An interaction record could not be normalized."""

    default_code = "interaction_normalization_error"
    default_message = "Unable to normalize interaction data."


# 3.5. Construction and table exceptions
# -----------------------------------------------------------------------------

class ReportBuildError(ReportError):
    """A report or report component could not be built."""

    default_code = "build_error"
    default_message = "Unable to build report."


class ReportSectionError(ReportBuildError):
    """A report section could not be built."""

    default_code = "section_error"
    default_message = "Unable to build report section."


class ReportTableError(ReportBuildError):
    """A report table could not be built."""

    default_code = "table_error"
    default_message = "Unable to build report table."


class ReportScoringError(ReportBuildError):
    """Scoring data could not be included in a report."""

    default_code = "scoring_error"
    default_message = "Unable to process scoring data."


class ReportMultiposeError(ReportBuildError):
    """Multipose data could not be summarized or ranked."""

    default_code = "multipose_error"
    default_message = "Unable to process multipose data."


class ReportProvenanceError(ReportBuildError):
    """Provenance data could not be collected."""

    default_code = "provenance_error"
    default_message = "Unable to collect report provenance."


# 3.6. Rendering and serialization exceptions
# -----------------------------------------------------------------------------

class ReportRenderError(ReportError):
    """Base exception for report rendering."""

    default_code = "render_error"
    default_message = "Unable to render report."


class TextRenderError(ReportRenderError):
    """Plain-text rendering failed."""

    default_code = "text_render_error"
    default_message = "Unable to render plain-text report."


class MarkdownRenderError(ReportRenderError):
    """Markdown rendering failed."""

    default_code = "markdown_render_error"
    default_message = "Unable to render Markdown report."


class HTMLRenderError(ReportRenderError):
    """HTML rendering failed."""

    default_code = "html_render_error"
    default_message = "Unable to render HTML report."


class JSONRenderError(ReportRenderError):
    """JSON rendering failed."""

    default_code = "json_render_error"
    default_message = "Unable to render JSON report."


class ReportSerializationError(ReportRenderError):
    """A report value could not be converted safely."""

    default_code = "serialization_error"
    default_message = "Unable to serialize report data."


# 3.7. Writing and export exceptions
# -----------------------------------------------------------------------------

class ReportWriteError(ReportError):
    """Report output could not be written."""

    default_code = "write_error"
    default_message = "Unable to write report output."


class ReportPathError(ReportWriteError):
    """An output path is invalid or unusable."""

    default_code = "path_error"
    default_message = "Invalid report output path."


class ReportOverwriteError(ReportWriteError):
    """Writing would overwrite an existing file."""

    default_code = "overwrite_error"

    def __init__(self, path: PathLike, **kwargs: Any) -> None:
        super().__init__(
            "Report output already exists and overwrite is disabled.",
            path=path,
            **kwargs,
        )


class ReportExportError(ReportError):
    """Integration with the export layer failed."""

    default_code = "export_error"
    default_message = "Unable to export report data."


class ReportIntegrationError(ReportError):
    """Integration with another DockAnalyzer component failed."""

    default_code = "integration_error"
    default_message = "Report integration failed."


# 3.8. Validation, ChimeraX and introspection exceptions
# -----------------------------------------------------------------------------

class ReportValidationError(ReportError):
    """Report validation failed."""

    default_code = "validation_error"
    default_message = "Report validation failed."


class ReportSchemaError(ReportValidationError):
    """Report data does not match the expected schema."""

    default_code = "schema_error"
    default_message = "Invalid report schema."


class ChimeraXReportError(ReportIntegrationError):
    """A ChimeraX-specific report operation failed."""

    default_code = "chimerax_error"
    default_message = "ChimeraX report integration failed."


class ReportIntrospectionError(ReportError):
    """Report introspection failed."""

    default_code = "introspection_error"
    default_message = "Unable to inspect report capabilities."


class ReportSelfTestError(ReportError):
    """A report self-test failed."""

    default_code = "self_test_error"
    default_message = "Report self-test failed."

    def __init__(
        self,
        test_name: Optional[str] = None,
        message: Any = None,
        **kwargs: Any,
    ) -> None:
        context = dict(kwargs.pop("context", {}) or {})
        if test_name is not None:
            context["test_name"] = test_name
        super().__init__(
            message or self.default_message,
            context=context,
            **kwargs,
        )
        self.test_name = test_name


# 3.9. Aggregated errors
# -----------------------------------------------------------------------------

class ReportAggregateError(ReportError):
    """Multiple report errors collected during permissive processing."""

    default_code = "aggregate_error"

    def __init__(
        self,
        errors: Iterable[BaseException],
        message: Any = None,
        **kwargs: Any,
    ) -> None:
        collected = tuple(errors)
        context = dict(kwargs.pop("context", {}) or {})
        context["error_count"] = len(collected)
        default_message = (
            f"{len(collected)} report error"
            f"{'' if len(collected) == 1 else 's'} collected."
        )
        super().__init__(
            message or default_message,
            context=context,
            **kwargs,
        )
        self.errors = collected

    def __len__(self) -> int:
        return len(self.errors)

    def __iter__(self) -> Iterator[BaseException]:
        return iter(self.errors)

    def to_dict(
        self,
        *,
        include_context: bool = True,
        include_cause: bool = True,
    ) -> Dict[str, Any]:
        record = super().to_dict(
            include_context=include_context,
            include_cause=include_cause,
        )
        record["errors"] = [
            error.to_dict(
                include_context=include_context,
                include_cause=include_cause,
            )
            if isinstance(error, ReportError)
            else {
                "type": _exception_name(error),
                "message": _exception_message(error),
            }
            for error in self.errors
        ]
        return record


# 3.10. Warning categories
# -----------------------------------------------------------------------------

class ReportWarning(UserWarning):
    """Base warning emitted by the report layer."""


class ReportDataWarning(ReportWarning):
    """Potentially incomplete or inconsistent report data."""


class ReportRenderWarning(ReportWarning):
    """Non-fatal rendering issue."""


class ReportWriteWarning(ReportWarning):
    """Non-fatal output-writing issue."""


class ReportIntegrationWarning(ReportWarning):
    """Non-fatal integration issue."""


class ChimeraXReportWarning(ReportIntegrationWarning):
    """Non-fatal ChimeraX integration issue."""


# 3.11. Public exception interface
# -----------------------------------------------------------------------------

_SECTION_3_PUBLIC_NAMES: Final[Tuple[str, ...]] = (
    "ReportError",
    "ReportConfigurationError",
    "ReportInputError",
    "ReportAccessError",
    "MissingReportFieldError",
    "ReportDependencyError",
    "ReportFormatError",
    "UnsupportedReportFormatError",
    "UnsupportedReportDetailError",
    "ReportNormalizationError",
    "InteractionNormalizationError",
    "ReportBuildError",
    "ReportSectionError",
    "ReportTableError",
    "ReportScoringError",
    "ReportMultiposeError",
    "ReportProvenanceError",
    "ReportRenderError",
    "TextRenderError",
    "MarkdownRenderError",
    "HTMLRenderError",
    "JSONRenderError",
    "ReportSerializationError",
    "ReportWriteError",
    "ReportPathError",
    "ReportOverwriteError",
    "ReportExportError",
    "ReportIntegrationError",
    "ReportValidationError",
    "ReportSchemaError",
    "ChimeraXReportError",
    "ReportIntrospectionError",
    "ReportSelfTestError",
    "ReportAggregateError",
    "ReportWarning",
    "ReportDataWarning",
    "ReportRenderWarning",
    "ReportWriteWarning",
    "ReportIntegrationWarning",
    "ChimeraXReportWarning",
)

_register_public_names(_SECTION_3_PUBLIC_NAMES)

# =============================================================================
# End of Section 3
# =============================================================================

# =============================================================================
# Section 4 — Enums
# =============================================================================

# 4.1. Enum base
# -----------------------------------------------------------------------------

class _StringEnum(str, Enum):
    """String enum with tolerant coercion."""

    def __str__(self) -> str:
        return self.value

    @classmethod
    def values(cls) -> Tuple[str, ...]:
        """Return values in declaration order."""

        return tuple(member.value for member in cls)

    @classmethod
    def aliases(cls) -> Mapping[str, str]:
        """Return accepted aliases."""

        return _EMPTY_METADATA

    @classmethod
    def coerce(
        cls,
        value: Any,
        *,
        default: Optional["_StringEnum"] = None,
    ) -> "_StringEnum":
        """Coerce strings and aliases into an enum member."""

        if isinstance(value, cls):
            return value
        if value is None and default is not None:
            return default
        token = str(value).strip().lower()
        normalized = token.replace("-", "_").replace(" ", "_")
        canonical = cls.aliases().get(token, cls.aliases().get(normalized, normalized))
        for member in cls:
            if canonical in {
                member.value.lower(),
                member.name.lower(),
            }:
                return member
        raise ValueError(f"Unsupported {cls.__name__}: {value!r}.")


# 4.2. Report and rendering enums
# -----------------------------------------------------------------------------

class ReportFormat(_StringEnum):
    """Output report format."""

    TEXT = REPORT_FORMAT_TEXT
    MARKDOWN = REPORT_FORMAT_MARKDOWN
    HTML = REPORT_FORMAT_HTML
    JSON = REPORT_FORMAT_JSON

    @classmethod
    def aliases(cls) -> Mapping[str, str]:
        return REPORT_FORMAT_ALIASES


class ReportDetail(_StringEnum):
    """Report detail level."""

    MINIMAL = REPORT_DETAIL_MINIMAL
    STANDARD = REPORT_DETAIL_STANDARD
    DETAILED = REPORT_DETAIL_DETAILED
    FULL = REPORT_DETAIL_FULL

    @classmethod
    def aliases(cls) -> Mapping[str, str]:
        return REPORT_DETAIL_ALIASES


class ReportSectionID(_StringEnum):
    """Standard report section identifier."""

    OVERVIEW = SECTION_OVERVIEW
    INPUTS = SECTION_INPUTS
    INTERACTIONS = SECTION_INTERACTIONS
    RESIDUES = SECTION_RESIDUES
    HOTSPOTS = SECTION_HOTSPOTS
    SCORING = SECTION_SCORING
    MULTIPOSE = SECTION_MULTIPOSE
    PROVENANCE = SECTION_PROVENANCE
    WARNINGS = SECTION_WARNINGS
    ERRORS = SECTION_ERRORS


class ReportBlockKind(_StringEnum):
    """Structured report block kind."""

    PARAGRAPH = BLOCK_PARAGRAPH
    KEY_VALUE = BLOCK_KEY_VALUE
    TABLE = BLOCK_TABLE
    LIST = BLOCK_LIST
    CODE = BLOCK_CODE
    NOTICE = BLOCK_NOTICE
    SEPARATOR = BLOCK_SEPARATOR


class ErrorMode(_StringEnum):
    """Error handling mode."""

    RAISE = ERROR_MODE_RAISE
    WARN = ERROR_MODE_WARN
    COLLECT = ERROR_MODE_COLLECT
    IGNORE = ERROR_MODE_IGNORE


class Severity(_StringEnum):
    """Diagnostic severity."""

    INFO = SEVERITY_INFO
    WARNING = SEVERITY_WARNING
    ERROR = SEVERITY_ERROR
    CRITICAL = SEVERITY_CRITICAL

    @property
    def weight(self) -> int:
        """Return sortable severity weight."""

        return SEVERITY_ORDER[self.value]


# 4.3. Interaction and scoring enums
# -----------------------------------------------------------------------------

class InteractionFamily(_StringEnum):
    """Canonical interaction family."""

    CONTACT = INTERACTION_CONTACT
    HYDROGEN_BOND = INTERACTION_HBOND
    HYDROPHOBIC = INTERACTION_HYDROPHOBIC
    PI = INTERACTION_PI
    SALT_BRIDGE = INTERACTION_SALT_BRIDGE
    CLASH = INTERACTION_CLASH
    UNKNOWN = INTERACTION_UNKNOWN

    @classmethod
    def aliases(cls) -> Mapping[str, str]:
        return INTERACTION_FAMILY_ALIASES

    @property
    def label(self) -> str:
        """Return the human-readable family label."""

        return INTERACTION_FAMILY_LABELS[self.value]

    @property
    def favorable(self) -> bool:
        """Return whether the family is favorable by default."""

        return self.value in FAVORABLE_INTERACTION_FAMILIES

    @property
    def penalty(self) -> bool:
        """Return whether the family is penalized by default."""

        return self.value in PENALTY_INTERACTION_FAMILIES


class InteractionStrength(_StringEnum):
    """Qualitative interaction strength."""

    STRONG = STRENGTH_STRONG
    MODERATE = STRENGTH_MODERATE
    WEAK = STRENGTH_WEAK
    UNCLASSIFIED = STRENGTH_UNCLASSIFIED
    UNKNOWN = STRENGTH_UNKNOWN

    @property
    def weight(self) -> int:
        return STRENGTH_ORDER[self.value]


class InteractionQuality(_StringEnum):
    """Qualitative geometric quality."""

    OPTIMAL = QUALITY_OPTIMAL
    FAVORABLE = QUALITY_FAVORABLE
    BORDERLINE = QUALITY_BORDERLINE
    REJECTED = QUALITY_REJECTED
    UNKNOWN = QUALITY_UNKNOWN

    @property
    def weight(self) -> int:
        return QUALITY_ORDER[self.value]


class RankDirection(_StringEnum):
    """Ranking direction."""

    HIGHER_IS_BETTER = RANK_DIRECTION_HIGHER_IS_BETTER
    LOWER_IS_BETTER = RANK_DIRECTION_LOWER_IS_BETTER

    @classmethod
    def aliases(cls) -> Mapping[str, str]:
        return MappingProxyType(
            {
                "ascending": RANK_DIRECTION_LOWER_IS_BETTER,
                "asc": RANK_DIRECTION_LOWER_IS_BETTER,
                "lower": RANK_DIRECTION_LOWER_IS_BETTER,
                "minimum": RANK_DIRECTION_LOWER_IS_BETTER,
                "descending": RANK_DIRECTION_HIGHER_IS_BETTER,
                "desc": RANK_DIRECTION_HIGHER_IS_BETTER,
                "higher": RANK_DIRECTION_HIGHER_IS_BETTER,
                "maximum": RANK_DIRECTION_HIGHER_IS_BETTER,
            }
        )

    @property
    def reverse(self) -> bool:
        """Return the reverse flag for ``sorted``."""

        return self is RankDirection.HIGHER_IS_BETTER


class WriteMode(_StringEnum):
    """Text-file write mode."""

    OVERWRITE = WRITE_MODE_TEXT
    EXCLUSIVE = WRITE_MODE_EXCLUSIVE

    @classmethod
    def aliases(cls) -> Mapping[str, str]:
        return MappingProxyType(
            {
                "write": WRITE_MODE_TEXT,
                "replace": WRITE_MODE_TEXT,
                "overwrite": WRITE_MODE_TEXT,
                "create": WRITE_MODE_EXCLUSIVE,
                "exclusive": WRITE_MODE_EXCLUSIVE,
            }
        )


# 4.4. Public enum interface
# -----------------------------------------------------------------------------

_SECTION_4_PUBLIC_NAMES: Final[Tuple[str, ...]] = (
    "ReportFormat",
    "ReportDetail",
    "ReportSectionID",
    "ReportBlockKind",
    "ErrorMode",
    "Severity",
    "InteractionFamily",
    "InteractionStrength",
    "InteractionQuality",
    "RankDirection",
    "WriteMode",
)

_register_public_names(_SECTION_4_PUBLIC_NAMES)

# =============================================================================
# End of Section 4
# =============================================================================


# =============================================================================
# Section 5 — Configuration dataclasses
# =============================================================================

# 5.1. Configuration helpers
# -----------------------------------------------------------------------------

def _config_int(
    value: Any,
    name: str,
    *,
    minimum: Optional[int] = None,
    maximum: Optional[int] = None,
) -> int:
    """Validate an integer configuration value."""

    if isinstance(value, bool):
        raise ReportConfigurationError(f"{name} must be an integer.")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ReportConfigurationError(
            f"{name} must be an integer.",
            context={"field": name, "value": value},
            cause=error,
        ) from error
    if minimum is not None and result < minimum:
        raise ReportConfigurationError(
            f"{name} must be at least {minimum}.",
            context={"field": name, "value": result},
        )
    if maximum is not None and result > maximum:
        raise ReportConfigurationError(
            f"{name} must not exceed {maximum}.",
            context={"field": name, "value": result},
        )
    return result


def _config_float(
    value: Any,
    name: str,
    *,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> float:
    """Validate a finite float configuration value."""

    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ReportConfigurationError(
            f"{name} must be numeric.",
            context={"field": name, "value": value},
            cause=error,
        ) from error
    if not math.isfinite(result):
        raise ReportConfigurationError(
            f"{name} must be finite.",
            context={"field": name, "value": value},
        )
    if minimum is not None and result < minimum:
        raise ReportConfigurationError(
            f"{name} must be at least {minimum}.",
            context={"field": name, "value": result},
        )
    if maximum is not None and result > maximum:
        raise ReportConfigurationError(
            f"{name} must not exceed {maximum}.",
            context={"field": name, "value": result},
        )
    return result


def _config_text(
    value: Any,
    name: str,
    *,
    allow_empty: bool = True,
) -> str:
    """Validate a text configuration value."""

    if value is None:
        value = ""
    result = str(value)
    if not allow_empty and not result.strip():
        raise ReportConfigurationError(
            f"{name} must not be empty.",
            context={"field": name},
        )
    return result


def _freeze_config_mapping(
    value: Optional[Mapping[Any, Any]],
) -> Mapping[str, Any]:
    """Copy a configuration mapping into an immutable mapping."""

    if not value:
        return _EMPTY_METADATA
    if not isinstance(value, Mapping):
        raise ReportConfigurationError("Configuration metadata must be a mapping.")
    return MappingProxyType({str(key): item for key, item in value.items()})


def _freeze_config_strings(
    values: Optional[Iterable[Any]],
    *,
    unique: bool = True,
) -> Tuple[str, ...]:
    """Normalize an iterable of strings."""

    if values is None:
        return ()
    if isinstance(values, str):
        values = (values,)
    result = tuple(str(value).strip() for value in values if str(value).strip())
    return tuple(dict.fromkeys(result)) if unique else result


def _coerce_enum(
    enum_type: Any,
    value: Any,
    field_name: str,
    *,
    default: Optional[_StringEnum] = None,
) -> _StringEnum:
    """Coerce an enum configuration value."""

    try:
        return enum_type.coerce(value, default=default)
    except ValueError as error:
        raise ReportConfigurationError(
            str(error),
            context={"field": field_name, "value": value},
            cause=error,
        ) from error


def _config_to_dict(value: Any) -> Any:
    """Convert nested configuration values into plain containers."""

    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {
            field_info.name: _config_to_dict(getattr(value, field_info.name))
            for field_info in fields(value)
        }
    if isinstance(value, Mapping):
        return {
            str(key): _config_to_dict(item)
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_config_to_dict(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


class _ConfigMixin:
    """Shared immutable configuration utilities."""

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain configuration mapping."""

        return _config_to_dict(self)

    def with_updates(self: T, **changes: Any) -> T:
        """Return a validated copy with selected fields replaced."""

        return replace(self, **changes)


# 5.2. Formatting configuration
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class FormattingConfig(_ConfigMixin):
    """Safe value-formatting options."""

    float_digits: int = DEFAULT_FLOAT_DIGITS
    score_digits: int = DEFAULT_SCORE_DIGITS
    distance_digits: int = DEFAULT_DISTANCE_DIGITS
    angle_digits: int = DEFAULT_ANGLE_DIGITS
    percent_digits: int = DEFAULT_PERCENT_DIGITS
    missing_text: str = DEFAULT_MISSING_TEXT
    unknown_text: str = DEFAULT_UNKNOWN_TEXT
    not_applicable_text: str = DEFAULT_NOT_APPLICABLE_TEXT
    truncation_marker: str = DEFAULT_TRUNCATION_MARKER
    true_text: str = "Yes"
    false_text: str = "No"
    strip_text: bool = True
    normalize_negative_zero: bool = True

    def __post_init__(self) -> None:
        for name in (
            "float_digits",
            "score_digits",
            "distance_digits",
            "angle_digits",
            "percent_digits",
        ):
            object.__setattr__(
                self,
                name,
                _config_int(getattr(self, name), name, minimum=0, maximum=12),
            )
        for name in (
            "missing_text",
            "unknown_text",
            "not_applicable_text",
            "truncation_marker",
            "true_text",
            "false_text",
        ):
            object.__setattr__(
                self,
                name,
                _config_text(getattr(self, name), name),
            )


# 5.3. Table configuration
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class TableConfig(_ConfigMixin):
    """Internal table limits and ordering."""

    max_rows: int = DEFAULT_MAX_ROWS
    max_cell_length: int = DEFAULT_MAX_CELL_LENGTH
    include_empty_columns: bool = False
    include_row_numbers: bool = False
    sort_columns: bool = False
    column_labels: Mapping[str, str] = field(
        default_factory=lambda: COLUMN_LABELS
    )
    preferred_columns: Mapping[str, Sequence[str]] = field(
        default_factory=lambda: MappingProxyType(
            {
                TABLE_INTERACTIONS: INTERACTION_TABLE_COLUMNS,
                TABLE_RESIDUES: RESIDUE_TABLE_COLUMNS,
                TABLE_HOTSPOTS: HOTSPOT_TABLE_COLUMNS,
                TABLE_RANKING: RANKING_TABLE_COLUMNS,
            }
        )
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "max_rows",
            _config_int(self.max_rows, "max_rows", minimum=1),
        )
        object.__setattr__(
            self,
            "max_cell_length",
            _config_int(self.max_cell_length, "max_cell_length", minimum=8),
        )
        labels = {
            str(key): str(value)
            for key, value in dict(self.column_labels).items()
        }
        preferred = {
            str(key): _freeze_config_strings(value)
            for key, value in dict(self.preferred_columns).items()
        }
        object.__setattr__(self, "column_labels", MappingProxyType(labels))
        object.__setattr__(
            self,
            "preferred_columns",
            MappingProxyType(preferred),
        )


# 5.4. Rendering configuration
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class RenderConfig(_ConfigMixin):
    """Report rendering options."""

    format: ReportFormat = ReportFormat.TEXT
    detail: ReportDetail = ReportDetail.STANDARD
    language: str = DEFAULT_REPORT_LANGUAGE
    width: int = DEFAULT_TEXT_WIDTH
    indent: int = DEFAULT_INDENT
    include_title: bool = True
    include_subtitle: bool = True
    include_generated_at: bool = True
    include_table_of_contents: bool = False
    include_empty_sections: bool = False
    compact: bool = False
    html_css: str = DEFAULT_HTML_CSS
    html_full_document: bool = True
    json_indent: Optional[int] = DEFAULT_JSON_INDENT
    json_sort_keys: bool = DEFAULT_JSON_SORT_KEYS
    json_ensure_ascii: bool = DEFAULT_JSON_ENSURE_ASCII
    newline: str = DEFAULT_NEWLINE

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "format",
            _coerce_enum(ReportFormat, self.format, "format"),
        )
        object.__setattr__(
            self,
            "detail",
            _coerce_enum(ReportDetail, self.detail, "detail"),
        )
        object.__setattr__(
            self,
            "language",
            _config_text(self.language, "language", allow_empty=False),
        )
        object.__setattr__(
            self,
            "width",
            _config_int(
                self.width,
                "width",
                minimum=MIN_TEXT_WIDTH,
                maximum=MAX_TEXT_WIDTH,
            ),
        )
        object.__setattr__(
            self,
            "indent",
            _config_int(self.indent, "indent", minimum=0, maximum=16),
        )
        if self.json_indent is not None:
            object.__setattr__(
                self,
                "json_indent",
                _config_int(
                    self.json_indent,
                    "json_indent",
                    minimum=0,
                    maximum=16,
                ),
            )
        object.__setattr__(self, "html_css", str(self.html_css or ""))
        newline = str(self.newline)
        if newline not in {"\n", "\r\n", "\r"}:
            raise ReportConfigurationError(
                "newline must be '\\n', '\\r\\n' or '\\r'.",
                context={"field": "newline", "value": newline},
            )


# 5.5. Interaction configuration
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class InteractionReportConfig(_ConfigMixin):
    """Interaction normalization and display options."""

    families: Tuple[InteractionFamily, ...] = field(
        default_factory=lambda: tuple(
            InteractionFamily(value)
            for value in (
                INTERACTION_CONTACT,
                INTERACTION_HBOND,
                INTERACTION_HYDROPHOBIC,
                INTERACTION_PI,
                INTERACTION_SALT_BRIDGE,
                INTERACTION_CLASH,
            )
        )
    )
    include_unknown: bool = True
    include_atoms: bool = True
    include_geometry: bool = True
    include_metadata: bool = False
    include_empty_families: bool = False
    deduplicate: bool = True
    sort_interactions: bool = True
    max_interactions: int = DEFAULT_MAX_ROWS

    def __post_init__(self) -> None:
        values: List[InteractionFamily] = []
        for value in self.families:
            family = _coerce_enum(
                InteractionFamily,
                value,
                "families",
            )
            if family not in values:
                values.append(family)
        if self.include_unknown and InteractionFamily.UNKNOWN not in values:
            values.append(InteractionFamily.UNKNOWN)
        object.__setattr__(self, "families", tuple(values))
        object.__setattr__(
            self,
            "max_interactions",
            _config_int(
                self.max_interactions,
                "max_interactions",
                minimum=1,
            ),
        )


# 5.6. Scoring and multipose configuration
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class ScoringReportConfig(_ConfigMixin):
    """Scoring and explainability options."""

    include: bool = True
    include_components: bool = True
    include_explainability: bool = True
    include_external_affinity: bool = True
    recalculate: bool = False
    top_components: int = DEFAULT_MAX_ITEMS
    score_direction: RankDirection = RankDirection.HIGHER_IS_BETTER
    affinity_direction: RankDirection = RankDirection.LOWER_IS_BETTER

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "top_components",
            _config_int(self.top_components, "top_components", minimum=1),
        )
        object.__setattr__(
            self,
            "score_direction",
            _coerce_enum(
                RankDirection,
                self.score_direction,
                "score_direction",
            ),
        )
        object.__setattr__(
            self,
            "affinity_direction",
            _coerce_enum(
                RankDirection,
                self.affinity_direction,
                "affinity_direction",
            ),
        )


@dataclass(frozen=True)
class MultiposeReportConfig(_ConfigMixin):
    """Multipose summary and ranking options."""

    include: bool = True
    top_poses: int = DEFAULT_TOP_POSES
    top_residues: int = DEFAULT_TOP_RESIDUES
    top_hotspots: int = DEFAULT_TOP_HOTSPOTS
    include_consensus: bool = True
    include_persistence: bool = True
    include_diversity: bool = True
    include_complementarity: bool = True
    preserve_input_order: bool = False
    rank_direction: RankDirection = RankDirection.HIGHER_IS_BETTER
    tie_tolerance: float = REPORT_COMPARISON_TOLERANCE

    def __post_init__(self) -> None:
        for name in ("top_poses", "top_residues", "top_hotspots"):
            object.__setattr__(
                self,
                name,
                _config_int(getattr(self, name), name, minimum=1),
            )
        object.__setattr__(
            self,
            "rank_direction",
            _coerce_enum(
                RankDirection,
                self.rank_direction,
                "rank_direction",
            ),
        )
        object.__setattr__(
            self,
            "tie_tolerance",
            _config_float(
                self.tie_tolerance,
                "tie_tolerance",
                minimum=0.0,
            ),
        )


# 5.7. Provenance, writing and integration configuration
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class ProvenanceConfig(_ConfigMixin):
    """Provenance collection options."""

    include: bool = True
    include_platform: bool = True
    include_python: bool = True
    include_dependencies: bool = True
    include_chimerax: bool = True
    include_source_files: bool = True
    include_parameters: bool = True
    include_checksums: bool = False
    checksum_algorithm: str = CHECKSUM_ALGORITHM
    extra: Mapping[str, Any] = field(default_factory=lambda: _EMPTY_METADATA)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "checksum_algorithm",
            _config_text(
                self.checksum_algorithm,
                "checksum_algorithm",
                allow_empty=False,
            ).lower(),
        )
        object.__setattr__(
            self,
            "extra",
            _freeze_config_mapping(self.extra),
        )


@dataclass(frozen=True)
class WriteConfig(_ConfigMixin):
    """Safe report-writing options."""

    encoding: str = DEFAULT_ENCODING
    overwrite: bool = DEFAULT_OVERWRITE
    atomic: bool = DEFAULT_ATOMIC_WRITE
    create_parents: bool = DEFAULT_CREATE_PARENTS
    backup: bool = False
    backup_suffix: str = DEFAULT_BACKUP_SUFFIX
    temp_suffix: str = DEFAULT_TEMP_SUFFIX
    newline: Optional[str] = None
    fsync: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "encoding",
            _config_text(self.encoding, "encoding", allow_empty=False),
        )
        object.__setattr__(
            self,
            "backup_suffix",
            _config_text(self.backup_suffix, "backup_suffix"),
        )
        object.__setattr__(
            self,
            "temp_suffix",
            _config_text(self.temp_suffix, "temp_suffix"),
        )
        if self.newline not in {None, "\n", "\r\n", "\r"}:
            raise ReportConfigurationError(
                "newline must be None, '\\n', '\\r\\n' or '\\r'.",
                context={"field": "newline", "value": self.newline},
            )


@dataclass(frozen=True)
class ErrorHandlingConfig(_ConfigMixin):
    """Strict and permissive error behavior."""

    mode: ErrorMode = ErrorMode.RAISE
    include_warnings: bool = True
    include_errors: bool = True
    include_tracebacks: bool = False
    max_messages: int = DEFAULT_MAX_ITEMS
    warning_category: type = ReportWarning

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "mode",
            _coerce_enum(ErrorMode, self.mode, "mode"),
        )
        object.__setattr__(
            self,
            "max_messages",
            _config_int(self.max_messages, "max_messages", minimum=1),
        )
        category = self.warning_category
        if not inspect.isclass(category) or not issubclass(category, Warning):
            raise ReportConfigurationError(
                "warning_category must be a Warning subclass.",
                context={"field": "warning_category", "value": category},
            )


@dataclass(frozen=True)
class ChimeraXConfig(_ConfigMixin):
    """Optional ChimeraX report behavior."""

    enabled: bool = True
    include_model_spec: bool = True
    include_atom_specs: bool = False
    include_selection_commands: bool = False
    include_visualization_commands: bool = False
    include_session_metadata: bool = True
    command_prefix: str = ""
    model_spec_attribute: str = "atomspec"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "command_prefix",
            _config_text(self.command_prefix, "command_prefix"),
        )
        object.__setattr__(
            self,
            "model_spec_attribute",
            _config_text(
                self.model_spec_attribute,
                "model_spec_attribute",
                allow_empty=False,
            ),
        )


# 5.8. Main report configuration
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class ReportConfig(_ConfigMixin):
    """Complete report-generation configuration."""

    title: str = DEFAULT_REPORT_TITLE
    subtitle: str = DEFAULT_REPORT_SUBTITLE
    description: str = ""
    section_order: Tuple[ReportSectionID, ...] = field(
        default_factory=lambda: tuple(
            ReportSectionID(value) for value in DEFAULT_SECTION_ORDER
        )
    )
    enabled_sections: FrozenSet[ReportSectionID] = field(
        default_factory=lambda: frozenset(
            ReportSectionID(value) for value in DEFAULT_ENABLED_SECTIONS
        )
    )
    formatting: FormattingConfig = field(default_factory=FormattingConfig)
    tables: TableConfig = field(default_factory=TableConfig)
    rendering: RenderConfig = field(default_factory=RenderConfig)
    interactions: InteractionReportConfig = field(
        default_factory=InteractionReportConfig
    )
    scoring: ScoringReportConfig = field(default_factory=ScoringReportConfig)
    multipose: MultiposeReportConfig = field(
        default_factory=MultiposeReportConfig
    )
    provenance: ProvenanceConfig = field(default_factory=ProvenanceConfig)
    writing: WriteConfig = field(default_factory=WriteConfig)
    errors: ErrorHandlingConfig = field(default_factory=ErrorHandlingConfig)
    chimerax: ChimeraXConfig = field(default_factory=ChimeraXConfig)
    metadata: Mapping[str, Any] = field(default_factory=lambda: _EMPTY_METADATA)

    def __post_init__(self) -> None:
        for name in ("title", "subtitle", "description"):
            object.__setattr__(
                self,
                name,
                _config_text(getattr(self, name), name),
            )

        order: List[ReportSectionID] = []
        for value in self.section_order:
            member = _coerce_enum(
                ReportSectionID,
                value,
                "section_order",
            )
            if member not in order:
                order.append(member)

        enabled = frozenset(
            _coerce_enum(
                ReportSectionID,
                value,
                "enabled_sections",
            )
            for value in self.enabled_sections
        )
        for member in enabled:
            if member not in order:
                order.append(member)

        object.__setattr__(self, "section_order", tuple(order))
        object.__setattr__(self, "enabled_sections", enabled)
        object.__setattr__(
            self,
            "metadata",
            _freeze_config_mapping(self.metadata),
        )

        expected_types = {
            "formatting": FormattingConfig,
            "tables": TableConfig,
            "rendering": RenderConfig,
            "interactions": InteractionReportConfig,
            "scoring": ScoringReportConfig,
            "multipose": MultiposeReportConfig,
            "provenance": ProvenanceConfig,
            "writing": WriteConfig,
            "errors": ErrorHandlingConfig,
            "chimerax": ChimeraXConfig,
        }
        for name, expected in expected_types.items():
            if not isinstance(getattr(self, name), expected):
                raise ReportConfigurationError(
                    f"{name} must be {expected.__name__}.",
                    context={"field": name},
                )

    def is_section_enabled(self, section: Any) -> bool:
        """Return whether a standard section is enabled."""

        member = _coerce_enum(
            ReportSectionID,
            section,
            "section",
        )
        return member in self.enabled_sections

    def ordered_sections(
        self,
        *,
        enabled_only: bool = True,
    ) -> Tuple[ReportSectionID, ...]:
        """Return configured sections in render order."""

        if not enabled_only:
            return self.section_order
        return tuple(
            section
            for section in self.section_order
            if section in self.enabled_sections
        )


DEFAULT_FORMATTING_CONFIG: Final[FormattingConfig] = FormattingConfig()
DEFAULT_TABLE_CONFIG: Final[TableConfig] = TableConfig()
DEFAULT_RENDER_CONFIG: Final[RenderConfig] = RenderConfig()
DEFAULT_INTERACTION_REPORT_CONFIG: Final[InteractionReportConfig] = (
    InteractionReportConfig()
)
DEFAULT_SCORING_REPORT_CONFIG: Final[ScoringReportConfig] = ScoringReportConfig()
DEFAULT_MULTIPOSE_REPORT_CONFIG: Final[MultiposeReportConfig] = (
    MultiposeReportConfig()
)
DEFAULT_PROVENANCE_CONFIG: Final[ProvenanceConfig] = ProvenanceConfig()
DEFAULT_WRITE_CONFIG: Final[WriteConfig] = WriteConfig()
DEFAULT_ERROR_HANDLING_CONFIG: Final[ErrorHandlingConfig] = (
    ErrorHandlingConfig()
)
DEFAULT_CHIMERAX_CONFIG: Final[ChimeraXConfig] = ChimeraXConfig()
DEFAULT_REPORT_CONFIG: Final[ReportConfig] = ReportConfig()

# 5.9. Public configuration interface
# -----------------------------------------------------------------------------

_SECTION_5_PUBLIC_NAMES: Final[Tuple[str, ...]] = (
    "FormattingConfig",
    "TableConfig",
    "RenderConfig",
    "InteractionReportConfig",
    "ScoringReportConfig",
    "MultiposeReportConfig",
    "ProvenanceConfig",
    "WriteConfig",
    "ErrorHandlingConfig",
    "ChimeraXConfig",
    "ReportConfig",
    "DEFAULT_FORMATTING_CONFIG",
    "DEFAULT_TABLE_CONFIG",
    "DEFAULT_RENDER_CONFIG",
    "DEFAULT_INTERACTION_REPORT_CONFIG",
    "DEFAULT_SCORING_REPORT_CONFIG",
    "DEFAULT_MULTIPOSE_REPORT_CONFIG",
    "DEFAULT_PROVENANCE_CONFIG",
    "DEFAULT_WRITE_CONFIG",
    "DEFAULT_ERROR_HANDLING_CONFIG",
    "DEFAULT_CHIMERAX_CONFIG",
    "DEFAULT_REPORT_CONFIG",
)

_register_public_names(_SECTION_5_PUBLIC_NAMES)

# =============================================================================
# End of Section 5
# =============================================================================


# =============================================================================
# Section 6 — Generic object access
# =============================================================================

# 6.1. Missing-value sentinel and access records
# -----------------------------------------------------------------------------

class _MissingValue:
    """Sentinel distinct from ``None``."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "MISSING"

    def __bool__(self) -> bool:
        return False


MISSING: Final[Any] = _MissingValue()


class AccessMatch(NamedTuple):
    """Result of a generic field lookup."""

    found: bool
    value: Any
    requested_name: str
    matched_name: Optional[str]
    source: Optional[str]


# 6.2. Object classification
# -----------------------------------------------------------------------------

def is_dataclass_instance(value: Any) -> bool:
    """Return whether a value is a dataclass instance."""

    return is_dataclass(value) and not inspect.isclass(value)


def is_mapping_like(value: Any) -> bool:
    """Return whether a value exposes mapping semantics."""

    return isinstance(value, Mapping)


def is_sequence_like(value: Any) -> bool:
    """Return whether a value is a non-text sequence."""

    return isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    )


def is_object_like(value: Any) -> bool:
    """Return whether a value can expose named fields."""

    if value is None:
        return False
    return (
        is_mapping_like(value)
        or is_dataclass_instance(value)
        or hasattr(value, "__dict__")
        or hasattr(type(value), "__slots__")
    )


# 6.3. Field-name normalization
# -----------------------------------------------------------------------------

_FIELD_TOKEN_PATTERN: Final[re.Pattern[str]] = re.compile(r"[^a-z0-9]+")


def normalize_field_name(name: Any) -> str:
    """Normalize a field name for tolerant matching."""

    text = str(name).strip().lower()
    return _FIELD_TOKEN_PATTERN.sub("_", text).strip("_")


def field_aliases(name: Any) -> Tuple[str, ...]:
    """Return canonical and configured aliases for a field."""

    normalized = normalize_field_name(name)
    candidates: List[str] = [str(name), normalized]

    if normalized in FIELD_ALIASES:
        candidates.extend(FIELD_ALIASES[normalized])
    else:
        for canonical, aliases in FIELD_ALIASES.items():
            normalized_aliases = {
                normalize_field_name(alias)
                for alias in aliases
            }
            if normalized in normalized_aliases:
                candidates.extend((canonical, *aliases))
                break

    output: List[str] = []
    seen: Set[str] = set()
    for candidate in candidates:
        text = str(candidate)
        key = normalize_field_name(text)
        if key and key not in seen:
            output.append(text)
            seen.add(key)
    return tuple(output)


def _candidate_field_names(
    name: Any,
    *,
    aliases: bool = True,
) -> Tuple[str, ...]:
    """Return field lookup candidates."""

    return field_aliases(name) if aliases else (str(name),)


# 6.4. Mapping and attribute lookup
# -----------------------------------------------------------------------------

def _mapping_match(
    mapping: Mapping[Any, Any],
    candidates: Sequence[str],
) -> AccessMatch:
    """Find a key by exact or normalized name."""

    for candidate in candidates:
        if candidate in mapping:
            return AccessMatch(
                True,
                mapping[candidate],
                str(candidates[0]),
                str(candidate),
                "mapping",
            )

    normalized_candidates = {
        normalize_field_name(candidate)
        for candidate in candidates
    }
    for key, value in mapping.items():
        if normalize_field_name(key) in normalized_candidates:
            return AccessMatch(
                True,
                value,
                str(candidates[0]),
                str(key),
                "mapping",
            )

    return AccessMatch(False, MISSING, str(candidates[0]), None, None)


def _attribute_match(
    obj: Any,
    candidates: Sequence[str],
) -> AccessMatch:
    """Find an attribute without propagating descriptor errors."""

    for candidate in candidates:
        try:
            value = getattr(obj, candidate)
        except (AttributeError, KeyError):
            continue
        except Exception:
            continue
        return AccessMatch(
            True,
            value,
            str(candidates[0]),
            candidate,
            "attribute",
        )

    try:
        names = dir(obj)
    except Exception:
        names = ()

    normalized = {
        normalize_field_name(candidate)
        for candidate in candidates
    }
    for name in names:
        if normalize_field_name(name) not in normalized:
            continue
        try:
            value = getattr(obj, name)
        except Exception:
            continue
        return AccessMatch(
            True,
            value,
            str(candidates[0]),
            name,
            "attribute",
        )

    return AccessMatch(False, MISSING, str(candidates[0]), None, None)


def find_object_field(
    obj: Any,
    name: Any,
    *,
    aliases: bool = True,
) -> AccessMatch:
    """Locate a named field on mappings or objects."""

    candidates = _candidate_field_names(name, aliases=aliases)
    if obj is None:
        return AccessMatch(False, MISSING, str(name), None, None)

    if isinstance(obj, Mapping):
        match = _mapping_match(obj, candidates)
        if match.found:
            return match

    return _attribute_match(obj, candidates)


def has_object_field(
    obj: Any,
    name: Any,
    *,
    aliases: bool = True,
) -> bool:
    """Return whether a field can be resolved."""

    return find_object_field(obj, name, aliases=aliases).found


# 6.5. Safe callable resolution
# -----------------------------------------------------------------------------

def can_call_without_arguments(value: Any) -> bool:
    """Return whether a callable accepts no required arguments."""

    if not callable(value):
        return False
    try:
        signature = inspect.signature(value)
    except (TypeError, ValueError):
        return False
    for parameter in signature.parameters.values():
        if parameter.kind in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }:
            continue
        if (
            parameter.default is inspect.Parameter.empty
            and parameter.kind
            not in {
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            }
        ):
            return False
        if (
            parameter.default is inspect.Parameter.empty
            and parameter.kind
            in {
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            }
        ):
            return False
    return True


def call_zero_argument(
    value: Any,
    *,
    default: Any = MISSING,
    suppress: bool = True,
) -> Any:
    """Call a zero-argument callable safely."""

    if not callable(value):
        return value
    if not can_call_without_arguments(value):
        if suppress:
            return default
        raise ReportAccessError("Callable requires arguments.")
    try:
        return value()
    except Exception as error:
        if suppress:
            return default
        raise ReportAccessError(
            "Unable to call report input attribute.",
            cause=error,
        ) from error


# 6.6. Generic field access
# -----------------------------------------------------------------------------

def get_object_field(
    obj: Any,
    name: Any,
    default: Any = None,
    *,
    aliases: bool = True,
    call: bool = False,
    call_errors: str = "default",
) -> Any:
    """Return a field from a mapping or object."""

    match = find_object_field(obj, name, aliases=aliases)
    if not match.found:
        return default

    value = match.value
    if call and callable(value):
        suppress = call_errors != ERROR_MODE_RAISE
        result = call_zero_argument(
            value,
            default=MISSING,
            suppress=suppress,
        )
        return default if result is MISSING else result
    return value


def require_object_field(
    obj: Any,
    name: Any,
    *,
    aliases: bool = True,
    call: bool = False,
    location: Optional[str] = None,
) -> Any:
    """Return a field or raise ``MissingReportFieldError``."""

    value = get_object_field(
        obj,
        name,
        MISSING,
        aliases=aliases,
        call=call,
        call_errors=ERROR_MODE_RAISE,
    )
    if value is MISSING:
        raise MissingReportFieldError(
            str(name),
            location=location,
        )
    return value


def get_first_object_field(
    obj: Any,
    names: Iterable[Any],
    default: Any = None,
    *,
    aliases: bool = True,
    call: bool = False,
    skip_none: bool = False,
) -> Any:
    """Return the first available field value."""

    for name in names:
        value = get_object_field(
            obj,
            name,
            MISSING,
            aliases=aliases,
            call=call,
        )
        if value is MISSING:
            continue
        if skip_none and value is None:
            continue
        return value
    return default


def find_first_object_field(
    obj: Any,
    names: Iterable[Any],
    *,
    aliases: bool = True,
) -> AccessMatch:
    """Return the first matching field record."""

    requested = tuple(names)
    for name in requested:
        match = find_object_field(obj, name, aliases=aliases)
        if match.found:
            return match
    label = str(requested[0]) if requested else ""
    return AccessMatch(False, MISSING, label, None, None)


# 6.7. Indexed and nested access
# -----------------------------------------------------------------------------

_PATH_PART_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"""
    (?:
        ^|[.]
    )
    ([^.\[\]]+)
    |
    \[
        (?:
            (-?\d+)
            |
            ["']([^"']+)["']
        )
    \]
    """,
    re.VERBOSE,
)


def split_object_path(path: Union[str, Sequence[Any]]) -> Tuple[Any, ...]:
    """Split dotted and indexed access paths."""

    if not isinstance(path, str):
        return tuple(path)

    text = path.strip()
    if not text:
        return ()

    parts: List[Any] = []
    position = 0
    for match in _PATH_PART_PATTERN.finditer(text):
        if match.start() != position:
            raise ReportAccessError(
                f"Invalid object path: {path!r}.",
                context={"path": path},
            )
        field_name, numeric_index, string_key = match.groups()
        if field_name is not None:
            parts.append(field_name)
        elif numeric_index is not None:
            parts.append(int(numeric_index))
        else:
            parts.append(string_key)
        position = match.end()

    if position != len(text):
        raise ReportAccessError(
            f"Invalid object path: {path!r}.",
            context={"path": path},
        )
    return tuple(parts)


def get_indexed_value(
    value: Any,
    key: Any,
    default: Any = None,
    *,
    aliases: bool = True,
) -> Any:
    """Resolve a mapping key, sequence index or object field."""

    if isinstance(key, int):
        try:
            return value[key]
        except (IndexError, KeyError, TypeError):
            return default
        except Exception:
            return default

    if isinstance(value, Mapping):
        match = _mapping_match(
            value,
            _candidate_field_names(key, aliases=aliases),
        )
        return match.value if match.found else default

    return get_object_field(
        value,
        key,
        default,
        aliases=aliases,
    )


def get_object_path(
    obj: Any,
    path: Union[str, Sequence[Any]],
    default: Any = None,
    *,
    aliases: bool = True,
) -> Any:
    """Resolve a dotted or indexed path."""

    current = obj
    for part in split_object_path(path):
        current = get_indexed_value(
            current,
            part,
            MISSING,
            aliases=aliases,
        )
        if current is MISSING:
            return default
    return current


def require_object_path(
    obj: Any,
    path: Union[str, Sequence[Any]],
    *,
    aliases: bool = True,
) -> Any:
    """Resolve a path or raise ``MissingReportFieldError``."""

    value = get_object_path(
        obj,
        path,
        MISSING,
        aliases=aliases,
    )
    if value is MISSING:
        raise MissingReportFieldError(
            str(path),
            location="object path",
        )
    return value


# 6.8. Object iteration and shallow conversion
# -----------------------------------------------------------------------------

def iter_object_items(
    obj: Any,
    *,
    include_private: bool = False,
    include_properties: bool = False,
    include_callables: bool = False,
) -> Iterator[Tuple[str, Any]]:
    """Yield accessible named values from an object."""

    if obj is None:
        return

    if isinstance(obj, Mapping):
        for key, value in obj.items():
            name = str(key)
            if include_private or not name.startswith("_"):
                if include_callables or not callable(value):
                    yield name, value
        return

    if is_dataclass_instance(obj):
        for field_info in fields(obj):
            name = field_info.name
            if not include_private and name.startswith("_"):
                continue
            try:
                value = getattr(obj, name)
            except Exception:
                continue
            if include_callables or not callable(value):
                yield name, value
        return

    seen: Set[str] = set()
    try:
        direct = vars(obj)
    except (TypeError, AttributeError):
        direct = {}

    for name, value in direct.items():
        text = str(name)
        if not include_private and text.startswith("_"):
            continue
        if not include_callables and callable(value):
            continue
        seen.add(text)
        yield text, value

    if not include_properties:
        return

    try:
        names = dir(obj)
    except Exception:
        return

    for name in names:
        if name in seen:
            continue
        if not include_private and name.startswith("_"):
            continue
        try:
            value = getattr(obj, name)
        except Exception:
            continue
        if not include_callables and callable(value):
            continue
        yield name, value


def object_field_names(
    obj: Any,
    *,
    include_private: bool = False,
    include_properties: bool = False,
) -> Tuple[str, ...]:
    """Return accessible object field names."""

    return tuple(
        name
        for name, _ in iter_object_items(
            obj,
            include_private=include_private,
            include_properties=include_properties,
            include_callables=True,
        )
    )


def object_to_shallow_dict(
    obj: Any,
    *,
    include_private: bool = False,
    include_properties: bool = False,
    include_callables: bool = False,
) -> Dict[str, Any]:
    """Convert accessible fields into a shallow dictionary."""

    return dict(
        iter_object_items(
            obj,
            include_private=include_private,
            include_properties=include_properties,
            include_callables=include_callables,
        )
    )


# 6.9. Collection access helpers
# -----------------------------------------------------------------------------

def iter_object_collection(
    value: Any,
    *,
    mapping_values: bool = True,
    scalar_as_single: bool = True,
) -> Iterator[Any]:
    """Iterate heterogeneous collection-like values."""

    if value is None or value is MISSING:
        return
    if isinstance(value, Mapping):
        iterable = value.values() if mapping_values else value.items()
        yield from iterable
        return
    if isinstance(value, (str, bytes, bytearray)):
        if scalar_as_single:
            yield value
        return
    try:
        iterator = iter(value)
    except TypeError:
        if scalar_as_single:
            yield value
        return
    yield from iterator


def collection_size(value: Any) -> int:
    """Return a best-effort collection size."""

    if value is None or value is MISSING:
        return 0
    try:
        return len(value)
    except (TypeError, AttributeError):
        return sum(1 for _ in iter_object_collection(value))


def first_collection_item(
    value: Any,
    default: Any = None,
) -> Any:
    """Return the first collection item."""

    return next(iter_object_collection(value), default)


# 6.10. DockModel-oriented access helpers
# -----------------------------------------------------------------------------

def get_model_identifier(obj: Any, default: Any = None) -> Any:
    """Return a model identifier from common fields."""

    return get_first_object_field(
        obj,
        FIELD_ALIASES["model_id"],
        default,
        skip_none=True,
    )


def get_pose_identifier(obj: Any, default: Any = None) -> Any:
    """Return a pose identifier from common fields."""

    return get_first_object_field(
        obj,
        FIELD_ALIASES["pose_id"],
        default,
        skip_none=True,
    )


def get_object_metadata(
    obj: Any,
    default: Optional[Mapping[str, Any]] = None,
) -> Mapping[str, Any]:
    """Return metadata as a mapping."""

    value = get_first_object_field(
        obj,
        FIELD_ALIASES["metadata"],
        MISSING,
        skip_none=True,
    )
    if value is MISSING:
        return default if default is not None else _EMPTY_METADATA
    if isinstance(value, Mapping):
        return value
    return default if default is not None else _EMPTY_METADATA


def get_interaction_container(
    obj: Any,
    family: Any,
    default: Any = None,
) -> Any:
    """Return an interaction-family container."""

    try:
        canonical = InteractionFamily.coerce(family).value
    except ValueError:
        canonical = INTERACTION_UNKNOWN
    names = INTERACTION_CONTAINER_ATTRIBUTES.get(canonical, ())
    return get_first_object_field(
        obj,
        names,
        default,
        aliases=False,
    )


def iter_interaction_containers(
    obj: Any,
    *,
    include_empty: bool = False,
) -> Iterator[Tuple[InteractionFamily, Any]]:
    """Yield recognized interaction containers."""

    for family in InteractionFamily:
        if family is InteractionFamily.UNKNOWN:
            continue
        value = get_interaction_container(obj, family, MISSING)
        if value is MISSING:
            continue
        if not include_empty and collection_size(value) == 0:
            continue
        yield family, value


# 6.11. Public access interface
# -----------------------------------------------------------------------------

_SECTION_6_PUBLIC_NAMES: Final[Tuple[str, ...]] = (
    "MISSING",
    "AccessMatch",
    "is_dataclass_instance",
    "is_mapping_like",
    "is_sequence_like",
    "is_object_like",
    "normalize_field_name",
    "field_aliases",
    "find_object_field",
    "has_object_field",
    "can_call_without_arguments",
    "call_zero_argument",
    "get_object_field",
    "require_object_field",
    "get_first_object_field",
    "find_first_object_field",
    "split_object_path",
    "get_indexed_value",
    "get_object_path",
    "require_object_path",
    "iter_object_items",
    "object_field_names",
    "object_to_shallow_dict",
    "iter_object_collection",
    "collection_size",
    "first_collection_item",
    "get_model_identifier",
    "get_pose_identifier",
    "get_object_metadata",
    "get_interaction_container",
    "iter_interaction_containers",
)

_register_public_names(_SECTION_6_PUBLIC_NAMES)

# =============================================================================
# End of Section 6
# =============================================================================

# =============================================================================
# Section 7 — Safe formatting
# =============================================================================

# 7.1. Scalar classification
# -----------------------------------------------------------------------------

def is_missing_value(value: Any) -> bool:
    """Return whether a value represents missing data."""

    return value is None or value is MISSING


def is_numpy_scalar(value: Any) -> bool:
    """Return whether a value is a NumPy scalar."""

    return bool(
        NUMPY_AVAILABLE
        and np is not None
        and isinstance(value, np.generic)
    )


def unwrap_scalar(value: Any) -> Any:
    """Convert supported scalar wrappers to Python values."""

    if is_numpy_scalar(value):
        try:
            return value.item()
        except Exception:
            return value
    return value


def is_boolean_value(value: Any) -> bool:
    """Return whether a value is boolean-like."""

    value = unwrap_scalar(value)
    return isinstance(value, bool)


def is_numeric_value(
    value: Any,
    *,
    include_bool: bool = False,
) -> bool:
    """Return whether a value is a real numeric scalar."""

    value = unwrap_scalar(value)
    if isinstance(value, bool):
        return include_bool
    return isinstance(value, Real)


def to_finite_float(
    value: Any,
    default: Any = None,
    *,
    allow_bool: bool = False,
) -> Any:
    """Convert a value to a finite float."""

    value = unwrap_scalar(value)
    if value is None or value is MISSING:
        return default
    if isinstance(value, bool) and not allow_bool:
        return default
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return result if math.isfinite(result) else default


def to_safe_int(
    value: Any,
    default: Any = None,
    *,
    allow_bool: bool = False,
) -> Any:
    """Convert an integer-like value without silent truncation."""

    value = unwrap_scalar(value)
    if value is None or value is MISSING:
        return default
    if isinstance(value, bool) and not allow_bool:
        return default
    if isinstance(value, int):
        return value
    numeric = to_finite_float(value, MISSING, allow_bool=allow_bool)
    if numeric is MISSING or not float(numeric).is_integer():
        return default
    return int(numeric)


def normalize_negative_zero(
    value: float,
    *,
    tolerance: float = REPORT_EPSILON,
) -> float:
    """Replace near-zero values with positive zero."""

    return 0.0 if abs(value) <= tolerance else value


# 7.2. Safe text conversion
# -----------------------------------------------------------------------------

_CONTROL_CHARACTER_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]"
)
_MULTILINE_WHITESPACE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"[ \t\f\v]+"
)
_MULTI_BLANK_LINE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\n{3,}"
)


def safe_string(
    value: Any,
    default: str = DEFAULT_EMPTY_TEXT,
    *,
    strip: bool = True,
    collapse_whitespace: bool = False,
    preserve_newlines: bool = True,
    remove_controls: bool = True,
) -> str:
    """Convert any value to safe display text."""

    if value is None or value is MISSING:
        return default

    value = unwrap_scalar(value)
    if isinstance(value, bytes):
        try:
            text = value.decode(DEFAULT_ENCODING, errors="replace")
        except Exception:
            text = repr(value)
    elif isinstance(value, Path):
        text = str(value)
    elif isinstance(value, Enum):
        text = str(value.value)
    else:
        try:
            text = str(value)
        except Exception:
            try:
                text = repr(value)
            except Exception:
                return default

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if remove_controls:
        text = _CONTROL_CHARACTER_PATTERN.sub("", text)

    if collapse_whitespace:
        if preserve_newlines:
            lines = [
                WHITESPACE_PATTERN.sub(" ", line).strip()
                for line in text.split("\n")
            ]
            text = "\n".join(lines)
            text = _MULTI_BLANK_LINE_PATTERN.sub("\n\n", text)
        else:
            text = WHITESPACE_PATTERN.sub(" ", text)

    return text.strip() if strip else text


def single_line_text(
    value: Any,
    default: str = DEFAULT_EMPTY_TEXT,
) -> str:
    """Return compact single-line text."""

    return safe_string(
        value,
        default,
        strip=True,
        collapse_whitespace=True,
        preserve_newlines=False,
    )


def truncate_text(
    value: Any,
    max_length: int = DEFAULT_MAX_CELL_LENGTH,
    *,
    marker: str = DEFAULT_TRUNCATION_MARKER,
    preserve_words: bool = True,
) -> str:
    """Truncate text to a bounded display length."""

    text = safe_string(value)
    limit = max(0, int(max_length))
    if len(text) <= limit:
        return text
    if limit == 0:
        return ""
    marker = safe_string(marker, "", strip=False)
    if len(marker) >= limit:
        return marker[:limit]

    available = limit - len(marker)
    prefix = text[:available]
    if preserve_words and available > 4:
        candidate = prefix.rsplit(None, 1)[0]
        if len(candidate) >= max(1, available // 2):
            prefix = candidate
    return prefix.rstrip() + marker


def safe_identifier(
    value: Any,
    default: str = DEFAULT_UNKNOWN_TEXT,
    *,
    max_length: int = DEFAULT_MAX_TITLE_LENGTH,
) -> str:
    """Return a compact identifier-like string."""

    text = single_line_text(value, default)
    return truncate_text(text, max_length, preserve_words=False)


def title_case_label(value: Any) -> str:
    """Convert a field identifier to a display label."""

    text = normalize_field_name(value).replace("_", " ")
    return text[:1].upper() + text[1:] if text else ""


def field_label(name: Any) -> str:
    """Return a configured or generated column label."""

    normalized = normalize_field_name(name)
    return COLUMN_LABELS.get(normalized, title_case_label(normalized))


# 7.3. Numeric formatting
# -----------------------------------------------------------------------------

def format_number(
    value: Any,
    digits: int = DEFAULT_FLOAT_DIGITS,
    *,
    missing: str = DEFAULT_MISSING_TEXT,
    trim_zeros: bool = False,
    signed: bool = False,
    thousands: bool = False,
    normalize_zero: bool = True,
) -> str:
    """Format a finite number safely."""

    numeric = to_finite_float(value, MISSING)
    if numeric is MISSING:
        return missing

    digits = max(0, int(digits))
    if normalize_zero:
        numeric = normalize_negative_zero(numeric)

    sign = "+" if signed else ""
    grouping = "," if thousands else ""
    text = format(numeric, f"{sign}{grouping}.{digits}f")
    if trim_zeros and "." in text:
        text = text.rstrip("0").rstrip(".")
        if text in {"-0", "+0"}:
            text = text[-1]
    return text


def format_integer(
    value: Any,
    *,
    missing: str = DEFAULT_MISSING_TEXT,
    thousands: bool = False,
) -> str:
    """Format an integer-like value safely."""

    integer = to_safe_int(value, MISSING)
    if integer is MISSING:
        return missing
    return format(integer, ",d" if thousands else "d")


def format_score(
    value: Any,
    config: FormattingConfig = DEFAULT_FORMATTING_CONFIG,
) -> str:
    """Format a score."""

    return format_number(
        value,
        config.score_digits,
        missing=config.missing_text,
        normalize_zero=config.normalize_negative_zero,
    )


def format_distance(
    value: Any,
    config: FormattingConfig = DEFAULT_FORMATTING_CONFIG,
    *,
    include_unit: bool = False,
) -> str:
    """Format an angstrom distance."""

    text = format_number(
        value,
        config.distance_digits,
        missing=config.missing_text,
        normalize_zero=config.normalize_negative_zero,
    )
    if include_unit and text != config.missing_text:
        return f"{text} Å"
    return text


def format_angle(
    value: Any,
    config: FormattingConfig = DEFAULT_FORMATTING_CONFIG,
    *,
    include_unit: bool = False,
) -> str:
    """Format an angle in degrees."""

    text = format_number(
        value,
        config.angle_digits,
        missing=config.missing_text,
        normalize_zero=config.normalize_negative_zero,
    )
    if include_unit and text != config.missing_text:
        return f"{text}°"
    return text


def format_percent(
    value: Any,
    config: FormattingConfig = DEFAULT_FORMATTING_CONFIG,
    *,
    fraction: bool = False,
    include_symbol: bool = True,
) -> str:
    """Format a percentage."""

    numeric = to_finite_float(value, MISSING)
    if numeric is MISSING:
        return config.missing_text
    if fraction:
        numeric *= 100.0
    text = format_number(
        numeric,
        config.percent_digits,
        missing=config.missing_text,
        normalize_zero=config.normalize_negative_zero,
    )
    return f"{text}%" if include_symbol else text


def format_range(
    minimum: Any,
    maximum: Any,
    *,
    digits: int = DEFAULT_FLOAT_DIGITS,
    separator: str = "–",
    missing: str = DEFAULT_MISSING_TEXT,
) -> str:
    """Format a numeric interval."""

    low = to_finite_float(minimum, MISSING)
    high = to_finite_float(maximum, MISSING)
    if low is MISSING and high is MISSING:
        return missing
    if low is MISSING:
        return f"≤ {format_number(high, digits, missing=missing)}"
    if high is MISSING:
        return f"≥ {format_number(low, digits, missing=missing)}"
    return (
        f"{format_number(low, digits, missing=missing)}"
        f"{separator}"
        f"{format_number(high, digits, missing=missing)}"
    )


# 7.4. Date, time and path formatting
# -----------------------------------------------------------------------------

def normalize_datetime(value: Any) -> Optional[datetime]:
    """Convert supported values to a datetime."""

    if value is None or value is MISSING:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time())
    if isinstance(value, Real) and not isinstance(value, bool):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
        try:
            return datetime.fromisoformat(normalized)
        except ValueError:
            return None
    return None


def format_datetime(
    value: Any,
    *,
    missing: str = DEFAULT_MISSING_TEXT,
    use_utc: bool = False,
    timespec: str = ISO_DATETIME_TIMESPEC,
) -> str:
    """Format a date or datetime in ISO form."""

    moment = normalize_datetime(value)
    if moment is None:
        return missing
    if use_utc:
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        else:
            moment = moment.astimezone(timezone.utc)
    try:
        text = moment.isoformat(timespec=timespec)
    except ValueError:
        text = moment.isoformat()
    if use_utc and text.endswith("+00:00"):
        text = text[:-6] + UTC_SUFFIX
    return text


def format_path(
    value: Any,
    *,
    missing: str = DEFAULT_MISSING_TEXT,
    resolve: bool = False,
    home_relative: bool = False,
) -> str:
    """Format a path without requiring it to exist."""

    if value is None or value is MISSING:
        return missing
    try:
        path = Path(os.fspath(value))
    except (TypeError, ValueError):
        return single_line_text(value, missing)
    if resolve:
        try:
            path = path.expanduser().resolve(strict=False)
        except Exception:
            path = path.expanduser()
    if home_relative:
        try:
            path = Path("~") / path.relative_to(Path.home())
        except (ValueError, OSError):
            pass
    return str(path)


# 7.5. Atom and residue formatting
# -----------------------------------------------------------------------------

def atom_display_name(
    atom: Any,
    *,
    include_residue: bool = False,
    include_model: bool = False,
    missing: str = DEFAULT_MISSING_TEXT,
) -> str:
    """Return a compact atom label."""

    if atom is None or atom is MISSING:
        return missing
    if isinstance(atom, str):
        return single_line_text(atom, missing)

    name = get_first_object_field(
        atom,
        ("atom_name", "name"),
        None,
        skip_none=True,
    )
    serial = get_first_object_field(
        atom,
        ("serial_number", "serial", "atom_id", "index"),
        None,
        skip_none=True,
    )
    residue = get_first_object_field(
        atom,
        ("residue", "parent_residue"),
        None,
        skip_none=True,
    )
    model = get_first_object_field(
        atom,
        ("structure", "model"),
        None,
        skip_none=True,
    )

    parts: List[str] = []
    if include_model and model is not None:
        model_name = get_first_object_field(
            model,
            ("model_id", "name", "id"),
            None,
            skip_none=True,
        )
        if model_name is not None:
            parts.append(safe_identifier(model_name))
    if include_residue and residue is not None:
        parts.append(residue_display_name(residue, missing=""))
    atom_text = safe_identifier(name, "") if name is not None else ""
    if serial is not None:
        serial_text = safe_identifier(serial, "")
        atom_text = f"{atom_text}#{serial_text}" if atom_text else serial_text
    if atom_text:
        parts.append(atom_text)
    return ":".join(part for part in parts if part) or missing


def residue_display_name(
    residue: Any,
    *,
    include_chain: bool = True,
    missing: str = DEFAULT_MISSING_TEXT,
) -> str:
    """Return a compact residue label."""

    if residue is None or residue is MISSING:
        return missing
    if isinstance(residue, str):
        return single_line_text(residue, missing)

    name = get_first_object_field(
        residue,
        ("residue_name", "name", "resname"),
        None,
        skip_none=True,
    )
    number = get_first_object_field(
        residue,
        ("residue_number", "number", "resnum", "id"),
        None,
        skip_none=True,
    )
    chain = get_first_object_field(
        residue,
        ("chain_id", "chain", "chain_identifier"),
        None,
        skip_none=True,
    )

    if chain is not None and not isinstance(chain, str):
        chain = get_first_object_field(
            chain,
            ("chain_id", "id", "name"),
            chain,
            skip_none=True,
        )

    core = "".join(
        part
        for part in (
            safe_identifier(name, "") if name is not None else "",
            safe_identifier(number, "") if number is not None else "",
        )
        if part
    )
    if include_chain and chain is not None:
        chain_text = safe_identifier(chain, "")
        if chain_text:
            core = f"{chain_text}:{core}" if core else chain_text
    return core or missing


# 7.6. Container and generic value formatting
# -----------------------------------------------------------------------------

def format_sequence(
    values: Any,
    *,
    formatter: Optional[ValueFormatter] = None,
    separator: str = ", ",
    max_items: int = DEFAULT_MAX_ITEMS,
    missing: str = DEFAULT_MISSING_TEXT,
) -> str:
    """Format a heterogeneous sequence."""

    if values is None or values is MISSING:
        return missing
    formatter = formatter or safe_string
    items = list(iter_object_collection(values))
    if not items:
        return ""
    limit = max(0, int(max_items))
    visible = items[:limit]
    text = separator.join(formatter(item) for item in visible)
    if len(items) > limit:
        text += f"{separator}{DEFAULT_TRUNCATION_MARKER} (+{len(items) - limit})"
    return text


def format_mapping(
    value: Any,
    *,
    item_separator: str = ", ",
    key_value_separator: str = TEXT_KEY_VALUE_SEPARATOR,
    max_items: int = DEFAULT_MAX_ITEMS,
    missing: str = DEFAULT_MISSING_TEXT,
) -> str:
    """Format a mapping as compact key-value text."""

    if value is None or value is MISSING:
        return missing
    if not isinstance(value, Mapping):
        return safe_string(value, missing)

    items = list(value.items())
    visible = items[:max(0, int(max_items))]
    text = item_separator.join(
        f"{field_label(key)}{key_value_separator}{format_value(item)}"
        for key, item in visible
    )
    if len(items) > len(visible):
        text += (
            f"{item_separator}{DEFAULT_TRUNCATION_MARKER}"
            f" (+{len(items) - len(visible)})"
        )
    return text


def format_value(
    value: Any,
    *,
    config: FormattingConfig = DEFAULT_FORMATTING_CONFIG,
    max_length: Optional[int] = None,
) -> str:
    """Format an arbitrary report value safely."""

    value = unwrap_scalar(value)
    if value is None or value is MISSING:
        text = config.missing_text
    elif isinstance(value, bool):
        text = config.true_text if value else config.false_text
    elif isinstance(value, Enum):
        text = safe_string(value.value, config.missing_text)
    elif isinstance(value, datetime):
        text = format_datetime(value, missing=config.missing_text)
    elif isinstance(value, date):
        text = value.isoformat()
    elif isinstance(value, Path):
        text = format_path(value, missing=config.missing_text)
    elif is_numeric_value(value):
        text = format_number(
            value,
            config.float_digits,
            missing=config.missing_text,
            trim_zeros=True,
            normalize_zero=config.normalize_negative_zero,
        )
    elif isinstance(value, Mapping):
        text = format_mapping(value, missing=config.missing_text)
    elif is_sequence_like(value) or isinstance(value, (set, frozenset)):
        text = format_sequence(value, missing=config.missing_text)
    else:
        text = safe_string(
            value,
            config.missing_text,
            strip=config.strip_text,
        )

    if max_length is not None:
        return truncate_text(
            text,
            max_length,
            marker=config.truncation_marker,
        )
    return text


def format_key_value(
    key: Any,
    value: Any,
    *,
    config: FormattingConfig = DEFAULT_FORMATTING_CONFIG,
    separator: str = TEXT_KEY_VALUE_SEPARATOR,
) -> str:
    """Format one labeled report value."""

    return f"{field_label(key)}{separator}{format_value(value, config=config)}"


# 7.7. Escaping helpers
# -----------------------------------------------------------------------------

def escape_markdown(value: Any) -> str:
    """Escape Markdown-sensitive characters."""

    text = safe_string(value, "", strip=False)
    return MARKDOWN_ESCAPE_PATTERN.sub(r"\\\1", text)


def escape_markdown_cell(value: Any) -> str:
    """Escape a Markdown table cell."""

    return escape_markdown(value).replace("\n", "<br>")


def escape_html(value: Any, *, quote: bool = True) -> str:
    """Escape text for HTML output."""

    return html_escape(safe_string(value, "", strip=False), quote=quote)


def escape_json_string(value: Any) -> str:
    """Return a JSON-escaped string without outer quotes."""

    encoded = json.dumps(
        safe_string(value, "", strip=False),
        ensure_ascii=DEFAULT_JSON_ENSURE_ASCII,
    )
    return encoded[1:-1]


# 7.8. Safe formatter object
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class SafeFormatter:
    """Reusable report value formatter."""

    config: FormattingConfig = field(default_factory=FormattingConfig)

    def value(self, value: Any, *, max_length: Optional[int] = None) -> str:
        return format_value(value, config=self.config, max_length=max_length)

    def number(self, value: Any, *, digits: Optional[int] = None) -> str:
        return format_number(
            value,
            self.config.float_digits if digits is None else digits,
            missing=self.config.missing_text,
            normalize_zero=self.config.normalize_negative_zero,
        )

    def score(self, value: Any) -> str:
        return format_score(value, self.config)

    def distance(self, value: Any, *, include_unit: bool = False) -> str:
        return format_distance(
            value,
            self.config,
            include_unit=include_unit,
        )

    def angle(self, value: Any, *, include_unit: bool = False) -> str:
        return format_angle(
            value,
            self.config,
            include_unit=include_unit,
        )

    def percent(
        self,
        value: Any,
        *,
        fraction: bool = False,
        include_symbol: bool = True,
    ) -> str:
        return format_percent(
            value,
            self.config,
            fraction=fraction,
            include_symbol=include_symbol,
        )

    def residue(self, value: Any) -> str:
        return residue_display_name(
            value,
            missing=self.config.missing_text,
        )

    def atom(self, value: Any) -> str:
        return atom_display_name(
            value,
            missing=self.config.missing_text,
        )


DEFAULT_SAFE_FORMATTER: Final[SafeFormatter] = SafeFormatter()

# 7.9. Public formatting interface
# -----------------------------------------------------------------------------

_SECTION_7_PUBLIC_NAMES: Final[Tuple[str, ...]] = (
    "is_missing_value",
    "is_numpy_scalar",
    "unwrap_scalar",
    "is_boolean_value",
    "is_numeric_value",
    "to_finite_float",
    "to_safe_int",
    "normalize_negative_zero",
    "safe_string",
    "single_line_text",
    "truncate_text",
    "safe_identifier",
    "title_case_label",
    "field_label",
    "format_number",
    "format_integer",
    "format_score",
    "format_distance",
    "format_angle",
    "format_percent",
    "format_range",
    "normalize_datetime",
    "format_datetime",
    "format_path",
    "atom_display_name",
    "residue_display_name",
    "format_sequence",
    "format_mapping",
    "format_value",
    "format_key_value",
    "escape_markdown",
    "escape_markdown_cell",
    "escape_html",
    "escape_json_string",
    "SafeFormatter",
    "DEFAULT_SAFE_FORMATTER",
)

_register_public_names(_SECTION_7_PUBLIC_NAMES)

# =============================================================================
# End of Section 7
# =============================================================================


# =============================================================================
# Section 8 — Interaction normalization
# =============================================================================

# 8.1. Normalized interaction record
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class NormalizedInteraction:
    """Canonical report representation of one interaction."""

    id: str
    family: InteractionFamily
    type: str
    subtype: str = ""
    source: str = ""
    pose_id: Any = None
    model_id: Any = None
    ligand_atom: str = ""
    receptor_atom: str = ""
    ligand_residue: str = ""
    receptor_residue: str = ""
    chain_id: str = ""
    distance: Optional[float] = None
    angle: Optional[float] = None
    strength: str = STRENGTH_UNKNOWN
    classification: str = ""
    score: Optional[float] = None
    metadata: Mapping[str, Any] = field(default_factory=lambda: _EMPTY_METADATA)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "family",
            _coerce_enum(
                InteractionFamily,
                self.family,
                "family",
                default=InteractionFamily.UNKNOWN,
            ),
        )
        for name in (
            "id",
            "type",
            "subtype",
            "source",
            "ligand_atom",
            "receptor_atom",
            "ligand_residue",
            "receptor_residue",
            "chain_id",
            "strength",
            "classification",
        ):
            object.__setattr__(
                self,
                name,
                single_line_text(getattr(self, name), ""),
            )
        object.__setattr__(
            self,
            "distance",
            to_finite_float(self.distance, None),
        )
        object.__setattr__(
            self,
            "angle",
            to_finite_float(self.angle, None),
        )
        object.__setattr__(
            self,
            "score",
            to_finite_float(self.score, None),
        )
        object.__setattr__(
            self,
            "metadata",
            _freeze_config_mapping(self.metadata),
        )

    def to_dict(
        self,
        *,
        include_metadata: bool = True,
        include_empty: bool = False,
    ) -> Dict[str, Any]:
        """Return a canonical interaction dictionary."""

        record: Dict[str, Any] = {
            KEY_ID: self.id,
            KEY_SOURCE: self.source,
            KEY_FAMILY: self.family.value,
            KEY_TYPE: self.type,
            KEY_SUBTYPE: self.subtype,
            KEY_POSE_ID: self.pose_id,
            KEY_MODEL_ID: self.model_id,
            KEY_LIGAND_ATOM: self.ligand_atom,
            KEY_RECEPTOR_ATOM: self.receptor_atom,
            KEY_LIGAND_RESIDUE: self.ligand_residue,
            KEY_RECEPTOR_RESIDUE: self.receptor_residue,
            KEY_CHAIN_ID: self.chain_id,
            KEY_DISTANCE: self.distance,
            KEY_ANGLE: self.angle,
            KEY_STRENGTH: self.strength,
            KEY_CLASSIFICATION: self.classification,
            KEY_SCORE: self.score,
        }
        if include_metadata:
            record[KEY_METADATA] = dict(self.metadata)
        if not include_empty:
            record = {
                key: value
                for key, value in record.items()
                if value not in (None, "", (), [], {})
            }
        return record

    def fingerprint(self) -> Tuple[Any, ...]:
        """Return a stable deduplication key."""

        return (
            self.family.value,
            normalize_field_name(self.type),
            normalize_field_name(self.subtype),
            self.pose_id,
            self.model_id,
            normalize_field_name(self.ligand_atom),
            normalize_field_name(self.receptor_atom),
            normalize_field_name(self.ligand_residue),
            normalize_field_name(self.receptor_residue),
            normalize_field_name(self.chain_id),
            None if self.distance is None else round(self.distance, 6),
            None if self.angle is None else round(self.angle, 6),
        )


# 8.2. Family and type inference
# -----------------------------------------------------------------------------

def normalize_interaction_family(
    value: Any,
    *,
    default: InteractionFamily = InteractionFamily.UNKNOWN,
) -> InteractionFamily:
    """Normalize an interaction family."""

    if value is None or value is MISSING:
        return default
    try:
        return InteractionFamily.coerce(value)
    except ValueError:
        token = normalize_field_name(value)
        for alias, canonical in INTERACTION_FAMILY_ALIASES.items():
            if token == normalize_field_name(alias):
                return InteractionFamily(canonical)
        return default


def infer_interaction_family(
    interaction: Any,
    *,
    family_hint: Any = None,
) -> InteractionFamily:
    """Infer interaction family from hints, fields and class names."""

    if family_hint is not None:
        family = normalize_interaction_family(family_hint)
        if family is not InteractionFamily.UNKNOWN:
            return family

    explicit = get_first_object_field(
        interaction,
        (
            "family",
            "interaction_family",
            "interaction_type",
            "type",
            "kind",
            "category",
            "source",
        ),
        None,
        skip_none=True,
    )
    if explicit is not None:
        family = normalize_interaction_family(explicit)
        if family is not InteractionFamily.UNKNOWN:
            return family

    class_name = normalize_field_name(type(interaction).__name__)
    module_name = normalize_field_name(type(interaction).__module__)
    searchable = f"{module_name}_{class_name}"

    patterns: Tuple[Tuple[InteractionFamily, Tuple[str, ...]], ...] = (
        (
            InteractionFamily.HYDROGEN_BOND,
            ("hbond", "hydrogen_bond", "hydrogenbond"),
        ),
        (
            InteractionFamily.HYDROPHOBIC,
            ("hydrophobic", "alkyl_contact"),
        ),
        (
            InteractionFamily.PI,
            ("pi_interaction", "pistacking", "cation_pi", "anion_pi"),
        ),
        (
            InteractionFamily.SALT_BRIDGE,
            ("salt_bridge", "saltbridge", "ionic_interaction"),
        ),
        (
            InteractionFamily.CLASH,
            ("clash", "steric"),
        ),
        (
            InteractionFamily.CONTACT,
            ("contact", "atomic_contact"),
        ),
    )
    for family, terms in patterns:
        if any(term in searchable for term in terms):
            return family
    return InteractionFamily.UNKNOWN


def normalize_interaction_type(
    interaction: Any,
    family: InteractionFamily,
) -> str:
    """Return a normalized interaction type."""

    value = get_first_object_field(
        interaction,
        (
            "interaction_type",
            "type",
            "kind",
            "name",
            "category",
        ),
        None,
        skip_none=True,
    )
    if value is None:
        return family.value
    token = normalize_field_name(value)
    family_alias = normalize_interaction_family(token)
    return family.value if family_alias is family else token or family.value


def normalize_interaction_subtype(interaction: Any) -> str:
    """Return a normalized interaction subtype."""

    value = get_first_object_field(
        interaction,
        (
            "interaction_subtype",
            "subtype",
            "classification",
            "geometry_class",
            "geometry_type",
        ),
        "",
        skip_none=True,
    )
    return normalize_field_name(value) if value else ""


# 8.3. Atom and residue extraction
# -----------------------------------------------------------------------------

def _interaction_atom(
    interaction: Any,
    side: str,
) -> Any:
    """Return a ligand- or receptor-side atom."""

    if side == "ligand":
        names = (
            "ligand_atom",
            "atom1",
            "first_atom",
            "donor_atom",
            "cation_atom",
            "source_atom",
        )
    else:
        names = (
            "receptor_atom",
            "protein_atom",
            "atom2",
            "second_atom",
            "acceptor_atom",
            "anion_atom",
            "target_atom",
        )
    return get_first_object_field(
        interaction,
        names,
        None,
        skip_none=True,
    )


def _interaction_residue(
    interaction: Any,
    side: str,
    atom: Any,
) -> Any:
    """Return a ligand- or receptor-side residue."""

    names = (
        ("ligand_residue", "residue1", "first_residue")
        if side == "ligand"
        else (
            "receptor_residue",
            "protein_residue",
            "residue2",
            "second_residue",
            "residue",
        )
    )
    residue = get_first_object_field(
        interaction,
        names,
        None,
        skip_none=True,
    )
    if residue is None and atom is not None:
        residue = get_first_object_field(
            atom,
            ("residue", "parent_residue"),
            None,
            skip_none=True,
        )
    return residue


def _interaction_chain_id(
    interaction: Any,
    receptor_residue: Any,
) -> str:
    """Return the receptor chain identifier."""

    chain = get_first_object_field(
        interaction,
        ("chain_id", "chain", "receptor_chain"),
        None,
        skip_none=True,
    )
    if chain is None and receptor_residue is not None:
        chain = get_first_object_field(
            receptor_residue,
            ("chain_id", "chain", "chain_identifier"),
            None,
            skip_none=True,
        )
    if chain is not None and not isinstance(chain, str):
        chain = get_first_object_field(
            chain,
            ("chain_id", "id", "name"),
            chain,
            skip_none=True,
        )
    return safe_identifier(chain, "") if chain is not None else ""


# 8.4. Geometry, score and metadata extraction
# -----------------------------------------------------------------------------

def _interaction_distance(interaction: Any) -> Optional[float]:
    """Return the primary interaction distance."""

    value = get_first_object_field(
        interaction,
        (
            "distance",
            "distance_angstrom",
            "distance_a",
            "separation",
            "centroid_distance",
            "minimum_distance",
            "min_distance",
        ),
        None,
        skip_none=True,
    )
    return to_finite_float(value, None)


def _interaction_angle(interaction: Any) -> Optional[float]:
    """Return the primary interaction angle."""

    value = get_first_object_field(
        interaction,
        (
            "angle",
            "angle_degrees",
            "theta",
            "donor_angle",
            "plane_angle",
            "normal_angle",
        ),
        None,
        skip_none=True,
    )
    return to_finite_float(value, None)


def _interaction_score(interaction: Any) -> Optional[float]:
    """Return an interaction score if present."""

    value = get_first_object_field(
        interaction,
        (
            "score",
            "interaction_score",
            "weighted_score",
            "normalized_score",
        ),
        None,
        skip_none=True,
    )
    return to_finite_float(value, None)


def _interaction_strength(interaction: Any) -> str:
    """Return a normalized strength label."""

    value = get_first_object_field(
        interaction,
        ("strength", "interaction_strength", "quality"),
        STRENGTH_UNKNOWN,
        skip_none=True,
    )
    token = normalize_field_name(value)
    aliases = {
        "high": STRENGTH_STRONG,
        "strong": STRENGTH_STRONG,
        "medium": STRENGTH_MODERATE,
        "moderate": STRENGTH_MODERATE,
        "low": STRENGTH_WEAK,
        "weak": STRENGTH_WEAK,
        "unclassified": STRENGTH_UNCLASSIFIED,
        "unknown": STRENGTH_UNKNOWN,
    }
    return aliases.get(token, token or STRENGTH_UNKNOWN)


def _interaction_classification(interaction: Any) -> str:
    """Return a qualitative classification."""

    value = get_first_object_field(
        interaction,
        (
            "classification",
            "geometry_class",
            "quality",
            "class_name",
        ),
        "",
        skip_none=True,
    )
    return normalize_field_name(value) if value else ""


def _interaction_metadata(
    interaction: Any,
    *,
    consumed_fields: Iterable[str] = (),
    include_unmapped: bool = False,
) -> Mapping[str, Any]:
    """Collect explicit and optionally unmapped metadata."""

    metadata: Dict[str, Any] = {}
    explicit = get_object_metadata(interaction)
    if explicit:
        metadata.update(explicit)

    if include_unmapped:
        consumed = {
            normalize_field_name(name)
            for name in consumed_fields
        }
        for name, value in iter_object_items(interaction):
            if normalize_field_name(name) in consumed:
                continue
            if callable(value):
                continue
            metadata.setdefault(name, value)
    return MappingProxyType(metadata) if metadata else _EMPTY_METADATA


# 8.5. Stable identity
# -----------------------------------------------------------------------------

def interaction_fingerprint_data(
    *,
    family: Any,
    interaction_type: Any,
    subtype: Any = "",
    pose_id: Any = None,
    model_id: Any = None,
    ligand_atom: Any = "",
    receptor_atom: Any = "",
    ligand_residue: Any = "",
    receptor_residue: Any = "",
    chain_id: Any = "",
    distance: Any = None,
    angle: Any = None,
) -> Tuple[Any, ...]:
    """Return canonical identity data for one interaction."""

    normalized_family = normalize_interaction_family(family)
    numeric_distance = to_finite_float(distance, None)
    numeric_angle = to_finite_float(angle, None)
    return (
        normalized_family.value,
        normalize_field_name(interaction_type),
        normalize_field_name(subtype),
        pose_id,
        model_id,
        normalize_field_name(ligand_atom),
        normalize_field_name(receptor_atom),
        normalize_field_name(ligand_residue),
        normalize_field_name(receptor_residue),
        normalize_field_name(chain_id),
        None if numeric_distance is None else round(numeric_distance, 6),
        None if numeric_angle is None else round(numeric_angle, 6),
    )


def make_interaction_id(
    fingerprint: Sequence[Any],
    *,
    prefix: str = "int",
    length: int = 16,
) -> str:
    """Create a deterministic short interaction identifier."""

    payload = json.dumps(
        list(fingerprint),
        ensure_ascii=True,
        sort_keys=False,
        default=str,
        separators=DEFAULT_JSON_COMPACT_SEPARATORS,
    )
    digest = sha256(payload.encode(DEFAULT_ENCODING)).hexdigest()
    return f"{safe_identifier(prefix, 'int')}-{digest[:max(4, int(length))]}"


# 8.6. Single-interaction normalization
# -----------------------------------------------------------------------------

_INTERACTION_CONSUMED_FIELDS: Final[Tuple[str, ...]] = (
    *NORMALIZED_INTERACTION_FIELDS,
    "interaction_family",
    "interaction_type",
    "interaction_subtype",
    "kind",
    "category",
    "name",
    "atom1",
    "atom2",
    "first_atom",
    "second_atom",
    "protein_atom",
    "residue1",
    "residue2",
    "first_residue",
    "second_residue",
    "protein_residue",
    "distance_angstrom",
    "distance_a",
    "separation",
    "centroid_distance",
    "minimum_distance",
    "angle_degrees",
    "theta",
    "geometry_class",
    "weighted_score",
)


def normalize_interaction(
    interaction: Any,
    *,
    family_hint: Any = None,
    pose_id: Any = None,
    model_id: Any = None,
    source: Optional[str] = None,
    include_metadata: bool = False,
    strict: bool = False,
) -> NormalizedInteraction:
    """Normalize one heterogeneous interaction object."""

    if isinstance(interaction, NormalizedInteraction):
        updates: Dict[str, Any] = {}
        if pose_id is not None and interaction.pose_id is None:
            updates["pose_id"] = pose_id
        if model_id is not None and interaction.model_id is None:
            updates["model_id"] = model_id
        if source and not interaction.source:
            updates["source"] = source
        return replace(interaction, **updates) if updates else interaction

    if interaction is None or interaction is MISSING:
        raise InteractionNormalizationError(
            "Interaction value is missing."
        )

    try:
        family = infer_interaction_family(
            interaction,
            family_hint=family_hint,
        )
        interaction_type = normalize_interaction_type(
            interaction,
            family,
        )
        subtype = normalize_interaction_subtype(interaction)

        ligand_atom_obj = _interaction_atom(interaction, "ligand")
        receptor_atom_obj = _interaction_atom(interaction, "receptor")
        ligand_residue_obj = _interaction_residue(
            interaction,
            "ligand",
            ligand_atom_obj,
        )
        receptor_residue_obj = _interaction_residue(
            interaction,
            "receptor",
            receptor_atom_obj,
        )

        resolved_pose_id = pose_id
        if resolved_pose_id is None:
            resolved_pose_id = get_pose_identifier(interaction, None)

        resolved_model_id = model_id
        if resolved_model_id is None:
            resolved_model_id = get_model_identifier(interaction, None)

        resolved_source = source
        if not resolved_source:
            resolved_source = get_first_object_field(
                interaction,
                ("source", "module", "detector", "origin"),
                "",
                skip_none=True,
            )

        ligand_atom = atom_display_name(
            ligand_atom_obj,
            include_residue=False,
            missing="",
        )
        receptor_atom = atom_display_name(
            receptor_atom_obj,
            include_residue=False,
            missing="",
        )
        ligand_residue = residue_display_name(
            ligand_residue_obj,
            include_chain=True,
            missing="",
        )
        receptor_residue = residue_display_name(
            receptor_residue_obj,
            include_chain=False,
            missing="",
        )
        chain_id = _interaction_chain_id(
            interaction,
            receptor_residue_obj,
        )
        distance = _interaction_distance(interaction)
        angle = _interaction_angle(interaction)
        score = _interaction_score(interaction)
        strength = _interaction_strength(interaction)
        classification = _interaction_classification(interaction)

        fingerprint = interaction_fingerprint_data(
            family=family,
            interaction_type=interaction_type,
            subtype=subtype,
            pose_id=resolved_pose_id,
            model_id=resolved_model_id,
            ligand_atom=ligand_atom,
            receptor_atom=receptor_atom,
            ligand_residue=ligand_residue,
            receptor_residue=receptor_residue,
            chain_id=chain_id,
            distance=distance,
            angle=angle,
        )
        explicit_id = get_first_object_field(
            interaction,
            ("interaction_id", "id", "uid"),
            None,
            skip_none=True,
        )
        interaction_id = (
            safe_identifier(explicit_id, "")
            if explicit_id is not None
            else make_interaction_id(fingerprint, prefix=family.value)
        )

        metadata = _interaction_metadata(
            interaction,
            consumed_fields=_INTERACTION_CONSUMED_FIELDS,
            include_unmapped=include_metadata,
        )

        return NormalizedInteraction(
            id=interaction_id,
            family=family,
            type=interaction_type,
            subtype=subtype,
            source=safe_identifier(resolved_source, ""),
            pose_id=resolved_pose_id,
            model_id=resolved_model_id,
            ligand_atom=ligand_atom,
            receptor_atom=receptor_atom,
            ligand_residue=ligand_residue,
            receptor_residue=receptor_residue,
            chain_id=chain_id,
            distance=distance,
            angle=angle,
            strength=strength,
            classification=classification,
            score=score,
            metadata=metadata,
        )
    except ReportError:
        raise
    except Exception as error:
        wrapped = InteractionNormalizationError(
            "Unable to normalize interaction.",
            context={
                "interaction_type": type(interaction).__name__,
                "family_hint": family_hint,
            },
            cause=error,
        )
        if strict:
            raise wrapped from error
        raise wrapped from error


# 8.7. Collection normalization
# -----------------------------------------------------------------------------

def interaction_sort_key(
    interaction: NormalizedInteraction,
) -> Tuple[Any, ...]:
    """Return a stable interaction sort key."""

    return (
        interaction.pose_id is None,
        safe_string(interaction.pose_id, ""),
        interaction.family.value,
        interaction.chain_id,
        interaction.receptor_residue,
        interaction.ligand_residue,
        interaction.type,
        interaction.subtype,
        math.inf if interaction.distance is None else interaction.distance,
        interaction.id,
    )


def deduplicate_interactions(
    interactions: Iterable[NormalizedInteraction],
) -> List[NormalizedInteraction]:
    """Remove duplicate normalized interactions."""

    seen: Set[Tuple[Any, ...]] = set()
    output: List[NormalizedInteraction] = []
    for interaction in interactions:
        key = interaction.fingerprint()
        if key in seen:
            continue
        seen.add(key)
        output.append(interaction)
    return output


def normalize_interactions(
    interactions: Any,
    *,
    family_hint: Any = None,
    pose_id: Any = None,
    model_id: Any = None,
    source: Optional[str] = None,
    config: InteractionReportConfig = DEFAULT_INTERACTION_REPORT_CONFIG,
    strict: bool = False,
    errors: Optional[List[ReportError]] = None,
) -> List[NormalizedInteraction]:
    """Normalize a heterogeneous interaction collection."""

    output: List[NormalizedInteraction] = []
    allowed = set(config.families)

    for raw in iter_object_collection(interactions):
        try:
            normalized = normalize_interaction(
                raw,
                family_hint=family_hint,
                pose_id=pose_id,
                model_id=model_id,
                source=source,
                include_metadata=config.include_metadata,
                strict=strict,
            )
        except ReportError as error:
            if strict:
                raise
            if errors is not None:
                errors.append(error)
            continue

        if (
            normalized.family is InteractionFamily.UNKNOWN
            and not config.include_unknown
        ):
            continue
        if normalized.family not in allowed:
            continue
        output.append(normalized)
        if len(output) >= config.max_interactions:
            break

    if config.deduplicate:
        output = deduplicate_interactions(output)
    if config.sort_interactions:
        output.sort(key=interaction_sort_key)
    return output


def normalize_interaction_containers(
    obj: Any,
    *,
    pose_id: Any = None,
    model_id: Any = None,
    config: InteractionReportConfig = DEFAULT_INTERACTION_REPORT_CONFIG,
    strict: bool = False,
    errors: Optional[List[ReportError]] = None,
) -> List[NormalizedInteraction]:
    """Normalize all recognized interaction containers on an object."""

    resolved_pose_id = (
        get_pose_identifier(obj, None) if pose_id is None else pose_id
    )
    resolved_model_id = (
        get_model_identifier(obj, None) if model_id is None else model_id
    )

    output: List[NormalizedInteraction] = []
    for family, container in iter_interaction_containers(
        obj,
        include_empty=config.include_empty_families,
    ):
        remaining = config.max_interactions - len(output)
        if remaining <= 0:
            break
        local_config = replace(config, max_interactions=remaining)
        output.extend(
            normalize_interactions(
                container,
                family_hint=family,
                pose_id=resolved_pose_id,
                model_id=resolved_model_id,
                source=family.value,
                config=local_config,
                strict=strict,
                errors=errors,
            )
        )

    if config.deduplicate:
        output = deduplicate_interactions(output)
    if config.sort_interactions:
        output.sort(key=interaction_sort_key)
    return output[: config.max_interactions]


def normalize_interaction_input(
    value: Any,
    *,
    config: InteractionReportConfig = DEFAULT_INTERACTION_REPORT_CONFIG,
    strict: bool = False,
    errors: Optional[List[ReportError]] = None,
) -> List[NormalizedInteraction]:
    """Normalize direct collections, family mappings or model-like objects."""

    if value is None or value is MISSING:
        return []

    if isinstance(value, Mapping):
        recognized: List[Tuple[InteractionFamily, Any]] = []
        for key, item in value.items():
            family = normalize_interaction_family(key)
            if family is not InteractionFamily.UNKNOWN:
                recognized.append((family, item))
        if recognized:
            output: List[NormalizedInteraction] = []
            for family, item in recognized:
                output.extend(
                    normalize_interactions(
                        item,
                        family_hint=family,
                        config=replace(
                            config,
                            max_interactions=max(
                                1,
                                config.max_interactions - len(output),
                            ),
                        ),
                        strict=strict,
                        errors=errors,
                    )
                )
                if len(output) >= config.max_interactions:
                    break
            if config.deduplicate:
                output = deduplicate_interactions(output)
            if config.sort_interactions:
                output.sort(key=interaction_sort_key)
            return output[: config.max_interactions]

    containers = list(iter_interaction_containers(value))
    if containers:
        return normalize_interaction_containers(
            value,
            config=config,
            strict=strict,
            errors=errors,
        )

    if is_sequence_like(value) or isinstance(value, (set, frozenset)):
        return normalize_interactions(
            value,
            config=config,
            strict=strict,
            errors=errors,
        )

    return normalize_interactions(
        (value,),
        config=config,
        strict=strict,
        errors=errors,
    )


# 8.8. Interaction grouping and statistics
# -----------------------------------------------------------------------------

def group_interactions_by_family(
    interactions: Iterable[NormalizedInteraction],
) -> Dict[InteractionFamily, List[NormalizedInteraction]]:
    """Group interactions by family."""

    grouped: Dict[InteractionFamily, List[NormalizedInteraction]] = {
        family: [] for family in InteractionFamily
    }
    for interaction in interactions:
        grouped.setdefault(interaction.family, []).append(interaction)
    return {
        family: values
        for family, values in grouped.items()
        if values
    }


def interaction_family_counts(
    interactions: Iterable[NormalizedInteraction],
) -> Dict[str, int]:
    """Count interactions by canonical family."""

    counts = Counter(
        interaction.family.value
        for interaction in interactions
    )
    return {
        family.value: counts.get(family.value, 0)
        for family in InteractionFamily
        if counts.get(family.value, 0)
    }


def interaction_type_counts(
    interactions: Iterable[NormalizedInteraction],
) -> Dict[str, int]:
    """Count interactions by normalized type."""

    counts = Counter(interaction.type for interaction in interactions)
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def interaction_residue_counts(
    interactions: Iterable[NormalizedInteraction],
) -> Dict[str, int]:
    """Count interactions by receptor residue."""

    counts = Counter(
        interaction.receptor_residue
        for interaction in interactions
        if interaction.receptor_residue
    )
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def interaction_distance_statistics(
    interactions: Iterable[NormalizedInteraction],
) -> Dict[str, Optional[float]]:
    """Summarize available interaction distances."""

    values = [
        interaction.distance
        for interaction in interactions
        if interaction.distance is not None
    ]
    if not values:
        return {
            "count": 0,
            "minimum": None,
            "maximum": None,
            "mean": None,
            "median": None,
        }
    return {
        "count": len(values),
        "minimum": min(values),
        "maximum": max(values),
        "mean": fmean(values),
        "median": median(values),
    }


def normalized_interactions_to_dicts(
    interactions: Iterable[NormalizedInteraction],
    *,
    include_metadata: bool = True,
    include_empty: bool = False,
) -> List[Dict[str, Any]]:
    """Convert normalized interactions into dictionaries."""

    return [
        interaction.to_dict(
            include_metadata=include_metadata,
            include_empty=include_empty,
        )
        for interaction in interactions
    ]


# 8.9. Public normalization interface
# -----------------------------------------------------------------------------

_SECTION_8_PUBLIC_NAMES: Final[Tuple[str, ...]] = (
    "NormalizedInteraction",
    "normalize_interaction_family",
    "infer_interaction_family",
    "normalize_interaction_type",
    "normalize_interaction_subtype",
    "interaction_fingerprint_data",
    "make_interaction_id",
    "normalize_interaction",
    "interaction_sort_key",
    "deduplicate_interactions",
    "normalize_interactions",
    "normalize_interaction_containers",
    "normalize_interaction_input",
    "group_interactions_by_family",
    "interaction_family_counts",
    "interaction_type_counts",
    "interaction_residue_counts",
    "interaction_distance_statistics",
    "normalized_interactions_to_dicts",
)

_register_public_names(_SECTION_8_PUBLIC_NAMES)

# =============================================================================
# End of Section 8
# =============================================================================


# =============================================================================
# Section 9 — Pose overview
# =============================================================================

# 9.1. Pose overview dataclass
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class PoseOverview:
    """General report summary for one docking pose."""

    pose_id: Any = None
    pose_name: str = ""
    model_id: Any = None
    model_name: str = ""
    ligand_name: str = ""
    receptor_name: str = ""
    source_path: str = ""
    affinity: Optional[float] = None
    total_score: Optional[float] = None
    interaction_count: int = 0
    residue_count: int = 0
    favorable_count: int = 0
    penalty_count: int = 0
    family_counts: Mapping[str, int] = field(
        default_factory=lambda: _EMPTY_METADATA
    )
    type_counts: Mapping[str, int] = field(
        default_factory=lambda: _EMPTY_METADATA
    )
    distance_statistics: Mapping[str, Any] = field(
        default_factory=lambda: _EMPTY_METADATA
    )
    warnings: Tuple[str, ...] = ()
    errors: Tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(
        default_factory=lambda: _EMPTY_METADATA
    )

    def __post_init__(self) -> None:
        for name in (
            "pose_name",
            "model_name",
            "ligand_name",
            "receptor_name",
            "source_path",
        ):
            object.__setattr__(
                self,
                name,
                single_line_text(getattr(self, name), ""),
            )
        object.__setattr__(
            self,
            "affinity",
            to_finite_float(self.affinity, None),
        )
        object.__setattr__(
            self,
            "total_score",
            to_finite_float(self.total_score, None),
        )
        for name in (
            "interaction_count",
            "residue_count",
            "favorable_count",
            "penalty_count",
        ):
            object.__setattr__(
                self,
                name,
                max(0, to_safe_int(getattr(self, name), 0)),
            )
        object.__setattr__(
            self,
            "family_counts",
            MappingProxyType(
                {
                    str(key): max(0, to_safe_int(value, 0))
                    for key, value in dict(self.family_counts).items()
                }
            ),
        )
        object.__setattr__(
            self,
            "type_counts",
            MappingProxyType(
                {
                    str(key): max(0, to_safe_int(value, 0))
                    for key, value in dict(self.type_counts).items()
                }
            ),
        )
        object.__setattr__(
            self,
            "distance_statistics",
            _freeze_config_mapping(self.distance_statistics),
        )
        object.__setattr__(
            self,
            "warnings",
            _freeze_config_strings(self.warnings, unique=False),
        )
        object.__setattr__(
            self,
            "errors",
            _freeze_config_strings(self.errors, unique=False),
        )
        object.__setattr__(
            self,
            "metadata",
            _freeze_config_mapping(self.metadata),
        )

    def to_dict(
        self,
        *,
        include_empty: bool = False,
    ) -> Dict[str, Any]:
        """Return a plain overview dictionary."""

        record: Dict[str, Any] = {
            KEY_POSE_ID: self.pose_id,
            "pose_name": self.pose_name,
            KEY_MODEL_ID: self.model_id,
            "model_name": self.model_name,
            "ligand_name": self.ligand_name,
            "receptor_name": self.receptor_name,
            "source_path": self.source_path,
            KEY_AFFINITY: self.affinity,
            KEY_TOTAL_SCORE: self.total_score,
            KEY_TOTAL_INTERACTIONS: self.interaction_count,
            KEY_TOTAL_RESIDUES: self.residue_count,
            "favorable_interactions": self.favorable_count,
            "penalty_interactions": self.penalty_count,
            "interaction_families": dict(self.family_counts),
            "interaction_types": dict(self.type_counts),
            "distance_statistics": dict(self.distance_statistics),
            KEY_WARNINGS: list(self.warnings),
            KEY_ERRORS: list(self.errors),
            KEY_METADATA: dict(self.metadata),
        }
        if include_empty:
            return record
        return {
            key: value
            for key, value in record.items()
            if value not in (None, "", (), [], {})
        }


# 9.2. Pose identity extraction
# -----------------------------------------------------------------------------

def get_pose_name(obj: Any, default: str = "") -> str:
    """Return a pose display name."""

    value = get_first_object_field(
        obj,
        ("pose_name", "pose_label", "name", "title"),
        default,
        skip_none=True,
    )
    return single_line_text(value, default)


def get_model_name(obj: Any, default: str = "") -> str:
    """Return a model display name."""

    value = get_first_object_field(
        obj,
        ("model_name", "structure_name", "name", "title"),
        default,
        skip_none=True,
    )
    return single_line_text(value, default)


def get_ligand_name(obj: Any, default: str = "") -> str:
    """Return a ligand display name."""

    ligand = get_first_object_field(
        obj,
        ("ligand", "ligand_model", "small_molecule"),
        None,
        skip_none=True,
    )
    value = get_first_object_field(
        obj,
        ("ligand_name", "compound_name", "ligand_id"),
        None,
        skip_none=True,
    )
    if value is None and ligand is not None:
        value = get_first_object_field(
            ligand,
            ("name", "title", "id"),
            None,
            skip_none=True,
        )
    return single_line_text(value, default)


def get_receptor_name(obj: Any, default: str = "") -> str:
    """Return a receptor display name."""

    receptor = get_first_object_field(
        obj,
        ("receptor", "target", "protein"),
        None,
        skip_none=True,
    )
    value = get_first_object_field(
        obj,
        ("receptor_name", "target_name", "protein_name"),
        None,
        skip_none=True,
    )
    if value is None and receptor is not None:
        value = get_first_object_field(
            receptor,
            ("name", "title", "id"),
            None,
            skip_none=True,
        )
    return single_line_text(value, default)


def get_pose_source_path(obj: Any, default: str = "") -> str:
    """Return the primary pose input path."""

    value = get_first_object_field(
        obj,
        (
            "source_path",
            "file_path",
            "filepath",
            "path",
            "input_path",
            "source_file",
            "filename",
        ),
        None,
        skip_none=True,
    )
    return format_path(value, missing=default) if value is not None else default


# 9.3. Pose score and affinity extraction
# -----------------------------------------------------------------------------

def get_pose_affinity(obj: Any, default: Any = None) -> Any:
    """Return an external docking affinity."""

    value = get_first_object_field(
        obj,
        FIELD_ALIASES["affinity"],
        MISSING,
        skip_none=True,
    )
    if value is MISSING:
        metadata = get_object_metadata(obj)
        value = get_first_object_field(
            metadata,
            FIELD_ALIASES["affinity"],
            MISSING,
            skip_none=True,
        )
    return to_finite_float(value, default)


def get_pose_total_score(obj: Any, default: Any = None) -> Any:
    """Return an existing aggregate score."""

    direct = get_first_object_field(
        obj,
        (
            "total_score",
            "score",
            "combined_score",
            "aggregate_score",
            "normalized_score",
        ),
        MISSING,
        skip_none=True,
    )
    if direct is not MISSING:
        numeric = to_finite_float(direct, MISSING)
        if numeric is not MISSING:
            return numeric

    scoring = get_first_object_field(
        obj,
        FIELD_ALIASES["scoring"],
        MISSING,
        skip_none=True,
    )
    if scoring is not MISSING:
        nested = get_first_object_field(
            scoring,
            (
                "total_score",
                "score",
                "combined_score",
                "normalized_score",
            ),
            MISSING,
            skip_none=True,
        )
        numeric = to_finite_float(nested, MISSING)
        if numeric is not MISSING:
            return numeric
    return default


# 9.4. Warning, error and metadata extraction
# -----------------------------------------------------------------------------

def _message_texts(value: Any) -> Tuple[str, ...]:
    """Normalize warning or error messages."""

    messages: List[str] = []
    for item in iter_object_collection(value):
        if isinstance(item, BaseException):
            text = _exception_message(item)
        elif isinstance(item, Mapping):
            text = get_first_object_field(
                item,
                ("message", "text", "detail"),
                item,
                skip_none=True,
            )
            text = single_line_text(text, "")
        else:
            text = single_line_text(item, "")
        if text:
            messages.append(text)
    return tuple(messages)


def get_pose_warnings(obj: Any) -> Tuple[str, ...]:
    """Return pose warning messages."""

    value = get_first_object_field(
        obj,
        FIELD_ALIASES["warnings"],
        (),
        skip_none=True,
    )
    return _message_texts(value)


def get_pose_errors(obj: Any) -> Tuple[str, ...]:
    """Return pose error messages."""

    value = get_first_object_field(
        obj,
        FIELD_ALIASES["errors"],
        (),
        skip_none=True,
    )
    return _message_texts(value)


def pose_overview_metadata(obj: Any) -> Mapping[str, Any]:
    """Return selected non-structural pose metadata."""

    metadata = dict(get_object_metadata(obj))
    for key in (
        "method",
        "engine",
        "docking_engine",
        "exhaustiveness",
        "seed",
        "replicate",
        "run_id",
        "timestamp",
        "notes",
    ):
        value = get_object_field(obj, key, MISSING)
        if value is not MISSING and value is not None:
            metadata.setdefault(key, value)
    return MappingProxyType(metadata) if metadata else _EMPTY_METADATA


# 9.5. Pose overview construction
# -----------------------------------------------------------------------------

def build_pose_overview(
    pose: Any,
    *,
    interactions: Optional[Iterable[NormalizedInteraction]] = None,
    interaction_config: InteractionReportConfig = (
        DEFAULT_INTERACTION_REPORT_CONFIG
    ),
    strict: bool = False,
    include_metadata: bool = True,
) -> PoseOverview:
    """Build the general summary for one pose."""

    normalization_errors: List[ReportError] = []
    if interactions is None:
        normalized = normalize_interaction_input(
            pose,
            config=interaction_config,
            strict=strict,
            errors=normalization_errors,
        )
    else:
        normalized = [
            item
            if isinstance(item, NormalizedInteraction)
            else normalize_interaction(
                item,
                pose_id=get_pose_identifier(pose, None),
                model_id=get_model_identifier(pose, None),
                include_metadata=interaction_config.include_metadata,
                strict=strict,
            )
            for item in interactions
        ]
        if interaction_config.deduplicate:
            normalized = deduplicate_interactions(normalized)
        if interaction_config.sort_interactions:
            normalized.sort(key=interaction_sort_key)

    family_counts = interaction_family_counts(normalized)
    type_counts = interaction_type_counts(normalized)
    residues = {
        (
            interaction.chain_id,
            interaction.receptor_residue,
        )
        for interaction in normalized
        if interaction.receptor_residue
    }
    favorable_count = sum(
        1 for interaction in normalized if interaction.family.favorable
    )
    penalty_count = sum(
        1 for interaction in normalized if interaction.family.penalty
    )

    errors = list(get_pose_errors(pose))
    errors.extend(str(error) for error in normalization_errors)

    return PoseOverview(
        pose_id=get_pose_identifier(pose, None),
        pose_name=get_pose_name(pose),
        model_id=get_model_identifier(pose, None),
        model_name=get_model_name(pose),
        ligand_name=get_ligand_name(pose),
        receptor_name=get_receptor_name(pose),
        source_path=get_pose_source_path(pose),
        affinity=get_pose_affinity(pose, None),
        total_score=get_pose_total_score(pose, None),
        interaction_count=len(normalized),
        residue_count=len(residues),
        favorable_count=favorable_count,
        penalty_count=penalty_count,
        family_counts=family_counts,
        type_counts=type_counts,
        distance_statistics=interaction_distance_statistics(normalized),
        warnings=get_pose_warnings(pose),
        errors=tuple(errors),
        metadata=pose_overview_metadata(pose) if include_metadata else {},
    )


def summarize_pose(
    pose: Any,
    *,
    interactions: Optional[Iterable[NormalizedInteraction]] = None,
    config: ReportConfig = DEFAULT_REPORT_CONFIG,
    strict: Optional[bool] = None,
) -> PoseOverview:
    """Convenience wrapper for a pose overview."""

    if strict is None:
        strict = config.errors.mode is ErrorMode.RAISE
    return build_pose_overview(
        pose,
        interactions=interactions,
        interaction_config=config.interactions,
        strict=strict,
        include_metadata=(
            config.rendering.detail
            in {ReportDetail.DETAILED, ReportDetail.FULL}
        ),
    )


def pose_overview_to_dict(
    overview: Union[PoseOverview, Any],
    *,
    include_empty: bool = False,
    config: ReportConfig = DEFAULT_REPORT_CONFIG,
) -> Dict[str, Any]:
    """Convert a pose or overview to a dictionary."""

    if not isinstance(overview, PoseOverview):
        overview = summarize_pose(overview, config=config)
    return overview.to_dict(include_empty=include_empty)


def pose_overview_rows(
    overview: Union[PoseOverview, Any],
    *,
    config: ReportConfig = DEFAULT_REPORT_CONFIG,
) -> ReportRows:
    """Convert a pose overview into key-value rows."""

    if not isinstance(overview, PoseOverview):
        overview = summarize_pose(overview, config=config)

    record = overview.to_dict(include_empty=False)
    preferred = (
        KEY_POSE_ID,
        "pose_name",
        KEY_MODEL_ID,
        "model_name",
        "ligand_name",
        "receptor_name",
        "source_path",
        KEY_AFFINITY,
        KEY_TOTAL_SCORE,
        KEY_TOTAL_INTERACTIONS,
        KEY_TOTAL_RESIDUES,
        "favorable_interactions",
        "penalty_interactions",
    )
    rows: ReportRows = []
    for key in preferred:
        if key not in record:
            continue
        rows.append(
            {
                "key": key,
                "label": field_label(key),
                "value": record[key],
                "formatted": format_value(
                    record[key],
                    config=config.formatting,
                ),
            }
        )
    return rows


def pose_overview_text(
    overview: Union[PoseOverview, Any],
    *,
    config: ReportConfig = DEFAULT_REPORT_CONFIG,
) -> str:
    """Return a compact plain-text pose summary."""

    rows = pose_overview_rows(overview, config=config)
    return config.rendering.newline.join(
        f"{row['label']}{TEXT_KEY_VALUE_SEPARATOR}{row['formatted']}"
        for row in rows
    )


# 9.6. Public pose-overview interface
# -----------------------------------------------------------------------------

_SECTION_9_PUBLIC_NAMES: Final[Tuple[str, ...]] = (
    "PoseOverview",
    "get_pose_name",
    "get_model_name",
    "get_ligand_name",
    "get_receptor_name",
    "get_pose_source_path",
    "get_pose_affinity",
    "get_pose_total_score",
    "get_pose_warnings",
    "get_pose_errors",
    "pose_overview_metadata",
    "build_pose_overview",
    "summarize_pose",
    "pose_overview_to_dict",
    "pose_overview_rows",
    "pose_overview_text",
)

_register_public_names(_SECTION_9_PUBLIC_NAMES)

# =============================================================================
# End of Section 9
# =============================================================================

# =============================================================================
# Section 10 — Input summary
# =============================================================================

# 10.1. Input records
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class InputRecord:
    """Normalized description of one report input."""

    role: str
    name: str = ""
    identifier: Any = None
    path: str = ""
    format: str = ""
    object_type: str = ""
    exists: Optional[bool] = None
    size_bytes: Optional[int] = None
    modified_at: Optional[str] = None
    model_id: Any = None
    atom_count: Optional[int] = None
    residue_count: Optional[int] = None
    metadata: Mapping[str, Any] = field(
        default_factory=lambda: _EMPTY_METADATA
    )

    def __post_init__(self) -> None:
        for name in (
            "role",
            "name",
            "path",
            "format",
            "object_type",
            "modified_at",
        ):
            object.__setattr__(
                self,
                name,
                single_line_text(getattr(self, name), ""),
            )
        if self.exists is not None:
            object.__setattr__(self, "exists", bool(self.exists))
        object.__setattr__(
            self,
            "size_bytes",
            to_safe_int(self.size_bytes, None),
        )
        object.__setattr__(
            self,
            "atom_count",
            to_safe_int(self.atom_count, None),
        )
        object.__setattr__(
            self,
            "residue_count",
            to_safe_int(self.residue_count, None),
        )
        object.__setattr__(
            self,
            "metadata",
            _freeze_config_mapping(self.metadata),
        )

    def to_dict(self, *, include_empty: bool = False) -> Dict[str, Any]:
        """Return a plain input record."""

        record: Dict[str, Any] = {
            "role": self.role,
            "name": self.name,
            KEY_ID: self.identifier,
            "path": self.path,
            "format": self.format,
            "object_type": self.object_type,
            "exists": self.exists,
            "size_bytes": self.size_bytes,
            "modified_at": self.modified_at,
            KEY_MODEL_ID: self.model_id,
            "atom_count": self.atom_count,
            "residue_count": self.residue_count,
            KEY_METADATA: dict(self.metadata),
        }
        if include_empty:
            return record
        return {
            key: value
            for key, value in record.items()
            if value not in (None, "", (), [], {})
        }


@dataclass(frozen=True)
class InputSummary:
    """Summary of all report inputs."""

    records: Tuple[InputRecord, ...] = ()
    total_inputs: int = 0
    existing_paths: int = 0
    missing_paths: int = 0
    total_size_bytes: int = 0
    formats: Mapping[str, int] = field(
        default_factory=lambda: _EMPTY_METADATA
    )
    roles: Mapping[str, int] = field(
        default_factory=lambda: _EMPTY_METADATA
    )
    warnings: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "records", tuple(self.records))
        for name in (
            "total_inputs",
            "existing_paths",
            "missing_paths",
            "total_size_bytes",
        ):
            object.__setattr__(
                self,
                name,
                max(0, to_safe_int(getattr(self, name), 0)),
            )
        object.__setattr__(
            self,
            "formats",
            MappingProxyType(
                {
                    str(key): max(0, to_safe_int(value, 0))
                    for key, value in dict(self.formats).items()
                }
            ),
        )
        object.__setattr__(
            self,
            "roles",
            MappingProxyType(
                {
                    str(key): max(0, to_safe_int(value, 0))
                    for key, value in dict(self.roles).items()
                }
            ),
        )
        object.__setattr__(
            self,
            "warnings",
            _freeze_config_strings(self.warnings, unique=False),
        )

    def to_dict(self, *, include_empty: bool = False) -> Dict[str, Any]:
        """Return a plain input summary."""

        record: Dict[str, Any] = {
            "records": [
                item.to_dict(include_empty=include_empty)
                for item in self.records
            ],
            "total_inputs": self.total_inputs,
            "existing_paths": self.existing_paths,
            "missing_paths": self.missing_paths,
            "total_size_bytes": self.total_size_bytes,
            "formats": dict(self.formats),
            "roles": dict(self.roles),
            KEY_WARNINGS: list(self.warnings),
        }
        if include_empty:
            return record
        return {
            key: value
            for key, value in record.items()
            if value not in (None, "", (), [], {})
        }


# 10.2. Input-role inference
# -----------------------------------------------------------------------------

_INPUT_ROLE_ALIASES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "ligand": "ligand",
        "compound": "ligand",
        "small_molecule": "ligand",
        "receptor": "receptor",
        "target": "receptor",
        "protein": "receptor",
        "pose": "pose",
        "dock_pose": "pose",
        "model": "model",
        "structure": "model",
        "input": "input",
        "file": "input",
        "unknown": "input",
    }
)


def normalize_input_role(
    value: Any,
    *,
    default: str = "input",
) -> str:
    """Normalize an input role."""

    token = normalize_field_name(value)
    return _INPUT_ROLE_ALIASES.get(token, token or default)


def infer_input_role(
    value: Any,
    *,
    role_hint: Any = None,
) -> str:
    """Infer the role of a report input."""

    if role_hint is not None:
        return normalize_input_role(role_hint)

    explicit = get_first_object_field(
        value,
        ("role", "input_role", "kind", "category"),
        None,
        skip_none=True,
    )
    if explicit is not None:
        token = normalize_input_role(explicit)
        if token != "input":
            return token

    class_name = normalize_field_name(type(value).__name__)
    for token, role in (
        ("ligand", "ligand"),
        ("compound", "ligand"),
        ("receptor", "receptor"),
        ("protein", "receptor"),
        ("pose", "pose"),
        ("model", "model"),
        ("structure", "model"),
    ):
        if token in class_name:
            return role

    return "input"


# 10.3. Path and format extraction
# -----------------------------------------------------------------------------

_INPUT_PATH_FIELDS: Final[Tuple[str, ...]] = (
    "source_path",
    "file_path",
    "filepath",
    "path",
    "input_path",
    "source_file",
    "filename",
    "file_name",
)

_INPUT_NAME_FIELDS: Final[Tuple[str, ...]] = (
    "input_name",
    "name",
    "title",
    "label",
    "filename",
)

_INPUT_ID_FIELDS: Final[Tuple[str, ...]] = (
    "input_id",
    "identifier",
    "id",
    "uid",
)

_FORMAT_ALIASES: Final[Mapping[str, str]] = MappingProxyType(
    {
        ".pdb": "pdb",
        ".pdbqt": "pdbqt",
        ".mol": "mol",
        ".mol2": "mol2",
        ".sdf": "sdf",
        ".json": "json",
        ".csv": "csv",
        ".tsv": "tsv",
        ".txt": "text",
        ".md": "markdown",
        ".html": "html",
    }
)


def input_path_value(value: Any) -> Optional[Path]:
    """Return a path represented by an input value."""

    if isinstance(value, (str, os.PathLike, Path)):
        try:
            return Path(os.fspath(value))
        except (TypeError, ValueError):
            return None

    raw = get_first_object_field(
        value,
        _INPUT_PATH_FIELDS,
        None,
        skip_none=True,
    )
    if raw is None:
        return None
    try:
        return Path(os.fspath(raw))
    except (TypeError, ValueError):
        return None


def infer_input_format(
    value: Any,
    *,
    path: Optional[Path] = None,
) -> str:
    """Infer an input file or object format."""

    if path is not None:
        suffixes = path.suffixes
        if suffixes:
            suffix = suffixes[-1].lower()
            return _FORMAT_ALIASES.get(suffix, suffix.lstrip("."))

    if isinstance(value, (str, bytes, os.PathLike, Path)):
        return ""

    explicit = get_first_object_field(
        value,
        ("file_format", "format", "extension", "suffix"),
        None,
        skip_none=True,
    )
    if explicit is None or callable(explicit):
        return ""
    return normalize_field_name(str(explicit).lstrip("."))


def input_object_type(value: Any) -> str:
    """Return the concrete input object type."""

    if value is None:
        return "NoneType"
    return type(value).__name__


# 10.4. File metadata
# -----------------------------------------------------------------------------

def safe_path_stat(path: Optional[Path]) -> Mapping[str, Any]:
    """Return non-raising path metadata."""

    if path is None:
        return _EMPTY_METADATA
    try:
        exists = path.exists()
    except OSError:
        return MappingProxyType({"exists": None})

    data: Dict[str, Any] = {"exists": exists}
    if not exists:
        return MappingProxyType(data)

    try:
        stat = path.stat()
    except OSError:
        return MappingProxyType(data)

    data["size_bytes"] = stat.st_size
    data["modified_at"] = format_datetime(
        stat.st_mtime,
        use_utc=True,
    )
    data["is_file"] = path.is_file()
    data["is_dir"] = path.is_dir()
    return MappingProxyType(data)


def format_byte_size(
    value: Any,
    *,
    missing: str = DEFAULT_MISSING_TEXT,
    digits: int = 1,
) -> str:
    """Format a byte count using binary units."""

    size = to_finite_float(value, MISSING)
    if size is MISSING or size < 0:
        return missing

    units = ("B", "KiB", "MiB", "GiB", "TiB")
    index = 0
    while size >= 1024.0 and index < len(units) - 1:
        size /= 1024.0
        index += 1

    if index == 0:
        return f"{int(size)} {units[index]}"
    return f"{format_number(size, digits, trim_zeros=True)} {units[index]}"


# 10.5. Structural counts
# -----------------------------------------------------------------------------

def _safe_object_count(value: Any, names: Iterable[str]) -> Optional[int]:
    """Return a non-negative count from common attributes."""

    direct = get_first_object_field(
        value,
        names,
        MISSING,
        skip_none=True,
    )
    if direct is MISSING:
        return None

    numeric = to_safe_int(direct, MISSING)
    if numeric is not MISSING:
        return max(0, numeric)

    try:
        return max(0, len(direct))
    except (TypeError, AttributeError):
        return None


def input_atom_count(value: Any) -> Optional[int]:
    """Return the number of atoms represented by an input."""

    return _safe_object_count(
        value,
        (
            "atom_count",
            "num_atoms",
            "n_atoms",
            "atoms",
        ),
    )


def input_residue_count(value: Any) -> Optional[int]:
    """Return the number of residues represented by an input."""

    return _safe_object_count(
        value,
        (
            "residue_count",
            "num_residues",
            "n_residues",
            "residues",
        ),
    )


# 10.6. Single-input normalization
# -----------------------------------------------------------------------------

_INPUT_METADATA_FIELDS: Final[Tuple[str, ...]] = (
    "source",
    "origin",
    "engine",
    "method",
    "chain_id",
    "pose_id",
    "rank",
    "replicate",
    "run_id",
)


def normalize_input_record(
    value: Any,
    *,
    role_hint: Any = None,
    name_hint: Any = None,
    include_metadata: bool = True,
) -> InputRecord:
    """Normalize one report input."""

    if isinstance(value, InputRecord):
        updates: Dict[str, Any] = {}
        if role_hint is not None:
            updates["role"] = normalize_input_role(role_hint)
        if name_hint is not None and not value.name:
            updates["name"] = single_line_text(name_hint, "")
        return replace(value, **updates) if updates else value

    role = infer_input_role(value, role_hint=role_hint)
    path = input_path_value(value)
    stat = safe_path_stat(path)

    explicit_name = name_hint
    if explicit_name is None:
        explicit_name = get_first_object_field(
            value,
            _INPUT_NAME_FIELDS,
            None,
            skip_none=True,
        )
    if explicit_name is None and path is not None:
        explicit_name = path.name

    identifier = get_first_object_field(
        value,
        _INPUT_ID_FIELDS,
        None,
        skip_none=True,
    )
    model_id = get_model_identifier(value, None)

    metadata: Dict[str, Any] = {}
    if include_metadata:
        metadata.update(get_object_metadata(value))
        for field_name in _INPUT_METADATA_FIELDS:
            field_value = get_object_field(value, field_name, MISSING)
            if field_value is not MISSING and field_value is not None:
                metadata.setdefault(field_name, field_value)
        if stat:
            for key in ("is_file", "is_dir"):
                if key in stat:
                    metadata.setdefault(key, stat[key])

    return InputRecord(
        role=role,
        name=single_line_text(explicit_name, ""),
        identifier=identifier,
        path=str(path) if path is not None else "",
        format=infer_input_format(value, path=path),
        object_type=input_object_type(value),
        exists=stat.get("exists") if stat else None,
        size_bytes=stat.get("size_bytes") if stat else None,
        modified_at=stat.get("modified_at") if stat else None,
        model_id=model_id,
        atom_count=input_atom_count(value),
        residue_count=input_residue_count(value),
        metadata=metadata,
    )


# 10.7. Input collection normalization
# -----------------------------------------------------------------------------

def _iter_named_inputs(value: Any) -> Iterator[Tuple[Optional[str], Any]]:
    """Yield optional role hints and input values."""

    if value is None or value is MISSING:
        return

    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = normalize_field_name(key)
            if normalized in {
                "ligand",
                "receptor",
                "pose",
                "model",
                "input",
            }:
                if is_sequence_like(item) and not isinstance(
                    item,
                    (str, bytes, bytearray),
                ):
                    for child in item:
                        yield normalized, child
                else:
                    yield normalized, item
            else:
                yield None, item
        return

    for item in iter_object_collection(value):
        yield None, item


def summarize_inputs(
    inputs: Any,
    *,
    include_metadata: bool = True,
) -> InputSummary:
    """Build a normalized summary of report inputs."""

    records: List[InputRecord] = []
    warnings_out: List[str] = []

    for role_hint, value in _iter_named_inputs(inputs):
        try:
            records.append(
                normalize_input_record(
                    value,
                    role_hint=role_hint,
                    include_metadata=include_metadata,
                )
            )
        except Exception as error:
            warnings_out.append(
                f"Unable to summarize input: {_exception_message(error)}"
            )

    formats = Counter(
        record.format
        for record in records
        if record.format
    )
    roles = Counter(record.role for record in records if record.role)
    existing = sum(record.exists is True for record in records)
    missing = sum(record.exists is False for record in records)
    total_size = sum(record.size_bytes or 0 for record in records)

    return InputSummary(
        records=tuple(records),
        total_inputs=len(records),
        existing_paths=existing,
        missing_paths=missing,
        total_size_bytes=total_size,
        formats=dict(formats),
        roles=dict(roles),
        warnings=tuple(warnings_out),
    )


def collect_pose_inputs(pose: Any) -> Tuple[Tuple[str, Any], ...]:
    """Collect receptor, ligand, pose and explicit input objects."""

    collected: List[Tuple[str, Any]] = []

    for role, names in (
        ("receptor", ("receptor", "target", "protein")),
        ("ligand", ("ligand", "ligand_model", "small_molecule")),
    ):
        value = get_first_object_field(
            pose,
            names,
            MISSING,
            skip_none=True,
        )
        if value is not MISSING:
            collected.append((role, value))

    explicit = get_first_object_field(
        pose,
        (
            "inputs",
            "input_files",
            "source_files",
        ),
        MISSING,
        skip_none=True,
    )
    if explicit is not MISSING:
        for role_hint, value in _iter_named_inputs(explicit):
            collected.append((role_hint or "input", value))

    source_path = input_path_value(pose)
    if source_path is not None:
        collected.append(("pose", source_path))

    return tuple(collected)


def summarize_pose_inputs(
    pose: Any,
    *,
    include_metadata: bool = True,
) -> InputSummary:
    """Summarize inputs associated with one pose."""

    records = [
        normalize_input_record(
            value,
            role_hint=role,
            include_metadata=include_metadata,
        )
        for role, value in collect_pose_inputs(pose)
    ]
    return summarize_inputs(
        records,
        include_metadata=include_metadata,
    )


def input_summary_rows(
    summary: Union[InputSummary, Any],
) -> ReportRows:
    """Convert input records to table rows."""

    if not isinstance(summary, InputSummary):
        summary = summarize_inputs(summary)

    rows: ReportRows = []
    for index, record in enumerate(summary.records, start=1):
        rows.append(
            {
                KEY_RANK: index,
                "role": record.role,
                "name": record.name,
                KEY_ID: record.identifier,
                "path": record.path,
                "format": record.format,
                "object_type": record.object_type,
                "exists": record.exists,
                "size_bytes": record.size_bytes,
                "size": (
                    format_byte_size(record.size_bytes, missing="")
                    if record.size_bytes is not None
                    else ""
                ),
                KEY_MODEL_ID: record.model_id,
                "atom_count": record.atom_count,
                "residue_count": record.residue_count,
            }
        )
    return rows


# 10.8. Public input-summary interface
# -----------------------------------------------------------------------------

_SECTION_10_PUBLIC_NAMES: Final[Tuple[str, ...]] = (
    "InputRecord",
    "InputSummary",
    "normalize_input_role",
    "infer_input_role",
    "input_path_value",
    "infer_input_format",
    "input_object_type",
    "safe_path_stat",
    "format_byte_size",
    "input_atom_count",
    "input_residue_count",
    "normalize_input_record",
    "summarize_inputs",
    "collect_pose_inputs",
    "summarize_pose_inputs",
    "input_summary_rows",
)

_register_public_names(_SECTION_10_PUBLIC_NAMES)

# =============================================================================
# End of Section 10
# =============================================================================


# =============================================================================
# Section 11 — Interaction section
# =============================================================================

# 11.1. Interaction family summaries
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class InteractionFamilySummary:
    """Summary of one interaction family."""

    family: InteractionFamily
    label: str
    count: int = 0
    percent: float = 0.0
    score_total: Optional[float] = None
    residues: int = 0
    distance_statistics: Mapping[str, Any] = field(
        default_factory=lambda: _EMPTY_METADATA
    )
    type_counts: Mapping[str, int] = field(
        default_factory=lambda: _EMPTY_METADATA
    )
    strength_counts: Mapping[str, int] = field(
        default_factory=lambda: _EMPTY_METADATA
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "family",
            _coerce_enum(
                InteractionFamily,
                self.family,
                "family",
                default=InteractionFamily.UNKNOWN,
            ),
        )
        object.__setattr__(
            self,
            "label",
            single_line_text(self.label, self.family.label),
        )
        object.__setattr__(
            self,
            "count",
            max(0, to_safe_int(self.count, 0)),
        )
        object.__setattr__(
            self,
            "percent",
            max(0.0, to_finite_float(self.percent, 0.0)),
        )
        object.__setattr__(
            self,
            "score_total",
            to_finite_float(self.score_total, None),
        )
        object.__setattr__(
            self,
            "residues",
            max(0, to_safe_int(self.residues, 0)),
        )
        object.__setattr__(
            self,
            "distance_statistics",
            _freeze_config_mapping(self.distance_statistics),
        )
        for name in ("type_counts", "strength_counts"):
            object.__setattr__(
                self,
                name,
                MappingProxyType(
                    {
                        str(key): max(0, to_safe_int(value, 0))
                        for key, value in dict(getattr(self, name)).items()
                    }
                ),
            )

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain family summary."""

        return {
            KEY_FAMILY: self.family.value,
            "label": self.label,
            KEY_COUNT: self.count,
            KEY_PERCENT: self.percent,
            KEY_TOTAL_SCORE: self.score_total,
            KEY_TOTAL_RESIDUES: self.residues,
            "distance_statistics": dict(self.distance_statistics),
            "type_counts": dict(self.type_counts),
            "strength_counts": dict(self.strength_counts),
        }


@dataclass(frozen=True)
class InteractionSection:
    """Structured interaction section for one pose or collection."""

    interactions: Tuple[NormalizedInteraction, ...] = ()
    families: Tuple[InteractionFamilySummary, ...] = ()
    total_interactions: int = 0
    favorable_interactions: int = 0
    penalty_interactions: int = 0
    total_score: Optional[float] = None
    residue_count: int = 0
    distance_statistics: Mapping[str, Any] = field(
        default_factory=lambda: _EMPTY_METADATA
    )
    warnings: Tuple[str, ...] = ()
    errors: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "interactions",
            tuple(self.interactions),
        )
        object.__setattr__(self, "families", tuple(self.families))
        for name in (
            "total_interactions",
            "favorable_interactions",
            "penalty_interactions",
            "residue_count",
        ):
            object.__setattr__(
                self,
                name,
                max(0, to_safe_int(getattr(self, name), 0)),
            )
        object.__setattr__(
            self,
            "total_score",
            to_finite_float(self.total_score, None),
        )
        object.__setattr__(
            self,
            "distance_statistics",
            _freeze_config_mapping(self.distance_statistics),
        )
        object.__setattr__(
            self,
            "warnings",
            _freeze_config_strings(self.warnings, unique=False),
        )
        object.__setattr__(
            self,
            "errors",
            _freeze_config_strings(self.errors, unique=False),
        )

    def to_dict(
        self,
        *,
        include_interactions: bool = True,
        include_metadata: bool = True,
    ) -> Dict[str, Any]:
        """Return a plain interaction section."""

        record: Dict[str, Any] = {
            KEY_TOTAL_INTERACTIONS: self.total_interactions,
            "favorable_interactions": self.favorable_interactions,
            "penalty_interactions": self.penalty_interactions,
            KEY_TOTAL_SCORE: self.total_score,
            KEY_TOTAL_RESIDUES: self.residue_count,
            "distance_statistics": dict(self.distance_statistics),
            "families": [family.to_dict() for family in self.families],
            KEY_WARNINGS: list(self.warnings),
            KEY_ERRORS: list(self.errors),
        }
        if include_interactions:
            record[KEY_INTERACTIONS] = normalized_interactions_to_dicts(
                self.interactions,
                include_metadata=include_metadata,
            )
        return record


# 11.2. Interaction summary helpers
# -----------------------------------------------------------------------------

def interaction_score_total(
    interactions: Iterable[NormalizedInteraction],
) -> Optional[float]:
    """Sum available interaction scores."""

    scores = [
        interaction.score
        for interaction in interactions
        if interaction.score is not None
    ]
    return sum(scores) if scores else None


def interaction_strength_counts(
    interactions: Iterable[NormalizedInteraction],
) -> Dict[str, int]:
    """Count qualitative interaction strengths."""

    counts = Counter(
        interaction.strength or STRENGTH_UNKNOWN
        for interaction in interactions
    )
    return dict(
        sorted(
            counts.items(),
            key=lambda item: (
                -STRENGTH_ORDER.get(item[0], -1),
                item[0],
            ),
        )
    )


def interaction_unique_residues(
    interactions: Iterable[NormalizedInteraction],
) -> Set[Tuple[str, str]]:
    """Return unique receptor residue keys."""

    return {
        (interaction.chain_id, interaction.receptor_residue)
        for interaction in interactions
        if interaction.receptor_residue
    }


def summarize_interaction_family(
    family: Any,
    interactions: Iterable[NormalizedInteraction],
    *,
    total_count: Optional[int] = None,
) -> InteractionFamilySummary:
    """Summarize one interaction family."""

    member = normalize_interaction_family(family)
    values = [
        interaction
        for interaction in interactions
        if interaction.family is member
    ]
    denominator = len(values) if total_count is None else max(0, total_count)
    percent = (
        (len(values) / denominator) * 100.0
        if denominator
        else 0.0
    )

    return InteractionFamilySummary(
        family=member,
        label=member.label,
        count=len(values),
        percent=percent,
        score_total=interaction_score_total(values),
        residues=len(interaction_unique_residues(values)),
        distance_statistics=interaction_distance_statistics(values),
        type_counts=interaction_type_counts(values),
        strength_counts=interaction_strength_counts(values),
    )


# 11.3. Section construction
# -----------------------------------------------------------------------------

def build_interaction_section(
    value: Any,
    *,
    interactions: Optional[Iterable[NormalizedInteraction]] = None,
    config: InteractionReportConfig = DEFAULT_INTERACTION_REPORT_CONFIG,
    strict: bool = False,
) -> InteractionSection:
    """Build the structured interaction section."""

    errors_out: List[ReportError] = []
    warnings_out: List[str] = []

    if interactions is None:
        normalized = normalize_interaction_input(
            value,
            config=config,
            strict=strict,
            errors=errors_out,
        )
    else:
        normalized = normalize_interactions(
            interactions,
            config=config,
            strict=strict,
            errors=errors_out,
        )

    if len(normalized) >= config.max_interactions:
        warnings_out.append(
            f"Interaction output limited to {config.max_interactions} records."
        )

    family_summaries: List[InteractionFamilySummary] = []
    for family in config.families:
        summary = summarize_interaction_family(
            family,
            normalized,
            total_count=len(normalized),
        )
        if summary.count or config.include_empty_families:
            family_summaries.append(summary)

    favorable = sum(
        1 for interaction in normalized if interaction.family.favorable
    )
    penalties = sum(
        1 for interaction in normalized if interaction.family.penalty
    )

    return InteractionSection(
        interactions=tuple(normalized),
        families=tuple(family_summaries),
        total_interactions=len(normalized),
        favorable_interactions=favorable,
        penalty_interactions=penalties,
        total_score=interaction_score_total(normalized),
        residue_count=len(interaction_unique_residues(normalized)),
        distance_statistics=interaction_distance_statistics(normalized),
        warnings=tuple(warnings_out),
        errors=tuple(str(error) for error in errors_out),
    )


def summarize_interactions(
    value: Any,
    *,
    config: ReportConfig = DEFAULT_REPORT_CONFIG,
    strict: Optional[bool] = None,
) -> InteractionSection:
    """Convenience wrapper for interaction-section construction."""

    if strict is None:
        strict = config.errors.mode is ErrorMode.RAISE
    return build_interaction_section(
        value,
        config=config.interactions,
        strict=strict,
    )


# 11.4. Interaction rows
# -----------------------------------------------------------------------------

def interaction_to_row(
    interaction: NormalizedInteraction,
    *,
    index: Optional[int] = None,
    formatting: FormattingConfig = DEFAULT_FORMATTING_CONFIG,
) -> ReportRow:
    """Convert one interaction into a table row."""

    row: ReportRow = {
        KEY_ID: interaction.id,
        KEY_POSE_ID: interaction.pose_id,
        KEY_MODEL_ID: interaction.model_id,
        KEY_FAMILY: interaction.family.value,
        KEY_TYPE: interaction.type,
        KEY_SUBTYPE: interaction.subtype,
        KEY_LIGAND_ATOM: interaction.ligand_atom,
        KEY_RECEPTOR_ATOM: interaction.receptor_atom,
        KEY_LIGAND_RESIDUE: interaction.ligand_residue,
        KEY_RECEPTOR_RESIDUE: interaction.receptor_residue,
        KEY_CHAIN_ID: interaction.chain_id,
        KEY_DISTANCE: interaction.distance,
        "distance_text": format_distance(
            interaction.distance,
            formatting,
        ),
        KEY_ANGLE: interaction.angle,
        "angle_text": format_angle(
            interaction.angle,
            formatting,
        ),
        KEY_STRENGTH: interaction.strength,
        KEY_CLASSIFICATION: interaction.classification,
        KEY_SCORE: interaction.score,
        "score_text": format_score(
            interaction.score,
            formatting,
        ),
        KEY_SOURCE: interaction.source,
    }
    if index is not None:
        row[KEY_RANK] = index
    return row


def interaction_rows(
    section: Union[InteractionSection, Iterable[NormalizedInteraction], Any],
    *,
    config: ReportConfig = DEFAULT_REPORT_CONFIG,
) -> ReportRows:
    """Return interaction table rows."""

    if isinstance(section, InteractionSection):
        interactions = section.interactions
    elif (
        is_sequence_like(section)
        and all(
            isinstance(item, NormalizedInteraction)
            for item in section
        )
    ):
        interactions = tuple(section)
    else:
        interactions = summarize_interactions(
            section,
            config=config,
        ).interactions

    return [
        interaction_to_row(
            interaction,
            index=index,
            formatting=config.formatting,
        )
        for index, interaction in enumerate(interactions, start=1)
    ]


def interaction_family_rows(
    section: Union[InteractionSection, Any],
    *,
    config: ReportConfig = DEFAULT_REPORT_CONFIG,
) -> ReportRows:
    """Return one summary row per interaction family."""

    if not isinstance(section, InteractionSection):
        section = summarize_interactions(section, config=config)

    rows: ReportRows = []
    for summary in section.families:
        rows.append(
            {
                KEY_FAMILY: summary.family.value,
                "label": summary.label,
                KEY_COUNT: summary.count,
                KEY_PERCENT: summary.percent,
                "percent_text": format_percent(
                    summary.percent,
                    config.formatting,
                ),
                KEY_TOTAL_SCORE: summary.score_total,
                KEY_TOTAL_RESIDUES: summary.residues,
                "minimum_distance": summary.distance_statistics.get("minimum"),
                "mean_distance": summary.distance_statistics.get("mean"),
                "maximum_distance": summary.distance_statistics.get("maximum"),
            }
        )
    return rows


def split_interaction_rows_by_family(
    section: Union[InteractionSection, Any],
    *,
    config: ReportConfig = DEFAULT_REPORT_CONFIG,
) -> Dict[str, ReportRows]:
    """Return interaction rows grouped by family."""

    if not isinstance(section, InteractionSection):
        section = summarize_interactions(section, config=config)

    grouped = group_interactions_by_family(section.interactions)
    return {
        family.value: [
            interaction_to_row(
                interaction,
                index=index,
                formatting=config.formatting,
            )
            for index, interaction in enumerate(values, start=1)
        ]
        for family, values in grouped.items()
    }


# 11.5. Public interaction-section interface
# -----------------------------------------------------------------------------

_SECTION_11_PUBLIC_NAMES: Final[Tuple[str, ...]] = (
    "InteractionFamilySummary",
    "InteractionSection",
    "interaction_score_total",
    "interaction_strength_counts",
    "interaction_unique_residues",
    "summarize_interaction_family",
    "build_interaction_section",
    "summarize_interactions",
    "interaction_to_row",
    "interaction_rows",
    "interaction_family_rows",
    "split_interaction_rows_by_family",
)

_register_public_names(_SECTION_11_PUBLIC_NAMES)

# =============================================================================
# End of Section 11
# =============================================================================


# =============================================================================
# Section 12 — Residue summary
# =============================================================================

# 12.1. Residue key and summary
# -----------------------------------------------------------------------------

@dataclass(frozen=True, order=True)
class ResidueKey:
    """Stable receptor residue identifier."""

    chain_id: str = ""
    residue_name: str = ""
    residue_number: str = ""
    label: str = ""

    def __post_init__(self) -> None:
        for name in (
            "chain_id",
            "residue_name",
            "residue_number",
            "label",
        ):
            object.__setattr__(
                self,
                name,
                single_line_text(getattr(self, name), ""),
            )

    @property
    def canonical(self) -> str:
        """Return a compact canonical residue key."""

        core = "".join(
            part
            for part in (self.residue_name, self.residue_number)
            if part
        )
        if self.chain_id:
            return f"{self.chain_id}:{core or self.label}"
        return core or self.label


@dataclass(frozen=True)
class ResidueSummary:
    """Aggregated interaction data for one receptor residue."""

    key: ResidueKey
    rank: Optional[int] = None
    interaction_count: int = 0
    favorable_count: int = 0
    penalty_count: int = 0
    score_total: Optional[float] = None
    score_mean: Optional[float] = None
    family_counts: Mapping[str, int] = field(
        default_factory=lambda: _EMPTY_METADATA
    )
    type_counts: Mapping[str, int] = field(
        default_factory=lambda: _EMPTY_METADATA
    )
    strength_counts: Mapping[str, int] = field(
        default_factory=lambda: _EMPTY_METADATA
    )
    ligand_atoms: Tuple[str, ...] = ()
    receptor_atoms: Tuple[str, ...] = ()
    distances: Tuple[float, ...] = ()
    distance_statistics: Mapping[str, Any] = field(
        default_factory=lambda: _EMPTY_METADATA
    )
    interaction_ids: Tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(
        default_factory=lambda: _EMPTY_METADATA
    )

    def __post_init__(self) -> None:
        if not isinstance(self.key, ResidueKey):
            raise ReportConfigurationError("key must be ResidueKey.")
        if self.rank is not None:
            object.__setattr__(
                self,
                "rank",
                max(1, to_safe_int(self.rank, 1)),
            )
        for name in (
            "interaction_count",
            "favorable_count",
            "penalty_count",
        ):
            object.__setattr__(
                self,
                name,
                max(0, to_safe_int(getattr(self, name), 0)),
            )
        object.__setattr__(
            self,
            "score_total",
            to_finite_float(self.score_total, None),
        )
        object.__setattr__(
            self,
            "score_mean",
            to_finite_float(self.score_mean, None),
        )
        for name in (
            "family_counts",
            "type_counts",
            "strength_counts",
        ):
            object.__setattr__(
                self,
                name,
                MappingProxyType(
                    {
                        str(key): max(0, to_safe_int(value, 0))
                        for key, value in dict(getattr(self, name)).items()
                    }
                ),
            )
        for name in (
            "ligand_atoms",
            "receptor_atoms",
            "interaction_ids",
        ):
            object.__setattr__(
                self,
                name,
                _freeze_config_strings(getattr(self, name)),
            )
        object.__setattr__(
            self,
            "distances",
            tuple(
                numeric
                for numeric in (
                    to_finite_float(value, None)
                    for value in self.distances
                )
                if numeric is not None
            ),
        )
        object.__setattr__(
            self,
            "distance_statistics",
            _freeze_config_mapping(self.distance_statistics),
        )
        object.__setattr__(
            self,
            "metadata",
            _freeze_config_mapping(self.metadata),
        )

    @property
    def residue(self) -> str:
        """Return the display residue label."""

        return self.key.canonical

    def to_dict(self, *, include_details: bool = True) -> Dict[str, Any]:
        """Return a plain residue summary."""

        record: Dict[str, Any] = {
            KEY_RANK: self.rank,
            KEY_RECEPTOR_RESIDUE: self.residue,
            KEY_CHAIN_ID: self.key.chain_id,
            "residue_name": self.key.residue_name,
            "residue_number": self.key.residue_number,
            KEY_COUNT: self.interaction_count,
            "favorable_interactions": self.favorable_count,
            "penalty_interactions": self.penalty_count,
            KEY_TOTAL_SCORE: self.score_total,
            "mean_score": self.score_mean,
            "family_counts": dict(self.family_counts),
            "type_counts": dict(self.type_counts),
            "strength_counts": dict(self.strength_counts),
            "distance_statistics": dict(self.distance_statistics),
        }
        if include_details:
            record.update(
                {
                    "ligand_atoms": list(self.ligand_atoms),
                    "receptor_atoms": list(self.receptor_atoms),
                    "distances": list(self.distances),
                    "interaction_ids": list(self.interaction_ids),
                    KEY_METADATA: dict(self.metadata),
                }
            )
        return record


# 12.2. Residue parsing
# -----------------------------------------------------------------------------

_RESIDUE_LABEL_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"""
    ^
    (?:
        (?P<chain>[^:]+):
    )?
    (?P<name>[A-Za-z]{1,4})?
    (?P<number>-?\d+[A-Za-z]?)?
    $
    """,
    re.VERBOSE,
)


def parse_residue_label(
    value: Any,
    *,
    chain_hint: Any = None,
) -> ResidueKey:
    """Parse a compact residue label."""

    label = single_line_text(value, "")
    chain = single_line_text(chain_hint, "")
    residue_name = ""
    residue_number = ""

    match = _RESIDUE_LABEL_PATTERN.match(label)
    if match:
        parsed_chain = match.group("chain")
        if parsed_chain:
            chain = parsed_chain
        residue_name = match.group("name") or ""
        residue_number = match.group("number") or ""

    return ResidueKey(
        chain_id=chain,
        residue_name=residue_name.upper(),
        residue_number=residue_number,
        label=label,
    )


def residue_key_from_interaction(
    interaction: NormalizedInteraction,
) -> ResidueKey:
    """Return the receptor residue key for an interaction."""

    return parse_residue_label(
        interaction.receptor_residue,
        chain_hint=interaction.chain_id,
    )


# 12.3. Residue aggregation
# -----------------------------------------------------------------------------

def group_interactions_by_residue(
    interactions: Iterable[NormalizedInteraction],
    *,
    include_unassigned: bool = False,
) -> Dict[ResidueKey, List[NormalizedInteraction]]:
    """Group normalized interactions by receptor residue."""

    grouped: Dict[ResidueKey, List[NormalizedInteraction]] = defaultdict(list)
    for interaction in interactions:
        key = residue_key_from_interaction(interaction)
        if not key.canonical and not include_unassigned:
            continue
        grouped[key].append(interaction)
    return dict(grouped)


def residue_score_values(
    interactions: Iterable[NormalizedInteraction],
) -> List[float]:
    """Return available residue interaction scores."""

    return [
        interaction.score
        for interaction in interactions
        if interaction.score is not None
    ]


def build_residue_summary(
    key: ResidueKey,
    interactions: Iterable[NormalizedInteraction],
    *,
    rank: Optional[int] = None,
    include_details: bool = True,
) -> ResidueSummary:
    """Build one residue summary."""

    values = list(interactions)
    scores = residue_score_values(values)
    distances = tuple(
        interaction.distance
        for interaction in values
        if interaction.distance is not None
    )

    metadata: Dict[str, Any] = {}
    if include_details:
        metadata["classifications"] = dict(
            Counter(
                interaction.classification
                for interaction in values
                if interaction.classification
            )
        )
        metadata["subtypes"] = dict(
            Counter(
                interaction.subtype
                for interaction in values
                if interaction.subtype
            )
        )

    return ResidueSummary(
        key=key,
        rank=rank,
        interaction_count=len(values),
        favorable_count=sum(
            interaction.family.favorable
            for interaction in values
        ),
        penalty_count=sum(
            interaction.family.penalty
            for interaction in values
        ),
        score_total=sum(scores) if scores else None,
        score_mean=fmean(scores) if scores else None,
        family_counts=interaction_family_counts(values),
        type_counts=interaction_type_counts(values),
        strength_counts=interaction_strength_counts(values),
        ligand_atoms=tuple(
            interaction.ligand_atom
            for interaction in values
            if interaction.ligand_atom
        ),
        receptor_atoms=tuple(
            interaction.receptor_atom
            for interaction in values
            if interaction.receptor_atom
        ),
        distances=distances,
        distance_statistics=interaction_distance_statistics(values),
        interaction_ids=tuple(interaction.id for interaction in values),
        metadata=metadata,
    )


def residue_summary_sort_key(
    summary: ResidueSummary,
    *,
    score_direction: RankDirection = RankDirection.HIGHER_IS_BETTER,
) -> Tuple[Any, ...]:
    """Return a ranking key for residue summaries."""

    score = summary.score_total
    if score is None:
        score_component = math.inf
    elif score_direction is RankDirection.HIGHER_IS_BETTER:
        score_component = -score
    else:
        score_component = score

    return (
        -summary.interaction_count,
        -summary.favorable_count,
        summary.penalty_count,
        score_component,
        summary.key.chain_id,
        summary.key.residue_number,
        summary.key.residue_name,
    )


def summarize_residues(
    value: Any,
    *,
    interactions: Optional[Iterable[NormalizedInteraction]] = None,
    interaction_config: InteractionReportConfig = (
        DEFAULT_INTERACTION_REPORT_CONFIG
    ),
    top_n: Optional[int] = None,
    include_unassigned: bool = False,
    include_details: bool = True,
    score_direction: RankDirection = RankDirection.HIGHER_IS_BETTER,
    strict: bool = False,
) -> List[ResidueSummary]:
    """Aggregate interactions by receptor residue."""

    if interactions is None:
        normalized = normalize_interaction_input(
            value,
            config=interaction_config,
            strict=strict,
        )
    else:
        normalized = [
            item
            if isinstance(item, NormalizedInteraction)
            else normalize_interaction(item, strict=strict)
            for item in interactions
        ]

    grouped = group_interactions_by_residue(
        normalized,
        include_unassigned=include_unassigned,
    )
    summaries = [
        build_residue_summary(
            key,
            values,
            include_details=include_details,
        )
        for key, values in grouped.items()
    ]
    summaries.sort(
        key=lambda summary: residue_summary_sort_key(
            summary,
            score_direction=score_direction,
        )
    )

    if top_n is not None:
        summaries = summaries[: max(0, int(top_n))]

    return [
        replace(summary, rank=index)
        for index, summary in enumerate(summaries, start=1)
    ]


# 12.4. Residue totals and rows
# -----------------------------------------------------------------------------

def residue_summary_totals(
    summaries: Iterable[ResidueSummary],
) -> Dict[str, Any]:
    """Return aggregate totals across residue summaries."""

    values = list(summaries)
    scores = [
        summary.score_total
        for summary in values
        if summary.score_total is not None
    ]
    return {
        KEY_TOTAL_RESIDUES: len(values),
        KEY_TOTAL_INTERACTIONS: sum(
            summary.interaction_count
            for summary in values
        ),
        "favorable_interactions": sum(
            summary.favorable_count
            for summary in values
        ),
        "penalty_interactions": sum(
            summary.penalty_count
            for summary in values
        ),
        KEY_TOTAL_SCORE: sum(scores) if scores else None,
    }


def residue_summary_to_row(
    summary: ResidueSummary,
    *,
    total_interactions: Optional[int] = None,
    formatting: FormattingConfig = DEFAULT_FORMATTING_CONFIG,
) -> ReportRow:
    """Convert one residue summary to a table row."""

    percent = (
        (summary.interaction_count / total_interactions) * 100.0
        if total_interactions
        else 0.0
    )
    minimum = summary.distance_statistics.get("minimum")
    mean_distance = summary.distance_statistics.get("mean")
    maximum = summary.distance_statistics.get("maximum")

    return {
        KEY_RANK: summary.rank,
        KEY_RECEPTOR_RESIDUE: summary.residue,
        KEY_CHAIN_ID: summary.key.chain_id,
        "residue_name": summary.key.residue_name,
        "residue_number": summary.key.residue_number,
        KEY_COUNT: summary.interaction_count,
        "favorable_interactions": summary.favorable_count,
        "penalty_interactions": summary.penalty_count,
        KEY_TOTAL_SCORE: summary.score_total,
        "score_text": format_score(
            summary.score_total,
            formatting,
        ),
        "mean_score": summary.score_mean,
        KEY_PERCENT: percent,
        "percent_text": format_percent(
            percent,
            formatting,
        ),
        "minimum_distance": minimum,
        "mean_distance": mean_distance,
        "maximum_distance": maximum,
        "distance_range": format_range(
            minimum,
            maximum,
            digits=formatting.distance_digits,
            missing=formatting.missing_text,
        ),
        "families": format_mapping(
            summary.family_counts,
            missing="",
        ),
        "types": format_mapping(
            summary.type_counts,
            missing="",
        ),
        "strengths": format_mapping(
            summary.strength_counts,
            missing="",
        ),
        "ligand_atoms": format_sequence(
            summary.ligand_atoms,
            missing="",
        ),
        "receptor_atoms": format_sequence(
            summary.receptor_atoms,
            missing="",
        ),
    }


def residue_summary_rows(
    summaries: Union[Iterable[ResidueSummary], Any],
    *,
    config: ReportConfig = DEFAULT_REPORT_CONFIG,
) -> ReportRows:
    """Return residue-summary table rows."""

    if (
        isinstance(summaries, Iterable)
        and not isinstance(summaries, (str, bytes, Mapping))
    ):
        candidate = list(summaries)
    else:
        candidate = []

    if candidate and all(
        isinstance(item, ResidueSummary)
        for item in candidate
    ):
        values = candidate
    elif not candidate and isinstance(summaries, (list, tuple, set, frozenset)):
        values = []
    else:
        values = summarize_residues(
            summaries,
            interaction_config=config.interactions,
            top_n=config.multipose.top_residues,
            score_direction=config.scoring.score_direction,
            strict=config.errors.mode is ErrorMode.RAISE,
        )

    total_interactions = sum(
        summary.interaction_count
        for summary in values
    )
    return [
        residue_summary_to_row(
            summary,
            total_interactions=total_interactions,
            formatting=config.formatting,
        )
        for summary in values
    ]


def residue_summary_map(
    summaries: Iterable[ResidueSummary],
) -> Dict[str, ResidueSummary]:
    """Index residue summaries by canonical label."""

    return {
        summary.residue: summary
        for summary in summaries
    }


# 12.5. Public residue-summary interface
# -----------------------------------------------------------------------------

_SECTION_12_PUBLIC_NAMES: Final[Tuple[str, ...]] = (
    "ResidueKey",
    "ResidueSummary",
    "parse_residue_label",
    "residue_key_from_interaction",
    "group_interactions_by_residue",
    "residue_score_values",
    "build_residue_summary",
    "residue_summary_sort_key",
    "summarize_residues",
    "residue_summary_totals",
    "residue_summary_to_row",
    "residue_summary_rows",
    "residue_summary_map",
)

_register_public_names(_SECTION_12_PUBLIC_NAMES)

# =============================================================================
# End of Section 12
# =============================================================================

# =============================================================================
# Section 13 — Hotspots
# =============================================================================

# 13.1. Hotspot records
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class HotspotSummary:
    """Ranked receptor hotspot."""

    residue: ResidueKey
    rank: Optional[int] = None
    hotspot_score: float = 0.0
    interaction_count: int = 0
    favorable_count: int = 0
    penalty_count: int = 0
    family_diversity: int = 0
    type_diversity: int = 0
    score_total: Optional[float] = None
    persistence: Optional[float] = None
    pose_count: Optional[int] = None
    total_poses: Optional[int] = None
    family_counts: Mapping[str, int] = field(
        default_factory=lambda: _EMPTY_METADATA
    )
    type_counts: Mapping[str, int] = field(
        default_factory=lambda: _EMPTY_METADATA
    )
    evidence: Tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(
        default_factory=lambda: _EMPTY_METADATA
    )

    def __post_init__(self) -> None:
        if not isinstance(self.residue, ResidueKey):
            raise ReportConfigurationError("residue must be ResidueKey.")
        if self.rank is not None:
            object.__setattr__(
                self,
                "rank",
                max(1, to_safe_int(self.rank, 1)),
            )
        object.__setattr__(
            self,
            "hotspot_score",
            to_finite_float(self.hotspot_score, 0.0),
        )
        for name in (
            "interaction_count",
            "favorable_count",
            "penalty_count",
            "family_diversity",
            "type_diversity",
        ):
            object.__setattr__(
                self,
                name,
                max(0, to_safe_int(getattr(self, name), 0)),
            )
        object.__setattr__(
            self,
            "score_total",
            to_finite_float(self.score_total, None),
        )
        object.__setattr__(
            self,
            "persistence",
            to_finite_float(self.persistence, None),
        )
        for name in ("pose_count", "total_poses"):
            value = getattr(self, name)
            object.__setattr__(
                self,
                name,
                None if value is None else max(0, to_safe_int(value, 0)),
            )
        for name in ("family_counts", "type_counts"):
            object.__setattr__(
                self,
                name,
                MappingProxyType(
                    {
                        str(key): max(0, to_safe_int(value, 0))
                        for key, value in dict(getattr(self, name)).items()
                    }
                ),
            )
        object.__setattr__(
            self,
            "evidence",
            _freeze_config_strings(self.evidence, unique=True),
        )
        object.__setattr__(
            self,
            "metadata",
            _freeze_config_mapping(self.metadata),
        )

    @property
    def label(self) -> str:
        """Return the canonical residue label."""

        return self.residue.canonical

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain hotspot record."""

        return {
            KEY_RANK: self.rank,
            KEY_RECEPTOR_RESIDUE: self.label,
            KEY_CHAIN_ID: self.residue.chain_id,
            "hotspot_score": self.hotspot_score,
            KEY_COUNT: self.interaction_count,
            "favorable_interactions": self.favorable_count,
            "penalty_interactions": self.penalty_count,
            "family_diversity": self.family_diversity,
            "type_diversity": self.type_diversity,
            KEY_TOTAL_SCORE: self.score_total,
            KEY_PERSISTENCE: self.persistence,
            "pose_count": self.pose_count,
            KEY_TOTAL_POSES: self.total_poses,
            "family_counts": dict(self.family_counts),
            "type_counts": dict(self.type_counts),
            "evidence": list(self.evidence),
            KEY_METADATA: dict(self.metadata),
        }


@dataclass(frozen=True)
class HotspotSection:
    """Hotspot collection and thresholds."""

    hotspots: Tuple[HotspotSummary, ...] = ()
    total_candidates: int = 0
    selected_count: int = 0
    minimum_interactions: int = 1
    minimum_score: Optional[float] = None
    minimum_persistence: Optional[float] = None
    warnings: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "hotspots", tuple(self.hotspots))
        for name in (
            "total_candidates",
            "selected_count",
            "minimum_interactions",
        ):
            object.__setattr__(
                self,
                name,
                max(0, to_safe_int(getattr(self, name), 0)),
            )
        object.__setattr__(
            self,
            "minimum_score",
            to_finite_float(self.minimum_score, None),
        )
        object.__setattr__(
            self,
            "minimum_persistence",
            to_finite_float(self.minimum_persistence, None),
        )
        object.__setattr__(
            self,
            "warnings",
            _freeze_config_strings(self.warnings, unique=False),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain hotspot section."""

        return {
            KEY_HOTSPOTS: [item.to_dict() for item in self.hotspots],
            "total_candidates": self.total_candidates,
            "selected_count": self.selected_count,
            "minimum_interactions": self.minimum_interactions,
            "minimum_score": self.minimum_score,
            "minimum_persistence": self.minimum_persistence,
            KEY_WARNINGS: list(self.warnings),
        }


# 13.2. Hotspot scoring
# -----------------------------------------------------------------------------

DEFAULT_HOTSPOT_WEIGHTS: Final[Mapping[str, float]] = MappingProxyType(
    {
        "interaction_count": 1.0,
        "favorable_count": 1.5,
        "penalty_count": -2.0,
        "family_diversity": 1.0,
        "type_diversity": 0.5,
        "score_total": 1.0,
        "persistence": 2.0,
    }
)


def hotspot_score(
    summary: ResidueSummary,
    *,
    persistence: Optional[float] = None,
    weights: Mapping[str, float] = DEFAULT_HOTSPOT_WEIGHTS,
) -> float:
    """Calculate a transparent hotspot score."""

    score_total = summary.score_total or 0.0
    persistence_value = persistence or 0.0
    components = {
        "interaction_count": summary.interaction_count,
        "favorable_count": summary.favorable_count,
        "penalty_count": summary.penalty_count,
        "family_diversity": len(summary.family_counts),
        "type_diversity": len(summary.type_counts),
        "score_total": score_total,
        "persistence": persistence_value,
    }
    return sum(
        to_finite_float(weights.get(name), 0.0) * value
        for name, value in components.items()
    )


def hotspot_evidence(
    summary: ResidueSummary,
    *,
    persistence: Optional[float] = None,
) -> Tuple[str, ...]:
    """Generate concise hotspot evidence labels."""

    evidence: List[str] = []
    if summary.interaction_count:
        evidence.append(f"{summary.interaction_count} interactions")
    if summary.favorable_count:
        evidence.append(f"{summary.favorable_count} favorable")
    if len(summary.family_counts) > 1:
        evidence.append(f"{len(summary.family_counts)} families")
    if summary.score_total is not None:
        evidence.append(f"score {format_score(summary.score_total)}")
    if persistence is not None:
        evidence.append(
            f"persistence {format_percent(persistence, fraction=True)}"
        )
    if summary.penalty_count:
        evidence.append(f"{summary.penalty_count} penalties")
    return tuple(evidence)


def build_hotspot_summary(
    summary: ResidueSummary,
    *,
    rank: Optional[int] = None,
    persistence: Optional[float] = None,
    pose_count: Optional[int] = None,
    total_poses: Optional[int] = None,
    weights: Mapping[str, float] = DEFAULT_HOTSPOT_WEIGHTS,
) -> HotspotSummary:
    """Convert a residue summary into a hotspot record."""

    return HotspotSummary(
        residue=summary.key,
        rank=rank,
        hotspot_score=hotspot_score(
            summary,
            persistence=persistence,
            weights=weights,
        ),
        interaction_count=summary.interaction_count,
        favorable_count=summary.favorable_count,
        penalty_count=summary.penalty_count,
        family_diversity=len(summary.family_counts),
        type_diversity=len(summary.type_counts),
        score_total=summary.score_total,
        persistence=persistence,
        pose_count=pose_count,
        total_poses=total_poses,
        family_counts=summary.family_counts,
        type_counts=summary.type_counts,
        evidence=hotspot_evidence(
            summary,
            persistence=persistence,
        ),
        metadata={
            "mean_score": summary.score_mean,
            "distance_statistics": dict(summary.distance_statistics),
        },
    )


def hotspot_sort_key(item: HotspotSummary) -> Tuple[Any, ...]:
    """Return a stable hotspot ranking key."""

    persistence = (
        -item.persistence
        if item.persistence is not None
        else math.inf
    )
    return (
        -item.hotspot_score,
        persistence,
        -item.interaction_count,
        -item.family_diversity,
        item.penalty_count,
        item.residue.chain_id,
        item.residue.residue_number,
        item.residue.residue_name,
    )


# 13.3. Hotspot selection
# -----------------------------------------------------------------------------

def identify_hotspots(
    summaries: Iterable[ResidueSummary],
    *,
    top_n: int = DEFAULT_TOP_HOTSPOTS,
    minimum_interactions: int = 1,
    minimum_score: Optional[float] = None,
    minimum_persistence: Optional[float] = None,
    persistence: Optional[Mapping[str, float]] = None,
    pose_counts: Optional[Mapping[str, int]] = None,
    total_poses: Optional[int] = None,
    weights: Mapping[str, float] = DEFAULT_HOTSPOT_WEIGHTS,
) -> HotspotSection:
    """Rank and filter residue hotspots."""

    minimum_interactions = max(0, int(minimum_interactions))
    candidates: List[HotspotSummary] = []

    for summary in summaries:
        label = summary.residue
        persistence_value = (
            to_finite_float(persistence.get(label), None)
            if persistence is not None and label in persistence
            else None
        )
        pose_count = (
            to_safe_int(pose_counts.get(label), None)
            if pose_counts is not None and label in pose_counts
            else None
        )
        candidate = build_hotspot_summary(
            summary,
            persistence=persistence_value,
            pose_count=pose_count,
            total_poses=total_poses,
            weights=weights,
        )
        if candidate.interaction_count < minimum_interactions:
            continue
        if (
            minimum_score is not None
            and candidate.hotspot_score < minimum_score
        ):
            continue
        if (
            minimum_persistence is not None
            and (
                candidate.persistence is None
                or candidate.persistence < minimum_persistence
            )
        ):
            continue
        candidates.append(candidate)

    candidates.sort(key=hotspot_sort_key)
    selected = candidates[: max(0, int(top_n))]
    ranked = tuple(
        replace(item, rank=index)
        for index, item in enumerate(selected, start=1)
    )

    return HotspotSection(
        hotspots=ranked,
        total_candidates=len(candidates),
        selected_count=len(ranked),
        minimum_interactions=minimum_interactions,
        minimum_score=minimum_score,
        minimum_persistence=minimum_persistence,
    )


def summarize_hotspots(
    value: Any,
    *,
    residue_summaries: Optional[Iterable[ResidueSummary]] = None,
    config: ReportConfig = DEFAULT_REPORT_CONFIG,
    strict: Optional[bool] = None,
) -> HotspotSection:
    """Build hotspots from a pose or residue summaries."""

    if residue_summaries is None:
        if strict is None:
            strict = config.errors.mode is ErrorMode.RAISE
        residue_summaries = summarize_residues(
            value,
            interaction_config=config.interactions,
            top_n=None,
            score_direction=config.scoring.score_direction,
            strict=strict,
        )
    return identify_hotspots(
        residue_summaries,
        top_n=config.multipose.top_hotspots,
    )


def hotspot_rows(
    section: Union[HotspotSection, Iterable[HotspotSummary], Any],
    *,
    config: ReportConfig = DEFAULT_REPORT_CONFIG,
) -> ReportRows:
    """Return hotspot table rows."""

    if isinstance(section, HotspotSection):
        values = section.hotspots
    elif (
        isinstance(section, Iterable)
        and not isinstance(section, (str, bytes, Mapping))
    ):
        values = tuple(section)
    else:
        values = summarize_hotspots(section, config=config).hotspots

    rows: ReportRows = []
    for item in values:
        rows.append(
            {
                KEY_RANK: item.rank,
                KEY_RECEPTOR_RESIDUE: item.label,
                KEY_CHAIN_ID: item.residue.chain_id,
                "hotspot_score": item.hotspot_score,
                "hotspot_score_text": format_score(
                    item.hotspot_score,
                    config.formatting,
                ),
                KEY_COUNT: item.interaction_count,
                "favorable_interactions": item.favorable_count,
                "penalty_interactions": item.penalty_count,
                "family_diversity": item.family_diversity,
                "type_diversity": item.type_diversity,
                KEY_TOTAL_SCORE: item.score_total,
                KEY_PERSISTENCE: item.persistence,
                "persistence_text": (
                    format_percent(
                        item.persistence,
                        config.formatting,
                        fraction=True,
                    )
                    if item.persistence is not None
                    else config.formatting.missing_text
                ),
                "evidence": format_sequence(item.evidence, missing=""),
            }
        )
    return rows


# 13.4. Public hotspot interface
# -----------------------------------------------------------------------------

_SECTION_13_PUBLIC_NAMES: Final[Tuple[str, ...]] = (
    "HotspotSummary",
    "HotspotSection",
    "DEFAULT_HOTSPOT_WEIGHTS",
    "hotspot_score",
    "hotspot_evidence",
    "build_hotspot_summary",
    "hotspot_sort_key",
    "identify_hotspots",
    "summarize_hotspots",
    "hotspot_rows",
)

_register_public_names(_SECTION_13_PUBLIC_NAMES)

# =============================================================================
# End of Section 13
# =============================================================================


# =============================================================================
# Section 14 — Scoring and explainability
# =============================================================================

# 14.1. Scoring records
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class ScoreComponentSummary:
    """Normalized score component."""

    name: str
    value: Optional[float] = None
    weight: Optional[float] = None
    contribution: Optional[float] = None
    direction: RankDirection = RankDirection.HIGHER_IS_BETTER
    source: str = ""
    description: str = ""
    metadata: Mapping[str, Any] = field(
        default_factory=lambda: _EMPTY_METADATA
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "name",
            single_line_text(self.name, DEFAULT_UNKNOWN_TEXT),
        )
        object.__setattr__(
            self,
            "value",
            to_finite_float(self.value, None),
        )
        object.__setattr__(
            self,
            "weight",
            to_finite_float(self.weight, None),
        )
        contribution = to_finite_float(self.contribution, None)
        if contribution is None and self.value is not None:
            contribution = (
                self.value
                if self.weight is None
                else self.value * self.weight
            )
        object.__setattr__(self, "contribution", contribution)
        object.__setattr__(
            self,
            "direction",
            _coerce_enum(
                RankDirection,
                self.direction,
                "direction",
            ),
        )
        object.__setattr__(
            self,
            "source",
            single_line_text(self.source, ""),
        )
        object.__setattr__(
            self,
            "description",
            safe_string(self.description, ""),
        )
        object.__setattr__(
            self,
            "metadata",
            _freeze_config_mapping(self.metadata),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain score component."""

        return {
            "name": self.name,
            KEY_SCORE: self.value,
            "weight": self.weight,
            "contribution": self.contribution,
            "direction": self.direction.value,
            KEY_SOURCE: self.source,
            KEY_DESCRIPTION: self.description,
            KEY_METADATA: dict(self.metadata),
        }


@dataclass(frozen=True)
class ExplanationItem:
    """One human-readable score explanation."""

    label: str
    text: str
    impact: Optional[float] = None
    favorable: Optional[bool] = None
    source: str = ""
    evidence: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "label",
            single_line_text(self.label, DEFAULT_UNKNOWN_TEXT),
        )
        object.__setattr__(
            self,
            "text",
            safe_string(self.text, ""),
        )
        object.__setattr__(
            self,
            "impact",
            to_finite_float(self.impact, None),
        )
        object.__setattr__(
            self,
            "source",
            single_line_text(self.source, ""),
        )
        object.__setattr__(
            self,
            "evidence",
            _freeze_config_strings(self.evidence, unique=True),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain explanation item."""

        return {
            "label": self.label,
            "text": self.text,
            "impact": self.impact,
            "favorable": self.favorable,
            KEY_SOURCE: self.source,
            "evidence": list(self.evidence),
        }


@dataclass(frozen=True)
class ScoringSection:
    """Scoring summary and explanation."""

    total_score: Optional[float] = None
    raw_score: Optional[float] = None
    normalized_score: Optional[float] = None
    affinity: Optional[float] = None
    components: Tuple[ScoreComponentSummary, ...] = ()
    explanations: Tuple[ExplanationItem, ...] = ()
    favorable_components: int = 0
    unfavorable_components: int = 0
    source: str = ""
    recalculated: bool = False
    metadata: Mapping[str, Any] = field(
        default_factory=lambda: _EMPTY_METADATA
    )
    warnings: Tuple[str, ...] = ()
    errors: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "total_score",
            "raw_score",
            "normalized_score",
            "affinity",
        ):
            object.__setattr__(
                self,
                name,
                to_finite_float(getattr(self, name), None),
            )
        object.__setattr__(
            self,
            "components",
            tuple(self.components),
        )
        object.__setattr__(
            self,
            "explanations",
            tuple(self.explanations),
        )
        for name in (
            "favorable_components",
            "unfavorable_components",
        ):
            object.__setattr__(
                self,
                name,
                max(0, to_safe_int(getattr(self, name), 0)),
            )
        object.__setattr__(
            self,
            "source",
            single_line_text(self.source, ""),
        )
        object.__setattr__(
            self,
            "metadata",
            _freeze_config_mapping(self.metadata),
        )
        object.__setattr__(
            self,
            "warnings",
            _freeze_config_strings(self.warnings, unique=False),
        )
        object.__setattr__(
            self,
            "errors",
            _freeze_config_strings(self.errors, unique=False),
        )

    def to_dict(
        self,
        *,
        include_components: bool = True,
        include_explanations: bool = True,
    ) -> Dict[str, Any]:
        """Return a plain scoring section."""

        record: Dict[str, Any] = {
            KEY_TOTAL_SCORE: self.total_score,
            KEY_RAW_SCORE: self.raw_score,
            KEY_NORMALIZED_SCORE: self.normalized_score,
            KEY_AFFINITY: self.affinity,
            "favorable_components": self.favorable_components,
            "unfavorable_components": self.unfavorable_components,
            KEY_SOURCE: self.source,
            "recalculated": self.recalculated,
            KEY_METADATA: dict(self.metadata),
            KEY_WARNINGS: list(self.warnings),
            KEY_ERRORS: list(self.errors),
        }
        if include_components:
            record[KEY_SCORE_COMPONENTS] = [
                item.to_dict() for item in self.components
            ]
        if include_explanations:
            record[KEY_EXPLAINABILITY] = [
                item.to_dict() for item in self.explanations
            ]
        return record


# 14.2. Score component normalization
# -----------------------------------------------------------------------------

_SCORE_COMPONENT_VALUE_FIELDS: Final[Tuple[str, ...]] = (
    "value",
    "score",
    "raw_value",
    "component_score",
)

_SCORE_COMPONENT_WEIGHT_FIELDS: Final[Tuple[str, ...]] = (
    "weight",
    "coefficient",
    "multiplier",
)

_SCORE_COMPONENT_CONTRIBUTION_FIELDS: Final[Tuple[str, ...]] = (
    "contribution",
    "weighted_score",
    "weighted_value",
    "impact",
)


def normalize_score_component(
    value: Any,
    *,
    name_hint: Any = None,
    source: str = "",
) -> ScoreComponentSummary:
    """Normalize one score component."""

    if isinstance(value, ScoreComponentSummary):
        if source and not value.source:
            return replace(value, source=source)
        return value

    if isinstance(value, Real) and not isinstance(value, bool):
        return ScoreComponentSummary(
            name=single_line_text(name_hint, "component"),
            value=float(value),
            source=source,
        )

    name = name_hint
    if name is None:
        name = get_first_object_field(
            value,
            ("name", "label", "component", "key"),
            "component",
            skip_none=True,
        )

    component_value = get_first_object_field(
        value,
        _SCORE_COMPONENT_VALUE_FIELDS,
        None,
        skip_none=True,
    )
    weight = get_first_object_field(
        value,
        _SCORE_COMPONENT_WEIGHT_FIELDS,
        None,
        skip_none=True,
    )
    contribution = get_first_object_field(
        value,
        _SCORE_COMPONENT_CONTRIBUTION_FIELDS,
        None,
        skip_none=True,
    )
    direction = get_first_object_field(
        value,
        ("direction", "rank_direction", "optimization"),
        RankDirection.HIGHER_IS_BETTER,
        skip_none=True,
    )
    description = get_first_object_field(
        value,
        ("description", "explanation", "text"),
        "",
        skip_none=True,
    )

    return ScoreComponentSummary(
        name=single_line_text(name, "component"),
        value=component_value,
        weight=weight,
        contribution=contribution,
        direction=direction,
        source=source or single_line_text(
            get_first_object_field(
                value,
                ("source", "module"),
                "",
                skip_none=True,
            ),
            "",
        ),
        description=safe_string(description, ""),
        metadata=get_object_metadata(value),
    )


def normalize_score_components(
    value: Any,
    *,
    source: str = "",
    max_items: int = DEFAULT_MAX_ITEMS,
) -> List[ScoreComponentSummary]:
    """Normalize score components from mappings or sequences."""

    if value is None or value is MISSING:
        return []

    components: List[ScoreComponentSummary] = []
    if isinstance(value, Mapping):
        iterator = value.items()
    else:
        iterator = ((None, item) for item in iter_object_collection(value))

    for name_hint, item in iterator:
        try:
            components.append(
                normalize_score_component(
                    item,
                    name_hint=name_hint,
                    source=source,
                )
            )
        except Exception:
            continue
        if len(components) >= max(0, int(max_items)):
            break

    return components


# 14.3. Scoring extraction
# -----------------------------------------------------------------------------

def extract_scoring_object(value: Any) -> Any:
    """Return the scoring payload associated with an object."""

    scoring = get_first_object_field(
        value,
        (
            "scoring",
            "scoring_result",
            "score_result",
            "score_summary",
        ),
        MISSING,
        skip_none=True,
    )
    return value if scoring is MISSING else scoring


def extract_score_components(
    value: Any,
    *,
    max_items: int = DEFAULT_MAX_ITEMS,
) -> List[ScoreComponentSummary]:
    """Extract normalized score components."""

    scoring = extract_scoring_object(value)
    components = get_first_object_field(
        scoring,
        (
            "components",
            "score_components",
            "component_scores",
            "terms",
            "contributions",
        ),
        MISSING,
        skip_none=True,
    )

    if components is not MISSING:
        return normalize_score_components(
            components,
            source=SOURCE_MODULE_SCORING,
            max_items=max_items,
        )

    if isinstance(scoring, Mapping):
        reserved = {
            normalize_field_name(name)
            for name in (
                "score",
                "total_score",
                "raw_score",
                "normalized_score",
                "affinity",
                "metadata",
                "warnings",
                "errors",
            )
        }
        candidate = {
            key: item
            for key, item in scoring.items()
            if normalize_field_name(key) not in reserved
            and to_finite_float(item, MISSING) is not MISSING
        }
        return normalize_score_components(
            candidate,
            source=SOURCE_MODULE_SCORING,
            max_items=max_items,
        )

    return []


def _score_value(
    value: Any,
    names: Iterable[str],
    default: Any = None,
) -> Any:
    """Return a score scalar from the scoring payload or parent."""

    scoring = extract_scoring_object(value)
    result = get_first_object_field(
        scoring,
        names,
        MISSING,
        skip_none=True,
    )
    if result is MISSING and scoring is not value:
        result = get_first_object_field(
            value,
            names,
            MISSING,
            skip_none=True,
        )
    return to_finite_float(result, default)


def _local_scoring_recalculation(value: Any) -> Any:
    """Try a compatible local scoring entry point."""

    try:
        try:
            from . import scoring as scoring_module
        except ImportError:
            import scoring as scoring_module
    except ImportError as error:
        raise ReportDependencyError(
            "scoring",
            purpose="score recalculation",
            cause=error,
        ) from error

    candidates = (
        "score_dock_model",
        "score_pose",
        "calculate_pose_score",
        "analyze_dock_model_scoring",
    )
    for name in candidates:
        function = getattr(scoring_module, name, None)
        if not callable(function):
            continue
        try:
            return function(value)
        except TypeError:
            continue
    raise ReportScoringError(
        "No compatible scoring recalculation entry point was found."
    )


# 14.4. Explanation generation
# -----------------------------------------------------------------------------

def component_is_favorable(
    component: ScoreComponentSummary,
) -> Optional[bool]:
    """Infer whether a component favors the pose."""

    impact = component.contribution
    if impact is None:
        return None
    if component.direction is RankDirection.HIGHER_IS_BETTER:
        return impact > 0
    return impact < 0


def explain_score_component(
    component: ScoreComponentSummary,
    *,
    formatting: FormattingConfig = DEFAULT_FORMATTING_CONFIG,
) -> ExplanationItem:
    """Generate one concise component explanation."""

    favorable = component_is_favorable(component)
    impact_text = (
        format_score(component.contribution, formatting)
        if component.contribution is not None
        else formatting.missing_text
    )
    direction_text = (
        "favorable"
        if favorable is True
        else "unfavorable"
        if favorable is False
        else "unclassified"
    )
    text = component.description or (
        f"{component.name} contributes {impact_text} "
        f"and is {direction_text}."
    )
    evidence = []
    if component.value is not None:
        evidence.append(f"value {format_score(component.value, formatting)}")
    if component.weight is not None:
        evidence.append(f"weight {format_score(component.weight, formatting)}")
    return ExplanationItem(
        label=component.name,
        text=text,
        impact=component.contribution,
        favorable=favorable,
        source=component.source,
        evidence=tuple(evidence),
    )


def explain_interaction_balance(
    interactions: Iterable[NormalizedInteraction],
) -> ExplanationItem:
    """Explain favorable and penalty interaction balance."""

    values = list(interactions)
    favorable = sum(item.family.favorable for item in values)
    penalties = sum(item.family.penalty for item in values)
    net = favorable - penalties
    return ExplanationItem(
        label="Interaction balance",
        text=(
            f"{favorable} favorable interactions and "
            f"{penalties} penalty interactions were identified."
        ),
        impact=float(net),
        favorable=net > 0 if net != 0 else None,
        source=SOURCE_MODULE_REPORT,
        evidence=tuple(
            f"{family}: {count}"
            for family, count in interaction_family_counts(values).items()
        ),
    )


def explain_top_residues(
    residues: Iterable[ResidueSummary],
    *,
    limit: int = 5,
) -> ExplanationItem:
    """Explain the highest-ranked receptor residues."""

    values = list(residues)[: max(0, int(limit))]
    labels = [item.residue for item in values]
    return ExplanationItem(
        label="Top residues",
        text=(
            "Highest-ranked receptor residues: "
            + (", ".join(labels) if labels else DEFAULT_MISSING_TEXT)
            + "."
        ),
        favorable=True if labels else None,
        source=SOURCE_MODULE_REPORT,
        evidence=tuple(
            f"{item.residue}: {item.interaction_count} interactions"
            for item in values
        ),
    )


def build_score_explanations(
    components: Iterable[ScoreComponentSummary],
    *,
    interactions: Iterable[NormalizedInteraction] = (),
    residues: Iterable[ResidueSummary] = (),
    formatting: FormattingConfig = DEFAULT_FORMATTING_CONFIG,
    max_items: int = DEFAULT_MAX_ITEMS,
) -> List[ExplanationItem]:
    """Build component and structural score explanations."""

    explanations = [
        explain_score_component(
            component,
            formatting=formatting,
        )
        for component in components
    ]

    interaction_values = list(interactions)
    if interaction_values:
        explanations.append(explain_interaction_balance(interaction_values))

    residue_values = list(residues)
    if residue_values:
        explanations.append(explain_top_residues(residue_values))

    explanations.sort(
        key=lambda item: (
            item.impact is None,
            -(abs(item.impact) if item.impact is not None else 0.0),
            item.label,
        )
    )
    return explanations[: max(0, int(max_items))]


# 14.5. Scoring section construction
# -----------------------------------------------------------------------------

def build_scoring_section(
    value: Any,
    *,
    interactions: Optional[Iterable[NormalizedInteraction]] = None,
    residues: Optional[Iterable[ResidueSummary]] = None,
    config: ScoringReportConfig = DEFAULT_SCORING_REPORT_CONFIG,
    formatting: FormattingConfig = DEFAULT_FORMATTING_CONFIG,
    strict: bool = False,
) -> ScoringSection:
    """Build scoring and explainability from existing results."""

    warnings_out: List[str] = []
    errors_out: List[str] = []
    recalculated = False
    scoring_value = value

    if config.recalculate:
        try:
            recalculated_value = _local_scoring_recalculation(value)
            if recalculated_value is not None:
                scoring_value = recalculated_value
                recalculated = True
        except ReportError as error:
            if strict:
                raise
            errors_out.append(str(error))

    components = extract_score_components(
        scoring_value,
        max_items=config.top_components,
    ) if config.include_components else []

    normalized_interactions = (
        list(interactions)
        if interactions is not None
        else normalize_interaction_input(value, strict=False)
    )
    residue_values = (
        list(residues)
        if residues is not None
        else summarize_residues(
            None,
            interactions=normalized_interactions,
            top_n=DEFAULT_TOP_RESIDUES,
            strict=False,
        )
    )

    total_score = _score_value(
        scoring_value,
        ("total_score", "score", "combined_score"),
        None,
    )
    raw_score = _score_value(
        scoring_value,
        ("raw_score", "unnormalized_score"),
        None,
    )
    normalized_score = _score_value(
        scoring_value,
        ("normalized_score", "norm_score"),
        None,
    )
    affinity = get_pose_affinity(value, None)

    if total_score is None:
        contributions = [
            component.contribution
            for component in components
            if component.contribution is not None
        ]
        if contributions:
            total_score = sum(contributions)
            warnings_out.append(
                "Total score inferred from component contributions."
            )

    explanations = (
        build_score_explanations(
            components,
            interactions=normalized_interactions,
            residues=residue_values,
            formatting=formatting,
            max_items=config.top_components,
        )
        if config.include_explainability
        else []
    )

    favorable_components = sum(
        component_is_favorable(component) is True
        for component in components
    )
    unfavorable_components = sum(
        component_is_favorable(component) is False
        for component in components
    )

    return ScoringSection(
        total_score=total_score,
        raw_score=raw_score,
        normalized_score=normalized_score,
        affinity=affinity if config.include_external_affinity else None,
        components=tuple(components),
        explanations=tuple(explanations),
        favorable_components=favorable_components,
        unfavorable_components=unfavorable_components,
        source=SOURCE_MODULE_SCORING,
        recalculated=recalculated,
        metadata={
            "component_count": len(components),
            "explanation_count": len(explanations),
        },
        warnings=tuple(warnings_out),
        errors=tuple(errors_out),
    )


def summarize_scoring(
    value: Any,
    *,
    config: ReportConfig = DEFAULT_REPORT_CONFIG,
    strict: Optional[bool] = None,
) -> ScoringSection:
    """Convenience scoring summary."""

    if strict is None:
        strict = config.errors.mode is ErrorMode.RAISE
    interactions = normalize_interaction_input(
        value,
        config=config.interactions,
        strict=False,
    )
    residues = summarize_residues(
        None,
        interactions=interactions,
        top_n=config.multipose.top_residues,
        score_direction=config.scoring.score_direction,
        strict=False,
    )
    return build_scoring_section(
        value,
        interactions=interactions,
        residues=residues,
        config=config.scoring,
        formatting=config.formatting,
        strict=strict,
    )


def score_component_rows(
    section: Union[ScoringSection, Any],
    *,
    config: ReportConfig = DEFAULT_REPORT_CONFIG,
) -> ReportRows:
    """Return score-component table rows."""

    if not isinstance(section, ScoringSection):
        section = summarize_scoring(section, config=config)

    rows: ReportRows = []
    for index, component in enumerate(section.components, start=1):
        rows.append(
            {
                KEY_RANK: index,
                "name": component.name,
                KEY_SCORE: component.value,
                "score_text": format_score(
                    component.value,
                    config.formatting,
                ),
                "weight": component.weight,
                "contribution": component.contribution,
                "contribution_text": format_score(
                    component.contribution,
                    config.formatting,
                ),
                "direction": component.direction.value,
                "favorable": component_is_favorable(component),
                KEY_SOURCE: component.source,
                KEY_DESCRIPTION: component.description,
            }
        )
    return rows


def explanation_rows(
    section: Union[ScoringSection, Any],
    *,
    config: ReportConfig = DEFAULT_REPORT_CONFIG,
) -> ReportRows:
    """Return explainability table rows."""

    if not isinstance(section, ScoringSection):
        section = summarize_scoring(section, config=config)

    return [
        {
            KEY_RANK: index,
            "label": item.label,
            "text": item.text,
            "impact": item.impact,
            "impact_text": format_score(
                item.impact,
                config.formatting,
            ),
            "favorable": item.favorable,
            KEY_SOURCE: item.source,
            "evidence": format_sequence(item.evidence, missing=""),
        }
        for index, item in enumerate(section.explanations, start=1)
    ]


# 14.6. Public scoring interface
# -----------------------------------------------------------------------------

_SECTION_14_PUBLIC_NAMES: Final[Tuple[str, ...]] = (
    "ScoreComponentSummary",
    "ExplanationItem",
    "ScoringSection",
    "normalize_score_component",
    "normalize_score_components",
    "extract_scoring_object",
    "extract_score_components",
    "component_is_favorable",
    "explain_score_component",
    "explain_interaction_balance",
    "explain_top_residues",
    "build_score_explanations",
    "build_scoring_section",
    "summarize_scoring",
    "score_component_rows",
    "explanation_rows",
)

_register_public_names(_SECTION_14_PUBLIC_NAMES)

# =============================================================================
# End of Section 14
# =============================================================================


# =============================================================================
# Section 15 — Multipose and ranking
# =============================================================================

# 15.1. Pose ranking records
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class PoseRankingEntry:
    """Ranked pose with structural and scoring summaries."""

    pose_id: Any
    pose_name: str = ""
    rank: Optional[int] = None
    total_score: Optional[float] = None
    normalized_score: Optional[float] = None
    affinity: Optional[float] = None
    interaction_count: int = 0
    residue_count: int = 0
    favorable_count: int = 0
    penalty_count: int = 0
    hotspot_count: int = 0
    tie_group: Optional[int] = None
    source_index: int = 0
    overview: Optional[PoseOverview] = None
    scoring: Optional[ScoringSection] = None
    metadata: Mapping[str, Any] = field(
        default_factory=lambda: _EMPTY_METADATA
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "pose_name",
            single_line_text(self.pose_name, ""),
        )
        for name in (
            "total_score",
            "normalized_score",
            "affinity",
        ):
            object.__setattr__(
                self,
                name,
                to_finite_float(getattr(self, name), None),
            )
        for name in (
            "interaction_count",
            "residue_count",
            "favorable_count",
            "penalty_count",
            "hotspot_count",
            "source_index",
        ):
            object.__setattr__(
                self,
                name,
                max(0, to_safe_int(getattr(self, name), 0)),
            )
        for name in ("rank", "tie_group"):
            value = getattr(self, name)
            object.__setattr__(
                self,
                name,
                None if value is None else max(1, to_safe_int(value, 1)),
            )
        object.__setattr__(
            self,
            "metadata",
            _freeze_config_mapping(self.metadata),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain pose ranking entry."""

        return {
            KEY_RANK: self.rank,
            KEY_POSE_ID: self.pose_id,
            "pose_name": self.pose_name,
            KEY_TOTAL_SCORE: self.total_score,
            KEY_NORMALIZED_SCORE: self.normalized_score,
            KEY_AFFINITY: self.affinity,
            KEY_TOTAL_INTERACTIONS: self.interaction_count,
            KEY_TOTAL_RESIDUES: self.residue_count,
            "favorable_interactions": self.favorable_count,
            "penalty_interactions": self.penalty_count,
            "hotspot_count": self.hotspot_count,
            "tie_group": self.tie_group,
            "source_index": self.source_index,
            KEY_METADATA: dict(self.metadata),
        }


@dataclass(frozen=True)
class MultiposeSummary:
    """Multipose ranking, consensus and persistence."""

    poses: Tuple[PoseRankingEntry, ...] = ()
    total_poses: int = 0
    best_pose_id: Any = None
    ranking_metric: str = KEY_TOTAL_SCORE
    rank_direction: RankDirection = RankDirection.HIGHER_IS_BETTER
    consensus_residues: Tuple[str, ...] = ()
    residue_pose_counts: Mapping[str, int] = field(
        default_factory=lambda: _EMPTY_METADATA
    )
    residue_persistence: Mapping[str, float] = field(
        default_factory=lambda: _EMPTY_METADATA
    )
    family_pose_counts: Mapping[str, int] = field(
        default_factory=lambda: _EMPTY_METADATA
    )
    family_persistence: Mapping[str, float] = field(
        default_factory=lambda: _EMPTY_METADATA
    )
    score_statistics: Mapping[str, Any] = field(
        default_factory=lambda: _EMPTY_METADATA
    )
    affinity_statistics: Mapping[str, Any] = field(
        default_factory=lambda: _EMPTY_METADATA
    )
    warnings: Tuple[str, ...] = ()
    errors: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "poses", tuple(self.poses))
        object.__setattr__(
            self,
            "total_poses",
            max(0, to_safe_int(self.total_poses, 0)),
        )
        object.__setattr__(
            self,
            "ranking_metric",
            single_line_text(self.ranking_metric, KEY_TOTAL_SCORE),
        )
        object.__setattr__(
            self,
            "rank_direction",
            _coerce_enum(
                RankDirection,
                self.rank_direction,
                "rank_direction",
            ),
        )
        object.__setattr__(
            self,
            "consensus_residues",
            _freeze_config_strings(self.consensus_residues),
        )
        for name in (
            "residue_pose_counts",
            "family_pose_counts",
        ):
            object.__setattr__(
                self,
                name,
                MappingProxyType(
                    {
                        str(key): max(0, to_safe_int(value, 0))
                        for key, value in dict(getattr(self, name)).items()
                    }
                ),
            )
        for name in (
            "residue_persistence",
            "family_persistence",
        ):
            object.__setattr__(
                self,
                name,
                MappingProxyType(
                    {
                        str(key): max(0.0, to_finite_float(value, 0.0))
                        for key, value in dict(getattr(self, name)).items()
                    }
                ),
            )
        for name in ("score_statistics", "affinity_statistics"):
            object.__setattr__(
                self,
                name,
                _freeze_config_mapping(getattr(self, name)),
            )
        object.__setattr__(
            self,
            "warnings",
            _freeze_config_strings(self.warnings, unique=False),
        )
        object.__setattr__(
            self,
            "errors",
            _freeze_config_strings(self.errors, unique=False),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain multipose summary."""

        return {
            KEY_TOTAL_POSES: self.total_poses,
            "best_pose_id": self.best_pose_id,
            "ranking_metric": self.ranking_metric,
            "rank_direction": self.rank_direction.value,
            KEY_RANKING: [pose.to_dict() for pose in self.poses],
            KEY_CONSENSUS: list(self.consensus_residues),
            "residue_pose_counts": dict(self.residue_pose_counts),
            "residue_persistence": dict(self.residue_persistence),
            "family_pose_counts": dict(self.family_pose_counts),
            "family_persistence": dict(self.family_persistence),
            "score_statistics": dict(self.score_statistics),
            "affinity_statistics": dict(self.affinity_statistics),
            KEY_WARNINGS: list(self.warnings),
            KEY_ERRORS: list(self.errors),
        }


# 15.2. Numeric summary helpers
# -----------------------------------------------------------------------------

def numeric_statistics(values: Iterable[Any]) -> Dict[str, Any]:
    """Summarize finite numeric values."""

    numeric = [
        value
        for value in (
            to_finite_float(item, None)
            for item in values
        )
        if value is not None
    ]
    if not numeric:
        return {
            "count": 0,
            "minimum": None,
            "maximum": None,
            "mean": None,
            "median": None,
        }
    return {
        "count": len(numeric),
        "minimum": min(numeric),
        "maximum": max(numeric),
        "mean": fmean(numeric),
        "median": median(numeric),
    }


# 15.3. Pose entry construction
# -----------------------------------------------------------------------------

def build_pose_ranking_entry(
    pose: Any,
    *,
    source_index: int = 0,
    config: ReportConfig = DEFAULT_REPORT_CONFIG,
) -> PoseRankingEntry:
    """Build one ranking entry from a pose."""

    interactions = normalize_interaction_input(
        pose,
        config=config.interactions,
        strict=False,
    )
    overview = build_pose_overview(
        pose,
        interactions=interactions,
        interaction_config=config.interactions,
        strict=False,
        include_metadata=False,
    )
    scoring = build_scoring_section(
        pose,
        interactions=interactions,
        residues=summarize_residues(
            None,
            interactions=interactions,
            top_n=config.multipose.top_residues,
            score_direction=config.scoring.score_direction,
            strict=False,
        ),
        config=config.scoring,
        formatting=config.formatting,
        strict=False,
    )
    hotspots = summarize_hotspots(
        pose,
        residue_summaries=summarize_residues(
            None,
            interactions=interactions,
            top_n=None,
            score_direction=config.scoring.score_direction,
            strict=False,
        ),
        config=config,
        strict=False,
    )

    total_score = (
        scoring.total_score
        if scoring.total_score is not None
        else overview.total_score
    )
    return PoseRankingEntry(
        pose_id=overview.pose_id
        if overview.pose_id is not None
        else source_index + 1,
        pose_name=overview.pose_name,
        total_score=total_score,
        normalized_score=scoring.normalized_score,
        affinity=scoring.affinity
        if scoring.affinity is not None
        else overview.affinity,
        interaction_count=overview.interaction_count,
        residue_count=overview.residue_count,
        favorable_count=overview.favorable_count,
        penalty_count=overview.penalty_count,
        hotspot_count=hotspots.selected_count,
        source_index=source_index,
        overview=overview,
        scoring=scoring,
    )


# 15.4. Ranking
# -----------------------------------------------------------------------------

def pose_primary_metric(
    entry: PoseRankingEntry,
    *,
    metric: str = KEY_TOTAL_SCORE,
) -> Optional[float]:
    """Return the selected pose ranking metric."""

    normalized = normalize_field_name(metric)
    mapping = {
        normalize_field_name(KEY_TOTAL_SCORE): entry.total_score,
        normalize_field_name(KEY_NORMALIZED_SCORE): entry.normalized_score,
        normalize_field_name(KEY_AFFINITY): entry.affinity,
        normalize_field_name(KEY_TOTAL_INTERACTIONS): float(
            entry.interaction_count
        ),
        normalize_field_name(KEY_TOTAL_RESIDUES): float(entry.residue_count),
    }
    return mapping.get(normalized, entry.total_score)


def pose_ranking_sort_key(
    entry: PoseRankingEntry,
    *,
    metric: str = KEY_TOTAL_SCORE,
    direction: RankDirection = RankDirection.HIGHER_IS_BETTER,
) -> Tuple[Any, ...]:
    """Return a stable pose ranking key."""

    primary = pose_primary_metric(entry, metric=metric)
    if primary is None:
        primary_key = math.inf
    elif direction is RankDirection.HIGHER_IS_BETTER:
        primary_key = -primary
    else:
        primary_key = primary

    affinity_key = (
        math.inf if entry.affinity is None else entry.affinity
    )
    return (
        primary_key,
        -entry.favorable_count,
        entry.penalty_count,
        affinity_key,
        -entry.interaction_count,
        entry.source_index,
    )


def assign_pose_ranks(
    entries: Iterable[PoseRankingEntry],
    *,
    metric: str = KEY_TOTAL_SCORE,
    direction: RankDirection = RankDirection.HIGHER_IS_BETTER,
    tolerance: float = REPORT_COMPARISON_TOLERANCE,
) -> List[PoseRankingEntry]:
    """Sort poses and assign competition-style ranks."""

    values = sorted(
        entries,
        key=lambda entry: pose_ranking_sort_key(
            entry,
            metric=metric,
            direction=direction,
        ),
    )
    ranked: List[PoseRankingEntry] = []
    previous_value: Optional[float] = None
    current_rank = 0
    tie_group = 0

    for position, entry in enumerate(values, start=1):
        current_value = pose_primary_metric(entry, metric=metric)
        same_tie = (
            previous_value is not None
            and current_value is not None
            and math.isclose(
                current_value,
                previous_value,
                rel_tol=tolerance,
                abs_tol=tolerance,
            )
        )
        if not same_tie:
            current_rank = position
            tie_group += 1
        ranked.append(
            replace(
                entry,
                rank=current_rank,
                tie_group=tie_group,
            )
        )
        previous_value = current_value

    return ranked


# 15.5. Consensus and persistence
# -----------------------------------------------------------------------------

def pose_residue_set(entry: PoseRankingEntry) -> Set[str]:
    """Return receptor residues represented by one pose entry."""

    if entry.overview is None:
        return set()
    metadata_residues = entry.metadata.get("residues")
    if metadata_residues:
        return {
            single_line_text(item, "")
            for item in iter_object_collection(metadata_residues)
            if single_line_text(item, "")
        }
    return set()


def multipose_residue_counts(
    poses: Iterable[Any],
    *,
    config: ReportConfig = DEFAULT_REPORT_CONFIG,
) -> Dict[str, int]:
    """Count in how many poses each receptor residue occurs."""

    counts: Counter[str] = Counter()
    for pose in poses:
        interactions = normalize_interaction_input(
            pose,
            config=config.interactions,
            strict=False,
        )
        residues = {
            residue_key_from_interaction(item).canonical
            for item in interactions
            if residue_key_from_interaction(item).canonical
        }
        counts.update(residues)
    return dict(
        sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    )


def multipose_family_counts(
    poses: Iterable[Any],
    *,
    config: ReportConfig = DEFAULT_REPORT_CONFIG,
) -> Dict[str, int]:
    """Count in how many poses each interaction family occurs."""

    counts: Counter[str] = Counter()
    for pose in poses:
        interactions = normalize_interaction_input(
            pose,
            config=config.interactions,
            strict=False,
        )
        counts.update(
            {
                interaction.family.value
                for interaction in interactions
            }
        )
    return dict(
        sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    )


def persistence_from_counts(
    counts: Mapping[str, int],
    total: int,
) -> Dict[str, float]:
    """Convert occurrence counts to fractions."""

    if total <= 0:
        return {str(key): 0.0 for key in counts}
    return {
        str(key): count / total
        for key, count in counts.items()
    }


def consensus_from_persistence(
    persistence: Mapping[str, float],
    *,
    threshold: float = 0.5,
) -> Tuple[str, ...]:
    """Return entries meeting a persistence threshold."""

    return tuple(
        key
        for key, value in sorted(
            persistence.items(),
            key=lambda item: (-item[1], item[0]),
        )
        if value >= threshold
    )


# 15.6. Multipose construction
# -----------------------------------------------------------------------------

def build_multipose_summary(
    poses: Any,
    *,
    config: ReportConfig = DEFAULT_REPORT_CONFIG,
    ranking_metric: str = KEY_TOTAL_SCORE,
    consensus_threshold: float = 0.5,
) -> MultiposeSummary:
    """Build multipose ranking, consensus and persistence."""

    pose_values = list(iter_object_collection(poses))
    entries = [
        build_pose_ranking_entry(
            pose,
            source_index=index,
            config=config,
        )
        for index, pose in enumerate(pose_values)
    ]

    direction = (
        config.scoring.affinity_direction
        if normalize_field_name(ranking_metric)
        == normalize_field_name(KEY_AFFINITY)
        else config.multipose.rank_direction
    )
    ranked = assign_pose_ranks(
        entries,
        metric=ranking_metric,
        direction=direction,
        tolerance=config.multipose.tie_tolerance,
    )

    if config.multipose.top_poses:
        ranked = ranked[: config.multipose.top_poses]

    residue_counts = multipose_residue_counts(
        pose_values,
        config=config,
    )
    family_counts = multipose_family_counts(
        pose_values,
        config=config,
    )
    residue_persistence = persistence_from_counts(
        residue_counts,
        len(pose_values),
    )
    family_persistence = persistence_from_counts(
        family_counts,
        len(pose_values),
    )
    consensus = (
        consensus_from_persistence(
            residue_persistence,
            threshold=consensus_threshold,
        )
        if config.multipose.include_consensus
        else ()
    )

    return MultiposeSummary(
        poses=tuple(ranked),
        total_poses=len(pose_values),
        best_pose_id=ranked[0].pose_id if ranked else None,
        ranking_metric=ranking_metric,
        rank_direction=direction,
        consensus_residues=consensus,
        residue_pose_counts=residue_counts,
        residue_persistence=residue_persistence
        if config.multipose.include_persistence
        else {},
        family_pose_counts=family_counts,
        family_persistence=family_persistence
        if config.multipose.include_persistence
        else {},
        score_statistics=numeric_statistics(
            entry.total_score for entry in entries
        ),
        affinity_statistics=numeric_statistics(
            entry.affinity for entry in entries
        ),
    )


def summarize_multipose(
    poses: Any,
    *,
    config: ReportConfig = DEFAULT_REPORT_CONFIG,
    ranking_metric: str = KEY_TOTAL_SCORE,
) -> MultiposeSummary:
    """Convenience multipose summary."""

    return build_multipose_summary(
        poses,
        config=config,
        ranking_metric=ranking_metric,
    )


def multipose_ranking_rows(
    summary: Union[MultiposeSummary, Any],
    *,
    config: ReportConfig = DEFAULT_REPORT_CONFIG,
) -> ReportRows:
    """Return multipose ranking rows."""

    if not isinstance(summary, MultiposeSummary):
        summary = summarize_multipose(summary, config=config)

    return [
        {
            KEY_RANK: entry.rank,
            "tie_group": entry.tie_group,
            KEY_POSE_ID: entry.pose_id,
            "pose_name": entry.pose_name,
            KEY_TOTAL_SCORE: entry.total_score,
            "score_text": format_score(
                entry.total_score,
                config.formatting,
            ),
            KEY_NORMALIZED_SCORE: entry.normalized_score,
            KEY_AFFINITY: entry.affinity,
            "affinity_text": format_score(
                entry.affinity,
                config.formatting,
            ),
            KEY_TOTAL_INTERACTIONS: entry.interaction_count,
            KEY_TOTAL_RESIDUES: entry.residue_count,
            "favorable_interactions": entry.favorable_count,
            "penalty_interactions": entry.penalty_count,
            "hotspot_count": entry.hotspot_count,
        }
        for entry in summary.poses
    ]


def persistence_rows(
    persistence: Mapping[str, float],
    *,
    counts: Optional[Mapping[str, int]] = None,
    total: Optional[int] = None,
    formatting: FormattingConfig = DEFAULT_FORMATTING_CONFIG,
) -> ReportRows:
    """Return persistence table rows."""

    rows: ReportRows = []
    for rank, (key, value) in enumerate(
        sorted(
            persistence.items(),
            key=lambda item: (-item[1], item[0]),
        ),
        start=1,
    ):
        rows.append(
            {
                KEY_RANK: rank,
                "item": key,
                KEY_COUNT: (
                    counts.get(key)
                    if counts is not None
                    else None
                ),
                KEY_TOTAL_POSES: total,
                KEY_PERSISTENCE: value,
                "persistence_text": format_percent(
                    value,
                    formatting,
                    fraction=True,
                ),
            }
        )
    return rows


# 15.7. Public multipose interface
# -----------------------------------------------------------------------------

_SECTION_15_PUBLIC_NAMES: Final[Tuple[str, ...]] = (
    "PoseRankingEntry",
    "MultiposeSummary",
    "numeric_statistics",
    "build_pose_ranking_entry",
    "pose_primary_metric",
    "pose_ranking_sort_key",
    "assign_pose_ranks",
    "pose_residue_set",
    "multipose_residue_counts",
    "multipose_family_counts",
    "persistence_from_counts",
    "consensus_from_persistence",
    "build_multipose_summary",
    "summarize_multipose",
    "multipose_ranking_rows",
    "persistence_rows",
)

_register_public_names(_SECTION_15_PUBLIC_NAMES)

# =============================================================================
# End of Section 15
# =============================================================================


# =============================================================================
# Section 16 — Provenance
# =============================================================================

# 16.1. Provenance records
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class ProvenanceItem:
    """One provenance key-value entry."""

    category: str
    key: str
    value: Any
    source: str = ""
    available: bool = True
    metadata: Mapping[str, Any] = field(
        default_factory=lambda: _EMPTY_METADATA
    )

    def __post_init__(self) -> None:
        for name in ("category", "key", "source"):
            object.__setattr__(
                self,
                name,
                single_line_text(getattr(self, name), ""),
            )
        object.__setattr__(self, "available", bool(self.available))
        object.__setattr__(
            self,
            "metadata",
            _freeze_config_mapping(self.metadata),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain provenance item."""

        return {
            "category": self.category,
            "key": self.key,
            "value": self.value,
            KEY_SOURCE: self.source,
            "available": self.available,
            KEY_METADATA: dict(self.metadata),
        }


@dataclass(frozen=True)
class ProvenanceSection:
    """Generation environment and source provenance."""

    generated_at: str
    generator: str = REPORT_GENERATOR_NAME
    generator_version: str = __version__
    schema_name: str = REPORT_SCHEMA_NAME
    schema_version: str = REPORT_SCHEMA_VERSION
    items: Tuple[ProvenanceItem, ...] = ()
    source_files: Tuple[Mapping[str, Any], ...] = ()
    parameters: Mapping[str, Any] = field(
        default_factory=lambda: _EMPTY_METADATA
    )
    checksums: Mapping[str, str] = field(
        default_factory=lambda: _EMPTY_METADATA
    )
    metadata: Mapping[str, Any] = field(
        default_factory=lambda: _EMPTY_METADATA
    )
    warnings: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "generated_at",
            "generator",
            "generator_version",
            "schema_name",
            "schema_version",
        ):
            object.__setattr__(
                self,
                name,
                single_line_text(getattr(self, name), ""),
            )
        object.__setattr__(self, "items", tuple(self.items))
        object.__setattr__(
            self,
            "source_files",
            tuple(
                MappingProxyType(dict(item))
                for item in self.source_files
            ),
        )
        object.__setattr__(
            self,
            "parameters",
            _freeze_config_mapping(self.parameters),
        )
        object.__setattr__(
            self,
            "checksums",
            MappingProxyType(
                {
                    str(key): str(value)
                    for key, value in dict(self.checksums).items()
                }
            ),
        )
        object.__setattr__(
            self,
            "metadata",
            _freeze_config_mapping(self.metadata),
        )
        object.__setattr__(
            self,
            "warnings",
            _freeze_config_strings(self.warnings, unique=False),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain provenance section."""

        return {
            KEY_GENERATED_AT: self.generated_at,
            KEY_GENERATOR: self.generator,
            KEY_GENERATOR_VERSION: self.generator_version,
            KEY_SCHEMA_NAME: self.schema_name,
            KEY_SCHEMA_VERSION: self.schema_version,
            "items": [item.to_dict() for item in self.items],
            KEY_SOURCE_FILES: [dict(item) for item in self.source_files],
            KEY_PARAMETERS: dict(self.parameters),
            "checksums": dict(self.checksums),
            KEY_METADATA: dict(self.metadata),
            KEY_WARNINGS: list(self.warnings),
        }


# 16.2. Version and environment helpers
# -----------------------------------------------------------------------------

def current_utc_timestamp() -> str:
    """Return the current UTC timestamp."""

    return format_datetime(
        datetime.now(timezone.utc),
        use_utc=True,
    )


def module_version(
    module_or_name: Any,
    default: Any = None,
) -> Any:
    """Return a module version without importing optional packages eagerly."""

    module = module_or_name
    if isinstance(module_or_name, str):
        module = sys.modules.get(module_or_name)
        if module is None:
            try:
                module = __import__(module_or_name)
            except ImportError:
                return default
    return get_first_object_field(
        module,
        ("__version__", "version", "VERSION"),
        default,
        skip_none=True,
    )


def python_environment_items() -> List[ProvenanceItem]:
    """Return Python and operating-system provenance items."""

    return [
        ProvenanceItem(
            category="environment",
            key=KEY_PYTHON_VERSION,
            value=platform.python_version(),
            source="platform",
        ),
        ProvenanceItem(
            category="environment",
            key=KEY_PLATFORM,
            value=platform.platform(),
            source="platform",
        ),
        ProvenanceItem(
            category="environment",
            key="implementation",
            value=platform.python_implementation(),
            source="platform",
        ),
        ProvenanceItem(
            category="environment",
            key="executable",
            value=sys.executable,
            source="sys",
        ),
    ]


def dependency_provenance_items() -> List[ProvenanceItem]:
    """Return versions of available core dependencies."""

    items: List[ProvenanceItem] = []
    dependencies = (
        ("numpy", NUMPY_AVAILABLE),
    )
    for name, available in dependencies:
        items.append(
            ProvenanceItem(
                category="dependency",
                key=f"{name}_version",
                value=module_version(name, None) if available else None,
                source=name,
                available=available,
            )
        )
    return items


def chimerax_version(default: Any = None) -> Any:
    """Return the ChimeraX version when available."""

    candidates = (
        "chimerax.core",
        "chimerax",
    )
    for name in candidates:
        module = sys.modules.get(name)
        if module is None:
            try:
                module = __import__(name, fromlist=["*"])
            except ImportError:
                continue
        version = module_version(module, None)
        if version is not None:
            return version
        buildinfo = getattr(module, "buildinfo", None)
        if buildinfo is not None:
            version = get_first_object_field(
                buildinfo,
                ("version", "version_string"),
                None,
                skip_none=True,
            )
            if version is not None:
                return version
    return default


def chimerax_provenance_items() -> List[ProvenanceItem]:
    """Return optional ChimeraX provenance."""

    version = chimerax_version(None)
    return [
        ProvenanceItem(
            category="environment",
            key=KEY_CHIMERAX_VERSION,
            value=version,
            source="chimerax",
            available=version is not None,
        )
    ]


# 16.3. Source-file provenance
# -----------------------------------------------------------------------------

def file_checksum(
    path: PathLike,
    *,
    algorithm: str = CHECKSUM_ALGORITHM,
    chunk_size: int = 1024 * 1024,
) -> str:
    """Calculate a streaming file checksum."""

    try:
        digest = __import__("hashlib").new(algorithm)
    except (ValueError, AttributeError) as error:
        raise ReportProvenanceError(
            f"Unsupported checksum algorithm: {algorithm}.",
            cause=error,
        ) from error

    file_path = Path(path)
    try:
        with file_path.open("rb") as handle:
            while True:
                chunk = handle.read(max(1, int(chunk_size)))
                if not chunk:
                    break
                digest.update(chunk)
    except OSError as error:
        raise ReportProvenanceError(
            "Unable to calculate source-file checksum.",
            path=file_path,
            cause=error,
        ) from error
    return digest.hexdigest()


def source_file_provenance(
    value: Any,
    *,
    include_checksum: bool = False,
    algorithm: str = CHECKSUM_ALGORITHM,
) -> Mapping[str, Any]:
    """Return provenance for one path-like input."""

    path = input_path_value(value)
    if path is None:
        raise ReportProvenanceError(
            "Input does not expose a source path."
        )

    stat = safe_path_stat(path)
    record: Dict[str, Any] = {
        "path": str(path),
        "name": path.name,
        "format": infer_input_format(value, path=path),
        "exists": stat.get("exists"),
        "size_bytes": stat.get("size_bytes"),
        "modified_at": stat.get("modified_at"),
    }
    if include_checksum and stat.get("exists") and path.is_file():
        record[KEY_CHECKSUM] = file_checksum(
            path,
            algorithm=algorithm,
        )
        record["checksum_algorithm"] = algorithm
    return MappingProxyType(record)


def collect_source_files(
    value: Any,
    *,
    include_checksum: bool = False,
    algorithm: str = CHECKSUM_ALGORITHM,
) -> Tuple[Mapping[str, Any], ...]:
    """Collect unique source-file provenance records."""

    records: List[Mapping[str, Any]] = []
    seen: Set[str] = set()

    candidates: List[Any] = []
    if isinstance(value, InputSummary):
        candidates.extend(record.path for record in value.records if record.path)
    else:
        path = input_path_value(value)
        if path is not None:
            candidates.append(path)
        explicit = get_first_object_field(
            value,
            ("source_files", "input_files", "inputs"),
            MISSING,
            skip_none=True,
        )
        if explicit is not MISSING:
            candidates.extend(iter_object_collection(explicit))

    for candidate in candidates:
        path = input_path_value(candidate)
        if path is None:
            continue
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        try:
            records.append(
                source_file_provenance(
                    candidate,
                    include_checksum=include_checksum,
                    algorithm=algorithm,
                )
            )
        except ReportError:
            continue

    return tuple(records)


# 16.4. Parameter provenance
# -----------------------------------------------------------------------------

_PROVENANCE_PARAMETER_FIELDS: Final[Tuple[str, ...]] = (
    "engine",
    "method",
    "exhaustiveness",
    "seed",
    "grid_center",
    "grid_size",
    "distance_cutoff",
    "angle_cutoff",
    "configuration",
    "config",
    "parameters",
    "settings",
)


def collect_parameters(value: Any) -> Mapping[str, Any]:
    """Collect common analysis parameters."""

    parameters: Dict[str, Any] = {}
    for name in _PROVENANCE_PARAMETER_FIELDS:
        item = get_object_field(value, name, MISSING)
        if item is MISSING or item is None:
            continue
        if name in {"configuration", "config", "parameters", "settings"}:
            if is_dataclass_instance(item):
                parameters.update(_config_to_dict(item))
            elif isinstance(item, Mapping):
                parameters.update(dict(item))
            else:
                parameters[name] = safe_string(item, "")
        else:
            parameters[name] = item
    return MappingProxyType(parameters) if parameters else _EMPTY_METADATA


# 16.5. Provenance construction
# -----------------------------------------------------------------------------

def build_provenance_section(
    value: Any = None,
    *,
    config: ProvenanceConfig = DEFAULT_PROVENANCE_CONFIG,
    report_config: Optional[ReportConfig] = None,
) -> ProvenanceSection:
    """Build report-generation provenance."""

    items: List[ProvenanceItem] = []
    warnings_out: List[str] = []

    if config.include_platform or config.include_python:
        for item in python_environment_items():
            if item.key == KEY_PLATFORM and not config.include_platform:
                continue
            if item.key in {
                KEY_PYTHON_VERSION,
                "implementation",
                "executable",
            } and not config.include_python:
                continue
            items.append(item)

    if config.include_dependencies:
        items.extend(dependency_provenance_items())

    if config.include_chimerax:
        items.extend(chimerax_provenance_items())

    source_files = (
        collect_source_files(
            value,
            include_checksum=config.include_checksums,
            algorithm=config.checksum_algorithm,
        )
        if config.include_source_files and value is not None
        else ()
    )

    checksums = {
        record["path"]: record[KEY_CHECKSUM]
        for record in source_files
        if KEY_CHECKSUM in record
    }

    parameters: Mapping[str, Any] = _EMPTY_METADATA
    if config.include_parameters:
        parameters = collect_parameters(value) if value is not None else {}
        if report_config is not None:
            parameters = MappingProxyType(
                {
                    **dict(parameters),
                    "report_config": report_config.to_dict(),
                }
            )

    return ProvenanceSection(
        generated_at=current_utc_timestamp(),
        items=tuple(items),
        source_files=source_files,
        parameters=parameters,
        checksums=checksums,
        metadata={
            **dict(config.extra),
            "module": _MODULE_NAME,
            "module_description": _MODULE_DESCRIPTION,
        },
        warnings=tuple(warnings_out),
    )


def summarize_provenance(
    value: Any = None,
    *,
    config: ReportConfig = DEFAULT_REPORT_CONFIG,
) -> ProvenanceSection:
    """Convenience provenance summary."""

    return build_provenance_section(
        value,
        config=config.provenance,
        report_config=config,
    )


def provenance_rows(
    section: Union[ProvenanceSection, Any],
    *,
    config: ReportConfig = DEFAULT_REPORT_CONFIG,
) -> ReportRows:
    """Return provenance table rows."""

    if not isinstance(section, ProvenanceSection):
        section = summarize_provenance(section, config=config)

    rows: ReportRows = [
        {
            "category": "report",
            "key": KEY_GENERATED_AT,
            "label": field_label(KEY_GENERATED_AT),
            "value": section.generated_at,
            "available": True,
            KEY_SOURCE: SOURCE_MODULE_REPORT,
        },
        {
            "category": "report",
            "key": KEY_GENERATOR,
            "label": field_label(KEY_GENERATOR),
            "value": section.generator,
            "available": True,
            KEY_SOURCE: SOURCE_MODULE_REPORT,
        },
        {
            "category": "report",
            "key": KEY_GENERATOR_VERSION,
            "label": field_label(KEY_GENERATOR_VERSION),
            "value": section.generator_version,
            "available": True,
            KEY_SOURCE: SOURCE_MODULE_REPORT,
        },
    ]
    for item in section.items:
        rows.append(
            {
                "category": item.category,
                "key": item.key,
                "label": field_label(item.key),
                "value": item.value,
                "formatted": format_value(
                    item.value,
                    config=config.formatting,
                ),
                "available": item.available,
                KEY_SOURCE: item.source,
            }
        )
    return rows


def source_file_rows(
    section: Union[ProvenanceSection, Any],
    *,
    config: ReportConfig = DEFAULT_REPORT_CONFIG,
) -> ReportRows:
    """Return source-file provenance rows."""

    if not isinstance(section, ProvenanceSection):
        section = summarize_provenance(section, config=config)

    rows: ReportRows = []
    for index, record in enumerate(section.source_files, start=1):
        rows.append(
            {
                KEY_RANK: index,
                "name": record.get("name"),
                "path": record.get("path"),
                "format": record.get("format"),
                "exists": record.get("exists"),
                "size_bytes": record.get("size_bytes"),
                "size": (
                    format_byte_size(record.get("size_bytes"), missing="")
                    if record.get("size_bytes") is not None
                    else ""
                ),
                "modified_at": record.get("modified_at"),
                KEY_CHECKSUM: record.get(KEY_CHECKSUM),
            }
        )
    return rows


# 16.6. Public provenance interface
# -----------------------------------------------------------------------------

_SECTION_16_PUBLIC_NAMES: Final[Tuple[str, ...]] = (
    "ProvenanceItem",
    "ProvenanceSection",
    "current_utc_timestamp",
    "module_version",
    "python_environment_items",
    "dependency_provenance_items",
    "chimerax_version",
    "chimerax_provenance_items",
    "file_checksum",
    "source_file_provenance",
    "collect_source_files",
    "collect_parameters",
    "build_provenance_section",
    "summarize_provenance",
    "provenance_rows",
    "source_file_rows",
)

_register_public_names(_SECTION_16_PUBLIC_NAMES)

# =============================================================================
# End of Section 16
# =============================================================================

# =============================================================================
# Section 17 — Section construction
# =============================================================================

# 17.1. Structured report blocks and sections
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class ReportBlock:
    """One structured unit inside a report section."""

    kind: ReportBlockKind
    content: Any = None
    title: str = ""
    name: str = ""
    level: int = 0
    visible: bool = True
    metadata: Mapping[str, Any] = field(
        default_factory=lambda: _EMPTY_METADATA
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "kind",
            _coerce_enum(
                ReportBlockKind,
                self.kind,
                "kind",
            ),
        )
        for field_name in ("title", "name"):
            object.__setattr__(
                self,
                field_name,
                single_line_text(getattr(self, field_name), ""),
            )
        object.__setattr__(
            self,
            "level",
            max(0, to_safe_int(self.level, 0)),
        )
        object.__setattr__(self, "visible", bool(self.visible))
        object.__setattr__(
            self,
            "metadata",
            _freeze_config_mapping(self.metadata),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain block record."""

        return {
            "kind": self.kind.value,
            "content": self.content,
            KEY_TITLE: self.title,
            "name": self.name,
            "level": self.level,
            "visible": self.visible,
            KEY_METADATA: dict(self.metadata),
        }


@dataclass(frozen=True)
class ReportSection:
    """Structured report section."""

    id: ReportSectionID
    title: str
    description: str = ""
    blocks: Tuple[ReportBlock, ...] = ()
    order: int = 0
    enabled: bool = True
    empty: bool = False
    metadata: Mapping[str, Any] = field(
        default_factory=lambda: _EMPTY_METADATA
    )
    warnings: Tuple[str, ...] = ()
    errors: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "id",
            _coerce_enum(
                ReportSectionID,
                self.id,
                "id",
            ),
        )
        object.__setattr__(
            self,
            "title",
            single_line_text(
                self.title,
                SECTION_TITLES.get(self.id.value, self.id.value),
            ),
        )
        object.__setattr__(
            self,
            "description",
            safe_string(self.description, ""),
        )
        object.__setattr__(self, "blocks", tuple(self.blocks))
        object.__setattr__(
            self,
            "order",
            max(0, to_safe_int(self.order, 0)),
        )
        object.__setattr__(self, "enabled", bool(self.enabled))
        object.__setattr__(self, "empty", bool(self.empty))
        object.__setattr__(
            self,
            "metadata",
            _freeze_config_mapping(self.metadata),
        )
        object.__setattr__(
            self,
            "warnings",
            _freeze_config_strings(self.warnings, unique=False),
        )
        object.__setattr__(
            self,
            "errors",
            _freeze_config_strings(self.errors, unique=False),
        )

    @property
    def visible_blocks(self) -> Tuple[ReportBlock, ...]:
        """Return visible blocks only."""

        return tuple(block for block in self.blocks if block.visible)

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain section record."""

        return {
            KEY_ID: self.id.value,
            KEY_TITLE: self.title,
            KEY_DESCRIPTION: self.description,
            "order": self.order,
            "enabled": self.enabled,
            "empty": self.empty,
            "blocks": [block.to_dict() for block in self.blocks],
            KEY_METADATA: dict(self.metadata),
            KEY_WARNINGS: list(self.warnings),
            KEY_ERRORS: list(self.errors),
        }


@dataclass(frozen=True)
class ReportDocument:
    """Complete structured report before rendering."""

    title: str = DEFAULT_REPORT_TITLE
    subtitle: str = DEFAULT_REPORT_SUBTITLE
    description: str = ""
    generated_at: str = ""
    sections: Tuple[ReportSection, ...] = ()
    config: ReportConfig = field(default_factory=ReportConfig)
    schema_name: str = REPORT_SCHEMA_NAME
    schema_version: str = REPORT_SCHEMA_VERSION
    generator: str = REPORT_GENERATOR_NAME
    generator_version: str = __version__
    metadata: Mapping[str, Any] = field(
        default_factory=lambda: _EMPTY_METADATA
    )
    warnings: Tuple[str, ...] = ()
    errors: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field_name in (
            "title",
            "subtitle",
            "schema_name",
            "schema_version",
            "generator",
            "generator_version",
        ):
            object.__setattr__(
                self,
                field_name,
                single_line_text(getattr(self, field_name), ""),
            )
        object.__setattr__(
            self,
            "description",
            safe_string(self.description, ""),
        )
        object.__setattr__(
            self,
            "generated_at",
            self.generated_at or current_utc_timestamp(),
        )
        object.__setattr__(self, "sections", tuple(self.sections))
        if not isinstance(self.config, ReportConfig):
            raise ReportConfigurationError(
                "config must be ReportConfig."
            )
        object.__setattr__(
            self,
            "metadata",
            _freeze_config_mapping(self.metadata),
        )
        object.__setattr__(
            self,
            "warnings",
            _freeze_config_strings(self.warnings, unique=False),
        )
        object.__setattr__(
            self,
            "errors",
            _freeze_config_strings(self.errors, unique=False),
        )

    @property
    def visible_sections(self) -> Tuple[ReportSection, ...]:
        """Return enabled non-empty sections according to configuration."""

        include_empty = self.config.rendering.include_empty_sections
        return tuple(
            section
            for section in sorted(
                self.sections,
                key=lambda item: (item.order, item.id.value),
            )
            if section.enabled and (include_empty or not section.empty)
        )

    def get_section(self, section_id: Any) -> Optional[ReportSection]:
        """Return a section by identifier."""

        member = _coerce_enum(
            ReportSectionID,
            section_id,
            "section_id",
        )
        return next(
            (
                section
                for section in self.sections
                if section.id is member
            ),
            None,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain report document."""

        return {
            KEY_SCHEMA_NAME: self.schema_name,
            KEY_SCHEMA_VERSION: self.schema_version,
            KEY_GENERATOR: self.generator,
            KEY_GENERATOR_VERSION: self.generator_version,
            KEY_GENERATED_AT: self.generated_at,
            KEY_TITLE: self.title,
            KEY_SUBTITLE: self.subtitle,
            KEY_DESCRIPTION: self.description,
            KEY_SECTIONS: [
                section.to_dict() for section in self.sections
            ],
            KEY_METADATA: dict(self.metadata),
            KEY_WARNINGS: list(self.warnings),
            KEY_ERRORS: list(self.errors),
        }


# 17.2. Build context
# -----------------------------------------------------------------------------

@dataclass
class ReportBuildContext:
    """Mutable cache used while constructing report sections."""

    value: Any
    config: ReportConfig = field(default_factory=ReportConfig)
    overview: Optional[PoseOverview] = None
    inputs: Optional[InputSummary] = None
    interactions: Optional[InteractionSection] = None
    residues: Optional[List[ResidueSummary]] = None
    hotspots: Optional[HotspotSection] = None
    scoring: Optional[ScoringSection] = None
    multipose: Optional[MultiposeSummary] = None
    provenance: Optional[ProvenanceSection] = None
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def strict(self) -> bool:
        """Return whether section construction is strict."""

        return self.config.errors.mode is ErrorMode.RAISE


# 17.3. Block factories
# -----------------------------------------------------------------------------

def paragraph_block(
    content: Any,
    *,
    title: str = "",
    visible: bool = True,
    metadata: Optional[Mapping[str, Any]] = None,
) -> ReportBlock:
    """Create a paragraph block."""

    return ReportBlock(
        kind=ReportBlockKind.PARAGRAPH,
        content=safe_string(content, ""),
        title=title,
        visible=visible,
        metadata=metadata or {},
    )


def key_value_block(
    content: Mapping[str, Any],
    *,
    title: str = "",
    visible: bool = True,
    metadata: Optional[Mapping[str, Any]] = None,
) -> ReportBlock:
    """Create a key-value block."""

    return ReportBlock(
        kind=ReportBlockKind.KEY_VALUE,
        content=dict(content),
        title=title,
        visible=visible,
        metadata=metadata or {},
    )


def table_block(
    content: Any,
    *,
    title: str = "",
    name: str = "",
    visible: bool = True,
    metadata: Optional[Mapping[str, Any]] = None,
) -> ReportBlock:
    """Create a table block."""

    return ReportBlock(
        kind=ReportBlockKind.TABLE,
        content=content,
        title=title,
        name=name,
        visible=visible,
        metadata=metadata or {},
    )


def list_block(
    content: Any,
    *,
    title: str = "",
    ordered: bool = False,
    visible: bool = True,
    metadata: Optional[Mapping[str, Any]] = None,
) -> ReportBlock:
    """Create a list block."""

    merged = dict(metadata or {})
    merged["ordered"] = bool(ordered)
    return ReportBlock(
        kind=ReportBlockKind.LIST,
        content=tuple(iter_object_collection(content)),
        title=title,
        visible=visible,
        metadata=merged,
    )


def notice_block(
    content: Any,
    *,
    title: str = "",
    severity: Severity = Severity.INFO,
    visible: bool = True,
) -> ReportBlock:
    """Create a diagnostic notice block."""

    return ReportBlock(
        kind=ReportBlockKind.NOTICE,
        content=safe_string(content, ""),
        title=title,
        visible=visible,
        metadata={"severity": Severity.coerce(severity).value},
    )


# 17.4. Section builders
# -----------------------------------------------------------------------------

def _section_order(
    config: ReportConfig,
    section_id: ReportSectionID,
) -> int:
    """Return the configured section order."""

    try:
        return config.section_order.index(section_id)
    except ValueError:
        return len(config.section_order)


def _section_base(
    section_id: Any,
    blocks: Iterable[ReportBlock],
    *,
    config: ReportConfig,
    description: Optional[str] = None,
    warnings: Iterable[str] = (),
    errors: Iterable[str] = (),
    metadata: Optional[Mapping[str, Any]] = None,
) -> ReportSection:
    """Build a standard section shell."""

    member = _coerce_enum(
        ReportSectionID,
        section_id,
        "section_id",
    )
    block_values = tuple(blocks)
    visible = tuple(block for block in block_values if block.visible)
    return ReportSection(
        id=member,
        title=SECTION_TITLES[member.value],
        description=(
            SECTION_DESCRIPTIONS.get(member.value, "")
            if description is None
            else description
        ),
        blocks=block_values,
        order=_section_order(config, member),
        enabled=config.is_section_enabled(member),
        empty=not visible,
        metadata=metadata or {},
        warnings=tuple(warnings),
        errors=tuple(errors),
    )


def build_overview_report_section(
    context: ReportBuildContext,
) -> ReportSection:
    """Build the overview report section."""

    if context.overview is None:
        context.overview = summarize_pose(
            context.value,
            config=context.config,
            strict=context.strict,
        )
    overview = context.overview
    blocks = [
        key_value_block(
            {
                row["label"]: row["value"]
                for row in pose_overview_rows(
                    overview,
                    config=context.config,
                )
            },
            title="Pose summary",
        )
    ]
    if overview.family_counts:
        blocks.append(
            table_block(
                [
                    {
                        KEY_FAMILY: family,
                        KEY_COUNT: count,
                    }
                    for family, count in overview.family_counts.items()
                ],
                title="Interaction families",
                name=TABLE_OVERVIEW,
            )
        )
    return _section_base(
        ReportSectionID.OVERVIEW,
        blocks,
        config=context.config,
        warnings=overview.warnings,
        errors=overview.errors,
    )


def build_inputs_report_section(
    context: ReportBuildContext,
) -> ReportSection:
    """Build the input-summary report section."""

    if context.inputs is None:
        pose_inputs = collect_pose_inputs(context.value)
        context.inputs = summarize_inputs(
            [
                normalize_input_record(
                    value,
                    role_hint=role,
                    include_metadata=(
                        context.config.rendering.detail
                        in {ReportDetail.DETAILED, ReportDetail.FULL}
                    ),
                )
                for role, value in pose_inputs
            ]
        )
    summary = context.inputs
    blocks: List[ReportBlock] = [
        key_value_block(
            {
                "Total inputs": summary.total_inputs,
                "Existing paths": summary.existing_paths,
                "Missing paths": summary.missing_paths,
                "Total size": format_byte_size(
                    summary.total_size_bytes,
                    missing="0 B",
                ),
            },
            title="Input summary",
        )
    ]
    rows = input_summary_rows(summary)
    if rows:
        blocks.append(
            table_block(
                rows,
                title="Input records",
                name=TABLE_INPUTS,
            )
        )
    return _section_base(
        ReportSectionID.INPUTS,
        blocks,
        config=context.config,
        warnings=summary.warnings,
    )


def build_interactions_report_section(
    context: ReportBuildContext,
) -> ReportSection:
    """Build the interaction report section."""

    if context.interactions is None:
        context.interactions = summarize_interactions(
            context.value,
            config=context.config,
            strict=context.strict,
        )
    section = context.interactions
    blocks: List[ReportBlock] = [
        key_value_block(
            {
                "Total interactions": section.total_interactions,
                "Favorable interactions": section.favorable_interactions,
                "Penalty interactions": section.penalty_interactions,
                "Residues": section.residue_count,
                "Total score": section.total_score,
            },
            title="Interaction summary",
        )
    ]
    family_rows = interaction_family_rows(
        section,
        config=context.config,
    )
    if family_rows:
        blocks.append(
            table_block(
                family_rows,
                title="Interaction families",
                name=TABLE_INTERACTIONS,
            )
        )
    detail_rows = interaction_rows(
        section,
        config=context.config,
    )
    if detail_rows:
        blocks.append(
            table_block(
                detail_rows,
                title="Interaction details",
                name=TABLE_INTERACTIONS,
                metadata={"detail": True},
            )
        )
    return _section_base(
        ReportSectionID.INTERACTIONS,
        blocks,
        config=context.config,
        warnings=section.warnings,
        errors=section.errors,
    )


def build_residues_report_section(
    context: ReportBuildContext,
) -> ReportSection:
    """Build the residue-summary report section."""

    if context.interactions is None:
        context.interactions = summarize_interactions(
            context.value,
            config=context.config,
            strict=context.strict,
        )
    if context.residues is None:
        context.residues = summarize_residues(
            None,
            interactions=context.interactions.interactions,
            interaction_config=context.config.interactions,
            top_n=context.config.multipose.top_residues,
            include_details=(
                context.config.rendering.detail
                in {ReportDetail.DETAILED, ReportDetail.FULL}
            ),
            score_direction=context.config.scoring.score_direction,
            strict=context.strict,
        )
    rows = residue_summary_rows(
        context.residues,
        config=context.config,
    )
    totals = residue_summary_totals(context.residues)
    blocks: List[ReportBlock] = [
        key_value_block(
            {
                "Total residues": totals[KEY_TOTAL_RESIDUES],
                "Total interactions": totals[KEY_TOTAL_INTERACTIONS],
                "Favorable interactions": totals[
                    "favorable_interactions"
                ],
                "Penalty interactions": totals["penalty_interactions"],
                "Total score": totals[KEY_TOTAL_SCORE],
            },
            title="Residue totals",
        )
    ]
    if rows:
        blocks.append(
            table_block(
                rows,
                title="Residue ranking",
                name=TABLE_RESIDUES,
            )
        )
    return _section_base(
        ReportSectionID.RESIDUES,
        blocks,
        config=context.config,
    )


def build_hotspots_report_section(
    context: ReportBuildContext,
) -> ReportSection:
    """Build the hotspot report section."""

    if context.residues is None:
        build_residues_report_section(context)
    if context.hotspots is None:
        context.hotspots = identify_hotspots(
            context.residues or (),
            top_n=context.config.multipose.top_hotspots,
        )
    section = context.hotspots
    blocks: List[ReportBlock] = [
        key_value_block(
            {
                "Candidates": section.total_candidates,
                "Selected hotspots": section.selected_count,
                "Minimum interactions": section.minimum_interactions,
            },
            title="Hotspot summary",
        )
    ]
    rows = hotspot_rows(section, config=context.config)
    if rows:
        blocks.append(
            table_block(
                rows,
                title="Ranked hotspots",
                name=TABLE_HOTSPOTS,
            )
        )
    return _section_base(
        ReportSectionID.HOTSPOTS,
        blocks,
        config=context.config,
        warnings=section.warnings,
    )


def build_scoring_report_section(
    context: ReportBuildContext,
) -> ReportSection:
    """Build the scoring and explainability report section."""

    if context.interactions is None:
        build_interactions_report_section(context)
    if context.residues is None:
        build_residues_report_section(context)
    if context.scoring is None:
        context.scoring = build_scoring_section(
            context.value,
            interactions=(
                context.interactions.interactions
                if context.interactions is not None
                else ()
            ),
            residues=context.residues or (),
            config=context.config.scoring,
            formatting=context.config.formatting,
            strict=context.strict,
        )
    section = context.scoring
    blocks: List[ReportBlock] = [
        key_value_block(
            {
                "Total score": section.total_score,
                "Raw score": section.raw_score,
                "Normalized score": section.normalized_score,
                "Docking affinity": section.affinity,
                "Favorable components": section.favorable_components,
                "Unfavorable components": section.unfavorable_components,
            },
            title="Score summary",
        )
    ]
    component_rows = score_component_rows(
        section,
        config=context.config,
    )
    if component_rows:
        blocks.append(
            table_block(
                component_rows,
                title="Score components",
                name=TABLE_SCORE_COMPONENTS,
            )
        )
    explanation_values = explanation_rows(
        section,
        config=context.config,
    )
    if explanation_values:
        blocks.append(
            table_block(
                explanation_values,
                title="Explainability",
                name=TABLE_EXPLAINABILITY,
            )
        )
    return _section_base(
        ReportSectionID.SCORING,
        blocks,
        config=context.config,
        warnings=section.warnings,
        errors=section.errors,
    )


def build_multipose_report_section(
    context: ReportBuildContext,
) -> ReportSection:
    """Build the multipose report section."""

    values = list(iter_object_collection(context.value))
    is_multi = (
        len(values) > 1
        and not isinstance(context.value, Mapping)
    )
    if not is_multi:
        return _section_base(
            ReportSectionID.MULTIPOSE,
            (),
            config=context.config,
            metadata={"applicable": False},
        )

    if context.multipose is None:
        context.multipose = summarize_multipose(
            values,
            config=context.config,
        )
    section = context.multipose
    blocks: List[ReportBlock] = [
        key_value_block(
            {
                "Total poses": section.total_poses,
                "Best pose": section.best_pose_id,
                "Ranking metric": section.ranking_metric,
                "Direction": section.rank_direction.value,
            },
            title="Multipose summary",
        )
    ]
    ranking_rows = multipose_ranking_rows(
        section,
        config=context.config,
    )
    if ranking_rows:
        blocks.append(
            table_block(
                ranking_rows,
                title="Pose ranking",
                name=TABLE_RANKING,
            )
        )
    if section.residue_persistence:
        blocks.append(
            table_block(
                persistence_rows(
                    section.residue_persistence,
                    counts=section.residue_pose_counts,
                    total=section.total_poses,
                    formatting=context.config.formatting,
                ),
                title="Residue persistence",
                name=TABLE_PERSISTENCE,
            )
        )
    return _section_base(
        ReportSectionID.MULTIPOSE,
        blocks,
        config=context.config,
        warnings=section.warnings,
        errors=section.errors,
    )


def build_provenance_report_section(
    context: ReportBuildContext,
) -> ReportSection:
    """Build the provenance report section."""

    if context.provenance is None:
        context.provenance = summarize_provenance(
            context.value,
            config=context.config,
        )
    section = context.provenance
    blocks: List[ReportBlock] = [
        table_block(
            provenance_rows(
                section,
                config=context.config,
            ),
            title="Environment",
            name=TABLE_PROVENANCE,
        )
    ]
    file_rows = source_file_rows(
        section,
        config=context.config,
    )
    if file_rows:
        blocks.append(
            table_block(
                file_rows,
                title="Source files",
                name=TABLE_PROVENANCE,
            )
        )
    if section.parameters:
        blocks.append(
            key_value_block(
                section.parameters,
                title="Parameters",
            )
        )
    return _section_base(
        ReportSectionID.PROVENANCE,
        blocks,
        config=context.config,
        warnings=section.warnings,
    )


def build_diagnostics_report_section(
    context: ReportBuildContext,
    section_id: ReportSectionID,
) -> ReportSection:
    """Build warnings or errors section."""

    values = (
        context.warnings
        if section_id is ReportSectionID.WARNINGS
        else context.errors
    )
    severity = (
        Severity.WARNING
        if section_id is ReportSectionID.WARNINGS
        else Severity.ERROR
    )
    blocks = [
        notice_block(
            message,
            severity=severity,
        )
        for message in values[
            : context.config.errors.max_messages
        ]
    ]
    return _section_base(
        section_id,
        blocks,
        config=context.config,
    )


# 17.5. Section registry and report construction
# -----------------------------------------------------------------------------

_SECTION_BUILDERS: Final[
    Mapping[ReportSectionID, Callable[[ReportBuildContext], ReportSection]]
] = MappingProxyType(
    {
        ReportSectionID.OVERVIEW: build_overview_report_section,
        ReportSectionID.INPUTS: build_inputs_report_section,
        ReportSectionID.INTERACTIONS: build_interactions_report_section,
        ReportSectionID.RESIDUES: build_residues_report_section,
        ReportSectionID.HOTSPOTS: build_hotspots_report_section,
        ReportSectionID.SCORING: build_scoring_report_section,
        ReportSectionID.MULTIPOSE: build_multipose_report_section,
        ReportSectionID.PROVENANCE: build_provenance_report_section,
    }
)


def build_report_section(
    section_id: Any,
    context: ReportBuildContext,
) -> ReportSection:
    """Build one configured report section."""

    member = _coerce_enum(
        ReportSectionID,
        section_id,
        "section_id",
    )
    if member is ReportSectionID.WARNINGS:
        return build_diagnostics_report_section(context, member)
    if member is ReportSectionID.ERRORS:
        return build_diagnostics_report_section(context, member)

    builder = _SECTION_BUILDERS.get(member)
    if builder is None:
        raise ReportSectionError(
            f"No builder registered for section {member.value!r}.",
            section=member.value,
        )
    try:
        return builder(context)
    except ReportError as error:
        if context.strict:
            raise
        context.errors.append(str(error))
        return _section_base(
            member,
            (
                notice_block(
                    str(error),
                    severity=Severity.ERROR,
                ),
            ),
            config=context.config,
            errors=(str(error),),
        )
    except Exception as error:
        wrapped = ReportSectionError(
            "Unable to build report section.",
            section=member.value,
            cause=error,
        )
        if context.strict:
            raise wrapped from error
        context.errors.append(str(wrapped))
        return _section_base(
            member,
            (
                notice_block(
                    str(wrapped),
                    severity=Severity.ERROR,
                ),
            ),
            config=context.config,
            errors=(str(wrapped),),
        )


def build_report_sections(
    value: Any,
    *,
    config: ReportConfig = DEFAULT_REPORT_CONFIG,
) -> Tuple[ReportSection, ...]:
    """Build all configured report sections."""

    context = ReportBuildContext(value=value, config=config)
    sections: List[ReportSection] = []

    for section_id in config.ordered_sections(enabled_only=False):
        sections.append(
            build_report_section(section_id, context)
        )

    if (
        context.warnings
        and ReportSectionID.WARNINGS not in config.section_order
    ):
        sections.append(
            build_diagnostics_report_section(
                context,
                ReportSectionID.WARNINGS,
            )
        )
    if (
        context.errors
        and ReportSectionID.ERRORS not in config.section_order
    ):
        sections.append(
            build_diagnostics_report_section(
                context,
                ReportSectionID.ERRORS,
            )
        )
    return tuple(sections)


def build_report_document(
    value: Any,
    *,
    config: ReportConfig = DEFAULT_REPORT_CONFIG,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    description: Optional[str] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> ReportDocument:
    """Build a complete structured report document."""

    sections = build_report_sections(value, config=config)
    warnings_out = tuple(
        message
        for section in sections
        for message in section.warnings
    )
    errors_out = tuple(
        message
        for section in sections
        for message in section.errors
    )
    return ReportDocument(
        title=config.title if title is None else title,
        subtitle=config.subtitle if subtitle is None else subtitle,
        description=(
            config.description
            if description is None
            else description
        ),
        sections=sections,
        config=config,
        metadata={
            **dict(config.metadata),
            **dict(metadata or {}),
        },
        warnings=warnings_out,
        errors=errors_out,
    )


# 17.6. Public construction interface
# -----------------------------------------------------------------------------

_SECTION_17_PUBLIC_NAMES: Final[Tuple[str, ...]] = (
    "ReportBlock",
    "ReportSection",
    "ReportDocument",
    "ReportBuildContext",
    "paragraph_block",
    "key_value_block",
    "table_block",
    "list_block",
    "notice_block",
    "build_overview_report_section",
    "build_inputs_report_section",
    "build_interactions_report_section",
    "build_residues_report_section",
    "build_hotspots_report_section",
    "build_scoring_report_section",
    "build_multipose_report_section",
    "build_provenance_report_section",
    "build_diagnostics_report_section",
    "build_report_section",
    "build_report_sections",
    "build_report_document",
)

_register_public_names(_SECTION_17_PUBLIC_NAMES)

# =============================================================================
# End of Section 17
# =============================================================================


# =============================================================================
# Section 18 — Internal tables
# =============================================================================

# 18.1. Table columns and tables
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class ReportColumn:
    """Internal table-column definition."""

    key: str
    label: str = ""
    align: str = "left"
    formatter: Optional[ValueFormatter] = None
    visible: bool = True
    minimum_width: int = 0
    maximum_width: Optional[int] = None
    metadata: Mapping[str, Any] = field(
        default_factory=lambda: _EMPTY_METADATA
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "key",
            single_line_text(self.key, ""),
        )
        object.__setattr__(
            self,
            "label",
            single_line_text(
                self.label,
                field_label(self.key),
            ),
        )
        align = normalize_field_name(self.align)
        if align not in {"left", "right", "center"}:
            raise ReportTableError(
                f"Unsupported column alignment: {self.align!r}."
            )
        object.__setattr__(self, "align", align)
        object.__setattr__(self, "visible", bool(self.visible))
        object.__setattr__(
            self,
            "minimum_width",
            max(0, to_safe_int(self.minimum_width, 0)),
        )
        if self.maximum_width is not None:
            object.__setattr__(
                self,
                "maximum_width",
                max(
                    self.minimum_width,
                    to_safe_int(
                        self.maximum_width,
                        self.minimum_width,
                    ),
                ),
            )
        object.__setattr__(
            self,
            "metadata",
            _freeze_config_mapping(self.metadata),
        )

    def format(
        self,
        value: Any,
        *,
        config: FormattingConfig = DEFAULT_FORMATTING_CONFIG,
    ) -> str:
        """Format one cell value."""

        if self.formatter is not None:
            try:
                return safe_string(self.formatter(value), "")
            except Exception:
                return format_value(value, config=config)
        return format_value(value, config=config)


@dataclass(frozen=True)
class ReportTable:
    """Normalized internal table."""

    name: str
    columns: Tuple[ReportColumn, ...]
    rows: Tuple[Mapping[str, Any], ...] = ()
    title: str = ""
    caption: str = ""
    truncated: bool = False
    total_rows: Optional[int] = None
    metadata: Mapping[str, Any] = field(
        default_factory=lambda: _EMPTY_METADATA
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "name",
            single_line_text(self.name, "table"),
        )
        object.__setattr__(self, "columns", tuple(self.columns))
        object.__setattr__(
            self,
            "rows",
            tuple(MappingProxyType(dict(row)) for row in self.rows),
        )
        object.__setattr__(
            self,
            "title",
            single_line_text(self.title, ""),
        )
        object.__setattr__(
            self,
            "caption",
            safe_string(self.caption, ""),
        )
        object.__setattr__(self, "truncated", bool(self.truncated))
        object.__setattr__(
            self,
            "total_rows",
            (
                len(self.rows)
                if self.total_rows is None
                else max(0, to_safe_int(self.total_rows, len(self.rows)))
            ),
        )
        object.__setattr__(
            self,
            "metadata",
            _freeze_config_mapping(self.metadata),
        )

    @property
    def visible_columns(self) -> Tuple[ReportColumn, ...]:
        """Return visible columns only."""

        return tuple(column for column in self.columns if column.visible)

    @property
    def empty(self) -> bool:
        """Return whether the table has no rows."""

        return not self.rows

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain table record."""

        return {
            "name": self.name,
            KEY_TITLE: self.title,
            "caption": self.caption,
            "columns": [
                {
                    "key": column.key,
                    "label": column.label,
                    "align": column.align,
                    "visible": column.visible,
                }
                for column in self.columns
            ],
            "rows": [dict(row) for row in self.rows],
            "truncated": self.truncated,
            "total_rows": self.total_rows,
            KEY_METADATA: dict(self.metadata),
        }


# 18.2. Column inference
# -----------------------------------------------------------------------------

_NUMERIC_COLUMN_KEYS: Final[FrozenSet[str]] = frozenset(
    {
        KEY_RANK,
        KEY_COUNT,
        KEY_SCORE,
        KEY_RAW_SCORE,
        KEY_NORMALIZED_SCORE,
        KEY_AFFINITY,
        KEY_DISTANCE,
        KEY_ANGLE,
        KEY_PERCENT,
        KEY_TOTAL_SCORE,
        KEY_TOTAL_INTERACTIONS,
        KEY_TOTAL_RESIDUES,
        KEY_TOTAL_POSES,
        "interaction_count",
        "residue_count",
        "atom_count",
        "size_bytes",
        "hotspot_score",
        "family_diversity",
        "type_diversity",
        "favorable_interactions",
        "penalty_interactions",
        "minimum_distance",
        "mean_distance",
        "maximum_distance",
        "mean_score",
        "contribution",
        "weight",
        "impact",
    }
)


def infer_column_alignment(
    key: Any,
    values: Iterable[Any] = (),
) -> str:
    """Infer table-column alignment."""

    normalized = normalize_field_name(key)
    if normalized in _NUMERIC_COLUMN_KEYS:
        return "right"
    non_missing = [
        value
        for value in values
        if value not in (None, "", MISSING)
    ]
    if non_missing and all(is_numeric_value(value) for value in non_missing):
        return "right"
    if non_missing and all(is_boolean_value(value) for value in non_missing):
        return "center"
    return "left"


def infer_table_columns(
    rows: Sequence[Mapping[str, Any]],
    *,
    preferred: Sequence[str] = (),
    labels: Mapping[str, str] = COLUMN_LABELS,
    include_empty: bool = False,
) -> Tuple[ReportColumn, ...]:
    """Infer ordered columns from heterogeneous rows."""

    discovered: List[str] = []
    for row in rows:
        for key in row:
            text = str(key)
            if text not in discovered:
                discovered.append(text)

    ordered: List[str] = []
    for key in preferred:
        if key in discovered and key not in ordered:
            ordered.append(key)
    for key in discovered:
        if key not in ordered:
            ordered.append(key)

    columns: List[ReportColumn] = []
    for key in ordered:
        values = [row.get(key) for row in rows]
        if not include_empty and all(
            value in (None, "", (), [], {}, MISSING)
            for value in values
        ):
            continue
        columns.append(
            ReportColumn(
                key=key,
                label=labels.get(
                    normalize_field_name(key),
                    field_label(key),
                ),
                align=infer_column_alignment(key, values),
            )
        )
    return tuple(columns)


# 18.3. Row normalization
# -----------------------------------------------------------------------------

def normalize_table_row(
    row: Any,
    *,
    columns: Optional[Sequence[ReportColumn]] = None,
) -> ReportRow:
    """Normalize one table row."""

    if isinstance(row, Mapping):
        record = dict(row)
    elif is_dataclass_instance(row):
        record = object_to_shallow_dict(row)
    else:
        record = object_to_shallow_dict(
            row,
            include_properties=False,
        )
        if not record:
            record = {"value": row}

    if columns is None:
        return {
            str(key): value
            for key, value in record.items()
        }

    return {
        column.key: record.get(column.key)
        for column in columns
    }


def normalize_table_rows(
    rows: Any,
    *,
    columns: Optional[Sequence[ReportColumn]] = None,
    max_rows: int = DEFAULT_MAX_ROWS,
) -> Tuple[Tuple[Mapping[str, Any], ...], bool, int]:
    """Normalize and limit table rows."""

    source_rows = list(iter_object_collection(rows))
    total = len(source_rows)
    limit = max(0, int(max_rows))
    visible = source_rows[:limit]
    normalized = tuple(
        MappingProxyType(
            normalize_table_row(
                row,
                columns=columns,
            )
        )
        for row in visible
    )
    return normalized, total > len(normalized), total


# 18.4. Table construction
# -----------------------------------------------------------------------------

def build_report_table(
    rows: Any,
    *,
    name: str = "table",
    title: str = "",
    caption: str = "",
    columns: Optional[Sequence[Union[ReportColumn, str]]] = None,
    preferred_columns: Sequence[str] = (),
    config: TableConfig = DEFAULT_TABLE_CONFIG,
    metadata: Optional[Mapping[str, Any]] = None,
) -> ReportTable:
    """Build an internal report table."""

    source_rows = list(iter_object_collection(rows))

    explicit_columns: Optional[Tuple[ReportColumn, ...]] = None
    if columns is not None:
        explicit_columns = tuple(
            column
            if isinstance(column, ReportColumn)
            else ReportColumn(key=str(column))
            for column in columns
        )

    if explicit_columns is None:
        explicit_columns = infer_table_columns(
            [
                normalize_table_row(row)
                for row in source_rows
            ],
            preferred=preferred_columns,
            labels=config.column_labels,
            include_empty=config.include_empty_columns,
        )

    normalized_rows, truncated, total = normalize_table_rows(
        source_rows,
        columns=explicit_columns,
        max_rows=config.max_rows,
    )

    return ReportTable(
        name=name,
        columns=explicit_columns,
        rows=normalized_rows,
        title=title,
        caption=caption,
        truncated=truncated,
        total_rows=total,
        metadata=metadata or {},
    )


def coerce_report_table(
    value: Any,
    *,
    name: str = "table",
    title: str = "",
    config: TableConfig = DEFAULT_TABLE_CONFIG,
) -> ReportTable:
    """Return a report table from a table or row collection."""

    if isinstance(value, ReportTable):
        return value
    preferred = config.preferred_columns.get(name, ())
    return build_report_table(
        value,
        name=name,
        title=title,
        preferred_columns=preferred,
        config=config,
    )


def table_from_block(
    block: ReportBlock,
    *,
    config: TableConfig = DEFAULT_TABLE_CONFIG,
) -> ReportTable:
    """Convert a table block to an internal table."""

    if block.kind is not ReportBlockKind.TABLE:
        raise ReportTableError(
            "Report block is not a table.",
            context={"kind": block.kind.value},
        )
    return coerce_report_table(
        block.content,
        name=block.name or "table",
        title=block.title,
        config=config,
    )


# 18.5. Standard tables from report sections
# -----------------------------------------------------------------------------

def section_tables(
    section: ReportSection,
    *,
    config: ReportConfig = DEFAULT_REPORT_CONFIG,
) -> Tuple[ReportTable, ...]:
    """Return all internal tables from a section."""

    tables: List[ReportTable] = []
    for block in section.visible_blocks:
        if block.kind is not ReportBlockKind.TABLE:
            continue
        tables.append(
            table_from_block(
                block,
                config=config.tables,
            )
        )
    return tuple(tables)


def report_tables(
    report: ReportDocument,
) -> Dict[str, Tuple[ReportTable, ...]]:
    """Return internal tables indexed by section."""

    return {
        section.id.value: section_tables(
            section,
            config=report.config,
        )
        for section in report.visible_sections
        if section_tables(section, config=report.config)
    }


# 18.6. Cell formatting and widths
# -----------------------------------------------------------------------------

def formatted_table_rows(
    table: ReportTable,
    *,
    formatting: FormattingConfig = DEFAULT_FORMATTING_CONFIG,
    max_cell_length: int = DEFAULT_MAX_CELL_LENGTH,
) -> Tuple[Tuple[str, ...], ...]:
    """Return formatted visible table cells."""

    columns = table.visible_columns
    return tuple(
        tuple(
            truncate_text(
                column.format(
                    row.get(column.key),
                    config=formatting,
                ),
                max_cell_length,
                marker=formatting.truncation_marker,
            )
            for column in columns
        )
        for row in table.rows
    )


def table_column_widths(
    table: ReportTable,
    *,
    formatting: FormattingConfig = DEFAULT_FORMATTING_CONFIG,
    max_cell_length: int = DEFAULT_MAX_CELL_LENGTH,
) -> Tuple[int, ...]:
    """Calculate display widths for visible columns."""

    columns = table.visible_columns
    rows = formatted_table_rows(
        table,
        formatting=formatting,
        max_cell_length=max_cell_length,
    )
    widths: List[int] = []
    for index, column in enumerate(columns):
        width = max(
            len(column.label),
            *(
                len(row[index])
                for row in rows
            ),
            column.minimum_width,
        )
        if column.maximum_width is not None:
            width = min(width, column.maximum_width)
        width = min(width, max_cell_length)
        widths.append(width)
    return tuple(widths)


def align_table_cell(
    text: Any,
    width: int,
    align: str = "left",
) -> str:
    """Align one table cell."""

    value = safe_string(text, "", strip=False)
    if len(value) > width:
        value = truncate_text(value, width)
    if align == "right":
        return value.rjust(width)
    if align == "center":
        return value.center(width)
    return value.ljust(width)


# 18.7. Public table interface
# -----------------------------------------------------------------------------

_SECTION_18_PUBLIC_NAMES: Final[Tuple[str, ...]] = (
    "ReportColumn",
    "ReportTable",
    "infer_column_alignment",
    "infer_table_columns",
    "normalize_table_row",
    "normalize_table_rows",
    "build_report_table",
    "coerce_report_table",
    "table_from_block",
    "section_tables",
    "report_tables",
    "formatted_table_rows",
    "table_column_widths",
    "align_table_cell",
)

_register_public_names(_SECTION_18_PUBLIC_NAMES)

# =============================================================================
# End of Section 18
# =============================================================================


# =============================================================================
# Section 19 — Text rendering
# =============================================================================

# 19.1. Text primitives
# -----------------------------------------------------------------------------

def render_text_heading(
    text: Any,
    *,
    level: int = 1,
    width: int = DEFAULT_TEXT_WIDTH,
) -> str:
    """Render a plain-text heading."""

    title = single_line_text(text, "")
    if not title:
        return ""
    index = min(
        max(0, int(level) - 1),
        len(TEXT_HEADING_CHARACTERS) - 1,
    )
    underline = TEXT_HEADING_CHARACTERS[index] * min(
        len(title),
        max(1, int(width)),
    )
    return f"{title}\n{underline}"


def wrap_text(
    text: Any,
    *,
    width: int = DEFAULT_TEXT_WIDTH,
    initial_indent: str = "",
    subsequent_indent: str = "",
) -> str:
    """Wrap plain text while preserving paragraphs."""

    import textwrap

    value = safe_string(text, "")
    if not value:
        return ""
    paragraphs = value.split("\n\n")
    return "\n\n".join(
        textwrap.fill(
            paragraph.replace("\n", " "),
            width=max(MIN_TEXT_WIDTH, int(width)),
            initial_indent=initial_indent,
            subsequent_indent=subsequent_indent,
            break_long_words=False,
            break_on_hyphens=False,
        )
        if paragraph.strip()
        else ""
        for paragraph in paragraphs
    )


def render_text_key_values(
    values: Mapping[str, Any],
    *,
    formatting: FormattingConfig = DEFAULT_FORMATTING_CONFIG,
    width: int = DEFAULT_TEXT_WIDTH,
) -> str:
    """Render aligned key-value text."""

    if not values:
        return ""
    labels = [single_line_text(key, "") for key in values]
    label_width = min(
        max(len(label) for label in labels),
        max(10, width // 3),
    )
    lines: List[str] = []
    for label, value in zip(labels, values.values()):
        rendered = format_value(value, config=formatting)
        prefix = f"{label.rjust(label_width)}{TEXT_KEY_VALUE_SEPARATOR}"
        if "\n" in rendered:
            rendered = rendered.replace(
                "\n",
                "\n" + " " * len(prefix),
            )
        lines.append(prefix + rendered)
    return "\n".join(lines)


def render_text_list(
    values: Any,
    *,
    ordered: bool = False,
    width: int = DEFAULT_TEXT_WIDTH,
    indent: int = DEFAULT_INDENT,
    formatting: FormattingConfig = DEFAULT_FORMATTING_CONFIG,
) -> str:
    """Render a plain-text list."""

    lines: List[str] = []
    for index, item in enumerate(
        iter_object_collection(values),
        start=1,
    ):
        marker = (
            f"{index}{TEXT_ORDERED_LIST_SUFFIX}"
            if ordered
            else TEXT_BULLET
        )
        prefix = " " * max(0, indent) + marker + " "
        lines.append(
            wrap_text(
                format_value(item, config=formatting),
                width=width,
                initial_indent=prefix,
                subsequent_indent=" " * len(prefix),
            )
        )
    return "\n".join(lines)


# 19.2. Text table rendering
# -----------------------------------------------------------------------------

def render_text_table(
    table: Union[ReportTable, Any],
    *,
    config: ReportConfig = DEFAULT_REPORT_CONFIG,
    name: str = "table",
    title: str = "",
) -> str:
    """Render an internal table as plain text."""

    table = coerce_report_table(
        table,
        name=name,
        title=title,
        config=config.tables,
    )
    columns = table.visible_columns
    if not columns:
        return ""

    rows = formatted_table_rows(
        table,
        formatting=config.formatting,
        max_cell_length=config.tables.max_cell_length,
    )
    widths = table_column_widths(
        table,
        formatting=config.formatting,
        max_cell_length=config.tables.max_cell_length,
    )

    header = TEXT_COLUMN_SEPARATOR.join(
        align_table_cell(column.label, widths[index], column.align)
        for index, column in enumerate(columns)
    )
    separator = TEXT_COLUMN_SEPARATOR.join(
        TEXT_TABLE_BORDER * width
        for width in widths
    )
    lines = [header, separator]

    for row in rows:
        lines.append(
            TEXT_COLUMN_SEPARATOR.join(
                align_table_cell(
                    row[index],
                    widths[index],
                    columns[index].align,
                )
                for index in range(len(columns))
            )
        )

    if not rows:
        lines.append("(no rows)")
    if table.truncated:
        lines.append(
            f"{DEFAULT_TRUNCATION_MARKER} "
            f"{len(table.rows)} of {table.total_rows} rows shown"
        )
    if table.caption:
        lines.append(wrap_text(table.caption, width=config.rendering.width))
    return "\n".join(lines)


# 19.3. Text block and section rendering
# -----------------------------------------------------------------------------

def render_text_notice(
    block: ReportBlock,
    *,
    config: ReportConfig = DEFAULT_REPORT_CONFIG,
) -> str:
    """Render a diagnostic notice."""

    severity = block.metadata.get("severity", SEVERITY_INFO)
    prefix = {
        SEVERITY_INFO: "Info:",
        SEVERITY_WARNING: TEXT_WARNING_PREFIX,
        SEVERITY_ERROR: TEXT_ERROR_PREFIX,
        SEVERITY_CRITICAL: "Critical:",
    }.get(str(severity), "Info:")
    return wrap_text(
        f"{prefix} {safe_string(block.content, '')}",
        width=config.rendering.width,
    )


def render_text_block(
    block: ReportBlock,
    *,
    config: ReportConfig = DEFAULT_REPORT_CONFIG,
) -> str:
    """Render one report block as plain text."""

    if not block.visible:
        return ""

    content = ""
    if block.kind is ReportBlockKind.PARAGRAPH:
        content = wrap_text(
            block.content,
            width=config.rendering.width,
        )
    elif block.kind is ReportBlockKind.KEY_VALUE:
        values = (
            block.content
            if isinstance(block.content, Mapping)
            else {"Value": block.content}
        )
        content = render_text_key_values(
            values,
            formatting=config.formatting,
            width=config.rendering.width,
        )
    elif block.kind is ReportBlockKind.TABLE:
        content = render_text_table(
            table_from_block(
                block,
                config=config.tables,
            ),
            config=config,
        )
    elif block.kind is ReportBlockKind.LIST:
        content = render_text_list(
            block.content,
            ordered=bool(block.metadata.get("ordered")),
            width=config.rendering.width,
            indent=config.rendering.indent,
            formatting=config.formatting,
        )
    elif block.kind is ReportBlockKind.CODE:
        content = safe_string(block.content, "", strip=False)
    elif block.kind is ReportBlockKind.NOTICE:
        content = render_text_notice(block, config=config)
    elif block.kind is ReportBlockKind.SEPARATOR:
        content = TEXT_TABLE_BORDER * config.rendering.width

    if not content:
        return ""
    if block.title:
        return (
            render_text_heading(
                block.title,
                level=max(3, block.level or 3),
                width=config.rendering.width,
            )
            + "\n"
            + content
        )
    return content


def render_text_section(
    section: ReportSection,
    *,
    config: ReportConfig = DEFAULT_REPORT_CONFIG,
) -> str:
    """Render one report section as plain text."""

    if not section.enabled:
        return ""
    rendered_blocks = [
        render_text_block(block, config=config)
        for block in section.visible_blocks
    ]
    rendered_blocks = [block for block in rendered_blocks if block]
    if not rendered_blocks and not config.rendering.include_empty_sections:
        return ""

    parts = [
        render_text_heading(
            section.title,
            level=2,
            width=config.rendering.width,
        )
    ]
    if section.description and (
        config.rendering.detail
        in {ReportDetail.DETAILED, ReportDetail.FULL}
    ):
        parts.append(
            wrap_text(
                section.description,
                width=config.rendering.width,
            )
        )
    parts.extend(rendered_blocks)
    return (
        config.rendering.newline
        * TEXT_SECTION_SPACING
    ).join(parts)


# 19.4. Complete text report
# -----------------------------------------------------------------------------

def render_report_text(
    report: Union[ReportDocument, Any],
    *,
    config: Optional[ReportConfig] = None,
) -> str:
    """Render a complete report as plain text."""

    if isinstance(report, ReportDocument):
        document = report
        active_config = config or report.config
    else:
        active_config = config or DEFAULT_REPORT_CONFIG
        document = build_report_document(
            report,
            config=active_config,
        )

    parts: List[str] = []
    if active_config.rendering.include_title and document.title:
        parts.append(
            render_text_heading(
                document.title,
                level=1,
                width=active_config.rendering.width,
            )
        )
    if (
        active_config.rendering.include_subtitle
        and document.subtitle
    ):
        parts.append(
            wrap_text(
                document.subtitle,
                width=active_config.rendering.width,
            )
        )
    if document.description:
        parts.append(
            wrap_text(
                document.description,
                width=active_config.rendering.width,
            )
        )
    if active_config.rendering.include_generated_at:
        parts.append(
            f"Generated at{TEXT_KEY_VALUE_SEPARATOR}"
            f"{document.generated_at}"
        )

    parts.extend(
        rendered
        for rendered in (
            render_text_section(
                section,
                config=active_config,
            )
            for section in document.visible_sections
        )
        if rendered
    )

    separator = (
        active_config.rendering.newline
        * TEXT_SECTION_SPACING
    )
    return separator.join(parts).rstrip() + active_config.rendering.newline


render_text_report = render_report_text

# 19.5. Public text-rendering interface
# -----------------------------------------------------------------------------

_SECTION_19_PUBLIC_NAMES: Final[Tuple[str, ...]] = (
    "render_text_heading",
    "wrap_text",
    "render_text_key_values",
    "render_text_list",
    "render_text_table",
    "render_text_notice",
    "render_text_block",
    "render_text_section",
    "render_report_text",
    "render_text_report",
)

_register_public_names(_SECTION_19_PUBLIC_NAMES)

# =============================================================================
# End of Section 19
# =============================================================================


# =============================================================================
# Section 20 — Markdown rendering
# =============================================================================

# 20.1. Markdown primitives
# -----------------------------------------------------------------------------

def render_markdown_heading(
    text: Any,
    *,
    level: int = 1,
) -> str:
    """Render a Markdown heading."""

    title = single_line_text(text, "")
    if not title:
        return ""
    level = min(
        max(1, int(level)),
        len(MARKDOWN_HEADING_PREFIXES),
    )
    return f"{MARKDOWN_HEADING_PREFIXES[level - 1]} {title}"


def markdown_anchor(value: Any) -> str:
    """Return a simple Markdown heading anchor."""

    text = normalize_field_name(value).replace("_", "-")
    return text.strip("-")


def render_markdown_key_values(
    values: Mapping[str, Any],
    *,
    formatting: FormattingConfig = DEFAULT_FORMATTING_CONFIG,
) -> str:
    """Render key-value pairs as a Markdown list."""

    return "\n".join(
        f"{MARKDOWN_BULLET} **{escape_markdown(key)}:** "
        f"{escape_markdown(format_value(value, config=formatting))}"
        for key, value in values.items()
    )


def render_markdown_list(
    values: Any,
    *,
    ordered: bool = False,
    formatting: FormattingConfig = DEFAULT_FORMATTING_CONFIG,
) -> str:
    """Render a Markdown list."""

    lines: List[str] = []
    for index, item in enumerate(
        iter_object_collection(values),
        start=1,
    ):
        marker = f"{index}." if ordered else MARKDOWN_BULLET
        text = escape_markdown(
            format_value(item, config=formatting)
        ).replace("\n", "  \n  ")
        lines.append(f"{marker} {text}")
    return "\n".join(lines)


def render_markdown_code(
    value: Any,
    *,
    language: str = "",
) -> str:
    """Render a fenced Markdown code block."""

    text = safe_string(value, "", strip=False)
    fence = MARKDOWN_CODE_FENCE
    while fence in text:
        fence += "`"
    return f"{fence}{language}\n{text}\n{fence}"


# 20.2. Markdown table rendering
# -----------------------------------------------------------------------------

def _markdown_alignment_marker(
    column: ReportColumn,
) -> str:
    """Return a Markdown alignment marker."""

    if column.align == "right":
        return f"{MARKDOWN_TABLE_SEPARATOR}:"
    if column.align == "center":
        return f":{MARKDOWN_TABLE_SEPARATOR}:"
    return f":{MARKDOWN_TABLE_SEPARATOR}"


def render_markdown_table(
    table: Union[ReportTable, Any],
    *,
    config: ReportConfig = DEFAULT_REPORT_CONFIG,
    name: str = "table",
    title: str = "",
) -> str:
    """Render an internal table as Markdown."""

    table = coerce_report_table(
        table,
        name=name,
        title=title,
        config=config.tables,
    )
    columns = table.visible_columns
    if not columns:
        return ""

    header = "| " + " | ".join(
        escape_markdown_cell(column.label)
        for column in columns
    ) + " |"
    separator = "| " + " | ".join(
        _markdown_alignment_marker(column)
        for column in columns
    ) + " |"

    lines = [header, separator]
    for row in table.rows:
        lines.append(
            "| "
            + " | ".join(
                escape_markdown_cell(
                    column.format(
                        row.get(column.key),
                        config=config.formatting,
                    )
                )
                for column in columns
            )
            + " |"
        )

    if not table.rows:
        lines.append(
            "| "
            + " | ".join(
                "—" if index == 0 else ""
                for index in range(len(columns))
            )
            + " |"
        )
    if table.truncated:
        lines.append("")
        lines.append(
            f"*{DEFAULT_TRUNCATION_MARKER} "
            f"{len(table.rows)} of {table.total_rows} rows shown.*"
        )
    if table.caption:
        lines.append("")
        lines.append(f"*{escape_markdown(table.caption)}*")
    return "\n".join(lines)


# 20.3. Markdown block and section rendering
# -----------------------------------------------------------------------------

def render_markdown_notice(
    block: ReportBlock,
) -> str:
    """Render a Markdown blockquote notice."""

    severity = str(
        block.metadata.get("severity", SEVERITY_INFO)
    )
    label = {
        SEVERITY_INFO: "Info",
        SEVERITY_WARNING: "Warning",
        SEVERITY_ERROR: "Error",
        SEVERITY_CRITICAL: "Critical",
    }.get(severity, "Info")
    lines = safe_string(block.content, "").splitlines() or [""]
    return "\n".join(
        f"> **{label}:** {escape_markdown(line)}"
        if index == 0
        else f"> {escape_markdown(line)}"
        for index, line in enumerate(lines)
    )


def render_markdown_block(
    block: ReportBlock,
    *,
    config: ReportConfig = DEFAULT_REPORT_CONFIG,
    heading_level: int = 3,
) -> str:
    """Render one report block as Markdown."""

    if not block.visible:
        return ""

    if block.kind is ReportBlockKind.PARAGRAPH:
        content = escape_markdown(
            safe_string(block.content, "")
        ).replace("\n\n", "\n\n")
    elif block.kind is ReportBlockKind.KEY_VALUE:
        values = (
            block.content
            if isinstance(block.content, Mapping)
            else {"Value": block.content}
        )
        content = render_markdown_key_values(
            values,
            formatting=config.formatting,
        )
    elif block.kind is ReportBlockKind.TABLE:
        content = render_markdown_table(
            table_from_block(
                block,
                config=config.tables,
            ),
            config=config,
        )
    elif block.kind is ReportBlockKind.LIST:
        content = render_markdown_list(
            block.content,
            ordered=bool(block.metadata.get("ordered")),
            formatting=config.formatting,
        )
    elif block.kind is ReportBlockKind.CODE:
        content = render_markdown_code(
            block.content,
            language=str(block.metadata.get("language", "")),
        )
    elif block.kind is ReportBlockKind.NOTICE:
        content = render_markdown_notice(block)
    elif block.kind is ReportBlockKind.SEPARATOR:
        content = "---"
    else:
        content = ""

    if not content:
        return ""
    if block.title:
        return (
            render_markdown_heading(
                block.title,
                level=max(1, heading_level),
            )
            + "\n\n"
            + content
        )
    return content


def render_markdown_section(
    section: ReportSection,
    *,
    config: ReportConfig = DEFAULT_REPORT_CONFIG,
) -> str:
    """Render one report section as Markdown."""

    if not section.enabled:
        return ""
    rendered_blocks = [
        render_markdown_block(
            block,
            config=config,
            heading_level=3,
        )
        for block in section.visible_blocks
    ]
    rendered_blocks = [block for block in rendered_blocks if block]
    if not rendered_blocks and not config.rendering.include_empty_sections:
        return ""

    parts = [
        render_markdown_heading(
            section.title,
            level=2,
        )
    ]
    if section.description and (
        config.rendering.detail
        in {ReportDetail.DETAILED, ReportDetail.FULL}
    ):
        parts.append(escape_markdown(section.description))
    parts.extend(rendered_blocks)
    return "\n\n".join(parts)


# 20.4. Markdown table of contents
# -----------------------------------------------------------------------------

def render_markdown_toc(
    report: ReportDocument,
) -> str:
    """Render a Markdown table of contents."""

    lines = [
        render_markdown_heading("Contents", level=2)
    ]
    for section in report.visible_sections:
        lines.append(
            f"{MARKDOWN_BULLET} "
            f"[{escape_markdown(section.title)}]"
            f"(#{markdown_anchor(section.title)})"
        )
    return "\n".join(lines)


# 20.5. Complete Markdown report
# -----------------------------------------------------------------------------

def render_report_markdown(
    report: Union[ReportDocument, Any],
    *,
    config: Optional[ReportConfig] = None,
) -> str:
    """Render a complete report as Markdown."""

    if isinstance(report, ReportDocument):
        document = report
        active_config = config or report.config
    else:
        active_config = config or DEFAULT_REPORT_CONFIG
        document = build_report_document(
            report,
            config=active_config,
        )

    parts: List[str] = []
    if active_config.rendering.include_title and document.title:
        parts.append(
            render_markdown_heading(
                document.title,
                level=1,
            )
        )
    if (
        active_config.rendering.include_subtitle
        and document.subtitle
    ):
        parts.append(f"*{escape_markdown(document.subtitle)}*")
    if document.description:
        parts.append(escape_markdown(document.description))
    if active_config.rendering.include_generated_at:
        parts.append(
            f"**Generated at:** "
            f"{escape_markdown(document.generated_at)}"
        )
    if active_config.rendering.include_table_of_contents:
        parts.append(render_markdown_toc(document))

    parts.extend(
        rendered
        for rendered in (
            render_markdown_section(
                section,
                config=active_config,
            )
            for section in document.visible_sections
        )
        if rendered
    )

    return "\n\n".join(parts).rstrip() + active_config.rendering.newline


render_markdown_report = render_report_markdown

# 20.6. Public Markdown-rendering interface
# -----------------------------------------------------------------------------

_SECTION_20_PUBLIC_NAMES: Final[Tuple[str, ...]] = (
    "render_markdown_heading",
    "markdown_anchor",
    "render_markdown_key_values",
    "render_markdown_list",
    "render_markdown_code",
    "render_markdown_table",
    "render_markdown_notice",
    "render_markdown_block",
    "render_markdown_section",
    "render_markdown_toc",
    "render_report_markdown",
    "render_markdown_report",
)

_register_public_names(_SECTION_20_PUBLIC_NAMES)

# =============================================================================
# End of Section 20
# =============================================================================

# =============================================================================
# Section 21 — HTML rendering
# =============================================================================

# 21.1. HTML primitives
# -----------------------------------------------------------------------------

def html_attributes(
    attributes: Optional[Mapping[str, Any]] = None,
) -> str:
    """Render safe HTML attributes."""

    if not attributes:
        return ""

    rendered: List[str] = []
    for key, value in attributes.items():
        name = normalize_field_name(key).replace("_", "-")
        if not name or value is None or value is False:
            continue
        if value is True:
            rendered.append(name)
            continue
        rendered.append(
            f'{name}="{escape_html(value, quote=True)}"'
        )
    return (" " + " ".join(rendered)) if rendered else ""


def html_tag(
    name: str,
    content: Any = "",
    *,
    attributes: Optional[Mapping[str, Any]] = None,
    escape_content: bool = False,
    self_closing: bool = False,
) -> str:
    """Build one HTML element."""

    tag_name = normalize_field_name(name).replace("_", "-")
    if not tag_name:
        raise HTMLRenderError("HTML tag name must not be empty.")

    attrs = html_attributes(attributes)
    if self_closing:
        return f"<{tag_name}{attrs}>"

    body = (
        escape_html(content)
        if escape_content
        else safe_string(content, "", strip=False)
    )
    return f"<{tag_name}{attrs}>{body}</{tag_name}>"


def render_html_heading(
    text: Any,
    *,
    level: int = 1,
    element_id: Optional[str] = None,
    css_class: Optional[str] = None,
) -> str:
    """Render an HTML heading."""

    level = min(max(1, int(level)), 6)
    title = single_line_text(text, "")
    if not title:
        return ""

    attributes: Dict[str, Any] = {}
    if element_id:
        attributes["id"] = element_id
    if css_class:
        attributes["class"] = css_class

    return html_tag(
        f"h{level}",
        title,
        attributes=attributes,
        escape_content=True,
    )


def render_html_paragraph(
    value: Any,
    *,
    css_class: Optional[str] = None,
) -> str:
    """Render escaped paragraphs with line breaks."""

    text = safe_string(value, "")
    if not text:
        return ""

    attributes = {"class": css_class} if css_class else None
    return "\n".join(
        html_tag(
            "p",
            escape_html(paragraph).replace("\n", "<br>"),
            attributes=attributes,
        )
        for paragraph in text.split("\n\n")
        if paragraph.strip()
    )


def render_html_key_values(
    values: Mapping[str, Any],
    *,
    formatting: FormattingConfig = DEFAULT_FORMATTING_CONFIG,
    css_class: str = "dockanalyzer-key-values",
) -> str:
    """Render key-value pairs as a definition list."""

    if not values:
        return ""

    items: List[str] = []
    for key, value in values.items():
        items.append(
            html_tag("dt", field_label(key), escape_content=True)
        )
        items.append(
            html_tag(
                "dd",
                format_value(value, config=formatting),
                escape_content=True,
            )
        )
    return html_tag(
        "dl",
        "\n".join(items),
        attributes={"class": css_class},
    )


def render_html_list(
    values: Any,
    *,
    ordered: bool = False,
    formatting: FormattingConfig = DEFAULT_FORMATTING_CONFIG,
) -> str:
    """Render an ordered or unordered HTML list."""

    items = [
        html_tag(
            "li",
            format_value(item, config=formatting),
            escape_content=True,
        )
        for item in iter_object_collection(values)
    ]
    if not items:
        return ""
    return html_tag(
        "ol" if ordered else "ul",
        "\n".join(items),
    )


def render_html_code(
    value: Any,
    *,
    language: str = "",
) -> str:
    """Render a safe HTML code block."""

    code_class = (
        f"language-{normalize_field_name(language)}"
        if language
        else None
    )
    code = html_tag(
        "code",
        safe_string(value, "", strip=False),
        attributes={"class": code_class} if code_class else None,
        escape_content=True,
    )
    return html_tag("pre", code)


# 21.2. HTML table rendering
# -----------------------------------------------------------------------------

def render_html_table(
    table: Union[ReportTable, Any],
    *,
    config: ReportConfig = DEFAULT_REPORT_CONFIG,
    name: str = "table",
    title: str = "",
) -> str:
    """Render an internal table as semantic HTML."""

    table = coerce_report_table(
        table,
        name=name,
        title=title,
        config=config.tables,
    )
    columns = table.visible_columns
    if not columns:
        return ""

    header_cells = [
        html_tag(
            "th",
            column.label,
            attributes={
                "scope": "col",
                "class": f"align-{column.align}",
            },
            escape_content=True,
        )
        for column in columns
    ]
    head = html_tag(
        "thead",
        html_tag("tr", "\n".join(header_cells)),
    )

    body_rows: List[str] = []
    for row in table.rows:
        cells = [
            html_tag(
                "td",
                column.format(
                    row.get(column.key),
                    config=config.formatting,
                ),
                attributes={
                    "class": f"align-{column.align}",
                    "data-column": column.key,
                },
                escape_content=True,
            )
            for column in columns
        ]
        body_rows.append(html_tag("tr", "\n".join(cells)))

    if not body_rows:
        body_rows.append(
            html_tag(
                "tr",
                html_tag(
                    "td",
                    "No rows",
                    attributes={"colspan": len(columns)},
                    escape_content=True,
                ),
            )
        )

    body = html_tag("tbody", "\n".join(body_rows))
    table_html = html_tag(
        "table",
        head + "\n" + body,
        attributes={
            "class": HTML_TABLE_CLASS,
            "data-table": table.name,
        },
    )

    parts: List[str] = []
    if table.title:
        parts.append(
            render_html_heading(
                table.title,
                level=4,
                css_class="dockanalyzer-table-title",
            )
        )
    parts.append(table_html)

    if table.truncated:
        parts.append(
            html_tag(
                "p",
                (
                    f"{len(table.rows)} of "
                    f"{table.total_rows} rows shown."
                ),
                attributes={"class": "dockanalyzer-table-note"},
                escape_content=True,
            )
        )
    if table.caption:
        parts.append(
            html_tag(
                "p",
                table.caption,
                attributes={"class": "dockanalyzer-table-caption"},
                escape_content=True,
            )
        )
    return "\n".join(parts)


# 21.3. HTML block and section rendering
# -----------------------------------------------------------------------------

def render_html_notice(
    block: ReportBlock,
) -> str:
    """Render a diagnostic HTML notice."""

    severity = str(
        block.metadata.get("severity", SEVERITY_INFO)
    )
    css_class = {
        SEVERITY_INFO: HTML_NOTICE_CLASS,
        SEVERITY_WARNING: (
            f"{HTML_NOTICE_CLASS} {HTML_WARNING_CLASS}"
        ),
        SEVERITY_ERROR: (
            f"{HTML_NOTICE_CLASS} {HTML_ERROR_CLASS}"
        ),
        SEVERITY_CRITICAL: (
            f"{HTML_NOTICE_CLASS} {HTML_ERROR_CLASS}"
        ),
    }.get(severity, HTML_NOTICE_CLASS)

    label = {
        SEVERITY_INFO: "Info",
        SEVERITY_WARNING: "Warning",
        SEVERITY_ERROR: "Error",
        SEVERITY_CRITICAL: "Critical",
    }.get(severity, "Info")

    content = (
        html_tag("strong", f"{label}: ", escape_content=True)
        + escape_html(block.content)
    )
    return html_tag(
        "div",
        content,
        attributes={
            "class": css_class,
            "role": "alert" if severity != SEVERITY_INFO else "note",
        },
    )


def render_html_block(
    block: ReportBlock,
    *,
    config: ReportConfig = DEFAULT_REPORT_CONFIG,
    heading_level: int = 3,
) -> str:
    """Render one structured block as HTML."""

    if not block.visible:
        return ""

    if block.kind is ReportBlockKind.PARAGRAPH:
        content = render_html_paragraph(block.content)
    elif block.kind is ReportBlockKind.KEY_VALUE:
        values = (
            block.content
            if isinstance(block.content, Mapping)
            else {"Value": block.content}
        )
        content = render_html_key_values(
            values,
            formatting=config.formatting,
        )
    elif block.kind is ReportBlockKind.TABLE:
        content = render_html_table(
            table_from_block(
                block,
                config=config.tables,
            ),
            config=config,
        )
    elif block.kind is ReportBlockKind.LIST:
        content = render_html_list(
            block.content,
            ordered=bool(block.metadata.get("ordered")),
            formatting=config.formatting,
        )
    elif block.kind is ReportBlockKind.CODE:
        content = render_html_code(
            block.content,
            language=str(block.metadata.get("language", "")),
        )
    elif block.kind is ReportBlockKind.NOTICE:
        content = render_html_notice(block)
    elif block.kind is ReportBlockKind.SEPARATOR:
        content = "<hr>"
    else:
        content = ""

    if not content:
        return ""

    if block.title and block.kind is not ReportBlockKind.TABLE:
        return (
            render_html_heading(
                block.title,
                level=max(1, heading_level),
            )
            + "\n"
            + content
        )
    return content


def render_html_section(
    section: ReportSection,
    *,
    config: ReportConfig = DEFAULT_REPORT_CONFIG,
) -> str:
    """Render one report section as HTML."""

    if not section.enabled:
        return ""

    rendered_blocks = [
        render_html_block(
            block,
            config=config,
            heading_level=3,
        )
        for block in section.visible_blocks
    ]
    rendered_blocks = [block for block in rendered_blocks if block]
    if not rendered_blocks and not config.rendering.include_empty_sections:
        return ""

    section_id = markdown_anchor(section.title)
    parts = [
        render_html_heading(
            section.title,
            level=2,
            element_id=section_id,
        )
    ]
    if section.description and (
        config.rendering.detail
        in {ReportDetail.DETAILED, ReportDetail.FULL}
    ):
        parts.append(
            render_html_paragraph(
                section.description,
                css_class="dockanalyzer-section-description",
            )
        )
    parts.extend(rendered_blocks)

    return html_tag(
        "section",
        "\n".join(parts),
        attributes={
            "class": HTML_SECTION_CLASS,
            "data-section": section.id.value,
        },
    )


# 21.4. HTML navigation and document shell
# -----------------------------------------------------------------------------

def render_html_toc(
    report: ReportDocument,
) -> str:
    """Render an HTML table of contents."""

    items = [
        html_tag(
            "li",
            html_tag(
                "a",
                section.title,
                attributes={
                    "href": f"#{markdown_anchor(section.title)}"
                },
                escape_content=True,
            ),
        )
        for section in report.visible_sections
    ]
    if not items:
        return ""
    return html_tag(
        "nav",
        render_html_heading("Contents", level=2)
        + "\n"
        + html_tag("ul", "\n".join(items)),
        attributes={
            "class": "dockanalyzer-toc",
            "aria-label": "Table of contents",
        },
    )


def render_html_head(
    report: ReportDocument,
    *,
    config: ReportConfig = DEFAULT_REPORT_CONFIG,
) -> str:
    """Render the HTML document head."""

    title = report.title or DEFAULT_REPORT_TITLE
    parts = [
        html_tag(
            "meta",
            attributes={"charset": HTML_CHARSET},
            self_closing=True,
        ),
        html_tag(
            "meta",
            attributes={
                "name": "viewport",
                "content": HTML_VIEWPORT,
            },
            self_closing=True,
        ),
        html_tag("title", title, escape_content=True),
    ]
    if config.rendering.html_css:
        parts.append(
            html_tag(
                "style",
                config.rendering.html_css,
                attributes={"type": "text/css"},
            )
        )
    return html_tag("head", "\n".join(parts))


def render_html_body(
    report: ReportDocument,
    *,
    config: ReportConfig = DEFAULT_REPORT_CONFIG,
) -> str:
    """Render the HTML document body."""

    parts: List[str] = []
    if config.rendering.include_title and report.title:
        parts.append(render_html_heading(report.title, level=1))
    if config.rendering.include_subtitle and report.subtitle:
        parts.append(
            html_tag(
                "p",
                report.subtitle,
                attributes={"class": "dockanalyzer-subtitle"},
                escape_content=True,
            )
        )
    if report.description:
        parts.append(
            render_html_paragraph(
                report.description,
                css_class="dockanalyzer-description",
            )
        )
    if config.rendering.include_generated_at:
        parts.append(
            html_tag(
                "p",
                f"Generated at: {report.generated_at}",
                attributes={"class": "dockanalyzer-generated-at"},
                escape_content=True,
            )
        )
    if config.rendering.include_table_of_contents:
        toc = render_html_toc(report)
        if toc:
            parts.append(toc)

    parts.extend(
        rendered
        for rendered in (
            render_html_section(section, config=config)
            for section in report.visible_sections
        )
        if rendered
    )

    return html_tag(
        "body",
        "\n".join(parts),
        attributes={"class": "dockanalyzer-report"},
    )


# 21.5. Complete HTML report
# -----------------------------------------------------------------------------

def render_report_html(
    report: Union[ReportDocument, Any],
    *,
    config: Optional[ReportConfig] = None,
) -> str:
    """Render a complete report as HTML."""

    if isinstance(report, ReportDocument):
        document = report
        active_config = config or report.config
    else:
        active_config = config or DEFAULT_REPORT_CONFIG
        document = build_report_document(
            report,
            config=active_config,
        )

    body = render_html_body(
        document,
        config=active_config,
    )
    if not active_config.rendering.html_full_document:
        return body + active_config.rendering.newline

    html_document = (
        HTML_DOCUMENT_TYPE
        + active_config.rendering.newline
        + html_tag(
            "html",
            render_html_head(
                document,
                config=active_config,
            )
            + active_config.rendering.newline
            + body,
            attributes={
                "lang": (
                    active_config.rendering.language
                    or HTML_DEFAULT_LANG
                )
            },
        )
    )
    return html_document.rstrip() + active_config.rendering.newline


render_html_report = render_report_html

# 21.6. Public HTML-rendering interface
# -----------------------------------------------------------------------------

_SECTION_21_PUBLIC_NAMES: Final[Tuple[str, ...]] = (
    "html_attributes",
    "html_tag",
    "render_html_heading",
    "render_html_paragraph",
    "render_html_key_values",
    "render_html_list",
    "render_html_code",
    "render_html_table",
    "render_html_notice",
    "render_html_block",
    "render_html_section",
    "render_html_toc",
    "render_html_head",
    "render_html_body",
    "render_report_html",
    "render_html_report",
)

_register_public_names(_SECTION_21_PUBLIC_NAMES)

# =============================================================================
# End of Section 21
# =============================================================================


# =============================================================================
# Section 22 — JSON representation
# =============================================================================

# 22.1. JSON conversion state
# -----------------------------------------------------------------------------

@dataclass
class JSONConversionState:
    """Mutable state for safe recursive JSON conversion."""

    max_depth: int = 20
    max_items: int = DEFAULT_MAX_ITEMS
    seen: Set[int] = field(default_factory=set)
    warnings: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.max_depth = max(1, to_safe_int(self.max_depth, 20))
        self.max_items = max(1, to_safe_int(
            self.max_items,
            DEFAULT_MAX_ITEMS,
        ))


def is_json_primitive(value: Any) -> bool:
    """Return whether a value is a JSON primitive."""

    return value is None or isinstance(
        value,
        (str, int, float, bool),
    )


def _json_number(value: Any) -> Optional[Union[int, float]]:
    """Return a finite JSON number or None."""

    value = unwrap_scalar(value)
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, Real):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    return None


def _json_cycle_record(value: Any) -> Dict[str, Any]:
    """Return a cycle marker."""

    return {
        JSON_TYPE_KEY: type(value).__name__,
        JSON_VALUE_KEY: "<recursive-reference>",
    }


def _json_truncation_record(
    total: Optional[int] = None,
) -> Dict[str, Any]:
    """Return a truncation marker."""

    record: Dict[str, Any] = {
        JSON_TYPE_KEY: "truncated",
        JSON_VALUE_KEY: DEFAULT_TRUNCATION_MARKER,
    }
    if total is not None:
        record["total_items"] = total
    return record


# 22.2. Recursive JSON-safe conversion
# -----------------------------------------------------------------------------

def to_json_safe(
    value: Any,
    *,
    state: Optional[JSONConversionState] = None,
    depth: int = 0,
    include_private: bool = False,
) -> JSONValue:
    """Convert heterogeneous report values into JSON-safe values."""

    if state is None:
        state = JSONConversionState()

    value = unwrap_scalar(value)

    if value is MISSING:
        return None
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, Real):
        return _json_number(value)
    if isinstance(value, Enum):
        return to_json_safe(
            value.value,
            state=state,
            depth=depth + 1,
            include_private=include_private,
        )
    if isinstance(value, (datetime, date)):
        return format_datetime(value) if isinstance(
            value,
            datetime,
        ) else value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return value.decode(DEFAULT_ENCODING, errors="replace")
    if isinstance(value, BaseException):
        if isinstance(value, ReportError):
            return to_json_safe(
                value.to_dict(),
                state=state,
                depth=depth + 1,
                include_private=include_private,
            )
        return {
            "type": _exception_name(value),
            "message": _exception_message(value),
        }

    if depth >= state.max_depth:
        state.warnings.append(
            f"JSON conversion depth limited at {type(value).__name__}."
        )
        return {
            JSON_TYPE_KEY: type(value).__name__,
            JSON_VALUE_KEY: "<maximum-depth>",
        }

    track_identity = isinstance(
        value,
        (Mapping, list, tuple, set, frozenset),
    ) or is_dataclass_instance(value) or hasattr(value, "__dict__")

    identity = id(value)
    if track_identity:
        if identity in state.seen:
            state.warnings.append(
                f"Recursive reference detected for {type(value).__name__}."
            )
            return _json_cycle_record(value)
        state.seen.add(identity)

    try:
        if isinstance(value, Mapping):
            items = list(value.items())
            result: Dict[str, JSONValue] = {}
            for index, (key, item) in enumerate(items):
                if index >= state.max_items:
                    result[DEFAULT_TRUNCATION_MARKER] = (
                        _json_truncation_record(len(items))
                    )
                    break
                result[safe_string(key, DEFAULT_UNKNOWN_TEXT)] = (
                    to_json_safe(
                        item,
                        state=state,
                        depth=depth + 1,
                        include_private=include_private,
                    )
                )
            return result

        if isinstance(value, (list, tuple, set, frozenset)):
            items = list(value)
            result_list: List[JSONValue] = []
            for index, item in enumerate(items):
                if index >= state.max_items:
                    result_list.append(
                        _json_truncation_record(len(items))
                    )
                    break
                result_list.append(
                    to_json_safe(
                        item,
                        state=state,
                        depth=depth + 1,
                        include_private=include_private,
                    )
                )
            return result_list

        if is_dataclass_instance(value):
            record = {
                field_info.name: getattr(value, field_info.name)
                for field_info in fields(value)
                if include_private
                or not field_info.name.startswith("_")
            }
            return to_json_safe(
                record,
                state=state,
                depth=depth + 1,
                include_private=include_private,
            )

        if hasattr(value, "to_dict") and callable(value.to_dict):
            try:
                record = value.to_dict()
            except TypeError:
                record = value.to_dict
            except Exception:
                record = MISSING
            if record is not MISSING and not callable(record):
                return to_json_safe(
                    record,
                    state=state,
                    depth=depth + 1,
                    include_private=include_private,
                )

        record = object_to_shallow_dict(
            value,
            include_private=include_private,
            include_properties=False,
            include_callables=False,
        )
        if record:
            converted = to_json_safe(
                record,
                state=state,
                depth=depth + 1,
                include_private=include_private,
            )
            if isinstance(converted, dict):
                converted.setdefault(
                    JSON_TYPE_KEY,
                    type(value).__name__,
                )
            return converted

        return safe_string(value, repr(value))
    finally:
        if track_identity:
            state.seen.discard(identity)


# 22.3. Report JSON representation
# -----------------------------------------------------------------------------

def report_to_json_data(
    report: Union[ReportDocument, Any],
    *,
    config: Optional[ReportConfig] = None,
    include_config: bool = False,
    include_private: bool = False,
    max_depth: int = 30,
    max_items: int = DEFAULT_MAX_ROWS,
) -> Dict[str, JSONValue]:
    """Return a complete JSON-safe report mapping."""

    if isinstance(report, ReportDocument):
        document = report
        active_config = config or report.config
    else:
        active_config = config or DEFAULT_REPORT_CONFIG
        document = build_report_document(
            report,
            config=active_config,
        )

    state = JSONConversionState(
        max_depth=max_depth,
        max_items=max_items,
    )
    data = to_json_safe(
        document.to_dict(),
        state=state,
        include_private=include_private,
    )
    if not isinstance(data, dict):
        raise ReportSerializationError(
            "Report JSON representation must be a mapping."
        )

    if include_config:
        data["config"] = to_json_safe(
            active_config.to_dict(),
            state=state,
            include_private=include_private,
        )
    if state.warnings:
        existing = data.get(KEY_WARNINGS)
        warnings_list = (
            list(existing)
            if isinstance(existing, list)
            else []
        )
        warnings_list.extend(state.warnings)
        data[KEY_WARNINGS] = warnings_list
    return data


def json_default_encoder(value: Any) -> JSONValue:
    """JSON default callback for report-compatible values."""

    converted = to_json_safe(value)
    if converted is value:
        raise TypeError(
            f"Object of type {type(value).__name__} "
            "is not JSON serializable."
        )
    return converted


def render_report_json(
    report: Union[ReportDocument, Any],
    *,
    config: Optional[ReportConfig] = None,
    indent: Optional[int] = None,
    sort_keys: Optional[bool] = None,
    ensure_ascii: Optional[bool] = None,
    include_config: bool = False,
) -> str:
    """Render a complete report as JSON."""

    active_config = (
        config
        or (
            report.config
            if isinstance(report, ReportDocument)
            else DEFAULT_REPORT_CONFIG
        )
    )
    render_config = active_config.rendering
    data = report_to_json_data(
        report,
        config=active_config,
        include_config=include_config,
        max_items=active_config.tables.max_rows,
    )

    resolved_indent = (
        render_config.json_indent
        if indent is None
        else indent
    )
    resolved_sort_keys = (
        render_config.json_sort_keys
        if sort_keys is None
        else bool(sort_keys)
    )
    resolved_ensure_ascii = (
        render_config.json_ensure_ascii
        if ensure_ascii is None
        else bool(ensure_ascii)
    )

    try:
        return (
            json.dumps(
                data,
                indent=resolved_indent,
                sort_keys=resolved_sort_keys,
                ensure_ascii=resolved_ensure_ascii,
                allow_nan=DEFAULT_JSON_ALLOW_NAN,
                default=json_default_encoder,
                separators=(
                    DEFAULT_JSON_PRETTY_SEPARATORS
                    if resolved_indent is not None
                    else DEFAULT_JSON_COMPACT_SEPARATORS
                ),
            )
            + render_config.newline
        )
    except (TypeError, ValueError, OverflowError) as error:
        raise JSONRenderError(
            "Unable to render report JSON.",
            cause=error,
        ) from error


render_json_report = render_report_json

# 22.4. JSON parsing and validation helpers
# -----------------------------------------------------------------------------

def parse_report_json(
    value: Union[str, bytes, bytearray],
) -> Dict[str, Any]:
    """Parse a JSON report mapping."""

    if isinstance(value, (bytes, bytearray)):
        value = bytes(value).decode(
            DEFAULT_ENCODING,
            errors="strict",
        )
    try:
        data = json.loads(value)
    except (TypeError, json.JSONDecodeError) as error:
        raise ReportSerializationError(
            "Invalid report JSON.",
            cause=error,
        ) from error
    if not isinstance(data, dict):
        raise ReportSchemaError(
            "Report JSON root must be an object."
        )
    return data


def report_json_schema_summary(
    value: Mapping[str, Any],
) -> Dict[str, Any]:
    """Return a minimal schema summary for JSON data."""

    missing = [
        key
        for key in REQUIRED_REPORT_KEYS
        if key not in value
    ]
    sections = value.get(KEY_SECTIONS)
    return {
        "valid_root": isinstance(value, Mapping),
        "missing_required_keys": missing,
        "section_count": (
            len(sections)
            if isinstance(sections, list)
            else 0
        ),
        KEY_SCHEMA_NAME: value.get(KEY_SCHEMA_NAME),
        KEY_SCHEMA_VERSION: value.get(KEY_SCHEMA_VERSION),
    }


# 22.5. Public JSON interface
# -----------------------------------------------------------------------------

_SECTION_22_PUBLIC_NAMES: Final[Tuple[str, ...]] = (
    "JSONConversionState",
    "is_json_primitive",
    "to_json_safe",
    "report_to_json_data",
    "json_default_encoder",
    "render_report_json",
    "render_json_report",
    "parse_report_json",
    "report_json_schema_summary",
)

_register_public_names(_SECTION_22_PUBLIC_NAMES)

# =============================================================================
# End of Section 22
# =============================================================================


# =============================================================================
# Section 23 — Safe writing
# =============================================================================

# 23.1. Output records and filename helpers
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class WriteResult:
    """Result of a successful report write."""

    path: str
    format: ReportFormat
    bytes_written: int
    atomic: bool
    overwritten: bool
    backup_path: Optional[str] = None
    checksum: Optional[str] = None
    encoding: str = DEFAULT_ENCODING
    metadata: Mapping[str, Any] = field(
        default_factory=lambda: _EMPTY_METADATA
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "path",
            single_line_text(self.path, ""),
        )
        object.__setattr__(
            self,
            "format",
            _coerce_enum(
                ReportFormat,
                self.format,
                "format",
            ),
        )
        object.__setattr__(
            self,
            "bytes_written",
            max(0, to_safe_int(self.bytes_written, 0)),
        )
        object.__setattr__(self, "atomic", bool(self.atomic))
        object.__setattr__(
            self,
            "overwritten",
            bool(self.overwritten),
        )
        object.__setattr__(
            self,
            "backup_path",
            (
                None
                if self.backup_path is None
                else single_line_text(self.backup_path, "")
            ),
        )
        object.__setattr__(
            self,
            "checksum",
            (
                None
                if self.checksum is None
                else single_line_text(self.checksum, "")
            ),
        )
        object.__setattr__(
            self,
            "encoding",
            single_line_text(self.encoding, DEFAULT_ENCODING),
        )
        object.__setattr__(
            self,
            "metadata",
            _freeze_config_mapping(self.metadata),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain write result."""

        return {
            "path": self.path,
            "format": self.format.value,
            "bytes_written": self.bytes_written,
            "atomic": self.atomic,
            "overwritten": self.overwritten,
            "backup_path": self.backup_path,
            KEY_CHECKSUM: self.checksum,
            "encoding": self.encoding,
            KEY_METADATA: dict(self.metadata),
        }


def sanitize_filename(
    value: Any,
    *,
    default: str = DEFAULT_REPORT_BASENAME,
    max_length: int = 240,
    replacement: str = "_",
) -> str:
    """Return a portable filename component."""

    text = single_line_text(value, default)
    text = INVALID_FILENAME_PATTERN.sub(replacement, text)
    text = WHITESPACE_PATTERN.sub(replacement, text)
    text = MULTIPLE_UNDERSCORES_PATTERN.sub(replacement, text)
    text = TRAILING_DOT_SPACE_PATTERN.sub("", text)
    text = text.strip(" ._")

    if not text:
        text = default
    if text.upper() in WINDOWS_RESERVED_NAMES:
        text = f"_{text}"
    return truncate_text(
        text,
        max(1, int(max_length)),
        marker="",
        preserve_words=False,
    )


def normalize_report_format(
    value: Any,
    *,
    default: ReportFormat = ReportFormat.TEXT,
) -> ReportFormat:
    """Normalize a report format for rendering or writing."""

    try:
        return ReportFormat.coerce(value, default=default)
    except ValueError as error:
        raise UnsupportedReportFormatError(value) from error


def infer_report_format_from_path(
    path: PathLike,
    *,
    default: ReportFormat = ReportFormat.TEXT,
) -> ReportFormat:
    """Infer report format from a path suffix."""

    suffix = Path(os.fspath(path)).suffix.lower()
    if suffix:
        try:
            return ReportFormat.coerce(suffix.lstrip("."))
        except ValueError:
            pass
    return default


def ensure_report_suffix(
    path: PathLike,
    report_format: Any,
) -> Path:
    """Ensure the standard suffix for a report format."""

    member = normalize_report_format(report_format)
    output = Path(os.fspath(path))
    suffix = REPORT_FILE_SUFFIXES[member.value]
    if output.suffix.lower() == suffix:
        return output
    if output.suffix:
        return output.with_suffix(suffix)
    return Path(str(output) + suffix)


def default_report_path(
    *,
    directory: Optional[PathLike] = None,
    basename: Any = DEFAULT_REPORT_BASENAME,
    report_format: Any = DEFAULT_REPORT_FORMAT,
) -> Path:
    """Build a default output path."""

    member = normalize_report_format(report_format)
    filename = (
        sanitize_filename(basename)
        + REPORT_FILE_SUFFIXES[member.value]
    )
    return (
        Path(os.fspath(directory)) / filename
        if directory is not None
        else Path(filename)
    )


# 23.2. Output rendering dispatch
# -----------------------------------------------------------------------------

def render_report(
    report: Union[ReportDocument, Any],
    *,
    report_format: Any = DEFAULT_REPORT_FORMAT,
    config: Optional[ReportConfig] = None,
) -> str:
    """Render a report in a supported output format."""

    member = normalize_report_format(report_format)
    if member is ReportFormat.TEXT:
        return render_report_text(report, config=config)
    if member is ReportFormat.MARKDOWN:
        return render_report_markdown(report, config=config)
    if member is ReportFormat.HTML:
        return render_report_html(report, config=config)
    if member is ReportFormat.JSON:
        return render_report_json(report, config=config)
    raise UnsupportedReportFormatError(member.value)


# 23.3. Path validation and backup
# -----------------------------------------------------------------------------

def prepare_output_path(
    path: PathLike,
    *,
    config: WriteConfig = DEFAULT_WRITE_CONFIG,
) -> Path:
    """Validate and prepare an output path."""

    try:
        output = Path(os.fspath(path)).expanduser()
    except (TypeError, ValueError) as error:
        raise ReportPathError(
            "Invalid report output path.",
            path=str(path),
            cause=error,
        ) from error

    if not output.name:
        raise ReportPathError(
            "Report output path must include a filename.",
            path=output,
        )

    parent = output.parent
    if config.create_parents:
        try:
            parent.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise ReportPathError(
                "Unable to create report output directory.",
                path=parent,
                cause=error,
            ) from error

    if not parent.exists() or not parent.is_dir():
        raise ReportPathError(
            "Report output directory does not exist.",
            path=parent,
        )

    if output.exists() and output.is_dir():
        raise ReportPathError(
            "Report output path refers to a directory.",
            path=output,
        )

    if output.exists() and not config.overwrite:
        raise ReportOverwriteError(output)

    return output


def create_backup_file(
    path: Path,
    *,
    suffix: str = DEFAULT_BACKUP_SUFFIX,
) -> Optional[Path]:
    """Create a non-destructive backup of an existing file."""

    if not path.exists() or not path.is_file():
        return None

    candidate = Path(str(path) + suffix)
    index = 1
    while candidate.exists():
        candidate = Path(f"{path}{suffix}.{index}")
        index += 1

    try:
        candidate.write_bytes(path.read_bytes())
    except OSError as error:
        raise ReportWriteError(
            "Unable to create report backup.",
            path=candidate,
            cause=error,
        ) from error
    return candidate


# 23.4. Atomic and direct text writing
# -----------------------------------------------------------------------------

def _write_text_handle(
    handle: TextIO,
    content: str,
    *,
    fsync: bool = False,
) -> int:
    """Write text and optionally synchronize it."""

    written = handle.write(content)
    handle.flush()
    if fsync:
        os.fsync(handle.fileno())
    return written


def write_text_direct(
    path: Path,
    content: str,
    *,
    config: WriteConfig = DEFAULT_WRITE_CONFIG,
) -> int:
    """Write text directly to its final path."""

    mode = (
        WRITE_MODE_TEXT
        if config.overwrite
        else WRITE_MODE_EXCLUSIVE
    )
    try:
        with path.open(
            mode,
            encoding=config.encoding,
            newline=config.newline,
        ) as handle:
            _write_text_handle(
                handle,
                content,
                fsync=config.fsync,
            )
    except FileExistsError as error:
        raise ReportOverwriteError(path, cause=error) from error
    except (OSError, UnicodeError) as error:
        raise ReportWriteError(
            "Unable to write report output.",
            path=path,
            cause=error,
        ) from error

    try:
        return path.stat().st_size
    except OSError:
        return len(content.encode(config.encoding))


def write_text_atomic(
    path: Path,
    content: str,
    *,
    config: WriteConfig = DEFAULT_WRITE_CONFIG,
) -> int:
    """Write text through a temporary file and atomic replace."""

    temporary_path: Optional[Path] = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding=config.encoding,
            newline=config.newline,
            prefix=f".{path.name}.",
            suffix=config.temp_suffix,
            dir=str(path.parent),
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            _write_text_handle(
                handle,
                content,
                fsync=config.fsync,
            )

        if path.exists() and not config.overwrite:
            raise ReportOverwriteError(path)

        os.replace(temporary_path, path)
        temporary_path = None
    except ReportError:
        raise
    except (OSError, UnicodeError) as error:
        raise ReportWriteError(
            "Unable to write report atomically.",
            path=path,
            cause=error,
        ) from error
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass

    try:
        return path.stat().st_size
    except OSError:
        return len(content.encode(config.encoding))


# 23.5. Generic content writing
# -----------------------------------------------------------------------------

def write_report_content(
    content: str,
    path: PathLike,
    *,
    report_format: Any = DEFAULT_REPORT_FORMAT,
    config: WriteConfig = DEFAULT_WRITE_CONFIG,
    calculate_checksum: bool = False,
) -> WriteResult:
    """Write already-rendered report content safely."""

    member = normalize_report_format(report_format)
    output = prepare_output_path(path, config=config)
    existed = output.exists()

    backup_path: Optional[Path] = None
    if existed and config.backup:
        backup_path = create_backup_file(
            output,
            suffix=config.backup_suffix,
        )

    bytes_written = (
        write_text_atomic(output, content, config=config)
        if config.atomic
        else write_text_direct(output, content, config=config)
    )

    checksum = (
        file_checksum(output)
        if calculate_checksum
        else None
    )

    return WriteResult(
        path=str(output),
        format=member,
        bytes_written=bytes_written,
        atomic=config.atomic,
        overwritten=existed,
        backup_path=(
            str(backup_path)
            if backup_path is not None
            else None
        ),
        checksum=checksum,
        encoding=config.encoding,
        metadata={
            "mime_type": REPORT_MIME_TYPES[member.value],
            "suffix": output.suffix,
        },
    )


def write_report(
    report: Union[ReportDocument, Any],
    path: PathLike,
    *,
    report_format: Any = None,
    config: Optional[ReportConfig] = None,
    write_config: Optional[WriteConfig] = None,
    ensure_suffix: bool = False,
    calculate_checksum: bool = False,
) -> WriteResult:
    """Render and safely write a report."""

    active_config = (
        config
        or (
            report.config
            if isinstance(report, ReportDocument)
            else DEFAULT_REPORT_CONFIG
        )
    )
    output_format = (
        infer_report_format_from_path(
            path,
            default=active_config.rendering.format,
        )
        if report_format is None
        else normalize_report_format(report_format)
    )
    output_path = (
        ensure_report_suffix(path, output_format)
        if ensure_suffix
        else Path(os.fspath(path))
    )
    active_write_config = write_config or active_config.writing

    content = render_report(
        report,
        report_format=output_format,
        config=active_config,
    )
    return write_report_content(
        content,
        output_path,
        report_format=output_format,
        config=active_write_config,
        calculate_checksum=calculate_checksum,
    )


# 23.6. Multi-format writing
# -----------------------------------------------------------------------------

def write_report_formats(
    report: Union[ReportDocument, Any],
    directory: PathLike,
    *,
    formats: Iterable[Any] = SUPPORTED_REPORT_FORMATS,
    basename: Any = DEFAULT_REPORT_BASENAME,
    config: Optional[ReportConfig] = None,
    write_config: Optional[WriteConfig] = None,
    calculate_checksum: bool = False,
) -> Dict[str, WriteResult]:
    """Write a report in multiple supported formats."""

    active_config = (
        config
        or (
            report.config
            if isinstance(report, ReportDocument)
            else DEFAULT_REPORT_CONFIG
        )
    )
    active_write_config = write_config or active_config.writing
    output_directory = Path(os.fspath(directory))
    results: Dict[str, WriteResult] = {}

    for value in formats:
        member = normalize_report_format(value)
        path = default_report_path(
            directory=output_directory,
            basename=basename,
            report_format=member,
        )
        results[member.value] = write_report(
            report,
            path,
            report_format=member,
            config=active_config,
            write_config=active_write_config,
            calculate_checksum=calculate_checksum,
        )
    return results


# 23.7. Public writing interface
# -----------------------------------------------------------------------------

_SECTION_23_PUBLIC_NAMES: Final[Tuple[str, ...]] = (
    "WriteResult",
    "sanitize_filename",
    "normalize_report_format",
    "infer_report_format_from_path",
    "ensure_report_suffix",
    "default_report_path",
    "render_report",
    "prepare_output_path",
    "create_backup_file",
    "write_text_direct",
    "write_text_atomic",
    "write_report_content",
    "write_report",
    "write_report_formats",
)

_register_public_names(_SECTION_23_PUBLIC_NAMES)

# =============================================================================
# End of Section 23
# =============================================================================

# =============================================================================
# Section 24 — Integration with export.py
# =============================================================================

# 24.1. Integration constants and records
# -----------------------------------------------------------------------------

REPORT_EXPORT_INTEGRATION_SCHEMA: Final[str] = "dockanalyzer.report-export"
EXPORT_INTEGRATION_SCHEMA_VERSION: Final[str] = "1.0"

EXPORT_MODULE_CANDIDATES: Final[Tuple[str, ...]] = (
    f"{__package__}.export" if __package__ else "",
    "export",
)

EXPORT_CAPABILITY_NAMES: Final[Tuple[str, ...]] = (
    "available_export_formats",
    "normalize_export_format",
    "export_data",
    "write_json",
    "write_json_lines",
    "write_csv",
    "write_tsv",
    "write_excel",
    "write_text",
    "build_table",
    "TableCollection",
    "ExportedFile",
    "ExportResult",
)

REPORT_TO_EXPORT_TABLE_NAMES: Final[Mapping[str, str]] = MappingProxyType(
    {
        SECTION_OVERVIEW: "summary",
        SECTION_INPUTS: "files",
        SECTION_INTERACTIONS: TABLE_INTERACTIONS,
        SECTION_RESIDUES: TABLE_RESIDUES,
        SECTION_HOTSPOTS: TABLE_HOTSPOTS,
        SECTION_SCORING: TABLE_SCORES,
        SECTION_MULTIPOSE: TABLE_RANKING,
        SECTION_PROVENANCE: TABLE_PROVENANCE,
        SECTION_WARNINGS: TABLE_WARNINGS,
        SECTION_ERRORS: TABLE_ERRORS,
    }
)

REPORT_EXPORT_FORMAT_ALIASES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "json": "json",
        ".json": "json",
        "jsonl": "jsonl",
        ".jsonl": "jsonl",
        "ndjson": "jsonl",
        "csv": "csv",
        ".csv": "csv",
        "tsv": "tsv",
        ".tsv": "tsv",
        "xlsx": "xlsx",
        ".xlsx": "xlsx",
        "excel": "xlsx",
        "txt": "txt",
        ".txt": "txt",
        "text": "txt",
    }
)

REPORT_EXPORT_PAYLOAD_FORMATS: Final[FrozenSet[str]] = frozenset(
    {"json", "jsonl", "txt"}
)
REPORT_EXPORT_TABLE_FORMATS: Final[FrozenSet[str]] = frozenset(
    {"csv", "tsv", "xlsx"}
)


@dataclass(frozen=True)
class ExportIntegrationCapabilities:
    """Detected public capabilities of ``export.py``."""

    available: bool
    module_name: str = ""
    module_version: str = ""
    formats: Tuple[str, ...] = ()
    functions: Tuple[str, ...] = ()
    classes: Tuple[str, ...] = ()
    missing: Tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(
        default_factory=lambda: _EMPTY_METADATA
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "available", bool(self.available))
        for name in ("module_name", "module_version"):
            object.__setattr__(
                self,
                name,
                single_line_text(getattr(self, name), ""),
            )
        for name in (
            "formats",
            "functions",
            "classes",
            "missing",
        ):
            object.__setattr__(
                self,
                name,
                _freeze_config_strings(getattr(self, name)),
            )
        object.__setattr__(
            self,
            "metadata",
            _freeze_config_mapping(self.metadata),
        )

    def supports(self, name: str) -> bool:
        """Return whether a public capability is available."""

        return name in self.functions or name in self.classes

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain capability record."""

        return {
            "available": self.available,
            "module_name": self.module_name,
            "module_version": self.module_version,
            "formats": list(self.formats),
            "functions": list(self.functions),
            "classes": list(self.classes),
            "missing": list(self.missing),
            KEY_METADATA: dict(self.metadata),
        }


@dataclass(frozen=True)
class ReportExportResult:
    """Normalized result returned by report/export integration."""

    status: str
    format: str
    mode: str
    files: Tuple[str, ...] = ()
    output_path: Optional[str] = None
    payload: Optional[Mapping[str, Any]] = None
    tables: Mapping[str, Tuple[Mapping[str, Any], ...]] = field(
        default_factory=lambda: _EMPTY_METADATA
    )
    native_result: Any = None
    warnings: Tuple[str, ...] = ()
    errors: Tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(
        default_factory=lambda: _EMPTY_METADATA
    )

    def __post_init__(self) -> None:
        for name in ("status", "format", "mode"):
            object.__setattr__(
                self,
                name,
                single_line_text(getattr(self, name), ""),
            )
        object.__setattr__(
            self,
            "files",
            _freeze_config_strings(self.files),
        )
        object.__setattr__(
            self,
            "output_path",
            (
                None
                if self.output_path is None
                else single_line_text(self.output_path, "")
            ),
        )
        if self.payload is not None:
            object.__setattr__(
                self,
                "payload",
                MappingProxyType(dict(self.payload)),
            )
        normalized_tables = {
            str(name): tuple(
                MappingProxyType(dict(row))
                for row in rows
            )
            for name, rows in dict(self.tables).items()
        }
        object.__setattr__(
            self,
            "tables",
            MappingProxyType(normalized_tables),
        )
        object.__setattr__(
            self,
            "warnings",
            _freeze_config_strings(self.warnings, unique=False),
        )
        object.__setattr__(
            self,
            "errors",
            _freeze_config_strings(self.errors, unique=False),
        )
        object.__setattr__(
            self,
            "metadata",
            _freeze_config_mapping(self.metadata),
        )

    @property
    def succeeded(self) -> bool:
        """Return whether the integration completed successfully."""

        return self.status in {"success", "partial"} and not self.errors

    def to_dict(self) -> Dict[str, Any]:
        """Return a serializable integration result."""

        return {
            "status": self.status,
            "format": self.format,
            "mode": self.mode,
            "files": list(self.files),
            "output_path": self.output_path,
            "payload": (
                dict(self.payload)
                if self.payload is not None
                else None
            ),
            "tables": {
                name: [dict(row) for row in rows]
                for name, rows in self.tables.items()
            },
            KEY_WARNINGS: list(self.warnings),
            KEY_ERRORS: list(self.errors),
            KEY_METADATA: dict(self.metadata),
        }


# 24.2. Local export-module loading
# -----------------------------------------------------------------------------

def load_export_module(
    *,
    required: bool = True,
) -> Any:
    """Import ``export.py`` locally to avoid circular imports."""

    errors: List[BaseException] = []

    if __package__:
        try:
            from . import export as export_module
            return export_module
        except (ImportError, AttributeError) as error:
            errors.append(error)

    try:
        import export as export_module
        return export_module
    except ImportError as error:
        errors.append(error)

    if required:
        cause = errors[-1] if errors else None
        raise ReportExportError(
            "DockAnalyzer export module is unavailable.",
            cause=cause,
        )
    return None


def export_module_available() -> bool:
    """Return whether ``export.py`` can be imported."""

    return load_export_module(required=False) is not None


def inspect_export_capabilities(
    export_module: Any = None,
) -> ExportIntegrationCapabilities:
    """Inspect the compatible public API exposed by ``export.py``."""

    module = export_module or load_export_module(required=False)
    if module is None:
        return ExportIntegrationCapabilities(
            available=False,
            missing=EXPORT_CAPABILITY_NAMES,
        )

    functions: List[str] = []
    classes: List[str] = []
    missing: List[str] = []

    for name in EXPORT_CAPABILITY_NAMES:
        value = getattr(module, name, MISSING)
        if value is MISSING:
            missing.append(name)
        elif inspect.isclass(value):
            classes.append(name)
        elif callable(value):
            functions.append(name)
        else:
            missing.append(name)

    formats: Tuple[str, ...] = ()
    provider = getattr(module, "available_export_formats", None)
    if callable(provider):
        try:
            formats = tuple(str(item) for item in provider())
        except Exception:
            formats = ()
    if not formats:
        supported = getattr(module, "SUPPORTED_EXPORT_FORMATS", ())
        formats = tuple(sorted(str(item) for item in supported))

    return ExportIntegrationCapabilities(
        available=True,
        module_name=getattr(module, "__name__", "export"),
        module_version=safe_string(
            getattr(module, "__version__", ""),
            "",
        ),
        formats=formats,
        functions=tuple(functions),
        classes=tuple(classes),
        missing=tuple(missing),
        metadata={
            "schema_version": getattr(
                module,
                "EXPORT_SCHEMA_VERSION",
                None,
            ),
            "schema_name": getattr(
                module,
                "EXPORT_SCHEMA_NAME",
                None,
            ),
        },
    )


# 24.3. Export format normalization
# -----------------------------------------------------------------------------

def normalize_report_export_format(
    value: Any,
    *,
    export_module: Any = None,
) -> str:
    """Return an ``export.py`` format name."""

    module = export_module or load_export_module(required=False)
    if module is not None:
        normalizer = getattr(module, "normalize_export_format", None)
        if callable(normalizer):
            try:
                return str(normalizer(value))
            except Exception:
                pass

    token = str(value or "").strip().lower()
    normalized = REPORT_EXPORT_FORMAT_ALIASES.get(
        token,
        REPORT_EXPORT_FORMAT_ALIASES.get(
            token.lstrip("."),
            token.lstrip("."),
        ),
    )
    supported = (
        REPORT_EXPORT_PAYLOAD_FORMATS
        | REPORT_EXPORT_TABLE_FORMATS
    )
    if normalized not in supported:
        raise ReportExportError(
            f"Unsupported export.py format: {value!r}.",
            context={"supported": sorted(supported)},
        )
    return normalized


def report_export_mode(
    format_name: Any,
    *,
    mode: str = "auto",
    export_module: Any = None,
) -> str:
    """Resolve payload or table export mode."""

    normalized_mode = normalize_field_name(mode)
    if normalized_mode in {"payload", "tables"}:
        return normalized_mode
    if normalized_mode != "auto":
        raise ReportConfigurationError(
            f"Unsupported report export mode: {mode!r}."
        )

    format_value = normalize_report_export_format(
        format_name,
        export_module=export_module,
    )
    return (
        "tables"
        if format_value in REPORT_EXPORT_TABLE_FORMATS
        else "payload"
    )


# 24.4. Report payload conversion
# -----------------------------------------------------------------------------

def report_to_export_payload(
    report: Union[ReportDocument, Any],
    *,
    config: Optional[ReportConfig] = None,
    include_config: bool = False,
    include_rendered: bool = False,
    rendered_formats: Iterable[Any] = (),
) -> Dict[str, Any]:
    """Convert a report to a portable ``export.py`` payload."""

    if isinstance(report, ReportDocument):
        document = report
        active_config = config or report.config
    else:
        active_config = config or DEFAULT_REPORT_CONFIG
        document = build_report_document(
            report,
            config=active_config,
        )

    payload = report_to_json_data(
        document,
        config=active_config,
        include_config=include_config,
        max_items=active_config.tables.max_rows,
    )
    payload["integration"] = {
        "schema_name": REPORT_EXPORT_INTEGRATION_SCHEMA,
        "schema_version": EXPORT_INTEGRATION_SCHEMA_VERSION,
        "source_module": SOURCE_MODULE_REPORT,
        "target_module": SOURCE_MODULE_EXPORT,
    }

    if include_rendered:
        rendered: Dict[str, str] = {}
        requested = tuple(rendered_formats) or (
            ReportFormat.TEXT,
            ReportFormat.MARKDOWN,
            ReportFormat.HTML,
        )
        for value in requested:
            member = normalize_report_format(value)
            if member is ReportFormat.JSON:
                continue
            rendered[member.value] = render_report(
                document,
                report_format=member,
                config=active_config,
            )
        payload["rendered"] = rendered

    return payload


# 24.5. Report table conversion
# -----------------------------------------------------------------------------

def export_table_name(
    section_id: Any,
    table: ReportTable,
    *,
    index: int = 1,
    used: Optional[Set[str]] = None,
) -> str:
    """Return a stable export table name."""

    section = normalize_field_name(section_id)
    default_name = REPORT_TO_EXPORT_TABLE_NAMES.get(
        section,
        table.name or "table",
    )
    explicit = normalize_field_name(table.name or "")
    if explicit and explicit not in {"table", section}:
        base = explicit
    else:
        base = default_name

    candidate = base
    occupied = used if used is not None else set()
    suffix = 2
    while candidate in occupied:
        candidate = f"{base}_{suffix}"
        suffix += 1
    occupied.add(candidate)
    return candidate


def report_table_to_export_rows(
    table: ReportTable,
    *,
    include_formatted: bool = False,
    formatting: FormattingConfig = DEFAULT_FORMATTING_CONFIG,
) -> List[Dict[str, Any]]:
    """Convert an internal report table to export rows."""

    columns = table.visible_columns
    rows: List[Dict[str, Any]] = []

    for source in table.rows:
        row: Dict[str, Any] = {}
        for column in columns:
            value = source.get(column.key)
            row[column.key] = to_json_safe(value)
            if include_formatted:
                formatted_key = f"{column.key}_formatted"
                if formatted_key not in row:
                    row[formatted_key] = column.format(
                        value,
                        config=formatting,
                    )
        rows.append(row)
    return rows


def report_to_export_tables(
    report: Union[ReportDocument, Any],
    *,
    config: Optional[ReportConfig] = None,
    include_formatted: bool = False,
    include_empty: bool = False,
) -> Dict[str, List[Dict[str, Any]]]:
    """Convert all report tables to an export table mapping."""

    if isinstance(report, ReportDocument):
        document = report
        active_config = config or report.config
    else:
        active_config = config or DEFAULT_REPORT_CONFIG
        document = build_report_document(
            report,
            config=active_config,
        )

    output: Dict[str, List[Dict[str, Any]]] = {}
    used: Set[str] = set()

    for section in document.visible_sections:
        for index, table in enumerate(
            section_tables(section, config=active_config),
            start=1,
        ):
            if table.empty and not include_empty:
                continue
            name = export_table_name(
                section.id.value,
                table,
                index=index,
                used=used,
            )
            output[name] = report_table_to_export_rows(
                table,
                include_formatted=include_formatted,
                formatting=active_config.formatting,
            )

    if include_empty:
        for section in document.visible_sections:
            default_name = REPORT_TO_EXPORT_TABLE_NAMES.get(
                section.id.value
            )
            if default_name and default_name not in output:
                output[default_name] = []

    return output


def build_export_table_collection(
    report: Union[ReportDocument, Any],
    *,
    config: Optional[ReportConfig] = None,
    include_formatted: bool = False,
    include_empty: bool = False,
    export_module: Any = None,
) -> Any:
    """Build an ``export.py`` ``TableCollection`` when available."""

    module = export_module or load_export_module()
    capabilities = inspect_export_capabilities(module)
    if not capabilities.supports("build_table"):
        raise ReportExportError(
            "export.py does not expose build_table()."
        )

    tables = report_to_export_tables(
        report,
        config=config,
        include_formatted=include_formatted,
        include_empty=include_empty,
    )

    collection_class = getattr(module, "TableCollection", None)
    if collection_class is None:
        return {
            name: module.build_table(rows, name=name)
            for name, rows in tables.items()
        }

    collection = collection_class()
    for name, rows in tables.items():
        table = module.build_table(
            rows,
            name=name,
            metadata={
                "source": SOURCE_MODULE_REPORT,
                "integration_schema": (
                    EXPORT_INTEGRATION_SCHEMA_VERSION
                ),
            },
        )
        add = getattr(collection, "add", None)
        if callable(add):
            add(table)
        elif hasattr(collection, "tables"):
            collection.tables[name] = table
        else:
            raise ReportExportError(
                "Unsupported export.py TableCollection interface."
            )

    metadata = getattr(collection, "metadata", None)
    if isinstance(metadata, dict):
        metadata.update(
            {
                "source": SOURCE_MODULE_REPORT,
                "schema_name": REPORT_SCHEMA_NAME,
                "schema_version": REPORT_SCHEMA_VERSION,
            }
        )
    return collection


# 24.6. Compatible function invocation
# -----------------------------------------------------------------------------

def _supported_kwargs(
    function: Callable[..., Any],
    kwargs: Mapping[str, Any],
) -> Dict[str, Any]:
    """Filter keyword arguments using a callable signature."""

    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return dict(kwargs)

    accepts_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if accepts_kwargs:
        return dict(kwargs)

    return {
        key: value
        for key, value in kwargs.items()
        if key in signature.parameters
    }


def call_export_function(
    function: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Invoke an ``export.py`` function with compatible keywords."""

    filtered = _supported_kwargs(function, kwargs)
    try:
        return function(*args, **filtered)
    except ReportError:
        raise
    except Exception as error:
        raise ReportExportError(
            f"export.py call failed: "
            f"{getattr(function, '__name__', type(function).__name__)}.",
            context={
                "function": getattr(
                    function,
                    "__name__",
                    type(function).__name__,
                ),
                "arguments": tuple(filtered),
            },
            cause=error,
        ) from error


# 24.7. Native-result adaptation
# -----------------------------------------------------------------------------

def exported_file_paths(value: Any) -> Tuple[str, ...]:
    """Extract output paths from export.py result objects."""

    paths: List[str] = []

    direct_path = get_object_field(value, "path", MISSING)
    if direct_path is not MISSING and direct_path is not None:
        paths.append(str(direct_path))

    files = get_object_field(value, "files", MISSING)
    if files is not MISSING:
        for item in iter_object_collection(files):
            item_path = get_object_field(item, "path", MISSING)
            if item_path is not MISSING and item_path is not None:
                paths.append(str(item_path))
            elif isinstance(item, (str, os.PathLike, Path)):
                paths.append(str(item))

    return tuple(dict.fromkeys(paths))


def export_result_messages(
    value: Any,
    field_name: str,
) -> Tuple[str, ...]:
    """Extract warning or error messages from a native result."""

    messages = get_object_field(value, field_name, ())
    return _message_texts(messages)


def adapt_export_result(
    value: Any,
    *,
    format_name: str,
    mode: str,
    payload: Optional[Mapping[str, Any]] = None,
    tables: Optional[Mapping[str, Sequence[Mapping[str, Any]]]] = None,
    output_path: Optional[PathLike] = None,
) -> ReportExportResult:
    """Normalize an ``export.py`` result object."""

    status = get_object_field(value, "status", None)
    if status is None:
        status = "success" if exported_file_paths(value) else "unknown"

    return ReportExportResult(
        status=safe_string(status, "unknown"),
        format=format_name,
        mode=mode,
        files=exported_file_paths(value),
        output_path=(
            str(output_path)
            if output_path is not None
            else None
        ),
        payload=payload,
        tables={
            str(name): tuple(dict(row) for row in rows)
            for name, rows in dict(tables or {}).items()
        },
        native_result=value,
        warnings=export_result_messages(value, "warnings"),
        errors=export_result_messages(value, "errors"),
        metadata={
            "native_type": type(value).__name__,
            "integration_schema": EXPORT_INTEGRATION_SCHEMA_VERSION,
        },
    )


# 24.8. Payload export through export.py
# -----------------------------------------------------------------------------

def export_report_payload(
    report: Union[ReportDocument, Any],
    path: PathLike,
    *,
    format_name: Any = None,
    config: Optional[ReportConfig] = None,
    include_config: bool = False,
    include_rendered: bool = False,
    overwrite: Any = "overwrite",
    export_module: Any = None,
    **kwargs: Any,
) -> ReportExportResult:
    """Export a report payload through ``export.py``."""

    module = export_module or load_export_module()
    resolved_format = normalize_report_export_format(
        format_name or Path(path).suffix or "json",
        export_module=module,
    )
    payload = report_to_export_payload(
        report,
        config=config,
        include_config=include_config,
        include_rendered=include_rendered,
    )

    function = getattr(module, "export_data", None)
    if callable(function):
        native = call_export_function(
            function,
            payload,
            path,
            format=resolved_format,
            overwrite=overwrite,
            **kwargs,
        )
    else:
        writer_name = {
            "json": "write_json",
            "jsonl": "write_json_lines",
            "txt": "write_text",
        }.get(resolved_format)
        writer = getattr(module, writer_name or "", None)
        if not callable(writer):
            raise ReportExportError(
                f"export.py cannot write {resolved_format!r} payloads."
            )
        source = (
            [payload]
            if resolved_format == "jsonl"
            else payload
        )
        native = call_export_function(
            writer,
            source,
            path=path,
            overwrite=overwrite,
            **kwargs,
        )

    return adapt_export_result(
        native,
        format_name=resolved_format,
        mode="payload",
        payload=payload,
        output_path=path,
    )


# 24.9. Table export through export.py
# -----------------------------------------------------------------------------

def _table_output_basename(
    path: Path,
    table_name: str,
) -> Path:
    """Build a path for one delimited table."""

    suffix = path.suffix
    base = path.stem if suffix else path.name
    directory = path.parent
    return directory / f"{base}_{sanitize_filename(table_name)}{suffix}"


def export_report_tables(
    report: Union[ReportDocument, Any],
    path: PathLike,
    *,
    format_name: Any = None,
    config: Optional[ReportConfig] = None,
    include_formatted: bool = False,
    include_empty: bool = False,
    overwrite: Any = "overwrite",
    export_module: Any = None,
    **kwargs: Any,
) -> ReportExportResult:
    """Export report tables through ``export.py``."""

    module = export_module or load_export_module()
    output = Path(path)
    resolved_format = normalize_report_export_format(
        format_name or output.suffix or "xlsx",
        export_module=module,
    )
    if resolved_format not in REPORT_EXPORT_TABLE_FORMATS:
        raise ReportExportError(
            f"Table mode does not support {resolved_format!r}."
        )

    table_mapping = report_to_export_tables(
        report,
        config=config,
        include_formatted=include_formatted,
        include_empty=include_empty,
    )

    if resolved_format == "xlsx":
        collection = build_export_table_collection(
            report,
            config=config,
            include_formatted=include_formatted,
            include_empty=include_empty,
            export_module=module,
        )
        writer = getattr(module, "write_excel", None)
        if not callable(writer):
            raise ReportExportError(
                "export.py does not expose write_excel()."
            )
        native = call_export_function(
            writer,
            collection,
            path=output,
            overwrite=overwrite,
            **kwargs,
        )
        return adapt_export_result(
            native,
            format_name=resolved_format,
            mode="tables",
            tables=table_mapping,
            output_path=output,
        )

    writer_name = (
        "write_csv"
        if resolved_format == "csv"
        else "write_tsv"
    )
    writer = getattr(module, writer_name, None)
    if not callable(writer):
        raise ReportExportError(
            f"export.py does not expose {writer_name}()."
        )

    native_files: List[Any] = []
    for table_name, rows in table_mapping.items():
        table_path = _table_output_basename(
            output,
            table_name,
        )
        native_files.append(
            call_export_function(
                writer,
                rows,
                path=table_path,
                table_name=table_name,
                overwrite=overwrite,
                **kwargs,
            )
        )

    synthetic = __import__("types").SimpleNamespace(
        status="success",
        files=native_files,
        warnings=[],
        errors=[],
    )
    return adapt_export_result(
        synthetic,
        format_name=resolved_format,
        mode="tables",
        tables=table_mapping,
        output_path=output,
    )


# 24.10. Unified report export interface
# -----------------------------------------------------------------------------

def export_report_with_export_module(
    report: Union[ReportDocument, Any],
    path: PathLike,
    *,
    format_name: Any = None,
    mode: str = "auto",
    config: Optional[ReportConfig] = None,
    include_config: bool = False,
    include_rendered: bool = False,
    include_formatted_tables: bool = False,
    include_empty_tables: bool = False,
    overwrite: Any = "overwrite",
    export_module: Any = None,
    **kwargs: Any,
) -> ReportExportResult:
    """Export report data using the shared ``export.py`` layer."""

    module = export_module or load_export_module()
    resolved_format = normalize_report_export_format(
        format_name or Path(path).suffix or "json",
        export_module=module,
    )
    resolved_mode = report_export_mode(
        resolved_format,
        mode=mode,
        export_module=module,
    )

    if resolved_mode == "payload":
        return export_report_payload(
            report,
            path,
            format_name=resolved_format,
            config=config,
            include_config=include_config,
            include_rendered=include_rendered,
            overwrite=overwrite,
            export_module=module,
            **kwargs,
        )

    return export_report_tables(
        report,
        path,
        format_name=resolved_format,
        config=config,
        include_formatted=include_formatted_tables,
        include_empty=include_empty_tables,
        overwrite=overwrite,
        export_module=module,
        **kwargs,
    )


def export_report_bundle(
    report: Union[ReportDocument, Any],
    directory: PathLike,
    *,
    basename: Any = DEFAULT_REPORT_BASENAME,
    formats: Iterable[Any] = (
        "json",
        "csv",
        "xlsx",
    ),
    config: Optional[ReportConfig] = None,
    overwrite: Any = "overwrite",
    export_module: Any = None,
) -> Dict[str, ReportExportResult]:
    """Export a report bundle through ``export.py``."""

    module = export_module or load_export_module()
    output_dir = Path(directory)
    results: Dict[str, ReportExportResult] = {}

    capabilities = inspect_export_capabilities(module)
    suffixes = {
        "json": ".json",
        "jsonl": ".jsonl",
        "csv": ".csv",
        "tsv": ".tsv",
        "xlsx": ".xlsx",
        "txt": ".txt",
    }

    for value in formats:
        normalized = normalize_report_export_format(
            value,
            export_module=module,
        )
        if capabilities.formats and normalized not in capabilities.formats:
            raise ReportExportError(
                f"export.py does not advertise format {normalized!r}."
            )
        path = output_dir / (
            sanitize_filename(basename)
            + suffixes[normalized]
        )
        results[normalized] = export_report_with_export_module(
            report,
            path,
            format_name=normalized,
            mode="auto",
            config=config,
            overwrite=overwrite,
            export_module=module,
        )

    return results


# 24.11. Public export integration interface
# -----------------------------------------------------------------------------

_SECTION_24_PUBLIC_NAMES: Final[Tuple[str, ...]] = (
    "REPORT_EXPORT_INTEGRATION_SCHEMA",
    "EXPORT_INTEGRATION_SCHEMA_VERSION",
    "ExportIntegrationCapabilities",
    "ReportExportResult",
    "load_export_module",
    "export_module_available",
    "inspect_export_capabilities",
    "normalize_report_export_format",
    "report_export_mode",
    "report_to_export_payload",
    "export_table_name",
    "report_table_to_export_rows",
    "report_to_export_tables",
    "build_export_table_collection",
    "call_export_function",
    "exported_file_paths",
    "export_result_messages",
    "adapt_export_result",
    "export_report_payload",
    "export_report_tables",
    "export_report_with_export_module",
    "export_report_bundle",
)

_register_public_names(_SECTION_24_PUBLIC_NAMES)

# =============================================================================
# End of Section 24
# =============================================================================

# =============================================================================
# Section 25 — Permissive mode and errors
# =============================================================================

# 25.1. Diagnostic records
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class DiagnosticRecord:
    """Normalized warning or error captured during reporting."""

    severity: Severity
    code: str
    message: str
    section: str = ""
    path: str = ""
    exception_type: str = ""
    traceback_text: str = ""
    timestamp: str = ""
    context: Mapping[str, Any] = field(
        default_factory=lambda: _EMPTY_METADATA
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "severity",
            _coerce_enum(
                Severity,
                self.severity,
                "severity",
            ),
        )
        for name in (
            "code",
            "message",
            "section",
            "path",
            "exception_type",
        ):
            object.__setattr__(
                self,
                name,
                single_line_text(getattr(self, name), ""),
            )
        object.__setattr__(
            self,
            "traceback_text",
            safe_string(self.traceback_text, "", strip=False),
        )
        object.__setattr__(
            self,
            "timestamp",
            self.timestamp or current_utc_timestamp(),
        )
        object.__setattr__(
            self,
            "context",
            _freeze_config_mapping(self.context),
        )

    @classmethod
    def from_exception(
        cls,
        error: BaseException,
        *,
        severity: Severity = Severity.ERROR,
        code: Optional[str] = None,
        section: Optional[str] = None,
        path: Optional[PathLike] = None,
        include_traceback: bool = False,
        context: Optional[Mapping[str, Any]] = None,
    ) -> "DiagnosticRecord":
        """Create a diagnostic from an exception."""

        report_error = (
            error
            if isinstance(error, ReportError)
            else ReportError.from_exception(error)
        )
        error_context = dict(
            getattr(report_error, "context", _EMPTY_METADATA)
        )
        error_context.update(dict(context or {}))
        traceback_text = ""
        if include_traceback:
            traceback_text = "".join(
                traceback.format_exception(
                    type(error),
                    error,
                    error.__traceback__,
                )
            )
        return cls(
            severity=severity,
            code=code or getattr(
                report_error,
                "code",
                type(error).__name__,
            ),
            message=_exception_message(error),
            section=section
            or getattr(report_error, "section", "")
            or "",
            path=(
                str(path)
                if path is not None
                else str(getattr(report_error, "path", "") or "")
            ),
            exception_type=type(error).__name__,
            traceback_text=traceback_text,
            context=error_context,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain diagnostic record."""

        return {
            "severity": self.severity.value,
            "code": self.code,
            "message": self.message,
            "section": self.section,
            "path": self.path,
            "exception_type": self.exception_type,
            "traceback": self.traceback_text,
            KEY_TIMESTAMP: self.timestamp,
            "context": dict(self.context),
        }


@dataclass(frozen=True)
class ErrorPolicy:
    """Resolved error-handling behavior."""

    mode: ErrorMode = ErrorMode.RAISE
    include_tracebacks: bool = False
    max_messages: int = DEFAULT_MAX_ITEMS
    warning_category: type = ReportWarning

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "mode",
            _coerce_enum(ErrorMode, self.mode, "mode"),
        )
        object.__setattr__(
            self,
            "max_messages",
            max(1, to_safe_int(self.max_messages, DEFAULT_MAX_ITEMS)),
        )
        category = self.warning_category
        if not inspect.isclass(category) or not issubclass(category, Warning):
            raise ReportConfigurationError(
                "warning_category must be a Warning subclass."
            )

    @classmethod
    def from_config(
        cls,
        value: Union[ReportConfig, ErrorHandlingConfig, "ErrorPolicy"],
    ) -> "ErrorPolicy":
        """Resolve a policy from report configuration."""

        if isinstance(value, cls):
            return value
        error_config = (
            value.errors
            if isinstance(value, ReportConfig)
            else value
        )
        if not isinstance(error_config, ErrorHandlingConfig):
            raise ReportConfigurationError(
                "Expected ReportConfig or ErrorHandlingConfig."
            )
        return cls(
            mode=error_config.mode,
            include_tracebacks=error_config.include_tracebacks,
            max_messages=error_config.max_messages,
            warning_category=error_config.warning_category,
        )


@dataclass
class DiagnosticCollector:
    """Mutable diagnostic accumulator."""

    max_items: int = DEFAULT_MAX_ITEMS
    include_tracebacks: bool = False
    records: List[DiagnosticRecord] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.max_items = max(
            1,
            to_safe_int(self.max_items, DEFAULT_MAX_ITEMS),
        )
        self.include_tracebacks = bool(self.include_tracebacks)

    def add(
        self,
        record: DiagnosticRecord,
    ) -> DiagnosticRecord:
        """Add one diagnostic while respecting the item limit."""

        if not isinstance(record, DiagnosticRecord):
            raise ReportConfigurationError(
                "record must be DiagnosticRecord."
            )
        if len(self.records) < self.max_items:
            self.records.append(record)
        return record

    def capture(
        self,
        error: BaseException,
        *,
        severity: Severity = Severity.ERROR,
        code: Optional[str] = None,
        section: Optional[str] = None,
        path: Optional[PathLike] = None,
        context: Optional[Mapping[str, Any]] = None,
    ) -> DiagnosticRecord:
        """Capture one exception."""

        return self.add(
            DiagnosticRecord.from_exception(
                error,
                severity=severity,
                code=code,
                section=section,
                path=path,
                include_traceback=self.include_tracebacks,
                context=context,
            )
        )

    def warning(
        self,
        message: Any,
        *,
        code: str = "report_warning",
        section: str = "",
        context: Optional[Mapping[str, Any]] = None,
    ) -> DiagnosticRecord:
        """Add a warning diagnostic."""

        return self.add(
            DiagnosticRecord(
                severity=Severity.WARNING,
                code=code,
                message=safe_string(message, ""),
                section=section,
                context=context or {},
            )
        )

    def error(
        self,
        message: Any,
        *,
        code: str = "report_error",
        section: str = "",
        context: Optional[Mapping[str, Any]] = None,
    ) -> DiagnosticRecord:
        """Add an error diagnostic."""

        return self.add(
            DiagnosticRecord(
                severity=Severity.ERROR,
                code=code,
                message=safe_string(message, ""),
                section=section,
                context=context or {},
            )
        )

    def extend(
        self,
        records: Iterable[DiagnosticRecord],
    ) -> None:
        """Append multiple diagnostics."""

        for record in records:
            self.add(record)

    @property
    def warnings(self) -> Tuple[DiagnosticRecord, ...]:
        """Return warning diagnostics."""

        return tuple(
            item
            for item in self.records
            if item.severity is Severity.WARNING
        )

    @property
    def errors(self) -> Tuple[DiagnosticRecord, ...]:
        """Return error and critical diagnostics."""

        return tuple(
            item
            for item in self.records
            if item.severity in {
                Severity.ERROR,
                Severity.CRITICAL,
            }
        )

    @property
    def has_errors(self) -> bool:
        """Return whether error diagnostics were captured."""

        return bool(self.errors)

    def raise_if_errors(self) -> None:
        """Raise one aggregate error when errors are present."""

        if not self.errors:
            return
        errors = tuple(
            ReportError(
                item.message,
                code=item.code,
                section=item.section or None,
                path=item.path or None,
                context=item.context,
            )
            for item in self.errors
        )
        raise ReportAggregateError(errors)

    def to_dict(self) -> Dict[str, Any]:
        """Return serialized diagnostics."""

        return {
            "count": len(self.records),
            "warning_count": len(self.warnings),
            "error_count": len(self.errors),
            "records": [record.to_dict() for record in self.records],
        }


# 25.2. Error normalization and handling
# -----------------------------------------------------------------------------

def normalize_error_policy(
    value: Union[
        None,
        ErrorMode,
        str,
        ErrorPolicy,
        ErrorHandlingConfig,
        ReportConfig,
    ] = None,
) -> ErrorPolicy:
    """Normalize error-mode input into a policy."""

    if value is None:
        return ErrorPolicy.from_config(DEFAULT_ERROR_HANDLING_CONFIG)
    if isinstance(value, ErrorPolicy):
        return value
    if isinstance(value, (ReportConfig, ErrorHandlingConfig)):
        return ErrorPolicy.from_config(value)
    try:
        mode = ErrorMode.coerce(value)
    except ValueError as error:
        raise ReportConfigurationError(
            f"Unsupported error mode: {value!r}.",
            cause=error,
        ) from error
    return ErrorPolicy(mode=mode)


def wrap_report_exception(
    error: BaseException,
    *,
    error_type: type = ReportError,
    message: Optional[str] = None,
    section: Optional[str] = None,
    path: Optional[PathLike] = None,
    context: Optional[Mapping[str, Any]] = None,
) -> ReportError:
    """Wrap arbitrary exceptions in a report exception."""

    if isinstance(error, ReportError):
        return error
    if not inspect.isclass(error_type) or not issubclass(
        error_type,
        ReportError,
    ):
        error_type = ReportError
    return error_type(
        message or _exception_message(error),
        section=section,
        path=path,
        context=context,
        cause=error,
    )


def emit_report_warning(
    error: Union[BaseException, str],
    *,
    policy: Union[
        ErrorPolicy,
        ErrorHandlingConfig,
        ReportConfig,
        ErrorMode,
        str,
    ] = DEFAULT_ERROR_HANDLING_CONFIG,
) -> None:
    """Emit one warning using the configured warning category."""

    resolved = normalize_error_policy(policy)
    message = (
        _exception_message(error)
        if isinstance(error, BaseException)
        else safe_string(error, "")
    )
    warnings.warn(
        message,
        resolved.warning_category,
        stacklevel=2,
    )


def handle_report_error(
    error: BaseException,
    *,
    policy: Union[
        ErrorPolicy,
        ErrorHandlingConfig,
        ReportConfig,
        ErrorMode,
        str,
    ] = DEFAULT_ERROR_HANDLING_CONFIG,
    collector: Optional[DiagnosticCollector] = None,
    fallback: Any = None,
    severity: Severity = Severity.ERROR,
    section: Optional[str] = None,
    path: Optional[PathLike] = None,
    context: Optional[Mapping[str, Any]] = None,
) -> Any:
    """Apply raise, warn, collect or ignore behavior."""

    resolved = normalize_error_policy(policy)
    wrapped = wrap_report_exception(
        error,
        section=section,
        path=path,
        context=context,
    )

    if resolved.mode is ErrorMode.RAISE:
        raise wrapped from error

    if collector is not None and resolved.mode in {
        ErrorMode.WARN,
        ErrorMode.COLLECT,
    }:
        collector.capture(
            wrapped,
            severity=severity,
            section=section,
            path=path,
            context=context,
        )

    if resolved.mode is ErrorMode.WARN:
        emit_report_warning(wrapped, policy=resolved)

    return fallback


def permissive_call(
    function: Callable[..., T],
    *args: Any,
    policy: Union[
        ErrorPolicy,
        ErrorHandlingConfig,
        ReportConfig,
        ErrorMode,
        str,
    ] = DEFAULT_ERROR_HANDLING_CONFIG,
    collector: Optional[DiagnosticCollector] = None,
    fallback: Any = None,
    error_type: type = ReportError,
    message: Optional[str] = None,
    section: Optional[str] = None,
    path: Optional[PathLike] = None,
    context: Optional[Mapping[str, Any]] = None,
    **kwargs: Any,
) -> Any:
    """Execute a callable under a report error policy."""

    try:
        return function(*args, **kwargs)
    except Exception as error:
        wrapped = wrap_report_exception(
            error,
            error_type=error_type,
            message=message,
            section=section,
            path=path,
            context=context,
        )
        return handle_report_error(
            wrapped,
            policy=policy,
            collector=collector,
            fallback=fallback,
            section=section,
            path=path,
            context=context,
        )


class ReportErrorBoundary:
    """Context manager applying one report error policy."""

    def __init__(
        self,
        *,
        policy: Union[
            ErrorPolicy,
            ErrorHandlingConfig,
            ReportConfig,
            ErrorMode,
            str,
        ] = DEFAULT_ERROR_HANDLING_CONFIG,
        collector: Optional[DiagnosticCollector] = None,
        section: Optional[str] = None,
        path: Optional[PathLike] = None,
        context: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.policy = normalize_error_policy(policy)
        self.collector = collector
        self.section = section
        self.path = path
        self.context = dict(context or {})
        self.error: Optional[ReportError] = None

    def __enter__(self) -> "ReportErrorBoundary":
        return self

    def __exit__(
        self,
        error_type: Any,
        error: Optional[BaseException],
        traceback_object: Any,
    ) -> bool:
        if error is None:
            return False
        self.error = wrap_report_exception(
            error,
            section=self.section,
            path=self.path,
            context=self.context,
        )
        if self.policy.mode is ErrorMode.RAISE:
            return False
        handle_report_error(
            self.error,
            policy=self.policy,
            collector=self.collector,
            section=self.section,
            path=self.path,
            context=self.context,
        )
        return True


# 25.3. Report diagnostic extraction
# -----------------------------------------------------------------------------

def report_diagnostics(
    report: ReportDocument,
    *,
    include_sections: bool = True,
    include_tracebacks: bool = False,
    max_items: int = DEFAULT_MAX_ITEMS,
) -> DiagnosticCollector:
    """Collect warnings and errors already attached to a report."""

    collector = DiagnosticCollector(
        max_items=max_items,
        include_tracebacks=include_tracebacks,
    )
    for message in report.warnings:
        collector.warning(
            message,
            code="document_warning",
        )
    for message in report.errors:
        collector.error(
            message,
            code="document_error",
        )

    if include_sections:
        for section in report.sections:
            for message in section.warnings:
                collector.warning(
                    message,
                    code="section_warning",
                    section=section.id.value,
                )
            for message in section.errors:
                collector.error(
                    message,
                    code="section_error",
                    section=section.id.value,
                )
    return collector


def diagnostics_to_notice_blocks(
    diagnostics: Union[
        DiagnosticCollector,
        Iterable[DiagnosticRecord],
    ],
) -> Tuple[ReportBlock, ...]:
    """Convert diagnostics to visible report notices."""

    records = (
        diagnostics.records
        if isinstance(diagnostics, DiagnosticCollector)
        else list(diagnostics)
    )
    return tuple(
        notice_block(
            record.message,
            severity=record.severity,
            title=record.code,
        )
        for record in records
    )


def merge_report_diagnostics(
    report: ReportDocument,
    diagnostics: Union[
        DiagnosticCollector,
        Iterable[DiagnosticRecord],
    ],
) -> ReportDocument:
    """Return a report with diagnostic messages attached."""

    records = (
        diagnostics.records
        if isinstance(diagnostics, DiagnosticCollector)
        else list(diagnostics)
    )
    warning_messages = tuple(
        record.message
        for record in records
        if record.severity is Severity.WARNING
    )
    error_messages = tuple(
        record.message
        for record in records
        if record.severity in {
            Severity.ERROR,
            Severity.CRITICAL,
        }
    )
    return replace(
        report,
        warnings=tuple(
            dict.fromkeys((*report.warnings, *warning_messages))
        ),
        errors=tuple(
            dict.fromkeys((*report.errors, *error_messages))
        ),
    )


# 25.4. Public permissive-mode interface
# -----------------------------------------------------------------------------

_SECTION_25_PUBLIC_NAMES: Final[Tuple[str, ...]] = (
    "DiagnosticRecord",
    "ErrorPolicy",
    "DiagnosticCollector",
    "normalize_error_policy",
    "wrap_report_exception",
    "emit_report_warning",
    "handle_report_error",
    "permissive_call",
    "ReportErrorBoundary",
    "report_diagnostics",
    "diagnostics_to_notice_blocks",
    "merge_report_diagnostics",
)

_register_public_names(_SECTION_25_PUBLIC_NAMES)

# =============================================================================
# End of Section 25
# =============================================================================


# =============================================================================
# Section 26 — Validation
# =============================================================================

# 26.1. Validation records
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class ValidationIssue:
    """One structural or semantic validation issue."""

    severity: Severity
    code: str
    message: str
    location: str = ""
    expected: Any = None
    actual: Any = None
    context: Mapping[str, Any] = field(
        default_factory=lambda: _EMPTY_METADATA
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "severity",
            _coerce_enum(
                Severity,
                self.severity,
                "severity",
            ),
        )
        for name in ("code", "message", "location"):
            object.__setattr__(
                self,
                name,
                single_line_text(getattr(self, name), ""),
            )
        object.__setattr__(
            self,
            "context",
            _freeze_config_mapping(self.context),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain validation issue."""

        return {
            "severity": self.severity.value,
            "code": self.code,
            "message": self.message,
            "location": self.location,
            "expected": to_json_safe(self.expected),
            "actual": to_json_safe(self.actual),
            "context": dict(self.context),
        }


@dataclass(frozen=True)
class ValidationResult:
    """Complete validation result."""

    target_type: str
    issues: Tuple[ValidationIssue, ...] = ()
    checked_at: str = ""
    metadata: Mapping[str, Any] = field(
        default_factory=lambda: _EMPTY_METADATA
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "target_type",
            single_line_text(self.target_type, DEFAULT_UNKNOWN_TEXT),
        )
        object.__setattr__(self, "issues", tuple(self.issues))
        object.__setattr__(
            self,
            "checked_at",
            self.checked_at or current_utc_timestamp(),
        )
        object.__setattr__(
            self,
            "metadata",
            _freeze_config_mapping(self.metadata),
        )

    @property
    def errors(self) -> Tuple[ValidationIssue, ...]:
        """Return error and critical issues."""

        return tuple(
            issue
            for issue in self.issues
            if issue.severity in {
                Severity.ERROR,
                Severity.CRITICAL,
            }
        )

    @property
    def warnings(self) -> Tuple[ValidationIssue, ...]:
        """Return warning issues."""

        return tuple(
            issue
            for issue in self.issues
            if issue.severity is Severity.WARNING
        )

    @property
    def valid(self) -> bool:
        """Return whether no errors were found."""

        return not self.errors

    def raise_for_errors(self) -> None:
        """Raise one validation error when invalid."""

        if self.valid:
            return
        first = self.errors[0]
        raise ReportValidationError(
            first.message,
            context={
                "issue_count": len(self.issues),
                "errors": [
                    issue.to_dict() for issue in self.errors
                ],
            },
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain validation result."""

        return {
            "valid": self.valid,
            "target_type": self.target_type,
            "checked_at": self.checked_at,
            "issue_count": len(self.issues),
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "issues": [issue.to_dict() for issue in self.issues],
            KEY_METADATA: dict(self.metadata),
        }


# 26.2. Validation helpers
# -----------------------------------------------------------------------------

def validation_issue(
    code: str,
    message: str,
    *,
    severity: Severity = Severity.ERROR,
    location: str = "",
    expected: Any = None,
    actual: Any = None,
    context: Optional[Mapping[str, Any]] = None,
) -> ValidationIssue:
    """Create one validation issue."""

    return ValidationIssue(
        severity=severity,
        code=code,
        message=message,
        location=location,
        expected=expected,
        actual=actual,
        context=context or {},
    )


def combine_validation_results(
    *results: ValidationResult,
    target_type: str = "combined",
    metadata: Optional[Mapping[str, Any]] = None,
) -> ValidationResult:
    """Merge multiple validation results."""

    return ValidationResult(
        target_type=target_type,
        issues=tuple(
            issue
            for result in results
            for issue in result.issues
        ),
        metadata=metadata or {},
    )


def _validate_nonnegative(
    value: Any,
    *,
    location: str,
    code: str,
) -> List[ValidationIssue]:
    """Validate a non-negative integer-like value."""

    integer = to_safe_int(value, MISSING)
    if integer is MISSING or integer < 0:
        return [
            validation_issue(
                code,
                f"{location} must be a non-negative integer.",
                location=location,
                expected="integer >= 0",
                actual=value,
            )
        ]
    return []


# 26.3. Configuration validation
# -----------------------------------------------------------------------------

def validate_report_config(
    config: Any,
) -> ValidationResult:
    """Validate a report configuration."""

    issues: List[ValidationIssue] = []
    if not isinstance(config, ReportConfig):
        issues.append(
            validation_issue(
                "invalid_config_type",
                "Configuration must be ReportConfig.",
                location="config",
                expected="ReportConfig",
                actual=type(config).__name__,
            )
        )
        return ValidationResult(
            target_type=type(config).__name__,
            issues=tuple(issues),
        )

    if not config.title.strip():
        issues.append(
            validation_issue(
                "empty_title",
                "Report title is empty.",
                severity=Severity.WARNING,
                location="config.title",
            )
        )

    if len(config.section_order) != len(set(config.section_order)):
        issues.append(
            validation_issue(
                "duplicate_section_order",
                "Section order contains duplicates.",
                location="config.section_order",
            )
        )

    missing_enabled = (
        set(config.enabled_sections) - set(config.section_order)
    )
    if missing_enabled:
        issues.append(
            validation_issue(
                "enabled_section_missing_order",
                "Enabled sections are missing from section order.",
                location="config.enabled_sections",
                expected="subset of section_order",
                actual=tuple(
                    section.value for section in missing_enabled
                ),
            )
        )

    if config.tables.max_rows <= 0:
        issues.append(
            validation_issue(
                "invalid_max_rows",
                "Table row limit must be positive.",
                location="config.tables.max_rows",
                actual=config.tables.max_rows,
            )
        )

    if config.rendering.width < MIN_TEXT_WIDTH:
        issues.append(
            validation_issue(
                "invalid_text_width",
                "Text width is below the supported minimum.",
                location="config.rendering.width",
                expected=f">= {MIN_TEXT_WIDTH}",
                actual=config.rendering.width,
            )
        )

    if config.scoring.recalculate:
        issues.append(
            validation_issue(
                "score_recalculation_enabled",
                "Scoring recalculation is enabled.",
                severity=Severity.WARNING,
                location="config.scoring.recalculate",
                context={
                    "note": (
                        "Reports normally consume existing scores."
                    )
                },
            )
        )

    return ValidationResult(
        target_type="ReportConfig",
        issues=tuple(issues),
        metadata={
            "enabled_sections": len(config.enabled_sections),
            "section_order": len(config.section_order),
        },
    )


# 26.4. Interaction validation
# -----------------------------------------------------------------------------

def validate_normalized_interaction(
    interaction: Any,
    *,
    location: str = "interaction",
) -> ValidationResult:
    """Validate one normalized interaction."""

    issues: List[ValidationIssue] = []
    if not isinstance(interaction, NormalizedInteraction):
        issues.append(
            validation_issue(
                "invalid_interaction_type",
                "Interaction must be NormalizedInteraction.",
                location=location,
                expected="NormalizedInteraction",
                actual=type(interaction).__name__,
            )
        )
        return ValidationResult(
            target_type=type(interaction).__name__,
            issues=tuple(issues),
        )

    if not interaction.id:
        issues.append(
            validation_issue(
                "missing_interaction_id",
                "Normalized interaction has no identifier.",
                location=f"{location}.id",
            )
        )
    if interaction.family is InteractionFamily.UNKNOWN:
        issues.append(
            validation_issue(
                "unknown_interaction_family",
                "Interaction family is unknown.",
                severity=Severity.WARNING,
                location=f"{location}.family",
            )
        )
    if not interaction.type:
        issues.append(
            validation_issue(
                "missing_interaction_type",
                "Interaction type is empty.",
                location=f"{location}.type",
            )
        )
    if interaction.distance is not None and interaction.distance < 0:
        issues.append(
            validation_issue(
                "negative_distance",
                "Interaction distance must not be negative.",
                location=f"{location}.distance",
                expected=">= 0",
                actual=interaction.distance,
            )
        )
    if interaction.angle is not None and not (
        0.0 <= interaction.angle <= 360.0
    ):
        issues.append(
            validation_issue(
                "angle_out_of_range",
                "Interaction angle is outside 0–360 degrees.",
                severity=Severity.WARNING,
                location=f"{location}.angle",
                expected="0 <= angle <= 360",
                actual=interaction.angle,
            )
        )
    if not interaction.receptor_residue:
        issues.append(
            validation_issue(
                "missing_receptor_residue",
                "Receptor residue is unavailable.",
                severity=Severity.WARNING,
                location=f"{location}.receptor_residue",
            )
        )

    return ValidationResult(
        target_type="NormalizedInteraction",
        issues=tuple(issues),
    )


def validate_interaction_collection(
    interactions: Iterable[Any],
    *,
    max_items: int = DEFAULT_MAX_ROWS,
) -> ValidationResult:
    """Validate a normalized interaction collection."""

    values = list(interactions)
    results = [
        validate_normalized_interaction(
            item,
            location=f"interactions[{index}]",
        )
        for index, item in enumerate(
            values[: max(0, int(max_items))]
        )
    ]

    issues = [
        issue
        for result in results
        for issue in result.issues
    ]
    identifiers = [
        item.id
        for item in values
        if isinstance(item, NormalizedInteraction) and item.id
    ]
    duplicates = [
        identifier
        for identifier, count in Counter(identifiers).items()
        if count > 1
    ]
    if duplicates:
        issues.append(
            validation_issue(
                "duplicate_interaction_ids",
                "Duplicate normalized interaction identifiers found.",
                location="interactions",
                actual=duplicates,
            )
        )

    return ValidationResult(
        target_type="interaction_collection",
        issues=tuple(issues),
        metadata={"count": len(values)},
    )


# 26.5. Table validation
# -----------------------------------------------------------------------------

def validate_report_table(
    table: Any,
    *,
    location: str = "table",
) -> ValidationResult:
    """Validate one internal report table."""

    issues: List[ValidationIssue] = []
    if not isinstance(table, ReportTable):
        issues.append(
            validation_issue(
                "invalid_table_type",
                "Table must be ReportTable.",
                location=location,
                expected="ReportTable",
                actual=type(table).__name__,
            )
        )
        return ValidationResult(
            target_type=type(table).__name__,
            issues=tuple(issues),
        )

    column_keys = [column.key for column in table.columns]
    duplicates = [
        key
        for key, count in Counter(column_keys).items()
        if count > 1
    ]
    if duplicates:
        issues.append(
            validation_issue(
                "duplicate_table_columns",
                "Table contains duplicate column keys.",
                location=f"{location}.columns",
                actual=duplicates,
            )
        )

    if table.total_rows is not None and table.total_rows < len(table.rows):
        issues.append(
            validation_issue(
                "invalid_total_rows",
                "Table total_rows is smaller than visible rows.",
                location=f"{location}.total_rows",
                expected=f">= {len(table.rows)}",
                actual=table.total_rows,
            )
        )

    known = set(column_keys)
    for index, row in enumerate(table.rows):
        unknown = set(row) - known
        if unknown:
            issues.append(
                validation_issue(
                    "unknown_table_cells",
                    "Row contains keys without column definitions.",
                    severity=Severity.WARNING,
                    location=f"{location}.rows[{index}]",
                    actual=sorted(unknown),
                )
            )

    return ValidationResult(
        target_type="ReportTable",
        issues=tuple(issues),
        metadata={
            "columns": len(table.columns),
            "rows": len(table.rows),
        },
    )


# 26.6. Section validation
# -----------------------------------------------------------------------------

def validate_report_section(
    section: Any,
    *,
    config: Optional[ReportConfig] = None,
    location: str = "section",
) -> ValidationResult:
    """Validate one structured report section."""

    issues: List[ValidationIssue] = []
    if not isinstance(section, ReportSection):
        issues.append(
            validation_issue(
                "invalid_section_type",
                "Section must be ReportSection.",
                location=location,
                expected="ReportSection",
                actual=type(section).__name__,
            )
        )
        return ValidationResult(
            target_type=type(section).__name__,
            issues=tuple(issues),
        )

    if not section.title:
        issues.append(
            validation_issue(
                "missing_section_title",
                "Report section title is empty.",
                location=f"{location}.title",
            )
        )

    if section.empty != (not section.visible_blocks):
        issues.append(
            validation_issue(
                "section_empty_flag_mismatch",
                "Section empty flag does not match visible blocks.",
                severity=Severity.WARNING,
                location=f"{location}.empty",
                expected=not section.visible_blocks,
                actual=section.empty,
            )
        )

    if config is not None:
        expected_enabled = config.is_section_enabled(section.id)
        if section.enabled != expected_enabled:
            issues.append(
                validation_issue(
                    "section_enabled_mismatch",
                    "Section enabled state differs from configuration.",
                    severity=Severity.WARNING,
                    location=f"{location}.enabled",
                    expected=expected_enabled,
                    actual=section.enabled,
                )
            )

    for index, block in enumerate(section.blocks):
        block_location = f"{location}.blocks[{index}]"
        if not isinstance(block, ReportBlock):
            issues.append(
                validation_issue(
                    "invalid_block_type",
                    "Section block must be ReportBlock.",
                    location=block_location,
                    expected="ReportBlock",
                    actual=type(block).__name__,
                )
            )
            continue
        if block.kind is ReportBlockKind.TABLE:
            try:
                table = table_from_block(
                    block,
                    config=(
                        config.tables
                        if config is not None
                        else DEFAULT_TABLE_CONFIG
                    ),
                )
            except ReportError as error:
                issues.append(
                    validation_issue(
                        "invalid_table_block",
                        str(error),
                        location=block_location,
                    )
                )
            else:
                table_result = validate_report_table(
                    table,
                    location=block_location,
                )
                issues.extend(table_result.issues)

    return ValidationResult(
        target_type="ReportSection",
        issues=tuple(issues),
        metadata={
            "section_id": section.id.value,
            "block_count": len(section.blocks),
        },
    )


# 26.7. Document validation
# -----------------------------------------------------------------------------

def validate_report_document(
    report: Any,
    *,
    validate_tables: bool = True,
) -> ValidationResult:
    """Validate a complete structured report document."""

    issues: List[ValidationIssue] = []
    if not isinstance(report, ReportDocument):
        issues.append(
            validation_issue(
                "invalid_document_type",
                "Report must be ReportDocument.",
                location="report",
                expected="ReportDocument",
                actual=type(report).__name__,
            )
        )
        return ValidationResult(
            target_type=type(report).__name__,
            issues=tuple(issues),
        )

    config_result = validate_report_config(report.config)
    issues.extend(config_result.issues)

    if report.schema_name != REPORT_SCHEMA_NAME:
        issues.append(
            validation_issue(
                "schema_name_mismatch",
                "Unexpected report schema name.",
                location="report.schema_name",
                expected=REPORT_SCHEMA_NAME,
                actual=report.schema_name,
            )
        )
    if report.schema_version != REPORT_SCHEMA_VERSION:
        issues.append(
            validation_issue(
                "schema_version_mismatch",
                "Unexpected report schema version.",
                severity=Severity.WARNING,
                location="report.schema_version",
                expected=REPORT_SCHEMA_VERSION,
                actual=report.schema_version,
            )
        )
    if not report.generated_at:
        issues.append(
            validation_issue(
                "missing_generated_at",
                "Report generation timestamp is missing.",
                location="report.generated_at",
            )
        )
    if not report.title:
        issues.append(
            validation_issue(
                "missing_report_title",
                "Report title is empty.",
                severity=Severity.WARNING,
                location="report.title",
            )
        )

    section_ids = [section.id.value for section in report.sections]
    duplicates = [
        section_id
        for section_id, count in Counter(section_ids).items()
        if count > 1
    ]
    if duplicates:
        issues.append(
            validation_issue(
                "duplicate_sections",
                "Report contains duplicate section identifiers.",
                location="report.sections",
                actual=duplicates,
            )
        )

    expected_order = {
        section.value: index
        for index, section in enumerate(report.config.section_order)
    }
    positions = [
        expected_order.get(section.id.value, len(expected_order))
        for section in report.sections
    ]
    if positions != sorted(positions):
        issues.append(
            validation_issue(
                "section_order_mismatch",
                "Report sections are not in configured order.",
                severity=Severity.WARNING,
                location="report.sections",
            )
        )

    for index, section in enumerate(report.sections):
        section_result = validate_report_section(
            section,
            config=report.config,
            location=f"report.sections[{index}]",
        )
        if not validate_tables:
            issues.extend(
                issue
                for issue in section_result.issues
                if "table" not in issue.code
            )
        else:
            issues.extend(section_result.issues)

    return ValidationResult(
        target_type="ReportDocument",
        issues=tuple(issues),
        metadata={
            "section_count": len(report.sections),
            "visible_section_count": len(report.visible_sections),
        },
    )


# 26.8. JSON and file validation
# -----------------------------------------------------------------------------

def validate_report_json_data(
    value: Any,
) -> ValidationResult:
    """Validate a report JSON mapping."""

    issues: List[ValidationIssue] = []
    if not isinstance(value, Mapping):
        issues.append(
            validation_issue(
                "invalid_json_root",
                "Report JSON root must be an object.",
                location="$",
                expected="mapping",
                actual=type(value).__name__,
            )
        )
        return ValidationResult(
            target_type=type(value).__name__,
            issues=tuple(issues),
        )

    for key in REQUIRED_REPORT_KEYS:
        if key not in value:
            issues.append(
                validation_issue(
                    "missing_required_key",
                    f"Required report key is missing: {key}.",
                    location=f"$.{key}",
                    expected="present",
                    actual="missing",
                )
            )

    if (
        KEY_SCHEMA_NAME in value
        and value[KEY_SCHEMA_NAME] != REPORT_SCHEMA_NAME
    ):
        issues.append(
            validation_issue(
                "json_schema_name_mismatch",
                "JSON schema name does not match.",
                location=f"$.{KEY_SCHEMA_NAME}",
                expected=REPORT_SCHEMA_NAME,
                actual=value[KEY_SCHEMA_NAME],
            )
        )

    sections = value.get(KEY_SECTIONS)
    if sections is not None and not isinstance(sections, list):
        issues.append(
            validation_issue(
                "invalid_json_sections",
                "JSON sections must be an array.",
                location=f"$.{KEY_SECTIONS}",
                expected="list",
                actual=type(sections).__name__,
            )
        )
    elif isinstance(sections, list):
        identifiers: List[str] = []
        for index, section in enumerate(sections):
            if not isinstance(section, Mapping):
                issues.append(
                    validation_issue(
                        "invalid_json_section",
                        "JSON section must be an object.",
                        location=f"$.sections[{index}]",
                        expected="mapping",
                        actual=type(section).__name__,
                    )
                )
                continue
            section_id = section.get(KEY_ID)
            if not section_id:
                issues.append(
                    validation_issue(
                        "missing_json_section_id",
                        "JSON section identifier is missing.",
                        location=f"$.sections[{index}].id",
                    )
                )
            else:
                identifiers.append(str(section_id))

        duplicates = [
            identifier
            for identifier, count in Counter(identifiers).items()
            if count > 1
        ]
        if duplicates:
            issues.append(
                validation_issue(
                    "duplicate_json_sections",
                    "JSON report contains duplicate sections.",
                    location="$.sections",
                    actual=duplicates,
                )
            )

    return ValidationResult(
        target_type="report_json",
        issues=tuple(issues),
        metadata={
            "keys": len(value),
            "sections": len(sections) if isinstance(sections, list) else 0,
        },
    )


def validate_report_file(
    path: PathLike,
    *,
    report_format: Any = None,
) -> ValidationResult:
    """Validate a report file according to its format."""

    file_path = Path(path)
    issues: List[ValidationIssue] = []
    if not file_path.exists():
        issues.append(
            validation_issue(
                "missing_report_file",
                "Report file does not exist.",
                location=str(file_path),
            )
        )
        return ValidationResult(
            target_type="report_file",
            issues=tuple(issues),
        )
    if not file_path.is_file():
        issues.append(
            validation_issue(
                "invalid_report_file",
                "Report path is not a regular file.",
                location=str(file_path),
            )
        )
        return ValidationResult(
            target_type="report_file",
            issues=tuple(issues),
        )

    member = (
        normalize_report_format(report_format)
        if report_format is not None
        else infer_report_format_from_path(file_path)
    )

    try:
        content = file_path.read_text(
            encoding=DEFAULT_ENCODING,
        )
    except (OSError, UnicodeError) as error:
        issues.append(
            validation_issue(
                "unreadable_report_file",
                "Report file could not be read.",
                location=str(file_path),
                actual=_exception_message(error),
            )
        )
        return ValidationResult(
            target_type="report_file",
            issues=tuple(issues),
        )

    if not content.strip():
        issues.append(
            validation_issue(
                "empty_report_file",
                "Report file is empty.",
                location=str(file_path),
            )
        )
    elif member is ReportFormat.JSON:
        try:
            parsed = parse_report_json(content)
        except ReportError as error:
            issues.append(
                validation_issue(
                    "invalid_report_json",
                    str(error),
                    location=str(file_path),
                )
            )
        else:
            issues.extend(
                validate_report_json_data(parsed).issues
            )
    elif member is ReportFormat.HTML:
        lowered = content.lower()
        if "<html" not in lowered and "<body" not in lowered:
            issues.append(
                validation_issue(
                    "invalid_report_html",
                    "HTML report has no html or body element.",
                    location=str(file_path),
                )
            )
    elif member is ReportFormat.MARKDOWN:
        if not any(
            line.startswith("#")
            for line in content.splitlines()
        ):
            issues.append(
                validation_issue(
                    "markdown_heading_missing",
                    "Markdown report has no heading.",
                    severity=Severity.WARNING,
                    location=str(file_path),
                )
            )

    return ValidationResult(
        target_type="report_file",
        issues=tuple(issues),
        metadata={
            "path": str(file_path),
            "format": member.value,
            "size_bytes": file_path.stat().st_size,
        },
    )


# 26.9. Unified validation
# -----------------------------------------------------------------------------

def validate_report(
    value: Any,
    *,
    config: Optional[ReportConfig] = None,
    strict: bool = False,
) -> ValidationResult:
    """Validate a report-related object by type."""

    if isinstance(value, ReportDocument):
        result = validate_report_document(value)
    elif isinstance(value, ReportSection):
        result = validate_report_section(
            value,
            config=config,
        )
    elif isinstance(value, ReportTable):
        result = validate_report_table(value)
    elif isinstance(value, NormalizedInteraction):
        result = validate_normalized_interaction(value)
    elif isinstance(value, ReportConfig):
        result = validate_report_config(value)
    elif isinstance(value, Mapping):
        result = validate_report_json_data(value)
    elif isinstance(value, (str, os.PathLike, Path)) and Path(value).exists():
        result = validate_report_file(value)
    else:
        active_config = config or DEFAULT_REPORT_CONFIG
        document = build_report_document(
            value,
            config=active_config,
        )
        result = validate_report_document(document)

    if strict:
        result.raise_for_errors()
    return result


def validation_rows(
    result: ValidationResult,
) -> ReportRows:
    """Return validation issues as table rows."""

    return [
        {
            KEY_RANK: index,
            "severity": issue.severity.value,
            "code": issue.code,
            "message": issue.message,
            "location": issue.location,
            "expected": issue.expected,
            "actual": issue.actual,
        }
        for index, issue in enumerate(result.issues, start=1)
    ]


# 26.10. Public validation interface
# -----------------------------------------------------------------------------

_SECTION_26_PUBLIC_NAMES: Final[Tuple[str, ...]] = (
    "ValidationIssue",
    "ValidationResult",
    "validation_issue",
    "combine_validation_results",
    "validate_report_config",
    "validate_normalized_interaction",
    "validate_interaction_collection",
    "validate_report_table",
    "validate_report_section",
    "validate_report_document",
    "validate_report_json_data",
    "validate_report_file",
    "validate_report",
    "validation_rows",
)

_register_public_names(_SECTION_26_PUBLIC_NAMES)

# =============================================================================
# End of Section 26
# =============================================================================


# =============================================================================
# Section 27 — Public convenience interface
# =============================================================================

# 27.1. Convenience configuration
# -----------------------------------------------------------------------------

def configure_report(
    *,
    base: ReportConfig = DEFAULT_REPORT_CONFIG,
    report_format: Any = None,
    detail: Any = None,
    error_mode: Any = None,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    description: Optional[str] = None,
    include_table_of_contents: Optional[bool] = None,
    include_empty_sections: Optional[bool] = None,
    overwrite: Optional[bool] = None,
    atomic_write: Optional[bool] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> ReportConfig:
    """Return a report configuration with common overrides."""

    if not isinstance(base, ReportConfig):
        raise ReportConfigurationError("base must be ReportConfig.")

    rendering_changes: Dict[str, Any] = {}
    if report_format is not None:
        rendering_changes["format"] = normalize_report_format(
            report_format
        )
    if detail is not None:
        rendering_changes["detail"] = _coerce_enum(
            ReportDetail,
            detail,
            "detail",
        )
    if include_table_of_contents is not None:
        rendering_changes["include_table_of_contents"] = bool(
            include_table_of_contents
        )
    if include_empty_sections is not None:
        rendering_changes["include_empty_sections"] = bool(
            include_empty_sections
        )

    writing_changes: Dict[str, Any] = {}
    if overwrite is not None:
        writing_changes["overwrite"] = bool(overwrite)
    if atomic_write is not None:
        writing_changes["atomic"] = bool(atomic_write)

    error_changes: Dict[str, Any] = {}
    if error_mode is not None:
        error_changes["mode"] = _coerce_enum(
            ErrorMode,
            error_mode,
            "error_mode",
        )

    changes: Dict[str, Any] = {
        "rendering": (
            base.rendering.with_updates(**rendering_changes)
            if rendering_changes
            else base.rendering
        ),
        "writing": (
            base.writing.with_updates(**writing_changes)
            if writing_changes
            else base.writing
        ),
        "errors": (
            base.errors.with_updates(**error_changes)
            if error_changes
            else base.errors
        ),
    }
    if title is not None:
        changes["title"] = title
    if subtitle is not None:
        changes["subtitle"] = subtitle
    if description is not None:
        changes["description"] = description
    if metadata is not None:
        changes["metadata"] = {
            **dict(base.metadata),
            **dict(metadata),
        }
    return base.with_updates(**changes)


# 27.2. High-level report creation
# -----------------------------------------------------------------------------

def create_report(
    value: Any,
    *,
    config: ReportConfig = DEFAULT_REPORT_CONFIG,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    description: Optional[str] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    validate: bool = False,
    strict_validation: bool = False,
) -> ReportDocument:
    """Create a structured report document."""

    document = build_report_document(
        value,
        config=config,
        title=title,
        subtitle=subtitle,
        description=description,
        metadata=metadata,
    )
    if validate:
        validate_report_document(document).raise_for_errors() if (
            strict_validation
        ) else validate_report_document(document)
    return document


def create_pose_report(
    pose: Any,
    *,
    config: ReportConfig = DEFAULT_REPORT_CONFIG,
    **kwargs: Any,
) -> ReportDocument:
    """Create a report for one pose."""

    return create_report(
        pose,
        config=config,
        **kwargs,
    )


def create_multipose_report(
    poses: Any,
    *,
    config: ReportConfig = DEFAULT_REPORT_CONFIG,
    **kwargs: Any,
) -> ReportDocument:
    """Create a report for multiple poses."""

    pose_values = list(iter_object_collection(poses))
    if not pose_values:
        raise ReportInputError("No poses were provided.")
    return create_report(
        pose_values,
        config=config,
        **kwargs,
    )


# 27.3. High-level rendering
# -----------------------------------------------------------------------------

def render_dock_report(
    value: Any,
    *,
    report_format: Any = None,
    config: ReportConfig = DEFAULT_REPORT_CONFIG,
    validate: bool = False,
) -> str:
    """Build and render a DockAnalyzer report."""

    document = (
        value
        if isinstance(value, ReportDocument)
        else create_report(
            value,
            config=config,
            validate=validate,
        )
    )
    member = (
        config.rendering.format
        if report_format is None
        else normalize_report_format(report_format)
    )
    return render_report(
        document,
        report_format=member,
        config=config,
    )


def render_dock_report_formats(
    value: Any,
    *,
    formats: Iterable[Any] = SUPPORTED_REPORT_FORMATS,
    config: ReportConfig = DEFAULT_REPORT_CONFIG,
) -> Dict[str, str]:
    """Render one report in multiple formats."""

    document = (
        value
        if isinstance(value, ReportDocument)
        else create_report(value, config=config)
    )
    output: Dict[str, str] = {}
    for format_value in formats:
        member = normalize_report_format(format_value)
        output[member.value] = render_report(
            document,
            report_format=member,
            config=config,
        )
    return output


generate_report = render_dock_report
generate_report_formats = render_dock_report_formats


# 27.4. High-level writing and export
# -----------------------------------------------------------------------------

def save_dock_report(
    value: Any,
    path: PathLike,
    *,
    report_format: Any = None,
    config: ReportConfig = DEFAULT_REPORT_CONFIG,
    write_config: Optional[WriteConfig] = None,
    ensure_suffix: bool = False,
    validate: bool = True,
    calculate_checksum: bool = False,
) -> WriteResult:
    """Create, validate and save one report."""

    document = (
        value
        if isinstance(value, ReportDocument)
        else create_report(value, config=config)
    )
    if validate:
        validate_report_document(document).raise_for_errors()
    return write_report(
        document,
        path,
        report_format=report_format,
        config=config,
        write_config=write_config,
        ensure_suffix=ensure_suffix,
        calculate_checksum=calculate_checksum,
    )


def save_dock_report_formats(
    value: Any,
    directory: PathLike,
    *,
    formats: Iterable[Any] = SUPPORTED_REPORT_FORMATS,
    basename: Any = DEFAULT_REPORT_BASENAME,
    config: ReportConfig = DEFAULT_REPORT_CONFIG,
    write_config: Optional[WriteConfig] = None,
    validate: bool = True,
    calculate_checksum: bool = False,
) -> Dict[str, WriteResult]:
    """Create and save a report in multiple formats."""

    document = (
        value
        if isinstance(value, ReportDocument)
        else create_report(value, config=config)
    )
    if validate:
        validate_report_document(document).raise_for_errors()
    return write_report_formats(
        document,
        directory,
        formats=formats,
        basename=basename,
        config=config,
        write_config=write_config,
        calculate_checksum=calculate_checksum,
    )


def export_dock_report(
    value: Any,
    path: PathLike,
    *,
    format_name: Any = None,
    mode: str = "auto",
    config: ReportConfig = DEFAULT_REPORT_CONFIG,
    validate: bool = True,
    export_module: Any = None,
    **kwargs: Any,
) -> ReportExportResult:
    """Create and export a report through ``export.py``."""

    document = (
        value
        if isinstance(value, ReportDocument)
        else create_report(value, config=config)
    )
    if validate:
        validate_report_document(document).raise_for_errors()
    return export_report_with_export_module(
        document,
        path,
        format_name=format_name,
        mode=mode,
        config=config,
        export_module=export_module,
        **kwargs,
    )


save_report = save_dock_report
save_report_formats = save_dock_report_formats
export_report = export_dock_report


# 27.5. Facade object
# -----------------------------------------------------------------------------

class DockReport:
    """Convenience facade with lazy report construction."""

    def __init__(
        self,
        value: Any,
        *,
        config: ReportConfig = DEFAULT_REPORT_CONFIG,
        title: Optional[str] = None,
        subtitle: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.value = value
        self.config = config
        self.title = title
        self.subtitle = subtitle
        self.description = description
        self.metadata = dict(metadata or {})
        self._document: Optional[ReportDocument] = None

    @property
    def document(self) -> ReportDocument:
        """Return the cached structured report."""

        if self._document is None:
            self._document = create_report(
                self.value,
                config=self.config,
                title=self.title,
                subtitle=self.subtitle,
                description=self.description,
                metadata=self.metadata,
            )
        return self._document

    def rebuild(self) -> ReportDocument:
        """Rebuild and return the structured report."""

        self._document = None
        return self.document

    def render(
        self,
        report_format: Any = None,
    ) -> str:
        """Render the report in one format."""

        return render_dock_report(
            self.document,
            report_format=report_format,
            config=self.config,
        )

    def render_all(
        self,
        formats: Iterable[Any] = SUPPORTED_REPORT_FORMATS,
    ) -> Dict[str, str]:
        """Render the report in multiple formats."""

        return render_dock_report_formats(
            self.document,
            formats=formats,
            config=self.config,
        )

    def save(
        self,
        path: PathLike,
        *,
        report_format: Any = None,
        write_config: Optional[WriteConfig] = None,
        ensure_suffix: bool = False,
        calculate_checksum: bool = False,
    ) -> WriteResult:
        """Save the report."""

        return save_dock_report(
            self.document,
            path,
            report_format=report_format,
            config=self.config,
            write_config=write_config,
            ensure_suffix=ensure_suffix,
            calculate_checksum=calculate_checksum,
        )

    def save_all(
        self,
        directory: PathLike,
        *,
        formats: Iterable[Any] = SUPPORTED_REPORT_FORMATS,
        basename: Any = DEFAULT_REPORT_BASENAME,
        write_config: Optional[WriteConfig] = None,
        calculate_checksum: bool = False,
    ) -> Dict[str, WriteResult]:
        """Save the report in multiple formats."""

        return save_dock_report_formats(
            self.document,
            directory,
            formats=formats,
            basename=basename,
            config=self.config,
            write_config=write_config,
            calculate_checksum=calculate_checksum,
        )

    def export(
        self,
        path: PathLike,
        *,
        format_name: Any = None,
        mode: str = "auto",
        export_module: Any = None,
        **kwargs: Any,
    ) -> ReportExportResult:
        """Export through ``export.py``."""

        return export_dock_report(
            self.document,
            path,
            format_name=format_name,
            mode=mode,
            config=self.config,
            export_module=export_module,
            **kwargs,
        )

    def validate(self, *, strict: bool = False) -> ValidationResult:
        """Validate the structured report."""

        return validate_report(
            self.document,
            strict=strict,
        )

    def diagnostics(self) -> DiagnosticCollector:
        """Return attached report diagnostics."""

        return report_diagnostics(
            self.document,
            include_tracebacks=self.config.errors.include_tracebacks,
            max_items=self.config.errors.max_messages,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return the structured report mapping."""

        return self.document.to_dict()

    def to_json(self) -> str:
        """Return the rendered JSON report."""

        return render_report_json(
            self.document,
            config=self.config,
        )

    def __str__(self) -> str:
        return self.render(self.config.rendering.format)


# 27.6. Compact summaries and aliases
# -----------------------------------------------------------------------------

def report_summary(
    value: Any,
    *,
    config: ReportConfig = DEFAULT_REPORT_CONFIG,
) -> Dict[str, Any]:
    """Return a compact programmatic report summary."""

    document = (
        value
        if isinstance(value, ReportDocument)
        else create_report(value, config=config)
    )
    overview_section = document.get_section(
        ReportSectionID.OVERVIEW
    )
    interaction_section = document.get_section(
        ReportSectionID.INTERACTIONS
    )
    scoring_section = document.get_section(
        ReportSectionID.SCORING
    )
    return {
        KEY_TITLE: document.title,
        KEY_GENERATED_AT: document.generated_at,
        "section_count": len(document.visible_sections),
        "has_overview": overview_section is not None,
        "has_interactions": interaction_section is not None,
        "has_scoring": scoring_section is not None,
        "warning_count": len(document.warnings),
        "error_count": len(document.errors),
        "valid": validate_report_document(document).valid,
    }


def report_pose(
    pose: Any,
    *,
    report_format: Any = DEFAULT_REPORT_FORMAT,
    config: ReportConfig = DEFAULT_REPORT_CONFIG,
) -> str:
    """Create and render one pose report."""

    return render_dock_report(
        pose,
        report_format=report_format,
        config=config,
    )


def report_multiple_poses(
    poses: Any,
    *,
    report_format: Any = DEFAULT_REPORT_FORMAT,
    config: ReportConfig = DEFAULT_REPORT_CONFIG,
) -> str:
    """Create and render a multipose report."""

    document = create_multipose_report(
        poses,
        config=config,
    )
    return render_dock_report(
        document,
        report_format=report_format,
        config=config,
    )


# 27.7. Public convenience interface
# -----------------------------------------------------------------------------

_SECTION_27_PUBLIC_NAMES: Final[Tuple[str, ...]] = (
    "configure_report",
    "create_report",
    "create_pose_report",
    "create_multipose_report",
    "render_dock_report",
    "render_dock_report_formats",
    "generate_report",
    "generate_report_formats",
    "save_dock_report",
    "save_dock_report_formats",
    "export_dock_report",
    "save_report",
    "save_report_formats",
    "export_report",
    "DockReport",
    "report_summary",
    "report_pose",
    "report_multiple_poses",
)

_register_public_names(_SECTION_27_PUBLIC_NAMES)

# =============================================================================
# End of Section 27
# =============================================================================

# =============================================================================
# Section 28 — ChimeraX compatibility
# =============================================================================

# 28.1. Compatibility constants
# -----------------------------------------------------------------------------

CHIMERAX_MODULE_NAMES: Final[Tuple[str, ...]] = (
    "chimerax",
    "chimerax.core",
    "chimerax.core.commands",
    "chimerax.atomic",
)

CHIMERAX_OBJECT_MODULE_PREFIXES: Final[Tuple[str, ...]] = (
    "chimerax.",
    "bundles.",
)

CHIMERAX_MODEL_SPEC_PREFIX: Final[str] = "#"
CHIMERAX_CHAIN_SPEC_PREFIX: Final[str] = "/"
CHIMERAX_RESIDUE_SPEC_PREFIX: Final[str] = ":"
CHIMERAX_ATOM_SPEC_PREFIX: Final[str] = "@"
CHIMERAX_SPEC_SEPARATOR: Final[str] = ""
CHIMERAX_COMMAND_SEPARATOR: Final[str] = "; "
CHIMERAX_SELECT_COMMAND: Final[str] = "select"
CHIMERAX_DISTANCE_COMMAND: Final[str] = "distance"
CHIMERAX_SHOW_COMMAND: Final[str] = "show"
CHIMERAX_HIDE_COMMAND: Final[str] = "hide"
CHIMERAX_COLOR_COMMAND: Final[str] = "color"
CHIMERAX_STYLE_COMMAND: Final[str] = "style"

CHIMERAX_REPORT_METADATA_KEY: Final[str] = "chimerax"
CHIMERAX_COMMANDS_METADATA_KEY: Final[str] = "chimerax_commands"

CHIMERAX_SAFE_SPEC_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^[#/:@A-Za-z0-9_.+\-*,'\"()\[\]]+$"
)

CHIMERAX_COMMAND_TYPES: Final[Tuple[str, ...]] = (
    "selection",
    "visualization",
    "distance",
    "show",
    "hide",
    "style",
    "color",
    "custom",
)


# 28.2. ChimeraX compatibility records
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class ChimeraXCapabilities:
    """Detected ChimeraX runtime capabilities."""

    available: bool
    version: str = ""
    session_available: bool = False
    commands_available: bool = False
    atomic_available: bool = False
    modules: Tuple[str, ...] = ()
    missing_modules: Tuple[str, ...] = ()
    object_types: Tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(
        default_factory=lambda: _EMPTY_METADATA
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "available", bool(self.available))
        object.__setattr__(
            self,
            "version",
            single_line_text(self.version, ""),
        )
        object.__setattr__(
            self,
            "session_available",
            bool(self.session_available),
        )
        object.__setattr__(
            self,
            "commands_available",
            bool(self.commands_available),
        )
        object.__setattr__(
            self,
            "atomic_available",
            bool(self.atomic_available),
        )
        for name in (
            "modules",
            "missing_modules",
            "object_types",
        ):
            object.__setattr__(
                self,
                name,
                _freeze_config_strings(getattr(self, name)),
            )
        object.__setattr__(
            self,
            "metadata",
            _freeze_config_mapping(self.metadata),
        )

    def supports(self, capability: str) -> bool:
        """Return whether a named ChimeraX capability is available."""

        token = normalize_field_name(capability)
        mapping = {
            "available": self.available,
            "session": self.session_available,
            "session_available": self.session_available,
            "commands": self.commands_available,
            "commands_available": self.commands_available,
            "atomic": self.atomic_available,
            "atomic_available": self.atomic_available,
        }
        return bool(mapping.get(token, False))

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain capability record."""

        return {
            "available": self.available,
            "version": self.version,
            "session_available": self.session_available,
            "commands_available": self.commands_available,
            "atomic_available": self.atomic_available,
            "modules": list(self.modules),
            "missing_modules": list(self.missing_modules),
            "object_types": list(self.object_types),
            KEY_METADATA: dict(self.metadata),
        }


@dataclass(frozen=True)
class ChimeraXCommand:
    """One generated ChimeraX command."""

    command: str
    kind: str = "custom"
    label: str = ""
    atomspecs: Tuple[str, ...] = ()
    enabled: bool = True
    metadata: Mapping[str, Any] = field(
        default_factory=lambda: _EMPTY_METADATA
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "command",
            safe_string(self.command, ""),
        )
        kind = normalize_field_name(self.kind)
        if kind not in CHIMERAX_COMMAND_TYPES:
            kind = "custom"
        object.__setattr__(self, "kind", kind)
        object.__setattr__(
            self,
            "label",
            single_line_text(self.label, ""),
        )
        object.__setattr__(
            self,
            "atomspecs",
            _freeze_config_strings(self.atomspecs),
        )
        object.__setattr__(self, "enabled", bool(self.enabled))
        object.__setattr__(
            self,
            "metadata",
            _freeze_config_mapping(self.metadata),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain command record."""

        return {
            "command": self.command,
            "kind": self.kind,
            "label": self.label,
            "atomspecs": list(self.atomspecs),
            "enabled": self.enabled,
            KEY_METADATA: dict(self.metadata),
        }


@dataclass(frozen=True)
class ChimeraXSessionSummary:
    """Non-invasive summary of a ChimeraX session."""

    available: bool
    version: str = ""
    model_count: int = 0
    model_specs: Tuple[str, ...] = ()
    model_names: Tuple[str, ...] = ()
    selected_atom_count: Optional[int] = None
    selected_residue_count: Optional[int] = None
    attributes: Mapping[str, Any] = field(
        default_factory=lambda: _EMPTY_METADATA
    )
    warnings: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "available", bool(self.available))
        object.__setattr__(
            self,
            "version",
            single_line_text(self.version, ""),
        )
        object.__setattr__(
            self,
            "model_count",
            max(0, to_safe_int(self.model_count, 0)),
        )
        for name in ("model_specs", "model_names", "warnings"):
            object.__setattr__(
                self,
                name,
                _freeze_config_strings(
                    getattr(self, name),
                    unique=(name != "warnings"),
                ),
            )
        for name in (
            "selected_atom_count",
            "selected_residue_count",
        ):
            value = getattr(self, name)
            object.__setattr__(
                self,
                name,
                None if value is None else max(0, to_safe_int(value, 0)),
            )
        object.__setattr__(
            self,
            "attributes",
            _freeze_config_mapping(self.attributes),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain session summary."""

        return {
            "available": self.available,
            "version": self.version,
            "model_count": self.model_count,
            "model_specs": list(self.model_specs),
            "model_names": list(self.model_names),
            "selected_atom_count": self.selected_atom_count,
            "selected_residue_count": self.selected_residue_count,
            "attributes": dict(self.attributes),
            KEY_WARNINGS: list(self.warnings),
        }


@dataclass(frozen=True)
class ChimeraXReportData:
    """ChimeraX-specific data generated for a report."""

    capabilities: ChimeraXCapabilities
    session: ChimeraXSessionSummary
    model_specs: Tuple[str, ...] = ()
    atom_specs: Tuple[str, ...] = ()
    residue_specs: Tuple[str, ...] = ()
    selection_commands: Tuple[ChimeraXCommand, ...] = ()
    visualization_commands: Tuple[ChimeraXCommand, ...] = ()
    interaction_specs: Tuple[Mapping[str, Any], ...] = ()
    metadata: Mapping[str, Any] = field(
        default_factory=lambda: _EMPTY_METADATA
    )
    warnings: Tuple[str, ...] = ()
    errors: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.capabilities, ChimeraXCapabilities):
            raise ReportConfigurationError(
                "capabilities must be ChimeraXCapabilities."
            )
        if not isinstance(self.session, ChimeraXSessionSummary):
            raise ReportConfigurationError(
                "session must be ChimeraXSessionSummary."
            )
        for name in (
            "model_specs",
            "atom_specs",
            "residue_specs",
            "warnings",
            "errors",
        ):
            object.__setattr__(
                self,
                name,
                _freeze_config_strings(
                    getattr(self, name),
                    unique=name not in {"warnings", "errors"},
                ),
            )
        object.__setattr__(
            self,
            "selection_commands",
            tuple(self.selection_commands),
        )
        object.__setattr__(
            self,
            "visualization_commands",
            tuple(self.visualization_commands),
        )
        object.__setattr__(
            self,
            "interaction_specs",
            tuple(
                MappingProxyType(dict(item))
                for item in self.interaction_specs
            ),
        )
        object.__setattr__(
            self,
            "metadata",
            _freeze_config_mapping(self.metadata),
        )

    @property
    def commands(self) -> Tuple[ChimeraXCommand, ...]:
        """Return all generated commands."""

        return (
            self.selection_commands
            + self.visualization_commands
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain ChimeraX report record."""

        return {
            "capabilities": self.capabilities.to_dict(),
            "session": self.session.to_dict(),
            "model_specs": list(self.model_specs),
            "atom_specs": list(self.atom_specs),
            "residue_specs": list(self.residue_specs),
            "selection_commands": [
                item.to_dict() for item in self.selection_commands
            ],
            "visualization_commands": [
                item.to_dict() for item in self.visualization_commands
            ],
            "interaction_specs": [
                dict(item) for item in self.interaction_specs
            ],
            KEY_METADATA: dict(self.metadata),
            KEY_WARNINGS: list(self.warnings),
            KEY_ERRORS: list(self.errors),
        }


# 28.3. ChimeraX module and runtime detection
# -----------------------------------------------------------------------------

def load_chimerax_module(
    module_name: str = "chimerax",
    *,
    required: bool = False,
) -> Any:
    """Import a ChimeraX module without creating a hard dependency."""

    try:
        return __import__(module_name, fromlist=["*"])
    except ImportError as error:
        if required:
            raise ChimeraXReportError(
                f"ChimeraX module is unavailable: {module_name}.",
                context={"module": module_name},
                cause=error,
            ) from error
        return None


def chimerax_available() -> bool:
    """Return whether a ChimeraX runtime can be imported."""

    return load_chimerax_module("chimerax", required=False) is not None


def detect_chimerax_capabilities(
    *,
    session: Any = None,
) -> ChimeraXCapabilities:
    """Detect optional ChimeraX modules and command support."""

    loaded: List[str] = []
    missing: List[str] = []
    modules: Dict[str, Any] = {}

    for module_name in CHIMERAX_MODULE_NAMES:
        module = load_chimerax_module(
            module_name,
            required=False,
        )
        if module is None:
            missing.append(module_name)
        else:
            loaded.append(module_name)
            modules[module_name] = module

    root = modules.get("chimerax")
    commands_module = modules.get("chimerax.core.commands")
    atomic_module = modules.get("chimerax.atomic")
    command_runner = getattr(commands_module, "run", None)

    version = ""
    if root is not None:
        version = safe_string(
            getattr(root, "__version__", ""),
            "",
        )
    if not version:
        version = safe_string(chimerax_version(""), "")

    object_types: List[str] = []
    if atomic_module is not None:
        for name in (
            "AtomicStructure",
            "Structure",
            "Atom",
            "Atoms",
            "Residue",
            "Residues",
            "Chain",
            "Chains",
        ):
            if getattr(atomic_module, name, None) is not None:
                object_types.append(name)

    return ChimeraXCapabilities(
        available=bool(root or commands_module or atomic_module),
        version=version,
        session_available=session is not None,
        commands_available=callable(command_runner),
        atomic_available=atomic_module is not None,
        modules=tuple(loaded),
        missing_modules=tuple(missing),
        object_types=tuple(object_types),
        metadata={
            "command_runner": (
                getattr(command_runner, "__name__", "")
                if callable(command_runner)
                else ""
            ),
        },
    )


def is_chimerax_object(value: Any) -> bool:
    """Return whether an object appears to originate from ChimeraX."""

    if value is None or value is MISSING:
        return False
    module_name = safe_string(
        getattr(type(value), "__module__", ""),
        "",
    )
    if module_name.startswith(CHIMERAX_OBJECT_MODULE_PREFIXES):
        return True
    return any(
        hasattr(value, attribute)
        for attribute in (
            "atomspec",
            "id_string",
            "session",
            "structure",
            "scene_coord",
        )
    ) and (
        "chimerax" in module_name.lower()
        or hasattr(value, "atomspec")
    )


def is_chimerax_session(value: Any) -> bool:
    """Return whether an object resembles a ChimeraX session."""

    return (
        value is not None
        and hasattr(value, "models")
        and (
            hasattr(value, "logger")
            or hasattr(value, "selection")
            or "session" in type(value).__name__.lower()
        )
    )


def resolve_chimerax_session(
    value: Any = None,
    *,
    session: Any = None,
) -> Any:
    """Resolve a session from explicit input or related objects."""

    if session is not None:
        return session
    if is_chimerax_session(value):
        return value
    candidate = get_object_field(value, "session", None)
    if is_chimerax_session(candidate):
        return candidate

    for attribute in (
        "structure",
        "model",
        "receptor",
        "ligand",
    ):
        child = get_object_field(value, attribute, None)
        candidate = get_object_field(child, "session", None)
        if is_chimerax_session(candidate):
            return candidate
    return None


# 28.4. Generic ChimeraX object collections
# -----------------------------------------------------------------------------

def chimerax_session_models(
    session: Any,
) -> Tuple[Any, ...]:
    """Return models from a ChimeraX-like session."""

    if session is None:
        return ()
    models = get_object_field(session, "models", None)
    if models is None:
        return ()

    list_method = getattr(models, "list", None)
    if callable(list_method):
        try:
            return tuple(list_method())
        except Exception:
            pass

    return tuple(iter_object_collection(models))


def chimerax_selected_atoms(
    session: Any,
) -> Tuple[Any, ...]:
    """Return selected atoms when exposed by the session."""

    if session is None:
        return ()

    selection = get_object_field(session, "selection", None)
    for source in (selection, session):
        if source is None:
            continue
        for name in (
            "atoms",
            "selected_atoms",
            "items",
        ):
            value = get_object_field(source, name, MISSING)
            if value is MISSING:
                continue
            if callable(value):
                try:
                    value = value()
                except Exception:
                    continue
            result = tuple(iter_object_collection(value))
            if result:
                return result
    return ()


def chimerax_selected_residues(
    session: Any,
) -> Tuple[Any, ...]:
    """Return selected residues when exposed by the session."""

    if session is None:
        return ()

    selection = get_object_field(session, "selection", None)
    for source in (selection, session):
        if source is None:
            continue
        for name in (
            "residues",
            "selected_residues",
        ):
            value = get_object_field(source, name, MISSING)
            if value is MISSING:
                continue
            if callable(value):
                try:
                    value = value()
                except Exception:
                    continue
            result = tuple(iter_object_collection(value))
            if result:
                return result
    return ()


# 28.5. Atom-specification helpers
# -----------------------------------------------------------------------------

def sanitize_chimerax_spec(
    value: Any,
    *,
    strict: bool = False,
) -> str:
    """Normalize a ChimeraX atom specification."""

    spec = single_line_text(value, "")
    if not spec:
        return ""
    if CHIMERAX_SAFE_SPEC_PATTERN.match(spec):
        return spec
    if strict:
        raise ChimeraXReportError(
            "Unsafe or invalid ChimeraX atom specification.",
            context={"atomspec": spec},
        )
    return "".join(
        character
        for character in spec
        if character.isalnum()
        or character in "#/:@_.+-*,'\"()[]"
    )


def chimerax_model_spec(
    model: Any,
    *,
    attribute: str = "atomspec",
) -> str:
    """Return a ChimeraX model specification."""

    if model is None or model is MISSING:
        return ""

    explicit = get_object_field(model, attribute, None)
    if explicit:
        spec = sanitize_chimerax_spec(explicit)
        if spec.startswith(CHIMERAX_MODEL_SPEC_PREFIX):
            return spec

    id_string = get_first_object_field(
        model,
        ("id_string", "model_id_string"),
        None,
        skip_none=True,
    )
    if id_string is not None:
        text = single_line_text(id_string, "").lstrip("#")
        return sanitize_chimerax_spec(
            f"{CHIMERAX_MODEL_SPEC_PREFIX}{text}"
        )

    identifier = get_object_field(model, "id", None)
    if isinstance(identifier, (tuple, list)):
        components = [
            single_line_text(item, "")
            for item in identifier
        ]
        if all(components):
            return sanitize_chimerax_spec(
                CHIMERAX_MODEL_SPEC_PREFIX
                + ".".join(components)
            )
    if identifier is not None and not callable(identifier):
        text = single_line_text(identifier, "")
        if text:
            return sanitize_chimerax_spec(
                CHIMERAX_MODEL_SPEC_PREFIX
                + text.lstrip("#")
            )
    return ""


def chimerax_chain_spec(
    chain: Any,
    *,
    model: Any = None,
) -> str:
    """Return a ChimeraX chain specification."""

    if chain is None or chain is MISSING:
        return ""
    explicit = get_object_field(chain, "atomspec", None)
    if explicit:
        return sanitize_chimerax_spec(explicit)

    chain_id = get_first_object_field(
        chain,
        ("chain_id", "id", "name"),
        None,
        skip_none=True,
    )
    if chain_id is None and isinstance(chain, str):
        chain_id = chain
    chain_text = single_line_text(chain_id, "")
    if not chain_text:
        return ""

    model_value = model or get_first_object_field(
        chain,
        ("structure", "model"),
        None,
        skip_none=True,
    )
    return sanitize_chimerax_spec(
        chimerax_model_spec(model_value)
        + CHIMERAX_CHAIN_SPEC_PREFIX
        + chain_text
    )


def chimerax_residue_spec(
    residue: Any,
    *,
    model: Any = None,
) -> str:
    """Return a ChimeraX residue specification."""

    if residue is None or residue is MISSING:
        return ""
    explicit = get_object_field(residue, "atomspec", None)
    if explicit:
        return sanitize_chimerax_spec(explicit)

    structure = model or get_first_object_field(
        residue,
        ("structure", "model"),
        None,
        skip_none=True,
    )
    chain = get_first_object_field(
        residue,
        ("chain", "parent_chain"),
        None,
        skip_none=True,
    )
    chain_id = get_first_object_field(
        residue,
        ("chain_id", "chain_identifier"),
        None,
        skip_none=True,
    )
    if chain_id is None and chain is not None:
        chain_id = get_first_object_field(
            chain,
            ("chain_id", "id", "name"),
            None,
            skip_none=True,
        )

    number = get_first_object_field(
        residue,
        (
            "number",
            "residue_number",
            "resnum",
            "id",
        ),
        None,
        skip_none=True,
    )
    insertion = get_first_object_field(
        residue,
        (
            "insertion_code",
            "insert",
            "icode",
        ),
        "",
        skip_none=True,
    )

    prefix = chimerax_model_spec(structure)
    if chain_id is not None:
        prefix += (
            CHIMERAX_CHAIN_SPEC_PREFIX
            + single_line_text(chain_id, "")
        )

    number_text = single_line_text(number, "")
    insertion_text = single_line_text(insertion, "")
    if not number_text:
        return sanitize_chimerax_spec(prefix)
    return sanitize_chimerax_spec(
        prefix
        + CHIMERAX_RESIDUE_SPEC_PREFIX
        + number_text
        + insertion_text
    )


def chimerax_atom_spec(
    atom: Any,
    *,
    model: Any = None,
) -> str:
    """Return a ChimeraX atom specification."""

    if atom is None or atom is MISSING:
        return ""
    explicit = get_object_field(atom, "atomspec", None)
    if explicit:
        return sanitize_chimerax_spec(explicit)

    residue = get_first_object_field(
        atom,
        ("residue", "parent_residue"),
        None,
        skip_none=True,
    )
    structure = model or get_first_object_field(
        atom,
        ("structure", "model"),
        None,
        skip_none=True,
    )
    name = get_first_object_field(
        atom,
        ("name", "atom_name"),
        None,
        skip_none=True,
    )
    name_text = single_line_text(name, "")
    prefix = (
        chimerax_residue_spec(residue, model=structure)
        if residue is not None
        else chimerax_model_spec(structure)
    )
    if not name_text:
        return sanitize_chimerax_spec(prefix)
    return sanitize_chimerax_spec(
        prefix
        + CHIMERAX_ATOM_SPEC_PREFIX
        + name_text
    )


def chimerax_object_spec(
    value: Any,
    *,
    config: ChimeraXConfig = DEFAULT_CHIMERAX_CONFIG,
) -> str:
    """Return the best ChimeraX specification for an object."""

    if value is None or value is MISSING:
        return ""
    if isinstance(value, str):
        return sanitize_chimerax_spec(value)

    explicit = get_object_field(
        value,
        config.model_spec_attribute,
        None,
    )
    if explicit:
        return sanitize_chimerax_spec(explicit)

    class_name = type(value).__name__.lower()
    if "atom" in class_name and "atomic" not in class_name:
        return chimerax_atom_spec(value)
    if "residue" in class_name:
        return chimerax_residue_spec(value)
    if "chain" in class_name:
        return chimerax_chain_spec(value)

    if get_object_field(value, "residue", None) is not None:
        return chimerax_atom_spec(value)
    if get_object_field(value, "chain_id", None) is not None:
        return chimerax_residue_spec(value)
    return chimerax_model_spec(
        value,
        attribute=config.model_spec_attribute,
    )


# 28.6. Interaction specs and commands
# -----------------------------------------------------------------------------

def interaction_chimerax_specs(
    interaction: Any,
    *,
    config: ChimeraXConfig = DEFAULT_CHIMERAX_CONFIG,
) -> Mapping[str, Any]:
    """Return ligand and receptor atom specs for one interaction."""

    normalized = (
        interaction
        if isinstance(interaction, NormalizedInteraction)
        else normalize_interaction(interaction)
    )

    ligand_atom_object = (
        _interaction_atom(interaction, "ligand")
        if not isinstance(interaction, NormalizedInteraction)
        else None
    )
    receptor_atom_object = (
        _interaction_atom(interaction, "receptor")
        if not isinstance(interaction, NormalizedInteraction)
        else None
    )

    ligand_spec = (
        chimerax_atom_spec(ligand_atom_object)
        if ligand_atom_object is not None
        else sanitize_chimerax_spec(
            normalized.metadata.get("ligand_atomspec", "")
        )
    )
    receptor_spec = (
        chimerax_atom_spec(receptor_atom_object)
        if receptor_atom_object is not None
        else sanitize_chimerax_spec(
            normalized.metadata.get("receptor_atomspec", "")
        )
    )

    if not ligand_spec and normalized.ligand_atom.startswith(
        (CHIMERAX_MODEL_SPEC_PREFIX, CHIMERAX_CHAIN_SPEC_PREFIX)
    ):
        ligand_spec = sanitize_chimerax_spec(normalized.ligand_atom)
    if not receptor_spec and normalized.receptor_atom.startswith(
        (CHIMERAX_MODEL_SPEC_PREFIX, CHIMERAX_CHAIN_SPEC_PREFIX)
    ):
        receptor_spec = sanitize_chimerax_spec(
            normalized.receptor_atom
        )

    return MappingProxyType(
        {
            KEY_ID: normalized.id,
            KEY_FAMILY: normalized.family.value,
            KEY_TYPE: normalized.type,
            "ligand_atomspec": ligand_spec,
            "receptor_atomspec": receptor_spec,
            KEY_DISTANCE: normalized.distance,
            KEY_ANGLE: normalized.angle,
        }
    )


def chimerax_selection_command(
    specs: Iterable[Any],
    *,
    add: bool = False,
    label: str = "",
) -> ChimeraXCommand:
    """Build a ChimeraX selection command."""

    normalized = tuple(
        spec
        for spec in (
            sanitize_chimerax_spec(item)
            for item in specs
        )
        if spec
    )
    command_name = (
        f"{CHIMERAX_SELECT_COMMAND} add"
        if add
        else CHIMERAX_SELECT_COMMAND
    )
    command = (
        f"{command_name} {' '.join(normalized)}"
        if normalized
        else ""
    )
    return ChimeraXCommand(
        command=command,
        kind="selection",
        label=label or "Select report objects",
        atomspecs=normalized,
        enabled=bool(normalized),
    )


def chimerax_distance_command(
    first_spec: Any,
    second_spec: Any,
    *,
    label: str = "",
    metadata: Optional[Mapping[str, Any]] = None,
) -> ChimeraXCommand:
    """Build a ChimeraX distance command."""

    first = sanitize_chimerax_spec(first_spec)
    second = sanitize_chimerax_spec(second_spec)
    specs = tuple(item for item in (first, second) if item)
    command = (
        f"{CHIMERAX_DISTANCE_COMMAND} {first} {second}"
        if len(specs) == 2
        else ""
    )
    return ChimeraXCommand(
        command=command,
        kind="distance",
        label=label or "Display interaction distance",
        atomspecs=specs,
        enabled=len(specs) == 2,
        metadata=metadata or {},
    )


def chimerax_show_command(
    specs: Iterable[Any],
    *,
    representation: str = "atoms",
) -> ChimeraXCommand:
    """Build a ChimeraX show command."""

    normalized = tuple(
        spec
        for spec in (
            sanitize_chimerax_spec(item)
            for item in specs
        )
        if spec
    )
    representation_text = normalize_field_name(
        representation
    ).replace("_", " ")
    command = (
        f"{CHIMERAX_SHOW_COMMAND} "
        f"{' '.join(normalized)} {representation_text}"
        if normalized
        else ""
    )
    return ChimeraXCommand(
        command=command,
        kind="show",
        label="Show report objects",
        atomspecs=normalized,
        enabled=bool(normalized),
        metadata={"representation": representation_text},
    )


def chimerax_style_command(
    specs: Iterable[Any],
    *,
    style: str = "stick",
) -> ChimeraXCommand:
    """Build a ChimeraX style command."""

    normalized = tuple(
        spec
        for spec in (
            sanitize_chimerax_spec(item)
            for item in specs
        )
        if spec
    )
    style_text = normalize_field_name(style).replace("_", " ")
    command = (
        f"{CHIMERAX_STYLE_COMMAND} "
        f"{' '.join(normalized)} {style_text}"
        if normalized
        else ""
    )
    return ChimeraXCommand(
        command=command,
        kind="style",
        label="Style report objects",
        atomspecs=normalized,
        enabled=bool(normalized),
        metadata={"style": style_text},
    )


def interaction_chimerax_commands(
    interaction: Any,
    *,
    config: ChimeraXConfig = DEFAULT_CHIMERAX_CONFIG,
) -> Tuple[ChimeraXCommand, ...]:
    """Build reviewable ChimeraX commands for one interaction."""

    specs = interaction_chimerax_specs(
        interaction,
        config=config,
    )
    ligand_spec = specs.get("ligand_atomspec", "")
    receptor_spec = specs.get("receptor_atomspec", "")
    commands: List[ChimeraXCommand] = []

    if config.include_selection_commands:
        commands.append(
            chimerax_selection_command(
                (ligand_spec, receptor_spec),
                label=f"Select {specs.get(KEY_ID, 'interaction')}",
            )
        )
    if config.include_visualization_commands:
        commands.append(
            chimerax_distance_command(
                ligand_spec,
                receptor_spec,
                label=f"Distance {specs.get(KEY_ID, 'interaction')}",
                metadata={
                    KEY_ID: specs.get(KEY_ID),
                    KEY_FAMILY: specs.get(KEY_FAMILY),
                },
            )
        )
    return tuple(
        command for command in commands if command.enabled
    )


def report_chimerax_commands(
    interactions: Iterable[Any],
    *,
    config: ChimeraXConfig = DEFAULT_CHIMERAX_CONFIG,
    deduplicate: bool = True,
) -> Tuple[ChimeraXCommand, ...]:
    """Build commands for an interaction collection."""

    commands: List[ChimeraXCommand] = []
    for interaction in interactions:
        commands.extend(
            interaction_chimerax_commands(
                interaction,
                config=config,
            )
        )

    if not deduplicate:
        return tuple(commands)

    seen: Set[str] = set()
    unique: List[ChimeraXCommand] = []
    for command in commands:
        if command.command in seen:
            continue
        seen.add(command.command)
        unique.append(command)
    return tuple(unique)


# 28.7. Session metadata
# -----------------------------------------------------------------------------

def summarize_chimerax_session(
    session: Any,
) -> ChimeraXSessionSummary:
    """Create a non-invasive session summary."""

    if session is None:
        return ChimeraXSessionSummary(
            available=False,
            version=safe_string(chimerax_version(""), ""),
        )

    models = chimerax_session_models(session)
    model_specs = tuple(
        spec
        for spec in (
            chimerax_model_spec(model)
            for model in models
        )
        if spec
    )
    model_names = tuple(
        single_line_text(
            get_first_object_field(
                model,
                ("name", "display_name", "title"),
                "",
                skip_none=True,
            ),
            "",
        )
        for model in models
    )
    selected_atoms = chimerax_selected_atoms(session)
    selected_residues = chimerax_selected_residues(session)

    session_attributes: Dict[str, Any] = {}
    for name in (
        "in_script",
        "ui",
        "undo",
        "triggers",
    ):
        value = get_object_field(session, name, MISSING)
        if value is not MISSING:
            session_attributes[name] = type(value).__name__

    return ChimeraXSessionSummary(
        available=True,
        version=safe_string(chimerax_version(""), ""),
        model_count=len(models),
        model_specs=model_specs,
        model_names=tuple(
            item for item in model_names if item
        ),
        selected_atom_count=len(selected_atoms),
        selected_residue_count=len(selected_residues),
        attributes=session_attributes,
    )


# 28.8. Report data construction and enrichment
# -----------------------------------------------------------------------------

def build_chimerax_report_data(
    value: Any,
    *,
    session: Any = None,
    interactions: Optional[Iterable[Any]] = None,
    config: ChimeraXConfig = DEFAULT_CHIMERAX_CONFIG,
) -> ChimeraXReportData:
    """Build optional ChimeraX report metadata and commands."""

    resolved_session = resolve_chimerax_session(
        value,
        session=session,
    )
    capabilities = detect_chimerax_capabilities(
        session=resolved_session,
    )
    session_summary = summarize_chimerax_session(
        resolved_session
    )
    warnings_out: List[str] = []
    errors_out: List[str] = []

    if not config.enabled:
        warnings_out.append("ChimeraX compatibility is disabled.")

    if interactions is None:
        raw_interactions: List[Any] = []
        containers = list(iter_interaction_containers(value))
        if containers:
            for _, container in containers:
                raw_interactions.extend(
                    iter_object_collection(container)
                )
        elif isinstance(value, InteractionSection):
            raw_interactions.extend(value.interactions)
        elif isinstance(value, NormalizedInteraction):
            raw_interactions.append(value)
    else:
        raw_interactions = list(interactions)

    model_values: List[Any] = []
    for name in (
        "receptor",
        "ligand",
        "model",
        "structure",
    ):
        candidate = get_object_field(value, name, MISSING)
        if candidate is not MISSING and candidate is not None:
            model_values.append(candidate)
    model_values.extend(chimerax_session_models(resolved_session))

    model_specs = tuple(
        dict.fromkeys(
            spec
            for spec in (
                chimerax_object_spec(
                    model,
                    config=config,
                )
                for model in model_values
            )
            if spec
        )
    )

    interaction_specs: List[Mapping[str, Any]] = []
    atom_specs: List[str] = []
    residue_specs: List[str] = []
    commands: List[ChimeraXCommand] = []

    for interaction in raw_interactions:
        try:
            specs = interaction_chimerax_specs(
                interaction,
                config=config,
            )
        except ReportError as error:
            errors_out.append(str(error))
            continue
        interaction_specs.append(specs)
        for key in ("ligand_atomspec", "receptor_atomspec"):
            spec = specs.get(key, "")
            if spec:
                atom_specs.append(spec)
        commands.extend(
            interaction_chimerax_commands(
                interaction,
                config=config,
            )
        )

        normalized = (
            interaction
            if isinstance(interaction, NormalizedInteraction)
            else normalize_interaction(interaction)
        )
        if normalized.receptor_residue:
            residue_specs.append(
                sanitize_chimerax_spec(
                    (
                        CHIMERAX_CHAIN_SPEC_PREFIX
                        + normalized.chain_id
                        if normalized.chain_id
                        else ""
                    )
                    + CHIMERAX_RESIDUE_SPEC_PREFIX
                    + normalized.receptor_residue
                )
            )

    selection_commands = tuple(
        command
        for command in commands
        if command.kind == "selection"
    )
    visualization_commands = tuple(
        command
        for command in commands
        if command.kind != "selection"
    )

    return ChimeraXReportData(
        capabilities=capabilities,
        session=session_summary,
        model_specs=tuple(dict.fromkeys(model_specs)),
        atom_specs=tuple(dict.fromkeys(atom_specs)),
        residue_specs=tuple(
            dict.fromkeys(
                spec for spec in residue_specs if spec
            )
        ),
        selection_commands=selection_commands,
        visualization_commands=visualization_commands,
        interaction_specs=tuple(interaction_specs),
        metadata={
            "enabled": config.enabled,
            "model_spec_attribute": config.model_spec_attribute,
            "interaction_count": len(raw_interactions),
        },
        warnings=tuple(warnings_out),
        errors=tuple(errors_out),
    )


def chimerax_report_rows(
    data: ChimeraXReportData,
) -> ReportRows:
    """Return ChimeraX compatibility rows."""

    if not isinstance(data, ChimeraXReportData):
        raise ReportConfigurationError(
            "data must be ChimeraXReportData."
        )
    return [
        {
            "category": "runtime",
            "key": "available",
            "value": data.capabilities.available,
        },
        {
            "category": "runtime",
            "key": "version",
            "value": data.capabilities.version,
        },
        {
            "category": "runtime",
            "key": "commands_available",
            "value": data.capabilities.commands_available,
        },
        {
            "category": "session",
            "key": "model_count",
            "value": data.session.model_count,
        },
        {
            "category": "report",
            "key": "model_specs",
            "value": format_sequence(data.model_specs, missing=""),
        },
        {
            "category": "report",
            "key": "atom_specs",
            "value": format_sequence(data.atom_specs, missing=""),
        },
        {
            "category": "report",
            "key": "commands",
            "value": len(data.commands),
        },
    ]


def chimerax_command_rows(
    data: ChimeraXReportData,
) -> ReportRows:
    """Return generated command rows."""

    return [
        {
            KEY_RANK: index,
            "kind": command.kind,
            "label": command.label,
            "command": command.command,
            "atomspecs": format_sequence(
                command.atomspecs,
                missing="",
            ),
            "enabled": command.enabled,
        }
        for index, command in enumerate(data.commands, start=1)
    ]


def enrich_report_with_chimerax(
    report: ReportDocument,
    *,
    value: Any = None,
    session: Any = None,
    data: Optional[ChimeraXReportData] = None,
) -> ReportDocument:
    """Attach ChimeraX metadata and optional commands to a report."""

    if not isinstance(report, ReportDocument):
        raise ReportConfigurationError(
            "report must be ReportDocument."
        )
    chimerax_data = data or build_chimerax_report_data(
        report if value is None else value,
        session=session,
        config=report.config.chimerax,
    )

    metadata = {
        **dict(report.metadata),
        CHIMERAX_REPORT_METADATA_KEY: chimerax_data.to_dict(),
    }

    sections = list(report.sections)
    provenance_index = next(
        (
            index
            for index, section in enumerate(sections)
            if section.id is ReportSectionID.PROVENANCE
        ),
        None,
    )
    if provenance_index is not None:
        provenance = sections[provenance_index]
        blocks = list(provenance.blocks)
        rows = chimerax_report_rows(chimerax_data)
        if rows:
            blocks.append(
                table_block(
                    rows,
                    title="ChimeraX compatibility",
                    name="chimerax",
                )
            )
        command_rows = chimerax_command_rows(chimerax_data)
        if command_rows:
            blocks.append(
                table_block(
                    command_rows,
                    title="ChimeraX commands",
                    name="chimerax_commands",
                )
            )
        sections[provenance_index] = replace(
            provenance,
            blocks=tuple(blocks),
            empty=not any(block.visible for block in blocks),
        )

    return replace(
        report,
        sections=tuple(sections),
        metadata=metadata,
        warnings=tuple(
            dict.fromkeys(
                (*report.warnings, *chimerax_data.warnings)
            )
        ),
        errors=tuple(
            dict.fromkeys(
                (*report.errors, *chimerax_data.errors)
            )
        ),
    )


# 28.9. Command execution
# -----------------------------------------------------------------------------

def execute_chimerax_command(
    session: Any,
    command: Union[ChimeraXCommand, str],
    *,
    log: bool = True,
    required: bool = True,
) -> Any:
    """Execute one command through ChimeraX's command runner."""

    if session is None:
        raise ChimeraXReportError(
            "A ChimeraX session is required to run commands."
        )

    command_text = (
        command.command
        if isinstance(command, ChimeraXCommand)
        else safe_string(command, "")
    )
    if not command_text:
        raise ChimeraXReportError(
            "ChimeraX command is empty."
        )

    commands_module = load_chimerax_module(
        "chimerax.core.commands",
        required=required,
    )
    runner = getattr(commands_module, "run", None)
    if not callable(runner):
        raise ChimeraXReportError(
            "ChimeraX command runner is unavailable."
        )

    try:
        return call_export_function(
            runner,
            session,
            command_text,
            log=log,
        )
    except ReportError as error:
        raise ChimeraXReportError(
            "Unable to execute ChimeraX command.",
            context={"command": command_text},
            cause=error,
        ) from error


def execute_chimerax_commands(
    session: Any,
    commands: Iterable[Union[ChimeraXCommand, str]],
    *,
    log: bool = True,
    stop_on_error: bool = True,
) -> Tuple[Any, ...]:
    """Execute multiple explicitly supplied ChimeraX commands."""

    results: List[Any] = []
    errors: List[ReportError] = []

    for command in commands:
        if isinstance(command, ChimeraXCommand) and not command.enabled:
            continue
        try:
            results.append(
                execute_chimerax_command(
                    session,
                    command,
                    log=log,
                )
            )
        except ReportError as error:
            if stop_on_error:
                raise
            errors.append(error)

    if errors and not results:
        raise ReportAggregateError(errors)
    return tuple(results)


# 28.10. Public ChimeraX interface
# -----------------------------------------------------------------------------

_SECTION_28_PUBLIC_NAMES: Final[Tuple[str, ...]] = (
    "CHIMERAX_MODULE_NAMES",
    "CHIMERAX_COMMAND_TYPES",
    "ChimeraXCapabilities",
    "ChimeraXCommand",
    "ChimeraXSessionSummary",
    "ChimeraXReportData",
    "load_chimerax_module",
    "chimerax_available",
    "detect_chimerax_capabilities",
    "is_chimerax_object",
    "is_chimerax_session",
    "resolve_chimerax_session",
    "chimerax_session_models",
    "chimerax_selected_atoms",
    "chimerax_selected_residues",
    "sanitize_chimerax_spec",
    "chimerax_model_spec",
    "chimerax_chain_spec",
    "chimerax_residue_spec",
    "chimerax_atom_spec",
    "chimerax_object_spec",
    "interaction_chimerax_specs",
    "chimerax_selection_command",
    "chimerax_distance_command",
    "chimerax_show_command",
    "chimerax_style_command",
    "interaction_chimerax_commands",
    "report_chimerax_commands",
    "summarize_chimerax_session",
    "build_chimerax_report_data",
    "chimerax_report_rows",
    "chimerax_command_rows",
    "enrich_report_with_chimerax",
    "execute_chimerax_command",
    "execute_chimerax_commands",
)

_register_public_names(_SECTION_28_PUBLIC_NAMES)

# =============================================================================
# End of Section 28
# =============================================================================


# =============================================================================
# Section 29 — Introspection
# =============================================================================

# 29.1. Introspection constants and records
# -----------------------------------------------------------------------------

PUBLIC_SYMBOL_KINDS: Final[Tuple[str, ...]] = (
    "class",
    "function",
    "constant",
    "enum",
    "dataclass",
    "module",
    "object",
)

REPORT_RENDERER_NAMES: Final[Mapping[str, str]] = MappingProxyType(
    {
        REPORT_FORMAT_TEXT: "render_report_text",
        REPORT_FORMAT_MARKDOWN: "render_report_markdown",
        REPORT_FORMAT_HTML: "render_report_html",
        REPORT_FORMAT_JSON: "render_report_json",
    }
)

REPORT_WRITER_NAMES: Final[Mapping[str, str]] = MappingProxyType(
    {
        REPORT_FORMAT_TEXT: "write_report",
        REPORT_FORMAT_MARKDOWN: "write_report",
        REPORT_FORMAT_HTML: "write_report",
        REPORT_FORMAT_JSON: "write_report",
    }
)


@dataclass(frozen=True)
class PublicSymbolInfo:
    """Introspection metadata for one public symbol."""

    name: str
    kind: str
    section: Optional[int] = None
    module: str = ""
    qualified_name: str = ""
    signature: str = ""
    summary: str = ""
    public: bool = True
    callable: bool = False
    dataclass: bool = False
    enum: bool = False
    metadata: Mapping[str, Any] = field(
        default_factory=lambda: _EMPTY_METADATA
    )

    def __post_init__(self) -> None:
        for name in (
            "name",
            "kind",
            "module",
            "qualified_name",
            "signature",
            "summary",
        ):
            object.__setattr__(
                self,
                name,
                single_line_text(getattr(self, name), ""),
            )
        if self.section is not None:
            object.__setattr__(
                self,
                "section",
                max(1, to_safe_int(self.section, 1)),
            )
        object.__setattr__(self, "public", bool(self.public))
        object.__setattr__(self, "callable", bool(self.callable))
        object.__setattr__(self, "dataclass", bool(self.dataclass))
        object.__setattr__(self, "enum", bool(self.enum))
        object.__setattr__(
            self,
            "metadata",
            _freeze_config_mapping(self.metadata),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain symbol record."""

        return {
            "name": self.name,
            "kind": self.kind,
            "section": self.section,
            "module": self.module,
            "qualified_name": self.qualified_name,
            "signature": self.signature,
            "summary": self.summary,
            "public": self.public,
            "callable": self.callable,
            "dataclass": self.dataclass,
            "enum": self.enum,
            KEY_METADATA: dict(self.metadata),
        }


@dataclass(frozen=True)
class ReportModuleCapabilities:
    """High-level capabilities exposed by report.py."""

    module_name: str
    version: str
    schema_name: str
    schema_version: str
    formats: Tuple[str, ...]
    sections: Tuple[str, ...]
    renderers: Mapping[str, str]
    writers: Mapping[str, str]
    exporters: Tuple[str, ...]
    public_symbol_count: int
    public_class_count: int
    public_function_count: int
    chimerax: ChimeraXCapabilities
    export_integration: ExportIntegrationCapabilities
    optional_dependencies: Mapping[str, bool] = field(
        default_factory=lambda: _EMPTY_METADATA
    )
    metadata: Mapping[str, Any] = field(
        default_factory=lambda: _EMPTY_METADATA
    )

    def __post_init__(self) -> None:
        for name in (
            "module_name",
            "version",
            "schema_name",
            "schema_version",
        ):
            object.__setattr__(
                self,
                name,
                single_line_text(getattr(self, name), ""),
            )
        for name in (
            "formats",
            "sections",
            "exporters",
        ):
            object.__setattr__(
                self,
                name,
                _freeze_config_strings(getattr(self, name)),
            )
        for name in (
            "public_symbol_count",
            "public_class_count",
            "public_function_count",
        ):
            object.__setattr__(
                self,
                name,
                max(0, to_safe_int(getattr(self, name), 0)),
            )
        object.__setattr__(
            self,
            "renderers",
            _freeze_config_mapping(self.renderers),
        )
        object.__setattr__(
            self,
            "writers",
            _freeze_config_mapping(self.writers),
        )
        object.__setattr__(
            self,
            "optional_dependencies",
            MappingProxyType(
                {
                    str(key): bool(value)
                    for key, value in dict(
                        self.optional_dependencies
                    ).items()
                }
            ),
        )
        object.__setattr__(
            self,
            "metadata",
            _freeze_config_mapping(self.metadata),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain capability record."""

        return {
            "module_name": self.module_name,
            "version": self.version,
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "formats": list(self.formats),
            "sections": list(self.sections),
            "renderers": dict(self.renderers),
            "writers": dict(self.writers),
            "exporters": list(self.exporters),
            "public_symbol_count": self.public_symbol_count,
            "public_class_count": self.public_class_count,
            "public_function_count": self.public_function_count,
            "chimerax": self.chimerax.to_dict(),
            "export_integration": self.export_integration.to_dict(),
            "optional_dependencies": dict(
                self.optional_dependencies
            ),
            KEY_METADATA: dict(self.metadata),
        }


@dataclass(frozen=True)
class PublicAPIValidation:
    """Validation result for the registered public API."""

    valid: bool
    missing: Tuple[str, ...] = ()
    duplicate: Tuple[str, ...] = ()
    unregistered: Tuple[str, ...] = ()
    invalid_section_exports: Tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(
        default_factory=lambda: _EMPTY_METADATA
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "valid", bool(self.valid))
        for name in (
            "missing",
            "duplicate",
            "unregistered",
            "invalid_section_exports",
        ):
            object.__setattr__(
                self,
                name,
                _freeze_config_strings(getattr(self, name)),
            )
        object.__setattr__(
            self,
            "metadata",
            _freeze_config_mapping(self.metadata),
        )

    def raise_for_errors(self) -> None:
        """Raise when the public API is inconsistent."""

        if self.valid:
            return
        raise ReportIntrospectionError(
            "Public report API is inconsistent.",
            context=self.to_dict(),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain API validation result."""

        return {
            "valid": self.valid,
            "missing": list(self.missing),
            "duplicate": list(self.duplicate),
            "unregistered": list(self.unregistered),
            "invalid_section_exports": list(
                self.invalid_section_exports
            ),
            KEY_METADATA: dict(self.metadata),
        }


# 29.2. Namespace and section discovery
# -----------------------------------------------------------------------------

def report_namespace(
    namespace: Optional[Mapping[str, Any]] = None,
) -> Mapping[str, Any]:
    """Return the namespace used for report introspection."""

    return globals() if namespace is None else namespace


def public_api_names(
    namespace: Optional[Mapping[str, Any]] = None,
) -> Tuple[str, ...]:
    """Return registered public names."""

    source = report_namespace(namespace)
    names = source.get("__all__", __all__)
    return tuple(str(name) for name in names)


def section_public_name_map(
    namespace: Optional[Mapping[str, Any]] = None,
) -> Dict[int, Tuple[str, ...]]:
    """Return public symbols indexed by implementation section."""

    source = report_namespace(namespace)
    output: Dict[int, Tuple[str, ...]] = {}

    for name, value in source.items():
        match = re.fullmatch(
            r"_SECTION_(\d+)_PUBLIC_NAMES",
            str(name),
        )
        if not match:
            continue
        section_number = int(match.group(1))
        output[section_number] = tuple(
            str(item)
            for item in iter_object_collection(value)
        )
    return dict(sorted(output.items()))


def public_name_section(
    name: str,
    namespace: Optional[Mapping[str, Any]] = None,
) -> Optional[int]:
    """Return the implementation section for a public name."""

    for section, names in section_public_name_map(
        namespace
    ).items():
        if name in names:
            return section
    return None


# 29.3. Symbol classification
# -----------------------------------------------------------------------------

def safe_callable_signature(value: Any) -> str:
    """Return a callable signature without raising."""

    if not callable(value):
        return ""
    try:
        return str(inspect.signature(value))
    except (TypeError, ValueError):
        return ""


def doc_summary(value: Any) -> str:
    """Return the first non-empty docstring paragraph."""

    documentation = inspect.getdoc(value) or ""
    if not documentation:
        return ""
    paragraph = documentation.split("\n\n", 1)[0]
    return single_line_text(paragraph, "")


def public_symbol_kind(value: Any) -> str:
    """Classify a public symbol."""

    if inspect.ismodule(value):
        return "module"
    if inspect.isclass(value):
        if issubclass(value, Enum):
            return "enum"
        if is_dataclass(value):
            return "dataclass"
        return "class"
    if inspect.isfunction(value) or inspect.ismethod(value):
        return "function"
    if callable(value):
        return "function"
    if isinstance(value, (str, int, float, bool, tuple, frozenset, Mapping)):
        return "constant"
    return "object"


def inspect_public_symbol(
    name: str,
    *,
    namespace: Optional[Mapping[str, Any]] = None,
) -> PublicSymbolInfo:
    """Inspect one registered public symbol."""

    source = report_namespace(namespace)
    if name not in source:
        raise ReportIntrospectionError(
            f"Public symbol is unavailable: {name}.",
            context={"name": name},
        )

    value = source[name]
    kind = public_symbol_kind(value)
    module_name = safe_string(
        getattr(value, "__module__", ""),
        "",
    )
    qualified_name = safe_string(
        getattr(value, "__qualname__", name),
        name,
    )

    metadata: Dict[str, Any] = {}
    if inspect.isclass(value):
        metadata["bases"] = [
            base.__name__
            for base in getattr(value, "__bases__", ())
        ]
        metadata["abstract"] = inspect.isabstract(value)
    elif isinstance(value, Mapping):
        metadata["size"] = len(value)
    elif isinstance(value, (tuple, list, set, frozenset)):
        metadata["size"] = len(value)

    return PublicSymbolInfo(
        name=name,
        kind=kind,
        section=public_name_section(
            name,
            namespace=source,
        ),
        module=module_name,
        qualified_name=qualified_name,
        signature=safe_callable_signature(value),
        summary=doc_summary(value),
        public=name in public_api_names(source),
        callable=callable(value),
        dataclass=inspect.isclass(value) and is_dataclass(value),
        enum=inspect.isclass(value) and issubclass(value, Enum),
        metadata=metadata,
    )


def inspect_public_api(
    *,
    namespace: Optional[Mapping[str, Any]] = None,
    kind: Optional[str] = None,
    section: Optional[int] = None,
) -> Tuple[PublicSymbolInfo, ...]:
    """Inspect all registered public symbols."""

    source = report_namespace(namespace)
    kind_filter = normalize_field_name(kind) if kind else ""
    output: List[PublicSymbolInfo] = []

    for name in public_api_names(source):
        if name not in source:
            continue
        info = inspect_public_symbol(
            name,
            namespace=source,
        )
        if kind_filter and info.kind != kind_filter:
            continue
        if section is not None and info.section != int(section):
            continue
        output.append(info)
    return tuple(output)


def search_public_api(
    query: str,
    *,
    namespace: Optional[Mapping[str, Any]] = None,
    kind: Optional[str] = None,
    section: Optional[int] = None,
) -> Tuple[PublicSymbolInfo, ...]:
    """Search names, signatures and summaries."""

    token = normalize_field_name(query)
    if not token:
        return inspect_public_api(
            namespace=namespace,
            kind=kind,
            section=section,
        )

    matches: List[PublicSymbolInfo] = []
    for info in inspect_public_api(
        namespace=namespace,
        kind=kind,
        section=section,
    ):
        haystack = normalize_field_name(
            " ".join(
                (
                    info.name,
                    info.kind,
                    info.signature,
                    info.summary,
                )
            )
        )
        if token in haystack:
            matches.append(info)
    return tuple(matches)


# 29.4. Callable compatibility
# -----------------------------------------------------------------------------

def callable_parameters(
    value: Any,
) -> Mapping[str, inspect.Parameter]:
    """Return a callable's signature parameters."""

    if not callable(value):
        return MappingProxyType({})
    try:
        signature = inspect.signature(value)
    except (TypeError, ValueError):
        return MappingProxyType({})
    return MappingProxyType(dict(signature.parameters))


def callable_accepts_parameter(
    value: Any,
    name: str,
) -> bool:
    """Return whether a callable accepts one named parameter."""

    parameters = callable_parameters(value)
    if name in parameters:
        return True
    return any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )


def callable_required_parameters(
    value: Any,
) -> Tuple[str, ...]:
    """Return required callable parameters."""

    return tuple(
        name
        for name, parameter in callable_parameters(value).items()
        if parameter.default is inspect.Parameter.empty
        and parameter.kind
        in {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }
    )


def callable_compatible_with(
    value: Any,
    *,
    positional_count: int = 0,
    keyword_names: Iterable[str] = (),
) -> bool:
    """Check whether a callable can accept a proposed call shape."""

    if not callable(value):
        return False
    try:
        signature = inspect.signature(value)
        args = [None] * max(0, int(positional_count))
        kwargs = {
            str(name): None
            for name in keyword_names
        }
        signature.bind(*args, **kwargs)
        return True
    except (TypeError, ValueError):
        return False


# 29.5. Available report capabilities
# -----------------------------------------------------------------------------

def available_report_formats() -> Tuple[str, ...]:
    """Return supported rendered report formats."""

    return tuple(SUPPORTED_REPORT_FORMATS)


def available_report_sections() -> Tuple[str, ...]:
    """Return available report section identifiers."""

    return tuple(
        section.value for section in ReportSectionID
    )


def available_report_renderers(
    *,
    namespace: Optional[Mapping[str, Any]] = None,
) -> Mapping[str, str]:
    """Return available renderer names by format."""

    source = report_namespace(namespace)
    return MappingProxyType(
        {
            format_name: function_name
            for format_name, function_name
            in REPORT_RENDERER_NAMES.items()
            if callable(source.get(function_name))
        }
    )


def available_report_writers(
    *,
    namespace: Optional[Mapping[str, Any]] = None,
) -> Mapping[str, str]:
    """Return available writer names by format."""

    source = report_namespace(namespace)
    return MappingProxyType(
        {
            format_name: function_name
            for format_name, function_name
            in REPORT_WRITER_NAMES.items()
            if callable(source.get(function_name))
        }
    )


def available_report_exporters(
    *,
    namespace: Optional[Mapping[str, Any]] = None,
) -> Tuple[str, ...]:
    """Return available high-level export functions."""

    source = report_namespace(namespace)
    candidates = (
        "write_report",
        "write_report_formats",
        "export_report_with_export_module",
        "export_report_bundle",
        "save_dock_report",
        "save_dock_report_formats",
        "export_dock_report",
    )
    return tuple(
        name
        for name in candidates
        if callable(source.get(name))
    )


def inspect_report_capabilities(
    *,
    namespace: Optional[Mapping[str, Any]] = None,
    session: Any = None,
    export_module: Any = None,
) -> ReportModuleCapabilities:
    """Inspect the complete high-level report feature set."""

    source = report_namespace(namespace)
    symbols = inspect_public_api(namespace=source)
    class_count = sum(
        info.kind in {"class", "dataclass", "enum"}
        for info in symbols
    )
    function_count = sum(
        info.kind == "function"
        for info in symbols
    )

    return ReportModuleCapabilities(
        module_name=_MODULE_NAME,
        version=__version__,
        schema_name=REPORT_SCHEMA_NAME,
        schema_version=REPORT_SCHEMA_VERSION,
        formats=available_report_formats(),
        sections=available_report_sections(),
        renderers=available_report_renderers(
            namespace=source,
        ),
        writers=available_report_writers(
            namespace=source,
        ),
        exporters=available_report_exporters(
            namespace=source,
        ),
        public_symbol_count=len(symbols),
        public_class_count=class_count,
        public_function_count=function_count,
        chimerax=detect_chimerax_capabilities(
            session=session,
        ),
        export_integration=inspect_export_capabilities(
            export_module
        ),
        optional_dependencies={
            "numpy": NUMPY_AVAILABLE,
            "chimerax": chimerax_available(),
            "export": export_module_available()
            if export_module is None
            else True,
        },
        metadata={
            "status": __status__,
            "license": __license__,
            "author": __author__,
        },
    )


# 29.6. Public API manifest
# -----------------------------------------------------------------------------

def report_api_manifest(
    *,
    namespace: Optional[Mapping[str, Any]] = None,
    include_symbols: bool = True,
    include_private_metadata: bool = False,
) -> Dict[str, Any]:
    """Return a machine-readable public API manifest."""

    source = report_namespace(namespace)
    capabilities = inspect_report_capabilities(
        namespace=source,
    )
    manifest: Dict[str, Any] = {
        "module": _MODULE_NAME,
        "description": _MODULE_DESCRIPTION,
        "version": __version__,
        "schema_name": REPORT_SCHEMA_NAME,
        "schema_version": REPORT_SCHEMA_VERSION,
        "capabilities": capabilities.to_dict(),
        "sections": {
            str(section): list(names)
            for section, names in section_public_name_map(
                source
            ).items()
        },
    }
    if include_symbols:
        manifest["symbols"] = [
            info.to_dict()
            for info in inspect_public_api(
                namespace=source,
            )
        ]
    if include_private_metadata:
        manifest["metadata"] = {
            "module_file": globals().get("__file__", ""),
            "package": __package__,
            "python": platform.python_version(),
        }
    return manifest


def render_report_api_manifest_json(
    *,
    namespace: Optional[Mapping[str, Any]] = None,
    indent: int = DEFAULT_JSON_INDENT,
) -> str:
    """Render the public API manifest as JSON."""

    try:
        return json.dumps(
            to_json_safe(
                report_api_manifest(
                    namespace=namespace,
                )
            ),
            indent=indent,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
        ) + DEFAULT_NEWLINE
    except (TypeError, ValueError) as error:
        raise ReportIntrospectionError(
            "Unable to render report API manifest.",
            cause=error,
        ) from error


def public_api_rows(
    *,
    namespace: Optional[Mapping[str, Any]] = None,
    kind: Optional[str] = None,
    section: Optional[int] = None,
) -> ReportRows:
    """Return public API symbols as table rows."""

    return [
        {
            KEY_RANK: index,
            "name": info.name,
            "kind": info.kind,
            "section": info.section,
            "signature": info.signature,
            "summary": info.summary,
            "module": info.module,
        }
        for index, info in enumerate(
            inspect_public_api(
                namespace=namespace,
                kind=kind,
                section=section,
            ),
            start=1,
        )
    ]


def capability_rows(
    capabilities: Optional[ReportModuleCapabilities] = None,
) -> ReportRows:
    """Return high-level capabilities as table rows."""

    value = capabilities or inspect_report_capabilities()
    rows: ReportRows = [
        {
            "category": "module",
            "capability": "version",
            "value": value.version,
        },
        {
            "category": "module",
            "capability": "schema",
            "value": (
                f"{value.schema_name} "
                f"{value.schema_version}"
            ),
        },
        {
            "category": "formats",
            "capability": "rendered",
            "value": format_sequence(value.formats, missing=""),
        },
        {
            "category": "sections",
            "capability": "available",
            "value": format_sequence(value.sections, missing=""),
        },
        {
            "category": "api",
            "capability": "public_symbols",
            "value": value.public_symbol_count,
        },
        {
            "category": "api",
            "capability": "public_functions",
            "value": value.public_function_count,
        },
        {
            "category": "api",
            "capability": "public_classes",
            "value": value.public_class_count,
        },
        {
            "category": "integration",
            "capability": "chimerax",
            "value": value.chimerax.available,
        },
        {
            "category": "integration",
            "capability": "export",
            "value": value.export_integration.available,
        },
    ]
    return rows


# 29.7. Public API validation
# -----------------------------------------------------------------------------

def validate_public_api(
    *,
    namespace: Optional[Mapping[str, Any]] = None,
    include_unregistered: bool = False,
) -> PublicAPIValidation:
    """Validate public-name registration and section ownership."""

    source = report_namespace(namespace)
    names = list(public_api_names(source))
    missing = tuple(
        name for name in names if name not in source
    )
    duplicate = tuple(
        name
        for name, count in Counter(names).items()
        if count > 1
    )

    section_map = section_public_name_map(source)
    invalid_section_exports: List[str] = []
    section_names: List[str] = []
    for section, values in section_map.items():
        for name in values:
            section_names.append(name)
            if name not in source:
                invalid_section_exports.append(
                    f"{section}:{name}"
                )

    unregistered: Tuple[str, ...] = ()
    if include_unregistered:
        public_candidates = {
            name
            for name, value in source.items()
            if not name.startswith("_")
            and (
                inspect.isclass(value)
                or inspect.isfunction(value)
            )
            and getattr(value, "__module__", None)
            == source.get("__name__", __name__)
        }
        unregistered = tuple(
            sorted(public_candidates - set(names))
        )

    valid = not (
        missing
        or duplicate
        or invalid_section_exports
    )
    return PublicAPIValidation(
        valid=valid,
        missing=missing,
        duplicate=duplicate,
        unregistered=unregistered,
        invalid_section_exports=tuple(
            invalid_section_exports
        ),
        metadata={
            "public_count": len(names),
            "section_count": len(section_map),
            "section_export_count": len(section_names),
        },
    )


def describe_report_api(
    name: Optional[str] = None,
    *,
    namespace: Optional[Mapping[str, Any]] = None,
) -> str:
    """Return a compact human-readable API description."""

    source = report_namespace(namespace)
    if name is not None:
        info = inspect_public_symbol(
            name,
            namespace=source,
        )
        parts = [
            f"{info.name} ({info.kind})",
            f"Section: {info.section or DEFAULT_MISSING_TEXT}",
        ]
        if info.signature:
            parts.append(f"Signature: {info.signature}")
        if info.summary:
            parts.append(info.summary)
        return DEFAULT_NEWLINE.join(parts)

    capabilities = inspect_report_capabilities(
        namespace=source,
    )
    return DEFAULT_NEWLINE.join(
        (
            f"{capabilities.module_name} {capabilities.version}",
            (
                "Formats: "
                + ", ".join(capabilities.formats)
            ),
            (
                "Sections: "
                + ", ".join(capabilities.sections)
            ),
            (
                "Public API: "
                f"{capabilities.public_symbol_count} symbols, "
                f"{capabilities.public_function_count} functions, "
                f"{capabilities.public_class_count} classes"
            ),
            (
                "ChimeraX: "
                + (
                    "available"
                    if capabilities.chimerax.available
                    else "unavailable"
                )
            ),
            (
                "export.py: "
                + (
                    "available"
                    if capabilities.export_integration.available
                    else "unavailable"
                )
            ),
        )
    )


# 29.8. Public introspection interface
# -----------------------------------------------------------------------------

_SECTION_29_PUBLIC_NAMES: Final[Tuple[str, ...]] = (
    "PUBLIC_SYMBOL_KINDS",
    "PublicSymbolInfo",
    "ReportModuleCapabilities",
    "PublicAPIValidation",
    "report_namespace",
    "public_api_names",
    "section_public_name_map",
    "public_name_section",
    "safe_callable_signature",
    "doc_summary",
    "public_symbol_kind",
    "inspect_public_symbol",
    "inspect_public_api",
    "search_public_api",
    "callable_parameters",
    "callable_accepts_parameter",
    "callable_required_parameters",
    "callable_compatible_with",
    "available_report_formats",
    "available_report_sections",
    "available_report_renderers",
    "available_report_writers",
    "available_report_exporters",
    "inspect_report_capabilities",
    "report_api_manifest",
    "render_report_api_manifest_json",
    "public_api_rows",
    "capability_rows",
    "validate_public_api",
    "describe_report_api",
)

_register_public_names(_SECTION_29_PUBLIC_NAMES)

# =============================================================================
# End of Section 29
# =============================================================================

# =============================================================================
# Section 30 — Self-tests
# =============================================================================

# =============================================================================
# Section 30.1 — Self-test infrastructure
# =============================================================================

# 30.1.1. Constants and statuses
# -----------------------------------------------------------------------------

SELF_TEST_SECTION_INFRASTRUCTURE: Final[str] = "30.1"
SELF_TEST_SECTION_FORMATTING: Final[str] = "30.2"
SELF_TEST_SECTION_INTERACTIONS: Final[str] = "30.3"
SELF_TEST_SECTION_SINGLE_POSE: Final[str] = "30.4"
SELF_TEST_SECTION_MULTIPOSE: Final[str] = "30.5"
SELF_TEST_SECTION_RENDERING: Final[str] = "30.6"
SELF_TEST_SECTION_WRITING_EXPORT: Final[str] = "30.7"
SELF_TEST_SECTION_FINAL_RUNNER: Final[str] = "30.8"

SELF_TEST_STATUS_PASS: Final[str] = "pass"
SELF_TEST_STATUS_FAIL: Final[str] = "fail"
SELF_TEST_STATUS_ERROR: Final[str] = "error"
SELF_TEST_STATUS_SKIP: Final[str] = "skip"

SELF_TEST_STATUSES: Final[Tuple[str, ...]] = (
    SELF_TEST_STATUS_PASS,
    SELF_TEST_STATUS_FAIL,
    SELF_TEST_STATUS_ERROR,
    SELF_TEST_STATUS_SKIP,
)

SELF_TEST_DEFAULT_TOLERANCE: Final[float] = 1e-9
SELF_TEST_DEFAULT_RELATIVE_TOLERANCE: Final[float] = 1e-9
SELF_TEST_DEFAULT_MAX_FAILURES: Final[int] = 100


class SelfTestStatus(_StringEnum):
    """Self-test execution status."""

    PASS = SELF_TEST_STATUS_PASS
    FAIL = SELF_TEST_STATUS_FAIL
    ERROR = SELF_TEST_STATUS_ERROR
    SKIP = SELF_TEST_STATUS_SKIP


class SelfTestFailure(ReportSelfTestError):
    """Raised by a failed self-test assertion."""

    default_code = "self_test_assertion_failed"


class SelfTestSkip(ReportSelfTestError):
    """Raised when a self-test is intentionally skipped."""

    default_code = "self_test_skipped"


# 30.1.2. Test records
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class SelfTestCase:
    """Registered self-test definition."""

    name: str
    section: str
    function: Callable[[], Any]
    description: str = ""
    tags: Tuple[str, ...] = ()
    enabled: bool = True
    metadata: Mapping[str, Any] = field(
        default_factory=lambda: _EMPTY_METADATA
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "name",
            safe_identifier(self.name, "unnamed_test"),
        )
        object.__setattr__(
            self,
            "section",
            single_line_text(self.section, SELF_TEST_SECTION_INFRASTRUCTURE),
        )
        if not callable(self.function):
            raise ReportConfigurationError(
                "SelfTestCase.function must be callable."
            )
        object.__setattr__(
            self,
            "description",
            safe_string(self.description, ""),
        )
        object.__setattr__(
            self,
            "tags",
            _freeze_config_strings(self.tags),
        )
        object.__setattr__(self, "enabled", bool(self.enabled))
        object.__setattr__(
            self,
            "metadata",
            _freeze_config_mapping(self.metadata),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain test-case record."""

        return {
            "name": self.name,
            "section": self.section,
            "description": self.description,
            "tags": list(self.tags),
            "enabled": self.enabled,
            "function": getattr(
                self.function,
                "__qualname__",
                getattr(self.function, "__name__", ""),
            ),
            KEY_METADATA: dict(self.metadata),
        }


@dataclass(frozen=True)
class SelfTestResult:
    """Result of one self-test."""

    name: str
    section: str
    status: SelfTestStatus
    duration_seconds: float = 0.0
    message: str = ""
    exception_type: str = ""
    traceback_text: str = ""
    details: Mapping[str, Any] = field(
        default_factory=lambda: _EMPTY_METADATA
    )

    def __post_init__(self) -> None:
        for name in ("name", "section", "message", "exception_type"):
            object.__setattr__(
                self,
                name,
                single_line_text(getattr(self, name), ""),
            )
        object.__setattr__(
            self,
            "status",
            _coerce_enum(
                SelfTestStatus,
                self.status,
                "status",
            ),
        )
        object.__setattr__(
            self,
            "duration_seconds",
            max(0.0, to_finite_float(self.duration_seconds, 0.0)),
        )
        object.__setattr__(
            self,
            "traceback_text",
            safe_string(self.traceback_text, "", strip=False),
        )
        object.__setattr__(
            self,
            "details",
            _freeze_config_mapping(self.details),
        )

    @property
    def passed(self) -> bool:
        """Return whether the test passed."""

        return self.status is SelfTestStatus.PASS

    @property
    def failed(self) -> bool:
        """Return whether the test failed or errored."""

        return self.status in {
            SelfTestStatus.FAIL,
            SelfTestStatus.ERROR,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain result record."""

        return {
            "name": self.name,
            "section": self.section,
            "status": self.status.value,
            "duration_seconds": self.duration_seconds,
            "message": self.message,
            "exception_type": self.exception_type,
            "traceback": self.traceback_text,
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class SelfTestReport:
    """Aggregate report for a self-test run."""

    results: Tuple[SelfTestResult, ...] = ()
    started_at: str = ""
    finished_at: str = ""
    duration_seconds: float = 0.0
    selected_sections: Tuple[str, ...] = ()
    selected_tags: Tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(
        default_factory=lambda: _EMPTY_METADATA
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "results", tuple(self.results))
        object.__setattr__(
            self,
            "started_at",
            self.started_at or current_utc_timestamp(),
        )
        object.__setattr__(
            self,
            "finished_at",
            self.finished_at or current_utc_timestamp(),
        )
        object.__setattr__(
            self,
            "duration_seconds",
            max(0.0, to_finite_float(self.duration_seconds, 0.0)),
        )
        object.__setattr__(
            self,
            "selected_sections",
            _freeze_config_strings(self.selected_sections),
        )
        object.__setattr__(
            self,
            "selected_tags",
            _freeze_config_strings(self.selected_tags),
        )
        object.__setattr__(
            self,
            "metadata",
            _freeze_config_mapping(self.metadata),
        )

    @property
    def passed(self) -> int:
        """Return the number of passed tests."""

        return sum(result.passed for result in self.results)

    @property
    def failed(self) -> int:
        """Return the number of assertion failures."""

        return sum(
            result.status is SelfTestStatus.FAIL
            for result in self.results
        )

    @property
    def errors(self) -> int:
        """Return the number of unexpected errors."""

        return sum(
            result.status is SelfTestStatus.ERROR
            for result in self.results
        )

    @property
    def skipped(self) -> int:
        """Return the number of skipped tests."""

        return sum(
            result.status is SelfTestStatus.SKIP
            for result in self.results
        )

    @property
    def total(self) -> int:
        """Return the number of executed or skipped tests."""

        return len(self.results)

    @property
    def successful(self) -> bool:
        """Return whether no test failed or errored."""

        return self.failed == 0 and self.errors == 0

    def raise_for_failures(self) -> None:
        """Raise an aggregate error when the run was unsuccessful."""

        failures = [
            ReportSelfTestError(
                result.name,
                result.message or result.status.value,
                context=result.details,
            )
            for result in self.results
            if result.failed
        ]
        if failures:
            raise ReportAggregateError(
                failures,
                message=(
                    f"{len(failures)} report self-test"
                    f"{'' if len(failures) == 1 else 's'} failed."
                ),
            )

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain self-test report."""

        return {
            "successful": self.successful,
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "errors": self.errors,
            "skipped": self.skipped,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_seconds": self.duration_seconds,
            "selected_sections": list(self.selected_sections),
            "selected_tags": list(self.selected_tags),
            "results": [result.to_dict() for result in self.results],
            KEY_METADATA: dict(self.metadata),
        }


# 30.1.3. Registry
# -----------------------------------------------------------------------------

_SELF_TEST_REGISTRY: Dict[str, SelfTestCase] = {}


def register_self_test(
    function: Callable[[], Any],
    *,
    name: Optional[str] = None,
    section: str = SELF_TEST_SECTION_INFRASTRUCTURE,
    description: Optional[str] = None,
    tags: Iterable[str] = (),
    enabled: bool = True,
    replace_existing: bool = False,
    metadata: Optional[Mapping[str, Any]] = None,
) -> SelfTestCase:
    """Register a self-test callable."""

    test_name = safe_identifier(
        name or getattr(function, "__name__", "unnamed_test"),
        "unnamed_test",
    )
    if test_name in _SELF_TEST_REGISTRY and not replace_existing:
        raise ReportSelfTestError(
            test_name,
            "Self-test name is already registered.",
        )

    case = SelfTestCase(
        name=test_name,
        section=section,
        function=function,
        description=description or doc_summary(function),
        tags=tuple(tags),
        enabled=enabled,
        metadata=metadata or {},
    )
    _SELF_TEST_REGISTRY[test_name] = case
    return case


def self_test(
    *,
    name: Optional[str] = None,
    section: str = SELF_TEST_SECTION_INFRASTRUCTURE,
    description: Optional[str] = None,
    tags: Iterable[str] = (),
    enabled: bool = True,
) -> Callable[[Callable[[], Any]], Callable[[], Any]]:
    """Decorator registering a report self-test."""

    def decorator(
        function: Callable[[], Any],
    ) -> Callable[[], Any]:
        register_self_test(
            function,
            name=name,
            section=section,
            description=description,
            tags=tags,
            enabled=enabled,
        )
        return function

    return decorator


def unregister_self_test(name: str) -> Optional[SelfTestCase]:
    """Remove and return a registered self-test."""

    return _SELF_TEST_REGISTRY.pop(str(name), None)


def get_self_test(name: str) -> SelfTestCase:
    """Return one registered self-test."""

    try:
        return _SELF_TEST_REGISTRY[str(name)]
    except KeyError as error:
        raise ReportSelfTestError(
            str(name),
            "Self-test is not registered.",
            cause=error,
        ) from error


def list_self_tests(
    *,
    sections: Iterable[str] = (),
    tags: Iterable[str] = (),
    enabled_only: bool = True,
) -> Tuple[SelfTestCase, ...]:
    """Return registered tests matching filters."""

    section_filter = {
        single_line_text(value, "")
        for value in sections
        if single_line_text(value, "")
    }
    tag_filter = {
        normalize_field_name(value)
        for value in tags
        if normalize_field_name(value)
    }

    output: List[SelfTestCase] = []
    for case in _SELF_TEST_REGISTRY.values():
        if enabled_only and not case.enabled:
            continue
        if section_filter and case.section not in section_filter:
            continue
        case_tags = {
            normalize_field_name(tag)
            for tag in case.tags
        }
        if tag_filter and not tag_filter.issubset(case_tags):
            continue
        output.append(case)

    return tuple(
        sorted(
            output,
            key=lambda case: (
                tuple(
                    to_safe_int(part, 0)
                    for part in case.section.split(".")
                ),
                case.name,
            ),
        )
    )


# 30.1.4. Assertion helpers
# -----------------------------------------------------------------------------

def _self_test_message(
    message: Optional[str],
    default: str,
) -> str:
    """Return a self-test assertion message."""

    return safe_string(message, "") or default


def self_test_fail(
    message: str,
    *,
    expected: Any = MISSING,
    actual: Any = MISSING,
    context: Optional[Mapping[str, Any]] = None,
) -> None:
    """Raise one assertion failure."""

    details = dict(context or {})
    if expected is not MISSING:
        details["expected"] = expected
    if actual is not MISSING:
        details["actual"] = actual
    raise SelfTestFailure(
        message=message,
        context=details,
    )


def self_test_skip(
    message: str,
    *,
    context: Optional[Mapping[str, Any]] = None,
) -> None:
    """Skip the current self-test."""

    raise SelfTestSkip(
        message=message,
        context=context or {},
    )


def assert_true(
    condition: Any,
    message: Optional[str] = None,
) -> None:
    """Assert that a value is truthy."""

    if not condition:
        self_test_fail(
            _self_test_message(
                message,
                "Expected condition to be true.",
            ),
            expected=True,
            actual=condition,
        )


def assert_false(
    condition: Any,
    message: Optional[str] = None,
) -> None:
    """Assert that a value is falsy."""

    if condition:
        self_test_fail(
            _self_test_message(
                message,
                "Expected condition to be false.",
            ),
            expected=False,
            actual=condition,
        )


def assert_equal(
    actual: Any,
    expected: Any,
    message: Optional[str] = None,
) -> None:
    """Assert equality."""

    if actual != expected:
        self_test_fail(
            _self_test_message(
                message,
                "Values are not equal.",
            ),
            expected=expected,
            actual=actual,
        )


def assert_not_equal(
    actual: Any,
    expected: Any,
    message: Optional[str] = None,
) -> None:
    """Assert inequality."""

    if actual == expected:
        self_test_fail(
            _self_test_message(
                message,
                "Values unexpectedly match.",
            ),
            expected=f"not {expected!r}",
            actual=actual,
        )


def assert_almost_equal(
    actual: Any,
    expected: Any,
    *,
    relative_tolerance: float = SELF_TEST_DEFAULT_RELATIVE_TOLERANCE,
    absolute_tolerance: float = SELF_TEST_DEFAULT_TOLERANCE,
    message: Optional[str] = None,
) -> None:
    """Assert numeric proximity."""

    first = to_finite_float(actual, MISSING)
    second = to_finite_float(expected, MISSING)
    if (
        first is MISSING
        or second is MISSING
        or not math.isclose(
            first,
            second,
            rel_tol=max(0.0, relative_tolerance),
            abs_tol=max(0.0, absolute_tolerance),
        )
    ):
        self_test_fail(
            _self_test_message(
                message,
                "Numeric values are not sufficiently close.",
            ),
            expected=expected,
            actual=actual,
            context={
                "relative_tolerance": relative_tolerance,
                "absolute_tolerance": absolute_tolerance,
            },
        )


def assert_is(
    actual: Any,
    expected: Any,
    message: Optional[str] = None,
) -> None:
    """Assert object identity."""

    if actual is not expected:
        self_test_fail(
            _self_test_message(
                message,
                "Objects are not identical.",
            ),
            expected=repr(expected),
            actual=repr(actual),
        )


def assert_is_instance(
    value: Any,
    expected_type: Any,
    message: Optional[str] = None,
) -> None:
    """Assert an object's type."""

    if not isinstance(value, expected_type):
        self_test_fail(
            _self_test_message(
                message,
                "Object has an unexpected type.",
            ),
            expected=getattr(
                expected_type,
                "__name__",
                repr(expected_type),
            ),
            actual=type(value).__name__,
        )


def assert_contains(
    container: Any,
    member: Any,
    message: Optional[str] = None,
) -> None:
    """Assert membership."""

    try:
        found = member in container
    except Exception:
        found = False
    if not found:
        self_test_fail(
            _self_test_message(
                message,
                "Expected member was not found.",
            ),
            expected=member,
            actual=container,
        )


def assert_sequence_equal(
    actual: Iterable[Any],
    expected: Iterable[Any],
    message: Optional[str] = None,
) -> None:
    """Assert sequence equality after tuple conversion."""

    assert_equal(
        tuple(actual),
        tuple(expected),
        message or "Sequences are not equal.",
    )


def assert_mapping_has_keys(
    mapping: Any,
    keys: Iterable[Any],
    message: Optional[str] = None,
) -> None:
    """Assert that a mapping contains all requested keys."""

    if not isinstance(mapping, Mapping):
        self_test_fail(
            _self_test_message(
                message,
                "Value is not a mapping.",
            ),
            expected="Mapping",
            actual=type(mapping).__name__,
        )
    missing = [
        key for key in keys if key not in mapping
    ]
    if missing:
        self_test_fail(
            _self_test_message(
                message,
                "Mapping is missing required keys.",
            ),
            expected=list(keys),
            actual=list(mapping),
            context={"missing": missing},
        )


def assert_raises(
    expected_exception: Any,
    function: Callable[..., Any],
    *args: Any,
    message: Optional[str] = None,
    **kwargs: Any,
) -> BaseException:
    """Assert that a callable raises an expected exception."""

    try:
        function(*args, **kwargs)
    except expected_exception as error:
        return error
    except Exception as error:
        self_test_fail(
            _self_test_message(
                message,
                "Callable raised an unexpected exception.",
            ),
            expected=getattr(
                expected_exception,
                "__name__",
                repr(expected_exception),
            ),
            actual=type(error).__name__,
            context={"message": _exception_message(error)},
        )
    self_test_fail(
        _self_test_message(
            message,
            "Callable did not raise.",
        ),
        expected=getattr(
            expected_exception,
            "__name__",
            repr(expected_exception),
        ),
        actual="no exception",
    )
    raise AssertionError("unreachable")


# 30.1.5. Synthetic molecular objects
# -----------------------------------------------------------------------------

@dataclass
class SelfTestChain:
    """Minimal chain used by report self-tests."""

    chain_id: str
    structure: Any = None


@dataclass
class SelfTestResidue:
    """Minimal residue used by report self-tests."""

    name: str
    number: Any
    chain: SelfTestChain
    structure: Any = None
    insertion_code: str = ""

    @property
    def chain_id(self) -> str:
        return self.chain.chain_id


@dataclass
class SelfTestAtom:
    """Minimal atom used by report self-tests."""

    name: str
    residue: SelfTestResidue
    serial_number: Optional[int] = None
    structure: Any = None


@dataclass
class SelfTestInteractionObject:
    """Object-style interaction used for normalization tests."""

    type: str
    ligand_atom: Any
    receptor_atom: Any
    distance: Optional[float] = None
    angle: Optional[float] = None
    subtype: str = ""
    strength: str = STRENGTH_UNKNOWN
    score: Optional[float] = None
    classification: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class SelfTestPose:
    """Minimal DockModel-like pose for report tests."""

    pose_id: Any = 1
    pose_name: str = "Self-test pose"
    model_id: Any = "self-test-model"
    model_name: str = "Self-test model"
    ligand_name: str = "LIG"
    receptor_name: str = "REC"
    affinity: Optional[float] = -7.0
    total_score: Optional[float] = 3.0
    contacts: List[Any] = field(default_factory=list)
    hbonds: List[Any] = field(default_factory=list)
    hydrophobic: List[Any] = field(default_factory=list)
    pi: List[Any] = field(default_factory=list)
    saltbridge: List[Any] = field(default_factory=list)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    scoring: Any = None


def make_self_test_atoms() -> Mapping[str, Any]:
    """Create a compact ligand/receptor atom system."""

    ligand_chain = SelfTestChain("L")
    receptor_chain = SelfTestChain("A")
    ligand_residue = SelfTestResidue(
        "LIG",
        1,
        ligand_chain,
    )
    tyr_residue = SelfTestResidue(
        "TYR",
        123,
        receptor_chain,
    )
    phe_residue = SelfTestResidue(
        "PHE",
        77,
        receptor_chain,
    )
    ligand_atom = SelfTestAtom(
        "C1",
        ligand_residue,
        serial_number=1,
    )
    receptor_atom = SelfTestAtom(
        "OH",
        tyr_residue,
        serial_number=42,
    )
    aromatic_atom = SelfTestAtom(
        "CZ",
        phe_residue,
        serial_number=55,
    )
    return MappingProxyType(
        {
            "ligand_chain": ligand_chain,
            "receptor_chain": receptor_chain,
            "ligand_residue": ligand_residue,
            "tyr_residue": tyr_residue,
            "phe_residue": phe_residue,
            "ligand_atom": ligand_atom,
            "receptor_atom": receptor_atom,
            "aromatic_atom": aromatic_atom,
        }
    )


def make_self_test_interactions() -> Mapping[str, Tuple[Any, ...]]:
    """Create representative interaction families."""

    atoms = make_self_test_atoms()
    hbond = {
        "type": "hbond",
        "subtype": "donor_acceptor",
        "ligand_atom": atoms["ligand_atom"],
        "receptor_atom": atoms["receptor_atom"],
        "distance": 2.8,
        "angle": 165.0,
        "strength": "strong",
        "score": 1.5,
    }
    contact = {
        "interaction_type": "contact",
        "ligand_atom": atoms["ligand_atom"],
        "receptor_atom": atoms["receptor_atom"],
        "distance": 3.4,
        "strength": "weak",
        "score": 0.4,
    }
    hydrophobic_interaction = SelfTestInteractionObject(
        type="hydrophobic",
        ligand_atom=atoms["ligand_atom"],
        receptor_atom=atoms["aromatic_atom"],
        distance=3.9,
        strength="moderate",
        score=0.8,
    )
    pi_interaction = {
        "type": "pi",
        "subtype": "parallel",
        "ligand_atom": atoms["ligand_atom"],
        "receptor_atom": atoms["aromatic_atom"],
        "distance": 4.8,
        "angle": 12.0,
        "strength": "moderate",
        "score": 1.1,
    }
    salt_bridge = {
        "type": "salt_bridge",
        "subtype": "attractive",
        "ligand_atom": atoms["ligand_atom"],
        "receptor_atom": atoms["receptor_atom"],
        "distance": 3.0,
        "strength": "strong",
        "score": 1.3,
    }
    clash = {
        "type": "clash",
        "ligand_atom": atoms["ligand_atom"],
        "receptor_atom": atoms["receptor_atom"],
        "distance": 1.2,
        "strength": "strong",
        "score": -2.0,
    }
    return MappingProxyType(
        {
            "contacts": (contact,),
            "hbonds": (hbond,),
            "hydrophobic": (hydrophobic_interaction,),
            "pi": (pi_interaction,),
            "saltbridge": (salt_bridge,),
            "clashes": (clash,),
        }
    )


def make_self_test_pose() -> SelfTestPose:
    """Create a populated DockModel-like pose."""

    interactions = make_self_test_interactions()
    pose = SelfTestPose(
        contacts=list(interactions["contacts"]),
        hbonds=list(interactions["hbonds"]),
        hydrophobic=list(interactions["hydrophobic"]),
        pi=list(interactions["pi"]),
        saltbridge=list(interactions["saltbridge"]),
        metadata={"engine": "self-test"},
    )
    pose.scoring = {
        "total_score": pose.total_score,
        "components": {
            "contacts": {"value": 1.0, "weight": 0.4},
            "hbonds": {"value": 1.0, "weight": 1.5},
            "hydrophobic": {"value": 1.0, "weight": 0.8},
        },
    }
    return pose


# 30.1.6. Test execution
# -----------------------------------------------------------------------------

def run_self_test_case(
    case: SelfTestCase,
    *,
    include_traceback: bool = False,
) -> SelfTestResult:
    """Execute one registered self-test."""

    if not isinstance(case, SelfTestCase):
        raise ReportConfigurationError(
            "case must be SelfTestCase."
        )
    if not case.enabled:
        return SelfTestResult(
            name=case.name,
            section=case.section,
            status=SelfTestStatus.SKIP,
            message="Test is disabled.",
        )

    from time import perf_counter

    started = perf_counter()
    try:
        returned = case.function()
    except SelfTestSkip as error:
        status = SelfTestStatus.SKIP
        message = error.message
        exception_type = type(error).__name__
        details = dict(error.context)
        traceback_text = ""
    except SelfTestFailure as error:
        status = SelfTestStatus.FAIL
        message = error.message
        exception_type = type(error).__name__
        details = dict(error.context)
        traceback_text = (
            "".join(
                traceback.format_exception(
                    type(error),
                    error,
                    error.__traceback__,
                )
            )
            if include_traceback
            else ""
        )
    except Exception as error:
        status = SelfTestStatus.ERROR
        message = _exception_message(error)
        exception_type = type(error).__name__
        details = {}
        traceback_text = (
            "".join(
                traceback.format_exception(
                    type(error),
                    error,
                    error.__traceback__,
                )
            )
            if include_traceback
            else ""
        )
    else:
        status = SelfTestStatus.PASS
        message = ""
        exception_type = ""
        traceback_text = ""
        details = (
            dict(returned)
            if isinstance(returned, Mapping)
            else {}
        )

    return SelfTestResult(
        name=case.name,
        section=case.section,
        status=status,
        duration_seconds=perf_counter() - started,
        message=message,
        exception_type=exception_type,
        traceback_text=traceback_text,
        details=details,
    )


def run_registered_self_tests(
    *,
    sections: Iterable[str] = (),
    tags: Iterable[str] = (),
    names: Iterable[str] = (),
    include_tracebacks: bool = False,
    stop_on_failure: bool = False,
    raise_on_failure: bool = False,
    enabled_only: bool = True,
) -> SelfTestReport:
    """Run selected registered report self-tests."""

    from time import perf_counter

    started_at = current_utc_timestamp()
    started = perf_counter()
    name_filter = {
        safe_identifier(name, "")
        for name in names
        if safe_identifier(name, "")
    }
    cases = list_self_tests(
        sections=sections,
        tags=tags,
        enabled_only=enabled_only,
    )
    if name_filter:
        cases = tuple(
            case for case in cases if case.name in name_filter
        )

    results: List[SelfTestResult] = []
    for case in cases:
        result = run_self_test_case(
            case,
            include_traceback=include_tracebacks,
        )
        results.append(result)
        if stop_on_failure and result.failed:
            break

    report = SelfTestReport(
        results=tuple(results),
        started_at=started_at,
        finished_at=current_utc_timestamp(),
        duration_seconds=perf_counter() - started,
        selected_sections=tuple(sections),
        selected_tags=tuple(tags),
        metadata={
            "registered_tests": len(_SELF_TEST_REGISTRY),
            "selected_tests": len(cases),
        },
    )
    if raise_on_failure:
        report.raise_for_failures()
    return report


def run_report_self_tests(
    *,
    sections: Iterable[str] = (
        SELF_TEST_SECTION_INFRASTRUCTURE,
        SELF_TEST_SECTION_FORMATTING,
        SELF_TEST_SECTION_INTERACTIONS,
        SELF_TEST_SECTION_SINGLE_POSE,
        SELF_TEST_SECTION_MULTIPOSE,
        SELF_TEST_SECTION_RENDERING,
        SELF_TEST_SECTION_WRITING_EXPORT,
        SELF_TEST_SECTION_FINAL_RUNNER,
    ),
    **kwargs: Any,
) -> SelfTestReport:
    """Run the currently implemented report self-tests."""

    return run_registered_self_tests(
        sections=sections,
        **kwargs,
    )


def self_test_result_rows(
    report: SelfTestReport,
) -> ReportRows:
    """Return self-test results as table rows."""

    return [
        {
            KEY_RANK: index,
            "name": result.name,
            "section": result.section,
            "status": result.status.value,
            "duration_seconds": result.duration_seconds,
            "message": result.message,
            "exception_type": result.exception_type,
        }
        for index, result in enumerate(report.results, start=1)
    ]


def format_self_test_report(
    report: SelfTestReport,
) -> str:
    """Return a compact text summary of a test run."""

    header = (
        f"Report self-tests: "
        f"{report.passed}/{report.total} passed, "
        f"{report.failed} failed, "
        f"{report.errors} errors, "
        f"{report.skipped} skipped"
    )
    failed_lines = [
        (
            f"- {result.section} {result.name}: "
            f"{result.status.value}"
            + (f" — {result.message}" if result.message else "")
        )
        for result in report.results
        if result.failed
    ]
    return DEFAULT_NEWLINE.join((header, *failed_lines))


# 30.1.7. Infrastructure self-tests
# -----------------------------------------------------------------------------

@self_test(
    section=SELF_TEST_SECTION_INFRASTRUCTURE,
    tags=("infrastructure", "assertions"),
)
def test_self_test_assertion_helpers() -> None:
    """Verify core assertion helper behavior."""

    assert_true(True)
    assert_false(False)
    assert_equal({"a": 1}, {"a": 1})
    assert_not_equal(1, 2)
    assert_almost_equal(0.1 + 0.2, 0.3)
    marker = object()
    assert_is(marker, marker)
    assert_is_instance([], list)
    assert_contains(("a", "b"), "b")
    assert_sequence_equal([1, 2], (1, 2))
    assert_mapping_has_keys({"a": 1, "b": 2}, ("a", "b"))
    error = assert_raises(ValueError, int, "not-an-int")
    assert_is_instance(error, ValueError)


@self_test(
    section=SELF_TEST_SECTION_INFRASTRUCTURE,
    tags=("infrastructure", "registry"),
)
def test_self_test_registry_helpers() -> None:
    """Verify registry lookup and filtering."""

    current = get_self_test("test_self_test_registry_helpers")
    assert_equal(current.section, SELF_TEST_SECTION_INFRASTRUCTURE)
    assert_contains(current.tags, "registry")
    infrastructure = list_self_tests(
        sections=(SELF_TEST_SECTION_INFRASTRUCTURE,)
    )
    assert_true(len(infrastructure) >= 2)
    assert_true(
        all(
            case.section == SELF_TEST_SECTION_INFRASTRUCTURE
            for case in infrastructure
        )
    )


@self_test(
    section=SELF_TEST_SECTION_INFRASTRUCTURE,
    tags=("infrastructure", "fixtures"),
)
def test_self_test_synthetic_objects() -> None:
    """Verify reusable molecular fixtures."""

    atoms = make_self_test_atoms()
    assert_mapping_has_keys(
        atoms,
        (
            "ligand_atom",
            "receptor_atom",
            "aromatic_atom",
        ),
    )
    assert_equal(
        atoms["receptor_atom"].residue.chain_id,
        "A",
    )
    interactions = make_self_test_interactions()
    assert_equal(len(interactions["hbonds"]), 1)
    assert_equal(len(interactions["pi"]), 1)
    pose = make_self_test_pose()
    assert_equal(len(pose.contacts), 1)
    assert_equal(len(pose.hbonds), 1)
    assert_equal(pose.metadata["engine"], "self-test")


@self_test(
    section=SELF_TEST_SECTION_INFRASTRUCTURE,
    tags=("infrastructure", "records"),
)
def test_self_test_records() -> None:
    """Verify result and report accounting."""

    result = SelfTestResult(
        name="sample",
        section=SELF_TEST_SECTION_INFRASTRUCTURE,
        status=SelfTestStatus.PASS,
        duration_seconds=0.01,
    )
    report = SelfTestReport(results=(result,))
    assert_true(result.passed)
    assert_false(result.failed)
    assert_true(report.successful)
    assert_equal(report.total, 1)
    assert_equal(report.passed, 1)
    assert_mapping_has_keys(
        report.to_dict(),
        ("successful", "results", "passed"),
    )


# 30.1.8. Public infrastructure interface
# -----------------------------------------------------------------------------

_SECTION_30_1_PUBLIC_NAMES: Final[Tuple[str, ...]] = (
    "SELF_TEST_SECTION_INFRASTRUCTURE",
    "SELF_TEST_SECTION_FORMATTING",
    "SELF_TEST_SECTION_INTERACTIONS",
    "SELF_TEST_SECTION_SINGLE_POSE",
    "SELF_TEST_SECTION_MULTIPOSE",
    "SELF_TEST_SECTION_RENDERING",
    "SELF_TEST_SECTION_WRITING_EXPORT",
    "SELF_TEST_SECTION_FINAL_RUNNER",
    "SelfTestStatus",
    "SelfTestFailure",
    "SelfTestSkip",
    "SelfTestCase",
    "SelfTestResult",
    "SelfTestReport",
    "register_self_test",
    "self_test",
    "unregister_self_test",
    "get_self_test",
    "list_self_tests",
    "self_test_fail",
    "self_test_skip",
    "assert_true",
    "assert_false",
    "assert_equal",
    "assert_not_equal",
    "assert_almost_equal",
    "assert_is",
    "assert_is_instance",
    "assert_contains",
    "assert_sequence_equal",
    "assert_mapping_has_keys",
    "assert_raises",
    "SelfTestChain",
    "SelfTestResidue",
    "SelfTestAtom",
    "SelfTestInteractionObject",
    "SelfTestPose",
    "make_self_test_atoms",
    "make_self_test_interactions",
    "make_self_test_pose",
    "run_self_test_case",
    "run_registered_self_tests",
    "run_report_self_tests",
    "self_test_result_rows",
    "format_self_test_report",
)

_register_public_names(_SECTION_30_1_PUBLIC_NAMES)

# =============================================================================
# End of Section 30.1
# =============================================================================


# =============================================================================
# Section 30.2 — Formatting and helper self-tests
# =============================================================================

@self_test(
    section=SELF_TEST_SECTION_FORMATTING,
    tags=("formatting", "numbers"),
)
def test_formatting_numeric_conversion() -> None:
    """Verify safe numeric conversion and zero normalization."""

    assert_almost_equal(to_finite_float("1.25"), 1.25)
    assert_equal(to_finite_float(float("nan"), None), None)
    assert_equal(to_finite_float(float("inf"), None), None)
    assert_equal(to_safe_int("7"), 7)
    assert_equal(to_safe_int("7.5", None), None)
    assert_equal(to_safe_int(True, None), None)
    assert_equal(normalize_negative_zero(-0.0), 0.0)
    assert_equal(normalize_negative_zero(1e-20), 0.0)


@self_test(
    section=SELF_TEST_SECTION_FORMATTING,
    tags=("formatting", "text"),
)
def test_formatting_safe_text() -> None:
    """Verify safe text conversion and whitespace handling."""

    class BrokenString:
        def __str__(self) -> str:
            raise RuntimeError("broken str")

        def __repr__(self) -> str:
            return "<broken>"

    assert_equal(safe_string(None, "missing"), "missing")
    assert_equal(safe_string(b"abc"), "abc")
    assert_equal(safe_string(Path("a/b")), "a/b")
    assert_equal(safe_string(BrokenString()), "<broken>")
    assert_equal(
        single_line_text("  alpha \n beta\tgamma  "),
        "alpha beta gamma",
    )
    assert_equal(
        safe_string("a\x00b", remove_controls=True),
        "ab",
    )


@self_test(
    section=SELF_TEST_SECTION_FORMATTING,
    tags=("formatting", "labels"),
)
def test_formatting_truncation_and_labels() -> None:
    """Verify truncation, identifiers and labels."""

    assert_equal(truncate_text("abcdefgh", 5), "abcd…")
    assert_equal(truncate_text("abc", 0), "")
    assert_equal(
        safe_identifier("  alpha beta  "),
        "alpha beta",
    )
    assert_equal(title_case_label("total_score"), "Total score")
    assert_equal(field_label(KEY_TOTAL_SCORE), "Total score")
    assert_equal(normalize_field_name("Total Score"), "total_score")


@self_test(
    section=SELF_TEST_SECTION_FORMATTING,
    tags=("formatting", "numbers"),
)
def test_formatting_numeric_output() -> None:
    """Verify numbers, scores and scientific units."""

    assert_equal(format_number(1.23456, 2), "1.23")
    assert_equal(
        format_number(1.2, 3, trim_zeros=True),
        "1.2",
    )
    assert_equal(
        format_number(1234, 0, thousands=True),
        "1,234",
    )
    assert_equal(format_integer("4"), "4")
    assert_equal(format_score(2.345), "2.3450")
    assert_equal(
        format_distance(3.125, include_unit=True),
        "3.125 Å",
    )
    assert_equal(
        format_angle(165.44, include_unit=True),
        "165.4°",
    )
    assert_equal(
        format_percent(0.25, fraction=True),
        "25.0%",
    )
    assert_equal(format_range(2.0, 4.0, digits=1), "2.0–4.0")


@self_test(
    section=SELF_TEST_SECTION_FORMATTING,
    tags=("formatting", "datetime", "path"),
)
def test_formatting_datetime_and_path() -> None:
    """Verify deterministic date, time and path formatting."""

    moment = datetime(
        2026,
        7,
        28,
        12,
        30,
        15,
        tzinfo=timezone.utc,
    )
    assert_equal(
        format_datetime(moment, use_utc=True),
        "2026-07-28T12:30:15Z",
    )
    assert_equal(
        format_datetime("2026-07-28T12:30:15+00:00", use_utc=True),
        "2026-07-28T12:30:15Z",
    )
    assert_true(format_path(Path("folder/file.pdb")).endswith(
        "folder/file.pdb"
    ))
    assert_equal(format_datetime("not-a-date"), DEFAULT_MISSING_TEXT)


@self_test(
    section=SELF_TEST_SECTION_FORMATTING,
    tags=("formatting", "collections"),
)
def test_formatting_collections_and_values() -> None:
    """Verify collection and generic value formatting."""

    assert_equal(
        format_sequence([1, 2, 3]),
        "1, 2, 3",
    )
    assert_contains(
        format_sequence([1, 2, 3], max_items=2),
        DEFAULT_TRUNCATION_MARKER,
    )
    mapping_text = format_mapping({"total_score": 1.5})
    assert_contains(mapping_text, "Total score")
    assert_contains(mapping_text, "1.5")
    assert_equal(
        format_value(True),
        DEFAULT_FORMATTING_CONFIG.true_text,
    )
    assert_equal(
        format_value(None),
        DEFAULT_FORMATTING_CONFIG.missing_text,
    )
    assert_equal(
        SafeFormatter().distance(2.5, include_unit=True),
        "2.500 Å",
    )


@self_test(
    section=SELF_TEST_SECTION_FORMATTING,
    tags=("formatting", "escaping"),
)
def test_formatting_escaping() -> None:
    """Verify Markdown, HTML and JSON escaping."""

    markdown = escape_markdown("*a|b*")
    assert_contains(markdown, r"\*")
    assert_contains(
        escape_markdown_cell("a|b"),
        r"\|",
    )
    assert_equal(escape_html("<tag>"), "&lt;tag&gt;")
    assert_equal(
        escape_json_string('a"b'),
        r'a\"b',
    )


@self_test(
    section=SELF_TEST_SECTION_FORMATTING,
    tags=("helpers", "object_access"),
)
def test_helper_object_access() -> None:
    """Verify tolerant field access and nested paths."""

    class AccessObject:
        def __init__(self) -> None:
            self.total_score = 3.5
            self.child = {
                "items": [
                    {"value": 7},
                ]
            }

        def zero_argument(self) -> str:
            return "called"

        def requires_argument(self, value: Any) -> Any:
            return value

    obj = AccessObject()
    assert_equal(get_object_field(obj, "total_score"), 3.5)
    assert_true(has_object_field(obj, "TOTAL SCORE"))
    assert_equal(
        get_object_path(obj, "child.items[0].value"),
        7,
    )
    assert_equal(
        get_object_field(
            obj,
            "zero_argument",
            call=True,
        ),
        "called",
    )
    assert_equal(
        get_object_field(
            obj,
            "requires_argument",
            "fallback",
            call=True,
        ),
        "fallback",
    )
    assert_raises(
        MissingReportFieldError,
        require_object_path,
        obj,
        "child.missing",
    )


@self_test(
    section=SELF_TEST_SECTION_FORMATTING,
    tags=("helpers", "enums", "config"),
)
def test_helper_enums_and_configuration() -> None:
    """Verify enum coercion and immutable configuration."""

    assert_is(ReportFormat.coerce("md"), ReportFormat.MARKDOWN)
    assert_is(ReportDetail.coerce("verbose"), ReportDetail.FULL)
    assert_is(
        InteractionFamily.coerce("hbonds"),
        InteractionFamily.HYDROGEN_BOND,
    )
    assert_true(
        Severity.CRITICAL.weight > Severity.WARNING.weight
    )
    config = FormattingConfig(float_digits=4)
    assert_equal(config.float_digits, 4)
    assert_raises(
        (AttributeError, TypeError),
        setattr,
        config,
        "float_digits",
        2,
    )


@self_test(
    section=SELF_TEST_SECTION_FORMATTING,
    tags=("helpers", "filename"),
)
def test_helper_filename_and_format_resolution() -> None:
    """Verify safe filenames and format inference."""

    assert_equal(
        sanitize_filename(' bad:name?.txt '),
        "bad_name_.txt",
    )
    assert_equal(
        normalize_report_format("md"),
        ReportFormat.MARKDOWN,
    )
    assert_equal(
        infer_report_format_from_path("report.html"),
        ReportFormat.HTML,
    )
    assert_equal(
        ensure_report_suffix("report.txt", "json").name,
        "report.json",
    )


# 30.2. Public formatting/helper test interface
# -----------------------------------------------------------------------------

def run_formatting_self_tests(
    **kwargs: Any,
) -> SelfTestReport:
    """Run Section 30.2 self-tests."""

    return run_registered_self_tests(
        sections=(SELF_TEST_SECTION_FORMATTING,),
        **kwargs,
    )


_SECTION_30_2_PUBLIC_NAMES: Final[Tuple[str, ...]] = (
    "run_formatting_self_tests",
)

_register_public_names(_SECTION_30_2_PUBLIC_NAMES)

# =============================================================================
# End of Section 30.2
# =============================================================================


# =============================================================================
# Section 30.3 — Interaction self-tests
# =============================================================================

@self_test(
    section=SELF_TEST_SECTION_INTERACTIONS,
    tags=("interactions", "family"),
)
def test_interaction_family_normalization() -> None:
    """Verify aliases and family inference."""

    assert_is(
        normalize_interaction_family("hbond"),
        InteractionFamily.HYDROGEN_BOND,
    )
    assert_is(
        normalize_interaction_family("saltbridge"),
        InteractionFamily.SALT_BRIDGE,
    )
    assert_is(
        normalize_interaction_family("unknown value"),
        InteractionFamily.UNKNOWN,
    )
    assert_is(
        infer_interaction_family(
            {"type": "pi"},
        ),
        InteractionFamily.PI,
    )
    assert_is(
        infer_interaction_family(
            {"distance": 3.2},
            family_hint="contact",
        ),
        InteractionFamily.CONTACT,
    )


@self_test(
    section=SELF_TEST_SECTION_INTERACTIONS,
    tags=("interactions", "normalization", "mapping"),
)
def test_interaction_mapping_normalization() -> None:
    """Verify complete mapping-based normalization."""

    raw = make_self_test_interactions()["hbonds"][0]
    normalized = normalize_interaction(
        raw,
        pose_id=1,
        model_id="model-1",
        source="self-test",
        include_metadata=True,
    )
    assert_is_instance(normalized, NormalizedInteraction)
    assert_is(
        normalized.family,
        InteractionFamily.HYDROGEN_BOND,
    )
    assert_equal(normalized.type, "hydrogen_bond")
    assert_equal(normalized.subtype, "donor_acceptor")
    assert_equal(normalized.pose_id, 1)
    assert_equal(normalized.model_id, "model-1")
    assert_equal(normalized.source, "self-test")
    assert_equal(normalized.ligand_atom, "C1#1")
    assert_equal(normalized.receptor_atom, "OH#42")
    assert_equal(normalized.ligand_residue, "L:LIG1")
    assert_equal(normalized.receptor_residue, "TYR123")
    assert_equal(normalized.chain_id, "A")
    assert_almost_equal(normalized.distance, 2.8)
    assert_almost_equal(normalized.angle, 165.0)
    assert_almost_equal(normalized.score, 1.5)
    assert_true(bool(normalized.id))


@self_test(
    section=SELF_TEST_SECTION_INTERACTIONS,
    tags=("interactions", "normalization", "object"),
)
def test_interaction_object_normalization() -> None:
    """Verify object-style interaction normalization."""

    raw = make_self_test_interactions()["hydrophobic"][0]
    normalized = normalize_interaction(raw)
    assert_is(
        normalized.family,
        InteractionFamily.HYDROPHOBIC,
    )
    assert_equal(normalized.type, "hydrophobic")
    assert_equal(normalized.receptor_residue, "PHE77")
    assert_almost_equal(normalized.distance, 3.9)
    assert_equal(normalized.strength, "moderate")


@self_test(
    section=SELF_TEST_SECTION_INTERACTIONS,
    tags=("interactions", "identity"),
)
def test_interaction_identity_and_fingerprint() -> None:
    """Verify deterministic IDs and fingerprints."""

    raw = make_self_test_interactions()["hbonds"][0]
    first = normalize_interaction(raw, pose_id=1)
    second = normalize_interaction(dict(raw), pose_id=1)
    assert_equal(first.id, second.id)
    assert_equal(first.fingerprint(), second.fingerprint())
    assert_equal(
        make_interaction_id(
            interaction_fingerprint_data(
                family=first.family,
                interaction_type=first.type,
                subtype=first.subtype,
                pose_id=first.pose_id,
                model_id=first.model_id,
                ligand_atom=first.ligand_atom,
                receptor_atom=first.receptor_atom,
                ligand_residue=first.ligand_residue,
                receptor_residue=first.receptor_residue,
                chain_id=first.chain_id,
                distance=first.distance,
                angle=first.angle,
            ),
            prefix=first.family.value,
        ),
        first.id,
    )


@self_test(
    section=SELF_TEST_SECTION_INTERACTIONS,
    tags=("interactions", "collection", "deduplication"),
)
def test_interaction_collection_normalization() -> None:
    """Verify collection normalization, sorting and deduplication."""

    raw = make_self_test_interactions()["hbonds"][0]
    normalized = normalize_interactions(
        [raw, dict(raw)],
        pose_id=2,
    )
    assert_equal(len(normalized), 1)
    assert_equal(normalized[0].pose_id, 2)

    all_raw = make_self_test_interactions()
    combined = normalize_interaction_input(all_raw)
    assert_equal(len(combined), 6)
    families = {
        item.family for item in combined
    }
    assert_contains(families, InteractionFamily.CONTACT)
    assert_contains(families, InteractionFamily.HYDROGEN_BOND)
    assert_contains(families, InteractionFamily.HYDROPHOBIC)
    assert_contains(families, InteractionFamily.PI)
    assert_contains(families, InteractionFamily.SALT_BRIDGE)
    assert_contains(families, InteractionFamily.CLASH)
    assert_equal(
        combined,
        sorted(combined, key=interaction_sort_key),
    )


@self_test(
    section=SELF_TEST_SECTION_INTERACTIONS,
    tags=("interactions", "containers", "pose"),
)
def test_interaction_pose_containers() -> None:
    """Verify DockModel-like interaction container discovery."""

    pose = make_self_test_pose()
    containers = list(iter_interaction_containers(pose))
    families = {family for family, _ in containers}
    assert_contains(families, InteractionFamily.CONTACT)
    assert_contains(families, InteractionFamily.HYDROGEN_BOND)
    assert_contains(families, InteractionFamily.HYDROPHOBIC)
    assert_contains(families, InteractionFamily.PI)
    assert_contains(families, InteractionFamily.SALT_BRIDGE)

    normalized = normalize_interaction_input(pose)
    assert_equal(len(normalized), 5)
    assert_true(
        all(item.pose_id == pose.pose_id for item in normalized)
    )
    assert_true(
        all(item.model_id == pose.model_id for item in normalized)
    )


@self_test(
    section=SELF_TEST_SECTION_INTERACTIONS,
    tags=("interactions", "grouping", "statistics"),
)
def test_interaction_grouping_and_statistics() -> None:
    """Verify family/type/residue counts and distance statistics."""

    values = normalize_interaction_input(
        make_self_test_interactions()
    )
    family_counts = interaction_family_counts(values)
    assert_equal(family_counts["contact"], 1)
    assert_equal(family_counts["hydrogen_bond"], 1)
    assert_equal(family_counts["clash"], 1)

    grouped = group_interactions_by_family(values)
    assert_equal(
        len(grouped[InteractionFamily.HYDROGEN_BOND]),
        1,
    )
    type_counts = interaction_type_counts(values)
    assert_equal(sum(type_counts.values()), len(values))
    residue_counts = interaction_residue_counts(values)
    assert_equal(residue_counts["TYR123"], 4)
    assert_equal(residue_counts["PHE77"], 2)

    statistics = interaction_distance_statistics(values)
    assert_equal(statistics["count"], 6)
    assert_almost_equal(statistics["minimum"], 1.2)
    assert_almost_equal(statistics["maximum"], 4.8)
    assert_almost_equal(
        statistics["mean"],
        sum((3.4, 2.8, 3.9, 4.8, 3.0, 1.2)) / 6,
    )


@self_test(
    section=SELF_TEST_SECTION_INTERACTIONS,
    tags=("interactions", "serialization"),
)
def test_interaction_serialization() -> None:
    """Verify interaction dictionary conversion."""

    normalized = normalize_interaction(
        make_self_test_interactions()["pi"][0],
        include_metadata=True,
    )
    record = normalized.to_dict(
        include_metadata=True,
        include_empty=False,
    )
    assert_mapping_has_keys(
        record,
        (
            KEY_ID,
            KEY_FAMILY,
            KEY_TYPE,
            KEY_RECEPTOR_RESIDUE,
            KEY_DISTANCE,
        ),
    )
    assert_equal(record[KEY_FAMILY], "pi")
    assert_equal(record[KEY_RECEPTOR_RESIDUE], "PHE77")

    records = normalized_interactions_to_dicts(
        (normalized,),
    )
    assert_equal(len(records), 1)
    assert_equal(records[0][KEY_ID], normalized.id)


@self_test(
    section=SELF_TEST_SECTION_INTERACTIONS,
    tags=("interactions", "errors"),
)
def test_interaction_strict_and_permissive_errors() -> None:
    """Verify strict and permissive normalization behavior."""

    assert_raises(
        InteractionNormalizationError,
        normalize_interaction,
        None,
        strict=True,
    )

    errors: List[ReportError] = []
    values = normalize_interactions(
        [None, make_self_test_interactions()["contacts"][0]],
        strict=False,
        errors=errors,
    )
    assert_equal(len(values), 1)
    assert_equal(len(errors), 1)
    assert_is_instance(
        errors[0],
        InteractionNormalizationError,
    )

    assert_raises(
        InteractionNormalizationError,
        normalize_interactions,
        [None],
        strict=True,
    )


@self_test(
    section=SELF_TEST_SECTION_INTERACTIONS,
    tags=("interactions", "section", "rows"),
)
def test_interaction_section_and_rows() -> None:
    """Verify structured interaction summaries and rows."""

    section = build_interaction_section(
        make_self_test_interactions()
    )
    assert_equal(section.total_interactions, 6)
    assert_equal(section.favorable_interactions, 5)
    assert_equal(section.penalty_interactions, 1)
    assert_equal(section.residue_count, 2)
    assert_almost_equal(section.total_score, 3.1)

    rows = interaction_rows(section)
    assert_equal(len(rows), 6)
    assert_mapping_has_keys(
        rows[0],
        (
            KEY_FAMILY,
            KEY_TYPE,
            KEY_DISTANCE,
            KEY_SCORE,
        ),
    )
    family_rows = interaction_family_rows(section)
    assert_equal(
        sum(row[KEY_COUNT] for row in family_rows),
        6,
    )
    split = split_interaction_rows_by_family(section)
    assert_contains(split, "hydrogen_bond")
    assert_contains(split, "clash")


@self_test(
    section=SELF_TEST_SECTION_INTERACTIONS,
    tags=("interactions", "validation"),
)
def test_interaction_validation() -> None:
    """Verify interaction validation accepts valid records."""

    valid = normalize_interaction(
        make_self_test_interactions()["hbonds"][0]
    )
    validation = validate_normalized_interaction(valid)
    assert_true(validation.valid)

    invalid = NormalizedInteraction(
        id="",
        family=InteractionFamily.UNKNOWN,
        type="",
        distance=-1.0,
    )
    invalid_validation = validate_normalized_interaction(invalid)
    assert_false(invalid_validation.valid)
    assert_true(len(invalid_validation.errors) >= 2)


# 30.3. Public interaction-test interface
# -----------------------------------------------------------------------------

def run_interaction_self_tests(
    **kwargs: Any,
) -> SelfTestReport:
    """Run Section 30.3 self-tests."""

    return run_registered_self_tests(
        sections=(SELF_TEST_SECTION_INTERACTIONS,),
        **kwargs,
    )


_SECTION_30_3_PUBLIC_NAMES: Final[Tuple[str, ...]] = (
    "run_interaction_self_tests",
)

_register_public_names(_SECTION_30_3_PUBLIC_NAMES)

# =============================================================================
# End of Section 30.3
# =============================================================================

# =============================================================================
# Section 30.4 — Single-pose self-tests
# =============================================================================

# 30.4.1. Single-pose fixtures
# -----------------------------------------------------------------------------

def _make_single_pose_diagnostics_fixture() -> SelfTestPose:
    """Create a pose carrying warning and error messages."""

    pose = make_self_test_pose()
    pose.warnings = ["synthetic warning"]
    pose.errors = ["synthetic error"]
    return pose


# 30.4.2. Pose overview tests
# -----------------------------------------------------------------------------

@self_test(
    section=SELF_TEST_SECTION_SINGLE_POSE,
    tags=("pose", "overview"),
)
def test_single_pose_overview_values() -> None:
    """Verify core single-pose overview values."""

    pose = make_self_test_pose()
    overview = summarize_pose(pose)
    assert_is_instance(overview, PoseOverview)
    assert_equal(overview.pose_id, 1)
    assert_equal(overview.pose_name, "Self-test pose")
    assert_equal(overview.model_id, "self-test-model")
    assert_equal(overview.model_name, "Self-test model")
    assert_equal(overview.ligand_name, "LIG")
    assert_equal(overview.receptor_name, "REC")
    assert_almost_equal(overview.affinity, -7.0)
    assert_almost_equal(overview.total_score, 3.0)
    assert_equal(overview.interaction_count, 5)
    assert_equal(overview.residue_count, 2)
    assert_equal(overview.favorable_count, 5)
    assert_equal(overview.penalty_count, 0)
    assert_equal(overview.family_counts["hydrogen_bond"], 1)
    assert_equal(overview.distance_statistics["count"], 5)


@self_test(
    section=SELF_TEST_SECTION_SINGLE_POSE,
    tags=("pose", "overview", "serialization"),
)
def test_single_pose_overview_rows_and_text() -> None:
    """Verify overview serialization and compact text."""

    overview = summarize_pose(make_self_test_pose())
    record = pose_overview_to_dict(overview)
    assert_mapping_has_keys(
        record,
        (
            KEY_POSE_ID,
            KEY_MODEL_ID,
            KEY_AFFINITY,
            KEY_TOTAL_SCORE,
            KEY_TOTAL_INTERACTIONS,
            KEY_TOTAL_RESIDUES,
        ),
    )
    rows = pose_overview_rows(overview)
    assert_true(len(rows) >= 10)
    assert_equal(rows[0]["key"], KEY_POSE_ID)
    text = pose_overview_text(overview)
    assert_contains(text, "Pose: 1")
    assert_contains(text, "Total interactions: 5")
    assert_contains(text, "Total residues: 2")


@self_test(
    section=SELF_TEST_SECTION_SINGLE_POSE,
    tags=("pose", "overview", "metadata"),
)
def test_single_pose_metadata_detail_levels() -> None:
    """Verify metadata inclusion follows detail configuration."""

    pose = make_self_test_pose()
    standard = summarize_pose(pose)
    assert_equal(dict(standard.metadata), {})

    detailed_config = DEFAULT_REPORT_CONFIG.with_updates(
        rendering=DEFAULT_RENDER_CONFIG.with_updates(
            detail=ReportDetail.DETAILED,
        )
    )
    detailed = summarize_pose(
        pose,
        config=detailed_config,
    )
    assert_equal(detailed.metadata["engine"], "self-test")

    explicit = build_pose_overview(
        pose,
        include_metadata=True,
    )
    assert_equal(explicit.metadata["engine"], "self-test")


# 30.4.3. Residue, hotspot and scoring tests
# -----------------------------------------------------------------------------

@self_test(
    section=SELF_TEST_SECTION_SINGLE_POSE,
    tags=("pose", "residues"),
)
def test_single_pose_residue_summary() -> None:
    """Verify aggregation by receptor residue."""

    summaries = summarize_residues(make_self_test_pose())
    assert_equal(len(summaries), 2)
    assert_equal(summaries[0].rank, 1)
    assert_equal(summaries[0].residue, "A:TYR123")
    assert_equal(summaries[0].interaction_count, 3)
    assert_almost_equal(summaries[0].score_total, 3.2)
    assert_equal(summaries[1].residue, "A:PHE77")
    assert_equal(summaries[1].interaction_count, 2)
    assert_almost_equal(summaries[1].score_total, 1.9)

    totals = residue_summary_totals(summaries)
    assert_equal(totals[KEY_TOTAL_RESIDUES], 2)
    assert_equal(totals[KEY_TOTAL_INTERACTIONS], 5)
    assert_almost_equal(totals[KEY_TOTAL_SCORE], 5.1)

    rows = residue_summary_rows(summaries)
    assert_equal(len(rows), 2)
    assert_equal(rows[0][KEY_RECEPTOR_RESIDUE], "A:TYR123")
    assert_true(rows[0][KEY_PERCENT] > rows[1][KEY_PERCENT])


@self_test(
    section=SELF_TEST_SECTION_SINGLE_POSE,
    tags=("pose", "hotspots"),
)
def test_single_pose_hotspots() -> None:
    """Verify hotspot ranking and evidence."""

    section = summarize_hotspots(make_self_test_pose())
    assert_equal(section.selected_count, 2)
    assert_equal(section.total_candidates, 2)
    first, second = section.hotspots
    assert_equal(first.rank, 1)
    assert_equal(first.label, "A:TYR123")
    assert_true(first.hotspot_score > second.hotspot_score)
    assert_contains(first.evidence, "3 interactions")
    rows = hotspot_rows(section)
    assert_equal(len(rows), 2)
    assert_equal(rows[0][KEY_RECEPTOR_RESIDUE], "A:TYR123")
    assert_equal(rows[0][KEY_COUNT], 3)


@self_test(
    section=SELF_TEST_SECTION_SINGLE_POSE,
    tags=("pose", "scoring"),
)
def test_single_pose_scoring_and_explainability() -> None:
    """Verify existing score extraction and explanations."""

    scoring = summarize_scoring(make_self_test_pose())
    assert_is_instance(scoring, ScoringSection)
    assert_almost_equal(scoring.total_score, 3.0)
    assert_almost_equal(scoring.affinity, -7.0)
    assert_equal(len(scoring.components), 3)
    assert_equal(scoring.favorable_components, 3)
    assert_equal(scoring.unfavorable_components, 0)
    assert_equal(len(scoring.explanations), 5)
    assert_true(
        any(
            item.label == "Interaction balance"
            for item in scoring.explanations
        )
    )
    assert_true(
        any(
            item.label == "Top residues"
            for item in scoring.explanations
        )
    )
    assert_equal(len(score_component_rows(scoring)), 3)
    assert_equal(len(explanation_rows(scoring)), 5)


@self_test(
    section=SELF_TEST_SECTION_SINGLE_POSE,
    tags=("pose", "scoring", "fallback"),
)
def test_single_pose_inferred_component_score() -> None:
    """Verify score inference when no total is stored."""

    pose = make_self_test_pose()
    pose.total_score = None
    pose.scoring = {
        "components": {
            "contacts": {"value": 1.0, "weight": 0.4},
            "hbonds": {"value": 1.0, "weight": 1.5},
            "hydrophobic": {"value": 1.0, "weight": 0.8},
        }
    }
    scoring = summarize_scoring(pose)
    assert_almost_equal(scoring.total_score, 2.7)
    assert_true(scoring.warnings)
    assert_contains(
        scoring.warnings[0],
        "inferred from component contributions",
    )


# 30.4.4. Single-pose document tests
# -----------------------------------------------------------------------------

@self_test(
    section=SELF_TEST_SECTION_SINGLE_POSE,
    tags=("pose", "sections", "document"),
)
def test_single_pose_section_construction() -> None:
    """Verify all applicable single-pose sections."""

    document = build_report_document(make_self_test_pose())
    expected_visible = (
        SECTION_OVERVIEW,
        SECTION_INPUTS,
        SECTION_INTERACTIONS,
        SECTION_RESIDUES,
        SECTION_HOTSPOTS,
        SECTION_SCORING,
        SECTION_PROVENANCE,
    )
    assert_sequence_equal(
        tuple(section.id.value for section in document.visible_sections),
        expected_visible,
    )

    multipose = document.get_section(ReportSectionID.MULTIPOSE)
    assert_is_instance(multipose, ReportSection)
    assert_true(multipose.empty)
    assert_equal(len(multipose.blocks), 0)

    interactions = document.get_section(
        ReportSectionID.INTERACTIONS
    )
    assert_equal(len(interactions.visible_blocks), 3)
    assert_true(
        any(
            block.kind is ReportBlockKind.TABLE
            for block in interactions.blocks
        )
    )


@self_test(
    section=SELF_TEST_SECTION_SINGLE_POSE,
    tags=("pose", "document", "validation"),
)
def test_single_pose_document_and_convenience_api() -> None:
    """Verify high-level creation, validation and facade access."""

    pose = make_self_test_pose()
    document = create_pose_report(
        pose,
        title="Single-pose self-test",
    )
    assert_equal(document.title, "Single-pose self-test")
    validation = validate_report_document(document)
    assert_true(validation.valid)
    assert_true(validate_report(document).valid)

    summary = report_summary(document)
    assert_true(summary["valid"])
    assert_true(summary["has_overview"])
    assert_true(summary["has_interactions"])
    assert_true(summary["has_scoring"])

    facade = DockReport(pose)
    assert_is(facade.document, facade.document)
    assert_true(facade.validate().valid)
    assert_equal(
        facade.to_dict()[KEY_SCHEMA_NAME],
        REPORT_SCHEMA_NAME,
    )


@self_test(
    section=SELF_TEST_SECTION_SINGLE_POSE,
    tags=("pose", "diagnostics"),
)
def test_single_pose_warning_and_error_capture() -> None:
    """Verify pose diagnostics are preserved in the overview."""

    pose = _make_single_pose_diagnostics_fixture()
    overview = build_pose_overview(
        pose,
        strict=False,
    )
    assert_contains(overview.warnings, "synthetic warning")
    assert_contains(overview.errors, "synthetic error")

    document = build_report_document(pose)
    diagnostics = report_diagnostics(document)
    assert_true(bool(diagnostics.warnings))
    assert_true(bool(diagnostics.errors))


def run_single_pose_self_tests(
    **kwargs: Any,
) -> SelfTestReport:
    """Run Section 30.4 self-tests."""

    return run_registered_self_tests(
        sections=(SELF_TEST_SECTION_SINGLE_POSE,),
        **kwargs,
    )


_SECTION_30_4_PUBLIC_NAMES: Final[Tuple[str, ...]] = (
    "run_single_pose_self_tests",
)

_register_public_names(_SECTION_30_4_PUBLIC_NAMES)

# =============================================================================
# End of Section 30.4
# =============================================================================


# =============================================================================
# Section 30.5 — Multipose self-tests
# =============================================================================

# 30.5.1. Multipose fixtures
# -----------------------------------------------------------------------------

def _make_multipose_fixture() -> Tuple[SelfTestPose, ...]:
    """Create three poses with scores, ties and varied persistence."""

    first = make_self_test_pose()
    first.pose_id = 1
    first.pose_name = "Pose 1"
    first.model_id = "model-1"
    first.total_score = 5.0
    first.affinity = -7.5
    first.scoring = {
        "total_score": 5.0,
        "normalized_score": 0.50,
        "components": {"combined": 5.0},
    }

    second = make_self_test_pose()
    second.pose_id = 2
    second.pose_name = "Pose 2"
    second.model_id = "model-2"
    second.total_score = 3.0
    second.affinity = -8.0
    second.hydrophobic = []
    second.pi = []
    second.scoring = {
        "total_score": 3.0,
        "normalized_score": 0.30,
        "components": {"combined": 3.0},
    }

    third = make_self_test_pose()
    third.pose_id = 3
    third.pose_name = "Pose 3"
    third.model_id = "model-3"
    third.total_score = 5.0
    third.affinity = -7.0
    third.scoring = {
        "total_score": 5.0,
        "normalized_score": 0.50,
        "components": {"combined": 5.0},
    }

    return first, second, third


# 30.5.2. Ranking helpers
# -----------------------------------------------------------------------------

@self_test(
    section=SELF_TEST_SECTION_MULTIPOSE,
    tags=("multipose", "statistics", "persistence"),
)
def test_multipose_numeric_and_persistence_helpers() -> None:
    """Verify numeric and persistence helper functions."""

    statistics = numeric_statistics((1, 2, 3, None))
    assert_equal(statistics["count"], 3)
    assert_almost_equal(statistics["minimum"], 1.0)
    assert_almost_equal(statistics["maximum"], 3.0)
    assert_almost_equal(statistics["mean"], 2.0)
    assert_almost_equal(statistics["median"], 2.0)

    persistence = persistence_from_counts(
        {"A": 3, "B": 2},
        3,
    )
    assert_almost_equal(persistence["A"], 1.0)
    assert_almost_equal(persistence["B"], 2.0 / 3.0)
    assert_sequence_equal(
        consensus_from_persistence(
            persistence,
            threshold=0.75,
        ),
        ("A",),
    )


@self_test(
    section=SELF_TEST_SECTION_MULTIPOSE,
    tags=("multipose", "entry"),
)
def test_multipose_pose_ranking_entry() -> None:
    """Verify one pose ranking entry."""

    pose = _make_multipose_fixture()[0]
    entry = build_pose_ranking_entry(
        pose,
        source_index=0,
    )
    assert_is_instance(entry, PoseRankingEntry)
    assert_equal(entry.pose_id, 1)
    assert_almost_equal(entry.total_score, 5.0)
    assert_almost_equal(entry.normalized_score, 0.5)
    assert_almost_equal(entry.affinity, -7.5)
    assert_equal(entry.interaction_count, 5)
    assert_equal(entry.residue_count, 2)
    assert_equal(entry.favorable_count, 5)
    assert_equal(entry.penalty_count, 0)
    assert_equal(entry.source_index, 0)


@self_test(
    section=SELF_TEST_SECTION_MULTIPOSE,
    tags=("multipose", "ranking", "ties"),
)
def test_multipose_score_ranking_and_ties() -> None:
    """Verify total-score ranking and competition ties."""

    entries = [
        build_pose_ranking_entry(
            pose,
            source_index=index,
        )
        for index, pose in enumerate(_make_multipose_fixture())
    ]
    ranked = assign_pose_ranks(entries)
    assert_sequence_equal(
        tuple(entry.pose_id for entry in ranked),
        (1, 3, 2),
    )
    assert_sequence_equal(
        tuple(entry.rank for entry in ranked),
        (1, 1, 3),
    )
    assert_equal(ranked[0].tie_group, ranked[1].tie_group)
    assert_true(ranked[2].tie_group > ranked[1].tie_group)


@self_test(
    section=SELF_TEST_SECTION_MULTIPOSE,
    tags=("multipose", "ranking", "affinity"),
)
def test_multipose_affinity_ranking() -> None:
    """Verify lower-is-better affinity ranking."""

    summary = build_multipose_summary(
        _make_multipose_fixture(),
        ranking_metric=KEY_AFFINITY,
    )
    assert_is(
        summary.rank_direction,
        RankDirection.LOWER_IS_BETTER,
    )
    assert_equal(summary.best_pose_id, 2)
    assert_sequence_equal(
        tuple(entry.pose_id for entry in summary.poses),
        (2, 1, 3),
    )
    assert_sequence_equal(
        tuple(entry.rank for entry in summary.poses),
        (1, 2, 3),
    )


# 30.5.3. Consensus and summary tests
# -----------------------------------------------------------------------------

@self_test(
    section=SELF_TEST_SECTION_MULTIPOSE,
    tags=("multipose", "consensus", "persistence"),
)
def test_multipose_consensus_and_family_persistence() -> None:
    """Verify residue and family persistence across poses."""

    poses = _make_multipose_fixture()
    residue_counts = multipose_residue_counts(poses)
    assert_equal(residue_counts["A:TYR123"], 3)
    assert_equal(residue_counts["A:PHE77"], 2)

    family_counts = multipose_family_counts(poses)
    assert_equal(family_counts["hydrogen_bond"], 3)
    assert_equal(family_counts["pi"], 2)

    summary = build_multipose_summary(
        poses,
        consensus_threshold=0.5,
    )
    assert_almost_equal(
        summary.residue_persistence["A:TYR123"],
        1.0,
    )
    assert_almost_equal(
        summary.residue_persistence["A:PHE77"],
        2.0 / 3.0,
    )
    assert_contains(summary.consensus_residues, "A:TYR123")
    assert_contains(summary.consensus_residues, "A:PHE77")
    assert_almost_equal(
        summary.family_persistence["pi"],
        2.0 / 3.0,
    )


@self_test(
    section=SELF_TEST_SECTION_MULTIPOSE,
    tags=("multipose", "summary"),
)
def test_multipose_complete_summary() -> None:
    """Verify the complete score-ranked multipose summary."""

    summary = summarize_multipose(
        _make_multipose_fixture()
    )
    assert_is_instance(summary, MultiposeSummary)
    assert_equal(summary.total_poses, 3)
    assert_equal(summary.best_pose_id, 1)
    assert_equal(summary.ranking_metric, KEY_TOTAL_SCORE)
    assert_is(
        summary.rank_direction,
        RankDirection.HIGHER_IS_BETTER,
    )
    assert_sequence_equal(
        tuple(entry.pose_id for entry in summary.poses),
        (1, 3, 2),
    )
    assert_equal(summary.score_statistics["count"], 3)
    assert_almost_equal(
        summary.score_statistics["mean"],
        13.0 / 3.0,
    )
    assert_almost_equal(
        summary.affinity_statistics["minimum"],
        -8.0,
    )
    record = summary.to_dict()
    assert_equal(record[KEY_TOTAL_POSES], 3)
    assert_equal(len(record[KEY_RANKING]), 3)


@self_test(
    section=SELF_TEST_SECTION_MULTIPOSE,
    tags=("multipose", "configuration", "limit"),
)
def test_multipose_top_pose_limit() -> None:
    """Verify configured ranking truncation."""

    config = DEFAULT_REPORT_CONFIG.with_updates(
        multipose=DEFAULT_MULTIPOSE_REPORT_CONFIG.with_updates(
            top_poses=2,
        )
    )
    summary = build_multipose_summary(
        _make_multipose_fixture(),
        config=config,
    )
    assert_equal(summary.total_poses, 3)
    assert_equal(len(summary.poses), 2)
    assert_sequence_equal(
        tuple(entry.pose_id for entry in summary.poses),
        (1, 3),
    )


@self_test(
    section=SELF_TEST_SECTION_MULTIPOSE,
    tags=("multipose", "rows"),
)
def test_multipose_rows() -> None:
    """Verify ranking and persistence table rows."""

    summary = build_multipose_summary(
        _make_multipose_fixture()
    )
    ranking = multipose_ranking_rows(summary)
    assert_equal(len(ranking), 3)
    assert_equal(ranking[0][KEY_POSE_ID], 1)
    assert_equal(ranking[0][KEY_RANK], 1)
    assert_mapping_has_keys(
        ranking[0],
        (
            KEY_TOTAL_SCORE,
            KEY_AFFINITY,
            KEY_TOTAL_INTERACTIONS,
            KEY_TOTAL_RESIDUES,
        ),
    )

    rows = persistence_rows(
        summary.residue_persistence,
        counts=summary.residue_pose_counts,
        total=summary.total_poses,
    )
    assert_equal(rows[0]["item"], "A:TYR123")
    assert_equal(rows[0][KEY_COUNT], 3)
    assert_almost_equal(rows[0][KEY_PERSISTENCE], 1.0)
    assert_equal(rows[0]["persistence_text"], "100.0%")


@self_test(
    section=SELF_TEST_SECTION_MULTIPOSE,
    tags=("multipose", "document", "section"),
)
def test_multipose_report_document() -> None:
    """Verify the multipose section in a complete document."""

    document = create_multipose_report(
        _make_multipose_fixture()
    )
    section = document.get_section(
        ReportSectionID.MULTIPOSE
    )
    assert_is_instance(section, ReportSection)
    assert_false(section.empty)
    assert_true(section.enabled)
    assert_equal(len(section.visible_blocks), 3)
    assert_true(
        any(
            block.name == TABLE_RANKING
            for block in section.blocks
        )
    )
    assert_true(
        any(
            block.name == TABLE_PERSISTENCE
            for block in section.blocks
        )
    )
    assert_true(validate_report_document(document).valid)


@self_test(
    section=SELF_TEST_SECTION_MULTIPOSE,
    tags=("multipose", "errors"),
)
def test_multipose_empty_input_error() -> None:
    """Verify the public multipose API rejects empty input."""

    assert_raises(
        ReportInputError,
        create_multipose_report,
        (),
    )


def run_multipose_self_tests(
    **kwargs: Any,
) -> SelfTestReport:
    """Run Section 30.5 self-tests."""

    return run_registered_self_tests(
        sections=(SELF_TEST_SECTION_MULTIPOSE,),
        **kwargs,
    )


_SECTION_30_5_PUBLIC_NAMES: Final[Tuple[str, ...]] = (
    "run_multipose_self_tests",
)

_register_public_names(_SECTION_30_5_PUBLIC_NAMES)

# =============================================================================
# End of Section 30.5
# =============================================================================


# =============================================================================
# Section 30.6 — Rendering self-tests
# =============================================================================

# 30.6.1. Rendering fixtures
# -----------------------------------------------------------------------------

def _make_rendering_config(
    *,
    report_format: ReportFormat = ReportFormat.TEXT,
    table_of_contents: bool = False,
    full_html: bool = True,
    include_generated_at: bool = False,
    detail: ReportDetail = ReportDetail.STANDARD,
) -> ReportConfig:
    """Create deterministic rendering configuration."""

    return DEFAULT_REPORT_CONFIG.with_updates(
        rendering=DEFAULT_RENDER_CONFIG.with_updates(
            format=report_format,
            include_table_of_contents=table_of_contents,
            html_full_document=full_html,
            include_generated_at=include_generated_at,
            detail=detail,
        )
    )


def _make_rendering_document(
    *,
    config: Optional[ReportConfig] = None,
) -> ReportDocument:
    """Create a deterministic rendering test document."""

    active_config = config or _make_rendering_config()
    document = build_report_document(
        make_self_test_pose(),
        config=active_config,
        title="Rendering <test>",
        subtitle="Format & escaping",
        description="A deterministic rendering fixture.",
    )
    return replace(
        document,
        generated_at="2026-07-28T12:30:15Z",
    )


# 30.6.2. Internal block and table tests
# -----------------------------------------------------------------------------

@self_test(
    section=SELF_TEST_SECTION_RENDERING,
    tags=("rendering", "blocks"),
)
def test_rendering_block_factories() -> None:
    """Verify structured block factories."""

    paragraph = paragraph_block("Alpha")
    key_values = key_value_block({"Score": 1.5})
    table = table_block(
        [{"name": "A", "score": 1.0}],
        name="scores",
    )
    values = list_block(("A", "B"), ordered=True)
    notice = notice_block(
        "Warning text",
        severity=Severity.WARNING,
    )

    assert_is(paragraph.kind, ReportBlockKind.PARAGRAPH)
    assert_is(key_values.kind, ReportBlockKind.KEY_VALUE)
    assert_is(table.kind, ReportBlockKind.TABLE)
    assert_is(values.kind, ReportBlockKind.LIST)
    assert_true(values.metadata["ordered"])
    assert_is(notice.kind, ReportBlockKind.NOTICE)
    assert_equal(notice.metadata["severity"], SEVERITY_WARNING)


@self_test(
    section=SELF_TEST_SECTION_RENDERING,
    tags=("rendering", "tables", "truncation"),
)
def test_rendering_internal_table_model() -> None:
    """Verify table inference, alignment and truncation."""

    table_config = DEFAULT_TABLE_CONFIG.with_updates(max_rows=2)
    table = build_report_table(
        (
            {"name": "Alpha", "score": 1.2},
            {"name": "Beta", "score": 2.3},
            {"name": "Gamma", "score": 3.4},
        ),
        name="scores",
        preferred_columns=("name", "score"),
        config=table_config,
    )
    assert_equal(len(table.columns), 2)
    assert_equal(table.columns[0].key, "name")
    assert_equal(table.columns[0].align, "left")
    assert_equal(table.columns[1].key, "score")
    assert_equal(table.columns[1].align, "right")
    assert_equal(len(table.rows), 2)
    assert_true(table.truncated)
    assert_equal(table.total_rows, 3)
    assert_equal(len(formatted_table_rows(table)), 2)
    assert_equal(len(table_column_widths(table)), 2)
    assert_equal(align_table_cell("x", 3, "right"), "  x")
    assert_true(validate_report_table(table).valid)


# 30.6.3. Text rendering tests
# -----------------------------------------------------------------------------

@self_test(
    section=SELF_TEST_SECTION_RENDERING,
    tags=("rendering", "text", "primitives"),
)
def test_text_rendering_primitives() -> None:
    """Verify plain-text headings, lists and key-values."""

    heading = render_text_heading("Title", level=1)
    assert_equal(heading, "Title\n=====")
    wrapped = wrap_text(
        "alpha beta gamma delta " * 4,
        width=MIN_TEXT_WIDTH,
    )
    assert_true(
        all(
            len(line) <= MIN_TEXT_WIDTH
            for line in wrapped.splitlines()
        )
    )
    key_values = render_text_key_values(
        {"Score": 1.5, "Count": 2}
    )
    assert_contains(key_values, "Score")
    assert_contains(key_values, "1.5")
    listed = render_text_list(
        ("A", "B"),
        ordered=True,
    )
    assert_contains(listed, "1. A")
    assert_contains(listed, "2. B")


@self_test(
    section=SELF_TEST_SECTION_RENDERING,
    tags=("rendering", "text", "table"),
)
def test_text_table_rendering() -> None:
    """Verify plain-text table output and truncation note."""

    config = _make_rendering_config()
    table = build_report_table(
        (
            {"name": "Alpha", "score": 1.2},
            {"name": "Beta", "score": 2.3},
            {"name": "Gamma", "score": 3.4},
        ),
        name="scores",
        preferred_columns=("name", "score"),
        config=DEFAULT_TABLE_CONFIG.with_updates(max_rows=2),
    )
    rendered = render_text_table(table, config=config)
    assert_contains(rendered, "Name")
    assert_contains(rendered, "Score")
    assert_contains(rendered, "Alpha")
    assert_contains(rendered, "2 of 3 rows shown")


@self_test(
    section=SELF_TEST_SECTION_RENDERING,
    tags=("rendering", "text", "report"),
)
def test_complete_text_report_rendering() -> None:
    """Verify a complete plain-text report."""

    config = _make_rendering_config(
        report_format=ReportFormat.TEXT,
    )
    document = _make_rendering_document(config=config)
    rendered = render_report_text(document)
    assert_true(rendered.startswith("Rendering <test>\n"))
    assert_contains(rendered, "Format & escaping")
    assert_contains(rendered, "Overview")
    assert_contains(rendered, "Interactions")
    assert_contains(rendered, "Residue Summary")
    assert_contains(rendered, "Scoring and Explainability")
    header = rendered.split("Overview", 1)[0]
    assert_false("Generated at:" in header)
    assert_true(rendered.endswith(DEFAULT_NEWLINE))


# 30.6.4. Markdown rendering tests
# -----------------------------------------------------------------------------

@self_test(
    section=SELF_TEST_SECTION_RENDERING,
    tags=("rendering", "markdown", "primitives"),
)
def test_markdown_rendering_primitives() -> None:
    """Verify Markdown headings, code and escaping."""

    assert_equal(
        render_markdown_heading("Title", level=2),
        "## Title",
    )
    assert_equal(markdown_anchor("Score & Details"), "score-details")
    assert_contains(
        render_markdown_key_values({"A": "*x*"}),
        r"\*x\*",
    )
    assert_contains(
        render_markdown_list(("A", "B"), ordered=True),
        "1. A",
    )
    code = render_markdown_code("print('x')", language="python")
    assert_true(code.startswith("```python"))
    assert_true(code.endswith("```"))


@self_test(
    section=SELF_TEST_SECTION_RENDERING,
    tags=("rendering", "markdown", "table"),
)
def test_markdown_table_rendering() -> None:
    """Verify Markdown table escaping and alignment."""

    table = build_report_table(
        (
            {"name": "A|B", "score": 1.2},
            {"name": "*C*", "score": 2.3},
        ),
        name="scores",
        preferred_columns=("name", "score"),
    )
    rendered = render_markdown_table(table)
    assert_contains(rendered, "| Name | Score |")
    assert_contains(rendered, "| :--- | ---: |")
    assert_contains(rendered, r"A\|B")
    assert_contains(rendered, r"\*C\*")


@self_test(
    section=SELF_TEST_SECTION_RENDERING,
    tags=("rendering", "markdown", "report", "toc"),
)
def test_complete_markdown_report_rendering() -> None:
    """Verify complete Markdown and table of contents."""

    config = _make_rendering_config(
        report_format=ReportFormat.MARKDOWN,
        table_of_contents=True,
    )
    document = _make_rendering_document(config=config)
    rendered = render_report_markdown(document)
    assert_true(rendered.startswith("# Rendering <test>"))
    assert_contains(rendered, "*Format & escaping*")
    assert_contains(rendered, "## Contents")
    assert_contains(rendered, "[Overview](#overview)")
    assert_contains(rendered, "## Interactions")
    assert_contains(rendered, "| Family | Count |")
    header = rendered.split("## Overview", 1)[0]
    assert_false("**Generated at:**" in header)
    assert_true(rendered.endswith(DEFAULT_NEWLINE))


# 30.6.5. HTML rendering tests
# -----------------------------------------------------------------------------

@self_test(
    section=SELF_TEST_SECTION_RENDERING,
    tags=("rendering", "html", "primitives"),
)
def test_html_rendering_primitives() -> None:
    """Verify HTML tags, attributes and escaping."""

    assert_equal(
        html_attributes({"data_value": 'a&"b'}),
        ' data-value="a&amp;&quot;b"',
    )
    assert_equal(
        html_tag("p", "<x>", escape_content=True),
        "<p>&lt;x&gt;</p>",
    )
    assert_equal(
        render_html_heading("A < B", level=2),
        "<h2>A &lt; B</h2>",
    )
    assert_contains(
        render_html_paragraph("A & B"),
        "A &amp; B",
    )
    assert_contains(
        render_html_code("<tag>", language="xml"),
        "&lt;tag&gt;",
    )


@self_test(
    section=SELF_TEST_SECTION_RENDERING,
    tags=("rendering", "html", "table"),
)
def test_html_table_rendering() -> None:
    """Verify semantic HTML table output."""

    table = build_report_table(
        (
            {"name": "<Alpha>", "score": 1.2},
            {"name": "Beta", "score": 2.3},
        ),
        name="scores",
        preferred_columns=("name", "score"),
    )
    rendered = render_html_table(table)
    assert_contains(rendered, '<table class="dockanalyzer-table"')
    assert_contains(rendered, "<thead>")
    assert_contains(rendered, "<tbody>")
    assert_contains(rendered, "&lt;Alpha&gt;")
    assert_contains(rendered, 'class="align-right"')


@self_test(
    section=SELF_TEST_SECTION_RENDERING,
    tags=("rendering", "html", "report"),
)
def test_complete_html_report_rendering() -> None:
    """Verify a complete standalone HTML document."""

    config = _make_rendering_config(
        report_format=ReportFormat.HTML,
        table_of_contents=True,
        full_html=True,
    )
    document = _make_rendering_document(config=config)
    rendered = render_report_html(document)
    assert_true(rendered.startswith(HTML_DOCUMENT_TYPE))
    assert_contains(rendered, '<html lang="en">')
    assert_contains(rendered, "<title>Rendering &lt;test&gt;</title>")
    assert_contains(rendered, "<h1>Rendering &lt;test&gt;</h1>")
    assert_contains(rendered, "Format &amp; escaping")
    assert_contains(rendered, '<nav class="dockanalyzer-toc"')
    assert_contains(rendered, 'data-section="interactions"')
    assert_true(rendered.endswith(DEFAULT_NEWLINE))


@self_test(
    section=SELF_TEST_SECTION_RENDERING,
    tags=("rendering", "html", "fragment"),
)
def test_html_fragment_rendering() -> None:
    """Verify body-only HTML rendering."""

    config = _make_rendering_config(
        report_format=ReportFormat.HTML,
        full_html=False,
    )
    document = _make_rendering_document(config=config)
    rendered = render_report_html(document)
    assert_true(rendered.startswith("<body"))
    assert_false(HTML_DOCUMENT_TYPE in rendered)
    assert_false("<html " in rendered)
    assert_contains(rendered, 'class="dockanalyzer-report"')


# 30.6.6. JSON and dispatch tests
# -----------------------------------------------------------------------------

@self_test(
    section=SELF_TEST_SECTION_RENDERING,
    tags=("rendering", "json", "safe_conversion"),
)
def test_json_safe_conversion() -> None:
    """Verify JSON conversion of special and recursive values."""

    assert_equal(to_json_safe(float("nan")), None)
    assert_equal(to_json_safe(float("inf")), None)
    assert_equal(to_json_safe(Path("/tmp/report")), "/tmp/report")
    assert_equal(
        to_json_safe(ReportFormat.MARKDOWN),
        "markdown",
    )

    recursive: List[Any] = []
    recursive.append(recursive)
    state = JSONConversionState()
    converted = to_json_safe(recursive, state=state)
    assert_equal(
        converted[0][JSON_VALUE_KEY],
        "<recursive-reference>",
    )
    assert_true(bool(state.warnings))


@self_test(
    section=SELF_TEST_SECTION_RENDERING,
    tags=("rendering", "json", "report"),
)
def test_complete_json_report_rendering() -> None:
    """Verify complete JSON representation and schema."""

    config = _make_rendering_config(
        report_format=ReportFormat.JSON,
    )
    document = _make_rendering_document(config=config)
    rendered = render_report_json(
        document,
        include_config=True,
    )
    parsed = parse_report_json(rendered)
    assert_equal(parsed[KEY_SCHEMA_NAME], REPORT_SCHEMA_NAME)
    assert_equal(parsed[KEY_SCHEMA_VERSION], REPORT_SCHEMA_VERSION)
    assert_equal(parsed[KEY_TITLE], "Rendering <test>")
    assert_true(isinstance(parsed[KEY_SECTIONS], list))
    assert_true(len(parsed[KEY_SECTIONS]) >= 7)
    assert_contains(parsed, "config")
    schema = report_json_schema_summary(parsed)
    assert_equal(schema["missing_required_keys"], [])
    assert_true(validate_report_json_data(parsed).valid)
    assert_true(rendered.endswith(DEFAULT_NEWLINE))


@self_test(
    section=SELF_TEST_SECTION_RENDERING,
    tags=("rendering", "dispatch", "formats"),
)
def test_rendering_dispatch_and_multiple_formats() -> None:
    """Verify unified and multi-format rendering interfaces."""

    document = _make_rendering_document()
    text = render_report(
        document,
        report_format=ReportFormat.TEXT,
    )
    markdown = render_report(
        document,
        report_format=ReportFormat.MARKDOWN,
    )
    html = render_report(
        document,
        report_format=ReportFormat.HTML,
    )
    json_text = render_report(
        document,
        report_format=ReportFormat.JSON,
    )
    assert_true(text.startswith("Rendering <test>"))
    assert_true(markdown.startswith("# Rendering <test>"))
    assert_true(html.startswith(HTML_DOCUMENT_TYPE))
    assert_equal(
        parse_report_json(json_text)[KEY_TITLE],
        "Rendering <test>",
    )

    outputs = render_dock_report_formats(
        document,
        formats=("text", "markdown", "html", "json"),
        config=document.config,
    )
    assert_equal(
        set(outputs),
        {"text", "markdown", "html", "json"},
    )


@self_test(
    section=SELF_TEST_SECTION_RENDERING,
    tags=("rendering", "configuration"),
)
def test_rendering_configuration_controls() -> None:
    """Verify title, subtitle, timestamp and detail controls."""

    config = DEFAULT_REPORT_CONFIG.with_updates(
        rendering=DEFAULT_RENDER_CONFIG.with_updates(
            include_title=False,
            include_subtitle=False,
            include_generated_at=True,
            detail=ReportDetail.DETAILED,
        )
    )
    document = _make_rendering_document(config=config)
    text = render_report_text(document)
    assert_false("Rendering <test>" in text)
    assert_false("Format & escaping" in text)
    assert_contains(text, "Generated at: 2026-07-28T12:30:15Z")
    assert_contains(
        text,
        SECTION_DESCRIPTIONS[SECTION_OVERVIEW],
    )


def run_rendering_self_tests(
    **kwargs: Any,
) -> SelfTestReport:
    """Run Section 30.6 self-tests."""

    return run_registered_self_tests(
        sections=(SELF_TEST_SECTION_RENDERING,),
        **kwargs,
    )


_SECTION_30_6_PUBLIC_NAMES: Final[Tuple[str, ...]] = (
    "run_rendering_self_tests",
)

_register_public_names(_SECTION_30_6_PUBLIC_NAMES)

# =============================================================================
# End of Section 30.6
# =============================================================================

# =============================================================================
# Section 30.7 — Writing and export self-tests
# =============================================================================

# 30.7.1. Synthetic export.py implementation
# -----------------------------------------------------------------------------

def _make_self_test_export_module() -> Any:
    """Create a compatible in-memory export.py substitute."""

    module_type = __import__("types").ModuleType
    module = module_type("report_self_test_export")
    module.__version__ = "1.0-self-test"
    module.EXPORT_SCHEMA_NAME = "dockanalyzer.export.self-test"
    module.EXPORT_SCHEMA_VERSION = "1.0"
    module.SUPPORTED_EXPORT_FORMATS = frozenset(
        {"json", "jsonl", "csv", "tsv", "xlsx", "txt"}
    )

    class ExportedFile:
        def __init__(
            self,
            path: PathLike,
            format_name: str,
            *,
            table: Optional[str] = None,
        ) -> None:
            self.path = Path(path)
            self.format = format_name
            self.table = table
            self.warnings: List[str] = []
            self.errors: List[str] = []

    class ExportResult:
        def __init__(
            self,
            files: Iterable[Any],
            *,
            status: str = "success",
        ) -> None:
            self.status = status
            self.files = list(files)
            self.warnings: List[str] = []
            self.errors: List[str] = []

    class TableData:
        def __init__(
            self,
            name: str,
            rows: Iterable[Mapping[str, Any]],
            metadata: Optional[Mapping[str, Any]] = None,
        ) -> None:
            self.name = name
            self.rows = [dict(row) for row in rows]
            self.columns = (
                list(self.rows[0])
                if self.rows
                else []
            )
            self.metadata = dict(metadata or {})

    class TableCollection:
        def __init__(self) -> None:
            self.tables: Dict[str, Any] = {}
            self.metadata: Dict[str, Any] = {}

        def add(
            self,
            table: Any,
            replace: bool = False,
        ) -> Any:
            if table.name in self.tables and not replace:
                raise ValueError(table.name)
            self.tables[table.name] = table
            return table

    def available_export_formats() -> Tuple[str, ...]:
        return tuple(sorted(module.SUPPORTED_EXPORT_FORMATS))

    def normalize_export_format(value: Any) -> str:
        token = str(value or "").strip().lower().lstrip(".")
        token = {
            "excel": "xlsx",
            "text": "txt",
            "ndjson": "jsonl",
        }.get(token, token)
        if token not in module.SUPPORTED_EXPORT_FORMATS:
            raise ValueError(token)
        return token

    def _prepare_path(path: PathLike) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        return output

    def _write_marker(
        path: PathLike,
        marker: str,
    ) -> Path:
        output = _prepare_path(path)
        output.write_text(marker, encoding=DEFAULT_ENCODING)
        return output

    def build_table(
        values: Iterable[Mapping[str, Any]],
        *,
        name: str = "table",
        metadata: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        return TableData(name, values, metadata=metadata)

    def export_data(
        value: Any,
        path: PathLike,
        *,
        format: Any = None,
        overwrite: Any = "overwrite",
        **kwargs: Any,
    ) -> Any:
        format_name = normalize_export_format(
            format or Path(path).suffix
        )
        output = _prepare_path(path)
        if output.exists() and overwrite in {
            False,
            "error",
            "raise",
        }:
            raise FileExistsError(output)

        if format_name == "json":
            text = json.dumps(
                to_json_safe(value),
                ensure_ascii=False,
                sort_keys=True,
            )
        elif format_name == "jsonl":
            values = (
                value
                if isinstance(value, (list, tuple))
                else [value]
            )
            text = DEFAULT_NEWLINE.join(
                json.dumps(
                    to_json_safe(item),
                    ensure_ascii=False,
                    sort_keys=True,
                )
                for item in values
            )
        else:
            text = safe_string(value, "")
        output.write_text(text, encoding=DEFAULT_ENCODING)
        return ExportedFile(output, format_name)

    def write_json(
        value: Any,
        path: PathLike = None,
        **kwargs: Any,
    ) -> Any:
        return export_data(
            value,
            path,
            format="json",
            **kwargs,
        )

    def write_json_lines(
        value: Any,
        path: PathLike = None,
        **kwargs: Any,
    ) -> Any:
        return export_data(
            value,
            path,
            format="jsonl",
            **kwargs,
        )

    def write_text(
        value: Any,
        path: PathLike = None,
        **kwargs: Any,
    ) -> Any:
        return export_data(
            value,
            path,
            format="txt",
            **kwargs,
        )

    def _write_delimited(
        values: Iterable[Mapping[str, Any]],
        path: PathLike,
        *,
        delimiter: str,
        format_name: str,
        table_name: str,
    ) -> Any:
        rows = [dict(row) for row in values]
        keys = list(rows[0]) if rows else []
        lines: List[str] = []
        if keys:
            lines.append(delimiter.join(keys))
            for row in rows:
                lines.append(
                    delimiter.join(
                        safe_string(row.get(key), "")
                        for key in keys
                    )
                )
        output = _write_marker(
            path,
            DEFAULT_NEWLINE.join(lines),
        )
        return ExportedFile(
            output,
            format_name,
            table=table_name,
        )

    def write_csv(
        values: Iterable[Mapping[str, Any]],
        path: PathLike = None,
        *,
        table_name: str = "data",
        **kwargs: Any,
    ) -> Any:
        return _write_delimited(
            values,
            path,
            delimiter=",",
            format_name="csv",
            table_name=table_name,
        )

    def write_tsv(
        values: Iterable[Mapping[str, Any]],
        path: PathLike = None,
        *,
        table_name: str = "data",
        **kwargs: Any,
    ) -> Any:
        return _write_delimited(
            values,
            path,
            delimiter="\t",
            format_name="tsv",
            table_name=table_name,
        )

    def write_excel(
        value: Any,
        path: PathLike = None,
        **kwargs: Any,
    ) -> Any:
        table_names = sorted(
            getattr(value, "tables", {}).keys()
        )
        output = _write_marker(
            path,
            "xlsx:" + ",".join(table_names),
        )
        return ExportedFile(output, "xlsx")

    module.ExportedFile = ExportedFile
    module.ExportResult = ExportResult
    module.TableCollection = TableCollection
    module.available_export_formats = available_export_formats
    module.normalize_export_format = normalize_export_format
    module.build_table = build_table
    module.export_data = export_data
    module.write_json = write_json
    module.write_json_lines = write_json_lines
    module.write_text = write_text
    module.write_csv = write_csv
    module.write_tsv = write_tsv
    module.write_excel = write_excel
    return module


def _make_writing_test_document() -> ReportDocument:
    """Create a deterministic document for file-output tests."""

    config = DEFAULT_REPORT_CONFIG.with_updates(
        rendering=DEFAULT_RENDER_CONFIG.with_updates(
            include_generated_at=False,
        )
    )
    document = build_report_document(
        make_self_test_pose(),
        config=config,
        title="Writing self-test",
    )
    return replace(
        document,
        generated_at="2026-07-28T12:30:15Z",
    )


# 30.7.2. Safe path and low-level writing tests
# -----------------------------------------------------------------------------

@self_test(
    section=SELF_TEST_SECTION_WRITING_EXPORT,
    tags=("writing", "paths"),
)
def test_writing_output_path_preparation() -> None:
    """Verify parent creation and path rejection."""

    TemporaryDirectory = __import__(
        "tempfile",
        fromlist=["TemporaryDirectory"],
    ).TemporaryDirectory

    with TemporaryDirectory() as directory:
        root = Path(directory)
        output = root / "nested" / "report.txt"
        prepared = prepare_output_path(
            output,
            config=WriteConfig(
                overwrite=False,
                create_parents=True,
            ),
        )
        assert_equal(prepared, output)
        assert_true(output.parent.is_dir())

        assert_raises(
            ReportPathError,
            prepare_output_path,
            output.parent,
            config=WriteConfig(
                overwrite=True,
                create_parents=True,
            ),
        )

        missing_parent = root / "missing" / "report.txt"
        assert_raises(
            ReportPathError,
            prepare_output_path,
            missing_parent,
            config=WriteConfig(
                overwrite=False,
                create_parents=False,
            ),
        )


@self_test(
    section=SELF_TEST_SECTION_WRITING_EXPORT,
    tags=("writing", "direct", "atomic"),
)
def test_writing_direct_and_atomic_text() -> None:
    """Verify direct and atomic text writing."""

    TemporaryDirectory = __import__(
        "tempfile",
        fromlist=["TemporaryDirectory"],
    ).TemporaryDirectory

    with TemporaryDirectory() as directory:
        root = Path(directory)
        direct_path = root / "direct.txt"
        atomic_path = root / "atomic.txt"

        direct_bytes = write_text_direct(
            direct_path,
            "direct content",
            config=WriteConfig(
                overwrite=False,
                atomic=False,
            ),
        )
        assert_equal(
            direct_path.read_text(encoding=DEFAULT_ENCODING),
            "direct content",
        )
        assert_equal(direct_bytes, direct_path.stat().st_size)

        atomic_bytes = write_text_atomic(
            atomic_path,
            "atomic content",
            config=WriteConfig(
                overwrite=False,
                atomic=True,
                temp_suffix=".selftest.tmp",
            ),
        )
        assert_equal(
            atomic_path.read_text(encoding=DEFAULT_ENCODING),
            "atomic content",
        )
        assert_equal(atomic_bytes, atomic_path.stat().st_size)
        assert_false(
            any(
                path.name.endswith(".selftest.tmp")
                for path in root.iterdir()
            )
        )


@self_test(
    section=SELF_TEST_SECTION_WRITING_EXPORT,
    tags=("writing", "overwrite"),
)
def test_writing_overwrite_protection() -> None:
    """Verify existing files are protected by default."""

    TemporaryDirectory = __import__(
        "tempfile",
        fromlist=["TemporaryDirectory"],
    ).TemporaryDirectory

    with TemporaryDirectory() as directory:
        path = Path(directory) / "protected.txt"
        path.write_text("original", encoding=DEFAULT_ENCODING)

        assert_raises(
            ReportOverwriteError,
            write_report_content,
            "replacement",
            path,
            report_format=ReportFormat.TEXT,
            config=WriteConfig(
                overwrite=False,
                atomic=True,
            ),
        )
        assert_equal(
            path.read_text(encoding=DEFAULT_ENCODING),
            "original",
        )


@self_test(
    section=SELF_TEST_SECTION_WRITING_EXPORT,
    tags=("writing", "backup", "checksum"),
)
def test_writing_backup_and_checksum() -> None:
    """Verify non-destructive backup and checksum output."""

    TemporaryDirectory = __import__(
        "tempfile",
        fromlist=["TemporaryDirectory"],
    ).TemporaryDirectory

    with TemporaryDirectory() as directory:
        path = Path(directory) / "report.txt"
        path.write_text("old content", encoding=DEFAULT_ENCODING)

        result = write_report_content(
            "new content",
            path,
            report_format=ReportFormat.TEXT,
            config=WriteConfig(
                overwrite=True,
                atomic=True,
                backup=True,
                backup_suffix=".bak",
            ),
            calculate_checksum=True,
        )
        assert_true(result.overwritten)
        assert_true(result.atomic)
        assert_true(bool(result.backup_path))
        assert_true(Path(result.backup_path).is_file())
        assert_equal(
            Path(result.backup_path).read_text(
                encoding=DEFAULT_ENCODING
            ),
            "old content",
        )
        assert_equal(
            path.read_text(encoding=DEFAULT_ENCODING),
            "new content",
        )
        assert_equal(result.checksum, file_checksum(path))
        assert_equal(result.to_dict()["format"], "text")


# 30.7.3. Report writing tests
# -----------------------------------------------------------------------------

@self_test(
    section=SELF_TEST_SECTION_WRITING_EXPORT,
    tags=("writing", "report", "formats"),
)
def test_writing_single_report_formats() -> None:
    """Verify text, Markdown, HTML and JSON file output."""

    TemporaryDirectory = __import__(
        "tempfile",
        fromlist=["TemporaryDirectory"],
    ).TemporaryDirectory

    document = _make_writing_test_document()
    with TemporaryDirectory() as directory:
        root = Path(directory)
        write_config = WriteConfig(
            overwrite=True,
            atomic=True,
            create_parents=True,
        )
        expectations = {
            "text": ("report.txt", "Writing self-test"),
            "markdown": ("report.md", "# Writing self-test"),
            "html": ("report.html", HTML_DOCUMENT_TYPE),
            "json": ("report.json", REPORT_SCHEMA_NAME),
        }
        for format_name, (filename, marker) in expectations.items():
            result = write_report(
                document,
                root / filename,
                report_format=format_name,
                write_config=write_config,
                calculate_checksum=True,
            )
            output = Path(result.path)
            assert_true(output.is_file())
            assert_true(result.bytes_written > 0)
            assert_true(bool(result.checksum))
            assert_contains(
                output.read_text(encoding=DEFAULT_ENCODING),
                marker,
            )
            assert_true(
                validate_report_file(
                    output,
                    report_format=format_name,
                ).valid
            )


@self_test(
    section=SELF_TEST_SECTION_WRITING_EXPORT,
    tags=("writing", "report", "suffix"),
)
def test_writing_suffix_inference_and_enforcement() -> None:
    """Verify format inference and suffix replacement."""

    TemporaryDirectory = __import__(
        "tempfile",
        fromlist=["TemporaryDirectory"],
    ).TemporaryDirectory

    document = _make_writing_test_document()
    with TemporaryDirectory() as directory:
        root = Path(directory)
        result = write_report(
            document,
            root / "report.unknown",
            report_format="json",
            ensure_suffix=True,
            write_config=WriteConfig(
                overwrite=True,
                create_parents=True,
            ),
        )
        assert_true(result.path.endswith(".json"))
        assert_equal(
            parse_report_json(
                Path(result.path).read_text(
                    encoding=DEFAULT_ENCODING
                )
            )[KEY_TITLE],
            "Writing self-test",
        )

        inferred = write_report(
            document,
            root / "inferred.md",
            write_config=WriteConfig(overwrite=True),
        )
        assert_is(inferred.format, ReportFormat.MARKDOWN)


@self_test(
    section=SELF_TEST_SECTION_WRITING_EXPORT,
    tags=("writing", "report", "multiple"),
)
def test_writing_multiple_report_formats() -> None:
    """Verify multi-format output and convenience aliases."""

    TemporaryDirectory = __import__(
        "tempfile",
        fromlist=["TemporaryDirectory"],
    ).TemporaryDirectory

    document = _make_writing_test_document()
    with TemporaryDirectory() as directory:
        root = Path(directory)
        results = write_report_formats(
            document,
            root / "reports",
            formats=("text", "markdown", "html", "json"),
            basename="dock report",
            write_config=WriteConfig(
                overwrite=True,
                atomic=True,
                create_parents=True,
            ),
            calculate_checksum=True,
        )
        assert_equal(
            set(results),
            {"text", "markdown", "html", "json"},
        )
        assert_true(
            all(Path(result.path).is_file() for result in results.values())
        )
        assert_true(
            all(bool(result.checksum) for result in results.values())
        )
        assert_equal(
            Path(results["markdown"].path).name,
            "dock_report.md",
        )

        convenience = save_dock_report_formats(
            document,
            root / "convenience",
            formats=("text", "json"),
            basename="pose",
            write_config=WriteConfig(
                overwrite=True,
                atomic=True,
                create_parents=True,
            ),
        )
        assert_equal(set(convenience), {"text", "json"})


# 30.7.4. export.py capability and conversion tests
# -----------------------------------------------------------------------------

@self_test(
    section=SELF_TEST_SECTION_WRITING_EXPORT,
    tags=("export", "capabilities"),
)
def test_export_capabilities_and_formats() -> None:
    """Verify compatible export.py capability detection."""

    module = _make_self_test_export_module()
    capabilities = inspect_export_capabilities(module)
    assert_true(capabilities.available)
    assert_equal(
        capabilities.module_version,
        "1.0-self-test",
    )
    assert_contains(capabilities.formats, "json")
    assert_contains(capabilities.formats, "xlsx")
    assert_true(capabilities.supports("export_data"))
    assert_true(capabilities.supports("TableCollection"))

    assert_equal(
        normalize_report_export_format(
            "excel",
            export_module=module,
        ),
        "xlsx",
    )
    assert_equal(
        report_export_mode(
            "json",
            export_module=module,
        ),
        "payload",
    )
    assert_equal(
        report_export_mode(
            "csv",
            export_module=module,
        ),
        "tables",
    )


@self_test(
    section=SELF_TEST_SECTION_WRITING_EXPORT,
    tags=("export", "payload", "tables"),
)
def test_export_payload_and_table_conversion() -> None:
    """Verify report conversion to export payloads and tables."""

    document = _make_writing_test_document()
    payload = report_to_export_payload(
        document,
        include_config=True,
        include_rendered=True,
    )
    assert_equal(payload[KEY_SCHEMA_NAME], REPORT_SCHEMA_NAME)
    assert_equal(
        payload["integration"]["target_module"],
        SOURCE_MODULE_EXPORT,
    )
    assert_contains(payload, "config")
    assert_contains(payload, "rendered")
    assert_contains(payload["rendered"], "text")
    assert_contains(payload["rendered"], "markdown")
    assert_contains(payload["rendered"], "html")

    tables = report_to_export_tables(document)
    assert_contains(tables, "summary")
    assert_contains(tables, TABLE_INTERACTIONS)
    assert_contains(tables, TABLE_RESIDUES)
    assert_true(
        all(isinstance(rows, list) for rows in tables.values())
    )
    assert_true(
        all(
            isinstance(row, dict)
            for rows in tables.values()
            for row in rows
        )
    )


@self_test(
    section=SELF_TEST_SECTION_WRITING_EXPORT,
    tags=("export", "collection"),
)
def test_export_table_collection() -> None:
    """Verify TableCollection adaptation."""

    module = _make_self_test_export_module()
    document = _make_writing_test_document()
    collection = build_export_table_collection(
        document,
        export_module=module,
    )
    assert_is_instance(collection, module.TableCollection)
    assert_contains(collection.tables, "summary")
    assert_contains(collection.tables, TABLE_INTERACTIONS)
    assert_contains(collection.tables, TABLE_RESIDUES)
    assert_equal(
        collection.metadata["source"],
        SOURCE_MODULE_REPORT,
    )


# 30.7.5. export.py output tests
# -----------------------------------------------------------------------------

@self_test(
    section=SELF_TEST_SECTION_WRITING_EXPORT,
    tags=("export", "payload", "json"),
)
def test_export_json_payload_output() -> None:
    """Verify JSON payload export through export.py."""

    TemporaryDirectory = __import__(
        "tempfile",
        fromlist=["TemporaryDirectory"],
    ).TemporaryDirectory

    module = _make_self_test_export_module()
    document = _make_writing_test_document()
    with TemporaryDirectory() as directory:
        output = Path(directory) / "report.json"
        result = export_report_with_export_module(
            document,
            output,
            export_module=module,
            include_config=True,
        )
        assert_true(result.succeeded)
        assert_equal(result.mode, "payload")
        assert_equal(result.format, "json")
        assert_equal(len(result.files), 1)
        assert_true(Path(result.files[0]).is_file())
        data = json.loads(
            Path(result.files[0]).read_text(
                encoding=DEFAULT_ENCODING
            )
        )
        assert_equal(data[KEY_SCHEMA_NAME], REPORT_SCHEMA_NAME)
        assert_contains(data, "config")


@self_test(
    section=SELF_TEST_SECTION_WRITING_EXPORT,
    tags=("export", "tables", "delimited"),
)
def test_export_csv_and_tsv_tables() -> None:
    """Verify one delimited file per report table."""

    TemporaryDirectory = __import__(
        "tempfile",
        fromlist=["TemporaryDirectory"],
    ).TemporaryDirectory

    module = _make_self_test_export_module()
    document = _make_writing_test_document()
    expected_tables = report_to_export_tables(document)

    with TemporaryDirectory() as directory:
        root = Path(directory)
        csv_result = export_report_tables(
            document,
            root / "report.csv",
            format_name="csv",
            export_module=module,
        )
        tsv_result = export_report_tables(
            document,
            root / "report.tsv",
            format_name="tsv",
            export_module=module,
        )
        assert_true(csv_result.succeeded)
        assert_true(tsv_result.succeeded)
        assert_equal(csv_result.mode, "tables")
        assert_equal(tsv_result.mode, "tables")
        assert_equal(len(csv_result.files), len(expected_tables))
        assert_equal(len(tsv_result.files), len(expected_tables))
        assert_true(
            all(Path(path).is_file() for path in csv_result.files)
        )
        assert_true(
            all(Path(path).is_file() for path in tsv_result.files)
        )
        assert_true(
            any(
                TABLE_INTERACTIONS in Path(path).stem
                for path in csv_result.files
            )
        )


@self_test(
    section=SELF_TEST_SECTION_WRITING_EXPORT,
    tags=("export", "tables", "excel"),
)
def test_export_excel_collection_output() -> None:
    """Verify Excel export through TableCollection."""

    TemporaryDirectory = __import__(
        "tempfile",
        fromlist=["TemporaryDirectory"],
    ).TemporaryDirectory

    module = _make_self_test_export_module()
    document = _make_writing_test_document()

    with TemporaryDirectory() as directory:
        output = Path(directory) / "report.xlsx"
        result = export_report_tables(
            document,
            output,
            format_name="xlsx",
            export_module=module,
        )
        assert_true(result.succeeded)
        assert_equal(result.format, "xlsx")
        assert_equal(result.mode, "tables")
        assert_equal(len(result.files), 1)
        assert_true(Path(result.files[0]).is_file())
        marker = Path(result.files[0]).read_text(
            encoding=DEFAULT_ENCODING
        )
        assert_true(marker.startswith("xlsx:"))
        assert_contains(marker, "summary")
        assert_contains(marker, TABLE_INTERACTIONS)


@self_test(
    section=SELF_TEST_SECTION_WRITING_EXPORT,
    tags=("export", "bundle", "convenience"),
)
def test_export_bundle_and_convenience_api() -> None:
    """Verify bundle and high-level export interfaces."""

    TemporaryDirectory = __import__(
        "tempfile",
        fromlist=["TemporaryDirectory"],
    ).TemporaryDirectory

    module = _make_self_test_export_module()
    document = _make_writing_test_document()

    with TemporaryDirectory() as directory:
        root = Path(directory)
        bundle = export_report_bundle(
            document,
            root / "bundle",
            basename="dock report",
            formats=("json", "csv", "xlsx"),
            export_module=module,
        )
        assert_equal(set(bundle), {"json", "csv", "xlsx"})
        assert_true(
            all(result.succeeded for result in bundle.values())
        )

        convenience = export_dock_report(
            document,
            root / "convenience.json",
            export_module=module,
        )
        assert_true(convenience.succeeded)
        assert_equal(convenience.mode, "payload")
        assert_true(Path(convenience.files[0]).is_file())


@self_test(
    section=SELF_TEST_SECTION_WRITING_EXPORT,
    tags=("export", "signature", "adapter"),
)
def test_export_signature_filtering_and_adapter() -> None:
    """Verify signature filtering and result adaptation."""

    module = _make_self_test_export_module()

    def limited(
        value: Any,
        *,
        path: PathLike = None,
    ) -> Any:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("limited", encoding=DEFAULT_ENCODING)
        return module.ExportedFile(output, "json")

    TemporaryDirectory = __import__(
        "tempfile",
        fromlist=["TemporaryDirectory"],
    ).TemporaryDirectory

    with TemporaryDirectory() as directory:
        output = Path(directory) / "limited.json"
        native = call_export_function(
            limited,
            {},
            path=output,
            unsupported=True,
        )
        assert_true(output.is_file())
        adapted = adapt_export_result(
            native,
            format_name="json",
            mode="payload",
            output_path=output,
        )
        assert_true(adapted.succeeded)
        assert_equal(adapted.status, "success")
        assert_equal(adapted.files, (str(output),))
        assert_equal(adapted.to_dict()["format"], "json")


def run_writing_export_self_tests(
    **kwargs: Any,
) -> SelfTestReport:
    """Run Section 30.7 self-tests."""

    return run_registered_self_tests(
        sections=(SELF_TEST_SECTION_WRITING_EXPORT,),
        **kwargs,
    )


_SECTION_30_7_PUBLIC_NAMES: Final[Tuple[str, ...]] = (
    "run_writing_export_self_tests",
)

_register_public_names(_SECTION_30_7_PUBLIC_NAMES)

# =============================================================================
# End of Section 30.7
# =============================================================================


# =============================================================================
# Section 30.8 — Final runner
# =============================================================================

# 30.8.1. Final runner configuration
# -----------------------------------------------------------------------------

FINAL_SELF_TEST_SECTIONS: Final[Tuple[str, ...]] = (
    SELF_TEST_SECTION_INFRASTRUCTURE,
    SELF_TEST_SECTION_FORMATTING,
    SELF_TEST_SECTION_INTERACTIONS,
    SELF_TEST_SECTION_SINGLE_POSE,
    SELF_TEST_SECTION_MULTIPOSE,
    SELF_TEST_SECTION_RENDERING,
    SELF_TEST_SECTION_WRITING_EXPORT,
    SELF_TEST_SECTION_FINAL_RUNNER,
)

FINAL_SELF_TEST_SUCCESS_EXIT_CODE: Final[int] = 0
FINAL_SELF_TEST_FAILURE_EXIT_CODE: Final[int] = 1
FINAL_SELF_TEST_USAGE_EXIT_CODE: Final[int] = 2


def available_self_test_sections() -> Tuple[str, ...]:
    """Return registered self-test section identifiers."""

    return tuple(
        sorted(
            {
                case.section
                for case in _SELF_TEST_REGISTRY.values()
            },
            key=lambda value: tuple(
                to_safe_int(part, 0)
                for part in value.split(".")
            ),
        )
    )


def self_test_exit_code(
    report: SelfTestReport,
) -> int:
    """Return a process exit code for a self-test report."""

    if not isinstance(report, SelfTestReport):
        raise ReportConfigurationError(
            "report must be SelfTestReport."
        )
    return (
        FINAL_SELF_TEST_SUCCESS_EXIT_CODE
        if report.successful
        else FINAL_SELF_TEST_FAILURE_EXIT_CODE
    )


def render_final_self_test_report(
    report: SelfTestReport,
    *,
    output_format: str = "text",
    include_passed: bool = False,
) -> str:
    """Render the final self-test result as text or JSON."""

    format_name = normalize_field_name(output_format)
    if format_name == "json":
        return json.dumps(
            to_json_safe(report.to_dict()),
            indent=DEFAULT_JSON_INDENT,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        ) + DEFAULT_NEWLINE
    if format_name != "text":
        raise ReportConfigurationError(
            f"Unsupported self-test output format: {output_format!r}."
        )

    lines = [
        format_self_test_report(report),
        f"Duration: {report.duration_seconds:.6f} s",
    ]
    for result in report.results:
        if result.passed and not include_passed:
            continue
        marker = {
            SelfTestStatus.PASS: "PASS",
            SelfTestStatus.FAIL: "FAIL",
            SelfTestStatus.ERROR: "ERROR",
            SelfTestStatus.SKIP: "SKIP",
        }[result.status]
        line = (
            f"[{marker}] {result.section} {result.name}"
            f" ({result.duration_seconds:.6f} s)"
        )
        if result.message:
            line += f" — {result.message}"
        lines.append(line)
        if result.traceback_text:
            lines.append(result.traceback_text.rstrip())
    return DEFAULT_NEWLINE.join(lines) + DEFAULT_NEWLINE


def run_self_tests(
    *,
    sections: Iterable[str] = FINAL_SELF_TEST_SECTIONS,
    tags: Iterable[str] = (),
    names: Iterable[str] = (),
    include_tracebacks: bool = False,
    stop_on_failure: bool = False,
    raise_on_failure: bool = False,
) -> SelfTestReport:
    """Run the complete report.py self-test suite."""

    return run_registered_self_tests(
        sections=sections,
        tags=tags,
        names=names,
        include_tracebacks=include_tracebacks,
        stop_on_failure=stop_on_failure,
        raise_on_failure=raise_on_failure,
    )


# 30.8.2. Final runner tests
# -----------------------------------------------------------------------------

@self_test(
    section=SELF_TEST_SECTION_FINAL_RUNNER,
    tags=("runner", "sections"),
)
def test_final_runner_sections() -> None:
    """Verify the complete section registry."""

    sections = available_self_test_sections()
    assert_sequence_equal(
        sections,
        FINAL_SELF_TEST_SECTIONS,
    )
    assert_equal(len(sections), 8)
    assert_true(
        all(
            list_self_tests(sections=(section,))
            for section in sections
        )
    )


@self_test(
    section=SELF_TEST_SECTION_FINAL_RUNNER,
    tags=("runner", "exit_code"),
)
def test_final_runner_exit_codes() -> None:
    """Verify success and failure process codes."""

    passed = SelfTestReport(
        results=(
            SelfTestResult(
                name="pass",
                section=SELF_TEST_SECTION_FINAL_RUNNER,
                status=SelfTestStatus.PASS,
            ),
        )
    )
    failed = SelfTestReport(
        results=(
            SelfTestResult(
                name="fail",
                section=SELF_TEST_SECTION_FINAL_RUNNER,
                status=SelfTestStatus.FAIL,
                message="failure",
            ),
        )
    )
    assert_equal(
        self_test_exit_code(passed),
        FINAL_SELF_TEST_SUCCESS_EXIT_CODE,
    )
    assert_equal(
        self_test_exit_code(failed),
        FINAL_SELF_TEST_FAILURE_EXIT_CODE,
    )


@self_test(
    section=SELF_TEST_SECTION_FINAL_RUNNER,
    tags=("runner", "rendering"),
)
def test_final_runner_report_rendering() -> None:
    """Verify text and JSON final-runner summaries."""

    report = SelfTestReport(
        results=(
            SelfTestResult(
                name="sample",
                section=SELF_TEST_SECTION_FINAL_RUNNER,
                status=SelfTestStatus.PASS,
                duration_seconds=0.01,
            ),
        ),
        duration_seconds=0.01,
    )
    text = render_final_self_test_report(
        report,
        include_passed=True,
    )
    assert_contains(text, "1/1 passed")
    assert_contains(text, "[PASS]")
    assert_contains(text, "sample")

    json_text = render_final_self_test_report(
        report,
        output_format="json",
    )
    parsed = json.loads(json_text)
    assert_true(parsed["successful"])
    assert_equal(parsed["total"], 1)
    assert_equal(parsed["results"][0]["name"], "sample")


@self_test(
    section=SELF_TEST_SECTION_FINAL_RUNNER,
    tags=("runner", "selection"),
)
def test_final_runner_named_selection() -> None:
    """Verify selection by a registered test name."""

    selected_name = "test_self_test_records"
    report = run_self_tests(
        sections=(SELF_TEST_SECTION_INFRASTRUCTURE,),
        names=(selected_name,),
    )
    assert_true(report.successful)
    assert_equal(report.total, 1)
    assert_equal(report.results[0].name, selected_name)


# 30.8.3. Command-line runner
# -----------------------------------------------------------------------------

def _build_self_test_argument_parser() -> Any:
    """Create the report self-test argument parser."""

    argparse = __import__("argparse")
    parser = argparse.ArgumentParser(
        prog=Path(globals().get("__file__", "report.py")).name,
        description="Run DockAnalyzer report.py self-tests.",
    )
    parser.add_argument(
        "--section",
        action="append",
        dest="sections",
        default=[],
        help="Run only one section; repeat to select several.",
    )
    parser.add_argument(
        "--tag",
        action="append",
        dest="tags",
        default=[],
        help="Require a test tag; repeat for several tags.",
    )
    parser.add_argument(
        "--name",
        action="append",
        dest="names",
        default=[],
        help="Run a named test; repeat for several tests.",
    )
    parser.add_argument(
        "--traceback",
        action="store_true",
        help="Include tracebacks for failures and errors.",
    )
    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="Stop after the first failure or error.",
    )
    parser.add_argument(
        "--raise-on-failure",
        action="store_true",
        help="Raise ReportAggregateError on failure.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_tests",
        help="List matching tests without running them.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Render the final result as JSON.",
    )
    parser.add_argument(
        "--show-passed",
        action="store_true",
        help="Include passed tests in text output.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress successful text output.",
    )
    return parser


def main(
    argv: Optional[Sequence[str]] = None,
) -> int:
    """Run report.py self-tests from the command line."""

    parser = _build_self_test_argument_parser()
    arguments = parser.parse_args(argv)

    sections = tuple(arguments.sections) or FINAL_SELF_TEST_SECTIONS
    unknown_sections = (
        set(sections) - set(available_self_test_sections())
    )
    if unknown_sections:
        parser.error(
            "Unknown self-test section(s): "
            + ", ".join(sorted(unknown_sections))
        )

    if arguments.list_tests:
        cases = list_self_tests(
            sections=sections,
            tags=arguments.tags,
        )
        name_filter = set(arguments.names)
        if name_filter:
            cases = tuple(
                case for case in cases if case.name in name_filter
            )
        for case in cases:
            tags = (
                f" [{', '.join(case.tags)}]"
                if case.tags
                else ""
            )
            print(f"{case.section} {case.name}{tags}")
        return FINAL_SELF_TEST_SUCCESS_EXIT_CODE

    report = run_self_tests(
        sections=sections,
        tags=arguments.tags,
        names=arguments.names,
        include_tracebacks=arguments.traceback,
        stop_on_failure=arguments.stop_on_failure,
        raise_on_failure=arguments.raise_on_failure,
    )

    if arguments.json_output:
        print(
            render_final_self_test_report(
                report,
                output_format="json",
            ),
            end="",
        )
    elif not arguments.quiet or not report.successful:
        print(
            render_final_self_test_report(
                report,
                include_passed=arguments.show_passed,
            ),
            end="",
        )
    return self_test_exit_code(report)


_SECTION_30_8_PUBLIC_NAMES: Final[Tuple[str, ...]] = (
    "FINAL_SELF_TEST_SECTIONS",
    "FINAL_SELF_TEST_SUCCESS_EXIT_CODE",
    "FINAL_SELF_TEST_FAILURE_EXIT_CODE",
    "FINAL_SELF_TEST_USAGE_EXIT_CODE",
    "available_self_test_sections",
    "self_test_exit_code",
    "render_final_self_test_report",
    "run_self_tests",
    "main",
)

_register_public_names(_SECTION_30_8_PUBLIC_NAMES)

# =============================================================================
# End of Section 30.8
# =============================================================================


if __name__ == "__main__":
    raise SystemExit(main())

