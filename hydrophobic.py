# =============================================================================
# DockAnalyzer — Hydrophobic-interaction analysis
# Section 1 — Infrastructure, imports, aliases and public interface
# =============================================================================

"""
Hydrophobic-interaction detection and analysis for DockAnalyzer.

This module provides tools for identifying, validating, classifying,
grouping and summarizing hydrophobic interactions between docked ligands
and receptor structures.

The implementation is designed to operate with ChimeraX atomic objects
while remaining compatible with synthetic Python objects used in tests.

Hydrophobic interactions are treated separately from general atomic
contacts. The :mod:`contacts` module identifies spatially close atom
pairs, whereas this module evaluates whether the participating atoms and
their local chemical environments are compatible with hydrophobic
interaction formation.

Notes
-----
A hydrophobic interaction must not be inferred solely because two carbon
atoms are spatially close. Later sections of this module evaluate:

1. atomic element;
2. local bonding environment;
3. formal or partial charge information, when available;
4. neighboring heteroatoms;
5. aromatic or aliphatic character;
6. residue context;
7. receptor-ligand distance;
8. local contact density.

Specialized aromatic interactions such as pi-stacking, cation-pi and
pi-anion interactions are intentionally handled by their respective
modules. Aromatic atoms may nevertheless participate in non-directional
hydrophobic contacts when the corresponding geometry does not satisfy a
specialized aromatic-interaction definition.
"""

from __future__ import annotations


# -----------------------------------------------------------------------------
# Standard-library imports
# -----------------------------------------------------------------------------

from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import (
    Any,
    Callable,
    Dict,
    Final,
    FrozenSet,
    Iterable,
    Iterator,
    List,
    Literal,
    Mapping,
    MutableMapping,
    Optional,
    Protocol,
    Sequence,
    Set,
    Tuple,
    TypeAlias,
    TypeVar,
    Union,
    runtime_checkable,
)


# -----------------------------------------------------------------------------
# Third-party imports
# -----------------------------------------------------------------------------

import numpy as np
from numpy.typing import NDArray


# -----------------------------------------------------------------------------
# DockAnalyzer imports
# -----------------------------------------------------------------------------

try:
    from . import config

    from .contacts import (
        AtomContact,
        ContactAnalysisResult,
        ResidueContact,
        ResidueContactKey,
        atom_coordinates,
        filter_atoms,
        get_atom_atomic_number,
        get_atom_coordinate,
        get_atom_element,
        get_atom_identifier,
        get_atom_index,
        get_atom_name,
        get_atom_residue,
        get_atom_structure,
        get_dock_model_identifier,
        get_dock_model_pose,
        get_dock_model_receptor,
        get_pose_identifier,
        get_residue_contact_key,
        is_atom_like,
        is_heavy_atom,
        is_hydrogen_atom,
        select_contact_collections,
        validate_atom,
        validate_atom_collection,
    )

    from .geometry import (
        distance,
    )

    from .utils import (
        DockLogger,
        DockModel,
    )

except ImportError:
    import config

    from contacts import (
        AtomContact,
        ContactAnalysisResult,
        ResidueContact,
        ResidueContactKey,
        atom_coordinates,
        filter_atoms,
        get_atom_atomic_number,
        get_atom_coordinate,
        get_atom_element,
        get_atom_identifier,
        get_atom_index,
        get_atom_name,
        get_atom_residue,
        get_atom_structure,
        get_dock_model_identifier,
        get_dock_model_pose,
        get_dock_model_receptor,
        get_pose_identifier,
        get_residue_contact_key,
        is_atom_like,
        is_heavy_atom,
        is_hydrogen_atom,
        select_contact_collections,
        validate_atom,
        validate_atom_collection,
    )

    from geometry import (
        distance,
    )

    from utils import (
        DockLogger,
        DockModel,
    )


# -----------------------------------------------------------------------------
# Module metadata
# -----------------------------------------------------------------------------

__author__: Final[str] = "DockAnalyzer contributors"
__version__: Final[str] = "0.1.0"
__license__: Final[str] = "MIT"

_MODULE_NAME: Final[str] = "hydrophobic"
_LOGGER: Final[DockLogger] = DockLogger(_MODULE_NAME)


# -----------------------------------------------------------------------------
# Public interface
# -----------------------------------------------------------------------------

__all__: List[str] = []


# -----------------------------------------------------------------------------
# Generic type variables
# -----------------------------------------------------------------------------

T = TypeVar("T")
AtomT = TypeVar("AtomT")
ResidueT = TypeVar("ResidueT")
StructureT = TypeVar("StructureT")
InteractionT = TypeVar("InteractionT")


# -----------------------------------------------------------------------------
# Numeric aliases
# -----------------------------------------------------------------------------

Number: TypeAlias = Union[
    int,
    float,
    np.integer,
    np.floating,
]

FloatArray: TypeAlias = NDArray[np.float64]
IntegerArray: TypeAlias = NDArray[np.int64]
BooleanArray: TypeAlias = NDArray[np.bool_]

Coordinate: TypeAlias = FloatArray
CoordinateCollection: TypeAlias = FloatArray
DistanceMatrix: TypeAlias = FloatArray


# -----------------------------------------------------------------------------
# ChimeraX-compatible object aliases
# -----------------------------------------------------------------------------

AtomLike: TypeAlias = Any
ResidueLike: TypeAlias = Any
StructureLike: TypeAlias = Any
LigandLike: TypeAlias = Any
ReceptorLike: TypeAlias = Any
BondLike: TypeAlias = Any
ElementLike: TypeAlias = Any


# -----------------------------------------------------------------------------
# Collection aliases
# -----------------------------------------------------------------------------

AtomCollection: TypeAlias = Sequence[AtomLike]
ResidueCollection: TypeAlias = Sequence[ResidueLike]
StructureCollection: TypeAlias = Sequence[StructureLike]

AtomPair: TypeAlias = Tuple[
    AtomLike,
    AtomLike,
]

IndexedAtomPair: TypeAlias = Tuple[
    int,
    int,
]

AtomPairCollection: TypeAlias = Sequence[AtomPair]
IndexedAtomPairCollection: TypeAlias = Sequence[IndexedAtomPair]


# -----------------------------------------------------------------------------
# Hydrophobic-interaction semantic aliases
# -----------------------------------------------------------------------------

HydrophobicInteractionDirection: TypeAlias = Literal[
    "ligand_receptor",
    "receptor_ligand",
    "unknown",
]

HydrophobicInteractionType: TypeAlias = Literal[
    "aliphatic_aliphatic",
    "aliphatic_aromatic",
    "aromatic_aliphatic",
    "aromatic_aromatic",
    "mixed",
    "unknown",
]

HydrophobicClassification: TypeAlias = Literal[
    "very_strong",
    "strong",
    "moderate",
    "weak",
    "marginal",
    "unknown",
]

HydrophobicAtomType: TypeAlias = Literal[
    "aliphatic",
    "aromatic",
    "mixed",
    "non_hydrophobic",
    "unknown",
]

HydrophobicAtomRole: TypeAlias = Literal[
    "receptor",
    "ligand",
    "unknown",
]

HydrophobicDetectionMethod: TypeAlias = Literal[
    "atomic",
    "grouped",
    "inferred",
    "unknown",
]

HydrophobicInteractionIdentifier: TypeAlias = Tuple[
    str,
    str,
    Optional[str],
]

HydrophobicResidueIdentifier: TypeAlias = Tuple[
    str,
    Optional[str],
    Optional[int],
    Optional[str],
]


# -----------------------------------------------------------------------------
# Mapping, metadata and statistics aliases
# -----------------------------------------------------------------------------

Metadata: TypeAlias = Mapping[
    str,
    Any,
]

MutableMetadata: TypeAlias = MutableMapping[
    str,
    Any,
]

Statistics: TypeAlias = Dict[
    str,
    Any,
]

HydrophobicMetadata: TypeAlias = Mapping[
    str,
    Any,
]

MutableHydrophobicMetadata: TypeAlias = MutableMapping[
    str,
    Any,
]

ElementSet: TypeAlias = FrozenSet[str]
ResidueNameSet: TypeAlias = FrozenSet[str]
AtomNameSet: TypeAlias = FrozenSet[str]
AtomicNumberSet: TypeAlias = FrozenSet[int]


# -----------------------------------------------------------------------------
# Callable aliases
# -----------------------------------------------------------------------------

AtomPredicate: TypeAlias = Callable[
    [AtomLike],
    bool,
]

AtomPairPredicate: TypeAlias = Callable[
    [AtomLike, AtomLike],
    bool,
]

ResiduePredicate: TypeAlias = Callable[
    [ResidueLike],
    bool,
]

NeighborResolver: TypeAlias = Callable[
    [AtomLike],
    Iterable[AtomLike],
]

BondResolver: TypeAlias = Callable[
    [AtomLike],
    Iterable[AtomLike],
]

CoordinateResolver: TypeAlias = Callable[
    [AtomLike],
    Coordinate,
]


# -----------------------------------------------------------------------------
# Minimal structural protocols
# -----------------------------------------------------------------------------

@runtime_checkable
class CoordinateProvider(Protocol):
    """
    Protocol for objects exposing Cartesian coordinates.

    ChimeraX atoms do not need to formally inherit from this protocol.
    Structural compatibility is sufficient.
    """

    @property
    def coord(self) -> Any:
        """Return the Cartesian coordinate of the object."""


@runtime_checkable
class ElementProvider(Protocol):
    """Protocol for atom-like objects exposing element information."""

    @property
    def element(self) -> Any:
        """Return an element-like object."""


@runtime_checkable
class AtomNameProvider(Protocol):
    """Protocol for atom-like objects exposing an atom name."""

    @property
    def name(self) -> Any:
        """Return the atom name."""


@runtime_checkable
class AtomicNumberProvider(Protocol):
    """Protocol for objects exposing an atomic number."""

    @property
    def atomic_number(self) -> Any:
        """Return the atomic number."""


@runtime_checkable
class ResidueProvider(Protocol):
    """Protocol for atom-like objects exposing a parent residue."""

    @property
    def residue(self) -> Any:
        """Return the parent residue."""


@runtime_checkable
class ResidueNameProvider(Protocol):
    """Protocol for residue-like objects exposing a residue name."""

    @property
    def name(self) -> Any:
        """Return the residue name."""


@runtime_checkable
class ResidueNumberProvider(Protocol):
    """Protocol for residue-like objects exposing a residue number."""

    @property
    def number(self) -> Any:
        """Return the residue number."""


@runtime_checkable
class ChainProvider(Protocol):
    """Protocol for residue-like objects exposing chain information."""

    @property
    def chain_id(self) -> Any:
        """Return the chain identifier."""


@runtime_checkable
class NeighborProvider(Protocol):
    """Protocol for atom-like objects exposing bonded neighbors."""

    @property
    def neighbors(self) -> Any:
        """Return bonded neighboring atoms."""


@runtime_checkable
class BondsProvider(Protocol):
    """Protocol for atom-like objects exposing bonded objects."""

    @property
    def bonds(self) -> Any:
        """Return bonds associated with the atom."""


@runtime_checkable
class FormalChargeProvider(Protocol):
    """Protocol for atom-like objects exposing a formal charge."""

    @property
    def formal_charge(self) -> Any:
        """Return the formal charge."""


@runtime_checkable
class ChargeProvider(Protocol):
    """Protocol for atom-like objects exposing a partial charge."""

    @property
    def charge(self) -> Any:
        """Return the partial atomic charge."""


@runtime_checkable
class StructureProvider(Protocol):
    """Protocol for atom-like objects exposing a parent structure."""

    @property
    def structure(self) -> Any:
        """Return the parent molecular structure."""


# -----------------------------------------------------------------------------
# Empty immutable objects
# -----------------------------------------------------------------------------

_EMPTY_METADATA: Final[Mapping[str, Any]] = MappingProxyType({})

_EMPTY_ATOM_TUPLE: Final[Tuple[AtomLike, ...]] = ()
_EMPTY_RESIDUE_TUPLE: Final[Tuple[ResidueLike, ...]] = ()
_EMPTY_ATOM_PAIR_TUPLE: Final[Tuple[AtomPair, ...]] = ()
_EMPTY_INDEXED_ATOM_PAIR_TUPLE: Final[Tuple[IndexedAtomPair, ...]] = ()
_EMPTY_HYDROPHOBIC_TUPLE: Final[Tuple[Any, ...]] = ()
_EMPTY_RESIDUE_KEY_TUPLE: Final[Tuple[ResidueContactKey, ...]] = ()


# -----------------------------------------------------------------------------
# Initial public names
# -----------------------------------------------------------------------------

_SECTION_1_PUBLIC_NAMES: Final[Tuple[str, ...]] = (
    # Numeric aliases
    "Number",
    "FloatArray",
    "IntegerArray",
    "BooleanArray",
    "Coordinate",
    "CoordinateCollection",
    "DistanceMatrix",

    # ChimeraX-compatible aliases
    "AtomLike",
    "ResidueLike",
    "StructureLike",
    "LigandLike",
    "ReceptorLike",
    "BondLike",
    "ElementLike",

    # Collection aliases
    "AtomCollection",
    "ResidueCollection",
    "StructureCollection",
    "AtomPair",
    "IndexedAtomPair",
    "AtomPairCollection",
    "IndexedAtomPairCollection",

    # Hydrophobic semantic aliases
    "HydrophobicInteractionDirection",
    "HydrophobicInteractionType",
    "HydrophobicClassification",
    "HydrophobicAtomType",
    "HydrophobicAtomRole",
    "HydrophobicDetectionMethod",
    "HydrophobicInteractionIdentifier",
    "HydrophobicResidueIdentifier",

    # Mapping and metadata aliases
    "Metadata",
    "MutableMetadata",
    "Statistics",
    "HydrophobicMetadata",
    "MutableHydrophobicMetadata",
    "ElementSet",
    "ResidueNameSet",
    "AtomNameSet",
    "AtomicNumberSet",

    # Callable aliases
    "AtomPredicate",
    "AtomPairPredicate",
    "ResiduePredicate",
    "NeighborResolver",
    "BondResolver",
    "CoordinateResolver",

    # Structural protocols
    "CoordinateProvider",
    "ElementProvider",
    "AtomNameProvider",
    "AtomicNumberProvider",
    "ResidueProvider",
    "ResidueNameProvider",
    "ResidueNumberProvider",
    "ChainProvider",
    "NeighborProvider",
    "BondsProvider",
    "FormalChargeProvider",
    "ChargeProvider",
    "StructureProvider",
)

for public_name in _SECTION_1_PUBLIC_NAMES:
    if public_name not in __all__:
        __all__.append(public_name)


# =============================================================================
# End of Section 1
# =============================================================================


# =============================================================================
# Section 2 — Geometric and chemical constants
# =============================================================================


# -----------------------------------------------------------------------------
# Hydrophobic-interaction directions
# -----------------------------------------------------------------------------

HYDROPHOBIC_DIRECTION_UNKNOWN: Final[
    HydrophobicInteractionDirection
] = "unknown"

HYDROPHOBIC_DIRECTION_LIGAND_RECEPTOR: Final[
    HydrophobicInteractionDirection
] = "ligand_receptor"

HYDROPHOBIC_DIRECTION_RECEPTOR_LIGAND: Final[
    HydrophobicInteractionDirection
] = "receptor_ligand"

_VALID_HYDROPHOBIC_DIRECTIONS: Final[FrozenSet[str]] = frozenset(
    {
        HYDROPHOBIC_DIRECTION_UNKNOWN,
        HYDROPHOBIC_DIRECTION_LIGAND_RECEPTOR,
        HYDROPHOBIC_DIRECTION_RECEPTOR_LIGAND,
    }
)


# -----------------------------------------------------------------------------
# Hydrophobic-interaction types
# -----------------------------------------------------------------------------

HYDROPHOBIC_TYPE_UNKNOWN: Final[
    HydrophobicInteractionType
] = "unknown"

HYDROPHOBIC_TYPE_ALIPHATIC_ALIPHATIC: Final[
    HydrophobicInteractionType
] = "aliphatic_aliphatic"

HYDROPHOBIC_TYPE_ALIPHATIC_AROMATIC: Final[
    HydrophobicInteractionType
] = "aliphatic_aromatic"

HYDROPHOBIC_TYPE_AROMATIC_ALIPHATIC: Final[
    HydrophobicInteractionType
] = "aromatic_aliphatic"

HYDROPHOBIC_TYPE_AROMATIC_AROMATIC: Final[
    HydrophobicInteractionType
] = "aromatic_aromatic"

HYDROPHOBIC_TYPE_MIXED: Final[
    HydrophobicInteractionType
] = "mixed"

_VALID_HYDROPHOBIC_TYPES: Final[FrozenSet[str]] = frozenset(
    {
        HYDROPHOBIC_TYPE_UNKNOWN,
        HYDROPHOBIC_TYPE_ALIPHATIC_ALIPHATIC,
        HYDROPHOBIC_TYPE_ALIPHATIC_AROMATIC,
        HYDROPHOBIC_TYPE_AROMATIC_ALIPHATIC,
        HYDROPHOBIC_TYPE_AROMATIC_AROMATIC,
        HYDROPHOBIC_TYPE_MIXED,
    }
)


# -----------------------------------------------------------------------------
# Hydrophobic classifications
# -----------------------------------------------------------------------------

HYDROPHOBIC_CLASS_UNKNOWN: Final[
    HydrophobicClassification
] = "unknown"

HYDROPHOBIC_CLASS_MARGINAL: Final[
    HydrophobicClassification
] = "marginal"

HYDROPHOBIC_CLASS_WEAK: Final[
    HydrophobicClassification
] = "weak"

HYDROPHOBIC_CLASS_MODERATE: Final[
    HydrophobicClassification
] = "moderate"

HYDROPHOBIC_CLASS_STRONG: Final[
    HydrophobicClassification
] = "strong"

HYDROPHOBIC_CLASS_VERY_STRONG: Final[
    HydrophobicClassification
] = "very_strong"

_VALID_HYDROPHOBIC_CLASSIFICATIONS: Final[FrozenSet[str]] = frozenset(
    {
        HYDROPHOBIC_CLASS_UNKNOWN,
        HYDROPHOBIC_CLASS_MARGINAL,
        HYDROPHOBIC_CLASS_WEAK,
        HYDROPHOBIC_CLASS_MODERATE,
        HYDROPHOBIC_CLASS_STRONG,
        HYDROPHOBIC_CLASS_VERY_STRONG,
    }
)


# -----------------------------------------------------------------------------
# Hydrophobic atom types
# -----------------------------------------------------------------------------

HYDROPHOBIC_ATOM_TYPE_UNKNOWN: Final[
    HydrophobicAtomType
] = "unknown"

HYDROPHOBIC_ATOM_TYPE_NON_HYDROPHOBIC: Final[
    HydrophobicAtomType
] = "non_hydrophobic"

HYDROPHOBIC_ATOM_TYPE_ALIPHATIC: Final[
    HydrophobicAtomType
] = "aliphatic"

HYDROPHOBIC_ATOM_TYPE_AROMATIC: Final[
    HydrophobicAtomType
] = "aromatic"

HYDROPHOBIC_ATOM_TYPE_MIXED: Final[
    HydrophobicAtomType
] = "mixed"

_VALID_HYDROPHOBIC_ATOM_TYPES: Final[FrozenSet[str]] = frozenset(
    {
        HYDROPHOBIC_ATOM_TYPE_UNKNOWN,
        HYDROPHOBIC_ATOM_TYPE_NON_HYDROPHOBIC,
        HYDROPHOBIC_ATOM_TYPE_ALIPHATIC,
        HYDROPHOBIC_ATOM_TYPE_AROMATIC,
        HYDROPHOBIC_ATOM_TYPE_MIXED,
    }
)


# -----------------------------------------------------------------------------
# Hydrophobic atom roles
# -----------------------------------------------------------------------------

HYDROPHOBIC_ROLE_UNKNOWN: Final[
    HydrophobicAtomRole
] = "unknown"

HYDROPHOBIC_ROLE_RECEPTOR: Final[
    HydrophobicAtomRole
] = "receptor"

HYDROPHOBIC_ROLE_LIGAND: Final[
    HydrophobicAtomRole
] = "ligand"

_VALID_HYDROPHOBIC_ROLES: Final[FrozenSet[str]] = frozenset(
    {
        HYDROPHOBIC_ROLE_UNKNOWN,
        HYDROPHOBIC_ROLE_RECEPTOR,
        HYDROPHOBIC_ROLE_LIGAND,
    }
)


# -----------------------------------------------------------------------------
# Hydrophobic detection methods
# -----------------------------------------------------------------------------

HYDROPHOBIC_METHOD_UNKNOWN: Final[
    HydrophobicDetectionMethod
] = "unknown"

HYDROPHOBIC_METHOD_ATOMIC: Final[
    HydrophobicDetectionMethod
] = "atomic"

HYDROPHOBIC_METHOD_GROUPED: Final[
    HydrophobicDetectionMethod
] = "grouped"

HYDROPHOBIC_METHOD_INFERRED: Final[
    HydrophobicDetectionMethod
] = "inferred"

_VALID_HYDROPHOBIC_METHODS: Final[FrozenSet[str]] = frozenset(
    {
        HYDROPHOBIC_METHOD_UNKNOWN,
        HYDROPHOBIC_METHOD_ATOMIC,
        HYDROPHOBIC_METHOD_GROUPED,
        HYDROPHOBIC_METHOD_INFERRED,
    }
)


# -----------------------------------------------------------------------------
# Chemically relevant elements
# -----------------------------------------------------------------------------

HYDROGEN_ELEMENT: Final[str] = "H"
CARBON_ELEMENT: Final[str] = "C"
NITROGEN_ELEMENT: Final[str] = "N"
OXYGEN_ELEMENT: Final[str] = "O"
SULFUR_ELEMENT: Final[str] = "S"
PHOSPHORUS_ELEMENT: Final[str] = "P"
FLUORINE_ELEMENT: Final[str] = "F"
CHLORINE_ELEMENT: Final[str] = "CL"
BROMINE_ELEMENT: Final[str] = "BR"
IODINE_ELEMENT: Final[str] = "I"

HALOGEN_ELEMENTS: Final[ElementSet] = frozenset(
    {
        FLUORINE_ELEMENT,
        CHLORINE_ELEMENT,
        BROMINE_ELEMENT,
        IODINE_ELEMENT,
    }
)

POLAR_HETEROATOM_ELEMENTS: Final[ElementSet] = frozenset(
    {
        NITROGEN_ELEMENT,
        OXYGEN_ELEMENT,
        SULFUR_ELEMENT,
        PHOSPHORUS_ELEMENT,
    }
)

COMMON_ORGANIC_ELEMENTS: Final[ElementSet] = frozenset(
    {
        HYDROGEN_ELEMENT,
        CARBON_ELEMENT,
        NITROGEN_ELEMENT,
        OXYGEN_ELEMENT,
        SULFUR_ELEMENT,
        PHOSPHORUS_ELEMENT,
        FLUORINE_ELEMENT,
        CHLORINE_ELEMENT,
        BROMINE_ELEMENT,
        IODINE_ELEMENT,
    }
)

PRIMARY_HYDROPHOBIC_ELEMENTS: Final[ElementSet] = frozenset(
    {
        CARBON_ELEMENT,
    }
)

CONDITIONALLY_HYDROPHOBIC_ELEMENTS: Final[ElementSet] = frozenset(
    {
        SULFUR_ELEMENT,
        FLUORINE_ELEMENT,
        CHLORINE_ELEMENT,
        BROMINE_ELEMENT,
        IODINE_ELEMENT,
    }
)

SUPPORTED_HYDROPHOBIC_ELEMENTS: Final[ElementSet] = frozenset(
    set(PRIMARY_HYDROPHOBIC_ELEMENTS)
    | set(CONDITIONALLY_HYDROPHOBIC_ELEMENTS)
)


# -----------------------------------------------------------------------------
# Atomic-number constants
# -----------------------------------------------------------------------------

HYDROGEN_ATOMIC_NUMBER: Final[int] = 1
CARBON_ATOMIC_NUMBER: Final[int] = 6
NITROGEN_ATOMIC_NUMBER: Final[int] = 7
OXYGEN_ATOMIC_NUMBER: Final[int] = 8
FLUORINE_ATOMIC_NUMBER: Final[int] = 9
PHOSPHORUS_ATOMIC_NUMBER: Final[int] = 15
SULFUR_ATOMIC_NUMBER: Final[int] = 16
CHLORINE_ATOMIC_NUMBER: Final[int] = 17
BROMINE_ATOMIC_NUMBER: Final[int] = 35
IODINE_ATOMIC_NUMBER: Final[int] = 53

PRIMARY_HYDROPHOBIC_ATOMIC_NUMBERS: Final[AtomicNumberSet] = frozenset(
    {
        CARBON_ATOMIC_NUMBER,
    }
)

CONDITIONALLY_HYDROPHOBIC_ATOMIC_NUMBERS: Final[
    AtomicNumberSet
] = frozenset(
    {
        SULFUR_ATOMIC_NUMBER,
        FLUORINE_ATOMIC_NUMBER,
        CHLORINE_ATOMIC_NUMBER,
        BROMINE_ATOMIC_NUMBER,
        IODINE_ATOMIC_NUMBER,
    }
)

SUPPORTED_HYDROPHOBIC_ATOMIC_NUMBERS: Final[
    AtomicNumberSet
] = frozenset(
    set(PRIMARY_HYDROPHOBIC_ATOMIC_NUMBERS)
    | set(CONDITIONALLY_HYDROPHOBIC_ATOMIC_NUMBERS)
)


# -----------------------------------------------------------------------------
# Standard residue classes
# -----------------------------------------------------------------------------

STANDARD_AMINO_ACID_NAMES: Final[ResidueNameSet] = frozenset(
    {
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
)

DEFAULT_HYDROPHOBIC_RESIDUE_NAMES: Final[ResidueNameSet] = frozenset(
    {
        "ALA",
        "VAL",
        "LEU",
        "ILE",
        "MET",
        "PRO",
        "PHE",
        "TRP",
        "TYR",
    }
)

STRONGLY_HYDROPHOBIC_RESIDUE_NAMES: Final[
    ResidueNameSet
] = frozenset(
    {
        "VAL",
        "LEU",
        "ILE",
        "MET",
        "PHE",
        "TRP",
    }
)

MODERATELY_HYDROPHOBIC_RESIDUE_NAMES: Final[
    ResidueNameSet
] = frozenset(
    {
        "ALA",
        "PRO",
        "TYR",
    }
)

AROMATIC_RESIDUE_NAMES: Final[ResidueNameSet] = frozenset(
    {
        "PHE",
        "TYR",
        "TRP",
        "HIS",
        "HID",
        "HIE",
        "HIP",
        "HSD",
        "HSE",
        "HSP",
    }
)

ALIPHATIC_RESIDUE_NAMES: Final[ResidueNameSet] = frozenset(
    {
        "ALA",
        "VAL",
        "LEU",
        "ILE",
        "MET",
        "PRO",
    }
)

HISTIDINE_RESIDUE_NAMES: Final[ResidueNameSet] = frozenset(
    {
        "HIS",
        "HID",
        "HIE",
        "HIP",
        "HSD",
        "HSE",
        "HSP",
    }
)

CYSTEINE_RESIDUE_NAMES: Final[ResidueNameSet] = frozenset(
    {
        "CYS",
        "CYM",
        "CYX",
    }
)

WATER_RESIDUE_NAMES: Final[ResidueNameSet] = frozenset(
    {
        "HOH",
        "WAT",
        "H2O",
        "SOL",
        "TIP3",
        "TIP3P",
        "TIP4",
        "TIP4P",
        "TIP5",
        "TIP5P",
        "SPC",
        "SPCE",
    }
)

NUCLEIC_ACID_RESIDUE_NAMES: Final[ResidueNameSet] = frozenset(
    {
        "A",
        "C",
        "G",
        "T",
        "U",
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
)


# -----------------------------------------------------------------------------
# Protein atom-name classes
# -----------------------------------------------------------------------------

PROTEIN_BACKBONE_ATOM_NAMES: Final[AtomNameSet] = frozenset(
    {
        "N",
        "CA",
        "C",
        "O",
        "OXT",
        "H",
        "HN",
        "HA",
        "HA2",
        "HA3",
    }
)

PROTEIN_BACKBONE_CARBON_NAMES: Final[AtomNameSet] = frozenset(
    {
        "CA",
        "C",
    }
)

PROTEIN_CARBONYL_CARBON_NAMES: Final[AtomNameSet] = frozenset(
    {
        "C",
    }
)

PROTEIN_ALIPHATIC_CARBON_NAMES: Final[AtomNameSet] = frozenset(
    {
        "CA",
        "CB",
        "CG",
        "CG1",
        "CG2",
        "CD",
        "CD1",
        "CD2",
        "CE",
        "CE1",
        "CE2",
        "CE3",
        "CZ",
        "CZ2",
        "CZ3",
        "CH2",
    }
)

ALA_HYDROPHOBIC_ATOM_NAMES: Final[AtomNameSet] = frozenset(
    {
        "CB",
    }
)

VAL_HYDROPHOBIC_ATOM_NAMES: Final[AtomNameSet] = frozenset(
    {
        "CB",
        "CG1",
        "CG2",
    }
)

LEU_HYDROPHOBIC_ATOM_NAMES: Final[AtomNameSet] = frozenset(
    {
        "CB",
        "CG",
        "CD1",
        "CD2",
    }
)

ILE_HYDROPHOBIC_ATOM_NAMES: Final[AtomNameSet] = frozenset(
    {
        "CB",
        "CG1",
        "CG2",
        "CD1",
    }
)

MET_HYDROPHOBIC_ATOM_NAMES: Final[AtomNameSet] = frozenset(
    {
        "CB",
        "CG",
        "SD",
        "CE",
    }
)

PRO_HYDROPHOBIC_ATOM_NAMES: Final[AtomNameSet] = frozenset(
    {
        "CB",
        "CG",
        "CD",
    }
)

PHE_HYDROPHOBIC_ATOM_NAMES: Final[AtomNameSet] = frozenset(
    {
        "CB",
        "CG",
        "CD1",
        "CD2",
        "CE1",
        "CE2",
        "CZ",
    }
)

TYR_HYDROPHOBIC_ATOM_NAMES: Final[AtomNameSet] = frozenset(
    {
        "CB",
        "CG",
        "CD1",
        "CD2",
        "CE1",
        "CE2",
        "CZ",
    }
)

TRP_HYDROPHOBIC_ATOM_NAMES: Final[AtomNameSet] = frozenset(
    {
        "CB",
        "CG",
        "CD1",
        "CD2",
        "CE2",
        "CE3",
        "CZ2",
        "CZ3",
        "CH2",
    }
)

HIS_HYDROPHOBIC_ATOM_NAMES: Final[AtomNameSet] = frozenset(
    {
        "CB",
        "CG",
        "CD2",
        "CE1",
    }
)

CYS_HYDROPHOBIC_ATOM_NAMES: Final[AtomNameSet] = frozenset(
    {
        "CB",
        "SG",
    }
)

HYDROPHOBIC_PROTEIN_ATOMS_BY_RESIDUE: Final[
    Mapping[str, AtomNameSet]
] = MappingProxyType(
    {
        "ALA": ALA_HYDROPHOBIC_ATOM_NAMES,
        "VAL": VAL_HYDROPHOBIC_ATOM_NAMES,
        "LEU": LEU_HYDROPHOBIC_ATOM_NAMES,
        "ILE": ILE_HYDROPHOBIC_ATOM_NAMES,
        "MET": MET_HYDROPHOBIC_ATOM_NAMES,
        "PRO": PRO_HYDROPHOBIC_ATOM_NAMES,
        "PHE": PHE_HYDROPHOBIC_ATOM_NAMES,
        "TYR": TYR_HYDROPHOBIC_ATOM_NAMES,
        "TRP": TRP_HYDROPHOBIC_ATOM_NAMES,
        "HIS": HIS_HYDROPHOBIC_ATOM_NAMES,
        "HID": HIS_HYDROPHOBIC_ATOM_NAMES,
        "HIE": HIS_HYDROPHOBIC_ATOM_NAMES,
        "HIP": HIS_HYDROPHOBIC_ATOM_NAMES,
        "HSD": HIS_HYDROPHOBIC_ATOM_NAMES,
        "HSE": HIS_HYDROPHOBIC_ATOM_NAMES,
        "HSP": HIS_HYDROPHOBIC_ATOM_NAMES,
        "CYS": CYS_HYDROPHOBIC_ATOM_NAMES,
        "CYM": CYS_HYDROPHOBIC_ATOM_NAMES,
        "CYX": CYS_HYDROPHOBIC_ATOM_NAMES,
    }
)


# -----------------------------------------------------------------------------
# Aromatic atom-name definitions
# -----------------------------------------------------------------------------

PHE_AROMATIC_ATOM_NAMES: Final[AtomNameSet] = frozenset(
    {
        "CG",
        "CD1",
        "CD2",
        "CE1",
        "CE2",
        "CZ",
    }
)

TYR_AROMATIC_ATOM_NAMES: Final[AtomNameSet] = frozenset(
    {
        "CG",
        "CD1",
        "CD2",
        "CE1",
        "CE2",
        "CZ",
    }
)

TRP_AROMATIC_ATOM_NAMES: Final[AtomNameSet] = frozenset(
    {
        "CG",
        "CD1",
        "CD2",
        "NE1",
        "CE2",
        "CE3",
        "CZ2",
        "CZ3",
        "CH2",
    }
)

HIS_AROMATIC_ATOM_NAMES: Final[AtomNameSet] = frozenset(
    {
        "CG",
        "ND1",
        "CD2",
        "CE1",
        "NE2",
    }
)

AROMATIC_PROTEIN_ATOMS_BY_RESIDUE: Final[
    Mapping[str, AtomNameSet]
] = MappingProxyType(
    {
        "PHE": PHE_AROMATIC_ATOM_NAMES,
        "TYR": TYR_AROMATIC_ATOM_NAMES,
        "TRP": TRP_AROMATIC_ATOM_NAMES,
        "HIS": HIS_AROMATIC_ATOM_NAMES,
        "HID": HIS_AROMATIC_ATOM_NAMES,
        "HIE": HIS_AROMATIC_ATOM_NAMES,
        "HIP": HIS_AROMATIC_ATOM_NAMES,
        "HSD": HIS_AROMATIC_ATOM_NAMES,
        "HSE": HIS_AROMATIC_ATOM_NAMES,
        "HSP": HIS_AROMATIC_ATOM_NAMES,
    }
)


# -----------------------------------------------------------------------------
# Polar or chemically excluded protein atoms
# -----------------------------------------------------------------------------

POLAR_PROTEIN_ATOM_NAMES: Final[AtomNameSet] = frozenset(
    {
        "N",
        "O",
        "OXT",
        "OG",
        "OG1",
        "OH",
        "OD1",
        "OD2",
        "OE1",
        "OE2",
        "ND1",
        "ND2",
        "NE",
        "NE1",
        "NE2",
        "NH1",
        "NH2",
        "NZ",
        "SG",
        "SD",
    }
)

CHARGED_PROTEIN_ATOM_NAMES: Final[AtomNameSet] = frozenset(
    {
        "OD1",
        "OD2",
        "OE1",
        "OE2",
        "NZ",
        "NH1",
        "NH2",
        "NE",
    }
)

CARBONYL_CARBON_ATOM_NAMES: Final[AtomNameSet] = frozenset(
    {
        "C",
        "CG",
        "CD",
    }
)


# -----------------------------------------------------------------------------
# Geometric defaults
# -----------------------------------------------------------------------------

DEFAULT_MINIMUM_HYDROPHOBIC_DISTANCE: Final[np.float64] = np.float64(
    2.50
)

DEFAULT_OPTIMAL_HYDROPHOBIC_DISTANCE: Final[np.float64] = np.float64(
    3.80
)

DEFAULT_MAXIMUM_HYDROPHOBIC_DISTANCE: Final[np.float64] = np.float64(
    4.50
)

DEFAULT_HYDROPHOBIC_DISTANCE_TOLERANCE: Final[np.float64] = np.float64(
    0.05
)

DEFAULT_GROUPING_DISTANCE: Final[np.float64] = np.float64(
    5.00
)

DEFAULT_LOCAL_ENVIRONMENT_RADIUS: Final[np.float64] = np.float64(
    2.20
)

DEFAULT_POLAR_NEIGHBOR_RADIUS: Final[np.float64] = np.float64(
    1.95
)

DEFAULT_MAXIMUM_ABSOLUTE_PARTIAL_CHARGE: Final[
    np.float64
] = np.float64(
    0.50
)

DEFAULT_MAXIMUM_ABSOLUTE_FORMAL_CHARGE: Final[int] = 0

DEFAULT_MINIMUM_LOCAL_CONTACT_COUNT: Final[int] = 1

DEFAULT_MINIMUM_HOTSPOT_CONTACT_COUNT: Final[int] = 3

DEFAULT_MAXIMUM_POLAR_NEIGHBORS: Final[int] = 1


# -----------------------------------------------------------------------------
# Distance thresholds used for classification
# -----------------------------------------------------------------------------

HYDROPHOBIC_VERY_STRONG_MAX_DISTANCE: Final[
    np.float64
] = np.float64(
    3.60
)

HYDROPHOBIC_STRONG_MAX_DISTANCE: Final[np.float64] = np.float64(
    3.90
)

HYDROPHOBIC_MODERATE_MAX_DISTANCE: Final[np.float64] = np.float64(
    4.20
)

HYDROPHOBIC_WEAK_MAX_DISTANCE: Final[np.float64] = np.float64(
    4.50
)

HYDROPHOBIC_MARGINAL_MAX_DISTANCE: Final[np.float64] = np.float64(
    5.00
)


# -----------------------------------------------------------------------------
# Local-density thresholds
# -----------------------------------------------------------------------------

HYDROPHOBIC_VERY_STRONG_MIN_LOCAL_CONTACTS: Final[int] = 4
HYDROPHOBIC_STRONG_MIN_LOCAL_CONTACTS: Final[int] = 3
HYDROPHOBIC_MODERATE_MIN_LOCAL_CONTACTS: Final[int] = 2
HYDROPHOBIC_WEAK_MIN_LOCAL_CONTACTS: Final[int] = 1


# -----------------------------------------------------------------------------
# Geometric-score weights
# -----------------------------------------------------------------------------

HYDROPHOBIC_DISTANCE_SCORE_WEIGHT: Final[np.float64] = np.float64(
    0.60
)

HYDROPHOBIC_DENSITY_SCORE_WEIGHT: Final[np.float64] = np.float64(
    0.25
)

HYDROPHOBIC_CHEMISTRY_SCORE_WEIGHT: Final[np.float64] = np.float64(
    0.15
)

HYDROPHOBIC_SCORE_WEIGHTS: Final[Mapping[str, np.float64]] = (
    MappingProxyType(
        {
            "distance": HYDROPHOBIC_DISTANCE_SCORE_WEIGHT,
            "density": HYDROPHOBIC_DENSITY_SCORE_WEIGHT,
            "chemistry": HYDROPHOBIC_CHEMISTRY_SCORE_WEIGHT,
        }
    )
)


# -----------------------------------------------------------------------------
# Atom-type score modifiers
# -----------------------------------------------------------------------------

HYDROPHOBIC_ATOM_TYPE_SCORE_MODIFIERS: Final[
    Mapping[str, np.float64]
] = MappingProxyType(
    {
        HYDROPHOBIC_ATOM_TYPE_ALIPHATIC: np.float64(1.00),
        HYDROPHOBIC_ATOM_TYPE_AROMATIC: np.float64(1.00),
        HYDROPHOBIC_ATOM_TYPE_MIXED: np.float64(0.90),
        HYDROPHOBIC_ATOM_TYPE_UNKNOWN: np.float64(0.50),
        HYDROPHOBIC_ATOM_TYPE_NON_HYDROPHOBIC: np.float64(0.00),
    }
)


# -----------------------------------------------------------------------------
# Interaction-type score modifiers
# -----------------------------------------------------------------------------

HYDROPHOBIC_INTERACTION_TYPE_SCORE_MODIFIERS: Final[
    Mapping[str, np.float64]
] = MappingProxyType(
    {
        HYDROPHOBIC_TYPE_ALIPHATIC_ALIPHATIC: np.float64(1.00),
        HYDROPHOBIC_TYPE_ALIPHATIC_AROMATIC: np.float64(1.00),
        HYDROPHOBIC_TYPE_AROMATIC_ALIPHATIC: np.float64(1.00),
        HYDROPHOBIC_TYPE_AROMATIC_AROMATIC: np.float64(0.95),
        HYDROPHOBIC_TYPE_MIXED: np.float64(0.85),
        HYDROPHOBIC_TYPE_UNKNOWN: np.float64(0.50),
    }
)


# -----------------------------------------------------------------------------
# Classification ranks and labels
# -----------------------------------------------------------------------------

HYDROPHOBIC_CLASSIFICATION_RANKS: Final[Mapping[str, int]] = (
    MappingProxyType(
        {
            HYDROPHOBIC_CLASS_UNKNOWN: 0,
            HYDROPHOBIC_CLASS_MARGINAL: 1,
            HYDROPHOBIC_CLASS_WEAK: 2,
            HYDROPHOBIC_CLASS_MODERATE: 3,
            HYDROPHOBIC_CLASS_STRONG: 4,
            HYDROPHOBIC_CLASS_VERY_STRONG: 5,
        }
    )
)

HYDROPHOBIC_CLASSIFICATION_BASE_SCORES: Final[
    Mapping[str, np.float64]
] = MappingProxyType(
    {
        HYDROPHOBIC_CLASS_UNKNOWN: np.float64(0.00),
        HYDROPHOBIC_CLASS_MARGINAL: np.float64(0.20),
        HYDROPHOBIC_CLASS_WEAK: np.float64(0.40),
        HYDROPHOBIC_CLASS_MODERATE: np.float64(0.60),
        HYDROPHOBIC_CLASS_STRONG: np.float64(0.80),
        HYDROPHOBIC_CLASS_VERY_STRONG: np.float64(1.00),
    }
)


# -----------------------------------------------------------------------------
# Numerically safe limits
# -----------------------------------------------------------------------------

MINIMUM_POSITIVE_DISTANCE: Final[np.float64] = np.float64(
    1.0e-8
)

MINIMUM_VECTOR_NORM: Final[np.float64] = np.float64(
    1.0e-12
)

MINIMUM_SCORE: Final[np.float64] = np.float64(
    0.0
)

MAXIMUM_SCORE: Final[np.float64] = np.float64(
    1.0
)

DEFAULT_COORDINATE_DECIMALS: Final[int] = 6
DEFAULT_DISTANCE_DECIMALS: Final[int] = 3
DEFAULT_SCORE_DECIMALS: Final[int] = 4


# -----------------------------------------------------------------------------
# Search and processing limits
# -----------------------------------------------------------------------------

DEFAULT_MAXIMUM_PAIR_ELEMENTS: Final[int] = 4_000_000
DEFAULT_HYDROPHOBIC_BLOCK_SIZE: Final[int] = 1_024
DEFAULT_MAXIMUM_HYDROPHOBIC_INTERACTIONS: Final[Optional[int]] = None


# -----------------------------------------------------------------------------
# Configuration lookup helpers
# -----------------------------------------------------------------------------

def _get_config_value(
    candidate_names: Sequence[str],
    default: T,
) -> T:
    """
    Retrieve the first available value from the configuration module.

    Parameters
    ----------
    candidate_names
        Candidate configuration attribute names, checked in order.
    default
        Value returned if no valid configuration attribute is found.

    Returns
    -------
    T
        Configured value or ``default``.

    Notes
    -----
    Multiple candidate names are accepted to preserve compatibility with
    future revisions of ``config.py``.
    """

    for candidate_name in candidate_names:
        try:
            value = getattr(
                config,
                candidate_name,
            )
        except (AttributeError, TypeError):
            continue

        if value is not None:
            return value

    return default


def _coerce_positive_float(
    value: Any,
    *,
    name: str,
    default: np.float64,
    allow_zero: bool = False,
) -> np.float64:
    """
    Convert a candidate value to a finite positive float.

    Invalid values are replaced with ``default``.
    """

    try:
        numeric_value = np.float64(value)
    except (TypeError, ValueError, OverflowError):
        try:
            _LOGGER.warning(
                f"Invalid configured value for {name!r}; "
                f"using default {float(default):g}."
            )
        except Exception:
            pass

        return np.float64(default)

    minimum_allowed = (
        np.float64(0.0)
        if allow_zero
        else MINIMUM_POSITIVE_DISTANCE
    )

    if (
        not np.isfinite(numeric_value)
        or numeric_value < minimum_allowed
    ):
        try:
            _LOGGER.warning(
                f"Configured value for {name!r} is outside the "
                f"accepted range; using default {float(default):g}."
            )
        except Exception:
            pass

        return np.float64(default)

    return numeric_value


def _coerce_non_negative_integer(
    value: Any,
    *,
    name: str,
    default: int,
) -> int:
    """
    Convert a candidate value to a non-negative integer.
    """

    if isinstance(value, bool):
        return int(default)

    try:
        integer_value = int(value)
    except (TypeError, ValueError, OverflowError):
        try:
            _LOGGER.warning(
                f"Invalid configured value for {name!r}; "
                f"using default {default}."
            )
        except Exception:
            pass

        return int(default)

    if integer_value < 0:
        try:
            _LOGGER.warning(
                f"Configured value for {name!r} must be "
                f"non-negative; using default {default}."
            )
        except Exception:
            pass

        return int(default)

    return integer_value


def _coerce_positive_integer(
    value: Any,
    *,
    name: str,
    default: int,
) -> int:
    """
    Convert a candidate value to a strictly positive integer.
    """

    integer_value = _coerce_non_negative_integer(
        value,
        name=name,
        default=default,
    )

    if integer_value == 0:
        try:
            _LOGGER.warning(
                f"Configured value for {name!r} must be positive; "
                f"using default {default}."
            )
        except Exception:
            pass

        return int(default)

    return integer_value


def _coerce_optional_positive_integer(
    value: Any,
    *,
    name: str,
    default: Optional[int],
) -> Optional[int]:
    """
    Convert a candidate value to ``None`` or a positive integer.
    """

    if value is None:
        return default

    if isinstance(value, bool):
        return default

    try:
        integer_value = int(value)
    except (TypeError, ValueError, OverflowError):
        try:
            _LOGGER.warning(
                f"Invalid configured value for {name!r}; "
                f"using default {default!r}."
            )
        except Exception:
            pass

        return default

    if integer_value <= 0:
        try:
            _LOGGER.warning(
                f"Configured value for {name!r} must be positive "
                f"or None; using default {default!r}."
            )
        except Exception:
            pass

        return default

    return integer_value


def _coerce_residue_name_set(
    value: Any,
    *,
    name: str,
    default: ResidueNameSet,
) -> ResidueNameSet:
    """
    Normalize a collection of residue names.

    Names are stripped and converted to uppercase.
    """

    if value is None:
        return default

    if isinstance(value, str):
        candidate_values: Iterable[Any] = (value,)
    else:
        try:
            candidate_values = tuple(value)
        except TypeError:
            try:
                _LOGGER.warning(
                    f"Invalid configured residue collection for "
                    f"{name!r}; using defaults."
                )
            except Exception:
                pass

            return default

    normalized_names: Set[str] = set()

    for candidate in candidate_values:
        if candidate is None:
            continue

        try:
            normalized_name = str(candidate).strip().upper()
        except Exception:
            continue

        if normalized_name:
            normalized_names.add(normalized_name)

    if not normalized_names:
        try:
            _LOGGER.warning(
                f"Configured residue collection for {name!r} is "
                "empty; using defaults."
            )
        except Exception:
            pass

        return default

    return frozenset(normalized_names)


# -----------------------------------------------------------------------------
# Public default-resolution functions
# -----------------------------------------------------------------------------

def get_default_minimum_hydrophobic_distance() -> np.float64:
    """
    Return the configured minimum hydrophobic-contact distance.
    """

    value = _get_config_value(
        (
            "HYDROPHOBIC_MIN_DISTANCE",
            "MIN_HYDROPHOBIC_DISTANCE",
            "DEFAULT_MINIMUM_HYDROPHOBIC_DISTANCE",
        ),
        DEFAULT_MINIMUM_HYDROPHOBIC_DISTANCE,
    )

    return _coerce_positive_float(
        value,
        name="minimum hydrophobic distance",
        default=DEFAULT_MINIMUM_HYDROPHOBIC_DISTANCE,
    )


def get_default_optimal_hydrophobic_distance() -> np.float64:
    """
    Return the configured optimal hydrophobic-contact distance.
    """

    value = _get_config_value(
        (
            "HYDROPHOBIC_OPTIMAL_DISTANCE",
            "OPTIMAL_HYDROPHOBIC_DISTANCE",
            "DEFAULT_OPTIMAL_HYDROPHOBIC_DISTANCE",
        ),
        DEFAULT_OPTIMAL_HYDROPHOBIC_DISTANCE,
    )

    return _coerce_positive_float(
        value,
        name="optimal hydrophobic distance",
        default=DEFAULT_OPTIMAL_HYDROPHOBIC_DISTANCE,
    )


def get_default_maximum_hydrophobic_distance() -> np.float64:
    """
    Return the configured maximum hydrophobic-contact distance.
    """

    value = _get_config_value(
        (
            "HYDROPHOBIC_DISTANCE",
            "HYDROPHOBIC_MAX_DISTANCE",
            "HYDROPHOBIC_DISTANCE_CUTOFF",
            "DEFAULT_MAXIMUM_HYDROPHOBIC_DISTANCE",
        ),
        DEFAULT_MAXIMUM_HYDROPHOBIC_DISTANCE,
    )

    return _coerce_positive_float(
        value,
        name="maximum hydrophobic distance",
        default=DEFAULT_MAXIMUM_HYDROPHOBIC_DISTANCE,
    )


def get_default_hydrophobic_distance_tolerance() -> np.float64:
    """
    Return the configured hydrophobic-distance tolerance.
    """

    value = _get_config_value(
        (
            "HYDROPHOBIC_DISTANCE_TOLERANCE",
            "DEFAULT_HYDROPHOBIC_DISTANCE_TOLERANCE",
            "DEFAULT_DISTANCE_TOLERANCE",
        ),
        DEFAULT_HYDROPHOBIC_DISTANCE_TOLERANCE,
    )

    return _coerce_positive_float(
        value,
        name="hydrophobic distance tolerance",
        default=DEFAULT_HYDROPHOBIC_DISTANCE_TOLERANCE,
        allow_zero=True,
    )


def get_default_grouping_distance() -> np.float64:
    """
    Return the distance used to group nearby hydrophobic contacts.
    """

    value = _get_config_value(
        (
            "HYDROPHOBIC_GROUPING_DISTANCE",
            "HYDROPHOBIC_GROUP_DISTANCE",
            "DEFAULT_HYDROPHOBIC_GROUPING_DISTANCE",
            "DEFAULT_GROUPING_DISTANCE",
        ),
        DEFAULT_GROUPING_DISTANCE,
    )

    return _coerce_positive_float(
        value,
        name="hydrophobic grouping distance",
        default=DEFAULT_GROUPING_DISTANCE,
    )


def get_default_local_environment_radius() -> np.float64:
    """
    Return the local radius used during chemical perception.
    """

    value = _get_config_value(
        (
            "HYDROPHOBIC_LOCAL_ENVIRONMENT_RADIUS",
            "HYDROPHOBIC_ENVIRONMENT_RADIUS",
            "DEFAULT_LOCAL_ENVIRONMENT_RADIUS",
        ),
        DEFAULT_LOCAL_ENVIRONMENT_RADIUS,
    )

    return _coerce_positive_float(
        value,
        name="hydrophobic local-environment radius",
        default=DEFAULT_LOCAL_ENVIRONMENT_RADIUS,
    )


def get_default_polar_neighbor_radius() -> np.float64:
    """
    Return the radius used to identify nearby polar heteroatoms.
    """

    value = _get_config_value(
        (
            "HYDROPHOBIC_POLAR_NEIGHBOR_RADIUS",
            "HYDROPHOBIC_POLAR_RADIUS",
            "DEFAULT_POLAR_NEIGHBOR_RADIUS",
        ),
        DEFAULT_POLAR_NEIGHBOR_RADIUS,
    )

    return _coerce_positive_float(
        value,
        name="hydrophobic polar-neighbor radius",
        default=DEFAULT_POLAR_NEIGHBOR_RADIUS,
    )


def get_default_maximum_absolute_partial_charge() -> np.float64:
    """
    Return the maximum absolute partial charge accepted by default.
    """

    value = _get_config_value(
        (
            "HYDROPHOBIC_MAX_ABSOLUTE_PARTIAL_CHARGE",
            "HYDROPHOBIC_MAX_PARTIAL_CHARGE",
            "DEFAULT_MAXIMUM_ABSOLUTE_PARTIAL_CHARGE",
        ),
        DEFAULT_MAXIMUM_ABSOLUTE_PARTIAL_CHARGE,
    )

    return _coerce_positive_float(
        value,
        name="maximum absolute hydrophobic partial charge",
        default=DEFAULT_MAXIMUM_ABSOLUTE_PARTIAL_CHARGE,
        allow_zero=True,
    )


def get_default_maximum_polar_neighbors() -> int:
    """
    Return the maximum number of polar neighbors accepted by default.
    """

    value = _get_config_value(
        (
            "HYDROPHOBIC_MAX_POLAR_NEIGHBORS",
            "DEFAULT_MAXIMUM_POLAR_NEIGHBORS",
        ),
        DEFAULT_MAXIMUM_POLAR_NEIGHBORS,
    )

    return _coerce_non_negative_integer(
        value,
        name="maximum polar neighbors",
        default=DEFAULT_MAXIMUM_POLAR_NEIGHBORS,
    )


def get_default_hydrophobic_residue_names() -> ResidueNameSet:
    """
    Return the configured set of hydrophobic receptor residues.
    """

    value = _get_config_value(
        (
            "HYDROPHOBIC_RESIDUES",
            "HYDROPHOBIC_RESIDUE_NAMES",
            "DEFAULT_HYDROPHOBIC_RESIDUES",
        ),
        DEFAULT_HYDROPHOBIC_RESIDUE_NAMES,
    )

    return _coerce_residue_name_set(
        value,
        name="hydrophobic residue names",
        default=DEFAULT_HYDROPHOBIC_RESIDUE_NAMES,
    )


def get_default_hydrophobic_block_size() -> int:
    """
    Return the configured pair-processing block size.
    """

    value = _get_config_value(
        (
            "HYDROPHOBIC_BLOCK_SIZE",
            "DEFAULT_HYDROPHOBIC_BLOCK_SIZE",
            "DEFAULT_BLOCK_SIZE",
        ),
        DEFAULT_HYDROPHOBIC_BLOCK_SIZE,
    )

    return _coerce_positive_integer(
        value,
        name="hydrophobic processing block size",
        default=DEFAULT_HYDROPHOBIC_BLOCK_SIZE,
    )


def get_default_maximum_pair_elements() -> int:
    """
    Return the maximum allowed full pair-matrix size.
    """

    value = _get_config_value(
        (
            "HYDROPHOBIC_MAXIMUM_PAIR_ELEMENTS",
            "HYDROPHOBIC_MAX_MATRIX_ELEMENTS",
            "DEFAULT_HYDROPHOBIC_MAXIMUM_PAIR_ELEMENTS",
            "DEFAULT_MAXIMUM_PAIR_ELEMENTS",
        ),
        DEFAULT_MAXIMUM_PAIR_ELEMENTS,
    )

    return _coerce_positive_integer(
        value,
        name="maximum hydrophobic pair elements",
        default=DEFAULT_MAXIMUM_PAIR_ELEMENTS,
    )


def get_default_maximum_hydrophobic_interactions() -> Optional[int]:
    """
    Return the optional maximum number of retained interactions.
    """

    value = _get_config_value(
        (
            "MAXIMUM_HYDROPHOBIC_INTERACTIONS",
            "HYDROPHOBIC_MAX_INTERACTIONS",
            "DEFAULT_MAXIMUM_HYDROPHOBIC_INTERACTIONS",
        ),
        DEFAULT_MAXIMUM_HYDROPHOBIC_INTERACTIONS,
    )

    return _coerce_optional_positive_integer(
        value,
        name="maximum hydrophobic interactions",
        default=DEFAULT_MAXIMUM_HYDROPHOBIC_INTERACTIONS,
    )


# -----------------------------------------------------------------------------
# Semantic validation helpers
# -----------------------------------------------------------------------------

def validate_hydrophobic_direction(
    direction: str,
) -> HydrophobicInteractionDirection:
    """
    Validate and normalize a hydrophobic-interaction direction.
    """

    if not isinstance(direction, str):
        raise TypeError(
            "Hydrophobic-interaction direction must be a string."
        )

    normalized_direction = direction.strip().lower()

    if normalized_direction not in _VALID_HYDROPHOBIC_DIRECTIONS:
        valid_directions = ", ".join(
            sorted(_VALID_HYDROPHOBIC_DIRECTIONS)
        )

        raise ValueError(
            f"Unsupported hydrophobic direction {direction!r}. "
            f"Expected one of: {valid_directions}."
        )

    return normalized_direction  # type: ignore[return-value]


def validate_hydrophobic_interaction_type(
    interaction_type: str,
) -> HydrophobicInteractionType:
    """
    Validate and normalize a hydrophobic-interaction type.
    """

    if not isinstance(interaction_type, str):
        raise TypeError(
            "Hydrophobic-interaction type must be a string."
        )

    normalized_type = interaction_type.strip().lower()

    if normalized_type not in _VALID_HYDROPHOBIC_TYPES:
        valid_types = ", ".join(
            sorted(_VALID_HYDROPHOBIC_TYPES)
        )

        raise ValueError(
            f"Unsupported hydrophobic-interaction type "
            f"{interaction_type!r}. Expected one of: {valid_types}."
        )

    return normalized_type  # type: ignore[return-value]


def validate_hydrophobic_classification(
    classification: str,
) -> HydrophobicClassification:
    """
    Validate and normalize a hydrophobic classification.
    """

    if not isinstance(classification, str):
        raise TypeError(
            "Hydrophobic classification must be a string."
        )

    normalized_classification = classification.strip().lower()

    if (
        normalized_classification
        not in _VALID_HYDROPHOBIC_CLASSIFICATIONS
    ):
        valid_classifications = ", ".join(
            sorted(_VALID_HYDROPHOBIC_CLASSIFICATIONS)
        )

        raise ValueError(
            f"Unsupported hydrophobic classification "
            f"{classification!r}. Expected one of: "
            f"{valid_classifications}."
        )

    return normalized_classification  # type: ignore[return-value]


def validate_hydrophobic_atom_type(
    atom_type: str,
) -> HydrophobicAtomType:
    """
    Validate and normalize a hydrophobic atom type.
    """

    if not isinstance(atom_type, str):
        raise TypeError(
            "Hydrophobic atom type must be a string."
        )

    normalized_type = atom_type.strip().lower()

    if normalized_type not in _VALID_HYDROPHOBIC_ATOM_TYPES:
        valid_types = ", ".join(
            sorted(_VALID_HYDROPHOBIC_ATOM_TYPES)
        )

        raise ValueError(
            f"Unsupported hydrophobic atom type {atom_type!r}. "
            f"Expected one of: {valid_types}."
        )

    return normalized_type  # type: ignore[return-value]


def validate_hydrophobic_atom_role(
    role: str,
) -> HydrophobicAtomRole:
    """
    Validate and normalize a hydrophobic atom role.
    """

    if not isinstance(role, str):
        raise TypeError(
            "Hydrophobic atom role must be a string."
        )

    normalized_role = role.strip().lower()

    if normalized_role not in _VALID_HYDROPHOBIC_ROLES:
        valid_roles = ", ".join(
            sorted(_VALID_HYDROPHOBIC_ROLES)
        )

        raise ValueError(
            f"Unsupported hydrophobic atom role {role!r}. "
            f"Expected one of: {valid_roles}."
        )

    return normalized_role  # type: ignore[return-value]


def validate_hydrophobic_detection_method(
    method: str,
) -> HydrophobicDetectionMethod:
    """
    Validate and normalize a hydrophobic detection method.
    """

    if not isinstance(method, str):
        raise TypeError(
            "Hydrophobic detection method must be a string."
        )

    normalized_method = method.strip().lower()

    if normalized_method not in _VALID_HYDROPHOBIC_METHODS:
        valid_methods = ", ".join(
            sorted(_VALID_HYDROPHOBIC_METHODS)
        )

        raise ValueError(
            f"Unsupported hydrophobic detection method "
            f"{method!r}. Expected one of: {valid_methods}."
        )

    return normalized_method  # type: ignore[return-value]


# -----------------------------------------------------------------------------
# Geometric validation helpers
# -----------------------------------------------------------------------------

def validate_hydrophobic_distance_limits(
    minimum_distance: Number,
    maximum_distance: Number,
) -> Tuple[np.float64, np.float64]:
    """
    Validate minimum and maximum hydrophobic distances.

    Raises
    ------
    ValueError
        If the minimum distance exceeds the maximum distance.
    """

    validated_minimum = _coerce_positive_float(
        minimum_distance,
        name="minimum hydrophobic distance",
        default=DEFAULT_MINIMUM_HYDROPHOBIC_DISTANCE,
    )

    validated_maximum = _coerce_positive_float(
        maximum_distance,
        name="maximum hydrophobic distance",
        default=DEFAULT_MAXIMUM_HYDROPHOBIC_DISTANCE,
    )

    if validated_minimum > validated_maximum:
        raise ValueError(
            "Minimum hydrophobic distance cannot exceed the "
            "maximum hydrophobic distance."
        )

    return (
        validated_minimum,
        validated_maximum,
    )


def validate_hydrophobic_score(
    score: Number,
) -> np.float64:
    """
    Validate and clamp a normalized hydrophobic score to [0, 1].
    """

    try:
        numeric_score = np.float64(score)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(
            "Hydrophobic score must be numeric."
        ) from exc

    if not np.isfinite(numeric_score):
        raise ValueError(
            "Hydrophobic score must be finite."
        )

    return np.float64(
        np.clip(
            numeric_score,
            MINIMUM_SCORE,
            MAXIMUM_SCORE,
        )
    )


# -----------------------------------------------------------------------------
# Section 2 public names
# -----------------------------------------------------------------------------

_SECTION_2_PUBLIC_NAMES: Final[Tuple[str, ...]] = (
    # Directions
    "HYDROPHOBIC_DIRECTION_UNKNOWN",
    "HYDROPHOBIC_DIRECTION_LIGAND_RECEPTOR",
    "HYDROPHOBIC_DIRECTION_RECEPTOR_LIGAND",

    # Interaction types
    "HYDROPHOBIC_TYPE_UNKNOWN",
    "HYDROPHOBIC_TYPE_ALIPHATIC_ALIPHATIC",
    "HYDROPHOBIC_TYPE_ALIPHATIC_AROMATIC",
    "HYDROPHOBIC_TYPE_AROMATIC_ALIPHATIC",
    "HYDROPHOBIC_TYPE_AROMATIC_AROMATIC",
    "HYDROPHOBIC_TYPE_MIXED",

    # Classifications
    "HYDROPHOBIC_CLASS_UNKNOWN",
    "HYDROPHOBIC_CLASS_MARGINAL",
    "HYDROPHOBIC_CLASS_WEAK",
    "HYDROPHOBIC_CLASS_MODERATE",
    "HYDROPHOBIC_CLASS_STRONG",
    "HYDROPHOBIC_CLASS_VERY_STRONG",

    # Atom types
    "HYDROPHOBIC_ATOM_TYPE_UNKNOWN",
    "HYDROPHOBIC_ATOM_TYPE_NON_HYDROPHOBIC",
    "HYDROPHOBIC_ATOM_TYPE_ALIPHATIC",
    "HYDROPHOBIC_ATOM_TYPE_AROMATIC",
    "HYDROPHOBIC_ATOM_TYPE_MIXED",

    # Roles
    "HYDROPHOBIC_ROLE_UNKNOWN",
    "HYDROPHOBIC_ROLE_RECEPTOR",
    "HYDROPHOBIC_ROLE_LIGAND",

    # Detection methods
    "HYDROPHOBIC_METHOD_UNKNOWN",
    "HYDROPHOBIC_METHOD_ATOMIC",
    "HYDROPHOBIC_METHOD_GROUPED",
    "HYDROPHOBIC_METHOD_INFERRED",

    # Elements
    "HYDROGEN_ELEMENT",
    "CARBON_ELEMENT",
    "NITROGEN_ELEMENT",
    "OXYGEN_ELEMENT",
    "SULFUR_ELEMENT",
    "PHOSPHORUS_ELEMENT",
    "HALOGEN_ELEMENTS",
    "POLAR_HETEROATOM_ELEMENTS",
    "PRIMARY_HYDROPHOBIC_ELEMENTS",
    "CONDITIONALLY_HYDROPHOBIC_ELEMENTS",
    "SUPPORTED_HYDROPHOBIC_ELEMENTS",

    # Atomic numbers
    "HYDROGEN_ATOMIC_NUMBER",
    "CARBON_ATOMIC_NUMBER",
    "NITROGEN_ATOMIC_NUMBER",
    "OXYGEN_ATOMIC_NUMBER",
    "SULFUR_ATOMIC_NUMBER",
    "PHOSPHORUS_ATOMIC_NUMBER",
    "PRIMARY_HYDROPHOBIC_ATOMIC_NUMBERS",
    "CONDITIONALLY_HYDROPHOBIC_ATOMIC_NUMBERS",
    "SUPPORTED_HYDROPHOBIC_ATOMIC_NUMBERS",

    # Residue classes
    "STANDARD_AMINO_ACID_NAMES",
    "DEFAULT_HYDROPHOBIC_RESIDUE_NAMES",
    "STRONGLY_HYDROPHOBIC_RESIDUE_NAMES",
    "MODERATELY_HYDROPHOBIC_RESIDUE_NAMES",
    "AROMATIC_RESIDUE_NAMES",
    "ALIPHATIC_RESIDUE_NAMES",
    "HISTIDINE_RESIDUE_NAMES",
    "CYSTEINE_RESIDUE_NAMES",
    "WATER_RESIDUE_NAMES",
    "NUCLEIC_ACID_RESIDUE_NAMES",

    # Protein atom-name definitions
    "PROTEIN_BACKBONE_ATOM_NAMES",
    "PROTEIN_BACKBONE_CARBON_NAMES",
    "PROTEIN_ALIPHATIC_CARBON_NAMES",
    "HYDROPHOBIC_PROTEIN_ATOMS_BY_RESIDUE",
    "AROMATIC_PROTEIN_ATOMS_BY_RESIDUE",
    "POLAR_PROTEIN_ATOM_NAMES",
    "CHARGED_PROTEIN_ATOM_NAMES",

    # Geometric defaults
    "DEFAULT_MINIMUM_HYDROPHOBIC_DISTANCE",
    "DEFAULT_OPTIMAL_HYDROPHOBIC_DISTANCE",
    "DEFAULT_MAXIMUM_HYDROPHOBIC_DISTANCE",
    "DEFAULT_HYDROPHOBIC_DISTANCE_TOLERANCE",
    "DEFAULT_GROUPING_DISTANCE",
    "DEFAULT_LOCAL_ENVIRONMENT_RADIUS",
    "DEFAULT_POLAR_NEIGHBOR_RADIUS",
    "DEFAULT_MAXIMUM_ABSOLUTE_PARTIAL_CHARGE",
    "DEFAULT_MAXIMUM_ABSOLUTE_FORMAL_CHARGE",
    "DEFAULT_MINIMUM_LOCAL_CONTACT_COUNT",
    "DEFAULT_MINIMUM_HOTSPOT_CONTACT_COUNT",
    "DEFAULT_MAXIMUM_POLAR_NEIGHBORS",

    # Classification thresholds
    "HYDROPHOBIC_VERY_STRONG_MAX_DISTANCE",
    "HYDROPHOBIC_STRONG_MAX_DISTANCE",
    "HYDROPHOBIC_MODERATE_MAX_DISTANCE",
    "HYDROPHOBIC_WEAK_MAX_DISTANCE",
    "HYDROPHOBIC_MARGINAL_MAX_DISTANCE",

    # Scores
    "HYDROPHOBIC_SCORE_WEIGHTS",
    "HYDROPHOBIC_ATOM_TYPE_SCORE_MODIFIERS",
    "HYDROPHOBIC_INTERACTION_TYPE_SCORE_MODIFIERS",
    "HYDROPHOBIC_CLASSIFICATION_RANKS",
    "HYDROPHOBIC_CLASSIFICATION_BASE_SCORES",

    # Processing defaults
    "DEFAULT_MAXIMUM_PAIR_ELEMENTS",
    "DEFAULT_HYDROPHOBIC_BLOCK_SIZE",
    "DEFAULT_MAXIMUM_HYDROPHOBIC_INTERACTIONS",

    # Configuration functions
    "get_default_minimum_hydrophobic_distance",
    "get_default_optimal_hydrophobic_distance",
    "get_default_maximum_hydrophobic_distance",
    "get_default_hydrophobic_distance_tolerance",
    "get_default_grouping_distance",
    "get_default_local_environment_radius",
    "get_default_polar_neighbor_radius",
    "get_default_maximum_absolute_partial_charge",
    "get_default_maximum_polar_neighbors",
    "get_default_hydrophobic_residue_names",
    "get_default_hydrophobic_block_size",
    "get_default_maximum_pair_elements",
    "get_default_maximum_hydrophobic_interactions",

    # Validators
    "validate_hydrophobic_direction",
    "validate_hydrophobic_interaction_type",
    "validate_hydrophobic_classification",
    "validate_hydrophobic_atom_type",
    "validate_hydrophobic_atom_role",
    "validate_hydrophobic_detection_method",
    "validate_hydrophobic_distance_limits",
    "validate_hydrophobic_score",
)

for public_name in _SECTION_2_PUBLIC_NAMES:
    if public_name not in __all__:
        __all__.append(public_name)


# =============================================================================
# End of Section 2
# =============================================================================

# =============================================================================
# Section 3 — Result dataclasses
# =============================================================================


# -----------------------------------------------------------------------------
# Local normalization and serialization helpers
# -----------------------------------------------------------------------------

def _freeze_metadata(
    metadata: Optional[Mapping[str, Any]],
) -> Mapping[str, Any]:
    """
    Create an immutable shallow copy of a metadata mapping.

    Parameters
    ----------
    metadata
        Mapping to normalize. ``None`` produces an empty mapping.

    Returns
    -------
    mapping
        Read-only metadata mapping.
    """

    if metadata is None:
        return _EMPTY_METADATA

    if not isinstance(metadata, Mapping):
        raise TypeError(
            "metadata must be a mapping or None."
        )

    if not metadata:
        return _EMPTY_METADATA

    return MappingProxyType(dict(metadata))


def _freeze_string_mapping(
    values: Optional[Mapping[Any, Any]],
) -> Mapping[str, Any]:
    """
    Normalize mapping keys to non-empty strings and freeze the result.
    """

    if values is None:
        return _EMPTY_METADATA

    if not isinstance(values, Mapping):
        raise TypeError(
            "Expected a mapping or None."
        )

    normalized: Dict[str, Any] = {}

    for key, value in values.items():
        normalized_key = str(key).strip()

        if not normalized_key:
            raise ValueError(
                "Mapping keys cannot be empty."
            )

        normalized[normalized_key] = value

    if not normalized:
        return _EMPTY_METADATA

    return MappingProxyType(normalized)


def _finite_float(
    value: Number,
    *,
    name: str,
) -> np.float64:
    """
    Convert a numeric value to a finite ``numpy.float64``.
    """

    if isinstance(value, bool):
        raise TypeError(
            f"{name} must be numeric, not boolean."
        )

    try:
        numeric_value = np.float64(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(
            f"{name} must be numeric."
        ) from exc

    if not np.isfinite(numeric_value):
        raise ValueError(
            f"{name} must be finite."
        )

    return numeric_value


def _nonnegative_float(
    value: Number,
    *,
    name: str,
) -> np.float64:
    """
    Convert a numeric value to a finite non-negative float.
    """

    numeric_value = _finite_float(
        value,
        name=name,
    )

    if numeric_value < 0.0:
        raise ValueError(
            f"{name} cannot be negative."
        )

    return numeric_value


def _positive_float(
    value: Number,
    *,
    name: str,
) -> np.float64:
    """
    Convert a numeric value to a finite strictly positive float.
    """

    numeric_value = _finite_float(
        value,
        name=name,
    )

    if numeric_value <= 0.0:
        raise ValueError(
            f"{name} must be greater than zero."
        )

    return numeric_value


def _optional_finite_float(
    value: Optional[Number],
    *,
    name: str,
) -> Optional[np.float64]:
    """
    Normalize an optional finite float.
    """

    if value is None:
        return None

    return _finite_float(
        value,
        name=name,
    )


def _optional_nonnegative_float(
    value: Optional[Number],
    *,
    name: str,
) -> Optional[np.float64]:
    """
    Normalize an optional finite non-negative float.
    """

    if value is None:
        return None

    return _nonnegative_float(
        value,
        name=name,
    )


def _nonnegative_integer(
    value: Any,
    *,
    name: str,
) -> int:
    """
    Normalize a non-negative integer.
    """

    if isinstance(value, bool):
        raise TypeError(
            f"{name} must be an integer, not boolean."
        )

    try:
        integer_value = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(
            f"{name} must be an integer."
        ) from exc

    if integer_value < 0:
        raise ValueError(
            f"{name} cannot be negative."
        )

    return integer_value


def _optional_nonnegative_integer(
    value: Optional[Any],
    *,
    name: str,
) -> Optional[int]:
    """
    Normalize an optional non-negative integer.
    """

    if value is None:
        return None

    return _nonnegative_integer(
        value,
        name=name,
    )


def _normalize_optional_string(
    value: Optional[Any],
) -> Optional[str]:
    """
    Normalize an optional string-like value.
    """

    if value is None:
        return None

    normalized = str(value).strip()

    return normalized or None


def _normalize_required_string(
    value: Any,
    *,
    name: str,
) -> str:
    """
    Normalize a required non-empty string.
    """

    normalized = str(value).strip()

    if not normalized:
        raise ValueError(
            f"{name} cannot be empty."
        )

    return normalized


def _normalize_element_symbol(
    value: Any,
) -> str:
    """
    Normalize an atomic element symbol.

    Examples
    --------
    ``"cl"`` becomes ``"CL"`` and ``" C "`` becomes ``"C"``.
    """

    normalized = str(value).strip().upper()

    return normalized or "UNKNOWN"


def _normalize_residue_key(
    key: Optional[Any],
) -> Optional[ResidueContactKey]:
    """
    Normalize a residue key without assuming its concrete representation.

    ``contacts.py`` is the canonical provider of residue keys. This helper
    only guarantees tuple storage and preserves compatibility with current
    and future key layouts.
    """

    if key is None:
        return None

    if isinstance(key, tuple):
        return key  # type: ignore[return-value]

    if isinstance(key, list):
        return tuple(key)  # type: ignore[return-value]

    try:
        return tuple(key)  # type: ignore[arg-type, return-value]
    except TypeError:
        return (str(key),)  # type: ignore[return-value]


def _safe_atom_identifier(
    atom: Optional[AtomLike],
    *,
    fallback: Optional[str] = None,
) -> Optional[str]:
    """
    Return a serializable atom identifier without propagating API errors.
    """

    if atom is None:
        return fallback

    try:
        identifier = get_atom_identifier(atom)
    except Exception:
        identifier = None

    normalized = _normalize_optional_string(identifier)

    if normalized is not None:
        return normalized

    try:
        atom_name = get_atom_name(atom)
    except Exception:
        atom_name = None

    normalized_name = _normalize_optional_string(atom_name)

    if normalized_name is not None:
        return normalized_name

    return fallback


def _safe_atom_index(
    atom: Optional[AtomLike],
) -> Optional[int]:
    """
    Return an atom index when one can be determined safely.
    """

    if atom is None:
        return None

    try:
        value = get_atom_index(atom)
    except Exception:
        return None

    try:
        return _optional_nonnegative_integer(
            value,
            name="atom index",
        )
    except (TypeError, ValueError):
        return None


def _safe_atom_element(
    atom: Optional[AtomLike],
) -> str:
    """
    Return a normalized atom element symbol.
    """

    if atom is None:
        return "UNKNOWN"

    try:
        element = get_atom_element(atom)
    except Exception:
        element = None

    if element is None:
        return "UNKNOWN"

    return _normalize_element_symbol(element)


def _safe_atom_residue(
    atom: Optional[AtomLike],
) -> Optional[ResidueLike]:
    """
    Return the parent residue of an atom, if available.
    """

    if atom is None:
        return None

    try:
        return get_atom_residue(atom)
    except Exception:
        return None


def _safe_residue_key_from_atom(
    atom: Optional[AtomLike],
) -> Optional[ResidueContactKey]:
    """
    Return the normalized residue key associated with an atom.
    """

    if atom is None:
        return None

    try:
        key = get_residue_contact_key(atom)
    except Exception:
        key = None

    return _normalize_residue_key(key)


def _safe_residue_identifier(
    residue: Optional[ResidueLike],
    key: Optional[ResidueContactKey] = None,
) -> Optional[str]:
    """
    Return a compact serializable residue identifier.
    """

    if key is not None:
        return ":".join(
            str(part)
            for part in key
            if part is not None and str(part).strip()
        ) or None

    if residue is None:
        return None

    residue_name = _normalize_optional_string(
        getattr(residue, "name", None)
    )

    residue_number = _normalize_optional_string(
        getattr(residue, "number", None)
    )

    chain_id = _normalize_optional_string(
        getattr(residue, "chain_id", None)
    )

    insertion_code = _normalize_optional_string(
        getattr(residue, "insertion_code", None)
    )

    components = tuple(
        component
        for component in (
            chain_id,
            residue_name,
            residue_number,
            insertion_code,
        )
        if component is not None
    )

    return ":".join(components) or None


def _atom_reference_dict(
    atom: AtomLike,
    *,
    identifier: Optional[str],
    index: Optional[int],
    element: str,
    residue_key: Optional[ResidueContactKey],
    include_atom: bool = False,
) -> Dict[str, Any]:
    """
    Create a serializable atom-reference dictionary.
    """

    result: Dict[str, Any] = {
        "identifier": identifier,
        "index": index,
        "element": element,
        "residue_key": residue_key,
    }

    if include_atom:
        result["atom"] = atom

    return result


# -----------------------------------------------------------------------------
# Hydrophobic atom descriptor
# -----------------------------------------------------------------------------

@dataclass(
    frozen=True,
    slots=True,
)
class HydrophobicAtom:
    """
    Chemical descriptor for one atom considered during hydrophobic analysis.

    Parameters
    ----------
    atom
        Underlying ChimeraX-compatible atom object.
    role
        Whether the atom belongs to the receptor, ligand or an unknown side.
    atom_type
        Hydrophobic chemical type assigned to the atom.
    is_hydrophobic
        Whether the atom passed hydrophobic chemical-perception rules.
    is_aromatic
        Whether the atom is part of an aromatic environment.
    is_aliphatic
        Whether the atom is part of an aliphatic environment.
    element
        Normalized atomic element symbol. When omitted, it is inferred.
    atomic_number
        Atomic number, when available.
    atom_index
        Index in the original analyzed atom collection.
    identifier
        Stable serializable atom identifier.
    residue
        Parent residue.
    residue_key
        Normalized residue key from :mod:`contacts`.
    formal_charge
        Formal atomic charge, when available.
    partial_charge
        Partial atomic charge, when available.
    polar_neighbor_count
        Number of directly bonded or locally associated polar atoms.
    heavy_neighbor_count
        Number of heavy-atom neighbors.
    metadata
        Additional chemical-perception metadata.
    """

    atom: AtomLike

    role: HydrophobicAtomRole = HYDROPHOBIC_ROLE_UNKNOWN
    atom_type: HydrophobicAtomType = HYDROPHOBIC_ATOM_TYPE_UNKNOWN

    is_hydrophobic: bool = False
    is_aromatic: bool = False
    is_aliphatic: bool = False

    element: Optional[str] = None
    atomic_number: Optional[int] = None
    atom_index: Optional[int] = None
    identifier: Optional[str] = None

    residue: Optional[ResidueLike] = None
    residue_key: Optional[ResidueContactKey] = None

    formal_charge: Optional[np.float64] = None
    partial_charge: Optional[np.float64] = None

    polar_neighbor_count: int = 0
    heavy_neighbor_count: int = 0

    metadata: Mapping[str, Any] = field(
        default_factory=lambda: _EMPTY_METADATA,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        """Validate and normalize the atom descriptor."""

        if self.atom is None:
            raise ValueError(
                "HydrophobicAtom.atom cannot be None."
            )

        normalized_role = validate_hydrophobic_atom_role(
            self.role
        )

        normalized_atom_type = validate_hydrophobic_atom_type(
            self.atom_type
        )

        element = (
            _safe_atom_element(self.atom)
            if self.element is None
            else _normalize_element_symbol(self.element)
        )

        atomic_number = self.atomic_number

        if atomic_number is None:
            try:
                atomic_number = get_atom_atomic_number(self.atom)
            except Exception:
                atomic_number = None

        normalized_atomic_number = (
            None
            if atomic_number is None
            else _nonnegative_integer(
                atomic_number,
                name="atomic number",
            )
        )

        atom_index = (
            _safe_atom_index(self.atom)
            if self.atom_index is None
            else _optional_nonnegative_integer(
                self.atom_index,
                name="atom index",
            )
        )

        identifier = (
            _safe_atom_identifier(
                self.atom,
                fallback=(
                    None
                    if atom_index is None
                    else f"atom-{atom_index}"
                ),
            )
            if self.identifier is None
            else _normalize_required_string(
                self.identifier,
                name="atom identifier",
            )
        )

        residue = (
            _safe_atom_residue(self.atom)
            if self.residue is None
            else self.residue
        )

        residue_key = _normalize_residue_key(
            self.residue_key
        )

        if residue_key is None:
            residue_key = _safe_residue_key_from_atom(
                self.atom
            )

        formal_charge = _optional_finite_float(
            self.formal_charge,
            name="formal charge",
        )

        partial_charge = _optional_finite_float(
            self.partial_charge,
            name="partial charge",
        )

        polar_neighbor_count = _nonnegative_integer(
            self.polar_neighbor_count,
            name="polar neighbor count",
        )

        heavy_neighbor_count = _nonnegative_integer(
            self.heavy_neighbor_count,
            name="heavy neighbor count",
        )

        if (
            normalized_atom_type
            == HYDROPHOBIC_ATOM_TYPE_NON_HYDROPHOBIC
            and self.is_hydrophobic
        ):
            raise ValueError(
                "A non-hydrophobic atom type cannot have "
                "is_hydrophobic=True."
            )

        if self.is_aromatic and self.is_aliphatic:
            if (
                normalized_atom_type
                not in {
                    HYDROPHOBIC_ATOM_TYPE_MIXED,
                    HYDROPHOBIC_ATOM_TYPE_UNKNOWN,
                }
            ):
                normalized_atom_type = (
                    HYDROPHOBIC_ATOM_TYPE_MIXED
                )

        elif self.is_aromatic:
            if normalized_atom_type == HYDROPHOBIC_ATOM_TYPE_UNKNOWN:
                normalized_atom_type = (
                    HYDROPHOBIC_ATOM_TYPE_AROMATIC
                )

        elif self.is_aliphatic:
            if normalized_atom_type == HYDROPHOBIC_ATOM_TYPE_UNKNOWN:
                normalized_atom_type = (
                    HYDROPHOBIC_ATOM_TYPE_ALIPHATIC
                )

        object.__setattr__(
            self,
            "role",
            normalized_role,
        )

        object.__setattr__(
            self,
            "atom_type",
            normalized_atom_type,
        )

        object.__setattr__(
            self,
            "element",
            element,
        )

        object.__setattr__(
            self,
            "atomic_number",
            normalized_atomic_number,
        )

        object.__setattr__(
            self,
            "atom_index",
            atom_index,
        )

        object.__setattr__(
            self,
            "identifier",
            identifier,
        )

        object.__setattr__(
            self,
            "residue",
            residue,
        )

        object.__setattr__(
            self,
            "residue_key",
            residue_key,
        )

        object.__setattr__(
            self,
            "formal_charge",
            formal_charge,
        )

        object.__setattr__(
            self,
            "partial_charge",
            partial_charge,
        )

        object.__setattr__(
            self,
            "polar_neighbor_count",
            polar_neighbor_count,
        )

        object.__setattr__(
            self,
            "heavy_neighbor_count",
            heavy_neighbor_count,
        )

        object.__setattr__(
            self,
            "metadata",
            _freeze_metadata(self.metadata),
        )

    @property
    def residue_identifier(self) -> Optional[str]:
        """Return a compact parent-residue identifier."""

        return _safe_residue_identifier(
            self.residue,
            self.residue_key,
        )

    @property
    def has_charge_information(self) -> bool:
        """Return whether formal or partial charge data are present."""

        return (
            self.formal_charge is not None
            or self.partial_charge is not None
        )

    @property
    def absolute_partial_charge(self) -> Optional[np.float64]:
        """Return the absolute partial charge, when available."""

        if self.partial_charge is None:
            return None

        return np.float64(abs(self.partial_charge))

    @property
    def is_conditionally_hydrophobic(self) -> bool:
        """
        Return whether the element requires contextual interpretation.
        """

        return self.element in CONDITIONALLY_HYDROPHOBIC_ELEMENTS

    def to_dict(
        self,
        *,
        include_atom: bool = False,
        include_residue: bool = False,
    ) -> Dict[str, Any]:
        """
        Serialize the atom descriptor.

        Parameters
        ----------
        include_atom
            Include the raw atom object.
        include_residue
            Include the raw parent-residue object.
        """

        result: Dict[str, Any] = {
            "identifier": self.identifier,
            "atom_index": self.atom_index,
            "element": self.element,
            "atomic_number": self.atomic_number,
            "role": self.role,
            "atom_type": self.atom_type,
            "is_hydrophobic": self.is_hydrophobic,
            "is_aromatic": self.is_aromatic,
            "is_aliphatic": self.is_aliphatic,
            "formal_charge": (
                None
                if self.formal_charge is None
                else float(self.formal_charge)
            ),
            "partial_charge": (
                None
                if self.partial_charge is None
                else float(self.partial_charge)
            ),
            "polar_neighbor_count": self.polar_neighbor_count,
            "heavy_neighbor_count": self.heavy_neighbor_count,
            "residue_key": self.residue_key,
            "residue_identifier": self.residue_identifier,
            "metadata": dict(self.metadata),
        }

        if include_atom:
            result["atom"] = self.atom

        if include_residue:
            result["residue"] = self.residue

        return result


# -----------------------------------------------------------------------------
# Individual hydrophobic interaction
# -----------------------------------------------------------------------------

@dataclass(
    frozen=True,
    slots=True,
)
class HydrophobicInteraction:
    """
    Representation of one receptor-ligand hydrophobic interaction.

    Parameters
    ----------
    receptor_atom
        Receptor atom participating in the interaction.
    ligand_atom
        Ligand atom participating in the interaction.
    distance
        Receptor-ligand atomic distance in angstroms.
    receptor_descriptor
        Optional precomputed receptor atom descriptor.
    ligand_descriptor
        Optional precomputed ligand atom descriptor.
    receptor_residue
        Parent receptor residue.
    receptor_residue_key
        Normalized receptor residue key.
    interaction_type
        Chemical combination represented by the pair.
    classification
        Geometric strength class.
    strength
        Continuous normalized geometric strength in ``[0, 1]``.
    score
        Final normalized interaction score in ``[0, 1]``.
    detection_method
        Method by which the interaction was detected.
    direction
        Directional receptor-ligand representation.
    local_contact_count
        Number of related local atom-pair contacts.
    polar_penalty
        Normalized penalty due to local polar character.
    receptor_atom_index
        Receptor atom index in its analyzed collection.
    ligand_atom_index
        Ligand atom index in its analyzed collection.
    receptor_atom_identifier
        Stable serializable receptor atom identifier.
    ligand_atom_identifier
        Stable serializable ligand atom identifier.
    interaction_identifier
        Stable serializable identifier for the complete interaction.
    metadata
        Additional interaction metadata.
    """

    receptor_atom: AtomLike
    ligand_atom: AtomLike
    distance: np.float64

    receptor_descriptor: Optional[HydrophobicAtom] = None
    ligand_descriptor: Optional[HydrophobicAtom] = None

    receptor_residue: Optional[ResidueLike] = None
    receptor_residue_key: Optional[ResidueContactKey] = None

    interaction_type: HydrophobicInteractionType = (
        HYDROPHOBIC_TYPE_UNKNOWN
    )

    classification: HydrophobicClassification = (
        HYDROPHOBIC_CLASS_UNKNOWN
    )

    strength: np.float64 = np.float64(0.0)
    score: np.float64 = np.float64(0.0)

    detection_method: HydrophobicDetectionMethod = (
        HYDROPHOBIC_METHOD_UNKNOWN
    )

    direction: HydrophobicInteractionDirection = (
        HYDROPHOBIC_DIRECTION_LIGAND_RECEPTOR
    )

    local_contact_count: int = 1
    polar_penalty: np.float64 = np.float64(0.0)

    receptor_atom_index: Optional[int] = None
    ligand_atom_index: Optional[int] = None

    receptor_atom_identifier: Optional[str] = None
    ligand_atom_identifier: Optional[str] = None
    interaction_identifier: Optional[str] = None

    metadata: Mapping[str, Any] = field(
        default_factory=lambda: _EMPTY_METADATA,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        """Validate and normalize the interaction."""

        if self.receptor_atom is None:
            raise ValueError(
                "receptor_atom cannot be None."
            )

        if self.ligand_atom is None:
            raise ValueError(
                "ligand_atom cannot be None."
            )

        if self.receptor_atom is self.ligand_atom:
            raise ValueError(
                "Receptor and ligand atoms must be different objects."
            )

        normalized_distance = _positive_float(
            self.distance,
            name="hydrophobic interaction distance",
        )

        normalized_type = validate_hydrophobic_interaction_type(
            self.interaction_type
        )

        normalized_classification = (
            validate_hydrophobic_classification(
                self.classification
            )
        )

        normalized_method = (
            validate_hydrophobic_detection_method(
                self.detection_method
            )
        )

        normalized_direction = validate_hydrophobic_direction(
            self.direction
        )

        normalized_strength = validate_hydrophobic_score(
            self.strength
        )

        normalized_score = validate_hydrophobic_score(
            self.score
        )

        normalized_polar_penalty = validate_hydrophobic_score(
            self.polar_penalty
        )

        normalized_local_contact_count = _nonnegative_integer(
            self.local_contact_count,
            name="local contact count",
        )

        if normalized_local_contact_count == 0:
            raise ValueError(
                "local_contact_count must be at least one."
            )

        receptor_descriptor = self.receptor_descriptor

        if receptor_descriptor is not None:
            if not isinstance(
                receptor_descriptor,
                HydrophobicAtom,
            ):
                raise TypeError(
                    "receptor_descriptor must be a "
                    "HydrophobicAtom instance or None."
                )

            if receptor_descriptor.atom is not self.receptor_atom:
                raise ValueError(
                    "receptor_descriptor.atom must refer to "
                    "receptor_atom."
                )

        ligand_descriptor = self.ligand_descriptor

        if ligand_descriptor is not None:
            if not isinstance(
                ligand_descriptor,
                HydrophobicAtom,
            ):
                raise TypeError(
                    "ligand_descriptor must be a "
                    "HydrophobicAtom instance or None."
                )

            if ligand_descriptor.atom is not self.ligand_atom:
                raise ValueError(
                    "ligand_descriptor.atom must refer to ligand_atom."
                )

        receptor_residue = self.receptor_residue

        if receptor_residue is None:
            receptor_residue = (
                receptor_descriptor.residue
                if receptor_descriptor is not None
                else _safe_atom_residue(self.receptor_atom)
            )

        receptor_residue_key = _normalize_residue_key(
            self.receptor_residue_key
        )

        if receptor_residue_key is None:
            receptor_residue_key = (
                receptor_descriptor.residue_key
                if receptor_descriptor is not None
                else _safe_residue_key_from_atom(
                    self.receptor_atom
                )
            )

        receptor_atom_index = (
            receptor_descriptor.atom_index
            if (
                self.receptor_atom_index is None
                and receptor_descriptor is not None
            )
            else (
                _safe_atom_index(self.receptor_atom)
                if self.receptor_atom_index is None
                else _optional_nonnegative_integer(
                    self.receptor_atom_index,
                    name="receptor atom index",
                )
            )
        )

        ligand_atom_index = (
            ligand_descriptor.atom_index
            if (
                self.ligand_atom_index is None
                and ligand_descriptor is not None
            )
            else (
                _safe_atom_index(self.ligand_atom)
                if self.ligand_atom_index is None
                else _optional_nonnegative_integer(
                    self.ligand_atom_index,
                    name="ligand atom index",
                )
            )
        )

        receptor_identifier = (
            receptor_descriptor.identifier
            if (
                self.receptor_atom_identifier is None
                and receptor_descriptor is not None
            )
            else (
                _safe_atom_identifier(
                    self.receptor_atom,
                    fallback=(
                        None
                        if receptor_atom_index is None
                        else f"receptor-{receptor_atom_index}"
                    ),
                )
                if self.receptor_atom_identifier is None
                else _normalize_required_string(
                    self.receptor_atom_identifier,
                    name="receptor atom identifier",
                )
            )
        )

        ligand_identifier = (
            ligand_descriptor.identifier
            if (
                self.ligand_atom_identifier is None
                and ligand_descriptor is not None
            )
            else (
                _safe_atom_identifier(
                    self.ligand_atom,
                    fallback=(
                        None
                        if ligand_atom_index is None
                        else f"ligand-{ligand_atom_index}"
                    ),
                )
                if self.ligand_atom_identifier is None
                else _normalize_required_string(
                    self.ligand_atom_identifier,
                    name="ligand atom identifier",
                )
            )
        )

        residue_identifier = _safe_residue_identifier(
            receptor_residue,
            receptor_residue_key,
        )

        interaction_identifier = (
            _normalize_optional_string(
                self.interaction_identifier
            )
        )

        if interaction_identifier is None:
            identifier_components = (
                receptor_identifier or "receptor-atom",
                ligand_identifier or "ligand-atom",
                residue_identifier or "unknown-residue",
            )

            interaction_identifier = "|".join(
                identifier_components
            )

        object.__setattr__(
            self,
            "distance",
            normalized_distance,
        )

        object.__setattr__(
            self,
            "interaction_type",
            normalized_type,
        )

        object.__setattr__(
            self,
            "classification",
            normalized_classification,
        )

        object.__setattr__(
            self,
            "detection_method",
            normalized_method,
        )

        object.__setattr__(
            self,
            "direction",
            normalized_direction,
        )

        object.__setattr__(
            self,
            "strength",
            normalized_strength,
        )

        object.__setattr__(
            self,
            "score",
            normalized_score,
        )

        object.__setattr__(
            self,
            "polar_penalty",
            normalized_polar_penalty,
        )

        object.__setattr__(
            self,
            "local_contact_count",
            normalized_local_contact_count,
        )

        object.__setattr__(
            self,
            "receptor_residue",
            receptor_residue,
        )

        object.__setattr__(
            self,
            "receptor_residue_key",
            receptor_residue_key,
        )

        object.__setattr__(
            self,
            "receptor_atom_index",
            receptor_atom_index,
        )

        object.__setattr__(
            self,
            "ligand_atom_index",
            ligand_atom_index,
        )

        object.__setattr__(
            self,
            "receptor_atom_identifier",
            receptor_identifier,
        )

        object.__setattr__(
            self,
            "ligand_atom_identifier",
            ligand_identifier,
        )

        object.__setattr__(
            self,
            "interaction_identifier",
            interaction_identifier,
        )

        object.__setattr__(
            self,
            "metadata",
            _freeze_metadata(self.metadata),
        )

    @property
    def receptor_element(self) -> str:
        """Return the receptor atom element."""

        if self.receptor_descriptor is not None:
            return self.receptor_descriptor.element or "UNKNOWN"

        return _safe_atom_element(self.receptor_atom)

    @property
    def ligand_element(self) -> str:
        """Return the ligand atom element."""

        if self.ligand_descriptor is not None:
            return self.ligand_descriptor.element or "UNKNOWN"

        return _safe_atom_element(self.ligand_atom)

    @property
    def receptor_atom_type(self) -> HydrophobicAtomType:
        """Return the receptor hydrophobic atom type."""

        if self.receptor_descriptor is None:
            return HYDROPHOBIC_ATOM_TYPE_UNKNOWN

        return self.receptor_descriptor.atom_type

    @property
    def ligand_atom_type(self) -> HydrophobicAtomType:
        """Return the ligand hydrophobic atom type."""

        if self.ligand_descriptor is None:
            return HYDROPHOBIC_ATOM_TYPE_UNKNOWN

        return self.ligand_descriptor.atom_type

    @property
    def receptor_residue_identifier(self) -> Optional[str]:
        """Return a serializable receptor-residue identifier."""

        return _safe_residue_identifier(
            self.receptor_residue,
            self.receptor_residue_key,
        )

    @property
    def is_aromatic_contact(self) -> bool:
        """Return whether at least one side is aromatic."""

        return self.interaction_type in {
            HYDROPHOBIC_TYPE_ALIPHATIC_AROMATIC,
            HYDROPHOBIC_TYPE_AROMATIC_ALIPHATIC,
            HYDROPHOBIC_TYPE_AROMATIC_AROMATIC,
        }

    @property
    def is_purely_aliphatic(self) -> bool:
        """Return whether both atoms have aliphatic character."""

        return (
            self.interaction_type
            == HYDROPHOBIC_TYPE_ALIPHATIC_ALIPHATIC
        )

    @property
    def classification_rank(self) -> int:
        """Return the ordinal rank of the classification."""

        return HYDROPHOBIC_CLASSIFICATION_RANKS[
            self.classification
        ]

    @property
    def is_strong(self) -> bool:
        """Return whether the interaction is strong or very strong."""

        return self.classification in {
            HYDROPHOBIC_CLASS_STRONG,
            HYDROPHOBIC_CLASS_VERY_STRONG,
        }

    def to_dict(
        self,
        *,
        include_atoms: bool = False,
        include_residue: bool = False,
        include_descriptors: bool = True,
    ) -> Dict[str, Any]:
        """
        Serialize the hydrophobic interaction.
        """

        result: Dict[str, Any] = {
            "interaction_identifier": self.interaction_identifier,
            "receptor_atom_identifier": (
                self.receptor_atom_identifier
            ),
            "ligand_atom_identifier": (
                self.ligand_atom_identifier
            ),
            "receptor_atom_index": self.receptor_atom_index,
            "ligand_atom_index": self.ligand_atom_index,
            "receptor_element": self.receptor_element,
            "ligand_element": self.ligand_element,
            "receptor_atom_type": self.receptor_atom_type,
            "ligand_atom_type": self.ligand_atom_type,
            "receptor_residue_key": self.receptor_residue_key,
            "receptor_residue_identifier": (
                self.receptor_residue_identifier
            ),
            "distance": float(self.distance),
            "interaction_type": self.interaction_type,
            "classification": self.classification,
            "classification_rank": self.classification_rank,
            "strength": float(self.strength),
            "score": float(self.score),
            "polar_penalty": float(self.polar_penalty),
            "local_contact_count": self.local_contact_count,
            "detection_method": self.detection_method,
            "direction": self.direction,
            "is_aromatic_contact": self.is_aromatic_contact,
            "is_purely_aliphatic": self.is_purely_aliphatic,
            "is_strong": self.is_strong,
            "metadata": dict(self.metadata),
        }

        if include_descriptors:
            result["receptor_descriptor"] = (
                None
                if self.receptor_descriptor is None
                else self.receptor_descriptor.to_dict(
                    include_atom=include_atoms,
                    include_residue=include_residue,
                )
            )

            result["ligand_descriptor"] = (
                None
                if self.ligand_descriptor is None
                else self.ligand_descriptor.to_dict(
                    include_atom=include_atoms,
                    include_residue=include_residue,
                )
            )

        if include_atoms:
            result["receptor_atom"] = self.receptor_atom
            result["ligand_atom"] = self.ligand_atom

        if include_residue:
            result["receptor_residue"] = self.receptor_residue

        return result


# -----------------------------------------------------------------------------
# Residue-grouped hydrophobic interactions
# -----------------------------------------------------------------------------

@dataclass(
    frozen=True,
    slots=True,
)
class HydrophobicResidueGroup:
    """
    Hydrophobic interactions grouped by one receptor residue.

    Parameters
    ----------
    residue
        Receptor residue represented by this group.
    residue_key
        Normalized residue key.
    interactions
        Hydrophobic interactions assigned to the residue.
    residue_identifier
        Stable serializable residue identifier.
    group_score
        Optional precomputed residue-level score.
    metadata
        Additional residue-group metadata.
    """

    residue: Optional[ResidueLike]
    residue_key: Optional[ResidueContactKey]

    interactions: Sequence[HydrophobicInteraction] = field(
        default_factory=tuple
    )

    residue_identifier: Optional[str] = None
    group_score: Optional[np.float64] = None

    metadata: Mapping[str, Any] = field(
        default_factory=lambda: _EMPTY_METADATA,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        """Validate and normalize the residue group."""

        normalized_interactions = tuple(self.interactions)

        for index, interaction in enumerate(
            normalized_interactions
        ):
            if not isinstance(
                interaction,
                HydrophobicInteraction,
            ):
                raise TypeError(
                    "All interactions must be "
                    "HydrophobicInteraction instances. "
                    f"Invalid entry at index {index}."
                )

        normalized_key = _normalize_residue_key(
            self.residue_key
        )

        if normalized_key is None and normalized_interactions:
            normalized_key = (
                normalized_interactions[
                    0
                ].receptor_residue_key
            )

        residue = self.residue

        if residue is None and normalized_interactions:
            residue = (
                normalized_interactions[
                    0
                ].receptor_residue
            )

        for interaction in normalized_interactions:
            interaction_key = interaction.receptor_residue_key

            if (
                normalized_key is not None
                and interaction_key is not None
                and interaction_key != normalized_key
            ):
                raise ValueError(
                    "Every grouped interaction must belong to "
                    "the same receptor residue."
                )

        residue_identifier = (
            _normalize_optional_string(
                self.residue_identifier
            )
        )

        if residue_identifier is None:
            residue_identifier = _safe_residue_identifier(
                residue,
                normalized_key,
            )

        group_score = _optional_nonnegative_float(
            self.group_score,
            name="hydrophobic residue group score",
        )

        if group_score is None:
            group_score = np.float64(
                sum(
                    float(interaction.score)
                    for interaction in normalized_interactions
                )
            )

        object.__setattr__(
            self,
            "residue",
            residue,
        )

        object.__setattr__(
            self,
            "residue_key",
            normalized_key,
        )

        object.__setattr__(
            self,
            "interactions",
            normalized_interactions,
        )

        object.__setattr__(
            self,
            "residue_identifier",
            residue_identifier,
        )

        object.__setattr__(
            self,
            "group_score",
            group_score,
        )

        object.__setattr__(
            self,
            "metadata",
            _freeze_metadata(self.metadata),
        )

    def __len__(self) -> int:
        """Return the number of grouped interactions."""

        return len(self.interactions)

    def __iter__(self) -> Iterator[HydrophobicInteraction]:
        """Iterate over grouped interactions."""

        return iter(self.interactions)

    def __getitem__(
        self,
        index: int,
    ) -> HydrophobicInteraction:
        """Return one grouped interaction by index."""

        return self.interactions[index]

    @property
    def interaction_count(self) -> int:
        """Return the number of interactions."""

        return len(self.interactions)

    @property
    def unique_receptor_atom_count(self) -> int:
        """Return the number of unique receptor atoms."""

        return len(
            {
                (
                    interaction.receptor_atom_identifier
                    or id(interaction.receptor_atom)
                )
                for interaction in self.interactions
            }
        )

    @property
    def unique_ligand_atom_count(self) -> int:
        """Return the number of unique ligand atoms."""

        return len(
            {
                (
                    interaction.ligand_atom_identifier
                    or id(interaction.ligand_atom)
                )
                for interaction in self.interactions
            }
        )

    @property
    def minimum_distance(self) -> Optional[np.float64]:
        """Return the minimum grouped interaction distance."""

        if not self.interactions:
            return None

        return np.float64(
            min(
                interaction.distance
                for interaction in self.interactions
            )
        )

    @property
    def mean_distance(self) -> Optional[np.float64]:
        """Return the mean grouped interaction distance."""

        if not self.interactions:
            return None

        return np.float64(
            np.mean(
                [
                    interaction.distance
                    for interaction in self.interactions
                ]
            )
        )

    @property
    def maximum_distance(self) -> Optional[np.float64]:
        """Return the maximum grouped interaction distance."""

        if not self.interactions:
            return None

        return np.float64(
            max(
                interaction.distance
                for interaction in self.interactions
            )
        )

    @property
    def mean_score(self) -> Optional[np.float64]:
        """Return the mean interaction score."""

        if not self.interactions:
            return None

        return np.float64(
            np.mean(
                [
                    interaction.score
                    for interaction in self.interactions
                ]
            )
        )

    @property
    def maximum_score(self) -> Optional[np.float64]:
        """Return the maximum individual interaction score."""

        if not self.interactions:
            return None

        return np.float64(
            max(
                interaction.score
                for interaction in self.interactions
            )
        )

    @property
    def strong_count(self) -> int:
        """Return the number of strong or very strong interactions."""

        return sum(
            interaction.is_strong
            for interaction in self.interactions
        )

    @property
    def aromatic_contact_count(self) -> int:
        """Return the number of contacts involving aromatic atoms."""

        return sum(
            interaction.is_aromatic_contact
            for interaction in self.interactions
        )

    @property
    def classifications(self) -> FrozenSet[str]:
        """Return all geometric classifications represented."""

        return frozenset(
            interaction.classification
            for interaction in self.interactions
        )

    @property
    def interaction_types(self) -> FrozenSet[str]:
        """Return all interaction types represented."""

        return frozenset(
            interaction.interaction_type
            for interaction in self.interactions
        )

    @property
    def is_hotspot(self) -> bool:
        """Return whether the residue satisfies the default hotspot count."""

        return (
            self.interaction_count
            >= DEFAULT_MINIMUM_HOTSPOT_CONTACT_COUNT
        )

    def interactions_by_classification(
        self,
        classification: HydrophobicClassification,
    ) -> Tuple[HydrophobicInteraction, ...]:
        """
        Return interactions matching a geometric classification.
        """

        normalized = validate_hydrophobic_classification(
            classification
        )

        return tuple(
            interaction
            for interaction in self.interactions
            if interaction.classification == normalized
        )

    def interactions_by_type(
        self,
        interaction_type: HydrophobicInteractionType,
    ) -> Tuple[HydrophobicInteraction, ...]:
        """
        Return interactions matching a chemical type.
        """

        normalized = validate_hydrophobic_interaction_type(
            interaction_type
        )

        return tuple(
            interaction
            for interaction in self.interactions
            if interaction.interaction_type == normalized
        )

    def to_dict(
        self,
        *,
        include_interactions: bool = True,
        include_atoms: bool = False,
        include_residue: bool = False,
        include_descriptors: bool = True,
    ) -> Dict[str, Any]:
        """
        Serialize the residue-level hydrophobic result.
        """

        result: Dict[str, Any] = {
            "residue_key": self.residue_key,
            "residue_identifier": self.residue_identifier,
            "interaction_count": self.interaction_count,
            "unique_receptor_atom_count": (
                self.unique_receptor_atom_count
            ),
            "unique_ligand_atom_count": (
                self.unique_ligand_atom_count
            ),
            "minimum_distance": (
                None
                if self.minimum_distance is None
                else float(self.minimum_distance)
            ),
            "mean_distance": (
                None
                if self.mean_distance is None
                else float(self.mean_distance)
            ),
            "maximum_distance": (
                None
                if self.maximum_distance is None
                else float(self.maximum_distance)
            ),
            "group_score": (
                None
                if self.group_score is None
                else float(self.group_score)
            ),
            "mean_score": (
                None
                if self.mean_score is None
                else float(self.mean_score)
            ),
            "maximum_score": (
                None
                if self.maximum_score is None
                else float(self.maximum_score)
            ),
            "strong_count": self.strong_count,
            "aromatic_contact_count": (
                self.aromatic_contact_count
            ),
            "classifications": sorted(self.classifications),
            "interaction_types": sorted(
                self.interaction_types
            ),
            "is_hotspot": self.is_hotspot,
            "metadata": dict(self.metadata),
        }

        if include_interactions:
            result["interactions"] = [
                interaction.to_dict(
                    include_atoms=include_atoms,
                    include_residue=include_residue,
                    include_descriptors=include_descriptors,
                )
                for interaction in self.interactions
            ]

        if include_residue:
            result["residue"] = self.residue

        return result


# -----------------------------------------------------------------------------
# Aggregate hydrophobic statistics
# -----------------------------------------------------------------------------

@dataclass(
    frozen=True,
    slots=True,
)
class HydrophobicStatistics:
    """
    Aggregate summary of a hydrophobic-interaction analysis.

    This structure stores computed values only. Statistical calculation is
    implemented in later sections; this dataclass validates, freezes and
    serializes the resulting summary.
    """

    interaction_count: int = 0
    residue_count: int = 0

    receptor_atom_count: int = 0
    ligand_atom_count: int = 0

    very_strong_count: int = 0
    strong_count: int = 0
    moderate_count: int = 0
    weak_count: int = 0
    marginal_count: int = 0
    unknown_count: int = 0

    aliphatic_aliphatic_count: int = 0
    aliphatic_aromatic_count: int = 0
    aromatic_aliphatic_count: int = 0
    aromatic_aromatic_count: int = 0
    mixed_count: int = 0

    hotspot_count: int = 0

    minimum_distance: Optional[np.float64] = None
    mean_distance: Optional[np.float64] = None
    median_distance: Optional[np.float64] = None
    maximum_distance: Optional[np.float64] = None
    distance_standard_deviation: Optional[np.float64] = None

    minimum_score: Optional[np.float64] = None
    mean_score: Optional[np.float64] = None
    median_score: Optional[np.float64] = None
    maximum_score: Optional[np.float64] = None
    total_score: np.float64 = np.float64(0.0)

    minimum_strength: Optional[np.float64] = None
    mean_strength: Optional[np.float64] = None
    maximum_strength: Optional[np.float64] = None

    classification_counts: Mapping[str, int] = field(
        default_factory=dict
    )

    interaction_type_counts: Mapping[str, int] = field(
        default_factory=dict
    )

    residue_interaction_counts: Mapping[str, int] = field(
        default_factory=dict
    )

    residue_scores: Mapping[str, float] = field(
        default_factory=dict
    )

    metadata: Mapping[str, Any] = field(
        default_factory=lambda: _EMPTY_METADATA,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        """Validate and normalize aggregate statistics."""

        count_fields = (
            "interaction_count",
            "residue_count",
            "receptor_atom_count",
            "ligand_atom_count",
            "very_strong_count",
            "strong_count",
            "moderate_count",
            "weak_count",
            "marginal_count",
            "unknown_count",
            "aliphatic_aliphatic_count",
            "aliphatic_aromatic_count",
            "aromatic_aliphatic_count",
            "aromatic_aromatic_count",
            "mixed_count",
            "hotspot_count",
        )

        for field_name in count_fields:
            normalized_value = _nonnegative_integer(
                getattr(self, field_name),
                name=field_name.replace("_", " "),
            )

            object.__setattr__(
                self,
                field_name,
                normalized_value,
            )

        optional_nonnegative_fields = (
            "minimum_distance",
            "mean_distance",
            "median_distance",
            "maximum_distance",
            "distance_standard_deviation",
            "minimum_score",
            "mean_score",
            "median_score",
            "maximum_score",
            "minimum_strength",
            "mean_strength",
            "maximum_strength",
        )

        for field_name in optional_nonnegative_fields:
            normalized_value = _optional_nonnegative_float(
                getattr(self, field_name),
                name=field_name.replace("_", " "),
            )

            object.__setattr__(
                self,
                field_name,
                normalized_value,
            )

        total_score = _nonnegative_float(
            self.total_score,
            name="total score",
        )

        classification_counts: Dict[str, int] = {}

        for key, value in self.classification_counts.items():
            normalized_key = validate_hydrophobic_classification(
                str(key)
            )

            classification_counts[normalized_key] = (
                _nonnegative_integer(
                    value,
                    name=(
                        f"classification count for "
                        f"{normalized_key}"
                    ),
                )
            )

        interaction_type_counts: Dict[str, int] = {}

        for key, value in self.interaction_type_counts.items():
            normalized_key = (
                validate_hydrophobic_interaction_type(
                    str(key)
                )
            )

            interaction_type_counts[normalized_key] = (
                _nonnegative_integer(
                    value,
                    name=(
                        f"interaction type count for "
                        f"{normalized_key}"
                    ),
                )
            )

        residue_interaction_counts: Dict[str, int] = {}

        for key, value in self.residue_interaction_counts.items():
            normalized_key = _normalize_required_string(
                key,
                name="residue interaction count key",
            )

            residue_interaction_counts[normalized_key] = (
                _nonnegative_integer(
                    value,
                    name=(
                        f"interaction count for residue "
                        f"{normalized_key}"
                    ),
                )
            )

        residue_scores: Dict[str, float] = {}

        for key, value in self.residue_scores.items():
            normalized_key = _normalize_required_string(
                key,
                name="residue score key",
            )

            residue_scores[normalized_key] = float(
                _nonnegative_float(
                    value,
                    name=f"score for residue {normalized_key}",
                )
            )

        if (
            self.minimum_distance is not None
            and self.maximum_distance is not None
            and self.minimum_distance > self.maximum_distance
        ):
            raise ValueError(
                "minimum_distance cannot exceed maximum_distance."
            )

        if (
            self.minimum_score is not None
            and self.maximum_score is not None
            and self.minimum_score > self.maximum_score
        ):
            raise ValueError(
                "minimum_score cannot exceed maximum_score."
            )

        if (
            self.minimum_strength is not None
            and self.maximum_strength is not None
            and self.minimum_strength > self.maximum_strength
        ):
            raise ValueError(
                "minimum_strength cannot exceed maximum_strength."
            )

        object.__setattr__(
            self,
            "total_score",
            total_score,
        )

        object.__setattr__(
            self,
            "classification_counts",
            MappingProxyType(classification_counts),
        )

        object.__setattr__(
            self,
            "interaction_type_counts",
            MappingProxyType(interaction_type_counts),
        )

        object.__setattr__(
            self,
            "residue_interaction_counts",
            MappingProxyType(residue_interaction_counts),
        )

        object.__setattr__(
            self,
            "residue_scores",
            MappingProxyType(residue_scores),
        )

        object.__setattr__(
            self,
            "metadata",
            _freeze_metadata(self.metadata),
        )

    @property
    def has_interactions(self) -> bool:
        """Return whether at least one interaction was detected."""

        return self.interaction_count > 0

    @property
    def classified_count(self) -> int:
        """Return the number of non-unknown classifications."""

        return (
            self.very_strong_count
            + self.strong_count
            + self.moderate_count
            + self.weak_count
            + self.marginal_count
        )

    @property
    def strong_or_better_count(self) -> int:
        """Return the strong plus very strong interaction count."""

        return (
            self.strong_count
            + self.very_strong_count
        )

    @property
    def aromatic_contact_count(self) -> int:
        """Return the number of interactions involving aromatic atoms."""

        return (
            self.aliphatic_aromatic_count
            + self.aromatic_aliphatic_count
            + self.aromatic_aromatic_count
        )

    @property
    def mean_interactions_per_residue(
        self,
    ) -> Optional[np.float64]:
        """Return the mean interaction count per contacted residue."""

        if self.residue_count == 0:
            return None

        return np.float64(
            self.interaction_count
            / self.residue_count
        )

    @property
    def strong_fraction(self) -> Optional[np.float64]:
        """Return the fraction classified as strong or very strong."""

        if self.interaction_count == 0:
            return None

        return np.float64(
            self.strong_or_better_count
            / self.interaction_count
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the aggregate statistics."""

        return {
            "interaction_count": self.interaction_count,
            "residue_count": self.residue_count,
            "receptor_atom_count": self.receptor_atom_count,
            "ligand_atom_count": self.ligand_atom_count,
            "very_strong_count": self.very_strong_count,
            "strong_count": self.strong_count,
            "moderate_count": self.moderate_count,
            "weak_count": self.weak_count,
            "marginal_count": self.marginal_count,
            "unknown_count": self.unknown_count,
            "classified_count": self.classified_count,
            "strong_or_better_count": (
                self.strong_or_better_count
            ),
            "aliphatic_aliphatic_count": (
                self.aliphatic_aliphatic_count
            ),
            "aliphatic_aromatic_count": (
                self.aliphatic_aromatic_count
            ),
            "aromatic_aliphatic_count": (
                self.aromatic_aliphatic_count
            ),
            "aromatic_aromatic_count": (
                self.aromatic_aromatic_count
            ),
            "mixed_count": self.mixed_count,
            "aromatic_contact_count": (
                self.aromatic_contact_count
            ),
            "hotspot_count": self.hotspot_count,
            "minimum_distance": (
                None
                if self.minimum_distance is None
                else float(self.minimum_distance)
            ),
            "mean_distance": (
                None
                if self.mean_distance is None
                else float(self.mean_distance)
            ),
            "median_distance": (
                None
                if self.median_distance is None
                else float(self.median_distance)
            ),
            "maximum_distance": (
                None
                if self.maximum_distance is None
                else float(self.maximum_distance)
            ),
            "distance_standard_deviation": (
                None
                if self.distance_standard_deviation is None
                else float(self.distance_standard_deviation)
            ),
            "minimum_score": (
                None
                if self.minimum_score is None
                else float(self.minimum_score)
            ),
            "mean_score": (
                None
                if self.mean_score is None
                else float(self.mean_score)
            ),
            "median_score": (
                None
                if self.median_score is None
                else float(self.median_score)
            ),
            "maximum_score": (
                None
                if self.maximum_score is None
                else float(self.maximum_score)
            ),
            "total_score": float(self.total_score),
            "minimum_strength": (
                None
                if self.minimum_strength is None
                else float(self.minimum_strength)
            ),
            "mean_strength": (
                None
                if self.mean_strength is None
                else float(self.mean_strength)
            ),
            "maximum_strength": (
                None
                if self.maximum_strength is None
                else float(self.maximum_strength)
            ),
            "mean_interactions_per_residue": (
                None
                if self.mean_interactions_per_residue is None
                else float(
                    self.mean_interactions_per_residue
                )
            ),
            "strong_fraction": (
                None
                if self.strong_fraction is None
                else float(self.strong_fraction)
            ),
            "classification_counts": dict(
                self.classification_counts
            ),
            "interaction_type_counts": dict(
                self.interaction_type_counts
            ),
            "residue_interaction_counts": dict(
                self.residue_interaction_counts
            ),
            "residue_scores": dict(self.residue_scores),
            "has_interactions": self.has_interactions,
            "metadata": dict(self.metadata),
        }


# -----------------------------------------------------------------------------
# Complete hydrophobic-analysis result
# -----------------------------------------------------------------------------

@dataclass(
    frozen=True,
    slots=True,
)
class HydrophobicAnalysisResult:
    """
    Complete result of a receptor-ligand hydrophobic analysis.

    Parameters
    ----------
    interactions
        Detected individual hydrophobic interactions.
    residue_groups
        Interactions grouped by receptor residue.
    receptor_hydrophobic_atoms
        Receptor atom descriptors accepted as hydrophobic.
    ligand_hydrophobic_atoms
        Ligand atom descriptors accepted as hydrophobic.
    receptor_atoms
        Original receptor atoms submitted to analysis.
    ligand_atoms
        Original ligand atoms submitted to analysis.
    minimum_distance
        Minimum accepted interaction distance.
    maximum_distance
        Maximum accepted interaction distance.
    grouping_distance
        Distance used for local grouping.
    statistics
        Aggregate statistics object.
    analysis_identifier
        Optional serializable analysis identifier.
    receptor_identifier
        Optional receptor-model identifier.
    ligand_identifier
        Optional ligand or pose identifier.
    metadata
        Additional analysis metadata.
    """

    interactions: Sequence[HydrophobicInteraction] = field(
        default_factory=tuple
    )

    residue_groups: Sequence[HydrophobicResidueGroup] = field(
        default_factory=tuple
    )

    receptor_hydrophobic_atoms: Sequence[HydrophobicAtom] = field(
        default_factory=tuple
    )

    ligand_hydrophobic_atoms: Sequence[HydrophobicAtom] = field(
        default_factory=tuple
    )

    receptor_atoms: Sequence[AtomLike] = field(
        default_factory=tuple
    )

    ligand_atoms: Sequence[AtomLike] = field(
        default_factory=tuple
    )

    minimum_distance: np.float64 = (
        DEFAULT_MINIMUM_HYDROPHOBIC_DISTANCE
    )

    maximum_distance: np.float64 = (
        DEFAULT_MAXIMUM_HYDROPHOBIC_DISTANCE
    )

    grouping_distance: np.float64 = DEFAULT_GROUPING_DISTANCE

    statistics: HydrophobicStatistics = field(
        default_factory=HydrophobicStatistics
    )

    analysis_identifier: Optional[str] = None
    receptor_identifier: Optional[str] = None
    ligand_identifier: Optional[str] = None

    metadata: Mapping[str, Any] = field(
        default_factory=lambda: _EMPTY_METADATA,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        """Validate and normalize the complete result."""

        normalized_interactions = tuple(self.interactions)

        for index, interaction in enumerate(
            normalized_interactions
        ):
            if not isinstance(
                interaction,
                HydrophobicInteraction,
            ):
                raise TypeError(
                    "All interactions entries must be "
                    "HydrophobicInteraction instances. "
                    f"Invalid entry at index {index}."
                )

        normalized_residue_groups = tuple(
            self.residue_groups
        )

        for index, residue_group in enumerate(
            normalized_residue_groups
        ):
            if not isinstance(
                residue_group,
                HydrophobicResidueGroup,
            ):
                raise TypeError(
                    "All residue_groups entries must be "
                    "HydrophobicResidueGroup instances. "
                    f"Invalid entry at index {index}."
                )

        normalized_receptor_descriptors = tuple(
            self.receptor_hydrophobic_atoms
        )

        for index, descriptor in enumerate(
            normalized_receptor_descriptors
        ):
            if not isinstance(
                descriptor,
                HydrophobicAtom,
            ):
                raise TypeError(
                    "All receptor_hydrophobic_atoms entries "
                    "must be HydrophobicAtom instances. "
                    f"Invalid entry at index {index}."
                )

        normalized_ligand_descriptors = tuple(
            self.ligand_hydrophobic_atoms
        )

        for index, descriptor in enumerate(
            normalized_ligand_descriptors
        ):
            if not isinstance(
                descriptor,
                HydrophobicAtom,
            ):
                raise TypeError(
                    "All ligand_hydrophobic_atoms entries "
                    "must be HydrophobicAtom instances. "
                    f"Invalid entry at index {index}."
                )

        normalized_receptor_atoms = tuple(
            self.receptor_atoms
        )

        normalized_ligand_atoms = tuple(
            self.ligand_atoms
        )

        minimum_distance, maximum_distance = (
            validate_hydrophobic_distance_limits(
                self.minimum_distance,
                self.maximum_distance,
            )
        )

        grouping_distance = _positive_float(
            self.grouping_distance,
            name="hydrophobic grouping distance",
        )

        if not isinstance(
            self.statistics,
            HydrophobicStatistics,
        ):
            raise TypeError(
                "statistics must be a HydrophobicStatistics instance."
            )

        interaction_identity_set = {
            interaction.interaction_identifier
            for interaction in normalized_interactions
        }

        grouped_identity_set = {
            interaction.interaction_identifier
            for group in normalized_residue_groups
            for interaction in group.interactions
        }

        unknown_grouped_interactions = (
            grouped_identity_set
            - interaction_identity_set
        )

        if unknown_grouped_interactions:
            raise ValueError(
                "residue_groups contain interactions absent from "
                "the complete interactions collection."
            )

        analysis_identifier = _normalize_optional_string(
            self.analysis_identifier
        )

        receptor_identifier = _normalize_optional_string(
            self.receptor_identifier
        )

        ligand_identifier = _normalize_optional_string(
            self.ligand_identifier
        )

        if analysis_identifier is None:
            identifier_parts = tuple(
                part
                for part in (
                    receptor_identifier,
                    ligand_identifier,
                )
                if part is not None
            )

            if identifier_parts:
                analysis_identifier = "|".join(
                    identifier_parts
                )

        object.__setattr__(
            self,
            "interactions",
            normalized_interactions,
        )

        object.__setattr__(
            self,
            "residue_groups",
            normalized_residue_groups,
        )

        object.__setattr__(
            self,
            "receptor_hydrophobic_atoms",
            normalized_receptor_descriptors,
        )

        object.__setattr__(
            self,
            "ligand_hydrophobic_atoms",
            normalized_ligand_descriptors,
        )

        object.__setattr__(
            self,
            "receptor_atoms",
            normalized_receptor_atoms,
        )

        object.__setattr__(
            self,
            "ligand_atoms",
            normalized_ligand_atoms,
        )

        object.__setattr__(
            self,
            "minimum_distance",
            minimum_distance,
        )

        object.__setattr__(
            self,
            "maximum_distance",
            maximum_distance,
        )

        object.__setattr__(
            self,
            "grouping_distance",
            grouping_distance,
        )

        object.__setattr__(
            self,
            "analysis_identifier",
            analysis_identifier,
        )

        object.__setattr__(
            self,
            "receptor_identifier",
            receptor_identifier,
        )

        object.__setattr__(
            self,
            "ligand_identifier",
            ligand_identifier,
        )

        object.__setattr__(
            self,
            "metadata",
            _freeze_metadata(self.metadata),
        )

    def __len__(self) -> int:
        """Return the number of detected interactions."""

        return len(self.interactions)

    def __iter__(self) -> Iterator[HydrophobicInteraction]:
        """Iterate over detected interactions."""

        return iter(self.interactions)

    def __getitem__(
        self,
        index: int,
    ) -> HydrophobicInteraction:
        """Return one interaction by index."""

        return self.interactions[index]

    @property
    def interaction_count(self) -> int:
        """Return the number of detected interactions."""

        return len(self.interactions)

    @property
    def residue_count(self) -> int:
        """Return the number of contacted receptor residues."""

        return len(self.residue_groups)

    @property
    def receptor_hydrophobic_atom_count(self) -> int:
        """Return the number of accepted receptor hydrophobic atoms."""

        return len(self.receptor_hydrophobic_atoms)

    @property
    def ligand_hydrophobic_atom_count(self) -> int:
        """Return the number of accepted ligand hydrophobic atoms."""

        return len(self.ligand_hydrophobic_atoms)

    @property
    def receptor_atom_count(self) -> int:
        """Return the number of submitted receptor atoms."""

        return len(self.receptor_atoms)

    @property
    def ligand_atom_count(self) -> int:
        """Return the number of submitted ligand atoms."""

        return len(self.ligand_atoms)

    @property
    def has_interactions(self) -> bool:
        """Return whether at least one interaction was detected."""

        return bool(self.interactions)

    @property
    def has_residue_groups(self) -> bool:
        """Return whether residue grouping has been performed."""

        return bool(self.residue_groups)

    @property
    def minimum_detected_distance(
        self,
    ) -> Optional[np.float64]:
        """Return the shortest detected interaction distance."""

        if not self.interactions:
            return None

        return np.float64(
            min(
                interaction.distance
                for interaction in self.interactions
            )
        )

    @property
    def mean_detected_distance(
        self,
    ) -> Optional[np.float64]:
        """Return the mean detected interaction distance."""

        if not self.interactions:
            return None

        return np.float64(
            np.mean(
                [
                    interaction.distance
                    for interaction in self.interactions
                ]
            )
        )

    @property
    def total_score(self) -> np.float64:
        """Return the sum of individual interaction scores."""

        return np.float64(
            sum(
                float(interaction.score)
                for interaction in self.interactions
            )
        )

    @property
    def strong_interaction_count(self) -> int:
        """Return the number of strong or very strong interactions."""

        return sum(
            interaction.is_strong
            for interaction in self.interactions
        )

    @property
    def hotspot_count(self) -> int:
        """Return the number of residue groups identified as hotspots."""

        return sum(
            group.is_hotspot
            for group in self.residue_groups
        )

    def get_residue_group(
        self,
        residue_key: ResidueContactKey,
    ) -> Optional[HydrophobicResidueGroup]:
        """
        Return a residue group by normalized residue key.
        """

        normalized_key = _normalize_residue_key(
            residue_key
        )

        for group in self.residue_groups:
            if group.residue_key == normalized_key:
                return group

        return None

    def interactions_by_classification(
        self,
        classification: HydrophobicClassification,
    ) -> Tuple[HydrophobicInteraction, ...]:
        """
        Return interactions matching a geometric classification.
        """

        normalized = validate_hydrophobic_classification(
            classification
        )

        return tuple(
            interaction
            for interaction in self.interactions
            if interaction.classification == normalized
        )

    def interactions_by_type(
        self,
        interaction_type: HydrophobicInteractionType,
    ) -> Tuple[HydrophobicInteraction, ...]:
        """
        Return interactions matching a chemical type.
        """

        normalized = validate_hydrophobic_interaction_type(
            interaction_type
        )

        return tuple(
            interaction
            for interaction in self.interactions
            if interaction.interaction_type == normalized
        )

    def to_dict(
        self,
        *,
        include_interactions: bool = True,
        include_residue_groups: bool = True,
        include_atom_descriptors: bool = True,
        include_atoms: bool = False,
        include_residues: bool = False,
    ) -> Dict[str, Any]:
        """
        Serialize the complete analysis result.
        """

        result: Dict[str, Any] = {
            "analysis_identifier": self.analysis_identifier,
            "receptor_identifier": self.receptor_identifier,
            "ligand_identifier": self.ligand_identifier,
            "interaction_count": self.interaction_count,
            "residue_count": self.residue_count,
            "receptor_atom_count": self.receptor_atom_count,
            "ligand_atom_count": self.ligand_atom_count,
            "receptor_hydrophobic_atom_count": (
                self.receptor_hydrophobic_atom_count
            ),
            "ligand_hydrophobic_atom_count": (
                self.ligand_hydrophobic_atom_count
            ),
            "minimum_distance_cutoff": float(
                self.minimum_distance
            ),
            "maximum_distance_cutoff": float(
                self.maximum_distance
            ),
            "grouping_distance": float(
                self.grouping_distance
            ),
            "minimum_detected_distance": (
                None
                if self.minimum_detected_distance is None
                else float(self.minimum_detected_distance)
            ),
            "mean_detected_distance": (
                None
                if self.mean_detected_distance is None
                else float(self.mean_detected_distance)
            ),
            "total_score": float(self.total_score),
            "strong_interaction_count": (
                self.strong_interaction_count
            ),
            "hotspot_count": self.hotspot_count,
            "has_interactions": self.has_interactions,
            "has_residue_groups": self.has_residue_groups,
            "statistics": self.statistics.to_dict(),
            "metadata": dict(self.metadata),
        }

        if include_interactions:
            result["interactions"] = [
                interaction.to_dict(
                    include_atoms=include_atoms,
                    include_residue=include_residues,
                    include_descriptors=(
                        include_atom_descriptors
                    ),
                )
                for interaction in self.interactions
            ]

        if include_residue_groups:
            result["residue_groups"] = [
                group.to_dict(
                    include_interactions=include_interactions,
                    include_atoms=include_atoms,
                    include_residue=include_residues,
                    include_descriptors=(
                        include_atom_descriptors
                    ),
                )
                for group in self.residue_groups
            ]

        if include_atom_descriptors:
            result["receptor_hydrophobic_atoms"] = [
                descriptor.to_dict(
                    include_atom=include_atoms,
                    include_residue=include_residues,
                )
                for descriptor in self.receptor_hydrophobic_atoms
            ]

            result["ligand_hydrophobic_atoms"] = [
                descriptor.to_dict(
                    include_atom=include_atoms,
                    include_residue=include_residues,
                )
                for descriptor in self.ligand_hydrophobic_atoms
            ]

        if include_atoms:
            result["receptor_atoms"] = self.receptor_atoms
            result["ligand_atoms"] = self.ligand_atoms

        return result


# -----------------------------------------------------------------------------
# Empty typed result collections
# -----------------------------------------------------------------------------

_EMPTY_HYDROPHOBIC_ATOMS: Final[
    Tuple[HydrophobicAtom, ...]
] = ()

_EMPTY_HYDROPHOBIC_INTERACTIONS: Final[
    Tuple[HydrophobicInteraction, ...]
] = ()

_EMPTY_HYDROPHOBIC_RESIDUE_GROUPS: Final[
    Tuple[HydrophobicResidueGroup, ...]
] = ()


# -----------------------------------------------------------------------------
# Section 3 public names
# -----------------------------------------------------------------------------

_SECTION_3_PUBLIC_NAMES: Final[Tuple[str, ...]] = (
    "HydrophobicAtom",
    "HydrophobicInteraction",
    "HydrophobicResidueGroup",
    "HydrophobicStatistics",
    "HydrophobicAnalysisResult",
)

for public_name in _SECTION_3_PUBLIC_NAMES:
    if public_name not in __all__:
        __all__.append(public_name)


# =============================================================================
# End of Section 3
# =============================================================================

# =============================================================================
# Section 4 — Hydrophobic-atom perception
# =============================================================================


# -----------------------------------------------------------------------------
# Generic chemical-perception helpers
# -----------------------------------------------------------------------------

def _safe_getattr(
    obj: Any,
    names: Sequence[str],
    default: Any = None,
) -> Any:
    """
    Return the first successfully resolved attribute.

    Callable attributes without required arguments are evaluated
    automatically. Attribute-access errors are ignored to preserve
    compatibility with ChimeraX objects and lightweight test doubles.
    """

    if obj is None:
        return default

    for name in names:
        try:
            value = getattr(obj, name)
        except Exception:
            continue

        if callable(value):
            try:
                value = value()
            except TypeError:
                continue
            except Exception:
                continue

        if value is not None:
            return value

    return default


def _safe_boolean_attribute(
    obj: Any,
    names: Sequence[str],
) -> Optional[bool]:
    """
    Resolve an optional boolean-like attribute.

    Returns ``None`` when no reliable value can be obtained.
    """

    value = _safe_getattr(
        obj,
        names,
        default=None,
    )

    if value is None:
        return None

    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    if isinstance(value, (int, np.integer)):
        return bool(value)

    if isinstance(value, str):
        normalized = value.strip().lower()

        if normalized in {
            "true",
            "yes",
            "y",
            "1",
            "aromatic",
            "aro",
        }:
            return True

        if normalized in {
            "false",
            "no",
            "n",
            "0",
            "nonaromatic",
            "non-aromatic",
        }:
            return False

    return None


def _normalize_atom_name(
    atom: Optional[AtomLike],
) -> str:
    """Return an uppercase atom name or an empty string."""

    if atom is None:
        return ""

    try:
        name = get_atom_name(atom)
    except Exception:
        name = _safe_getattr(
            atom,
            (
                "name",
                "atom_name",
                "idatm_name",
            ),
            default="",
        )

    return str(name).strip().upper() if name is not None else ""


def _normalize_residue_name(
    residue: Optional[ResidueLike],
) -> str:
    """Return an uppercase residue name or an empty string."""

    if residue is None:
        return ""

    name = _safe_getattr(
        residue,
        (
            "name",
            "resname",
            "residue_name",
            "type",
        ),
        default="",
    )

    return str(name).strip().upper() if name is not None else ""


def get_atom_residue_name(
    atom: AtomLike,
) -> str:
    """Return the normalized parent-residue name of an atom."""

    residue = _safe_atom_residue(atom)

    return _normalize_residue_name(residue)


def get_atom_formal_charge(
    atom: AtomLike,
) -> Optional[np.float64]:
    """
    Return an atom's formal charge, when available.

    Several common property names are inspected because ChimeraX models,
    PDB-derived structures and ligand formats may expose charge data
    differently.
    """

    value = _safe_getattr(
        atom,
        (
            "formal_charge",
            "formalCharge",
            "integer_charge",
            "valence_charge",
        ),
        default=None,
    )

    if value is None:
        return None

    try:
        return _finite_float(
            value,
            name="formal charge",
        )
    except (TypeError, ValueError):
        return None


def get_atom_partial_charge(
    atom: AtomLike,
) -> Optional[np.float64]:
    """
    Return an atom's partial charge, when available.
    """

    value = _safe_getattr(
        atom,
        (
            "charge",
            "partial_charge",
            "partialCharge",
            "gasteiger_charge",
            "gasteigerCharge",
        ),
        default=None,
    )

    if value is None:
        return None

    try:
        return _finite_float(
            value,
            name="partial charge",
        )
    except (TypeError, ValueError):
        return None


def _iterable_to_tuple(
    value: Any,
) -> Tuple[Any, ...]:
    """Convert an arbitrary collection-like value to a tuple."""

    if value is None:
        return ()

    if isinstance(value, tuple):
        return value

    if isinstance(value, (str, bytes)):
        return ()

    try:
        return tuple(value)
    except TypeError:
        return (value,)


def get_bonded_neighbors(
    atom: AtomLike,
) -> Tuple[AtomLike, ...]:
    """
    Return directly bonded neighboring atoms.

    The function first checks atom-level neighbor collections and then
    attempts to infer opposite atoms from bond objects.
    """

    if atom is None:
        return ()

    neighbors_value = _safe_getattr(
        atom,
        (
            "neighbors",
            "bonded_atoms",
            "bondedAtoms",
            "connected_atoms",
            "connectedAtoms",
        ),
        default=None,
    )

    neighbors: List[AtomLike] = []

    for neighbor in _iterable_to_tuple(neighbors_value):
        if (
            neighbor is not None
            and neighbor is not atom
            and neighbor not in neighbors
        ):
            neighbors.append(neighbor)

    if neighbors:
        return tuple(neighbors)

    bonds_value = _safe_getattr(
        atom,
        (
            "bonds",
            "bond_objects",
            "bondObjects",
        ),
        default=None,
    )

    for bond in _iterable_to_tuple(bonds_value):
        bond_atoms = _safe_getattr(
            bond,
            (
                "atoms",
                "endpoints",
                "atom_pair",
            ),
            default=None,
        )

        bond_atom_tuple = _iterable_to_tuple(bond_atoms)

        if len(bond_atom_tuple) >= 2:
            for candidate in bond_atom_tuple:
                if (
                    candidate is not None
                    and candidate is not atom
                    and candidate not in neighbors
                ):
                    neighbors.append(candidate)

            continue

        for candidate_name in (
            "atom1",
            "atom2",
            "a1",
            "a2",
            "first_atom",
            "second_atom",
        ):
            candidate = _safe_getattr(
                bond,
                (candidate_name,),
                default=None,
            )

            if (
                candidate is not None
                and candidate is not atom
                and candidate not in neighbors
            ):
                neighbors.append(candidate)

    return tuple(neighbors)


def get_heavy_neighbors(
    atom: AtomLike,
) -> Tuple[AtomLike, ...]:
    """Return directly bonded non-hydrogen neighbors."""

    return tuple(
        neighbor
        for neighbor in get_bonded_neighbors(atom)
        if is_heavy_atom(neighbor)
    )


def get_polar_neighbors(
    atom: AtomLike,
) -> Tuple[AtomLike, ...]:
    """
    Return directly bonded polar heteroatom neighbors.
    """

    polar_neighbors: List[AtomLike] = []

    for neighbor in get_bonded_neighbors(atom):
        element = _safe_atom_element(neighbor)

        if element in POLAR_HETEROATOM_ELEMENTS:
            polar_neighbors.append(neighbor)

    return tuple(polar_neighbors)


def count_heavy_neighbors(
    atom: AtomLike,
) -> int:
    """Return the number of directly bonded heavy atoms."""

    return len(get_heavy_neighbors(atom))


def count_polar_neighbors(
    atom: AtomLike,
) -> int:
    """Return the number of directly bonded polar heteroatoms."""

    return len(get_polar_neighbors(atom))


# -----------------------------------------------------------------------------
# Bond inspection
# -----------------------------------------------------------------------------

def _get_bonds_between(
    atom_a: AtomLike,
    atom_b: AtomLike,
) -> Tuple[Any, ...]:
    """Return bond objects connecting two atoms, when available."""

    matching_bonds: List[Any] = []

    bonds = _iterable_to_tuple(
        _safe_getattr(
            atom_a,
            (
                "bonds",
                "bond_objects",
                "bondObjects",
            ),
            default=None,
        )
    )

    for bond in bonds:
        atoms = _iterable_to_tuple(
            _safe_getattr(
                bond,
                (
                    "atoms",
                    "endpoints",
                    "atom_pair",
                ),
                default=None,
            )
        )

        if (
            len(atoms) >= 2
            and atom_a in atoms
            and atom_b in atoms
        ):
            matching_bonds.append(bond)
            continue

        first_atom = _safe_getattr(
            bond,
            (
                "atom1",
                "a1",
                "first_atom",
            ),
            default=None,
        )

        second_atom = _safe_getattr(
            bond,
            (
                "atom2",
                "a2",
                "second_atom",
            ),
            default=None,
        )

        if {
            id(first_atom),
            id(second_atom),
        } == {
            id(atom_a),
            id(atom_b),
        }:
            matching_bonds.append(bond)

    return tuple(matching_bonds)


def get_bond_order(
    atom_a: AtomLike,
    atom_b: AtomLike,
) -> Optional[np.float64]:
    """
    Return the bond order between two atoms, when available.

    Aromatic bond values are normalized to ``1.5``.
    """

    direct_value = None

    bond_orders = _safe_getattr(
        atom_a,
        (
            "bond_orders",
            "bondOrders",
        ),
        default=None,
    )

    if isinstance(bond_orders, Mapping):
        for key in (
            atom_b,
            id(atom_b),
            _safe_atom_identifier(atom_b),
        ):
            try:
                if key in bond_orders:
                    direct_value = bond_orders[key]
                    break
            except Exception:
                continue

    if direct_value is not None:
        try:
            return _finite_float(
                direct_value,
                name="bond order",
            )
        except (TypeError, ValueError):
            pass

    for bond in _get_bonds_between(
        atom_a,
        atom_b,
    ):
        aromatic = _safe_boolean_attribute(
            bond,
            (
                "aromatic",
                "is_aromatic",
                "isAromatic",
            ),
        )

        if aromatic:
            return np.float64(1.5)

        value = _safe_getattr(
            bond,
            (
                "order",
                "bond_order",
                "bondOrder",
                "type",
            ),
            default=None,
        )

        if isinstance(value, str):
            normalized = value.strip().lower()

            string_orders: Mapping[str, float] = {
                "single": 1.0,
                "double": 2.0,
                "triple": 3.0,
                "aromatic": 1.5,
                "amide": 1.0,
            }

            if normalized in string_orders:
                return np.float64(
                    string_orders[normalized]
                )

        if value is not None:
            try:
                return _finite_float(
                    value,
                    name="bond order",
                )
            except (TypeError, ValueError):
                continue

    return None


def is_double_bonded_to_element(
    atom: AtomLike,
    element: str,
) -> bool:
    """
    Return whether an atom has a double bond to a selected element.
    """

    normalized_element = _normalize_element_symbol(element)

    for neighbor in get_bonded_neighbors(atom):
        if _safe_atom_element(neighbor) != normalized_element:
            continue

        bond_order = get_bond_order(
            atom,
            neighbor,
        )

        if (
            bond_order is not None
            and bond_order >= 1.75
        ):
            return True

    return False


# -----------------------------------------------------------------------------
# Protein and ligand context
# -----------------------------------------------------------------------------

def is_standard_protein_atom(
    atom: AtomLike,
) -> bool:
    """
    Return whether the atom belongs to a recognized amino-acid residue.
    """

    residue_name = get_atom_residue_name(atom)

    return (
        residue_name in STANDARD_AMINO_ACID_NAMES
        or residue_name in HISTIDINE_RESIDUE_NAMES
        or residue_name in CYSTEINE_RESIDUE_NAMES
    )


def is_water_atom(
    atom: AtomLike,
) -> bool:
    """Return whether the atom belongs to a recognized water residue."""

    return get_atom_residue_name(atom) in WATER_RESIDUE_NAMES


def is_nucleic_acid_atom(
    atom: AtomLike,
) -> bool:
    """Return whether the atom belongs to a recognized nucleic acid."""

    return (
        get_atom_residue_name(atom)
        in NUCLEIC_ACID_RESIDUE_NAMES
    )


def infer_hydrophobic_atom_role(
    atom: AtomLike,
    *,
    receptor_atoms: Optional[Iterable[AtomLike]] = None,
    ligand_atoms: Optional[Iterable[AtomLike]] = None,
    default: HydrophobicAtomRole = HYDROPHOBIC_ROLE_UNKNOWN,
) -> HydrophobicAtomRole:
    """
    Infer whether an atom belongs to the receptor or ligand.

    Explicit collections take precedence. When neither collection is
    supplied, protein context is interpreted as receptor context.
    """

    normalized_default = validate_hydrophobic_atom_role(
        default
    )

    if receptor_atoms is not None:
        for receptor_atom in receptor_atoms:
            if receptor_atom is atom:
                return HYDROPHOBIC_ROLE_RECEPTOR

    if ligand_atoms is not None:
        for ligand_atom in ligand_atoms:
            if ligand_atom is atom:
                return HYDROPHOBIC_ROLE_LIGAND

    role_value = _safe_getattr(
        atom,
        (
            "hydrophobic_role",
            "molecular_role",
            "role",
        ),
        default=None,
    )

    if isinstance(role_value, str):
        normalized_role = role_value.strip().lower()

        role_aliases: Mapping[str, HydrophobicAtomRole] = {
            "protein": HYDROPHOBIC_ROLE_RECEPTOR,
            "receptor": HYDROPHOBIC_ROLE_RECEPTOR,
            "target": HYDROPHOBIC_ROLE_RECEPTOR,
            "ligand": HYDROPHOBIC_ROLE_LIGAND,
            "pose": HYDROPHOBIC_ROLE_LIGAND,
            "compound": HYDROPHOBIC_ROLE_LIGAND,
        }

        if normalized_role in role_aliases:
            return role_aliases[normalized_role]

    if is_standard_protein_atom(atom):
        return HYDROPHOBIC_ROLE_RECEPTOR

    return normalized_default


# -----------------------------------------------------------------------------
# Aromatic and aliphatic perception
# -----------------------------------------------------------------------------

def is_protein_aromatic_atom(
    atom: AtomLike,
) -> bool:
    """
    Return whether an atom is a recognized aromatic protein atom.
    """

    residue_name = get_atom_residue_name(atom)
    atom_name = _normalize_atom_name(atom)

    aromatic_names = AROMATIC_PROTEIN_ATOMS_BY_RESIDUE.get(
        residue_name
    )

    return (
        aromatic_names is not None
        and atom_name in aromatic_names
    )


def is_aromatic_atom(
    atom: AtomLike,
) -> bool:
    """
    Determine whether an atom has aromatic character.

    Explicit aromatic flags and atom-type annotations take precedence,
    followed by standard protein residue definitions.
    """

    explicit_aromatic = _safe_boolean_attribute(
        atom,
        (
            "is_aromatic",
            "aromatic",
            "isAromatic",
            "in_aromatic_ring",
            "inAromaticRing",
        ),
    )

    if explicit_aromatic is not None:
        return explicit_aromatic

    atom_type = _safe_getattr(
        atom,
        (
            "idatm_type",
            "idatmType",
            "atom_type",
            "atomType",
            "sybyl_type",
            "gaff_type",
        ),
        default=None,
    )

    if atom_type is not None:
        normalized_type = str(atom_type).strip().lower()

        aromatic_markers = (
            ".ar",
            "aromatic",
            "car",
            "nar",
            "c.ar",
            "n.ar",
            "ca",
        )

        if any(
            marker == normalized_type
            or marker in normalized_type
            for marker in aromatic_markers
        ):
            return True

    return is_protein_aromatic_atom(atom)


def is_protein_aliphatic_atom(
    atom: AtomLike,
) -> bool:
    """
    Return whether an atom is a known aliphatic protein-side-chain atom.
    """

    residue_name = get_atom_residue_name(atom)
    atom_name = _normalize_atom_name(atom)

    if is_protein_aromatic_atom(atom):
        return False

    hydrophobic_names = HYDROPHOBIC_PROTEIN_ATOMS_BY_RESIDUE.get(
        residue_name
    )

    if hydrophobic_names is None:
        return False

    return atom_name in hydrophobic_names


def is_aliphatic_atom(
    atom: AtomLike,
) -> bool:
    """
    Determine whether an atom belongs to an aliphatic environment.
    """

    if is_aromatic_atom(atom):
        return False

    explicit_aliphatic = _safe_boolean_attribute(
        atom,
        (
            "is_aliphatic",
            "aliphatic",
            "isAliphatic",
        ),
    )

    if explicit_aliphatic is not None:
        return explicit_aliphatic

    if is_protein_aliphatic_atom(atom):
        return True

    element = _safe_atom_element(atom)

    if element != CARBON_ELEMENT:
        return False

    atom_type = _safe_getattr(
        atom,
        (
            "idatm_type",
            "idatmType",
            "atom_type",
            "atomType",
            "sybyl_type",
            "gaff_type",
        ),
        default=None,
    )

    if atom_type is not None:
        normalized_type = str(atom_type).strip().lower()

        aliphatic_markers = {
            "c3",
            "c2",
            "c1",
            "c.3",
            "c.2",
            "c.1",
            "ct",
            "c",
            "sp3",
            "sp2",
            "sp",
        }

        if normalized_type in aliphatic_markers:
            return True

    heavy_neighbors = get_heavy_neighbors(atom)

    return bool(heavy_neighbors)


# -----------------------------------------------------------------------------
# Polarized-carbon exclusions
# -----------------------------------------------------------------------------

def is_carbonyl_carbon(
    atom: AtomLike,
) -> bool:
    """
    Return whether an atom is a carbonyl carbon.

    Bond-order information is preferred. Standard protein backbone carbon
    names and common atom-type annotations are used as fallbacks.
    """

    if _safe_atom_element(atom) != CARBON_ELEMENT:
        return False

    if is_double_bonded_to_element(
        atom,
        OXYGEN_ELEMENT,
    ):
        return True

    atom_type = _safe_getattr(
        atom,
        (
            "idatm_type",
            "idatmType",
            "atom_type",
            "atomType",
            "sybyl_type",
        ),
        default=None,
    )

    if atom_type is not None:
        normalized_type = str(atom_type).strip().lower()

        if normalized_type in {
            "c.2",
            "c2",
            "c",
            "co",
            "carbonyl",
            "c_carbonyl",
        }:
            oxygen_neighbors = tuple(
                neighbor
                for neighbor in get_bonded_neighbors(atom)
                if _safe_atom_element(neighbor)
                == OXYGEN_ELEMENT
            )

            if oxygen_neighbors:
                return True

    if (
        is_standard_protein_atom(atom)
        and _normalize_atom_name(atom)
        in PROTEIN_CARBONYL_CARBON_NAMES
    ):
        return True

    return False


def is_carboxylate_carbon(
    atom: AtomLike,
) -> bool:
    """
    Return whether an atom is a carboxylate or carboxylic-acid carbon.
    """

    if _safe_atom_element(atom) != CARBON_ELEMENT:
        return False

    oxygen_neighbors = tuple(
        neighbor
        for neighbor in get_bonded_neighbors(atom)
        if _safe_atom_element(neighbor)
        == OXYGEN_ELEMENT
    )

    if len(oxygen_neighbors) < 2:
        return False

    double_bonded_oxygen_count = 0
    charged_oxygen_count = 0

    for oxygen in oxygen_neighbors:
        bond_order = get_bond_order(
            atom,
            oxygen,
        )

        if (
            bond_order is not None
            and bond_order >= 1.75
        ):
            double_bonded_oxygen_count += 1

        formal_charge = get_atom_formal_charge(oxygen)

        if (
            formal_charge is not None
            and formal_charge < 0.0
        ):
            charged_oxygen_count += 1

    if (
        double_bonded_oxygen_count >= 1
        or charged_oxygen_count >= 1
    ):
        return True

    residue_name = get_atom_residue_name(atom)
    atom_name = _normalize_atom_name(atom)

    known_carboxylate_atoms: Mapping[str, FrozenSet[str]] = {
        "ASP": frozenset({"CG"}),
        "GLU": frozenset({"CD"}),
    }

    return atom_name in known_carboxylate_atoms.get(
        residue_name,
        frozenset(),
    )


def is_amide_carbon(
    atom: AtomLike,
) -> bool:
    """
    Return whether an atom is the carbonyl carbon of an amide group.
    """

    if not is_carbonyl_carbon(atom):
        return False

    return any(
        _safe_atom_element(neighbor)
        == NITROGEN_ELEMENT
        for neighbor in get_bonded_neighbors(atom)
    )


def is_nitrile_carbon(
    atom: AtomLike,
) -> bool:
    """
    Return whether a carbon is triple-bonded to nitrogen.
    """

    if _safe_atom_element(atom) != CARBON_ELEMENT:
        return False

    for neighbor in get_bonded_neighbors(atom):
        if _safe_atom_element(neighbor) != NITROGEN_ELEMENT:
            continue

        bond_order = get_bond_order(
            atom,
            neighbor,
        )

        if (
            bond_order is not None
            and bond_order >= 2.75
        ):
            return True

    atom_type = _safe_getattr(
        atom,
        (
            "atom_type",
            "idatm_type",
            "sybyl_type",
        ),
        default=None,
    )

    if atom_type is None:
        return False

    return str(atom_type).strip().lower() in {
        "c.1",
        "c1",
        "cy",
        "nitrile",
    }


def is_carbon_bound_to_multiple_heteroatoms(
    atom: AtomLike,
    *,
    minimum_count: int = 2,
) -> bool:
    """
    Return whether a carbon is directly bonded to multiple heteroatoms.
    """

    if _safe_atom_element(atom) != CARBON_ELEMENT:
        return False

    heteroatom_count = sum(
        _safe_atom_element(neighbor)
        in POLAR_HETEROATOM_ELEMENTS
        for neighbor in get_heavy_neighbors(atom)
    )

    return heteroatom_count >= minimum_count


def is_strongly_polarized_carbon(
    atom: AtomLike,
    *,
    maximum_absolute_partial_charge: Optional[Number] = None,
) -> bool:
    """
    Return whether a carbon is too polarized for hydrophobic assignment.

    Carbonyl, carboxylate, amide and nitrile carbons are always excluded.
    Charge information and the number of directly bonded heteroatoms are
    then used as additional criteria.
    """

    if _safe_atom_element(atom) != CARBON_ELEMENT:
        return False

    if (
        is_carbonyl_carbon(atom)
        or is_carboxylate_carbon(atom)
        or is_amide_carbon(atom)
        or is_nitrile_carbon(atom)
    ):
        return True

    if is_carbon_bound_to_multiple_heteroatoms(atom):
        return True

    formal_charge = get_atom_formal_charge(atom)

    if (
        formal_charge is not None
        and abs(formal_charge)
        > DEFAULT_MAXIMUM_ABSOLUTE_FORMAL_CHARGE
    ):
        return True

    charge_limit = (
        get_default_maximum_absolute_partial_charge()
        if maximum_absolute_partial_charge is None
        else _nonnegative_float(
            maximum_absolute_partial_charge,
            name="maximum absolute partial charge",
        )
    )

    partial_charge = get_atom_partial_charge(atom)

    return (
        partial_charge is not None
        and abs(partial_charge) > charge_limit
    )


# -----------------------------------------------------------------------------
# Sulfur and halogen perception
# -----------------------------------------------------------------------------

def is_hydrophobic_sulfur(
    atom: AtomLike,
) -> bool:
    """
    Return whether a sulfur atom has a sufficiently nonpolar environment.
    """

    if _safe_atom_element(atom) != SULFUR_ELEMENT:
        return False

    formal_charge = get_atom_formal_charge(atom)

    if (
        formal_charge is not None
        and abs(formal_charge) > 0.0
    ):
        return False

    partial_charge = get_atom_partial_charge(atom)

    if (
        partial_charge is not None
        and abs(partial_charge)
        > get_default_maximum_absolute_partial_charge()
    ):
        return False

    heavy_neighbors = get_heavy_neighbors(atom)

    if not heavy_neighbors:
        return False

    neighbor_elements = tuple(
        _safe_atom_element(neighbor)
        for neighbor in heavy_neighbors
    )

    if any(
        element in {
            OXYGEN_ELEMENT,
            NITROGEN_ELEMENT,
            PHOSPHORUS_ELEMENT,
        }
        for element in neighbor_elements
    ):
        return False

    if get_atom_residue_name(atom) in {
        "MET",
        "CYS",
        "CYM",
        "CYX",
    }:
        return True

    return all(
        element == CARBON_ELEMENT
        for element in neighbor_elements
    )


def is_hydrophobic_halogen(
    atom: AtomLike,
) -> bool:
    """
    Return whether a halogen is covalently attached to a nonpolar carbon.
    """

    if _safe_atom_element(atom) not in HALOGEN_ELEMENTS:
        return False

    formal_charge = get_atom_formal_charge(atom)

    if (
        formal_charge is not None
        and abs(formal_charge) > 0.0
    ):
        return False

    heavy_neighbors = get_heavy_neighbors(atom)

    if len(heavy_neighbors) != 1:
        return False

    bonded_atom = heavy_neighbors[0]

    if _safe_atom_element(bonded_atom) != CARBON_ELEMENT:
        return False

    return not is_strongly_polarized_carbon(bonded_atom)


# -----------------------------------------------------------------------------
# Protein-specific hydrophobic perception
# -----------------------------------------------------------------------------

def is_known_hydrophobic_protein_atom(
    atom: AtomLike,
    *,
    hydrophobic_residue_names: Optional[Iterable[str]] = None,
) -> bool:
    """
    Return whether an atom matches a curated protein hydrophobic definition.

    The residue name alone is insufficient: the atom must also be present
    in the corresponding side-chain atom-name definition.
    """

    residue_name = get_atom_residue_name(atom)
    atom_name = _normalize_atom_name(atom)

    allowed_residues = (
        get_default_hydrophobic_residue_names()
        if hydrophobic_residue_names is None
        else frozenset(
            str(name).strip().upper()
            for name in hydrophobic_residue_names
            if str(name).strip()
        )
    )

    if (
        residue_name not in allowed_residues
        and residue_name not in HISTIDINE_RESIDUE_NAMES
        and residue_name not in CYSTEINE_RESIDUE_NAMES
    ):
        return False

    allowed_atoms = HYDROPHOBIC_PROTEIN_ATOMS_BY_RESIDUE.get(
        residue_name
    )

    if (
        allowed_atoms is None
        or atom_name not in allowed_atoms
    ):
        return False

    element = _safe_atom_element(atom)

    if element == CARBON_ELEMENT:
        return not is_strongly_polarized_carbon(atom)

    if element == SULFUR_ELEMENT:
        return is_hydrophobic_sulfur(atom)

    return False


# -----------------------------------------------------------------------------
# Complete hydrophobic eligibility assessment
# -----------------------------------------------------------------------------

def hydrophobic_exclusion_reasons(
    atom: AtomLike,
    *,
    role: HydrophobicAtomRole = HYDROPHOBIC_ROLE_UNKNOWN,
    maximum_absolute_partial_charge: Optional[Number] = None,
    maximum_polar_neighbors: Optional[int] = None,
    hydrophobic_residue_names: Optional[Iterable[str]] = None,
) -> Tuple[str, ...]:
    """
    Return reasons that prevent an atom from being hydrophobic.

    An empty tuple means that no exclusion criterion was detected.
    """

    if atom is None:
        return ("missing_atom",)

    normalized_role = validate_hydrophobic_atom_role(role)
    element = _safe_atom_element(atom)
    reasons: List[str] = []

    if is_hydrogen_atom(atom):
        reasons.append("hydrogen_atom")

    if is_water_atom(atom):
        reasons.append("water_atom")

    if element not in SUPPORTED_HYDROPHOBIC_ELEMENTS:
        reasons.append("unsupported_element")

    formal_charge = get_atom_formal_charge(atom)

    if (
        formal_charge is not None
        and abs(formal_charge)
        > DEFAULT_MAXIMUM_ABSOLUTE_FORMAL_CHARGE
    ):
        reasons.append("nonzero_formal_charge")

    partial_charge_limit = (
        get_default_maximum_absolute_partial_charge()
        if maximum_absolute_partial_charge is None
        else _nonnegative_float(
            maximum_absolute_partial_charge,
            name="maximum absolute partial charge",
        )
    )

    partial_charge = get_atom_partial_charge(atom)

    if (
        partial_charge is not None
        and abs(partial_charge) > partial_charge_limit
    ):
        reasons.append("excessive_partial_charge")

    polar_neighbor_limit = (
        get_default_maximum_polar_neighbors()
        if maximum_polar_neighbors is None
        else _nonnegative_integer(
            maximum_polar_neighbors,
            name="maximum polar neighbors",
        )
    )

    polar_neighbor_count = count_polar_neighbors(atom)

    if polar_neighbor_count > polar_neighbor_limit:
        reasons.append("too_many_polar_neighbors")

    if element == CARBON_ELEMENT:
        if is_carboxylate_carbon(atom):
            reasons.append("carboxylate_carbon")

        elif is_amide_carbon(atom):
            reasons.append("amide_carbonyl_carbon")

        elif is_carbonyl_carbon(atom):
            reasons.append("carbonyl_carbon")

        elif is_nitrile_carbon(atom):
            reasons.append("nitrile_carbon")

        elif is_carbon_bound_to_multiple_heteroatoms(atom):
            reasons.append("multiply_heteroatom_substituted_carbon")

        elif is_strongly_polarized_carbon(
            atom,
            maximum_absolute_partial_charge=(
                partial_charge_limit
            ),
        ):
            reasons.append("strongly_polarized_carbon")

    elif element == SULFUR_ELEMENT:
        if not is_hydrophobic_sulfur(atom):
            reasons.append("polar_or_unsupported_sulfur")

    elif element in HALOGEN_ELEMENTS:
        if not is_hydrophobic_halogen(atom):
            reasons.append("nonhydrophobic_halogen_environment")

    if (
        normalized_role == HYDROPHOBIC_ROLE_RECEPTOR
        and is_standard_protein_atom(atom)
        and not is_known_hydrophobic_protein_atom(
            atom,
            hydrophobic_residue_names=(
                hydrophobic_residue_names
            ),
        )
    ):
        reasons.append("nonhydrophobic_protein_atom")

    return tuple(dict.fromkeys(reasons))


def is_hydrophobic_atom(
    atom: AtomLike,
    *,
    role: HydrophobicAtomRole = HYDROPHOBIC_ROLE_UNKNOWN,
    receptor_atoms: Optional[Iterable[AtomLike]] = None,
    ligand_atoms: Optional[Iterable[AtomLike]] = None,
    maximum_absolute_partial_charge: Optional[Number] = None,
    maximum_polar_neighbors: Optional[int] = None,
    hydrophobic_residue_names: Optional[Iterable[str]] = None,
) -> bool:
    """
    Return whether an atom can participate in a hydrophobic interaction.

    Protein receptor atoms are checked against curated residue/atom-name
    definitions. Ligand atoms are evaluated from their local chemistry,
    charge, bonding and aromatic/aliphatic context.
    """

    if atom is None:
        return False

    inferred_role = infer_hydrophobic_atom_role(
        atom,
        receptor_atoms=receptor_atoms,
        ligand_atoms=ligand_atoms,
        default=role,
    )

    exclusions = hydrophobic_exclusion_reasons(
        atom,
        role=inferred_role,
        maximum_absolute_partial_charge=(
            maximum_absolute_partial_charge
        ),
        maximum_polar_neighbors=maximum_polar_neighbors,
        hydrophobic_residue_names=hydrophobic_residue_names,
    )

    if exclusions:
        return False

    element = _safe_atom_element(atom)

    if element == CARBON_ELEMENT:
        if (
            inferred_role == HYDROPHOBIC_ROLE_RECEPTOR
            and is_standard_protein_atom(atom)
        ):
            return is_known_hydrophobic_protein_atom(
                atom,
                hydrophobic_residue_names=(
                    hydrophobic_residue_names
                ),
            )

        return (
            is_aromatic_atom(atom)
            or is_aliphatic_atom(atom)
        )

    if element == SULFUR_ELEMENT:
        return is_hydrophobic_sulfur(atom)

    if element in HALOGEN_ELEMENTS:
        return is_hydrophobic_halogen(atom)

    return False


def classify_hydrophobic_atom_type(
    atom: AtomLike,
    *,
    role: HydrophobicAtomRole = HYDROPHOBIC_ROLE_UNKNOWN,
    receptor_atoms: Optional[Iterable[AtomLike]] = None,
    ligand_atoms: Optional[Iterable[AtomLike]] = None,
    maximum_absolute_partial_charge: Optional[Number] = None,
    maximum_polar_neighbors: Optional[int] = None,
    hydrophobic_residue_names: Optional[Iterable[str]] = None,
) -> HydrophobicAtomType:
    """
    Classify an atom as aromatic, aliphatic, mixed or non-hydrophobic.
    """

    hydrophobic = is_hydrophobic_atom(
        atom,
        role=role,
        receptor_atoms=receptor_atoms,
        ligand_atoms=ligand_atoms,
        maximum_absolute_partial_charge=(
            maximum_absolute_partial_charge
        ),
        maximum_polar_neighbors=maximum_polar_neighbors,
        hydrophobic_residue_names=hydrophobic_residue_names,
    )

    if not hydrophobic:
        return HYDROPHOBIC_ATOM_TYPE_NON_HYDROPHOBIC

    aromatic = is_aromatic_atom(atom)
    aliphatic = is_aliphatic_atom(atom)

    if aromatic and aliphatic:
        return HYDROPHOBIC_ATOM_TYPE_MIXED

    if aromatic:
        return HYDROPHOBIC_ATOM_TYPE_AROMATIC

    if aliphatic:
        return HYDROPHOBIC_ATOM_TYPE_ALIPHATIC

    element = _safe_atom_element(atom)

    if (
        element == SULFUR_ELEMENT
        or element in HALOGEN_ELEMENTS
    ):
        return HYDROPHOBIC_ATOM_TYPE_MIXED

    return HYDROPHOBIC_ATOM_TYPE_UNKNOWN


def perceive_hydrophobic_atom(
    atom: AtomLike,
    *,
    role: HydrophobicAtomRole = HYDROPHOBIC_ROLE_UNKNOWN,
    receptor_atoms: Optional[Iterable[AtomLike]] = None,
    ligand_atoms: Optional[Iterable[AtomLike]] = None,
    maximum_absolute_partial_charge: Optional[Number] = None,
    maximum_polar_neighbors: Optional[int] = None,
    hydrophobic_residue_names: Optional[Iterable[str]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> HydrophobicAtom:
    """
    Create a complete :class:`HydrophobicAtom` descriptor.

    The descriptor is returned even when the atom is non-hydrophobic,
    allowing rejected atoms and exclusion reasons to be inspected.
    """

    if atom is None:
        raise ValueError(
            "Cannot perceive a missing atom."
        )

    inferred_role = infer_hydrophobic_atom_role(
        atom,
        receptor_atoms=receptor_atoms,
        ligand_atoms=ligand_atoms,
        default=role,
    )

    exclusions = hydrophobic_exclusion_reasons(
        atom,
        role=inferred_role,
        maximum_absolute_partial_charge=(
            maximum_absolute_partial_charge
        ),
        maximum_polar_neighbors=maximum_polar_neighbors,
        hydrophobic_residue_names=hydrophobic_residue_names,
    )

    atom_type = classify_hydrophobic_atom_type(
        atom,
        role=inferred_role,
        receptor_atoms=receptor_atoms,
        ligand_atoms=ligand_atoms,
        maximum_absolute_partial_charge=(
            maximum_absolute_partial_charge
        ),
        maximum_polar_neighbors=maximum_polar_neighbors,
        hydrophobic_residue_names=hydrophobic_residue_names,
    )

    aromatic = is_aromatic_atom(atom)
    aliphatic = is_aliphatic_atom(atom)
    hydrophobic = not exclusions and (
        atom_type
        != HYDROPHOBIC_ATOM_TYPE_NON_HYDROPHOBIC
    )

    supplied_metadata = (
        {} if metadata is None else dict(metadata)
    )

    supplied_metadata.update(
        {
            "exclusion_reasons": exclusions,
            "residue_name": get_atom_residue_name(atom),
            "atom_name": _normalize_atom_name(atom),
            "is_protein_atom": is_standard_protein_atom(atom),
            "is_known_hydrophobic_protein_atom": (
                is_known_hydrophobic_protein_atom(
                    atom,
                    hydrophobic_residue_names=(
                        hydrophobic_residue_names
                    ),
                )
                if is_standard_protein_atom(atom)
                else False
            ),
            "is_carbonyl_carbon": is_carbonyl_carbon(atom),
            "is_carboxylate_carbon": (
                is_carboxylate_carbon(atom)
            ),
            "is_amide_carbon": is_amide_carbon(atom),
            "is_nitrile_carbon": is_nitrile_carbon(atom),
            "is_hydrophobic_sulfur": (
                is_hydrophobic_sulfur(atom)
                if _safe_atom_element(atom)
                == SULFUR_ELEMENT
                else False
            ),
            "is_hydrophobic_halogen": (
                is_hydrophobic_halogen(atom)
                if _safe_atom_element(atom)
                in HALOGEN_ELEMENTS
                else False
            ),
        }
    )

    atomic_number = None

    try:
        atomic_number = get_atom_atomic_number(atom)
    except Exception:
        atomic_number = None

    return HydrophobicAtom(
        atom=atom,
        role=inferred_role,
        atom_type=atom_type,
        is_hydrophobic=hydrophobic,
        is_aromatic=aromatic,
        is_aliphatic=aliphatic,
        element=_safe_atom_element(atom),
        atomic_number=atomic_number,
        atom_index=_safe_atom_index(atom),
        identifier=_safe_atom_identifier(atom),
        residue=_safe_atom_residue(atom),
        residue_key=_safe_residue_key_from_atom(atom),
        formal_charge=get_atom_formal_charge(atom),
        partial_charge=get_atom_partial_charge(atom),
        polar_neighbor_count=count_polar_neighbors(atom),
        heavy_neighbor_count=count_heavy_neighbors(atom),
        metadata=supplied_metadata,
    )


def perceive_hydrophobic_atoms(
    atoms: Iterable[AtomLike],
    *,
    role: HydrophobicAtomRole = HYDROPHOBIC_ROLE_UNKNOWN,
    receptor_atoms: Optional[Iterable[AtomLike]] = None,
    ligand_atoms: Optional[Iterable[AtomLike]] = None,
    include_non_hydrophobic: bool = False,
    maximum_absolute_partial_charge: Optional[Number] = None,
    maximum_polar_neighbors: Optional[int] = None,
    hydrophobic_residue_names: Optional[Iterable[str]] = None,
) -> Tuple[HydrophobicAtom, ...]:
    """
    Perceive hydrophobic character for a collection of atoms.

    Parameters
    ----------
    atoms
        Atom collection to analyze.
    role
        Default molecular role assigned to the atoms.
    include_non_hydrophobic
        Retain rejected descriptors when ``True``.
    """

    normalized_role = validate_hydrophobic_atom_role(role)

    atom_tuple = tuple(atoms)

    receptor_atom_tuple = (
        tuple(receptor_atoms)
        if receptor_atoms is not None
        else (
            atom_tuple
            if normalized_role == HYDROPHOBIC_ROLE_RECEPTOR
            else None
        )
    )

    ligand_atom_tuple = (
        tuple(ligand_atoms)
        if ligand_atoms is not None
        else (
            atom_tuple
            if normalized_role == HYDROPHOBIC_ROLE_LIGAND
            else None
        )
    )

    descriptors: List[HydrophobicAtom] = []
    seen_atom_ids: Set[int] = set()

    for atom in atom_tuple:
        if atom is None:
            continue

        atom_identity = id(atom)

        if atom_identity in seen_atom_ids:
            continue

        seen_atom_ids.add(atom_identity)

        descriptor = perceive_hydrophobic_atom(
            atom,
            role=normalized_role,
            receptor_atoms=receptor_atom_tuple,
            ligand_atoms=ligand_atom_tuple,
            maximum_absolute_partial_charge=(
                maximum_absolute_partial_charge
            ),
            maximum_polar_neighbors=maximum_polar_neighbors,
            hydrophobic_residue_names=(
                hydrophobic_residue_names
            ),
        )

        if (
            include_non_hydrophobic
            or descriptor.is_hydrophobic
        ):
            descriptors.append(descriptor)

    return tuple(descriptors)


def filter_hydrophobic_atoms(
    atoms: Iterable[AtomLike],
    *,
    role: HydrophobicAtomRole = HYDROPHOBIC_ROLE_UNKNOWN,
    receptor_atoms: Optional[Iterable[AtomLike]] = None,
    ligand_atoms: Optional[Iterable[AtomLike]] = None,
    maximum_absolute_partial_charge: Optional[Number] = None,
    maximum_polar_neighbors: Optional[int] = None,
    hydrophobic_residue_names: Optional[Iterable[str]] = None,
) -> Tuple[AtomLike, ...]:
    """
    Return only atoms accepted as hydrophobic.
    """

    descriptors = perceive_hydrophobic_atoms(
        atoms,
        role=role,
        receptor_atoms=receptor_atoms,
        ligand_atoms=ligand_atoms,
        include_non_hydrophobic=False,
        maximum_absolute_partial_charge=(
            maximum_absolute_partial_charge
        ),
        maximum_polar_neighbors=maximum_polar_neighbors,
        hydrophobic_residue_names=hydrophobic_residue_names,
    )

    return tuple(
        descriptor.atom
        for descriptor in descriptors
    )


def partition_hydrophobic_atoms(
    atoms: Iterable[AtomLike],
    *,
    role: HydrophobicAtomRole = HYDROPHOBIC_ROLE_UNKNOWN,
    receptor_atoms: Optional[Iterable[AtomLike]] = None,
    ligand_atoms: Optional[Iterable[AtomLike]] = None,
    maximum_absolute_partial_charge: Optional[Number] = None,
    maximum_polar_neighbors: Optional[int] = None,
    hydrophobic_residue_names: Optional[Iterable[str]] = None,
) -> Tuple[
    Tuple[HydrophobicAtom, ...],
    Tuple[HydrophobicAtom, ...],
]:
    """
    Partition atom descriptors into accepted and rejected collections.

    Returns
    -------
    accepted, rejected
        Two immutable descriptor tuples.
    """

    descriptors = perceive_hydrophobic_atoms(
        atoms,
        role=role,
        receptor_atoms=receptor_atoms,
        ligand_atoms=ligand_atoms,
        include_non_hydrophobic=True,
        maximum_absolute_partial_charge=(
            maximum_absolute_partial_charge
        ),
        maximum_polar_neighbors=maximum_polar_neighbors,
        hydrophobic_residue_names=hydrophobic_residue_names,
    )

    accepted = tuple(
        descriptor
        for descriptor in descriptors
        if descriptor.is_hydrophobic
    )

    rejected = tuple(
        descriptor
        for descriptor in descriptors
        if not descriptor.is_hydrophobic
    )

    return accepted, rejected


# -----------------------------------------------------------------------------
# Section 4 public names
# -----------------------------------------------------------------------------

_SECTION_4_PUBLIC_NAMES: Final[Tuple[str, ...]] = (
    # Generic accessors
    "get_atom_residue_name",
    "get_atom_formal_charge",
    "get_atom_partial_charge",
    "get_bonded_neighbors",
    "get_heavy_neighbors",
    "get_polar_neighbors",
    "count_heavy_neighbors",
    "count_polar_neighbors",
    "get_bond_order",

    # Molecular context
    "is_standard_protein_atom",
    "is_water_atom",
    "is_nucleic_acid_atom",
    "infer_hydrophobic_atom_role",

    # Aromatic and aliphatic perception
    "is_protein_aromatic_atom",
    "is_aromatic_atom",
    "is_protein_aliphatic_atom",
    "is_aliphatic_atom",

    # Polarized-carbon exclusions
    "is_double_bonded_to_element",
    "is_carbonyl_carbon",
    "is_carboxylate_carbon",
    "is_amide_carbon",
    "is_nitrile_carbon",
    "is_carbon_bound_to_multiple_heteroatoms",
    "is_strongly_polarized_carbon",

    # Conditional hydrophobic elements
    "is_hydrophobic_sulfur",
    "is_hydrophobic_halogen",

    # Protein definitions
    "is_known_hydrophobic_protein_atom",

    # Complete perception
    "hydrophobic_exclusion_reasons",
    "is_hydrophobic_atom",
    "classify_hydrophobic_atom_type",
    "perceive_hydrophobic_atom",
    "perceive_hydrophobic_atoms",
    "filter_hydrophobic_atoms",
    "partition_hydrophobic_atoms",
)

for public_name in _SECTION_4_PUBLIC_NAMES:
    if public_name not in __all__:
        __all__.append(public_name)


# =============================================================================
# End of Section 4
# =============================================================================

# =============================================================================
# Section 5 — Atom collections and preparation
# =============================================================================


# -----------------------------------------------------------------------------
# Collection-normalization constants
# -----------------------------------------------------------------------------

_ATOM_COLLECTION_ATTRIBUTE_NAMES: Final[Tuple[str, ...]] = (
    "atoms",
    "all_atoms",
    "allAtoms",
    "atom_collection",
    "atomCollection",
)

_RESIDUE_COLLECTION_ATTRIBUTE_NAMES: Final[Tuple[str, ...]] = (
    "residues",
    "all_residues",
    "allResidues",
    "residue_collection",
    "residueCollection",
)

_STRUCTURE_COLLECTION_ATTRIBUTE_NAMES: Final[Tuple[str, ...]] = (
    "structures",
    "models",
    "atomic_structures",
    "atomicStructures",
    "children",
)

_RECEPTOR_ATTRIBUTE_NAMES: Final[Tuple[str, ...]] = (
    "receptor",
    "receptor_model",
    "receptorModel",
    "protein",
    "protein_model",
    "proteinModel",
    "target",
    "target_model",
    "targetModel",
)

_LIGAND_ATTRIBUTE_NAMES: Final[Tuple[str, ...]] = (
    "ligand",
    "ligand_model",
    "ligandModel",
    "pose",
    "pose_model",
    "poseModel",
    "docked_ligand",
    "dockedLigand",
    "compound",
)

_DELETED_ATTRIBUTE_NAMES: Final[Tuple[str, ...]] = (
    "deleted",
    "is_deleted",
    "isDeleted",
)

_DISPLAY_ATTRIBUTE_NAMES: Final[Tuple[str, ...]] = (
    "display",
    "displayed",
    "shown",
    "visible",
    "is_visible",
    "isVisible",
)

_ALTLOC_ATTRIBUTE_NAMES: Final[Tuple[str, ...]] = (
    "alt_loc",
    "altLoc",
    "altloc",
    "alternate_location",
    "alternateLocation",
)

_OCCUPANCY_ATTRIBUTE_NAMES: Final[Tuple[str, ...]] = (
    "occupancy",
    "occ",
)

_DEFAULT_ALLOWED_ALTLOCS: Final[FrozenSet[str]] = frozenset(
    {
        "",
        "A",
        ".",
        "?",
    }
)


# -----------------------------------------------------------------------------
# Prepared collection dataclass
# -----------------------------------------------------------------------------

@dataclass(
    frozen=True,
    slots=True,
)
class HydrophobicAtomCollections:
    """
    Prepared receptor and ligand atom collections.

    This object records both the original normalized atom collections and
    the hydrophobic descriptors produced from them.

    Parameters
    ----------
    receptor_atoms
        Valid, normalized and deduplicated receptor atoms.
    ligand_atoms
        Valid, normalized and deduplicated ligand atoms.
    receptor_hydrophobic_atoms
        Receptor descriptors accepted as hydrophobic.
    ligand_hydrophobic_atoms
        Ligand descriptors accepted as hydrophobic.
    rejected_receptor_atoms
        Receptor descriptors rejected during chemical perception.
    rejected_ligand_atoms
        Ligand descriptors rejected during chemical perception.
    receptor_source
        Original receptor-side object.
    ligand_source
        Original ligand-side object.
    metadata
        Preparation metadata and collection counts.
    """

    receptor_atoms: Sequence[AtomLike] = field(
        default_factory=tuple
    )

    ligand_atoms: Sequence[AtomLike] = field(
        default_factory=tuple
    )

    receptor_hydrophobic_atoms: Sequence[HydrophobicAtom] = field(
        default_factory=tuple
    )

    ligand_hydrophobic_atoms: Sequence[HydrophobicAtom] = field(
        default_factory=tuple
    )

    rejected_receptor_atoms: Sequence[HydrophobicAtom] = field(
        default_factory=tuple
    )

    rejected_ligand_atoms: Sequence[HydrophobicAtom] = field(
        default_factory=tuple
    )

    receptor_source: Optional[Any] = field(
        default=None,
        repr=False,
        compare=False,
    )

    ligand_source: Optional[Any] = field(
        default=None,
        repr=False,
        compare=False,
    )

    metadata: Mapping[str, Any] = field(
        default_factory=lambda: _EMPTY_METADATA,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        """Validate and freeze all prepared collections."""

        receptor_atoms = tuple(self.receptor_atoms)
        ligand_atoms = tuple(self.ligand_atoms)

        receptor_descriptors = tuple(
            self.receptor_hydrophobic_atoms
        )

        ligand_descriptors = tuple(
            self.ligand_hydrophobic_atoms
        )

        rejected_receptor = tuple(
            self.rejected_receptor_atoms
        )

        rejected_ligand = tuple(
            self.rejected_ligand_atoms
        )

        for collection_name, descriptors in (
            (
                "receptor_hydrophobic_atoms",
                receptor_descriptors,
            ),
            (
                "ligand_hydrophobic_atoms",
                ligand_descriptors,
            ),
            (
                "rejected_receptor_atoms",
                rejected_receptor,
            ),
            (
                "rejected_ligand_atoms",
                rejected_ligand,
            ),
        ):
            for index, descriptor in enumerate(descriptors):
                if not isinstance(
                    descriptor,
                    HydrophobicAtom,
                ):
                    raise TypeError(
                        f"{collection_name}[{index}] must be a "
                        "HydrophobicAtom instance."
                    )

        for descriptor in receptor_descriptors:
            if not descriptor.is_hydrophobic:
                raise ValueError(
                    "receptor_hydrophobic_atoms cannot contain "
                    "rejected descriptors."
                )

            if descriptor.role not in {
                HYDROPHOBIC_ROLE_RECEPTOR,
                HYDROPHOBIC_ROLE_UNKNOWN,
            }:
                raise ValueError(
                    "A receptor hydrophobic descriptor cannot have "
                    f"role={descriptor.role!r}."
                )

        for descriptor in ligand_descriptors:
            if not descriptor.is_hydrophobic:
                raise ValueError(
                    "ligand_hydrophobic_atoms cannot contain "
                    "rejected descriptors."
                )

            if descriptor.role not in {
                HYDROPHOBIC_ROLE_LIGAND,
                HYDROPHOBIC_ROLE_UNKNOWN,
            }:
                raise ValueError(
                    "A ligand hydrophobic descriptor cannot have "
                    f"role={descriptor.role!r}."
                )

        for descriptor in (
            *rejected_receptor,
            *rejected_ligand,
        ):
            if descriptor.is_hydrophobic:
                raise ValueError(
                    "Rejected collections cannot contain accepted "
                    "hydrophobic descriptors."
                )

        receptor_atom_ids = {
            id(atom)
            for atom in receptor_atoms
        }

        ligand_atom_ids = {
            id(atom)
            for atom in ligand_atoms
        }

        for descriptor in (
            *receptor_descriptors,
            *rejected_receptor,
        ):
            if id(descriptor.atom) not in receptor_atom_ids:
                raise ValueError(
                    "A receptor descriptor references an atom absent "
                    "from receptor_atoms."
                )

        for descriptor in (
            *ligand_descriptors,
            *rejected_ligand,
        ):
            if id(descriptor.atom) not in ligand_atom_ids:
                raise ValueError(
                    "A ligand descriptor references an atom absent "
                    "from ligand_atoms."
                )

        object.__setattr__(
            self,
            "receptor_atoms",
            receptor_atoms,
        )

        object.__setattr__(
            self,
            "ligand_atoms",
            ligand_atoms,
        )

        object.__setattr__(
            self,
            "receptor_hydrophobic_atoms",
            receptor_descriptors,
        )

        object.__setattr__(
            self,
            "ligand_hydrophobic_atoms",
            ligand_descriptors,
        )

        object.__setattr__(
            self,
            "rejected_receptor_atoms",
            rejected_receptor,
        )

        object.__setattr__(
            self,
            "rejected_ligand_atoms",
            rejected_ligand,
        )

        object.__setattr__(
            self,
            "metadata",
            _freeze_metadata(self.metadata),
        )

    @property
    def receptor_atom_count(self) -> int:
        """Return the number of prepared receptor atoms."""

        return len(self.receptor_atoms)

    @property
    def ligand_atom_count(self) -> int:
        """Return the number of prepared ligand atoms."""

        return len(self.ligand_atoms)

    @property
    def receptor_hydrophobic_atom_count(self) -> int:
        """Return the number of accepted receptor descriptors."""

        return len(self.receptor_hydrophobic_atoms)

    @property
    def ligand_hydrophobic_atom_count(self) -> int:
        """Return the number of accepted ligand descriptors."""

        return len(self.ligand_hydrophobic_atoms)

    @property
    def rejected_receptor_atom_count(self) -> int:
        """Return the number of rejected receptor descriptors."""

        return len(self.rejected_receptor_atoms)

    @property
    def rejected_ligand_atom_count(self) -> int:
        """Return the number of rejected ligand descriptors."""

        return len(self.rejected_ligand_atoms)

    @property
    def has_receptor_atoms(self) -> bool:
        """Return whether receptor atoms are available."""

        return bool(self.receptor_atoms)

    @property
    def has_ligand_atoms(self) -> bool:
        """Return whether ligand atoms are available."""

        return bool(self.ligand_atoms)

    @property
    def is_ready(self) -> bool:
        """
        Return whether both sides contain hydrophobic atoms.

        A collection may be valid but not ready for interaction detection
        when one side contains no chemically accepted hydrophobic atoms.
        """

        return bool(
            self.receptor_hydrophobic_atoms
            and self.ligand_hydrophobic_atoms
        )

    @property
    def receptor_hydrophobic_fraction(
        self,
    ) -> np.float64:
        """Return the fraction of receptor atoms accepted as hydrophobic."""

        if not self.receptor_atoms:
            return np.float64(0.0)

        return np.float64(
            len(self.receptor_hydrophobic_atoms)
            / len(self.receptor_atoms)
        )

    @property
    def ligand_hydrophobic_fraction(
        self,
    ) -> np.float64:
        """Return the fraction of ligand atoms accepted as hydrophobic."""

        if not self.ligand_atoms:
            return np.float64(0.0)

        return np.float64(
            len(self.ligand_hydrophobic_atoms)
            / len(self.ligand_atoms)
        )

    def to_dict(
        self,
        *,
        include_atoms: bool = False,
        include_rejected: bool = True,
        include_descriptors: bool = True,
    ) -> Dict[str, Any]:
        """Serialize the prepared collections."""

        result: Dict[str, Any] = {
            "receptor_atom_count": self.receptor_atom_count,
            "ligand_atom_count": self.ligand_atom_count,
            "receptor_hydrophobic_atom_count": (
                self.receptor_hydrophobic_atom_count
            ),
            "ligand_hydrophobic_atom_count": (
                self.ligand_hydrophobic_atom_count
            ),
            "rejected_receptor_atom_count": (
                self.rejected_receptor_atom_count
            ),
            "rejected_ligand_atom_count": (
                self.rejected_ligand_atom_count
            ),
            "receptor_hydrophobic_fraction": float(
                self.receptor_hydrophobic_fraction
            ),
            "ligand_hydrophobic_fraction": float(
                self.ligand_hydrophobic_fraction
            ),
            "has_receptor_atoms": self.has_receptor_atoms,
            "has_ligand_atoms": self.has_ligand_atoms,
            "is_ready": self.is_ready,
            "metadata": dict(self.metadata),
        }

        if include_descriptors:
            result["receptor_hydrophobic_atoms"] = [
                descriptor.to_dict(
                    include_atom=include_atoms,
                    include_residue=include_atoms,
                )
                for descriptor
                in self.receptor_hydrophobic_atoms
            ]

            result["ligand_hydrophobic_atoms"] = [
                descriptor.to_dict(
                    include_atom=include_atoms,
                    include_residue=include_atoms,
                )
                for descriptor
                in self.ligand_hydrophobic_atoms
            ]

            if include_rejected:
                result["rejected_receptor_atoms"] = [
                    descriptor.to_dict(
                        include_atom=include_atoms,
                        include_residue=include_atoms,
                    )
                    for descriptor
                    in self.rejected_receptor_atoms
                ]

                result["rejected_ligand_atoms"] = [
                    descriptor.to_dict(
                        include_atom=include_atoms,
                        include_residue=include_atoms,
                    )
                    for descriptor
                    in self.rejected_ligand_atoms
                ]

        if include_atoms:
            result["receptor_atoms"] = self.receptor_atoms
            result["ligand_atoms"] = self.ligand_atoms

        return result


# -----------------------------------------------------------------------------
# Low-level collection inspection
# -----------------------------------------------------------------------------

def _is_scalar_collection_value(
    value: Any,
) -> bool:
    """
    Return whether a value should not be expanded as an atom collection.
    """

    return isinstance(
        value,
        (
            str,
            bytes,
            bytearray,
            int,
            float,
            complex,
            bool,
            np.number,
        ),
    )


def _is_mapping_like(
    value: Any,
) -> bool:
    """Return whether a value behaves as a mapping."""

    return isinstance(value, Mapping)


def _mapping_values(
    mapping: Mapping[Any, Any],
) -> Tuple[Any, ...]:
    """Return mapping values as an immutable tuple."""

    try:
        return tuple(mapping.values())
    except Exception:
        return ()


def _safe_iter_collection(
    value: Any,
) -> Tuple[Any, ...]:
    """
    Convert a collection-like object to a tuple safely.

    Atom-like objects are intentionally not expanded through arbitrary
    iteration because some external molecular objects expose surprising
    iterator behavior.
    """

    if value is None:
        return ()

    if _is_scalar_collection_value(value):
        return ()

    if _is_mapping_like(value):
        return _mapping_values(value)

    if isinstance(value, tuple):
        return value

    if isinstance(value, list):
        return tuple(value)

    if isinstance(value, set):
        return tuple(value)

    if isinstance(value, frozenset):
        return tuple(value)

    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return ()

        try:
            return tuple(value.tolist())
        except Exception:
            return tuple(value)

    try:
        return tuple(iter(value))
    except (TypeError, RuntimeError, ValueError):
        return ()


def _looks_like_atom(
    value: Any,
) -> bool:
    """
    Return whether an object is structurally compatible with an atom.

    The canonical ``contacts.is_atom_like`` helper is attempted first.
    Fallback detection supports simple test doubles with coordinates and
    element information.
    """

    if value is None:
        return False

    try:
        if is_atom_like(value):
            return True
    except Exception:
        pass

    coordinate = _safe_getattr(
        value,
        (
            "coord",
            "coords",
            "coordinate",
            "coordinates",
            "scene_coord",
            "sceneCoord",
        ),
        default=None,
    )

    element = _safe_getattr(
        value,
        (
            "element",
            "atomic_number",
            "atomicNumber",
            "element_name",
            "elementName",
            "symbol",
        ),
        default=None,
    )

    name = _safe_getattr(
        value,
        (
            "name",
            "atom_name",
            "atomName",
        ),
        default=None,
    )

    return (
        coordinate is not None
        and (
            element is not None
            or name is not None
        )
    )


def _looks_like_residue(
    value: Any,
) -> bool:
    """Return whether an object appears to be a residue."""

    if value is None or _looks_like_atom(value):
        return False

    atoms = _safe_getattr(
        value,
        _ATOM_COLLECTION_ATTRIBUTE_NAMES,
        default=None,
    )

    name = _safe_getattr(
        value,
        (
            "name",
            "resname",
            "residue_name",
        ),
        default=None,
    )

    return atoms is not None and name is not None


def _looks_like_structure(
    value: Any,
) -> bool:
    """Return whether an object appears to contain molecular atoms."""

    if (
        value is None
        or _looks_like_atom(value)
        or _looks_like_residue(value)
    ):
        return False

    return (
        _safe_getattr(
            value,
            _ATOM_COLLECTION_ATTRIBUTE_NAMES,
            default=None,
        )
        is not None
        or _safe_getattr(
            value,
            _RESIDUE_COLLECTION_ATTRIBUTE_NAMES,
            default=None,
        )
        is not None
    )


def _is_deleted_object(
    value: Any,
) -> bool:
    """
    Return whether an external molecular object is marked as deleted.
    """

    deleted = _safe_boolean_attribute(
        value,
        _DELETED_ATTRIBUTE_NAMES,
    )

    return bool(deleted)


def _is_displayed_object(
    value: Any,
) -> Optional[bool]:
    """
    Return an object's display status when available.
    """

    return _safe_boolean_attribute(
        value,
        _DISPLAY_ATTRIBUTE_NAMES,
    )


def _get_atom_altloc(
    atom: AtomLike,
) -> str:
    """Return a normalized alternate-location identifier."""

    value = _safe_getattr(
        atom,
        _ALTLOC_ATTRIBUTE_NAMES,
        default="",
    )

    if value is None:
        return ""

    return str(value).strip().upper()


def _get_atom_occupancy(
    atom: AtomLike,
) -> Optional[np.float64]:
    """Return an atom occupancy value when available."""

    value = _safe_getattr(
        atom,
        _OCCUPANCY_ATTRIBUTE_NAMES,
        default=None,
    )

    if value is None:
        return None

    try:
        occupancy = _finite_float(
            value,
            name="atom occupancy",
        )
    except (TypeError, ValueError):
        return None

    return occupancy


# -----------------------------------------------------------------------------
# Recursive atom extraction
# -----------------------------------------------------------------------------

def _extract_atoms_recursive(
    source: Any,
    *,
    maximum_depth: int,
    current_depth: int,
    seen_objects: Set[int],
) -> List[AtomLike]:
    """
    Recursively extract atoms from molecular collection objects.
    """

    if source is None:
        return []

    if current_depth > maximum_depth:
        return []

    source_identity = id(source)

    if source_identity in seen_objects:
        return []

    seen_objects.add(source_identity)

    if _looks_like_atom(source):
        return [source]

    extracted: List[AtomLike] = []

    # Direct atom container, common to ChimeraX structures and residues.
    direct_atoms = _safe_getattr(
        source,
        _ATOM_COLLECTION_ATTRIBUTE_NAMES,
        default=None,
    )

    if direct_atoms is not None and direct_atoms is not source:
        direct_values = _safe_iter_collection(direct_atoms)

        if direct_values:
            for candidate in direct_values:
                extracted.extend(
                    _extract_atoms_recursive(
                        candidate,
                        maximum_depth=maximum_depth,
                        current_depth=current_depth + 1,
                        seen_objects=seen_objects,
                    )
                )

            if extracted:
                return extracted

    # Residue containers.
    residues = _safe_getattr(
        source,
        _RESIDUE_COLLECTION_ATTRIBUTE_NAMES,
        default=None,
    )

    if residues is not None and residues is not source:
        for residue in _safe_iter_collection(residues):
            extracted.extend(
                _extract_atoms_recursive(
                    residue,
                    maximum_depth=maximum_depth,
                    current_depth=current_depth + 1,
                    seen_objects=seen_objects,
                )
            )

        if extracted:
            return extracted

    # Collections of structures/models.
    structures = _safe_getattr(
        source,
        _STRUCTURE_COLLECTION_ATTRIBUTE_NAMES,
        default=None,
    )

    if structures is not None and structures is not source:
        for structure in _safe_iter_collection(structures):
            extracted.extend(
                _extract_atoms_recursive(
                    structure,
                    maximum_depth=maximum_depth,
                    current_depth=current_depth + 1,
                    seen_objects=seen_objects,
                )
            )

        if extracted:
            return extracted

    # Generic Python sequences, generators and mappings.
    for candidate in _safe_iter_collection(source):
        if candidate is source:
            continue

        extracted.extend(
            _extract_atoms_recursive(
                candidate,
                maximum_depth=maximum_depth,
                current_depth=current_depth + 1,
                seen_objects=seen_objects,
            )
        )

    return extracted


def normalize_atom_collection(
    source: Any,
    *,
    allow_empty: bool = True,
    maximum_depth: int = 8,
) -> Tuple[AtomLike, ...]:
    """
    Normalize an atom, molecular object or collection into an atom tuple.

    Supported input forms include:

    - a single atom;
    - a Python sequence or generator of atoms;
    - a residue or sequence of residues;
    - an atomic structure exposing ``atoms``;
    - a collection of molecular structures;
    - mappings whose values contain atoms or structures;
    - lightweight synthetic objects used in self-tests.

    Parameters
    ----------
    source
        Object from which atoms will be extracted.
    allow_empty
        Return an empty tuple when no atoms are found. When ``False``, a
        ``ValueError`` is raised instead.
    maximum_depth
        Maximum recursive nesting depth.
    """

    maximum_depth = _nonnegative_integer(
        maximum_depth,
        name="maximum extraction depth",
    )

    if maximum_depth == 0:
        maximum_depth = 1

    atoms = _extract_atoms_recursive(
        source,
        maximum_depth=maximum_depth,
        current_depth=0,
        seen_objects=set(),
    )

    normalized = tuple(
        atom
        for atom in atoms
        if atom is not None
    )

    if not normalized and not allow_empty:
        raise ValueError(
            "No atom-like objects could be extracted from the "
            "provided collection."
        )

    return normalized


# -----------------------------------------------------------------------------
# Collection deduplication
# -----------------------------------------------------------------------------

def atom_deduplication_key(
    atom: AtomLike,
    *,
    strategy: Literal[
        "identity",
        "identifier",
        "index",
        "auto",
    ] = "auto",
) -> Tuple[Any, ...]:
    """
    Create a stable atom deduplication key.

    ``identity`` is always collision-safe during one Python session.
    ``identifier`` and ``index`` may merge equivalent wrappers referring
    to the same external atom. ``auto`` prefers explicit identifiers and
    falls back to object identity.
    """

    normalized_strategy = str(strategy).strip().lower()

    if normalized_strategy not in {
        "identity",
        "identifier",
        "index",
        "auto",
    }:
        raise ValueError(
            "strategy must be 'identity', 'identifier', 'index' "
            "or 'auto'."
        )

    if normalized_strategy == "identity":
        return (
            "identity",
            id(atom),
        )

    structure = None

    try:
        structure = get_atom_structure(atom)
    except Exception:
        structure = _safe_getattr(
            atom,
            (
                "structure",
                "model",
                "molecule",
            ),
            default=None,
        )

    structure_identity = (
        id(structure)
        if structure is not None
        else None
    )

    atom_identifier = _safe_atom_identifier(atom)
    atom_index = _safe_atom_index(atom)

    if normalized_strategy == "identifier":
        if atom_identifier is None:
            return (
                "identity",
                id(atom),
            )

        return (
            "identifier",
            structure_identity,
            atom_identifier,
        )

    if normalized_strategy == "index":
        if atom_index is None:
            return (
                "identity",
                id(atom),
            )

        return (
            "index",
            structure_identity,
            atom_index,
        )

    if atom_identifier is not None:
        return (
            "identifier",
            structure_identity,
            atom_identifier,
        )

    if atom_index is not None:
        return (
            "index",
            structure_identity,
            atom_index,
        )

    return (
        "identity",
        id(atom),
    )


def deduplicate_atoms(
    atoms: Iterable[AtomLike],
    *,
    strategy: Literal[
        "identity",
        "identifier",
        "index",
        "auto",
    ] = "auto",
) -> Tuple[AtomLike, ...]:
    """
    Remove duplicate atoms while preserving the original order.
    """

    unique_atoms: List[AtomLike] = []
    seen_keys: Set[Tuple[Any, ...]] = set()

    for atom in atoms:
        if atom is None:
            continue

        key = atom_deduplication_key(
            atom,
            strategy=strategy,
        )

        if key in seen_keys:
            continue

        seen_keys.add(key)
        unique_atoms.append(atom)

    return tuple(unique_atoms)


def descriptor_deduplication_key(
    descriptor: HydrophobicAtom,
) -> Tuple[Any, ...]:
    """Create a deduplication key for a hydrophobic descriptor."""

    if not isinstance(
        descriptor,
        HydrophobicAtom,
    ):
        raise TypeError(
            "descriptor must be a HydrophobicAtom instance."
        )

    return atom_deduplication_key(
        descriptor.atom,
        strategy="auto",
    )


def deduplicate_hydrophobic_descriptors(
    descriptors: Iterable[HydrophobicAtom],
    *,
    prefer_hydrophobic: bool = True,
) -> Tuple[HydrophobicAtom, ...]:
    """
    Remove duplicate hydrophobic descriptors.

    When duplicate descriptors disagree and ``prefer_hydrophobic=True``,
    an accepted descriptor replaces a rejected descriptor for the same
    atom.
    """

    ordered_keys: List[Tuple[Any, ...]] = []
    descriptor_map: Dict[
        Tuple[Any, ...],
        HydrophobicAtom,
    ] = {}

    for descriptor in descriptors:
        if not isinstance(
            descriptor,
            HydrophobicAtom,
        ):
            raise TypeError(
                "All descriptors must be HydrophobicAtom instances."
            )

        key = descriptor_deduplication_key(
            descriptor
        )

        existing = descriptor_map.get(key)

        if existing is None:
            ordered_keys.append(key)
            descriptor_map[key] = descriptor
            continue

        if (
            prefer_hydrophobic
            and descriptor.is_hydrophobic
            and not existing.is_hydrophobic
        ):
            descriptor_map[key] = descriptor

    return tuple(
        descriptor_map[key]
        for key in ordered_keys
    )


# -----------------------------------------------------------------------------
# Atom validation
# -----------------------------------------------------------------------------

def validate_hydrophobic_atom_candidate(
    atom: AtomLike,
    *,
    require_coordinates: bool = True,
    require_element: bool = True,
    allow_deleted: bool = False,
) -> bool:
    """
    Validate one atom before hydrophobic chemical perception.

    Returns
    -------
    bool
        ``True`` when the atom can safely proceed to perception.

    Notes
    -----
    This function does not determine hydrophobicity. It validates only the
    basic integrity of the atom object.
    """

    if atom is None:
        return False

    if not _looks_like_atom(atom):
        return False

    if (
        not allow_deleted
        and _is_deleted_object(atom)
    ):
        return False

    try:
        validation_result = validate_atom(atom)

        if validation_result is False:
            return False

    except Exception:
        # Synthetic atoms may not satisfy all canonical contacts.py
        # requirements. Explicit checks below remain authoritative.
        pass

    if require_coordinates:
        try:
            coordinate = get_atom_coordinate(atom)
        except Exception:
            coordinate = _safe_getattr(
                atom,
                (
                    "coord",
                    "coords",
                    "coordinate",
                    "coordinates",
                    "scene_coord",
                    "sceneCoord",
                ),
                default=None,
            )

        if coordinate is None:
            return False

        try:
            coordinate_array = np.asarray(
                coordinate,
                dtype=np.float64,
            ).reshape(-1)
        except (
            TypeError,
            ValueError,
            OverflowError,
        ):
            return False

        if (
            coordinate_array.size != 3
            or not np.all(np.isfinite(coordinate_array))
        ):
            return False

    if require_element:
        element = _safe_atom_element(atom)

        if element in {
            "",
            "UNKNOWN",
            "NONE",
        }:
            atomic_number = None

            try:
                atomic_number = get_atom_atomic_number(atom)
            except Exception:
                atomic_number = _safe_getattr(
                    atom,
                    (
                        "atomic_number",
                        "atomicNumber",
                    ),
                    default=None,
                )

            if atomic_number is None:
                return False

            try:
                atomic_number_value = int(atomic_number)
            except (
                TypeError,
                ValueError,
                OverflowError,
            ):
                return False

            if atomic_number_value <= 0:
                return False

    return True


def validate_prepared_atom_collection(
    atoms: Iterable[AtomLike],
    *,
    collection_name: str = "atom collection",
    allow_empty: bool = True,
    require_coordinates: bool = True,
    require_element: bool = True,
    allow_deleted: bool = False,
    raise_on_invalid: bool = True,
) -> Tuple[AtomLike, ...]:
    """
    Validate every atom in a normalized collection.

    Invalid entries are either rejected with ``ValueError`` or silently
    omitted, depending on ``raise_on_invalid``.
    """

    normalized_name = _normalize_required_string(
        collection_name,
        name="collection name",
    )

    atom_tuple = tuple(atoms)
    valid_atoms: List[AtomLike] = []
    invalid_indices: List[int] = []

    for index, atom in enumerate(atom_tuple):
        valid = validate_hydrophobic_atom_candidate(
            atom,
            require_coordinates=require_coordinates,
            require_element=require_element,
            allow_deleted=allow_deleted,
        )

        if valid:
            valid_atoms.append(atom)
        else:
            invalid_indices.append(index)

    if invalid_indices and raise_on_invalid:
        displayed_indices = ", ".join(
            str(index)
            for index in invalid_indices[:10]
        )

        if len(invalid_indices) > 10:
            displayed_indices += ", ..."

        raise ValueError(
            f"{normalized_name} contains invalid atom entries at "
            f"indices: {displayed_indices}."
        )

    if not valid_atoms and not allow_empty:
        raise ValueError(
            f"{normalized_name} contains no valid atoms."
        )

    return tuple(valid_atoms)


# -----------------------------------------------------------------------------
# General preparation filters
# -----------------------------------------------------------------------------

def filter_atom_collection(
    atoms: Iterable[AtomLike],
    *,
    remove_hydrogens: bool = True,
    remove_water: bool = True,
    remove_deleted: bool = True,
    displayed_only: bool = False,
    allowed_altlocs: Optional[Iterable[str]] = None,
    minimum_occupancy: Optional[Number] = None,
    predicate: Optional[AtomPredicate] = None,
) -> Tuple[AtomLike, ...]:
    """
    Apply general structural filters before hydrophobic perception.

    These filters concern atom availability and model state, not chemical
    hydrophobicity.
    """

    if predicate is not None and not callable(predicate):
        raise TypeError(
            "predicate must be callable or None."
        )

    normalized_altlocs = (
        _DEFAULT_ALLOWED_ALTLOCS
        if allowed_altlocs is None
        else frozenset(
            str(value).strip().upper()
            for value in allowed_altlocs
        )
    )

    occupancy_limit = (
        None
        if minimum_occupancy is None
        else _nonnegative_float(
            minimum_occupancy,
            name="minimum occupancy",
        )
    )

    filtered: List[AtomLike] = []

    for atom in atoms:
        if atom is None:
            continue

        if remove_deleted and _is_deleted_object(atom):
            continue

        if remove_hydrogens:
            try:
                if is_hydrogen_atom(atom):
                    continue
            except Exception:
                if _safe_atom_element(atom) == HYDROGEN_ELEMENT:
                    continue

        if remove_water and is_water_atom(atom):
            continue

        if displayed_only:
            display_state = _is_displayed_object(atom)

            if display_state is False:
                continue

        altloc = _get_atom_altloc(atom)

        if (
            normalized_altlocs
            and altloc not in normalized_altlocs
        ):
            continue

        if occupancy_limit is not None:
            occupancy = _get_atom_occupancy(atom)

            if (
                occupancy is not None
                and occupancy < occupancy_limit
            ):
                continue

        if predicate is not None:
            try:
                if not bool(predicate(atom)):
                    continue
            except Exception as exc:
                raise ValueError(
                    "The atom-filter predicate raised an exception."
                ) from exc

        filtered.append(atom)

    return tuple(filtered)


def prepare_atom_collection(
    source: Any,
    *,
    collection_name: str = "atom collection",
    allow_empty: bool = True,
    remove_hydrogens: bool = True,
    remove_water: bool = True,
    remove_deleted: bool = True,
    displayed_only: bool = False,
    allowed_altlocs: Optional[Iterable[str]] = None,
    minimum_occupancy: Optional[Number] = None,
    require_coordinates: bool = True,
    require_element: bool = True,
    raise_on_invalid: bool = False,
    deduplication_strategy: Literal[
        "identity",
        "identifier",
        "index",
        "auto",
    ] = "auto",
    predicate: Optional[AtomPredicate] = None,
) -> Tuple[AtomLike, ...]:
    """
    Normalize, validate, filter and deduplicate an atom collection.

    Processing order
    ----------------
    1. recursively extract atoms;
    2. remove duplicates;
    3. validate atom integrity;
    4. remove hydrogens, waters, deleted atoms and unwanted alternate
       locations;
    5. remove any duplicates introduced through external wrappers.
    """

    normalized = normalize_atom_collection(
        source,
        allow_empty=allow_empty,
    )

    normalized = deduplicate_atoms(
        normalized,
        strategy=deduplication_strategy,
    )

    validated = validate_prepared_atom_collection(
        normalized,
        collection_name=collection_name,
        allow_empty=allow_empty,
        require_coordinates=require_coordinates,
        require_element=require_element,
        allow_deleted=not remove_deleted,
        raise_on_invalid=raise_on_invalid,
    )

    filtered = filter_atom_collection(
        validated,
        remove_hydrogens=remove_hydrogens,
        remove_water=remove_water,
        remove_deleted=remove_deleted,
        displayed_only=displayed_only,
        allowed_altlocs=allowed_altlocs,
        minimum_occupancy=minimum_occupancy,
        predicate=predicate,
    )

    filtered = deduplicate_atoms(
        filtered,
        strategy=deduplication_strategy,
    )

    if not filtered and not allow_empty:
        raise ValueError(
            f"{collection_name} contains no atoms after preparation."
        )

    return filtered


# -----------------------------------------------------------------------------
# Receptor and ligand source extraction
# -----------------------------------------------------------------------------

def _extract_dock_model_receptor_source(
    dock_model: DockModel,
) -> Optional[Any]:
    """Extract a receptor-side object from a DockModel."""

    try:
        receptor = get_dock_model_receptor(
            dock_model
        )

        if receptor is not None:
            return receptor

    except Exception:
        pass

    return _safe_getattr(
        dock_model,
        _RECEPTOR_ATTRIBUTE_NAMES,
        default=None,
    )


def _extract_dock_model_ligand_source(
    dock_model: DockModel,
) -> Optional[Any]:
    """Extract a ligand or pose object from a DockModel."""

    try:
        pose = get_dock_model_pose(
            dock_model
        )

        if pose is not None:
            return pose

    except Exception:
        pass

    return _safe_getattr(
        dock_model,
        _LIGAND_ATTRIBUTE_NAMES,
        default=None,
    )


def extract_receptor_source(
    source: Any,
) -> Optional[Any]:
    """
    Resolve the receptor-side molecular object.

    The input may already be a receptor structure, or it may be a
    DockModel-like container.
    """

    if source is None:
        return None

    if isinstance(source, DockModel):
        receptor = _extract_dock_model_receptor_source(
            source
        )

        if receptor is not None:
            return receptor

    receptor = _safe_getattr(
        source,
        _RECEPTOR_ATTRIBUTE_NAMES,
        default=None,
    )

    if receptor is not None and receptor is not source:
        return receptor

    return source


def extract_ligand_source(
    source: Any,
) -> Optional[Any]:
    """
    Resolve the ligand-side molecular object.

    The input may already be a ligand/pose structure, or it may be a
    DockModel-like container.
    """

    if source is None:
        return None

    if isinstance(source, DockModel):
        ligand = _extract_dock_model_ligand_source(
            source
        )

        if ligand is not None:
            return ligand

    ligand = _safe_getattr(
        source,
        _LIGAND_ATTRIBUTE_NAMES,
        default=None,
    )

    if ligand is not None and ligand is not source:
        return ligand

    return source


def extract_receptor_atoms(
    source: Any,
    **preparation_options: Any,
) -> Tuple[AtomLike, ...]:
    """
    Extract and prepare receptor atoms from a structure or DockModel.
    """

    receptor_source = extract_receptor_source(
        source
    )

    return prepare_atom_collection(
        receptor_source,
        collection_name="receptor atom collection",
        **preparation_options,
    )


def extract_ligand_atoms(
    source: Any,
    **preparation_options: Any,
) -> Tuple[AtomLike, ...]:
    """
    Extract and prepare ligand atoms from a pose or DockModel.
    """

    ligand_source = extract_ligand_source(
        source
    )

    return prepare_atom_collection(
        ligand_source,
        collection_name="ligand atom collection",
        **preparation_options,
    )


# -----------------------------------------------------------------------------
# Receptor-ligand separation and overlap handling
# -----------------------------------------------------------------------------

def find_collection_overlap(
    receptor_atoms: Iterable[AtomLike],
    ligand_atoms: Iterable[AtomLike],
) -> Tuple[AtomLike, ...]:
    """
    Return atom objects present in both receptor and ligand collections.
    """

    ligand_identities = {
        id(atom)
        for atom in ligand_atoms
    }

    return tuple(
        atom
        for atom in receptor_atoms
        if id(atom) in ligand_identities
    )


def remove_collection_overlap(
    receptor_atoms: Iterable[AtomLike],
    ligand_atoms: Iterable[AtomLike],
    *,
    prefer: Literal[
        "receptor",
        "ligand",
        "error",
    ] = "error",
) -> Tuple[
    Tuple[AtomLike, ...],
    Tuple[AtomLike, ...],
]:
    """
    Resolve atoms assigned simultaneously to receptor and ligand.

    Parameters
    ----------
    prefer
        ``"receptor"`` removes overlaps from the ligand collection.
        ``"ligand"`` removes them from the receptor collection.
        ``"error"`` raises an exception.
    """

    normalized_preference = str(prefer).strip().lower()

    if normalized_preference not in {
        "receptor",
        "ligand",
        "error",
    }:
        raise ValueError(
            "prefer must be 'receptor', 'ligand' or 'error'."
        )

    receptor_tuple = tuple(receptor_atoms)
    ligand_tuple = tuple(ligand_atoms)

    overlapping_ids = {
        id(atom)
        for atom in find_collection_overlap(
            receptor_tuple,
            ligand_tuple,
        )
    }

    if not overlapping_ids:
        return receptor_tuple, ligand_tuple

    if normalized_preference == "error":
        raise ValueError(
            "Receptor and ligand collections share "
            f"{len(overlapping_ids)} atom object(s)."
        )

    if normalized_preference == "receptor":
        ligand_tuple = tuple(
            atom
            for atom in ligand_tuple
            if id(atom) not in overlapping_ids
        )

    else:
        receptor_tuple = tuple(
            atom
            for atom in receptor_tuple
            if id(atom) not in overlapping_ids
        )

    return receptor_tuple, ligand_tuple


# -----------------------------------------------------------------------------
# Hydrophobic descriptor preparation
# -----------------------------------------------------------------------------

def prepare_hydrophobic_descriptors(
    atoms: Iterable[AtomLike],
    *,
    role: HydrophobicAtomRole,
    include_rejected: bool = False,
    maximum_absolute_partial_charge: Optional[Number] = None,
    maximum_polar_neighbors: Optional[int] = None,
    hydrophobic_residue_names: Optional[Iterable[str]] = None,
) -> Union[
    Tuple[HydrophobicAtom, ...],
    Tuple[
        Tuple[HydrophobicAtom, ...],
        Tuple[HydrophobicAtom, ...],
    ],
]:
    """
    Create deduplicated hydrophobic descriptors for one molecular side.

    Returns
    -------
    tuple
        Accepted descriptors when ``include_rejected=False``.
    accepted, rejected
        Two descriptor tuples when ``include_rejected=True``.
    """

    normalized_role = validate_hydrophobic_atom_role(
        role
    )

    atom_tuple = deduplicate_atoms(
        tuple(atoms),
        strategy="auto",
    )

    if normalized_role == HYDROPHOBIC_ROLE_RECEPTOR:
        receptor_atoms = atom_tuple
        ligand_atoms = None

    elif normalized_role == HYDROPHOBIC_ROLE_LIGAND:
        receptor_atoms = None
        ligand_atoms = atom_tuple

    else:
        receptor_atoms = None
        ligand_atoms = None

    accepted, rejected = partition_hydrophobic_atoms(
        atom_tuple,
        role=normalized_role,
        receptor_atoms=receptor_atoms,
        ligand_atoms=ligand_atoms,
        maximum_absolute_partial_charge=(
            maximum_absolute_partial_charge
        ),
        maximum_polar_neighbors=maximum_polar_neighbors,
        hydrophobic_residue_names=hydrophobic_residue_names,
    )

    accepted = deduplicate_hydrophobic_descriptors(
        accepted
    )

    rejected = deduplicate_hydrophobic_descriptors(
        rejected
    )

    if include_rejected:
        return accepted, rejected

    return accepted


def prepare_hydrophobic_atom_collections(
    receptor: Any,
    ligand: Optional[Any] = None,
    *,
    allow_empty_receptor: bool = False,
    allow_empty_ligand: bool = False,
    remove_hydrogens: bool = True,
    remove_water: bool = True,
    remove_deleted: bool = True,
    displayed_only: bool = False,
    allowed_altlocs: Optional[Iterable[str]] = None,
    minimum_occupancy: Optional[Number] = None,
    require_coordinates: bool = True,
    require_element: bool = True,
    raise_on_invalid: bool = False,
    overlap_policy: Literal[
        "receptor",
        "ligand",
        "error",
    ] = "error",
    deduplication_strategy: Literal[
        "identity",
        "identifier",
        "index",
        "auto",
    ] = "auto",
    receptor_predicate: Optional[AtomPredicate] = None,
    ligand_predicate: Optional[AtomPredicate] = None,
    maximum_absolute_partial_charge: Optional[Number] = None,
    maximum_polar_neighbors: Optional[int] = None,
    hydrophobic_residue_names: Optional[Iterable[str]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> HydrophobicAtomCollections:
    """
    Prepare receptor and ligand collections for interaction detection.

    ``receptor`` may be a receptor structure or a complete DockModel. When
    a DockModel is supplied and ``ligand`` is omitted, both molecular sides
    are extracted from the DockModel.
    """

    supplied_dock_model = (
        receptor
        if isinstance(receptor, DockModel)
        else None
    )

    if supplied_dock_model is not None:
        receptor_source = extract_receptor_source(
            supplied_dock_model
        )

        ligand_source = (
            extract_ligand_source(supplied_dock_model)
            if ligand is None
            else extract_ligand_source(ligand)
        )

    else:
        receptor_source = extract_receptor_source(
            receptor
        )

        ligand_source = extract_ligand_source(
            ligand
        )

    receptor_atoms = prepare_atom_collection(
        receptor_source,
        collection_name="receptor atom collection",
        allow_empty=allow_empty_receptor,
        remove_hydrogens=remove_hydrogens,
        remove_water=remove_water,
        remove_deleted=remove_deleted,
        displayed_only=displayed_only,
        allowed_altlocs=allowed_altlocs,
        minimum_occupancy=minimum_occupancy,
        require_coordinates=require_coordinates,
        require_element=require_element,
        raise_on_invalid=raise_on_invalid,
        deduplication_strategy=deduplication_strategy,
        predicate=receptor_predicate,
    )

    ligand_atoms = prepare_atom_collection(
        ligand_source,
        collection_name="ligand atom collection",
        allow_empty=allow_empty_ligand,
        remove_hydrogens=remove_hydrogens,
        remove_water=remove_water,
        remove_deleted=remove_deleted,
        displayed_only=displayed_only,
        allowed_altlocs=allowed_altlocs,
        minimum_occupancy=minimum_occupancy,
        require_coordinates=require_coordinates,
        require_element=require_element,
        raise_on_invalid=raise_on_invalid,
        deduplication_strategy=deduplication_strategy,
        predicate=ligand_predicate,
    )

    receptor_atoms, ligand_atoms = remove_collection_overlap(
        receptor_atoms,
        ligand_atoms,
        prefer=overlap_policy,
    )

    receptor_descriptor_result = (
        prepare_hydrophobic_descriptors(
            receptor_atoms,
            role=HYDROPHOBIC_ROLE_RECEPTOR,
            include_rejected=True,
            maximum_absolute_partial_charge=(
                maximum_absolute_partial_charge
            ),
            maximum_polar_neighbors=maximum_polar_neighbors,
            hydrophobic_residue_names=(
                hydrophobic_residue_names
            ),
        )
    )

    ligand_descriptor_result = (
        prepare_hydrophobic_descriptors(
            ligand_atoms,
            role=HYDROPHOBIC_ROLE_LIGAND,
            include_rejected=True,
            maximum_absolute_partial_charge=(
                maximum_absolute_partial_charge
            ),
            maximum_polar_neighbors=maximum_polar_neighbors,
            hydrophobic_residue_names=(
                hydrophobic_residue_names
            ),
        )
    )

    receptor_hydrophobic_atoms, rejected_receptor_atoms = (
        receptor_descriptor_result
    )

    ligand_hydrophobic_atoms, rejected_ligand_atoms = (
        ligand_descriptor_result
    )

    preparation_metadata: Dict[str, Any] = (
        {} if metadata is None else dict(metadata)
    )

    preparation_metadata.update(
        {
            "source_is_dock_model": (
                supplied_dock_model is not None
            ),
            "remove_hydrogens": bool(remove_hydrogens),
            "remove_water": bool(remove_water),
            "remove_deleted": bool(remove_deleted),
            "displayed_only": bool(displayed_only),
            "allowed_altlocs": tuple(
                sorted(
                    _DEFAULT_ALLOWED_ALTLOCS
                    if allowed_altlocs is None
                    else {
                        str(value).strip().upper()
                        for value in allowed_altlocs
                    }
                )
            ),
            "minimum_occupancy": (
                None
                if minimum_occupancy is None
                else float(
                    _nonnegative_float(
                        minimum_occupancy,
                        name="minimum occupancy",
                    )
                )
            ),
            "require_coordinates": bool(
                require_coordinates
            ),
            "require_element": bool(require_element),
            "raise_on_invalid": bool(raise_on_invalid),
            "overlap_policy": overlap_policy,
            "deduplication_strategy": (
                deduplication_strategy
            ),
            "receptor_atom_count": len(
                receptor_atoms
            ),
            "ligand_atom_count": len(
                ligand_atoms
            ),
            "receptor_hydrophobic_atom_count": len(
                receptor_hydrophobic_atoms
            ),
            "ligand_hydrophobic_atom_count": len(
                ligand_hydrophobic_atoms
            ),
            "rejected_receptor_atom_count": len(
                rejected_receptor_atoms
            ),
            "rejected_ligand_atom_count": len(
                rejected_ligand_atoms
            ),
        }
    )

    return HydrophobicAtomCollections(
        receptor_atoms=receptor_atoms,
        ligand_atoms=ligand_atoms,
        receptor_hydrophobic_atoms=(
            receptor_hydrophobic_atoms
        ),
        ligand_hydrophobic_atoms=(
            ligand_hydrophobic_atoms
        ),
        rejected_receptor_atoms=(
            rejected_receptor_atoms
        ),
        rejected_ligand_atoms=(
            rejected_ligand_atoms
        ),
        receptor_source=receptor_source,
        ligand_source=ligand_source,
        metadata=preparation_metadata,
    )


# -----------------------------------------------------------------------------
# Synthetic objects for self-tests
# -----------------------------------------------------------------------------

@dataclass(
    slots=True,
)
class SyntheticElement:
    """
    Minimal element object used by hydrophobic self-tests.
    """

    name: str
    atomic_number: int

    @property
    def symbol(self) -> str:
        """Return the normalized element symbol."""

        return str(self.name).strip().upper()


@dataclass(
    slots=True,
)
class SyntheticResidue:
    """
    Minimal residue object compatible with the perception layer.
    """

    name: str
    number: int = 1
    chain_id: str = "A"
    insertion_code: str = ""
    atoms: List[Any] = field(
        default_factory=list
    )

    def add_atom(
        self,
        atom: Any,
    ) -> None:
        """Add an atom and assign this residue as its parent."""

        if atom not in self.atoms:
            self.atoms.append(atom)

        try:
            atom.residue = self
        except Exception:
            pass


@dataclass(
    slots=True,
)
class SyntheticBond:
    """
    Minimal bond object carrying atoms, order and aromaticity.
    """

    atom1: Any
    atom2: Any
    order: float = 1.0
    aromatic: bool = False

    @property
    def atoms(self) -> Tuple[Any, Any]:
        """Return both bonded atoms."""

        return (
            self.atom1,
            self.atom2,
        )


@dataclass(
    slots=True,
)
class SyntheticAtom:
    """
    Lightweight atom compatible with Sections 4 and 5.

    The class is intended only for internal self-tests and examples.
    """

    name: str
    element: Any
    coord: Any

    residue: Optional[SyntheticResidue] = None
    index: Optional[int] = None

    formal_charge: Optional[float] = None
    charge: Optional[float] = None

    aromatic: Optional[bool] = None
    aliphatic: Optional[bool] = None
    atom_type: Optional[str] = None

    occupancy: float = 1.0
    alt_loc: str = ""
    deleted: bool = False
    display: bool = True

    neighbors: List[Any] = field(
        default_factory=list
    )

    bonds: List[SyntheticBond] = field(
        default_factory=list
    )

    structure: Optional[Any] = None

    def __post_init__(self) -> None:
        """Normalize coordinates and register the parent residue."""

        self.coord = np.asarray(
            self.coord,
            dtype=np.float64,
        )

        if self.coord.shape != (3,):
            raise ValueError(
                "SyntheticAtom.coord must contain exactly "
                "three coordinates."
            )

        if self.residue is not None:
            self.residue.add_atom(self)

    def connect(
        self,
        other: "SyntheticAtom",
        *,
        order: Number = 1.0,
        aromatic: bool = False,
    ) -> SyntheticBond:
        """
        Create a reciprocal bond to another synthetic atom.
        """

        if not isinstance(
            other,
            SyntheticAtom,
        ):
            raise TypeError(
                "other must be a SyntheticAtom."
            )

        if other is self:
            raise ValueError(
                "An atom cannot be bonded to itself."
            )

        normalized_order = _positive_float(
            order,
            name="synthetic bond order",
        )

        bond = SyntheticBond(
            atom1=self,
            atom2=other,
            order=float(normalized_order),
            aromatic=bool(aromatic),
        )

        if other not in self.neighbors:
            self.neighbors.append(other)

        if self not in other.neighbors:
            other.neighbors.append(self)

        self.bonds.append(bond)
        other.bonds.append(bond)

        return bond


@dataclass(
    slots=True,
)
class SyntheticStructure:
    """
    Minimal atom-containing molecular structure for self-tests.
    """

    name: str
    atoms: List[SyntheticAtom] = field(
        default_factory=list
    )

    @property
    def residues(self) -> Tuple[SyntheticResidue, ...]:
        """Return unique residues represented by the structure."""

        result: List[SyntheticResidue] = []
        seen: Set[int] = set()

        for atom in self.atoms:
            residue = atom.residue

            if residue is None:
                continue

            residue_identity = id(residue)

            if residue_identity in seen:
                continue

            seen.add(residue_identity)
            result.append(residue)

        return tuple(result)

    def add_atom(
        self,
        atom: SyntheticAtom,
    ) -> None:
        """Add an atom to the structure."""

        if atom not in self.atoms:
            self.atoms.append(atom)

        atom.structure = self


def make_synthetic_atom(
    name: str,
    element: str,
    coordinate: Sequence[Number],
    *,
    atomic_number: Optional[int] = None,
    residue: Optional[SyntheticResidue] = None,
    index: Optional[int] = None,
    formal_charge: Optional[Number] = None,
    partial_charge: Optional[Number] = None,
    aromatic: Optional[bool] = None,
    aliphatic: Optional[bool] = None,
    atom_type: Optional[str] = None,
) -> SyntheticAtom:
    """
    Construct a synthetic atom with normalized element information.
    """

    normalized_element = _normalize_element_symbol(
        element
    )

    known_atomic_numbers: Mapping[str, int] = {
        HYDROGEN_ELEMENT: HYDROGEN_ATOMIC_NUMBER,
        CARBON_ELEMENT: CARBON_ATOMIC_NUMBER,
        NITROGEN_ELEMENT: NITROGEN_ATOMIC_NUMBER,
        OXYGEN_ELEMENT: OXYGEN_ATOMIC_NUMBER,
        FLUORINE_ELEMENT: FLUORINE_ATOMIC_NUMBER,
        PHOSPHORUS_ELEMENT: PHOSPHORUS_ATOMIC_NUMBER,
        SULFUR_ELEMENT: SULFUR_ATOMIC_NUMBER,
        CHLORINE_ELEMENT: CHLORINE_ATOMIC_NUMBER,
        BROMINE_ELEMENT: BROMINE_ATOMIC_NUMBER,
        IODINE_ELEMENT: IODINE_ATOMIC_NUMBER,
    }

    resolved_atomic_number = (
        known_atomic_numbers.get(
            normalized_element,
            0,
        )
        if atomic_number is None
        else _nonnegative_integer(
            atomic_number,
            name="atomic number",
        )
    )

    element_object = SyntheticElement(
        name=normalized_element,
        atomic_number=resolved_atomic_number,
    )

    return SyntheticAtom(
        name=_normalize_required_string(
            name,
            name="synthetic atom name",
        ),
        element=element_object,
        coord=coordinate,
        residue=residue,
        index=index,
        formal_charge=(
            None
            if formal_charge is None
            else float(
                _finite_float(
                    formal_charge,
                    name="formal charge",
                )
            )
        ),
        charge=(
            None
            if partial_charge is None
            else float(
                _finite_float(
                    partial_charge,
                    name="partial charge",
                )
            )
        ),
        aromatic=aromatic,
        aliphatic=aliphatic,
        atom_type=atom_type,
    )


# -----------------------------------------------------------------------------
# Empty prepared collections
# -----------------------------------------------------------------------------

_EMPTY_PREPARED_HYDROPHOBIC_COLLECTIONS: Final[
    HydrophobicAtomCollections
] = HydrophobicAtomCollections()


# -----------------------------------------------------------------------------
# Section 5 public names
# -----------------------------------------------------------------------------

_SECTION_5_PUBLIC_NAMES: Final[Tuple[str, ...]] = (
    # Prepared result
    "HydrophobicAtomCollections",

    # Collection normalization
    "normalize_atom_collection",
    "atom_deduplication_key",
    "deduplicate_atoms",
    "descriptor_deduplication_key",
    "deduplicate_hydrophobic_descriptors",

    # Validation
    "validate_hydrophobic_atom_candidate",
    "validate_prepared_atom_collection",

    # Filtering and preparation
    "filter_atom_collection",
    "prepare_atom_collection",

    # Receptor and ligand extraction
    "extract_receptor_source",
    "extract_ligand_source",
    "extract_receptor_atoms",
    "extract_ligand_atoms",

    # Collection separation
    "find_collection_overlap",
    "remove_collection_overlap",

    # Hydrophobic descriptors
    "prepare_hydrophobic_descriptors",
    "prepare_hydrophobic_atom_collections",

    # Synthetic testing objects
    "SyntheticElement",
    "SyntheticResidue",
    "SyntheticBond",
    "SyntheticAtom",
    "SyntheticStructure",
    "make_synthetic_atom",
)

for public_name in _SECTION_5_PUBLIC_NAMES:
    if public_name not in __all__:
        __all__.append(public_name)


# =============================================================================
# End of Section 5
# =============================================================================


# =============================================================================
# Section 6 — Hydrophobic-interaction geometry
# =============================================================================


# -----------------------------------------------------------------------------
# Geometry-related aliases
# -----------------------------------------------------------------------------

HydrophobicGeometryType: TypeAlias = Literal[
    "aliphatic_aliphatic",
    "aliphatic_aromatic",
    "aromatic_aliphatic",
    "aromatic_aromatic",
    "mixed",
    "unknown",
]

HydrophobicGroupInput: TypeAlias = Union[
    Sequence[AtomLike],
    Sequence[HydrophobicAtom],
]

HydrophobicCoordinateInput: TypeAlias = Union[
    AtomLike,
    HydrophobicAtom,
    Sequence[Number],
    NDArray[np.floating],
]


# -----------------------------------------------------------------------------
# Geometric defaults
# -----------------------------------------------------------------------------

DEFAULT_LOCAL_CONTACT_RADIUS: Final[np.float64] = np.float64(
    DEFAULT_MAXIMUM_HYDROPHOBIC_DISTANCE
)

DEFAULT_CONTACT_DENSITY_RADIUS: Final[np.float64] = np.float64(
    DEFAULT_GROUPING_DISTANCE
)

DEFAULT_CONTACT_AREA_ATOM_RADIUS: Final[np.float64] = np.float64(
    1.70
)

DEFAULT_MINIMUM_CONTACT_AREA: Final[np.float64] = np.float64(
    0.0
)

DEFAULT_MAXIMUM_CONTACT_AREA_PER_PAIR: Final[np.float64] = np.float64(
    25.0
)

DEFAULT_COMPACTION_REFERENCE_DISTANCE: Final[np.float64] = np.float64(
    DEFAULT_MAXIMUM_HYDROPHOBIC_DISTANCE
)

DEFAULT_DENSITY_NORMALIZATION_CONTACT_COUNT: Final[int] = 6


# -----------------------------------------------------------------------------
# Hydrophobic geometry dataclasses
# -----------------------------------------------------------------------------

@dataclass(
    frozen=True,
    slots=True,
)
class HydrophobicGroupGeometry:
    """
    Geometric description of one hydrophobic atom group.

    Parameters
    ----------
    atoms
        Raw atoms represented by the group.
    descriptors
        Hydrophobic descriptors associated with the atoms.
    centroid
        Arithmetic centroid of valid atomic coordinates.
    minimum_coordinate
        Lower Cartesian boundary of the group.
    maximum_coordinate
        Upper Cartesian boundary of the group.
    radius_of_gyration
        Root-mean-square distance from atoms to the centroid.
    maximum_centroid_distance
        Largest atom-to-centroid distance.
    aromatic_atom_count
        Number of aromatic descriptors.
    aliphatic_atom_count
        Number of aliphatic descriptors.
    metadata
        Additional group-level geometric information.
    """

    atoms: Sequence[AtomLike] = field(
        default_factory=tuple
    )

    descriptors: Sequence[HydrophobicAtom] = field(
        default_factory=tuple
    )

    centroid: Coordinate = field(
        default_factory=lambda: np.zeros(
            3,
            dtype=np.float64,
        )
    )

    minimum_coordinate: Coordinate = field(
        default_factory=lambda: np.zeros(
            3,
            dtype=np.float64,
        )
    )

    maximum_coordinate: Coordinate = field(
        default_factory=lambda: np.zeros(
            3,
            dtype=np.float64,
        )
    )

    radius_of_gyration: np.float64 = np.float64(0.0)
    maximum_centroid_distance: np.float64 = np.float64(0.0)

    aromatic_atom_count: int = 0
    aliphatic_atom_count: int = 0

    metadata: Mapping[str, Any] = field(
        default_factory=lambda: _EMPTY_METADATA,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        """Validate and freeze group geometry."""

        atoms = tuple(self.atoms)
        descriptors = tuple(self.descriptors)

        for index, descriptor in enumerate(descriptors):
            if not isinstance(
                descriptor,
                HydrophobicAtom,
            ):
                raise TypeError(
                    "descriptors must contain HydrophobicAtom "
                    f"instances; invalid entry at index {index}."
                )

        centroid = _normalize_coordinate(
            self.centroid,
            name="group centroid",
        )

        minimum_coordinate = _normalize_coordinate(
            self.minimum_coordinate,
            name="minimum group coordinate",
        )

        maximum_coordinate = _normalize_coordinate(
            self.maximum_coordinate,
            name="maximum group coordinate",
        )

        if np.any(
            minimum_coordinate > maximum_coordinate
        ):
            raise ValueError(
                "minimum_coordinate cannot exceed "
                "maximum_coordinate."
            )

        radius_of_gyration = _nonnegative_float(
            self.radius_of_gyration,
            name="radius of gyration",
        )

        maximum_centroid_distance = _nonnegative_float(
            self.maximum_centroid_distance,
            name="maximum centroid distance",
        )

        aromatic_atom_count = _nonnegative_integer(
            self.aromatic_atom_count,
            name="aromatic atom count",
        )

        aliphatic_atom_count = _nonnegative_integer(
            self.aliphatic_atom_count,
            name="aliphatic atom count",
        )

        object.__setattr__(
            self,
            "atoms",
            atoms,
        )

        object.__setattr__(
            self,
            "descriptors",
            descriptors,
        )

        object.__setattr__(
            self,
            "centroid",
            centroid,
        )

        object.__setattr__(
            self,
            "minimum_coordinate",
            minimum_coordinate,
        )

        object.__setattr__(
            self,
            "maximum_coordinate",
            maximum_coordinate,
        )

        object.__setattr__(
            self,
            "radius_of_gyration",
            radius_of_gyration,
        )

        object.__setattr__(
            self,
            "maximum_centroid_distance",
            maximum_centroid_distance,
        )

        object.__setattr__(
            self,
            "aromatic_atom_count",
            aromatic_atom_count,
        )

        object.__setattr__(
            self,
            "aliphatic_atom_count",
            aliphatic_atom_count,
        )

        object.__setattr__(
            self,
            "metadata",
            _freeze_metadata(self.metadata),
        )

    @property
    def atom_count(self) -> int:
        """Return the number of atoms in the group."""

        return len(self.atoms)

    @property
    def descriptor_count(self) -> int:
        """Return the number of descriptors in the group."""

        return len(self.descriptors)

    @property
    def bounding_box_size(self) -> Coordinate:
        """Return the Cartesian dimensions of the group."""

        return np.asarray(
            self.maximum_coordinate
            - self.minimum_coordinate,
            dtype=np.float64,
        )

    @property
    def bounding_box_volume(self) -> np.float64:
        """
        Return the axis-aligned bounding-box volume.

        Flat or single-atom groups may have a volume of zero.
        """

        size = self.bounding_box_size

        return np.float64(
            np.prod(
                np.maximum(
                    size,
                    0.0,
                )
            )
        )

    @property
    def is_aromatic(self) -> bool:
        """Return whether every classified atom is aromatic."""

        classified_count = (
            self.aromatic_atom_count
            + self.aliphatic_atom_count
        )

        return (
            classified_count > 0
            and self.aromatic_atom_count == classified_count
        )

    @property
    def is_aliphatic(self) -> bool:
        """Return whether every classified atom is aliphatic."""

        classified_count = (
            self.aromatic_atom_count
            + self.aliphatic_atom_count
        )

        return (
            classified_count > 0
            and self.aliphatic_atom_count == classified_count
        )

    @property
    def is_mixed(self) -> bool:
        """Return whether aromatic and aliphatic atoms coexist."""

        return (
            self.aromatic_atom_count > 0
            and self.aliphatic_atom_count > 0
        )

    def to_dict(
        self,
        *,
        include_atoms: bool = False,
        include_descriptors: bool = True,
    ) -> Dict[str, Any]:
        """Serialize group geometry."""

        result: Dict[str, Any] = {
            "atom_count": self.atom_count,
            "descriptor_count": self.descriptor_count,
            "centroid": self.centroid.tolist(),
            "minimum_coordinate": (
                self.minimum_coordinate.tolist()
            ),
            "maximum_coordinate": (
                self.maximum_coordinate.tolist()
            ),
            "bounding_box_size": (
                self.bounding_box_size.tolist()
            ),
            "bounding_box_volume": float(
                self.bounding_box_volume
            ),
            "radius_of_gyration": float(
                self.radius_of_gyration
            ),
            "maximum_centroid_distance": float(
                self.maximum_centroid_distance
            ),
            "aromatic_atom_count": (
                self.aromatic_atom_count
            ),
            "aliphatic_atom_count": (
                self.aliphatic_atom_count
            ),
            "is_aromatic": self.is_aromatic,
            "is_aliphatic": self.is_aliphatic,
            "is_mixed": self.is_mixed,
            "metadata": dict(self.metadata),
        }

        if include_descriptors:
            result["descriptors"] = [
                descriptor.to_dict(
                    include_atom=include_atoms,
                    include_residue=include_atoms,
                )
                for descriptor in self.descriptors
            ]

        if include_atoms:
            result["atoms"] = self.atoms

        return result


@dataclass(
    frozen=True,
    slots=True,
)
class HydrophobicPairGeometry:
    """
    Geometric description of one receptor–ligand hydrophobic pair.

    This object describes non-directional hydrophobic contact geometry.
    Aromatic–aromatic contact does not imply π-stacking.
    """

    receptor_atom: AtomLike
    ligand_atom: AtomLike

    receptor_descriptor: Optional[HydrophobicAtom] = None
    ligand_descriptor: Optional[HydrophobicAtom] = None

    distance: np.float64 = np.float64(0.0)

    geometry_type: HydrophobicGeometryType = (
        HYDROPHOBIC_TYPE_UNKNOWN
    )

    receptor_neighbor_count: int = 0
    ligand_neighbor_count: int = 0
    shared_local_contact_count: int = 0

    local_compaction: np.float64 = np.float64(0.0)
    contact_density: np.float64 = np.float64(0.0)
    approximate_contact_area: np.float64 = np.float64(0.0)

    receptor_local_centroid: Optional[Coordinate] = None
    ligand_local_centroid: Optional[Coordinate] = None
    local_centroid_distance: Optional[np.float64] = None

    metadata: Mapping[str, Any] = field(
        default_factory=lambda: _EMPTY_METADATA,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        """Validate and normalize pair geometry."""

        if self.receptor_atom is None:
            raise ValueError(
                "receptor_atom cannot be None."
            )

        if self.ligand_atom is None:
            raise ValueError(
                "ligand_atom cannot be None."
            )

        pair_distance = _nonnegative_float(
            self.distance,
            name="pair distance",
        )

        geometry_type = (
            validate_hydrophobic_interaction_type(
                self.geometry_type
            )
        )

        receptor_neighbor_count = _nonnegative_integer(
            self.receptor_neighbor_count,
            name="receptor neighbor count",
        )

        ligand_neighbor_count = _nonnegative_integer(
            self.ligand_neighbor_count,
            name="ligand neighbor count",
        )

        shared_local_contact_count = _nonnegative_integer(
            self.shared_local_contact_count,
            name="shared local contact count",
        )

        local_compaction = validate_hydrophobic_score(
            self.local_compaction
        )

        contact_density = validate_hydrophobic_score(
            self.contact_density
        )

        approximate_contact_area = _nonnegative_float(
            self.approximate_contact_area,
            name="approximate contact area",
        )

        receptor_local_centroid = (
            None
            if self.receptor_local_centroid is None
            else _normalize_coordinate(
                self.receptor_local_centroid,
                name="receptor local centroid",
            )
        )

        ligand_local_centroid = (
            None
            if self.ligand_local_centroid is None
            else _normalize_coordinate(
                self.ligand_local_centroid,
                name="ligand local centroid",
            )
        )

        local_centroid_distance = (
            None
            if self.local_centroid_distance is None
            else _nonnegative_float(
                self.local_centroid_distance,
                name="local centroid distance",
            )
        )

        object.__setattr__(
            self,
            "distance",
            pair_distance,
        )

        object.__setattr__(
            self,
            "geometry_type",
            geometry_type,
        )

        object.__setattr__(
            self,
            "receptor_neighbor_count",
            receptor_neighbor_count,
        )

        object.__setattr__(
            self,
            "ligand_neighbor_count",
            ligand_neighbor_count,
        )

        object.__setattr__(
            self,
            "shared_local_contact_count",
            shared_local_contact_count,
        )

        object.__setattr__(
            self,
            "local_compaction",
            local_compaction,
        )

        object.__setattr__(
            self,
            "contact_density",
            contact_density,
        )

        object.__setattr__(
            self,
            "approximate_contact_area",
            approximate_contact_area,
        )

        object.__setattr__(
            self,
            "receptor_local_centroid",
            receptor_local_centroid,
        )

        object.__setattr__(
            self,
            "ligand_local_centroid",
            ligand_local_centroid,
        )

        object.__setattr__(
            self,
            "local_centroid_distance",
            local_centroid_distance,
        )

        object.__setattr__(
            self,
            "metadata",
            _freeze_metadata(self.metadata),
        )

    @property
    def total_neighbor_count(self) -> int:
        """Return the combined local neighbor count."""

        return (
            self.receptor_neighbor_count
            + self.ligand_neighbor_count
        )

    @property
    def is_aliphatic_aliphatic(self) -> bool:
        """Return whether the pair is aliphatic–aliphatic."""

        return (
            self.geometry_type
            == HYDROPHOBIC_TYPE_ALIPHATIC_ALIPHATIC
        )

    @property
    def is_aromatic_aliphatic(self) -> bool:
        """Return whether exactly one side is aromatic."""

        return self.geometry_type in {
            HYDROPHOBIC_TYPE_ALIPHATIC_AROMATIC,
            HYDROPHOBIC_TYPE_AROMATIC_ALIPHATIC,
        }

    @property
    def is_aromatic_aromatic(self) -> bool:
        """
        Return whether both atoms have aromatic character.

        This property does not classify the contact as π-stacking.
        """

        return (
            self.geometry_type
            == HYDROPHOBIC_TYPE_AROMATIC_AROMATIC
        )

    def to_dict(
        self,
        *,
        include_atoms: bool = False,
        include_descriptors: bool = True,
    ) -> Dict[str, Any]:
        """Serialize pair geometry."""

        result: Dict[str, Any] = {
            "receptor_atom_identifier": (
                _safe_atom_identifier(
                    self.receptor_atom
                )
            ),
            "ligand_atom_identifier": (
                _safe_atom_identifier(
                    self.ligand_atom
                )
            ),
            "distance": float(self.distance),
            "geometry_type": self.geometry_type,
            "receptor_neighbor_count": (
                self.receptor_neighbor_count
            ),
            "ligand_neighbor_count": (
                self.ligand_neighbor_count
            ),
            "total_neighbor_count": (
                self.total_neighbor_count
            ),
            "shared_local_contact_count": (
                self.shared_local_contact_count
            ),
            "local_compaction": float(
                self.local_compaction
            ),
            "contact_density": float(
                self.contact_density
            ),
            "approximate_contact_area": float(
                self.approximate_contact_area
            ),
            "receptor_local_centroid": (
                None
                if self.receptor_local_centroid is None
                else self.receptor_local_centroid.tolist()
            ),
            "ligand_local_centroid": (
                None
                if self.ligand_local_centroid is None
                else self.ligand_local_centroid.tolist()
            ),
            "local_centroid_distance": (
                None
                if self.local_centroid_distance is None
                else float(self.local_centroid_distance)
            ),
            "is_aliphatic_aliphatic": (
                self.is_aliphatic_aliphatic
            ),
            "is_aromatic_aliphatic": (
                self.is_aromatic_aliphatic
            ),
            "is_aromatic_aromatic": (
                self.is_aromatic_aromatic
            ),
            "pi_stacking_assigned": False,
            "metadata": dict(self.metadata),
        }

        if include_descriptors:
            result["receptor_descriptor"] = (
                None
                if self.receptor_descriptor is None
                else self.receptor_descriptor.to_dict(
                    include_atom=include_atoms,
                    include_residue=include_atoms,
                )
            )

            result["ligand_descriptor"] = (
                None
                if self.ligand_descriptor is None
                else self.ligand_descriptor.to_dict(
                    include_atom=include_atoms,
                    include_residue=include_atoms,
                )
            )

        if include_atoms:
            result["receptor_atom"] = self.receptor_atom
            result["ligand_atom"] = self.ligand_atom

        return result


@dataclass(
    frozen=True,
    slots=True,
)
class HydrophobicGroupContactGeometry:
    """
    Geometry between receptor and ligand hydrophobic groups.
    """

    receptor_group: HydrophobicGroupGeometry
    ligand_group: HydrophobicGroupGeometry

    centroid_distance: np.float64
    minimum_distance: np.float64
    mean_contact_distance: Optional[np.float64] = None

    contact_pair_count: int = 0
    receptor_contact_atom_count: int = 0
    ligand_contact_atom_count: int = 0

    compaction: np.float64 = np.float64(0.0)
    contact_density: np.float64 = np.float64(0.0)
    approximate_contact_area: np.float64 = np.float64(0.0)

    geometry_type: HydrophobicGeometryType = (
        HYDROPHOBIC_TYPE_UNKNOWN
    )

    metadata: Mapping[str, Any] = field(
        default_factory=lambda: _EMPTY_METADATA,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        """Validate group-contact geometry."""

        if not isinstance(
            self.receptor_group,
            HydrophobicGroupGeometry,
        ):
            raise TypeError(
                "receptor_group must be a "
                "HydrophobicGroupGeometry."
            )

        if not isinstance(
            self.ligand_group,
            HydrophobicGroupGeometry,
        ):
            raise TypeError(
                "ligand_group must be a "
                "HydrophobicGroupGeometry."
            )

        centroid_distance = _nonnegative_float(
            self.centroid_distance,
            name="centroid distance",
        )

        minimum_distance = _nonnegative_float(
            self.minimum_distance,
            name="minimum group distance",
        )

        mean_contact_distance = (
            None
            if self.mean_contact_distance is None
            else _nonnegative_float(
                self.mean_contact_distance,
                name="mean contact distance",
            )
        )

        contact_pair_count = _nonnegative_integer(
            self.contact_pair_count,
            name="contact pair count",
        )

        receptor_contact_atom_count = _nonnegative_integer(
            self.receptor_contact_atom_count,
            name="receptor contact atom count",
        )

        ligand_contact_atom_count = _nonnegative_integer(
            self.ligand_contact_atom_count,
            name="ligand contact atom count",
        )

        compaction = validate_hydrophobic_score(
            self.compaction
        )

        contact_density = validate_hydrophobic_score(
            self.contact_density
        )

        approximate_contact_area = _nonnegative_float(
            self.approximate_contact_area,
            name="approximate contact area",
        )

        geometry_type = (
            validate_hydrophobic_interaction_type(
                self.geometry_type
            )
        )

        object.__setattr__(
            self,
            "centroid_distance",
            centroid_distance,
        )

        object.__setattr__(
            self,
            "minimum_distance",
            minimum_distance,
        )

        object.__setattr__(
            self,
            "mean_contact_distance",
            mean_contact_distance,
        )

        object.__setattr__(
            self,
            "contact_pair_count",
            contact_pair_count,
        )

        object.__setattr__(
            self,
            "receptor_contact_atom_count",
            receptor_contact_atom_count,
        )

        object.__setattr__(
            self,
            "ligand_contact_atom_count",
            ligand_contact_atom_count,
        )

        object.__setattr__(
            self,
            "compaction",
            compaction,
        )

        object.__setattr__(
            self,
            "contact_density",
            contact_density,
        )

        object.__setattr__(
            self,
            "approximate_contact_area",
            approximate_contact_area,
        )

        object.__setattr__(
            self,
            "geometry_type",
            geometry_type,
        )

        object.__setattr__(
            self,
            "metadata",
            _freeze_metadata(self.metadata),
        )

    @property
    def has_contacts(self) -> bool:
        """Return whether at least one atomic contact exists."""

        return self.contact_pair_count > 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize group-contact geometry."""

        return {
            "receptor_group": (
                self.receptor_group.to_dict(
                    include_atoms=False,
                    include_descriptors=False,
                )
            ),
            "ligand_group": (
                self.ligand_group.to_dict(
                    include_atoms=False,
                    include_descriptors=False,
                )
            ),
            "centroid_distance": float(
                self.centroid_distance
            ),
            "minimum_distance": float(
                self.minimum_distance
            ),
            "mean_contact_distance": (
                None
                if self.mean_contact_distance is None
                else float(self.mean_contact_distance)
            ),
            "contact_pair_count": self.contact_pair_count,
            "receptor_contact_atom_count": (
                self.receptor_contact_atom_count
            ),
            "ligand_contact_atom_count": (
                self.ligand_contact_atom_count
            ),
            "compaction": float(self.compaction),
            "contact_density": float(
                self.contact_density
            ),
            "approximate_contact_area": float(
                self.approximate_contact_area
            ),
            "geometry_type": self.geometry_type,
            "has_contacts": self.has_contacts,
            "pi_stacking_assigned": False,
            "metadata": dict(self.metadata),
        }


# -----------------------------------------------------------------------------
# Coordinate normalization
# -----------------------------------------------------------------------------

def _normalize_coordinate(
    coordinate: Any,
    *,
    name: str = "coordinate",
) -> Coordinate:
    """
    Normalize a Cartesian coordinate into a finite float64 array.
    """

    try:
        coordinate_array = np.asarray(
            coordinate,
            dtype=np.float64,
        ).reshape(-1)
    except (
        TypeError,
        ValueError,
        OverflowError,
    ) as exc:
        raise TypeError(
            f"{name} must be an array-like Cartesian coordinate."
        ) from exc

    if coordinate_array.size != 3:
        raise ValueError(
            f"{name} must contain exactly three values."
        )

    if not np.all(np.isfinite(coordinate_array)):
        raise ValueError(
            f"{name} must contain only finite values."
        )

    return np.asarray(
        coordinate_array,
        dtype=np.float64,
    )


def get_hydrophobic_coordinate(
    value: HydrophobicCoordinateInput,
) -> Coordinate:
    """
    Resolve a coordinate from an atom, descriptor or coordinate array.
    """

    if isinstance(value, HydrophobicAtom):
        value = value.atom

    if _looks_like_atom(value):
        try:
            coordinate = get_atom_coordinate(value)
        except Exception:
            coordinate = _safe_getattr(
                value,
                (
                    "coord",
                    "coords",
                    "coordinate",
                    "coordinates",
                    "scene_coord",
                    "sceneCoord",
                ),
                default=None,
            )

        if coordinate is None:
            raise ValueError(
                "No coordinate could be resolved from the atom."
            )

        return _normalize_coordinate(
            coordinate,
            name="atom coordinate",
        )

    return _normalize_coordinate(
        value,
        name="coordinate",
    )


def hydrophobic_atom_coordinates(
    values: Iterable[
        Union[
            AtomLike,
            HydrophobicAtom,
        ]
    ],
    *,
    allow_empty: bool = True,
    skip_invalid: bool = False,
) -> CoordinateCollection:
    """
    Return an ``N × 3`` coordinate matrix.

    Parameters
    ----------
    values
        Atoms or hydrophobic descriptors.
    allow_empty
        Allow an empty ``(0, 3)`` array.
    skip_invalid
        Omit entries whose coordinates cannot be resolved.
    """

    coordinates: List[Coordinate] = []

    for index, value in enumerate(values):
        try:
            coordinate = get_hydrophobic_coordinate(
                value
            )
        except (
            TypeError,
            ValueError,
            AttributeError,
        ):
            if skip_invalid:
                continue

            raise ValueError(
                "Could not resolve the coordinate of entry "
                f"{index}."
            )

        coordinates.append(coordinate)

    if not coordinates:
        if not allow_empty:
            raise ValueError(
                "The coordinate collection cannot be empty."
            )

        return np.empty(
            (0, 3),
            dtype=np.float64,
        )

    return np.asarray(
        coordinates,
        dtype=np.float64,
    ).reshape(-1, 3)


# -----------------------------------------------------------------------------
# Basic distances
# -----------------------------------------------------------------------------

def hydrophobic_distance(
    first: HydrophobicCoordinateInput,
    second: HydrophobicCoordinateInput,
) -> np.float64:
    """
    Calculate the Euclidean distance between two atoms or coordinates.

    The implementation attempts to use :func:`geometry.distance` first
    and falls back to a direct NumPy calculation.
    """

    first_coordinate = get_hydrophobic_coordinate(
        first
    )

    second_coordinate = get_hydrophobic_coordinate(
        second
    )

    try:
        measured_distance = distance(
            first_coordinate,
            second_coordinate,
        )
    except Exception:
        measured_distance = np.linalg.norm(
            first_coordinate
            - second_coordinate
        )

    try:
        normalized_distance = _nonnegative_float(
            measured_distance,
            name="hydrophobic distance",
        )
    except (TypeError, ValueError):
        normalized_distance = np.float64(
            np.linalg.norm(
                first_coordinate
                - second_coordinate
            )
        )

    return normalized_distance


def hydrophobic_distance_matrix(
    first_group: HydrophobicGroupInput,
    second_group: Optional[HydrophobicGroupInput] = None,
) -> DistanceMatrix:
    """
    Calculate a vectorized distance matrix between atom groups.

    When ``second_group`` is omitted, a symmetric intragroup matrix is
    returned.
    """

    first_coordinates = hydrophobic_atom_coordinates(
        first_group,
        allow_empty=True,
    )

    second_coordinates = (
        first_coordinates
        if second_group is None
        else hydrophobic_atom_coordinates(
            second_group,
            allow_empty=True,
        )
    )

    if (
        first_coordinates.shape[0] == 0
        or second_coordinates.shape[0] == 0
    ):
        return np.empty(
            (
                first_coordinates.shape[0],
                second_coordinates.shape[0],
            ),
            dtype=np.float64,
        )

    coordinate_differences = (
        first_coordinates[:, np.newaxis, :]
        - second_coordinates[np.newaxis, :, :]
    )

    squared_distances = np.einsum(
        "ijk,ijk->ij",
        coordinate_differences,
        coordinate_differences,
    )

    squared_distances = np.maximum(
        squared_distances,
        0.0,
    )

    return np.asarray(
        np.sqrt(squared_distances),
        dtype=np.float64,
    )


# -----------------------------------------------------------------------------
# Centroids and group dimensions
# -----------------------------------------------------------------------------

def hydrophobic_centroid(
    group: HydrophobicGroupInput,
) -> Coordinate:
    """
    Calculate the arithmetic centroid of an atom group.
    """

    coordinates = hydrophobic_atom_coordinates(
        group,
        allow_empty=False,
    )

    return np.asarray(
        np.mean(
            coordinates,
            axis=0,
        ),
        dtype=np.float64,
    )


def hydrophobic_centroid_distance(
    first_group: HydrophobicGroupInput,
    second_group: HydrophobicGroupInput,
) -> np.float64:
    """
    Calculate the distance between the centroids of two groups.
    """

    return hydrophobic_distance(
        hydrophobic_centroid(first_group),
        hydrophobic_centroid(second_group),
    )


def minimum_group_distance(
    first_group: HydrophobicGroupInput,
    second_group: HydrophobicGroupInput,
) -> np.float64:
    """
    Return the smallest atom-to-atom distance between two groups.
    """

    distance_matrix = hydrophobic_distance_matrix(
        first_group,
        second_group,
    )

    if distance_matrix.size == 0:
        raise ValueError(
            "Both groups must contain at least one valid atom."
        )

    return np.float64(
        np.min(distance_matrix)
    )


def maximum_group_distance(
    first_group: HydrophobicGroupInput,
    second_group: HydrophobicGroupInput,
) -> np.float64:
    """
    Return the largest atom-to-atom distance between two groups.
    """

    distance_matrix = hydrophobic_distance_matrix(
        first_group,
        second_group,
    )

    if distance_matrix.size == 0:
        raise ValueError(
            "Both groups must contain at least one valid atom."
        )

    return np.float64(
        np.max(distance_matrix)
    )


def mean_group_distance(
    first_group: HydrophobicGroupInput,
    second_group: HydrophobicGroupInput,
) -> np.float64:
    """
    Return the mean atom-to-atom distance between two groups.
    """

    distance_matrix = hydrophobic_distance_matrix(
        first_group,
        second_group,
    )

    if distance_matrix.size == 0:
        raise ValueError(
            "Both groups must contain at least one valid atom."
        )

    return np.float64(
        np.mean(distance_matrix)
    )


def radius_of_gyration(
    group: HydrophobicGroupInput,
) -> np.float64:
    """
    Calculate the unweighted radius of gyration of an atom group.
    """

    coordinates = hydrophobic_atom_coordinates(
        group,
        allow_empty=False,
    )

    centroid = np.mean(
        coordinates,
        axis=0,
    )

    squared_distances = np.sum(
        (
            coordinates
            - centroid
        ) ** 2,
        axis=1,
    )

    return np.float64(
        np.sqrt(
            np.mean(squared_distances)
        )
    )


def maximum_centroid_distance(
    group: HydrophobicGroupInput,
) -> np.float64:
    """
    Return the largest distance from the centroid to any group atom.
    """

    coordinates = hydrophobic_atom_coordinates(
        group,
        allow_empty=False,
    )

    centroid = np.mean(
        coordinates,
        axis=0,
    )

    atom_distances = np.linalg.norm(
        coordinates
        - centroid,
        axis=1,
    )

    return np.float64(
        np.max(atom_distances)
    )


# -----------------------------------------------------------------------------
# Group preparation
# -----------------------------------------------------------------------------

def _normalize_geometry_descriptors(
    group: HydrophobicGroupInput,
    *,
    role: HydrophobicAtomRole = HYDROPHOBIC_ROLE_UNKNOWN,
) -> Tuple[
    Tuple[AtomLike, ...],
    Tuple[HydrophobicAtom, ...],
]:
    """
    Normalize atoms and descriptors used in geometric calculations.
    """

    values = tuple(group)

    atoms: List[AtomLike] = []
    descriptors: List[HydrophobicAtom] = []

    for value in values:
        if isinstance(value, HydrophobicAtom):
            descriptors.append(value)
            atoms.append(value.atom)
            continue

        if _looks_like_atom(value):
            atoms.append(value)

    atoms_tuple = deduplicate_atoms(
        atoms,
        strategy="auto",
    )

    descriptor_map = {
        id(descriptor.atom): descriptor
        for descriptor in descriptors
    }

    for atom in atoms_tuple:
        if id(atom) in descriptor_map:
            continue

        try:
            descriptor = perceive_hydrophobic_atom(
                atom,
                role=role,
            )
        except Exception:
            continue

        descriptor_map[id(atom)] = descriptor

    descriptors_tuple = tuple(
        descriptor_map[id(atom)]
        for atom in atoms_tuple
        if id(atom) in descriptor_map
    )

    return (
        atoms_tuple,
        descriptors_tuple,
    )


def describe_hydrophobic_group_geometry(
    group: HydrophobicGroupInput,
    *,
    role: HydrophobicAtomRole = HYDROPHOBIC_ROLE_UNKNOWN,
    metadata: Optional[Mapping[str, Any]] = None,
) -> HydrophobicGroupGeometry:
    """
    Build a geometric description of a hydrophobic atom group.
    """

    atoms, descriptors = _normalize_geometry_descriptors(
        group,
        role=role,
    )

    if not atoms:
        raise ValueError(
            "Cannot describe an empty hydrophobic group."
        )

    coordinates = hydrophobic_atom_coordinates(
        atoms,
        allow_empty=False,
    )

    centroid = np.mean(
        coordinates,
        axis=0,
    )

    minimum_coordinate = np.min(
        coordinates,
        axis=0,
    )

    maximum_coordinate = np.max(
        coordinates,
        axis=0,
    )

    centered_coordinates = (
        coordinates
        - centroid
    )

    centroid_distances = np.linalg.norm(
        centered_coordinates,
        axis=1,
    )

    aromatic_count = sum(
        descriptor.is_aromatic
        for descriptor in descriptors
    )

    aliphatic_count = sum(
        descriptor.is_aliphatic
        for descriptor in descriptors
    )

    group_metadata: Dict[str, Any] = (
        {} if metadata is None else dict(metadata)
    )

    group_metadata.update(
        {
            "coordinate_count": int(
                coordinates.shape[0]
            ),
            "hydrophobic_descriptor_count": sum(
                descriptor.is_hydrophobic
                for descriptor in descriptors
            ),
        }
    )

    return HydrophobicGroupGeometry(
        atoms=atoms,
        descriptors=descriptors,
        centroid=np.asarray(
            centroid,
            dtype=np.float64,
        ),
        minimum_coordinate=np.asarray(
            minimum_coordinate,
            dtype=np.float64,
        ),
        maximum_coordinate=np.asarray(
            maximum_coordinate,
            dtype=np.float64,
        ),
        radius_of_gyration=np.float64(
            np.sqrt(
                np.mean(
                    centroid_distances ** 2
                )
            )
        ),
        maximum_centroid_distance=np.float64(
            np.max(centroid_distances)
        ),
        aromatic_atom_count=aromatic_count,
        aliphatic_atom_count=aliphatic_count,
        metadata=group_metadata,
    )


# -----------------------------------------------------------------------------
# Contact masks and pair indices
# -----------------------------------------------------------------------------

def hydrophobic_contact_mask(
    first_group: HydrophobicGroupInput,
    second_group: HydrophobicGroupInput,
    *,
    maximum_distance: Optional[Number] = None,
    minimum_distance: Optional[Number] = None,
    tolerance: Optional[Number] = None,
) -> BooleanArray:
    """
    Return a Boolean matrix identifying geometrically valid contacts.
    """

    maximum_cutoff = (
        get_default_maximum_hydrophobic_distance()
        if maximum_distance is None
        else _positive_float(
            maximum_distance,
            name="maximum hydrophobic contact distance",
        )
    )

    minimum_cutoff = (
        get_default_minimum_hydrophobic_distance()
        if minimum_distance is None
        else _nonnegative_float(
            minimum_distance,
            name="minimum hydrophobic contact distance",
        )
    )

    distance_tolerance = (
        get_default_hydrophobic_distance_tolerance()
        if tolerance is None
        else _nonnegative_float(
            tolerance,
            name="hydrophobic distance tolerance",
        )
    )

    if minimum_cutoff > maximum_cutoff:
        raise ValueError(
            "minimum_distance cannot exceed maximum_distance."
        )

    distance_matrix = hydrophobic_distance_matrix(
        first_group,
        second_group,
    )

    lower_limit = np.maximum(
        minimum_cutoff
        - distance_tolerance,
        0.0,
    )

    upper_limit = (
        maximum_cutoff
        + distance_tolerance
    )

    return np.asarray(
        (
            distance_matrix >= lower_limit
        )
        & (
            distance_matrix <= upper_limit
        ),
        dtype=np.bool_,
    )


def hydrophobic_contact_indices(
    first_group: HydrophobicGroupInput,
    second_group: HydrophobicGroupInput,
    *,
    maximum_distance: Optional[Number] = None,
    minimum_distance: Optional[Number] = None,
    tolerance: Optional[Number] = None,
) -> Tuple[IndexedAtomPair, ...]:
    """
    Return indices of atom pairs satisfying the distance criteria.
    """

    contact_mask = hydrophobic_contact_mask(
        first_group,
        second_group,
        maximum_distance=maximum_distance,
        minimum_distance=minimum_distance,
        tolerance=tolerance,
    )

    row_indices, column_indices = np.nonzero(
        contact_mask
    )

    return tuple(
        (
            int(row_index),
            int(column_index),
        )
        for row_index, column_index
        in zip(
            row_indices,
            column_indices,
        )
    )


def hydrophobic_contact_pairs(
    first_group: HydrophobicGroupInput,
    second_group: HydrophobicGroupInput,
    *,
    maximum_distance: Optional[Number] = None,
    minimum_distance: Optional[Number] = None,
    tolerance: Optional[Number] = None,
) -> Tuple[AtomPair, ...]:
    """
    Return atom pairs satisfying hydrophobic distance criteria.
    """

    first_atoms, _ = _normalize_geometry_descriptors(
        first_group
    )

    second_atoms, _ = _normalize_geometry_descriptors(
        second_group
    )

    pair_indices = hydrophobic_contact_indices(
        first_atoms,
        second_atoms,
        maximum_distance=maximum_distance,
        minimum_distance=minimum_distance,
        tolerance=tolerance,
    )

    return tuple(
        (
            first_atoms[first_index],
            second_atoms[second_index],
        )
        for first_index, second_index
        in pair_indices
    )


# -----------------------------------------------------------------------------
# Local-neighborhood geometry
# -----------------------------------------------------------------------------

def local_hydrophobic_neighbors(
    atom: Union[
        AtomLike,
        HydrophobicAtom,
    ],
    candidates: HydrophobicGroupInput,
    *,
    radius: Optional[Number] = None,
    include_self: bool = False,
) -> Tuple[AtomLike, ...]:
    """
    Return candidate atoms located within a local spatial radius.
    """

    local_radius = (
        DEFAULT_LOCAL_CONTACT_RADIUS
        if radius is None
        else _positive_float(
            radius,
            name="local contact radius",
        )
    )

    central_atom = (
        atom.atom
        if isinstance(atom, HydrophobicAtom)
        else atom
    )

    candidate_atoms, _ = _normalize_geometry_descriptors(
        candidates
    )

    neighbors: List[AtomLike] = []

    for candidate in candidate_atoms:
        if (
            not include_self
            and candidate is central_atom
        ):
            continue

        candidate_distance = hydrophobic_distance(
            central_atom,
            candidate,
        )

        if candidate_distance <= local_radius:
            neighbors.append(candidate)

    return tuple(neighbors)


def count_local_hydrophobic_neighbors(
    atom: Union[
        AtomLike,
        HydrophobicAtom,
    ],
    candidates: HydrophobicGroupInput,
    *,
    radius: Optional[Number] = None,
    include_self: bool = False,
) -> int:
    """
    Count neighboring atoms within a local spatial radius.
    """

    return len(
        local_hydrophobic_neighbors(
            atom,
            candidates,
            radius=radius,
            include_self=include_self,
        )
    )


def local_hydrophobic_centroid(
    atom: Union[
        AtomLike,
        HydrophobicAtom,
    ],
    candidates: HydrophobicGroupInput,
    *,
    radius: Optional[Number] = None,
    include_central_atom: bool = True,
) -> Coordinate:
    """
    Calculate the centroid of the local hydrophobic neighborhood.
    """

    central_atom = (
        atom.atom
        if isinstance(atom, HydrophobicAtom)
        else atom
    )

    neighbors = list(
        local_hydrophobic_neighbors(
            central_atom,
            candidates,
            radius=radius,
            include_self=False,
        )
    )

    if include_central_atom:
        neighbors.insert(
            0,
            central_atom,
        )

    if not neighbors:
        return get_hydrophobic_coordinate(
            central_atom
        )

    return hydrophobic_centroid(
        neighbors
    )


def count_cross_group_contacts(
    first_group: HydrophobicGroupInput,
    second_group: HydrophobicGroupInput,
    *,
    maximum_distance: Optional[Number] = None,
    minimum_distance: Optional[Number] = None,
) -> int:
    """
    Count atom-to-atom contacts between two groups.
    """

    return len(
        hydrophobic_contact_indices(
            first_group,
            second_group,
            maximum_distance=maximum_distance,
            minimum_distance=minimum_distance,
        )
    )


# -----------------------------------------------------------------------------
# Contact compaction
# -----------------------------------------------------------------------------

def distance_compaction_score(
    measured_distance: Number,
    *,
    reference_distance: Optional[Number] = None,
    minimum_distance: Optional[Number] = None,
) -> np.float64:
    """
    Convert distance into a normalized contact-compaction score.

    The score approaches one near the minimum accepted distance and
    approaches zero at or beyond the reference distance.
    """

    measured = _nonnegative_float(
        measured_distance,
        name="measured distance",
    )

    reference = (
        DEFAULT_COMPACTION_REFERENCE_DISTANCE
        if reference_distance is None
        else _positive_float(
            reference_distance,
            name="compaction reference distance",
        )
    )

    minimum = (
        get_default_minimum_hydrophobic_distance()
        if minimum_distance is None
        else _nonnegative_float(
            minimum_distance,
            name="minimum hydrophobic distance",
        )
    )

    if minimum >= reference:
        raise ValueError(
            "minimum_distance must be smaller than "
            "reference_distance."
        )

    if measured <= minimum:
        return np.float64(1.0)

    if measured >= reference:
        return np.float64(0.0)

    normalized_score = (
        reference
        - measured
    ) / (
        reference
        - minimum
    )

    return validate_hydrophobic_score(
        normalized_score
    )


def group_compaction_score(
    first_group: HydrophobicGroupInput,
    second_group: HydrophobicGroupInput,
    *,
    maximum_distance: Optional[Number] = None,
    minimum_distance: Optional[Number] = None,
) -> np.float64:
    """
    Estimate how tightly two groups form a hydrophobic contact.

    The score combines:

    - minimum intergroup distance;
    - mean distance of contacting pairs;
    - fraction of possible atom pairs that are in contact.
    """

    first_atoms, _ = _normalize_geometry_descriptors(
        first_group
    )

    second_atoms, _ = _normalize_geometry_descriptors(
        second_group
    )

    if not first_atoms or not second_atoms:
        return np.float64(0.0)

    maximum_cutoff = (
        get_default_maximum_hydrophobic_distance()
        if maximum_distance is None
        else _positive_float(
            maximum_distance,
            name="maximum contact distance",
        )
    )

    minimum_cutoff = (
        get_default_minimum_hydrophobic_distance()
        if minimum_distance is None
        else _nonnegative_float(
            minimum_distance,
            name="minimum contact distance",
        )
    )

    distance_matrix = hydrophobic_distance_matrix(
        first_atoms,
        second_atoms,
    )

    contact_mask = (
        distance_matrix >= minimum_cutoff
    ) & (
        distance_matrix <= maximum_cutoff
    )

    if not np.any(contact_mask):
        return np.float64(0.0)

    contact_distances = distance_matrix[
        contact_mask
    ]

    minimum_distance_score = distance_compaction_score(
        np.min(contact_distances),
        reference_distance=maximum_cutoff,
        minimum_distance=minimum_cutoff,
    )

    mean_distance_score = distance_compaction_score(
        np.mean(contact_distances),
        reference_distance=maximum_cutoff,
        minimum_distance=minimum_cutoff,
    )

    possible_pair_count = (
        len(first_atoms)
        * len(second_atoms)
    )

    contact_fraction = (
        contact_distances.size
        / possible_pair_count
    )

    combined_score = (
        0.45 * float(minimum_distance_score)
        + 0.35 * float(mean_distance_score)
        + 0.20 * float(contact_fraction)
    )

    return validate_hydrophobic_score(
        combined_score
    )


# -----------------------------------------------------------------------------
# Contact density
# -----------------------------------------------------------------------------

def contact_pair_density(
    first_group: HydrophobicGroupInput,
    second_group: HydrophobicGroupInput,
    *,
    maximum_distance: Optional[Number] = None,
    minimum_distance: Optional[Number] = None,
) -> np.float64:
    """
    Return the fraction of possible cross-group pairs in contact.

    This metric lies in ``[0, 1]``.
    """

    first_atoms, _ = _normalize_geometry_descriptors(
        first_group
    )

    second_atoms, _ = _normalize_geometry_descriptors(
        second_group
    )

    possible_pair_count = (
        len(first_atoms)
        * len(second_atoms)
    )

    if possible_pair_count == 0:
        return np.float64(0.0)

    contact_count = count_cross_group_contacts(
        first_atoms,
        second_atoms,
        maximum_distance=maximum_distance,
        minimum_distance=minimum_distance,
    )

    return validate_hydrophobic_score(
        contact_count
        / possible_pair_count
    )


def contacted_atom_fraction(
    first_group: HydrophobicGroupInput,
    second_group: HydrophobicGroupInput,
    *,
    maximum_distance: Optional[Number] = None,
    minimum_distance: Optional[Number] = None,
) -> np.float64:
    """
    Return the mean fraction of contacted atoms on both sides.
    """

    first_atoms, _ = _normalize_geometry_descriptors(
        first_group
    )

    second_atoms, _ = _normalize_geometry_descriptors(
        second_group
    )

    if not first_atoms or not second_atoms:
        return np.float64(0.0)

    pair_indices = hydrophobic_contact_indices(
        first_atoms,
        second_atoms,
        maximum_distance=maximum_distance,
        minimum_distance=minimum_distance,
    )

    if not pair_indices:
        return np.float64(0.0)

    contacted_first_indices = {
        first_index
        for first_index, _
        in pair_indices
    }

    contacted_second_indices = {
        second_index
        for _, second_index
        in pair_indices
    }

    first_fraction = (
        len(contacted_first_indices)
        / len(first_atoms)
    )

    second_fraction = (
        len(contacted_second_indices)
        / len(second_atoms)
    )

    return validate_hydrophobic_score(
        (
            first_fraction
            + second_fraction
        ) / 2.0
    )


def approximate_contact_density(
    first_group: HydrophobicGroupInput,
    second_group: HydrophobicGroupInput,
    *,
    maximum_distance: Optional[Number] = None,
    minimum_distance: Optional[Number] = None,
) -> np.float64:
    """
    Estimate normalized local contact density.

    The metric combines pair density and the fractions of atoms involved
    in at least one contact.
    """

    pair_density = contact_pair_density(
        first_group,
        second_group,
        maximum_distance=maximum_distance,
        minimum_distance=minimum_distance,
    )

    atom_fraction = contacted_atom_fraction(
        first_group,
        second_group,
        maximum_distance=maximum_distance,
        minimum_distance=minimum_distance,
    )

    return validate_hydrophobic_score(
        0.60 * float(pair_density)
        + 0.40 * float(atom_fraction)
    )


# -----------------------------------------------------------------------------
# Approximate contact area
# -----------------------------------------------------------------------------

def approximate_pair_contact_area(
    measured_distance: Number,
    *,
    atom_radius: Number = DEFAULT_CONTACT_AREA_ATOM_RADIUS,
    maximum_distance: Optional[Number] = None,
) -> np.float64:
    """
    Estimate a relative contact area for one atom pair.

    This is a geometric approximation rather than a solvent-accessible
    surface-area calculation. Two equal spheres are assumed, and the
    overlap depth is converted into a circular contact-cap area.

    The result is expressed in approximate square angstroms.
    """

    pair_distance = _nonnegative_float(
        measured_distance,
        name="pair distance",
    )

    radius = _positive_float(
        atom_radius,
        name="contact atom radius",
    )

    maximum_cutoff = (
        get_default_maximum_hydrophobic_distance()
        if maximum_distance is None
        else _positive_float(
            maximum_distance,
            name="maximum contact distance",
        )
    )

    if pair_distance >= maximum_cutoff:
        return np.float64(
            DEFAULT_MINIMUM_CONTACT_AREA
        )

    normalized_overlap = np.clip(
        (
            maximum_cutoff
            - pair_distance
        ) / maximum_cutoff,
        0.0,
        1.0,
    )

    maximum_disc_area = (
        np.pi
        * radius ** 2
    )

    estimated_area = (
        maximum_disc_area
        * normalized_overlap
    )

    return np.float64(
        np.clip(
            estimated_area,
            DEFAULT_MINIMUM_CONTACT_AREA,
            DEFAULT_MAXIMUM_CONTACT_AREA_PER_PAIR,
        )
    )


def approximate_group_contact_area(
    first_group: HydrophobicGroupInput,
    second_group: HydrophobicGroupInput,
    *,
    maximum_distance: Optional[Number] = None,
    minimum_distance: Optional[Number] = None,
    atom_radius: Number = DEFAULT_CONTACT_AREA_ATOM_RADIUS,
) -> np.float64:
    """
    Sum approximate contact areas for valid cross-group atom pairs.

    The result may overestimate the physical surface because overlapping
    atom-pair contact discs are not explicitly subtracted.
    """

    maximum_cutoff = (
        get_default_maximum_hydrophobic_distance()
        if maximum_distance is None
        else _positive_float(
            maximum_distance,
            name="maximum contact distance",
        )
    )

    minimum_cutoff = (
        get_default_minimum_hydrophobic_distance()
        if minimum_distance is None
        else _nonnegative_float(
            minimum_distance,
            name="minimum contact distance",
        )
    )

    distance_matrix = hydrophobic_distance_matrix(
        first_group,
        second_group,
    )

    if distance_matrix.size == 0:
        return np.float64(0.0)

    contact_mask = (
        distance_matrix >= minimum_cutoff
    ) & (
        distance_matrix <= maximum_cutoff
    )

    contact_distances = distance_matrix[
        contact_mask
    ]

    if contact_distances.size == 0:
        return np.float64(0.0)

    pair_areas = [
        approximate_pair_contact_area(
            measured_distance,
            atom_radius=atom_radius,
            maximum_distance=maximum_cutoff,
        )
        for measured_distance in contact_distances
    ]

    return np.float64(
        np.sum(pair_areas)
    )


# -----------------------------------------------------------------------------
# Contact-type classification
# -----------------------------------------------------------------------------

def classify_hydrophobic_geometry_type(
    receptor: Union[
        AtomLike,
        HydrophobicAtom,
        HydrophobicGroupGeometry,
        Sequence[
            Union[
                AtomLike,
                HydrophobicAtom,
            ]
        ],
    ],
    ligand: Union[
        AtomLike,
        HydrophobicAtom,
        HydrophobicGroupGeometry,
        Sequence[
            Union[
                AtomLike,
                HydrophobicAtom,
            ]
        ],
    ],
) -> HydrophobicInteractionType:
    """
    Classify the chemical geometry of a hydrophobic contact.

    Aromatic–aromatic classification means only that both sides have
    aromatic character. No π-stacking assignment is made here.
    """

    def resolve_character(
        value: Any,
    ) -> Tuple[bool, bool]:
        if isinstance(
            value,
            HydrophobicGroupGeometry,
        ):
            return (
                value.aromatic_atom_count > 0,
                value.aliphatic_atom_count > 0,
            )

        if isinstance(
            value,
            HydrophobicAtom,
        ):
            return (
                value.is_aromatic,
                value.is_aliphatic,
            )

        if _looks_like_atom(value):
            return (
                is_aromatic_atom(value),
                is_aliphatic_atom(value),
            )

        values = tuple(value)

        aromatic = False
        aliphatic = False

        for item in values:
            item_aromatic, item_aliphatic = (
                resolve_character(item)
            )

            aromatic = (
                aromatic
                or item_aromatic
            )

            aliphatic = (
                aliphatic
                or item_aliphatic
            )

        return (
            aromatic,
            aliphatic,
        )

    receptor_aromatic, receptor_aliphatic = (
        resolve_character(receptor)
    )

    ligand_aromatic, ligand_aliphatic = (
        resolve_character(ligand)
    )

    receptor_mixed = (
        receptor_aromatic
        and receptor_aliphatic
    )

    ligand_mixed = (
        ligand_aromatic
        and ligand_aliphatic
    )

    if receptor_mixed or ligand_mixed:
        return HYDROPHOBIC_TYPE_MIXED

    if (
        receptor_aromatic
        and ligand_aromatic
    ):
        return (
            HYDROPHOBIC_TYPE_AROMATIC_AROMATIC
        )

    if (
        receptor_aliphatic
        and ligand_aromatic
    ):
        return (
            HYDROPHOBIC_TYPE_ALIPHATIC_AROMATIC
        )

    if (
        receptor_aromatic
        and ligand_aliphatic
    ):
        return (
            HYDROPHOBIC_TYPE_AROMATIC_ALIPHATIC
        )

    if (
        receptor_aliphatic
        and ligand_aliphatic
    ):
        return (
            HYDROPHOBIC_TYPE_ALIPHATIC_ALIPHATIC
        )

    return HYDROPHOBIC_TYPE_UNKNOWN


def is_aliphatic_aliphatic_contact(
    receptor: Union[
        AtomLike,
        HydrophobicAtom,
    ],
    ligand: Union[
        AtomLike,
        HydrophobicAtom,
    ],
) -> bool:
    """Return whether both sides are aliphatic."""

    return (
        classify_hydrophobic_geometry_type(
            receptor,
            ligand,
        )
        == HYDROPHOBIC_TYPE_ALIPHATIC_ALIPHATIC
    )


def is_aromatic_aliphatic_contact(
    receptor: Union[
        AtomLike,
        HydrophobicAtom,
    ],
    ligand: Union[
        AtomLike,
        HydrophobicAtom,
    ],
) -> bool:
    """Return whether exactly one side is aromatic."""

    return (
        classify_hydrophobic_geometry_type(
            receptor,
            ligand,
        )
        in {
            HYDROPHOBIC_TYPE_ALIPHATIC_AROMATIC,
            HYDROPHOBIC_TYPE_AROMATIC_ALIPHATIC,
        }
    )


def is_aromatic_aromatic_hydrophobic_contact(
    receptor: Union[
        AtomLike,
        HydrophobicAtom,
    ],
    ligand: Union[
        AtomLike,
        HydrophobicAtom,
    ],
) -> bool:
    """
    Return whether both sides are aromatic hydrophobic atoms.

    This function deliberately does not evaluate ring-plane orientation,
    centroid offset or angular geometry and therefore does not indicate
    π-stacking.
    """

    return (
        classify_hydrophobic_geometry_type(
            receptor,
            ligand,
        )
        == HYDROPHOBIC_TYPE_AROMATIC_AROMATIC
    )


# -----------------------------------------------------------------------------
# Pair-level geometric analysis
# -----------------------------------------------------------------------------

def analyze_hydrophobic_pair_geometry(
    receptor: Union[
        AtomLike,
        HydrophobicAtom,
    ],
    ligand: Union[
        AtomLike,
        HydrophobicAtom,
    ],
    *,
    receptor_candidates: Optional[
        HydrophobicGroupInput
    ] = None,
    ligand_candidates: Optional[
        HydrophobicGroupInput
    ] = None,
    local_radius: Optional[Number] = None,
    maximum_contact_distance: Optional[Number] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> HydrophobicPairGeometry:
    """
    Calculate detailed geometry for one hydrophobic atom pair.
    """

    receptor_descriptor = (
        receptor
        if isinstance(
            receptor,
            HydrophobicAtom,
        )
        else None
    )

    ligand_descriptor = (
        ligand
        if isinstance(
            ligand,
            HydrophobicAtom,
        )
        else None
    )

    receptor_atom = (
        receptor.atom
        if receptor_descriptor is not None
        else receptor
    )

    ligand_atom = (
        ligand.atom
        if ligand_descriptor is not None
        else ligand
    )

    pair_distance = hydrophobic_distance(
        receptor_atom,
        ligand_atom,
    )

    geometry_type = classify_hydrophobic_geometry_type(
        receptor_descriptor or receptor_atom,
        ligand_descriptor or ligand_atom,
    )

    resolved_local_radius = (
        DEFAULT_LOCAL_CONTACT_RADIUS
        if local_radius is None
        else _positive_float(
            local_radius,
            name="local radius",
        )
    )

    receptor_candidate_group = (
        (receptor_atom,)
        if receptor_candidates is None
        else receptor_candidates
    )

    ligand_candidate_group = (
        (ligand_atom,)
        if ligand_candidates is None
        else ligand_candidates
    )

    receptor_neighbors = local_hydrophobic_neighbors(
        receptor_atom,
        receptor_candidate_group,
        radius=resolved_local_radius,
        include_self=False,
    )

    ligand_neighbors = local_hydrophobic_neighbors(
        ligand_atom,
        ligand_candidate_group,
        radius=resolved_local_radius,
        include_self=False,
    )

    receptor_local_group = (
        receptor_atom,
        *receptor_neighbors,
    )

    ligand_local_group = (
        ligand_atom,
        *ligand_neighbors,
    )

    shared_local_contact_count = (
        count_cross_group_contacts(
            receptor_local_group,
            ligand_local_group,
            maximum_distance=maximum_contact_distance,
            minimum_distance=0.0,
        )
    )

    local_compaction = group_compaction_score(
        receptor_local_group,
        ligand_local_group,
        maximum_distance=maximum_contact_distance,
        minimum_distance=0.0,
    )

    contact_density = approximate_contact_density(
        receptor_local_group,
        ligand_local_group,
        maximum_distance=maximum_contact_distance,
        minimum_distance=0.0,
    )

    approximate_area = approximate_group_contact_area(
        receptor_local_group,
        ligand_local_group,
        maximum_distance=maximum_contact_distance,
        minimum_distance=0.0,
    )

    receptor_local_centroid = hydrophobic_centroid(
        receptor_local_group
    )

    ligand_local_centroid = hydrophobic_centroid(
        ligand_local_group
    )

    local_centroid_distance = hydrophobic_distance(
        receptor_local_centroid,
        ligand_local_centroid,
    )

    pair_metadata: Dict[str, Any] = (
        {} if metadata is None else dict(metadata)
    )

    pair_metadata.update(
        {
            "local_radius": float(
                resolved_local_radius
            ),
            "aromatic_aromatic_is_not_pi_stacking": (
                geometry_type
                == HYDROPHOBIC_TYPE_AROMATIC_AROMATIC
            ),
        }
    )

    return HydrophobicPairGeometry(
        receptor_atom=receptor_atom,
        ligand_atom=ligand_atom,
        receptor_descriptor=receptor_descriptor,
        ligand_descriptor=ligand_descriptor,
        distance=pair_distance,
        geometry_type=geometry_type,
        receptor_neighbor_count=len(
            receptor_neighbors
        ),
        ligand_neighbor_count=len(
            ligand_neighbors
        ),
        shared_local_contact_count=(
            shared_local_contact_count
        ),
        local_compaction=local_compaction,
        contact_density=contact_density,
        approximate_contact_area=approximate_area,
        receptor_local_centroid=(
            receptor_local_centroid
        ),
        ligand_local_centroid=(
            ligand_local_centroid
        ),
        local_centroid_distance=(
            local_centroid_distance
        ),
        metadata=pair_metadata,
    )


# -----------------------------------------------------------------------------
# Group-level geometric analysis
# -----------------------------------------------------------------------------

def analyze_hydrophobic_group_geometry(
    receptor_group: HydrophobicGroupInput,
    ligand_group: HydrophobicGroupInput,
    *,
    maximum_contact_distance: Optional[Number] = None,
    minimum_contact_distance: Optional[Number] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> HydrophobicGroupContactGeometry:
    """
    Calculate detailed geometry between two hydrophobic groups.
    """

    receptor_geometry = (
        describe_hydrophobic_group_geometry(
            receptor_group,
            role=HYDROPHOBIC_ROLE_RECEPTOR,
        )
    )

    ligand_geometry = (
        describe_hydrophobic_group_geometry(
            ligand_group,
            role=HYDROPHOBIC_ROLE_LIGAND,
        )
    )

    maximum_cutoff = (
        get_default_maximum_hydrophobic_distance()
        if maximum_contact_distance is None
        else _positive_float(
            maximum_contact_distance,
            name="maximum group contact distance",
        )
    )

    minimum_cutoff = (
        get_default_minimum_hydrophobic_distance()
        if minimum_contact_distance is None
        else _nonnegative_float(
            minimum_contact_distance,
            name="minimum group contact distance",
        )
    )

    distance_matrix = hydrophobic_distance_matrix(
        receptor_geometry.atoms,
        ligand_geometry.atoms,
    )

    contact_mask = (
        distance_matrix >= minimum_cutoff
    ) & (
        distance_matrix <= maximum_cutoff
    )

    contact_indices = np.argwhere(
        contact_mask
    )

    contact_pair_count = int(
        contact_indices.shape[0]
    )

    if contact_pair_count:
        contact_distances = distance_matrix[
            contact_mask
        ]

        mean_contact_distance = np.float64(
            np.mean(contact_distances)
        )

        receptor_contact_atom_count = len(
            set(
                int(index)
                for index
                in contact_indices[:, 0]
            )
        )

        ligand_contact_atom_count = len(
            set(
                int(index)
                for index
                in contact_indices[:, 1]
            )
        )

    else:
        mean_contact_distance = None
        receptor_contact_atom_count = 0
        ligand_contact_atom_count = 0

    centroid_distance = hydrophobic_distance(
        receptor_geometry.centroid,
        ligand_geometry.centroid,
    )

    minimum_distance = np.float64(
        np.min(distance_matrix)
    )

    compaction = group_compaction_score(
        receptor_geometry.atoms,
        ligand_geometry.atoms,
        maximum_distance=maximum_cutoff,
        minimum_distance=minimum_cutoff,
    )

    contact_density = approximate_contact_density(
        receptor_geometry.atoms,
        ligand_geometry.atoms,
        maximum_distance=maximum_cutoff,
        minimum_distance=minimum_cutoff,
    )

    approximate_area = approximate_group_contact_area(
        receptor_geometry.atoms,
        ligand_geometry.atoms,
        maximum_distance=maximum_cutoff,
        minimum_distance=minimum_cutoff,
    )

    geometry_type = classify_hydrophobic_geometry_type(
        receptor_geometry,
        ligand_geometry,
    )

    group_metadata: Dict[str, Any] = (
        {} if metadata is None else dict(metadata)
    )

    group_metadata.update(
        {
            "maximum_contact_distance": float(
                maximum_cutoff
            ),
            "minimum_contact_distance": float(
                minimum_cutoff
            ),
            "possible_pair_count": (
                len(receptor_geometry.atoms)
                * len(ligand_geometry.atoms)
            ),
            "aromatic_aromatic_is_not_pi_stacking": (
                geometry_type
                == HYDROPHOBIC_TYPE_AROMATIC_AROMATIC
            ),
        }
    )

    return HydrophobicGroupContactGeometry(
        receptor_group=receptor_geometry,
        ligand_group=ligand_geometry,
        centroid_distance=centroid_distance,
        minimum_distance=minimum_distance,
        mean_contact_distance=mean_contact_distance,
        contact_pair_count=contact_pair_count,
        receptor_contact_atom_count=(
            receptor_contact_atom_count
        ),
        ligand_contact_atom_count=(
            ligand_contact_atom_count
        ),
        compaction=compaction,
        contact_density=contact_density,
        approximate_contact_area=approximate_area,
        geometry_type=geometry_type,
        metadata=group_metadata,
    )


# -----------------------------------------------------------------------------
# Geometry compatibility checks
# -----------------------------------------------------------------------------

def is_within_hydrophobic_distance(
    receptor: HydrophobicCoordinateInput,
    ligand: HydrophobicCoordinateInput,
    *,
    minimum_distance: Optional[Number] = None,
    maximum_distance: Optional[Number] = None,
    tolerance: Optional[Number] = None,
) -> bool:
    """
    Return whether a pair satisfies hydrophobic distance limits.
    """

    minimum_cutoff = (
        get_default_minimum_hydrophobic_distance()
        if minimum_distance is None
        else _nonnegative_float(
            minimum_distance,
            name="minimum hydrophobic distance",
        )
    )

    maximum_cutoff = (
        get_default_maximum_hydrophobic_distance()
        if maximum_distance is None
        else _positive_float(
            maximum_distance,
            name="maximum hydrophobic distance",
        )
    )

    distance_tolerance = (
        get_default_hydrophobic_distance_tolerance()
        if tolerance is None
        else _nonnegative_float(
            tolerance,
            name="hydrophobic distance tolerance",
        )
    )

    if minimum_cutoff > maximum_cutoff:
        raise ValueError(
            "minimum_distance cannot exceed maximum_distance."
        )

    measured_distance = hydrophobic_distance(
        receptor,
        ligand,
    )

    return bool(
        measured_distance
        >= max(
            0.0,
            minimum_cutoff
            - distance_tolerance,
        )
        and measured_distance
        <= (
            maximum_cutoff
            + distance_tolerance
        )
    )


def is_geometrically_hydrophobic_pair(
    receptor: Union[
        AtomLike,
        HydrophobicAtom,
    ],
    ligand: Union[
        AtomLike,
        HydrophobicAtom,
    ],
    *,
    minimum_distance: Optional[Number] = None,
    maximum_distance: Optional[Number] = None,
    require_hydrophobic_atoms: bool = True,
) -> bool:
    """
    Return whether an atom pair is chemically and geometrically eligible.
    """

    receptor_descriptor = (
        receptor
        if isinstance(
            receptor,
            HydrophobicAtom,
        )
        else None
    )

    ligand_descriptor = (
        ligand
        if isinstance(
            ligand,
            HydrophobicAtom,
        )
        else None
    )

    receptor_atom = (
        receptor_descriptor.atom
        if receptor_descriptor is not None
        else receptor
    )

    ligand_atom = (
        ligand_descriptor.atom
        if ligand_descriptor is not None
        else ligand
    )

    if require_hydrophobic_atoms:
        receptor_is_hydrophobic = (
            receptor_descriptor.is_hydrophobic
            if receptor_descriptor is not None
            else is_hydrophobic_atom(
                receptor_atom,
                role=HYDROPHOBIC_ROLE_RECEPTOR,
            )
        )

        ligand_is_hydrophobic = (
            ligand_descriptor.is_hydrophobic
            if ligand_descriptor is not None
            else is_hydrophobic_atom(
                ligand_atom,
                role=HYDROPHOBIC_ROLE_LIGAND,
            )
        )

        if (
            not receptor_is_hydrophobic
            or not ligand_is_hydrophobic
        ):
            return False

    return is_within_hydrophobic_distance(
        receptor_atom,
        ligand_atom,
        minimum_distance=minimum_distance,
        maximum_distance=maximum_distance,
    )


# -----------------------------------------------------------------------------
# Section 6 public names
# -----------------------------------------------------------------------------

_SECTION_6_PUBLIC_NAMES: Final[Tuple[str, ...]] = (
    # Aliases
    "HydrophobicGeometryType",
    "HydrophobicGroupInput",
    "HydrophobicCoordinateInput",

    # Defaults
    "DEFAULT_LOCAL_CONTACT_RADIUS",
    "DEFAULT_CONTACT_DENSITY_RADIUS",
    "DEFAULT_CONTACT_AREA_ATOM_RADIUS",
    "DEFAULT_COMPACTION_REFERENCE_DISTANCE",
    "DEFAULT_DENSITY_NORMALIZATION_CONTACT_COUNT",

    # Dataclasses
    "HydrophobicGroupGeometry",
    "HydrophobicPairGeometry",
    "HydrophobicGroupContactGeometry",

    # Coordinates and distances
    "get_hydrophobic_coordinate",
    "hydrophobic_atom_coordinates",
    "hydrophobic_distance",
    "hydrophobic_distance_matrix",
    "hydrophobic_centroid",
    "hydrophobic_centroid_distance",
    "minimum_group_distance",
    "maximum_group_distance",
    "mean_group_distance",
    "radius_of_gyration",
    "maximum_centroid_distance",

    # Group geometry
    "describe_hydrophobic_group_geometry",

    # Contacts and local neighborhoods
    "hydrophobic_contact_mask",
    "hydrophobic_contact_indices",
    "hydrophobic_contact_pairs",
    "local_hydrophobic_neighbors",
    "count_local_hydrophobic_neighbors",
    "local_hydrophobic_centroid",
    "count_cross_group_contacts",

    # Compaction and density
    "distance_compaction_score",
    "group_compaction_score",
    "contact_pair_density",
    "contacted_atom_fraction",
    "approximate_contact_density",

    # Approximate contact area
    "approximate_pair_contact_area",
    "approximate_group_contact_area",

    # Chemical geometry type
    "classify_hydrophobic_geometry_type",
    "is_aliphatic_aliphatic_contact",
    "is_aromatic_aliphatic_contact",
    "is_aromatic_aromatic_hydrophobic_contact",

    # Complete analyses
    "analyze_hydrophobic_pair_geometry",
    "analyze_hydrophobic_group_geometry",

    # Eligibility
    "is_within_hydrophobic_distance",
    "is_geometrically_hydrophobic_pair",
)

for public_name in _SECTION_6_PUBLIC_NAMES:
    if public_name not in __all__:
        __all__.append(public_name)


# =============================================================================
# End of Section 6
# =============================================================================


# =============================================================================
# Section 7 — Hydrophobic-interaction detection
# =============================================================================


# -----------------------------------------------------------------------------
# Detection-related aliases
# -----------------------------------------------------------------------------

HydrophobicDescriptorPair: TypeAlias = Tuple[
    HydrophobicAtom,
    HydrophobicAtom,
]

HydrophobicDescriptorPairCollection: TypeAlias = Tuple[
    HydrophobicDescriptorPair,
    ...,
]

HydrophobicPairKey: TypeAlias = Tuple[
    Tuple[Any, ...],
    Tuple[Any, ...],
]


# -----------------------------------------------------------------------------
# Detection defaults
# -----------------------------------------------------------------------------

DEFAULT_REQUIRE_HYDROPHOBIC_DESCRIPTORS: Final[bool] = True
DEFAULT_REQUIRE_DISTINCT_ATOMS: Final[bool] = True
DEFAULT_REMOVE_DUPLICATE_PAIRS: Final[bool] = True
DEFAULT_SORT_HYDROPHOBIC_INTERACTIONS: Final[bool] = True
DEFAULT_INCLUDE_PAIR_GEOMETRY_METADATA: Final[bool] = True

DEFAULT_REJECT_ZERO_DISTANCE_PAIRS: Final[bool] = True

DEFAULT_MINIMUM_VALID_PAIR_DISTANCE: Final[np.float64] = np.float64(
    1.0e-6
)

DEFAULT_CHEMICAL_COMPATIBILITY_SCORE: Final[np.float64] = np.float64(
    1.0
)

DEFAULT_UNKNOWN_CHEMISTRY_SCORE: Final[np.float64] = np.float64(
    0.5
)


# -----------------------------------------------------------------------------
# Detection result dataclass
# -----------------------------------------------------------------------------

@dataclass(
    frozen=True,
    slots=True,
)
class HydrophobicDetectionResult:
    """
    Intermediate result produced by hydrophobic-contact detection.

    Unlike :class:`HydrophobicAnalysisResult`, this structure represents
    the output immediately after pair detection and interaction creation.
    Residue grouping and final statistics are added in later sections.

    Parameters
    ----------
    prepared_collections
        Prepared receptor and ligand atom collections.
    candidate_pairs
        Descriptor pairs satisfying the preliminary distance search.
    interactions
        Valid, deduplicated hydrophobic interactions.
    rejected_pair_count
        Number of candidate pairs rejected after detailed validation.
    duplicate_pair_count
        Number of duplicate candidate pairs removed.
    minimum_distance
        Minimum distance cutoff used during detection.
    maximum_distance
        Maximum distance cutoff used during detection.
    metadata
        Additional detection information.
    """

    prepared_collections: HydrophobicAtomCollections

    candidate_pairs: Sequence[
        HydrophobicDescriptorPair
    ] = field(
        default_factory=tuple
    )

    interactions: Sequence[
        HydrophobicInteraction
    ] = field(
        default_factory=tuple
    )

    rejected_pair_count: int = 0
    duplicate_pair_count: int = 0

    minimum_distance: np.float64 = (
        DEFAULT_MINIMUM_HYDROPHOBIC_DISTANCE
    )

    maximum_distance: np.float64 = (
        DEFAULT_MAXIMUM_HYDROPHOBIC_DISTANCE
    )

    metadata: Mapping[str, Any] = field(
        default_factory=lambda: _EMPTY_METADATA,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        """Validate and freeze the detection result."""

        if not isinstance(
            self.prepared_collections,
            HydrophobicAtomCollections,
        ):
            raise TypeError(
                "prepared_collections must be a "
                "HydrophobicAtomCollections instance."
            )

        candidate_pairs = tuple(
            self.candidate_pairs
        )

        for pair_index, pair in enumerate(
            candidate_pairs
        ):
            if (
                not isinstance(pair, tuple)
                or len(pair) != 2
            ):
                raise TypeError(
                    "Each candidate pair must be a tuple containing "
                    "two HydrophobicAtom descriptors. Invalid pair "
                    f"at index {pair_index}."
                )

            receptor_descriptor, ligand_descriptor = pair

            if not isinstance(
                receptor_descriptor,
                HydrophobicAtom,
            ):
                raise TypeError(
                    "The first member of each candidate pair must "
                    "be a HydrophobicAtom."
                )

            if not isinstance(
                ligand_descriptor,
                HydrophobicAtom,
            ):
                raise TypeError(
                    "The second member of each candidate pair must "
                    "be a HydrophobicAtom."
                )

        interactions = tuple(
            self.interactions
        )

        for interaction_index, interaction in enumerate(
            interactions
        ):
            if not isinstance(
                interaction,
                HydrophobicInteraction,
            ):
                raise TypeError(
                    "interactions must contain "
                    "HydrophobicInteraction instances. Invalid "
                    f"entry at index {interaction_index}."
                )

        rejected_pair_count = _nonnegative_integer(
            self.rejected_pair_count,
            name="rejected pair count",
        )

        duplicate_pair_count = _nonnegative_integer(
            self.duplicate_pair_count,
            name="duplicate pair count",
        )

        minimum_distance, maximum_distance = (
            validate_hydrophobic_distance_limits(
                self.minimum_distance,
                self.maximum_distance,
            )
        )

        object.__setattr__(
            self,
            "candidate_pairs",
            candidate_pairs,
        )

        object.__setattr__(
            self,
            "interactions",
            interactions,
        )

        object.__setattr__(
            self,
            "rejected_pair_count",
            rejected_pair_count,
        )

        object.__setattr__(
            self,
            "duplicate_pair_count",
            duplicate_pair_count,
        )

        object.__setattr__(
            self,
            "minimum_distance",
            minimum_distance,
        )

        object.__setattr__(
            self,
            "maximum_distance",
            maximum_distance,
        )

        object.__setattr__(
            self,
            "metadata",
            _freeze_metadata(self.metadata),
        )

    def __len__(self) -> int:
        """Return the number of detected interactions."""

        return len(self.interactions)

    def __iter__(self) -> Iterator[HydrophobicInteraction]:
        """Iterate over detected interactions."""

        return iter(self.interactions)

    @property
    def candidate_pair_count(self) -> int:
        """Return the number of preliminary candidate pairs."""

        return len(self.candidate_pairs)

    @property
    def interaction_count(self) -> int:
        """Return the number of valid interactions."""

        return len(self.interactions)

    @property
    def accepted_pair_fraction(self) -> np.float64:
        """Return the fraction of candidate pairs retained."""

        if not self.candidate_pairs:
            return np.float64(0.0)

        return np.float64(
            len(self.interactions)
            / len(self.candidate_pairs)
        )

    @property
    def has_interactions(self) -> bool:
        """Return whether at least one interaction was detected."""

        return bool(self.interactions)

    def to_dict(
        self,
        *,
        include_interactions: bool = True,
        include_candidate_pairs: bool = False,
        include_atoms: bool = False,
    ) -> Dict[str, Any]:
        """Serialize the intermediate detection result."""

        result: Dict[str, Any] = {
            "candidate_pair_count": (
                self.candidate_pair_count
            ),
            "interaction_count": self.interaction_count,
            "rejected_pair_count": (
                self.rejected_pair_count
            ),
            "duplicate_pair_count": (
                self.duplicate_pair_count
            ),
            "accepted_pair_fraction": float(
                self.accepted_pair_fraction
            ),
            "minimum_distance": float(
                self.minimum_distance
            ),
            "maximum_distance": float(
                self.maximum_distance
            ),
            "has_interactions": self.has_interactions,
            "prepared_collections": (
                self.prepared_collections.to_dict(
                    include_atoms=include_atoms,
                    include_rejected=True,
                    include_descriptors=True,
                )
            ),
            "metadata": dict(self.metadata),
        }

        if include_interactions:
            result["interactions"] = [
                interaction.to_dict(
                    include_atoms=include_atoms,
                    include_residue=include_atoms,
                    include_descriptors=True,
                )
                for interaction in self.interactions
            ]

        if include_candidate_pairs:
            result["candidate_pairs"] = [
                {
                    "receptor": receptor.to_dict(
                        include_atom=include_atoms,
                        include_residue=include_atoms,
                    ),
                    "ligand": ligand.to_dict(
                        include_atom=include_atoms,
                        include_residue=include_atoms,
                    ),
                }
                for receptor, ligand
                in self.candidate_pairs
            ]

        return result


# -----------------------------------------------------------------------------
# Pair identifiers and deduplication
# -----------------------------------------------------------------------------

def hydrophobic_descriptor_pair_key(
    receptor_descriptor: HydrophobicAtom,
    ligand_descriptor: HydrophobicAtom,
) -> HydrophobicPairKey:
    """
    Create a stable receptor–ligand descriptor-pair key.

    The orientation is preserved because receptor and ligand have
    different semantic roles.
    """

    if not isinstance(
        receptor_descriptor,
        HydrophobicAtom,
    ):
        raise TypeError(
            "receptor_descriptor must be a HydrophobicAtom."
        )

    if not isinstance(
        ligand_descriptor,
        HydrophobicAtom,
    ):
        raise TypeError(
            "ligand_descriptor must be a HydrophobicAtom."
        )

    receptor_key = atom_deduplication_key(
        receptor_descriptor.atom,
        strategy="auto",
    )

    ligand_key = atom_deduplication_key(
        ligand_descriptor.atom,
        strategy="auto",
    )

    return (
        receptor_key,
        ligand_key,
    )


def hydrophobic_interaction_pair_key(
    interaction: HydrophobicInteraction,
) -> HydrophobicPairKey:
    """
    Create a receptor–ligand atom-pair key from an interaction.
    """

    if not isinstance(
        interaction,
        HydrophobicInteraction,
    ):
        raise TypeError(
            "interaction must be a HydrophobicInteraction."
        )

    return (
        atom_deduplication_key(
            interaction.receptor_atom,
            strategy="auto",
        ),
        atom_deduplication_key(
            interaction.ligand_atom,
            strategy="auto",
        ),
    )


def deduplicate_hydrophobic_pairs(
    pairs: Iterable[
        HydrophobicDescriptorPair
    ],
) -> HydrophobicDescriptorPairCollection:
    """
    Remove duplicate descriptor pairs while preserving order.
    """

    unique_pairs: List[
        HydrophobicDescriptorPair
    ] = []

    seen_keys: Set[
        HydrophobicPairKey
    ] = set()

    for pair_index, pair in enumerate(pairs):
        try:
            receptor_descriptor, ligand_descriptor = pair
        except Exception as exc:
            raise TypeError(
                "Every pair must contain exactly two "
                "HydrophobicAtom descriptors."
            ) from exc

        key = hydrophobic_descriptor_pair_key(
            receptor_descriptor,
            ligand_descriptor,
        )

        if key in seen_keys:
            continue

        seen_keys.add(key)

        unique_pairs.append(
            (
                receptor_descriptor,
                ligand_descriptor,
            )
        )

    return tuple(unique_pairs)


def deduplicate_hydrophobic_interactions(
    interactions: Iterable[
        HydrophobicInteraction
    ],
    *,
    prefer_highest_score: bool = True,
) -> Tuple[
    HydrophobicInteraction,
    ...,
]:
    """
    Remove duplicate atom-pair interactions.

    When duplicate interactions represent the same receptor–ligand atom
    pair, the interaction with the highest score is retained by default.
    Distance is used as a secondary criterion.
    """

    interaction_map: Dict[
        HydrophobicPairKey,
        HydrophobicInteraction,
    ] = {}

    ordered_keys: List[
        HydrophobicPairKey
    ] = []

    for interaction in interactions:
        if not isinstance(
            interaction,
            HydrophobicInteraction,
        ):
            raise TypeError(
                "All entries must be HydrophobicInteraction "
                "instances."
            )

        key = hydrophobic_interaction_pair_key(
            interaction
        )

        current = interaction_map.get(key)

        if current is None:
            interaction_map[key] = interaction
            ordered_keys.append(key)
            continue

        if not prefer_highest_score:
            continue

        replace_current = bool(
            interaction.score > current.score
            or (
                np.isclose(
                    interaction.score,
                    current.score,
                )
                and interaction.distance
                < current.distance
            )
        )

        if replace_current:
            interaction_map[key] = interaction

    return tuple(
        interaction_map[key]
        for key in ordered_keys
    )


# -----------------------------------------------------------------------------
# Preliminary geometric classification
# -----------------------------------------------------------------------------

def classify_hydrophobic_distance(
    measured_distance: Number,
    *,
    maximum_distance: Optional[Number] = None,
) -> HydrophobicClassification:
    """
    Assign a preliminary geometric classification from distance.

    This is the initial classification used during detection. Section 9
    may refine it using local density, compactness and chemical context.
    """

    pair_distance = _nonnegative_float(
        measured_distance,
        name="hydrophobic distance",
    )

    maximum_cutoff = (
        get_default_maximum_hydrophobic_distance()
        if maximum_distance is None
        else _positive_float(
            maximum_distance,
            name="maximum hydrophobic distance",
        )
    )

    if (
        pair_distance
        <= HYDROPHOBIC_VERY_STRONG_MAX_DISTANCE
    ):
        return HYDROPHOBIC_CLASS_VERY_STRONG

    if (
        pair_distance
        <= HYDROPHOBIC_STRONG_MAX_DISTANCE
    ):
        return HYDROPHOBIC_CLASS_STRONG

    if (
        pair_distance
        <= HYDROPHOBIC_MODERATE_MAX_DISTANCE
    ):
        return HYDROPHOBIC_CLASS_MODERATE

    if pair_distance <= min(
        HYDROPHOBIC_WEAK_MAX_DISTANCE,
        maximum_cutoff,
    ):
        return HYDROPHOBIC_CLASS_WEAK

    if pair_distance <= max(
        HYDROPHOBIC_MARGINAL_MAX_DISTANCE,
        maximum_cutoff,
    ):
        return HYDROPHOBIC_CLASS_MARGINAL

    return HYDROPHOBIC_CLASS_UNKNOWN


def calculate_preliminary_hydrophobic_strength(
    measured_distance: Number,
    *,
    minimum_distance: Optional[Number] = None,
    maximum_distance: Optional[Number] = None,
) -> np.float64:
    """
    Calculate a normalized preliminary geometric strength.

    Strength is distance-based at this stage. The complete geometric and
    chemical score is calculated separately.
    """

    minimum_cutoff = (
        get_default_minimum_hydrophobic_distance()
        if minimum_distance is None
        else _nonnegative_float(
            minimum_distance,
            name="minimum hydrophobic distance",
        )
    )

    maximum_cutoff = (
        get_default_maximum_hydrophobic_distance()
        if maximum_distance is None
        else _positive_float(
            maximum_distance,
            name="maximum hydrophobic distance",
        )
    )

    if minimum_cutoff >= maximum_cutoff:
        raise ValueError(
            "minimum_distance must be smaller than "
            "maximum_distance."
        )

    return distance_compaction_score(
        measured_distance,
        reference_distance=maximum_cutoff,
        minimum_distance=minimum_cutoff,
    )


def hydrophobic_chemical_compatibility_score(
    interaction_type: HydrophobicInteractionType,
) -> np.float64:
    """
    Return the preliminary chemical compatibility score.
    """

    normalized_type = (
        validate_hydrophobic_interaction_type(
            interaction_type
        )
    )

    modifier = (
        HYDROPHOBIC_INTERACTION_TYPE_SCORE_MODIFIERS.get(
            normalized_type,
            DEFAULT_UNKNOWN_CHEMISTRY_SCORE,
        )
    )

    return validate_hydrophobic_score(
        modifier
    )


def calculate_preliminary_hydrophobic_score(
    geometry: HydrophobicPairGeometry,
    *,
    strength: Optional[Number] = None,
) -> np.float64:
    """
    Calculate the initial combined score for a detected interaction.

    The score combines:

    - atom-pair geometric strength;
    - local contact density;
    - chemical compatibility.

    Section 9 may replace or refine this preliminary score.
    """

    if not isinstance(
        geometry,
        HydrophobicPairGeometry,
    ):
        raise TypeError(
            "geometry must be a HydrophobicPairGeometry."
        )

    geometric_strength = (
        calculate_preliminary_hydrophobic_strength(
            geometry.distance
        )
        if strength is None
        else validate_hydrophobic_score(
            strength
        )
    )

    chemical_score = (
        hydrophobic_chemical_compatibility_score(
            geometry.geometry_type
        )
    )

    weighted_score = (
        float(HYDROPHOBIC_DISTANCE_SCORE_WEIGHT)
        * float(geometric_strength)
        + float(HYDROPHOBIC_DENSITY_SCORE_WEIGHT)
        * float(geometry.contact_density)
        + float(HYDROPHOBIC_CHEMISTRY_SCORE_WEIGHT)
        * float(chemical_score)
    )

    return validate_hydrophobic_score(
        weighted_score
    )


# -----------------------------------------------------------------------------
# Descriptor-pair validation
# -----------------------------------------------------------------------------

def hydrophobic_pair_exclusion_reasons(
    receptor_descriptor: HydrophobicAtom,
    ligand_descriptor: HydrophobicAtom,
    *,
    minimum_distance: Optional[Number] = None,
    maximum_distance: Optional[Number] = None,
    require_hydrophobic_descriptors: bool = (
        DEFAULT_REQUIRE_HYDROPHOBIC_DESCRIPTORS
    ),
    require_distinct_atoms: bool = (
        DEFAULT_REQUIRE_DISTINCT_ATOMS
    ),
    reject_zero_distance: bool = (
        DEFAULT_REJECT_ZERO_DISTANCE_PAIRS
    ),
) -> Tuple[str, ...]:
    """
    Return reasons why a descriptor pair must be rejected.

    An empty tuple means the pair is valid for detailed detection.
    """

    reasons: List[str] = []

    if not isinstance(
        receptor_descriptor,
        HydrophobicAtom,
    ):
        return (
            "invalid_receptor_descriptor",
        )

    if not isinstance(
        ligand_descriptor,
        HydrophobicAtom,
    ):
        return (
            "invalid_ligand_descriptor",
        )

    receptor_atom = receptor_descriptor.atom
    ligand_atom = ligand_descriptor.atom

    if receptor_atom is None:
        reasons.append(
            "missing_receptor_atom"
        )

    if ligand_atom is None:
        reasons.append(
            "missing_ligand_atom"
        )

    if reasons:
        return tuple(reasons)

    if (
        require_distinct_atoms
        and receptor_atom is ligand_atom
    ):
        reasons.append(
            "identical_receptor_and_ligand_atom"
        )

    if require_hydrophobic_descriptors:
        if not receptor_descriptor.is_hydrophobic:
            reasons.append(
                "nonhydrophobic_receptor_atom"
            )

        if not ligand_descriptor.is_hydrophobic:
            reasons.append(
                "nonhydrophobic_ligand_atom"
            )

    if receptor_descriptor.role not in {
        HYDROPHOBIC_ROLE_RECEPTOR,
        HYDROPHOBIC_ROLE_UNKNOWN,
    }:
        reasons.append(
            "invalid_receptor_role"
        )

    if ligand_descriptor.role not in {
        HYDROPHOBIC_ROLE_LIGAND,
        HYDROPHOBIC_ROLE_UNKNOWN,
    }:
        reasons.append(
            "invalid_ligand_role"
        )

    try:
        pair_distance = hydrophobic_distance(
            receptor_atom,
            ligand_atom,
        )
    except Exception:
        reasons.append(
            "invalid_pair_coordinates"
        )

        return tuple(
            dict.fromkeys(reasons)
        )

    if (
        reject_zero_distance
        and pair_distance
        <= DEFAULT_MINIMUM_VALID_PAIR_DISTANCE
    ):
        reasons.append(
            "zero_or_near_zero_distance"
        )

    minimum_cutoff = (
        get_default_minimum_hydrophobic_distance()
        if minimum_distance is None
        else _nonnegative_float(
            minimum_distance,
            name="minimum hydrophobic distance",
        )
    )

    maximum_cutoff = (
        get_default_maximum_hydrophobic_distance()
        if maximum_distance is None
        else _positive_float(
            maximum_distance,
            name="maximum hydrophobic distance",
        )
    )

    if minimum_cutoff > maximum_cutoff:
        raise ValueError(
            "minimum_distance cannot exceed maximum_distance."
        )

    tolerance = (
        get_default_hydrophobic_distance_tolerance()
    )

    if pair_distance < max(
        0.0,
        minimum_cutoff - tolerance,
    ):
        reasons.append(
            "distance_below_minimum"
        )

    if pair_distance > (
        maximum_cutoff
        + tolerance
    ):
        reasons.append(
            "distance_above_maximum"
        )

    interaction_type = (
        classify_hydrophobic_geometry_type(
            receptor_descriptor,
            ligand_descriptor,
        )
    )

    if (
        interaction_type
        == HYDROPHOBIC_TYPE_UNKNOWN
    ):
        reasons.append(
            "unknown_hydrophobic_geometry_type"
        )

    return tuple(
        dict.fromkeys(reasons)
    )


def is_valid_hydrophobic_pair(
    receptor_descriptor: HydrophobicAtom,
    ligand_descriptor: HydrophobicAtom,
    *,
    minimum_distance: Optional[Number] = None,
    maximum_distance: Optional[Number] = None,
    require_hydrophobic_descriptors: bool = (
        DEFAULT_REQUIRE_HYDROPHOBIC_DESCRIPTORS
    ),
    require_distinct_atoms: bool = (
        DEFAULT_REQUIRE_DISTINCT_ATOMS
    ),
    reject_zero_distance: bool = (
        DEFAULT_REJECT_ZERO_DISTANCE_PAIRS
    ),
) -> bool:
    """
    Return whether a descriptor pair is valid for interaction creation.
    """

    return not hydrophobic_pair_exclusion_reasons(
        receptor_descriptor,
        ligand_descriptor,
        minimum_distance=minimum_distance,
        maximum_distance=maximum_distance,
        require_hydrophobic_descriptors=(
            require_hydrophobic_descriptors
        ),
        require_distinct_atoms=require_distinct_atoms,
        reject_zero_distance=reject_zero_distance,
    )


# -----------------------------------------------------------------------------
# Candidate-pair search
# -----------------------------------------------------------------------------

def find_hydrophobic_pairs(
    receptor: Union[
        Any,
        HydrophobicAtomCollections,
        Sequence[HydrophobicAtom],
    ],
    ligand: Optional[
        Union[
            Any,
            Sequence[HydrophobicAtom],
        ]
    ] = None,
    *,
    minimum_distance: Optional[Number] = None,
    maximum_distance: Optional[Number] = None,
    prepared_collections: Optional[
        HydrophobicAtomCollections
    ] = None,
    remove_duplicates: bool = (
        DEFAULT_REMOVE_DUPLICATE_PAIRS
    ),
    validate_pairs: bool = True,
    require_hydrophobic_descriptors: bool = True,
    preparation_options: Optional[
        Mapping[str, Any]
    ] = None,
) -> HydrophobicDescriptorPairCollection:
    """
    Find receptor–ligand hydrophobic descriptor pairs within the cutoff.

    Parameters
    ----------
    receptor
        Receptor structure, DockModel, prepared collection or receptor
        descriptor sequence.
    ligand
        Ligand structure or ligand descriptor sequence.
    prepared_collections
        Optional precomputed atom preparation result.
    validate_pairs
        Apply complete chemical and geometric pair validation.
    preparation_options
        Options passed to
        :func:`prepare_hydrophobic_atom_collections`.
    """

    minimum_cutoff = (
        get_default_minimum_hydrophobic_distance()
        if minimum_distance is None
        else _nonnegative_float(
            minimum_distance,
            name="minimum hydrophobic distance",
        )
    )

    maximum_cutoff = (
        get_default_maximum_hydrophobic_distance()
        if maximum_distance is None
        else _positive_float(
            maximum_distance,
            name="maximum hydrophobic distance",
        )
    )

    if minimum_cutoff > maximum_cutoff:
        raise ValueError(
            "minimum_distance cannot exceed maximum_distance."
        )

    resolved_prepared = prepared_collections

    receptor_descriptors: Tuple[
        HydrophobicAtom,
        ...
    ]

    ligand_descriptors: Tuple[
        HydrophobicAtom,
        ...
    ]

    if resolved_prepared is not None:
        if not isinstance(
            resolved_prepared,
            HydrophobicAtomCollections,
        ):
            raise TypeError(
                "prepared_collections must be a "
                "HydrophobicAtomCollections instance."
            )

        receptor_descriptors = tuple(
            resolved_prepared.receptor_hydrophobic_atoms
        )

        ligand_descriptors = tuple(
            resolved_prepared.ligand_hydrophobic_atoms
        )

    elif isinstance(
        receptor,
        HydrophobicAtomCollections,
    ):
        receptor_descriptors = tuple(
            receptor.receptor_hydrophobic_atoms
        )

        ligand_descriptors = tuple(
            receptor.ligand_hydrophobic_atoms
        )

    else:
        receptor_values = tuple(
            receptor
        ) if (
            not isinstance(receptor, DockModel)
            and not _looks_like_atom(receptor)
            and not _looks_like_structure(receptor)
            and not _looks_like_residue(receptor)
            and not isinstance(receptor, Mapping)
            and not isinstance(receptor, str)
        ) else ()

        ligand_values = tuple(
            ligand
        ) if (
            ligand is not None
            and not _looks_like_atom(ligand)
            and not _looks_like_structure(ligand)
            and not _looks_like_residue(ligand)
            and not isinstance(ligand, Mapping)
            and not isinstance(ligand, str)
        ) else ()

        receptor_is_descriptor_sequence = bool(
            receptor_values
            and all(
                isinstance(
                    value,
                    HydrophobicAtom,
                )
                for value in receptor_values
            )
        )

        ligand_is_descriptor_sequence = bool(
            ligand_values
            and all(
                isinstance(
                    value,
                    HydrophobicAtom,
                )
                for value in ligand_values
            )
        )

        if (
            receptor_is_descriptor_sequence
            and ligand_is_descriptor_sequence
        ):
            receptor_descriptors = (
                deduplicate_hydrophobic_descriptors(
                    receptor_values
                )
            )

            ligand_descriptors = (
                deduplicate_hydrophobic_descriptors(
                    ligand_values
                )
            )

        else:
            options = (
                {}
                if preparation_options is None
                else dict(preparation_options)
            )

            resolved_prepared = (
                prepare_hydrophobic_atom_collections(
                    receptor,
                    ligand,
                    **options,
                )
            )

            receptor_descriptors = tuple(
                resolved_prepared.receptor_hydrophobic_atoms
            )

            ligand_descriptors = tuple(
                resolved_prepared.ligand_hydrophobic_atoms
            )

    if (
        not receptor_descriptors
        or not ligand_descriptors
    ):
        return ()

    pair_indices = hydrophobic_contact_indices(
        receptor_descriptors,
        ligand_descriptors,
        minimum_distance=minimum_cutoff,
        maximum_distance=maximum_cutoff,
    )

    candidate_pairs: List[
        HydrophobicDescriptorPair
    ] = []

    for receptor_index, ligand_index in pair_indices:
        receptor_descriptor = (
            receptor_descriptors[
                receptor_index
            ]
        )

        ligand_descriptor = (
            ligand_descriptors[
                ligand_index
            ]
        )

        if validate_pairs:
            if not is_valid_hydrophobic_pair(
                receptor_descriptor,
                ligand_descriptor,
                minimum_distance=minimum_cutoff,
                maximum_distance=maximum_cutoff,
                require_hydrophobic_descriptors=(
                    require_hydrophobic_descriptors
                ),
            ):
                continue

        candidate_pairs.append(
            (
                receptor_descriptor,
                ligand_descriptor,
            )
        )

    if remove_duplicates:
        return deduplicate_hydrophobic_pairs(
            candidate_pairs
        )

    return tuple(candidate_pairs)


# -----------------------------------------------------------------------------
# Detailed geometry analysis
# -----------------------------------------------------------------------------

def analyze_hydrophobic_geometry(
    receptor: Union[
        HydrophobicAtom,
        AtomLike,
        HydrophobicDescriptorPair,
    ],
    ligand: Optional[
        Union[
            HydrophobicAtom,
            AtomLike,
        ]
    ] = None,
    *,
    receptor_candidates: Optional[
        HydrophobicGroupInput
    ] = None,
    ligand_candidates: Optional[
        HydrophobicGroupInput
    ] = None,
    local_radius: Optional[Number] = None,
    maximum_distance: Optional[Number] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> HydrophobicPairGeometry:
    """
    Analyze the detailed geometry of one hydrophobic pair.

    The pair can be supplied either as two arguments or as one
    ``(receptor_descriptor, ligand_descriptor)`` tuple.
    """

    if ligand is None:
        if (
            isinstance(receptor, tuple)
            and len(receptor) == 2
        ):
            receptor_value, ligand_value = receptor

        else:
            raise ValueError(
                "When ligand is omitted, receptor must be a "
                "two-member receptor–ligand pair."
            )

    else:
        receptor_value = receptor
        ligand_value = ligand

    if isinstance(
        receptor_value,
        HydrophobicAtom,
    ):
        receptor_descriptor = receptor_value

    else:
        receptor_descriptor = perceive_hydrophobic_atom(
            receptor_value,
            role=HYDROPHOBIC_ROLE_RECEPTOR,
        )

    if isinstance(
        ligand_value,
        HydrophobicAtom,
    ):
        ligand_descriptor = ligand_value

    else:
        ligand_descriptor = perceive_hydrophobic_atom(
            ligand_value,
            role=HYDROPHOBIC_ROLE_LIGAND,
        )

    if not is_valid_hydrophobic_pair(
        receptor_descriptor,
        ligand_descriptor,
        minimum_distance=0.0,
        maximum_distance=(
            maximum_distance
            if maximum_distance is not None
            else get_default_maximum_hydrophobic_distance()
        ),
        reject_zero_distance=True,
    ):
        exclusion_reasons = (
            hydrophobic_pair_exclusion_reasons(
                receptor_descriptor,
                ligand_descriptor,
                minimum_distance=0.0,
                maximum_distance=(
                    maximum_distance
                    if maximum_distance is not None
                    else get_default_maximum_hydrophobic_distance()
                ),
            )
        )

        raise ValueError(
            "Invalid hydrophobic pair: "
            + ", ".join(exclusion_reasons)
        )

    return analyze_hydrophobic_pair_geometry(
        receptor_descriptor,
        ligand_descriptor,
        receptor_candidates=receptor_candidates,
        ligand_candidates=ligand_candidates,
        local_radius=local_radius,
        maximum_contact_distance=maximum_distance,
        metadata=metadata,
    )


# -----------------------------------------------------------------------------
# Interaction construction
# -----------------------------------------------------------------------------

def _build_hydrophobic_interaction_identifier(
    receptor_descriptor: HydrophobicAtom,
    ligand_descriptor: HydrophobicAtom,
) -> HydrophobicInteractionIdentifier:
    """
    Build a stable serializable interaction identifier.
    """

    receptor_identifier = (
        receptor_descriptor.identifier
        or _safe_atom_identifier(
            receptor_descriptor.atom,
            fallback="receptor-atom",
        )
        or "receptor-atom"
    )

    ligand_identifier = (
        ligand_descriptor.identifier
        or _safe_atom_identifier(
            ligand_descriptor.atom,
            fallback="ligand-atom",
        )
        or "ligand-atom"
    )

    residue_identifier = (
        receptor_descriptor.residue_identifier
        or "unknown-residue"
    )

    return (
        f"hydrophobic|{residue_identifier}|"
        f"{receptor_identifier}|{ligand_identifier}"
    )


def create_hydrophobic_interaction(
    receptor_descriptor: HydrophobicAtom,
    ligand_descriptor: HydrophobicAtom,
    *,
    geometry: Optional[
        HydrophobicPairGeometry
    ] = None,
    receptor_candidates: Optional[
        HydrophobicGroupInput
    ] = None,
    ligand_candidates: Optional[
        HydrophobicGroupInput
    ] = None,
    minimum_distance: Optional[Number] = None,
    maximum_distance: Optional[Number] = None,
    detection_method: HydrophobicDetectionMethod = (
        HYDROPHOBIC_METHOD_ATOMIC
    ),
    include_geometry_metadata: bool = (
        DEFAULT_INCLUDE_PAIR_GEOMETRY_METADATA
    ),
    metadata: Optional[Mapping[str, Any]] = None,
) -> HydrophobicInteraction:
    """
    Create one structured hydrophobic interaction.
    """

    if not isinstance(
        receptor_descriptor,
        HydrophobicAtom,
    ):
        raise TypeError(
            "receptor_descriptor must be a HydrophobicAtom."
        )

    if not isinstance(
        ligand_descriptor,
        HydrophobicAtom,
    ):
        raise TypeError(
            "ligand_descriptor must be a HydrophobicAtom."
        )

    minimum_cutoff = (
        get_default_minimum_hydrophobic_distance()
        if minimum_distance is None
        else _nonnegative_float(
            minimum_distance,
            name="minimum hydrophobic distance",
        )
    )

    maximum_cutoff = (
        get_default_maximum_hydrophobic_distance()
        if maximum_distance is None
        else _positive_float(
            maximum_distance,
            name="maximum hydrophobic distance",
        )
    )

    exclusion_reasons = hydrophobic_pair_exclusion_reasons(
        receptor_descriptor,
        ligand_descriptor,
        minimum_distance=minimum_cutoff,
        maximum_distance=maximum_cutoff,
    )

    if exclusion_reasons:
        raise ValueError(
            "Cannot create hydrophobic interaction: "
            + ", ".join(exclusion_reasons)
        )

    pair_geometry = geometry

    if pair_geometry is None:
        pair_geometry = analyze_hydrophobic_pair_geometry(
            receptor_descriptor,
            ligand_descriptor,
            receptor_candidates=(
                receptor_candidates
            ),
            ligand_candidates=ligand_candidates,
            maximum_contact_distance=maximum_cutoff,
        )

    if not isinstance(
        pair_geometry,
        HydrophobicPairGeometry,
    ):
        raise TypeError(
            "geometry must be a HydrophobicPairGeometry "
            "instance or None."
        )

    interaction_type = (
        pair_geometry.geometry_type
    )

    classification = classify_hydrophobic_distance(
        pair_geometry.distance,
        maximum_distance=maximum_cutoff,
    )

    strength = (
        calculate_preliminary_hydrophobic_strength(
            pair_geometry.distance,
            minimum_distance=minimum_cutoff,
            maximum_distance=maximum_cutoff,
        )
    )

    score = calculate_preliminary_hydrophobic_score(
        pair_geometry,
        strength=strength,
    )

    polar_neighbor_total = (
        receptor_descriptor.polar_neighbor_count
        + ligand_descriptor.polar_neighbor_count
    )

    maximum_polar_neighbors = max(
        get_default_maximum_polar_neighbors(),
        1,
    )

    polar_penalty = validate_hydrophobic_score(
        min(
            polar_neighbor_total
            / (
                2.0
                * maximum_polar_neighbors
            ),
            1.0,
        )
    )

    interaction_metadata: Dict[str, Any] = (
        {} if metadata is None else dict(metadata)
    )

    interaction_metadata.update(
        {
            "preliminary_classification": True,
            "preliminary_score": True,
            "distance_cutoffs": {
                "minimum": float(minimum_cutoff),
                "maximum": float(maximum_cutoff),
            },
            "receptor_exclusion_reasons": tuple(
                receptor_descriptor.metadata.get(
                    "exclusion_reasons",
                    (),
                )
            ),
            "ligand_exclusion_reasons": tuple(
                ligand_descriptor.metadata.get(
                    "exclusion_reasons",
                    (),
                )
            ),
            "aromatic_aromatic_is_not_pi_stacking": (
                interaction_type
                == HYDROPHOBIC_TYPE_AROMATIC_AROMATIC
            ),
        }
    )

    if include_geometry_metadata:
        interaction_metadata["geometry"] = (
            pair_geometry.to_dict(
                include_atoms=False,
                include_descriptors=False,
            )
        )

    return HydrophobicInteraction(
        receptor_atom=receptor_descriptor.atom,
        ligand_atom=ligand_descriptor.atom,
        distance=pair_geometry.distance,
        receptor_descriptor=receptor_descriptor,
        ligand_descriptor=ligand_descriptor,
        receptor_residue=(
            receptor_descriptor.residue
        ),
        receptor_residue_key=(
            receptor_descriptor.residue_key
        ),
        interaction_type=interaction_type,
        classification=classification,
        strength=strength,
        score=score,
        detection_method=detection_method,
        direction=(
            HYDROPHOBIC_DIRECTION_LIGAND_RECEPTOR
        ),
        local_contact_count=max(
            pair_geometry.shared_local_contact_count,
            1,
        ),
        polar_penalty=polar_penalty,
        receptor_atom_index=(
            receptor_descriptor.atom_index
        ),
        ligand_atom_index=(
            ligand_descriptor.atom_index
        ),
        receptor_atom_identifier=(
            receptor_descriptor.identifier
        ),
        ligand_atom_identifier=(
            ligand_descriptor.identifier
        ),
        interaction_identifier=(
            _build_hydrophobic_interaction_identifier(
                receptor_descriptor,
                ligand_descriptor,
            )
        ),
        metadata=interaction_metadata,
    )


# -----------------------------------------------------------------------------
# Contact detection
# -----------------------------------------------------------------------------

def detect_hydrophobic_contacts(
    receptor: Union[
        Any,
        HydrophobicAtomCollections,
    ],
    ligand: Optional[Any] = None,
    *,
    prepared_collections: Optional[
        HydrophobicAtomCollections
    ] = None,
    minimum_distance: Optional[Number] = None,
    maximum_distance: Optional[Number] = None,
    local_radius: Optional[Number] = None,
    remove_duplicates: bool = (
        DEFAULT_REMOVE_DUPLICATE_PAIRS
    ),
    sort_interactions: bool = (
        DEFAULT_SORT_HYDROPHOBIC_INTERACTIONS
    ),
    maximum_interactions: Optional[int] = None,
    include_geometry_metadata: bool = (
        DEFAULT_INCLUDE_PAIR_GEOMETRY_METADATA
    ),
    preparation_options: Optional[
        Mapping[str, Any]
    ] = None,
) -> Tuple[
    HydrophobicInteraction,
    ...,
]:
    """
    Detect individual receptor–ligand hydrophobic contacts.

    Processing steps
    ----------------
    1. prepare receptor and ligand atoms;
    2. perceive hydrophobic atoms;
    3. search pairs within the distance cutoff;
    4. validate each candidate pair;
    5. calculate detailed local geometry;
    6. classify chemical contact type;
    7. assign preliminary strength and score;
    8. remove duplicate interactions.
    """

    minimum_cutoff = (
        get_default_minimum_hydrophobic_distance()
        if minimum_distance is None
        else _nonnegative_float(
            minimum_distance,
            name="minimum hydrophobic distance",
        )
    )

    maximum_cutoff = (
        get_default_maximum_hydrophobic_distance()
        if maximum_distance is None
        else _positive_float(
            maximum_distance,
            name="maximum hydrophobic distance",
        )
    )

    if minimum_cutoff > maximum_cutoff:
        raise ValueError(
            "minimum_distance cannot exceed maximum_distance."
        )

    resolved_prepared = prepared_collections

    if resolved_prepared is None:
        if isinstance(
            receptor,
            HydrophobicAtomCollections,
        ):
            resolved_prepared = receptor

        else:
            options = (
                {}
                if preparation_options is None
                else dict(preparation_options)
            )

            resolved_prepared = (
                prepare_hydrophobic_atom_collections(
                    receptor,
                    ligand,
                    **options,
                )
            )

    if not isinstance(
        resolved_prepared,
        HydrophobicAtomCollections,
    ):
        raise TypeError(
            "Could not resolve prepared hydrophobic atom "
            "collections."
        )

    receptor_descriptors = tuple(
        resolved_prepared.receptor_hydrophobic_atoms
    )

    ligand_descriptors = tuple(
        resolved_prepared.ligand_hydrophobic_atoms
    )

    if (
        not receptor_descriptors
        or not ligand_descriptors
    ):
        return ()

    candidate_pairs = find_hydrophobic_pairs(
        resolved_prepared,
        minimum_distance=minimum_cutoff,
        maximum_distance=maximum_cutoff,
        remove_duplicates=remove_duplicates,
        validate_pairs=True,
    )

    interactions: List[
        HydrophobicInteraction
    ] = []

    for receptor_descriptor, ligand_descriptor in candidate_pairs:
        try:
            pair_geometry = (
                analyze_hydrophobic_pair_geometry(
                    receptor_descriptor,
                    ligand_descriptor,
                    receptor_candidates=(
                        receptor_descriptors
                    ),
                    ligand_candidates=(
                        ligand_descriptors
                    ),
                    local_radius=local_radius,
                    maximum_contact_distance=(
                        maximum_cutoff
                    ),
                )
            )

            interaction = (
                create_hydrophobic_interaction(
                    receptor_descriptor,
                    ligand_descriptor,
                    geometry=pair_geometry,
                    receptor_candidates=(
                        receptor_descriptors
                    ),
                    ligand_candidates=(
                        ligand_descriptors
                    ),
                    minimum_distance=minimum_cutoff,
                    maximum_distance=maximum_cutoff,
                    detection_method=(
                        HYDROPHOBIC_METHOD_ATOMIC
                    ),
                    include_geometry_metadata=(
                        include_geometry_metadata
                    ),
                )
            )

        except (
            TypeError,
            ValueError,
            AttributeError,
        ) as exc:
            try:
                _LOGGER.warning(
                    "A candidate hydrophobic pair was rejected "
                    f"during detailed analysis: {exc}"
                )
            except Exception:
                pass

            continue

        interactions.append(interaction)

    if remove_duplicates:
        interactions_tuple = (
            deduplicate_hydrophobic_interactions(
                interactions,
                prefer_highest_score=True,
            )
        )

    else:
        interactions_tuple = tuple(
            interactions
        )

    if sort_interactions:
        interactions_tuple = tuple(
            sorted(
                interactions_tuple,
                key=lambda interaction: (
                    -float(interaction.score),
                    float(interaction.distance),
                    interaction.interaction_identifier or "",
                ),
            )
        )

    interaction_limit = (
        get_default_maximum_hydrophobic_interactions()
        if maximum_interactions is None
        else _coerce_optional_positive_integer(
            maximum_interactions,
            name="maximum hydrophobic interactions",
            default=None,
        )
    )

    if interaction_limit is not None:
        interactions_tuple = interactions_tuple[
            :interaction_limit
        ]

    return interactions_tuple


# -----------------------------------------------------------------------------
# Basic statistics available during detection
# -----------------------------------------------------------------------------

def _build_detection_statistics(
    interactions: Sequence[
        HydrophobicInteraction
    ],
    *,
    receptor_atom_count: int,
    ligand_atom_count: int,
) -> HydrophobicStatistics:
    """
    Build basic statistics without residue grouping.

    Section 10 will provide the complete statistics implementation.
    """

    interactions_tuple = tuple(
        interactions
    )

    classification_counts = {
        classification: sum(
            interaction.classification
            == classification
            for interaction in interactions_tuple
        )
        for classification
        in _VALID_HYDROPHOBIC_CLASSIFICATIONS
    }

    interaction_type_counts = {
        interaction_type: sum(
            interaction.interaction_type
            == interaction_type
            for interaction in interactions_tuple
        )
        for interaction_type
        in _VALID_HYDROPHOBIC_TYPES
    }

    residue_identifiers = {
        interaction.receptor_residue_identifier
        for interaction in interactions_tuple
        if interaction.receptor_residue_identifier
        is not None
    }

    residue_interaction_counts: Dict[
        str,
        int,
    ] = {}

    residue_scores: Dict[
        str,
        float,
    ] = {}

    for interaction in interactions_tuple:
        residue_identifier = (
            interaction.receptor_residue_identifier
        )

        if residue_identifier is None:
            continue

        residue_interaction_counts[
            residue_identifier
        ] = (
            residue_interaction_counts.get(
                residue_identifier,
                0,
            )
            + 1
        )

        residue_scores[
            residue_identifier
        ] = (
            residue_scores.get(
                residue_identifier,
                0.0,
            )
            + float(interaction.score)
        )

    if interactions_tuple:
        distances = np.asarray(
            [
                interaction.distance
                for interaction in interactions_tuple
            ],
            dtype=np.float64,
        )

        scores = np.asarray(
            [
                interaction.score
                for interaction in interactions_tuple
            ],
            dtype=np.float64,
        )

        strengths = np.asarray(
            [
                interaction.strength
                for interaction in interactions_tuple
            ],
            dtype=np.float64,
        )

        minimum_distance = np.float64(
            np.min(distances)
        )

        mean_distance = np.float64(
            np.mean(distances)
        )

        median_distance = np.float64(
            np.median(distances)
        )

        maximum_distance = np.float64(
            np.max(distances)
        )

        distance_standard_deviation = np.float64(
            np.std(distances)
        )

        minimum_score = np.float64(
            np.min(scores)
        )

        mean_score = np.float64(
            np.mean(scores)
        )

        median_score = np.float64(
            np.median(scores)
        )

        maximum_score = np.float64(
            np.max(scores)
        )

        total_score = np.float64(
            np.sum(scores)
        )

        minimum_strength = np.float64(
            np.min(strengths)
        )

        mean_strength = np.float64(
            np.mean(strengths)
        )

        maximum_strength = np.float64(
            np.max(strengths)
        )

    else:
        minimum_distance = None
        mean_distance = None
        median_distance = None
        maximum_distance = None
        distance_standard_deviation = None

        minimum_score = None
        mean_score = None
        median_score = None
        maximum_score = None
        total_score = np.float64(0.0)

        minimum_strength = None
        mean_strength = None
        maximum_strength = None

    return HydrophobicStatistics(
        interaction_count=len(
            interactions_tuple
        ),
        residue_count=len(
            residue_identifiers
        ),
        receptor_atom_count=(
            receptor_atom_count
        ),
        ligand_atom_count=ligand_atom_count,
        very_strong_count=(
            classification_counts.get(
                HYDROPHOBIC_CLASS_VERY_STRONG,
                0,
            )
        ),
        strong_count=classification_counts.get(
            HYDROPHOBIC_CLASS_STRONG,
            0,
        ),
        moderate_count=classification_counts.get(
            HYDROPHOBIC_CLASS_MODERATE,
            0,
        ),
        weak_count=classification_counts.get(
            HYDROPHOBIC_CLASS_WEAK,
            0,
        ),
        marginal_count=classification_counts.get(
            HYDROPHOBIC_CLASS_MARGINAL,
            0,
        ),
        unknown_count=classification_counts.get(
            HYDROPHOBIC_CLASS_UNKNOWN,
            0,
        ),
        aliphatic_aliphatic_count=(
            interaction_type_counts.get(
                HYDROPHOBIC_TYPE_ALIPHATIC_ALIPHATIC,
                0,
            )
        ),
        aliphatic_aromatic_count=(
            interaction_type_counts.get(
                HYDROPHOBIC_TYPE_ALIPHATIC_AROMATIC,
                0,
            )
        ),
        aromatic_aliphatic_count=(
            interaction_type_counts.get(
                HYDROPHOBIC_TYPE_AROMATIC_ALIPHATIC,
                0,
            )
        ),
        aromatic_aromatic_count=(
            interaction_type_counts.get(
                HYDROPHOBIC_TYPE_AROMATIC_AROMATIC,
                0,
            )
        ),
        mixed_count=interaction_type_counts.get(
            HYDROPHOBIC_TYPE_MIXED,
            0,
        ),
        hotspot_count=0,
        minimum_distance=minimum_distance,
        mean_distance=mean_distance,
        median_distance=median_distance,
        maximum_distance=maximum_distance,
        distance_standard_deviation=(
            distance_standard_deviation
        ),
        minimum_score=minimum_score,
        mean_score=mean_score,
        median_score=median_score,
        maximum_score=maximum_score,
        total_score=total_score,
        minimum_strength=minimum_strength,
        mean_strength=mean_strength,
        maximum_strength=maximum_strength,
        classification_counts=(
            classification_counts
        ),
        interaction_type_counts=(
            interaction_type_counts
        ),
        residue_interaction_counts=(
            residue_interaction_counts
        ),
        residue_scores=residue_scores,
        metadata={
            "statistics_stage": "detection",
            "residue_grouping_completed": False,
        },
    )


# -----------------------------------------------------------------------------
# Complete detection workflow
# -----------------------------------------------------------------------------

def detect_hydrophobic_interactions(
    receptor: Union[
        Any,
        HydrophobicAtomCollections,
    ],
    ligand: Optional[Any] = None,
    *,
    prepared_collections: Optional[
        HydrophobicAtomCollections
    ] = None,
    minimum_distance: Optional[Number] = None,
    maximum_distance: Optional[Number] = None,
    grouping_distance: Optional[Number] = None,
    local_radius: Optional[Number] = None,
    remove_duplicates: bool = (
        DEFAULT_REMOVE_DUPLICATE_PAIRS
    ),
    sort_interactions: bool = (
        DEFAULT_SORT_HYDROPHOBIC_INTERACTIONS
    ),
    maximum_interactions: Optional[int] = None,
    include_geometry_metadata: bool = (
        DEFAULT_INCLUDE_PAIR_GEOMETRY_METADATA
    ),
    preparation_options: Optional[
        Mapping[str, Any]
    ] = None,
    receptor_identifier: Optional[str] = None,
    ligand_identifier: Optional[str] = None,
    analysis_identifier: Optional[str] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> HydrophobicAnalysisResult:
    """
    Run the complete atomic hydrophobic-interaction detection workflow.

    At this stage, the result contains individual interactions and basic
    statistics. Residue groups are intentionally empty until Section 8
    performs residue-level grouping.
    """

    minimum_cutoff = (
        get_default_minimum_hydrophobic_distance()
        if minimum_distance is None
        else _nonnegative_float(
            minimum_distance,
            name="minimum hydrophobic distance",
        )
    )

    maximum_cutoff = (
        get_default_maximum_hydrophobic_distance()
        if maximum_distance is None
        else _positive_float(
            maximum_distance,
            name="maximum hydrophobic distance",
        )
    )

    grouping_cutoff = (
        get_default_grouping_distance()
        if grouping_distance is None
        else _positive_float(
            grouping_distance,
            name="hydrophobic grouping distance",
        )
    )

    if minimum_cutoff > maximum_cutoff:
        raise ValueError(
            "minimum_distance cannot exceed maximum_distance."
        )

    resolved_prepared = prepared_collections

    if resolved_prepared is None:
        if isinstance(
            receptor,
            HydrophobicAtomCollections,
        ):
            resolved_prepared = receptor

        else:
            options = (
                {}
                if preparation_options is None
                else dict(preparation_options)
            )

            resolved_prepared = (
                prepare_hydrophobic_atom_collections(
                    receptor,
                    ligand,
                    **options,
                )
            )

    candidate_pairs = find_hydrophobic_pairs(
        resolved_prepared,
        minimum_distance=minimum_cutoff,
        maximum_distance=maximum_cutoff,
        remove_duplicates=remove_duplicates,
        validate_pairs=True,
    )

    interactions = detect_hydrophobic_contacts(
        resolved_prepared,
        prepared_collections=resolved_prepared,
        minimum_distance=minimum_cutoff,
        maximum_distance=maximum_cutoff,
        local_radius=local_radius,
        remove_duplicates=remove_duplicates,
        sort_interactions=sort_interactions,
        maximum_interactions=maximum_interactions,
        include_geometry_metadata=(
            include_geometry_metadata
        ),
    )

    statistics = _build_detection_statistics(
        interactions,
        receptor_atom_count=len(
            resolved_prepared.receptor_atoms
        ),
        ligand_atom_count=len(
            resolved_prepared.ligand_atoms
        ),
    )

    detected_pair_keys = {
        hydrophobic_interaction_pair_key(
            interaction
        )
        for interaction in interactions
    }

    candidate_pair_keys = {
        hydrophobic_descriptor_pair_key(
            receptor_descriptor,
            ligand_descriptor,
        )
        for receptor_descriptor, ligand_descriptor
        in candidate_pairs
    }

    rejected_pair_count = len(
        candidate_pair_keys
        - detected_pair_keys
    )

    analysis_metadata: Dict[str, Any] = (
        {} if metadata is None else dict(metadata)
    )

    analysis_metadata.update(
        {
            "analysis_stage": "atomic_detection",
            "candidate_pair_count": len(
                candidate_pairs
            ),
            "detected_interaction_count": len(
                interactions
            ),
            "rejected_pair_count": (
                rejected_pair_count
            ),
            "minimum_distance": float(
                minimum_cutoff
            ),
            "maximum_distance": float(
                maximum_cutoff
            ),
            "grouping_distance": float(
                grouping_cutoff
            ),
            "local_radius": (
                None
                if local_radius is None
                else float(
                    _positive_float(
                        local_radius,
                        name="local radius",
                    )
                )
            ),
            "duplicates_removed": bool(
                remove_duplicates
            ),
            "interactions_sorted": bool(
                sort_interactions
            ),
            "geometric_classification_is_preliminary": True,
            "score_is_preliminary": True,
            "residue_grouping_completed": False,
            "aromatic_aromatic_is_not_pi_stacking": True,
        }
    )

    return HydrophobicAnalysisResult(
        interactions=interactions,
        residue_groups=(),
        receptor_hydrophobic_atoms=(
            resolved_prepared.receptor_hydrophobic_atoms
        ),
        ligand_hydrophobic_atoms=(
            resolved_prepared.ligand_hydrophobic_atoms
        ),
        receptor_atoms=(
            resolved_prepared.receptor_atoms
        ),
        ligand_atoms=(
            resolved_prepared.ligand_atoms
        ),
        minimum_distance=minimum_cutoff,
        maximum_distance=maximum_cutoff,
        grouping_distance=grouping_cutoff,
        statistics=statistics,
        analysis_identifier=analysis_identifier,
        receptor_identifier=receptor_identifier,
        ligand_identifier=ligand_identifier,
        metadata=analysis_metadata,
    )


# -----------------------------------------------------------------------------
# Intermediate detection-result workflow
# -----------------------------------------------------------------------------

def run_hydrophobic_detection(
    receptor: Union[
        Any,
        HydrophobicAtomCollections,
    ],
    ligand: Optional[Any] = None,
    *,
    prepared_collections: Optional[
        HydrophobicAtomCollections
    ] = None,
    minimum_distance: Optional[Number] = None,
    maximum_distance: Optional[Number] = None,
    local_radius: Optional[Number] = None,
    remove_duplicates: bool = True,
    preparation_options: Optional[
        Mapping[str, Any]
    ] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> HydrophobicDetectionResult:
    """
    Return the intermediate atomic-detection result.

    This function is useful for debugging and self-tests because it keeps
    candidate pairs, rejected counts and prepared collections together.
    """

    minimum_cutoff = (
        get_default_minimum_hydrophobic_distance()
        if minimum_distance is None
        else _nonnegative_float(
            minimum_distance,
            name="minimum hydrophobic distance",
        )
    )

    maximum_cutoff = (
        get_default_maximum_hydrophobic_distance()
        if maximum_distance is None
        else _positive_float(
            maximum_distance,
            name="maximum hydrophobic distance",
        )
    )

    resolved_prepared = prepared_collections

    if resolved_prepared is None:
        if isinstance(
            receptor,
            HydrophobicAtomCollections,
        ):
            resolved_prepared = receptor

        else:
            options = (
                {}
                if preparation_options is None
                else dict(preparation_options)
            )

            resolved_prepared = (
                prepare_hydrophobic_atom_collections(
                    receptor,
                    ligand,
                    **options,
                )
            )

    raw_pairs = find_hydrophobic_pairs(
        resolved_prepared,
        minimum_distance=minimum_cutoff,
        maximum_distance=maximum_cutoff,
        remove_duplicates=False,
        validate_pairs=False,
    )

    valid_pairs: List[
        HydrophobicDescriptorPair
    ] = []

    rejected_pair_count = 0

    for receptor_descriptor, ligand_descriptor in raw_pairs:
        if is_valid_hydrophobic_pair(
            receptor_descriptor,
            ligand_descriptor,
            minimum_distance=minimum_cutoff,
            maximum_distance=maximum_cutoff,
        ):
            valid_pairs.append(
                (
                    receptor_descriptor,
                    ligand_descriptor,
                )
            )

        else:
            rejected_pair_count += 1

    if remove_duplicates:
        unique_pairs = deduplicate_hydrophobic_pairs(
            valid_pairs
        )

    else:
        unique_pairs = tuple(valid_pairs)

    duplicate_pair_count = (
        len(valid_pairs)
        - len(unique_pairs)
    )

    interactions = detect_hydrophobic_contacts(
        resolved_prepared,
        prepared_collections=resolved_prepared,
        minimum_distance=minimum_cutoff,
        maximum_distance=maximum_cutoff,
        local_radius=local_radius,
        remove_duplicates=remove_duplicates,
    )

    detection_metadata: Dict[str, Any] = (
        {} if metadata is None else dict(metadata)
    )

    detection_metadata.update(
        {
            "raw_pair_count": len(raw_pairs),
            "valid_pair_count_before_deduplication": len(
                valid_pairs
            ),
            "valid_pair_count_after_deduplication": len(
                unique_pairs
            ),
        }
    )

    return HydrophobicDetectionResult(
        prepared_collections=resolved_prepared,
        candidate_pairs=unique_pairs,
        interactions=interactions,
        rejected_pair_count=rejected_pair_count,
        duplicate_pair_count=duplicate_pair_count,
        minimum_distance=minimum_cutoff,
        maximum_distance=maximum_cutoff,
        metadata=detection_metadata,
    )


# -----------------------------------------------------------------------------
# Empty detection result
# -----------------------------------------------------------------------------

_EMPTY_HYDROPHOBIC_DETECTION_RESULT: Final[
    HydrophobicDetectionResult
] = HydrophobicDetectionResult(
    prepared_collections=(
        _EMPTY_PREPARED_HYDROPHOBIC_COLLECTIONS
    )
)


# -----------------------------------------------------------------------------
# Section 7 public names
# -----------------------------------------------------------------------------

_SECTION_7_PUBLIC_NAMES: Final[Tuple[str, ...]] = (
    # Aliases
    "HydrophobicDescriptorPair",
    "HydrophobicDescriptorPairCollection",
    "HydrophobicPairKey",

    # Dataclass
    "HydrophobicDetectionResult",

    # Pair identifiers and deduplication
    "hydrophobic_descriptor_pair_key",
    "hydrophobic_interaction_pair_key",
    "deduplicate_hydrophobic_pairs",
    "deduplicate_hydrophobic_interactions",

    # Preliminary classification and scores
    "classify_hydrophobic_distance",
    "calculate_preliminary_hydrophobic_strength",
    "hydrophobic_chemical_compatibility_score",
    "calculate_preliminary_hydrophobic_score",

    # Pair validation
    "hydrophobic_pair_exclusion_reasons",
    "is_valid_hydrophobic_pair",

    # Central detection functions
    "find_hydrophobic_pairs",
    "analyze_hydrophobic_geometry",
    "create_hydrophobic_interaction",
    "detect_hydrophobic_contacts",
    "detect_hydrophobic_interactions",
    "run_hydrophobic_detection",
)

for public_name in _SECTION_7_PUBLIC_NAMES:
    if public_name not in __all__:
        __all__.append(public_name)


# =============================================================================
# End of Section 7
# =============================================================================


# =============================================================================
# Section 8 — Grouping by residue, chain, pose and local region
# =============================================================================


# -----------------------------------------------------------------------------
# Grouping aliases and defaults
# -----------------------------------------------------------------------------

HydrophobicChainKey: TypeAlias = str
HydrophobicPoseKey: TypeAlias = str
HydrophobicLocalRegionKey: TypeAlias = Tuple[
    Optional[str],
    Tuple[str, ...],
]

HydrophobicInteractionCollection: TypeAlias = Sequence[
    HydrophobicInteraction
]

DEFAULT_LOCAL_INTERACTION_CLUSTER_DISTANCE: Final[np.float64] = (
    np.float64(
        DEFAULT_GROUPING_DISTANCE
    )
)

DEFAULT_HOTSPOT_MINIMUM_CONTACT_COUNT: Final[int] = (
    DEFAULT_MINIMUM_HOTSPOT_CONTACT_COUNT
)

DEFAULT_HOTSPOT_MINIMUM_SCORE: Final[np.float64] = np.float64(
    1.5
)

DEFAULT_HOTSPOT_MINIMUM_CONTACT_AREA: Final[np.float64] = np.float64(
    5.0
)

DEFAULT_HOTSPOT_MINIMUM_LIGAND_ATOM_COUNT: Final[int] = 2

DEFAULT_RESIDUE_GROUP_SORTING: Final[str] = "score"
DEFAULT_LOCAL_REGION_SORTING: Final[str] = "score"


# -----------------------------------------------------------------------------
# Chain, pose and residue identifier helpers
# -----------------------------------------------------------------------------

def get_residue_chain_identifier(
    residue: Optional[ResidueLike],
    *,
    default: str = "",
) -> str:
    """
    Return a normalized chain identifier for a residue.

    The function supports ChimeraX residues and synthetic objects exposing
    attributes such as ``chain_id``, ``chainId`` or ``chain``.
    """

    if residue is None:
        return str(default)

    chain_value = _safe_getattr(
        residue,
        (
            "chain_id",
            "chainId",
            "chain_identifier",
            "chainIdentifier",
        ),
        default=None,
    )

    if chain_value is not None:
        normalized = str(chain_value).strip()

        if normalized:
            return normalized

    chain_object = _safe_getattr(
        residue,
        (
            "chain",
            "parent_chain",
            "parentChain",
        ),
        default=None,
    )

    if chain_object is not None:
        chain_value = _safe_getattr(
            chain_object,
            (
                "chain_id",
                "chainId",
                "id",
                "name",
            ),
            default=None,
        )

        if chain_value is not None:
            normalized = str(chain_value).strip()

            if normalized:
                return normalized

    return str(default)


def get_interaction_chain_identifier(
    interaction: HydrophobicInteraction,
    *,
    default: str = "",
) -> str:
    """
    Return the receptor chain identifier for an interaction.
    """

    if not isinstance(
        interaction,
        HydrophobicInteraction,
    ):
        raise TypeError(
            "interaction must be a HydrophobicInteraction."
        )

    chain_identifier = get_residue_chain_identifier(
        interaction.receptor_residue,
        default="",
    )

    if chain_identifier:
        return chain_identifier

    residue_key = interaction.receptor_residue_key

    if residue_key:
        # The precise ResidueContactKey layout is owned by contacts.py.
        # The following fallback attempts to recover a likely chain field.
        for value in residue_key:
            if value is None:
                continue

            normalized = str(value).strip()

            if (
                normalized
                and len(normalized) <= 4
                and not normalized.lstrip("-").isdigit()
            ):
                return normalized

    metadata_chain = interaction.metadata.get(
        "chain_identifier"
    )

    if metadata_chain is not None:
        normalized = str(metadata_chain).strip()

        if normalized:
            return normalized

    return str(default)


def get_interaction_pose_identifier(
    interaction: HydrophobicInteraction,
    *,
    default: str = "pose-unknown",
) -> str:
    """
    Return a serializable pose identifier for an interaction.

    Pose information may be stored directly in interaction metadata, in
    ligand-descriptor metadata or in the ligand's parent structure.
    """

    if not isinstance(
        interaction,
        HydrophobicInteraction,
    ):
        raise TypeError(
            "interaction must be a HydrophobicInteraction."
        )

    metadata_names = (
        "pose_identifier",
        "pose_id",
        "pose",
        "ligand_pose",
        "ligand_identifier",
    )

    for name in metadata_names:
        value = interaction.metadata.get(name)

        normalized = _normalize_optional_string(value)

        if normalized is not None:
            return normalized

    ligand_descriptor = interaction.ligand_descriptor

    if ligand_descriptor is not None:
        for name in metadata_names:
            value = ligand_descriptor.metadata.get(name)

            normalized = _normalize_optional_string(value)

            if normalized is not None:
                return normalized

    ligand_structure = None

    try:
        ligand_structure = get_atom_structure(
            interaction.ligand_atom
        )
    except Exception:
        ligand_structure = _safe_getattr(
            interaction.ligand_atom,
            (
                "structure",
                "model",
                "molecule",
            ),
            default=None,
        )

    if ligand_structure is not None:
        value = _safe_getattr(
            ligand_structure,
            (
                "pose_identifier",
                "pose_id",
                "id_string",
                "id",
                "name",
            ),
            default=None,
        )

        normalized = _normalize_optional_string(value)

        if normalized is not None:
            return normalized

    return str(default)


def hydrophobic_residue_group_key(
    interaction: HydrophobicInteraction,
) -> Tuple[Any, ...]:
    """
    Return a stable receptor-residue grouping key.
    """

    if not isinstance(
        interaction,
        HydrophobicInteraction,
    ):
        raise TypeError(
            "interaction must be a HydrophobicInteraction."
        )

    residue_key = interaction.receptor_residue_key

    if residue_key is not None:
        return (
            "residue-key",
            *tuple(residue_key),
        )

    residue = interaction.receptor_residue

    if residue is not None:
        return (
            "residue-object",
            id(residue),
        )

    residue_identifier = (
        interaction.receptor_residue_identifier
    )

    if residue_identifier is not None:
        return (
            "residue-identifier",
            residue_identifier,
        )

    return (
        "receptor-atom",
        interaction.receptor_atom_identifier
        or id(interaction.receptor_atom),
    )


def hydrophobic_local_region_key(
    interactions: HydrophobicInteractionCollection,
) -> HydrophobicLocalRegionKey:
    """
    Create a serializable key for a local interaction region.
    """

    interaction_tuple = tuple(interactions)

    pose_identifiers = sorted(
        {
            get_interaction_pose_identifier(
                interaction
            )
            for interaction in interaction_tuple
        }
    )

    residue_identifiers = sorted(
        {
            interaction.receptor_residue_identifier
            or "residue-unknown"
            for interaction in interaction_tuple
        }
    )

    pose_identifier = (
        pose_identifiers[0]
        if len(pose_identifiers) == 1
        else "|".join(pose_identifiers)
    )

    return (
        pose_identifier,
        tuple(residue_identifiers),
    )


# -----------------------------------------------------------------------------
# Interaction centroid and local-distance helpers
# -----------------------------------------------------------------------------

def hydrophobic_interaction_midpoint(
    interaction: HydrophobicInteraction,
) -> Coordinate:
    """
    Return the midpoint between receptor and ligand atoms.
    """

    if not isinstance(
        interaction,
        HydrophobicInteraction,
    ):
        raise TypeError(
            "interaction must be a HydrophobicInteraction."
        )

    receptor_coordinate = get_hydrophobic_coordinate(
        interaction.receptor_atom
    )

    ligand_coordinate = get_hydrophobic_coordinate(
        interaction.ligand_atom
    )

    return np.asarray(
        (
            receptor_coordinate
            + ligand_coordinate
        ) / 2.0,
        dtype=np.float64,
    )


def hydrophobic_interaction_midpoint_distance(
    first: HydrophobicInteraction,
    second: HydrophobicInteraction,
) -> np.float64:
    """
    Return the distance between two interaction midpoints.
    """

    return hydrophobic_distance(
        hydrophobic_interaction_midpoint(first),
        hydrophobic_interaction_midpoint(second),
    )


def hydrophobic_interactions_share_atom(
    first: HydrophobicInteraction,
    second: HydrophobicInteraction,
) -> bool:
    """
    Return whether two interactions share a receptor or ligand atom.
    """

    return bool(
        first.receptor_atom is second.receptor_atom
        or first.ligand_atom is second.ligand_atom
        or (
            first.receptor_atom_identifier is not None
            and first.receptor_atom_identifier
            == second.receptor_atom_identifier
        )
        or (
            first.ligand_atom_identifier is not None
            and first.ligand_atom_identifier
            == second.ligand_atom_identifier
        )
    )


def are_hydrophobic_interactions_locally_connected(
    first: HydrophobicInteraction,
    second: HydrophobicInteraction,
    *,
    grouping_distance: Optional[Number] = None,
    require_same_pose: bool = True,
) -> bool:
    """
    Return whether two atom-pair contacts belong to one local region.

    Contacts are connected when they share an atom or when their
    interaction midpoints are within ``grouping_distance``.
    """

    cutoff = (
        DEFAULT_LOCAL_INTERACTION_CLUSTER_DISTANCE
        if grouping_distance is None
        else _positive_float(
            grouping_distance,
            name="local interaction grouping distance",
        )
    )

    if require_same_pose:
        if (
            get_interaction_pose_identifier(first)
            != get_interaction_pose_identifier(second)
        ):
            return False

    if hydrophobic_interactions_share_atom(
        first,
        second,
    ):
        return True

    midpoint_distance = (
        hydrophobic_interaction_midpoint_distance(
            first,
            second,
        )
    )

    return bool(
        midpoint_distance <= cutoff
    )


# -----------------------------------------------------------------------------
# Residue-group metrics
# -----------------------------------------------------------------------------

def select_closest_hydrophobic_interaction(
    interactions: HydrophobicInteractionCollection,
) -> Optional[HydrophobicInteraction]:
    """
    Return the shortest-distance interaction in a collection.

    Score and identifier are used as deterministic tie-breakers.
    """

    interaction_tuple = tuple(interactions)

    if not interaction_tuple:
        return None

    return min(
        interaction_tuple,
        key=lambda interaction: (
            float(interaction.distance),
            -float(interaction.score),
            interaction.interaction_identifier or "",
        ),
    )


def select_highest_scoring_hydrophobic_interaction(
    interactions: HydrophobicInteractionCollection,
) -> Optional[HydrophobicInteraction]:
    """
    Return the highest-scoring interaction in a collection.
    """

    interaction_tuple = tuple(interactions)

    if not interaction_tuple:
        return None

    return max(
        interaction_tuple,
        key=lambda interaction: (
            float(interaction.score),
            -float(interaction.distance),
            interaction.interaction_identifier or "",
        ),
    )


def approximate_interaction_contact_area(
    interaction: HydrophobicInteraction,
) -> np.float64:
    """
    Return the approximate area associated with an interaction.

    A value already calculated by Section 6 is preferred. When absent, the
    area is estimated directly from the atom-pair distance.
    """

    geometry = interaction.metadata.get(
        "geometry"
    )

    if isinstance(geometry, Mapping):
        area = geometry.get(
            "approximate_contact_area"
        )

        if area is not None:
            try:
                return _nonnegative_float(
                    area,
                    name="interaction contact area",
                )
            except (
                TypeError,
                ValueError,
            ):
                pass

    return approximate_pair_contact_area(
        interaction.distance,
        maximum_distance=(
            get_default_maximum_hydrophobic_distance()
        ),
    )


def approximate_hydrophobic_surface_area(
    interactions: HydrophobicInteractionCollection,
    *,
    reduce_shared_atoms: bool = True,
) -> np.float64:
    """
    Estimate the hydrophobic contact surface represented by interactions.

    The sum of atom-pair area estimates can overcount areas when one atom
    participates in several contacts. When ``reduce_shared_atoms=True``,
    repeated contributions from the same receptor or ligand atom are
    progressively discounted.
    """

    interaction_tuple = tuple(interactions)

    if not interaction_tuple:
        return np.float64(0.0)

    receptor_usage: Dict[
        Tuple[Any, ...],
        int,
    ] = {}

    ligand_usage: Dict[
        Tuple[Any, ...],
        int,
    ] = {}

    total_area = 0.0

    for interaction in sorted(
        interaction_tuple,
        key=lambda item: float(item.distance),
    ):
        pair_area = float(
            approximate_interaction_contact_area(
                interaction
            )
        )

        if not reduce_shared_atoms:
            total_area += pair_area
            continue

        receptor_key = atom_deduplication_key(
            interaction.receptor_atom,
            strategy="auto",
        )

        ligand_key = atom_deduplication_key(
            interaction.ligand_atom,
            strategy="auto",
        )

        receptor_count = receptor_usage.get(
            receptor_key,
            0,
        )

        ligand_count = ligand_usage.get(
            ligand_key,
            0,
        )

        reuse_count = max(
            receptor_count,
            ligand_count,
        )

        discount = 1.0 / (
            1.0
            + 0.5 * reuse_count
        )

        total_area += (
            pair_area
            * discount
        )

        receptor_usage[
            receptor_key
        ] = receptor_count + 1

        ligand_usage[
            ligand_key
        ] = ligand_count + 1

    return np.float64(total_area)


def calculate_residue_group_score(
    interactions: HydrophobicInteractionCollection,
) -> np.float64:
    """
    Calculate a residue-level hydrophobic score.

    Multiple contacts contribute with diminishing returns so that a
    residue is not rewarded linearly for several nearly redundant pairs.
    """

    interaction_tuple = tuple(interactions)

    if not interaction_tuple:
        return np.float64(0.0)

    ordered_scores = sorted(
        (
            float(interaction.score)
            for interaction in interaction_tuple
        ),
        reverse=True,
    )

    total_score = 0.0

    for rank, score in enumerate(
        ordered_scores
    ):
        diminishing_weight = 1.0 / (
            1.0
            + 0.35 * rank
        )

        total_score += (
            score
            * diminishing_weight
        )

    unique_ligand_atoms = len(
        {
            atom_deduplication_key(
                interaction.ligand_atom,
                strategy="auto",
            )
            for interaction in interaction_tuple
        }
    )

    diversity_bonus = min(
        0.25 * max(
            unique_ligand_atoms - 1,
            0,
        ),
        0.75,
    )

    return np.float64(
        total_score
        + diversity_bonus
    )


def is_hydrophobic_hotspot(
    interactions: HydrophobicInteractionCollection,
    *,
    minimum_contact_count: Optional[int] = None,
    minimum_group_score: Optional[Number] = None,
    minimum_contact_area: Optional[Number] = None,
    minimum_ligand_atom_count: Optional[int] = None,
) -> bool:
    """
    Determine whether an interaction group represents a hotspot.

    A hotspot must satisfy all configured criteria:

    - minimum number of atom-pair contacts;
    - minimum group score;
    - minimum approximate hydrophobic surface;
    - minimum number of distinct ligand atoms.
    """

    interaction_tuple = tuple(interactions)

    contact_count_limit = (
        DEFAULT_HOTSPOT_MINIMUM_CONTACT_COUNT
        if minimum_contact_count is None
        else _nonnegative_integer(
            minimum_contact_count,
            name="minimum hotspot contact count",
        )
    )

    group_score_limit = (
        DEFAULT_HOTSPOT_MINIMUM_SCORE
        if minimum_group_score is None
        else _nonnegative_float(
            minimum_group_score,
            name="minimum hotspot group score",
        )
    )

    area_limit = (
        DEFAULT_HOTSPOT_MINIMUM_CONTACT_AREA
        if minimum_contact_area is None
        else _nonnegative_float(
            minimum_contact_area,
            name="minimum hotspot contact area",
        )
    )

    ligand_atom_limit = (
        DEFAULT_HOTSPOT_MINIMUM_LIGAND_ATOM_COUNT
        if minimum_ligand_atom_count is None
        else _nonnegative_integer(
            minimum_ligand_atom_count,
            name="minimum hotspot ligand atom count",
        )
    )

    if len(interaction_tuple) < contact_count_limit:
        return False

    group_score = calculate_residue_group_score(
        interaction_tuple
    )

    if group_score < group_score_limit:
        return False

    contact_area = approximate_hydrophobic_surface_area(
        interaction_tuple
    )

    if contact_area < area_limit:
        return False

    unique_ligand_atom_count = len(
        {
            atom_deduplication_key(
                interaction.ligand_atom,
                strategy="auto",
            )
            for interaction in interaction_tuple
        }
    )

    return bool(
        unique_ligand_atom_count
        >= ligand_atom_limit
    )


# -----------------------------------------------------------------------------
# Extended grouping dataclasses
# -----------------------------------------------------------------------------

@dataclass(
    frozen=True,
    slots=True,
)
class HydrophobicLocalRegion:
    """
    Local cluster of hydrophobic atom-pair contacts.

    Several atomic contacts may describe one broader interaction region,
    especially when neighboring receptor and ligand atoms form a compact
    contact surface.
    """

    interactions: Sequence[
        HydrophobicInteraction
    ] = field(
        default_factory=tuple
    )

    region_identifier: Optional[str] = None
    pose_identifier: Optional[str] = None
    chain_identifiers: Sequence[str] = field(
        default_factory=tuple
    )
    residue_identifiers: Sequence[str] = field(
        default_factory=tuple
    )

    centroid: Optional[Coordinate] = None

    closest_interaction: Optional[
        HydrophobicInteraction
    ] = None

    representative_interaction: Optional[
        HydrophobicInteraction
    ] = None

    total_score: Optional[np.float64] = None
    approximate_contact_area: Optional[np.float64] = None
    contact_density: Optional[np.float64] = None

    is_hotspot: Optional[bool] = None

    metadata: Mapping[str, Any] = field(
        default_factory=lambda: _EMPTY_METADATA,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        """Validate and normalize a local region."""

        interactions = tuple(
            self.interactions
        )

        for index, interaction in enumerate(
            interactions
        ):
            if not isinstance(
                interaction,
                HydrophobicInteraction,
            ):
                raise TypeError(
                    "interactions must contain "
                    "HydrophobicInteraction instances. "
                    f"Invalid entry at index {index}."
                )

        pose_identifiers = {
            get_interaction_pose_identifier(
                interaction
            )
            for interaction in interactions
        }

        pose_identifier = _normalize_optional_string(
            self.pose_identifier
        )

        if pose_identifier is None:
            if len(pose_identifiers) == 1:
                pose_identifier = next(
                    iter(pose_identifiers)
                )

            elif pose_identifiers:
                pose_identifier = "|".join(
                    sorted(pose_identifiers)
                )

        chain_identifiers = tuple(
            sorted(
                {
                    get_interaction_chain_identifier(
                        interaction,
                        default="",
                    )
                    for interaction in interactions
                    if get_interaction_chain_identifier(
                        interaction,
                        default="",
                    )
                }
            )
        )

        if self.chain_identifiers:
            chain_identifiers = tuple(
                sorted(
                    {
                        str(value).strip()
                        for value in self.chain_identifiers
                        if str(value).strip()
                    }
                )
            )

        residue_identifiers = tuple(
            sorted(
                {
                    interaction.receptor_residue_identifier
                    or "residue-unknown"
                    for interaction in interactions
                }
            )
        )

        if self.residue_identifiers:
            residue_identifiers = tuple(
                sorted(
                    {
                        str(value).strip()
                        for value in self.residue_identifiers
                        if str(value).strip()
                    }
                )
            )

        centroid = self.centroid

        if centroid is None and interactions:
            centroid = np.mean(
                np.asarray(
                    [
                        hydrophobic_interaction_midpoint(
                            interaction
                        )
                        for interaction in interactions
                    ],
                    dtype=np.float64,
                ),
                axis=0,
            )

        normalized_centroid = (
            None
            if centroid is None
            else _normalize_coordinate(
                centroid,
                name="local region centroid",
            )
        )

        closest_interaction = (
            self.closest_interaction
            or select_closest_hydrophobic_interaction(
                interactions
            )
        )

        representative_interaction = (
            self.representative_interaction
            or select_highest_scoring_hydrophobic_interaction(
                interactions
            )
        )

        total_score = (
            calculate_residue_group_score(
                interactions
            )
            if self.total_score is None
            else _nonnegative_float(
                self.total_score,
                name="local region total score",
            )
        )

        contact_area = (
            approximate_hydrophobic_surface_area(
                interactions
            )
            if self.approximate_contact_area is None
            else _nonnegative_float(
                self.approximate_contact_area,
                name="local region contact area",
            )
        )

        density = self.contact_density

        if density is None:
            receptor_atoms = deduplicate_atoms(
                (
                    interaction.receptor_atom
                    for interaction in interactions
                ),
                strategy="auto",
            )

            ligand_atoms = deduplicate_atoms(
                (
                    interaction.ligand_atom
                    for interaction in interactions
                ),
                strategy="auto",
            )

            density = approximate_contact_density(
                receptor_atoms,
                ligand_atoms,
                minimum_distance=0.0,
                maximum_distance=(
                    get_default_maximum_hydrophobic_distance()
                ),
            ) if (
                receptor_atoms
                and ligand_atoms
            ) else np.float64(0.0)

        normalized_density = (
            validate_hydrophobic_score(
                density
            )
        )

        hotspot = (
            is_hydrophobic_hotspot(
                interactions
            )
            if self.is_hotspot is None
            else bool(self.is_hotspot)
        )

        region_identifier = (
            _normalize_optional_string(
                self.region_identifier
            )
        )

        if region_identifier is None:
            residue_component = (
                ",".join(residue_identifiers)
                if residue_identifiers
                else "residue-unknown"
            )

            region_identifier = (
                f"hydrophobic-region|"
                f"{pose_identifier or 'pose-unknown'}|"
                f"{residue_component}"
            )

        object.__setattr__(
            self,
            "interactions",
            interactions,
        )

        object.__setattr__(
            self,
            "region_identifier",
            region_identifier,
        )

        object.__setattr__(
            self,
            "pose_identifier",
            pose_identifier,
        )

        object.__setattr__(
            self,
            "chain_identifiers",
            chain_identifiers,
        )

        object.__setattr__(
            self,
            "residue_identifiers",
            residue_identifiers,
        )

        object.__setattr__(
            self,
            "centroid",
            normalized_centroid,
        )

        object.__setattr__(
            self,
            "closest_interaction",
            closest_interaction,
        )

        object.__setattr__(
            self,
            "representative_interaction",
            representative_interaction,
        )

        object.__setattr__(
            self,
            "total_score",
            total_score,
        )

        object.__setattr__(
            self,
            "approximate_contact_area",
            contact_area,
        )

        object.__setattr__(
            self,
            "contact_density",
            normalized_density,
        )

        object.__setattr__(
            self,
            "is_hotspot",
            hotspot,
        )

        object.__setattr__(
            self,
            "metadata",
            _freeze_metadata(self.metadata),
        )

    def __len__(self) -> int:
        """Return the number of atomic contacts in the region."""

        return len(self.interactions)

    def __iter__(self) -> Iterator[HydrophobicInteraction]:
        """Iterate over region interactions."""

        return iter(self.interactions)

    @property
    def interaction_count(self) -> int:
        """Return the number of atom-pair contacts."""

        return len(self.interactions)

    @property
    def residue_count(self) -> int:
        """Return the number of receptor residues represented."""

        return len(self.residue_identifiers)

    @property
    def unique_receptor_atom_count(self) -> int:
        """Return the number of unique receptor atoms."""

        return len(
            {
                atom_deduplication_key(
                    interaction.receptor_atom,
                    strategy="auto",
                )
                for interaction in self.interactions
            }
        )

    @property
    def unique_ligand_atom_count(self) -> int:
        """Return the number of unique ligand atoms."""

        return len(
            {
                atom_deduplication_key(
                    interaction.ligand_atom,
                    strategy="auto",
                )
                for interaction in self.interactions
            }
        )

    @property
    def minimum_distance(self) -> Optional[np.float64]:
        """Return the minimum atom-pair distance."""

        if self.closest_interaction is None:
            return None

        return self.closest_interaction.distance

    def to_dict(
        self,
        *,
        include_interactions: bool = True,
        include_atoms: bool = False,
    ) -> Dict[str, Any]:
        """Serialize the local interaction region."""

        result: Dict[str, Any] = {
            "region_identifier": self.region_identifier,
            "pose_identifier": self.pose_identifier,
            "chain_identifiers": list(
                self.chain_identifiers
            ),
            "residue_identifiers": list(
                self.residue_identifiers
            ),
            "interaction_count": self.interaction_count,
            "residue_count": self.residue_count,
            "unique_receptor_atom_count": (
                self.unique_receptor_atom_count
            ),
            "unique_ligand_atom_count": (
                self.unique_ligand_atom_count
            ),
            "centroid": (
                None
                if self.centroid is None
                else self.centroid.tolist()
            ),
            "minimum_distance": (
                None
                if self.minimum_distance is None
                else float(self.minimum_distance)
            ),
            "total_score": float(
                self.total_score or 0.0
            ),
            "approximate_contact_area": float(
                self.approximate_contact_area or 0.0
            ),
            "contact_density": float(
                self.contact_density or 0.0
            ),
            "is_hotspot": bool(self.is_hotspot),
            "closest_interaction_identifier": (
                None
                if self.closest_interaction is None
                else self.closest_interaction.interaction_identifier
            ),
            "representative_interaction_identifier": (
                None
                if self.representative_interaction is None
                else self.representative_interaction.interaction_identifier
            ),
            "metadata": dict(self.metadata),
        }

        if include_interactions:
            result["interactions"] = [
                interaction.to_dict(
                    include_atoms=include_atoms,
                    include_residue=include_atoms,
                    include_descriptors=True,
                )
                for interaction in self.interactions
            ]

        return result


@dataclass(
    frozen=True,
    slots=True,
)
class HydrophobicChainGroup:
    """
    Hydrophobic interactions grouped by receptor chain.
    """

    chain_identifier: str

    residue_groups: Sequence[
        HydrophobicResidueGroup
    ] = field(
        default_factory=tuple
    )

    interactions: Sequence[
        HydrophobicInteraction
    ] = field(
        default_factory=tuple
    )

    local_regions: Sequence[
        HydrophobicLocalRegion
    ] = field(
        default_factory=tuple
    )

    metadata: Mapping[str, Any] = field(
        default_factory=lambda: _EMPTY_METADATA,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        """Validate and normalize a chain group."""

        chain_identifier = str(
            self.chain_identifier
        ).strip()

        residue_groups = tuple(
            self.residue_groups
        )

        interactions = tuple(
            self.interactions
        )

        local_regions = tuple(
            self.local_regions
        )

        for group in residue_groups:
            if not isinstance(
                group,
                HydrophobicResidueGroup,
            ):
                raise TypeError(
                    "residue_groups must contain "
                    "HydrophobicResidueGroup instances."
                )

        for interaction in interactions:
            if not isinstance(
                interaction,
                HydrophobicInteraction,
            ):
                raise TypeError(
                    "interactions must contain "
                    "HydrophobicInteraction instances."
                )

        for region in local_regions:
            if not isinstance(
                region,
                HydrophobicLocalRegion,
            ):
                raise TypeError(
                    "local_regions must contain "
                    "HydrophobicLocalRegion instances."
                )

        object.__setattr__(
            self,
            "chain_identifier",
            chain_identifier,
        )

        object.__setattr__(
            self,
            "residue_groups",
            residue_groups,
        )

        object.__setattr__(
            self,
            "interactions",
            interactions,
        )

        object.__setattr__(
            self,
            "local_regions",
            local_regions,
        )

        object.__setattr__(
            self,
            "metadata",
            _freeze_metadata(self.metadata),
        )

    @property
    def interaction_count(self) -> int:
        """Return the chain interaction count."""

        return len(self.interactions)

    @property
    def residue_count(self) -> int:
        """Return the contacted residue count."""

        return len(self.residue_groups)

    @property
    def hotspot_count(self) -> int:
        """Return the number of hotspot residue groups."""

        return sum(
            bool(
                group.metadata.get(
                    "is_hotspot",
                    group.is_hotspot,
                )
            )
            for group in self.residue_groups
        )

    @property
    def total_score(self) -> np.float64:
        """Return the sum of residue-group scores."""

        return np.float64(
            sum(
                float(group.group_score or 0.0)
                for group in self.residue_groups
            )
        )

    @property
    def approximate_contact_area(self) -> np.float64:
        """Return the chain-level hydrophobic contact area."""

        return approximate_hydrophobic_surface_area(
            self.interactions
        )

    def to_dict(
        self,
        *,
        include_interactions: bool = False,
    ) -> Dict[str, Any]:
        """Serialize a chain group."""

        return {
            "chain_identifier": self.chain_identifier,
            "interaction_count": self.interaction_count,
            "residue_count": self.residue_count,
            "hotspot_count": self.hotspot_count,
            "total_score": float(self.total_score),
            "approximate_contact_area": float(
                self.approximate_contact_area
            ),
            "residue_groups": [
                group.to_dict(
                    include_interactions=(
                        include_interactions
                    ),
                    include_atoms=False,
                    include_residue=False,
                    include_descriptors=True,
                )
                for group in self.residue_groups
            ],
            "local_regions": [
                region.to_dict(
                    include_interactions=(
                        include_interactions
                    ),
                    include_atoms=False,
                )
                for region in self.local_regions
            ],
            "metadata": dict(self.metadata),
        }


@dataclass(
    frozen=True,
    slots=True,
)
class HydrophobicPoseGroup:
    """
    Hydrophobic interactions grouped by ligand pose.
    """

    pose_identifier: str

    interactions: Sequence[
        HydrophobicInteraction
    ] = field(
        default_factory=tuple
    )

    residue_groups: Sequence[
        HydrophobicResidueGroup
    ] = field(
        default_factory=tuple
    )

    chain_groups: Sequence[
        HydrophobicChainGroup
    ] = field(
        default_factory=tuple
    )

    local_regions: Sequence[
        HydrophobicLocalRegion
    ] = field(
        default_factory=tuple
    )

    metadata: Mapping[str, Any] = field(
        default_factory=lambda: _EMPTY_METADATA,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        """Validate and freeze a pose group."""

        pose_identifier = _normalize_required_string(
            self.pose_identifier,
            name="pose identifier",
        )

        interactions = tuple(
            self.interactions
        )

        residue_groups = tuple(
            self.residue_groups
        )

        chain_groups = tuple(
            self.chain_groups
        )

        local_regions = tuple(
            self.local_regions
        )

        object.__setattr__(
            self,
            "pose_identifier",
            pose_identifier,
        )

        object.__setattr__(
            self,
            "interactions",
            interactions,
        )

        object.__setattr__(
            self,
            "residue_groups",
            residue_groups,
        )

        object.__setattr__(
            self,
            "chain_groups",
            chain_groups,
        )

        object.__setattr__(
            self,
            "local_regions",
            local_regions,
        )

        object.__setattr__(
            self,
            "metadata",
            _freeze_metadata(self.metadata),
        )

    @property
    def interaction_count(self) -> int:
        """Return the pose interaction count."""

        return len(self.interactions)

    @property
    def residue_count(self) -> int:
        """Return the number of contacted residues."""

        return len(self.residue_groups)

    @property
    def hotspot_count(self) -> int:
        """Return the number of hotspot residues."""

        return sum(
            bool(
                group.metadata.get(
                    "is_hotspot",
                    group.is_hotspot,
                )
            )
            for group in self.residue_groups
        )

    @property
    def total_score(self) -> np.float64:
        """Return the sum of interaction scores."""

        return np.float64(
            sum(
                float(interaction.score)
                for interaction in self.interactions
            )
        )

    @property
    def approximate_contact_area(self) -> np.float64:
        """Return the pose-level contact area."""

        return approximate_hydrophobic_surface_area(
            self.interactions
        )

    def to_dict(
        self,
        *,
        include_interactions: bool = True,
    ) -> Dict[str, Any]:
        """Serialize the pose group."""

        result: Dict[str, Any] = {
            "pose_identifier": self.pose_identifier,
            "interaction_count": self.interaction_count,
            "residue_count": self.residue_count,
            "hotspot_count": self.hotspot_count,
            "total_score": float(self.total_score),
            "approximate_contact_area": float(
                self.approximate_contact_area
            ),
            "residue_groups": [
                group.to_dict(
                    include_interactions=False,
                    include_atoms=False,
                    include_residue=False,
                    include_descriptors=True,
                )
                for group in self.residue_groups
            ],
            "chain_groups": [
                group.to_dict(
                    include_interactions=False
                )
                for group in self.chain_groups
            ],
            "local_regions": [
                region.to_dict(
                    include_interactions=False,
                    include_atoms=False,
                )
                for region in self.local_regions
            ],
            "metadata": dict(self.metadata),
        }

        if include_interactions:
            result["interactions"] = [
                interaction.to_dict(
                    include_atoms=False,
                    include_residue=False,
                    include_descriptors=True,
                )
                for interaction in self.interactions
            ]

        return result


# -----------------------------------------------------------------------------
# Residue grouping
# -----------------------------------------------------------------------------

def group_hydrophobic_interactions_by_residue(
    interactions: HydrophobicInteractionCollection,
    *,
    identify_hotspots: bool = True,
    hotspot_minimum_contact_count: Optional[int] = None,
    hotspot_minimum_group_score: Optional[Number] = None,
    hotspot_minimum_contact_area: Optional[Number] = None,
    hotspot_minimum_ligand_atom_count: Optional[int] = None,
    sort_by: Literal[
        "score",
        "distance",
        "count",
        "identifier",
    ] = DEFAULT_RESIDUE_GROUP_SORTING,
) -> Tuple[
    HydrophobicResidueGroup,
    ...,
]:
    """
    Group hydrophobic interactions by receptor residue.
    """

    interaction_tuple = (
        deduplicate_hydrophobic_interactions(
            interactions
        )
    )

    grouped_interactions: Dict[
        Tuple[Any, ...],
        List[HydrophobicInteraction],
    ] = {}

    group_order: List[
        Tuple[Any, ...]
    ] = []

    for interaction in interaction_tuple:
        key = hydrophobic_residue_group_key(
            interaction
        )

        if key not in grouped_interactions:
            grouped_interactions[key] = []
            group_order.append(key)

        grouped_interactions[
            key
        ].append(interaction)

    residue_groups: List[
        HydrophobicResidueGroup
    ] = []

    for key in group_order:
        residue_interactions = tuple(
            grouped_interactions[key]
        )

        closest_interaction = (
            select_closest_hydrophobic_interaction(
                residue_interactions
            )
        )

        representative_interaction = (
            select_highest_scoring_hydrophobic_interaction(
                residue_interactions
            )
        )

        group_score = calculate_residue_group_score(
            residue_interactions
        )

        contact_area = (
            approximate_hydrophobic_surface_area(
                residue_interactions
            )
        )

        receptor_atoms = deduplicate_atoms(
            (
                interaction.receptor_atom
                for interaction in residue_interactions
            ),
            strategy="auto",
        )

        ligand_atoms = deduplicate_atoms(
            (
                interaction.ligand_atom
                for interaction in residue_interactions
            ),
            strategy="auto",
        )

        contact_density = (
            approximate_contact_density(
                receptor_atoms,
                ligand_atoms,
                minimum_distance=0.0,
                maximum_distance=(
                    get_default_maximum_hydrophobic_distance()
                ),
            )
            if receptor_atoms and ligand_atoms
            else np.float64(0.0)
        )

        hotspot = (
            is_hydrophobic_hotspot(
                residue_interactions,
                minimum_contact_count=(
                    hotspot_minimum_contact_count
                ),
                minimum_group_score=(
                    hotspot_minimum_group_score
                ),
                minimum_contact_area=(
                    hotspot_minimum_contact_area
                ),
                minimum_ligand_atom_count=(
                    hotspot_minimum_ligand_atom_count
                ),
            )
            if identify_hotspots
            else False
        )

        first_interaction = (
            residue_interactions[0]
        )

        group_metadata = {
            "chain_identifier": (
                get_interaction_chain_identifier(
                    first_interaction,
                    default="",
                )
            ),
            "pose_identifiers": tuple(
                sorted(
                    {
                        get_interaction_pose_identifier(
                            interaction
                        )
                        for interaction
                        in residue_interactions
                    }
                )
            ),
            "closest_interaction_identifier": (
                None
                if closest_interaction is None
                else closest_interaction.interaction_identifier
            ),
            "representative_interaction_identifier": (
                None
                if representative_interaction is None
                else representative_interaction.interaction_identifier
            ),
            "contact_count": len(
                residue_interactions
            ),
            "approximate_contact_area": float(
                contact_area
            ),
            "contact_density": float(
                contact_density
            ),
            "is_hotspot": hotspot,
            "unique_receptor_atom_count": len(
                receptor_atoms
            ),
            "unique_ligand_atom_count": len(
                ligand_atoms
            ),
        }

        residue_groups.append(
            HydrophobicResidueGroup(
                residue=(
                    first_interaction.receptor_residue
                ),
                residue_key=(
                    first_interaction.receptor_residue_key
                ),
                interactions=residue_interactions,
                residue_identifier=(
                    first_interaction.receptor_residue_identifier
                ),
                group_score=group_score,
                metadata=group_metadata,
            )
        )

    normalized_sorting = str(
        sort_by
    ).strip().lower()

    if normalized_sorting == "score":
        residue_groups.sort(
            key=lambda group: (
                -float(group.group_score or 0.0),
                float(
                    group.minimum_distance
                    if group.minimum_distance is not None
                    else np.inf
                ),
                group.residue_identifier or "",
            )
        )

    elif normalized_sorting == "distance":
        residue_groups.sort(
            key=lambda group: (
                float(
                    group.minimum_distance
                    if group.minimum_distance is not None
                    else np.inf
                ),
                -float(group.group_score or 0.0),
                group.residue_identifier or "",
            )
        )

    elif normalized_sorting == "count":
        residue_groups.sort(
            key=lambda group: (
                -group.interaction_count,
                -float(group.group_score or 0.0),
                group.residue_identifier or "",
            )
        )

    elif normalized_sorting == "identifier":
        residue_groups.sort(
            key=lambda group: (
                group.residue_identifier or ""
            )
        )

    else:
        raise ValueError(
            "sort_by must be 'score', 'distance', "
            "'count' or 'identifier'."
        )

    return tuple(residue_groups)


# -----------------------------------------------------------------------------
# Local-region clustering
# -----------------------------------------------------------------------------

def cluster_hydrophobic_local_regions(
    interactions: HydrophobicInteractionCollection,
    *,
    grouping_distance: Optional[Number] = None,
    require_same_pose: bool = True,
    require_same_chain: bool = False,
    identify_hotspots: bool = True,
    sort_by: Literal[
        "score",
        "distance",
        "count",
        "identifier",
    ] = DEFAULT_LOCAL_REGION_SORTING,
) -> Tuple[
    HydrophobicLocalRegion,
    ...,
]:
    """
    Cluster multiple atom-pair contacts into local interaction regions.

    A connected-components procedure is used. Two interactions belong to
    the same component when they share an atom or their pair midpoints are
    spatially close.
    """

    interaction_tuple = (
        deduplicate_hydrophobic_interactions(
            interactions
        )
    )

    if not interaction_tuple:
        return ()

    cutoff = (
        DEFAULT_LOCAL_INTERACTION_CLUSTER_DISTANCE
        if grouping_distance is None
        else _positive_float(
            grouping_distance,
            name="local region grouping distance",
        )
    )

    adjacency: Dict[
        int,
        Set[int],
    ] = {
        index: set()
        for index in range(
            len(interaction_tuple)
        )
    }

    for first_index in range(
        len(interaction_tuple)
    ):
        first = interaction_tuple[
            first_index
        ]

        for second_index in range(
            first_index + 1,
            len(interaction_tuple),
        ):
            second = interaction_tuple[
                second_index
            ]

            if require_same_chain:
                if (
                    get_interaction_chain_identifier(
                        first
                    )
                    != get_interaction_chain_identifier(
                        second
                    )
                ):
                    continue

            if are_hydrophobic_interactions_locally_connected(
                first,
                second,
                grouping_distance=cutoff,
                require_same_pose=require_same_pose,
            ):
                adjacency[
                    first_index
                ].add(second_index)

                adjacency[
                    second_index
                ].add(first_index)

    visited: Set[int] = set()
    components: List[
        Tuple[HydrophobicInteraction, ...]
    ] = []

    for starting_index in range(
        len(interaction_tuple)
    ):
        if starting_index in visited:
            continue

        stack = [
            starting_index
        ]

        component_indices: List[int] = []

        while stack:
            current_index = stack.pop()

            if current_index in visited:
                continue

            visited.add(current_index)
            component_indices.append(
                current_index
            )

            stack.extend(
                adjacency[current_index]
                - visited
            )

        component = tuple(
            interaction_tuple[index]
            for index in sorted(
                component_indices
            )
        )

        components.append(component)

    regions: List[
        HydrophobicLocalRegion
    ] = []

    for region_index, component in enumerate(
        components,
        start=1,
    ):
        pose_identifier, residue_identifiers = (
            hydrophobic_local_region_key(
                component
            )
        )

        region_metadata = {
            "grouping_distance": float(
                cutoff
            ),
            "require_same_pose": bool(
                require_same_pose
            ),
            "require_same_chain": bool(
                require_same_chain
            ),
            "atomic_pair_count": len(
                component
            ),
        }

        region = HydrophobicLocalRegion(
            interactions=component,
            region_identifier=(
                f"hydrophobic-region-"
                f"{region_index}|"
                f"{pose_identifier or 'pose-unknown'}"
            ),
            pose_identifier=pose_identifier,
            residue_identifiers=(
                residue_identifiers
            ),
            is_hotspot=(
                is_hydrophobic_hotspot(
                    component
                )
                if identify_hotspots
                else False
            ),
            metadata=region_metadata,
        )

        regions.append(region)

    normalized_sorting = str(
        sort_by
    ).strip().lower()

    if normalized_sorting == "score":
        regions.sort(
            key=lambda region: (
                -float(region.total_score or 0.0),
                float(
                    region.minimum_distance
                    if region.minimum_distance is not None
                    else np.inf
                ),
                region.region_identifier or "",
            )
        )

    elif normalized_sorting == "distance":
        regions.sort(
            key=lambda region: (
                float(
                    region.minimum_distance
                    if region.minimum_distance is not None
                    else np.inf
                ),
                -float(region.total_score or 0.0),
                region.region_identifier or "",
            )
        )

    elif normalized_sorting == "count":
        regions.sort(
            key=lambda region: (
                -region.interaction_count,
                -float(region.total_score or 0.0),
                region.region_identifier or "",
            )
        )

    elif normalized_sorting == "identifier":
        regions.sort(
            key=lambda region: (
                region.region_identifier or ""
            )
        )

    else:
        raise ValueError(
            "sort_by must be 'score', 'distance', "
            "'count' or 'identifier'."
        )

    return tuple(regions)


# -----------------------------------------------------------------------------
# Grouping by chain
# -----------------------------------------------------------------------------

def group_hydrophobic_interactions_by_chain(
    interactions: HydrophobicInteractionCollection,
    *,
    residue_groups: Optional[
        Sequence[HydrophobicResidueGroup]
    ] = None,
    local_regions: Optional[
        Sequence[HydrophobicLocalRegion]
    ] = None,
    include_unknown_chain: bool = True,
) -> Tuple[
    HydrophobicChainGroup,
    ...,
]:
    """
    Group hydrophobic interactions by receptor chain.
    """

    interaction_tuple = (
        deduplicate_hydrophobic_interactions(
            interactions
        )
    )

    resolved_residue_groups = (
        group_hydrophobic_interactions_by_residue(
            interaction_tuple
        )
        if residue_groups is None
        else tuple(residue_groups)
    )

    resolved_local_regions = (
        cluster_hydrophobic_local_regions(
            interaction_tuple,
            require_same_chain=True,
        )
        if local_regions is None
        else tuple(local_regions)
    )

    interaction_map: Dict[
        str,
        List[HydrophobicInteraction],
    ] = {}

    for interaction in interaction_tuple:
        chain_identifier = (
            get_interaction_chain_identifier(
                interaction,
                default="",
            )
        )

        if (
            not chain_identifier
            and not include_unknown_chain
        ):
            continue

        chain_key = (
            chain_identifier
            or "chain-unknown"
        )

        interaction_map.setdefault(
            chain_key,
            [],
        ).append(interaction)

    chain_groups: List[
        HydrophobicChainGroup
    ] = []

    for chain_identifier, chain_interactions in (
        interaction_map.items()
    ):
        chain_residue_groups = tuple(
            group
            for group in resolved_residue_groups
            if (
                str(
                    group.metadata.get(
                        "chain_identifier",
                        "",
                    )
                ).strip()
                or "chain-unknown"
            )
            == chain_identifier
        )

        chain_local_regions = tuple(
            region
            for region in resolved_local_regions
            if (
                chain_identifier
                in {
                    value or "chain-unknown"
                    for value in region.chain_identifiers
                }
                or (
                    chain_identifier == "chain-unknown"
                    and not region.chain_identifiers
                )
            )
        )

        chain_groups.append(
            HydrophobicChainGroup(
                chain_identifier=chain_identifier,
                residue_groups=chain_residue_groups,
                interactions=tuple(
                    chain_interactions
                ),
                local_regions=chain_local_regions,
                metadata={
                    "grouping_level": "chain",
                },
            )
        )

    chain_groups.sort(
        key=lambda group: (
            -float(group.total_score),
            -group.interaction_count,
            group.chain_identifier,
        )
    )

    return tuple(chain_groups)


# -----------------------------------------------------------------------------
# Grouping by pose
# -----------------------------------------------------------------------------

def group_hydrophobic_interactions_by_pose(
    interactions: HydrophobicInteractionCollection,
    *,
    include_unknown_pose: bool = True,
) -> Tuple[
    HydrophobicPoseGroup,
    ...,
]:
    """
    Group interactions by ligand pose identifier.
    """

    interaction_tuple = (
        deduplicate_hydrophobic_interactions(
            interactions
        )
    )

    pose_map: Dict[
        str,
        List[HydrophobicInteraction],
    ] = {}

    for interaction in interaction_tuple:
        pose_identifier = (
            get_interaction_pose_identifier(
                interaction
            )
        )

        if (
            pose_identifier == "pose-unknown"
            and not include_unknown_pose
        ):
            continue

        pose_map.setdefault(
            pose_identifier,
            [],
        ).append(interaction)

    pose_groups: List[
        HydrophobicPoseGroup
    ] = []

    for pose_identifier, pose_interactions in (
        pose_map.items()
    ):
        pose_interaction_tuple = tuple(
            pose_interactions
        )

        residue_groups = (
            group_hydrophobic_interactions_by_residue(
                pose_interaction_tuple
            )
        )

        local_regions = (
            cluster_hydrophobic_local_regions(
                pose_interaction_tuple,
                require_same_pose=True,
            )
        )

        chain_groups = (
            group_hydrophobic_interactions_by_chain(
                pose_interaction_tuple,
                residue_groups=residue_groups,
                local_regions=local_regions,
            )
        )

        pose_groups.append(
            HydrophobicPoseGroup(
                pose_identifier=pose_identifier,
                interactions=pose_interaction_tuple,
                residue_groups=residue_groups,
                chain_groups=chain_groups,
                local_regions=local_regions,
                metadata={
                    "grouping_level": "pose",
                },
            )
        )

    pose_groups.sort(
        key=lambda group: (
            -float(group.total_score),
            -group.interaction_count,
            group.pose_identifier,
        )
    )

    return tuple(pose_groups)


# -----------------------------------------------------------------------------
# Contact-count and hotspot summaries
# -----------------------------------------------------------------------------

def count_hydrophobic_contacts_by_residue(
    interactions: HydrophobicInteractionCollection,
) -> Mapping[str, int]:
    """
    Return the number of atomic contacts associated with each residue.
    """

    counts: Dict[str, int] = {}

    for group in group_hydrophobic_interactions_by_residue(
        interactions
    ):
        identifier = (
            group.residue_identifier
            or "residue-unknown"
        )

        counts[identifier] = (
            group.interaction_count
        )

    return MappingProxyType(counts)


def hydrophobic_surface_by_residue(
    interactions: HydrophobicInteractionCollection,
) -> Mapping[str, float]:
    """
    Return approximate hydrophobic contact area by receptor residue.
    """

    surfaces: Dict[str, float] = {}

    for group in group_hydrophobic_interactions_by_residue(
        interactions
    ):
        identifier = (
            group.residue_identifier
            or "residue-unknown"
        )

        surfaces[identifier] = float(
            approximate_hydrophobic_surface_area(
                group.interactions
            )
        )

    return MappingProxyType(surfaces)


def find_hydrophobic_hotspots(
    interactions: HydrophobicInteractionCollection,
    *,
    minimum_contact_count: Optional[int] = None,
    minimum_group_score: Optional[Number] = None,
    minimum_contact_area: Optional[Number] = None,
    minimum_ligand_atom_count: Optional[int] = None,
) -> Tuple[
    HydrophobicResidueGroup,
    ...,
]:
    """
    Return receptor-residue groups satisfying hotspot criteria.
    """

    groups = (
        group_hydrophobic_interactions_by_residue(
            interactions,
            identify_hotspots=True,
            hotspot_minimum_contact_count=(
                minimum_contact_count
            ),
            hotspot_minimum_group_score=(
                minimum_group_score
            ),
            hotspot_minimum_contact_area=(
                minimum_contact_area
            ),
            hotspot_minimum_ligand_atom_count=(
                minimum_ligand_atom_count
            ),
        )
    )

    return tuple(
        group
        for group in groups
        if bool(
            group.metadata.get(
                "is_hotspot",
                False,
            )
        )
    )


# -----------------------------------------------------------------------------
# Complete grouping result
# -----------------------------------------------------------------------------

@dataclass(
    frozen=True,
    slots=True,
)
class HydrophobicGroupingResult:
    """
    Complete residue, chain, pose and local-region grouping result.
    """

    interactions: Sequence[
        HydrophobicInteraction
    ] = field(
        default_factory=tuple
    )

    residue_groups: Sequence[
        HydrophobicResidueGroup
    ] = field(
        default_factory=tuple
    )

    chain_groups: Sequence[
        HydrophobicChainGroup
    ] = field(
        default_factory=tuple
    )

    pose_groups: Sequence[
        HydrophobicPoseGroup
    ] = field(
        default_factory=tuple
    )

    local_regions: Sequence[
        HydrophobicLocalRegion
    ] = field(
        default_factory=tuple
    )

    hotspot_groups: Sequence[
        HydrophobicResidueGroup
    ] = field(
        default_factory=tuple
    )

    metadata: Mapping[str, Any] = field(
        default_factory=lambda: _EMPTY_METADATA,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        """Validate and freeze the grouping result."""

        object.__setattr__(
            self,
            "interactions",
            tuple(self.interactions),
        )

        object.__setattr__(
            self,
            "residue_groups",
            tuple(self.residue_groups),
        )

        object.__setattr__(
            self,
            "chain_groups",
            tuple(self.chain_groups),
        )

        object.__setattr__(
            self,
            "pose_groups",
            tuple(self.pose_groups),
        )

        object.__setattr__(
            self,
            "local_regions",
            tuple(self.local_regions),
        )

        object.__setattr__(
            self,
            "hotspot_groups",
            tuple(self.hotspot_groups),
        )

        object.__setattr__(
            self,
            "metadata",
            _freeze_metadata(self.metadata),
        )

    @property
    def interaction_count(self) -> int:
        """Return the total interaction count."""

        return len(self.interactions)

    @property
    def residue_count(self) -> int:
        """Return the contacted residue count."""

        return len(self.residue_groups)

    @property
    def chain_count(self) -> int:
        """Return the contacted chain count."""

        return len(self.chain_groups)

    @property
    def pose_count(self) -> int:
        """Return the represented pose count."""

        return len(self.pose_groups)

    @property
    def local_region_count(self) -> int:
        """Return the local interaction-region count."""

        return len(self.local_regions)

    @property
    def hotspot_count(self) -> int:
        """Return the hotspot residue count."""

        return len(self.hotspot_groups)

    @property
    def approximate_contact_area(self) -> np.float64:
        """Return the total approximate contact area."""

        return approximate_hydrophobic_surface_area(
            self.interactions
        )

    def to_dict(
        self,
        *,
        include_interactions: bool = True,
    ) -> Dict[str, Any]:
        """Serialize the grouping result."""

        result: Dict[str, Any] = {
            "interaction_count": self.interaction_count,
            "residue_count": self.residue_count,
            "chain_count": self.chain_count,
            "pose_count": self.pose_count,
            "local_region_count": (
                self.local_region_count
            ),
            "hotspot_count": self.hotspot_count,
            "approximate_contact_area": float(
                self.approximate_contact_area
            ),
            "residue_groups": [
                group.to_dict(
                    include_interactions=False,
                    include_atoms=False,
                    include_residue=False,
                    include_descriptors=True,
                )
                for group in self.residue_groups
            ],
            "chain_groups": [
                group.to_dict(
                    include_interactions=False
                )
                for group in self.chain_groups
            ],
            "pose_groups": [
                group.to_dict(
                    include_interactions=False
                )
                for group in self.pose_groups
            ],
            "local_regions": [
                region.to_dict(
                    include_interactions=False,
                    include_atoms=False,
                )
                for region in self.local_regions
            ],
            "hotspot_residue_identifiers": [
                group.residue_identifier
                for group in self.hotspot_groups
            ],
            "metadata": dict(self.metadata),
        }

        if include_interactions:
            result["interactions"] = [
                interaction.to_dict(
                    include_atoms=False,
                    include_residue=False,
                    include_descriptors=True,
                )
                for interaction in self.interactions
            ]

        return result


# -----------------------------------------------------------------------------
# Complete grouping workflow
# -----------------------------------------------------------------------------

def group_hydrophobic_interactions(
    interactions: HydrophobicInteractionCollection,
    *,
    grouping_distance: Optional[Number] = None,
    identify_hotspots: bool = True,
    hotspot_minimum_contact_count: Optional[int] = None,
    hotspot_minimum_group_score: Optional[Number] = None,
    hotspot_minimum_contact_area: Optional[Number] = None,
    hotspot_minimum_ligand_atom_count: Optional[int] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> HydrophobicGroupingResult:
    """
    Perform all hydrophobic grouping operations.

    The workflow produces:

    - receptor-residue groups;
    - receptor-chain groups;
    - ligand-pose groups;
    - spatial local regions;
    - hotspot residue groups.
    """

    interaction_tuple = (
        deduplicate_hydrophobic_interactions(
            interactions
        )
    )

    residue_groups = (
        group_hydrophobic_interactions_by_residue(
            interaction_tuple,
            identify_hotspots=identify_hotspots,
            hotspot_minimum_contact_count=(
                hotspot_minimum_contact_count
            ),
            hotspot_minimum_group_score=(
                hotspot_minimum_group_score
            ),
            hotspot_minimum_contact_area=(
                hotspot_minimum_contact_area
            ),
            hotspot_minimum_ligand_atom_count=(
                hotspot_minimum_ligand_atom_count
            ),
        )
    )

    local_regions = (
        cluster_hydrophobic_local_regions(
            interaction_tuple,
            grouping_distance=grouping_distance,
            require_same_pose=True,
            identify_hotspots=identify_hotspots,
        )
    )

    chain_groups = (
        group_hydrophobic_interactions_by_chain(
            interaction_tuple,
            residue_groups=residue_groups,
            local_regions=local_regions,
        )
    )

    pose_groups = (
        group_hydrophobic_interactions_by_pose(
            interaction_tuple
        )
    )

    hotspot_groups = tuple(
        group
        for group in residue_groups
        if bool(
            group.metadata.get(
                "is_hotspot",
                False,
            )
        )
    )

    grouping_metadata: Dict[str, Any] = (
        {} if metadata is None else dict(metadata)
    )

    grouping_metadata.update(
        {
            "grouping_completed": True,
            "grouping_distance": float(
                DEFAULT_LOCAL_INTERACTION_CLUSTER_DISTANCE
                if grouping_distance is None
                else _positive_float(
                    grouping_distance,
                    name="grouping distance",
                )
            ),
            "interaction_count": len(
                interaction_tuple
            ),
            "residue_group_count": len(
                residue_groups
            ),
            "chain_group_count": len(
                chain_groups
            ),
            "pose_group_count": len(
                pose_groups
            ),
            "local_region_count": len(
                local_regions
            ),
            "hotspot_count": len(
                hotspot_groups
            ),
            "surface_is_approximate": True,
        }
    )

    return HydrophobicGroupingResult(
        interactions=interaction_tuple,
        residue_groups=residue_groups,
        chain_groups=chain_groups,
        pose_groups=pose_groups,
        local_regions=local_regions,
        hotspot_groups=hotspot_groups,
        metadata=grouping_metadata,
    )


def add_hydrophobic_grouping_to_result(
    result: HydrophobicAnalysisResult,
    *,
    grouping_distance: Optional[Number] = None,
    identify_hotspots: bool = True,
    hotspot_minimum_contact_count: Optional[int] = None,
    hotspot_minimum_group_score: Optional[Number] = None,
    hotspot_minimum_contact_area: Optional[Number] = None,
    hotspot_minimum_ligand_atom_count: Optional[int] = None,
) -> HydrophobicAnalysisResult:
    """
    Return a new analysis result containing residue grouping.

    Chain, pose and local-region summaries are stored in result metadata.
    The strongly typed ``residue_groups`` field is populated directly.
    """

    if not isinstance(
        result,
        HydrophobicAnalysisResult,
    ):
        raise TypeError(
            "result must be a HydrophobicAnalysisResult."
        )

    grouping = group_hydrophobic_interactions(
        result.interactions,
        grouping_distance=(
            grouping_distance
            if grouping_distance is not None
            else result.grouping_distance
        ),
        identify_hotspots=identify_hotspots,
        hotspot_minimum_contact_count=(
            hotspot_minimum_contact_count
        ),
        hotspot_minimum_group_score=(
            hotspot_minimum_group_score
        ),
        hotspot_minimum_contact_area=(
            hotspot_minimum_contact_area
        ),
        hotspot_minimum_ligand_atom_count=(
            hotspot_minimum_ligand_atom_count
        ),
    )

    updated_metadata = dict(
        result.metadata
    )

    updated_metadata.update(
        {
            "analysis_stage": (
                "residue_and_region_grouping"
            ),
            "residue_grouping_completed": True,
            "chain_group_count": (
                grouping.chain_count
            ),
            "pose_group_count": (
                grouping.pose_count
            ),
            "local_region_count": (
                grouping.local_region_count
            ),
            "hotspot_count": (
                grouping.hotspot_count
            ),
            "approximate_contact_area": float(
                grouping.approximate_contact_area
            ),
            "chain_groups": [
                chain_group.to_dict(
                    include_interactions=False
                )
                for chain_group
                in grouping.chain_groups
            ],
            "pose_groups": [
                pose_group.to_dict(
                    include_interactions=False
                )
                for pose_group
                in grouping.pose_groups
            ],
            "local_regions": [
                region.to_dict(
                    include_interactions=False,
                    include_atoms=False,
                )
                for region
                in grouping.local_regions
            ],
        }
    )

    preliminary_statistics = result.statistics

    updated_statistics = HydrophobicStatistics(
        interaction_count=(
            preliminary_statistics.interaction_count
        ),
        residue_count=(
            grouping.residue_count
        ),
        receptor_atom_count=(
            preliminary_statistics.receptor_atom_count
        ),
        ligand_atom_count=(
            preliminary_statistics.ligand_atom_count
        ),
        very_strong_count=(
            preliminary_statistics.very_strong_count
        ),
        strong_count=(
            preliminary_statistics.strong_count
        ),
        moderate_count=(
            preliminary_statistics.moderate_count
        ),
        weak_count=(
            preliminary_statistics.weak_count
        ),
        marginal_count=(
            preliminary_statistics.marginal_count
        ),
        unknown_count=(
            preliminary_statistics.unknown_count
        ),
        aliphatic_aliphatic_count=(
            preliminary_statistics.aliphatic_aliphatic_count
        ),
        aliphatic_aromatic_count=(
            preliminary_statistics.aliphatic_aromatic_count
        ),
        aromatic_aliphatic_count=(
            preliminary_statistics.aromatic_aliphatic_count
        ),
        aromatic_aromatic_count=(
            preliminary_statistics.aromatic_aromatic_count
        ),
        mixed_count=(
            preliminary_statistics.mixed_count
        ),
        hotspot_count=(
            grouping.hotspot_count
        ),
        minimum_distance=(
            preliminary_statistics.minimum_distance
        ),
        mean_distance=(
            preliminary_statistics.mean_distance
        ),
        median_distance=(
            preliminary_statistics.median_distance
        ),
        maximum_distance=(
            preliminary_statistics.maximum_distance
        ),
        distance_standard_deviation=(
            preliminary_statistics.distance_standard_deviation
        ),
        minimum_score=(
            preliminary_statistics.minimum_score
        ),
        mean_score=(
            preliminary_statistics.mean_score
        ),
        median_score=(
            preliminary_statistics.median_score
        ),
        maximum_score=(
            preliminary_statistics.maximum_score
        ),
        total_score=(
            preliminary_statistics.total_score
        ),
        minimum_strength=(
            preliminary_statistics.minimum_strength
        ),
        mean_strength=(
            preliminary_statistics.mean_strength
        ),
        maximum_strength=(
            preliminary_statistics.maximum_strength
        ),
        classification_counts=(
            preliminary_statistics.classification_counts
        ),
        interaction_type_counts=(
            preliminary_statistics.interaction_type_counts
        ),
        residue_interaction_counts={
            (
                group.residue_identifier
                or "residue-unknown"
            ): group.interaction_count
            for group in grouping.residue_groups
        },
        residue_scores={
            (
                group.residue_identifier
                or "residue-unknown"
            ): float(
                group.group_score or 0.0
            )
            for group in grouping.residue_groups
        },
        metadata={
            **dict(
                preliminary_statistics.metadata
            ),
            "statistics_stage": (
                "residue_and_region_grouping"
            ),
            "residue_grouping_completed": True,
            "approximate_contact_area": float(
                grouping.approximate_contact_area
            ),
        },
    )

    return HydrophobicAnalysisResult(
        interactions=result.interactions,
        residue_groups=grouping.residue_groups,
        receptor_hydrophobic_atoms=(
            result.receptor_hydrophobic_atoms
        ),
        ligand_hydrophobic_atoms=(
            result.ligand_hydrophobic_atoms
        ),
        receptor_atoms=result.receptor_atoms,
        ligand_atoms=result.ligand_atoms,
        minimum_distance=result.minimum_distance,
        maximum_distance=result.maximum_distance,
        grouping_distance=(
            result.grouping_distance
            if grouping_distance is None
            else _positive_float(
                grouping_distance,
                name="grouping distance",
            )
        ),
        statistics=updated_statistics,
        analysis_identifier=(
            result.analysis_identifier
        ),
        receptor_identifier=(
            result.receptor_identifier
        ),
        ligand_identifier=(
            result.ligand_identifier
        ),
        metadata=updated_metadata,
    )


# -----------------------------------------------------------------------------
# Empty grouping collections
# -----------------------------------------------------------------------------

_EMPTY_HYDROPHOBIC_LOCAL_REGIONS: Final[
    Tuple[HydrophobicLocalRegion, ...]
] = ()

_EMPTY_HYDROPHOBIC_CHAIN_GROUPS: Final[
    Tuple[HydrophobicChainGroup, ...]
] = ()

_EMPTY_HYDROPHOBIC_POSE_GROUPS: Final[
    Tuple[HydrophobicPoseGroup, ...]
] = ()

_EMPTY_HYDROPHOBIC_GROUPING_RESULT: Final[
    HydrophobicGroupingResult
] = HydrophobicGroupingResult()


# -----------------------------------------------------------------------------
# Section 8 public names
# -----------------------------------------------------------------------------

_SECTION_8_PUBLIC_NAMES: Final[Tuple[str, ...]] = (
    # Aliases
    "HydrophobicChainKey",
    "HydrophobicPoseKey",
    "HydrophobicLocalRegionKey",
    "HydrophobicInteractionCollection",

    # Identifier helpers
    "get_residue_chain_identifier",
    "get_interaction_chain_identifier",
    "get_interaction_pose_identifier",
    "hydrophobic_residue_group_key",
    "hydrophobic_local_region_key",

    # Local geometry
    "hydrophobic_interaction_midpoint",
    "hydrophobic_interaction_midpoint_distance",
    "hydrophobic_interactions_share_atom",
    "are_hydrophobic_interactions_locally_connected",

    # Group metrics
    "select_closest_hydrophobic_interaction",
    "select_highest_scoring_hydrophobic_interaction",
    "approximate_interaction_contact_area",
    "approximate_hydrophobic_surface_area",
    "calculate_residue_group_score",
    "is_hydrophobic_hotspot",

    # Dataclasses
    "HydrophobicLocalRegion",
    "HydrophobicChainGroup",
    "HydrophobicPoseGroup",
    "HydrophobicGroupingResult",

    # Grouping functions
    "group_hydrophobic_interactions_by_residue",
    "cluster_hydrophobic_local_regions",
    "group_hydrophobic_interactions_by_chain",
    "group_hydrophobic_interactions_by_pose",

    # Summaries
    "count_hydrophobic_contacts_by_residue",
    "hydrophobic_surface_by_residue",
    "find_hydrophobic_hotspots",

    # Complete workflows
    "group_hydrophobic_interactions",
    "add_hydrophobic_grouping_to_result",
)

for public_name in _SECTION_8_PUBLIC_NAMES:
    if public_name not in __all__:
        __all__.append(public_name)


# =============================================================================
# End of Section 8
# =============================================================================


# =============================================================================
# Section 9 — Geometric classification and interaction strength
# =============================================================================


# -----------------------------------------------------------------------------
# Classification and scoring aliases
# -----------------------------------------------------------------------------

HydrophobicScoreComponentName: TypeAlias = Literal[
    "distance",
    "minimum_distance",
    "compaction",
    "density",
    "contact_count",
    "atom_diversity",
    "chemical_type",
    "aromatic_character",
    "aliphatic_character",
    "group_size",
    "surface_area",
    "polar_penalty",
    "redundancy_penalty",
]

HydrophobicScoreComponentMap: TypeAlias = Mapping[
    str,
    np.float64,
]


# -----------------------------------------------------------------------------
# Final score weights
# -----------------------------------------------------------------------------

HYDROPHOBIC_FINAL_DISTANCE_WEIGHT: Final[np.float64] = np.float64(
    0.28
)

HYDROPHOBIC_FINAL_COMPACTION_WEIGHT: Final[np.float64] = np.float64(
    0.14
)

HYDROPHOBIC_FINAL_DENSITY_WEIGHT: Final[np.float64] = np.float64(
    0.14
)

HYDROPHOBIC_FINAL_CONTACT_COUNT_WEIGHT: Final[np.float64] = np.float64(
    0.10
)

HYDROPHOBIC_FINAL_ATOM_DIVERSITY_WEIGHT: Final[np.float64] = np.float64(
    0.09
)

HYDROPHOBIC_FINAL_CHEMICAL_TYPE_WEIGHT: Final[np.float64] = np.float64(
    0.08
)

HYDROPHOBIC_FINAL_GROUP_SIZE_WEIGHT: Final[np.float64] = np.float64(
    0.07
)

HYDROPHOBIC_FINAL_SURFACE_AREA_WEIGHT: Final[np.float64] = np.float64(
    0.05
)

HYDROPHOBIC_FINAL_AROMATIC_ALIPHATIC_WEIGHT: Final[
    np.float64
] = np.float64(
    0.05
)

HYDROPHOBIC_FINAL_SCORE_WEIGHTS: Final[
    Mapping[str, np.float64]
] = MappingProxyType(
    {
        "distance": HYDROPHOBIC_FINAL_DISTANCE_WEIGHT,
        "compaction": HYDROPHOBIC_FINAL_COMPACTION_WEIGHT,
        "density": HYDROPHOBIC_FINAL_DENSITY_WEIGHT,
        "contact_count": HYDROPHOBIC_FINAL_CONTACT_COUNT_WEIGHT,
        "atom_diversity": HYDROPHOBIC_FINAL_ATOM_DIVERSITY_WEIGHT,
        "chemical_type": HYDROPHOBIC_FINAL_CHEMICAL_TYPE_WEIGHT,
        "group_size": HYDROPHOBIC_FINAL_GROUP_SIZE_WEIGHT,
        "surface_area": HYDROPHOBIC_FINAL_SURFACE_AREA_WEIGHT,
        "aromatic_aliphatic_character": (
            HYDROPHOBIC_FINAL_AROMATIC_ALIPHATIC_WEIGHT
        ),
    }
)


# -----------------------------------------------------------------------------
# Penalty weights
# -----------------------------------------------------------------------------

HYDROPHOBIC_POLARITY_PENALTY_WEIGHT: Final[np.float64] = np.float64(
    0.18
)

HYDROPHOBIC_REDUNDANCY_PENALTY_WEIGHT: Final[np.float64] = np.float64(
    0.08
)

HYDROPHOBIC_UNKNOWN_TYPE_PENALTY: Final[np.float64] = np.float64(
    0.10
)

HYDROPHOBIC_EXCESSIVE_GROUP_SIZE_PENALTY: Final[
    np.float64
] = np.float64(
    0.05
)


# -----------------------------------------------------------------------------
# Normalization references
# -----------------------------------------------------------------------------

HYDROPHOBIC_CONTACT_COUNT_SATURATION: Final[int] = 6
HYDROPHOBIC_ATOM_DIVERSITY_SATURATION: Final[int] = 6
HYDROPHOBIC_GROUP_SIZE_SATURATION: Final[int] = 10

HYDROPHOBIC_CONTACT_AREA_SATURATION: Final[np.float64] = np.float64(
    30.0
)

HYDROPHOBIC_MAXIMUM_RECOMMENDED_GROUP_SIZE: Final[int] = 20

HYDROPHOBIC_MAXIMUM_POLAR_NEIGHBOR_REFERENCE: Final[int] = 4


# -----------------------------------------------------------------------------
# Final classification thresholds
# -----------------------------------------------------------------------------

HYDROPHOBIC_VERY_STRONG_MINIMUM_SCORE: Final[np.float64] = np.float64(
    0.82
)

HYDROPHOBIC_STRONG_MINIMUM_SCORE: Final[np.float64] = np.float64(
    0.68
)

HYDROPHOBIC_MODERATE_MINIMUM_SCORE: Final[np.float64] = np.float64(
    0.50
)

HYDROPHOBIC_WEAK_MINIMUM_SCORE: Final[np.float64] = np.float64(
    0.30
)

HYDROPHOBIC_MARGINAL_MINIMUM_SCORE: Final[np.float64] = np.float64(
    0.10
)


# -----------------------------------------------------------------------------
# Chemical-type contributions
# -----------------------------------------------------------------------------

HYDROPHOBIC_FINAL_TYPE_SCORES: Final[
    Mapping[HydrophobicInteractionType, np.float64]
] = MappingProxyType(
    {
        HYDROPHOBIC_TYPE_ALIPHATIC_ALIPHATIC: np.float64(
            0.90
        ),
        HYDROPHOBIC_TYPE_ALIPHATIC_AROMATIC: np.float64(
            0.95
        ),
        HYDROPHOBIC_TYPE_AROMATIC_ALIPHATIC: np.float64(
            0.95
        ),
        HYDROPHOBIC_TYPE_AROMATIC_AROMATIC: np.float64(
            0.92
        ),
        HYDROPHOBIC_TYPE_MIXED: np.float64(
            0.85
        ),
        HYDROPHOBIC_TYPE_UNKNOWN: np.float64(
            0.40
        ),
    }
)


# -----------------------------------------------------------------------------
# Strength-analysis dataclass
# -----------------------------------------------------------------------------

@dataclass(
    frozen=True,
    slots=True,
)
class HydrophobicStrengthAssessment:
    """
    Multifactor assessment of one hydrophobic interaction or local group.

    The final score is normalized to the interval ``[0, 1]``. Individual
    positive components and penalties are retained for interpretation,
    debugging and serialization.
    """

    classification: HydrophobicClassification

    strength: np.float64
    score: np.float64

    distance_component: np.float64
    minimum_distance_component: np.float64

    compaction_component: np.float64
    density_component: np.float64

    contact_count_component: np.float64
    atom_diversity_component: np.float64

    chemical_type_component: np.float64
    aromatic_character_component: np.float64
    aliphatic_character_component: np.float64

    group_size_component: np.float64
    surface_area_component: np.float64

    polar_penalty: np.float64
    redundancy_penalty: np.float64
    additional_penalty: np.float64

    contact_count: int
    unique_receptor_atom_count: int
    unique_ligand_atom_count: int

    minimum_distance: np.float64
    mean_distance: np.float64
    maximum_distance: np.float64

    approximate_contact_area: np.float64

    interaction_type: HydrophobicInteractionType

    metadata: Mapping[str, Any] = field(
        default_factory=lambda: _EMPTY_METADATA,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        """Validate and normalize the strength assessment."""

        classification = (
            validate_hydrophobic_classification(
                self.classification
            )
        )

        interaction_type = (
            validate_hydrophobic_interaction_type(
                self.interaction_type
            )
        )

        normalized_score_fields = (
            "strength",
            "score",
            "distance_component",
            "minimum_distance_component",
            "compaction_component",
            "density_component",
            "contact_count_component",
            "atom_diversity_component",
            "chemical_type_component",
            "aromatic_character_component",
            "aliphatic_character_component",
            "group_size_component",
            "surface_area_component",
            "polar_penalty",
            "redundancy_penalty",
            "additional_penalty",
        )

        for field_name in normalized_score_fields:
            normalized_value = validate_hydrophobic_score(
                getattr(
                    self,
                    field_name,
                )
            )

            object.__setattr__(
                self,
                field_name,
                normalized_value,
            )

        object.__setattr__(
            self,
            "contact_count",
            _nonnegative_integer(
                self.contact_count,
                name="contact count",
            ),
        )

        object.__setattr__(
            self,
            "unique_receptor_atom_count",
            _nonnegative_integer(
                self.unique_receptor_atom_count,
                name="unique receptor atom count",
            ),
        )

        object.__setattr__(
            self,
            "unique_ligand_atom_count",
            _nonnegative_integer(
                self.unique_ligand_atom_count,
                name="unique ligand atom count",
            ),
        )

        object.__setattr__(
            self,
            "minimum_distance",
            _nonnegative_float(
                self.minimum_distance,
                name="minimum distance",
            ),
        )

        object.__setattr__(
            self,
            "mean_distance",
            _nonnegative_float(
                self.mean_distance,
                name="mean distance",
            ),
        )

        object.__setattr__(
            self,
            "maximum_distance",
            _nonnegative_float(
                self.maximum_distance,
                name="maximum distance",
            ),
        )

        object.__setattr__(
            self,
            "approximate_contact_area",
            _nonnegative_float(
                self.approximate_contact_area,
                name="approximate contact area",
            ),
        )

        object.__setattr__(
            self,
            "classification",
            classification,
        )

        object.__setattr__(
            self,
            "interaction_type",
            interaction_type,
        )

        object.__setattr__(
            self,
            "metadata",
            _freeze_metadata(self.metadata),
        )

    @property
    def total_positive_contribution(self) -> np.float64:
        """Return the sum of unweighted positive components."""

        return np.float64(
            self.distance_component
            + self.compaction_component
            + self.density_component
            + self.contact_count_component
            + self.atom_diversity_component
            + self.chemical_type_component
            + self.group_size_component
            + self.surface_area_component
            + (
                self.aromatic_character_component
                + self.aliphatic_character_component
            ) / 2.0
        )

    @property
    def total_penalty(self) -> np.float64:
        """Return the normalized total penalty."""

        return validate_hydrophobic_score(
            self.polar_penalty
            + self.redundancy_penalty
            + self.additional_penalty
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the strength assessment."""

        return {
            "classification": self.classification,
            "strength": float(self.strength),
            "score": float(self.score),
            "components": {
                "distance": float(
                    self.distance_component
                ),
                "minimum_distance": float(
                    self.minimum_distance_component
                ),
                "compaction": float(
                    self.compaction_component
                ),
                "density": float(
                    self.density_component
                ),
                "contact_count": float(
                    self.contact_count_component
                ),
                "atom_diversity": float(
                    self.atom_diversity_component
                ),
                "chemical_type": float(
                    self.chemical_type_component
                ),
                "aromatic_character": float(
                    self.aromatic_character_component
                ),
                "aliphatic_character": float(
                    self.aliphatic_character_component
                ),
                "group_size": float(
                    self.group_size_component
                ),
                "surface_area": float(
                    self.surface_area_component
                ),
            },
            "penalties": {
                "polar": float(
                    self.polar_penalty
                ),
                "redundancy": float(
                    self.redundancy_penalty
                ),
                "additional": float(
                    self.additional_penalty
                ),
                "total": float(
                    self.total_penalty
                ),
            },
            "contact_count": self.contact_count,
            "unique_receptor_atom_count": (
                self.unique_receptor_atom_count
            ),
            "unique_ligand_atom_count": (
                self.unique_ligand_atom_count
            ),
            "minimum_distance": float(
                self.minimum_distance
            ),
            "mean_distance": float(
                self.mean_distance
            ),
            "maximum_distance": float(
                self.maximum_distance
            ),
            "approximate_contact_area": float(
                self.approximate_contact_area
            ),
            "interaction_type": self.interaction_type,
            "total_positive_contribution": float(
                self.total_positive_contribution
            ),
            "metadata": dict(self.metadata),
        }


# -----------------------------------------------------------------------------
# Generic score normalization
# -----------------------------------------------------------------------------

def saturating_hydrophobic_score(
    value: Number,
    saturation_value: Number,
) -> np.float64:
    """
    Normalize a nonnegative quantity using a saturating curve.

    The function rises rapidly at low values and approaches one as the
    quantity reaches or exceeds the selected reference.
    """

    normalized_value = _nonnegative_float(
        value,
        name="score value",
    )

    saturation = _positive_float(
        saturation_value,
        name="saturation value",
    )

    if normalized_value == 0.0:
        return np.float64(0.0)

    score = normalized_value / (
        normalized_value
        + 0.5 * saturation
    )

    return validate_hydrophobic_score(
        score
    )


def linear_saturating_hydrophobic_score(
    value: Number,
    saturation_value: Number,
) -> np.float64:
    """
    Normalize a quantity linearly up to a saturation reference.
    """

    normalized_value = _nonnegative_float(
        value,
        name="score value",
    )

    saturation = _positive_float(
        saturation_value,
        name="saturation value",
    )

    return validate_hydrophobic_score(
        min(
            normalized_value / saturation,
            1.0,
        )
    )


# -----------------------------------------------------------------------------
# Final classification
# -----------------------------------------------------------------------------

def classify_hydrophobic_strength_score(
    score: Number,
) -> HydrophobicClassification:
    """
    Convert a multifactor score into the final classification.
    """

    normalized_score = validate_hydrophobic_score(
        score
    )

    if (
        normalized_score
        >= HYDROPHOBIC_VERY_STRONG_MINIMUM_SCORE
    ):
        return HYDROPHOBIC_CLASS_VERY_STRONG

    if (
        normalized_score
        >= HYDROPHOBIC_STRONG_MINIMUM_SCORE
    ):
        return HYDROPHOBIC_CLASS_STRONG

    if (
        normalized_score
        >= HYDROPHOBIC_MODERATE_MINIMUM_SCORE
    ):
        return HYDROPHOBIC_CLASS_MODERATE

    if (
        normalized_score
        >= HYDROPHOBIC_WEAK_MINIMUM_SCORE
    ):
        return HYDROPHOBIC_CLASS_WEAK

    if (
        normalized_score
        >= HYDROPHOBIC_MARGINAL_MINIMUM_SCORE
    ):
        return HYDROPHOBIC_CLASS_MARGINAL

    return HYDROPHOBIC_CLASS_UNKNOWN


# -----------------------------------------------------------------------------
# Interaction collection helpers
# -----------------------------------------------------------------------------

def _normalize_strength_interactions(
    interactions: HydrophobicInteractionCollection,
) -> Tuple[HydrophobicInteraction, ...]:
    """Validate and deduplicate interactions used for scoring."""

    normalized = (
        deduplicate_hydrophobic_interactions(
            interactions,
            prefer_highest_score=True,
        )
    )

    for interaction in normalized:
        if not isinstance(
            interaction,
            HydrophobicInteraction,
        ):
            raise TypeError(
                "Strength calculations require "
                "HydrophobicInteraction instances."
            )

    return normalized


def _interaction_distance_array(
    interactions: HydrophobicInteractionCollection,
) -> NDArray[np.float64]:
    """Return interaction distances as a float64 array."""

    interaction_tuple = tuple(interactions)

    if not interaction_tuple:
        return np.empty(
            0,
            dtype=np.float64,
        )

    return np.asarray(
        [
            interaction.distance
            for interaction in interaction_tuple
        ],
        dtype=np.float64,
    )


def _unique_interaction_atoms(
    interactions: HydrophobicInteractionCollection,
) -> Tuple[
    Tuple[AtomLike, ...],
    Tuple[AtomLike, ...],
]:
    """Return unique receptor and ligand atoms."""

    interaction_tuple = tuple(interactions)

    receptor_atoms = deduplicate_atoms(
        (
            interaction.receptor_atom
            for interaction in interaction_tuple
        ),
        strategy="auto",
    )

    ligand_atoms = deduplicate_atoms(
        (
            interaction.ligand_atom
            for interaction in interaction_tuple
        ),
        strategy="auto",
    )

    return receptor_atoms, ligand_atoms


def find_related_hydrophobic_interactions(
    interaction: HydrophobicInteraction,
    interactions: HydrophobicInteractionCollection,
    *,
    grouping_distance: Optional[Number] = None,
    require_same_pose: bool = True,
) -> Tuple[HydrophobicInteraction, ...]:
    """
    Return interactions belonging to the same local contact environment.

    The selected interaction is always included in the result.
    """

    if not isinstance(
        interaction,
        HydrophobicInteraction,
    ):
        raise TypeError(
            "interaction must be a HydrophobicInteraction."
        )

    candidates = (
        _normalize_strength_interactions(
            interactions
        )
    )

    if interaction not in candidates:
        candidates = (
            interaction,
            *candidates,
        )

    connected: List[
        HydrophobicInteraction
    ] = []

    for candidate in candidates:
        if candidate is interaction:
            connected.append(candidate)
            continue

        if are_hydrophobic_interactions_locally_connected(
            interaction,
            candidate,
            grouping_distance=grouping_distance,
            require_same_pose=require_same_pose,
        ):
            connected.append(candidate)

    return tuple(connected)


# -----------------------------------------------------------------------------
# Distance contribution
# -----------------------------------------------------------------------------

def calculate_hydrophobic_distance_component(
    interactions: HydrophobicInteractionCollection,
    *,
    minimum_cutoff: Optional[Number] = None,
    maximum_cutoff: Optional[Number] = None,
) -> Tuple[np.float64, np.float64]:
    """
    Calculate mean-distance and minimum-distance components.

    The closest contact receives greater importance, but the mean distance
    prevents a large group of marginal pairs from appearing artificially
    strong because of one short pair.
    """

    interaction_tuple = tuple(interactions)

    if not interaction_tuple:
        return (
            np.float64(0.0),
            np.float64(0.0),
        )

    distances = _interaction_distance_array(
        interaction_tuple
    )

    minimum_distance = np.min(distances)
    mean_distance = np.mean(distances)

    minimum_limit = (
        get_default_minimum_hydrophobic_distance()
        if minimum_cutoff is None
        else _nonnegative_float(
            minimum_cutoff,
            name="minimum hydrophobic cutoff",
        )
    )

    maximum_limit = (
        get_default_maximum_hydrophobic_distance()
        if maximum_cutoff is None
        else _positive_float(
            maximum_cutoff,
            name="maximum hydrophobic cutoff",
        )
    )

    if minimum_limit >= maximum_limit:
        raise ValueError(
            "minimum cutoff must be smaller than maximum cutoff."
        )

    minimum_component = distance_compaction_score(
        minimum_distance,
        reference_distance=maximum_limit,
        minimum_distance=minimum_limit,
    )

    mean_component = distance_compaction_score(
        mean_distance,
        reference_distance=maximum_limit,
        minimum_distance=minimum_limit,
    )

    combined_distance_component = (
        0.65 * float(minimum_component)
        + 0.35 * float(mean_component)
    )

    return (
        validate_hydrophobic_score(
            combined_distance_component
        ),
        minimum_component,
    )


# -----------------------------------------------------------------------------
# Compaction and density contributions
# -----------------------------------------------------------------------------

def calculate_hydrophobic_compaction_component(
    interactions: HydrophobicInteractionCollection,
) -> np.float64:
    """
    Calculate group compactness from all unique receptor and ligand atoms.
    """

    receptor_atoms, ligand_atoms = (
        _unique_interaction_atoms(
            interactions
        )
    )

    if not receptor_atoms or not ligand_atoms:
        return np.float64(0.0)

    return group_compaction_score(
        receptor_atoms,
        ligand_atoms,
        minimum_distance=0.0,
        maximum_distance=(
            get_default_maximum_hydrophobic_distance()
        ),
    )


def calculate_hydrophobic_density_component(
    interactions: HydrophobicInteractionCollection,
) -> np.float64:
    """
    Calculate local receptor–ligand contact density.
    """

    receptor_atoms, ligand_atoms = (
        _unique_interaction_atoms(
            interactions
        )
    )

    if not receptor_atoms or not ligand_atoms:
        return np.float64(0.0)

    return approximate_contact_density(
        receptor_atoms,
        ligand_atoms,
        minimum_distance=0.0,
        maximum_distance=(
            get_default_maximum_hydrophobic_distance()
        ),
    )


# -----------------------------------------------------------------------------
# Contact count, diversity and group size
# -----------------------------------------------------------------------------

def calculate_hydrophobic_contact_count_component(
    interactions: HydrophobicInteractionCollection,
) -> np.float64:
    """
    Score the number of nonduplicated atom-pair contacts.
    """

    interaction_count = len(
        _normalize_strength_interactions(
            interactions
        )
    )

    return saturating_hydrophobic_score(
        interaction_count,
        HYDROPHOBIC_CONTACT_COUNT_SATURATION,
    )


def calculate_hydrophobic_atom_diversity_component(
    interactions: HydrophobicInteractionCollection,
) -> np.float64:
    """
    Score the diversity of receptor and ligand atoms participating.

    Repetition of the same atom in many pairs therefore contributes less
    than contacts distributed across a broader local surface.
    """

    receptor_atoms, ligand_atoms = (
        _unique_interaction_atoms(
            interactions
        )
    )

    receptor_diversity = saturating_hydrophobic_score(
        len(receptor_atoms),
        HYDROPHOBIC_ATOM_DIVERSITY_SATURATION,
    )

    ligand_diversity = saturating_hydrophobic_score(
        len(ligand_atoms),
        HYDROPHOBIC_ATOM_DIVERSITY_SATURATION,
    )

    return validate_hydrophobic_score(
        (
            float(receptor_diversity)
            + float(ligand_diversity)
        ) / 2.0
    )


def calculate_hydrophobic_group_size_component(
    interactions: HydrophobicInteractionCollection,
) -> np.float64:
    """
    Score the total size of the local interacting atom group.
    """

    receptor_atoms, ligand_atoms = (
        _unique_interaction_atoms(
            interactions
        )
    )

    total_atom_count = (
        len(receptor_atoms)
        + len(ligand_atoms)
    )

    return saturating_hydrophobic_score(
        total_atom_count,
        HYDROPHOBIC_GROUP_SIZE_SATURATION,
    )


# -----------------------------------------------------------------------------
# Contact type and atom-character contributions
# -----------------------------------------------------------------------------

def predominant_hydrophobic_interaction_type(
    interactions: HydrophobicInteractionCollection,
) -> HydrophobicInteractionType:
    """
    Return the most frequent interaction type.

    The shortest mean distance is used to resolve count ties.
    """

    interaction_tuple = tuple(interactions)

    if not interaction_tuple:
        return HYDROPHOBIC_TYPE_UNKNOWN

    grouped: Dict[
        HydrophobicInteractionType,
        List[HydrophobicInteraction],
    ] = {}

    for interaction in interaction_tuple:
        grouped.setdefault(
            interaction.interaction_type,
            [],
        ).append(interaction)

    ordered_types = sorted(
        grouped,
        key=lambda interaction_type: (
            -len(grouped[interaction_type]),
            float(
                np.mean(
                    [
                        interaction.distance
                        for interaction
                        in grouped[interaction_type]
                    ]
                )
            ),
            interaction_type,
        ),
    )

    return ordered_types[0]


def calculate_hydrophobic_chemical_type_component(
    interactions: HydrophobicInteractionCollection,
) -> Tuple[
    np.float64,
    HydrophobicInteractionType,
]:
    """
    Score the distribution of hydrophobic contact types.
    """

    interaction_tuple = tuple(interactions)

    if not interaction_tuple:
        return (
            np.float64(0.0),
            HYDROPHOBIC_TYPE_UNKNOWN,
        )

    type_scores = [
        HYDROPHOBIC_FINAL_TYPE_SCORES.get(
            interaction.interaction_type,
            np.float64(0.40),
        )
        for interaction in interaction_tuple
    ]

    predominant_type = (
        predominant_hydrophobic_interaction_type(
            interaction_tuple
        )
    )

    return (
        validate_hydrophobic_score(
            np.mean(type_scores)
        ),
        predominant_type,
    )


def calculate_hydrophobic_atom_character_components(
    interactions: HydrophobicInteractionCollection,
) -> Tuple[np.float64, np.float64]:
    """
    Calculate aromatic and aliphatic participation components.

    Aromatic–aromatic contacts remain classified only as hydrophobic
    contacts. No π-stacking assignment is made.
    """

    interaction_tuple = tuple(interactions)

    if not interaction_tuple:
        return (
            np.float64(0.0),
            np.float64(0.0),
        )

    receptor_atoms, ligand_atoms = (
        _unique_interaction_atoms(
            interaction_tuple
        )
    )

    all_atoms = (
        *receptor_atoms,
        *ligand_atoms,
    )

    if not all_atoms:
        return (
            np.float64(0.0),
            np.float64(0.0),
        )

    aromatic_count = sum(
        is_aromatic_atom(atom)
        for atom in all_atoms
    )

    aliphatic_count = sum(
        is_aliphatic_atom(atom)
        for atom in all_atoms
    )

    atom_count = len(all_atoms)

    aromatic_component = (
        aromatic_count
        / atom_count
    )

    aliphatic_component = (
        aliphatic_count
        / atom_count
    )

    return (
        validate_hydrophobic_score(
            aromatic_component
        ),
        validate_hydrophobic_score(
            aliphatic_component
        ),
    )


def combine_hydrophobic_atom_character(
    aromatic_component: Number,
    aliphatic_component: Number,
) -> np.float64:
    """
    Combine aromatic and aliphatic participation.

    A group is not penalized merely for being predominantly aromatic or
    predominantly aliphatic. Mixed participation receives a small
    diversity benefit.
    """

    aromatic = validate_hydrophobic_score(
        aromatic_component
    )

    aliphatic = validate_hydrophobic_score(
        aliphatic_component
    )

    dominant_character = max(
        aromatic,
        aliphatic,
    )

    mixed_character_bonus = min(
        aromatic,
        aliphatic,
    ) * 0.20

    return validate_hydrophobic_score(
        dominant_character
        + mixed_character_bonus
    )


# -----------------------------------------------------------------------------
# Surface-area contribution
# -----------------------------------------------------------------------------

def calculate_hydrophobic_surface_component(
    interactions: HydrophobicInteractionCollection,
) -> Tuple[np.float64, np.float64]:
    """
    Calculate normalized and absolute approximate contact surface.
    """

    surface_area = (
        approximate_hydrophobic_surface_area(
            interactions,
            reduce_shared_atoms=True,
        )
    )

    component = saturating_hydrophobic_score(
        surface_area,
        HYDROPHOBIC_CONTACT_AREA_SATURATION,
    )

    return (
        component,
        surface_area,
    )


# -----------------------------------------------------------------------------
# Polarity penalty
# -----------------------------------------------------------------------------

def _descriptor_polar_neighbor_count(
    descriptor: Optional[HydrophobicAtom],
) -> int:
    """Return a safe descriptor polar-neighbor count."""

    if descriptor is None:
        return 0

    try:
        return _nonnegative_integer(
            descriptor.polar_neighbor_count,
            name="polar neighbor count",
        )
    except (
        TypeError,
        ValueError,
    ):
        return 0


def calculate_hydrophobic_polar_penalty(
    interactions: HydrophobicInteractionCollection,
) -> np.float64:
    """
    Calculate the polarity penalty for an interaction group.

    The penalty considers:

    - polar neighbors of participating atoms;
    - atom-level partial charges;
    - pair-level polar penalties from detection.
    """

    interaction_tuple = tuple(interactions)

    if not interaction_tuple:
        return np.float64(0.0)

    receptor_atoms, ligand_atoms = (
        _unique_interaction_atoms(
            interaction_tuple
        )
    )

    all_atoms = (
        *receptor_atoms,
        *ligand_atoms,
    )

    polar_neighbor_total = 0

    for atom in all_atoms:
        polar_neighbor_total += count_polar_neighbors(
            atom
        )

    polar_neighbor_reference = max(
        len(all_atoms)
        * HYDROPHOBIC_MAXIMUM_POLAR_NEIGHBOR_REFERENCE,
        1,
    )

    neighbor_penalty = min(
        polar_neighbor_total
        / polar_neighbor_reference,
        1.0,
    )

    absolute_partial_charges: List[float] = []

    for atom in all_atoms:
        partial_charge = get_atom_partial_charge(
            atom
        )

        if partial_charge is not None:
            absolute_partial_charges.append(
                abs(
                    float(partial_charge)
                )
            )

    if absolute_partial_charges:
        partial_charge_limit = max(
            float(
                get_default_maximum_absolute_partial_charge()
            ),
            1.0e-6,
        )

        charge_penalty = min(
            float(
                np.mean(
                    absolute_partial_charges
                )
            )
            / partial_charge_limit,
            1.0,
        )

    else:
        charge_penalty = 0.0

    pair_penalty = float(
        np.mean(
            [
                interaction.polar_penalty
                for interaction
                in interaction_tuple
            ]
        )
    )

    combined_penalty = (
        0.45 * neighbor_penalty
        + 0.35 * charge_penalty
        + 0.20 * pair_penalty
    )

    return validate_hydrophobic_score(
        combined_penalty
    )


# -----------------------------------------------------------------------------
# Redundancy penalty
# -----------------------------------------------------------------------------

def calculate_hydrophobic_redundancy_penalty(
    interactions: HydrophobicInteractionCollection,
) -> np.float64:
    """
    Penalize excessive reuse of the same atoms across many pairs.
    """

    interaction_tuple = (
        _normalize_strength_interactions(
            interactions
        )
    )

    if len(interaction_tuple) <= 1:
        return np.float64(0.0)

    receptor_atoms, ligand_atoms = (
        _unique_interaction_atoms(
            interaction_tuple
        )
    )

    unique_atom_total = (
        len(receptor_atoms)
        + len(ligand_atoms)
    )

    maximum_unique_atom_total = (
        2 * len(interaction_tuple)
    )

    if maximum_unique_atom_total == 0:
        return np.float64(0.0)

    diversity_fraction = (
        unique_atom_total
        / maximum_unique_atom_total
    )

    return validate_hydrophobic_score(
        1.0
        - diversity_fraction
    )


# -----------------------------------------------------------------------------
# Additional penalties
# -----------------------------------------------------------------------------

def calculate_additional_hydrophobic_penalty(
    interactions: HydrophobicInteractionCollection,
    interaction_type: HydrophobicInteractionType,
) -> np.float64:
    """
    Calculate penalties not represented by explicit polarity/redundancy.
    """

    interaction_tuple = tuple(interactions)

    additional_penalty = 0.0

    if interaction_type == HYDROPHOBIC_TYPE_UNKNOWN:
        additional_penalty += float(
            HYDROPHOBIC_UNKNOWN_TYPE_PENALTY
        )

    receptor_atoms, ligand_atoms = (
        _unique_interaction_atoms(
            interaction_tuple
        )
    )

    total_group_size = (
        len(receptor_atoms)
        + len(ligand_atoms)
    )

    if (
        total_group_size
        > HYDROPHOBIC_MAXIMUM_RECOMMENDED_GROUP_SIZE
    ):
        excess_fraction = min(
            (
                total_group_size
                - HYDROPHOBIC_MAXIMUM_RECOMMENDED_GROUP_SIZE
            )
            / HYDROPHOBIC_MAXIMUM_RECOMMENDED_GROUP_SIZE,
            1.0,
        )

        additional_penalty += (
            float(
                HYDROPHOBIC_EXCESSIVE_GROUP_SIZE_PENALTY
            )
            * excess_fraction
        )

    return validate_hydrophobic_score(
        additional_penalty
    )


# -----------------------------------------------------------------------------
# Combined geometric contribution
# -----------------------------------------------------------------------------

def calculate_combined_hydrophobic_contribution(
    *,
    distance_component: Number,
    compaction_component: Number,
    density_component: Number,
    contact_count_component: Number,
    atom_diversity_component: Number,
    chemical_type_component: Number,
    group_size_component: Number,
    surface_area_component: Number,
    aromatic_aliphatic_component: Number,
    weights: Optional[Mapping[str, Number]] = None,
) -> np.float64:
    """
    Combine normalized positive components into one geometric score.
    """

    resolved_weights: Dict[str, float] = {
        key: float(value)
        for key, value
        in HYDROPHOBIC_FINAL_SCORE_WEIGHTS.items()
    }

    if weights is not None:
        for key, value in weights.items():
            if key not in resolved_weights:
                raise ValueError(
                    f"Unknown hydrophobic score weight: {key!r}."
                )

            resolved_weights[key] = float(
                _nonnegative_float(
                    value,
                    name=f"{key} score weight",
                )
            )

    total_weight = sum(
        resolved_weights.values()
    )

    if total_weight <= 0.0:
        raise ValueError(
            "The total hydrophobic score weight must be positive."
        )

    components = {
        "distance": validate_hydrophobic_score(
            distance_component
        ),
        "compaction": validate_hydrophobic_score(
            compaction_component
        ),
        "density": validate_hydrophobic_score(
            density_component
        ),
        "contact_count": validate_hydrophobic_score(
            contact_count_component
        ),
        "atom_diversity": validate_hydrophobic_score(
            atom_diversity_component
        ),
        "chemical_type": validate_hydrophobic_score(
            chemical_type_component
        ),
        "group_size": validate_hydrophobic_score(
            group_size_component
        ),
        "surface_area": validate_hydrophobic_score(
            surface_area_component
        ),
        "aromatic_aliphatic_character": (
            validate_hydrophobic_score(
                aromatic_aliphatic_component
            )
        ),
    }

    weighted_sum = sum(
        float(components[key])
        * resolved_weights[key]
        for key in components
    )

    return validate_hydrophobic_score(
        weighted_sum / total_weight
    )


# -----------------------------------------------------------------------------
# Complete strength assessment
# -----------------------------------------------------------------------------

def assess_hydrophobic_interaction_strength(
    interactions: Union[
        HydrophobicInteraction,
        HydrophobicInteractionCollection,
        HydrophobicResidueGroup,
        HydrophobicLocalRegion,
    ],
    *,
    weights: Optional[Mapping[str, Number]] = None,
    minimum_distance: Optional[Number] = None,
    maximum_distance: Optional[Number] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> HydrophobicStrengthAssessment:
    """
    Perform complete multifactor hydrophobic-strength assessment.

    Accepted inputs include:

    - one atomic interaction;
    - a sequence of interactions;
    - one residue group;
    - one local interaction region.
    """

    if isinstance(
        interactions,
        HydrophobicInteraction,
    ):
        interaction_tuple = (
            interactions,
        )

    elif isinstance(
        interactions,
        HydrophobicResidueGroup,
    ):
        interaction_tuple = tuple(
            interactions.interactions
        )

    elif isinstance(
        interactions,
        HydrophobicLocalRegion,
    ):
        interaction_tuple = tuple(
            interactions.interactions
        )

    else:
        interaction_tuple = tuple(
            interactions
        )

    interaction_tuple = (
        _normalize_strength_interactions(
            interaction_tuple
        )
    )

    if not interaction_tuple:
        raise ValueError(
            "At least one hydrophobic interaction is required."
        )

    distances = _interaction_distance_array(
        interaction_tuple
    )

    minimum_observed_distance = np.float64(
        np.min(distances)
    )

    mean_observed_distance = np.float64(
        np.mean(distances)
    )

    maximum_observed_distance = np.float64(
        np.max(distances)
    )

    (
        distance_component,
        minimum_distance_component,
    ) = calculate_hydrophobic_distance_component(
        interaction_tuple,
        minimum_cutoff=minimum_distance,
        maximum_cutoff=maximum_distance,
    )

    compaction_component = (
        calculate_hydrophobic_compaction_component(
            interaction_tuple
        )
    )

    density_component = (
        calculate_hydrophobic_density_component(
            interaction_tuple
        )
    )

    contact_count_component = (
        calculate_hydrophobic_contact_count_component(
            interaction_tuple
        )
    )

    atom_diversity_component = (
        calculate_hydrophobic_atom_diversity_component(
            interaction_tuple
        )
    )

    group_size_component = (
        calculate_hydrophobic_group_size_component(
            interaction_tuple
        )
    )

    (
        chemical_type_component,
        interaction_type,
    ) = calculate_hydrophobic_chemical_type_component(
        interaction_tuple
    )

    (
        aromatic_character_component,
        aliphatic_character_component,
    ) = calculate_hydrophobic_atom_character_components(
        interaction_tuple
    )

    aromatic_aliphatic_component = (
        combine_hydrophobic_atom_character(
            aromatic_character_component,
            aliphatic_character_component,
        )
    )

    (
        surface_area_component,
        approximate_contact_area,
    ) = calculate_hydrophobic_surface_component(
        interaction_tuple
    )

    polar_penalty = (
        calculate_hydrophobic_polar_penalty(
            interaction_tuple
        )
    )

    redundancy_penalty = (
        calculate_hydrophobic_redundancy_penalty(
            interaction_tuple
        )
    )

    additional_penalty = (
        calculate_additional_hydrophobic_penalty(
            interaction_tuple,
            interaction_type,
        )
    )

    positive_score = (
        calculate_combined_hydrophobic_contribution(
            distance_component=distance_component,
            compaction_component=compaction_component,
            density_component=density_component,
            contact_count_component=(
                contact_count_component
            ),
            atom_diversity_component=(
                atom_diversity_component
            ),
            chemical_type_component=(
                chemical_type_component
            ),
            group_size_component=(
                group_size_component
            ),
            surface_area_component=(
                surface_area_component
            ),
            aromatic_aliphatic_component=(
                aromatic_aliphatic_component
            ),
            weights=weights,
        )
    )

    weighted_polar_penalty = (
        float(polar_penalty)
        * float(
            HYDROPHOBIC_POLARITY_PENALTY_WEIGHT
        )
    )

    weighted_redundancy_penalty = (
        float(redundancy_penalty)
        * float(
            HYDROPHOBIC_REDUNDANCY_PENALTY_WEIGHT
        )
    )

    final_score = np.clip(
        float(positive_score)
        - weighted_polar_penalty
        - weighted_redundancy_penalty
        - float(additional_penalty),
        0.0,
        1.0,
    )

    # Strength emphasizes geometric quality more strongly, while the
    # final score also includes chemical type and group-level context.
    strength = (
        0.45 * float(distance_component)
        + 0.25 * float(compaction_component)
        + 0.20 * float(density_component)
        + 0.10 * float(atom_diversity_component)
    )

    strength -= (
        0.50 * weighted_polar_penalty
        + 0.25 * weighted_redundancy_penalty
    )

    strength = np.clip(
        strength,
        0.0,
        1.0,
    )

    classification = (
        classify_hydrophobic_strength_score(
            final_score
        )
    )

    receptor_atoms, ligand_atoms = (
        _unique_interaction_atoms(
            interaction_tuple
        )
    )

    assessment_metadata: Dict[str, Any] = (
        {} if metadata is None else dict(metadata)
    )

    assessment_metadata.update(
        {
            "assessment_method": (
                "multifactor_geometric_and_chemical"
            ),
            "distance_only_classification": False,
            "positive_score_before_penalties": float(
                positive_score
            ),
            "weighted_polar_penalty": (
                weighted_polar_penalty
            ),
            "weighted_redundancy_penalty": (
                weighted_redundancy_penalty
            ),
            "pi_stacking_assigned": False,
            "aromatic_aromatic_is_hydrophobic_only": True,
            "weights": (
                dict(
                    HYDROPHOBIC_FINAL_SCORE_WEIGHTS
                )
                if weights is None
                else {
                    **dict(
                        HYDROPHOBIC_FINAL_SCORE_WEIGHTS
                    ),
                    **dict(weights),
                }
            ),
        }
    )

    return HydrophobicStrengthAssessment(
        classification=classification,
        strength=validate_hydrophobic_score(
            strength
        ),
        score=validate_hydrophobic_score(
            final_score
        ),
        distance_component=distance_component,
        minimum_distance_component=(
            minimum_distance_component
        ),
        compaction_component=compaction_component,
        density_component=density_component,
        contact_count_component=(
            contact_count_component
        ),
        atom_diversity_component=(
            atom_diversity_component
        ),
        chemical_type_component=(
            chemical_type_component
        ),
        aromatic_character_component=(
            aromatic_character_component
        ),
        aliphatic_character_component=(
            aliphatic_character_component
        ),
        group_size_component=(
            group_size_component
        ),
        surface_area_component=(
            surface_area_component
        ),
        polar_penalty=polar_penalty,
        redundancy_penalty=(
            redundancy_penalty
        ),
        additional_penalty=(
            additional_penalty
        ),
        contact_count=len(
            interaction_tuple
        ),
        unique_receptor_atom_count=len(
            receptor_atoms
        ),
        unique_ligand_atom_count=len(
            ligand_atoms
        ),
        minimum_distance=(
            minimum_observed_distance
        ),
        mean_distance=mean_observed_distance,
        maximum_distance=(
            maximum_observed_distance
        ),
        approximate_contact_area=(
            approximate_contact_area
        ),
        interaction_type=interaction_type,
        metadata=assessment_metadata,
    )


# -----------------------------------------------------------------------------
# Interaction reconstruction
# -----------------------------------------------------------------------------

def rebuild_hydrophobic_interaction(
    interaction: HydrophobicInteraction,
    *,
    classification: Optional[
        HydrophobicClassification
    ] = None,
    strength: Optional[Number] = None,
    score: Optional[Number] = None,
    interaction_type: Optional[
        HydrophobicInteractionType
    ] = None,
    local_contact_count: Optional[int] = None,
    polar_penalty: Optional[Number] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> HydrophobicInteraction:
    """
    Return a modified copy of an immutable interaction.

    The constructor is used directly instead of relying on
    ``dataclasses.replace``, keeping this section independent from the
    exact imports used in Section 1.
    """

    updated_metadata = dict(
        interaction.metadata
    )

    if metadata is not None:
        updated_metadata.update(
            dict(metadata)
        )

    return HydrophobicInteraction(
        receptor_atom=interaction.receptor_atom,
        ligand_atom=interaction.ligand_atom,
        receptor_descriptor=(
            interaction.receptor_descriptor
        ),
        ligand_descriptor=(
            interaction.ligand_descriptor
        ),
        receptor_residue=(
            interaction.receptor_residue
        ),
        receptor_residue_key=(
            interaction.receptor_residue_key
        ),
        distance=interaction.distance,
        interaction_type=(
            interaction.interaction_type
            if interaction_type is None
            else interaction_type
        ),
        classification=(
            interaction.classification
            if classification is None
            else classification
        ),
        strength=(
            interaction.strength
            if strength is None
            else strength
        ),
        score=(
            interaction.score
            if score is None
            else score
        ),
        detection_method=(
            interaction.detection_method
        ),
        direction=interaction.direction,
        local_contact_count=(
            interaction.local_contact_count
            if local_contact_count is None
            else local_contact_count
        ),
        polar_penalty=(
            interaction.polar_penalty
            if polar_penalty is None
            else polar_penalty
        ),
        receptor_atom_index=(
            interaction.receptor_atom_index
        ),
        ligand_atom_index=(
            interaction.ligand_atom_index
        ),
        receptor_atom_identifier=(
            interaction.receptor_atom_identifier
        ),
        ligand_atom_identifier=(
            interaction.ligand_atom_identifier
        ),
        interaction_identifier=(
            interaction.interaction_identifier
        ),
        metadata=updated_metadata,
    )


# -----------------------------------------------------------------------------
# Refinement of one atomic interaction
# -----------------------------------------------------------------------------

def classify_hydrophobic_interaction(
    interaction: HydrophobicInteraction,
    *,
    related_interactions: Optional[
        HydrophobicInteractionCollection
    ] = None,
    grouping_distance: Optional[Number] = None,
    weights: Optional[Mapping[str, Number]] = None,
) -> HydrophobicStrengthAssessment:
    """
    Classify one interaction using its local contact environment.
    """

    if related_interactions is None:
        local_interactions = (
            interaction,
        )

    else:
        local_interactions = (
            find_related_hydrophobic_interactions(
                interaction,
                related_interactions,
                grouping_distance=grouping_distance,
                require_same_pose=True,
            )
        )

    return assess_hydrophobic_interaction_strength(
        local_interactions,
        weights=weights,
    )


def refine_hydrophobic_interaction(
    interaction: HydrophobicInteraction,
    *,
    related_interactions: Optional[
        HydrophobicInteractionCollection
    ] = None,
    grouping_distance: Optional[Number] = None,
    weights: Optional[Mapping[str, Number]] = None,
) -> HydrophobicInteraction:
    """
    Return one interaction with final classification, strength and score.
    """

    assessment = classify_hydrophobic_interaction(
        interaction,
        related_interactions=related_interactions,
        grouping_distance=grouping_distance,
        weights=weights,
    )

    refined_metadata = {
        "preliminary_classification": False,
        "preliminary_score": False,
        "classification_stage": (
            "final_multifactor"
        ),
        "strength_assessment": (
            assessment.to_dict()
        ),
        "local_group_contact_count": (
            assessment.contact_count
        ),
        "local_group_minimum_distance": float(
            assessment.minimum_distance
        ),
        "local_group_contact_area": float(
            assessment.approximate_contact_area
        ),
    }

    return rebuild_hydrophobic_interaction(
        interaction,
        classification=(
            assessment.classification
        ),
        strength=assessment.strength,
        score=assessment.score,
        interaction_type=(
            assessment.interaction_type
        ),
        local_contact_count=max(
            assessment.contact_count,
            1,
        ),
        polar_penalty=(
            assessment.polar_penalty
        ),
        metadata=refined_metadata,
    )


# -----------------------------------------------------------------------------
# Refinement of interaction collections
# -----------------------------------------------------------------------------

def refine_hydrophobic_interactions(
    interactions: HydrophobicInteractionCollection,
    *,
    grouping_distance: Optional[Number] = None,
    weights: Optional[Mapping[str, Number]] = None,
    sort_interactions: bool = True,
) -> Tuple[HydrophobicInteraction, ...]:
    """
    Refine all interactions using local-region context.

    Interactions belonging to the same local region receive the same
    group-context assessment, but each atomic interaction remains present
    as an independent structured object.
    """

    interaction_tuple = (
        _normalize_strength_interactions(
            interactions
        )
    )

    if not interaction_tuple:
        return ()

    regions = cluster_hydrophobic_local_regions(
        interaction_tuple,
        grouping_distance=grouping_distance,
        require_same_pose=True,
        identify_hotspots=False,
    )

    region_by_interaction_key: Dict[
        HydrophobicPairKey,
        Tuple[HydrophobicInteraction, ...],
    ] = {}

    for region in regions:
        region_interactions = tuple(
            region.interactions
        )

        for interaction in region_interactions:
            region_by_interaction_key[
                hydrophobic_interaction_pair_key(
                    interaction
                )
            ] = region_interactions

    refined: List[
        HydrophobicInteraction
    ] = []

    for interaction in interaction_tuple:
        interaction_key = (
            hydrophobic_interaction_pair_key(
                interaction
            )
        )

        related = region_by_interaction_key.get(
            interaction_key,
            (
                interaction,
            ),
        )

        refined.append(
            refine_hydrophobic_interaction(
                interaction,
                related_interactions=related,
                grouping_distance=grouping_distance,
                weights=weights,
            )
        )

    refined_tuple = (
        deduplicate_hydrophobic_interactions(
            refined,
            prefer_highest_score=True,
        )
    )

    if sort_interactions:
        refined_tuple = tuple(
            sorted(
                refined_tuple,
                key=lambda item: (
                    -float(item.score),
                    -float(item.strength),
                    float(item.distance),
                    item.interaction_identifier or "",
                ),
            )
        )

    return refined_tuple


# -----------------------------------------------------------------------------
# Group-level strength assessment
# -----------------------------------------------------------------------------

def classify_hydrophobic_residue_group(
    group: HydrophobicResidueGroup,
    *,
    weights: Optional[Mapping[str, Number]] = None,
) -> HydrophobicStrengthAssessment:
    """
    Assess the complete hydrophobic contribution of one receptor residue.
    """

    if not isinstance(
        group,
        HydrophobicResidueGroup,
    ):
        raise TypeError(
            "group must be a HydrophobicResidueGroup."
        )

    return assess_hydrophobic_interaction_strength(
        group.interactions,
        weights=weights,
        metadata={
            "assessment_level": "residue",
            "residue_identifier": (
                group.residue_identifier
            ),
        },
    )


def classify_hydrophobic_local_region(
    region: HydrophobicLocalRegion,
    *,
    weights: Optional[Mapping[str, Number]] = None,
) -> HydrophobicStrengthAssessment:
    """
    Assess the complete contribution of one local contact region.
    """

    if not isinstance(
        region,
        HydrophobicLocalRegion,
    ):
        raise TypeError(
            "region must be a HydrophobicLocalRegion."
        )

    return assess_hydrophobic_interaction_strength(
        region.interactions,
        weights=weights,
        metadata={
            "assessment_level": "local_region",
            "region_identifier": (
                region.region_identifier
            ),
        },
    )


# -----------------------------------------------------------------------------
# Rebuilding residue groups after refinement
# -----------------------------------------------------------------------------

def rebuild_hydrophobic_residue_groups(
    interactions: HydrophobicInteractionCollection,
    *,
    identify_hotspots: bool = True,
) -> Tuple[HydrophobicResidueGroup, ...]:
    """
    Recreate residue groups using final interaction scores.
    """

    return group_hydrophobic_interactions_by_residue(
        interactions,
        identify_hotspots=identify_hotspots,
        sort_by="score",
    )


# -----------------------------------------------------------------------------
# Complete result refinement
# -----------------------------------------------------------------------------

def add_hydrophobic_classification_to_result(
    result: HydrophobicAnalysisResult,
    *,
    grouping_distance: Optional[Number] = None,
    weights: Optional[Mapping[str, Number]] = None,
    regroup_interactions: bool = True,
    identify_hotspots: bool = True,
) -> HydrophobicAnalysisResult:
    """
    Return a new analysis result with final multifactor classifications.

    This function:

    1. refines all atomic interaction scores;
    2. rebuilds residue and local-region groups;
    3. updates result metadata;
    4. marks distance-only classifications as replaced.
    """

    if not isinstance(
        result,
        HydrophobicAnalysisResult,
    ):
        raise TypeError(
            "result must be a HydrophobicAnalysisResult."
        )

    resolved_grouping_distance = (
        result.grouping_distance
        if grouping_distance is None
        else _positive_float(
            grouping_distance,
            name="grouping distance",
        )
    )

    refined_interactions = (
        refine_hydrophobic_interactions(
            result.interactions,
            grouping_distance=(
                resolved_grouping_distance
            ),
            weights=weights,
            sort_interactions=True,
        )
    )

    if regroup_interactions:
        grouping = group_hydrophobic_interactions(
            refined_interactions,
            grouping_distance=(
                resolved_grouping_distance
            ),
            identify_hotspots=identify_hotspots,
        )

        residue_groups = (
            grouping.residue_groups
        )

    else:
        grouping = None

        residue_groups = (
            rebuild_hydrophobic_residue_groups(
                refined_interactions,
                identify_hotspots=identify_hotspots,
            )
        )

    updated_metadata = dict(
        result.metadata
    )

    updated_metadata.update(
        {
            "analysis_stage": (
                "final_geometric_classification"
            ),
            "geometric_classification_is_preliminary": False,
            "score_is_preliminary": False,
            "classification_method": (
                "multifactor_geometric_and_chemical"
            ),
            "classification_thresholds": {
                "very_strong": float(
                    HYDROPHOBIC_VERY_STRONG_MINIMUM_SCORE
                ),
                "strong": float(
                    HYDROPHOBIC_STRONG_MINIMUM_SCORE
                ),
                "moderate": float(
                    HYDROPHOBIC_MODERATE_MINIMUM_SCORE
                ),
                "weak": float(
                    HYDROPHOBIC_WEAK_MINIMUM_SCORE
                ),
                "marginal": float(
                    HYDROPHOBIC_MARGINAL_MINIMUM_SCORE
                ),
            },
            "score_weights": {
                key: float(value)
                for key, value
                in HYDROPHOBIC_FINAL_SCORE_WEIGHTS.items()
            },
            "polarity_penalty_weight": float(
                HYDROPHOBIC_POLARITY_PENALTY_WEIGHT
            ),
            "redundancy_penalty_weight": float(
                HYDROPHOBIC_REDUNDANCY_PENALTY_WEIGHT
            ),
            "aromatic_aromatic_is_not_pi_stacking": True,
        }
    )

    if grouping is not None:
        updated_metadata.update(
            {
                "residue_grouping_completed": True,
                "chain_group_count": (
                    grouping.chain_count
                ),
                "pose_group_count": (
                    grouping.pose_count
                ),
                "local_region_count": (
                    grouping.local_region_count
                ),
                "hotspot_count": (
                    grouping.hotspot_count
                ),
                "approximate_contact_area": float(
                    grouping.approximate_contact_area
                ),
                "chain_groups": [
                    chain_group.to_dict(
                        include_interactions=False
                    )
                    for chain_group
                    in grouping.chain_groups
                ],
                "pose_groups": [
                    pose_group.to_dict(
                        include_interactions=False
                    )
                    for pose_group
                    in grouping.pose_groups
                ],
                "local_regions": [
                    region.to_dict(
                        include_interactions=False,
                        include_atoms=False,
                    )
                    for region
                    in grouping.local_regions
                ],
            }
        )

    updated_statistics = (
        _build_detection_statistics(
            refined_interactions,
            receptor_atom_count=len(
                result.receptor_atoms
            ),
            ligand_atom_count=len(
                result.ligand_atoms
            ),
        )
    )

    statistic_metadata = dict(
        updated_statistics.metadata
    )

    statistic_metadata.update(
        {
            "statistics_stage": (
                "final_geometric_classification"
            ),
            "classification_is_final": True,
            "residue_group_count": len(
                residue_groups
            ),
            "hotspot_count": sum(
                bool(
                    group.metadata.get(
                        "is_hotspot",
                        False,
                    )
                )
                for group in residue_groups
            ),
        }
    )

    # HydrophobicStatistics is immutable, so it is reconstructed with
    # final residue and hotspot information.
    final_statistics = HydrophobicStatistics(
        interaction_count=(
            updated_statistics.interaction_count
        ),
        residue_count=len(
            residue_groups
        ),
        receptor_atom_count=(
            updated_statistics.receptor_atom_count
        ),
        ligand_atom_count=(
            updated_statistics.ligand_atom_count
        ),
        very_strong_count=(
            updated_statistics.very_strong_count
        ),
        strong_count=(
            updated_statistics.strong_count
        ),
        moderate_count=(
            updated_statistics.moderate_count
        ),
        weak_count=(
            updated_statistics.weak_count
        ),
        marginal_count=(
            updated_statistics.marginal_count
        ),
        unknown_count=(
            updated_statistics.unknown_count
        ),
        aliphatic_aliphatic_count=(
            updated_statistics.aliphatic_aliphatic_count
        ),
        aliphatic_aromatic_count=(
            updated_statistics.aliphatic_aromatic_count
        ),
        aromatic_aliphatic_count=(
            updated_statistics.aromatic_aliphatic_count
        ),
        aromatic_aromatic_count=(
            updated_statistics.aromatic_aromatic_count
        ),
        mixed_count=(
            updated_statistics.mixed_count
        ),
        hotspot_count=sum(
            bool(
                group.metadata.get(
                    "is_hotspot",
                    False,
                )
            )
            for group in residue_groups
        ),
        minimum_distance=(
            updated_statistics.minimum_distance
        ),
        mean_distance=(
            updated_statistics.mean_distance
        ),
        median_distance=(
            updated_statistics.median_distance
        ),
        maximum_distance=(
            updated_statistics.maximum_distance
        ),
        distance_standard_deviation=(
            updated_statistics.distance_standard_deviation
        ),
        minimum_score=(
            updated_statistics.minimum_score
        ),
        mean_score=(
            updated_statistics.mean_score
        ),
        median_score=(
            updated_statistics.median_score
        ),
        maximum_score=(
            updated_statistics.maximum_score
        ),
        total_score=(
            updated_statistics.total_score
        ),
        minimum_strength=(
            updated_statistics.minimum_strength
        ),
        mean_strength=(
            updated_statistics.mean_strength
        ),
        maximum_strength=(
            updated_statistics.maximum_strength
        ),
        classification_counts=(
            updated_statistics.classification_counts
        ),
        interaction_type_counts=(
            updated_statistics.interaction_type_counts
        ),
        residue_interaction_counts={
            (
                group.residue_identifier
                or "residue-unknown"
            ): group.interaction_count
            for group in residue_groups
        },
        residue_scores={
            (
                group.residue_identifier
                or "residue-unknown"
            ): float(
                group.group_score or 0.0
            )
            for group in residue_groups
        },
        metadata=statistic_metadata,
    )

    return HydrophobicAnalysisResult(
        interactions=refined_interactions,
        residue_groups=residue_groups,
        receptor_hydrophobic_atoms=(
            result.receptor_hydrophobic_atoms
        ),
        ligand_hydrophobic_atoms=(
            result.ligand_hydrophobic_atoms
        ),
        receptor_atoms=result.receptor_atoms,
        ligand_atoms=result.ligand_atoms,
        minimum_distance=result.minimum_distance,
        maximum_distance=result.maximum_distance,
        grouping_distance=(
            resolved_grouping_distance
        ),
        statistics=final_statistics,
        analysis_identifier=(
            result.analysis_identifier
        ),
        receptor_identifier=(
            result.receptor_identifier
        ),
        ligand_identifier=(
            result.ligand_identifier
        ),
        metadata=updated_metadata,
    )


# -----------------------------------------------------------------------------
# Combined detection, grouping and classification workflow
# -----------------------------------------------------------------------------

def analyze_and_classify_hydrophobic_interactions(
    receptor: Union[
        Any,
        HydrophobicAtomCollections,
    ],
    ligand: Optional[Any] = None,
    *,
    minimum_distance: Optional[Number] = None,
    maximum_distance: Optional[Number] = None,
    grouping_distance: Optional[Number] = None,
    local_radius: Optional[Number] = None,
    weights: Optional[Mapping[str, Number]] = None,
    preparation_options: Optional[
        Mapping[str, Any]
    ] = None,
    identify_hotspots: bool = True,
    receptor_identifier: Optional[str] = None,
    ligand_identifier: Optional[str] = None,
    analysis_identifier: Optional[str] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> HydrophobicAnalysisResult:
    """
    Run detection, grouping and final geometric classification.
    """

    detected_result = detect_hydrophobic_interactions(
        receptor,
        ligand,
        minimum_distance=minimum_distance,
        maximum_distance=maximum_distance,
        grouping_distance=grouping_distance,
        local_radius=local_radius,
        preparation_options=preparation_options,
        receptor_identifier=receptor_identifier,
        ligand_identifier=ligand_identifier,
        analysis_identifier=analysis_identifier,
        metadata=metadata,
    )

    grouped_result = add_hydrophobic_grouping_to_result(
        detected_result,
        grouping_distance=grouping_distance,
        identify_hotspots=identify_hotspots,
    )

    return add_hydrophobic_classification_to_result(
        grouped_result,
        grouping_distance=grouping_distance,
        weights=weights,
        regroup_interactions=True,
        identify_hotspots=identify_hotspots,
    )


# -----------------------------------------------------------------------------
# Section 9 public names
# -----------------------------------------------------------------------------

_SECTION_9_PUBLIC_NAMES: Final[Tuple[str, ...]] = (
    # Aliases
    "HydrophobicScoreComponentName",
    "HydrophobicScoreComponentMap",

    # Dataclass
    "HydrophobicStrengthAssessment",

    # Normalization
    "saturating_hydrophobic_score",
    "linear_saturating_hydrophobic_score",

    # Final classification
    "classify_hydrophobic_strength_score",

    # Local context
    "find_related_hydrophobic_interactions",

    # Positive score components
    "calculate_hydrophobic_distance_component",
    "calculate_hydrophobic_compaction_component",
    "calculate_hydrophobic_density_component",
    "calculate_hydrophobic_contact_count_component",
    "calculate_hydrophobic_atom_diversity_component",
    "calculate_hydrophobic_group_size_component",
    "predominant_hydrophobic_interaction_type",
    "calculate_hydrophobic_chemical_type_component",
    "calculate_hydrophobic_atom_character_components",
    "combine_hydrophobic_atom_character",
    "calculate_hydrophobic_surface_component",

    # Penalties
    "calculate_hydrophobic_polar_penalty",
    "calculate_hydrophobic_redundancy_penalty",
    "calculate_additional_hydrophobic_penalty",

    # Combined score
    "calculate_combined_hydrophobic_contribution",
    "assess_hydrophobic_interaction_strength",

    # Interaction reconstruction and refinement
    "rebuild_hydrophobic_interaction",
    "classify_hydrophobic_interaction",
    "refine_hydrophobic_interaction",
    "refine_hydrophobic_interactions",

    # Group classification
    "classify_hydrophobic_residue_group",
    "classify_hydrophobic_local_region",
    "rebuild_hydrophobic_residue_groups",

    # Result-level workflows
    "add_hydrophobic_classification_to_result",
    "analyze_and_classify_hydrophobic_interactions",
)

for public_name in _SECTION_9_PUBLIC_NAMES:
    if public_name not in __all__:
        __all__.append(public_name)


# =============================================================================
# End of Section 9
# =============================================================================




# =============================================================================
# Section 10 — Statistics, summaries and serializable tables
# =============================================================================


# -----------------------------------------------------------------------------
# Statistics-related aliases
# -----------------------------------------------------------------------------

HydrophobicSerializableRow: TypeAlias = Dict[str, Any]

HydrophobicSerializableTable: TypeAlias = Tuple[
    HydrophobicSerializableRow,
    ...,
]

HydrophobicDistribution: TypeAlias = Mapping[str, int]

HydrophobicOccupancyMap: TypeAlias = Mapping[
    str,
    np.float64,
]


# -----------------------------------------------------------------------------
# Statistics defaults
# -----------------------------------------------------------------------------

DEFAULT_HYDROPHOBIC_OCCUPANCY_THRESHOLD: Final[np.float64] = np.float64(
    0.0
)

DEFAULT_HYDROPHOBIC_ROUND_DIGITS: Final[int] = 6

DEFAULT_INCLUDE_ZERO_DISTRIBUTION_CLASSES: Final[bool] = True

DEFAULT_SERIALIZE_INTERACTION_METADATA: Final[bool] = False
DEFAULT_SERIALIZE_GROUP_METADATA: Final[bool] = False

DEFAULT_RESIDUE_TABLE_SORTING: Final[str] = "score"
DEFAULT_POSE_TABLE_SORTING: Final[str] = "pose"
DEFAULT_INTERACTION_TABLE_SORTING: Final[str] = "score"


# -----------------------------------------------------------------------------
# Statistical utility helpers
# -----------------------------------------------------------------------------

def _round_optional_float(
    value: Optional[Number],
    *,
    digits: int = DEFAULT_HYDROPHOBIC_ROUND_DIGITS,
) -> Optional[float]:
    """Return a rounded float or ``None``."""

    if value is None:
        return None

    normalized_digits = _nonnegative_integer(
        digits,
        name="rounding digits",
    )

    return round(
        float(
            _finite_float(
                value,
                name="serializable numeric value",
            )
        ),
        normalized_digits,
    )


def _safe_mean(
    values: Sequence[Number],
) -> Optional[np.float64]:
    """Return the arithmetic mean of finite values."""

    if not values:
        return None

    array = np.asarray(
        values,
        dtype=np.float64,
    )

    array = array[
        np.isfinite(array)
    ]

    if array.size == 0:
        return None

    return np.float64(
        np.mean(array)
    )


def _safe_median(
    values: Sequence[Number],
) -> Optional[np.float64]:
    """Return the median of finite values."""

    if not values:
        return None

    array = np.asarray(
        values,
        dtype=np.float64,
    )

    array = array[
        np.isfinite(array)
    ]

    if array.size == 0:
        return None

    return np.float64(
        np.median(array)
    )


def _safe_minimum(
    values: Sequence[Number],
) -> Optional[np.float64]:
    """Return the minimum finite value."""

    if not values:
        return None

    array = np.asarray(
        values,
        dtype=np.float64,
    )

    array = array[
        np.isfinite(array)
    ]

    if array.size == 0:
        return None

    return np.float64(
        np.min(array)
    )


def _safe_maximum(
    values: Sequence[Number],
) -> Optional[np.float64]:
    """Return the maximum finite value."""

    if not values:
        return None

    array = np.asarray(
        values,
        dtype=np.float64,
    )

    array = array[
        np.isfinite(array)
    ]

    if array.size == 0:
        return None

    return np.float64(
        np.max(array)
    )


def _safe_standard_deviation(
    values: Sequence[Number],
) -> Optional[np.float64]:
    """Return the population standard deviation of finite values."""

    if not values:
        return None

    array = np.asarray(
        values,
        dtype=np.float64,
    )

    array = array[
        np.isfinite(array)
    ]

    if array.size == 0:
        return None

    return np.float64(
        np.std(
            array,
            ddof=0,
        )
    )


def _safe_sum(
    values: Sequence[Number],
) -> np.float64:
    """Return the finite sum of numeric values."""

    if not values:
        return np.float64(0.0)

    array = np.asarray(
        values,
        dtype=np.float64,
    )

    array = array[
        np.isfinite(array)
    ]

    if array.size == 0:
        return np.float64(0.0)

    return np.float64(
        np.sum(array)
    )


def _normalized_fraction(
    numerator: Number,
    denominator: Number,
) -> np.float64:
    """Return a normalized fraction in the interval ``[0, 1]``."""

    normalized_numerator = _nonnegative_float(
        numerator,
        name="fraction numerator",
    )

    normalized_denominator = _nonnegative_float(
        denominator,
        name="fraction denominator",
    )

    if normalized_denominator == 0.0:
        return np.float64(0.0)

    return validate_hydrophobic_score(
        normalized_numerator
        / normalized_denominator
    )


def _sorted_mapping_proxy(
    values: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Return an immutable mapping sorted by key."""

    return MappingProxyType(
        {
            key: values[key]
            for key in sorted(values)
        }
    )


# -----------------------------------------------------------------------------
# Interaction and grouping resolution
# -----------------------------------------------------------------------------

def _resolve_statistics_interactions(
    source: Union[
        HydrophobicAnalysisResult,
        HydrophobicGroupingResult,
        HydrophobicDetectionResult,
        HydrophobicInteractionCollection,
    ],
) -> Tuple[HydrophobicInteraction, ...]:
    """Resolve deduplicated interactions from a supported source."""

    if isinstance(
        source,
        HydrophobicAnalysisResult,
    ):
        interactions = source.interactions

    elif isinstance(
        source,
        HydrophobicGroupingResult,
    ):
        interactions = source.interactions

    elif isinstance(
        source,
        HydrophobicDetectionResult,
    ):
        interactions = source.interactions

    else:
        interactions = source

    return deduplicate_hydrophobic_interactions(
        interactions,
        prefer_highest_score=True,
    )


def _resolve_statistics_grouping(
    source: Union[
        HydrophobicAnalysisResult,
        HydrophobicGroupingResult,
        HydrophobicDetectionResult,
        HydrophobicInteractionCollection,
    ],
    *,
    grouping_distance: Optional[Number] = None,
    identify_hotspots: bool = True,
) -> HydrophobicGroupingResult:
    """Resolve or create complete grouping information."""

    if isinstance(
        source,
        HydrophobicGroupingResult,
    ):
        return source

    interactions = _resolve_statistics_interactions(
        source
    )

    return group_hydrophobic_interactions(
        interactions,
        grouping_distance=grouping_distance,
        identify_hotspots=identify_hotspots,
    )


def _resolve_total_receptor_atom_count(
    source: Any,
    interactions: Sequence[HydrophobicInteraction],
) -> int:
    """Resolve the total receptor atom count."""

    if isinstance(
        source,
        HydrophobicAnalysisResult,
    ):
        return len(
            source.receptor_atoms
        )

    if isinstance(
        source,
        HydrophobicDetectionResult,
    ):
        return len(
            source.prepared_collections.receptor_atoms
        )

    return len(
        deduplicate_atoms(
            (
                interaction.receptor_atom
                for interaction in interactions
            ),
            strategy="auto",
        )
    )


def _resolve_total_ligand_atom_count(
    source: Any,
    interactions: Sequence[HydrophobicInteraction],
) -> int:
    """Resolve the total ligand atom count."""

    if isinstance(
        source,
        HydrophobicAnalysisResult,
    ):
        return len(
            source.ligand_atoms
        )

    if isinstance(
        source,
        HydrophobicDetectionResult,
    ):
        return len(
            source.prepared_collections.ligand_atoms
        )

    return len(
        deduplicate_atoms(
            (
                interaction.ligand_atom
                for interaction in interactions
            ),
            strategy="auto",
        )
    )


# -----------------------------------------------------------------------------
# Atomic-pair statistics
# -----------------------------------------------------------------------------

def count_hydrophobic_atomic_pairs(
    interactions: HydrophobicInteractionCollection,
) -> int:
    """Return the number of unique receptor–ligand atomic pairs."""

    return len(
        deduplicate_hydrophobic_interactions(
            interactions,
            prefer_highest_score=True,
        )
    )


def count_hydrophobic_interactions(
    interactions: HydrophobicInteractionCollection,
    *,
    grouping_distance: Optional[Number] = None,
    count_local_regions: bool = False,
) -> int:
    """
    Return the total interaction count.

    By default, one structured atomic contact is counted as one
    interaction. When ``count_local_regions=True``, spatially connected
    atomic pairs are counted as broader local interactions.
    """

    interaction_tuple = (
        deduplicate_hydrophobic_interactions(
            interactions
        )
    )

    if not count_local_regions:
        return len(interaction_tuple)

    return len(
        cluster_hydrophobic_local_regions(
            interaction_tuple,
            grouping_distance=grouping_distance,
            require_same_pose=True,
        )
    )


def hydrophobic_distance_statistics(
    interactions: HydrophobicInteractionCollection,
) -> Mapping[str, Optional[np.float64]]:
    """
    Return descriptive statistics for atomic-pair distances.
    """

    interaction_tuple = tuple(
        interactions
    )

    distances = [
        interaction.distance
        for interaction in interaction_tuple
    ]

    return MappingProxyType(
        {
            "minimum": _safe_minimum(
                distances
            ),
            "mean": _safe_mean(
                distances
            ),
            "median": _safe_median(
                distances
            ),
            "maximum": _safe_maximum(
                distances
            ),
            "standard_deviation": (
                _safe_standard_deviation(
                    distances
                )
            ),
        }
    )


def hydrophobic_score_statistics(
    interactions: HydrophobicInteractionCollection,
) -> Mapping[str, Optional[np.float64]]:
    """
    Return descriptive statistics for final interaction scores.
    """

    interaction_tuple = tuple(
        interactions
    )

    scores = [
        interaction.score
        for interaction in interaction_tuple
    ]

    return MappingProxyType(
        {
            "minimum": _safe_minimum(
                scores
            ),
            "mean": _safe_mean(
                scores
            ),
            "median": _safe_median(
                scores
            ),
            "maximum": _safe_maximum(
                scores
            ),
            "standard_deviation": (
                _safe_standard_deviation(
                    scores
                )
            ),
            "total": _safe_sum(
                scores
            ),
        }
    )


def hydrophobic_strength_statistics(
    interactions: HydrophobicInteractionCollection,
) -> Mapping[str, Optional[np.float64]]:
    """
    Return descriptive statistics for geometric strength values.
    """

    interaction_tuple = tuple(
        interactions
    )

    strengths = [
        interaction.strength
        for interaction in interaction_tuple
    ]

    return MappingProxyType(
        {
            "minimum": _safe_minimum(
                strengths
            ),
            "mean": _safe_mean(
                strengths
            ),
            "median": _safe_median(
                strengths
            ),
            "maximum": _safe_maximum(
                strengths
            ),
            "standard_deviation": (
                _safe_standard_deviation(
                    strengths
                )
            ),
            "total": _safe_sum(
                strengths
            ),
        }
    )


# -----------------------------------------------------------------------------
# Classification and type distributions
# -----------------------------------------------------------------------------

def hydrophobic_classification_distribution(
    interactions: HydrophobicInteractionCollection,
    *,
    include_zero_classes: bool = (
        DEFAULT_INCLUDE_ZERO_DISTRIBUTION_CLASSES
    ),
) -> Mapping[str, int]:
    """
    Return the interaction distribution by final strength class.
    """

    counts: Dict[str, int] = {
        classification: 0
        for classification
        in _VALID_HYDROPHOBIC_CLASSIFICATIONS
    }

    for interaction in interactions:
        counts[
            interaction.classification
        ] = (
            counts.get(
                interaction.classification,
                0,
            )
            + 1
        )

    if not include_zero_classes:
        counts = {
            key: value
            for key, value in counts.items()
            if value > 0
        }

    return _sorted_mapping_proxy(
        counts
    )


def hydrophobic_interaction_type_distribution(
    interactions: HydrophobicInteractionCollection,
    *,
    include_zero_types: bool = (
        DEFAULT_INCLUDE_ZERO_DISTRIBUTION_CLASSES
    ),
) -> Mapping[str, int]:
    """
    Return the distribution by hydrophobic contact type.
    """

    counts: Dict[str, int] = {
        interaction_type: 0
        for interaction_type
        in _VALID_HYDROPHOBIC_TYPES
    }

    for interaction in interactions:
        counts[
            interaction.interaction_type
        ] = (
            counts.get(
                interaction.interaction_type,
                0,
            )
            + 1
        )

    if not include_zero_types:
        counts = {
            key: value
            for key, value in counts.items()
            if value > 0
        }

    return _sorted_mapping_proxy(
        counts
    )


def hydrophobic_classification_fraction_distribution(
    interactions: HydrophobicInteractionCollection,
) -> Mapping[str, np.float64]:
    """
    Return normalized interaction fractions by strength class.
    """

    counts = hydrophobic_classification_distribution(
        interactions,
        include_zero_classes=True,
    )

    total = sum(
        counts.values()
    )

    return MappingProxyType(
        {
            classification: _normalized_fraction(
                count,
                total,
            )
            for classification, count
            in counts.items()
        }
    )


def hydrophobic_type_fraction_distribution(
    interactions: HydrophobicInteractionCollection,
) -> Mapping[str, np.float64]:
    """
    Return normalized interaction fractions by contact type.
    """

    counts = hydrophobic_interaction_type_distribution(
        interactions,
        include_zero_types=True,
    )

    total = sum(
        counts.values()
    )

    return MappingProxyType(
        {
            interaction_type: _normalized_fraction(
                count,
                total,
            )
            for interaction_type, count
            in counts.items()
        }
    )


# -----------------------------------------------------------------------------
# Residue-level statistics
# -----------------------------------------------------------------------------

def hydrophobic_score_by_residue(
    interactions: HydrophobicInteractionCollection,
    *,
    grouped_score: bool = True,
) -> Mapping[str, np.float64]:
    """
    Return the hydrophobic score associated with each receptor residue.

    When ``grouped_score=True``, diminishing returns and atom-diversity
    contributions from Section 8 are used. Otherwise, atomic scores are
    summed directly.
    """

    groups = group_hydrophobic_interactions_by_residue(
        interactions,
        identify_hotspots=True,
        sort_by="identifier",
    )

    scores: Dict[str, np.float64] = {}

    for group in groups:
        identifier = (
            group.residue_identifier
            or "residue-unknown"
        )

        if grouped_score:
            score = np.float64(
                group.group_score or 0.0
            )

        else:
            score = _safe_sum(
                [
                    interaction.score
                    for interaction
                    in group.interactions
                ]
            )

        scores[identifier] = score

    return _sorted_mapping_proxy(
        scores
    )


def hydrophobic_contact_count_by_residue(
    interactions: HydrophobicInteractionCollection,
) -> Mapping[str, int]:
    """Return atomic-pair contact counts by receptor residue."""

    return count_hydrophobic_contacts_by_residue(
        interactions
    )


def hydrophobic_minimum_distance_by_residue(
    interactions: HydrophobicInteractionCollection,
) -> Mapping[str, Optional[np.float64]]:
    """Return the closest contact distance for every residue."""

    values: Dict[
        str,
        Optional[np.float64],
    ] = {}

    for group in group_hydrophobic_interactions_by_residue(
        interactions,
        sort_by="identifier",
    ):
        identifier = (
            group.residue_identifier
            or "residue-unknown"
        )

        values[identifier] = (
            group.minimum_distance
        )

    return _sorted_mapping_proxy(
        values
    )


def hydrophobic_residue_identifiers(
    interactions: HydrophobicInteractionCollection,
) -> Tuple[str, ...]:
    """Return sorted unique receptor-residue identifiers."""

    identifiers = {
        interaction.receptor_residue_identifier
        or "residue-unknown"
        for interaction in interactions
    }

    return tuple(
        sorted(identifiers)
    )


def hydrophobic_chain_identifiers(
    interactions: HydrophobicInteractionCollection,
) -> Tuple[str, ...]:
    """Return sorted unique contacted receptor-chain identifiers."""

    identifiers = {
        get_interaction_chain_identifier(
            interaction,
            default="chain-unknown",
        )
        or "chain-unknown"
        for interaction in interactions
    }

    return tuple(
        sorted(identifiers)
    )


# -----------------------------------------------------------------------------
# Pose occupancy
# -----------------------------------------------------------------------------

def hydrophobic_pose_identifiers(
    interactions: HydrophobicInteractionCollection,
) -> Tuple[str, ...]:
    """Return sorted unique pose identifiers."""

    return tuple(
        sorted(
            {
                get_interaction_pose_identifier(
                    interaction
                )
                for interaction in interactions
            }
        )
    )


def hydrophobic_residue_pose_presence(
    interactions: HydrophobicInteractionCollection,
) -> Mapping[str, Tuple[str, ...]]:
    """
    Return poses in which each receptor residue has at least one contact.
    """

    presence: Dict[
        str,
        Set[str],
    ] = {}

    for interaction in interactions:
        residue_identifier = (
            interaction.receptor_residue_identifier
            or "residue-unknown"
        )

        pose_identifier = (
            get_interaction_pose_identifier(
                interaction
            )
        )

        presence.setdefault(
            residue_identifier,
            set(),
        ).add(
            pose_identifier
        )

    return MappingProxyType(
        {
            residue_identifier: tuple(
                sorted(pose_identifiers)
            )
            for residue_identifier, pose_identifiers
            in sorted(
                presence.items()
            )
        }
    )


def calculate_hydrophobic_residue_pose_occupancy(
    interactions: HydrophobicInteractionCollection,
    *,
    pose_identifiers: Optional[Iterable[str]] = None,
    minimum_group_score: Number = (
        DEFAULT_HYDROPHOBIC_OCCUPANCY_THRESHOLD
    ),
) -> Mapping[str, np.float64]:
    """
    Calculate residue occupancy across ligand poses.

    Occupancy is the fraction of represented poses in which the residue
    participates in at least one group whose score is above the optional
    threshold.
    """

    interaction_tuple = (
        deduplicate_hydrophobic_interactions(
            interactions
        )
    )

    minimum_score = _nonnegative_float(
        minimum_group_score,
        name="minimum occupancy group score",
    )

    resolved_pose_identifiers = (
        tuple(
            sorted(
                {
                    str(value).strip()
                    for value in pose_identifiers
                    if str(value).strip()
                }
            )
        )
        if pose_identifiers is not None
        else hydrophobic_pose_identifiers(
            interaction_tuple
        )
    )

    pose_count = len(
        resolved_pose_identifiers
    )

    if pose_count == 0:
        return MappingProxyType({})

    pose_identifier_set = set(
        resolved_pose_identifiers
    )

    residue_pose_presence: Dict[
        str,
        Set[str],
    ] = {}

    pose_groups = (
        group_hydrophobic_interactions_by_pose(
            interaction_tuple,
            include_unknown_pose=True,
        )
    )

    for pose_group in pose_groups:
        if (
            pose_group.pose_identifier
            not in pose_identifier_set
        ):
            continue

        for residue_group in pose_group.residue_groups:
            group_score = float(
                residue_group.group_score or 0.0
            )

            if group_score < minimum_score:
                continue

            residue_identifier = (
                residue_group.residue_identifier
                or "residue-unknown"
            )

            residue_pose_presence.setdefault(
                residue_identifier,
                set(),
            ).add(
                pose_group.pose_identifier
            )

    occupancy = {
        residue_identifier: np.float64(
            len(present_poses)
            / pose_count
        )
        for residue_identifier, present_poses
        in residue_pose_presence.items()
    }

    return _sorted_mapping_proxy(
        occupancy
    )


def calculate_hydrophobic_chain_pose_occupancy(
    interactions: HydrophobicInteractionCollection,
    *,
    pose_identifiers: Optional[Iterable[str]] = None,
) -> Mapping[str, np.float64]:
    """
    Calculate receptor-chain occupancy across poses.
    """

    interaction_tuple = tuple(
        interactions
    )

    resolved_pose_identifiers = (
        tuple(
            sorted(
                {
                    str(value).strip()
                    for value in pose_identifiers
                    if str(value).strip()
                }
            )
        )
        if pose_identifiers is not None
        else hydrophobic_pose_identifiers(
            interaction_tuple
        )
    )

    pose_count = len(
        resolved_pose_identifiers
    )

    if pose_count == 0:
        return MappingProxyType({})

    presence: Dict[
        str,
        Set[str],
    ] = {}

    for interaction in interaction_tuple:
        chain_identifier = (
            get_interaction_chain_identifier(
                interaction,
                default="chain-unknown",
            )
            or "chain-unknown"
        )

        pose_identifier = (
            get_interaction_pose_identifier(
                interaction
            )
        )

        presence.setdefault(
            chain_identifier,
            set(),
        ).add(
            pose_identifier
        )

    return _sorted_mapping_proxy(
        {
            chain_identifier: np.float64(
                len(present_poses)
                / pose_count
            )
            for chain_identifier, present_poses
            in presence.items()
        }
    )


def calculate_hydrophobic_interaction_pose_occupancy(
    interactions: HydrophobicInteractionCollection,
    *,
    pose_identifiers: Optional[Iterable[str]] = None,
) -> Mapping[str, np.float64]:
    """
    Calculate occupancy of receptor–ligand atom-pair signatures.

    Ligand atom identifiers can vary across poses. Therefore, the
    signature contains receptor residue, receptor atom and ligand atom
    name/identifier rather than Python object identity.
    """

    interaction_tuple = tuple(
        interactions
    )

    resolved_pose_identifiers = (
        tuple(
            sorted(
                {
                    str(value).strip()
                    for value in pose_identifiers
                    if str(value).strip()
                }
            )
        )
        if pose_identifiers is not None
        else hydrophobic_pose_identifiers(
            interaction_tuple
        )
    )

    pose_count = len(
        resolved_pose_identifiers
    )

    if pose_count == 0:
        return MappingProxyType({})

    signature_presence: Dict[
        str,
        Set[str],
    ] = {}

    for interaction in interaction_tuple:
        residue_identifier = (
            interaction.receptor_residue_identifier
            or "residue-unknown"
        )

        receptor_identifier = (
            interaction.receptor_atom_identifier
            or _safe_atom_identifier(
                interaction.receptor_atom,
                fallback="receptor-atom",
            )
            or "receptor-atom"
        )

        ligand_identifier = (
            interaction.ligand_atom_identifier
            or _safe_atom_identifier(
                interaction.ligand_atom,
                fallback="ligand-atom",
            )
            or "ligand-atom"
        )

        signature = (
            f"{residue_identifier}|"
            f"{receptor_identifier}|"
            f"{ligand_identifier}"
        )

        pose_identifier = (
            get_interaction_pose_identifier(
                interaction
            )
        )

        signature_presence.setdefault(
            signature,
            set(),
        ).add(
            pose_identifier
        )

    return _sorted_mapping_proxy(
        {
            signature: np.float64(
                len(present_poses)
                / pose_count
            )
            for signature, present_poses
            in signature_presence.items()
        }
    )


# -----------------------------------------------------------------------------
# Hotspot summaries
# -----------------------------------------------------------------------------

def summarize_hydrophobic_hotspots(
    interactions: HydrophobicInteractionCollection,
    *,
    minimum_contact_count: Optional[int] = None,
    minimum_group_score: Optional[Number] = None,
    minimum_contact_area: Optional[Number] = None,
    minimum_ligand_atom_count: Optional[int] = None,
) -> Tuple[HydrophobicResidueGroup, ...]:
    """Return hotspot residue groups sorted by score."""

    hotspots = find_hydrophobic_hotspots(
        interactions,
        minimum_contact_count=minimum_contact_count,
        minimum_group_score=minimum_group_score,
        minimum_contact_area=minimum_contact_area,
        minimum_ligand_atom_count=minimum_ligand_atom_count,
    )

    return tuple(
        sorted(
            hotspots,
            key=lambda group: (
                -float(
                    group.group_score or 0.0
                ),
                float(
                    group.minimum_distance
                    if group.minimum_distance is not None
                    else np.inf
                ),
                group.residue_identifier or "",
            ),
        )
    )


# -----------------------------------------------------------------------------
# Extended statistics dataclass
# -----------------------------------------------------------------------------

@dataclass(
    frozen=True,
    slots=True,
)
class HydrophobicSummary:
    """
    Complete statistical summary of hydrophobic interactions.

    This object complements the base :class:`HydrophobicStatistics`
    dataclass with pose occupancy, local-region information, contact
    surface estimates and directly serializable tables.
    """

    interaction_count: int = 0
    atomic_pair_count: int = 0
    local_interaction_count: int = 0

    residue_count: int = 0
    chain_count: int = 0
    pose_count: int = 0
    hotspot_count: int = 0

    receptor_atom_count: int = 0
    ligand_atom_count: int = 0

    contacted_receptor_atom_count: int = 0
    contacted_ligand_atom_count: int = 0

    minimum_distance: Optional[np.float64] = None
    mean_distance: Optional[np.float64] = None
    median_distance: Optional[np.float64] = None
    maximum_distance: Optional[np.float64] = None
    distance_standard_deviation: Optional[np.float64] = None

    minimum_score: Optional[np.float64] = None
    mean_score: Optional[np.float64] = None
    median_score: Optional[np.float64] = None
    maximum_score: Optional[np.float64] = None
    total_score: np.float64 = np.float64(0.0)

    minimum_strength: Optional[np.float64] = None
    mean_strength: Optional[np.float64] = None
    median_strength: Optional[np.float64] = None
    maximum_strength: Optional[np.float64] = None
    total_strength: np.float64 = np.float64(0.0)

    approximate_contact_area: np.float64 = np.float64(0.0)

    classification_counts: Mapping[str, int] = field(
        default_factory=lambda: MappingProxyType({})
    )

    classification_fractions: Mapping[
        str,
        np.float64,
    ] = field(
        default_factory=lambda: MappingProxyType({})
    )

    interaction_type_counts: Mapping[str, int] = field(
        default_factory=lambda: MappingProxyType({})
    )

    interaction_type_fractions: Mapping[
        str,
        np.float64,
    ] = field(
        default_factory=lambda: MappingProxyType({})
    )

    residue_contact_counts: Mapping[str, int] = field(
        default_factory=lambda: MappingProxyType({})
    )

    residue_scores: Mapping[
        str,
        np.float64,
    ] = field(
        default_factory=lambda: MappingProxyType({})
    )

    residue_minimum_distances: Mapping[
        str,
        Optional[np.float64],
    ] = field(
        default_factory=lambda: MappingProxyType({})
    )

    residue_contact_areas: Mapping[
        str,
        np.float64,
    ] = field(
        default_factory=lambda: MappingProxyType({})
    )

    residue_pose_occupancy: Mapping[
        str,
        np.float64,
    ] = field(
        default_factory=lambda: MappingProxyType({})
    )

    chain_pose_occupancy: Mapping[
        str,
        np.float64,
    ] = field(
        default_factory=lambda: MappingProxyType({})
    )

    interaction_pose_occupancy: Mapping[
        str,
        np.float64,
    ] = field(
        default_factory=lambda: MappingProxyType({})
    )

    residue_identifiers: Sequence[str] = field(
        default_factory=tuple
    )

    chain_identifiers: Sequence[str] = field(
        default_factory=tuple
    )

    pose_identifiers: Sequence[str] = field(
        default_factory=tuple
    )

    hotspot_residue_identifiers: Sequence[str] = field(
        default_factory=tuple
    )

    metadata: Mapping[str, Any] = field(
        default_factory=lambda: _EMPTY_METADATA,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        """Validate and freeze the complete summary."""

        integer_fields = (
            "interaction_count",
            "atomic_pair_count",
            "local_interaction_count",
            "residue_count",
            "chain_count",
            "pose_count",
            "hotspot_count",
            "receptor_atom_count",
            "ligand_atom_count",
            "contacted_receptor_atom_count",
            "contacted_ligand_atom_count",
        )

        for field_name in integer_fields:
            object.__setattr__(
                self,
                field_name,
                _nonnegative_integer(
                    getattr(
                        self,
                        field_name,
                    ),
                    name=field_name.replace(
                        "_",
                        " ",
                    ),
                ),
            )

        optional_nonnegative_fields = (
            "minimum_distance",
            "mean_distance",
            "median_distance",
            "maximum_distance",
            "distance_standard_deviation",
            "minimum_score",
            "mean_score",
            "median_score",
            "maximum_score",
            "minimum_strength",
            "mean_strength",
            "median_strength",
            "maximum_strength",
        )

        for field_name in optional_nonnegative_fields:
            value = getattr(
                self,
                field_name,
            )

            if value is None:
                continue

            object.__setattr__(
                self,
                field_name,
                _nonnegative_float(
                    value,
                    name=field_name.replace(
                        "_",
                        " ",
                    ),
                ),
            )

        object.__setattr__(
            self,
            "total_score",
            _nonnegative_float(
                self.total_score,
                name="total score",
            ),
        )

        object.__setattr__(
            self,
            "total_strength",
            _nonnegative_float(
                self.total_strength,
                name="total strength",
            ),
        )

        object.__setattr__(
            self,
            "approximate_contact_area",
            _nonnegative_float(
                self.approximate_contact_area,
                name="approximate contact area",
            ),
        )

        integer_mappings = (
            "classification_counts",
            "interaction_type_counts",
            "residue_contact_counts",
        )

        for field_name in integer_mappings:
            mapping = {
                str(key): _nonnegative_integer(
                    value,
                    name=f"{field_name} value",
                )
                for key, value
                in dict(
                    getattr(
                        self,
                        field_name,
                    )
                ).items()
            }

            object.__setattr__(
                self,
                field_name,
                _sorted_mapping_proxy(
                    mapping
                ),
            )

        score_mappings = (
            "classification_fractions",
            "interaction_type_fractions",
            "residue_scores",
            "residue_pose_occupancy",
            "chain_pose_occupancy",
            "interaction_pose_occupancy",
        )

        for field_name in score_mappings:
            mapping = {
                str(key): validate_hydrophobic_score(
                    value
                )
                if "occupancy" in field_name
                or "fraction" in field_name
                else _nonnegative_float(
                    value,
                    name=f"{field_name} value",
                )
                for key, value
                in dict(
                    getattr(
                        self,
                        field_name,
                    )
                ).items()
            }

            object.__setattr__(
                self,
                field_name,
                _sorted_mapping_proxy(
                    mapping
                ),
            )

        residue_minimum_distances = {
            str(key): (
                None
                if value is None
                else _nonnegative_float(
                    value,
                    name="residue minimum distance",
                )
            )
            for key, value
            in dict(
                self.residue_minimum_distances
            ).items()
        }

        object.__setattr__(
            self,
            "residue_minimum_distances",
            _sorted_mapping_proxy(
                residue_minimum_distances
            ),
        )

        residue_contact_areas = {
            str(key): _nonnegative_float(
                value,
                name="residue contact area",
            )
            for key, value
            in dict(
                self.residue_contact_areas
            ).items()
        }

        object.__setattr__(
            self,
            "residue_contact_areas",
            _sorted_mapping_proxy(
                residue_contact_areas
            ),
        )

        sequence_fields = (
            "residue_identifiers",
            "chain_identifiers",
            "pose_identifiers",
            "hotspot_residue_identifiers",
        )

        for field_name in sequence_fields:
            values = tuple(
                sorted(
                    {
                        str(value).strip()
                        for value
                        in getattr(
                            self,
                            field_name,
                        )
                        if str(value).strip()
                    }
                )
            )

            object.__setattr__(
                self,
                field_name,
                values,
            )

        object.__setattr__(
            self,
            "metadata",
            _freeze_metadata(
                self.metadata
            ),
        )

    @property
    def mean_atomic_pairs_per_local_interaction(
        self,
    ) -> np.float64:
        """Return the average number of atomic pairs per local region."""

        if self.local_interaction_count == 0:
            return np.float64(0.0)

        return np.float64(
            self.atomic_pair_count
            / self.local_interaction_count
        )

    @property
    def mean_contacts_per_residue(
        self,
    ) -> np.float64:
        """Return the average contact count per contacted residue."""

        if self.residue_count == 0:
            return np.float64(0.0)

        return np.float64(
            self.atomic_pair_count
            / self.residue_count
        )

    @property
    def receptor_contact_fraction(
        self,
    ) -> np.float64:
        """Return the fraction of receptor atoms involved in contacts."""

        return _normalized_fraction(
            self.contacted_receptor_atom_count,
            self.receptor_atom_count,
        )

    @property
    def ligand_contact_fraction(
        self,
    ) -> np.float64:
        """Return the fraction of ligand atoms involved in contacts."""

        return _normalized_fraction(
            self.contacted_ligand_atom_count,
            self.ligand_atom_count,
        )

    @property
    def mean_score_per_residue(
        self,
    ) -> np.float64:
        """Return the mean grouped score per contacted residue."""

        if not self.residue_scores:
            return np.float64(0.0)

        return np.float64(
            np.mean(
                list(
                    self.residue_scores.values()
                )
            )
        )

    def to_dict(
        self,
        *,
        round_digits: int = (
            DEFAULT_HYDROPHOBIC_ROUND_DIGITS
        ),
    ) -> Dict[str, Any]:
        """Serialize the complete statistical summary."""

        digits = _nonnegative_integer(
            round_digits,
            name="rounding digits",
        )

        return {
            "counts": {
                "interactions": self.interaction_count,
                "atomic_pairs": self.atomic_pair_count,
                "local_interactions": (
                    self.local_interaction_count
                ),
                "residues": self.residue_count,
                "chains": self.chain_count,
                "poses": self.pose_count,
                "hotspots": self.hotspot_count,
                "receptor_atoms": (
                    self.receptor_atom_count
                ),
                "ligand_atoms": (
                    self.ligand_atom_count
                ),
                "contacted_receptor_atoms": (
                    self.contacted_receptor_atom_count
                ),
                "contacted_ligand_atoms": (
                    self.contacted_ligand_atom_count
                ),
            },
            "distance": {
                "minimum": _round_optional_float(
                    self.minimum_distance,
                    digits=digits,
                ),
                "mean": _round_optional_float(
                    self.mean_distance,
                    digits=digits,
                ),
                "median": _round_optional_float(
                    self.median_distance,
                    digits=digits,
                ),
                "maximum": _round_optional_float(
                    self.maximum_distance,
                    digits=digits,
                ),
                "standard_deviation": (
                    _round_optional_float(
                        self.distance_standard_deviation,
                        digits=digits,
                    )
                ),
            },
            "score": {
                "minimum": _round_optional_float(
                    self.minimum_score,
                    digits=digits,
                ),
                "mean": _round_optional_float(
                    self.mean_score,
                    digits=digits,
                ),
                "median": _round_optional_float(
                    self.median_score,
                    digits=digits,
                ),
                "maximum": _round_optional_float(
                    self.maximum_score,
                    digits=digits,
                ),
                "total": _round_optional_float(
                    self.total_score,
                    digits=digits,
                ),
                "mean_per_residue": (
                    _round_optional_float(
                        self.mean_score_per_residue,
                        digits=digits,
                    )
                ),
            },
            "strength": {
                "minimum": _round_optional_float(
                    self.minimum_strength,
                    digits=digits,
                ),
                "mean": _round_optional_float(
                    self.mean_strength,
                    digits=digits,
                ),
                "median": _round_optional_float(
                    self.median_strength,
                    digits=digits,
                ),
                "maximum": _round_optional_float(
                    self.maximum_strength,
                    digits=digits,
                ),
                "total": _round_optional_float(
                    self.total_strength,
                    digits=digits,
                ),
            },
            "surface": {
                "approximate_contact_area": (
                    _round_optional_float(
                        self.approximate_contact_area,
                        digits=digits,
                    )
                ),
            },
            "averages": {
                "atomic_pairs_per_local_interaction": (
                    _round_optional_float(
                        self.mean_atomic_pairs_per_local_interaction,
                        digits=digits,
                    )
                ),
                "contacts_per_residue": (
                    _round_optional_float(
                        self.mean_contacts_per_residue,
                        digits=digits,
                    )
                ),
            },
            "fractions": {
                "receptor_atoms_contacted": (
                    _round_optional_float(
                        self.receptor_contact_fraction,
                        digits=digits,
                    )
                ),
                "ligand_atoms_contacted": (
                    _round_optional_float(
                        self.ligand_contact_fraction,
                        digits=digits,
                    )
                ),
            },
            "classification_counts": dict(
                self.classification_counts
            ),
            "classification_fractions": {
                key: _round_optional_float(
                    value,
                    digits=digits,
                )
                for key, value
                in self.classification_fractions.items()
            },
            "interaction_type_counts": dict(
                self.interaction_type_counts
            ),
            "interaction_type_fractions": {
                key: _round_optional_float(
                    value,
                    digits=digits,
                )
                for key, value
                in self.interaction_type_fractions.items()
            },
            "residue_contact_counts": dict(
                self.residue_contact_counts
            ),
            "residue_scores": {
                key: _round_optional_float(
                    value,
                    digits=digits,
                )
                for key, value
                in self.residue_scores.items()
            },
            "residue_minimum_distances": {
                key: _round_optional_float(
                    value,
                    digits=digits,
                )
                for key, value
                in self.residue_minimum_distances.items()
            },
            "residue_contact_areas": {
                key: _round_optional_float(
                    value,
                    digits=digits,
                )
                for key, value
                in self.residue_contact_areas.items()
            },
            "residue_pose_occupancy": {
                key: _round_optional_float(
                    value,
                    digits=digits,
                )
                for key, value
                in self.residue_pose_occupancy.items()
            },
            "chain_pose_occupancy": {
                key: _round_optional_float(
                    value,
                    digits=digits,
                )
                for key, value
                in self.chain_pose_occupancy.items()
            },
            "interaction_pose_occupancy": {
                key: _round_optional_float(
                    value,
                    digits=digits,
                )
                for key, value
                in self.interaction_pose_occupancy.items()
            },
            "residue_identifiers": list(
                self.residue_identifiers
            ),
            "chain_identifiers": list(
                self.chain_identifiers
            ),
            "pose_identifiers": list(
                self.pose_identifiers
            ),
            "hotspot_residue_identifiers": list(
                self.hotspot_residue_identifiers
            ),
            "metadata": dict(
                self.metadata
            ),
        }


# -----------------------------------------------------------------------------
# Complete summary calculation
# -----------------------------------------------------------------------------

def calculate_hydrophobic_summary(
    source: Union[
        HydrophobicAnalysisResult,
        HydrophobicGroupingResult,
        HydrophobicDetectionResult,
        HydrophobicInteractionCollection,
    ],
    *,
    grouping_distance: Optional[Number] = None,
    pose_identifiers: Optional[Iterable[str]] = None,
    occupancy_minimum_group_score: Number = (
        DEFAULT_HYDROPHOBIC_OCCUPANCY_THRESHOLD
    ),
    identify_hotspots: bool = True,
    metadata: Optional[Mapping[str, Any]] = None,
) -> HydrophobicSummary:
    """
    Calculate the complete hydrophobic statistical summary.
    """

    interactions = _resolve_statistics_interactions(
        source
    )

    grouping = _resolve_statistics_grouping(
        source,
        grouping_distance=grouping_distance,
        identify_hotspots=identify_hotspots,
    )

    distance_statistics = (
        hydrophobic_distance_statistics(
            interactions
        )
    )

    score_statistics = (
        hydrophobic_score_statistics(
            interactions
        )
    )

    strength_statistics = (
        hydrophobic_strength_statistics(
            interactions
        )
    )

    classification_counts = (
        hydrophobic_classification_distribution(
            interactions
        )
    )

    classification_fractions = (
        hydrophobic_classification_fraction_distribution(
            interactions
        )
    )

    interaction_type_counts = (
        hydrophobic_interaction_type_distribution(
            interactions
        )
    )

    interaction_type_fractions = (
        hydrophobic_type_fraction_distribution(
            interactions
        )
    )

    residue_contact_counts = (
        hydrophobic_contact_count_by_residue(
            interactions
        )
    )

    residue_scores = (
        hydrophobic_score_by_residue(
            interactions,
            grouped_score=True,
        )
    )

    residue_minimum_distances = (
        hydrophobic_minimum_distance_by_residue(
            interactions
        )
    )

    residue_contact_areas = (
        hydrophobic_surface_by_residue(
            interactions
        )
    )

    resolved_pose_identifiers = (
        tuple(
            sorted(
                {
                    str(value).strip()
                    for value in pose_identifiers
                    if str(value).strip()
                }
            )
        )
        if pose_identifiers is not None
        else hydrophobic_pose_identifiers(
            interactions
        )
    )

    residue_pose_occupancy = (
        calculate_hydrophobic_residue_pose_occupancy(
            interactions,
            pose_identifiers=(
                resolved_pose_identifiers
            ),
            minimum_group_score=(
                occupancy_minimum_group_score
            ),
        )
    )

    chain_pose_occupancy = (
        calculate_hydrophobic_chain_pose_occupancy(
            interactions,
            pose_identifiers=(
                resolved_pose_identifiers
            ),
        )
    )

    interaction_pose_occupancy = (
        calculate_hydrophobic_interaction_pose_occupancy(
            interactions,
            pose_identifiers=(
                resolved_pose_identifiers
            ),
        )
    )

    receptor_atoms, ligand_atoms = (
        _unique_interaction_atoms(
            interactions
        )
    )

    total_receptor_atom_count = (
        _resolve_total_receptor_atom_count(
            source,
            interactions,
        )
    )

    total_ligand_atom_count = (
        _resolve_total_ligand_atom_count(
            source,
            interactions,
        )
    )

    hotspot_identifiers = tuple(
        sorted(
            {
                group.residue_identifier
                or "residue-unknown"
                for group
                in grouping.hotspot_groups
            }
        )
    )

    summary_metadata: Dict[str, Any] = (
        {} if metadata is None else dict(metadata)
    )

    summary_metadata.update(
        {
            "statistics_stage": "complete_summary",
            "classification_is_final": all(
                not bool(
                    interaction.metadata.get(
                        "preliminary_classification",
                        False,
                    )
                )
                for interaction in interactions
            ),
            "score_is_final": all(
                not bool(
                    interaction.metadata.get(
                        "preliminary_score",
                        False,
                    )
                )
                for interaction in interactions
            ),
            "surface_is_approximate": True,
            "occupancy_denominator_pose_count": len(
                resolved_pose_identifiers
            ),
            "occupancy_minimum_group_score": float(
                _nonnegative_float(
                    occupancy_minimum_group_score,
                    name="occupancy minimum group score",
                )
            ),
            "atomic_interactions_are_deduplicated": True,
        }
    )

    return HydrophobicSummary(
        interaction_count=len(
            interactions
        ),
        atomic_pair_count=count_hydrophobic_atomic_pairs(
            interactions
        ),
        local_interaction_count=len(
            grouping.local_regions
        ),
        residue_count=len(
            grouping.residue_groups
        ),
        chain_count=len(
            grouping.chain_groups
        ),
        pose_count=len(
            resolved_pose_identifiers
        ),
        hotspot_count=len(
            grouping.hotspot_groups
        ),
        receptor_atom_count=(
            total_receptor_atom_count
        ),
        ligand_atom_count=(
            total_ligand_atom_count
        ),
        contacted_receptor_atom_count=len(
            receptor_atoms
        ),
        contacted_ligand_atom_count=len(
            ligand_atoms
        ),
        minimum_distance=(
            distance_statistics["minimum"]
        ),
        mean_distance=(
            distance_statistics["mean"]
        ),
        median_distance=(
            distance_statistics["median"]
        ),
        maximum_distance=(
            distance_statistics["maximum"]
        ),
        distance_standard_deviation=(
            distance_statistics[
                "standard_deviation"
            ]
        ),
        minimum_score=(
            score_statistics["minimum"]
        ),
        mean_score=(
            score_statistics["mean"]
        ),
        median_score=(
            score_statistics["median"]
        ),
        maximum_score=(
            score_statistics["maximum"]
        ),
        total_score=(
            score_statistics["total"]
            or np.float64(0.0)
        ),
        minimum_strength=(
            strength_statistics["minimum"]
        ),
        mean_strength=(
            strength_statistics["mean"]
        ),
        median_strength=(
            strength_statistics["median"]
        ),
        maximum_strength=(
            strength_statistics["maximum"]
        ),
        total_strength=(
            strength_statistics["total"]
            or np.float64(0.0)
        ),
        approximate_contact_area=(
            grouping.approximate_contact_area
        ),
        classification_counts=(
            classification_counts
        ),
        classification_fractions=(
            classification_fractions
        ),
        interaction_type_counts=(
            interaction_type_counts
        ),
        interaction_type_fractions=(
            interaction_type_fractions
        ),
        residue_contact_counts=(
            residue_contact_counts
        ),
        residue_scores=residue_scores,
        residue_minimum_distances=(
            residue_minimum_distances
        ),
        residue_contact_areas=(
            residue_contact_areas
        ),
        residue_pose_occupancy=(
            residue_pose_occupancy
        ),
        chain_pose_occupancy=(
            chain_pose_occupancy
        ),
        interaction_pose_occupancy=(
            interaction_pose_occupancy
        ),
        residue_identifiers=(
            hydrophobic_residue_identifiers(
                interactions
            )
        ),
        chain_identifiers=(
            hydrophobic_chain_identifiers(
                interactions
            )
        ),
        pose_identifiers=(
            resolved_pose_identifiers
        ),
        hotspot_residue_identifiers=(
            hotspot_identifiers
        ),
        metadata=summary_metadata,
    )


# -----------------------------------------------------------------------------
# Conversion to the base HydrophobicStatistics dataclass
# -----------------------------------------------------------------------------

def summary_to_hydrophobic_statistics(
    summary: HydrophobicSummary,
) -> HydrophobicStatistics:
    """
    Convert a complete summary into the base statistics dataclass.
    """

    if not isinstance(
        summary,
        HydrophobicSummary,
    ):
        raise TypeError(
            "summary must be a HydrophobicSummary."
        )

    classification_counts = dict(
        summary.classification_counts
    )

    type_counts = dict(
        summary.interaction_type_counts
    )

    return HydrophobicStatistics(
        interaction_count=(
            summary.interaction_count
        ),
        residue_count=summary.residue_count,
        receptor_atom_count=(
            summary.receptor_atom_count
        ),
        ligand_atom_count=(
            summary.ligand_atom_count
        ),
        very_strong_count=(
            classification_counts.get(
                HYDROPHOBIC_CLASS_VERY_STRONG,
                0,
            )
        ),
        strong_count=(
            classification_counts.get(
                HYDROPHOBIC_CLASS_STRONG,
                0,
            )
        ),
        moderate_count=(
            classification_counts.get(
                HYDROPHOBIC_CLASS_MODERATE,
                0,
            )
        ),
        weak_count=(
            classification_counts.get(
                HYDROPHOBIC_CLASS_WEAK,
                0,
            )
        ),
        marginal_count=(
            classification_counts.get(
                HYDROPHOBIC_CLASS_MARGINAL,
                0,
            )
        ),
        unknown_count=(
            classification_counts.get(
                HYDROPHOBIC_CLASS_UNKNOWN,
                0,
            )
        ),
        aliphatic_aliphatic_count=(
            type_counts.get(
                HYDROPHOBIC_TYPE_ALIPHATIC_ALIPHATIC,
                0,
            )
        ),
        aliphatic_aromatic_count=(
            type_counts.get(
                HYDROPHOBIC_TYPE_ALIPHATIC_AROMATIC,
                0,
            )
        ),
        aromatic_aliphatic_count=(
            type_counts.get(
                HYDROPHOBIC_TYPE_AROMATIC_ALIPHATIC,
                0,
            )
        ),
        aromatic_aromatic_count=(
            type_counts.get(
                HYDROPHOBIC_TYPE_AROMATIC_AROMATIC,
                0,
            )
        ),
        mixed_count=(
            type_counts.get(
                HYDROPHOBIC_TYPE_MIXED,
                0,
            )
        ),
        hotspot_count=summary.hotspot_count,
        minimum_distance=(
            summary.minimum_distance
        ),
        mean_distance=summary.mean_distance,
        median_distance=(
            summary.median_distance
        ),
        maximum_distance=(
            summary.maximum_distance
        ),
        distance_standard_deviation=(
            summary.distance_standard_deviation
        ),
        minimum_score=summary.minimum_score,
        mean_score=summary.mean_score,
        median_score=summary.median_score,
        maximum_score=summary.maximum_score,
        total_score=summary.total_score,
        minimum_strength=(
            summary.minimum_strength
        ),
        mean_strength=summary.mean_strength,
        maximum_strength=(
            summary.maximum_strength
        ),
        classification_counts=(
            summary.classification_counts
        ),
        interaction_type_counts=(
            summary.interaction_type_counts
        ),
        residue_interaction_counts=(
            summary.residue_contact_counts
        ),
        residue_scores=summary.residue_scores,
        metadata={
            **dict(summary.metadata),
            "atomic_pair_count": (
                summary.atomic_pair_count
            ),
            "local_interaction_count": (
                summary.local_interaction_count
            ),
            "chain_count": summary.chain_count,
            "pose_count": summary.pose_count,
            "contacted_receptor_atom_count": (
                summary.contacted_receptor_atom_count
            ),
            "contacted_ligand_atom_count": (
                summary.contacted_ligand_atom_count
            ),
            "approximate_contact_area": float(
                summary.approximate_contact_area
            ),
            "residue_pose_occupancy": {
                key: float(value)
                for key, value
                in summary.residue_pose_occupancy.items()
            },
        },
    )


# -----------------------------------------------------------------------------
# Interaction table
# -----------------------------------------------------------------------------

def hydrophobic_interaction_table(
    interactions: HydrophobicInteractionCollection,
    *,
    include_metadata: bool = (
        DEFAULT_SERIALIZE_INTERACTION_METADATA
    ),
    round_digits: int = (
        DEFAULT_HYDROPHOBIC_ROUND_DIGITS
    ),
    sort_by: Literal[
        "score",
        "distance",
        "residue",
        "pose",
    ] = DEFAULT_INTERACTION_TABLE_SORTING,
) -> HydrophobicSerializableTable:
    """
    Create a serializable row table of atomic interactions.
    """

    digits = _nonnegative_integer(
        round_digits,
        name="rounding digits",
    )

    interaction_tuple = (
        deduplicate_hydrophobic_interactions(
            interactions
        )
    )

    normalized_sorting = str(
        sort_by
    ).strip().lower()

    if normalized_sorting == "score":
        interaction_tuple = tuple(
            sorted(
                interaction_tuple,
                key=lambda interaction: (
                    -float(interaction.score),
                    float(interaction.distance),
                    interaction.interaction_identifier
                    or "",
                ),
            )
        )

    elif normalized_sorting == "distance":
        interaction_tuple = tuple(
            sorted(
                interaction_tuple,
                key=lambda interaction: (
                    float(interaction.distance),
                    -float(interaction.score),
                    interaction.interaction_identifier
                    or "",
                ),
            )
        )

    elif normalized_sorting == "residue":
        interaction_tuple = tuple(
            sorted(
                interaction_tuple,
                key=lambda interaction: (
                    interaction.receptor_residue_identifier
                    or "",
                    float(interaction.distance),
                ),
            )
        )

    elif normalized_sorting == "pose":
        interaction_tuple = tuple(
            sorted(
                interaction_tuple,
                key=lambda interaction: (
                    get_interaction_pose_identifier(
                        interaction
                    ),
                    -float(interaction.score),
                ),
            )
        )

    else:
        raise ValueError(
            "sort_by must be 'score', 'distance', "
            "'residue' or 'pose'."
        )

    rows: List[
        HydrophobicSerializableRow
    ] = []

    for interaction in interaction_tuple:
        row: HydrophobicSerializableRow = {
            "interaction_identifier": (
                interaction.interaction_identifier
            ),
            "pose_identifier": (
                get_interaction_pose_identifier(
                    interaction
                )
            ),
            "chain_identifier": (
                get_interaction_chain_identifier(
                    interaction,
                    default="chain-unknown",
                )
                or "chain-unknown"
            ),
            "residue_identifier": (
                interaction.receptor_residue_identifier
                or "residue-unknown"
            ),
            "receptor_atom_identifier": (
                interaction.receptor_atom_identifier
            ),
            "ligand_atom_identifier": (
                interaction.ligand_atom_identifier
            ),
            "receptor_atom_index": (
                interaction.receptor_atom_index
            ),
            "ligand_atom_index": (
                interaction.ligand_atom_index
            ),
            "distance": _round_optional_float(
                interaction.distance,
                digits=digits,
            ),
            "interaction_type": (
                interaction.interaction_type
            ),
            "classification": (
                interaction.classification
            ),
            "strength": _round_optional_float(
                interaction.strength,
                digits=digits,
            ),
            "score": _round_optional_float(
                interaction.score,
                digits=digits,
            ),
            "local_contact_count": (
                interaction.local_contact_count
            ),
            "polar_penalty": (
                _round_optional_float(
                    interaction.polar_penalty,
                    digits=digits,
                )
            ),
            "detection_method": (
                interaction.detection_method
            ),
            "direction": interaction.direction,
        }

        geometry = interaction.metadata.get(
            "geometry"
        )

        if isinstance(
            geometry,
            Mapping,
        ):
            row.update(
                {
                    "local_compaction": (
                        _round_optional_float(
                            geometry.get(
                                "local_compaction"
                            ),
                            digits=digits,
                        )
                    ),
                    "contact_density": (
                        _round_optional_float(
                            geometry.get(
                                "contact_density"
                            ),
                            digits=digits,
                        )
                    ),
                    "approximate_contact_area": (
                        _round_optional_float(
                            geometry.get(
                                "approximate_contact_area"
                            ),
                            digits=digits,
                        )
                    ),
                    "local_centroid_distance": (
                        _round_optional_float(
                            geometry.get(
                                "local_centroid_distance"
                            ),
                            digits=digits,
                        )
                    ),
                }
            )

        else:
            row.update(
                {
                    "local_compaction": None,
                    "contact_density": None,
                    "approximate_contact_area": (
                        _round_optional_float(
                            approximate_interaction_contact_area(
                                interaction
                            ),
                            digits=digits,
                        )
                    ),
                    "local_centroid_distance": None,
                }
            )

        if include_metadata:
            row["metadata"] = dict(
                interaction.metadata
            )

        rows.append(row)

    return tuple(rows)


# -----------------------------------------------------------------------------
# Residue table
# -----------------------------------------------------------------------------

def hydrophobic_residue_table(
    interactions: HydrophobicInteractionCollection,
    *,
    pose_identifiers: Optional[Iterable[str]] = None,
    include_metadata: bool = (
        DEFAULT_SERIALIZE_GROUP_METADATA
    ),
    round_digits: int = (
        DEFAULT_HYDROPHOBIC_ROUND_DIGITS
    ),
    sort_by: Literal[
        "score",
        "distance",
        "count",
        "occupancy",
        "identifier",
    ] = DEFAULT_RESIDUE_TABLE_SORTING,
) -> HydrophobicSerializableTable:
    """
    Create a serializable residue-level summary table.
    """

    digits = _nonnegative_integer(
        round_digits,
        name="rounding digits",
    )

    groups = list(
        group_hydrophobic_interactions_by_residue(
            interactions,
            identify_hotspots=True,
            sort_by="identifier",
        )
    )

    occupancy = (
        calculate_hydrophobic_residue_pose_occupancy(
            interactions,
            pose_identifiers=pose_identifiers,
        )
    )

    normalized_sorting = str(
        sort_by
    ).strip().lower()

    if normalized_sorting == "score":
        groups.sort(
            key=lambda group: (
                -float(group.group_score or 0.0),
                float(
                    group.minimum_distance
                    if group.minimum_distance is not None
                    else np.inf
                ),
            )
        )

    elif normalized_sorting == "distance":
        groups.sort(
            key=lambda group: (
                float(
                    group.minimum_distance
                    if group.minimum_distance is not None
                    else np.inf
                ),
                -float(group.group_score or 0.0),
            )
        )

    elif normalized_sorting == "count":
        groups.sort(
            key=lambda group: (
                -group.interaction_count,
                -float(group.group_score or 0.0),
            )
        )

    elif normalized_sorting == "occupancy":
        groups.sort(
            key=lambda group: (
                -float(
                    occupancy.get(
                        group.residue_identifier
                        or "residue-unknown",
                        0.0,
                    )
                ),
                -float(group.group_score or 0.0),
            )
        )

    elif normalized_sorting == "identifier":
        groups.sort(
            key=lambda group: (
                group.residue_identifier
                or ""
            )
        )

    else:
        raise ValueError(
            "sort_by must be 'score', 'distance', 'count', "
            "'occupancy' or 'identifier'."
        )

    rows: List[
        HydrophobicSerializableRow
    ] = []

    for group in groups:
        residue_identifier = (
            group.residue_identifier
            or "residue-unknown"
        )

        closest = (
            select_closest_hydrophobic_interaction(
                group.interactions
            )
        )

        representative = (
            select_highest_scoring_hydrophobic_interaction(
                group.interactions
            )
        )

        unique_receptor_atoms, unique_ligand_atoms = (
            _unique_interaction_atoms(
                group.interactions
            )
        )

        assessment = (
            assess_hydrophobic_interaction_strength(
                group.interactions,
                metadata={
                    "assessment_level": "residue_table",
                },
            )
        )

        row: HydrophobicSerializableRow = {
            "residue_identifier": residue_identifier,
            "chain_identifier": (
                str(
                    group.metadata.get(
                        "chain_identifier",
                        "",
                    )
                ).strip()
                or "chain-unknown"
            ),
            "pose_identifiers": list(
                group.metadata.get(
                    "pose_identifiers",
                    (),
                )
            ),
            "interaction_count": (
                group.interaction_count
            ),
            "atomic_pair_count": (
                group.interaction_count
            ),
            "unique_receptor_atom_count": len(
                unique_receptor_atoms
            ),
            "unique_ligand_atom_count": len(
                unique_ligand_atoms
            ),
            "minimum_distance": (
                _round_optional_float(
                    group.minimum_distance,
                    digits=digits,
                )
            ),
            "mean_distance": (
                _round_optional_float(
                    _safe_mean(
                        [
                            interaction.distance
                            for interaction
                            in group.interactions
                        ]
                    ),
                    digits=digits,
                )
            ),
            "maximum_distance": (
                _round_optional_float(
                    _safe_maximum(
                        [
                            interaction.distance
                            for interaction
                            in group.interactions
                        ]
                    ),
                    digits=digits,
                )
            ),
            "group_score": (
                _round_optional_float(
                    group.group_score,
                    digits=digits,
                )
            ),
            "final_strength": (
                _round_optional_float(
                    assessment.strength,
                    digits=digits,
                )
            ),
            "final_classification": (
                assessment.classification
            ),
            "predominant_interaction_type": (
                assessment.interaction_type
            ),
            "approximate_contact_area": (
                _round_optional_float(
                    approximate_hydrophobic_surface_area(
                        group.interactions
                    ),
                    digits=digits,
                )
            ),
            "contact_density": (
                _round_optional_float(
                    calculate_hydrophobic_density_component(
                        group.interactions
                    ),
                    digits=digits,
                )
            ),
            "pose_occupancy": (
                _round_optional_float(
                    occupancy.get(
                        residue_identifier,
                        0.0,
                    ),
                    digits=digits,
                )
            ),
            "is_hotspot": bool(
                group.metadata.get(
                    "is_hotspot",
                    False,
                )
            ),
            "closest_interaction_identifier": (
                None
                if closest is None
                else closest.interaction_identifier
            ),
            "representative_interaction_identifier": (
                None
                if representative is None
                else representative.interaction_identifier
            ),
        }

        if include_metadata:
            row["metadata"] = dict(
                group.metadata
            )

        rows.append(row)

    return tuple(rows)


# -----------------------------------------------------------------------------
# Pose table
# -----------------------------------------------------------------------------

def hydrophobic_pose_table(
    interactions: HydrophobicInteractionCollection,
    *,
    include_metadata: bool = (
        DEFAULT_SERIALIZE_GROUP_METADATA
    ),
    round_digits: int = (
        DEFAULT_HYDROPHOBIC_ROUND_DIGITS
    ),
    sort_by: Literal[
        "pose",
        "score",
        "count",
        "area",
    ] = DEFAULT_POSE_TABLE_SORTING,
) -> HydrophobicSerializableTable:
    """
    Create a serializable table with one row per ligand pose.
    """

    digits = _nonnegative_integer(
        round_digits,
        name="rounding digits",
    )

    pose_groups = list(
        group_hydrophobic_interactions_by_pose(
            interactions
        )
    )

    normalized_sorting = str(
        sort_by
    ).strip().lower()

    if normalized_sorting == "pose":
        pose_groups.sort(
            key=lambda group: (
                group.pose_identifier
            )
        )

    elif normalized_sorting == "score":
        pose_groups.sort(
            key=lambda group: (
                -float(group.total_score),
                group.pose_identifier,
            )
        )

    elif normalized_sorting == "count":
        pose_groups.sort(
            key=lambda group: (
                -group.interaction_count,
                -float(group.total_score),
            )
        )

    elif normalized_sorting == "area":
        pose_groups.sort(
            key=lambda group: (
                -float(
                    group.approximate_contact_area
                ),
                -float(group.total_score),
            )
        )

    else:
        raise ValueError(
            "sort_by must be 'pose', 'score', "
            "'count' or 'area'."
        )

    rows: List[
        HydrophobicSerializableRow
    ] = []

    for pose_group in pose_groups:
        distance_stats = (
            hydrophobic_distance_statistics(
                pose_group.interactions
            )
        )

        score_stats = (
            hydrophobic_score_statistics(
                pose_group.interactions
            )
        )

        classification_counts = (
            hydrophobic_classification_distribution(
                pose_group.interactions
            )
        )

        type_counts = (
            hydrophobic_interaction_type_distribution(
                pose_group.interactions
            )
        )

        row: HydrophobicSerializableRow = {
            "pose_identifier": (
                pose_group.pose_identifier
            ),
            "interaction_count": (
                pose_group.interaction_count
            ),
            "atomic_pair_count": (
                pose_group.interaction_count
            ),
            "residue_count": (
                pose_group.residue_count
            ),
            "chain_count": len(
                pose_group.chain_groups
            ),
            "local_region_count": len(
                pose_group.local_regions
            ),
            "hotspot_count": (
                pose_group.hotspot_count
            ),
            "minimum_distance": (
                _round_optional_float(
                    distance_stats["minimum"],
                    digits=digits,
                )
            ),
            "mean_distance": (
                _round_optional_float(
                    distance_stats["mean"],
                    digits=digits,
                )
            ),
            "maximum_distance": (
                _round_optional_float(
                    distance_stats["maximum"],
                    digits=digits,
                )
            ),
            "mean_score": (
                _round_optional_float(
                    score_stats["mean"],
                    digits=digits,
                )
            ),
            "total_score": (
                _round_optional_float(
                    score_stats["total"],
                    digits=digits,
                )
            ),
            "approximate_contact_area": (
                _round_optional_float(
                    pose_group.approximate_contact_area,
                    digits=digits,
                )
            ),
            "classification_counts": dict(
                classification_counts
            ),
            "interaction_type_counts": dict(
                type_counts
            ),
            "residue_identifiers": [
                group.residue_identifier
                or "residue-unknown"
                for group
                in pose_group.residue_groups
            ],
        }

        if include_metadata:
            row["metadata"] = dict(
                pose_group.metadata
            )

        rows.append(row)

    return tuple(rows)


# -----------------------------------------------------------------------------
# Hotspot table
# -----------------------------------------------------------------------------

def hydrophobic_hotspot_table(
    interactions: HydrophobicInteractionCollection,
    *,
    pose_identifiers: Optional[Iterable[str]] = None,
    include_metadata: bool = (
        DEFAULT_SERIALIZE_GROUP_METADATA
    ),
    round_digits: int = (
        DEFAULT_HYDROPHOBIC_ROUND_DIGITS
    ),
) -> HydrophobicSerializableTable:
    """
    Create a serializable table containing only hotspot residues.
    """

    residue_rows = hydrophobic_residue_table(
        interactions,
        pose_identifiers=pose_identifiers,
        include_metadata=include_metadata,
        round_digits=round_digits,
        sort_by="score",
    )

    return tuple(
        row
        for row in residue_rows
        if bool(
            row.get(
                "is_hotspot",
                False,
            )
        )
    )


# -----------------------------------------------------------------------------
# Local-region table
# -----------------------------------------------------------------------------

def hydrophobic_local_region_table(
    interactions: HydrophobicInteractionCollection,
    *,
    grouping_distance: Optional[Number] = None,
    include_metadata: bool = (
        DEFAULT_SERIALIZE_GROUP_METADATA
    ),
    round_digits: int = (
        DEFAULT_HYDROPHOBIC_ROUND_DIGITS
    ),
) -> HydrophobicSerializableTable:
    """
    Create a serializable table of local contact regions.
    """

    digits = _nonnegative_integer(
        round_digits,
        name="rounding digits",
    )

    regions = (
        cluster_hydrophobic_local_regions(
            interactions,
            grouping_distance=grouping_distance,
            identify_hotspots=True,
            sort_by="score",
        )
    )

    rows: List[
        HydrophobicSerializableRow
    ] = []

    for region in regions:
        assessment = (
            classify_hydrophobic_local_region(
                region
            )
        )

        row: HydrophobicSerializableRow = {
            "region_identifier": (
                region.region_identifier
            ),
            "pose_identifier": (
                region.pose_identifier
            ),
            "chain_identifiers": list(
                region.chain_identifiers
            ),
            "residue_identifiers": list(
                region.residue_identifiers
            ),
            "interaction_count": (
                region.interaction_count
            ),
            "atomic_pair_count": (
                region.interaction_count
            ),
            "residue_count": (
                region.residue_count
            ),
            "unique_receptor_atom_count": (
                region.unique_receptor_atom_count
            ),
            "unique_ligand_atom_count": (
                region.unique_ligand_atom_count
            ),
            "minimum_distance": (
                _round_optional_float(
                    region.minimum_distance,
                    digits=digits,
                )
            ),
            "total_score": (
                _round_optional_float(
                    region.total_score,
                    digits=digits,
                )
            ),
            "final_strength": (
                _round_optional_float(
                    assessment.strength,
                    digits=digits,
                )
            ),
            "final_classification": (
                assessment.classification
            ),
            "predominant_interaction_type": (
                assessment.interaction_type
            ),
            "contact_density": (
                _round_optional_float(
                    region.contact_density,
                    digits=digits,
                )
            ),
            "approximate_contact_area": (
                _round_optional_float(
                    region.approximate_contact_area,
                    digits=digits,
                )
            ),
            "is_hotspot": bool(
                region.is_hotspot
            ),
            "closest_interaction_identifier": (
                None
                if region.closest_interaction is None
                else region.closest_interaction.interaction_identifier
            ),
            "representative_interaction_identifier": (
                None
                if region.representative_interaction is None
                else region.representative_interaction.interaction_identifier
            ),
        }

        if include_metadata:
            row["metadata"] = dict(
                region.metadata
            )

        rows.append(row)

    return tuple(rows)


# -----------------------------------------------------------------------------
# Distribution tables
# -----------------------------------------------------------------------------

def hydrophobic_classification_table(
    interactions: HydrophobicInteractionCollection,
    *,
    round_digits: int = (
        DEFAULT_HYDROPHOBIC_ROUND_DIGITS
    ),
) -> HydrophobicSerializableTable:
    """Create a serializable strength-distribution table."""

    counts = hydrophobic_classification_distribution(
        interactions
    )

    fractions = (
        hydrophobic_classification_fraction_distribution(
            interactions
        )
    )

    ordered_classifications = (
        HYDROPHOBIC_CLASS_VERY_STRONG,
        HYDROPHOBIC_CLASS_STRONG,
        HYDROPHOBIC_CLASS_MODERATE,
        HYDROPHOBIC_CLASS_WEAK,
        HYDROPHOBIC_CLASS_MARGINAL,
        HYDROPHOBIC_CLASS_UNKNOWN,
    )

    return tuple(
        {
            "classification": classification,
            "count": counts.get(
                classification,
                0,
            ),
            "fraction": _round_optional_float(
                fractions.get(
                    classification,
                    0.0,
                ),
                digits=round_digits,
            ),
        }
        for classification
        in ordered_classifications
    )


def hydrophobic_type_table(
    interactions: HydrophobicInteractionCollection,
    *,
    round_digits: int = (
        DEFAULT_HYDROPHOBIC_ROUND_DIGITS
    ),
) -> HydrophobicSerializableTable:
    """Create a serializable contact-type distribution table."""

    counts = hydrophobic_interaction_type_distribution(
        interactions
    )

    fractions = (
        hydrophobic_type_fraction_distribution(
            interactions
        )
    )

    ordered_types = (
        HYDROPHOBIC_TYPE_ALIPHATIC_ALIPHATIC,
        HYDROPHOBIC_TYPE_ALIPHATIC_AROMATIC,
        HYDROPHOBIC_TYPE_AROMATIC_ALIPHATIC,
        HYDROPHOBIC_TYPE_AROMATIC_AROMATIC,
        HYDROPHOBIC_TYPE_MIXED,
        HYDROPHOBIC_TYPE_UNKNOWN,
    )

    return tuple(
        {
            "interaction_type": interaction_type,
            "count": counts.get(
                interaction_type,
                0,
            ),
            "fraction": _round_optional_float(
                fractions.get(
                    interaction_type,
                    0.0,
                ),
                digits=round_digits,
            ),
            "pi_stacking_assigned": False,
        }
        for interaction_type
        in ordered_types
    )


def hydrophobic_occupancy_table(
    interactions: HydrophobicInteractionCollection,
    *,
    pose_identifiers: Optional[Iterable[str]] = None,
    minimum_group_score: Number = (
        DEFAULT_HYDROPHOBIC_OCCUPANCY_THRESHOLD
    ),
    round_digits: int = (
        DEFAULT_HYDROPHOBIC_ROUND_DIGITS
    ),
) -> HydrophobicSerializableTable:
    """Create a serializable residue occupancy table."""

    occupancy = (
        calculate_hydrophobic_residue_pose_occupancy(
            interactions,
            pose_identifiers=pose_identifiers,
            minimum_group_score=(
                minimum_group_score
            ),
        )
    )

    presence = (
        hydrophobic_residue_pose_presence(
            interactions
        )
    )

    return tuple(
        {
            "residue_identifier": residue_identifier,
            "occupancy": _round_optional_float(
                occupancy_value,
                digits=round_digits,
            ),
            "pose_count_with_contact": len(
                presence.get(
                    residue_identifier,
                    (),
                )
            ),
            "poses_with_contact": list(
                presence.get(
                    residue_identifier,
                    (),
                )
            ),
        }
        for residue_identifier, occupancy_value
        in sorted(
            occupancy.items(),
            key=lambda item: (
                -float(item[1]),
                item[0],
            ),
        )
    )


# -----------------------------------------------------------------------------
# Complete table bundle
# -----------------------------------------------------------------------------

@dataclass(
    frozen=True,
    slots=True,
)
class HydrophobicSerializableTables:
    """
    Collection of directly serializable hydrophobic result tables.
    """

    summary: HydrophobicSerializableTable = field(
        default_factory=tuple
    )

    interactions: HydrophobicSerializableTable = field(
        default_factory=tuple
    )

    residues: HydrophobicSerializableTable = field(
        default_factory=tuple
    )

    poses: HydrophobicSerializableTable = field(
        default_factory=tuple
    )

    local_regions: HydrophobicSerializableTable = field(
        default_factory=tuple
    )

    hotspots: HydrophobicSerializableTable = field(
        default_factory=tuple
    )

    classification_distribution: HydrophobicSerializableTable = field(
        default_factory=tuple
    )

    type_distribution: HydrophobicSerializableTable = field(
        default_factory=tuple
    )

    pose_occupancy: HydrophobicSerializableTable = field(
        default_factory=tuple
    )

    metadata: Mapping[str, Any] = field(
        default_factory=lambda: _EMPTY_METADATA,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        """Freeze all tables."""

        table_fields = (
            "summary",
            "interactions",
            "residues",
            "poses",
            "local_regions",
            "hotspots",
            "classification_distribution",
            "type_distribution",
            "pose_occupancy",
        )

        for field_name in table_fields:
            rows = tuple(
                dict(row)
                for row
                in getattr(
                    self,
                    field_name,
                )
            )

            object.__setattr__(
                self,
                field_name,
                rows,
            )

        object.__setattr__(
            self,
            "metadata",
            _freeze_metadata(
                self.metadata
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize all tables as plain lists of dictionaries."""

        return {
            "summary": [
                dict(row)
                for row in self.summary
            ],
            "interactions": [
                dict(row)
                for row in self.interactions
            ],
            "residues": [
                dict(row)
                for row in self.residues
            ],
            "poses": [
                dict(row)
                for row in self.poses
            ],
            "local_regions": [
                dict(row)
                for row in self.local_regions
            ],
            "hotspots": [
                dict(row)
                for row in self.hotspots
            ],
            "classification_distribution": [
                dict(row)
                for row
                in self.classification_distribution
            ],
            "type_distribution": [
                dict(row)
                for row in self.type_distribution
            ],
            "pose_occupancy": [
                dict(row)
                for row in self.pose_occupancy
            ],
            "metadata": dict(
                self.metadata
            ),
        }


def build_hydrophobic_serializable_tables(
    source: Union[
        HydrophobicAnalysisResult,
        HydrophobicGroupingResult,
        HydrophobicDetectionResult,
        HydrophobicInteractionCollection,
    ],
    *,
    grouping_distance: Optional[Number] = None,
    pose_identifiers: Optional[Iterable[str]] = None,
    occupancy_minimum_group_score: Number = (
        DEFAULT_HYDROPHOBIC_OCCUPANCY_THRESHOLD
    ),
    include_interaction_metadata: bool = (
        DEFAULT_SERIALIZE_INTERACTION_METADATA
    ),
    include_group_metadata: bool = (
        DEFAULT_SERIALIZE_GROUP_METADATA
    ),
    round_digits: int = (
        DEFAULT_HYDROPHOBIC_ROUND_DIGITS
    ),
) -> HydrophobicSerializableTables:
    """
    Build all standard serializable hydrophobic tables.
    """

    interactions = _resolve_statistics_interactions(
        source
    )

    summary = calculate_hydrophobic_summary(
        source,
        grouping_distance=grouping_distance,
        pose_identifiers=pose_identifiers,
        occupancy_minimum_group_score=(
            occupancy_minimum_group_score
        ),
    )

    summary_row = summary.to_dict(
        round_digits=round_digits
    )

    return HydrophobicSerializableTables(
        summary=(
            summary_row,
        ),
        interactions=hydrophobic_interaction_table(
            interactions,
            include_metadata=(
                include_interaction_metadata
            ),
            round_digits=round_digits,
        ),
        residues=hydrophobic_residue_table(
            interactions,
            pose_identifiers=pose_identifiers,
            include_metadata=(
                include_group_metadata
            ),
            round_digits=round_digits,
        ),
        poses=hydrophobic_pose_table(
            interactions,
            include_metadata=(
                include_group_metadata
            ),
            round_digits=round_digits,
        ),
        local_regions=(
            hydrophobic_local_region_table(
                interactions,
                grouping_distance=(
                    grouping_distance
                ),
                include_metadata=(
                    include_group_metadata
                ),
                round_digits=round_digits,
            )
        ),
        hotspots=hydrophobic_hotspot_table(
            interactions,
            pose_identifiers=pose_identifiers,
            include_metadata=(
                include_group_metadata
            ),
            round_digits=round_digits,
        ),
        classification_distribution=(
            hydrophobic_classification_table(
                interactions,
                round_digits=round_digits,
            )
        ),
        type_distribution=(
            hydrophobic_type_table(
                interactions,
                round_digits=round_digits,
            )
        ),
        pose_occupancy=(
            hydrophobic_occupancy_table(
                interactions,
                pose_identifiers=pose_identifiers,
                minimum_group_score=(
                    occupancy_minimum_group_score
                ),
                round_digits=round_digits,
            )
        ),
        metadata={
            "table_format": (
                "tuple_of_serializable_dictionaries"
            ),
            "round_digits": int(
                round_digits
            ),
            "interaction_metadata_included": bool(
                include_interaction_metadata
            ),
            "group_metadata_included": bool(
                include_group_metadata
            ),
        },
    )


# -----------------------------------------------------------------------------
# Plain-text summary
# -----------------------------------------------------------------------------

def format_hydrophobic_summary(
    summary: HydrophobicSummary,
    *,
    include_distributions: bool = True,
    include_hotspots: bool = True,
    include_occupancy: bool = True,
    round_digits: int = 3,
) -> str:
    """
    Format a compact human-readable hydrophobic summary.
    """

    if not isinstance(
        summary,
        HydrophobicSummary,
    ):
        raise TypeError(
            "summary must be a HydrophobicSummary."
        )

    digits = _nonnegative_integer(
        round_digits,
        name="rounding digits",
    )

    lines = [
        "Hydrophobic interaction summary",
        "-------------------------------",
        (
            f"Interactions: "
            f"{summary.interaction_count}"
        ),
        (
            f"Atomic pairs: "
            f"{summary.atomic_pair_count}"
        ),
        (
            f"Local regions: "
            f"{summary.local_interaction_count}"
        ),
        (
            f"Residues: "
            f"{summary.residue_count}"
        ),
        (
            f"Chains: "
            f"{summary.chain_count}"
        ),
        (
            f"Poses: "
            f"{summary.pose_count}"
        ),
        (
            f"Hotspots: "
            f"{summary.hotspot_count}"
        ),
        (
            "Distance "
            f"(min/mean/max): "
            f"{_round_optional_float(summary.minimum_distance, digits=digits)} / "
            f"{_round_optional_float(summary.mean_distance, digits=digits)} / "
            f"{_round_optional_float(summary.maximum_distance, digits=digits)}"
        ),
        (
            f"Total score: "
            f"{_round_optional_float(summary.total_score, digits=digits)}"
        ),
        (
            f"Approximate contact area: "
            f"{_round_optional_float(summary.approximate_contact_area, digits=digits)}"
        ),
    ]

    if include_distributions:
        classification_text = ", ".join(
            f"{key}={value}"
            for key, value
            in summary.classification_counts.items()
        )

        type_text = ", ".join(
            f"{key}={value}"
            for key, value
            in summary.interaction_type_counts.items()
        )

        lines.extend(
            [
                (
                    "Strength distribution: "
                    + classification_text
                ),
                (
                    "Type distribution: "
                    + type_text
                ),
            ]
        )

    if include_hotspots:
        hotspot_text = (
            ", ".join(
                summary.hotspot_residue_identifiers
            )
            if summary.hotspot_residue_identifiers
            else "none"
        )

        lines.append(
            f"Hotspot residues: {hotspot_text}"
        )

    if include_occupancy:
        occupied_residues = sorted(
            summary.residue_pose_occupancy.items(),
            key=lambda item: (
                -float(item[1]),
                item[0],
            ),
        )

        occupancy_text = (
            ", ".join(
                (
                    f"{residue}="
                    f"{round(float(occupancy), digits)}"
                )
                for residue, occupancy
                in occupied_residues
            )
            if occupied_residues
            else "none"
        )

        lines.append(
            "Residue pose occupancy: "
            + occupancy_text
        )

    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Adding complete statistics to an analysis result
# -----------------------------------------------------------------------------

def add_hydrophobic_statistics_to_result(
    result: HydrophobicAnalysisResult,
    *,
    grouping_distance: Optional[Number] = None,
    pose_identifiers: Optional[Iterable[str]] = None,
    occupancy_minimum_group_score: Number = (
        DEFAULT_HYDROPHOBIC_OCCUPANCY_THRESHOLD
    ),
    include_serializable_tables: bool = True,
    round_digits: int = (
        DEFAULT_HYDROPHOBIC_ROUND_DIGITS
    ),
) -> HydrophobicAnalysisResult:
    """
    Return a new analysis result with final statistics and summaries.
    """

    if not isinstance(
        result,
        HydrophobicAnalysisResult,
    ):
        raise TypeError(
            "result must be a HydrophobicAnalysisResult."
        )

    resolved_grouping_distance = (
        result.grouping_distance
        if grouping_distance is None
        else _positive_float(
            grouping_distance,
            name="grouping distance",
        )
    )

    summary = calculate_hydrophobic_summary(
        result,
        grouping_distance=(
            resolved_grouping_distance
        ),
        pose_identifiers=pose_identifiers,
        occupancy_minimum_group_score=(
            occupancy_minimum_group_score
        ),
    )

    statistics = (
        summary_to_hydrophobic_statistics(
            summary
        )
    )

    grouping = group_hydrophobic_interactions(
        result.interactions,
        grouping_distance=(
            resolved_grouping_distance
        ),
        identify_hotspots=True,
    )

    updated_metadata = dict(
        result.metadata
    )

    updated_metadata.update(
        {
            "analysis_stage": (
                "complete_statistics_and_summaries"
            ),
            "statistics_completed": True,
            "summary": summary.to_dict(
                round_digits=round_digits
            ),
            "total_interactions": (
                summary.interaction_count
            ),
            "total_atomic_pairs": (
                summary.atomic_pair_count
            ),
            "residue_count": (
                summary.residue_count
            ),
            "chain_count": (
                summary.chain_count
            ),
            "pose_count": summary.pose_count,
            "hotspot_count": (
                summary.hotspot_count
            ),
            "total_score": float(
                summary.total_score
            ),
            "approximate_contact_area": float(
                summary.approximate_contact_area
            ),
            "residue_pose_occupancy": {
                key: float(value)
                for key, value
                in summary.residue_pose_occupancy.items()
            },
        }
    )

    if include_serializable_tables:
        tables = (
            build_hydrophobic_serializable_tables(
                result,
                grouping_distance=(
                    resolved_grouping_distance
                ),
                pose_identifiers=pose_identifiers,
                occupancy_minimum_group_score=(
                    occupancy_minimum_group_score
                ),
                round_digits=round_digits,
            )
        )

        updated_metadata[
            "serializable_tables"
        ] = tables.to_dict()

    return HydrophobicAnalysisResult(
        interactions=result.interactions,
        residue_groups=(
            grouping.residue_groups
        ),
        receptor_hydrophobic_atoms=(
            result.receptor_hydrophobic_atoms
        ),
        ligand_hydrophobic_atoms=(
            result.ligand_hydrophobic_atoms
        ),
        receptor_atoms=result.receptor_atoms,
        ligand_atoms=result.ligand_atoms,
        minimum_distance=result.minimum_distance,
        maximum_distance=result.maximum_distance,
        grouping_distance=(
            resolved_grouping_distance
        ),
        statistics=statistics,
        analysis_identifier=(
            result.analysis_identifier
        ),
        receptor_identifier=(
            result.receptor_identifier
        ),
        ligand_identifier=(
            result.ligand_identifier
        ),
        metadata=updated_metadata,
    )


# -----------------------------------------------------------------------------
# Complete workflow through Section 10
# -----------------------------------------------------------------------------

def analyze_hydrophobic_interactions(
    receptor: Union[
        Any,
        HydrophobicAtomCollections,
    ],
    ligand: Optional[Any] = None,
    *,
    minimum_distance: Optional[Number] = None,
    maximum_distance: Optional[Number] = None,
    grouping_distance: Optional[Number] = None,
    local_radius: Optional[Number] = None,
    weights: Optional[Mapping[str, Number]] = None,
    preparation_options: Optional[
        Mapping[str, Any]
    ] = None,
    identify_hotspots: bool = True,
    pose_identifiers: Optional[Iterable[str]] = None,
    occupancy_minimum_group_score: Number = (
        DEFAULT_HYDROPHOBIC_OCCUPANCY_THRESHOLD
    ),
    include_serializable_tables: bool = True,
    receptor_identifier: Optional[str] = None,
    ligand_identifier: Optional[str] = None,
    analysis_identifier: Optional[str] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> HydrophobicAnalysisResult:
    """
    Run detection, grouping, classification and statistical summaries.
    """

    classified_result = (
        analyze_and_classify_hydrophobic_interactions(
            receptor,
            ligand,
            minimum_distance=minimum_distance,
            maximum_distance=maximum_distance,
            grouping_distance=grouping_distance,
            local_radius=local_radius,
            weights=weights,
            preparation_options=(
                preparation_options
            ),
            identify_hotspots=identify_hotspots,
            receptor_identifier=(
                receptor_identifier
            ),
            ligand_identifier=(
                ligand_identifier
            ),
            analysis_identifier=(
                analysis_identifier
            ),
            metadata=metadata,
        )
    )

    return add_hydrophobic_statistics_to_result(
        classified_result,
        grouping_distance=grouping_distance,
        pose_identifiers=pose_identifiers,
        occupancy_minimum_group_score=(
            occupancy_minimum_group_score
        ),
        include_serializable_tables=(
            include_serializable_tables
        ),
    )


# -----------------------------------------------------------------------------
# Empty statistical objects
# -----------------------------------------------------------------------------

_EMPTY_HYDROPHOBIC_SUMMARY: Final[
    HydrophobicSummary
] = HydrophobicSummary()

_EMPTY_HYDROPHOBIC_SERIALIZABLE_TABLES: Final[
    HydrophobicSerializableTables
] = HydrophobicSerializableTables()


# -----------------------------------------------------------------------------
# Section 10 public names
# -----------------------------------------------------------------------------

_SECTION_10_PUBLIC_NAMES: Final[Tuple[str, ...]] = (
    # Aliases
    "HydrophobicSerializableRow",
    "HydrophobicSerializableTable",
    "HydrophobicDistribution",
    "HydrophobicOccupancyMap",

    # Dataclasses
    "HydrophobicSummary",
    "HydrophobicSerializableTables",

    # Atomic and distance statistics
    "count_hydrophobic_atomic_pairs",
    "count_hydrophobic_interactions",
    "hydrophobic_distance_statistics",
    "hydrophobic_score_statistics",
    "hydrophobic_strength_statistics",

    # Distributions
    "hydrophobic_classification_distribution",
    "hydrophobic_interaction_type_distribution",
    "hydrophobic_classification_fraction_distribution",
    "hydrophobic_type_fraction_distribution",

    # Residue summaries
    "hydrophobic_score_by_residue",
    "hydrophobic_contact_count_by_residue",
    "hydrophobic_minimum_distance_by_residue",
    "hydrophobic_residue_identifiers",
    "hydrophobic_chain_identifiers",

    # Pose occupancy
    "hydrophobic_pose_identifiers",
    "hydrophobic_residue_pose_presence",
    "calculate_hydrophobic_residue_pose_occupancy",
    "calculate_hydrophobic_chain_pose_occupancy",
    "calculate_hydrophobic_interaction_pose_occupancy",

    # Hotspots
    "summarize_hydrophobic_hotspots",

    # Complete summary
    "calculate_hydrophobic_summary",
    "summary_to_hydrophobic_statistics",

    # Serializable tables
    "hydrophobic_interaction_table",
    "hydrophobic_residue_table",
    "hydrophobic_pose_table",
    "hydrophobic_hotspot_table",
    "hydrophobic_local_region_table",
    "hydrophobic_classification_table",
    "hydrophobic_type_table",
    "hydrophobic_occupancy_table",
    "build_hydrophobic_serializable_tables",

    # Text formatting
    "format_hydrophobic_summary",

    # Result-level workflows
    "add_hydrophobic_statistics_to_result",
    "analyze_hydrophobic_interactions",
)

for public_name in _SECTION_10_PUBLIC_NAMES:
    if public_name not in __all__:
        __all__.append(public_name)


# =============================================================================
# End of Section 10
# =============================================================================


# =============================================================================
# Section 11 — DockModel integration
# =============================================================================


# -----------------------------------------------------------------------------
# DockModel integration aliases
# -----------------------------------------------------------------------------

DockModelLike: TypeAlias = Any

HydrophobicAttachmentMode: TypeAlias = Literal[
    "append",
    "replace",
    "merge",
]

HydrophobicDockModelResultCollection: TypeAlias = Tuple[
    HydrophobicAnalysisResult,
    ...,
]


# -----------------------------------------------------------------------------
# DockModel integration defaults
# -----------------------------------------------------------------------------

DEFAULT_HYDROPHOBIC_ATTACHMENT_MODE: Final[
    HydrophobicAttachmentMode
] = "merge"

DEFAULT_PRESERVE_PREVIOUS_HYDROPHOBIC_RESULTS: Final[bool] = True
DEFAULT_UPDATE_DOCK_MODEL_STATISTICS: Final[bool] = True
DEFAULT_UPDATE_DOCK_MODEL_SCORE: Final[bool] = True
DEFAULT_SERIALIZE_DOCK_MODEL_RESULTS: Final[bool] = True

DEFAULT_HYDROPHOBIC_SCORE_ATTRIBUTE_NAMES: Final[
    Tuple[str, ...]
] = (
    "hydrophobic_score",
    "hydrophobicScore",
)

DEFAULT_HYDROPHOBIC_STATISTICS_ATTRIBUTE_NAMES: Final[
    Tuple[str, ...]
] = (
    "hydrophobic_statistics",
    "hydrophobicStatistics",
)

DEFAULT_DOCK_MODEL_SCORE_ATTRIBUTE_NAMES: Final[
    Tuple[str, ...]
] = (
    "score",
    "total_score",
    "interaction_score",
)

DEFAULT_DOCK_MODEL_POSE_ATTRIBUTE_NAMES: Final[
    Tuple[str, ...]
] = (
    "pose_identifier",
    "pose_id",
    "pose",
    "model_id",
    "id",
    "name",
)

DEFAULT_DOCK_MODEL_RECEPTOR_ATTRIBUTE_NAMES: Final[
    Tuple[str, ...]
] = (
    "receptor",
    "receptor_model",
    "protein",
    "target",
    "macromolecule",
)

DEFAULT_DOCK_MODEL_LIGAND_ATTRIBUTE_NAMES: Final[
    Tuple[str, ...]
] = (
    "ligand",
    "ligand_model",
    "docked_ligand",
    "compound",
    "molecule",
)

DEFAULT_DOCK_MODEL_HYDROPHOBIC_ATTRIBUTE: Final[str] = (
    "hydrophobic"
)

HYDROPHOBIC_DOCK_MODEL_RESULT_SCHEMA_VERSION: Final[str] = (
    "1.0"
)


# -----------------------------------------------------------------------------
# DockModel validation and attribute helpers
# -----------------------------------------------------------------------------

def is_dock_model_like(
    value: Any,
) -> bool:
    """
    Return whether an object exposes the minimum DockModel interface.

    The integration does not depend on the concrete DockModel class.
    A compatible object must provide a writable ``hydrophobic``
    attribute or allow that attribute to be created.
    """

    if value is None:
        return False

    if isinstance(
        value,
        HydrophobicAnalysisResult,
    ):
        return False

    if isinstance(
        value,
        Mapping,
    ):
        return False

    if hasattr(
        value,
        DEFAULT_DOCK_MODEL_HYDROPHOBIC_ATTRIBUTE,
    ):
        return True

    return hasattr(
        value,
        "__dict__",
    )


def validate_dock_model_like(
    dock_model: DockModelLike,
) -> DockModelLike:
    """
    Validate and return a DockModel-compatible object.
    """

    if not is_dock_model_like(
        dock_model
    ):
        raise TypeError(
            "dock_model must expose a writable "
            "'hydrophobic' attribute or support dynamic attributes."
        )

    return dock_model


def _safe_set_dock_model_attribute(
    dock_model: DockModelLike,
    attribute_name: str,
    value: Any,
    *,
    strict: bool = False,
) -> bool:
    """
    Safely set an attribute on a DockModel-compatible object.

    Returns
    -------
    bool
        ``True`` when the attribute was updated successfully.
    """

    try:
        setattr(
            dock_model,
            attribute_name,
            value,
        )

    except Exception:
        if strict:
            raise

        return False

    return True


def _first_existing_dock_model_attribute(
    dock_model: DockModelLike,
    attribute_names: Iterable[str],
) -> Tuple[Optional[str], Any]:
    """
    Return the first existing DockModel attribute and its value.
    """

    for attribute_name in attribute_names:
        try:
            if hasattr(
                dock_model,
                attribute_name,
            ):
                return (
                    attribute_name,
                    getattr(
                        dock_model,
                        attribute_name,
                    ),
                )

        except Exception:
            continue

    return (
        None,
        None,
    )


def get_dock_model_pose_identifier(
    dock_model: DockModelLike,
    *,
    default: Optional[str] = None,
) -> str:
    """
    Resolve a stable pose identifier from a DockModel.
    """

    validate_dock_model_like(
        dock_model
    )

    attribute_name, value = (
        _first_existing_dock_model_attribute(
            dock_model,
            DEFAULT_DOCK_MODEL_POSE_ATTRIBUTE_NAMES,
        )
    )

    normalized_value = (
        _normalize_optional_string(
            value
        )
    )

    if normalized_value is not None:
        return normalized_value

    if default is not None:
        normalized_default = (
            _normalize_optional_string(
                default
            )
        )

        if normalized_default is not None:
            return normalized_default

    return (
        f"pose-{id(dock_model)}"
    )


def get_dock_model_receptor(
    dock_model: DockModelLike,
    *,
    default: Any = None,
) -> Any:
    """
    Resolve the receptor source stored in a DockModel.
    """

    validate_dock_model_like(
        dock_model
    )

    _, receptor = (
        _first_existing_dock_model_attribute(
            dock_model,
            DEFAULT_DOCK_MODEL_RECEPTOR_ATTRIBUTE_NAMES,
        )
    )

    if receptor is not None:
        return receptor

    return default


def get_dock_model_ligand(
    dock_model: DockModelLike,
    *,
    default: Any = None,
) -> Any:
    """
    Resolve the ligand source stored in a DockModel.
    """

    validate_dock_model_like(
        dock_model
    )

    _, ligand = (
        _first_existing_dock_model_attribute(
            dock_model,
            DEFAULT_DOCK_MODEL_LIGAND_ATTRIBUTE_NAMES,
        )
    )

    if ligand is not None:
        return ligand

    return default


# -----------------------------------------------------------------------------
# Existing hydrophobic-result normalization
# -----------------------------------------------------------------------------

def get_existing_dock_model_hydrophobic_results(
    dock_model: DockModelLike,
) -> Tuple[Any, ...]:
    """
    Return previously stored ``dock_model.hydrophobic`` entries.

    The function tolerates ``None``, a single object, tuples and other
    iterable containers.
    """

    validate_dock_model_like(
        dock_model
    )

    try:
        existing = getattr(
            dock_model,
            DEFAULT_DOCK_MODEL_HYDROPHOBIC_ATTRIBUTE,
            None,
        )

    except Exception:
        return ()

    if existing is None:
        return ()

    if isinstance(
        existing,
        tuple,
    ):
        return existing

    if isinstance(
        existing,
        list,
    ):
        return tuple(existing)

    if isinstance(
        existing,
        (
            str,
            bytes,
            Mapping,
            HydrophobicAnalysisResult,
            HydrophobicInteraction,
        ),
    ):
        return (
            existing,
        )

    try:
        return tuple(existing)

    except TypeError:
        return (
            existing,
        )


def hydrophobic_result_identity_key(
    value: Any,
) -> Tuple[Any, ...]:
    """
    Create a stable key for stored hydrophobic-result deduplication.
    """

    if isinstance(
        value,
        HydrophobicAnalysisResult,
    ):
        return (
            "analysis-result",
            value.analysis_identifier,
            value.receptor_identifier,
            value.ligand_identifier,
            value.metadata.get(
                "pose_identifier"
            ),
        )

    if isinstance(
        value,
        Mapping,
    ):
        return (
            "serialized-result",
            value.get(
                "schema"
            ),
            value.get(
                "analysis_identifier"
            ),
            value.get(
                "pose_identifier"
            ),
            value.get(
                "receptor_identifier"
            ),
            value.get(
                "ligand_identifier"
            ),
        )

    return (
        "object",
        id(value),
    )


def merge_hydrophobic_result_entries(
    previous_entries: Iterable[Any],
    new_entries: Iterable[Any],
) -> Tuple[Any, ...]:
    """
    Merge stored entries while preserving order and removing duplicates.

    New entries replace older entries with the same semantic identity.
    """

    merged: Dict[
        Tuple[Any, ...],
        Any,
    ] = {}

    ordered_keys: List[
        Tuple[Any, ...]
    ] = []

    for entry in previous_entries:
        key = hydrophobic_result_identity_key(
            entry
        )

        if key not in merged:
            ordered_keys.append(key)

        merged[key] = entry

    for entry in new_entries:
        key = hydrophobic_result_identity_key(
            entry
        )

        if key not in merged:
            ordered_keys.append(key)

        merged[key] = entry

    return tuple(
        merged[key]
        for key in ordered_keys
    )


# -----------------------------------------------------------------------------
# DockModel serialization
# -----------------------------------------------------------------------------

def serialize_hydrophobic_analysis_result_for_dock_model(
    result: HydrophobicAnalysisResult,
    *,
    pose_identifier: Optional[str] = None,
    include_interactions: bool = True,
    include_tables: bool = True,
    include_atoms: bool = False,
    round_digits: int = (
        DEFAULT_HYDROPHOBIC_ROUND_DIGITS
    ),
) -> Dict[str, Any]:
    """
    Serialize a hydrophobic result for storage in ``DockModel.hydrophobic``.

    The serialized form contains no requirement for DockModel to know the
    internal implementation of this module.
    """

    if not isinstance(
        result,
        HydrophobicAnalysisResult,
    ):
        raise TypeError(
            "result must be a HydrophobicAnalysisResult."
        )

    resolved_pose_identifier = (
        _normalize_optional_string(
            pose_identifier
        )
        or _normalize_optional_string(
            result.metadata.get(
                "pose_identifier"
            )
        )
        or result.ligand_identifier
        or "pose-unknown"
    )

    summary = calculate_hydrophobic_summary(
        result,
        pose_identifiers=(
            resolved_pose_identifier,
        ),
    )

    serialized: Dict[str, Any] = {
        "schema": "dockanalyzer.hydrophobic",
        "schema_version": (
            HYDROPHOBIC_DOCK_MODEL_RESULT_SCHEMA_VERSION
        ),
        "analysis_identifier": (
            result.analysis_identifier
        ),
        "receptor_identifier": (
            result.receptor_identifier
        ),
        "ligand_identifier": (
            result.ligand_identifier
        ),
        "pose_identifier": (
            resolved_pose_identifier
        ),
        "minimum_distance_cutoff": float(
            result.minimum_distance
        ),
        "maximum_distance_cutoff": float(
            result.maximum_distance
        ),
        "grouping_distance": float(
            result.grouping_distance
        ),
        "interaction_count": (
            summary.interaction_count
        ),
        "atomic_pair_count": (
            summary.atomic_pair_count
        ),
        "local_interaction_count": (
            summary.local_interaction_count
        ),
        "residue_count": (
            summary.residue_count
        ),
        "hotspot_count": (
            summary.hotspot_count
        ),
        "total_score": float(
            summary.total_score
        ),
        "mean_score": (
            None
            if summary.mean_score is None
            else float(summary.mean_score)
        ),
        "minimum_distance": (
            None
            if summary.minimum_distance is None
            else float(
                summary.minimum_distance
            )
        ),
        "mean_distance": (
            None
            if summary.mean_distance is None
            else float(
                summary.mean_distance
            )
        ),
        "maximum_distance": (
            None
            if summary.maximum_distance is None
            else float(
                summary.maximum_distance
            )
        ),
        "approximate_contact_area": float(
            summary.approximate_contact_area
        ),
        "classification_counts": dict(
            summary.classification_counts
        ),
        "interaction_type_counts": dict(
            summary.interaction_type_counts
        ),
        "residue_scores": {
            key: float(value)
            for key, value
            in summary.residue_scores.items()
        },
        "residue_pose_occupancy": {
            key: float(value)
            for key, value
            in summary.residue_pose_occupancy.items()
        },
        "hotspot_residue_identifiers": list(
            summary.hotspot_residue_identifiers
        ),
        "summary": summary.to_dict(
            round_digits=round_digits
        ),
        "metadata": {
            **dict(result.metadata),
            "dock_model_serialization": True,
            "atoms_included": bool(
                include_atoms
            ),
        },
    }

    if include_interactions:
        serialized["interactions"] = [
            interaction.to_dict(
                include_atoms=include_atoms,
                include_residue=include_atoms,
                include_descriptors=True,
            )
            for interaction in result.interactions
        ]

    if include_tables:
        tables = (
            build_hydrophobic_serializable_tables(
                result,
                pose_identifiers=(
                    resolved_pose_identifier,
                ),
                round_digits=round_digits,
            )
        )

        serialized["tables"] = (
            tables.to_dict()
        )

    return serialized


# -----------------------------------------------------------------------------
# DockModel score integration
# -----------------------------------------------------------------------------

def calculate_dock_model_hydrophobic_score(
    result: HydrophobicAnalysisResult,
    *,
    normalize_by_ligand_atom_count: bool = False,
    normalize_by_residue_count: bool = False,
) -> np.float64:
    """
    Calculate the score exposed to DockModel.

    By default, the complete hydrophobic total score is returned.
    Optional normalization can support comparisons between ligands of
    different sizes.
    """

    if not isinstance(
        result,
        HydrophobicAnalysisResult,
    ):
        raise TypeError(
            "result must be a HydrophobicAnalysisResult."
        )

    summary = calculate_hydrophobic_summary(
        result
    )

    score = float(
        summary.total_score
    )

    if normalize_by_ligand_atom_count:
        denominator = max(
            summary.contacted_ligand_atom_count,
            1,
        )

        score /= denominator

    if normalize_by_residue_count:
        denominator = max(
            summary.residue_count,
            1,
        )

        score /= denominator

    return np.float64(
        max(
            score,
            0.0,
        )
    )


def update_dock_model_hydrophobic_score(
    dock_model: DockModelLike,
    result: HydrophobicAnalysisResult,
    *,
    attribute_names: Sequence[str] = (
        DEFAULT_HYDROPHOBIC_SCORE_ATTRIBUTE_NAMES
    ),
    normalize_by_ligand_atom_count: bool = False,
    normalize_by_residue_count: bool = False,
    strict: bool = False,
) -> np.float64:
    """
    Update a dedicated hydrophobic score on DockModel.

    This function does not overwrite DockModel's general docking score
    unless one of the supplied attribute names explicitly points to it.
    """

    validate_dock_model_like(
        dock_model
    )

    hydrophobic_score = (
        calculate_dock_model_hydrophobic_score(
            result,
            normalize_by_ligand_atom_count=(
                normalize_by_ligand_atom_count
            ),
            normalize_by_residue_count=(
                normalize_by_residue_count
            ),
        )
    )

    updated = False

    for attribute_name in attribute_names:
        if _safe_set_dock_model_attribute(
            dock_model,
            attribute_name,
            float(hydrophobic_score),
            strict=False,
        ):
            updated = True
            break

    if not updated and strict:
        raise AttributeError(
            "Could not update a hydrophobic score attribute "
            "on DockModel."
        )

    return hydrophobic_score


def update_dock_model_combined_score(
    dock_model: DockModelLike,
    hydrophobic_score: Number,
    *,
    contribution_weight: Number = 1.0,
    preserve_original_score: bool = True,
    strict: bool = False,
) -> Optional[np.float64]:
    """
    Optionally add the hydrophobic contribution to DockModel's score.

    The original score is preserved in ``score_before_hydrophobic`` when
    possible. This operation is intentionally opt-in because DockModel's
    internal scoring semantics may vary.
    """

    validate_dock_model_like(
        dock_model
    )

    normalized_hydrophobic_score = (
        _nonnegative_float(
            hydrophobic_score,
            name="hydrophobic score",
        )
    )

    weight = _finite_float(
        contribution_weight,
        name="hydrophobic contribution weight",
    )

    score_attribute, current_score = (
        _first_existing_dock_model_attribute(
            dock_model,
            DEFAULT_DOCK_MODEL_SCORE_ATTRIBUTE_NAMES,
        )
    )

    if score_attribute is None:
        if strict:
            raise AttributeError(
                "No compatible DockModel score attribute was found."
            )

        return None

    try:
        normalized_current_score = _finite_float(
            current_score,
            name="DockModel score",
        )

    except (
        TypeError,
        ValueError,
    ):
        if strict:
            raise

        return None

    if preserve_original_score:
        _safe_set_dock_model_attribute(
            dock_model,
            "score_before_hydrophobic",
            float(normalized_current_score),
            strict=False,
        )

    combined_score = np.float64(
        normalized_current_score
        + weight
        * normalized_hydrophobic_score
    )

    updated = _safe_set_dock_model_attribute(
        dock_model,
        score_attribute,
        float(combined_score),
        strict=strict,
    )

    if not updated:
        return None

    return combined_score


# -----------------------------------------------------------------------------
# DockModel statistics integration
# -----------------------------------------------------------------------------

def update_dock_model_hydrophobic_statistics(
    dock_model: DockModelLike,
    result: HydrophobicAnalysisResult,
    *,
    attribute_names: Sequence[str] = (
        DEFAULT_HYDROPHOBIC_STATISTICS_ATTRIBUTE_NAMES
    ),
    serialize: bool = True,
    strict: bool = False,
) -> HydrophobicSummary:
    """
    Calculate and attach hydrophobic statistics to DockModel.
    """

    validate_dock_model_like(
        dock_model
    )

    summary = calculate_hydrophobic_summary(
        result
    )

    attached_value: Any = (
        summary.to_dict()
        if serialize
        else summary
    )

    updated = False

    for attribute_name in attribute_names:
        if _safe_set_dock_model_attribute(
            dock_model,
            attribute_name,
            attached_value,
            strict=False,
        ):
            updated = True
            break

    if not updated and strict:
        raise AttributeError(
            "Could not update a hydrophobic statistics "
            "attribute on DockModel."
        )

    return summary


# -----------------------------------------------------------------------------
# Attachment record
# -----------------------------------------------------------------------------

@dataclass(
    frozen=True,
    slots=True,
)
class HydrophobicDockModelAttachment:
    """
    Record describing a hydrophobic result attached to DockModel.
    """

    pose_identifier: str
    analysis_result: HydrophobicAnalysisResult

    serialized_result: Optional[
        Mapping[str, Any]
    ] = None

    previous_entry_count: int = 0
    final_entry_count: int = 0

    attachment_mode: HydrophobicAttachmentMode = (
        DEFAULT_HYDROPHOBIC_ATTACHMENT_MODE
    )

    previous_results_preserved: bool = True
    statistics_updated: bool = False
    hydrophobic_score_updated: bool = False
    combined_score_updated: bool = False

    hydrophobic_score: Optional[
        np.float64
    ] = None

    combined_score: Optional[
        np.float64
    ] = None

    metadata: Mapping[str, Any] = field(
        default_factory=lambda: _EMPTY_METADATA,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        """Validate and freeze the attachment record."""

        pose_identifier = (
            _normalize_required_string(
                self.pose_identifier,
                name="pose identifier",
            )
        )

        if not isinstance(
            self.analysis_result,
            HydrophobicAnalysisResult,
        ):
            raise TypeError(
                "analysis_result must be a "
                "HydrophobicAnalysisResult."
            )

        attachment_mode = str(
            self.attachment_mode
        ).strip().lower()

        if attachment_mode not in {
            "append",
            "replace",
            "merge",
        }:
            raise ValueError(
                "attachment_mode must be 'append', "
                "'replace' or 'merge'."
            )

        serialized_result = (
            None
            if self.serialized_result is None
            else _freeze_metadata(
                self.serialized_result
            )
        )

        hydrophobic_score = (
            None
            if self.hydrophobic_score is None
            else _nonnegative_float(
                self.hydrophobic_score,
                name="hydrophobic score",
            )
        )

        combined_score = (
            None
            if self.combined_score is None
            else _finite_float(
                self.combined_score,
                name="combined score",
            )
        )

        object.__setattr__(
            self,
            "pose_identifier",
            pose_identifier,
        )

        object.__setattr__(
            self,
            "serialized_result",
            serialized_result,
        )

        object.__setattr__(
            self,
            "previous_entry_count",
            _nonnegative_integer(
                self.previous_entry_count,
                name="previous entry count",
            ),
        )

        object.__setattr__(
            self,
            "final_entry_count",
            _nonnegative_integer(
                self.final_entry_count,
                name="final entry count",
            ),
        )

        object.__setattr__(
            self,
            "attachment_mode",
            attachment_mode,
        )

        object.__setattr__(
            self,
            "hydrophobic_score",
            hydrophobic_score,
        )

        object.__setattr__(
            self,
            "combined_score",
            combined_score,
        )

        object.__setattr__(
            self,
            "metadata",
            _freeze_metadata(
                self.metadata
            ),
        )

    def to_dict(
        self,
        *,
        include_analysis_result: bool = False,
    ) -> Dict[str, Any]:
        """Serialize the attachment record."""

        result: Dict[str, Any] = {
            "pose_identifier": self.pose_identifier,
            "previous_entry_count": (
                self.previous_entry_count
            ),
            "final_entry_count": (
                self.final_entry_count
            ),
            "attachment_mode": (
                self.attachment_mode
            ),
            "previous_results_preserved": (
                self.previous_results_preserved
            ),
            "statistics_updated": (
                self.statistics_updated
            ),
            "hydrophobic_score_updated": (
                self.hydrophobic_score_updated
            ),
            "combined_score_updated": (
                self.combined_score_updated
            ),
            "hydrophobic_score": (
                None
                if self.hydrophobic_score is None
                else float(
                    self.hydrophobic_score
                )
            ),
            "combined_score": (
                None
                if self.combined_score is None
                else float(
                    self.combined_score
                )
            ),
            "serialized_result": (
                None
                if self.serialized_result is None
                else dict(
                    self.serialized_result
                )
            ),
            "metadata": dict(
                self.metadata
            ),
        }

        if include_analysis_result:
            result["analysis_result"] = (
                self.analysis_result.to_dict(
                    include_interactions=True,
                    include_atoms=False,
                )
            )

        return result


# -----------------------------------------------------------------------------
# Attaching hydrophobic results
# -----------------------------------------------------------------------------

def attach_hydrophobic_results(
    dock_model: DockModelLike,
    result: HydrophobicAnalysisResult,
    *,
    mode: HydrophobicAttachmentMode = (
        DEFAULT_HYDROPHOBIC_ATTACHMENT_MODE
    ),
    preserve_previous: bool = (
        DEFAULT_PRESERVE_PREVIOUS_HYDROPHOBIC_RESULTS
    ),
    serialize: bool = (
        DEFAULT_SERIALIZE_DOCK_MODEL_RESULTS
    ),
    include_interactions: bool = True,
    include_tables: bool = True,
    include_atoms: bool = False,
    update_statistics: bool = (
        DEFAULT_UPDATE_DOCK_MODEL_STATISTICS
    ),
    update_hydrophobic_score: bool = (
        DEFAULT_UPDATE_DOCK_MODEL_SCORE
    ),
    update_combined_score: bool = False,
    combined_score_weight: Number = 1.0,
    preserve_original_score: bool = True,
    pose_identifier: Optional[str] = None,
    strict: bool = True,
) -> HydrophobicDockModelAttachment:
    """
    Attach a standardized hydrophobic result to ``DockModel.hydrophobic``.

    Parameters
    ----------
    mode
        ``"append"`` always adds the new entry;
        ``"replace"`` replaces the complete hydrophobic list;
        ``"merge"`` replaces only semantically equivalent entries.
    preserve_previous
        Preserve previous entries. When false, behavior is equivalent to
        replacement regardless of ``mode``.
    serialize
        Store a plain dictionary instead of the live result object.
    update_statistics
        Attach a separate hydrophobic statistics summary.
    update_hydrophobic_score
        Attach the dedicated hydrophobic score.
    update_combined_score
        Add the hydrophobic score to DockModel's general score. This is
        opt-in to avoid changing DockModel's internal algorithm silently.
    """

    validate_dock_model_like(
        dock_model
    )

    if not isinstance(
        result,
        HydrophobicAnalysisResult,
    ):
        raise TypeError(
            "result must be a HydrophobicAnalysisResult."
        )

    normalized_mode = str(
        mode
    ).strip().lower()

    if normalized_mode not in {
        "append",
        "replace",
        "merge",
    }:
        raise ValueError(
            "mode must be 'append', 'replace' or 'merge'."
        )

    resolved_pose_identifier = (
        _normalize_optional_string(
            pose_identifier
        )
        or get_dock_model_pose_identifier(
            dock_model
        )
    )

    previous_entries = (
        get_existing_dock_model_hydrophobic_results(
            dock_model
        )
    )

    serialized_result: Optional[
        Dict[str, Any]
    ] = None

    if serialize:
        serialized_result = (
            serialize_hydrophobic_analysis_result_for_dock_model(
                result,
                pose_identifier=(
                    resolved_pose_identifier
                ),
                include_interactions=(
                    include_interactions
                ),
                include_tables=include_tables,
                include_atoms=include_atoms,
            )
        )

        new_entry: Any = (
            serialized_result
        )

    else:
        new_entry = result

    if (
        not preserve_previous
        or normalized_mode == "replace"
    ):
        final_entries = (
            new_entry,
        )

    elif normalized_mode == "append":
        final_entries = (
            *previous_entries,
            new_entry,
        )

    else:
        final_entries = (
            merge_hydrophobic_result_entries(
                previous_entries,
                (
                    new_entry,
                ),
            )
        )

    attached = _safe_set_dock_model_attribute(
        dock_model,
        DEFAULT_DOCK_MODEL_HYDROPHOBIC_ATTRIBUTE,
        list(final_entries),
        strict=strict,
    )

    if not attached:
        raise AttributeError(
            "Could not update dock_model.hydrophobic."
        )

    statistics_updated = False

    if update_statistics:
        try:
            update_dock_model_hydrophobic_statistics(
                dock_model,
                result,
                serialize=True,
                strict=strict,
            )

            statistics_updated = True

        except Exception:
            if strict:
                raise

    hydrophobic_score: Optional[
        np.float64
    ] = None

    hydrophobic_score_updated = False

    if update_hydrophobic_score:
        try:
            hydrophobic_score = (
                update_dock_model_hydrophobic_score(
                    dock_model,
                    result,
                    strict=strict,
                )
            )

            hydrophobic_score_updated = True

        except Exception:
            if strict:
                raise

    combined_score: Optional[
        np.float64
    ] = None

    combined_score_updated = False

    if update_combined_score:
        if hydrophobic_score is None:
            hydrophobic_score = (
                calculate_dock_model_hydrophobic_score(
                    result
                )
            )

        combined_score = (
            update_dock_model_combined_score(
                dock_model,
                hydrophobic_score,
                contribution_weight=(
                    combined_score_weight
                ),
                preserve_original_score=(
                    preserve_original_score
                ),
                strict=strict,
            )
        )

        combined_score_updated = (
            combined_score is not None
        )

    return HydrophobicDockModelAttachment(
        pose_identifier=(
            resolved_pose_identifier
        ),
        analysis_result=result,
        serialized_result=(
            serialized_result
        ),
        previous_entry_count=len(
            previous_entries
        ),
        final_entry_count=len(
            final_entries
        ),
        attachment_mode=normalized_mode,
        previous_results_preserved=bool(
            preserve_previous
        ),
        statistics_updated=(
            statistics_updated
        ),
        hydrophobic_score_updated=(
            hydrophobic_score_updated
        ),
        combined_score_updated=(
            combined_score_updated
        ),
        hydrophobic_score=(
            hydrophobic_score
        ),
        combined_score=combined_score,
        metadata={
            "dock_model_type": type(
                dock_model
            ).__name__,
            "serialized": bool(serialize),
            "include_interactions": bool(
                include_interactions
            ),
            "include_tables": bool(
                include_tables
            ),
            "include_atoms": bool(
                include_atoms
            ),
        },
    )


# -----------------------------------------------------------------------------
# Single-pose DockModel analysis
# -----------------------------------------------------------------------------

def analyze_dock_model_hydrophobic(
    dock_model: DockModelLike,
    *,
    receptor: Any = None,
    ligand: Any = None,
    minimum_distance: Optional[Number] = None,
    maximum_distance: Optional[Number] = None,
    grouping_distance: Optional[Number] = None,
    local_radius: Optional[Number] = None,
    weights: Optional[Mapping[str, Number]] = None,
    preparation_options: Optional[
        Mapping[str, Any]
    ] = None,
    identify_hotspots: bool = True,
    occupancy_minimum_group_score: Number = (
        DEFAULT_HYDROPHOBIC_OCCUPANCY_THRESHOLD
    ),
    attach: bool = True,
    attachment_mode: HydrophobicAttachmentMode = (
        DEFAULT_HYDROPHOBIC_ATTACHMENT_MODE
    ),
    preserve_previous: bool = (
        DEFAULT_PRESERVE_PREVIOUS_HYDROPHOBIC_RESULTS
    ),
    serialize: bool = (
        DEFAULT_SERIALIZE_DOCK_MODEL_RESULTS
    ),
    include_interactions: bool = True,
    include_tables: bool = True,
    update_statistics: bool = (
        DEFAULT_UPDATE_DOCK_MODEL_STATISTICS
    ),
    update_hydrophobic_score: bool = (
        DEFAULT_UPDATE_DOCK_MODEL_SCORE
    ),
    update_combined_score: bool = False,
    combined_score_weight: Number = 1.0,
    preserve_original_score: bool = True,
    analysis_identifier: Optional[str] = None,
    receptor_identifier: Optional[str] = None,
    ligand_identifier: Optional[str] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> HydrophobicAnalysisResult:
    """
    Analyze one DockModel pose and optionally attach the result.

    The DockModel object is only used as a source of receptor/ligand
    structures and as a result container. Its internal algorithms are not
    modified.
    """

    validate_dock_model_like(
        dock_model
    )

    resolved_receptor = (
        receptor
        if receptor is not None
        else get_dock_model_receptor(
            dock_model
        )
    )

    resolved_ligand = (
        ligand
        if ligand is not None
        else get_dock_model_ligand(
            dock_model
        )
    )

    pose_identifier = (
        get_dock_model_pose_identifier(
            dock_model
        )
    )

    resolved_analysis_identifier = (
        _normalize_optional_string(
            analysis_identifier
        )
        or f"hydrophobic-analysis|{pose_identifier}"
    )

    resolved_ligand_identifier = (
        _normalize_optional_string(
            ligand_identifier
        )
        or pose_identifier
    )

    analysis_metadata: Dict[str, Any] = (
        {} if metadata is None else dict(metadata)
    )

    analysis_metadata.update(
        {
            "source": "DockModel",
            "pose_identifier": (
                pose_identifier
            ),
            "dock_model_type": type(
                dock_model
            ).__name__,
            "single_pose_analysis": True,
        }
    )

    result = analyze_hydrophobic_interactions(
        resolved_receptor
        if resolved_receptor is not None
        else dock_model,
        resolved_ligand,
        minimum_distance=minimum_distance,
        maximum_distance=maximum_distance,
        grouping_distance=grouping_distance,
        local_radius=local_radius,
        weights=weights,
        preparation_options=(
            preparation_options
        ),
        identify_hotspots=(
            identify_hotspots
        ),
        pose_identifiers=(
            pose_identifier,
        ),
        occupancy_minimum_group_score=(
            occupancy_minimum_group_score
        ),
        include_serializable_tables=(
            include_tables
        ),
        receptor_identifier=(
            receptor_identifier
        ),
        ligand_identifier=(
            resolved_ligand_identifier
        ),
        analysis_identifier=(
            resolved_analysis_identifier
        ),
        metadata=analysis_metadata,
    )

    if attach:
        attach_hydrophobic_results(
            dock_model,
            result,
            mode=attachment_mode,
            preserve_previous=(
                preserve_previous
            ),
            serialize=serialize,
            include_interactions=(
                include_interactions
            ),
            include_tables=include_tables,
            update_statistics=(
                update_statistics
            ),
            update_hydrophobic_score=(
                update_hydrophobic_score
            ),
            update_combined_score=(
                update_combined_score
            ),
            combined_score_weight=(
                combined_score_weight
            ),
            preserve_original_score=(
                preserve_original_score
            ),
            pose_identifier=(
                pose_identifier
            ),
        )

    return result


# -----------------------------------------------------------------------------
# Multipose analysis result
# -----------------------------------------------------------------------------

@dataclass(
    frozen=True,
    slots=True,
)
class HydrophobicMultiPoseResult:
    """
    Aggregated result from several DockModel poses.
    """

    results: Sequence[
        HydrophobicAnalysisResult
    ] = field(
        default_factory=tuple
    )

    pose_identifiers: Sequence[str] = field(
        default_factory=tuple
    )

    attachments: Sequence[
        HydrophobicDockModelAttachment
    ] = field(
        default_factory=tuple
    )

    combined_summary: Optional[
        HydrophobicSummary
    ] = None

    combined_tables: Optional[
        HydrophobicSerializableTables
    ] = None

    failed_pose_identifiers: Sequence[str] = field(
        default_factory=tuple
    )

    errors: Mapping[str, str] = field(
        default_factory=lambda: MappingProxyType({})
    )

    metadata: Mapping[str, Any] = field(
        default_factory=lambda: _EMPTY_METADATA,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        """Validate and freeze the multipose result."""

        results = tuple(
            self.results
        )

        attachments = tuple(
            self.attachments
        )

        for result in results:
            if not isinstance(
                result,
                HydrophobicAnalysisResult,
            ):
                raise TypeError(
                    "results must contain "
                    "HydrophobicAnalysisResult instances."
                )

        for attachment in attachments:
            if not isinstance(
                attachment,
                HydrophobicDockModelAttachment,
            ):
                raise TypeError(
                    "attachments must contain "
                    "HydrophobicDockModelAttachment instances."
                )

        pose_identifiers = tuple(
            str(value).strip()
            for value in self.pose_identifiers
            if str(value).strip()
        )

        failed_pose_identifiers = tuple(
            str(value).strip()
            for value
            in self.failed_pose_identifiers
            if str(value).strip()
        )

        object.__setattr__(
            self,
            "results",
            results,
        )

        object.__setattr__(
            self,
            "pose_identifiers",
            pose_identifiers,
        )

        object.__setattr__(
            self,
            "attachments",
            attachments,
        )

        object.__setattr__(
            self,
            "failed_pose_identifiers",
            failed_pose_identifiers,
        )

        object.__setattr__(
            self,
            "errors",
            MappingProxyType(
                {
                    str(key): str(value)
                    for key, value
                    in dict(self.errors).items()
                }
            ),
        )

        object.__setattr__(
            self,
            "metadata",
            _freeze_metadata(
                self.metadata
            ),
        )

    @property
    def pose_count(self) -> int:
        """Return the number of successful poses."""

        return len(self.results)

    @property
    def failed_pose_count(self) -> int:
        """Return the number of failed poses."""

        return len(
            self.failed_pose_identifiers
        )

    @property
    def interaction_count(self) -> int:
        """Return the total atomic interaction count."""

        return sum(
            len(result.interactions)
            for result in self.results
        )

    @property
    def total_score(self) -> np.float64:
        """Return the combined hydrophobic score."""

        if self.combined_summary is not None:
            return self.combined_summary.total_score

        return np.float64(
            sum(
                float(
                    result.statistics.total_score
                )
                for result in self.results
            )
        )

    def to_dict(
        self,
        *,
        include_results: bool = True,
        include_attachments: bool = True,
    ) -> Dict[str, Any]:
        """Serialize the multipose result."""

        serialized: Dict[str, Any] = {
            "pose_count": self.pose_count,
            "failed_pose_count": (
                self.failed_pose_count
            ),
            "interaction_count": (
                self.interaction_count
            ),
            "total_score": float(
                self.total_score
            ),
            "pose_identifiers": list(
                self.pose_identifiers
            ),
            "failed_pose_identifiers": list(
                self.failed_pose_identifiers
            ),
            "errors": dict(
                self.errors
            ),
            "combined_summary": (
                None
                if self.combined_summary is None
                else self.combined_summary.to_dict()
            ),
            "combined_tables": (
                None
                if self.combined_tables is None
                else self.combined_tables.to_dict()
            ),
            "metadata": dict(
                self.metadata
            ),
        }

        if include_results:
            serialized["results"] = [
                serialize_hydrophobic_analysis_result_for_dock_model(
                    result,
                    pose_identifier=(
                        result.metadata.get(
                            "pose_identifier"
                        )
                    ),
                    include_interactions=True,
                    include_tables=True,
                    include_atoms=False,
                )
                for result in self.results
            ]

        if include_attachments:
            serialized["attachments"] = [
                attachment.to_dict(
                    include_analysis_result=False
                )
                for attachment
                in self.attachments
            ]

        return serialized


# -----------------------------------------------------------------------------
# Multipose interaction aggregation
# -----------------------------------------------------------------------------

def combine_hydrophobic_pose_interactions(
    results: Iterable[
        HydrophobicAnalysisResult
    ],
) -> Tuple[HydrophobicInteraction, ...]:
    """
    Combine interactions from several poses.

    Pose identity is inserted into metadata so occupancy calculations
    remain valid even when atom identifiers repeat between poses.
    """

    combined: List[
        HydrophobicInteraction
    ] = []

    for result in results:
        if not isinstance(
            result,
            HydrophobicAnalysisResult,
        ):
            raise TypeError(
                "All results must be "
                "HydrophobicAnalysisResult instances."
            )

        pose_identifier = (
            _normalize_optional_string(
                result.metadata.get(
                    "pose_identifier"
                )
            )
            or result.ligand_identifier
            or "pose-unknown"
        )

        for interaction in result.interactions:
            metadata = dict(
                interaction.metadata
            )

            metadata[
                "pose_identifier"
            ] = pose_identifier

            combined.append(
                rebuild_hydrophobic_interaction(
                    interaction,
                    metadata=metadata,
                )
            )

    return tuple(combined)


def summarize_multiple_hydrophobic_results(
    results: Iterable[
        HydrophobicAnalysisResult
    ],
    *,
    pose_identifiers: Optional[
        Iterable[str]
    ] = None,
) -> Tuple[
    HydrophobicSummary,
    HydrophobicSerializableTables,
]:
    """
    Build combined multipose statistics and tables.
    """

    result_tuple = tuple(
        results
    )

    combined_interactions = (
        combine_hydrophobic_pose_interactions(
            result_tuple
        )
    )

    resolved_pose_identifiers = (
        tuple(
            pose_identifiers
        )
        if pose_identifiers is not None
        else tuple(
            (
                _normalize_optional_string(
                    result.metadata.get(
                        "pose_identifier"
                    )
                )
                or result.ligand_identifier
                or f"pose-{index + 1}"
            )
            for index, result
            in enumerate(result_tuple)
        )
    )

    summary = calculate_hydrophobic_summary(
        combined_interactions,
        pose_identifiers=(
            resolved_pose_identifiers
        ),
        metadata={
            "multipose_summary": True,
            "source_result_count": len(
                result_tuple
            ),
        },
    )

    tables = (
        build_hydrophobic_serializable_tables(
            combined_interactions,
            pose_identifiers=(
                resolved_pose_identifiers
            ),
        )
    )

    return (
        summary,
        tables,
    )


# -----------------------------------------------------------------------------
# Multiple DockModel analysis
# -----------------------------------------------------------------------------

def analyze_multiple_dock_models_hydrophobic(
    dock_models: Iterable[DockModelLike],
    *,
    receptor: Any = None,
    receptor_getter: Optional[
        Callable[[DockModelLike], Any]
    ] = None,
    ligand_getter: Optional[
        Callable[[DockModelLike], Any]
    ] = None,
    minimum_distance: Optional[Number] = None,
    maximum_distance: Optional[Number] = None,
    grouping_distance: Optional[Number] = None,
    local_radius: Optional[Number] = None,
    weights: Optional[Mapping[str, Number]] = None,
    preparation_options: Optional[
        Mapping[str, Any]
    ] = None,
    identify_hotspots: bool = True,
    attach: bool = True,
    attachment_mode: HydrophobicAttachmentMode = (
        DEFAULT_HYDROPHOBIC_ATTACHMENT_MODE
    ),
    preserve_previous: bool = (
        DEFAULT_PRESERVE_PREVIOUS_HYDROPHOBIC_RESULTS
    ),
    serialize: bool = (
        DEFAULT_SERIALIZE_DOCK_MODEL_RESULTS
    ),
    include_interactions: bool = True,
    include_tables: bool = True,
    update_statistics: bool = (
        DEFAULT_UPDATE_DOCK_MODEL_STATISTICS
    ),
    update_hydrophobic_score: bool = (
        DEFAULT_UPDATE_DOCK_MODEL_SCORE
    ),
    update_combined_score: bool = False,
    combined_score_weight: Number = 1.0,
    preserve_original_score: bool = True,
    continue_on_error: bool = True,
    metadata: Optional[Mapping[str, Any]] = None,
) -> HydrophobicMultiPoseResult:
    """
    Analyze hydrophobic contacts across several DockModel poses.

    Each DockModel is analyzed independently. Results are then combined
    to calculate residue occupancy and multipose summaries.
    """

    dock_model_tuple = tuple(
        dock_models
    )

    results: List[
        HydrophobicAnalysisResult
    ] = []

    attachments: List[
        HydrophobicDockModelAttachment
    ] = []

    pose_identifiers: List[str] = []
    failed_pose_identifiers: List[str] = []

    errors: Dict[str, str] = {}

    for pose_index, dock_model in enumerate(
        dock_model_tuple,
        start=1,
    ):
        try:
            validate_dock_model_like(
                dock_model
            )

            pose_identifier = (
                get_dock_model_pose_identifier(
                    dock_model,
                    default=f"pose-{pose_index}",
                )
            )

            resolved_receptor = (
                receptor_getter(
                    dock_model
                )
                if receptor_getter is not None
                else (
                    receptor
                    if receptor is not None
                    else get_dock_model_receptor(
                        dock_model
                    )
                )
            )

            resolved_ligand = (
                ligand_getter(
                    dock_model
                )
                if ligand_getter is not None
                else get_dock_model_ligand(
                    dock_model
                )
            )

            pose_metadata = {
                **(
                    {}
                    if metadata is None
                    else dict(metadata)
                ),
                "source": "DockModel",
                "pose_identifier": (
                    pose_identifier
                ),
                "pose_index": pose_index,
                "multipose_analysis": True,
            }

            result = (
                analyze_hydrophobic_interactions(
                    resolved_receptor
                    if resolved_receptor is not None
                    else dock_model,
                    resolved_ligand,
                    minimum_distance=(
                        minimum_distance
                    ),
                    maximum_distance=(
                        maximum_distance
                    ),
                    grouping_distance=(
                        grouping_distance
                    ),
                    local_radius=local_radius,
                    weights=weights,
                    preparation_options=(
                        preparation_options
                    ),
                    identify_hotspots=(
                        identify_hotspots
                    ),
                    pose_identifiers=(
                        pose_identifier,
                    ),
                    include_serializable_tables=(
                        include_tables
                    ),
                    ligand_identifier=(
                        pose_identifier
                    ),
                    analysis_identifier=(
                        f"hydrophobic-analysis|"
                        f"{pose_identifier}"
                    ),
                    metadata=pose_metadata,
                )
            )

            results.append(result)
            pose_identifiers.append(
                pose_identifier
            )

            if attach:
                attachment = (
                    attach_hydrophobic_results(
                        dock_model,
                        result,
                        mode=attachment_mode,
                        preserve_previous=(
                            preserve_previous
                        ),
                        serialize=serialize,
                        include_interactions=(
                            include_interactions
                        ),
                        include_tables=(
                            include_tables
                        ),
                        update_statistics=(
                            update_statistics
                        ),
                        update_hydrophobic_score=(
                            update_hydrophobic_score
                        ),
                        update_combined_score=(
                            update_combined_score
                        ),
                        combined_score_weight=(
                            combined_score_weight
                        ),
                        preserve_original_score=(
                            preserve_original_score
                        ),
                        pose_identifier=(
                            pose_identifier
                        ),
                    )
                )

                attachments.append(
                    attachment
                )

        except Exception as exc:
            failed_identifier = (
                get_dock_model_pose_identifier(
                    dock_model,
                    default=f"pose-{pose_index}",
                )
                if is_dock_model_like(
                    dock_model
                )
                else f"pose-{pose_index}"
            )

            failed_pose_identifiers.append(
                failed_identifier
            )

            errors[
                failed_identifier
            ] = (
                f"{type(exc).__name__}: {exc}"
            )

            if not continue_on_error:
                raise

    if results:
        (
            combined_summary,
            combined_tables,
        ) = summarize_multiple_hydrophobic_results(
            results,
            pose_identifiers=(
                pose_identifiers
            ),
        )

    else:
        combined_summary = (
            _EMPTY_HYDROPHOBIC_SUMMARY
        )

        combined_tables = (
            _EMPTY_HYDROPHOBIC_SERIALIZABLE_TABLES
        )

    multipose_metadata: Dict[str, Any] = (
        {} if metadata is None else dict(metadata)
    )

    multipose_metadata.update(
        {
            "analysis_stage": (
                "DockModel_multipose_integration"
            ),
            "requested_pose_count": len(
                dock_model_tuple
            ),
            "successful_pose_count": len(
                results
            ),
            "failed_pose_count": len(
                failed_pose_identifiers
            ),
            "results_attached": bool(
                attach
            ),
            "attachment_mode": (
                attachment_mode
            ),
            "previous_results_preserved": bool(
                preserve_previous
            ),
            "serialized": bool(
                serialize
            ),
        }
    )

    return HydrophobicMultiPoseResult(
        results=results,
        pose_identifiers=(
            pose_identifiers
        ),
        attachments=attachments,
        combined_summary=(
            combined_summary
        ),
        combined_tables=combined_tables,
        failed_pose_identifiers=(
            failed_pose_identifiers
        ),
        errors=errors,
        metadata=multipose_metadata,
    )


# -----------------------------------------------------------------------------
# Updating previously attached DockModel results
# -----------------------------------------------------------------------------

def refresh_dock_model_hydrophobic_results(
    dock_model: DockModelLike,
    *,
    preserve_unknown_entries: bool = True,
    include_tables: bool = True,
) -> Tuple[Any, ...]:
    """
    Re-serialize live HydrophobicAnalysisResult entries already attached.

    Unknown legacy entries are preserved by default.
    """

    existing_entries = (
        get_existing_dock_model_hydrophobic_results(
            dock_model
        )
    )

    refreshed: List[Any] = []

    for entry in existing_entries:
        if isinstance(
            entry,
            HydrophobicAnalysisResult,
        ):
            refreshed.append(
                serialize_hydrophobic_analysis_result_for_dock_model(
                    entry,
                    pose_identifier=(
                        entry.metadata.get(
                            "pose_identifier"
                        )
                    ),
                    include_interactions=True,
                    include_tables=(
                        include_tables
                    ),
                    include_atoms=False,
                )
            )

        elif preserve_unknown_entries:
            refreshed.append(entry)

    _safe_set_dock_model_attribute(
        dock_model,
        DEFAULT_DOCK_MODEL_HYDROPHOBIC_ATTRIBUTE,
        list(refreshed),
        strict=True,
    )

    return tuple(refreshed)


# -----------------------------------------------------------------------------
# Reading attached standardized results
# -----------------------------------------------------------------------------

def get_standardized_dock_model_hydrophobic_results(
    dock_model: DockModelLike,
) -> Tuple[Mapping[str, Any], ...]:
    """
    Return only standardized serialized hydrophobic entries.
    """

    standardized: List[
        Mapping[str, Any]
    ] = []

    for entry in (
        get_existing_dock_model_hydrophobic_results(
            dock_model
        )
    ):
        if not isinstance(
            entry,
            Mapping,
        ):
            continue

        if (
            entry.get(
                "schema"
            )
            != "dockanalyzer.hydrophobic"
        ):
            continue

        standardized.append(
            MappingProxyType(
                dict(entry)
            )
        )

    return tuple(
        standardized
    )


def get_latest_dock_model_hydrophobic_result(
    dock_model: DockModelLike,
) -> Optional[Any]:
    """
    Return the latest attached hydrophobic result.
    """

    entries = (
        get_existing_dock_model_hydrophobic_results(
            dock_model
        )
    )

    if not entries:
        return None

    return entries[-1]


# -----------------------------------------------------------------------------
# DockModel integration serialization bundle
# -----------------------------------------------------------------------------

def serialize_hydrophobic_multi_pose_result(
    result: HydrophobicMultiPoseResult,
    *,
    include_pose_results: bool = True,
    include_attachments: bool = True,
) -> Dict[str, Any]:
    """
    Serialize a complete multipose DockModel analysis.
    """

    if not isinstance(
        result,
        HydrophobicMultiPoseResult,
    ):
        raise TypeError(
            "result must be a HydrophobicMultiPoseResult."
        )

    serialized = result.to_dict(
        include_results=(
            include_pose_results
        ),
        include_attachments=(
            include_attachments
        ),
    )

    serialized.update(
        {
            "schema": (
                "dockanalyzer.hydrophobic.multipose"
            ),
            "schema_version": (
                HYDROPHOBIC_DOCK_MODEL_RESULT_SCHEMA_VERSION
            ),
        }
    )

    return serialized


# -----------------------------------------------------------------------------
# Empty DockModel integration objects
# -----------------------------------------------------------------------------

_EMPTY_HYDROPHOBIC_MULTIPOSE_RESULT: Final[
    HydrophobicMultiPoseResult
] = HydrophobicMultiPoseResult()


# -----------------------------------------------------------------------------
# Section 11 public names
# -----------------------------------------------------------------------------

_SECTION_11_PUBLIC_NAMES: Final[Tuple[str, ...]] = (
    # Aliases
    "DockModelLike",
    "HydrophobicAttachmentMode",
    "HydrophobicDockModelResultCollection",

    # Validation and source resolution
    "is_dock_model_like",
    "validate_dock_model_like",
    "get_dock_model_pose_identifier",
    "get_dock_model_receptor",
    "get_dock_model_ligand",

    # Existing results
    "get_existing_dock_model_hydrophobic_results",
    "hydrophobic_result_identity_key",
    "merge_hydrophobic_result_entries",
    "get_standardized_dock_model_hydrophobic_results",
    "get_latest_dock_model_hydrophobic_result",

    # Serialization
    "serialize_hydrophobic_analysis_result_for_dock_model",
    "serialize_hydrophobic_multi_pose_result",

    # Score and statistics integration
    "calculate_dock_model_hydrophobic_score",
    "update_dock_model_hydrophobic_score",
    "update_dock_model_combined_score",
    "update_dock_model_hydrophobic_statistics",

    # Attachment
    "HydrophobicDockModelAttachment",
    "attach_hydrophobic_results",
    "refresh_dock_model_hydrophobic_results",

    # Single-pose analysis
    "analyze_dock_model_hydrophobic",

    # Multipose analysis
    "HydrophobicMultiPoseResult",
    "combine_hydrophobic_pose_interactions",
    "summarize_multiple_hydrophobic_results",
    "analyze_multiple_dock_models_hydrophobic",
)

for public_name in _SECTION_11_PUBLIC_NAMES:
    if public_name not in __all__:
        __all__.append(public_name)


# =============================================================================
# End of Section 11
# =============================================================================


