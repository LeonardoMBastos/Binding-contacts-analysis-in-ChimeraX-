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


