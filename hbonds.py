# =============================================================================
# DockAnalyzer — Hydrogen-bond analysis
# Section 1 — Header, imports, aliases and public interface
# =============================================================================

"""
Hydrogen-bond detection and analysis for DockAnalyzer.

This module provides tools for identifying, validating, classifying and
summarizing hydrogen bonds between docked ligands and receptor structures.

The implementation is designed to operate with ChimeraX atomic objects while
remaining compatible with synthetic Python objects used in tests.

Hydrogen-bond detection is separated from the general contact classification
implemented in :mod:`contacts`. The ``contacts`` module identifies spatially
close atom pairs, whereas this module evaluates donor, hydrogen and acceptor
chemistry together with hydrogen-bond geometry.

Notes
-----
The module supports two analysis modes:

1. Explicit-hydrogen mode
   Uses donor-hydrogen-acceptor geometry when bonded hydrogen atoms are
   available.

2. Inferred-hydrogen mode
   Uses donor-acceptor geometry when explicit hydrogens are unavailable.
   Results obtained in this mode must be marked as inferred.

Specialized interaction types such as salt bridges, aromatic interactions and
hydrophobic contacts are intentionally handled by their respective modules.
"""

from __future__ import annotations

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
    NamedTuple,
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

import numpy as np
from numpy.typing import NDArray

try:
    from . import config

    from .contacts import (
        AtomContact,
        ContactAnalysisResult,
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
        angle,
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
        angle,
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

_MODULE_NAME: Final[str] = "hbonds"

_LOGGER: Final[DockLogger] = DockLogger(
    _MODULE_NAME
)


# -----------------------------------------------------------------------------
# Public interface
# -----------------------------------------------------------------------------

__all__: List[str] = []


# -----------------------------------------------------------------------------
# Generic type variables
# -----------------------------------------------------------------------------

T = TypeVar(
    "T"
)

AtomT = TypeVar(
    "AtomT"
)

ResidueT = TypeVar(
    "ResidueT"
)

StructureT = TypeVar(
    "StructureT"
)


# -----------------------------------------------------------------------------
# Numeric aliases
# -----------------------------------------------------------------------------

Number: TypeAlias = Union[
    int,
    float,
    np.integer,
    np.floating,
]

FloatArray: TypeAlias = NDArray[
    np.float64
]

IntegerArray: TypeAlias = NDArray[
    np.int64
]

BooleanArray: TypeAlias = NDArray[
    np.bool_
]

Coordinate: TypeAlias = FloatArray
CoordinateCollection: TypeAlias = FloatArray


# -----------------------------------------------------------------------------
# ChimeraX-compatible object aliases
# -----------------------------------------------------------------------------

AtomLike: TypeAlias = Any
ResidueLike: TypeAlias = Any
StructureLike: TypeAlias = Any
LigandLike: TypeAlias = Any
ReceptorLike: TypeAlias = Any
BondLike: TypeAlias = Any


# -----------------------------------------------------------------------------
# Collection aliases
# -----------------------------------------------------------------------------

AtomCollection: TypeAlias = Sequence[
    AtomLike
]

ResidueCollection: TypeAlias = Sequence[
    ResidueLike
]

StructureCollection: TypeAlias = Sequence[
    StructureLike
]

AtomPair: TypeAlias = Tuple[
    AtomLike,
    AtomLike,
]

AtomTriple: TypeAlias = Tuple[
    AtomLike,
    AtomLike,
    AtomLike,
]

IndexedAtomPair: TypeAlias = Tuple[
    int,
    int,
]

IndexedAtomTriple: TypeAlias = Tuple[
    int,
    int,
    int,
]


# -----------------------------------------------------------------------------
# Hydrogen-bond semantic aliases
# -----------------------------------------------------------------------------

HydrogenBondMode: TypeAlias = Literal[
    "explicit",
    "inferred",
]

HydrogenBondDirection: TypeAlias = Literal[
    "ligand_donor",
    "receptor_donor",
    "unknown",
]

HydrogenBondClassification: TypeAlias = Literal[
    "strong",
    "moderate",
    "weak",
    "unknown",
]

HydrogenBondRole: TypeAlias = Literal[
    "donor",
    "acceptor",
    "hydrogen",
    "none",
    "unknown",
]

HydrogenBondIdentifier: TypeAlias = Tuple[
    str,
    str,
    Optional[str],
]

DonorAcceptorPair: TypeAlias = Tuple[
    AtomLike,
    AtomLike,
]

DonorHydrogenAcceptorTriple: TypeAlias = Tuple[
    AtomLike,
    AtomLike,
    AtomLike,
]


# -----------------------------------------------------------------------------
# Mapping and metadata aliases
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

HydrogenBondMetadata: TypeAlias = Mapping[
    str,
    Any,
]

ElementSet: TypeAlias = FrozenSet[
    str
]

ResidueNameSet: TypeAlias = FrozenSet[
    str
]

AtomNameSet: TypeAlias = FrozenSet[
    str
]


# -----------------------------------------------------------------------------
# Callable aliases
# -----------------------------------------------------------------------------

AtomPredicate: TypeAlias = Callable[
    [
        AtomLike,
    ],
    bool,
]

AtomPairPredicate: TypeAlias = Callable[
    [
        AtomLike,
        AtomLike,
    ],
    bool,
]

BondResolver: TypeAlias = Callable[
    [
        AtomLike,
    ],
    Iterable[
        AtomLike
    ],
]


# -----------------------------------------------------------------------------
# Structural protocols
# -----------------------------------------------------------------------------

@runtime_checkable
class CoordinateProvider(
    Protocol
):
    """
    Protocol for objects exposing Cartesian coordinates.

    Notes
    -----
    ChimeraX atoms are not required to formally inherit from this protocol.
    Structural compatibility is sufficient.
    """

    @property
    def coord(
        self,
    ) -> Any:
        """Return the Cartesian coordinate."""


@runtime_checkable
class ElementProvider(
    Protocol
):
    """Protocol for objects exposing element information."""

    @property
    def element(
        self,
    ) -> Any:
        """Return an element-like object."""


@runtime_checkable
class ResidueProvider(
    Protocol
):
    """Protocol for objects exposing a parent residue."""

    @property
    def residue(
        self,
    ) -> Any:
        """Return the parent residue."""


@runtime_checkable
class NeighborProvider(
    Protocol
):
    """Protocol for atom-like objects exposing bonded neighbors."""

    @property
    def neighbors(
        self,
    ) -> Any:
        """Return bonded neighboring atoms."""


# -----------------------------------------------------------------------------
# Empty immutable objects
# -----------------------------------------------------------------------------

_EMPTY_METADATA: Final[
    Mapping[
        str,
        Any,
    ]
] = MappingProxyType(
    {}
)

_EMPTY_ATOM_TUPLE: Final[
    Tuple[
        AtomLike,
        ...,
    ]
] = ()

_EMPTY_HBOND_TUPLE: Final[
    Tuple[
        Any,
        ...,
    ]
] = ()

_EMPTY_RESIDUE_KEY_TUPLE: Final[
    Tuple[
        ResidueContactKey,
        ...,
    ]
] = ()


# -----------------------------------------------------------------------------
# Initial public names
# -----------------------------------------------------------------------------

_SECTION_1_PUBLIC_NAMES: Final[
    Tuple[
        str,
        ...,
    ]
] = (
    "Number",
    "FloatArray",
    "IntegerArray",
    "BooleanArray",
    "Coordinate",
    "CoordinateCollection",
    "AtomLike",
    "ResidueLike",
    "StructureLike",
    "LigandLike",
    "ReceptorLike",
    "BondLike",
    "AtomCollection",
    "ResidueCollection",
    "StructureCollection",
    "AtomPair",
    "AtomTriple",
    "IndexedAtomPair",
    "IndexedAtomTriple",
    "HydrogenBondMode",
    "HydrogenBondDirection",
    "HydrogenBondClassification",
    "HydrogenBondRole",
    "HydrogenBondIdentifier",
    "DonorAcceptorPair",
    "DonorHydrogenAcceptorTriple",
    "Metadata",
    "MutableMetadata",
    "Statistics",
    "HydrogenBondMetadata",
    "ElementSet",
    "ResidueNameSet",
    "AtomNameSet",
    "AtomPredicate",
    "AtomPairPredicate",
    "BondResolver",
    "CoordinateProvider",
    "ElementProvider",
    "ResidueProvider",
    "NeighborProvider",
)

for public_name in _SECTION_1_PUBLIC_NAMES:
    if public_name not in __all__:
        __all__.append(
            public_name
        )


# =============================================================================
# End of Section 1
# =============================================================================



# =============================================================================
# Section 2 — Geometric and chemical constants
# =============================================================================


# -----------------------------------------------------------------------------
# Hydrogen-bond modes
# -----------------------------------------------------------------------------

HBOND_MODE_EXPLICIT: Final[
    HydrogenBondMode
] = "explicit"

HBOND_MODE_INFERRED: Final[
    HydrogenBondMode
] = "inferred"


_VALID_HBOND_MODES: Final[
    FrozenSet[
        str
    ]
] = frozenset(
    {
        HBOND_MODE_EXPLICIT,
        HBOND_MODE_INFERRED,
    }
)


# -----------------------------------------------------------------------------
# Hydrogen-bond directions
# -----------------------------------------------------------------------------

HBOND_DIRECTION_UNKNOWN: Final[
    HydrogenBondDirection
] = "unknown"

HBOND_DIRECTION_LIGAND_DONOR: Final[
    HydrogenBondDirection
] = "ligand_donor"

HBOND_DIRECTION_RECEPTOR_DONOR: Final[
    HydrogenBondDirection
] = "receptor_donor"


_VALID_HBOND_DIRECTIONS: Final[
    FrozenSet[
        str
    ]
] = frozenset(
    {
        HBOND_DIRECTION_UNKNOWN,
        HBOND_DIRECTION_LIGAND_DONOR,
        HBOND_DIRECTION_RECEPTOR_DONOR,
    }
)


# -----------------------------------------------------------------------------
# Hydrogen-bond classifications
# -----------------------------------------------------------------------------

HBOND_TYPE_UNKNOWN: Final[
    HydrogenBondClassification
] = "unknown"

HBOND_TYPE_WEAK: Final[
    HydrogenBondClassification
] = "weak"

HBOND_TYPE_MODERATE: Final[
    HydrogenBondClassification
] = "moderate"

HBOND_TYPE_STRONG: Final[
    HydrogenBondClassification
] = "strong"


_VALID_HBOND_CLASSIFICATIONS: Final[
    FrozenSet[
        str
    ]
] = frozenset(
    {
        HBOND_TYPE_UNKNOWN,
        HBOND_TYPE_WEAK,
        HBOND_TYPE_MODERATE,
        HBOND_TYPE_STRONG,
    }
)


# -----------------------------------------------------------------------------
# Hydrogen-bond atom roles
# -----------------------------------------------------------------------------

HBOND_ROLE_DONOR: Final[
    HydrogenBondRole
] = "donor"

HBOND_ROLE_ACCEPTOR: Final[
    HydrogenBondRole
] = "acceptor"

HBOND_ROLE_HYDROGEN: Final[
    HydrogenBondRole
] = "hydrogen"

HBOND_ROLE_NONE: Final[
    HydrogenBondRole
] = "none"

HBOND_ROLE_UNKNOWN: Final[
    HydrogenBondRole
] = "unknown"


_VALID_HBOND_ROLES: Final[
    FrozenSet[
        str
    ]
] = frozenset(
    {
        HBOND_ROLE_DONOR,
        HBOND_ROLE_ACCEPTOR,
        HBOND_ROLE_HYDROGEN,
        HBOND_ROLE_NONE,
        HBOND_ROLE_UNKNOWN,
    }
)


# -----------------------------------------------------------------------------
# Universal angular constants
# -----------------------------------------------------------------------------

FULL_ROTATION_DEGREES: Final[
    np.float64
] = np.float64(
    360.0
)

STRAIGHT_ANGLE_DEGREES: Final[
    np.float64
] = np.float64(
    180.0
)

RIGHT_ANGLE_DEGREES: Final[
    np.float64
] = np.float64(
    90.0
)

MINIMUM_VALID_ANGLE_DEGREES: Final[
    np.float64
] = np.float64(
    0.0
)

MAXIMUM_VALID_ANGLE_DEGREES: Final[
    np.float64
] = STRAIGHT_ANGLE_DEGREES


# -----------------------------------------------------------------------------
# Default geometric thresholds
# -----------------------------------------------------------------------------

DEFAULT_DONOR_ACCEPTOR_DISTANCE: Final[
    np.float64
] = np.float64(
    3.50
)

DEFAULT_HYDROGEN_ACCEPTOR_DISTANCE: Final[
    np.float64
] = np.float64(
    2.50
)

DEFAULT_MINIMUM_DHA_ANGLE: Final[
    np.float64
] = np.float64(
    120.0
)

DEFAULT_MINIMUM_INFERRED_ANGLE: Final[
    np.float64
] = np.float64(
    90.0
)

DEFAULT_STRONG_DHA_ANGLE: Final[
    np.float64
] = np.float64(
    150.0
)

DEFAULT_MODERATE_DHA_ANGLE: Final[
    np.float64
] = np.float64(
    135.0
)

DEFAULT_WEAK_DHA_ANGLE: Final[
    np.float64
] = DEFAULT_MINIMUM_DHA_ANGLE

DEFAULT_DISTANCE_TOLERANCE: Final[
    np.float64
] = np.float64(
    0.05
)

DEFAULT_ANGLE_TOLERANCE: Final[
    np.float64
] = np.float64(
    1.0
)


# -----------------------------------------------------------------------------
# Distance thresholds used for geometric classification
# -----------------------------------------------------------------------------

HBOND_STRONG_MAX_DONOR_ACCEPTOR_DISTANCE: Final[
    np.float64
] = np.float64(
    3.00
)

HBOND_MODERATE_MAX_DONOR_ACCEPTOR_DISTANCE: Final[
    np.float64
] = np.float64(
    3.20
)

HBOND_WEAK_MAX_DONOR_ACCEPTOR_DISTANCE: Final[
    np.float64
] = DEFAULT_DONOR_ACCEPTOR_DISTANCE


HBOND_STRONG_MAX_HYDROGEN_ACCEPTOR_DISTANCE: Final[
    np.float64
] = np.float64(
    2.00
)

HBOND_MODERATE_MAX_HYDROGEN_ACCEPTOR_DISTANCE: Final[
    np.float64
] = np.float64(
    2.30
)

HBOND_WEAK_MAX_HYDROGEN_ACCEPTOR_DISTANCE: Final[
    np.float64
] = DEFAULT_HYDROGEN_ACCEPTOR_DISTANCE


# -----------------------------------------------------------------------------
# Angle thresholds used for geometric classification
# -----------------------------------------------------------------------------

HBOND_STRONG_MIN_DHA_ANGLE: Final[
    np.float64
] = DEFAULT_STRONG_DHA_ANGLE

HBOND_MODERATE_MIN_DHA_ANGLE: Final[
    np.float64
] = DEFAULT_MODERATE_DHA_ANGLE

HBOND_WEAK_MIN_DHA_ANGLE: Final[
    np.float64
] = DEFAULT_WEAK_DHA_ANGLE


# -----------------------------------------------------------------------------
# Chemically relevant elements
# -----------------------------------------------------------------------------

HYDROGEN_ELEMENT: Final[
    str
] = "H"

CARBON_ELEMENT: Final[
    str
] = "C"

NITROGEN_ELEMENT: Final[
    str
] = "N"

OXYGEN_ELEMENT: Final[
    str
] = "O"

SULFUR_ELEMENT: Final[
    str
] = "S"

PHOSPHORUS_ELEMENT: Final[
    str
] = "P"


DONOR_ELEMENTS: Final[
    ElementSet
] = frozenset(
    {
        NITROGEN_ELEMENT,
        OXYGEN_ELEMENT,
        SULFUR_ELEMENT,
    }
)


ACCEPTOR_ELEMENTS: Final[
    ElementSet
] = frozenset(
    {
        NITROGEN_ELEMENT,
        OXYGEN_ELEMENT,
        SULFUR_ELEMENT,
    }
)


WEAK_ACCEPTOR_ELEMENTS: Final[
    ElementSet
] = frozenset(
    {
        "F",
        "CL",
        "BR",
        "I",
    }
)


HALOGEN_ELEMENTS: Final[
    ElementSet
] = frozenset(
    {
        "F",
        "CL",
        "BR",
        "I",
    }
)


COMMON_METAL_ELEMENTS: Final[
    ElementSet
] = frozenset(
    {
        "LI",
        "NA",
        "K",
        "RB",
        "CS",
        "BE",
        "MG",
        "CA",
        "SR",
        "BA",
        "AL",
        "MN",
        "FE",
        "CO",
        "NI",
        "CU",
        "ZN",
        "CD",
        "HG",
    }
)


COMMON_METAL_ATOMIC_NUMBERS: Final[
    FrozenSet[
        int
    ]
] = frozenset(
    {
        3,
        4,
        11,
        12,
        13,
        19,
        20,
        25,
        26,
        27,
        28,
        29,
        30,
        37,
        38,
        48,
        55,
        56,
        80,
    }
)


# -----------------------------------------------------------------------------
# Atomic-number constants
# -----------------------------------------------------------------------------

HYDROGEN_ATOMIC_NUMBER: Final[
    int
] = 1

CARBON_ATOMIC_NUMBER: Final[
    int
] = 6

NITROGEN_ATOMIC_NUMBER: Final[
    int
] = 7

OXYGEN_ATOMIC_NUMBER: Final[
    int
] = 8

PHOSPHORUS_ATOMIC_NUMBER: Final[
    int
] = 15

SULFUR_ATOMIC_NUMBER: Final[
    int
] = 16


DONOR_ATOMIC_NUMBERS: Final[
    FrozenSet[
        int
    ]
] = frozenset(
    {
        NITROGEN_ATOMIC_NUMBER,
        OXYGEN_ATOMIC_NUMBER,
        SULFUR_ATOMIC_NUMBER,
    }
)


ACCEPTOR_ATOMIC_NUMBERS: Final[
    FrozenSet[
        int
    ]
] = frozenset(
    {
        NITROGEN_ATOMIC_NUMBER,
        OXYGEN_ATOMIC_NUMBER,
        SULFUR_ATOMIC_NUMBER,
    }
)


# -----------------------------------------------------------------------------
# Residue classes used by later chemical-perception routines
# -----------------------------------------------------------------------------

STANDARD_AMINO_ACID_NAMES: Final[
    ResidueNameSet
] = frozenset(
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


HISTIDINE_RESIDUE_NAMES: Final[
    ResidueNameSet
] = frozenset(
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


CYSTEINE_RESIDUE_NAMES: Final[
    ResidueNameSet
] = frozenset(
    {
        "CYS",
        "CYM",
        "CYX",
    }
)


WATER_RESIDUE_NAMES: Final[
    ResidueNameSet
] = frozenset(
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


# -----------------------------------------------------------------------------
# Generic atom-name constants
# -----------------------------------------------------------------------------

BACKBONE_NITROGEN_ATOM_NAME: Final[
    str
] = "N"

BACKBONE_OXYGEN_ATOM_NAME: Final[
    str
] = "O"

BACKBONE_TERMINAL_OXYGEN_ATOM_NAME: Final[
    str
] = "OXT"

BACKBONE_CARBON_ATOM_NAME: Final[
    str
] = "C"

ALPHA_CARBON_ATOM_NAME: Final[
    str
] = "CA"


# -----------------------------------------------------------------------------
# Numerically safe limits
# -----------------------------------------------------------------------------

MINIMUM_POSITIVE_DISTANCE: Final[
    np.float64
] = np.float64(
    1.0e-8
)

MINIMUM_VECTOR_NORM: Final[
    np.float64
] = np.float64(
    1.0e-12
)

DEFAULT_COORDINATE_DECIMALS: Final[
    int
] = 6

DEFAULT_DISTANCE_DECIMALS: Final[
    int
] = 3

DEFAULT_ANGLE_DECIMALS: Final[
    int
] = 2


# -----------------------------------------------------------------------------
# Search and processing limits
# -----------------------------------------------------------------------------

DEFAULT_MAXIMUM_PAIR_ELEMENTS: Final[
    int
] = 4_000_000

DEFAULT_HBOND_BLOCK_SIZE: Final[
    int
] = 1_024

DEFAULT_MAXIMUM_HYDROGEN_BONDS: Final[
    Optional[
        int
    ]
] = None


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
    candidate_names : sequence of str
        Candidate configuration attribute names.
    default : T
        Fallback value.

    Returns
    -------
    T
        Configured value or ``default``.

    Notes
    -----
    This helper intentionally accepts multiple names to preserve compatibility
    with possible future changes in ``config.py``.
    """

    for candidate_name in candidate_names:
        try:
            value = getattr(
                config,
                candidate_name,
            )

        except (
            AttributeError,
            TypeError,
        ):
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
    Convert a configuration value to a finite positive float.

    Parameters
    ----------
    value : Any
        Candidate numeric value.
    name : str
        Human-readable parameter name.
    default : numpy.float64
        Fallback value.
    allow_zero : bool, optional
        Whether zero is accepted.

    Returns
    -------
    numpy.float64
        Validated numeric value.
    """

    try:
        numeric_value = np.float64(
            value
        )

    except (
        TypeError,
        ValueError,
        OverflowError,
    ):
        try:
            _LOGGER.warning(
                f"Invalid configured value for {name!r}; "
                f"using default {float(default):g}."
            )

        except Exception:
            pass

        return np.float64(
            default
        )

    if not np.isfinite(
        numeric_value
    ):
        try:
            _LOGGER.warning(
                f"Non-finite configured value for {name!r}; "
                f"using default {float(default):g}."
            )

        except Exception:
            pass

        return np.float64(
            default
        )

    minimum_allowed = (
        np.float64(0.0)
        if allow_zero
        else MINIMUM_POSITIVE_DISTANCE
    )

    if numeric_value < minimum_allowed:
        try:
            _LOGGER.warning(
                f"Configured value for {name!r} is outside "
                f"the accepted range; using default "
                f"{float(default):g}."
            )

        except Exception:
            pass

        return np.float64(
            default
        )

    return numeric_value


def _coerce_angle(
    value: Any,
    *,
    name: str,
    default: np.float64,
) -> np.float64:
    """
    Convert a configuration value to a valid angle in degrees.

    Parameters
    ----------
    value : Any
        Candidate angle.
    name : str
        Human-readable parameter name.
    default : numpy.float64
        Fallback angle.

    Returns
    -------
    numpy.float64
        Angle between zero and 180 degrees.
    """

    try:
        numeric_value = np.float64(
            value
        )

    except (
        TypeError,
        ValueError,
        OverflowError,
    ):
        try:
            _LOGGER.warning(
                f"Invalid configured angle for {name!r}; "
                f"using default {float(default):g} degrees."
            )

        except Exception:
            pass

        return np.float64(
            default
        )

    if (
        not np.isfinite(
            numeric_value
        )
        or numeric_value
        < MINIMUM_VALID_ANGLE_DEGREES
        or numeric_value
        > MAXIMUM_VALID_ANGLE_DEGREES
    ):
        try:
            _LOGGER.warning(
                f"Configured angle for {name!r} is outside "
                "the valid interval [0, 180]; "
                f"using default {float(default):g} degrees."
            )

        except Exception:
            pass

        return np.float64(
            default
        )

    return numeric_value


def _coerce_positive_integer(
    value: Any,
    *,
    name: str,
    default: int,
) -> int:
    """
    Convert a configuration value to a positive integer.

    Parameters
    ----------
    value : Any
        Candidate value.
    name : str
        Human-readable parameter name.
    default : int
        Fallback integer.

    Returns
    -------
    int
        Positive integer.
    """

    if isinstance(
        value,
        bool,
    ):
        return int(
            default
        )

    try:
        integer_value = int(
            value
        )

    except (
        TypeError,
        ValueError,
        OverflowError,
    ):
        try:
            _LOGGER.warning(
                f"Invalid configured value for {name!r}; "
                f"using default {default}."
            )

        except Exception:
            pass

        return int(
            default
        )

    if integer_value <= 0:
        try:
            _LOGGER.warning(
                f"Configured value for {name!r} must be positive; "
                f"using default {default}."
            )

        except Exception:
            pass

        return int(
            default
        )

    return integer_value


# -----------------------------------------------------------------------------
# Public default-resolution functions
# -----------------------------------------------------------------------------

def get_default_donor_acceptor_distance(
) -> np.float64:
    """
    Return the configured donor-acceptor distance cutoff.

    Returns
    -------
    numpy.float64
        Donor-acceptor cutoff in angstroms.
    """

    value = _get_config_value(
        (
            "HBOND_DONOR_ACCEPTOR_DISTANCE",
            "HBOND_DONOR_ACCEPTOR_CUTOFF",
            "DEFAULT_HBOND_DONOR_ACCEPTOR_DISTANCE",
            "DEFAULT_DONOR_ACCEPTOR_DISTANCE",
        ),
        DEFAULT_DONOR_ACCEPTOR_DISTANCE,
    )

    return _coerce_positive_float(
        value,
        name="donor-acceptor distance",
        default=DEFAULT_DONOR_ACCEPTOR_DISTANCE,
    )


def get_default_hydrogen_acceptor_distance(
) -> np.float64:
    """
    Return the configured hydrogen-acceptor distance cutoff.

    Returns
    -------
    numpy.float64
        Hydrogen-acceptor cutoff in angstroms.
    """

    value = _get_config_value(
        (
            "HBOND_HYDROGEN_ACCEPTOR_DISTANCE",
            "HBOND_HYDROGEN_ACCEPTOR_CUTOFF",
            "DEFAULT_HBOND_HYDROGEN_ACCEPTOR_DISTANCE",
            "DEFAULT_HYDROGEN_ACCEPTOR_DISTANCE",
        ),
        DEFAULT_HYDROGEN_ACCEPTOR_DISTANCE,
    )

    return _coerce_positive_float(
        value,
        name="hydrogen-acceptor distance",
        default=DEFAULT_HYDROGEN_ACCEPTOR_DISTANCE,
    )


def get_default_minimum_dha_angle(
) -> np.float64:
    """
    Return the configured minimum donor-hydrogen-acceptor angle.

    Returns
    -------
    numpy.float64
        Minimum D-H...A angle in degrees.
    """

    value = _get_config_value(
        (
            "HBOND_MINIMUM_DHA_ANGLE",
            "HBOND_MIN_DHA_ANGLE",
            "DEFAULT_HBOND_MINIMUM_DHA_ANGLE",
            "DEFAULT_MINIMUM_DHA_ANGLE",
        ),
        DEFAULT_MINIMUM_DHA_ANGLE,
    )

    return _coerce_angle(
        value,
        name="minimum D-H-A angle",
        default=DEFAULT_MINIMUM_DHA_ANGLE,
    )


def get_default_minimum_inferred_angle(
) -> np.float64:
    """
    Return the minimum angle used for inferred hydrogen bonds.

    Returns
    -------
    numpy.float64
        Minimum inferred angle in degrees.
    """

    value = _get_config_value(
        (
            "HBOND_MINIMUM_INFERRED_ANGLE",
            "HBOND_MIN_INFERRED_ANGLE",
            "DEFAULT_HBOND_MINIMUM_INFERRED_ANGLE",
            "DEFAULT_MINIMUM_INFERRED_ANGLE",
        ),
        DEFAULT_MINIMUM_INFERRED_ANGLE,
    )

    return _coerce_angle(
        value,
        name="minimum inferred hydrogen-bond angle",
        default=DEFAULT_MINIMUM_INFERRED_ANGLE,
    )


def get_default_strong_dha_angle(
) -> np.float64:
    """
    Return the configured strong hydrogen-bond angle threshold.

    Returns
    -------
    numpy.float64
        Strong D-H...A angle threshold in degrees.
    """

    value = _get_config_value(
        (
            "HBOND_STRONG_DHA_ANGLE",
            "HBOND_STRONG_MINIMUM_ANGLE",
            "DEFAULT_HBOND_STRONG_DHA_ANGLE",
            "DEFAULT_STRONG_DHA_ANGLE",
        ),
        DEFAULT_STRONG_DHA_ANGLE,
    )

    return _coerce_angle(
        value,
        name="strong D-H-A angle",
        default=DEFAULT_STRONG_DHA_ANGLE,
    )


def get_default_moderate_dha_angle(
) -> np.float64:
    """
    Return the configured moderate hydrogen-bond angle threshold.

    Returns
    -------
    numpy.float64
        Moderate D-H...A angle threshold in degrees.
    """

    value = _get_config_value(
        (
            "HBOND_MODERATE_DHA_ANGLE",
            "HBOND_MODERATE_MINIMUM_ANGLE",
            "DEFAULT_HBOND_MODERATE_DHA_ANGLE",
            "DEFAULT_MODERATE_DHA_ANGLE",
        ),
        DEFAULT_MODERATE_DHA_ANGLE,
    )

    return _coerce_angle(
        value,
        name="moderate D-H-A angle",
        default=DEFAULT_MODERATE_DHA_ANGLE,
    )


def get_default_distance_tolerance(
) -> np.float64:
    """
    Return the configured geometric distance tolerance.

    Returns
    -------
    numpy.float64
        Distance tolerance in angstroms.
    """

    value = _get_config_value(
        (
            "HBOND_DISTANCE_TOLERANCE",
            "DEFAULT_HBOND_DISTANCE_TOLERANCE",
            "DEFAULT_DISTANCE_TOLERANCE",
        ),
        DEFAULT_DISTANCE_TOLERANCE,
    )

    return _coerce_positive_float(
        value,
        name="hydrogen-bond distance tolerance",
        default=DEFAULT_DISTANCE_TOLERANCE,
        allow_zero=True,
    )


def get_default_angle_tolerance(
) -> np.float64:
    """
    Return the configured geometric angle tolerance.

    Returns
    -------
    numpy.float64
        Angle tolerance in degrees.
    """

    value = _get_config_value(
        (
            "HBOND_ANGLE_TOLERANCE",
            "DEFAULT_HBOND_ANGLE_TOLERANCE",
            "DEFAULT_ANGLE_TOLERANCE",
        ),
        DEFAULT_ANGLE_TOLERANCE,
    )

    return _coerce_positive_float(
        value,
        name="hydrogen-bond angle tolerance",
        default=DEFAULT_ANGLE_TOLERANCE,
        allow_zero=True,
    )


def get_default_hbond_block_size(
) -> int:
    """
    Return the configured hydrogen-bond processing block size.

    Returns
    -------
    int
        Positive processing block size.
    """

    value = _get_config_value(
        (
            "HBOND_BLOCK_SIZE",
            "DEFAULT_HBOND_BLOCK_SIZE",
            "DEFAULT_BLOCK_SIZE",
        ),
        DEFAULT_HBOND_BLOCK_SIZE,
    )

    return _coerce_positive_integer(
        value,
        name="hydrogen-bond block size",
        default=DEFAULT_HBOND_BLOCK_SIZE,
    )


def get_default_maximum_pair_elements(
) -> int:
    """
    Return the configured full pair-matrix element limit.

    Returns
    -------
    int
        Maximum number of donor-acceptor matrix elements.
    """

    value = _get_config_value(
        (
            "HBOND_MAXIMUM_PAIR_ELEMENTS",
            "HBOND_MAX_MATRIX_ELEMENTS",
            "DEFAULT_HBOND_MAXIMUM_PAIR_ELEMENTS",
            "DEFAULT_MAXIMUM_PAIR_ELEMENTS",
        ),
        DEFAULT_MAXIMUM_PAIR_ELEMENTS,
    )

    return _coerce_positive_integer(
        value,
        name="maximum hydrogen-bond pair elements",
        default=DEFAULT_MAXIMUM_PAIR_ELEMENTS,
    )


# -----------------------------------------------------------------------------
# Validation helpers
# -----------------------------------------------------------------------------

def validate_hydrogen_bond_mode(
    mode: str,
) -> HydrogenBondMode:
    """
    Validate and normalize a hydrogen-bond analysis mode.

    Parameters
    ----------
    mode : str
        Hydrogen-bond mode.

    Returns
    -------
    HydrogenBondMode
        Normalized mode.

    Raises
    ------
    TypeError
        If ``mode`` is not a string.
    ValueError
        If the mode is unsupported.
    """

    if not isinstance(
        mode,
        str,
    ):
        raise TypeError(
            "Hydrogen-bond mode must be a string."
        )

    normalized_mode = mode.strip().lower()

    if normalized_mode not in _VALID_HBOND_MODES:
        valid_modes = ", ".join(
            sorted(
                _VALID_HBOND_MODES
            )
        )

        raise ValueError(
            f"Unsupported hydrogen-bond mode {mode!r}. "
            f"Expected one of: {valid_modes}."
        )

    return normalized_mode  # type: ignore[return-value]


def validate_hydrogen_bond_direction(
    direction: str,
) -> HydrogenBondDirection:
    """
    Validate and normalize a hydrogen-bond direction.

    Parameters
    ----------
    direction : str
        Hydrogen-bond direction.

    Returns
    -------
    HydrogenBondDirection
        Normalized direction.

    Raises
    ------
    TypeError
        If ``direction`` is not a string.
    ValueError
        If the direction is unsupported.
    """

    if not isinstance(
        direction,
        str,
    ):
        raise TypeError(
            "Hydrogen-bond direction must be a string."
        )

    normalized_direction = (
        direction.strip().lower()
    )

    if (
        normalized_direction
        not in _VALID_HBOND_DIRECTIONS
    ):
        valid_directions = ", ".join(
            sorted(
                _VALID_HBOND_DIRECTIONS
            )
        )

        raise ValueError(
            f"Unsupported hydrogen-bond direction "
            f"{direction!r}. Expected one of: "
            f"{valid_directions}."
        )

    return normalized_direction  # type: ignore[return-value]


def validate_hydrogen_bond_classification(
    classification: str,
) -> HydrogenBondClassification:
    """
    Validate and normalize a hydrogen-bond classification.

    Parameters
    ----------
    classification : str
        Hydrogen-bond classification.

    Returns
    -------
    HydrogenBondClassification
        Normalized classification.

    Raises
    ------
    TypeError
        If ``classification`` is not a string.
    ValueError
        If the classification is unsupported.
    """

    if not isinstance(
        classification,
        str,
    ):
        raise TypeError(
            "Hydrogen-bond classification must be a string."
        )

    normalized_classification = (
        classification.strip().lower()
    )

    if (
        normalized_classification
        not in _VALID_HBOND_CLASSIFICATIONS
    ):
        valid_classifications = ", ".join(
            sorted(
                _VALID_HBOND_CLASSIFICATIONS
            )
        )

        raise ValueError(
            f"Unsupported hydrogen-bond classification "
            f"{classification!r}. Expected one of: "
            f"{valid_classifications}."
        )

    return normalized_classification  # type: ignore[return-value]


def validate_hydrogen_bond_role(
    role: str,
) -> HydrogenBondRole:
    """
    Validate and normalize a hydrogen-bond atom role.

    Parameters
    ----------
    role : str
        Atom role.

    Returns
    -------
    HydrogenBondRole
        Normalized atom role.

    Raises
    ------
    TypeError
        If ``role`` is not a string.
    ValueError
        If the role is unsupported.
    """

    if not isinstance(
        role,
        str,
    ):
        raise TypeError(
            "Hydrogen-bond role must be a string."
        )

    normalized_role = role.strip().lower()

    if normalized_role not in _VALID_HBOND_ROLES:
        valid_roles = ", ".join(
            sorted(
                _VALID_HBOND_ROLES
            )
        )

        raise ValueError(
            f"Unsupported hydrogen-bond role {role!r}. "
            f"Expected one of: {valid_roles}."
        )

    return normalized_role  # type: ignore[return-value]


# -----------------------------------------------------------------------------
# Empty immutable collections
# -----------------------------------------------------------------------------

_EMPTY_HYDROGEN_BOND_LIST: Final[
    Tuple[
        Any,
        ...,
    ]
] = ()

_EMPTY_DONOR_LIST: Final[
    Tuple[
        AtomLike,
        ...,
    ]
] = ()

_EMPTY_ACCEPTOR_LIST: Final[
    Tuple[
        AtomLike,
        ...,
    ]
] = ()

_EMPTY_HYDROGEN_LIST: Final[
    Tuple[
        AtomLike,
        ...,
    ]
] = ()


# -----------------------------------------------------------------------------
# Public interface
# -----------------------------------------------------------------------------

_SECTION_2_PUBLIC_NAMES: Final[
    Tuple[
        str,
        ...,
    ]
] = (
    "HBOND_MODE_EXPLICIT",
    "HBOND_MODE_INFERRED",
    "HBOND_DIRECTION_UNKNOWN",
    "HBOND_DIRECTION_LIGAND_DONOR",
    "HBOND_DIRECTION_RECEPTOR_DONOR",
    "HBOND_TYPE_UNKNOWN",
    "HBOND_TYPE_WEAK",
    "HBOND_TYPE_MODERATE",
    "HBOND_TYPE_STRONG",
    "HBOND_ROLE_DONOR",
    "HBOND_ROLE_ACCEPTOR",
    "HBOND_ROLE_HYDROGEN",
    "HBOND_ROLE_NONE",
    "HBOND_ROLE_UNKNOWN",
    "FULL_ROTATION_DEGREES",
    "STRAIGHT_ANGLE_DEGREES",
    "RIGHT_ANGLE_DEGREES",
    "DEFAULT_DONOR_ACCEPTOR_DISTANCE",
    "DEFAULT_HYDROGEN_ACCEPTOR_DISTANCE",
    "DEFAULT_MINIMUM_DHA_ANGLE",
    "DEFAULT_MINIMUM_INFERRED_ANGLE",
    "DEFAULT_STRONG_DHA_ANGLE",
    "DEFAULT_MODERATE_DHA_ANGLE",
    "DEFAULT_WEAK_DHA_ANGLE",
    "DEFAULT_DISTANCE_TOLERANCE",
    "DEFAULT_ANGLE_TOLERANCE",
    "HBOND_STRONG_MAX_DONOR_ACCEPTOR_DISTANCE",
    "HBOND_MODERATE_MAX_DONOR_ACCEPTOR_DISTANCE",
    "HBOND_WEAK_MAX_DONOR_ACCEPTOR_DISTANCE",
    "HBOND_STRONG_MAX_HYDROGEN_ACCEPTOR_DISTANCE",
    "HBOND_MODERATE_MAX_HYDROGEN_ACCEPTOR_DISTANCE",
    "HBOND_WEAK_MAX_HYDROGEN_ACCEPTOR_DISTANCE",
    "HBOND_STRONG_MIN_DHA_ANGLE",
    "HBOND_MODERATE_MIN_DHA_ANGLE",
    "HBOND_WEAK_MIN_DHA_ANGLE",
    "HYDROGEN_ELEMENT",
    "CARBON_ELEMENT",
    "NITROGEN_ELEMENT",
    "OXYGEN_ELEMENT",
    "SULFUR_ELEMENT",
    "PHOSPHORUS_ELEMENT",
    "DONOR_ELEMENTS",
    "ACCEPTOR_ELEMENTS",
    "WEAK_ACCEPTOR_ELEMENTS",
    "HALOGEN_ELEMENTS",
    "COMMON_METAL_ELEMENTS",
    "COMMON_METAL_ATOMIC_NUMBERS",
    "HYDROGEN_ATOMIC_NUMBER",
    "CARBON_ATOMIC_NUMBER",
    "NITROGEN_ATOMIC_NUMBER",
    "OXYGEN_ATOMIC_NUMBER",
    "PHOSPHORUS_ATOMIC_NUMBER",
    "SULFUR_ATOMIC_NUMBER",
    "DONOR_ATOMIC_NUMBERS",
    "ACCEPTOR_ATOMIC_NUMBERS",
    "STANDARD_AMINO_ACID_NAMES",
    "HISTIDINE_RESIDUE_NAMES",
    "CYSTEINE_RESIDUE_NAMES",
    "WATER_RESIDUE_NAMES",
    "BACKBONE_NITROGEN_ATOM_NAME",
    "BACKBONE_OXYGEN_ATOM_NAME",
    "BACKBONE_TERMINAL_OXYGEN_ATOM_NAME",
    "BACKBONE_CARBON_ATOM_NAME",
    "ALPHA_CARBON_ATOM_NAME",
    "MINIMUM_POSITIVE_DISTANCE",
    "MINIMUM_VECTOR_NORM",
    "DEFAULT_COORDINATE_DECIMALS",
    "DEFAULT_DISTANCE_DECIMALS",
    "DEFAULT_ANGLE_DECIMALS",
    "DEFAULT_MAXIMUM_PAIR_ELEMENTS",
    "DEFAULT_HBOND_BLOCK_SIZE",
    "DEFAULT_MAXIMUM_HYDROGEN_BONDS",
    "get_default_donor_acceptor_distance",
    "get_default_hydrogen_acceptor_distance",
    "get_default_minimum_dha_angle",
    "get_default_minimum_inferred_angle",
    "get_default_strong_dha_angle",
    "get_default_moderate_dha_angle",
    "get_default_distance_tolerance",
    "get_default_angle_tolerance",
    "get_default_hbond_block_size",
    "get_default_maximum_pair_elements",
    "validate_hydrogen_bond_mode",
    "validate_hydrogen_bond_direction",
    "validate_hydrogen_bond_classification",
    "validate_hydrogen_bond_role",
)

for public_name in _SECTION_2_PUBLIC_NAMES:
    if public_name not in __all__:
        __all__.append(
            public_name
        )


# =============================================================================
# End of Section 2
# =============================================================================

# =============================================================================
# Section 3 — Result dataclasses
# =============================================================================


# -----------------------------------------------------------------------------
# Local normalization helpers
# -----------------------------------------------------------------------------

def _freeze_metadata(
    metadata: Optional[
        Mapping[
            str,
            Any,
        ]
    ],
) -> Mapping[
    str,
    Any,
]:
    """
    Create an immutable shallow copy of a metadata mapping.

    Parameters
    ----------
    metadata : mapping or None
        Metadata to normalize.

    Returns
    -------
    mapping
        Immutable metadata mapping.

    Raises
    ------
    TypeError
        If ``metadata`` is not a mapping.
    """

    if metadata is None:
        return _EMPTY_METADATA

    if not isinstance(
        metadata,
        Mapping,
    ):
        raise TypeError(
            "Metadata must be a mapping or None."
        )

    if not metadata:
        return _EMPTY_METADATA

    return MappingProxyType(
        dict(
            metadata
        )
    )


def _optional_float64(
    value: Optional[
        Number
    ],
    *,
    name: str,
    minimum: Optional[
        Number
    ] = None,
    maximum: Optional[
        Number
    ] = None,
) -> Optional[
    np.float64
]:
    """
    Normalize an optional numeric value as ``numpy.float64``.

    Parameters
    ----------
    value : Number or None
        Numeric value.
    name : str
        Human-readable field name.
    minimum : Number or None, optional
        Inclusive minimum value.
    maximum : Number or None, optional
        Inclusive maximum value.

    Returns
    -------
    numpy.float64 or None
        Normalized value.

    Raises
    ------
    TypeError
        If ``value`` is not numeric.
    ValueError
        If ``value`` is non-finite or outside the accepted interval.
    """

    if value is None:
        return None

    if isinstance(
        value,
        (
            bool,
            np.bool_,
        ),
    ):
        raise TypeError(
            f"{name} must be numeric, not boolean."
        )

    try:
        normalized = np.float64(
            value
        )

    except (
        TypeError,
        ValueError,
        OverflowError,
    ) as error:
        raise TypeError(
            f"{name} must be a valid numeric value."
        ) from error

    if not np.isfinite(
        normalized
    ):
        raise ValueError(
            f"{name} must be finite."
        )

    if (
        minimum is not None
        and normalized
        < np.float64(
            minimum
        )
    ):
        raise ValueError(
            f"{name} must be greater than or equal to "
            f"{float(np.float64(minimum)):g}."
        )

    if (
        maximum is not None
        and normalized
        > np.float64(
            maximum
        )
    ):
        raise ValueError(
            f"{name} must be less than or equal to "
            f"{float(np.float64(maximum)):g}."
        )

    return normalized


def _optional_nonnegative_integer(
    value: Optional[
        Number
    ],
    *,
    name: str,
) -> Optional[
    int
]:
    """
    Normalize an optional non-negative integer.

    Parameters
    ----------
    value : Number or None
        Candidate index.
    name : str
        Human-readable field name.

    Returns
    -------
    int or None
        Normalized integer.

    Raises
    ------
    TypeError
        If ``value`` is not integer-like.
    ValueError
        If ``value`` is negative.
    """

    if value is None:
        return None

    if isinstance(
        value,
        (
            bool,
            np.bool_,
        ),
    ):
        raise TypeError(
            f"{name} must be an integer, not boolean."
        )

    try:
        normalized_float = np.float64(
            value
        )

    except (
        TypeError,
        ValueError,
        OverflowError,
    ) as error:
        raise TypeError(
            f"{name} must be an integer or None."
        ) from error

    if (
        not np.isfinite(
            normalized_float
        )
        or not float(
            normalized_float
        ).is_integer()
    ):
        raise TypeError(
            f"{name} must be an integer or None."
        )

    normalized = int(
        normalized_float
    )

    if normalized < 0:
        raise ValueError(
            f"{name} cannot be negative."
        )

    return normalized


def _normalize_residue_key(
    key: Optional[
        ResidueContactKey
    ],
) -> Optional[
    ResidueContactKey
]:
    """
    Normalize an optional residue contact key.

    Parameters
    ----------
    key : ResidueContactKey or None
        Residue key in ``(name, number, chain_id)`` form.

    Returns
    -------
    ResidueContactKey or None
        Normalized key.

    Raises
    ------
    TypeError
        If the key does not have three components.
    """

    if key is None:
        return None

    if (
        not isinstance(
            key,
            (
                tuple,
                list,
            ),
        )
        or len(
            key
        )
        != 3
    ):
        raise TypeError(
            "A residue key must contain exactly three values: "
            "(name, number, chain_id)."
        )

    residue_name = str(
        key[
            0
        ]
        or ""
    ).strip().upper()

    try:
        residue_number = int(
            key[
                1
            ]
        )

    except (
        TypeError,
        ValueError,
        OverflowError,
    ) as error:
        raise TypeError(
            "The residue number in a residue key must be an integer."
        ) from error

    chain_id = str(
        key[
            2
        ]
        or ""
    ).strip()

    return (
        residue_name,
        residue_number,
        chain_id,
    )


def _safe_atom_identifier(
    atom: Optional[
        AtomLike
    ],
) -> Optional[
    str
]:
    """
    Return an atom identifier without propagating helper errors.

    Parameters
    ----------
    atom : atom-like or None
        Atom to identify.

    Returns
    -------
    str or None
        Atom identifier.
    """

    if atom is None:
        return None

    try:
        identifier = get_atom_identifier(
            atom
        )

    except Exception:
        try:
            identifier = get_atom_name(
                atom
            )

        except Exception:
            identifier = repr(
                atom
            )

    if identifier is None:
        return None

    return str(
        identifier
    )


def _safe_residue_key_from_atom(
    atom: Optional[
        AtomLike
    ],
) -> Optional[
    ResidueContactKey
]:
    """
    Resolve a residue contact key from an atom defensively.

    Parameters
    ----------
    atom : atom-like or None
        Atom whose residue should be identified.

    Returns
    -------
    ResidueContactKey or None
        Residue key when available.
    """

    if atom is None:
        return None

    try:
        residue = get_atom_residue(
            atom
        )

    except Exception:
        return None

    if residue is None:
        return None

    try:
        return get_residue_contact_key(
            residue
        )

    except Exception:
        return None


# -----------------------------------------------------------------------------
# Hydrogen-bond geometry
# -----------------------------------------------------------------------------

@dataclass(
    frozen=True,
    slots=True,
)
class HydrogenBondGeometry:
    """
    Geometric measurements associated with a hydrogen bond.

    Parameters
    ----------
    donor_acceptor_distance : Number
        Distance between donor and acceptor atoms in angstroms.
    hydrogen_acceptor_distance : Number or None, optional
        Distance between the explicit hydrogen and acceptor in angstroms.
        This value is normally unavailable for inferred hydrogen bonds.
    donor_hydrogen_distance : Number or None, optional
        Covalent donor-hydrogen distance in angstroms.
    dha_angle : Number or None, optional
        Donor-hydrogen-acceptor angle in degrees.
    donor_angle : Number or None, optional
        Auxiliary donor-side angle used by inferred geometric models.
    acceptor_angle : Number or None, optional
        Auxiliary acceptor-side angle.
    metadata : mapping, optional
        Additional geometric information.

    Notes
    -----
    All stored floating-point values are normalized to ``numpy.float64``.
    """

    donor_acceptor_distance: np.float64

    hydrogen_acceptor_distance: Optional[
        np.float64
    ] = None

    donor_hydrogen_distance: Optional[
        np.float64
    ] = None

    dha_angle: Optional[
        np.float64
    ] = None

    donor_angle: Optional[
        np.float64
    ] = None

    acceptor_angle: Optional[
        np.float64
    ] = None

    metadata: Mapping[
        str,
        Any,
    ] = field(
        default_factory=lambda: _EMPTY_METADATA,
        repr=False,
        compare=False,
    )

    def __post_init__(
        self,
    ) -> None:
        """Validate and normalize geometric measurements."""

        donor_acceptor_distance = (
            _optional_float64(
                self.donor_acceptor_distance,
                name="donor-acceptor distance",
                minimum=0.0,
            )
        )

        if donor_acceptor_distance is None:
            raise ValueError(
                "Donor-acceptor distance cannot be None."
            )

        object.__setattr__(
            self,
            "donor_acceptor_distance",
            donor_acceptor_distance,
        )

        object.__setattr__(
            self,
            "hydrogen_acceptor_distance",
            _optional_float64(
                self.hydrogen_acceptor_distance,
                name="hydrogen-acceptor distance",
                minimum=0.0,
            ),
        )

        object.__setattr__(
            self,
            "donor_hydrogen_distance",
            _optional_float64(
                self.donor_hydrogen_distance,
                name="donor-hydrogen distance",
                minimum=0.0,
            ),
        )

        for field_name in (
            "dha_angle",
            "donor_angle",
            "acceptor_angle",
        ):
            object.__setattr__(
                self,
                field_name,
                _optional_float64(
                    getattr(
                        self,
                        field_name,
                    ),
                    name=field_name.replace(
                        "_",
                        " ",
                    ),
                    minimum=MINIMUM_VALID_ANGLE_DEGREES,
                    maximum=MAXIMUM_VALID_ANGLE_DEGREES,
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
    def has_explicit_hydrogen_geometry(
        self,
    ) -> bool:
        """
        Whether explicit hydrogen-bond geometry is available.

        Returns
        -------
        bool
            ``True`` when both H...A distance and D-H...A angle exist.
        """

        return (
            self.hydrogen_acceptor_distance
            is not None
            and self.dha_angle
            is not None
        )

    @property
    def has_angular_geometry(
        self,
    ) -> bool:
        """
        Whether at least one angular measurement is available.

        Returns
        -------
        bool
            Angular-data availability.
        """

        return any(
            value is not None
            for value in (
                self.dha_angle,
                self.donor_angle,
                self.acceptor_angle,
            )
        )

    @property
    def linearity_deviation(
        self,
    ) -> Optional[
        np.float64
    ]:
        """
        Return the deviation of D-H...A from linear geometry.

        Returns
        -------
        numpy.float64 or None
            ``180° - D-H...A`` when the angle is available.
        """

        if self.dha_angle is None:
            return None

        return np.float64(
            STRAIGHT_ANGLE_DEGREES
            - self.dha_angle
        )

    def to_dict(
        self,
    ) -> Dict[
        str,
        Any,
    ]:
        """
        Serialize the geometry.

        Returns
        -------
        dict
            JSON-compatible geometric summary.
        """

        return {
            "donor_acceptor_distance": float(
                self.donor_acceptor_distance
            ),
            "hydrogen_acceptor_distance": (
                None
                if self.hydrogen_acceptor_distance is None
                else float(
                    self.hydrogen_acceptor_distance
                )
            ),
            "donor_hydrogen_distance": (
                None
                if self.donor_hydrogen_distance is None
                else float(
                    self.donor_hydrogen_distance
                )
            ),
            "dha_angle": (
                None
                if self.dha_angle is None
                else float(
                    self.dha_angle
                )
            ),
            "donor_angle": (
                None
                if self.donor_angle is None
                else float(
                    self.donor_angle
                )
            ),
            "acceptor_angle": (
                None
                if self.acceptor_angle is None
                else float(
                    self.acceptor_angle
                )
            ),
            "has_explicit_hydrogen_geometry": (
                self.has_explicit_hydrogen_geometry
            ),
            "linearity_deviation": (
                None
                if self.linearity_deviation is None
                else float(
                    self.linearity_deviation
                )
            ),
            "metadata": dict(
                self.metadata
            ),
        }


# -----------------------------------------------------------------------------
# Individual hydrogen-bond result
# -----------------------------------------------------------------------------

@dataclass(
    frozen=True,
    slots=True,
)
class HydrogenBond:
    """
    Representation of one donor-hydrogen-acceptor interaction.

    Parameters
    ----------
    donor : atom-like
        Donor atom.
    acceptor : atom-like
        Acceptor atom.
    geometry : HydrogenBondGeometry
        Calculated interaction geometry.
    hydrogen : atom-like or None, optional
        Explicit hydrogen atom. It is ``None`` for inferred interactions.
    mode : HydrogenBondMode, optional
        Explicit or inferred analysis mode.
    direction : HydrogenBondDirection, optional
        Whether the ligand or receptor acts as donor.
    classification : HydrogenBondClassification, optional
        Geometric strength classification.
    donor_index : int or None, optional
        Donor index in its analyzed atom collection.
    hydrogen_index : int or None, optional
        Hydrogen index in its analyzed atom collection.
    acceptor_index : int or None, optional
        Acceptor index in its analyzed atom collection.
    donor_residue : residue-like or None, optional
        Donor parent residue.
    acceptor_residue : residue-like or None, optional
        Acceptor parent residue.
    donor_residue_key : ResidueContactKey or None, optional
        Normalized donor residue key.
    acceptor_residue_key : ResidueContactKey or None, optional
        Normalized acceptor residue key.
    metadata : mapping, optional
        Additional interaction metadata.
    """

    donor: AtomLike
    acceptor: AtomLike
    geometry: HydrogenBondGeometry

    hydrogen: Optional[
        AtomLike
    ] = None

    mode: HydrogenBondMode = (
        HBOND_MODE_INFERRED
    )

    direction: HydrogenBondDirection = (
        HBOND_DIRECTION_UNKNOWN
    )

    classification: HydrogenBondClassification = (
        HBOND_TYPE_UNKNOWN
    )

    donor_index: Optional[
        int
    ] = None

    hydrogen_index: Optional[
        int
    ] = None

    acceptor_index: Optional[
        int
    ] = None

    donor_residue: Optional[
        ResidueLike
    ] = None

    acceptor_residue: Optional[
        ResidueLike
    ] = None

    donor_residue_key: Optional[
        ResidueContactKey
    ] = None

    acceptor_residue_key: Optional[
        ResidueContactKey
    ] = None

    metadata: Mapping[
        str,
        Any,
    ] = field(
        default_factory=lambda: _EMPTY_METADATA,
        repr=False,
        compare=False,
    )

    def __post_init__(
        self,
    ) -> None:
        """Validate and normalize the interaction."""

        if self.donor is None:
            raise ValueError(
                "Hydrogen-bond donor cannot be None."
            )

        if self.acceptor is None:
            raise ValueError(
                "Hydrogen-bond acceptor cannot be None."
            )

        if self.donor is self.acceptor:
            raise ValueError(
                "Hydrogen-bond donor and acceptor must be "
                "different atoms."
            )

        if not isinstance(
            self.geometry,
            HydrogenBondGeometry,
        ):
            raise TypeError(
                "geometry must be a HydrogenBondGeometry instance."
            )

        normalized_mode = (
            validate_hydrogen_bond_mode(
                self.mode
            )
        )

        normalized_direction = (
            validate_hydrogen_bond_direction(
                self.direction
            )
        )

        normalized_classification = (
            validate_hydrogen_bond_classification(
                self.classification
            )
        )

        if (
            normalized_mode
            == HBOND_MODE_EXPLICIT
            and self.hydrogen is None
        ):
            raise ValueError(
                "Explicit hydrogen-bond mode requires a "
                "hydrogen atom."
            )

        if (
            normalized_mode
            == HBOND_MODE_INFERRED
            and self.hydrogen is not None
        ):
            normalized_mode = (
                HBOND_MODE_EXPLICIT
            )

        if (
            self.hydrogen is not None
            and (
                self.hydrogen is self.donor
                or self.hydrogen is self.acceptor
            )
        ):
            raise ValueError(
                "The hydrogen atom must differ from the donor "
                "and acceptor atoms."
            )

        object.__setattr__(
            self,
            "mode",
            normalized_mode,
        )

        object.__setattr__(
            self,
            "direction",
            normalized_direction,
        )

        object.__setattr__(
            self,
            "classification",
            normalized_classification,
        )

        object.__setattr__(
            self,
            "donor_index",
            _optional_nonnegative_integer(
                self.donor_index,
                name="donor index",
            ),
        )

        object.__setattr__(
            self,
            "hydrogen_index",
            _optional_nonnegative_integer(
                self.hydrogen_index,
                name="hydrogen index",
            ),
        )

        object.__setattr__(
            self,
            "acceptor_index",
            _optional_nonnegative_integer(
                self.acceptor_index,
                name="acceptor index",
            ),
        )

        donor_residue = self.donor_residue

        if donor_residue is None:
            try:
                donor_residue = (
                    get_atom_residue(
                        self.donor
                    )
                )

            except Exception:
                donor_residue = None

        acceptor_residue = self.acceptor_residue

        if acceptor_residue is None:
            try:
                acceptor_residue = (
                    get_atom_residue(
                        self.acceptor
                    )
                )

            except Exception:
                acceptor_residue = None

        object.__setattr__(
            self,
            "donor_residue",
            donor_residue,
        )

        object.__setattr__(
            self,
            "acceptor_residue",
            acceptor_residue,
        )

        donor_key = (
            _normalize_residue_key(
                self.donor_residue_key
            )
        )

        if donor_key is None:
            donor_key = (
                _safe_residue_key_from_atom(
                    self.donor
                )
            )

        acceptor_key = (
            _normalize_residue_key(
                self.acceptor_residue_key
            )
        )

        if acceptor_key is None:
            acceptor_key = (
                _safe_residue_key_from_atom(
                    self.acceptor
                )
            )

        object.__setattr__(
            self,
            "donor_residue_key",
            donor_key,
        )

        object.__setattr__(
            self,
            "acceptor_residue_key",
            acceptor_key,
        )

        object.__setattr__(
            self,
            "metadata",
            _freeze_metadata(
                self.metadata
            ),
        )

    @property
    def donor_acceptor_distance(
        self,
    ) -> np.float64:
        """
        Return the donor-acceptor distance.

        Returns
        -------
        numpy.float64
            D...A distance in angstroms.
        """

        return (
            self.geometry
            .donor_acceptor_distance
        )

    @property
    def hydrogen_acceptor_distance(
        self,
    ) -> Optional[
        np.float64
    ]:
        """
        Return the hydrogen-acceptor distance.

        Returns
        -------
        numpy.float64 or None
            H...A distance in angstroms.
        """

        return (
            self.geometry
            .hydrogen_acceptor_distance
        )

    @property
    def dha_angle(
        self,
    ) -> Optional[
        np.float64
    ]:
        """
        Return the donor-hydrogen-acceptor angle.

        Returns
        -------
        numpy.float64 or None
            D-H...A angle in degrees.
        """

        return self.geometry.dha_angle

    @property
    def atom_pair(
        self,
    ) -> DonorAcceptorPair:
        """
        Return the donor-acceptor atom pair.

        Returns
        -------
        DonorAcceptorPair
            Donor and acceptor atoms.
        """

        return (
            self.donor,
            self.acceptor,
        )

    @property
    def atom_triple(
        self,
    ) -> Tuple[
        AtomLike,
        Optional[
            AtomLike
        ],
        AtomLike,
    ]:
        """
        Return donor, hydrogen and acceptor atoms.

        Returns
        -------
        tuple
            ``(donor, hydrogen, acceptor)``.
        """

        return (
            self.donor,
            self.hydrogen,
            self.acceptor,
        )

    @property
    def index_pair(
        self,
    ) -> Tuple[
        Optional[
            int
        ],
        Optional[
            int
        ],
    ]:
        """
        Return donor and acceptor indices.

        Returns
        -------
        tuple
            Donor and acceptor indices.
        """

        return (
            self.donor_index,
            self.acceptor_index,
        )

    @property
    def index_triple(
        self,
    ) -> Tuple[
        Optional[
            int
        ],
        Optional[
            int
        ],
        Optional[
            int
        ],
    ]:
        """
        Return donor, hydrogen and acceptor indices.

        Returns
        -------
        tuple
            D-H-A indices.
        """

        return (
            self.donor_index,
            self.hydrogen_index,
            self.acceptor_index,
        )

    @property
    def is_explicit(
        self,
    ) -> bool:
        """
        Whether this bond uses an explicit hydrogen.

        Returns
        -------
        bool
            Explicit-mode status.
        """

        return (
            self.mode
            == HBOND_MODE_EXPLICIT
            and self.hydrogen is not None
        )

    @property
    def is_inferred(
        self,
    ) -> bool:
        """
        Whether this bond was geometrically inferred.

        Returns
        -------
        bool
            Inferred-mode status.
        """

        return not self.is_explicit

    @property
    def is_strong(
        self,
    ) -> bool:
        """
        Whether the bond is classified as strong.

        Returns
        -------
        bool
            Strong-classification status.
        """

        return (
            self.classification
            == HBOND_TYPE_STRONG
        )

    @property
    def is_ligand_donor(
        self,
    ) -> bool:
        """
        Whether the ligand acts as donor.

        Returns
        -------
        bool
            Ligand-donor status.
        """

        return (
            self.direction
            == HBOND_DIRECTION_LIGAND_DONOR
        )

    @property
    def is_receptor_donor(
        self,
    ) -> bool:
        """
        Whether the receptor acts as donor.

        Returns
        -------
        bool
            Receptor-donor status.
        """

        return (
            self.direction
            == HBOND_DIRECTION_RECEPTOR_DONOR
        )

    @property
    def identifier(
        self,
    ) -> HydrogenBondIdentifier:
        """
        Return a stable descriptive hydrogen-bond identifier.

        Returns
        -------
        HydrogenBondIdentifier
            Donor, acceptor and optional hydrogen identifiers.
        """

        donor_identifier = (
            _safe_atom_identifier(
                self.donor
            )
            or "unknown_donor"
        )

        acceptor_identifier = (
            _safe_atom_identifier(
                self.acceptor
            )
            or "unknown_acceptor"
        )

        hydrogen_identifier = (
            _safe_atom_identifier(
                self.hydrogen
            )
        )

        return (
            donor_identifier,
            acceptor_identifier,
            hydrogen_identifier,
        )

    def to_dict(
        self,
        *,
        include_atoms: bool = False,
    ) -> Dict[
        str,
        Any,
    ]:
        """
        Serialize the hydrogen bond.

        Parameters
        ----------
        include_atoms : bool, optional
            Whether raw atom and residue objects should be included.

        Returns
        -------
        dict
            Serializable interaction representation.
        """

        result: Dict[
            str,
            Any,
        ] = {
            "identifier": self.identifier,
            "donor_identifier": (
                _safe_atom_identifier(
                    self.donor
                )
            ),
            "hydrogen_identifier": (
                _safe_atom_identifier(
                    self.hydrogen
                )
            ),
            "acceptor_identifier": (
                _safe_atom_identifier(
                    self.acceptor
                )
            ),
            "mode": self.mode,
            "direction": self.direction,
            "classification": (
                self.classification
            ),
            "donor_index": self.donor_index,
            "hydrogen_index": (
                self.hydrogen_index
            ),
            "acceptor_index": (
                self.acceptor_index
            ),
            "donor_residue_key": (
                self.donor_residue_key
            ),
            "acceptor_residue_key": (
                self.acceptor_residue_key
            ),
            "geometry": (
                self.geometry.to_dict()
            ),
            "metadata": dict(
                self.metadata
            ),
        }

        if include_atoms:
            result.update(
                {
                    "donor": self.donor,
                    "hydrogen": self.hydrogen,
                    "acceptor": self.acceptor,
                    "donor_residue": (
                        self.donor_residue
                    ),
                    "acceptor_residue": (
                        self.acceptor_residue
                    ),
                }
            )

        return result


# -----------------------------------------------------------------------------
# Residue-level hydrogen-bond result
# -----------------------------------------------------------------------------

@dataclass(
    frozen=True,
    slots=True,
)
class ResidueHydrogenBond:
    """
    Hydrogen bonds associated with one residue.

    Parameters
    ----------
    residue : residue-like
        Grouped residue.
    key : ResidueContactKey
        Normalized residue key.
    hydrogen_bonds : sequence of HydrogenBond
        Hydrogen bonds involving the residue.
    side : str, optional
        Grouping side: ``"donor"``, ``"acceptor"`` or ``"receptor"``.
    metadata : mapping, optional
        Additional residue-level metadata.
    """

    residue: ResidueLike
    key: ResidueContactKey

    hydrogen_bonds: Sequence[
        HydrogenBond
    ] = field(
        default_factory=tuple
    )

    side: str = "receptor"

    metadata: Mapping[
        str,
        Any,
    ] = field(
        default_factory=lambda: _EMPTY_METADATA,
        repr=False,
        compare=False,
    )

    def __post_init__(
        self,
    ) -> None:
        """Validate and normalize residue-level data."""

        if self.residue is None:
            raise ValueError(
                "ResidueHydrogenBond.residue cannot be None."
            )

        normalized_key = (
            _normalize_residue_key(
                self.key
            )
        )

        if normalized_key is None:
            raise ValueError(
                "ResidueHydrogenBond.key cannot be None."
            )

        normalized_bonds = tuple(
            self.hydrogen_bonds
        )

        for index, hydrogen_bond in enumerate(
            normalized_bonds
        ):
            if not isinstance(
                hydrogen_bond,
                HydrogenBond,
            ):
                raise TypeError(
                    "All residue hydrogen-bond entries must be "
                    "HydrogenBond instances. Invalid entry at "
                    f"index {index}."
                )

        normalized_side = str(
            self.side
        ).strip().lower()

        valid_sides = {
            "donor",
            "acceptor",
            "ligand",
            "receptor",
            "either",
        }

        if normalized_side not in valid_sides:
            raise ValueError(
                f"Unsupported residue grouping side "
                f"{self.side!r}. Expected one of: "
                f"{', '.join(sorted(valid_sides))}."
            )

        object.__setattr__(
            self,
            "key",
            normalized_key,
        )

        object.__setattr__(
            self,
            "hydrogen_bonds",
            normalized_bonds,
        )

        object.__setattr__(
            self,
            "side",
            normalized_side,
        )

        object.__setattr__(
            self,
            "metadata",
            _freeze_metadata(
                self.metadata
            ),
        )

    @property
    def hydrogen_bond_count(
        self,
    ) -> int:
        """
        Return the number of hydrogen bonds.

        Returns
        -------
        int
            Hydrogen-bond count.
        """

        return len(
            self.hydrogen_bonds
        )

    @property
    def minimum_distance(
        self,
    ) -> Optional[
        np.float64
    ]:
        """
        Return the minimum donor-acceptor distance.

        Returns
        -------
        numpy.float64 or None
            Minimum distance.
        """

        if not self.hydrogen_bonds:
            return None

        return np.float64(
            min(
                hydrogen_bond
                .donor_acceptor_distance
                for hydrogen_bond
                in self.hydrogen_bonds
            )
        )

    @property
    def mean_distance(
        self,
    ) -> Optional[
        np.float64
    ]:
        """
        Return the mean donor-acceptor distance.

        Returns
        -------
        numpy.float64 or None
            Mean distance.
        """

        if not self.hydrogen_bonds:
            return None

        return np.float64(
            np.mean(
                np.asarray(
                    [
                        hydrogen_bond
                        .donor_acceptor_distance
                        for hydrogen_bond
                        in self.hydrogen_bonds
                    ],
                    dtype=np.float64,
                )
            )
        )

    @property
    def maximum_dha_angle(
        self,
    ) -> Optional[
        np.float64
    ]:
        """
        Return the largest available D-H...A angle.

        Returns
        -------
        numpy.float64 or None
            Maximum angle.
        """

        angles = [
            hydrogen_bond.dha_angle
            for hydrogen_bond
            in self.hydrogen_bonds
            if hydrogen_bond.dha_angle
            is not None
        ]

        if not angles:
            return None

        return np.float64(
            max(
                angles
            )
        )

    @property
    def classifications(
        self,
    ) -> FrozenSet[
        HydrogenBondClassification
    ]:
        """
        Return classifications represented in the group.

        Returns
        -------
        frozenset
            Hydrogen-bond classifications.
        """

        return frozenset(
            hydrogen_bond.classification
            for hydrogen_bond
            in self.hydrogen_bonds
        )

    @property
    def directions(
        self,
    ) -> FrozenSet[
        HydrogenBondDirection
    ]:
        """
        Return directions represented in the group.

        Returns
        -------
        frozenset
            Hydrogen-bond directions.
        """

        return frozenset(
            hydrogen_bond.direction
            for hydrogen_bond
            in self.hydrogen_bonds
        )

    @property
    def explicit_count(
        self,
    ) -> int:
        """
        Return the number of explicit hydrogen bonds.

        Returns
        -------
        int
            Explicit-bond count.
        """

        return sum(
            hydrogen_bond.is_explicit
            for hydrogen_bond
            in self.hydrogen_bonds
        )

    @property
    def inferred_count(
        self,
    ) -> int:
        """
        Return the number of inferred hydrogen bonds.

        Returns
        -------
        int
            Inferred-bond count.
        """

        return (
            self.hydrogen_bond_count
            - self.explicit_count
        )

    @property
    def strong_count(
        self,
    ) -> int:
        """
        Return the number of strong hydrogen bonds.

        Returns
        -------
        int
            Strong-bond count.
        """

        return sum(
            hydrogen_bond.is_strong
            for hydrogen_bond
            in self.hydrogen_bonds
        )

    @property
    def has_strong_bond(
        self,
    ) -> bool:
        """
        Whether the residue forms a strong hydrogen bond.

        Returns
        -------
        bool
            Strong-bond presence.
        """

        return self.strong_count > 0

    def bonds_by_classification(
        self,
        classification: str,
    ) -> Tuple[
        HydrogenBond,
        ...,
    ]:
        """
        Select residue bonds by classification.

        Parameters
        ----------
        classification : str
            Requested classification.

        Returns
        -------
        tuple of HydrogenBond
            Matching bonds.
        """

        normalized = (
            validate_hydrogen_bond_classification(
                classification
            )
        )

        return tuple(
            hydrogen_bond
            for hydrogen_bond
            in self.hydrogen_bonds
            if hydrogen_bond.classification
            == normalized
        )

    def to_dict(
        self,
        *,
        include_bonds: bool = True,
        include_atoms: bool = False,
    ) -> Dict[
        str,
        Any,
    ]:
        """
        Serialize the residue-level hydrogen-bond result.

        Parameters
        ----------
        include_bonds : bool, optional
            Whether individual bonds should be included.
        include_atoms : bool, optional
            Whether raw atom objects should be included in bond data.

        Returns
        -------
        dict
            Residue-level summary.
        """

        result: Dict[
            str,
            Any,
        ] = {
            "key": self.key,
            "side": self.side,
            "hydrogen_bond_count": (
                self.hydrogen_bond_count
            ),
            "explicit_count": (
                self.explicit_count
            ),
            "inferred_count": (
                self.inferred_count
            ),
            "strong_count": (
                self.strong_count
            ),
            "minimum_distance": (
                None
                if self.minimum_distance is None
                else float(
                    self.minimum_distance
                )
            ),
            "mean_distance": (
                None
                if self.mean_distance is None
                else float(
                    self.mean_distance
                )
            ),
            "maximum_dha_angle": (
                None
                if self.maximum_dha_angle is None
                else float(
                    self.maximum_dha_angle
                )
            ),
            "classifications": sorted(
                self.classifications
            ),
            "directions": sorted(
                self.directions
            ),
            "metadata": dict(
                self.metadata
            ),
        }

        if include_bonds:
            result[
                "hydrogen_bonds"
            ] = [
                hydrogen_bond.to_dict(
                    include_atoms=include_atoms
                )
                for hydrogen_bond
                in self.hydrogen_bonds
            ]

        if include_atoms:
            result[
                "residue"
            ] = self.residue

        return result


# -----------------------------------------------------------------------------
# Complete hydrogen-bond analysis result
# -----------------------------------------------------------------------------

@dataclass(
    frozen=True,
    slots=True,
)
class HydrogenBondAnalysisResult:
    """
    Complete result of a ligand-receptor hydrogen-bond analysis.

    Parameters
    ----------
    hydrogen_bonds : sequence of HydrogenBond
        Detected hydrogen bonds.
    residue_hydrogen_bonds : sequence of ResidueHydrogenBond
        Residue-grouped interaction results.
    ligand_atoms : sequence of atom-like
        Ligand atoms used in the analysis.
    receptor_atoms : sequence of atom-like
        Receptor atoms used in the analysis.
    donor_acceptor_cutoff : Number
        Maximum D...A distance used by the analysis.
    hydrogen_acceptor_cutoff : Number or None
        Maximum H...A distance.
    minimum_dha_angle : Number or None
        Minimum explicit D-H...A angle.
    minimum_inferred_angle : Number or None
        Minimum angle used for inferred interactions.
    statistics : mapping, optional
        Precomputed summary statistics.
    metadata : mapping, optional
        Additional analysis metadata.
    """

    hydrogen_bonds: Sequence[
        HydrogenBond
    ] = field(
        default_factory=tuple
    )

    residue_hydrogen_bonds: Sequence[
        ResidueHydrogenBond
    ] = field(
        default_factory=tuple
    )

    ligand_atoms: Sequence[
        AtomLike
    ] = field(
        default_factory=tuple
    )

    receptor_atoms: Sequence[
        AtomLike
    ] = field(
        default_factory=tuple
    )

    donor_acceptor_cutoff: np.float64 = (
        DEFAULT_DONOR_ACCEPTOR_DISTANCE
    )

    hydrogen_acceptor_cutoff: Optional[
        np.float64
    ] = DEFAULT_HYDROGEN_ACCEPTOR_DISTANCE

    minimum_dha_angle: Optional[
        np.float64
    ] = DEFAULT_MINIMUM_DHA_ANGLE

    minimum_inferred_angle: Optional[
        np.float64
    ] = DEFAULT_MINIMUM_INFERRED_ANGLE

    statistics: Mapping[
        str,
        Any,
    ] = field(
        default_factory=lambda: _EMPTY_METADATA,
        repr=False,
        compare=False,
    )

    metadata: Mapping[
        str,
        Any,
    ] = field(
        default_factory=lambda: _EMPTY_METADATA,
        repr=False,
        compare=False,
    )

    def __post_init__(
        self,
    ) -> None:
        """Validate and normalize the complete result."""

        normalized_bonds = tuple(
            self.hydrogen_bonds
        )

        for index, hydrogen_bond in enumerate(
            normalized_bonds
        ):
            if not isinstance(
                hydrogen_bond,
                HydrogenBond,
            ):
                raise TypeError(
                    "All hydrogen_bonds entries must be "
                    "HydrogenBond instances. Invalid entry "
                    f"at index {index}."
                )

        normalized_residue_results = tuple(
            self.residue_hydrogen_bonds
        )

        for index, residue_result in enumerate(
            normalized_residue_results
        ):
            if not isinstance(
                residue_result,
                ResidueHydrogenBond,
            ):
                raise TypeError(
                    "All residue_hydrogen_bonds entries must be "
                    "ResidueHydrogenBond instances. Invalid entry "
                    f"at index {index}."
                )

        ligand_atoms = tuple(
            self.ligand_atoms
        )

        receptor_atoms = tuple(
            self.receptor_atoms
        )

        donor_acceptor_cutoff = (
            _optional_float64(
                self.donor_acceptor_cutoff,
                name="donor-acceptor cutoff",
                minimum=0.0,
            )
        )

        if donor_acceptor_cutoff is None:
            raise ValueError(
                "Donor-acceptor cutoff cannot be None."
            )

        object.__setattr__(
            self,
            "hydrogen_bonds",
            normalized_bonds,
        )

        object.__setattr__(
            self,
            "residue_hydrogen_bonds",
            normalized_residue_results,
        )

        object.__setattr__(
            self,
            "ligand_atoms",
            ligand_atoms,
        )

        object.__setattr__(
            self,
            "receptor_atoms",
            receptor_atoms,
        )

        object.__setattr__(
            self,
            "donor_acceptor_cutoff",
            donor_acceptor_cutoff,
        )

        object.__setattr__(
            self,
            "hydrogen_acceptor_cutoff",
            _optional_float64(
                self.hydrogen_acceptor_cutoff,
                name="hydrogen-acceptor cutoff",
                minimum=0.0,
            ),
        )

        object.__setattr__(
            self,
            "minimum_dha_angle",
            _optional_float64(
                self.minimum_dha_angle,
                name="minimum D-H-A angle",
                minimum=MINIMUM_VALID_ANGLE_DEGREES,
                maximum=MAXIMUM_VALID_ANGLE_DEGREES,
            ),
        )

        object.__setattr__(
            self,
            "minimum_inferred_angle",
            _optional_float64(
                self.minimum_inferred_angle,
                name="minimum inferred angle",
                minimum=MINIMUM_VALID_ANGLE_DEGREES,
                maximum=MAXIMUM_VALID_ANGLE_DEGREES,
            ),
        )

        object.__setattr__(
            self,
            "statistics",
            _freeze_metadata(
                self.statistics
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
    def hydrogen_bond_count(
        self,
    ) -> int:
        """
        Return the total hydrogen-bond count.

        Returns
        -------
        int
            Number of detected hydrogen bonds.
        """

        return len(
            self.hydrogen_bonds
        )

    @property
    def residue_count(
        self,
    ) -> int:
        """
        Return the number of grouped residues.

        Returns
        -------
        int
            Residue count.
        """

        return len(
            self.residue_hydrogen_bonds
        )

    @property
    def ligand_atom_count(
        self,
    ) -> int:
        """
        Return the number of analyzed ligand atoms.

        Returns
        -------
        int
            Ligand atom count.
        """

        return len(
            self.ligand_atoms
        )

    @property
    def receptor_atom_count(
        self,
    ) -> int:
        """
        Return the number of analyzed receptor atoms.

        Returns
        -------
        int
            Receptor atom count.
        """

        return len(
            self.receptor_atoms
        )

    @property
    def minimum_distance(
        self,
    ) -> Optional[
        np.float64
    ]:
        """
        Return the minimum donor-acceptor distance.

        Returns
        -------
        numpy.float64 or None
            Minimum D...A distance.
        """

        if not self.hydrogen_bonds:
            return None

        return np.float64(
            min(
                hydrogen_bond
                .donor_acceptor_distance
                for hydrogen_bond
                in self.hydrogen_bonds
            )
        )

    @property
    def mean_distance(
        self,
    ) -> Optional[
        np.float64
    ]:
        """
        Return the mean donor-acceptor distance.

        Returns
        -------
        numpy.float64 or None
            Mean D...A distance.
        """

        if not self.hydrogen_bonds:
            return None

        return np.float64(
            np.mean(
                np.asarray(
                    [
                        hydrogen_bond
                        .donor_acceptor_distance
                        for hydrogen_bond
                        in self.hydrogen_bonds
                    ],
                    dtype=np.float64,
                )
            )
        )

    @property
    def explicit_count(
        self,
    ) -> int:
        """
        Return the number of explicit hydrogen bonds.

        Returns
        -------
        int
            Explicit-bond count.
        """

        return sum(
            hydrogen_bond.is_explicit
            for hydrogen_bond
            in self.hydrogen_bonds
        )

    @property
    def inferred_count(
        self,
    ) -> int:
        """
        Return the number of inferred hydrogen bonds.

        Returns
        -------
        int
            Inferred-bond count.
        """

        return (
            self.hydrogen_bond_count
            - self.explicit_count
        )

    @property
    def strong_count(
        self,
    ) -> int:
        """
        Return the number of strong hydrogen bonds.

        Returns
        -------
        int
            Strong-bond count.
        """

        return sum(
            hydrogen_bond.is_strong
            for hydrogen_bond
            in self.hydrogen_bonds
        )

    @property
    def ligand_donor_count(
        self,
    ) -> int:
        """
        Return bonds in which the ligand is the donor.

        Returns
        -------
        int
            Ligand-donor bond count.
        """

        return sum(
            hydrogen_bond.is_ligand_donor
            for hydrogen_bond
            in self.hydrogen_bonds
        )

    @property
    def receptor_donor_count(
        self,
    ) -> int:
        """
        Return bonds in which the receptor is the donor.

        Returns
        -------
        int
            Receptor-donor bond count.
        """

        return sum(
            hydrogen_bond.is_receptor_donor
            for hydrogen_bond
            in self.hydrogen_bonds
        )

    @property
    def has_hydrogen_bonds(
        self,
    ) -> bool:
        """
        Whether at least one hydrogen bond was detected.

        Returns
        -------
        bool
            Hydrogen-bond presence.
        """

        return bool(
            self.hydrogen_bonds
        )

    @property
    def has_explicit_bonds(
        self,
    ) -> bool:
        """
        Whether explicit hydrogen bonds were detected.

        Returns
        -------
        bool
            Explicit-bond presence.
        """

        return self.explicit_count > 0

    @property
    def has_inferred_bonds(
        self,
    ) -> bool:
        """
        Whether inferred hydrogen bonds were detected.

        Returns
        -------
        bool
            Inferred-bond presence.
        """

        return self.inferred_count > 0

    @property
    def has_strong_bonds(
        self,
    ) -> bool:
        """
        Whether strong hydrogen bonds were detected.

        Returns
        -------
        bool
            Strong-bond presence.
        """

        return self.strong_count > 0

    def bonds_by_classification(
        self,
        classification: str,
    ) -> Tuple[
        HydrogenBond,
        ...,
    ]:
        """
        Return hydrogen bonds with one classification.

        Parameters
        ----------
        classification : str
            Requested classification.

        Returns
        -------
        tuple of HydrogenBond
            Matching interactions.
        """

        normalized = (
            validate_hydrogen_bond_classification(
                classification
            )
        )

        return tuple(
            hydrogen_bond
            for hydrogen_bond
            in self.hydrogen_bonds
            if hydrogen_bond.classification
            == normalized
        )

    def bonds_by_mode(
        self,
        mode: str,
    ) -> Tuple[
        HydrogenBond,
        ...,
    ]:
        """
        Return hydrogen bonds detected in one mode.

        Parameters
        ----------
        mode : str
            Explicit or inferred mode.

        Returns
        -------
        tuple of HydrogenBond
            Matching interactions.
        """

        normalized = (
            validate_hydrogen_bond_mode(
                mode
            )
        )

        return tuple(
            hydrogen_bond
            for hydrogen_bond
            in self.hydrogen_bonds
            if hydrogen_bond.mode
            == normalized
        )

    def bonds_by_direction(
        self,
        direction: str,
    ) -> Tuple[
        HydrogenBond,
        ...,
    ]:
        """
        Return hydrogen bonds with one donor direction.

        Parameters
        ----------
        direction : str
            Requested direction.

        Returns
        -------
        tuple of HydrogenBond
            Matching interactions.
        """

        normalized = (
            validate_hydrogen_bond_direction(
                direction
            )
        )

        return tuple(
            hydrogen_bond
            for hydrogen_bond
            in self.hydrogen_bonds
            if hydrogen_bond.direction
            == normalized
        )

    def get_residue_hydrogen_bond(
        self,
        key: ResidueContactKey,
    ) -> Optional[
        ResidueHydrogenBond
    ]:
        """
        Return the residue-level result associated with a key.

        Parameters
        ----------
        key : ResidueContactKey
            Residue key.

        Returns
        -------
        ResidueHydrogenBond or None
            Matching result.
        """

        normalized_key = (
            _normalize_residue_key(
                key
            )
        )

        for residue_result in (
            self.residue_hydrogen_bonds
        ):
            if (
                residue_result.key
                == normalized_key
            ):
                return residue_result

        return None

    def to_dict(
        self,
        *,
        include_bonds: bool = True,
        include_residues: bool = True,
        include_atoms: bool = False,
    ) -> Dict[
        str,
        Any,
    ]:
        """
        Serialize the complete analysis result.

        Parameters
        ----------
        include_bonds : bool, optional
            Whether individual hydrogen bonds should be included.
        include_residues : bool, optional
            Whether residue-grouped results should be included.
        include_atoms : bool, optional
            Whether raw atom and residue objects should be included.

        Returns
        -------
        dict
            Serializable analysis representation.
        """

        result: Dict[
            str,
            Any,
        ] = {
            "hydrogen_bond_count": (
                self.hydrogen_bond_count
            ),
            "residue_count": (
                self.residue_count
            ),
            "ligand_atom_count": (
                self.ligand_atom_count
            ),
            "receptor_atom_count": (
                self.receptor_atom_count
            ),
            "explicit_count": (
                self.explicit_count
            ),
            "inferred_count": (
                self.inferred_count
            ),
            "strong_count": (
                self.strong_count
            ),
            "ligand_donor_count": (
                self.ligand_donor_count
            ),
            "receptor_donor_count": (
                self.receptor_donor_count
            ),
            "minimum_distance": (
                None
                if self.minimum_distance is None
                else float(
                    self.minimum_distance
                )
            ),
            "mean_distance": (
                None
                if self.mean_distance is None
                else float(
                    self.mean_distance
                )
            ),
            "donor_acceptor_cutoff": float(
                self.donor_acceptor_cutoff
            ),
            "hydrogen_acceptor_cutoff": (
                None
                if self.hydrogen_acceptor_cutoff
                is None
                else float(
                    self.hydrogen_acceptor_cutoff
                )
            ),
            "minimum_dha_angle": (
                None
                if self.minimum_dha_angle
                is None
                else float(
                    self.minimum_dha_angle
                )
            ),
            "minimum_inferred_angle": (
                None
                if self.minimum_inferred_angle
                is None
                else float(
                    self.minimum_inferred_angle
                )
            ),
            "has_hydrogen_bonds": (
                self.has_hydrogen_bonds
            ),
            "has_explicit_bonds": (
                self.has_explicit_bonds
            ),
            "has_inferred_bonds": (
                self.has_inferred_bonds
            ),
            "has_strong_bonds": (
                self.has_strong_bonds
            ),
            "statistics": dict(
                self.statistics
            ),
            "metadata": dict(
                self.metadata
            ),
        }

        if include_bonds:
            result[
                "hydrogen_bonds"
            ] = [
                hydrogen_bond.to_dict(
                    include_atoms=include_atoms
                )
                for hydrogen_bond
                in self.hydrogen_bonds
            ]

        if include_residues:
            result[
                "residue_hydrogen_bonds"
            ] = [
                residue_result.to_dict(
                    include_bonds=include_bonds,
                    include_atoms=include_atoms,
                )
                for residue_result
                in self.residue_hydrogen_bonds
            ]

        if include_atoms:
            result[
                "ligand_atoms"
            ] = self.ligand_atoms

            result[
                "receptor_atoms"
            ] = self.receptor_atoms

        return result


# -----------------------------------------------------------------------------
# Empty typed result collections
# -----------------------------------------------------------------------------

_EMPTY_HYDROGEN_BONDS: Final[
    Tuple[
        HydrogenBond,
        ...,
    ]
] = ()

_EMPTY_RESIDUE_HYDROGEN_BONDS: Final[
    Tuple[
        ResidueHydrogenBond,
        ...,
    ]
] = ()


# -----------------------------------------------------------------------------
# Public interface
# -----------------------------------------------------------------------------

_SECTION_3_PUBLIC_NAMES: Final[
    Tuple[
        str,
        ...,
    ]
] = (
    "HydrogenBondGeometry",
    "HydrogenBond",
    "ResidueHydrogenBond",
    "HydrogenBondAnalysisResult",
)

for public_name in _SECTION_3_PUBLIC_NAMES:
    if public_name not in __all__:
        __all__.append(
            public_name
        )


# =============================================================================
# End of Section 3
# =============================================================================


# =============================================================================
# Section 4 — Donor and acceptor identification
# =============================================================================


# -----------------------------------------------------------------------------
# Protein chemical-perception tables
# -----------------------------------------------------------------------------

# Backbone atoms are treated separately because their hydrogen-bond roles are
# largely independent of the amino-acid side chain.
_PROTEIN_BACKBONE_DONOR_ATOM_NAMES: Final[
    AtomNameSet
] = frozenset(
    {
        "N",
        "NT",
        "NTER",
    }
)

_PROTEIN_BACKBONE_ACCEPTOR_ATOM_NAMES: Final[
    AtomNameSet
] = frozenset(
    {
        "O",
        "OXT",
        "OT1",
        "OT2",
        "O1",
        "O2",
    }
)


# Residue-specific side-chain donor atoms.
_PROTEIN_SIDECHAIN_DONOR_ATOMS: Final[
    Mapping[
        str,
        AtomNameSet,
    ]
] = MappingProxyType(
    {
        "ARG": frozenset(
            {
                "NE",
                "NH1",
                "NH2",
            }
        ),
        "ASN": frozenset(
            {
                "ND2",
            }
        ),
        "CYS": frozenset(
            {
                "SG",
            }
        ),
        "CYM": frozenset(),
        "CYX": frozenset(),
        "GLN": frozenset(
            {
                "NE2",
            }
        ),
        "HIS": frozenset(
            {
                "ND1",
                "NE2",
            }
        ),
        "HID": frozenset(
            {
                "ND1",
            }
        ),
        "HIE": frozenset(
            {
                "NE2",
            }
        ),
        "HIP": frozenset(
            {
                "ND1",
                "NE2",
            }
        ),
        "HSD": frozenset(
            {
                "ND1",
            }
        ),
        "HSE": frozenset(
            {
                "NE2",
            }
        ),
        "HSP": frozenset(
            {
                "ND1",
                "NE2",
            }
        ),
        "LYS": frozenset(
            {
                "NZ",
            }
        ),
        "LYN": frozenset(
            {
                "NZ",
            }
        ),
        "SER": frozenset(
            {
                "OG",
            }
        ),
        "THR": frozenset(
            {
                "OG1",
            }
        ),
        "TRP": frozenset(
            {
                "NE1",
            }
        ),
        "TYR": frozenset(
            {
                "OH",
            }
        ),
    }
)


# Residue-specific side-chain acceptor atoms.
_PROTEIN_SIDECHAIN_ACCEPTOR_ATOMS: Final[
    Mapping[
        str,
        AtomNameSet,
    ]
] = MappingProxyType(
    {
        "ASN": frozenset(
            {
                "OD1",
            }
        ),
        "ASP": frozenset(
            {
                "OD1",
                "OD2",
            }
        ),
        "ASH": frozenset(
            {
                "OD1",
                "OD2",
            }
        ),
        "CYS": frozenset(
            {
                "SG",
            }
        ),
        "CYM": frozenset(
            {
                "SG",
            }
        ),
        "CYX": frozenset(),
        "GLN": frozenset(
            {
                "OE1",
            }
        ),
        "GLU": frozenset(
            {
                "OE1",
                "OE2",
            }
        ),
        "GLH": frozenset(
            {
                "OE1",
                "OE2",
            }
        ),
        "HIS": frozenset(
            {
                "ND1",
                "NE2",
            }
        ),
        "HID": frozenset(
            {
                "NE2",
            }
        ),
        "HIE": frozenset(
            {
                "ND1",
            }
        ),
        "HIP": frozenset(),
        "HSD": frozenset(
            {
                "NE2",
            }
        ),
        "HSE": frozenset(
            {
                "ND1",
            }
        ),
        "HSP": frozenset(),
        "LYN": frozenset(
            {
                "NZ",
            }
        ),
        "MET": frozenset(
            {
                "SD",
            }
        ),
        "SER": frozenset(
            {
                "OG",
            }
        ),
        "THR": frozenset(
            {
                "OG1",
            }
        ),
        "TYR": frozenset(
            {
                "OH",
            }
        ),
    }
)


# Residue variants that should still be recognized as protein residues.
_ADDITIONAL_PROTEIN_RESIDUE_NAMES: Final[
    ResidueNameSet
] = frozenset(
    {
        "ACE",
        "NME",
        "ASH",
        "GLH",
        "LYN",
        "CYM",
        "CYX",
        "HID",
        "HIE",
        "HIP",
        "HSD",
        "HSE",
        "HSP",
    }
)

_PROTEIN_RESIDUE_NAMES: Final[
    ResidueNameSet
] = frozenset(
    set(
        STANDARD_AMINO_ACID_NAMES
    )
    | set(
        _ADDITIONAL_PROTEIN_RESIDUE_NAMES
    )
)


# Proline backbone nitrogen normally lacks a transferable hydrogen.
_NON_DONOR_BACKBONE_RESIDUES: Final[
    ResidueNameSet
] = frozenset(
    {
        "PRO",
        "HYP",
    }
)


# Permanently or usually protonated residue forms whose listed nitrogen atoms
# should not be treated as acceptors.
_PROTONATED_PROTEIN_NITROGENS: Final[
    Mapping[
        str,
        AtomNameSet,
    ]
] = MappingProxyType(
    {
        "ARG": frozenset(
            {
                "NE",
                "NH1",
                "NH2",
            }
        ),
        "LYS": frozenset(
            {
                "NZ",
            }
        ),
        "HIP": frozenset(
            {
                "ND1",
                "NE2",
            }
        ),
        "HSP": frozenset(
            {
                "ND1",
                "NE2",
            }
        ),
    }
)


# Amide-like nitrogens are donors when protonated but not acceptors.
_PROTEIN_AMIDE_NITROGENS: Final[
    Mapping[
        str,
        AtomNameSet,
    ]
] = MappingProxyType(
    {
        "ASN": frozenset(
            {
                "ND2",
            }
        ),
        "GLN": frozenset(
            {
                "NE2",
            }
        ),
    }
)


# -----------------------------------------------------------------------------
# Generic chemical-perception constants
# -----------------------------------------------------------------------------

# Elements that may act as donors in ordinary molecular docking systems.
_GENERIC_DONOR_ELEMENTS: Final[
    ElementSet
] = frozenset(
    {
        "N",
        "O",
        "S",
    }
)

# Elements that may act as conventional acceptors. Halogens are deliberately
# excluded because they are not standard hydrogen-bond acceptors in this model.
_GENERIC_ACCEPTOR_ELEMENTS: Final[
    ElementSet
] = frozenset(
    {
        "N",
        "O",
        "S",
    }
)

# Approximate maximum heavy-atom valences used only as defensive fallbacks.
_GENERIC_MAXIMUM_VALENCE: Final[
    Mapping[
        str,
        int,
    ]
] = MappingProxyType(
    {
        "N": 4,
        "O": 3,
        "S": 6,
    }
)

# Common atom-type strings encountered in MOL2, PDBQT and force-field objects.
_AMIDE_NITROGEN_TYPES: Final[
    FrozenSet[
        str
    ]
] = frozenset(
    {
        "N.AM",
        "NAM",
        "AMIDE_N",
        "AMIDE-N",
        "N_AMIDE",
    }
)

_AROMATIC_NITROGEN_DONOR_TYPES: Final[
    FrozenSet[
        str
    ]
] = frozenset(
    {
        "N.AR",
        "NAR",
        "N.H",
        "N.PYRROLE",
        "PYRROLE_N",
    }
)

_AROMATIC_NITROGEN_ACCEPTOR_TYPES: Final[
    FrozenSet[
        str
    ]
] = frozenset(
    {
        "N.AR",
        "NAR",
        "N.PYR",
        "N.PYRIDINE",
        "PYRIDINE_N",
    }
)

_POSITIVELY_CHARGED_NITROGEN_TYPES: Final[
    FrozenSet[
        str
    ]
] = frozenset(
    {
        "N.4",
        "N4",
        "N.PL3+",
        "N+",
        "QUATERNARY_N",
        "PROTONATED_N",
    }
)

_NEGATIVELY_CHARGED_OXYGEN_TYPES: Final[
    FrozenSet[
        str
    ]
] = frozenset(
    {
        "O.CO2",
        "O-",
        "O.MINUS",
        "CARBOXYLATE_O",
    }
)

_NON_ACCEPTOR_OXYGEN_TYPES: Final[
    FrozenSet[
        str
    ]
] = frozenset(
    {
        "O+",
        "O.PLUS",
        "PROTONATED_O",
        "OXONIUM_O",
    }
)

_NON_ACCEPTOR_SULFUR_TYPES: Final[
    FrozenSet[
        str
    ]
] = frozenset(
    {
        "S+",
        "S.PLUS",
        "SULFONIUM_S",
    }
)


# -----------------------------------------------------------------------------
# Generic object-value access
# -----------------------------------------------------------------------------

_MISSING_CHEMICAL_VALUE: Final[
    object
] = object()


def _get_chemical_object_value(
    object_: Any,
    names: Sequence[str],
    *,
    default: Any = None,
) -> Any:
    """
    Retrieve a chemical attribute from an object or mapping.

    Parameters
    ----------
    object_ : Any
        Object or mapping.
    names : sequence of str
        Candidate names, checked in order.
    default : Any, optional
        Value returned when no candidate is available.

    Returns
    -------
    Any
        First available value or ``default``.

    Notes
    -----
    Zero, ``False`` and empty collections are considered valid values.
    Callables without required arguments are invoked automatically.
    """

    if object_ is None:
        return default

    for name in names:
        value: Any = _MISSING_CHEMICAL_VALUE

        if isinstance(
            object_,
            Mapping,
        ):
            if name in object_:
                value = object_[
                    name
                ]

        else:
            try:
                value = getattr(
                    object_,
                    name,
                )

            except (
                AttributeError,
                TypeError,
            ):
                continue

        if value is _MISSING_CHEMICAL_VALUE:
            continue

        if callable(
            value
        ):
            try:
                value = value()

            except TypeError:
                continue

            except Exception:
                continue

        return value

    return default


def _normalize_chemical_text(
    value: Any,
) -> str:
    """
    Normalize a chemical text value.

    Parameters
    ----------
    value : Any
        Value to normalize.

    Returns
    -------
    str
        Uppercase stripped text.
    """

    if value is None:
        return ""

    try:
        text = str(
            value
        )

    except Exception:
        return ""

    return text.strip().upper()


# -----------------------------------------------------------------------------
# Residue and atom descriptors
# -----------------------------------------------------------------------------

def get_hbond_residue_name(
    atom_or_residue: Any,
) -> str:
    """
    Return the normalized residue name.

    Parameters
    ----------
    atom_or_residue : Any
        Atom-like or residue-like object.

    Returns
    -------
    str
        Uppercase residue name, or an empty string.
    """

    residue = atom_or_residue

    try:
        candidate_residue = get_atom_residue(
            atom_or_residue
        )

    except Exception:
        candidate_residue = None

    if candidate_residue is not None:
        residue = candidate_residue

    value = _get_chemical_object_value(
        residue,
        (
            "name",
            "resname",
            "residue_name",
            "type",
        ),
        default="",
    )

    return _normalize_chemical_text(
        value
    )


def get_atom_type(
    atom: AtomLike,
) -> str:
    """
    Return a normalized chemical atom type.

    Parameters
    ----------
    atom : atom-like
        Atom to inspect.

    Returns
    -------
    str
        Uppercase atom type, or an empty string.

    Notes
    -----
    Candidate attributes include ChimeraX, MOL2, PDBQT and force-field naming
    conventions.
    """

    value = _get_chemical_object_value(
        atom,
        (
            "idatm_type",
            "atom_type",
            "gaff_type",
            "sybyl_type",
            "mol2_type",
            "type_name",
            "type",
        ),
        default="",
    )

    return _normalize_chemical_text(
        value
    )


def get_atom_formal_charge(
    atom: AtomLike,
) -> Optional[
    np.float64
]:
    """
    Return an atom's formal charge when available.

    Parameters
    ----------
    atom : atom-like
        Atom to inspect.

    Returns
    -------
    numpy.float64 or None
        Formal charge.

    Notes
    -----
    Partial-charge fields such as PDBQT Gasteiger charges are not used as
    formal charges unless the object exposes no more specific field and marks
    that value explicitly as formal.
    """

    value = _get_chemical_object_value(
        atom,
        (
            "formal_charge",
            "formalCharge",
            "formalcharge",
            "integer_charge",
        ),
        default=None,
    )

    if value is None:
        element = _get_chemical_object_value(
            atom,
            (
                "element",
            ),
            default=None,
        )

        value = _get_chemical_object_value(
            element,
            (
                "formal_charge",
                "formalCharge",
            ),
            default=None,
        )

    if value is None:
        return None

    if isinstance(
        value,
        (
            bool,
            np.bool_,
        ),
    ):
        return None

    try:
        numeric_value = np.float64(
            value
        )

    except (
        TypeError,
        ValueError,
        OverflowError,
    ):
        return None

    if not np.isfinite(
        numeric_value
    ):
        return None

    return numeric_value


def atom_is_aromatic(
    atom: AtomLike,
) -> bool:
    """
    Determine whether an atom is aromatic.

    Parameters
    ----------
    atom : atom-like
        Atom to inspect.

    Returns
    -------
    bool
        Aromaticity status.
    """

    value = _get_chemical_object_value(
        atom,
        (
            "is_aromatic",
            "aromatic",
            "isAromatic",
        ),
        default=None,
    )

    if value is not None:
        try:
            return bool(
                value
            )

        except Exception:
            pass

    atom_type = get_atom_type(
        atom
    )

    return (
        ".AR" in atom_type
        or atom_type.endswith(
            "AR"
        )
        or "AROMATIC" in atom_type
        or atom_type
        in (
            _AROMATIC_NITROGEN_DONOR_TYPES
            | _AROMATIC_NITROGEN_ACCEPTOR_TYPES
        )
    )


# -----------------------------------------------------------------------------
# Bonded-neighbor access
# -----------------------------------------------------------------------------

def _extract_atoms_from_bond(
    bond: Any,
) -> Tuple[
    AtomLike,
    ...,
]:
    """
    Extract atoms from a bond-like object.

    Parameters
    ----------
    bond : Any
        Bond-like object.

    Returns
    -------
    tuple of atom-like
        Bond atoms.
    """

    if bond is None:
        return ()

    atoms = _get_chemical_object_value(
        bond,
        (
            "atoms",
            "atom_pair",
            "ends",
        ),
        default=None,
    )

    if atoms is not None:
        try:
            return tuple(
                atoms
            )

        except TypeError:
            pass

    atom_1 = _get_chemical_object_value(
        bond,
        (
            "atom1",
            "atom_1",
            "a1",
        ),
        default=None,
    )

    atom_2 = _get_chemical_object_value(
        bond,
        (
            "atom2",
            "atom_2",
            "a2",
        ),
        default=None,
    )

    return tuple(
        atom
        for atom in (
            atom_1,
            atom_2,
        )
        if atom is not None
    )


def get_bonded_neighbors(
    atom: AtomLike,
    *,
    bond_resolver: Optional[
        BondResolver
    ] = None,
) -> Tuple[
    AtomLike,
    ...,
]:
    """
    Return atoms covalently bonded to an atom.

    Parameters
    ----------
    atom : atom-like
        Central atom.
    bond_resolver : callable or None, optional
        Custom neighbor resolver.

    Returns
    -------
    tuple of atom-like
        Unique bonded neighbors.

    Notes
    -----
    The function supports direct neighbor collections, bond collections,
    synthetic objects and mappings. Returned order follows the source order.
    """

    validate_atom(
        atom,
        require_coordinate=False,
    )

    if bond_resolver is not None:
        try:
            raw_neighbors = bond_resolver(
                atom
            )

        except Exception as error:
            raise ValueError(
                "The custom bond resolver failed."
            ) from error

        try:
            candidates = tuple(
                raw_neighbors
            )

        except TypeError as error:
            raise TypeError(
                "The custom bond resolver must return an iterable."
            ) from error

    else:
        raw_neighbors = _get_chemical_object_value(
            atom,
            (
                "neighbors",
                "bonded_neighbors",
                "bonded_atoms",
                "connected_atoms",
            ),
            default=None,
        )

        if raw_neighbors is not None:
            try:
                candidates = tuple(
                    raw_neighbors
                )

            except TypeError:
                candidates = ()

        else:
            raw_bonds = _get_chemical_object_value(
                atom,
                (
                    "bonds",
                    "bond_objects",
                ),
                default=(),
            )

            try:
                bonds = tuple(
                    raw_bonds
                )

            except TypeError:
                bonds = ()

            extracted_neighbors: List[
                AtomLike
            ] = []

            for bond in bonds:
                for bonded_atom in _extract_atoms_from_bond(
                    bond
                ):
                    if bonded_atom is not atom:
                        extracted_neighbors.append(
                            bonded_atom
                        )

            candidates = tuple(
                extracted_neighbors
            )

    unique_neighbors: List[
        AtomLike
    ] = []

    seen_ids: Set[
        int
    ] = set()

    for candidate in candidates:
        if candidate is None or candidate is atom:
            continue

        candidate_identity = id(
            candidate
        )

        if candidate_identity in seen_ids:
            continue

        if not is_atom_like(
            candidate
        ):
            continue

        seen_ids.add(
            candidate_identity
        )

        unique_neighbors.append(
            candidate
        )

    return tuple(
        unique_neighbors
    )


def get_bonded_hydrogens(
    atom: AtomLike,
    *,
    bond_resolver: Optional[
        BondResolver
    ] = None,
) -> Tuple[
    AtomLike,
    ...,
]:
    """
    Return explicit hydrogen atoms bonded to an atom.

    Parameters
    ----------
    atom : atom-like
        Heavy atom.
    bond_resolver : callable or None, optional
        Custom bonded-neighbor resolver.

    Returns
    -------
    tuple of atom-like
        Bonded hydrogens.
    """

    return tuple(
        neighbor
        for neighbor in get_bonded_neighbors(
            atom,
            bond_resolver=bond_resolver,
        )
        if is_hydrogen_atom(
            neighbor
        )
    )


def get_bonded_heavy_atoms(
    atom: AtomLike,
    *,
    bond_resolver: Optional[
        BondResolver
    ] = None,
) -> Tuple[
    AtomLike,
    ...,
]:
    """
    Return explicit heavy atoms bonded to an atom.

    Parameters
    ----------
    atom : atom-like
        Atom to inspect.
    bond_resolver : callable or None, optional
        Custom bonded-neighbor resolver.

    Returns
    -------
    tuple of atom-like
        Bonded heavy atoms.
    """

    return tuple(
        neighbor
        for neighbor in get_bonded_neighbors(
            atom,
            bond_resolver=bond_resolver,
        )
        if is_heavy_atom(
            neighbor
        )
    )


def atom_has_explicit_hydrogen(
    atom: AtomLike,
    *,
    bond_resolver: Optional[
        BondResolver
    ] = None,
) -> bool:
    """
    Determine whether an atom has a bonded explicit hydrogen.

    Parameters
    ----------
    atom : atom-like
        Atom to inspect.
    bond_resolver : callable or None, optional
        Custom bonded-neighbor resolver.

    Returns
    -------
    bool
        Explicit-hydrogen status.
    """

    return bool(
        get_bonded_hydrogens(
            atom,
            bond_resolver=bond_resolver,
        )
    )


# -----------------------------------------------------------------------------
# Bond descriptors
# -----------------------------------------------------------------------------

def get_bond_order(
    atom_1: AtomLike,
    atom_2: AtomLike,
) -> Optional[
    np.float64
]:
    """
    Return the bond order between two atoms when available.

    Parameters
    ----------
    atom_1 : atom-like
        First atom.
    atom_2 : atom-like
        Second atom.

    Returns
    -------
    numpy.float64 or None
        Bond order.

    Notes
    -----
    Aromatic bonds represented by text are returned as ``1.5``.
    """

    raw_bonds = _get_chemical_object_value(
        atom_1,
        (
            "bonds",
            "bond_objects",
        ),
        default=(),
    )

    try:
        bonds = tuple(
            raw_bonds
        )

    except TypeError:
        bonds = ()

    for bond in bonds:
        bond_atoms = _extract_atoms_from_bond(
            bond
        )

        if (
            atom_1 not in bond_atoms
            or atom_2 not in bond_atoms
        ):
            continue

        raw_order = _get_chemical_object_value(
            bond,
            (
                "order",
                "bond_order",
                "bondOrder",
                "type",
            ),
            default=None,
        )

        if raw_order is None:
            return None

        normalized_text = _normalize_chemical_text(
            raw_order
        )

        if normalized_text in {
            "AR",
            "AROMATIC",
            "1.5",
        }:
            return np.float64(
                1.5
            )

        try:
            numeric_order = np.float64(
                raw_order
            )

        except (
            TypeError,
            ValueError,
            OverflowError,
        ):
            return None

        if np.isfinite(
            numeric_order
        ):
            return numeric_order

        return None

    return None


def _is_bonded_to_carbonyl_carbon(
    atom: AtomLike,
    *,
    bond_resolver: Optional[
        BondResolver
    ] = None,
) -> bool:
    """
    Determine whether an atom is attached to a carbonyl-like carbon.

    Parameters
    ----------
    atom : atom-like
        Atom to inspect.
    bond_resolver : callable or None, optional
        Custom neighbor resolver.

    Returns
    -------
    bool
        Carbonyl-adjacency status.
    """

    for neighbor in get_bonded_heavy_atoms(
        atom,
        bond_resolver=bond_resolver,
    ):
        if get_atom_element(
            neighbor
        ) != CARBON_ELEMENT:
            continue

        neighbor_type = get_atom_type(
            neighbor
        )

        if neighbor_type in {
            "C.2",
            "C.CAR",
            "CARBONYL_C",
            "AMIDE_C",
        }:
            return True

        for second_neighbor in get_bonded_heavy_atoms(
            neighbor,
            bond_resolver=bond_resolver,
        ):
            if second_neighbor is atom:
                continue

            if get_atom_element(
                second_neighbor
            ) != OXYGEN_ELEMENT:
                continue

            bond_order = get_bond_order(
                neighbor,
                second_neighbor,
            )

            if (
                bond_order is not None
                and bond_order
                >= np.float64(
                    1.75
                )
            ):
                return True

    return False


# -----------------------------------------------------------------------------
# Protein-residue perception
# -----------------------------------------------------------------------------

def is_protein_residue(
    atom_or_residue: Any,
) -> bool:
    """
    Determine whether an object belongs to a recognized protein residue.

    Parameters
    ----------
    atom_or_residue : Any
        Atom-like or residue-like object.

    Returns
    -------
    bool
        Protein-residue status.
    """

    residue_name = get_hbond_residue_name(
        atom_or_residue
    )

    if residue_name in _PROTEIN_RESIDUE_NAMES:
        return True

    residue = atom_or_residue

    try:
        candidate_residue = get_atom_residue(
            atom_or_residue
        )

    except Exception:
        candidate_residue = None

    if candidate_residue is not None:
        residue = candidate_residue

    polymer_type = _normalize_chemical_text(
        _get_chemical_object_value(
            residue,
            (
                "polymer_type",
                "polymerType",
                "category",
            ),
            default="",
        )
    )

    return polymer_type in {
        "AMINO",
        "AMINO_ACID",
        "PROTEIN",
        "PEPTIDE",
    }


def is_protein_backbone_atom(
    atom: AtomLike,
) -> bool:
    """
    Determine whether an atom is part of a protein backbone.

    Parameters
    ----------
    atom : atom-like
        Atom to inspect.

    Returns
    -------
    bool
        Protein-backbone status.
    """

    if not is_protein_residue(
        atom
    ):
        return False

    atom_name = get_atom_name(
        atom
    ).strip().upper()

    return atom_name in {
        "N",
        "CA",
        "C",
        "O",
        "OXT",
        "OT1",
        "OT2",
        "H",
        "HN",
        "H1",
        "H2",
        "H3",
        "HA",
    }


def _protein_atom_is_known_donor(
    atom: AtomLike,
) -> Optional[
    bool
]:
    """
    Resolve donor status using protein residue tables.

    Parameters
    ----------
    atom : atom-like
        Protein atom.

    Returns
    -------
    bool or None
        ``True`` or ``False`` when resolved, otherwise ``None``.
    """

    residue_name = get_hbond_residue_name(
        atom
    )

    atom_name = get_atom_name(
        atom
    ).strip().upper()

    if atom_name in _PROTEIN_BACKBONE_DONOR_ATOM_NAMES:
        return (
            residue_name
            not in _NON_DONOR_BACKBONE_RESIDUES
        )

    residue_donors = (
        _PROTEIN_SIDECHAIN_DONOR_ATOMS.get(
            residue_name
        )
    )

    if residue_donors is None:
        return None

    return atom_name in residue_donors


def _protein_atom_is_known_acceptor(
    atom: AtomLike,
) -> Optional[
    bool
]:
    """
    Resolve acceptor status using protein residue tables.

    Parameters
    ----------
    atom : atom-like
        Protein atom.

    Returns
    -------
    bool or None
        ``True`` or ``False`` when resolved, otherwise ``None``.
    """

    residue_name = get_hbond_residue_name(
        atom
    )

    atom_name = get_atom_name(
        atom
    ).strip().upper()

    if atom_name in _PROTEIN_BACKBONE_ACCEPTOR_ATOM_NAMES:
        return True

    if atom_name in _PROTEIN_BACKBONE_DONOR_ATOM_NAMES:
        return False

    amide_nitrogens = (
        _PROTEIN_AMIDE_NITROGENS.get(
            residue_name,
            frozenset(),
        )
    )

    if atom_name in amide_nitrogens:
        return False

    protonated_nitrogens = (
        _PROTONATED_PROTEIN_NITROGENS.get(
            residue_name,
            frozenset(),
        )
    )

    if atom_name in protonated_nitrogens:
        return False

    residue_acceptors = (
        _PROTEIN_SIDECHAIN_ACCEPTOR_ATOMS.get(
            residue_name
        )
    )

    if residue_acceptors is None:
        return None

    return atom_name in residue_acceptors


# -----------------------------------------------------------------------------
# Generic donor perception
# -----------------------------------------------------------------------------

def _generic_atom_can_donate(
    atom: AtomLike,
    *,
    require_explicit_hydrogen: bool,
    bond_resolver: Optional[
        BondResolver
    ],
) -> bool:
    """
    Apply generic donor chemical-perception rules.

    Parameters
    ----------
    atom : atom-like
        Atom to inspect.
    require_explicit_hydrogen : bool
        Whether a bonded explicit hydrogen is mandatory.
    bond_resolver : callable or None
        Custom bonded-neighbor resolver.

    Returns
    -------
    bool
        Donor status.
    """

    element = get_atom_element(
        atom
    )

    if element not in _GENERIC_DONOR_ELEMENTS:
        return False

    formal_charge = get_atom_formal_charge(
        atom
    )

    if (
        formal_charge is not None
        and formal_charge
        <= np.float64(
            -1.0
        )
    ):
        # Strongly anionic N/O/S atoms are generally not proton donors unless
        # the explicit topology proves otherwise.
        if not atom_has_explicit_hydrogen(
            atom,
            bond_resolver=bond_resolver,
        ):
            return False

    has_hydrogen = atom_has_explicit_hydrogen(
        atom,
        bond_resolver=bond_resolver,
    )

    if require_explicit_hydrogen:
        return has_hydrogen

    if has_hydrogen:
        return True

    atom_type = get_atom_type(
        atom
    )

    if atom_type in _AROMATIC_NITROGEN_DONOR_TYPES:
        return True

    if element == NITROGEN_ELEMENT:
        heavy_neighbor_count = len(
            get_bonded_heavy_atoms(
                atom,
                bond_resolver=bond_resolver,
            )
        )

        if (
            formal_charge is not None
            and formal_charge > 0.0
        ):
            # Protonated amines and guanidinium-like nitrogens may donate.
            return heavy_neighbor_count <= 3

        # Neutral N with fewer than three known heavy neighbors may possess
        # an implicit hydrogen.
        return heavy_neighbor_count < 3

    if element == OXYGEN_ELEMENT:
        heavy_neighbor_count = len(
            get_bonded_heavy_atoms(
                atom,
                bond_resolver=bond_resolver,
            )
        )

        # Hydroxyl-like oxygen generally has one heavy neighbor. Water oxygen
        # has zero heavy neighbors.
        return heavy_neighbor_count <= 1

    if element == SULFUR_ELEMENT:
        heavy_neighbor_count = len(
            get_bonded_heavy_atoms(
                atom,
                bond_resolver=bond_resolver,
            )
        )

        # Thiol-like sulfur.
        return heavy_neighbor_count <= 1

    return False


def is_hbond_donor(
    atom: AtomLike,
    *,
    require_explicit_hydrogen: bool = False,
    use_protein_templates: bool = True,
    bond_resolver: Optional[
        BondResolver
    ] = None,
    custom_predicate: Optional[
        AtomPredicate
    ] = None,
) -> bool:
    """
    Determine whether an atom can act as a hydrogen-bond donor.

    Parameters
    ----------
    atom : atom-like
        Atom to inspect.
    require_explicit_hydrogen : bool, optional
        Whether the donor must have a bonded explicit hydrogen.
    use_protein_templates : bool, optional
        Whether residue and atom-name templates should be used for proteins.
    bond_resolver : callable or None, optional
        Custom bonded-neighbor resolver.
    custom_predicate : callable or None, optional
        Additional donor predicate. A ``True`` result accepts the atom after
        basic element validation; a ``False`` result rejects it.

    Returns
    -------
    bool
        Donor status.
    """

    try:
        validate_atom(
            atom,
            require_coordinate=False,
        )

    except Exception:
        return False

    if is_hydrogen_atom(
        atom
    ):
        return False

    element = get_atom_element(
        atom
    )

    if element not in _GENERIC_DONOR_ELEMENTS:
        return False

    if element in COMMON_METAL_ELEMENTS:
        return False

    if custom_predicate is not None:
        try:
            custom_result = bool(
                custom_predicate(
                    atom
                )
            )

        except Exception as error:
            raise ValueError(
                "The custom donor predicate failed."
            ) from error

        if not custom_result:
            return False

    if (
        use_protein_templates
        and is_protein_residue(
            atom
        )
    ):
        template_result = (
            _protein_atom_is_known_donor(
                atom
            )
        )

        if template_result is not None:
            if not template_result:
                return False

            if require_explicit_hydrogen:
                return atom_has_explicit_hydrogen(
                    atom,
                    bond_resolver=bond_resolver,
                )

            return True

    return _generic_atom_can_donate(
        atom,
        require_explicit_hydrogen=(
            require_explicit_hydrogen
        ),
        bond_resolver=bond_resolver,
    )


# -----------------------------------------------------------------------------
# Generic acceptor perception
# -----------------------------------------------------------------------------

def _generic_nitrogen_is_acceptor(
    atom: AtomLike,
    *,
    bond_resolver: Optional[
        BondResolver
    ],
) -> bool:
    """
    Determine generic nitrogen acceptor status.

    Parameters
    ----------
    atom : atom-like
        Nitrogen atom.
    bond_resolver : callable or None
        Custom neighbor resolver.

    Returns
    -------
    bool
        Acceptor status.
    """

    formal_charge = get_atom_formal_charge(
        atom
    )

    atom_type = get_atom_type(
        atom
    )

    if (
        formal_charge is not None
        and formal_charge > 0.0
    ):
        return False

    if atom_type in _POSITIVELY_CHARGED_NITROGEN_TYPES:
        return False

    if atom_type in _AMIDE_NITROGEN_TYPES:
        return False

    if _is_bonded_to_carbonyl_carbon(
        atom,
        bond_resolver=bond_resolver,
    ):
        return False

    bonded_hydrogens = get_bonded_hydrogens(
        atom,
        bond_resolver=bond_resolver,
    )

    heavy_neighbors = get_bonded_heavy_atoms(
        atom,
        bond_resolver=bond_resolver,
    )

    total_known_valence = (
        len(
            bonded_hydrogens
        )
        + len(
            heavy_neighbors
        )
    )

    if total_known_valence >= 4:
        return False

    if atom_is_aromatic(
        atom
    ):
        if atom_type in _AROMATIC_NITROGEN_DONOR_TYPES:
            # Pyrrole-like aromatic nitrogen contributes its lone pair to the
            # aromatic system and is not an acceptor.
            if bonded_hydrogens:
                return False

        if atom_type in _AROMATIC_NITROGEN_ACCEPTOR_TYPES:
            return not bonded_hydrogens

        # Aromatic N-H is pyrrole-like; aromatic N without H is pyridine-like.
        return not bonded_hydrogens

    # Neutral amines, imines and nitriles may accept unless excluded above.
    return True


def _generic_oxygen_is_acceptor(
    atom: AtomLike,
    *,
    bond_resolver: Optional[
        BondResolver
    ],
) -> bool:
    """
    Determine generic oxygen acceptor status.

    Parameters
    ----------
    atom : atom-like
        Oxygen atom.
    bond_resolver : callable or None
        Custom neighbor resolver.

    Returns
    -------
    bool
        Acceptor status.
    """

    formal_charge = get_atom_formal_charge(
        atom
    )

    atom_type = get_atom_type(
        atom
    )

    if atom_type in _NON_ACCEPTOR_OXYGEN_TYPES:
        return False

    if (
        formal_charge is not None
        and formal_charge > 0.0
    ):
        return False

    if atom_type in _NEGATIVELY_CHARGED_OXYGEN_TYPES:
        return True

    heavy_neighbors = get_bonded_heavy_atoms(
        atom,
        bond_resolver=bond_resolver,
    )

    bonded_hydrogens = get_bonded_hydrogens(
        atom,
        bond_resolver=bond_resolver,
    )

    # Neutral hydroxyl oxygen remains a weak-to-moderate acceptor in the
    # simplified docking model, except when positively charged.
    if (
        len(
            heavy_neighbors
        )
        <= 2
        and len(
            bonded_hydrogens
        )
        <= 2
    ):
        return True

    return len(
        heavy_neighbors
    ) <= 2


def _generic_sulfur_is_acceptor(
    atom: AtomLike,
    *,
    bond_resolver: Optional[
        BondResolver
    ],
) -> bool:
    """
    Determine generic sulfur acceptor status.

    Parameters
    ----------
    atom : atom-like
        Sulfur atom.
    bond_resolver : callable or None
        Custom neighbor resolver.

    Returns
    -------
    bool
        Acceptor status.
    """

    formal_charge = get_atom_formal_charge(
        atom
    )

    atom_type = get_atom_type(
        atom
    )

    if atom_type in _NON_ACCEPTOR_SULFUR_TYPES:
        return False

    if (
        formal_charge is not None
        and formal_charge > 0.0
    ):
        return False

    heavy_neighbors = get_bonded_heavy_atoms(
        atom,
        bond_resolver=bond_resolver,
    )

    if len(
        heavy_neighbors
    ) > _GENERIC_MAXIMUM_VALENCE[
        "S"
    ]:
        return False

    return True


def is_hbond_acceptor(
    atom: AtomLike,
    *,
    use_protein_templates: bool = True,
    bond_resolver: Optional[
        BondResolver
    ] = None,
    custom_predicate: Optional[
        AtomPredicate
    ] = None,
) -> bool:
    """
    Determine whether an atom can act as a hydrogen-bond acceptor.

    Parameters
    ----------
    atom : atom-like
        Atom to inspect.
    use_protein_templates : bool, optional
        Whether residue and atom-name templates should be used for proteins.
    bond_resolver : callable or None, optional
        Custom bonded-neighbor resolver.
    custom_predicate : callable or None, optional
        Additional acceptor predicate. A ``False`` result rejects the atom.

    Returns
    -------
    bool
        Acceptor status.
    """

    try:
        validate_atom(
            atom,
            require_coordinate=False,
        )

    except Exception:
        return False

    if is_hydrogen_atom(
        atom
    ):
        return False

    element = get_atom_element(
        atom
    )

    if element not in _GENERIC_ACCEPTOR_ELEMENTS:
        return False

    if element in COMMON_METAL_ELEMENTS:
        return False

    if custom_predicate is not None:
        try:
            custom_result = bool(
                custom_predicate(
                    atom
                )
            )

        except Exception as error:
            raise ValueError(
                "The custom acceptor predicate failed."
            ) from error

        if not custom_result:
            return False

    if (
        use_protein_templates
        and is_protein_residue(
            atom
        )
    ):
        template_result = (
            _protein_atom_is_known_acceptor(
                atom
            )
        )

        if template_result is not None:
            return template_result

    if element == NITROGEN_ELEMENT:
        return _generic_nitrogen_is_acceptor(
            atom,
            bond_resolver=bond_resolver,
        )

    if element == OXYGEN_ELEMENT:
        return _generic_oxygen_is_acceptor(
            atom,
            bond_resolver=bond_resolver,
        )

    if element == SULFUR_ELEMENT:
        return _generic_sulfur_is_acceptor(
            atom,
            bond_resolver=bond_resolver,
        )

    return False


# -----------------------------------------------------------------------------
# Combined role perception
# -----------------------------------------------------------------------------

def get_hbond_atom_roles(
    atom: AtomLike,
    *,
    require_explicit_hydrogen: bool = False,
    use_protein_templates: bool = True,
    bond_resolver: Optional[
        BondResolver
    ] = None,
    donor_predicate: Optional[
        AtomPredicate
    ] = None,
    acceptor_predicate: Optional[
        AtomPredicate
    ] = None,
) -> FrozenSet[
    HydrogenBondRole
]:
    """
    Return all hydrogen-bond roles supported by an atom.

    Parameters
    ----------
    atom : atom-like
        Atom to inspect.
    require_explicit_hydrogen : bool, optional
        Whether donor perception requires a bonded hydrogen.
    use_protein_templates : bool, optional
        Whether protein templates should be applied.
    bond_resolver : callable or None, optional
        Custom bonded-neighbor resolver.
    donor_predicate : callable or None, optional
        Additional donor predicate.
    acceptor_predicate : callable or None, optional
        Additional acceptor predicate.

    Returns
    -------
    frozenset of HydrogenBondRole
        Supported roles. Hydrogen atoms return ``{"hydrogen"}``; atoms with no
        supported role return ``{"none"}``.
    """

    if is_hydrogen_atom(
        atom
    ):
        return frozenset(
            {
                HBOND_ROLE_HYDROGEN,
            }
        )

    roles: Set[
        HydrogenBondRole
    ] = set()

    if is_hbond_donor(
        atom,
        require_explicit_hydrogen=(
            require_explicit_hydrogen
        ),
        use_protein_templates=(
            use_protein_templates
        ),
        bond_resolver=bond_resolver,
        custom_predicate=donor_predicate,
    ):
        roles.add(
            HBOND_ROLE_DONOR
        )

    if is_hbond_acceptor(
        atom,
        use_protein_templates=(
            use_protein_templates
        ),
        bond_resolver=bond_resolver,
        custom_predicate=acceptor_predicate,
    ):
        roles.add(
            HBOND_ROLE_ACCEPTOR
        )

    if not roles:
        roles.add(
            HBOND_ROLE_NONE
        )

    return frozenset(
        roles
    )


def get_primary_hbond_atom_role(
    atom: AtomLike,
    *,
    require_explicit_hydrogen: bool = False,
    use_protein_templates: bool = True,
    bond_resolver: Optional[
        BondResolver
    ] = None,
) -> HydrogenBondRole:
    """
    Return one primary hydrogen-bond role for an atom.

    Parameters
    ----------
    atom : atom-like
        Atom to inspect.
    require_explicit_hydrogen : bool, optional
        Whether donor perception requires an explicit hydrogen.
    use_protein_templates : bool, optional
        Whether protein templates should be applied.
    bond_resolver : callable or None, optional
        Custom neighbor resolver.

    Returns
    -------
    HydrogenBondRole
        Primary role.

    Notes
    -----
    Priority is hydrogen, donor, acceptor and none. Use
    :func:`get_hbond_atom_roles` when dual donor/acceptor behavior must be
    preserved.
    """

    roles = get_hbond_atom_roles(
        atom,
        require_explicit_hydrogen=(
            require_explicit_hydrogen
        ),
        use_protein_templates=(
            use_protein_templates
        ),
        bond_resolver=bond_resolver,
    )

    for role in (
        HBOND_ROLE_HYDROGEN,
        HBOND_ROLE_DONOR,
        HBOND_ROLE_ACCEPTOR,
        HBOND_ROLE_NONE,
    ):
        if role in roles:
            return role

    return HBOND_ROLE_UNKNOWN


# -----------------------------------------------------------------------------
# Collection selection
# -----------------------------------------------------------------------------

def select_hbond_donors(
    atoms: Iterable[
        AtomLike
    ],
    *,
    require_explicit_hydrogen: bool = False,
    use_protein_templates: bool = True,
    bond_resolver: Optional[
        BondResolver
    ] = None,
    custom_predicate: Optional[
        AtomPredicate
    ] = None,
) -> Tuple[
    AtomLike,
    ...,
]:
    """
    Select hydrogen-bond donors from an atom collection.

    Parameters
    ----------
    atoms : iterable of atom-like
        Atom collection.
    require_explicit_hydrogen : bool, optional
        Whether each donor must have a bonded explicit hydrogen.
    use_protein_templates : bool, optional
        Whether protein templates should be applied.
    bond_resolver : callable or None, optional
        Custom bonded-neighbor resolver.
    custom_predicate : callable or None, optional
        Additional donor predicate.

    Returns
    -------
    tuple of atom-like
        Donor atoms.
    """

    validated_atoms = validate_atom_collection(
        atoms,
        allow_empty=True,
        require_coordinate=False,
    )

    return tuple(
        atom
        for atom in validated_atoms
        if is_hbond_donor(
            atom,
            require_explicit_hydrogen=(
                require_explicit_hydrogen
            ),
            use_protein_templates=(
                use_protein_templates
            ),
            bond_resolver=bond_resolver,
            custom_predicate=custom_predicate,
        )
    )


def select_hbond_acceptors(
    atoms: Iterable[
        AtomLike
    ],
    *,
    use_protein_templates: bool = True,
    bond_resolver: Optional[
        BondResolver
    ] = None,
    custom_predicate: Optional[
        AtomPredicate
    ] = None,
) -> Tuple[
    AtomLike,
    ...,
]:
    """
    Select hydrogen-bond acceptors from an atom collection.

    Parameters
    ----------
    atoms : iterable of atom-like
        Atom collection.
    use_protein_templates : bool, optional
        Whether protein templates should be applied.
    bond_resolver : callable or None, optional
        Custom bonded-neighbor resolver.
    custom_predicate : callable or None, optional
        Additional acceptor predicate.

    Returns
    -------
    tuple of atom-like
        Acceptor atoms.
    """

    validated_atoms = validate_atom_collection(
        atoms,
        allow_empty=True,
        require_coordinate=False,
    )

    return tuple(
        atom
        for atom in validated_atoms
        if is_hbond_acceptor(
            atom,
            use_protein_templates=(
                use_protein_templates
            ),
            bond_resolver=bond_resolver,
            custom_predicate=custom_predicate,
        )
    )


def classify_hbond_atom_roles(
    atoms: Iterable[
        AtomLike
    ],
    *,
    require_explicit_hydrogen: bool = False,
    use_protein_templates: bool = True,
    bond_resolver: Optional[
        BondResolver
    ] = None,
) -> Dict[
    HydrogenBondRole,
    Tuple[
        AtomLike,
        ...,
    ],
]:
    """
    Classify atoms into hydrogen-bond role collections.

    Parameters
    ----------
    atoms : iterable of atom-like
        Atom collection.
    require_explicit_hydrogen : bool, optional
        Whether donor classification requires explicit hydrogen.
    use_protein_templates : bool, optional
        Whether protein templates should be applied.
    bond_resolver : callable or None, optional
        Custom bonded-neighbor resolver.

    Returns
    -------
    dict
        Mapping from role to atom tuples.

    Notes
    -----
    Dual-role atoms appear in both donor and acceptor collections.
    """

    validated_atoms = validate_atom_collection(
        atoms,
        allow_empty=True,
        require_coordinate=False,
    )

    role_lists: Dict[
        HydrogenBondRole,
        List[
            AtomLike
        ],
    ] = {
        HBOND_ROLE_DONOR: [],
        HBOND_ROLE_ACCEPTOR: [],
        HBOND_ROLE_HYDROGEN: [],
        HBOND_ROLE_NONE: [],
        HBOND_ROLE_UNKNOWN: [],
    }

    for atom in validated_atoms:
        roles = get_hbond_atom_roles(
            atom,
            require_explicit_hydrogen=(
                require_explicit_hydrogen
            ),
            use_protein_templates=(
                use_protein_templates
            ),
            bond_resolver=bond_resolver,
        )

        for role in roles:
            role_lists[
                role
            ].append(
                atom
            )

    return {
        role: tuple(
            role_atoms
        )
        for role, role_atoms
        in role_lists.items()
    }


def hbond_role_counts(
    atoms: Iterable[
        AtomLike
    ],
    *,
    require_explicit_hydrogen: bool = False,
    use_protein_templates: bool = True,
    bond_resolver: Optional[
        BondResolver
    ] = None,
) -> Dict[
    str,
    int,
]:
    """
    Count hydrogen-bond roles in an atom collection.

    Parameters
    ----------
    atoms : iterable of atom-like
        Atom collection.
    require_explicit_hydrogen : bool, optional
        Whether donors require explicit hydrogen.
    use_protein_templates : bool, optional
        Whether protein templates should be applied.
    bond_resolver : callable or None, optional
        Custom bonded-neighbor resolver.

    Returns
    -------
    dict
        Role counts. Dual-role atoms contribute to both applicable roles.
    """

    classified = classify_hbond_atom_roles(
        atoms,
        require_explicit_hydrogen=(
            require_explicit_hydrogen
        ),
        use_protein_templates=(
            use_protein_templates
        ),
        bond_resolver=bond_resolver,
    )

    return {
        role: len(
            role_atoms
        )
        for role, role_atoms
        in classified.items()
    }


# -----------------------------------------------------------------------------
# Public interface
# -----------------------------------------------------------------------------

_SECTION_4_PUBLIC_NAMES: Final[
    Tuple[
        str,
        ...,
    ]
] = (
    "get_hbond_residue_name",
    "get_atom_type",
    "get_atom_formal_charge",
    "atom_is_aromatic",
    "get_bonded_neighbors",
    "get_bonded_hydrogens",
    "get_bonded_heavy_atoms",
    "atom_has_explicit_hydrogen",
    "get_bond_order",
    "is_protein_residue",
    "is_protein_backbone_atom",
    "is_hbond_donor",
    "is_hbond_acceptor",
    "get_hbond_atom_roles",
    "get_primary_hbond_atom_role",
    "select_hbond_donors",
    "select_hbond_acceptors",
    "classify_hbond_atom_roles",
    "hbond_role_counts",
)

for public_name in _SECTION_4_PUBLIC_NAMES:
    if public_name not in __all__:
        __all__.append(
            public_name
        )


# =============================================================================
# End of Section 4
# =============================================================================


# =============================================================================
# Section 5 — Bonded-hydrogen identification
# =============================================================================


# -----------------------------------------------------------------------------
# Hydrogen-assignment constants
# -----------------------------------------------------------------------------

DEFAULT_MAXIMUM_DONOR_HYDROGEN_DISTANCE: Final[
    np.float64
] = np.float64(
    1.30
)

DEFAULT_MINIMUM_DONOR_HYDROGEN_DISTANCE: Final[
    np.float64
] = np.float64(
    0.50
)

DEFAULT_HYDROGEN_ASSIGNMENT_TOLERANCE: Final[
    np.float64
] = np.float64(
    0.15
)

DEFAULT_ALLOW_DISTANCE_BASED_HYDROGEN_ASSIGNMENT: Final[
    bool
] = True


# Approximate maximum covalent D-H distances used only when explicit bond
# topology is unavailable.
_MAXIMUM_DONOR_HYDROGEN_DISTANCE_BY_ELEMENT: Final[
    Mapping[
        str,
        np.float64,
    ]
] = MappingProxyType(
    {
        "N": np.float64(
            1.25
        ),
        "O": np.float64(
            1.20
        ),
        "S": np.float64(
            1.55
        ),
    }
)


# Approximate minimum distances avoid assigning coincident or corrupt
# coordinates as covalently bonded atoms.
_MINIMUM_DONOR_HYDROGEN_DISTANCE_BY_ELEMENT: Final[
    Mapping[
        str,
        np.float64,
    ]
] = MappingProxyType(
    {
        "N": np.float64(
            0.55
        ),
        "O": np.float64(
            0.50
        ),
        "S": np.float64(
            0.70
        ),
    }
)


# Typical upper limits for the number of hydrogens bonded to a donor atom.
_MAXIMUM_BOUND_HYDROGENS_BY_ELEMENT: Final[
    Mapping[
        str,
        int,
    ]
] = MappingProxyType(
    {
        "N": 3,
        "O": 2,
        "S": 1,
    }
)


# -----------------------------------------------------------------------------
# Hydrogen-assignment result dataclass
# -----------------------------------------------------------------------------

@dataclass(
    frozen=True,
    slots=True,
)
class DonorHydrogenAssignment:
    """
    Association between a donor atom and one explicit hydrogen.

    Parameters
    ----------
    donor : atom-like
        Hydrogen-bond donor atom.
    hydrogen : atom-like
        Explicit hydrogen associated with the donor.
    distance : Number
        D-H distance in angstroms.
    donor_index : int or None, optional
        Donor index in the original atom collection.
    hydrogen_index : int or None, optional
        Hydrogen index in the original atom collection.
    assignment_method : str, optional
        Assignment source, normally ``"topology"`` or ``"distance"``.
    is_ambiguous : bool, optional
        Whether the hydrogen could reasonably belong to multiple donors.
    alternative_donors : sequence of atom-like, optional
        Other donor candidates considered for the hydrogen.
    metadata : mapping, optional
        Additional assignment information.

    Notes
    -----
    The object stores references to the original atom objects and normalizes
    numeric fields to ``numpy.float64``.
    """

    donor: AtomLike
    hydrogen: AtomLike
    distance: np.float64

    donor_index: Optional[
        int
    ] = None

    hydrogen_index: Optional[
        int
    ] = None

    assignment_method: str = "topology"

    is_ambiguous: bool = False

    alternative_donors: Sequence[
        AtomLike
    ] = field(
        default_factory=tuple
    )

    metadata: Mapping[
        str,
        Any,
    ] = field(
        default_factory=lambda: _EMPTY_METADATA,
        repr=False,
        compare=False,
    )

    def __post_init__(
        self,
    ) -> None:
        """Validate and normalize the assignment."""

        if self.donor is None:
            raise ValueError(
                "DonorHydrogenAssignment.donor cannot be None."
            )

        if self.hydrogen is None:
            raise ValueError(
                "DonorHydrogenAssignment.hydrogen cannot be None."
            )

        if self.donor is self.hydrogen:
            raise ValueError(
                "Donor and hydrogen must be different atoms."
            )

        if not is_hydrogen_atom(
            self.hydrogen
        ):
            raise ValueError(
                "The assigned hydrogen atom is not recognized as hydrogen."
            )

        if is_hydrogen_atom(
            self.donor
        ):
            raise ValueError(
                "The donor atom cannot itself be hydrogen."
            )

        normalized_distance = _optional_float64(
            self.distance,
            name="donor-hydrogen distance",
            minimum=0.0,
        )

        if normalized_distance is None:
            raise ValueError(
                "Donor-hydrogen distance cannot be None."
            )

        normalized_method = str(
            self.assignment_method
        ).strip().lower()

        if normalized_method not in {
            "topology",
            "distance",
            "custom",
        }:
            raise ValueError(
                "assignment_method must be 'topology', 'distance' "
                "or 'custom'."
            )

        normalized_alternatives = tuple(
            atom
            for atom in self.alternative_donors
            if atom is not None
            and atom is not self.donor
            and atom is not self.hydrogen
        )

        for index, alternative in enumerate(
            normalized_alternatives
        ):
            if not is_atom_like(
                alternative
            ):
                raise TypeError(
                    "All alternative donors must be atom-like objects. "
                    f"Invalid entry at index {index}."
                )

        object.__setattr__(
            self,
            "distance",
            normalized_distance,
        )

        object.__setattr__(
            self,
            "donor_index",
            _optional_nonnegative_integer(
                self.donor_index,
                name="donor index",
            ),
        )

        object.__setattr__(
            self,
            "hydrogen_index",
            _optional_nonnegative_integer(
                self.hydrogen_index,
                name="hydrogen index",
            ),
        )

        object.__setattr__(
            self,
            "assignment_method",
            normalized_method,
        )

        object.__setattr__(
            self,
            "is_ambiguous",
            bool(
                self.is_ambiguous
                or normalized_alternatives
            ),
        )

        object.__setattr__(
            self,
            "alternative_donors",
            normalized_alternatives,
        )

        object.__setattr__(
            self,
            "metadata",
            _freeze_metadata(
                self.metadata
            ),
        )

    @property
    def atom_pair(
        self,
    ) -> Tuple[
        AtomLike,
        AtomLike,
    ]:
        """
        Return the donor-hydrogen pair.

        Returns
        -------
        tuple
            ``(donor, hydrogen)``.
        """

        return (
            self.donor,
            self.hydrogen,
        )

    @property
    def index_pair(
        self,
    ) -> Tuple[
        Optional[
            int
        ],
        Optional[
            int
        ],
    ]:
        """
        Return donor and hydrogen indices.

        Returns
        -------
        tuple
            ``(donor_index, hydrogen_index)``.
        """

        return (
            self.donor_index,
            self.hydrogen_index,
        )

    @property
    def is_topology_based(
        self,
    ) -> bool:
        """
        Whether the assignment came from explicit bond topology.

        Returns
        -------
        bool
            Topology-assignment status.
        """

        return (
            self.assignment_method
            == "topology"
        )

    @property
    def is_distance_based(
        self,
    ) -> bool:
        """
        Whether the assignment was inferred by distance.

        Returns
        -------
        bool
            Distance-assignment status.
        """

        return (
            self.assignment_method
            == "distance"
        )

    def to_dict(
        self,
        *,
        include_atoms: bool = False,
    ) -> Dict[
        str,
        Any,
    ]:
        """
        Serialize the assignment.

        Parameters
        ----------
        include_atoms : bool, optional
            Whether raw atom objects should be included.

        Returns
        -------
        dict
            Serializable assignment representation.
        """

        result: Dict[
            str,
            Any,
        ] = {
            "donor_identifier": (
                _safe_atom_identifier(
                    self.donor
                )
            ),
            "hydrogen_identifier": (
                _safe_atom_identifier(
                    self.hydrogen
                )
            ),
            "distance": float(
                self.distance
            ),
            "donor_index": (
                self.donor_index
            ),
            "hydrogen_index": (
                self.hydrogen_index
            ),
            "assignment_method": (
                self.assignment_method
            ),
            "is_ambiguous": (
                self.is_ambiguous
            ),
            "alternative_donor_identifiers": [
                _safe_atom_identifier(
                    atom
                )
                for atom in self.alternative_donors
            ],
            "metadata": dict(
                self.metadata
            ),
        }

        if include_atoms:
            result.update(
                {
                    "donor": self.donor,
                    "hydrogen": self.hydrogen,
                    "alternative_donors": (
                        self.alternative_donors
                    ),
                }
            )

        return result


# -----------------------------------------------------------------------------
# Distance helpers
# -----------------------------------------------------------------------------

def _calculate_atom_distance(
    atom_1: AtomLike,
    atom_2: AtomLike,
) -> np.float64:
    """
    Calculate the Euclidean distance between two atoms.

    Parameters
    ----------
    atom_1 : atom-like
        First atom.
    atom_2 : atom-like
        Second atom.

    Returns
    -------
    numpy.float64
        Distance in angstroms.

    Raises
    ------
    ValueError
        If valid three-dimensional coordinates cannot be obtained.
    """

    coordinate_1 = np.asarray(
        get_atom_coordinate(
            atom_1
        ),
        dtype=np.float64,
    )

    coordinate_2 = np.asarray(
        get_atom_coordinate(
            atom_2
        ),
        dtype=np.float64,
    )

    if (
        coordinate_1.shape
        != (
            3,
        )
        or coordinate_2.shape
        != (
            3,
        )
    ):
        raise ValueError(
            "Atom coordinates must contain exactly three values."
        )

    if (
        not np.all(
            np.isfinite(
                coordinate_1
            )
        )
        or not np.all(
            np.isfinite(
                coordinate_2
            )
        )
    ):
        raise ValueError(
            "Atom coordinates must be finite."
        )

    return np.float64(
        np.linalg.norm(
            coordinate_1
            - coordinate_2
        )
    )


def get_donor_hydrogen_distance_bounds(
    donor: AtomLike,
    *,
    tolerance: Number = DEFAULT_HYDROGEN_ASSIGNMENT_TOLERANCE,
) -> Tuple[
    np.float64,
    np.float64,
]:
    """
    Return valid distance bounds for assigning hydrogen to a donor.

    Parameters
    ----------
    donor : atom-like
        Candidate donor atom.
    tolerance : Number, optional
        Additional tolerance added to the maximum distance.

    Returns
    -------
    tuple of numpy.float64
        Minimum and maximum D-H distances.
    """

    validate_atom(
        donor,
        require_coordinate=False,
    )

    element = get_atom_element(
        donor
    )

    normalized_tolerance = _optional_float64(
        tolerance,
        name="hydrogen-assignment tolerance",
        minimum=0.0,
    )

    if normalized_tolerance is None:
        normalized_tolerance = np.float64(
            0.0
        )

    minimum_distance = (
        _MINIMUM_DONOR_HYDROGEN_DISTANCE_BY_ELEMENT.get(
            element,
            DEFAULT_MINIMUM_DONOR_HYDROGEN_DISTANCE,
        )
    )

    maximum_distance = (
        _MAXIMUM_DONOR_HYDROGEN_DISTANCE_BY_ELEMENT.get(
            element,
            DEFAULT_MAXIMUM_DONOR_HYDROGEN_DISTANCE,
        )
    )

    return (
        np.float64(
            minimum_distance
        ),
        np.float64(
            maximum_distance
            + normalized_tolerance
        ),
    )


def is_valid_donor_hydrogen_distance(
    donor: AtomLike,
    hydrogen: AtomLike,
    *,
    tolerance: Number = DEFAULT_HYDROGEN_ASSIGNMENT_TOLERANCE,
) -> bool:
    """
    Test whether a donor-hydrogen distance is chemically plausible.

    Parameters
    ----------
    donor : atom-like
        Candidate donor.
    hydrogen : atom-like
        Candidate bonded hydrogen.
    tolerance : Number, optional
        Additional maximum-distance tolerance.

    Returns
    -------
    bool
        ``True`` when the distance lies within accepted bounds.
    """

    if donor is None or hydrogen is None:
        return False

    if donor is hydrogen:
        return False

    if is_hydrogen_atom(
        donor
    ):
        return False

    if not is_hydrogen_atom(
        hydrogen
    ):
        return False

    try:
        donor_hydrogen_distance = (
            _calculate_atom_distance(
                donor,
                hydrogen,
            )
        )

        minimum_distance, maximum_distance = (
            get_donor_hydrogen_distance_bounds(
                donor,
                tolerance=tolerance,
            )
        )

    except Exception:
        return False

    return bool(
        minimum_distance
        <= donor_hydrogen_distance
        <= maximum_distance
    )


# -----------------------------------------------------------------------------
# Explicit-topology hydrogen assignment
# -----------------------------------------------------------------------------

def get_topology_bonded_hydrogens(
    donor: AtomLike,
    *,
    bond_resolver: Optional[
        BondResolver
    ] = None,
    validate_distance: bool = True,
    tolerance: Number = DEFAULT_HYDROGEN_ASSIGNMENT_TOLERANCE,
) -> Tuple[
    AtomLike,
    ...,
]:
    """
    Return hydrogens explicitly bonded to a donor by topology.

    Parameters
    ----------
    donor : atom-like
        Candidate donor atom.
    bond_resolver : callable or None, optional
        Custom topology resolver.
    validate_distance : bool, optional
        Whether implausible D-H distances should be rejected.
    tolerance : Number, optional
        Distance tolerance.

    Returns
    -------
    tuple of atom-like
        Explicit bonded hydrogens.
    """

    if not is_hbond_donor(
        donor,
        require_explicit_hydrogen=False,
        bond_resolver=bond_resolver,
    ):
        return ()

    hydrogens = get_bonded_hydrogens(
        donor,
        bond_resolver=bond_resolver,
    )

    if not validate_distance:
        return hydrogens

    return tuple(
        hydrogen
        for hydrogen in hydrogens
        if is_valid_donor_hydrogen_distance(
            donor,
            hydrogen,
            tolerance=tolerance,
        )
    )


def build_topology_hydrogen_assignments(
    donors: Iterable[
        AtomLike
    ],
    *,
    atom_indices: Optional[
        Mapping[
            int,
            int,
        ]
    ] = None,
    bond_resolver: Optional[
        BondResolver
    ] = None,
    validate_distance: bool = True,
    tolerance: Number = DEFAULT_HYDROGEN_ASSIGNMENT_TOLERANCE,
) -> Tuple[
    DonorHydrogenAssignment,
    ...,
]:
    """
    Build donor-hydrogen assignments from explicit topology.

    Parameters
    ----------
    donors : iterable of atom-like
        Candidate donors.
    atom_indices : mapping or None, optional
        Mapping from ``id(atom)`` to source atom index.
    bond_resolver : callable or None, optional
        Custom topology resolver.
    validate_distance : bool, optional
        Whether D-H distance should be validated.
    tolerance : Number, optional
        Distance tolerance.

    Returns
    -------
    tuple of DonorHydrogenAssignment
        Topology-based assignments.
    """

    validated_donors = validate_atom_collection(
        donors,
        allow_empty=True,
        require_coordinate=False,
    )

    assignments: List[
        DonorHydrogenAssignment
    ] = []

    seen_pairs: Set[
        Tuple[
            int,
            int,
        ]
    ] = set()

    for donor in validated_donors:
        hydrogens = get_topology_bonded_hydrogens(
            donor,
            bond_resolver=bond_resolver,
            validate_distance=validate_distance,
            tolerance=tolerance,
        )

        for hydrogen in hydrogens:
            pair_key = (
                id(
                    donor
                ),
                id(
                    hydrogen
                ),
            )

            if pair_key in seen_pairs:
                continue

            seen_pairs.add(
                pair_key
            )

            try:
                donor_hydrogen_distance = (
                    _calculate_atom_distance(
                        donor,
                        hydrogen,
                    )
                )

            except Exception:
                if validate_distance:
                    continue

                donor_hydrogen_distance = np.float64(
                    np.nan
                )

            donor_index = None
            hydrogen_index = None

            if atom_indices is not None:
                donor_index = atom_indices.get(
                    id(
                        donor
                    )
                )

                hydrogen_index = atom_indices.get(
                    id(
                        hydrogen
                    )
                )

            assignments.append(
                DonorHydrogenAssignment(
                    donor=donor,
                    hydrogen=hydrogen,
                    distance=(
                        donor_hydrogen_distance
                    ),
                    donor_index=donor_index,
                    hydrogen_index=(
                        hydrogen_index
                    ),
                    assignment_method="topology",
                )
            )

    return tuple(
        assignments
    )


# -----------------------------------------------------------------------------
# Distance-based hydrogen assignment
# -----------------------------------------------------------------------------

def find_candidate_donors_for_hydrogen(
    hydrogen: AtomLike,
    donors: Iterable[
        AtomLike
    ],
    *,
    tolerance: Number = DEFAULT_HYDROGEN_ASSIGNMENT_TOLERANCE,
    require_donor_perception: bool = True,
    bond_resolver: Optional[
        BondResolver
    ] = None,
) -> Tuple[
    Tuple[
        AtomLike,
        np.float64,
    ],
    ...,
]:
    """
    Find plausible donor atoms for one hydrogen by distance.

    Parameters
    ----------
    hydrogen : atom-like
        Explicit hydrogen atom.
    donors : iterable of atom-like
        Candidate donors.
    tolerance : Number, optional
        Maximum-distance tolerance.
    require_donor_perception : bool, optional
        Whether candidate atoms must pass donor perception.
    bond_resolver : callable or None, optional
        Custom topology resolver.

    Returns
    -------
    tuple
        ``(donor, distance)`` pairs sorted by increasing distance.
    """

    if not is_hydrogen_atom(
        hydrogen
    ):
        return ()

    validated_donors = validate_atom_collection(
        donors,
        allow_empty=True,
        require_coordinate=True,
    )

    candidates: List[
        Tuple[
            AtomLike,
            np.float64,
        ]
    ] = []

    for donor in validated_donors:
        if donor is hydrogen:
            continue

        if require_donor_perception and not is_hbond_donor(
            donor,
            require_explicit_hydrogen=False,
            bond_resolver=bond_resolver,
        ):
            continue

        if not is_valid_donor_hydrogen_distance(
            donor,
            hydrogen,
            tolerance=tolerance,
        ):
            continue

        try:
            donor_hydrogen_distance = (
                _calculate_atom_distance(
                    donor,
                    hydrogen,
                )
            )

        except Exception:
            continue

        candidates.append(
            (
                donor,
                donor_hydrogen_distance,
            )
        )

    candidates.sort(
        key=lambda item: float(
            item[
                1
            ]
        )
    )

    return tuple(
        candidates
    )


def assign_hydrogen_by_distance(
    hydrogen: AtomLike,
    donors: Iterable[
        AtomLike
    ],
    *,
    tolerance: Number = DEFAULT_HYDROGEN_ASSIGNMENT_TOLERANCE,
    ambiguity_tolerance: Number = 0.10,
    require_donor_perception: bool = True,
    bond_resolver: Optional[
        BondResolver
    ] = None,
    donor_indices: Optional[
        Mapping[
            int,
            int,
        ]
    ] = None,
    hydrogen_index: Optional[
        int
    ] = None,
) -> Optional[
    DonorHydrogenAssignment
]:
    """
    Assign one hydrogen to its most plausible donor by distance.

    Parameters
    ----------
    hydrogen : atom-like
        Hydrogen to assign.
    donors : iterable of atom-like
        Candidate donor atoms.
    tolerance : Number, optional
        Maximum-distance tolerance.
    ambiguity_tolerance : Number, optional
        Distance difference below which alternative donors are considered
        ambiguous.
    require_donor_perception : bool, optional
        Whether donors must pass chemical perception.
    bond_resolver : callable or None, optional
        Custom topology resolver.
    donor_indices : mapping or None, optional
        Mapping from ``id(donor)`` to source index.
    hydrogen_index : int or None, optional
        Source hydrogen index.

    Returns
    -------
    DonorHydrogenAssignment or None
        Best assignment, or ``None`` when no valid donor is found.
    """

    candidates = find_candidate_donors_for_hydrogen(
        hydrogen,
        donors,
        tolerance=tolerance,
        require_donor_perception=(
            require_donor_perception
        ),
        bond_resolver=bond_resolver,
    )

    if not candidates:
        return None

    normalized_ambiguity_tolerance = (
        _optional_float64(
            ambiguity_tolerance,
            name="hydrogen-assignment ambiguity tolerance",
            minimum=0.0,
        )
    )

    if normalized_ambiguity_tolerance is None:
        normalized_ambiguity_tolerance = np.float64(
            0.0
        )

    selected_donor, selected_distance = (
        candidates[
            0
        ]
    )

    alternative_donors = tuple(
        donor
        for donor, candidate_distance
        in candidates[
            1:
        ]
        if (
            candidate_distance
            - selected_distance
        )
        <= normalized_ambiguity_tolerance
    )

    donor_index = None

    if donor_indices is not None:
        donor_index = donor_indices.get(
            id(
                selected_donor
            )
        )

    return DonorHydrogenAssignment(
        donor=selected_donor,
        hydrogen=hydrogen,
        distance=selected_distance,
        donor_index=donor_index,
        hydrogen_index=hydrogen_index,
        assignment_method="distance",
        is_ambiguous=bool(
            alternative_donors
        ),
        alternative_donors=(
            alternative_donors
        ),
    )


# -----------------------------------------------------------------------------
# Complete explicit-hydrogen mapping
# -----------------------------------------------------------------------------

def identify_donor_hydrogen_assignments(
    atoms: Iterable[
        AtomLike
    ],
    *,
    donors: Optional[
        Iterable[
            AtomLike
        ]
    ] = None,
    bond_resolver: Optional[
        BondResolver
    ] = None,
    allow_distance_assignment: bool = (
        DEFAULT_ALLOW_DISTANCE_BASED_HYDROGEN_ASSIGNMENT
    ),
    validate_topology_distance: bool = True,
    distance_tolerance: Number = (
        DEFAULT_HYDROGEN_ASSIGNMENT_TOLERANCE
    ),
    ambiguity_tolerance: Number = 0.10,
    include_ambiguous: bool = True,
) -> Tuple[
    DonorHydrogenAssignment,
    ...,
]:
    """
    Identify all explicit donor-hydrogen assignments in a structure.

    Parameters
    ----------
    atoms : iterable of atom-like
        Complete atom collection.
    donors : iterable of atom-like or None, optional
        Preselected donors. When omitted, donors are identified automatically.
    bond_resolver : callable or None, optional
        Custom topology resolver.
    allow_distance_assignment : bool, optional
        Whether unassigned hydrogens may be assigned by distance.
    validate_topology_distance : bool, optional
        Whether topology-derived assignments must pass D-H distance checks.
    distance_tolerance : Number, optional
        Maximum-distance tolerance.
    ambiguity_tolerance : Number, optional
        Difference used to mark distance assignments as ambiguous.
    include_ambiguous : bool, optional
        Whether ambiguous distance assignments should be retained.

    Returns
    -------
    tuple of DonorHydrogenAssignment
        Unique donor-hydrogen assignments.

    Notes
    -----
    Explicit topology always takes priority over distance inference.
    """

    validated_atoms = validate_atom_collection(
        atoms,
        allow_empty=True,
        require_coordinate=False,
    )

    if not validated_atoms:
        return ()

    atom_indices: Dict[
        int,
        int,
    ] = {
        id(
            atom
        ): index
        for index, atom
        in enumerate(
            validated_atoms
        )
    }

    if donors is None:
        selected_donors = select_hbond_donors(
            validated_atoms,
            require_explicit_hydrogen=False,
            bond_resolver=bond_resolver,
        )

    else:
        selected_donors = (
            validate_atom_collection(
                donors,
                allow_empty=True,
                require_coordinate=False,
            )
        )

    topology_assignments = (
        build_topology_hydrogen_assignments(
            selected_donors,
            atom_indices=atom_indices,
            bond_resolver=bond_resolver,
            validate_distance=(
                validate_topology_distance
            ),
            tolerance=distance_tolerance,
        )
    )

    assignments: List[
        DonorHydrogenAssignment
    ] = list(
        topology_assignments
    )

    assigned_hydrogen_ids: Set[
        int
    ] = {
        id(
            assignment.hydrogen
        )
        for assignment in topology_assignments
    }

    assigned_pair_ids: Set[
        Tuple[
            int,
            int,
        ]
    ] = {
        (
            id(
                assignment.donor
            ),
            id(
                assignment.hydrogen
            ),
        )
        for assignment in topology_assignments
    }

    if not allow_distance_assignment:
        return tuple(
            assignments
        )

    explicit_hydrogens = tuple(
        atom
        for atom in validated_atoms
        if is_hydrogen_atom(
            atom
        )
    )

    for hydrogen in explicit_hydrogens:
        if id(
            hydrogen
        ) in assigned_hydrogen_ids:
            continue

        try:
            assignment = assign_hydrogen_by_distance(
                hydrogen,
                selected_donors,
                tolerance=distance_tolerance,
                ambiguity_tolerance=(
                    ambiguity_tolerance
                ),
                require_donor_perception=True,
                bond_resolver=bond_resolver,
                donor_indices=atom_indices,
                hydrogen_index=atom_indices.get(
                    id(
                        hydrogen
                    )
                ),
            )

        except Exception:
            assignment = None

        if assignment is None:
            continue

        if (
            assignment.is_ambiguous
            and not include_ambiguous
        ):
            continue

        pair_key = (
            id(
                assignment.donor
            ),
            id(
                assignment.hydrogen
            ),
        )

        if pair_key in assigned_pair_ids:
            continue

        assigned_pair_ids.add(
            pair_key
        )

        assigned_hydrogen_ids.add(
            id(
                assignment.hydrogen
            )
        )

        assignments.append(
            assignment
        )

    assignments.sort(
        key=lambda assignment: (
            assignment.donor_index
            if assignment.donor_index is not None
            else sys.maxsize,
            assignment.hydrogen_index
            if assignment.hydrogen_index is not None
            else sys.maxsize,
            float(
                assignment.distance
            ),
        )
    )

    return tuple(
        assignments
    )


# -----------------------------------------------------------------------------
# Assignment grouping
# -----------------------------------------------------------------------------

def group_hydrogen_assignments_by_donor(
    assignments: Iterable[
        DonorHydrogenAssignment
    ],
) -> Dict[
    AtomLike,
    Tuple[
        DonorHydrogenAssignment,
        ...,
    ],
]:
    """
    Group hydrogen assignments by donor atom.

    Parameters
    ----------
    assignments : iterable of DonorHydrogenAssignment
        Assignments to group.

    Returns
    -------
    dict
        Donor-to-assignment mapping.
    """

    grouped: Dict[
        AtomLike,
        List[
            DonorHydrogenAssignment
        ],
    ] = {}

    for index, assignment in enumerate(
        assignments
    ):
        if not isinstance(
            assignment,
            DonorHydrogenAssignment,
        ):
            raise TypeError(
                "All assignments must be DonorHydrogenAssignment "
                f"instances. Invalid entry at index {index}."
            )

        grouped.setdefault(
            assignment.donor,
            [],
        ).append(
            assignment
        )

    return {
        donor: tuple(
            donor_assignments
        )
        for donor, donor_assignments
        in grouped.items()
    }


def group_hydrogens_by_donor(
    assignments: Iterable[
        DonorHydrogenAssignment
    ],
) -> Dict[
    AtomLike,
    Tuple[
        AtomLike,
        ...,
    ],
]:
    """
    Group explicit hydrogen atoms by donor.

    Parameters
    ----------
    assignments : iterable of DonorHydrogenAssignment
        Assignments to group.

    Returns
    -------
    dict
        Donor-to-hydrogen mapping.
    """

    grouped_assignments = (
        group_hydrogen_assignments_by_donor(
            assignments
        )
    )

    return {
        donor: tuple(
            assignment.hydrogen
            for assignment
            in donor_assignments
        )
        for donor, donor_assignments
        in grouped_assignments.items()
    }


def get_assignments_for_donor(
    donor: AtomLike,
    assignments: Iterable[
        DonorHydrogenAssignment
    ],
) -> Tuple[
    DonorHydrogenAssignment,
    ...,
]:
    """
    Return all explicit-hydrogen assignments for one donor.

    Parameters
    ----------
    donor : atom-like
        Donor atom.
    assignments : iterable of DonorHydrogenAssignment
        Available assignments.

    Returns
    -------
    tuple of DonorHydrogenAssignment
        Matching assignments.
    """

    return tuple(
        assignment
        for assignment in assignments
        if (
            isinstance(
                assignment,
                DonorHydrogenAssignment,
            )
            and assignment.donor
            is donor
        )
    )


def get_assigned_hydrogens_for_donor(
    donor: AtomLike,
    assignments: Iterable[
        DonorHydrogenAssignment
    ],
) -> Tuple[
    AtomLike,
    ...,
]:
    """
    Return assigned explicit hydrogens for one donor.

    Parameters
    ----------
    donor : atom-like
        Donor atom.
    assignments : iterable of DonorHydrogenAssignment
        Available assignments.

    Returns
    -------
    tuple of atom-like
        Assigned hydrogen atoms.
    """

    return tuple(
        assignment.hydrogen
        for assignment
        in get_assignments_for_donor(
            donor,
            assignments,
        )
    )


# -----------------------------------------------------------------------------
# Orphan and ambiguous hydrogen detection
# -----------------------------------------------------------------------------

def find_unassigned_hydrogens(
    atoms: Iterable[
        AtomLike
    ],
    assignments: Iterable[
        DonorHydrogenAssignment
    ],
) -> Tuple[
    AtomLike,
    ...,
]:
    """
    Return explicit hydrogens without donor assignments.

    Parameters
    ----------
    atoms : iterable of atom-like
        Complete atom collection.
    assignments : iterable of DonorHydrogenAssignment
        Existing assignments.

    Returns
    -------
    tuple of atom-like
        Unassigned hydrogen atoms.
    """

    validated_atoms = validate_atom_collection(
        atoms,
        allow_empty=True,
        require_coordinate=False,
    )

    assigned_hydrogen_ids = {
        id(
            assignment.hydrogen
        )
        for assignment in assignments
        if isinstance(
            assignment,
            DonorHydrogenAssignment,
        )
    }

    return tuple(
        atom
        for atom in validated_atoms
        if (
            is_hydrogen_atom(
                atom
            )
            and id(
                atom
            )
            not in assigned_hydrogen_ids
        )
    )


def find_ambiguous_hydrogen_assignments(
    assignments: Iterable[
        DonorHydrogenAssignment
    ],
) -> Tuple[
    DonorHydrogenAssignment,
    ...,
]:
    """
    Return assignments marked as ambiguous.

    Parameters
    ----------
    assignments : iterable of DonorHydrogenAssignment
        Assignments to inspect.

    Returns
    -------
    tuple of DonorHydrogenAssignment
        Ambiguous assignments.
    """

    return tuple(
        assignment
        for assignment in assignments
        if (
            isinstance(
                assignment,
                DonorHydrogenAssignment,
            )
            and assignment.is_ambiguous
        )
    )


def donor_has_assigned_hydrogen(
    donor: AtomLike,
    assignments: Iterable[
        DonorHydrogenAssignment
    ],
) -> bool:
    """
    Determine whether a donor has at least one assigned hydrogen.

    Parameters
    ----------
    donor : atom-like
        Donor atom.
    assignments : iterable of DonorHydrogenAssignment
        Available assignments.

    Returns
    -------
    bool
        Assigned-hydrogen status.
    """

    return any(
        isinstance(
            assignment,
            DonorHydrogenAssignment,
        )
        and assignment.donor
        is donor
        for assignment in assignments
    )


# -----------------------------------------------------------------------------
# Assignment statistics
# -----------------------------------------------------------------------------

def hydrogen_assignment_statistics(
    atoms: Iterable[
        AtomLike
    ],
    assignments: Iterable[
        DonorHydrogenAssignment
    ],
) -> Dict[
    str,
    Any,
]:
    """
    Calculate summary statistics for donor-hydrogen assignments.

    Parameters
    ----------
    atoms : iterable of atom-like
        Complete atom collection.
    assignments : iterable of DonorHydrogenAssignment
        Hydrogen assignments.

    Returns
    -------
    dict
        Assignment statistics.
    """

    validated_atoms = validate_atom_collection(
        atoms,
        allow_empty=True,
        require_coordinate=False,
    )

    normalized_assignments = tuple(
        assignments
    )

    for index, assignment in enumerate(
        normalized_assignments
    ):
        if not isinstance(
            assignment,
            DonorHydrogenAssignment,
        ):
            raise TypeError(
                "All assignments must be DonorHydrogenAssignment "
                f"instances. Invalid entry at index {index}."
            )

    explicit_hydrogen_count = sum(
        is_hydrogen_atom(
            atom
        )
        for atom in validated_atoms
    )

    assigned_hydrogen_ids = {
        id(
            assignment.hydrogen
        )
        for assignment in normalized_assignments
    }

    donor_ids = {
        id(
            assignment.donor
        )
        for assignment in normalized_assignments
    }

    distances = np.asarray(
        [
            assignment.distance
            for assignment
            in normalized_assignments
            if np.isfinite(
                assignment.distance
            )
        ],
        dtype=np.float64,
    )

    return {
        "atom_count": len(
            validated_atoms
        ),
        "explicit_hydrogen_count": (
            explicit_hydrogen_count
        ),
        "assignment_count": len(
            normalized_assignments
        ),
        "assigned_hydrogen_count": len(
            assigned_hydrogen_ids
        ),
        "unassigned_hydrogen_count": max(
            0,
            explicit_hydrogen_count
            - len(
                assigned_hydrogen_ids
            ),
        ),
        "donor_with_hydrogen_count": len(
            donor_ids
        ),
        "topology_assignment_count": sum(
            assignment.is_topology_based
            for assignment
            in normalized_assignments
        ),
        "distance_assignment_count": sum(
            assignment.is_distance_based
            for assignment
            in normalized_assignments
        ),
        "ambiguous_assignment_count": sum(
            assignment.is_ambiguous
            for assignment
            in normalized_assignments
        ),
        "minimum_donor_hydrogen_distance": (
            None
            if distances.size == 0
            else float(
                np.min(
                    distances
                )
            )
        ),
        "maximum_donor_hydrogen_distance": (
            None
            if distances.size == 0
            else float(
                np.max(
                    distances
                )
            )
        ),
        "mean_donor_hydrogen_distance": (
            None
            if distances.size == 0
            else float(
                np.mean(
                    distances
                )
            )
        ),
        "median_donor_hydrogen_distance": (
            None
            if distances.size == 0
            else float(
                np.median(
                    distances
                )
            )
        ),
    }


# -----------------------------------------------------------------------------
# Public interface
# -----------------------------------------------------------------------------

_SECTION_5_PUBLIC_NAMES: Final[
    Tuple[
        str,
        ...,
    ]
] = (
    "DEFAULT_MAXIMUM_DONOR_HYDROGEN_DISTANCE",
    "DEFAULT_MINIMUM_DONOR_HYDROGEN_DISTANCE",
    "DEFAULT_HYDROGEN_ASSIGNMENT_TOLERANCE",
    "DEFAULT_ALLOW_DISTANCE_BASED_HYDROGEN_ASSIGNMENT",
    "DonorHydrogenAssignment",
    "get_donor_hydrogen_distance_bounds",
    "is_valid_donor_hydrogen_distance",
    "get_topology_bonded_hydrogens",
    "build_topology_hydrogen_assignments",
    "find_candidate_donors_for_hydrogen",
    "assign_hydrogen_by_distance",
    "identify_donor_hydrogen_assignments",
    "group_hydrogen_assignments_by_donor",
    "group_hydrogens_by_donor",
    "get_assignments_for_donor",
    "get_assigned_hydrogens_for_donor",
    "find_unassigned_hydrogens",
    "find_ambiguous_hydrogen_assignments",
    "donor_has_assigned_hydrogen",
    "hydrogen_assignment_statistics",
)

for public_name in _SECTION_5_PUBLIC_NAMES:
    if public_name not in __all__:
        __all__.append(
            public_name
        )


# =============================================================================
# End of Section 5
# =============================================================================


# =============================================================================
# Section 6 — D-H...A and D...A geometry
# =============================================================================


# -----------------------------------------------------------------------------
# Geometric-evaluation constants
# -----------------------------------------------------------------------------

DEFAULT_REQUIRE_DONOR_ACCEPTOR_DISTANCE: Final[
    bool
] = True

DEFAULT_REQUIRE_HYDROGEN_ACCEPTOR_DISTANCE: Final[
    bool
] = True

DEFAULT_REQUIRE_DHA_ANGLE: Final[
    bool
] = True

DEFAULT_REQUIRE_INFERRED_ANGLE: Final[
    bool
] = True

DEFAULT_INFERRED_DONOR_VECTOR_METHOD: Final[
    str
] = "opposite_neighbors"

DEFAULT_INFERRED_ACCEPTOR_VECTOR_METHOD: Final[
    str
] = "opposite_neighbors"

VALID_INFERRED_VECTOR_METHODS: Final[
    FrozenSet[
        str
    ]
] = frozenset(
    {
        "opposite_neighbors",
        "nearest_neighbor",
        "centroid",
        "none",
    }
)


# -----------------------------------------------------------------------------
# Geometry evaluation result
# -----------------------------------------------------------------------------

@dataclass(
    frozen=True,
    slots=True,
)
class HydrogenBondGeometryEvaluation:
    """
    Result of applying hydrogen-bond geometric criteria.

    Parameters
    ----------
    geometry : HydrogenBondGeometry
        Calculated hydrogen-bond geometry.
    mode : HydrogenBondMode
        Explicit or inferred geometry mode.
    donor_acceptor_valid : bool
        Whether the D...A distance criterion was satisfied.
    hydrogen_acceptor_valid : bool or None
        Whether the H...A distance criterion was satisfied.
    dha_angle_valid : bool or None
        Whether the D-H...A angular criterion was satisfied.
    donor_angle_valid : bool or None
        Whether the inferred donor-angle criterion was satisfied.
    acceptor_angle_valid : bool or None
        Whether the optional acceptor-angle criterion was satisfied.
    is_valid : bool
        Overall geometric validity.
    failed_criteria : sequence of str, optional
        Names of criteria that failed.
    metadata : mapping, optional
        Additional evaluation information.
    """

    geometry: HydrogenBondGeometry
    mode: HydrogenBondMode

    donor_acceptor_valid: bool

    hydrogen_acceptor_valid: Optional[
        bool
    ] = None

    dha_angle_valid: Optional[
        bool
    ] = None

    donor_angle_valid: Optional[
        bool
    ] = None

    acceptor_angle_valid: Optional[
        bool
    ] = None

    is_valid: bool = False

    failed_criteria: Sequence[
        str
    ] = field(
        default_factory=tuple
    )

    metadata: Mapping[
        str,
        Any,
    ] = field(
        default_factory=lambda: _EMPTY_METADATA,
        repr=False,
        compare=False,
    )

    def __post_init__(
        self,
    ) -> None:
        """Validate and normalize the geometry evaluation."""

        if not isinstance(
            self.geometry,
            HydrogenBondGeometry,
        ):
            raise TypeError(
                "geometry must be a HydrogenBondGeometry instance."
            )

        object.__setattr__(
            self,
            "mode",
            validate_hydrogen_bond_mode(
                self.mode
            ),
        )

        object.__setattr__(
            self,
            "donor_acceptor_valid",
            bool(
                self.donor_acceptor_valid
            ),
        )

        for field_name in (
            "hydrogen_acceptor_valid",
            "dha_angle_valid",
            "donor_angle_valid",
            "acceptor_angle_valid",
        ):
            value = getattr(
                self,
                field_name,
            )

            object.__setattr__(
                self,
                field_name,
                None
                if value is None
                else bool(
                    value
                ),
            )

        normalized_failed_criteria = tuple(
            str(
                criterion
            ).strip()
            for criterion in self.failed_criteria
            if str(
                criterion
            ).strip()
        )

        object.__setattr__(
            self,
            "failed_criteria",
            normalized_failed_criteria,
        )

        calculated_validity = not bool(
            normalized_failed_criteria
        )

        object.__setattr__(
            self,
            "is_valid",
            bool(
                self.is_valid
                and calculated_validity
            )
            if self.is_valid
            else calculated_validity,
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
    ) -> Dict[
        str,
        Any,
    ]:
        """
        Serialize the geometric evaluation.

        Returns
        -------
        dict
            Serializable evaluation representation.
        """

        return {
            "geometry": self.geometry.to_dict(),
            "mode": self.mode,
            "donor_acceptor_valid": (
                self.donor_acceptor_valid
            ),
            "hydrogen_acceptor_valid": (
                self.hydrogen_acceptor_valid
            ),
            "dha_angle_valid": (
                self.dha_angle_valid
            ),
            "donor_angle_valid": (
                self.donor_angle_valid
            ),
            "acceptor_angle_valid": (
                self.acceptor_angle_valid
            ),
            "is_valid": self.is_valid,
            "failed_criteria": list(
                self.failed_criteria
            ),
            "metadata": dict(
                self.metadata
            ),
        }


# -----------------------------------------------------------------------------
# Coordinate and vector normalization
# -----------------------------------------------------------------------------

def _as_coordinate_vector(
    value: Any,
    *,
    name: str,
) -> FloatArray:
    """
    Normalize a value as a finite three-dimensional coordinate.

    Parameters
    ----------
    value : Any
        Coordinate-like value.
    name : str
        Human-readable value name.

    Returns
    -------
    numpy.ndarray
        One-dimensional ``float64`` array with three elements.

    Raises
    ------
    ValueError
        If the coordinate is invalid.
    """

    try:
        coordinate = np.asarray(
            value,
            dtype=np.float64,
        )

    except (
        TypeError,
        ValueError,
        OverflowError,
    ) as error:
        raise ValueError(
            f"{name} must be a valid numeric coordinate."
        ) from error

    coordinate = np.ravel(
        coordinate
    )

    if coordinate.shape != (
        3,
    ):
        raise ValueError(
            f"{name} must contain exactly three values."
        )

    if not np.all(
        np.isfinite(
            coordinate
        )
    ):
        raise ValueError(
            f"{name} must contain only finite values."
        )

    return coordinate.astype(
        np.float64,
        copy=False,
    )


def _get_hbond_atom_coordinate(
    atom: AtomLike,
    *,
    name: str,
) -> FloatArray:
    """
    Return a validated atom coordinate.

    Parameters
    ----------
    atom : atom-like
        Atom whose coordinate should be retrieved.
    name : str
        Human-readable atom role.

    Returns
    -------
    numpy.ndarray
        Three-dimensional coordinate.
    """

    if atom is None:
        raise ValueError(
            f"{name} atom cannot be None."
        )

    try:
        coordinate = get_atom_coordinate(
            atom
        )

    except Exception as error:
        raise ValueError(
            f"Could not retrieve the {name} atom coordinate."
        ) from error

    return _as_coordinate_vector(
        coordinate,
        name=f"{name} coordinate",
    )


def _normalize_vector(
    vector: Any,
    *,
    name: str,
    allow_zero: bool = False,
) -> Optional[
    FloatArray
]:
    """
    Normalize a three-dimensional vector to unit length.

    Parameters
    ----------
    vector : Any
        Vector-like value.
    name : str
        Human-readable vector name.
    allow_zero : bool, optional
        Whether a zero-length vector should return ``None``.

    Returns
    -------
    numpy.ndarray or None
        Unit vector.

    Raises
    ------
    ValueError
        If the vector is invalid or has zero length.
    """

    normalized_vector = _as_coordinate_vector(
        vector,
        name=name,
    )

    vector_norm = np.float64(
        np.linalg.norm(
            normalized_vector
        )
    )

    if (
        not np.isfinite(
            vector_norm
        )
        or vector_norm
        <= MINIMUM_VECTOR_NORM
    ):
        if allow_zero:
            return None

        raise ValueError(
            f"{name} has zero or near-zero length."
        )

    return np.asarray(
        normalized_vector
        / vector_norm,
        dtype=np.float64,
    )


# -----------------------------------------------------------------------------
# Fundamental distance and angle calculations
# -----------------------------------------------------------------------------

def calculate_hbond_distance(
    atom_1: AtomLike,
    atom_2: AtomLike,
) -> np.float64:
    """
    Calculate the Euclidean distance between two atoms.

    Parameters
    ----------
    atom_1 : atom-like
        First atom.
    atom_2 : atom-like
        Second atom.

    Returns
    -------
    numpy.float64
        Distance in angstroms.
    """

    coordinate_1 = _get_hbond_atom_coordinate(
        atom_1,
        name="first",
    )

    coordinate_2 = _get_hbond_atom_coordinate(
        atom_2,
        name="second",
    )

    return np.float64(
        np.linalg.norm(
            coordinate_1
            - coordinate_2
        )
    )


def calculate_vector_angle(
    vector_1: Any,
    vector_2: Any,
) -> np.float64:
    """
    Calculate the angle between two vectors.

    Parameters
    ----------
    vector_1 : array-like
        First three-dimensional vector.
    vector_2 : array-like
        Second three-dimensional vector.

    Returns
    -------
    numpy.float64
        Angle in degrees within ``[0, 180]``.

    Raises
    ------
    ValueError
        If either vector is invalid or has zero length.
    """

    normalized_vector_1 = _normalize_vector(
        vector_1,
        name="first vector",
    )

    normalized_vector_2 = _normalize_vector(
        vector_2,
        name="second vector",
    )

    if (
        normalized_vector_1 is None
        or normalized_vector_2 is None
    ):
        raise ValueError(
            "Cannot calculate an angle from a zero-length vector."
        )

    cosine = np.float64(
        np.dot(
            normalized_vector_1,
            normalized_vector_2,
        )
    )

    cosine = np.float64(
        np.clip(
            cosine,
            -1.0,
            1.0,
        )
    )

    return np.float64(
        np.degrees(
            np.arccos(
                cosine
            )
        )
    )


def calculate_three_atom_angle(
    atom_1: AtomLike,
    vertex_atom: AtomLike,
    atom_3: AtomLike,
) -> np.float64:
    """
    Calculate the angle formed by three atoms.

    Parameters
    ----------
    atom_1 : atom-like
        First terminal atom.
    vertex_atom : atom-like
        Central atom defining the angle vertex.
    atom_3 : atom-like
        Second terminal atom.

    Returns
    -------
    numpy.float64
        Angle ``atom_1-vertex_atom-atom_3`` in degrees.
    """

    coordinate_1 = _get_hbond_atom_coordinate(
        atom_1,
        name="first terminal",
    )

    vertex_coordinate = _get_hbond_atom_coordinate(
        vertex_atom,
        name="angle vertex",
    )

    coordinate_3 = _get_hbond_atom_coordinate(
        atom_3,
        name="second terminal",
    )

    return calculate_vector_angle(
        coordinate_1
        - vertex_coordinate,
        coordinate_3
        - vertex_coordinate,
    )


def calculate_donor_acceptor_distance(
    donor: AtomLike,
    acceptor: AtomLike,
) -> np.float64:
    """
    Calculate the donor-acceptor distance.

    Parameters
    ----------
    donor : atom-like
        Donor atom.
    acceptor : atom-like
        Acceptor atom.

    Returns
    -------
    numpy.float64
        D...A distance in angstroms.
    """

    if donor is acceptor:
        raise ValueError(
            "Donor and acceptor must be different atoms."
        )

    return calculate_hbond_distance(
        donor,
        acceptor,
    )


def calculate_hydrogen_acceptor_distance(
    hydrogen: AtomLike,
    acceptor: AtomLike,
) -> np.float64:
    """
    Calculate the hydrogen-acceptor distance.

    Parameters
    ----------
    hydrogen : atom-like
        Explicit donor-bound hydrogen.
    acceptor : atom-like
        Acceptor atom.

    Returns
    -------
    numpy.float64
        H...A distance in angstroms.
    """

    if not is_hydrogen_atom(
        hydrogen
    ):
        raise ValueError(
            "The supplied hydrogen atom is not recognized as hydrogen."
        )

    if hydrogen is acceptor:
        raise ValueError(
            "Hydrogen and acceptor must be different atoms."
        )

    return calculate_hbond_distance(
        hydrogen,
        acceptor,
    )


def calculate_donor_hydrogen_distance(
    donor: AtomLike,
    hydrogen: AtomLike,
) -> np.float64:
    """
    Calculate the donor-hydrogen distance.

    Parameters
    ----------
    donor : atom-like
        Donor atom.
    hydrogen : atom-like
        Explicit hydrogen atom.

    Returns
    -------
    numpy.float64
        D-H distance in angstroms.
    """

    if donor is hydrogen:
        raise ValueError(
            "Donor and hydrogen must be different atoms."
        )

    if not is_hydrogen_atom(
        hydrogen
    ):
        raise ValueError(
            "The supplied hydrogen atom is not recognized as hydrogen."
        )

    return calculate_hbond_distance(
        donor,
        hydrogen,
    )


def calculate_dha_angle(
    donor: AtomLike,
    hydrogen: AtomLike,
    acceptor: AtomLike,
) -> np.float64:
    """
    Calculate the donor-hydrogen-acceptor angle.

    Parameters
    ----------
    donor : atom-like
        Donor atom.
    hydrogen : atom-like
        Explicit donor-bound hydrogen.
    acceptor : atom-like
        Acceptor atom.

    Returns
    -------
    numpy.float64
        D-H...A angle in degrees.

    Notes
    -----
    The hydrogen is the angle vertex. A linear hydrogen bond approaches
    180 degrees.
    """

    if (
        donor is hydrogen
        or donor is acceptor
        or hydrogen is acceptor
    ):
        raise ValueError(
            "Donor, hydrogen and acceptor must be distinct atoms."
        )

    if not is_hydrogen_atom(
        hydrogen
    ):
        raise ValueError(
            "The central atom must be hydrogen."
        )

    return calculate_three_atom_angle(
        donor,
        hydrogen,
        acceptor,
    )


# -----------------------------------------------------------------------------
# Neighbor-coordinate collection
# -----------------------------------------------------------------------------

def _get_valid_heavy_neighbor_coordinates(
    atom: AtomLike,
    *,
    excluded_atoms: Sequence[
        AtomLike
    ] = (),
    bond_resolver: Optional[
        BondResolver
    ] = None,
) -> Tuple[
    FloatArray,
    ...,
]:
    """
    Return valid heavy-neighbor coordinates for an atom.

    Parameters
    ----------
    atom : atom-like
        Central atom.
    excluded_atoms : sequence of atom-like, optional
        Neighbors that should be ignored.
    bond_resolver : callable or None, optional
        Custom bonded-neighbor resolver.

    Returns
    -------
    tuple of numpy.ndarray
        Valid heavy-neighbor coordinates.
    """

    excluded_ids = {
        id(
            excluded_atom
        )
        for excluded_atom in excluded_atoms
        if excluded_atom is not None
    }

    coordinates: List[
        FloatArray
    ] = []

    for neighbor in get_bonded_heavy_atoms(
        atom,
        bond_resolver=bond_resolver,
    ):
        if id(
            neighbor
        ) in excluded_ids:
            continue

        try:
            coordinate = (
                _get_hbond_atom_coordinate(
                    neighbor,
                    name="bonded neighbor",
                )
            )

        except Exception:
            continue

        coordinates.append(
            coordinate
        )

    return tuple(
        coordinates
    )


# -----------------------------------------------------------------------------
# Inferred donor and acceptor vectors
# -----------------------------------------------------------------------------

def validate_inferred_vector_method(
    method: str,
) -> str:
    """
    Validate an inferred-vector construction method.

    Parameters
    ----------
    method : str
        Vector method.

    Returns
    -------
    str
        Normalized method.

    Raises
    ------
    TypeError
        If ``method`` is not a string.
    ValueError
        If the method is unsupported.
    """

    if not isinstance(
        method,
        str,
    ):
        raise TypeError(
            "Inferred-vector method must be a string."
        )

    normalized_method = (
        method.strip().lower()
    )

    if (
        normalized_method
        not in VALID_INFERRED_VECTOR_METHODS
    ):
        raise ValueError(
            f"Unsupported inferred-vector method {method!r}. "
            "Expected one of: "
            f"{', '.join(sorted(VALID_INFERRED_VECTOR_METHODS))}."
        )

    return normalized_method


def infer_open_valence_vector(
    central_atom: AtomLike,
    *,
    excluded_atoms: Sequence[
        AtomLike
    ] = (),
    method: str = DEFAULT_INFERRED_DONOR_VECTOR_METHOD,
    bond_resolver: Optional[
        BondResolver
    ] = None,
) -> Optional[
    FloatArray
]:
    """
    Infer the direction of an open valence from bonded heavy atoms.

    Parameters
    ----------
    central_atom : atom-like
        Atom for which an outward vector should be inferred.
    excluded_atoms : sequence of atom-like, optional
        Bonded atoms to ignore.
    method : str, optional
        Vector-construction method.
    bond_resolver : callable or None, optional
        Custom bonded-neighbor resolver.

    Returns
    -------
    numpy.ndarray or None
        Unit vector pointing toward the inferred open valence.

    Notes
    -----
    ``"opposite_neighbors"`` computes the direction opposite to the sum of
    normalized bond vectors.

    ``"nearest_neighbor"`` uses the direction opposite to the nearest bonded
    heavy atom.

    ``"centroid"`` points away from the centroid of bonded heavy atoms.
    """

    normalized_method = (
        validate_inferred_vector_method(
            method
        )
    )

    if normalized_method == "none":
        return None

    central_coordinate = (
        _get_hbond_atom_coordinate(
            central_atom,
            name="central atom",
        )
    )

    neighbor_coordinates = (
        _get_valid_heavy_neighbor_coordinates(
            central_atom,
            excluded_atoms=excluded_atoms,
            bond_resolver=bond_resolver,
        )
    )

    if not neighbor_coordinates:
        return None

    if normalized_method == "nearest_neighbor":
        distances_and_coordinates = [
            (
                np.float64(
                    np.linalg.norm(
                        neighbor_coordinate
                        - central_coordinate
                    )
                ),
                neighbor_coordinate,
            )
            for neighbor_coordinate
            in neighbor_coordinates
        ]

        distances_and_coordinates = [
            item
            for item in distances_and_coordinates
            if (
                np.isfinite(
                    item[
                        0
                    ]
                )
                and item[
                    0
                ]
                > MINIMUM_VECTOR_NORM
            )
        ]

        if not distances_and_coordinates:
            return None

        _, nearest_coordinate = min(
            distances_and_coordinates,
            key=lambda item: float(
                item[
                    0
                ]
            ),
        )

        return _normalize_vector(
            central_coordinate
            - nearest_coordinate,
            name="nearest-neighbor open-valence vector",
            allow_zero=True,
        )

    if normalized_method == "centroid":
        neighbor_centroid = np.mean(
            np.vstack(
                neighbor_coordinates
            ),
            axis=0,
            dtype=np.float64,
        )

        return _normalize_vector(
            central_coordinate
            - neighbor_centroid,
            name="centroid open-valence vector",
            allow_zero=True,
        )

    normalized_bond_vectors: List[
        FloatArray
    ] = []

    for neighbor_coordinate in neighbor_coordinates:
        normalized_bond_vector = (
            _normalize_vector(
                neighbor_coordinate
                - central_coordinate,
                name="central-to-neighbor bond vector",
                allow_zero=True,
            )
        )

        if normalized_bond_vector is not None:
            normalized_bond_vectors.append(
                normalized_bond_vector
            )

    if not normalized_bond_vectors:
        return None

    summed_bond_vector = np.sum(
        np.vstack(
            normalized_bond_vectors
        ),
        axis=0,
        dtype=np.float64,
    )

    open_valence_vector = (
        _normalize_vector(
            -summed_bond_vector,
            name="opposite-neighbor open-valence vector",
            allow_zero=True,
        )
    )

    if open_valence_vector is not None:
        return open_valence_vector

    # Symmetric arrangements can produce a zero vector. Fall back to the
    # direction opposite the nearest bonded atom.
    return infer_open_valence_vector(
        central_atom,
        excluded_atoms=excluded_atoms,
        method="nearest_neighbor",
        bond_resolver=bond_resolver,
    )


def calculate_inferred_donor_angle(
    donor: AtomLike,
    acceptor: AtomLike,
    *,
    method: str = DEFAULT_INFERRED_DONOR_VECTOR_METHOD,
    bond_resolver: Optional[
        BondResolver
    ] = None,
) -> Optional[
    np.float64
]:
    """
    Calculate an inferred donor-direction angle.

    Parameters
    ----------
    donor : atom-like
        Donor atom lacking an explicit hydrogen.
    acceptor : atom-like
        Candidate acceptor atom.
    method : str, optional
        Open-valence-vector method.
    bond_resolver : callable or None, optional
        Custom bonded-neighbor resolver.

    Returns
    -------
    numpy.float64 or None
        Angle between the inferred donor open valence and D...A.

    Notes
    -----
    A value near zero means that the inferred donor open valence points
    directly toward the acceptor. This differs from the explicit D-H...A
    convention, in which ideal geometry approaches 180 degrees.
    """

    if donor is acceptor:
        raise ValueError(
            "Donor and acceptor must be different atoms."
        )

    donor_coordinate = (
        _get_hbond_atom_coordinate(
            donor,
            name="donor",
        )
    )

    acceptor_coordinate = (
        _get_hbond_atom_coordinate(
            acceptor,
            name="acceptor",
        )
    )

    donor_open_vector = infer_open_valence_vector(
        donor,
        excluded_atoms=(
            acceptor,
        ),
        method=method,
        bond_resolver=bond_resolver,
    )

    if donor_open_vector is None:
        return None

    donor_acceptor_vector = (
        _normalize_vector(
            acceptor_coordinate
            - donor_coordinate,
            name="donor-acceptor vector",
            allow_zero=True,
        )
    )

    if donor_acceptor_vector is None:
        return None

    return calculate_vector_angle(
        donor_open_vector,
        donor_acceptor_vector,
    )


def calculate_inferred_acceptor_angle(
    donor: AtomLike,
    acceptor: AtomLike,
    *,
    method: str = DEFAULT_INFERRED_ACCEPTOR_VECTOR_METHOD,
    bond_resolver: Optional[
        BondResolver
    ] = None,
) -> Optional[
    np.float64
]:
    """
    Calculate an inferred acceptor-direction angle.

    Parameters
    ----------
    donor : atom-like
        Candidate donor atom.
    acceptor : atom-like
        Acceptor atom.
    method : str, optional
        Open-valence-vector method.
    bond_resolver : callable or None, optional
        Custom bonded-neighbor resolver.

    Returns
    -------
    numpy.float64 or None
        Angle between the inferred acceptor open valence and A...D.

    Notes
    -----
    A value near zero indicates that the inferred acceptor lone-pair direction
    points toward the donor.
    """

    if donor is acceptor:
        raise ValueError(
            "Donor and acceptor must be different atoms."
        )

    donor_coordinate = (
        _get_hbond_atom_coordinate(
            donor,
            name="donor",
        )
    )

    acceptor_coordinate = (
        _get_hbond_atom_coordinate(
            acceptor,
            name="acceptor",
        )
    )

    acceptor_open_vector = (
        infer_open_valence_vector(
            acceptor,
            excluded_atoms=(
                donor,
            ),
            method=method,
            bond_resolver=bond_resolver,
        )
    )

    if acceptor_open_vector is None:
        return None

    acceptor_donor_vector = (
        _normalize_vector(
            donor_coordinate
            - acceptor_coordinate,
            name="acceptor-donor vector",
            allow_zero=True,
        )
    )

    if acceptor_donor_vector is None:
        return None

    return calculate_vector_angle(
        acceptor_open_vector,
        acceptor_donor_vector,
    )


# -----------------------------------------------------------------------------
# Explicit hydrogen-bond geometry
# -----------------------------------------------------------------------------

def calculate_explicit_hbond_geometry(
    donor: AtomLike,
    hydrogen: AtomLike,
    acceptor: AtomLike,
    *,
    calculate_acceptor_angle: bool = True,
    acceptor_vector_method: str = (
        DEFAULT_INFERRED_ACCEPTOR_VECTOR_METHOD
    ),
    bond_resolver: Optional[
        BondResolver
    ] = None,
    metadata: Optional[
        Mapping[
            str,
            Any,
        ]
    ] = None,
) -> HydrogenBondGeometry:
    """
    Calculate complete geometry for an explicit D-H...A interaction.

    Parameters
    ----------
    donor : atom-like
        Donor atom.
    hydrogen : atom-like
        Donor-bound explicit hydrogen.
    acceptor : atom-like
        Acceptor atom.
    calculate_acceptor_angle : bool, optional
        Whether an auxiliary acceptor-side angle should be calculated.
    acceptor_vector_method : str, optional
        Method used to infer the acceptor open-valence direction.
    bond_resolver : callable or None, optional
        Custom bonded-neighbor resolver.
    metadata : mapping or None, optional
        Additional geometry metadata.

    Returns
    -------
    HydrogenBondGeometry
        Calculated geometry.
    """

    if (
        donor is hydrogen
        or donor is acceptor
        or hydrogen is acceptor
    ):
        raise ValueError(
            "Donor, hydrogen and acceptor must be distinct atoms."
        )

    if not is_hydrogen_atom(
        hydrogen
    ):
        raise ValueError(
            "The explicit hydrogen atom is not recognized as hydrogen."
        )

    donor_acceptor_distance = (
        calculate_donor_acceptor_distance(
            donor,
            acceptor,
        )
    )

    hydrogen_acceptor_distance = (
        calculate_hydrogen_acceptor_distance(
            hydrogen,
            acceptor,
        )
    )

    donor_hydrogen_distance = (
        calculate_donor_hydrogen_distance(
            donor,
            hydrogen,
        )
    )

    dha_angle = calculate_dha_angle(
        donor,
        hydrogen,
        acceptor,
    )

    acceptor_angle: Optional[
        np.float64
    ] = None

    if calculate_acceptor_angle:
        acceptor_angle = (
            calculate_inferred_acceptor_angle(
                donor,
                acceptor,
                method=acceptor_vector_method,
                bond_resolver=bond_resolver,
            )
        )

    geometry_metadata: Dict[
        str,
        Any,
    ] = {
        "geometry_mode": (
            HBOND_MODE_EXPLICIT
        ),
        "acceptor_vector_method": (
            acceptor_vector_method
            if calculate_acceptor_angle
            else None
        ),
    }

    if metadata:
        geometry_metadata.update(
            metadata
        )

    return HydrogenBondGeometry(
        donor_acceptor_distance=(
            donor_acceptor_distance
        ),
        hydrogen_acceptor_distance=(
            hydrogen_acceptor_distance
        ),
        donor_hydrogen_distance=(
            donor_hydrogen_distance
        ),
        dha_angle=dha_angle,
        donor_angle=None,
        acceptor_angle=acceptor_angle,
        metadata=geometry_metadata,
    )


def calculate_assignment_hbond_geometry(
    assignment: DonorHydrogenAssignment,
    acceptor: AtomLike,
    *,
    calculate_acceptor_angle: bool = True,
    acceptor_vector_method: str = (
        DEFAULT_INFERRED_ACCEPTOR_VECTOR_METHOD
    ),
    bond_resolver: Optional[
        BondResolver
    ] = None,
    metadata: Optional[
        Mapping[
            str,
            Any,
        ]
    ] = None,
) -> HydrogenBondGeometry:
    """
    Calculate explicit geometry from a donor-hydrogen assignment.

    Parameters
    ----------
    assignment : DonorHydrogenAssignment
        Donor-hydrogen assignment.
    acceptor : atom-like
        Candidate acceptor.
    calculate_acceptor_angle : bool, optional
        Whether the auxiliary acceptor angle should be calculated.
    acceptor_vector_method : str, optional
        Acceptor-vector method.
    bond_resolver : callable or None, optional
        Custom bonded-neighbor resolver.
    metadata : mapping or None, optional
        Additional geometry metadata.

    Returns
    -------
    HydrogenBondGeometry
        Calculated explicit geometry.
    """

    if not isinstance(
        assignment,
        DonorHydrogenAssignment,
    ):
        raise TypeError(
            "assignment must be a DonorHydrogenAssignment instance."
        )

    assignment_metadata: Dict[
        str,
        Any,
    ] = {
        "assignment_method": (
            assignment.assignment_method
        ),
        "assignment_ambiguous": (
            assignment.is_ambiguous
        ),
    }

    if metadata:
        assignment_metadata.update(
            metadata
        )

    return calculate_explicit_hbond_geometry(
        assignment.donor,
        assignment.hydrogen,
        acceptor,
        calculate_acceptor_angle=(
            calculate_acceptor_angle
        ),
        acceptor_vector_method=(
            acceptor_vector_method
        ),
        bond_resolver=bond_resolver,
        metadata=assignment_metadata,
    )


# -----------------------------------------------------------------------------
# Inferred hydrogen-bond geometry
# -----------------------------------------------------------------------------

def calculate_inferred_hbond_geometry(
    donor: AtomLike,
    acceptor: AtomLike,
    *,
    donor_vector_method: str = (
        DEFAULT_INFERRED_DONOR_VECTOR_METHOD
    ),
    acceptor_vector_method: str = (
        DEFAULT_INFERRED_ACCEPTOR_VECTOR_METHOD
    ),
    calculate_acceptor_angle: bool = True,
    bond_resolver: Optional[
        BondResolver
    ] = None,
    metadata: Optional[
        Mapping[
            str,
            Any,
        ]
    ] = None,
) -> HydrogenBondGeometry:
    """
    Calculate inferred D...A geometry without an explicit hydrogen.

    Parameters
    ----------
    donor : atom-like
        Donor atom.
    acceptor : atom-like
        Acceptor atom.
    donor_vector_method : str, optional
        Method used to infer the donor open-valence direction.
    acceptor_vector_method : str, optional
        Method used to infer the acceptor open-valence direction.
    calculate_acceptor_angle : bool, optional
        Whether the acceptor-side angle should be calculated.
    bond_resolver : callable or None, optional
        Custom bonded-neighbor resolver.
    metadata : mapping or None, optional
        Additional geometry metadata.

    Returns
    -------
    HydrogenBondGeometry
        Inferred geometry.

    Notes
    -----
    ``donor_angle`` and ``acceptor_angle`` are deviation angles. Values closer
    to zero indicate better alignment toward the interaction partner.
    """

    donor_acceptor_distance = (
        calculate_donor_acceptor_distance(
            donor,
            acceptor,
        )
    )

    donor_angle = calculate_inferred_donor_angle(
        donor,
        acceptor,
        method=donor_vector_method,
        bond_resolver=bond_resolver,
    )

    acceptor_angle: Optional[
        np.float64
    ] = None

    if calculate_acceptor_angle:
        acceptor_angle = (
            calculate_inferred_acceptor_angle(
                donor,
                acceptor,
                method=acceptor_vector_method,
                bond_resolver=bond_resolver,
            )
        )

    geometry_metadata: Dict[
        str,
        Any,
    ] = {
        "geometry_mode": (
            HBOND_MODE_INFERRED
        ),
        "donor_vector_method": (
            donor_vector_method
        ),
        "acceptor_vector_method": (
            acceptor_vector_method
            if calculate_acceptor_angle
            else None
        ),
        "angle_convention": (
            "zero_is_optimal"
        ),
    }

    if metadata:
        geometry_metadata.update(
            metadata
        )

    return HydrogenBondGeometry(
        donor_acceptor_distance=(
            donor_acceptor_distance
        ),
        hydrogen_acceptor_distance=None,
        donor_hydrogen_distance=None,
        dha_angle=None,
        donor_angle=donor_angle,
        acceptor_angle=acceptor_angle,
        metadata=geometry_metadata,
    )


# -----------------------------------------------------------------------------
# Geometry validation
# -----------------------------------------------------------------------------

def _distance_within_cutoff(
    value: Number,
    cutoff: Number,
    *,
    tolerance: Number = 0.0,
) -> bool:
    """
    Test whether a distance is within a cutoff.

    Parameters
    ----------
    value : Number
        Observed distance.
    cutoff : Number
        Maximum accepted distance.
    tolerance : Number, optional
        Additional accepted tolerance.

    Returns
    -------
    bool
        Cutoff status.
    """

    normalized_value = _optional_float64(
        value,
        name="distance",
        minimum=0.0,
    )

    normalized_cutoff = _optional_float64(
        cutoff,
        name="distance cutoff",
        minimum=0.0,
    )

    normalized_tolerance = _optional_float64(
        tolerance,
        name="distance tolerance",
        minimum=0.0,
    )

    if (
        normalized_value is None
        or normalized_cutoff is None
    ):
        return False

    if normalized_tolerance is None:
        normalized_tolerance = np.float64(
            0.0
        )

    return bool(
        normalized_value
        <= (
            normalized_cutoff
            + normalized_tolerance
        )
    )


def _angle_above_cutoff(
    value: Number,
    cutoff: Number,
    *,
    tolerance: Number = 0.0,
) -> bool:
    """
    Test whether a conventional angle reaches a minimum cutoff.

    Parameters
    ----------
    value : Number
        Observed angle.
    cutoff : Number
        Minimum accepted angle.
    tolerance : Number, optional
        Angular tolerance subtracted from the cutoff.

    Returns
    -------
    bool
        Cutoff status.
    """

    normalized_value = _optional_float64(
        value,
        name="angle",
        minimum=MINIMUM_VALID_ANGLE_DEGREES,
        maximum=MAXIMUM_VALID_ANGLE_DEGREES,
    )

    normalized_cutoff = _optional_float64(
        cutoff,
        name="minimum angle",
        minimum=MINIMUM_VALID_ANGLE_DEGREES,
        maximum=MAXIMUM_VALID_ANGLE_DEGREES,
    )

    normalized_tolerance = _optional_float64(
        tolerance,
        name="angle tolerance",
        minimum=0.0,
    )

    if (
        normalized_value is None
        or normalized_cutoff is None
    ):
        return False

    if normalized_tolerance is None:
        normalized_tolerance = np.float64(
            0.0
        )

    effective_cutoff = np.float64(
        max(
            0.0,
            float(
                normalized_cutoff
                - normalized_tolerance
            ),
        )
    )

    return bool(
        normalized_value
        >= effective_cutoff
    )


def _deviation_angle_within_cutoff(
    value: Number,
    maximum_deviation: Number,
    *,
    tolerance: Number = 0.0,
) -> bool:
    """
    Test whether an inferred deviation angle is sufficiently small.

    Parameters
    ----------
    value : Number
        Observed deviation angle.
    maximum_deviation : Number
        Maximum accepted deviation.
    tolerance : Number, optional
        Additional angular tolerance.

    Returns
    -------
    bool
        Cutoff status.
    """

    normalized_value = _optional_float64(
        value,
        name="deviation angle",
        minimum=MINIMUM_VALID_ANGLE_DEGREES,
        maximum=MAXIMUM_VALID_ANGLE_DEGREES,
    )

    normalized_cutoff = _optional_float64(
        maximum_deviation,
        name="maximum deviation angle",
        minimum=MINIMUM_VALID_ANGLE_DEGREES,
        maximum=MAXIMUM_VALID_ANGLE_DEGREES,
    )

    normalized_tolerance = _optional_float64(
        tolerance,
        name="angle tolerance",
        minimum=0.0,
    )

    if (
        normalized_value is None
        or normalized_cutoff is None
    ):
        return False

    if normalized_tolerance is None:
        normalized_tolerance = np.float64(
            0.0
        )

    return bool(
        normalized_value
        <= (
            normalized_cutoff
            + normalized_tolerance
        )
    )


def evaluate_explicit_hbond_geometry(
    geometry: HydrogenBondGeometry,
    *,
    donor_acceptor_cutoff: Number = (
        DEFAULT_DONOR_ACCEPTOR_DISTANCE
    ),
    hydrogen_acceptor_cutoff: Number = (
        DEFAULT_HYDROGEN_ACCEPTOR_DISTANCE
    ),
    minimum_dha_angle: Number = (
        DEFAULT_MINIMUM_DHA_ANGLE
    ),
    distance_tolerance: Number = (
        DEFAULT_DISTANCE_TOLERANCE
    ),
    angle_tolerance: Number = (
        DEFAULT_ANGLE_TOLERANCE
    ),
    require_hydrogen_acceptor_distance: bool = (
        DEFAULT_REQUIRE_HYDROGEN_ACCEPTOR_DISTANCE
    ),
    require_dha_angle: bool = (
        DEFAULT_REQUIRE_DHA_ANGLE
    ),
) -> HydrogenBondGeometryEvaluation:
    """
    Evaluate explicit D-H...A geometry.

    Parameters
    ----------
    geometry : HydrogenBondGeometry
        Explicit geometry.
    donor_acceptor_cutoff : Number, optional
        Maximum D...A distance.
    hydrogen_acceptor_cutoff : Number, optional
        Maximum H...A distance.
    minimum_dha_angle : Number, optional
        Minimum D-H...A angle.
    distance_tolerance : Number, optional
        Distance tolerance.
    angle_tolerance : Number, optional
        Angular tolerance.
    require_hydrogen_acceptor_distance : bool, optional
        Whether missing or invalid H...A distance rejects the interaction.
    require_dha_angle : bool, optional
        Whether missing or invalid D-H...A angle rejects the interaction.

    Returns
    -------
    HydrogenBondGeometryEvaluation
        Geometric evaluation.
    """

    if not isinstance(
        geometry,
        HydrogenBondGeometry,
    ):
        raise TypeError(
            "geometry must be a HydrogenBondGeometry instance."
        )

    donor_acceptor_valid = (
        _distance_within_cutoff(
            geometry.donor_acceptor_distance,
            donor_acceptor_cutoff,
            tolerance=distance_tolerance,
        )
    )

    hydrogen_acceptor_valid: Optional[
        bool
    ]

    if geometry.hydrogen_acceptor_distance is None:
        hydrogen_acceptor_valid = None

    else:
        hydrogen_acceptor_valid = (
            _distance_within_cutoff(
                geometry.hydrogen_acceptor_distance,
                hydrogen_acceptor_cutoff,
                tolerance=distance_tolerance,
            )
        )

    dha_angle_valid: Optional[
        bool
    ]

    if geometry.dha_angle is None:
        dha_angle_valid = None

    else:
        dha_angle_valid = _angle_above_cutoff(
            geometry.dha_angle,
            minimum_dha_angle,
            tolerance=angle_tolerance,
        )

    failed_criteria: List[
        str
    ] = []

    if not donor_acceptor_valid:
        failed_criteria.append(
            "donor_acceptor_distance"
        )

    if require_hydrogen_acceptor_distance:
        if hydrogen_acceptor_valid is not True:
            failed_criteria.append(
                "hydrogen_acceptor_distance"
            )

    elif hydrogen_acceptor_valid is False:
        failed_criteria.append(
            "hydrogen_acceptor_distance"
        )

    if require_dha_angle:
        if dha_angle_valid is not True:
            failed_criteria.append(
                "dha_angle"
            )

    elif dha_angle_valid is False:
        failed_criteria.append(
            "dha_angle"
        )

    return HydrogenBondGeometryEvaluation(
        geometry=geometry,
        mode=HBOND_MODE_EXPLICIT,
        donor_acceptor_valid=(
            donor_acceptor_valid
        ),
        hydrogen_acceptor_valid=(
            hydrogen_acceptor_valid
        ),
        dha_angle_valid=dha_angle_valid,
        donor_angle_valid=None,
        acceptor_angle_valid=None,
        is_valid=not failed_criteria,
        failed_criteria=failed_criteria,
        metadata={
            "donor_acceptor_cutoff": float(
                np.float64(
                    donor_acceptor_cutoff
                )
            ),
            "hydrogen_acceptor_cutoff": float(
                np.float64(
                    hydrogen_acceptor_cutoff
                )
            ),
            "minimum_dha_angle": float(
                np.float64(
                    minimum_dha_angle
                )
            ),
        },
    )


def evaluate_inferred_hbond_geometry(
    geometry: HydrogenBondGeometry,
    *,
    donor_acceptor_cutoff: Number = (
        DEFAULT_DONOR_ACCEPTOR_DISTANCE
    ),
    maximum_donor_deviation: Optional[
        Number
    ] = None,
    maximum_acceptor_deviation: Optional[
        Number
    ] = None,
    minimum_inferred_angle: Number = (
        DEFAULT_MINIMUM_INFERRED_ANGLE
    ),
    distance_tolerance: Number = (
        DEFAULT_DISTANCE_TOLERANCE
    ),
    angle_tolerance: Number = (
        DEFAULT_ANGLE_TOLERANCE
    ),
    require_donor_angle: bool = (
        DEFAULT_REQUIRE_INFERRED_ANGLE
    ),
    require_acceptor_angle: bool = False,
) -> HydrogenBondGeometryEvaluation:
    """
    Evaluate inferred D...A hydrogen-bond geometry.

    Parameters
    ----------
    geometry : HydrogenBondGeometry
        Inferred geometry.
    donor_acceptor_cutoff : Number, optional
        Maximum D...A distance.
    maximum_donor_deviation : Number or None, optional
        Maximum donor-direction deviation. When omitted, it is derived as
        ``180 - minimum_inferred_angle``.
    maximum_acceptor_deviation : Number or None, optional
        Maximum acceptor-direction deviation. When omitted, the donor limit is
        reused.
    minimum_inferred_angle : Number, optional
        Compatibility parameter expressed in conventional linear-angle form.
    distance_tolerance : Number, optional
        Distance tolerance.
    angle_tolerance : Number, optional
        Angular tolerance.
    require_donor_angle : bool, optional
        Whether an inferred donor angle is mandatory.
    require_acceptor_angle : bool, optional
        Whether an inferred acceptor angle is mandatory.

    Returns
    -------
    HydrogenBondGeometryEvaluation
        Geometric evaluation.

    Notes
    -----
    Inferred angles stored in ``HydrogenBondGeometry`` are deviation angles:
    zero degrees is optimal. The traditional minimum inferred angle is
    converted to a maximum deviation using ``180 - minimum_angle``.
    """

    if not isinstance(
        geometry,
        HydrogenBondGeometry,
    ):
        raise TypeError(
            "geometry must be a HydrogenBondGeometry instance."
        )

    normalized_minimum_inferred_angle = (
        _optional_float64(
            minimum_inferred_angle,
            name="minimum inferred angle",
            minimum=MINIMUM_VALID_ANGLE_DEGREES,
            maximum=MAXIMUM_VALID_ANGLE_DEGREES,
        )
    )

    if normalized_minimum_inferred_angle is None:
        normalized_minimum_inferred_angle = (
            DEFAULT_MINIMUM_INFERRED_ANGLE
        )

    if maximum_donor_deviation is None:
        donor_deviation_cutoff = np.float64(
            STRAIGHT_ANGLE_DEGREES
            - normalized_minimum_inferred_angle
        )

    else:
        donor_deviation_cutoff = (
            _optional_float64(
                maximum_donor_deviation,
                name="maximum donor deviation",
                minimum=MINIMUM_VALID_ANGLE_DEGREES,
                maximum=MAXIMUM_VALID_ANGLE_DEGREES,
            )
        )

        if donor_deviation_cutoff is None:
            raise ValueError(
                "Maximum donor deviation cannot be None."
            )

    if maximum_acceptor_deviation is None:
        acceptor_deviation_cutoff = (
            donor_deviation_cutoff
        )

    else:
        acceptor_deviation_cutoff = (
            _optional_float64(
                maximum_acceptor_deviation,
                name="maximum acceptor deviation",
                minimum=MINIMUM_VALID_ANGLE_DEGREES,
                maximum=MAXIMUM_VALID_ANGLE_DEGREES,
            )
        )

        if acceptor_deviation_cutoff is None:
            raise ValueError(
                "Maximum acceptor deviation cannot be None."
            )

    donor_acceptor_valid = (
        _distance_within_cutoff(
            geometry.donor_acceptor_distance,
            donor_acceptor_cutoff,
            tolerance=distance_tolerance,
        )
    )

    donor_angle_valid: Optional[
        bool
    ]

    if geometry.donor_angle is None:
        donor_angle_valid = None

    else:
        donor_angle_valid = (
            _deviation_angle_within_cutoff(
                geometry.donor_angle,
                donor_deviation_cutoff,
                tolerance=angle_tolerance,
            )
        )

    acceptor_angle_valid: Optional[
        bool
    ]

    if geometry.acceptor_angle is None:
        acceptor_angle_valid = None

    else:
        acceptor_angle_valid = (
            _deviation_angle_within_cutoff(
                geometry.acceptor_angle,
                acceptor_deviation_cutoff,
                tolerance=angle_tolerance,
            )
        )

    failed_criteria: List[
        str
    ] = []

    if not donor_acceptor_valid:
        failed_criteria.append(
            "donor_acceptor_distance"
        )

    if require_donor_angle:
        if donor_angle_valid is not True:
            failed_criteria.append(
                "donor_angle"
            )

    elif donor_angle_valid is False:
        failed_criteria.append(
            "donor_angle"
        )

    if require_acceptor_angle:
        if acceptor_angle_valid is not True:
            failed_criteria.append(
                "acceptor_angle"
            )

    elif acceptor_angle_valid is False:
        failed_criteria.append(
            "acceptor_angle"
        )

    return HydrogenBondGeometryEvaluation(
        geometry=geometry,
        mode=HBOND_MODE_INFERRED,
        donor_acceptor_valid=(
            donor_acceptor_valid
        ),
        hydrogen_acceptor_valid=None,
        dha_angle_valid=None,
        donor_angle_valid=(
            donor_angle_valid
        ),
        acceptor_angle_valid=(
            acceptor_angle_valid
        ),
        is_valid=not failed_criteria,
        failed_criteria=failed_criteria,
        metadata={
            "donor_acceptor_cutoff": float(
                np.float64(
                    donor_acceptor_cutoff
                )
            ),
            "maximum_donor_deviation": float(
                donor_deviation_cutoff
            ),
            "maximum_acceptor_deviation": float(
                acceptor_deviation_cutoff
            ),
            "minimum_inferred_angle": float(
                normalized_minimum_inferred_angle
            ),
        },
    )


def evaluate_hbond_geometry(
    geometry: HydrogenBondGeometry,
    *,
    mode: Optional[
        str
    ] = None,
    donor_acceptor_cutoff: Number = (
        DEFAULT_DONOR_ACCEPTOR_DISTANCE
    ),
    hydrogen_acceptor_cutoff: Number = (
        DEFAULT_HYDROGEN_ACCEPTOR_DISTANCE
    ),
    minimum_dha_angle: Number = (
        DEFAULT_MINIMUM_DHA_ANGLE
    ),
    minimum_inferred_angle: Number = (
        DEFAULT_MINIMUM_INFERRED_ANGLE
    ),
    maximum_donor_deviation: Optional[
        Number
    ] = None,
    maximum_acceptor_deviation: Optional[
        Number
    ] = None,
    distance_tolerance: Number = (
        DEFAULT_DISTANCE_TOLERANCE
    ),
    angle_tolerance: Number = (
        DEFAULT_ANGLE_TOLERANCE
    ),
    require_acceptor_angle: bool = False,
) -> HydrogenBondGeometryEvaluation:
    """
    Evaluate explicit or inferred hydrogen-bond geometry.

    Parameters
    ----------
    geometry : HydrogenBondGeometry
        Geometry to evaluate.
    mode : str or None, optional
        Explicit or inferred mode. When omitted, the mode is inferred from
        hydrogen-specific measurements.
    donor_acceptor_cutoff : Number, optional
        Maximum D...A distance.
    hydrogen_acceptor_cutoff : Number, optional
        Maximum H...A distance.
    minimum_dha_angle : Number, optional
        Minimum D-H...A angle.
    minimum_inferred_angle : Number, optional
        Minimum conventional inferred angle.
    maximum_donor_deviation : Number or None, optional
        Maximum inferred donor deviation.
    maximum_acceptor_deviation : Number or None, optional
        Maximum inferred acceptor deviation.
    distance_tolerance : Number, optional
        Distance tolerance.
    angle_tolerance : Number, optional
        Angular tolerance.
    require_acceptor_angle : bool, optional
        Whether the acceptor angle is mandatory.

    Returns
    -------
    HydrogenBondGeometryEvaluation
        Geometry evaluation.
    """

    if mode is None:
        normalized_mode = (
            HBOND_MODE_EXPLICIT
            if (
                geometry.hydrogen_acceptor_distance
                is not None
                or geometry.dha_angle
                is not None
            )
            else HBOND_MODE_INFERRED
        )

    else:
        normalized_mode = (
            validate_hydrogen_bond_mode(
                mode
            )
        )

    if normalized_mode == HBOND_MODE_EXPLICIT:
        return evaluate_explicit_hbond_geometry(
            geometry,
            donor_acceptor_cutoff=(
                donor_acceptor_cutoff
            ),
            hydrogen_acceptor_cutoff=(
                hydrogen_acceptor_cutoff
            ),
            minimum_dha_angle=(
                minimum_dha_angle
            ),
            distance_tolerance=(
                distance_tolerance
            ),
            angle_tolerance=(
                angle_tolerance
            ),
        )

    return evaluate_inferred_hbond_geometry(
        geometry,
        donor_acceptor_cutoff=(
            donor_acceptor_cutoff
        ),
        maximum_donor_deviation=(
            maximum_donor_deviation
        ),
        maximum_acceptor_deviation=(
            maximum_acceptor_deviation
        ),
        minimum_inferred_angle=(
            minimum_inferred_angle
        ),
        distance_tolerance=(
            distance_tolerance
        ),
        angle_tolerance=(
            angle_tolerance
        ),
        require_acceptor_angle=(
            require_acceptor_angle
        ),
    )


# -----------------------------------------------------------------------------
# Batch geometry calculations
# -----------------------------------------------------------------------------

def calculate_explicit_geometries_for_acceptors(
    assignment: DonorHydrogenAssignment,
    acceptors: Iterable[
        AtomLike
    ],
    *,
    donor_acceptor_cutoff: Optional[
        Number
    ] = None,
    calculate_acceptor_angle: bool = True,
    acceptor_vector_method: str = (
        DEFAULT_INFERRED_ACCEPTOR_VECTOR_METHOD
    ),
    bond_resolver: Optional[
        BondResolver
    ] = None,
) -> Tuple[
    Tuple[
        AtomLike,
        HydrogenBondGeometry,
    ],
    ...,
]:
    """
    Calculate explicit geometry against multiple acceptors.

    Parameters
    ----------
    assignment : DonorHydrogenAssignment
        Donor-hydrogen assignment.
    acceptors : iterable of atom-like
        Candidate acceptors.
    donor_acceptor_cutoff : Number or None, optional
        Optional prefiltering D...A cutoff.
    calculate_acceptor_angle : bool, optional
        Whether acceptor angles should be calculated.
    acceptor_vector_method : str, optional
        Acceptor-vector construction method.
    bond_resolver : callable or None, optional
        Custom bonded-neighbor resolver.

    Returns
    -------
    tuple
        ``(acceptor, geometry)`` pairs.
    """

    if not isinstance(
        assignment,
        DonorHydrogenAssignment,
    ):
        raise TypeError(
            "assignment must be a DonorHydrogenAssignment instance."
        )

    validated_acceptors = (
        validate_atom_collection(
            acceptors,
            allow_empty=True,
            require_coordinate=True,
        )
    )

    normalized_cutoff: Optional[
        np.float64
    ] = None

    if donor_acceptor_cutoff is not None:
        normalized_cutoff = (
            _optional_float64(
                donor_acceptor_cutoff,
                name="donor-acceptor cutoff",
                minimum=0.0,
            )
        )

    results: List[
        Tuple[
            AtomLike,
            HydrogenBondGeometry,
        ]
    ] = []

    for acceptor in validated_acceptors:
        if (
            acceptor is assignment.donor
            or acceptor is assignment.hydrogen
        ):
            continue

        if normalized_cutoff is not None:
            try:
                distance = (
                    calculate_donor_acceptor_distance(
                        assignment.donor,
                        acceptor,
                    )
                )

            except Exception:
                continue

            if distance > normalized_cutoff:
                continue

        try:
            geometry = (
                calculate_assignment_hbond_geometry(
                    assignment,
                    acceptor,
                    calculate_acceptor_angle=(
                        calculate_acceptor_angle
                    ),
                    acceptor_vector_method=(
                        acceptor_vector_method
                    ),
                    bond_resolver=bond_resolver,
                )
            )

        except Exception:
            continue

        results.append(
            (
                acceptor,
                geometry,
            )
        )

    return tuple(
        results
    )


def calculate_inferred_geometries_for_acceptors(
    donor: AtomLike,
    acceptors: Iterable[
        AtomLike
    ],
    *,
    donor_acceptor_cutoff: Optional[
        Number
    ] = None,
    donor_vector_method: str = (
        DEFAULT_INFERRED_DONOR_VECTOR_METHOD
    ),
    acceptor_vector_method: str = (
        DEFAULT_INFERRED_ACCEPTOR_VECTOR_METHOD
    ),
    calculate_acceptor_angle: bool = True,
    bond_resolver: Optional[
        BondResolver
    ] = None,
) -> Tuple[
    Tuple[
        AtomLike,
        HydrogenBondGeometry,
    ],
    ...,
]:
    """
    Calculate inferred geometry against multiple acceptors.

    Parameters
    ----------
    donor : atom-like
        Donor atom.
    acceptors : iterable of atom-like
        Candidate acceptors.
    donor_acceptor_cutoff : Number or None, optional
        Optional D...A prefilter.
    donor_vector_method : str, optional
        Donor-vector construction method.
    acceptor_vector_method : str, optional
        Acceptor-vector construction method.
    calculate_acceptor_angle : bool, optional
        Whether acceptor angles should be calculated.
    bond_resolver : callable or None, optional
        Custom bonded-neighbor resolver.

    Returns
    -------
    tuple
        ``(acceptor, geometry)`` pairs.
    """

    validate_atom(
        donor,
        require_coordinate=True,
    )

    validated_acceptors = (
        validate_atom_collection(
            acceptors,
            allow_empty=True,
            require_coordinate=True,
        )
    )

    normalized_cutoff: Optional[
        np.float64
    ] = None

    if donor_acceptor_cutoff is not None:
        normalized_cutoff = (
            _optional_float64(
                donor_acceptor_cutoff,
                name="donor-acceptor cutoff",
                minimum=0.0,
            )
        )

    results: List[
        Tuple[
            AtomLike,
            HydrogenBondGeometry,
        ]
    ] = []

    for acceptor in validated_acceptors:
        if acceptor is donor:
            continue

        if normalized_cutoff is not None:
            try:
                distance = (
                    calculate_donor_acceptor_distance(
                        donor,
                        acceptor,
                    )
                )

            except Exception:
                continue

            if distance > normalized_cutoff:
                continue

        try:
            geometry = (
                calculate_inferred_hbond_geometry(
                    donor,
                    acceptor,
                    donor_vector_method=(
                        donor_vector_method
                    ),
                    acceptor_vector_method=(
                        acceptor_vector_method
                    ),
                    calculate_acceptor_angle=(
                        calculate_acceptor_angle
                    ),
                    bond_resolver=bond_resolver,
                )
            )

        except Exception:
            continue

        results.append(
            (
                acceptor,
                geometry,
            )
        )

    return tuple(
        results
    )


# -----------------------------------------------------------------------------
# Convenience geometry predicates
# -----------------------------------------------------------------------------

def explicit_hbond_geometry_is_valid(
    donor: AtomLike,
    hydrogen: AtomLike,
    acceptor: AtomLike,
    *,
    donor_acceptor_cutoff: Number = (
        DEFAULT_DONOR_ACCEPTOR_DISTANCE
    ),
    hydrogen_acceptor_cutoff: Number = (
        DEFAULT_HYDROGEN_ACCEPTOR_DISTANCE
    ),
    minimum_dha_angle: Number = (
        DEFAULT_MINIMUM_DHA_ANGLE
    ),
    distance_tolerance: Number = (
        DEFAULT_DISTANCE_TOLERANCE
    ),
    angle_tolerance: Number = (
        DEFAULT_ANGLE_TOLERANCE
    ),
    bond_resolver: Optional[
        BondResolver
    ] = None,
) -> bool:
    """
    Determine whether an explicit D-H...A arrangement is geometrically valid.

    Parameters
    ----------
    donor : atom-like
        Donor atom.
    hydrogen : atom-like
        Explicit hydrogen.
    acceptor : atom-like
        Acceptor atom.
    donor_acceptor_cutoff : Number, optional
        Maximum D...A distance.
    hydrogen_acceptor_cutoff : Number, optional
        Maximum H...A distance.
    minimum_dha_angle : Number, optional
        Minimum D-H...A angle.
    distance_tolerance : Number, optional
        Distance tolerance.
    angle_tolerance : Number, optional
        Angular tolerance.
    bond_resolver : callable or None, optional
        Custom bonded-neighbor resolver.

    Returns
    -------
    bool
        Geometric validity.
    """

    try:
        geometry = calculate_explicit_hbond_geometry(
            donor,
            hydrogen,
            acceptor,
            calculate_acceptor_angle=False,
            bond_resolver=bond_resolver,
        )

        evaluation = evaluate_explicit_hbond_geometry(
            geometry,
            donor_acceptor_cutoff=(
                donor_acceptor_cutoff
            ),
            hydrogen_acceptor_cutoff=(
                hydrogen_acceptor_cutoff
            ),
            minimum_dha_angle=(
                minimum_dha_angle
            ),
            distance_tolerance=(
                distance_tolerance
            ),
            angle_tolerance=(
                angle_tolerance
            ),
        )

    except Exception:
        return False

    return evaluation.is_valid


def inferred_hbond_geometry_is_valid(
    donor: AtomLike,
    acceptor: AtomLike,
    *,
    donor_acceptor_cutoff: Number = (
        DEFAULT_DONOR_ACCEPTOR_DISTANCE
    ),
    minimum_inferred_angle: Number = (
        DEFAULT_MINIMUM_INFERRED_ANGLE
    ),
    donor_vector_method: str = (
        DEFAULT_INFERRED_DONOR_VECTOR_METHOD
    ),
    require_donor_angle: bool = True,
    bond_resolver: Optional[
        BondResolver
    ] = None,
) -> bool:
    """
    Determine whether an inferred D...A arrangement is geometrically valid.

    Parameters
    ----------
    donor : atom-like
        Donor atom.
    acceptor : atom-like
        Acceptor atom.
    donor_acceptor_cutoff : Number, optional
        Maximum D...A distance.
    minimum_inferred_angle : Number, optional
        Minimum conventional inferred angle.
    donor_vector_method : str, optional
        Donor-vector construction method.
    require_donor_angle : bool, optional
        Whether a donor direction must be available.
    bond_resolver : callable or None, optional
        Custom bonded-neighbor resolver.

    Returns
    -------
    bool
        Geometric validity.
    """

    try:
        geometry = calculate_inferred_hbond_geometry(
            donor,
            acceptor,
            donor_vector_method=(
                donor_vector_method
            ),
            calculate_acceptor_angle=False,
            bond_resolver=bond_resolver,
        )

        evaluation = evaluate_inferred_hbond_geometry(
            geometry,
            donor_acceptor_cutoff=(
                donor_acceptor_cutoff
            ),
            minimum_inferred_angle=(
                minimum_inferred_angle
            ),
            require_donor_angle=(
                require_donor_angle
            ),
            require_acceptor_angle=False,
        )

    except Exception:
        return False

    return evaluation.is_valid


# -----------------------------------------------------------------------------
# Public interface
# -----------------------------------------------------------------------------

_SECTION_6_PUBLIC_NAMES: Final[
    Tuple[
        str,
        ...,
    ]
] = (
    "DEFAULT_REQUIRE_DONOR_ACCEPTOR_DISTANCE",
    "DEFAULT_REQUIRE_HYDROGEN_ACCEPTOR_DISTANCE",
    "DEFAULT_REQUIRE_DHA_ANGLE",
    "DEFAULT_REQUIRE_INFERRED_ANGLE",
    "DEFAULT_INFERRED_DONOR_VECTOR_METHOD",
    "DEFAULT_INFERRED_ACCEPTOR_VECTOR_METHOD",
    "VALID_INFERRED_VECTOR_METHODS",
    "HydrogenBondGeometryEvaluation",
    "calculate_hbond_distance",
    "calculate_vector_angle",
    "calculate_three_atom_angle",
    "calculate_donor_acceptor_distance",
    "calculate_hydrogen_acceptor_distance",
    "calculate_donor_hydrogen_distance",
    "calculate_dha_angle",
    "validate_inferred_vector_method",
    "infer_open_valence_vector",
    "calculate_inferred_donor_angle",
    "calculate_inferred_acceptor_angle",
    "calculate_explicit_hbond_geometry",
    "calculate_assignment_hbond_geometry",
    "calculate_inferred_hbond_geometry",
    "evaluate_explicit_hbond_geometry",
    "evaluate_inferred_hbond_geometry",
    "evaluate_hbond_geometry",
    "calculate_explicit_geometries_for_acceptors",
    "calculate_inferred_geometries_for_acceptors",
    "explicit_hbond_geometry_is_valid",
    "inferred_hbond_geometry_is_valid",
)

for public_name in _SECTION_6_PUBLIC_NAMES:
    if public_name not in __all__:
        __all__.append(
            public_name
        )


# =============================================================================
# End of Section 6
# =============================================================================


# =============================================================================
# Section 7 — Hydrogen-bond detection
# =============================================================================


# -----------------------------------------------------------------------------
# Detection constants
# -----------------------------------------------------------------------------

DEFAULT_HBOND_ANALYSIS_MODE: Final[
    HydrogenBondMode
] = HBOND_MODE_EXPLICIT

DEFAULT_ALLOW_INFERRED_FALLBACK: Final[
    bool
] = True

DEFAULT_INCLUDE_AMBIGUOUS_HYDROGEN_ASSIGNMENTS: Final[
    bool
] = True

DEFAULT_REQUIRE_ACCEPTOR_ANGLE: Final[
    bool
] = False

DEFAULT_ALLOW_INTRAMOLECULAR_HBONDS: Final[
    bool
] = False

DEFAULT_DEDUPLICATE_HYDROGEN_BONDS: Final[
    bool
] = True

DEFAULT_STOP_AFTER_FIRST_VALID_HYDROGEN: Final[
    bool
] = False

DEFAULT_INCLUDE_UNKNOWN_DIRECTION: Final[
    bool
] = False


# -----------------------------------------------------------------------------
# Detection helper aliases
# -----------------------------------------------------------------------------

HydrogenBondPairKey = Tuple[
    int,
    int,
    Optional[
        int
    ],
]

HydrogenBondDetectionRecord = Tuple[
    AtomLike,
    Optional[
        AtomLike
    ],
    AtomLike,
    HydrogenBondGeometryEvaluation,
]


# -----------------------------------------------------------------------------
# Collection and identity helpers
# -----------------------------------------------------------------------------

def _normalize_hbond_atom_collection(
    atoms: Iterable[
        AtomLike
    ],
    *,
    name: str,
    require_coordinate: bool = True,
) -> Tuple[
    AtomLike,
    ...,
]:
    """
    Validate and normalize an atom collection for hydrogen-bond detection.

    Parameters
    ----------
    atoms : iterable of atom-like
        Atom collection.
    name : str
        Human-readable collection name.
    require_coordinate : bool, optional
        Whether valid coordinates are mandatory.

    Returns
    -------
    tuple of atom-like
        Validated atom collection.

    Raises
    ------
    TypeError
        If the collection is invalid.
    ValueError
        If an atom lacks required information.
    """

    try:
        normalized_atoms = validate_atom_collection(
            atoms,
            allow_empty=True,
            require_coordinate=require_coordinate,
        )

    except Exception as error:
        raise ValueError(
            f"Could not validate the {name} atom collection."
        ) from error

    return tuple(
        normalized_atoms
    )


def _build_atom_identity_index(
    atoms: Sequence[
        AtomLike
    ],
) -> Dict[
    int,
    int,
]:
    """
    Build an identity-based atom index.

    Parameters
    ----------
    atoms : sequence of atom-like
        Atom collection.

    Returns
    -------
    dict
        Mapping from ``id(atom)`` to collection index.
    """

    return {
        id(
            atom
        ): index
        for index, atom
        in enumerate(
            atoms
        )
    }


def _collections_share_atoms(
    atoms_1: Sequence[
        AtomLike
    ],
    atoms_2: Sequence[
        AtomLike
    ],
) -> bool:
    """
    Determine whether two atom collections share object identities.

    Parameters
    ----------
    atoms_1 : sequence of atom-like
        First collection.
    atoms_2 : sequence of atom-like
        Second collection.

    Returns
    -------
    bool
        Shared-atom status.
    """

    atom_ids_1 = {
        id(
            atom
        )
        for atom in atoms_1
    }

    return any(
        id(
            atom
        ) in atom_ids_1
        for atom in atoms_2
    )


def _atoms_belong_to_same_structure(
    atom_1: AtomLike,
    atom_2: AtomLike,
) -> bool:
    """
    Determine whether two atoms belong to the same structure.

    Parameters
    ----------
    atom_1 : atom-like
        First atom.
    atom_2 : atom-like
        Second atom.

    Returns
    -------
    bool
        Same-structure status.

    Notes
    -----
    When structural ownership cannot be resolved, object identity is used as
    the conservative fallback.
    """

    if atom_1 is atom_2:
        return True

    structure_1 = _get_chemical_object_value(
        atom_1,
        (
            "structure",
            "molecule",
            "model",
            "parent",
        ),
        default=None,
    )

    structure_2 = _get_chemical_object_value(
        atom_2,
        (
            "structure",
            "molecule",
            "model",
            "parent",
        ),
        default=None,
    )

    if (
        structure_1 is not None
        and structure_2 is not None
    ):
        return structure_1 is structure_2

    residue_1 = None
    residue_2 = None

    try:
        residue_1 = get_atom_residue(
            atom_1
        )

    except Exception:
        pass

    try:
        residue_2 = get_atom_residue(
            atom_2
        )

    except Exception:
        pass

    if (
        residue_1 is not None
        and residue_2 is not None
    ):
        residue_structure_1 = (
            _get_chemical_object_value(
                residue_1,
                (
                    "structure",
                    "molecule",
                    "model",
                    "parent",
                ),
                default=None,
            )
        )

        residue_structure_2 = (
            _get_chemical_object_value(
                residue_2,
                (
                    "structure",
                    "molecule",
                    "model",
                    "parent",
                ),
                default=None,
            )
        )

        if (
            residue_structure_1 is not None
            and residue_structure_2 is not None
        ):
            return (
                residue_structure_1
                is residue_structure_2
            )

    return False


# -----------------------------------------------------------------------------
# Donor-acceptor candidate prefiltering
# -----------------------------------------------------------------------------

def _coordinates_from_atoms(
    atoms: Sequence[
        AtomLike
    ],
) -> FloatArray:
    """
    Extract coordinates from an atom collection.

    Parameters
    ----------
    atoms : sequence of atom-like
        Atoms with valid coordinates.

    Returns
    -------
    numpy.ndarray
        Array with shape ``(n_atoms, 3)``.
    """

    if not atoms:
        return np.empty(
            (
                0,
                3,
            ),
            dtype=np.float64,
        )

    coordinates = np.empty(
        (
            len(
                atoms
            ),
            3,
        ),
        dtype=np.float64,
    )

    for index, atom in enumerate(
        atoms
    ):
        coordinates[
            index
        ] = _get_hbond_atom_coordinate(
            atom,
            name=f"atom {index}",
        )

    return coordinates


def _iter_candidate_pair_indices(
    donor_coordinates: FloatArray,
    acceptor_coordinates: FloatArray,
    *,
    maximum_distance: Number,
    block_size: int,
) -> Iterator[
    Tuple[
        int,
        int,
        np.float64,
    ]
]:
    """
    Yield donor-acceptor index pairs within a distance cutoff.

    Parameters
    ----------
    donor_coordinates : numpy.ndarray
        Donor coordinates with shape ``(n_donors, 3)``.
    acceptor_coordinates : numpy.ndarray
        Acceptor coordinates with shape ``(n_acceptors, 3)``.
    maximum_distance : Number
        Maximum D...A distance.
    block_size : int
        Donor processing block size.

    Yields
    ------
    tuple
        ``(donor_index, acceptor_index, distance)``.
    """

    normalized_cutoff = _optional_float64(
        maximum_distance,
        name="maximum donor-acceptor distance",
        minimum=0.0,
    )

    if normalized_cutoff is None:
        return

    normalized_block_size = _coerce_positive_integer(
        block_size,
        name="hydrogen-bond block size",
        default=DEFAULT_HBOND_BLOCK_SIZE,
    )

    if (
        donor_coordinates.size == 0
        or acceptor_coordinates.size == 0
    ):
        return

    squared_cutoff = np.float64(
        normalized_cutoff
        * normalized_cutoff
    )

    donor_count = donor_coordinates.shape[
        0
    ]

    for block_start in range(
        0,
        donor_count,
        normalized_block_size,
    ):
        block_end = min(
            donor_count,
            block_start
            + normalized_block_size,
        )

        donor_block = donor_coordinates[
            block_start:block_end
        ]

        differences = (
            donor_block[
                :,
                np.newaxis,
                :,
            ]
            - acceptor_coordinates[
                np.newaxis,
                :,
                :,
            ]
        )

        squared_distances = np.einsum(
            "ijk,ijk->ij",
            differences,
            differences,
            dtype=np.float64,
        )

        local_donor_indices, acceptor_indices = (
            np.nonzero(
                squared_distances
                <= squared_cutoff
            )
        )

        for local_donor_index, acceptor_index in zip(
            local_donor_indices,
            acceptor_indices,
        ):
            squared_distance = squared_distances[
                local_donor_index,
                acceptor_index,
            ]

            if squared_distance < 0.0:
                squared_distance = np.float64(
                    0.0
                )

            yield (
                block_start
                + int(
                    local_donor_index
                ),
                int(
                    acceptor_index
                ),
                np.float64(
                    np.sqrt(
                        squared_distance
                    )
                ),
            )


def find_hbond_candidate_pairs(
    donors: Iterable[
        AtomLike
    ],
    acceptors: Iterable[
        AtomLike
    ],
    *,
    maximum_distance: Number = (
        DEFAULT_DONOR_ACCEPTOR_DISTANCE
    ),
    distance_tolerance: Number = (
        DEFAULT_DISTANCE_TOLERANCE
    ),
    block_size: Optional[
        int
    ] = None,
    allow_same_atom: bool = False,
    allow_intramolecular: bool = (
        DEFAULT_ALLOW_INTRAMOLECULAR_HBONDS
    ),
) -> Tuple[
    Tuple[
        AtomLike,
        AtomLike,
        np.float64,
    ],
    ...,
]:
    """
    Find donor-acceptor pairs passing the D...A distance prefilter.

    Parameters
    ----------
    donors : iterable of atom-like
        Candidate donor atoms.
    acceptors : iterable of atom-like
        Candidate acceptor atoms.
    maximum_distance : Number, optional
        Maximum donor-acceptor distance.
    distance_tolerance : Number, optional
        Additional distance tolerance.
    block_size : int or None, optional
        Number of donors processed per vectorized block.
    allow_same_atom : bool, optional
        Whether a donor may be paired with itself.
    allow_intramolecular : bool, optional
        Whether atoms belonging to the same structure may be paired.

    Returns
    -------
    tuple
        ``(donor, acceptor, distance)`` entries.
    """

    normalized_donors = (
        _normalize_hbond_atom_collection(
            donors,
            name="donor",
            require_coordinate=True,
        )
    )

    normalized_acceptors = (
        _normalize_hbond_atom_collection(
            acceptors,
            name="acceptor",
            require_coordinate=True,
        )
    )

    if (
        not normalized_donors
        or not normalized_acceptors
    ):
        return ()

    normalized_distance = _optional_float64(
        maximum_distance,
        name="maximum donor-acceptor distance",
        minimum=0.0,
    )

    normalized_tolerance = _optional_float64(
        distance_tolerance,
        name="distance tolerance",
        minimum=0.0,
    )

    if normalized_distance is None:
        raise ValueError(
            "Maximum donor-acceptor distance cannot be None."
        )

    if normalized_tolerance is None:
        normalized_tolerance = np.float64(
            0.0
        )

    effective_cutoff = np.float64(
        normalized_distance
        + normalized_tolerance
    )

    effective_block_size = (
        get_default_hbond_block_size()
        if block_size is None
        else _coerce_positive_integer(
            block_size,
            name="hydrogen-bond block size",
            default=DEFAULT_HBOND_BLOCK_SIZE,
        )
    )

    donor_coordinates = _coordinates_from_atoms(
        normalized_donors
    )

    acceptor_coordinates = _coordinates_from_atoms(
        normalized_acceptors
    )

    candidate_pairs: List[
        Tuple[
            AtomLike,
            AtomLike,
            np.float64,
        ]
    ] = []

    for (
        donor_index,
        acceptor_index,
        distance,
    ) in _iter_candidate_pair_indices(
        donor_coordinates,
        acceptor_coordinates,
        maximum_distance=effective_cutoff,
        block_size=effective_block_size,
    ):
        donor = normalized_donors[
            donor_index
        ]

        acceptor = normalized_acceptors[
            acceptor_index
        ]

        if (
            not allow_same_atom
            and donor is acceptor
        ):
            continue

        if (
            not allow_intramolecular
            and _atoms_belong_to_same_structure(
                donor,
                acceptor,
            )
        ):
            continue

        candidate_pairs.append(
            (
                donor,
                acceptor,
                distance,
            )
        )

    return tuple(
        candidate_pairs
    )


# -----------------------------------------------------------------------------
# Explicit hydrogen-bond detection
# -----------------------------------------------------------------------------

def detect_explicit_hydrogen_bonds(
    donors: Iterable[
        AtomLike
    ],
    acceptors: Iterable[
        AtomLike
    ],
    assignments: Iterable[
        DonorHydrogenAssignment
    ],
    *,
    direction: HydrogenBondDirection = (
        HBOND_DIRECTION_UNKNOWN
    ),
    donor_indices: Optional[
        Mapping[
            int,
            int,
        ]
    ] = None,
    acceptor_indices: Optional[
        Mapping[
            int,
            int,
        ]
    ] = None,
    donor_acceptor_cutoff: Number = (
        DEFAULT_DONOR_ACCEPTOR_DISTANCE
    ),
    hydrogen_acceptor_cutoff: Number = (
        DEFAULT_HYDROGEN_ACCEPTOR_DISTANCE
    ),
    minimum_dha_angle: Number = (
        DEFAULT_MINIMUM_DHA_ANGLE
    ),
    distance_tolerance: Number = (
        DEFAULT_DISTANCE_TOLERANCE
    ),
    angle_tolerance: Number = (
        DEFAULT_ANGLE_TOLERANCE
    ),
    calculate_acceptor_angle: bool = False,
    require_acceptor_angle: bool = False,
    acceptor_vector_method: str = (
        DEFAULT_INFERRED_ACCEPTOR_VECTOR_METHOD
    ),
    bond_resolver: Optional[
        BondResolver
    ] = None,
    allow_intramolecular: bool = (
        DEFAULT_ALLOW_INTRAMOLECULAR_HBONDS
    ),
    stop_after_first_valid_hydrogen: bool = (
        DEFAULT_STOP_AFTER_FIRST_VALID_HYDROGEN
    ),
    maximum_hydrogen_bonds: Optional[
        int
    ] = DEFAULT_MAXIMUM_HYDROGEN_BONDS,
) -> Tuple[
    HydrogenBond,
    ...,
]:
    """
    Detect hydrogen bonds using explicit donor-bound hydrogens.

    Parameters
    ----------
    donors : iterable of atom-like
        Donor atoms.
    acceptors : iterable of atom-like
        Acceptor atoms.
    assignments : iterable of DonorHydrogenAssignment
        Donor-hydrogen assignments.
    direction : HydrogenBondDirection, optional
        Interaction direction.
    donor_indices : mapping or None, optional
        Mapping from ``id(donor)`` to donor index.
    acceptor_indices : mapping or None, optional
        Mapping from ``id(acceptor)`` to acceptor index.
    donor_acceptor_cutoff : Number, optional
        Maximum D...A distance.
    hydrogen_acceptor_cutoff : Number, optional
        Maximum H...A distance.
    minimum_dha_angle : Number, optional
        Minimum D-H...A angle.
    distance_tolerance : Number, optional
        Distance tolerance.
    angle_tolerance : Number, optional
        Angular tolerance.
    calculate_acceptor_angle : bool, optional
        Whether to calculate the auxiliary acceptor angle.
    require_acceptor_angle : bool, optional
        Whether a valid acceptor angle is mandatory.
    acceptor_vector_method : str, optional
        Acceptor open-valence vector method.
    bond_resolver : callable or None, optional
        Custom bond resolver.
    allow_intramolecular : bool, optional
        Whether intramolecular interactions are allowed.
    stop_after_first_valid_hydrogen : bool, optional
        Whether only the first valid hydrogen per donor-acceptor pair is kept.
    maximum_hydrogen_bonds : int or None, optional
        Optional result limit.

    Returns
    -------
    tuple of HydrogenBond
        Detected explicit hydrogen bonds.
    """

    normalized_direction = (
        validate_hydrogen_bond_direction(
            direction
        )
    )

    normalized_donors = (
        _normalize_hbond_atom_collection(
            donors,
            name="donor",
            require_coordinate=True,
        )
    )

    normalized_acceptors = (
        _normalize_hbond_atom_collection(
            acceptors,
            name="acceptor",
            require_coordinate=True,
        )
    )

    normalized_assignments = tuple(
        assignments
    )

    for index, assignment in enumerate(
        normalized_assignments
    ):
        if not isinstance(
            assignment,
            DonorHydrogenAssignment,
        ):
            raise TypeError(
                "All assignments must be DonorHydrogenAssignment "
                f"instances. Invalid entry at index {index}."
            )

    assignments_by_donor_id: Dict[
        int,
        List[
            DonorHydrogenAssignment
        ],
    ] = {}

    for assignment in normalized_assignments:
        assignments_by_donor_id.setdefault(
            id(
                assignment.donor
            ),
            [],
        ).append(
            assignment
        )

    candidate_pairs = find_hbond_candidate_pairs(
        normalized_donors,
        normalized_acceptors,
        maximum_distance=donor_acceptor_cutoff,
        distance_tolerance=distance_tolerance,
        allow_intramolecular=(
            allow_intramolecular
        ),
    )

    detected_bonds: List[
        HydrogenBond
    ] = []

    for (
        donor,
        acceptor,
        prefilter_distance,
    ) in candidate_pairs:
        donor_assignments = (
            assignments_by_donor_id.get(
                id(
                    donor
                ),
                [],
            )
        )

        if not donor_assignments:
            continue

        for assignment in donor_assignments:
            hydrogen = assignment.hydrogen

            if (
                hydrogen is acceptor
                or hydrogen is donor
            ):
                continue

            try:
                geometry = (
                    calculate_assignment_hbond_geometry(
                        assignment,
                        acceptor,
                        calculate_acceptor_angle=(
                            calculate_acceptor_angle
                            or require_acceptor_angle
                        ),
                        acceptor_vector_method=(
                            acceptor_vector_method
                        ),
                        bond_resolver=bond_resolver,
                        metadata={
                            "prefilter_distance": float(
                                prefilter_distance
                            ),
                        },
                    )
                )

                evaluation = (
                    evaluate_explicit_hbond_geometry(
                        geometry,
                        donor_acceptor_cutoff=(
                            donor_acceptor_cutoff
                        ),
                        hydrogen_acceptor_cutoff=(
                            hydrogen_acceptor_cutoff
                        ),
                        minimum_dha_angle=(
                            minimum_dha_angle
                        ),
                        distance_tolerance=(
                            distance_tolerance
                        ),
                        angle_tolerance=(
                            angle_tolerance
                        ),
                    )
                )

            except Exception:
                continue

            if not evaluation.is_valid:
                continue

            if require_acceptor_angle:
                if geometry.acceptor_angle is None:
                    continue

                maximum_acceptor_deviation = (
                    np.float64(
                        STRAIGHT_ANGLE_DEGREES
                        - np.float64(
                            DEFAULT_MINIMUM_INFERRED_ANGLE
                        )
                    )
                )

                if not _deviation_angle_within_cutoff(
                    geometry.acceptor_angle,
                    maximum_acceptor_deviation,
                    tolerance=angle_tolerance,
                ):
                    continue

            donor_index = (
                assignment.donor_index
            )

            if (
                donor_index is None
                and donor_indices is not None
            ):
                donor_index = donor_indices.get(
                    id(
                        donor
                    )
                )

            hydrogen_index = (
                assignment.hydrogen_index
            )

            acceptor_index = None

            if acceptor_indices is not None:
                acceptor_index = (
                    acceptor_indices.get(
                        id(
                            acceptor
                        )
                    )
                )

            detected_bonds.append(
                HydrogenBond(
                    donor=donor,
                    hydrogen=hydrogen,
                    acceptor=acceptor,
                    geometry=geometry,
                    mode=HBOND_MODE_EXPLICIT,
                    direction=normalized_direction,
                    classification=(
                        HBOND_TYPE_UNKNOWN
                    ),
                    donor_index=donor_index,
                    hydrogen_index=(
                        hydrogen_index
                    ),
                    acceptor_index=(
                        acceptor_index
                    ),
                    metadata={
                        "geometry_valid": True,
                        "failed_criteria": (),
                        "assignment_method": (
                            assignment
                            .assignment_method
                        ),
                        "assignment_ambiguous": (
                            assignment
                            .is_ambiguous
                        ),
                    },
                )
            )

            if (
                maximum_hydrogen_bonds
                is not None
                and len(
                    detected_bonds
                )
                >= maximum_hydrogen_bonds
            ):
                return tuple(
                    detected_bonds
                )

            if stop_after_first_valid_hydrogen:
                break

    return tuple(
        detected_bonds
    )


# -----------------------------------------------------------------------------
# Inferred hydrogen-bond detection
# -----------------------------------------------------------------------------

def detect_inferred_hydrogen_bonds(
    donors: Iterable[
        AtomLike
    ],
    acceptors: Iterable[
        AtomLike
    ],
    *,
    direction: HydrogenBondDirection = (
        HBOND_DIRECTION_UNKNOWN
    ),
    donor_indices: Optional[
        Mapping[
            int,
            int,
        ]
    ] = None,
    acceptor_indices: Optional[
        Mapping[
            int,
            int,
        ]
    ] = None,
    donor_acceptor_cutoff: Number = (
        DEFAULT_DONOR_ACCEPTOR_DISTANCE
    ),
    minimum_inferred_angle: Number = (
        DEFAULT_MINIMUM_INFERRED_ANGLE
    ),
    maximum_donor_deviation: Optional[
        Number
    ] = None,
    maximum_acceptor_deviation: Optional[
        Number
    ] = None,
    distance_tolerance: Number = (
        DEFAULT_DISTANCE_TOLERANCE
    ),
    angle_tolerance: Number = (
        DEFAULT_ANGLE_TOLERANCE
    ),
    donor_vector_method: str = (
        DEFAULT_INFERRED_DONOR_VECTOR_METHOD
    ),
    acceptor_vector_method: str = (
        DEFAULT_INFERRED_ACCEPTOR_VECTOR_METHOD
    ),
    require_donor_angle: bool = True,
    require_acceptor_angle: bool = (
        DEFAULT_REQUIRE_ACCEPTOR_ANGLE
    ),
    bond_resolver: Optional[
        BondResolver
    ] = None,
    allow_intramolecular: bool = (
        DEFAULT_ALLOW_INTRAMOLECULAR_HBONDS
    ),
    excluded_donor_ids: Optional[
        AbstractSet[
            int
        ]
    ] = None,
    maximum_hydrogen_bonds: Optional[
        int
    ] = DEFAULT_MAXIMUM_HYDROGEN_BONDS,
) -> Tuple[
    HydrogenBond,
    ...,
]:
    """
    Detect hydrogen bonds without explicit hydrogens.

    Parameters
    ----------
    donors : iterable of atom-like
        Donor atoms.
    acceptors : iterable of atom-like
        Acceptor atoms.
    direction : HydrogenBondDirection, optional
        Interaction direction.
    donor_indices : mapping or None, optional
        Donor identity-to-index mapping.
    acceptor_indices : mapping or None, optional
        Acceptor identity-to-index mapping.
    donor_acceptor_cutoff : Number, optional
        Maximum D...A distance.
    minimum_inferred_angle : Number, optional
        Minimum conventional inferred angle.
    maximum_donor_deviation : Number or None, optional
        Maximum donor-vector deviation.
    maximum_acceptor_deviation : Number or None, optional
        Maximum acceptor-vector deviation.
    distance_tolerance : Number, optional
        Distance tolerance.
    angle_tolerance : Number, optional
        Angular tolerance.
    donor_vector_method : str, optional
        Donor open-valence vector method.
    acceptor_vector_method : str, optional
        Acceptor open-valence vector method.
    require_donor_angle : bool, optional
        Whether donor angular information is mandatory.
    require_acceptor_angle : bool, optional
        Whether acceptor angular information is mandatory.
    bond_resolver : callable or None, optional
        Custom bonded-neighbor resolver.
    allow_intramolecular : bool, optional
        Whether intramolecular interactions are allowed.
    excluded_donor_ids : set of int or None, optional
        Donor identities that should not use inferred detection.
    maximum_hydrogen_bonds : int or None, optional
        Optional result limit.

    Returns
    -------
    tuple of HydrogenBond
        Detected inferred hydrogen bonds.
    """

    normalized_direction = (
        validate_hydrogen_bond_direction(
            direction
        )
    )

    normalized_donors = (
        _normalize_hbond_atom_collection(
            donors,
            name="donor",
            require_coordinate=True,
        )
    )

    normalized_acceptors = (
        _normalize_hbond_atom_collection(
            acceptors,
            name="acceptor",
            require_coordinate=True,
        )
    )

    excluded_ids = (
        frozenset()
        if excluded_donor_ids is None
        else frozenset(
            int(
                identity
            )
            for identity in excluded_donor_ids
        )
    )

    filtered_donors = tuple(
        donor
        for donor in normalized_donors
        if id(
            donor
        ) not in excluded_ids
    )

    candidate_pairs = find_hbond_candidate_pairs(
        filtered_donors,
        normalized_acceptors,
        maximum_distance=donor_acceptor_cutoff,
        distance_tolerance=distance_tolerance,
        allow_intramolecular=(
            allow_intramolecular
        ),
    )

    detected_bonds: List[
        HydrogenBond
    ] = []

    for (
        donor,
        acceptor,
        prefilter_distance,
    ) in candidate_pairs:
        try:
            geometry = (
                calculate_inferred_hbond_geometry(
                    donor,
                    acceptor,
                    donor_vector_method=(
                        donor_vector_method
                    ),
                    acceptor_vector_method=(
                        acceptor_vector_method
                    ),
                    calculate_acceptor_angle=(
                        require_acceptor_angle
                    ),
                    bond_resolver=bond_resolver,
                    metadata={
                        "prefilter_distance": float(
                            prefilter_distance
                        ),
                    },
                )
            )

            evaluation = (
                evaluate_inferred_hbond_geometry(
                    geometry,
                    donor_acceptor_cutoff=(
                        donor_acceptor_cutoff
                    ),
                    maximum_donor_deviation=(
                        maximum_donor_deviation
                    ),
                    maximum_acceptor_deviation=(
                        maximum_acceptor_deviation
                    ),
                    minimum_inferred_angle=(
                        minimum_inferred_angle
                    ),
                    distance_tolerance=(
                        distance_tolerance
                    ),
                    angle_tolerance=(
                        angle_tolerance
                    ),
                    require_donor_angle=(
                        require_donor_angle
                    ),
                    require_acceptor_angle=(
                        require_acceptor_angle
                    ),
                )
            )

        except Exception:
            continue

        if not evaluation.is_valid:
            continue

        donor_index = None
        acceptor_index = None

        if donor_indices is not None:
            donor_index = donor_indices.get(
                id(
                    donor
                )
            )

        if acceptor_indices is not None:
            acceptor_index = acceptor_indices.get(
                id(
                    acceptor
                )
            )

        detected_bonds.append(
            HydrogenBond(
                donor=donor,
                hydrogen=None,
                acceptor=acceptor,
                geometry=geometry,
                mode=HBOND_MODE_INFERRED,
                direction=normalized_direction,
                classification=(
                    HBOND_TYPE_UNKNOWN
                ),
                donor_index=donor_index,
                hydrogen_index=None,
                acceptor_index=(
                    acceptor_index
                ),
                metadata={
                    "geometry_valid": True,
                    "failed_criteria": (),
                    "donor_angle_required": (
                        require_donor_angle
                    ),
                    "acceptor_angle_required": (
                        require_acceptor_angle
                    ),
                },
            )
        )

        if (
            maximum_hydrogen_bonds
            is not None
            and len(
                detected_bonds
            )
            >= maximum_hydrogen_bonds
        ):
            break

    return tuple(
        detected_bonds
    )


# -----------------------------------------------------------------------------
# Duplicate handling
# -----------------------------------------------------------------------------

def get_hydrogen_bond_pair_key(
    hydrogen_bond: HydrogenBond,
    *,
    include_hydrogen: bool = True,
) -> HydrogenBondPairKey:
    """
    Return an identity-based hydrogen-bond key.

    Parameters
    ----------
    hydrogen_bond : HydrogenBond
        Hydrogen bond.
    include_hydrogen : bool, optional
        Whether explicit hydrogen identity should distinguish interactions.

    Returns
    -------
    HydrogenBondPairKey
        Identity key.
    """

    if not isinstance(
        hydrogen_bond,
        HydrogenBond,
    ):
        raise TypeError(
            "hydrogen_bond must be a HydrogenBond instance."
        )

    hydrogen_identity: Optional[
        int
    ]

    if (
        include_hydrogen
        and hydrogen_bond.hydrogen is not None
    ):
        hydrogen_identity = id(
            hydrogen_bond.hydrogen
        )

    else:
        hydrogen_identity = None

    return (
        id(
            hydrogen_bond.donor
        ),
        id(
            hydrogen_bond.acceptor
        ),
        hydrogen_identity,
    )


def _hydrogen_bond_quality_key(
    hydrogen_bond: HydrogenBond,
) -> Tuple[
    int,
    np.float64,
    np.float64,
    np.float64,
]:
    """
    Return a sorting key that favors better hydrogen-bond geometry.

    Parameters
    ----------
    hydrogen_bond : HydrogenBond
        Hydrogen bond.

    Returns
    -------
    tuple
        Geometry quality key.

    Notes
    -----
    Explicit geometry is preferred over inferred geometry. Smaller distances
    and larger D-H...A angles are favored.
    """

    explicit_priority = (
        0
        if hydrogen_bond.is_explicit
        else 1
    )

    donor_acceptor_distance = np.float64(
        hydrogen_bond
        .donor_acceptor_distance
    )

    hydrogen_acceptor_distance = (
        hydrogen_bond
        .hydrogen_acceptor_distance
    )

    normalized_hydrogen_acceptor_distance = (
        np.float64(
            np.inf
        )
        if hydrogen_acceptor_distance is None
        else np.float64(
            hydrogen_acceptor_distance
        )
    )

    dha_angle = hydrogen_bond.dha_angle

    negative_dha_angle = (
        np.float64(
            0.0
        )
        if dha_angle is None
        else np.float64(
            -dha_angle
        )
    )

    return (
        explicit_priority,
        donor_acceptor_distance,
        normalized_hydrogen_acceptor_distance,
        negative_dha_angle,
    )


def deduplicate_hydrogen_bonds(
    hydrogen_bonds: Iterable[
        HydrogenBond
    ],
    *,
    include_hydrogen_in_key: bool = True,
    prefer_explicit: bool = True,
) -> Tuple[
    HydrogenBond,
    ...,
]:
    """
    Remove duplicate hydrogen bonds.

    Parameters
    ----------
    hydrogen_bonds : iterable of HydrogenBond
        Hydrogen bonds to deduplicate.
    include_hydrogen_in_key : bool, optional
        Whether different donor-bound hydrogens are retained separately.
    prefer_explicit : bool, optional
        Whether explicit geometry should replace inferred geometry for the
        same donor-acceptor pair.

    Returns
    -------
    tuple of HydrogenBond
        Deduplicated hydrogen bonds.
    """

    normalized_bonds = tuple(
        hydrogen_bonds
    )

    for index, hydrogen_bond in enumerate(
        normalized_bonds
    ):
        if not isinstance(
            hydrogen_bond,
            HydrogenBond,
        ):
            raise TypeError(
                "All entries must be HydrogenBond instances. "
                f"Invalid entry at index {index}."
            )

    selected: Dict[
        HydrogenBondPairKey,
        HydrogenBond,
    ] = {}

    insertion_order: List[
        HydrogenBondPairKey
    ] = []

    for hydrogen_bond in normalized_bonds:
        key = get_hydrogen_bond_pair_key(
            hydrogen_bond,
            include_hydrogen=(
                include_hydrogen_in_key
            ),
        )

        existing = selected.get(
            key
        )

        if existing is None:
            selected[
                key
            ] = hydrogen_bond

            insertion_order.append(
                key
            )

            continue

        if prefer_explicit:
            existing_quality = (
                _hydrogen_bond_quality_key(
                    existing
                )
            )

            candidate_quality = (
                _hydrogen_bond_quality_key(
                    hydrogen_bond
                )
            )

            if candidate_quality < existing_quality:
                selected[
                    key
                ] = hydrogen_bond

        elif (
            hydrogen_bond
            .donor_acceptor_distance
            < existing
            .donor_acceptor_distance
        ):
            selected[
                key
            ] = hydrogen_bond

    return tuple(
        selected[
            key
        ]
        for key in insertion_order
    )


# -----------------------------------------------------------------------------
# One-direction detection orchestration
# -----------------------------------------------------------------------------

def detect_directional_hydrogen_bonds(
    donor_atoms: Iterable[
        AtomLike
    ],
    acceptor_atoms: Iterable[
        AtomLike
    ],
    *,
    direction: HydrogenBondDirection,
    mode: HydrogenBondMode = (
        DEFAULT_HBOND_ANALYSIS_MODE
    ),
    allow_inferred_fallback: bool = (
        DEFAULT_ALLOW_INFERRED_FALLBACK
    ),
    require_explicit_hydrogen: bool = False,
    bond_resolver: Optional[
        BondResolver
    ] = None,
    donor_acceptor_cutoff: Optional[
        Number
    ] = None,
    hydrogen_acceptor_cutoff: Optional[
        Number
    ] = None,
    minimum_dha_angle: Optional[
        Number
    ] = None,
    minimum_inferred_angle: Optional[
        Number
    ] = None,
    distance_tolerance: Optional[
        Number
    ] = None,
    angle_tolerance: Optional[
        Number
    ] = None,
    require_inferred_donor_angle: bool = True,
    require_acceptor_angle: bool = (
        DEFAULT_REQUIRE_ACCEPTOR_ANGLE
    ),
    donor_vector_method: str = (
        DEFAULT_INFERRED_DONOR_VECTOR_METHOD
    ),
    acceptor_vector_method: str = (
        DEFAULT_INFERRED_ACCEPTOR_VECTOR_METHOD
    ),
    allow_distance_hydrogen_assignment: bool = (
        DEFAULT_ALLOW_DISTANCE_BASED_HYDROGEN_ASSIGNMENT
    ),
    include_ambiguous_assignments: bool = (
        DEFAULT_INCLUDE_AMBIGUOUS_HYDROGEN_ASSIGNMENTS
    ),
    allow_intramolecular: bool = (
        DEFAULT_ALLOW_INTRAMOLECULAR_HBONDS
    ),
    deduplicate: bool = (
        DEFAULT_DEDUPLICATE_HYDROGEN_BONDS
    ),
    maximum_hydrogen_bonds: Optional[
        int
    ] = DEFAULT_MAXIMUM_HYDROGEN_BONDS,
) -> Tuple[
    HydrogenBond,
    ...,
]:
    """
    Detect hydrogen bonds in one donor-to-acceptor direction.

    Parameters
    ----------
    donor_atoms : iterable of atom-like
        Atom collection containing possible donors.
    acceptor_atoms : iterable of atom-like
        Atom collection containing possible acceptors.
    direction : HydrogenBondDirection
        Ligand-donor or receptor-donor direction.
    mode : HydrogenBondMode, optional
        ``"explicit"`` or ``"inferred"``.
    allow_inferred_fallback : bool, optional
        In explicit mode, allow inferred detection for donors lacking assigned
        hydrogens.
    require_explicit_hydrogen : bool, optional
        Whether donor selection itself requires explicit hydrogen.
    bond_resolver : callable or None, optional
        Custom bond resolver.
    donor_acceptor_cutoff : Number or None, optional
        Maximum D...A distance.
    hydrogen_acceptor_cutoff : Number or None, optional
        Maximum H...A distance.
    minimum_dha_angle : Number or None, optional
        Minimum D-H...A angle.
    minimum_inferred_angle : Number or None, optional
        Minimum conventional inferred angle.
    distance_tolerance : Number or None, optional
        Distance tolerance.
    angle_tolerance : Number or None, optional
        Angular tolerance.
    require_inferred_donor_angle : bool, optional
        Whether inferred donor alignment is mandatory.
    require_acceptor_angle : bool, optional
        Whether acceptor alignment is mandatory.
    donor_vector_method : str, optional
        Donor inferred-vector method.
    acceptor_vector_method : str, optional
        Acceptor inferred-vector method.
    allow_distance_hydrogen_assignment : bool, optional
        Whether explicit hydrogens may be assigned by distance.
    include_ambiguous_assignments : bool, optional
        Whether ambiguous hydrogen assignments are allowed.
    allow_intramolecular : bool, optional
        Whether intramolecular interactions are allowed.
    deduplicate : bool, optional
        Whether duplicate interactions should be removed.
    maximum_hydrogen_bonds : int or None, optional
        Optional result limit.

    Returns
    -------
    tuple of HydrogenBond
        Directional hydrogen bonds.
    """

    normalized_mode = (
        validate_hydrogen_bond_mode(
            mode
        )
    )

    normalized_direction = (
        validate_hydrogen_bond_direction(
            direction
        )
    )

    all_donor_atoms = (
        _normalize_hbond_atom_collection(
            donor_atoms,
            name="source",
            require_coordinate=True,
        )
    )

    all_acceptor_atoms = (
        _normalize_hbond_atom_collection(
            acceptor_atoms,
            name="target",
            require_coordinate=True,
        )
    )

    if (
        not all_donor_atoms
        or not all_acceptor_atoms
    ):
        return ()

    donors = select_hbond_donors(
        all_donor_atoms,
        require_explicit_hydrogen=(
            require_explicit_hydrogen
        ),
        bond_resolver=bond_resolver,
    )

    acceptors = select_hbond_acceptors(
        all_acceptor_atoms,
        bond_resolver=bond_resolver,
    )

    if not donors or not acceptors:
        return ()

    donor_indices = _build_atom_identity_index(
        all_donor_atoms
    )

    acceptor_indices = (
        _build_atom_identity_index(
            all_acceptor_atoms
        )
    )

    resolved_da_cutoff = (
        get_default_donor_acceptor_distance()
        if donor_acceptor_cutoff is None
        else _coerce_positive_float(
            donor_acceptor_cutoff,
            name="donor-acceptor cutoff",
            default=(
                DEFAULT_DONOR_ACCEPTOR_DISTANCE
            ),
        )
    )

    resolved_ha_cutoff = (
        get_default_hydrogen_acceptor_distance()
        if hydrogen_acceptor_cutoff is None
        else _coerce_positive_float(
            hydrogen_acceptor_cutoff,
            name="hydrogen-acceptor cutoff",
            default=(
                DEFAULT_HYDROGEN_ACCEPTOR_DISTANCE
            ),
        )
    )

    resolved_dha_angle = (
        get_default_minimum_dha_angle()
        if minimum_dha_angle is None
        else _coerce_angle(
            minimum_dha_angle,
            name="minimum D-H-A angle",
            default=DEFAULT_MINIMUM_DHA_ANGLE,
        )
    )

    resolved_inferred_angle = (
        get_default_minimum_inferred_angle()
        if minimum_inferred_angle is None
        else _coerce_angle(
            minimum_inferred_angle,
            name="minimum inferred angle",
            default=(
                DEFAULT_MINIMUM_INFERRED_ANGLE
            ),
        )
    )

    resolved_distance_tolerance = (
        get_default_distance_tolerance()
        if distance_tolerance is None
        else _coerce_positive_float(
            distance_tolerance,
            name="distance tolerance",
            default=DEFAULT_DISTANCE_TOLERANCE,
            allow_zero=True,
        )
    )

    resolved_angle_tolerance = (
        get_default_angle_tolerance()
        if angle_tolerance is None
        else _coerce_positive_float(
            angle_tolerance,
            name="angle tolerance",
            default=DEFAULT_ANGLE_TOLERANCE,
            allow_zero=True,
        )
    )

    detected_bonds: List[
        HydrogenBond
    ] = []

    donor_ids_with_assignments: Set[
        int
    ] = set()

    if normalized_mode == HBOND_MODE_EXPLICIT:
        assignments = (
            identify_donor_hydrogen_assignments(
                all_donor_atoms,
                donors=donors,
                bond_resolver=bond_resolver,
                allow_distance_assignment=(
                    allow_distance_hydrogen_assignment
                ),
                distance_tolerance=(
                    DEFAULT_HYDROGEN_ASSIGNMENT_TOLERANCE
                ),
                include_ambiguous=(
                    include_ambiguous_assignments
                ),
            )
        )

        donor_ids_with_assignments = {
            id(
                assignment.donor
            )
            for assignment in assignments
        }

        explicit_bonds = (
            detect_explicit_hydrogen_bonds(
                donors,
                acceptors,
                assignments,
                direction=normalized_direction,
                donor_indices=donor_indices,
                acceptor_indices=(
                    acceptor_indices
                ),
                donor_acceptor_cutoff=(
                    resolved_da_cutoff
                ),
                hydrogen_acceptor_cutoff=(
                    resolved_ha_cutoff
                ),
                minimum_dha_angle=(
                    resolved_dha_angle
                ),
                distance_tolerance=(
                    resolved_distance_tolerance
                ),
                angle_tolerance=(
                    resolved_angle_tolerance
                ),
                calculate_acceptor_angle=(
                    require_acceptor_angle
                ),
                require_acceptor_angle=(
                    require_acceptor_angle
                ),
                acceptor_vector_method=(
                    acceptor_vector_method
                ),
                bond_resolver=bond_resolver,
                allow_intramolecular=(
                    allow_intramolecular
                ),
                maximum_hydrogen_bonds=(
                    maximum_hydrogen_bonds
                ),
            )
        )

        detected_bonds.extend(
            explicit_bonds
        )

        if (
            allow_inferred_fallback
            and (
                maximum_hydrogen_bonds is None
                or len(
                    detected_bonds
                )
                < maximum_hydrogen_bonds
            )
        ):
            remaining_limit = (
                None
                if maximum_hydrogen_bonds is None
                else max(
                    0,
                    maximum_hydrogen_bonds
                    - len(
                        detected_bonds
                    ),
                )
            )

            inferred_bonds = (
                detect_inferred_hydrogen_bonds(
                    donors,
                    acceptors,
                    direction=(
                        normalized_direction
                    ),
                    donor_indices=(
                        donor_indices
                    ),
                    acceptor_indices=(
                        acceptor_indices
                    ),
                    donor_acceptor_cutoff=(
                        resolved_da_cutoff
                    ),
                    minimum_inferred_angle=(
                        resolved_inferred_angle
                    ),
                    distance_tolerance=(
                        resolved_distance_tolerance
                    ),
                    angle_tolerance=(
                        resolved_angle_tolerance
                    ),
                    donor_vector_method=(
                        donor_vector_method
                    ),
                    acceptor_vector_method=(
                        acceptor_vector_method
                    ),
                    require_donor_angle=(
                        require_inferred_donor_angle
                    ),
                    require_acceptor_angle=(
                        require_acceptor_angle
                    ),
                    bond_resolver=bond_resolver,
                    allow_intramolecular=(
                        allow_intramolecular
                    ),
                    excluded_donor_ids=(
                        donor_ids_with_assignments
                    ),
                    maximum_hydrogen_bonds=(
                        remaining_limit
                    ),
                )
            )

            detected_bonds.extend(
                inferred_bonds
            )

    else:
        inferred_bonds = (
            detect_inferred_hydrogen_bonds(
                donors,
                acceptors,
                direction=normalized_direction,
                donor_indices=donor_indices,
                acceptor_indices=(
                    acceptor_indices
                ),
                donor_acceptor_cutoff=(
                    resolved_da_cutoff
                ),
                minimum_inferred_angle=(
                    resolved_inferred_angle
                ),
                distance_tolerance=(
                    resolved_distance_tolerance
                ),
                angle_tolerance=(
                    resolved_angle_tolerance
                ),
                donor_vector_method=(
                    donor_vector_method
                ),
                acceptor_vector_method=(
                    acceptor_vector_method
                ),
                require_donor_angle=(
                    require_inferred_donor_angle
                ),
                require_acceptor_angle=(
                    require_acceptor_angle
                ),
                bond_resolver=bond_resolver,
                allow_intramolecular=(
                    allow_intramolecular
                ),
                maximum_hydrogen_bonds=(
                    maximum_hydrogen_bonds
                ),
            )
        )

        detected_bonds.extend(
            inferred_bonds
        )

    if deduplicate:
        return deduplicate_hydrogen_bonds(
            detected_bonds,
            include_hydrogen_in_key=True,
            prefer_explicit=True,
        )

    return tuple(
        detected_bonds
    )


# -----------------------------------------------------------------------------
# Bidirectional ligand-receptor detection
# -----------------------------------------------------------------------------

def detect_hydrogen_bonds(
    ligand_atoms: Iterable[
        AtomLike
    ],
    receptor_atoms: Iterable[
        AtomLike
    ],
    *,
    mode: HydrogenBondMode = (
        DEFAULT_HBOND_ANALYSIS_MODE
    ),
    analyze_ligand_as_donor: bool = True,
    analyze_receptor_as_donor: bool = True,
    allow_inferred_fallback: bool = (
        DEFAULT_ALLOW_INFERRED_FALLBACK
    ),
    bond_resolver: Optional[
        BondResolver
    ] = None,
    donor_acceptor_cutoff: Optional[
        Number
    ] = None,
    hydrogen_acceptor_cutoff: Optional[
        Number
    ] = None,
    minimum_dha_angle: Optional[
        Number
    ] = None,
    minimum_inferred_angle: Optional[
        Number
    ] = None,
    distance_tolerance: Optional[
        Number
    ] = None,
    angle_tolerance: Optional[
        Number
    ] = None,
    require_inferred_donor_angle: bool = True,
    require_acceptor_angle: bool = (
        DEFAULT_REQUIRE_ACCEPTOR_ANGLE
    ),
    donor_vector_method: str = (
        DEFAULT_INFERRED_DONOR_VECTOR_METHOD
    ),
    acceptor_vector_method: str = (
        DEFAULT_INFERRED_ACCEPTOR_VECTOR_METHOD
    ),
    allow_distance_hydrogen_assignment: bool = (
        DEFAULT_ALLOW_DISTANCE_BASED_HYDROGEN_ASSIGNMENT
    ),
    include_ambiguous_assignments: bool = (
        DEFAULT_INCLUDE_AMBIGUOUS_HYDROGEN_ASSIGNMENTS
    ),
    deduplicate: bool = (
        DEFAULT_DEDUPLICATE_HYDROGEN_BONDS
    ),
    maximum_hydrogen_bonds: Optional[
        int
    ] = DEFAULT_MAXIMUM_HYDROGEN_BONDS,
) -> Tuple[
    HydrogenBond,
    ...,
]:
    """
    Detect ligand-receptor hydrogen bonds in both directions.

    Parameters
    ----------
    ligand_atoms : iterable of atom-like
        Ligand atoms.
    receptor_atoms : iterable of atom-like
        Receptor atoms.
    mode : HydrogenBondMode, optional
        Explicit or inferred mode.
    analyze_ligand_as_donor : bool, optional
        Whether ligand-donor interactions should be detected.
    analyze_receptor_as_donor : bool, optional
        Whether receptor-donor interactions should be detected.
    allow_inferred_fallback : bool, optional
        Whether explicit mode may fall back to inferred geometry.
    bond_resolver : callable or None, optional
        Custom bond resolver.
    donor_acceptor_cutoff : Number or None, optional
        Maximum D...A distance.
    hydrogen_acceptor_cutoff : Number or None, optional
        Maximum H...A distance.
    minimum_dha_angle : Number or None, optional
        Minimum D-H...A angle.
    minimum_inferred_angle : Number or None, optional
        Minimum inferred angle.
    distance_tolerance : Number or None, optional
        Distance tolerance.
    angle_tolerance : Number or None, optional
        Angular tolerance.
    require_inferred_donor_angle : bool, optional
        Whether inferred donor alignment is mandatory.
    require_acceptor_angle : bool, optional
        Whether acceptor alignment is mandatory.
    donor_vector_method : str, optional
        Donor inferred-vector method.
    acceptor_vector_method : str, optional
        Acceptor inferred-vector method.
    allow_distance_hydrogen_assignment : bool, optional
        Whether hydrogen topology may be inferred by D-H distance.
    include_ambiguous_assignments : bool, optional
        Whether ambiguous D-H assignments are retained.
    deduplicate : bool, optional
        Whether duplicate interactions should be removed.
    maximum_hydrogen_bonds : int or None, optional
        Optional global result limit.

    Returns
    -------
    tuple of HydrogenBond
        Detected ligand-receptor hydrogen bonds.
    """

    normalized_ligand_atoms = (
        _normalize_hbond_atom_collection(
            ligand_atoms,
            name="ligand",
            require_coordinate=True,
        )
    )

    normalized_receptor_atoms = (
        _normalize_hbond_atom_collection(
            receptor_atoms,
            name="receptor",
            require_coordinate=True,
        )
    )

    if (
        not normalized_ligand_atoms
        or not normalized_receptor_atoms
    ):
        return ()

    if _collections_share_atoms(
        normalized_ligand_atoms,
        normalized_receptor_atoms,
    ):
        raise ValueError(
            "Ligand and receptor collections must not share atom objects."
        )

    detected_bonds: List[
        HydrogenBond
    ] = []

    if analyze_ligand_as_donor:
        remaining_limit = (
            None
            if maximum_hydrogen_bonds is None
            else max(
                0,
                maximum_hydrogen_bonds
                - len(
                    detected_bonds
                ),
            )
        )

        ligand_donor_bonds = (
            detect_directional_hydrogen_bonds(
                normalized_ligand_atoms,
                normalized_receptor_atoms,
                direction=(
                    HBOND_DIRECTION_LIGAND_DONOR
                ),
                mode=mode,
                allow_inferred_fallback=(
                    allow_inferred_fallback
                ),
                bond_resolver=bond_resolver,
                donor_acceptor_cutoff=(
                    donor_acceptor_cutoff
                ),
                hydrogen_acceptor_cutoff=(
                    hydrogen_acceptor_cutoff
                ),
                minimum_dha_angle=(
                    minimum_dha_angle
                ),
                minimum_inferred_angle=(
                    minimum_inferred_angle
                ),
                distance_tolerance=(
                    distance_tolerance
                ),
                angle_tolerance=(
                    angle_tolerance
                ),
                require_inferred_donor_angle=(
                    require_inferred_donor_angle
                ),
                require_acceptor_angle=(
                    require_acceptor_angle
                ),
                donor_vector_method=(
                    donor_vector_method
                ),
                acceptor_vector_method=(
                    acceptor_vector_method
                ),
                allow_distance_hydrogen_assignment=(
                    allow_distance_hydrogen_assignment
                ),
                include_ambiguous_assignments=(
                    include_ambiguous_assignments
                ),
                allow_intramolecular=False,
                deduplicate=deduplicate,
                maximum_hydrogen_bonds=(
                    remaining_limit
                ),
            )
        )

        detected_bonds.extend(
            ligand_donor_bonds
        )

    if (
        analyze_receptor_as_donor
        and (
            maximum_hydrogen_bonds is None
            or len(
                detected_bonds
            )
            < maximum_hydrogen_bonds
        )
    ):
        remaining_limit = (
            None
            if maximum_hydrogen_bonds is None
            else max(
                0,
                maximum_hydrogen_bonds
                - len(
                    detected_bonds
                ),
            )
        )

        receptor_donor_bonds = (
            detect_directional_hydrogen_bonds(
                normalized_receptor_atoms,
                normalized_ligand_atoms,
                direction=(
                    HBOND_DIRECTION_RECEPTOR_DONOR
                ),
                mode=mode,
                allow_inferred_fallback=(
                    allow_inferred_fallback
                ),
                bond_resolver=bond_resolver,
                donor_acceptor_cutoff=(
                    donor_acceptor_cutoff
                ),
                hydrogen_acceptor_cutoff=(
                    hydrogen_acceptor_cutoff
                ),
                minimum_dha_angle=(
                    minimum_dha_angle
                ),
                minimum_inferred_angle=(
                    minimum_inferred_angle
                ),
                distance_tolerance=(
                    distance_tolerance
                ),
                angle_tolerance=(
                    angle_tolerance
                ),
                require_inferred_donor_angle=(
                    require_inferred_donor_angle
                ),
                require_acceptor_angle=(
                    require_acceptor_angle
                ),
                donor_vector_method=(
                    donor_vector_method
                ),
                acceptor_vector_method=(
                    acceptor_vector_method
                ),
                allow_distance_hydrogen_assignment=(
                    allow_distance_hydrogen_assignment
                ),
                include_ambiguous_assignments=(
                    include_ambiguous_assignments
                ),
                allow_intramolecular=False,
                deduplicate=deduplicate,
                maximum_hydrogen_bonds=(
                    remaining_limit
                ),
            )
        )

        detected_bonds.extend(
            receptor_donor_bonds
        )

    if deduplicate:
        detected_result = deduplicate_hydrogen_bonds(
            detected_bonds,
            include_hydrogen_in_key=True,
            prefer_explicit=True,
        )

    else:
        detected_result = tuple(
            detected_bonds
        )

    if maximum_hydrogen_bonds is not None:
        return detected_result[
            :maximum_hydrogen_bonds
        ]

    return detected_result


# -----------------------------------------------------------------------------
# Complete analysis-result construction
# -----------------------------------------------------------------------------

def analyze_hydrogen_bonds(
    ligand_atoms: Iterable[
        AtomLike
    ],
    receptor_atoms: Iterable[
        AtomLike
    ],
    *,
    mode: HydrogenBondMode = (
        DEFAULT_HBOND_ANALYSIS_MODE
    ),
    analyze_ligand_as_donor: bool = True,
    analyze_receptor_as_donor: bool = True,
    allow_inferred_fallback: bool = (
        DEFAULT_ALLOW_INFERRED_FALLBACK
    ),
    bond_resolver: Optional[
        BondResolver
    ] = None,
    donor_acceptor_cutoff: Optional[
        Number
    ] = None,
    hydrogen_acceptor_cutoff: Optional[
        Number
    ] = None,
    minimum_dha_angle: Optional[
        Number
    ] = None,
    minimum_inferred_angle: Optional[
        Number
    ] = None,
    distance_tolerance: Optional[
        Number
    ] = None,
    angle_tolerance: Optional[
        Number
    ] = None,
    require_inferred_donor_angle: bool = True,
    require_acceptor_angle: bool = (
        DEFAULT_REQUIRE_ACCEPTOR_ANGLE
    ),
    donor_vector_method: str = (
        DEFAULT_INFERRED_DONOR_VECTOR_METHOD
    ),
    acceptor_vector_method: str = (
        DEFAULT_INFERRED_ACCEPTOR_VECTOR_METHOD
    ),
    allow_distance_hydrogen_assignment: bool = (
        DEFAULT_ALLOW_DISTANCE_BASED_HYDROGEN_ASSIGNMENT
    ),
    include_ambiguous_assignments: bool = (
        DEFAULT_INCLUDE_AMBIGUOUS_HYDROGEN_ASSIGNMENTS
    ),
    deduplicate: bool = (
        DEFAULT_DEDUPLICATE_HYDROGEN_BONDS
    ),
    maximum_hydrogen_bonds: Optional[
        int
    ] = DEFAULT_MAXIMUM_HYDROGEN_BONDS,
    metadata: Optional[
        Mapping[
            str,
            Any,
        ]
    ] = None,
) -> HydrogenBondAnalysisResult:
    """
    Perform complete ligand-receptor hydrogen-bond detection.

    Parameters
    ----------
    ligand_atoms : iterable of atom-like
        Ligand atoms.
    receptor_atoms : iterable of atom-like
        Receptor atoms.
    mode : HydrogenBondMode, optional
        Explicit or inferred mode.
    analyze_ligand_as_donor : bool, optional
        Whether ligand-donor interactions are included.
    analyze_receptor_as_donor : bool, optional
        Whether receptor-donor interactions are included.
    allow_inferred_fallback : bool, optional
        Whether explicit mode can fall back to inferred geometry.
    bond_resolver : callable or None, optional
        Custom bonded-neighbor resolver.
    donor_acceptor_cutoff : Number or None, optional
        Maximum D...A distance.
    hydrogen_acceptor_cutoff : Number or None, optional
        Maximum H...A distance.
    minimum_dha_angle : Number or None, optional
        Minimum D-H...A angle.
    minimum_inferred_angle : Number or None, optional
        Minimum inferred angle.
    distance_tolerance : Number or None, optional
        Distance tolerance.
    angle_tolerance : Number or None, optional
        Angular tolerance.
    require_inferred_donor_angle : bool, optional
        Whether inferred donor alignment is mandatory.
    require_acceptor_angle : bool, optional
        Whether acceptor alignment is mandatory.
    donor_vector_method : str, optional
        Donor inferred-vector method.
    acceptor_vector_method : str, optional
        Acceptor inferred-vector method.
    allow_distance_hydrogen_assignment : bool, optional
        Whether explicit D-H assignments may be inferred by distance.
    include_ambiguous_assignments : bool, optional
        Whether ambiguous D-H assignments are retained.
    deduplicate : bool, optional
        Whether duplicate hydrogen bonds are removed.
    maximum_hydrogen_bonds : int or None, optional
        Optional interaction limit.
    metadata : mapping or None, optional
        Additional result metadata.

    Returns
    -------
    HydrogenBondAnalysisResult
        Complete detection result.

    Notes
    -----
    Residue grouping and detailed statistics are populated by later sections.
    At this stage, ``residue_hydrogen_bonds`` and ``statistics`` remain empty.
    """

    normalized_mode = (
        validate_hydrogen_bond_mode(
            mode
        )
    )

    normalized_ligand_atoms = (
        _normalize_hbond_atom_collection(
            ligand_atoms,
            name="ligand",
            require_coordinate=True,
        )
    )

    normalized_receptor_atoms = (
        _normalize_hbond_atom_collection(
            receptor_atoms,
            name="receptor",
            require_coordinate=True,
        )
    )

    resolved_da_cutoff = (
        get_default_donor_acceptor_distance()
        if donor_acceptor_cutoff is None
        else _coerce_positive_float(
            donor_acceptor_cutoff,
            name="donor-acceptor cutoff",
            default=(
                DEFAULT_DONOR_ACCEPTOR_DISTANCE
            ),
        )
    )

    resolved_ha_cutoff = (
        get_default_hydrogen_acceptor_distance()
        if hydrogen_acceptor_cutoff is None
        else _coerce_positive_float(
            hydrogen_acceptor_cutoff,
            name="hydrogen-acceptor cutoff",
            default=(
                DEFAULT_HYDROGEN_ACCEPTOR_DISTANCE
            ),
        )
    )

    resolved_dha_angle = (
        get_default_minimum_dha_angle()
        if minimum_dha_angle is None
        else _coerce_angle(
            minimum_dha_angle,
            name="minimum D-H-A angle",
            default=DEFAULT_MINIMUM_DHA_ANGLE,
        )
    )

    resolved_inferred_angle = (
        get_default_minimum_inferred_angle()
        if minimum_inferred_angle is None
        else _coerce_angle(
            minimum_inferred_angle,
            name="minimum inferred angle",
            default=(
                DEFAULT_MINIMUM_INFERRED_ANGLE
            ),
        )
    )

    detected_bonds = detect_hydrogen_bonds(
        normalized_ligand_atoms,
        normalized_receptor_atoms,
        mode=normalized_mode,
        analyze_ligand_as_donor=(
            analyze_ligand_as_donor
        ),
        analyze_receptor_as_donor=(
            analyze_receptor_as_donor
        ),
        allow_inferred_fallback=(
            allow_inferred_fallback
        ),
        bond_resolver=bond_resolver,
        donor_acceptor_cutoff=(
            resolved_da_cutoff
        ),
        hydrogen_acceptor_cutoff=(
            resolved_ha_cutoff
        ),
        minimum_dha_angle=(
            resolved_dha_angle
        ),
        minimum_inferred_angle=(
            resolved_inferred_angle
        ),
        distance_tolerance=(
            distance_tolerance
        ),
        angle_tolerance=(
            angle_tolerance
        ),
        require_inferred_donor_angle=(
            require_inferred_donor_angle
        ),
        require_acceptor_angle=(
            require_acceptor_angle
        ),
        donor_vector_method=(
            donor_vector_method
        ),
        acceptor_vector_method=(
            acceptor_vector_method
        ),
        allow_distance_hydrogen_assignment=(
            allow_distance_hydrogen_assignment
        ),
        include_ambiguous_assignments=(
            include_ambiguous_assignments
        ),
        deduplicate=deduplicate,
        maximum_hydrogen_bonds=(
            maximum_hydrogen_bonds
        ),
    )

    result_metadata: Dict[
        str,
        Any,
    ] = {
        "analysis_mode": normalized_mode,
        "analyze_ligand_as_donor": bool(
            analyze_ligand_as_donor
        ),
        "analyze_receptor_as_donor": bool(
            analyze_receptor_as_donor
        ),
        "allow_inferred_fallback": bool(
            allow_inferred_fallback
        ),
        "require_inferred_donor_angle": bool(
            require_inferred_donor_angle
        ),
        "require_acceptor_angle": bool(
            require_acceptor_angle
        ),
        "donor_vector_method": (
            donor_vector_method
        ),
        "acceptor_vector_method": (
            acceptor_vector_method
        ),
        "distance_hydrogen_assignment": bool(
            allow_distance_hydrogen_assignment
        ),
        "ambiguous_assignments_included": bool(
            include_ambiguous_assignments
        ),
        "deduplicated": bool(
            deduplicate
        ),
    }

    if metadata:
        result_metadata.update(
            metadata
        )

    return HydrogenBondAnalysisResult(
        hydrogen_bonds=detected_bonds,
        residue_hydrogen_bonds=(),
        ligand_atoms=normalized_ligand_atoms,
        receptor_atoms=normalized_receptor_atoms,
        donor_acceptor_cutoff=(
            resolved_da_cutoff
        ),
        hydrogen_acceptor_cutoff=(
            resolved_ha_cutoff
        ),
        minimum_dha_angle=(
            resolved_dha_angle
        ),
        minimum_inferred_angle=(
            resolved_inferred_angle
        ),
        statistics={},
        metadata=result_metadata,
    )


# -----------------------------------------------------------------------------
# Public interface
# -----------------------------------------------------------------------------

_SECTION_7_PUBLIC_NAMES: Final[
    Tuple[
        str,
        ...,
    ]
] = (
    "DEFAULT_HBOND_ANALYSIS_MODE",
    "DEFAULT_ALLOW_INFERRED_FALLBACK",
    "DEFAULT_INCLUDE_AMBIGUOUS_HYDROGEN_ASSIGNMENTS",
    "DEFAULT_REQUIRE_ACCEPTOR_ANGLE",
    "DEFAULT_ALLOW_INTRAMOLECULAR_HBONDS",
    "DEFAULT_DEDUPLICATE_HYDROGEN_BONDS",
    "DEFAULT_STOP_AFTER_FIRST_VALID_HYDROGEN",
    "DEFAULT_INCLUDE_UNKNOWN_DIRECTION",
    "find_hbond_candidate_pairs",
    "detect_explicit_hydrogen_bonds",
    "detect_inferred_hydrogen_bonds",
    "get_hydrogen_bond_pair_key",
    "deduplicate_hydrogen_bonds",
    "detect_directional_hydrogen_bonds",
    "detect_hydrogen_bonds",
    "analyze_hydrogen_bonds",
)

for public_name in _SECTION_7_PUBLIC_NAMES:
    if public_name not in __all__:
        __all__.append(
            public_name
        )


# =============================================================================
# End of Section 7
# =============================================================================


# =============================================================================
# Section 8 — Residue grouping
# =============================================================================


# -----------------------------------------------------------------------------
# Residue-grouping constants
# -----------------------------------------------------------------------------

RESIDUE_GROUP_SIDE_DONOR: Final[
    str
] = "donor"

RESIDUE_GROUP_SIDE_ACCEPTOR: Final[
    str
] = "acceptor"

RESIDUE_GROUP_SIDE_RECEPTOR: Final[
    str
] = "receptor"

RESIDUE_GROUP_SIDE_LIGAND: Final[
    str
] = "ligand"

RESIDUE_GROUP_SIDE_EITHER: Final[
    str
] = "either"


VALID_RESIDUE_GROUP_SIDES: Final[
    FrozenSet[
        str
    ]
] = frozenset(
    {
        RESIDUE_GROUP_SIDE_DONOR,
        RESIDUE_GROUP_SIDE_ACCEPTOR,
        RESIDUE_GROUP_SIDE_RECEPTOR,
        RESIDUE_GROUP_SIDE_LIGAND,
        RESIDUE_GROUP_SIDE_EITHER,
    }
)


DEFAULT_RESIDUE_GROUP_SIDE: Final[
    str
] = RESIDUE_GROUP_SIDE_RECEPTOR

DEFAULT_INCLUDE_RESIDUELESS_HBONDS: Final[
    bool
] = False

DEFAULT_SORT_RESIDUE_GROUPS: Final[
    bool
] = True

DEFAULT_SORT_BONDS_WITHIN_RESIDUE: Final[
    bool
] = True

DEFAULT_DEDUPLICATE_WITHIN_RESIDUE: Final[
    bool
] = False


# Synthetic residue key used only when the caller explicitly requests inclusion
# of hydrogen bonds whose residue cannot be resolved.
_UNRESOLVED_RESIDUE_KEY: Final[
    ResidueContactKey
] = (
    "UNK",
    -1,
    "",
)


# -----------------------------------------------------------------------------
# Residue-side validation
# -----------------------------------------------------------------------------

def validate_residue_group_side(
    side: str,
) -> str:
    """
    Validate a residue-grouping side.

    Parameters
    ----------
    side : str
        Grouping side.

    Returns
    -------
    str
        Normalized grouping side.

    Raises
    ------
    TypeError
        If ``side`` is not a string.
    ValueError
        If the grouping side is unsupported.
    """

    if not isinstance(
        side,
        str,
    ):
        raise TypeError(
            "Residue grouping side must be a string."
        )

    normalized_side = side.strip().lower()

    if normalized_side not in VALID_RESIDUE_GROUP_SIDES:
        raise ValueError(
            f"Unsupported residue grouping side {side!r}. "
            "Expected one of: "
            f"{', '.join(sorted(VALID_RESIDUE_GROUP_SIDES))}."
        )

    return normalized_side


# -----------------------------------------------------------------------------
# Residue access helpers
# -----------------------------------------------------------------------------

def _get_hbond_donor_residue(
    hydrogen_bond: HydrogenBond,
) -> Optional[
    ResidueLike
]:
    """
    Return the donor residue of a hydrogen bond.

    Parameters
    ----------
    hydrogen_bond : HydrogenBond
        Hydrogen bond.

    Returns
    -------
    residue-like or None
        Donor residue.
    """

    if hydrogen_bond.donor_residue is not None:
        return hydrogen_bond.donor_residue

    try:
        return get_atom_residue(
            hydrogen_bond.donor
        )

    except Exception:
        return None


def _get_hbond_acceptor_residue(
    hydrogen_bond: HydrogenBond,
) -> Optional[
    ResidueLike
]:
    """
    Return the acceptor residue of a hydrogen bond.

    Parameters
    ----------
    hydrogen_bond : HydrogenBond
        Hydrogen bond.

    Returns
    -------
    residue-like or None
        Acceptor residue.
    """

    if hydrogen_bond.acceptor_residue is not None:
        return hydrogen_bond.acceptor_residue

    try:
        return get_atom_residue(
            hydrogen_bond.acceptor
        )

    except Exception:
        return None


def _get_residue_key_defensively(
    residue: Optional[
        ResidueLike
    ],
) -> Optional[
    ResidueContactKey
]:
    """
    Return a normalized residue key defensively.

    Parameters
    ----------
    residue : residue-like or None
        Residue to identify.

    Returns
    -------
    ResidueContactKey or None
        Normalized residue key.
    """

    if residue is None:
        return None

    try:
        key = get_residue_contact_key(
            residue
        )

    except Exception:
        return None

    try:
        return _normalize_residue_key(
            key
        )

    except Exception:
        return None


def _get_hbond_donor_residue_key(
    hydrogen_bond: HydrogenBond,
) -> Optional[
    ResidueContactKey
]:
    """
    Return the donor residue key of a hydrogen bond.

    Parameters
    ----------
    hydrogen_bond : HydrogenBond
        Hydrogen bond.

    Returns
    -------
    ResidueContactKey or None
        Donor residue key.
    """

    if hydrogen_bond.donor_residue_key is not None:
        try:
            return _normalize_residue_key(
                hydrogen_bond.donor_residue_key
            )

        except Exception:
            pass

    return _get_residue_key_defensively(
        _get_hbond_donor_residue(
            hydrogen_bond
        )
    )


def _get_hbond_acceptor_residue_key(
    hydrogen_bond: HydrogenBond,
) -> Optional[
    ResidueContactKey
]:
    """
    Return the acceptor residue key of a hydrogen bond.

    Parameters
    ----------
    hydrogen_bond : HydrogenBond
        Hydrogen bond.

    Returns
    -------
    ResidueContactKey or None
        Acceptor residue key.
    """

    if hydrogen_bond.acceptor_residue_key is not None:
        try:
            return _normalize_residue_key(
                hydrogen_bond.acceptor_residue_key
            )

        except Exception:
            pass

    return _get_residue_key_defensively(
        _get_hbond_acceptor_residue(
            hydrogen_bond
        )
    )


# -----------------------------------------------------------------------------
# Ligand/receptor side resolution
# -----------------------------------------------------------------------------

def get_hbond_receptor_residue(
    hydrogen_bond: HydrogenBond,
) -> Optional[
    ResidueLike
]:
    """
    Return the receptor residue involved in a hydrogen bond.

    Parameters
    ----------
    hydrogen_bond : HydrogenBond
        Hydrogen bond.

    Returns
    -------
    residue-like or None
        Receptor residue.

    Notes
    -----
    The interaction direction determines which atom belongs to the receptor:

    - ``ligand_donor``: receptor is the acceptor side;
    - ``receptor_donor``: receptor is the donor side.
    """

    if not isinstance(
        hydrogen_bond,
        HydrogenBond,
    ):
        raise TypeError(
            "hydrogen_bond must be a HydrogenBond instance."
        )

    if (
        hydrogen_bond.direction
        == HBOND_DIRECTION_LIGAND_DONOR
    ):
        return _get_hbond_acceptor_residue(
            hydrogen_bond
        )

    if (
        hydrogen_bond.direction
        == HBOND_DIRECTION_RECEPTOR_DONOR
    ):
        return _get_hbond_donor_residue(
            hydrogen_bond
        )

    return None


def get_hbond_ligand_residue(
    hydrogen_bond: HydrogenBond,
) -> Optional[
    ResidueLike
]:
    """
    Return the ligand residue involved in a hydrogen bond.

    Parameters
    ----------
    hydrogen_bond : HydrogenBond
        Hydrogen bond.

    Returns
    -------
    residue-like or None
        Ligand residue.
    """

    if not isinstance(
        hydrogen_bond,
        HydrogenBond,
    ):
        raise TypeError(
            "hydrogen_bond must be a HydrogenBond instance."
        )

    if (
        hydrogen_bond.direction
        == HBOND_DIRECTION_LIGAND_DONOR
    ):
        return _get_hbond_donor_residue(
            hydrogen_bond
        )

    if (
        hydrogen_bond.direction
        == HBOND_DIRECTION_RECEPTOR_DONOR
    ):
        return _get_hbond_acceptor_residue(
            hydrogen_bond
        )

    return None


def get_hbond_receptor_residue_key(
    hydrogen_bond: HydrogenBond,
) -> Optional[
    ResidueContactKey
]:
    """
    Return the receptor residue key involved in a hydrogen bond.

    Parameters
    ----------
    hydrogen_bond : HydrogenBond
        Hydrogen bond.

    Returns
    -------
    ResidueContactKey or None
        Receptor residue key.
    """

    if (
        hydrogen_bond.direction
        == HBOND_DIRECTION_LIGAND_DONOR
    ):
        return _get_hbond_acceptor_residue_key(
            hydrogen_bond
        )

    if (
        hydrogen_bond.direction
        == HBOND_DIRECTION_RECEPTOR_DONOR
    ):
        return _get_hbond_donor_residue_key(
            hydrogen_bond
        )

    return None


def get_hbond_ligand_residue_key(
    hydrogen_bond: HydrogenBond,
) -> Optional[
    ResidueContactKey
]:
    """
    Return the ligand residue key involved in a hydrogen bond.

    Parameters
    ----------
    hydrogen_bond : HydrogenBond
        Hydrogen bond.

    Returns
    -------
    ResidueContactKey or None
        Ligand residue key.
    """

    if (
        hydrogen_bond.direction
        == HBOND_DIRECTION_LIGAND_DONOR
    ):
        return _get_hbond_donor_residue_key(
            hydrogen_bond
        )

    if (
        hydrogen_bond.direction
        == HBOND_DIRECTION_RECEPTOR_DONOR
    ):
        return _get_hbond_acceptor_residue_key(
            hydrogen_bond
        )

    return None


# -----------------------------------------------------------------------------
# Generic residue-target resolution
# -----------------------------------------------------------------------------

def get_hbond_residue_for_side(
    hydrogen_bond: HydrogenBond,
    side: str = DEFAULT_RESIDUE_GROUP_SIDE,
) -> Optional[
    ResidueLike
]:
    """
    Return the residue associated with a selected interaction side.

    Parameters
    ----------
    hydrogen_bond : HydrogenBond
        Hydrogen bond.
    side : str, optional
        ``"donor"``, ``"acceptor"``, ``"receptor"`` or ``"ligand"``.

    Returns
    -------
    residue-like or None
        Selected residue.

    Raises
    ------
    ValueError
        If ``side="either"`` is requested because one bond may then map to two
        residues. Use :func:`get_hbond_residue_entries` instead.
    """

    normalized_side = validate_residue_group_side(
        side
    )

    if normalized_side == RESIDUE_GROUP_SIDE_DONOR:
        return _get_hbond_donor_residue(
            hydrogen_bond
        )

    if normalized_side == RESIDUE_GROUP_SIDE_ACCEPTOR:
        return _get_hbond_acceptor_residue(
            hydrogen_bond
        )

    if normalized_side == RESIDUE_GROUP_SIDE_RECEPTOR:
        return get_hbond_receptor_residue(
            hydrogen_bond
        )

    if normalized_side == RESIDUE_GROUP_SIDE_LIGAND:
        return get_hbond_ligand_residue(
            hydrogen_bond
        )

    raise ValueError(
        "side='either' may resolve to more than one residue. "
        "Use get_hbond_residue_entries()."
    )


def get_hbond_residue_key_for_side(
    hydrogen_bond: HydrogenBond,
    side: str = DEFAULT_RESIDUE_GROUP_SIDE,
) -> Optional[
    ResidueContactKey
]:
    """
    Return the residue key associated with a selected interaction side.

    Parameters
    ----------
    hydrogen_bond : HydrogenBond
        Hydrogen bond.
    side : str, optional
        Selected grouping side.

    Returns
    -------
    ResidueContactKey or None
        Selected residue key.
    """

    normalized_side = validate_residue_group_side(
        side
    )

    if normalized_side == RESIDUE_GROUP_SIDE_DONOR:
        return _get_hbond_donor_residue_key(
            hydrogen_bond
        )

    if normalized_side == RESIDUE_GROUP_SIDE_ACCEPTOR:
        return _get_hbond_acceptor_residue_key(
            hydrogen_bond
        )

    if normalized_side == RESIDUE_GROUP_SIDE_RECEPTOR:
        return get_hbond_receptor_residue_key(
            hydrogen_bond
        )

    if normalized_side == RESIDUE_GROUP_SIDE_LIGAND:
        return get_hbond_ligand_residue_key(
            hydrogen_bond
        )

    raise ValueError(
        "side='either' may resolve to more than one residue. "
        "Use get_hbond_residue_entries()."
    )


def get_hbond_residue_entries(
    hydrogen_bond: HydrogenBond,
    *,
    side: str = DEFAULT_RESIDUE_GROUP_SIDE,
    include_unresolved: bool = (
        DEFAULT_INCLUDE_RESIDUELESS_HBONDS
    ),
) -> Tuple[
    Tuple[
        ResidueContactKey,
        Optional[
            ResidueLike
        ],
    ],
    ...,
]:
    """
    Return residue entries used to group a hydrogen bond.

    Parameters
    ----------
    hydrogen_bond : HydrogenBond
        Hydrogen bond.
    side : str, optional
        Grouping side.
    include_unresolved : bool, optional
        Whether unresolved residues should use a synthetic ``UNK`` key.

    Returns
    -------
    tuple
        ``(residue_key, residue)`` entries.

    Notes
    -----
    With ``side="either"``, a hydrogen bond may produce both donor-side and
    acceptor-side entries. Duplicate residue identities are removed.
    """

    if not isinstance(
        hydrogen_bond,
        HydrogenBond,
    ):
        raise TypeError(
            "hydrogen_bond must be a HydrogenBond instance."
        )

    normalized_side = validate_residue_group_side(
        side
    )

    entries: List[
        Tuple[
            Optional[
                ResidueContactKey
            ],
            Optional[
                ResidueLike
            ],
        ]
    ] = []

    if normalized_side == RESIDUE_GROUP_SIDE_EITHER:
        entries.extend(
            (
                (
                    _get_hbond_donor_residue_key(
                        hydrogen_bond
                    ),
                    _get_hbond_donor_residue(
                        hydrogen_bond
                    ),
                ),
                (
                    _get_hbond_acceptor_residue_key(
                        hydrogen_bond
                    ),
                    _get_hbond_acceptor_residue(
                        hydrogen_bond
                    ),
                ),
            )
        )

    else:
        entries.append(
            (
                get_hbond_residue_key_for_side(
                    hydrogen_bond,
                    normalized_side,
                ),
                get_hbond_residue_for_side(
                    hydrogen_bond,
                    normalized_side,
                ),
            )
        )

    normalized_entries: List[
        Tuple[
            ResidueContactKey,
            Optional[
                ResidueLike
            ],
        ]
    ] = []

    seen_entries: Set[
        Tuple[
            ResidueContactKey,
            Optional[
                int
            ],
        ]
    ] = set()

    for residue_key, residue in entries:
        if residue_key is None:
            if not include_unresolved:
                continue

            normalized_key = (
                _UNRESOLVED_RESIDUE_KEY
            )

        else:
            try:
                normalized_key = (
                    _normalize_residue_key(
                        residue_key
                    )
                )

            except Exception:
                if not include_unresolved:
                    continue

                normalized_key = (
                    _UNRESOLVED_RESIDUE_KEY
                )

        if normalized_key is None:
            continue

        entry_identity = (
            normalized_key,
            None
            if residue is None
            else id(
                residue
            ),
        )

        if entry_identity in seen_entries:
            continue

        seen_entries.add(
            entry_identity
        )

        normalized_entries.append(
            (
                normalized_key,
                residue,
            )
        )

    return tuple(
        normalized_entries
    )


# -----------------------------------------------------------------------------
# Sorting helpers
# -----------------------------------------------------------------------------

def _residue_key_sort_key(
    key: ResidueContactKey,
) -> Tuple[
    str,
    int,
    str,
]:
    """
    Return a deterministic sorting key for a residue key.

    Parameters
    ----------
    key : ResidueContactKey
        Residue key.

    Returns
    -------
    tuple
        Sorting key.
    """

    normalized_key = _normalize_residue_key(
        key
    )

    if normalized_key is None:
        normalized_key = _UNRESOLVED_RESIDUE_KEY

    residue_name, residue_number, chain_id = (
        normalized_key
    )

    return (
        str(
            chain_id
        ),
        int(
            residue_number
        ),
        str(
            residue_name
        ),
    )


def _hydrogen_bond_residue_sort_key(
    hydrogen_bond: HydrogenBond,
) -> Tuple[
    np.float64,
    np.float64,
    np.float64,
    str,
    str,
]:
    """
    Return a deterministic within-residue hydrogen-bond sorting key.

    Parameters
    ----------
    hydrogen_bond : HydrogenBond
        Hydrogen bond.

    Returns
    -------
    tuple
        Sorting key.

    Notes
    -----
    Interactions are ordered by:

    1. D...A distance;
    2. H...A distance;
    3. descending D-H...A angle;
    4. donor identifier;
    5. acceptor identifier.
    """

    hydrogen_acceptor_distance = (
        hydrogen_bond
        .hydrogen_acceptor_distance
    )

    dha_angle = hydrogen_bond.dha_angle

    return (
        np.float64(
            hydrogen_bond
            .donor_acceptor_distance
        ),
        (
            np.float64(
                np.inf
            )
            if hydrogen_acceptor_distance
            is None
            else np.float64(
                hydrogen_acceptor_distance
            )
        ),
        (
            np.float64(
                0.0
            )
            if dha_angle is None
            else np.float64(
                -dha_angle
            )
        ),
        (
            _safe_atom_identifier(
                hydrogen_bond.donor
            )
            or ""
        ),
        (
            _safe_atom_identifier(
                hydrogen_bond.acceptor
            )
            or ""
        ),
    )


# -----------------------------------------------------------------------------
# Primary residue-grouping function
# -----------------------------------------------------------------------------

def group_hydrogen_bonds_by_residue(
    hydrogen_bonds: Iterable[
        HydrogenBond
    ],
    *,
    side: str = DEFAULT_RESIDUE_GROUP_SIDE,
    include_unresolved: bool = (
        DEFAULT_INCLUDE_RESIDUELESS_HBONDS
    ),
    sort_groups: bool = (
        DEFAULT_SORT_RESIDUE_GROUPS
    ),
    sort_bonds: bool = (
        DEFAULT_SORT_BONDS_WITHIN_RESIDUE
    ),
    deduplicate_bonds: bool = (
        DEFAULT_DEDUPLICATE_WITHIN_RESIDUE
    ),
) -> Tuple[
    ResidueHydrogenBond,
    ...,
]:
    """
    Group hydrogen bonds by residue.

    Parameters
    ----------
    hydrogen_bonds : iterable of HydrogenBond
        Hydrogen bonds to group.
    side : str, optional
        Grouping side. The default groups by receptor residue.
    include_unresolved : bool, optional
        Whether interactions with unresolved residues should be retained under
        a synthetic ``("UNK", -1, "")`` key.
    sort_groups : bool, optional
        Whether residue groups should be sorted deterministically.
    sort_bonds : bool, optional
        Whether bonds inside each residue group should be sorted.
    deduplicate_bonds : bool, optional
        Whether repeated interaction objects should be removed inside each
        group.

    Returns
    -------
    tuple of ResidueHydrogenBond
        Residue-level hydrogen-bond groups.
    """

    normalized_side = validate_residue_group_side(
        side
    )

    normalized_bonds = tuple(
        hydrogen_bonds
    )

    for index, hydrogen_bond in enumerate(
        normalized_bonds
    ):
        if not isinstance(
            hydrogen_bond,
            HydrogenBond,
        ):
            raise TypeError(
                "All entries must be HydrogenBond instances. "
                f"Invalid entry at index {index}."
            )

    grouped_bonds: Dict[
        ResidueContactKey,
        List[
            HydrogenBond
        ],
    ] = {}

    grouped_residues: Dict[
        ResidueContactKey,
        Optional[
            ResidueLike
        ],
    ] = {}

    for hydrogen_bond in normalized_bonds:
        residue_entries = (
            get_hbond_residue_entries(
                hydrogen_bond,
                side=normalized_side,
                include_unresolved=(
                    include_unresolved
                ),
            )
        )

        for residue_key, residue in residue_entries:
            grouped_bonds.setdefault(
                residue_key,
                [],
            ).append(
                hydrogen_bond
            )

            existing_residue = (
                grouped_residues.get(
                    residue_key
                )
            )

            if (
                existing_residue is None
                and residue is not None
            ):
                grouped_residues[
                    residue_key
                ] = residue

            elif residue_key not in grouped_residues:
                grouped_residues[
                    residue_key
                ] = residue

    group_keys = list(
        grouped_bonds
    )

    if sort_groups:
        group_keys.sort(
            key=_residue_key_sort_key
        )

    grouped_results: List[
        ResidueHydrogenBond
    ] = []

    for residue_key in group_keys:
        residue_bonds = list(
            grouped_bonds[
                residue_key
            ]
        )

        if deduplicate_bonds:
            residue_bonds = list(
                deduplicate_hydrogen_bonds(
                    residue_bonds,
                    include_hydrogen_in_key=True,
                    prefer_explicit=True,
                )
            )

        if sort_bonds:
            residue_bonds.sort(
                key=(
                    _hydrogen_bond_residue_sort_key
                )
            )

        residue = grouped_residues.get(
            residue_key
        )

        if residue is None:
            if not include_unresolved:
                continue

            # ResidueHydrogenBond requires a non-None residue object. A small
            # immutable synthetic mapping is used only for unresolved groups.
            residue = MappingProxyType(
                {
                    "name": residue_key[
                        0
                    ],
                    "number": residue_key[
                        1
                    ],
                    "chain_id": residue_key[
                        2
                    ],
                    "synthetic": True,
                }
            )

        grouped_results.append(
            ResidueHydrogenBond(
                residue=residue,
                key=residue_key,
                hydrogen_bonds=tuple(
                    residue_bonds
                ),
                side=normalized_side,
                metadata={
                    "grouping_side": (
                        normalized_side
                    ),
                    "unresolved_residue": (
                        residue_key
                        == _UNRESOLVED_RESIDUE_KEY
                    ),
                    "sorted_bonds": bool(
                        sort_bonds
                    ),
                    "deduplicated_bonds": bool(
                        deduplicate_bonds
                    ),
                },
            )
        )

    return tuple(
        grouped_results
    )


# -----------------------------------------------------------------------------
# Mapping-based group representations
# -----------------------------------------------------------------------------

def map_hydrogen_bonds_by_residue_key(
    hydrogen_bonds: Iterable[
        HydrogenBond
    ],
    *,
    side: str = DEFAULT_RESIDUE_GROUP_SIDE,
    include_unresolved: bool = (
        DEFAULT_INCLUDE_RESIDUELESS_HBONDS
    ),
    sort_bonds: bool = (
        DEFAULT_SORT_BONDS_WITHIN_RESIDUE
    ),
) -> Dict[
    ResidueContactKey,
    Tuple[
        HydrogenBond,
        ...,
    ],
]:
    """
    Return a residue-key-to-hydrogen-bonds mapping.

    Parameters
    ----------
    hydrogen_bonds : iterable of HydrogenBond
        Hydrogen bonds to group.
    side : str, optional
        Grouping side.
    include_unresolved : bool, optional
        Whether unresolved residue groups should be included.
    sort_bonds : bool, optional
        Whether interactions inside each group should be sorted.

    Returns
    -------
    dict
        Mapping from residue keys to hydrogen-bond tuples.
    """

    residue_groups = group_hydrogen_bonds_by_residue(
        hydrogen_bonds,
        side=side,
        include_unresolved=(
            include_unresolved
        ),
        sort_groups=True,
        sort_bonds=sort_bonds,
        deduplicate_bonds=False,
    )

    return {
        group.key: group.hydrogen_bonds
        for group in residue_groups
    }


def map_residue_hydrogen_bond_results(
    residue_groups: Iterable[
        ResidueHydrogenBond
    ],
) -> Dict[
    ResidueContactKey,
    ResidueHydrogenBond,
]:
    """
    Convert residue-group results to a key-based mapping.

    Parameters
    ----------
    residue_groups : iterable of ResidueHydrogenBond
        Residue-group results.

    Returns
    -------
    dict
        Mapping from residue key to residue-group result.

    Raises
    ------
    ValueError
        If duplicate residue keys are present.
    """

    mapping: Dict[
        ResidueContactKey,
        ResidueHydrogenBond,
    ] = {}

    for index, residue_group in enumerate(
        residue_groups
    ):
        if not isinstance(
            residue_group,
            ResidueHydrogenBond,
        ):
            raise TypeError(
                "All entries must be ResidueHydrogenBond instances. "
                f"Invalid entry at index {index}."
            )

        if residue_group.key in mapping:
            raise ValueError(
                "Duplicate residue key encountered while creating "
                f"the residue-group mapping: {residue_group.key!r}."
            )

        mapping[
            residue_group.key
        ] = residue_group

    return mapping


# -----------------------------------------------------------------------------
# Group filtering
# -----------------------------------------------------------------------------

def filter_residue_hydrogen_bond_groups(
    residue_groups: Iterable[
        ResidueHydrogenBond
    ],
    *,
    minimum_bond_count: int = 1,
    maximum_mean_distance: Optional[
        Number
    ] = None,
    require_explicit: bool = False,
    require_strong: bool = False,
    direction: Optional[
        HydrogenBondDirection
    ] = None,
) -> Tuple[
    ResidueHydrogenBond,
    ...,
]:
    """
    Filter residue-level hydrogen-bond groups.

    Parameters
    ----------
    residue_groups : iterable of ResidueHydrogenBond
        Residue groups.
    minimum_bond_count : int, optional
        Minimum number of hydrogen bonds.
    maximum_mean_distance : Number or None, optional
        Maximum mean D...A distance.
    require_explicit : bool, optional
        Whether each retained group must contain an explicit interaction.
    require_strong : bool, optional
        Whether each retained group must contain a strong interaction.
    direction : HydrogenBondDirection or None, optional
        Required interaction direction.

    Returns
    -------
    tuple of ResidueHydrogenBond
        Filtered residue groups.
    """

    normalized_minimum_count = (
        _optional_nonnegative_integer(
            minimum_bond_count,
            name="minimum bond count",
        )
    )

    if normalized_minimum_count is None:
        normalized_minimum_count = 1

    normalized_maximum_distance: Optional[
        np.float64
    ] = None

    if maximum_mean_distance is not None:
        normalized_maximum_distance = (
            _optional_float64(
                maximum_mean_distance,
                name="maximum mean distance",
                minimum=0.0,
            )
        )

    normalized_direction: Optional[
        HydrogenBondDirection
    ] = None

    if direction is not None:
        normalized_direction = (
            validate_hydrogen_bond_direction(
                direction
            )
        )

    filtered_groups: List[
        ResidueHydrogenBond
    ] = []

    for index, residue_group in enumerate(
        residue_groups
    ):
        if not isinstance(
            residue_group,
            ResidueHydrogenBond,
        ):
            raise TypeError(
                "All entries must be ResidueHydrogenBond instances. "
                f"Invalid entry at index {index}."
            )

        if (
            residue_group.hydrogen_bond_count
            < normalized_minimum_count
        ):
            continue

        if normalized_maximum_distance is not None:
            mean_distance = (
                residue_group.mean_distance
            )

            if (
                mean_distance is None
                or mean_distance
                > normalized_maximum_distance
            ):
                continue

        if (
            require_explicit
            and residue_group.explicit_count
            == 0
        ):
            continue

        if (
            require_strong
            and not residue_group.has_strong_bond
        ):
            continue

        if (
            normalized_direction is not None
            and normalized_direction
            not in residue_group.directions
        ):
            continue

        filtered_groups.append(
            residue_group
        )

    return tuple(
        filtered_groups
    )


# -----------------------------------------------------------------------------
# Residue-group summaries
# -----------------------------------------------------------------------------

def residue_hydrogen_bond_counts(
    residue_groups: Iterable[
        ResidueHydrogenBond
    ],
) -> Dict[
    ResidueContactKey,
    int,
]:
    """
    Return hydrogen-bond counts by residue key.

    Parameters
    ----------
    residue_groups : iterable of ResidueHydrogenBond
        Residue groups.

    Returns
    -------
    dict
        Residue-key-to-count mapping.
    """

    counts: Dict[
        ResidueContactKey,
        int,
    ] = {}

    for index, residue_group in enumerate(
        residue_groups
    ):
        if not isinstance(
            residue_group,
            ResidueHydrogenBond,
        ):
            raise TypeError(
                "All entries must be ResidueHydrogenBond instances. "
                f"Invalid entry at index {index}."
            )

        counts[
            residue_group.key
        ] = residue_group.hydrogen_bond_count

    return counts


def get_residue_hydrogen_bond_group(
    residue_groups: Iterable[
        ResidueHydrogenBond
    ],
    key: ResidueContactKey,
) -> Optional[
    ResidueHydrogenBond
]:
    """
    Return one residue group by key.

    Parameters
    ----------
    residue_groups : iterable of ResidueHydrogenBond
        Available residue groups.
    key : ResidueContactKey
        Requested residue key.

    Returns
    -------
    ResidueHydrogenBond or None
        Matching residue group.
    """

    normalized_key = _normalize_residue_key(
        key
    )

    if normalized_key is None:
        return None

    for residue_group in residue_groups:
        if not isinstance(
            residue_group,
            ResidueHydrogenBond,
        ):
            continue

        if residue_group.key == normalized_key:
            return residue_group

    return None


def get_top_hbond_residues(
    residue_groups: Iterable[
        ResidueHydrogenBond
    ],
    *,
    limit: int = 10,
) -> Tuple[
    ResidueHydrogenBond,
    ...,
]:
    """
    Return the residues with the highest hydrogen-bond counts.

    Parameters
    ----------
    residue_groups : iterable of ResidueHydrogenBond
        Residue groups.
    limit : int, optional
        Maximum number of groups returned.

    Returns
    -------
    tuple of ResidueHydrogenBond
        Top residue groups.

    Notes
    -----
    Ties are resolved using minimum D...A distance and then residue key.
    """

    normalized_limit = _optional_nonnegative_integer(
        limit,
        name="residue-group limit",
    )

    if normalized_limit is None or normalized_limit == 0:
        return ()

    normalized_groups = tuple(
        residue_groups
    )

    for index, residue_group in enumerate(
        normalized_groups
    ):
        if not isinstance(
            residue_group,
            ResidueHydrogenBond,
        ):
            raise TypeError(
                "All entries must be ResidueHydrogenBond instances. "
                f"Invalid entry at index {index}."
            )

    ordered_groups = sorted(
        normalized_groups,
        key=lambda group: (
            -group.hydrogen_bond_count,
            (
                np.float64(
                    np.inf
                )
                if group.minimum_distance is None
                else group.minimum_distance
            ),
            _residue_key_sort_key(
                group.key
            ),
        ),
    )

    return tuple(
        ordered_groups[
            :normalized_limit
        ]
    )


# -----------------------------------------------------------------------------
# HydrogenBondAnalysisResult integration
# -----------------------------------------------------------------------------

def attach_residue_hydrogen_bond_groups(
    result: HydrogenBondAnalysisResult,
    residue_groups: Iterable[
        ResidueHydrogenBond
    ],
    *,
    replace: bool = True,
    metadata: Optional[
        Mapping[
            str,
            Any,
        ]
    ] = None,
) -> HydrogenBondAnalysisResult:
    """
    Attach residue groups to a hydrogen-bond analysis result.

    Parameters
    ----------
    result : HydrogenBondAnalysisResult
        Original analysis result.
    residue_groups : iterable of ResidueHydrogenBond
        Residue groups to attach.
    replace : bool, optional
        Whether existing groups should be replaced. When ``False``, old and new
        groups are merged by residue key.
    metadata : mapping or None, optional
        Additional result metadata.

    Returns
    -------
    HydrogenBondAnalysisResult
        New immutable analysis result containing residue groups.
    """

    if not isinstance(
        result,
        HydrogenBondAnalysisResult,
    ):
        raise TypeError(
            "result must be a HydrogenBondAnalysisResult instance."
        )

    normalized_groups = tuple(
        residue_groups
    )

    for index, residue_group in enumerate(
        normalized_groups
    ):
        if not isinstance(
            residue_group,
            ResidueHydrogenBond,
        ):
            raise TypeError(
                "All residue_groups entries must be "
                "ResidueHydrogenBond instances. "
                f"Invalid entry at index {index}."
            )

    if replace:
        final_groups = normalized_groups

    else:
        merged_bonds: Dict[
            Tuple[
                ResidueContactKey,
                str,
            ],
            List[
                HydrogenBond
            ],
        ] = {}

        merged_residues: Dict[
            Tuple[
                ResidueContactKey,
                str,
            ],
            ResidueLike,
        ] = {}

        for residue_group in (
            tuple(
                result.residue_hydrogen_bonds
            )
            + normalized_groups
        ):
            group_identity = (
                residue_group.key,
                residue_group.side,
            )

            merged_bonds.setdefault(
                group_identity,
                [],
            ).extend(
                residue_group.hydrogen_bonds
            )

            merged_residues[
                group_identity
            ] = residue_group.residue

        merged_groups: List[
            ResidueHydrogenBond
        ] = []

        for (
            residue_key,
            group_side,
        ), group_bonds in sorted(
            merged_bonds.items(),
            key=lambda item: (
                item[
                    0
                ][
                    1
                ],
                _residue_key_sort_key(
                    item[
                        0
                    ][
                        0
                    ]
                ),
            ),
        ):
            merged_groups.append(
                ResidueHydrogenBond(
                    residue=merged_residues[
                        (
                            residue_key,
                            group_side,
                        )
                    ],
                    key=residue_key,
                    hydrogen_bonds=(
                        deduplicate_hydrogen_bonds(
                            group_bonds,
                            include_hydrogen_in_key=True,
                            prefer_explicit=True,
                        )
                    ),
                    side=group_side,
                    metadata={
                        "merged": True,
                    },
                )
            )

        final_groups = tuple(
            merged_groups
        )

    updated_metadata = dict(
        result.metadata
    )

    updated_metadata.update(
        {
            "residue_grouping_attached": True,
            "residue_group_count": len(
                final_groups
            ),
        }
    )

    if metadata:
        updated_metadata.update(
            metadata
        )

    return HydrogenBondAnalysisResult(
        hydrogen_bonds=(
            result.hydrogen_bonds
        ),
        residue_hydrogen_bonds=(
            final_groups
        ),
        ligand_atoms=result.ligand_atoms,
        receptor_atoms=(
            result.receptor_atoms
        ),
        donor_acceptor_cutoff=(
            result.donor_acceptor_cutoff
        ),
        hydrogen_acceptor_cutoff=(
            result.hydrogen_acceptor_cutoff
        ),
        minimum_dha_angle=(
            result.minimum_dha_angle
        ),
        minimum_inferred_angle=(
            result.minimum_inferred_angle
        ),
        statistics=result.statistics,
        metadata=updated_metadata,
    )


def group_analysis_hydrogen_bonds_by_residue(
    result: HydrogenBondAnalysisResult,
    *,
    side: str = DEFAULT_RESIDUE_GROUP_SIDE,
    include_unresolved: bool = (
        DEFAULT_INCLUDE_RESIDUELESS_HBONDS
    ),
    sort_groups: bool = (
        DEFAULT_SORT_RESIDUE_GROUPS
    ),
    sort_bonds: bool = (
        DEFAULT_SORT_BONDS_WITHIN_RESIDUE
    ),
    deduplicate_bonds: bool = (
        DEFAULT_DEDUPLICATE_WITHIN_RESIDUE
    ),
) -> HydrogenBondAnalysisResult:
    """
    Group all interactions in an analysis result by residue.

    Parameters
    ----------
    result : HydrogenBondAnalysisResult
        Hydrogen-bond analysis result.
    side : str, optional
        Grouping side.
    include_unresolved : bool, optional
        Whether unresolved residue groups should be retained.
    sort_groups : bool, optional
        Whether groups should be sorted.
    sort_bonds : bool, optional
        Whether bonds inside groups should be sorted.
    deduplicate_bonds : bool, optional
        Whether bonds inside each group should be deduplicated.

    Returns
    -------
    HydrogenBondAnalysisResult
        New result containing residue groups.
    """

    if not isinstance(
        result,
        HydrogenBondAnalysisResult,
    ):
        raise TypeError(
            "result must be a HydrogenBondAnalysisResult instance."
        )

    normalized_side = validate_residue_group_side(
        side
    )

    residue_groups = group_hydrogen_bonds_by_residue(
        result.hydrogen_bonds,
        side=normalized_side,
        include_unresolved=(
            include_unresolved
        ),
        sort_groups=sort_groups,
        sort_bonds=sort_bonds,
        deduplicate_bonds=(
            deduplicate_bonds
        ),
    )

    return attach_residue_hydrogen_bond_groups(
        result,
        residue_groups,
        replace=True,
        metadata={
            "residue_grouping_side": (
                normalized_side
            ),
            "include_unresolved_residues": bool(
                include_unresolved
            ),
        },
    )


def analyze_and_group_hydrogen_bonds(
    ligand_atoms: Iterable[
        AtomLike
    ],
    receptor_atoms: Iterable[
        AtomLike
    ],
    *,
    residue_side: str = (
        DEFAULT_RESIDUE_GROUP_SIDE
    ),
    include_unresolved_residues: bool = (
        DEFAULT_INCLUDE_RESIDUELESS_HBONDS
    ),
    **analysis_kwargs: Any,
) -> HydrogenBondAnalysisResult:
    """
    Detect hydrogen bonds and immediately group them by residue.

    Parameters
    ----------
    ligand_atoms : iterable of atom-like
        Ligand atoms.
    receptor_atoms : iterable of atom-like
        Receptor atoms.
    residue_side : str, optional
        Residue grouping side.
    include_unresolved_residues : bool, optional
        Whether unresolved residues should be included.
    **analysis_kwargs : Any
        Additional arguments forwarded to :func:`analyze_hydrogen_bonds`.

    Returns
    -------
    HydrogenBondAnalysisResult
        Analysis result with residue groups attached.
    """

    result = analyze_hydrogen_bonds(
        ligand_atoms,
        receptor_atoms,
        **analysis_kwargs,
    )

    return group_analysis_hydrogen_bonds_by_residue(
        result,
        side=residue_side,
        include_unresolved=(
            include_unresolved_residues
        ),
    )


# -----------------------------------------------------------------------------
# Public interface
# -----------------------------------------------------------------------------

_SECTION_8_PUBLIC_NAMES: Final[
    Tuple[
        str,
        ...,
    ]
] = (
    "RESIDUE_GROUP_SIDE_DONOR",
    "RESIDUE_GROUP_SIDE_ACCEPTOR",
    "RESIDUE_GROUP_SIDE_RECEPTOR",
    "RESIDUE_GROUP_SIDE_LIGAND",
    "RESIDUE_GROUP_SIDE_EITHER",
    "VALID_RESIDUE_GROUP_SIDES",
    "DEFAULT_RESIDUE_GROUP_SIDE",
    "DEFAULT_INCLUDE_RESIDUELESS_HBONDS",
    "DEFAULT_SORT_RESIDUE_GROUPS",
    "DEFAULT_SORT_BONDS_WITHIN_RESIDUE",
    "DEFAULT_DEDUPLICATE_WITHIN_RESIDUE",
    "validate_residue_group_side",
    "get_hbond_receptor_residue",
    "get_hbond_ligand_residue",
    "get_hbond_receptor_residue_key",
    "get_hbond_ligand_residue_key",
    "get_hbond_residue_for_side",
    "get_hbond_residue_key_for_side",
    "get_hbond_residue_entries",
    "group_hydrogen_bonds_by_residue",
    "map_hydrogen_bonds_by_residue_key",
    "map_residue_hydrogen_bond_results",
    "filter_residue_hydrogen_bond_groups",
    "residue_hydrogen_bond_counts",
    "get_residue_hydrogen_bond_group",
    "get_top_hbond_residues",
    "attach_residue_hydrogen_bond_groups",
    "group_analysis_hydrogen_bonds_by_residue",
    "analyze_and_group_hydrogen_bonds",
)

for public_name in _SECTION_8_PUBLIC_NAMES:
    if public_name not in __all__:
        __all__.append(
            public_name
        )


# =============================================================================
# End of Section 8
# =============================================================================


# =============================================================================
# Section 9 — Geometric classification and strength
# =============================================================================


# -----------------------------------------------------------------------------
# Classification labels
# -----------------------------------------------------------------------------

HBOND_TYPE_STRONG: Final[
    HydrogenBondClassification
] = "strong"

HBOND_TYPE_MODERATE: Final[
    HydrogenBondClassification
] = "moderate"

HBOND_TYPE_WEAK: Final[
    HydrogenBondClassification
] = "weak"

HBOND_TYPE_GEOMETRIC_ONLY: Final[
    HydrogenBondClassification
] = "geometric_only"

HBOND_TYPE_REJECTED: Final[
    HydrogenBondClassification
] = "rejected"

HBOND_TYPE_UNKNOWN: Final[
    HydrogenBondClassification
] = "unknown"


VALID_HYDROGEN_BOND_CLASSIFICATIONS: Final[
    FrozenSet[
        HydrogenBondClassification
    ]
] = frozenset(
    {
        HBOND_TYPE_STRONG,
        HBOND_TYPE_MODERATE,
        HBOND_TYPE_WEAK,
        HBOND_TYPE_GEOMETRIC_ONLY,
        HBOND_TYPE_REJECTED,
        HBOND_TYPE_UNKNOWN,
    }
)


# -----------------------------------------------------------------------------
# Classification cutoffs
# -----------------------------------------------------------------------------

# Explicit hydrogen-bond thresholds.
#
# Strong:
#     D...A <= 3.0 Å
#     H...A <= 2.2 Å
#     D-H...A >= 150°
#
# Moderate:
#     D...A <= 3.3 Å
#     H...A <= 2.5 Å
#     D-H...A >= 130°
#
# Weak:
#     D...A <= 3.6 Å
#     H...A <= 2.8 Å
#     D-H...A >= 110°

DEFAULT_STRONG_DONOR_ACCEPTOR_DISTANCE: Final[
    np.float64
] = np.float64(
    3.00
)

DEFAULT_MODERATE_DONOR_ACCEPTOR_DISTANCE: Final[
    np.float64
] = np.float64(
    3.30
)

DEFAULT_WEAK_DONOR_ACCEPTOR_DISTANCE: Final[
    np.float64
] = np.float64(
    3.60
)


DEFAULT_STRONG_HYDROGEN_ACCEPTOR_DISTANCE: Final[
    np.float64
] = np.float64(
    2.20
)

DEFAULT_MODERATE_HYDROGEN_ACCEPTOR_DISTANCE: Final[
    np.float64
] = np.float64(
    2.50
)

DEFAULT_WEAK_HYDROGEN_ACCEPTOR_DISTANCE: Final[
    np.float64
] = np.float64(
    2.80
)


DEFAULT_STRONG_DHA_ANGLE: Final[
    np.float64
] = np.float64(
    150.0
)

DEFAULT_MODERATE_DHA_ANGLE: Final[
    np.float64
] = np.float64(
    130.0
)

DEFAULT_WEAK_DHA_ANGLE: Final[
    np.float64
] = np.float64(
    110.0
)


# In inferred mode, donor_angle and acceptor_angle are deviation angles:
#
#     0° = ideal alignment
#
# Therefore, smaller angles indicate better geometry.
DEFAULT_STRONG_INFERRED_DEVIATION: Final[
    np.float64
] = np.float64(
    30.0
)

DEFAULT_MODERATE_INFERRED_DEVIATION: Final[
    np.float64
] = np.float64(
    50.0
)

DEFAULT_WEAK_INFERRED_DEVIATION: Final[
    np.float64
] = np.float64(
    70.0
)


# -----------------------------------------------------------------------------
# Geometric-strength score constants
# -----------------------------------------------------------------------------

DEFAULT_DISTANCE_SCORE_WEIGHT: Final[
    np.float64
] = np.float64(
    0.45
)

DEFAULT_ANGLE_SCORE_WEIGHT: Final[
    np.float64
] = np.float64(
    0.40
)

DEFAULT_ACCEPTOR_ANGLE_SCORE_WEIGHT: Final[
    np.float64
] = np.float64(
    0.15
)


DEFAULT_EXPLICIT_MODE_SCORE_FACTOR: Final[
    np.float64
] = np.float64(
    1.00
)

DEFAULT_INFERRED_MODE_SCORE_FACTOR: Final[
    np.float64
] = np.float64(
    0.85
)


DEFAULT_AMBIGUOUS_ASSIGNMENT_SCORE_FACTOR: Final[
    np.float64
] = np.float64(
    0.90
)

DEFAULT_MISSING_ANGLE_SCORE_FACTOR: Final[
    np.float64
] = np.float64(
    0.80
)


DEFAULT_STRONG_SCORE_THRESHOLD: Final[
    np.float64
] = np.float64(
    0.78
)

DEFAULT_MODERATE_SCORE_THRESHOLD: Final[
    np.float64
] = np.float64(
    0.55
)

DEFAULT_WEAK_SCORE_THRESHOLD: Final[
    np.float64
] = np.float64(
    0.30
)


# -----------------------------------------------------------------------------
# Classification configuration
# -----------------------------------------------------------------------------

@dataclass(
    frozen=True,
    slots=True,
)
class HydrogenBondClassificationConfig:
    """
    Configuration for geometric hydrogen-bond classification.

    Parameters
    ----------
    strong_donor_acceptor_distance : Number
        Maximum strong D...A distance.
    moderate_donor_acceptor_distance : Number
        Maximum moderate D...A distance.
    weak_donor_acceptor_distance : Number
        Maximum weak D...A distance.
    strong_hydrogen_acceptor_distance : Number
        Maximum strong H...A distance.
    moderate_hydrogen_acceptor_distance : Number
        Maximum moderate H...A distance.
    weak_hydrogen_acceptor_distance : Number
        Maximum weak H...A distance.
    strong_dha_angle : Number
        Minimum strong D-H...A angle.
    moderate_dha_angle : Number
        Minimum moderate D-H...A angle.
    weak_dha_angle : Number
        Minimum weak D-H...A angle.
    strong_inferred_deviation : Number
        Maximum strong inferred deviation angle.
    moderate_inferred_deviation : Number
        Maximum moderate inferred deviation angle.
    weak_inferred_deviation : Number
        Maximum weak inferred deviation angle.
    strong_score_threshold : Number
        Minimum score for strong classification.
    moderate_score_threshold : Number
        Minimum score for moderate classification.
    weak_score_threshold : Number
        Minimum score for weak classification.
    distance_weight : Number
        Weight assigned to distance geometry.
    angle_weight : Number
        Weight assigned to donor-side angular geometry.
    acceptor_angle_weight : Number
        Weight assigned to acceptor-side angular geometry.
    inferred_mode_factor : Number
        Multiplicative penalty applied to inferred geometry.
    ambiguous_assignment_factor : Number
        Multiplicative penalty applied to ambiguous D-H assignments.
    missing_angle_factor : Number
        Multiplicative penalty applied when optional angular data are absent.
    require_threshold_consistency : bool
        Whether score and discrete geometric thresholds must agree.
    metadata : mapping
        Additional configuration metadata.

    Notes
    -----
    The resulting score is a normalized geometric quality index between zero
    and one. It is not an interaction energy or binding affinity.
    """

    strong_donor_acceptor_distance: np.float64 = (
        DEFAULT_STRONG_DONOR_ACCEPTOR_DISTANCE
    )

    moderate_donor_acceptor_distance: np.float64 = (
        DEFAULT_MODERATE_DONOR_ACCEPTOR_DISTANCE
    )

    weak_donor_acceptor_distance: np.float64 = (
        DEFAULT_WEAK_DONOR_ACCEPTOR_DISTANCE
    )

    strong_hydrogen_acceptor_distance: np.float64 = (
        DEFAULT_STRONG_HYDROGEN_ACCEPTOR_DISTANCE
    )

    moderate_hydrogen_acceptor_distance: np.float64 = (
        DEFAULT_MODERATE_HYDROGEN_ACCEPTOR_DISTANCE
    )

    weak_hydrogen_acceptor_distance: np.float64 = (
        DEFAULT_WEAK_HYDROGEN_ACCEPTOR_DISTANCE
    )

    strong_dha_angle: np.float64 = (
        DEFAULT_STRONG_DHA_ANGLE
    )

    moderate_dha_angle: np.float64 = (
        DEFAULT_MODERATE_DHA_ANGLE
    )

    weak_dha_angle: np.float64 = (
        DEFAULT_WEAK_DHA_ANGLE
    )

    strong_inferred_deviation: np.float64 = (
        DEFAULT_STRONG_INFERRED_DEVIATION
    )

    moderate_inferred_deviation: np.float64 = (
        DEFAULT_MODERATE_INFERRED_DEVIATION
    )

    weak_inferred_deviation: np.float64 = (
        DEFAULT_WEAK_INFERRED_DEVIATION
    )

    strong_score_threshold: np.float64 = (
        DEFAULT_STRONG_SCORE_THRESHOLD
    )

    moderate_score_threshold: np.float64 = (
        DEFAULT_MODERATE_SCORE_THRESHOLD
    )

    weak_score_threshold: np.float64 = (
        DEFAULT_WEAK_SCORE_THRESHOLD
    )

    distance_weight: np.float64 = (
        DEFAULT_DISTANCE_SCORE_WEIGHT
    )

    angle_weight: np.float64 = (
        DEFAULT_ANGLE_SCORE_WEIGHT
    )

    acceptor_angle_weight: np.float64 = (
        DEFAULT_ACCEPTOR_ANGLE_SCORE_WEIGHT
    )

    explicit_mode_factor: np.float64 = (
        DEFAULT_EXPLICIT_MODE_SCORE_FACTOR
    )

    inferred_mode_factor: np.float64 = (
        DEFAULT_INFERRED_MODE_SCORE_FACTOR
    )

    ambiguous_assignment_factor: np.float64 = (
        DEFAULT_AMBIGUOUS_ASSIGNMENT_SCORE_FACTOR
    )

    missing_angle_factor: np.float64 = (
        DEFAULT_MISSING_ANGLE_SCORE_FACTOR
    )

    require_threshold_consistency: bool = True

    metadata: Mapping[
        str,
        Any,
    ] = field(
        default_factory=lambda: _EMPTY_METADATA,
        repr=False,
        compare=False,
    )

    def __post_init__(
        self,
    ) -> None:
        """Validate and normalize classification parameters."""

        distance_fields = (
            "strong_donor_acceptor_distance",
            "moderate_donor_acceptor_distance",
            "weak_donor_acceptor_distance",
            "strong_hydrogen_acceptor_distance",
            "moderate_hydrogen_acceptor_distance",
            "weak_hydrogen_acceptor_distance",
        )

        angle_fields = (
            "strong_dha_angle",
            "moderate_dha_angle",
            "weak_dha_angle",
            "strong_inferred_deviation",
            "moderate_inferred_deviation",
            "weak_inferred_deviation",
        )

        score_fields = (
            "strong_score_threshold",
            "moderate_score_threshold",
            "weak_score_threshold",
            "distance_weight",
            "angle_weight",
            "acceptor_angle_weight",
            "explicit_mode_factor",
            "inferred_mode_factor",
            "ambiguous_assignment_factor",
            "missing_angle_factor",
        )

        for field_name in distance_fields:
            value = _optional_float64(
                getattr(
                    self,
                    field_name,
                ),
                name=field_name.replace(
                    "_",
                    " ",
                ),
                minimum=0.0,
            )

            if value is None:
                raise ValueError(
                    f"{field_name} cannot be None."
                )

            object.__setattr__(
                self,
                field_name,
                value,
            )

        for field_name in angle_fields:
            value = _optional_float64(
                getattr(
                    self,
                    field_name,
                ),
                name=field_name.replace(
                    "_",
                    " ",
                ),
                minimum=MINIMUM_VALID_ANGLE_DEGREES,
                maximum=MAXIMUM_VALID_ANGLE_DEGREES,
            )

            if value is None:
                raise ValueError(
                    f"{field_name} cannot be None."
                )

            object.__setattr__(
                self,
                field_name,
                value,
            )

        for field_name in score_fields:
            value = _optional_float64(
                getattr(
                    self,
                    field_name,
                ),
                name=field_name.replace(
                    "_",
                    " ",
                ),
                minimum=0.0,
                maximum=1.0,
            )

            if value is None:
                raise ValueError(
                    f"{field_name} cannot be None."
                )

            object.__setattr__(
                self,
                field_name,
                value,
            )

        if not (
            self.strong_donor_acceptor_distance
            <= self.moderate_donor_acceptor_distance
            <= self.weak_donor_acceptor_distance
        ):
            raise ValueError(
                "D...A distance thresholds must satisfy "
                "strong <= moderate <= weak."
            )

        if not (
            self.strong_hydrogen_acceptor_distance
            <= self.moderate_hydrogen_acceptor_distance
            <= self.weak_hydrogen_acceptor_distance
        ):
            raise ValueError(
                "H...A distance thresholds must satisfy "
                "strong <= moderate <= weak."
            )

        if not (
            self.strong_dha_angle
            >= self.moderate_dha_angle
            >= self.weak_dha_angle
        ):
            raise ValueError(
                "D-H...A angle thresholds must satisfy "
                "strong >= moderate >= weak."
            )

        if not (
            self.strong_inferred_deviation
            <= self.moderate_inferred_deviation
            <= self.weak_inferred_deviation
        ):
            raise ValueError(
                "Inferred deviation thresholds must satisfy "
                "strong <= moderate <= weak."
            )

        if not (
            self.strong_score_threshold
            >= self.moderate_score_threshold
            >= self.weak_score_threshold
        ):
            raise ValueError(
                "Score thresholds must satisfy "
                "strong >= moderate >= weak."
            )

        total_weight = np.float64(
            self.distance_weight
            + self.angle_weight
            + self.acceptor_angle_weight
        )

        if total_weight <= 0.0:
            raise ValueError(
                "At least one geometric score weight must be positive."
            )

        object.__setattr__(
            self,
            "require_threshold_consistency",
            bool(
                self.require_threshold_consistency
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
    def total_weight(
        self,
    ) -> np.float64:
        """
        Return the sum of geometric score weights.

        Returns
        -------
        numpy.float64
            Total weight.
        """

        return np.float64(
            self.distance_weight
            + self.angle_weight
            + self.acceptor_angle_weight
        )

    def to_dict(
        self,
    ) -> Dict[
        str,
        Any,
    ]:
        """
        Serialize the classification configuration.

        Returns
        -------
        dict
            Serializable configuration.
        """

        return {
            field_name: (
                bool(
                    value
                )
                if isinstance(
                    value,
                    (
                        bool,
                        np.bool_,
                    ),
                )
                else float(
                    value
                )
                if isinstance(
                    value,
                    (
                        float,
                        np.floating,
                    ),
                )
                else value
            )
            for field_name, value
            in (
                (
                    "strong_donor_acceptor_distance",
                    self.strong_donor_acceptor_distance,
                ),
                (
                    "moderate_donor_acceptor_distance",
                    self.moderate_donor_acceptor_distance,
                ),
                (
                    "weak_donor_acceptor_distance",
                    self.weak_donor_acceptor_distance,
                ),
                (
                    "strong_hydrogen_acceptor_distance",
                    self.strong_hydrogen_acceptor_distance,
                ),
                (
                    "moderate_hydrogen_acceptor_distance",
                    self.moderate_hydrogen_acceptor_distance,
                ),
                (
                    "weak_hydrogen_acceptor_distance",
                    self.weak_hydrogen_acceptor_distance,
                ),
                (
                    "strong_dha_angle",
                    self.strong_dha_angle,
                ),
                (
                    "moderate_dha_angle",
                    self.moderate_dha_angle,
                ),
                (
                    "weak_dha_angle",
                    self.weak_dha_angle,
                ),
                (
                    "strong_inferred_deviation",
                    self.strong_inferred_deviation,
                ),
                (
                    "moderate_inferred_deviation",
                    self.moderate_inferred_deviation,
                ),
                (
                    "weak_inferred_deviation",
                    self.weak_inferred_deviation,
                ),
                (
                    "strong_score_threshold",
                    self.strong_score_threshold,
                ),
                (
                    "moderate_score_threshold",
                    self.moderate_score_threshold,
                ),
                (
                    "weak_score_threshold",
                    self.weak_score_threshold,
                ),
                (
                    "distance_weight",
                    self.distance_weight,
                ),
                (
                    "angle_weight",
                    self.angle_weight,
                ),
                (
                    "acceptor_angle_weight",
                    self.acceptor_angle_weight,
                ),
                (
                    "explicit_mode_factor",
                    self.explicit_mode_factor,
                ),
                (
                    "inferred_mode_factor",
                    self.inferred_mode_factor,
                ),
                (
                    "ambiguous_assignment_factor",
                    self.ambiguous_assignment_factor,
                ),
                (
                    "missing_angle_factor",
                    self.missing_angle_factor,
                ),
                (
                    "require_threshold_consistency",
                    self.require_threshold_consistency,
                ),
            )
        } | {
            "metadata": dict(
                self.metadata
            )
        }


DEFAULT_HBOND_CLASSIFICATION_CONFIG: Final[
    HydrogenBondClassificationConfig
] = HydrogenBondClassificationConfig()


# -----------------------------------------------------------------------------
# Classification result
# -----------------------------------------------------------------------------

@dataclass(
    frozen=True,
    slots=True,
)
class HydrogenBondStrength:
    """
    Geometric-strength assessment for one hydrogen bond.

    Parameters
    ----------
    classification : HydrogenBondClassification
        Strong, moderate, weak, geometric-only, rejected or unknown.
    score : Number
        Normalized geometric score between zero and one.
    distance_score : Number
        Distance contribution.
    donor_angle_score : Number or None
        Donor-side angular contribution.
    acceptor_angle_score : Number or None
        Acceptor-side angular contribution.
    mode_factor : Number
        Explicit/inferred confidence factor.
    ambiguity_factor : Number
        Hydrogen-assignment ambiguity factor.
    threshold_classification : HydrogenBondClassification
        Classification based only on discrete geometric thresholds.
    score_classification : HydrogenBondClassification
        Classification based only on the continuous score.
    limiting_criterion : str or None
        Criterion that most strongly limited the result.
    passed_criteria : sequence of str
        Criteria satisfied.
    failed_criteria : sequence of str
        Criteria not satisfied.
    metadata : mapping
        Additional classification information.

    Notes
    -----
    The score is a descriptive geometric index and must not be interpreted as
    an energetic quantity.
    """

    classification: HydrogenBondClassification
    score: np.float64

    distance_score: np.float64

    donor_angle_score: Optional[
        np.float64
    ] = None

    acceptor_angle_score: Optional[
        np.float64
    ] = None

    mode_factor: np.float64 = np.float64(
        1.0
    )

    ambiguity_factor: np.float64 = np.float64(
        1.0
    )

    threshold_classification: HydrogenBondClassification = (
        HBOND_TYPE_UNKNOWN
    )

    score_classification: HydrogenBondClassification = (
        HBOND_TYPE_UNKNOWN
    )

    limiting_criterion: Optional[
        str
    ] = None

    passed_criteria: Sequence[
        str
    ] = field(
        default_factory=tuple
    )

    failed_criteria: Sequence[
        str
    ] = field(
        default_factory=tuple
    )

    metadata: Mapping[
        str,
        Any,
    ] = field(
        default_factory=lambda: _EMPTY_METADATA,
        repr=False,
        compare=False,
    )

    def __post_init__(
        self,
    ) -> None:
        """Validate and normalize strength information."""

        object.__setattr__(
            self,
            "classification",
            validate_hydrogen_bond_classification(
                self.classification
            ),
        )

        object.__setattr__(
            self,
            "threshold_classification",
            validate_hydrogen_bond_classification(
                self.threshold_classification
            ),
        )

        object.__setattr__(
            self,
            "score_classification",
            validate_hydrogen_bond_classification(
                self.score_classification
            ),
        )

        for field_name in (
            "score",
            "distance_score",
            "mode_factor",
            "ambiguity_factor",
        ):
            normalized_value = _optional_float64(
                getattr(
                    self,
                    field_name,
                ),
                name=field_name.replace(
                    "_",
                    " ",
                ),
                minimum=0.0,
                maximum=1.0,
            )

            if normalized_value is None:
                raise ValueError(
                    f"{field_name} cannot be None."
                )

            object.__setattr__(
                self,
                field_name,
                normalized_value,
            )

        for field_name in (
            "donor_angle_score",
            "acceptor_angle_score",
        ):
            value = getattr(
                self,
                field_name,
            )

            if value is None:
                continue

            normalized_value = _optional_float64(
                value,
                name=field_name.replace(
                    "_",
                    " ",
                ),
                minimum=0.0,
                maximum=1.0,
            )

            object.__setattr__(
                self,
                field_name,
                normalized_value,
            )

        object.__setattr__(
            self,
            "limiting_criterion",
            (
                None
                if self.limiting_criterion is None
                else str(
                    self.limiting_criterion
                ).strip()
                or None
            ),
        )

        object.__setattr__(
            self,
            "passed_criteria",
            tuple(
                str(
                    criterion
                ).strip()
                for criterion in self.passed_criteria
                if str(
                    criterion
                ).strip()
            ),
        )

        object.__setattr__(
            self,
            "failed_criteria",
            tuple(
                str(
                    criterion
                ).strip()
                for criterion in self.failed_criteria
                if str(
                    criterion
                ).strip()
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
    def is_strong(
        self,
    ) -> bool:
        """Whether the interaction is classified as strong."""

        return self.classification == HBOND_TYPE_STRONG

    @property
    def is_moderate(
        self,
    ) -> bool:
        """Whether the interaction is classified as moderate."""

        return self.classification == HBOND_TYPE_MODERATE

    @property
    def is_weak(
        self,
    ) -> bool:
        """Whether the interaction is classified as weak."""

        return self.classification == HBOND_TYPE_WEAK

    @property
    def is_accepted(
        self,
    ) -> bool:
        """Whether the interaction received an accepted classification."""

        return self.classification in {
            HBOND_TYPE_STRONG,
            HBOND_TYPE_MODERATE,
            HBOND_TYPE_WEAK,
            HBOND_TYPE_GEOMETRIC_ONLY,
        }

    def to_dict(
        self,
    ) -> Dict[
        str,
        Any,
    ]:
        """
        Serialize the strength result.

        Returns
        -------
        dict
            Serializable representation.
        """

        return {
            "classification": self.classification,
            "score": float(
                self.score
            ),
            "distance_score": float(
                self.distance_score
            ),
            "donor_angle_score": (
                None
                if self.donor_angle_score is None
                else float(
                    self.donor_angle_score
                )
            ),
            "acceptor_angle_score": (
                None
                if self.acceptor_angle_score is None
                else float(
                    self.acceptor_angle_score
                )
            ),
            "mode_factor": float(
                self.mode_factor
            ),
            "ambiguity_factor": float(
                self.ambiguity_factor
            ),
            "threshold_classification": (
                self.threshold_classification
            ),
            "score_classification": (
                self.score_classification
            ),
            "limiting_criterion": (
                self.limiting_criterion
            ),
            "passed_criteria": list(
                self.passed_criteria
            ),
            "failed_criteria": list(
                self.failed_criteria
            ),
            "metadata": dict(
                self.metadata
            ),
        }


# -----------------------------------------------------------------------------
# Classification validation
# -----------------------------------------------------------------------------

def validate_hydrogen_bond_classification(
    classification: str,
) -> HydrogenBondClassification:
    """
    Validate a hydrogen-bond classification label.

    Parameters
    ----------
    classification : str
        Classification label.

    Returns
    -------
    HydrogenBondClassification
        Normalized classification.

    Raises
    ------
    TypeError
        If the label is not a string.
    ValueError
        If the label is unsupported.
    """

    if not isinstance(
        classification,
        str,
    ):
        raise TypeError(
            "Hydrogen-bond classification must be a string."
        )

    normalized_classification = (
        classification.strip().lower()
    )

    if (
        normalized_classification
        not in VALID_HYDROGEN_BOND_CLASSIFICATIONS
    ):
        raise ValueError(
            "Unsupported hydrogen-bond classification "
            f"{classification!r}. Expected one of: "
            f"{', '.join(sorted(VALID_HYDROGEN_BOND_CLASSIFICATIONS))}."
        )

    return normalized_classification


# -----------------------------------------------------------------------------
# Continuous score helpers
# -----------------------------------------------------------------------------

def _clamp_unit_interval(
    value: Number,
) -> np.float64:
    """
    Clamp a numeric value to the interval ``[0, 1]``.

    Parameters
    ----------
    value : Number
        Numeric value.

    Returns
    -------
    numpy.float64
        Clamped value.
    """

    try:
        numeric_value = np.float64(
            value
        )

    except (
        TypeError,
        ValueError,
        OverflowError,
    ):
        return np.float64(
            0.0
        )

    if not np.isfinite(
        numeric_value
    ):
        return np.float64(
            0.0
        )

    return np.float64(
        np.clip(
            numeric_value,
            0.0,
            1.0,
        )
    )


def _linear_smaller_is_better_score(
    value: Number,
    *,
    ideal: Number,
    maximum: Number,
) -> np.float64:
    """
    Score a quantity for which smaller values are better.

    Parameters
    ----------
    value : Number
        Observed value.
    ideal : Number
        Value receiving a score of one.
    maximum : Number
        Value receiving a score of zero.

    Returns
    -------
    numpy.float64
        Normalized score.
    """

    normalized_value = np.float64(
        value
    )

    normalized_ideal = np.float64(
        ideal
    )

    normalized_maximum = np.float64(
        maximum
    )

    if (
        not np.isfinite(
            normalized_value
        )
        or not np.isfinite(
            normalized_ideal
        )
        or not np.isfinite(
            normalized_maximum
        )
        or normalized_maximum
        <= normalized_ideal
    ):
        return np.float64(
            0.0
        )

    if normalized_value <= normalized_ideal:
        return np.float64(
            1.0
        )

    if normalized_value >= normalized_maximum:
        return np.float64(
            0.0
        )

    return _clamp_unit_interval(
        (
            normalized_maximum
            - normalized_value
        )
        / (
            normalized_maximum
            - normalized_ideal
        )
    )


def _linear_larger_is_better_score(
    value: Number,
    *,
    minimum: Number,
    ideal: Number,
) -> np.float64:
    """
    Score a quantity for which larger values are better.

    Parameters
    ----------
    value : Number
        Observed value.
    minimum : Number
        Value receiving a score of zero.
    ideal : Number
        Value receiving a score of one.

    Returns
    -------
    numpy.float64
        Normalized score.
    """

    normalized_value = np.float64(
        value
    )

    normalized_minimum = np.float64(
        minimum
    )

    normalized_ideal = np.float64(
        ideal
    )

    if (
        not np.isfinite(
            normalized_value
        )
        or not np.isfinite(
            normalized_minimum
        )
        or not np.isfinite(
            normalized_ideal
        )
        or normalized_ideal
        <= normalized_minimum
    ):
        return np.float64(
            0.0
        )

    if normalized_value <= normalized_minimum:
        return np.float64(
            0.0
        )

    if normalized_value >= normalized_ideal:
        return np.float64(
            1.0
        )

    return _clamp_unit_interval(
        (
            normalized_value
            - normalized_minimum
        )
        / (
            normalized_ideal
            - normalized_minimum
        )
    )


def calculate_distance_geometry_score(
    geometry: HydrogenBondGeometry,
    *,
    config: HydrogenBondClassificationConfig = (
        DEFAULT_HBOND_CLASSIFICATION_CONFIG
    ),
) -> np.float64:
    """
    Calculate the distance component of geometric strength.

    Parameters
    ----------
    geometry : HydrogenBondGeometry
        Hydrogen-bond geometry.
    config : HydrogenBondClassificationConfig, optional
        Classification configuration.

    Returns
    -------
    numpy.float64
        Distance score between zero and one.

    Notes
    -----
    In explicit mode, D...A and H...A scores are averaged. In inferred mode,
    only D...A is available.
    """

    if not isinstance(
        geometry,
        HydrogenBondGeometry,
    ):
        raise TypeError(
            "geometry must be a HydrogenBondGeometry instance."
        )

    if not isinstance(
        config,
        HydrogenBondClassificationConfig,
    ):
        raise TypeError(
            "config must be a HydrogenBondClassificationConfig."
        )

    donor_acceptor_score = (
        _linear_smaller_is_better_score(
            geometry.donor_acceptor_distance,
            ideal=np.float64(
                2.60
            ),
            maximum=(
                config.weak_donor_acceptor_distance
            ),
        )
    )

    if geometry.hydrogen_acceptor_distance is None:
        return donor_acceptor_score

    hydrogen_acceptor_score = (
        _linear_smaller_is_better_score(
            geometry.hydrogen_acceptor_distance,
            ideal=np.float64(
                1.60
            ),
            maximum=(
                config.weak_hydrogen_acceptor_distance
            ),
        )
    )

    return np.float64(
        (
            donor_acceptor_score
            + hydrogen_acceptor_score
        )
        / np.float64(
            2.0
        )
    )


def calculate_donor_angle_geometry_score(
    geometry: HydrogenBondGeometry,
    *,
    config: HydrogenBondClassificationConfig = (
        DEFAULT_HBOND_CLASSIFICATION_CONFIG
    ),
) -> Optional[
    np.float64
]:
    """
    Calculate the donor-side angular score.

    Parameters
    ----------
    geometry : HydrogenBondGeometry
        Hydrogen-bond geometry.
    config : HydrogenBondClassificationConfig, optional
        Classification configuration.

    Returns
    -------
    numpy.float64 or None
        Donor angular score.

    Notes
    -----
    Explicit geometry uses the D-H...A angle, for which larger is better.
    Inferred geometry uses ``donor_angle``, a deviation for which smaller is
    better.
    """

    if geometry.dha_angle is not None:
        return _linear_larger_is_better_score(
            geometry.dha_angle,
            minimum=config.weak_dha_angle,
            ideal=STRAIGHT_ANGLE_DEGREES,
        )

    if geometry.donor_angle is not None:
        return _linear_smaller_is_better_score(
            geometry.donor_angle,
            ideal=np.float64(
                0.0
            ),
            maximum=(
                config.weak_inferred_deviation
            ),
        )

    return None


def calculate_acceptor_angle_geometry_score(
    geometry: HydrogenBondGeometry,
    *,
    config: HydrogenBondClassificationConfig = (
        DEFAULT_HBOND_CLASSIFICATION_CONFIG
    ),
) -> Optional[
    np.float64
]:
    """
    Calculate the acceptor-side angular score.

    Parameters
    ----------
    geometry : HydrogenBondGeometry
        Hydrogen-bond geometry.
    config : HydrogenBondClassificationConfig, optional
        Classification configuration.

    Returns
    -------
    numpy.float64 or None
        Acceptor angular score.

    Notes
    -----
    Acceptor angles are stored as deviation angles, with zero representing
    optimal alignment.
    """

    if geometry.acceptor_angle is None:
        return None

    return _linear_smaller_is_better_score(
        geometry.acceptor_angle,
        ideal=np.float64(
            0.0
        ),
        maximum=config.weak_inferred_deviation,
    )


def calculate_hydrogen_bond_geometric_score(
    hydrogen_bond: HydrogenBond,
    *,
    config: HydrogenBondClassificationConfig = (
        DEFAULT_HBOND_CLASSIFICATION_CONFIG
    ),
) -> HydrogenBondStrength:
    """
    Calculate a continuous geometric-strength score.

    Parameters
    ----------
    hydrogen_bond : HydrogenBond
        Hydrogen bond to score.
    config : HydrogenBondClassificationConfig, optional
        Classification configuration.

    Returns
    -------
    HydrogenBondStrength
        Continuous score and preliminary classifications.
    """

    if not isinstance(
        hydrogen_bond,
        HydrogenBond,
    ):
        raise TypeError(
            "hydrogen_bond must be a HydrogenBond instance."
        )

    geometry = hydrogen_bond.geometry

    distance_score = (
        calculate_distance_geometry_score(
            geometry,
            config=config,
        )
    )

    donor_angle_score = (
        calculate_donor_angle_geometry_score(
            geometry,
            config=config,
        )
    )

    acceptor_angle_score = (
        calculate_acceptor_angle_geometry_score(
            geometry,
            config=config,
        )
    )

    weighted_score_sum = np.float64(
        config.distance_weight
        * distance_score
    )

    used_weight = np.float64(
        config.distance_weight
    )

    missing_angle_count = 0

    if donor_angle_score is not None:
        weighted_score_sum += np.float64(
            config.angle_weight
            * donor_angle_score
        )

        used_weight += config.angle_weight

    else:
        missing_angle_count += 1

    if acceptor_angle_score is not None:
        weighted_score_sum += np.float64(
            config.acceptor_angle_weight
            * acceptor_angle_score
        )

        used_weight += (
            config.acceptor_angle_weight
        )

    elif config.acceptor_angle_weight > 0.0:
        missing_angle_count += 1

    if used_weight <= 0.0:
        base_score = np.float64(
            0.0
        )

    else:
        base_score = np.float64(
            weighted_score_sum
            / used_weight
        )

    if hydrogen_bond.mode == HBOND_MODE_EXPLICIT:
        mode_factor = (
            config.explicit_mode_factor
        )

    else:
        mode_factor = (
            config.inferred_mode_factor
        )

    assignment_ambiguous = bool(
        hydrogen_bond.metadata.get(
            "assignment_ambiguous",
            False,
        )
    )

    ambiguity_factor = (
        config.ambiguous_assignment_factor
        if assignment_ambiguous
        else np.float64(
            1.0
        )
    )

    missing_angle_factor = (
        np.float64(
            config.missing_angle_factor
            ** missing_angle_count
        )
        if missing_angle_count > 0
        else np.float64(
            1.0
        )
    )

    final_score = _clamp_unit_interval(
        base_score
        * mode_factor
        * ambiguity_factor
        * missing_angle_factor
    )

    score_classification = (
        classify_hydrogen_bond_score(
            final_score,
            config=config,
        )
    )

    threshold_classification = (
        classify_hydrogen_bond_by_thresholds(
            hydrogen_bond,
            config=config,
        )
    )

    final_classification = (
        _merge_hbond_classifications(
            threshold_classification,
            score_classification,
            require_consistency=(
                config.require_threshold_consistency
            ),
        )
    )

    component_scores: Dict[
        str,
        np.float64,
    ] = {
        "distance": distance_score,
    }

    if donor_angle_score is not None:
        component_scores[
            "donor_angle"
        ] = donor_angle_score

    if acceptor_angle_score is not None:
        component_scores[
            "acceptor_angle"
        ] = acceptor_angle_score

    limiting_criterion = (
        min(
            component_scores,
            key=lambda key: float(
                component_scores[
                    key
                ]
            ),
        )
        if component_scores
        else None
    )

    passed_criteria = tuple(
        criterion
        for criterion, component_score
        in component_scores.items()
        if component_score
        >= config.weak_score_threshold
    )

    failed_criteria = tuple(
        criterion
        for criterion, component_score
        in component_scores.items()
        if component_score
        < config.weak_score_threshold
    )

    return HydrogenBondStrength(
        classification=final_classification,
        score=final_score,
        distance_score=distance_score,
        donor_angle_score=donor_angle_score,
        acceptor_angle_score=(
            acceptor_angle_score
        ),
        mode_factor=mode_factor,
        ambiguity_factor=ambiguity_factor,
        threshold_classification=(
            threshold_classification
        ),
        score_classification=(
            score_classification
        ),
        limiting_criterion=(
            limiting_criterion
        ),
        passed_criteria=(
            passed_criteria
        ),
        failed_criteria=(
            failed_criteria
        ),
        metadata={
            "base_score": float(
                base_score
            ),
            "missing_angle_count": (
                missing_angle_count
            ),
            "missing_angle_factor": float(
                missing_angle_factor
            ),
            "assignment_ambiguous": (
                assignment_ambiguous
            ),
            "geometric_score_only": True,
        },
    )


# -----------------------------------------------------------------------------
# Discrete threshold classification
# -----------------------------------------------------------------------------

def _explicit_geometry_meets_level(
    geometry: HydrogenBondGeometry,
    *,
    donor_acceptor_cutoff: Number,
    hydrogen_acceptor_cutoff: Number,
    minimum_dha_angle: Number,
) -> bool:
    """
    Test explicit geometry against one classification level.

    Parameters
    ----------
    geometry : HydrogenBondGeometry
        Explicit geometry.
    donor_acceptor_cutoff : Number
        Maximum D...A distance.
    hydrogen_acceptor_cutoff : Number
        Maximum H...A distance.
    minimum_dha_angle : Number
        Minimum D-H...A angle.

    Returns
    -------
    bool
        Threshold status.
    """

    if (
        geometry.hydrogen_acceptor_distance is None
        or geometry.dha_angle is None
    ):
        return False

    return bool(
        geometry.donor_acceptor_distance
        <= np.float64(
            donor_acceptor_cutoff
        )
        and geometry.hydrogen_acceptor_distance
        <= np.float64(
            hydrogen_acceptor_cutoff
        )
        and geometry.dha_angle
        >= np.float64(
            minimum_dha_angle
        )
    )


def _inferred_geometry_meets_level(
    geometry: HydrogenBondGeometry,
    *,
    donor_acceptor_cutoff: Number,
    maximum_deviation: Number,
    require_acceptor_angle: bool = False,
) -> bool:
    """
    Test inferred geometry against one classification level.

    Parameters
    ----------
    geometry : HydrogenBondGeometry
        Inferred geometry.
    donor_acceptor_cutoff : Number
        Maximum D...A distance.
    maximum_deviation : Number
        Maximum inferred donor/acceptor deviation.
    require_acceptor_angle : bool, optional
        Whether acceptor alignment is mandatory.

    Returns
    -------
    bool
        Threshold status.
    """

    if geometry.donor_angle is None:
        return False

    if (
        geometry.donor_acceptor_distance
        > np.float64(
            donor_acceptor_cutoff
        )
    ):
        return False

    if (
        geometry.donor_angle
        > np.float64(
            maximum_deviation
        )
    ):
        return False

    if require_acceptor_angle:
        if geometry.acceptor_angle is None:
            return False

        if (
            geometry.acceptor_angle
            > np.float64(
                maximum_deviation
            )
        ):
            return False

    return True


def classify_hydrogen_bond_by_thresholds(
    hydrogen_bond: HydrogenBond,
    *,
    config: HydrogenBondClassificationConfig = (
        DEFAULT_HBOND_CLASSIFICATION_CONFIG
    ),
    require_inferred_acceptor_angle: bool = False,
) -> HydrogenBondClassification:
    """
    Classify a hydrogen bond using discrete geometric thresholds.

    Parameters
    ----------
    hydrogen_bond : HydrogenBond
        Hydrogen bond to classify.
    config : HydrogenBondClassificationConfig, optional
        Classification configuration.
    require_inferred_acceptor_angle : bool, optional
        Whether inferred acceptor alignment is mandatory.

    Returns
    -------
    HydrogenBondClassification
        Threshold-based classification.
    """

    if not isinstance(
        hydrogen_bond,
        HydrogenBond,
    ):
        raise TypeError(
            "hydrogen_bond must be a HydrogenBond instance."
        )

    geometry = hydrogen_bond.geometry

    if hydrogen_bond.mode == HBOND_MODE_EXPLICIT:
        levels = (
            (
                HBOND_TYPE_STRONG,
                config.strong_donor_acceptor_distance,
                config.strong_hydrogen_acceptor_distance,
                config.strong_dha_angle,
            ),
            (
                HBOND_TYPE_MODERATE,
                config.moderate_donor_acceptor_distance,
                config.moderate_hydrogen_acceptor_distance,
                config.moderate_dha_angle,
            ),
            (
                HBOND_TYPE_WEAK,
                config.weak_donor_acceptor_distance,
                config.weak_hydrogen_acceptor_distance,
                config.weak_dha_angle,
            ),
        )

        for (
            classification,
            donor_acceptor_cutoff,
            hydrogen_acceptor_cutoff,
            minimum_angle,
        ) in levels:
            if _explicit_geometry_meets_level(
                geometry,
                donor_acceptor_cutoff=(
                    donor_acceptor_cutoff
                ),
                hydrogen_acceptor_cutoff=(
                    hydrogen_acceptor_cutoff
                ),
                minimum_dha_angle=(
                    minimum_angle
                ),
            ):
                return classification

        if (
            geometry.donor_acceptor_distance
            <= config.weak_donor_acceptor_distance
        ):
            return HBOND_TYPE_GEOMETRIC_ONLY

        return HBOND_TYPE_REJECTED

    levels = (
        (
            HBOND_TYPE_STRONG,
            config.strong_donor_acceptor_distance,
            config.strong_inferred_deviation,
        ),
        (
            HBOND_TYPE_MODERATE,
            config.moderate_donor_acceptor_distance,
            config.moderate_inferred_deviation,
        ),
        (
            HBOND_TYPE_WEAK,
            config.weak_donor_acceptor_distance,
            config.weak_inferred_deviation,
        ),
    )

    for (
        classification,
        donor_acceptor_cutoff,
        maximum_deviation,
    ) in levels:
        if _inferred_geometry_meets_level(
            geometry,
            donor_acceptor_cutoff=(
                donor_acceptor_cutoff
            ),
            maximum_deviation=(
                maximum_deviation
            ),
            require_acceptor_angle=(
                require_inferred_acceptor_angle
            ),
        ):
            return classification

    if (
        geometry.donor_acceptor_distance
        <= config.weak_donor_acceptor_distance
    ):
        return HBOND_TYPE_GEOMETRIC_ONLY

    return HBOND_TYPE_REJECTED


def classify_hydrogen_bond_score(
    score: Number,
    *,
    config: HydrogenBondClassificationConfig = (
        DEFAULT_HBOND_CLASSIFICATION_CONFIG
    ),
) -> HydrogenBondClassification:
    """
    Convert a normalized geometric score into a classification.

    Parameters
    ----------
    score : Number
        Score between zero and one.
    config : HydrogenBondClassificationConfig, optional
        Classification configuration.

    Returns
    -------
    HydrogenBondClassification
        Score-based classification.
    """

    normalized_score = _clamp_unit_interval(
        score
    )

    if (
        normalized_score
        >= config.strong_score_threshold
    ):
        return HBOND_TYPE_STRONG

    if (
        normalized_score
        >= config.moderate_score_threshold
    ):
        return HBOND_TYPE_MODERATE

    if (
        normalized_score
        >= config.weak_score_threshold
    ):
        return HBOND_TYPE_WEAK

    if normalized_score > 0.0:
        return HBOND_TYPE_GEOMETRIC_ONLY

    return HBOND_TYPE_REJECTED


_CLASSIFICATION_RANK: Final[
    Mapping[
        HydrogenBondClassification,
        int,
    ]
] = MappingProxyType(
    {
        HBOND_TYPE_REJECTED: 0,
        HBOND_TYPE_UNKNOWN: 0,
        HBOND_TYPE_GEOMETRIC_ONLY: 1,
        HBOND_TYPE_WEAK: 2,
        HBOND_TYPE_MODERATE: 3,
        HBOND_TYPE_STRONG: 4,
    }
)


def _merge_hbond_classifications(
    threshold_classification: HydrogenBondClassification,
    score_classification: HydrogenBondClassification,
    *,
    require_consistency: bool,
) -> HydrogenBondClassification:
    """
    Merge discrete and score-based classifications.

    Parameters
    ----------
    threshold_classification : HydrogenBondClassification
        Discrete threshold classification.
    score_classification : HydrogenBondClassification
        Continuous-score classification.
    require_consistency : bool
        Whether the more conservative classification should be used.

    Returns
    -------
    HydrogenBondClassification
        Final classification.
    """

    normalized_threshold = (
        validate_hydrogen_bond_classification(
            threshold_classification
        )
    )

    normalized_score = (
        validate_hydrogen_bond_classification(
            score_classification
        )
    )

    if not require_consistency:
        return normalized_score

    threshold_rank = _CLASSIFICATION_RANK[
        normalized_threshold
    ]

    score_rank = _CLASSIFICATION_RANK[
        normalized_score
    ]

    if threshold_rank <= score_rank:
        return normalized_threshold

    return normalized_score


# -----------------------------------------------------------------------------
# HydrogenBond reconstruction
# -----------------------------------------------------------------------------

def classify_hydrogen_bond(
    hydrogen_bond: HydrogenBond,
    *,
    config: HydrogenBondClassificationConfig = (
        DEFAULT_HBOND_CLASSIFICATION_CONFIG
    ),
    require_inferred_acceptor_angle: bool = False,
) -> Tuple[
    HydrogenBond,
    HydrogenBondStrength,
]:
    """
    Classify one hydrogen bond and return an updated immutable object.

    Parameters
    ----------
    hydrogen_bond : HydrogenBond
        Hydrogen bond to classify.
    config : HydrogenBondClassificationConfig, optional
        Classification configuration.
    require_inferred_acceptor_angle : bool, optional
        Whether acceptor alignment is required in inferred mode.

    Returns
    -------
    tuple
        Updated ``HydrogenBond`` and corresponding ``HydrogenBondStrength``.
    """

    if not isinstance(
        hydrogen_bond,
        HydrogenBond,
    ):
        raise TypeError(
            "hydrogen_bond must be a HydrogenBond instance."
        )

    threshold_classification = (
        classify_hydrogen_bond_by_thresholds(
            hydrogen_bond,
            config=config,
            require_inferred_acceptor_angle=(
                require_inferred_acceptor_angle
            ),
        )
    )

    preliminary_strength = (
        calculate_hydrogen_bond_geometric_score(
            hydrogen_bond,
            config=config,
        )
    )

    final_classification = (
        _merge_hbond_classifications(
            threshold_classification,
            preliminary_strength.score_classification,
            require_consistency=(
                config.require_threshold_consistency
            ),
        )
    )

    strength = HydrogenBondStrength(
        classification=final_classification,
        score=preliminary_strength.score,
        distance_score=(
            preliminary_strength.distance_score
        ),
        donor_angle_score=(
            preliminary_strength.donor_angle_score
        ),
        acceptor_angle_score=(
            preliminary_strength.acceptor_angle_score
        ),
        mode_factor=(
            preliminary_strength.mode_factor
        ),
        ambiguity_factor=(
            preliminary_strength.ambiguity_factor
        ),
        threshold_classification=(
            threshold_classification
        ),
        score_classification=(
            preliminary_strength.score_classification
        ),
        limiting_criterion=(
            preliminary_strength.limiting_criterion
        ),
        passed_criteria=(
            preliminary_strength.passed_criteria
        ),
        failed_criteria=(
            preliminary_strength.failed_criteria
        ),
        metadata=preliminary_strength.metadata,
    )

    updated_metadata = dict(
        hydrogen_bond.metadata
    )

    updated_metadata.update(
        {
            "geometric_strength_score": float(
                strength.score
            ),
            "distance_score": float(
                strength.distance_score
            ),
            "donor_angle_score": (
                None
                if strength.donor_angle_score is None
                else float(
                    strength.donor_angle_score
                )
            ),
            "acceptor_angle_score": (
                None
                if strength.acceptor_angle_score is None
                else float(
                    strength.acceptor_angle_score
                )
            ),
            "threshold_classification": (
                strength.threshold_classification
            ),
            "score_classification": (
                strength.score_classification
            ),
            "limiting_criterion": (
                strength.limiting_criterion
            ),
            "classification_is_geometric": True,
        }
    )

    updated_hydrogen_bond = HydrogenBond(
        donor=hydrogen_bond.donor,
        acceptor=hydrogen_bond.acceptor,
        hydrogen=hydrogen_bond.hydrogen,
        geometry=hydrogen_bond.geometry,
        mode=hydrogen_bond.mode,
        direction=hydrogen_bond.direction,
        classification=final_classification,
        donor_index=hydrogen_bond.donor_index,
        acceptor_index=(
            hydrogen_bond.acceptor_index
        ),
        hydrogen_index=(
            hydrogen_bond.hydrogen_index
        ),
        donor_residue=(
            hydrogen_bond.donor_residue
        ),
        acceptor_residue=(
            hydrogen_bond.acceptor_residue
        ),
        donor_residue_key=(
            hydrogen_bond.donor_residue_key
        ),
        acceptor_residue_key=(
            hydrogen_bond.acceptor_residue_key
        ),
        metadata=updated_metadata,
    )

    return (
        updated_hydrogen_bond,
        strength,
    )


def classify_hydrogen_bonds(
    hydrogen_bonds: Iterable[
        HydrogenBond
    ],
    *,
    config: HydrogenBondClassificationConfig = (
        DEFAULT_HBOND_CLASSIFICATION_CONFIG
    ),
    require_inferred_acceptor_angle: bool = False,
    include_rejected: bool = True,
    sort_by_strength: bool = False,
) -> Tuple[
    HydrogenBond,
    ...,
]:
    """
    Classify a collection of hydrogen bonds.

    Parameters
    ----------
    hydrogen_bonds : iterable of HydrogenBond
        Hydrogen bonds to classify.
    config : HydrogenBondClassificationConfig, optional
        Classification configuration.
    require_inferred_acceptor_angle : bool, optional
        Whether inferred acceptor alignment is required.
    include_rejected : bool, optional
        Whether rejected interactions should be retained.
    sort_by_strength : bool, optional
        Whether results should be sorted by decreasing strength score.

    Returns
    -------
    tuple of HydrogenBond
        Classified hydrogen bonds.
    """

    classified_entries: List[
        Tuple[
            HydrogenBond,
            HydrogenBondStrength,
        ]
    ] = []

    for index, hydrogen_bond in enumerate(
        hydrogen_bonds
    ):
        if not isinstance(
            hydrogen_bond,
            HydrogenBond,
        ):
            raise TypeError(
                "All entries must be HydrogenBond instances. "
                f"Invalid entry at index {index}."
            )

        classified_hbond, strength = (
            classify_hydrogen_bond(
                hydrogen_bond,
                config=config,
                require_inferred_acceptor_angle=(
                    require_inferred_acceptor_angle
                ),
            )
        )

        if (
            not include_rejected
            and classified_hbond.classification
            == HBOND_TYPE_REJECTED
        ):
            continue

        classified_entries.append(
            (
                classified_hbond,
                strength,
            )
        )

    if sort_by_strength:
        classified_entries.sort(
            key=lambda entry: (
                -float(
                    entry[
                        1
                    ].score
                ),
                float(
                    entry[
                        0
                    ].donor_acceptor_distance
                ),
                (
                    _safe_atom_identifier(
                        entry[
                            0
                        ].donor
                    )
                    or ""
                ),
                (
                    _safe_atom_identifier(
                        entry[
                            0
                        ].acceptor
                    )
                    or ""
                ),
            )
        )

    return tuple(
        hydrogen_bond
        for hydrogen_bond, _
        in classified_entries
    )


# -----------------------------------------------------------------------------
# Strength extraction and filtering
# -----------------------------------------------------------------------------

def get_hydrogen_bond_strength(
    hydrogen_bond: HydrogenBond,
    *,
    config: HydrogenBondClassificationConfig = (
        DEFAULT_HBOND_CLASSIFICATION_CONFIG
    ),
) -> HydrogenBondStrength:
    """
    Return geometric-strength information for one hydrogen bond.

    Parameters
    ----------
    hydrogen_bond : HydrogenBond
        Hydrogen bond.
    config : HydrogenBondClassificationConfig, optional
        Classification configuration.

    Returns
    -------
    HydrogenBondStrength
        Strength assessment.
    """

    _, strength = classify_hydrogen_bond(
        hydrogen_bond,
        config=config,
    )

    return strength


def filter_hydrogen_bonds_by_classification(
    hydrogen_bonds: Iterable[
        HydrogenBond
    ],
    *,
    classifications: Iterable[
        HydrogenBondClassification
    ],
) -> Tuple[
    HydrogenBond,
    ...,
]:
    """
    Filter hydrogen bonds by classification.

    Parameters
    ----------
    hydrogen_bonds : iterable of HydrogenBond
        Hydrogen bonds.
    classifications : iterable of HydrogenBondClassification
        Accepted classifications.

    Returns
    -------
    tuple of HydrogenBond
        Filtered hydrogen bonds.
    """

    normalized_classifications = frozenset(
        validate_hydrogen_bond_classification(
            classification
        )
        for classification in classifications
    )

    return tuple(
        hydrogen_bond
        for hydrogen_bond in hydrogen_bonds
        if (
            isinstance(
                hydrogen_bond,
                HydrogenBond,
            )
            and hydrogen_bond.classification
            in normalized_classifications
        )
    )


def filter_hydrogen_bonds_by_strength(
    hydrogen_bonds: Iterable[
        HydrogenBond
    ],
    *,
    minimum_score: Number = (
        DEFAULT_WEAK_SCORE_THRESHOLD
    ),
    config: HydrogenBondClassificationConfig = (
        DEFAULT_HBOND_CLASSIFICATION_CONFIG
    ),
) -> Tuple[
    HydrogenBond,
    ...,
]:
    """
    Filter hydrogen bonds by geometric-strength score.

    Parameters
    ----------
    hydrogen_bonds : iterable of HydrogenBond
        Hydrogen bonds.
    minimum_score : Number, optional
        Minimum normalized geometric score.
    config : HydrogenBondClassificationConfig, optional
        Classification configuration.

    Returns
    -------
    tuple of HydrogenBond
        Interactions reaching the score threshold.
    """

    normalized_minimum_score = (
        _optional_float64(
            minimum_score,
            name="minimum geometric strength score",
            minimum=0.0,
            maximum=1.0,
        )
    )

    if normalized_minimum_score is None:
        raise ValueError(
            "Minimum geometric strength score cannot be None."
        )

    selected: List[
        HydrogenBond
    ] = []

    for hydrogen_bond in hydrogen_bonds:
        if not isinstance(
            hydrogen_bond,
            HydrogenBond,
        ):
            continue

        strength = get_hydrogen_bond_strength(
            hydrogen_bond,
            config=config,
        )

        if strength.score >= normalized_minimum_score:
            selected.append(
                hydrogen_bond
            )

    return tuple(
        selected
    )


# -----------------------------------------------------------------------------
# Classification statistics
# -----------------------------------------------------------------------------

def hydrogen_bond_classification_counts(
    hydrogen_bonds: Iterable[
        HydrogenBond
    ],
) -> Dict[
    HydrogenBondClassification,
    int,
]:
    """
    Count hydrogen bonds by classification.

    Parameters
    ----------
    hydrogen_bonds : iterable of HydrogenBond
        Hydrogen bonds.

    Returns
    -------
    dict
        Classification counts.
    """

    counts: Dict[
        HydrogenBondClassification,
        int,
    ] = {
        classification: 0
        for classification
        in VALID_HYDROGEN_BOND_CLASSIFICATIONS
    }

    for hydrogen_bond in hydrogen_bonds:
        if not isinstance(
            hydrogen_bond,
            HydrogenBond,
        ):
            continue

        classification = (
            validate_hydrogen_bond_classification(
                hydrogen_bond.classification
            )
        )

        counts[
            classification
        ] += 1

    return counts


def hydrogen_bond_strength_statistics(
    hydrogen_bonds: Iterable[
        HydrogenBond
    ],
    *,
    config: HydrogenBondClassificationConfig = (
        DEFAULT_HBOND_CLASSIFICATION_CONFIG
    ),
) -> Dict[
    str,
    Any,
]:
    """
    Calculate geometric-strength statistics.

    Parameters
    ----------
    hydrogen_bonds : iterable of HydrogenBond
        Hydrogen bonds.
    config : HydrogenBondClassificationConfig, optional
        Classification configuration.

    Returns
    -------
    dict
        Strength and classification statistics.
    """

    normalized_bonds = tuple(
        hydrogen_bonds
    )

    strengths = tuple(
        get_hydrogen_bond_strength(
            hydrogen_bond,
            config=config,
        )
        for hydrogen_bond
        in normalized_bonds
        if isinstance(
            hydrogen_bond,
            HydrogenBond,
        )
    )

    scores = np.asarray(
        [
            strength.score
            for strength in strengths
        ],
        dtype=np.float64,
    )

    classification_counts: Dict[
        HydrogenBondClassification,
        int,
    ] = {
        classification: 0
        for classification
        in VALID_HYDROGEN_BOND_CLASSIFICATIONS
    }

    for strength in strengths:
        classification_counts[
            strength.classification
        ] += 1

    if scores.size == 0:
        minimum_score = None
        maximum_score = None
        mean_score = None
        median_score = None
        standard_deviation = None

    else:
        minimum_score = float(
            np.min(
                scores
            )
        )

        maximum_score = float(
            np.max(
                scores
            )
        )

        mean_score = float(
            np.mean(
                scores
            )
        )

        median_score = float(
            np.median(
                scores
            )
        )

        standard_deviation = float(
            np.std(
                scores,
                dtype=np.float64,
            )
        )

    return {
        "hydrogen_bond_count": len(
            strengths
        ),
        "classification_counts": (
            classification_counts
        ),
        "accepted_count": sum(
            classification_counts[
                classification
            ]
            for classification in (
                HBOND_TYPE_STRONG,
                HBOND_TYPE_MODERATE,
                HBOND_TYPE_WEAK,
                HBOND_TYPE_GEOMETRIC_ONLY,
            )
        ),
        "rejected_count": (
            classification_counts[
                HBOND_TYPE_REJECTED
            ]
        ),
        "minimum_strength_score": (
            minimum_score
        ),
        "maximum_strength_score": (
            maximum_score
        ),
        "mean_strength_score": mean_score,
        "median_strength_score": (
            median_score
        ),
        "strength_score_standard_deviation": (
            standard_deviation
        ),
        "explicit_count": sum(
            hydrogen_bond.mode
            == HBOND_MODE_EXPLICIT
            for hydrogen_bond
            in normalized_bonds
            if isinstance(
                hydrogen_bond,
                HydrogenBond,
            )
        ),
        "inferred_count": sum(
            hydrogen_bond.mode
            == HBOND_MODE_INFERRED
            for hydrogen_bond
            in normalized_bonds
            if isinstance(
                hydrogen_bond,
                HydrogenBond,
            )
        ),
    }


# -----------------------------------------------------------------------------
# Residue-group classification integration
# -----------------------------------------------------------------------------

def classify_residue_hydrogen_bond_group(
    residue_group: ResidueHydrogenBond,
    *,
    config: HydrogenBondClassificationConfig = (
        DEFAULT_HBOND_CLASSIFICATION_CONFIG
    ),
    include_rejected: bool = True,
    sort_by_strength: bool = False,
) -> ResidueHydrogenBond:
    """
    Classify all hydrogen bonds in one residue group.

    Parameters
    ----------
    residue_group : ResidueHydrogenBond
        Residue-level group.
    config : HydrogenBondClassificationConfig, optional
        Classification configuration.
    include_rejected : bool, optional
        Whether rejected interactions should be retained.
    sort_by_strength : bool, optional
        Whether interactions should be strength-sorted.

    Returns
    -------
    ResidueHydrogenBond
        New group containing classified hydrogen bonds.
    """

    if not isinstance(
        residue_group,
        ResidueHydrogenBond,
    ):
        raise TypeError(
            "residue_group must be a ResidueHydrogenBond instance."
        )

    classified_bonds = classify_hydrogen_bonds(
        residue_group.hydrogen_bonds,
        config=config,
        include_rejected=include_rejected,
        sort_by_strength=sort_by_strength,
    )

    updated_metadata = dict(
        residue_group.metadata
    )

    updated_metadata.update(
        {
            "classified": True,
            "classification_counts": (
                hydrogen_bond_classification_counts(
                    classified_bonds
                )
            ),
        }
    )

    return ResidueHydrogenBond(
        residue=residue_group.residue,
        key=residue_group.key,
        hydrogen_bonds=classified_bonds,
        side=residue_group.side,
        metadata=updated_metadata,
    )


def classify_residue_hydrogen_bond_groups(
    residue_groups: Iterable[
        ResidueHydrogenBond
    ],
    *,
    config: HydrogenBondClassificationConfig = (
        DEFAULT_HBOND_CLASSIFICATION_CONFIG
    ),
    include_rejected: bool = True,
    sort_by_strength: bool = False,
) -> Tuple[
    ResidueHydrogenBond,
    ...,
]:
    """
    Classify multiple residue-level hydrogen-bond groups.

    Parameters
    ----------
    residue_groups : iterable of ResidueHydrogenBond
        Residue groups.
    config : HydrogenBondClassificationConfig, optional
        Classification configuration.
    include_rejected : bool, optional
        Whether rejected interactions should be retained.
    sort_by_strength : bool, optional
        Whether each group should be strength-sorted.

    Returns
    -------
    tuple of ResidueHydrogenBond
        Classified residue groups.
    """

    return tuple(
        classify_residue_hydrogen_bond_group(
            residue_group,
            config=config,
            include_rejected=include_rejected,
            sort_by_strength=sort_by_strength,
        )
        for residue_group
        in residue_groups
    )


# -----------------------------------------------------------------------------
# HydrogenBondAnalysisResult integration
# -----------------------------------------------------------------------------

def classify_hydrogen_bond_analysis_result(
    result: HydrogenBondAnalysisResult,
    *,
    config: HydrogenBondClassificationConfig = (
        DEFAULT_HBOND_CLASSIFICATION_CONFIG
    ),
    include_rejected: bool = True,
    sort_by_strength: bool = False,
    regroup_residues: bool = True,
) -> HydrogenBondAnalysisResult:
    """
    Classify every hydrogen bond in an analysis result.

    Parameters
    ----------
    result : HydrogenBondAnalysisResult
        Analysis result.
    config : HydrogenBondClassificationConfig, optional
        Classification configuration.
    include_rejected : bool, optional
        Whether rejected interactions should be retained.
    sort_by_strength : bool, optional
        Whether interactions should be sorted by decreasing score.
    regroup_residues : bool, optional
        Whether existing residue groups should be rebuilt using classified
        hydrogen-bond objects.

    Returns
    -------
    HydrogenBondAnalysisResult
        New result containing classified interactions.
    """

    if not isinstance(
        result,
        HydrogenBondAnalysisResult,
    ):
        raise TypeError(
            "result must be a HydrogenBondAnalysisResult instance."
        )

    classified_bonds = classify_hydrogen_bonds(
        result.hydrogen_bonds,
        config=config,
        include_rejected=include_rejected,
        sort_by_strength=sort_by_strength,
    )

    if (
        regroup_residues
        and result.residue_hydrogen_bonds
    ):
        grouping_side = str(
            result.metadata.get(
                "residue_grouping_side",
                DEFAULT_RESIDUE_GROUP_SIDE,
            )
        )

        try:
            normalized_grouping_side = (
                validate_residue_group_side(
                    grouping_side
                )
            )

        except Exception:
            normalized_grouping_side = (
                DEFAULT_RESIDUE_GROUP_SIDE
            )

        classified_residue_groups = (
            group_hydrogen_bonds_by_residue(
                classified_bonds,
                side=normalized_grouping_side,
                include_unresolved=bool(
                    result.metadata.get(
                        "include_unresolved_residues",
                        False,
                    )
                ),
                sort_groups=True,
                sort_bonds=not sort_by_strength,
                deduplicate_bonds=False,
            )
        )

        if sort_by_strength:
            classified_residue_groups = (
                classify_residue_hydrogen_bond_groups(
                    classified_residue_groups,
                    config=config,
                    include_rejected=True,
                    sort_by_strength=True,
                )
            )

    else:
        classified_residue_groups = (
            classify_residue_hydrogen_bond_groups(
                result.residue_hydrogen_bonds,
                config=config,
                include_rejected=include_rejected,
                sort_by_strength=sort_by_strength,
            )
            if result.residue_hydrogen_bonds
            else ()
        )

    strength_statistics = (
        hydrogen_bond_strength_statistics(
            classified_bonds,
            config=config,
        )
    )

    updated_statistics = dict(
        result.statistics
    )

    updated_statistics.update(
        strength_statistics
    )

    updated_metadata = dict(
        result.metadata
    )

    updated_metadata.update(
        {
            "hydrogen_bonds_classified": True,
            "classification_geometric_only": True,
            "classification_config": (
                config.to_dict()
            ),
            "rejected_interactions_included": bool(
                include_rejected
            ),
            "sorted_by_strength": bool(
                sort_by_strength
            ),
        }
    )

    return HydrogenBondAnalysisResult(
        hydrogen_bonds=classified_bonds,
        residue_hydrogen_bonds=(
            classified_residue_groups
        ),
        ligand_atoms=result.ligand_atoms,
        receptor_atoms=result.receptor_atoms,
        donor_acceptor_cutoff=(
            result.donor_acceptor_cutoff
        ),
        hydrogen_acceptor_cutoff=(
            result.hydrogen_acceptor_cutoff
        ),
        minimum_dha_angle=(
            result.minimum_dha_angle
        ),
        minimum_inferred_angle=(
            result.minimum_inferred_angle
        ),
        statistics=updated_statistics,
        metadata=updated_metadata,
    )


def analyze_group_and_classify_hydrogen_bonds(
    ligand_atoms: Iterable[
        AtomLike
    ],
    receptor_atoms: Iterable[
        AtomLike
    ],
    *,
    residue_side: str = (
        DEFAULT_RESIDUE_GROUP_SIDE
    ),
    classification_config: HydrogenBondClassificationConfig = (
        DEFAULT_HBOND_CLASSIFICATION_CONFIG
    ),
    include_rejected: bool = True,
    sort_by_strength: bool = False,
    **analysis_kwargs: Any,
) -> HydrogenBondAnalysisResult:
    """
    Detect, group and classify ligand-receptor hydrogen bonds.

    Parameters
    ----------
    ligand_atoms : iterable of atom-like
        Ligand atoms.
    receptor_atoms : iterable of atom-like
        Receptor atoms.
    residue_side : str, optional
        Residue grouping side.
    classification_config : HydrogenBondClassificationConfig, optional
        Classification configuration.
    include_rejected : bool, optional
        Whether rejected interactions should be retained.
    sort_by_strength : bool, optional
        Whether interactions should be sorted by strength.
    **analysis_kwargs : Any
        Arguments forwarded to :func:`analyze_hydrogen_bonds`.

    Returns
    -------
    HydrogenBondAnalysisResult
        Classified and residue-grouped result.
    """

    grouped_result = analyze_and_group_hydrogen_bonds(
        ligand_atoms,
        receptor_atoms,
        residue_side=residue_side,
        **analysis_kwargs,
    )

    return classify_hydrogen_bond_analysis_result(
        grouped_result,
        config=classification_config,
        include_rejected=include_rejected,
        sort_by_strength=sort_by_strength,
        regroup_residues=True,
    )


# -----------------------------------------------------------------------------
# Public interface
# -----------------------------------------------------------------------------

_SECTION_9_PUBLIC_NAMES: Final[
    Tuple[
        str,
        ...,
    ]
] = (
    "HBOND_TYPE_STRONG",
    "HBOND_TYPE_MODERATE",
    "HBOND_TYPE_WEAK",
    "HBOND_TYPE_GEOMETRIC_ONLY",
    "HBOND_TYPE_REJECTED",
    "HBOND_TYPE_UNKNOWN",
    "VALID_HYDROGEN_BOND_CLASSIFICATIONS",
    "DEFAULT_STRONG_DONOR_ACCEPTOR_DISTANCE",
    "DEFAULT_MODERATE_DONOR_ACCEPTOR_DISTANCE",
    "DEFAULT_WEAK_DONOR_ACCEPTOR_DISTANCE",
    "DEFAULT_STRONG_HYDROGEN_ACCEPTOR_DISTANCE",
    "DEFAULT_MODERATE_HYDROGEN_ACCEPTOR_DISTANCE",
    "DEFAULT_WEAK_HYDROGEN_ACCEPTOR_DISTANCE",
    "DEFAULT_STRONG_DHA_ANGLE",
    "DEFAULT_MODERATE_DHA_ANGLE",
    "DEFAULT_WEAK_DHA_ANGLE",
    "DEFAULT_STRONG_INFERRED_DEVIATION",
    "DEFAULT_MODERATE_INFERRED_DEVIATION",
    "DEFAULT_WEAK_INFERRED_DEVIATION",
    "DEFAULT_DISTANCE_SCORE_WEIGHT",
    "DEFAULT_ANGLE_SCORE_WEIGHT",
    "DEFAULT_ACCEPTOR_ANGLE_SCORE_WEIGHT",
    "DEFAULT_EXPLICIT_MODE_SCORE_FACTOR",
    "DEFAULT_INFERRED_MODE_SCORE_FACTOR",
    "DEFAULT_AMBIGUOUS_ASSIGNMENT_SCORE_FACTOR",
    "DEFAULT_MISSING_ANGLE_SCORE_FACTOR",
    "DEFAULT_STRONG_SCORE_THRESHOLD",
    "DEFAULT_MODERATE_SCORE_THRESHOLD",
    "DEFAULT_WEAK_SCORE_THRESHOLD",
    "HydrogenBondClassificationConfig",
    "DEFAULT_HBOND_CLASSIFICATION_CONFIG",
    "HydrogenBondStrength",
    "validate_hydrogen_bond_classification",
    "calculate_distance_geometry_score",
    "calculate_donor_angle_geometry_score",
    "calculate_acceptor_angle_geometry_score",
    "calculate_hydrogen_bond_geometric_score",
    "classify_hydrogen_bond_by_thresholds",
    "classify_hydrogen_bond_score",
    "classify_hydrogen_bond",
    "classify_hydrogen_bonds",
    "get_hydrogen_bond_strength",
    "filter_hydrogen_bonds_by_classification",
    "filter_hydrogen_bonds_by_strength",
    "hydrogen_bond_classification_counts",
    "hydrogen_bond_strength_statistics",
    "classify_residue_hydrogen_bond_group",
    "classify_residue_hydrogen_bond_groups",
    "classify_hydrogen_bond_analysis_result",
    "analyze_group_and_classify_hydrogen_bonds",
)

for public_name in _SECTION_9_PUBLIC_NAMES:
    if public_name not in __all__:
        __all__.append(
            public_name
        )


# =============================================================================
# End of Section 9
# =============================================================================

# =============================================================================
# Section 10 — Statistics and summaries
# =============================================================================


# -----------------------------------------------------------------------------
# Statistical constants
# -----------------------------------------------------------------------------

DEFAULT_STATISTICS_DDOF: Final[
    int
] = 0

DEFAULT_TOP_HYDROGEN_BOND_COUNT: Final[
    int
] = 10

DEFAULT_TOP_RESIDUE_COUNT: Final[
    int
] = 10

DEFAULT_SUMMARY_DECIMAL_PLACES: Final[
    int
] = 3

DEFAULT_INCLUDE_EMPTY_STATISTICS: Final[
    bool
] = True

DEFAULT_INCLUDE_HYDROGEN_BOND_RECORDS: Final[
    bool
] = False

DEFAULT_INCLUDE_RESIDUE_RECORDS: Final[
    bool
] = False

DEFAULT_RECALCULATE_STRENGTHS: Final[
    bool
] = False


# -----------------------------------------------------------------------------
# Numeric summary helpers
# -----------------------------------------------------------------------------

def _normalize_statistics_ddof(
    ddof: int,
) -> int:
    """
    Validate a statistical delta degrees of freedom.

    Parameters
    ----------
    ddof : int
        Delta degrees of freedom.

    Returns
    -------
    int
        Validated nonnegative integer.

    Raises
    ------
    TypeError
        If ``ddof`` is not an integer.
    ValueError
        If ``ddof`` is negative.
    """

    if isinstance(
        ddof,
        bool,
    ) or not isinstance(
        ddof,
        (
            int,
            np.integer,
        ),
    ):
        raise TypeError(
            "ddof must be an integer."
        )

    normalized_ddof = int(
        ddof
    )

    if normalized_ddof < 0:
        raise ValueError(
            "ddof must be nonnegative."
        )

    return normalized_ddof


def _normalize_summary_limit(
    value: int,
    *,
    name: str,
) -> int:
    """
    Validate a nonnegative summary limit.

    Parameters
    ----------
    value : int
        Requested limit.
    name : str
        Human-readable parameter name.

    Returns
    -------
    int
        Validated limit.
    """

    normalized_value = (
        _optional_nonnegative_integer(
            value,
            name=name,
        )
    )

    if normalized_value is None:
        return 0

    return normalized_value


def _finite_float_array(
    values: Iterable[
        Optional[
            Number
        ]
    ],
) -> FloatArray:
    """
    Convert values to a finite one-dimensional float array.

    Parameters
    ----------
    values : iterable of Number or None
        Numeric values.

    Returns
    -------
    numpy.ndarray
        Finite ``float64`` values.
    """

    normalized_values: List[
        np.float64
    ] = []

    for value in values:
        if value is None:
            continue

        try:
            numeric_value = np.float64(
                value
            )

        except (
            TypeError,
            ValueError,
            OverflowError,
        ):
            continue

        if np.isfinite(
            numeric_value
        ):
            normalized_values.append(
                numeric_value
            )

    return np.asarray(
        normalized_values,
        dtype=np.float64,
    )


def summarize_numeric_values(
    values: Iterable[
        Optional[
            Number
        ]
    ],
    *,
    ddof: int = DEFAULT_STATISTICS_DDOF,
    include_values: bool = False,
) -> Dict[
    str,
    Any,
]:
    """
    Calculate descriptive statistics for numeric values.

    Parameters
    ----------
    values : iterable of Number or None
        Values to summarize.
    ddof : int, optional
        Delta degrees of freedom for variance and standard deviation.
    include_values : bool, optional
        Whether the normalized values should be included.

    Returns
    -------
    dict
        Descriptive statistics.

    Notes
    -----
    Nonfinite values and ``None`` entries are ignored.
    """

    normalized_ddof = _normalize_statistics_ddof(
        ddof
    )

    numeric_values = _finite_float_array(
        values
    )

    count = int(
        numeric_values.size
    )

    if count == 0:
        result: Dict[
            str,
            Any,
        ] = {
            "count": 0,
            "minimum": None,
            "maximum": None,
            "mean": None,
            "median": None,
            "standard_deviation": None,
            "variance": None,
            "first_quartile": None,
            "third_quartile": None,
            "interquartile_range": None,
            "sum": np.float64(
                0.0
            ),
        }

        if include_values:
            result[
                "values"
            ] = ()

        return result

    first_quartile, median, third_quartile = (
        np.percentile(
            numeric_values,
            [
                25.0,
                50.0,
                75.0,
            ],
        )
    )

    if count > normalized_ddof:
        variance = np.float64(
            np.var(
                numeric_values,
                ddof=normalized_ddof,
                dtype=np.float64,
            )
        )

        standard_deviation = np.float64(
            np.sqrt(
                variance
            )
        )

    else:
        variance = None
        standard_deviation = None

    result = {
        "count": count,
        "minimum": np.float64(
            np.min(
                numeric_values
            )
        ),
        "maximum": np.float64(
            np.max(
                numeric_values
            )
        ),
        "mean": np.float64(
            np.mean(
                numeric_values,
                dtype=np.float64,
            )
        ),
        "median": np.float64(
            median
        ),
        "standard_deviation": (
            standard_deviation
        ),
        "variance": variance,
        "first_quartile": np.float64(
            first_quartile
        ),
        "third_quartile": np.float64(
            third_quartile
        ),
        "interquartile_range": np.float64(
            third_quartile
            - first_quartile
        ),
        "sum": np.float64(
            np.sum(
                numeric_values,
                dtype=np.float64,
            )
        ),
    }

    if include_values:
        result[
            "values"
        ] = tuple(
            np.float64(
                value
            )
            for value in numeric_values
        )

    return result


def _serialize_numeric_summary(
    summary: Mapping[
        str,
        Any,
    ],
) -> Dict[
    str,
    Any,
]:
    """
    Convert NumPy values in a numeric summary to Python scalars.

    Parameters
    ----------
    summary : mapping
        Numeric summary.

    Returns
    -------
    dict
        Serializable summary.
    """

    serialized: Dict[
        str,
        Any,
    ] = {}

    for key, value in summary.items():
        if isinstance(
            value,
            np.ndarray,
        ):
            serialized[
                key
            ] = value.tolist()

        elif isinstance(
            value,
            (
                np.integer,
            ),
        ):
            serialized[
                key
            ] = int(
                value
            )

        elif isinstance(
            value,
            (
                np.floating,
            ),
        ):
            serialized[
                key
            ] = float(
                value
            )

        elif isinstance(
            value,
            tuple,
        ):
            serialized[
                key
            ] = [
                float(
                    item
                )
                if isinstance(
                    item,
                    (
                        float,
                        np.floating,
                    ),
                )
                else item
                for item in value
            ]

        else:
            serialized[
                key
            ] = value

    return serialized


# -----------------------------------------------------------------------------
# Hydrogen-bond collection validation
# -----------------------------------------------------------------------------

def _normalize_hydrogen_bond_collection(
    hydrogen_bonds: Iterable[
        HydrogenBond
    ],
    *,
    name: str = "hydrogen bonds",
) -> Tuple[
    HydrogenBond,
    ...,
]:
    """
    Validate a hydrogen-bond collection.

    Parameters
    ----------
    hydrogen_bonds : iterable of HydrogenBond
        Hydrogen bonds.
    name : str, optional
        Human-readable collection name.

    Returns
    -------
    tuple of HydrogenBond
        Validated collection.
    """

    try:
        normalized_bonds = tuple(
            hydrogen_bonds
        )

    except TypeError as error:
        raise TypeError(
            f"{name} must be an iterable of HydrogenBond objects."
        ) from error

    for index, hydrogen_bond in enumerate(
        normalized_bonds
    ):
        if not isinstance(
            hydrogen_bond,
            HydrogenBond,
        ):
            raise TypeError(
                f"All {name} entries must be HydrogenBond instances. "
                f"Invalid entry at index {index}."
            )

    return normalized_bonds


def _normalize_residue_hbond_collection(
    residue_groups: Iterable[
        ResidueHydrogenBond
    ],
) -> Tuple[
    ResidueHydrogenBond,
    ...,
]:
    """
    Validate residue-level hydrogen-bond groups.

    Parameters
    ----------
    residue_groups : iterable of ResidueHydrogenBond
        Residue groups.

    Returns
    -------
    tuple of ResidueHydrogenBond
        Validated groups.
    """

    try:
        normalized_groups = tuple(
            residue_groups
        )

    except TypeError as error:
        raise TypeError(
            "residue_groups must be iterable."
        ) from error

    for index, residue_group in enumerate(
        normalized_groups
    ):
        if not isinstance(
            residue_group,
            ResidueHydrogenBond,
        ):
            raise TypeError(
                "All residue groups must be ResidueHydrogenBond "
                f"instances. Invalid entry at index {index}."
            )

    return normalized_groups


# -----------------------------------------------------------------------------
# Basic count summaries
# -----------------------------------------------------------------------------

def hydrogen_bond_mode_counts(
    hydrogen_bonds: Iterable[
        HydrogenBond
    ],
) -> Dict[
    HydrogenBondMode,
    int,
]:
    """
    Count hydrogen bonds by analysis mode.

    Parameters
    ----------
    hydrogen_bonds : iterable of HydrogenBond
        Hydrogen bonds.

    Returns
    -------
    dict
        Explicit and inferred counts.
    """

    normalized_bonds = (
        _normalize_hydrogen_bond_collection(
            hydrogen_bonds
        )
    )

    counts: Dict[
        HydrogenBondMode,
        int,
    ] = {
        HBOND_MODE_EXPLICIT: 0,
        HBOND_MODE_INFERRED: 0,
    }

    for hydrogen_bond in normalized_bonds:
        mode = validate_hydrogen_bond_mode(
            hydrogen_bond.mode
        )

        counts[
            mode
        ] = counts.get(
            mode,
            0,
        ) + 1

    return counts


def hydrogen_bond_direction_counts(
    hydrogen_bonds: Iterable[
        HydrogenBond
    ],
) -> Dict[
    HydrogenBondDirection,
    int,
]:
    """
    Count hydrogen bonds by donor direction.

    Parameters
    ----------
    hydrogen_bonds : iterable of HydrogenBond
        Hydrogen bonds.

    Returns
    -------
    dict
        Direction counts.
    """

    normalized_bonds = (
        _normalize_hydrogen_bond_collection(
            hydrogen_bonds
        )
    )

    counts: Dict[
        HydrogenBondDirection,
        int,
    ] = {
        HBOND_DIRECTION_LIGAND_DONOR: 0,
        HBOND_DIRECTION_RECEPTOR_DONOR: 0,
        HBOND_DIRECTION_UNKNOWN: 0,
    }

    for hydrogen_bond in normalized_bonds:
        direction = (
            validate_hydrogen_bond_direction(
                hydrogen_bond.direction
            )
        )

        counts[
            direction
        ] = counts.get(
            direction,
            0,
        ) + 1

    return counts


def hydrogen_bond_role_counts(
    hydrogen_bonds: Iterable[
        HydrogenBond
    ],
) -> Dict[
    str,
    int,
]:
    """
    Count ligand and receptor donor/acceptor roles.

    Parameters
    ----------
    hydrogen_bonds : iterable of HydrogenBond
        Hydrogen bonds.

    Returns
    -------
    dict
        Role counts.
    """

    direction_counts = (
        hydrogen_bond_direction_counts(
            hydrogen_bonds
        )
    )

    ligand_donor_count = direction_counts.get(
        HBOND_DIRECTION_LIGAND_DONOR,
        0,
    )

    receptor_donor_count = direction_counts.get(
        HBOND_DIRECTION_RECEPTOR_DONOR,
        0,
    )

    unknown_count = direction_counts.get(
        HBOND_DIRECTION_UNKNOWN,
        0,
    )

    return {
        "ligand_donor_count": (
            ligand_donor_count
        ),
        "ligand_acceptor_count": (
            receptor_donor_count
        ),
        "receptor_donor_count": (
            receptor_donor_count
        ),
        "receptor_acceptor_count": (
            ligand_donor_count
        ),
        "unknown_direction_count": (
            unknown_count
        ),
    }


def count_unique_hbond_atoms(
    hydrogen_bonds: Iterable[
        HydrogenBond
    ],
) -> Dict[
    str,
    int,
]:
    """
    Count unique atoms participating in hydrogen bonds.

    Parameters
    ----------
    hydrogen_bonds : iterable of HydrogenBond
        Hydrogen bonds.

    Returns
    -------
    dict
        Unique donor, acceptor and hydrogen counts.
    """

    normalized_bonds = (
        _normalize_hydrogen_bond_collection(
            hydrogen_bonds
        )
    )

    donor_ids = {
        id(
            hydrogen_bond.donor
        )
        for hydrogen_bond in normalized_bonds
    }

    acceptor_ids = {
        id(
            hydrogen_bond.acceptor
        )
        for hydrogen_bond in normalized_bonds
    }

    hydrogen_ids = {
        id(
            hydrogen_bond.hydrogen
        )
        for hydrogen_bond in normalized_bonds
        if hydrogen_bond.hydrogen is not None
    }

    all_heavy_atom_ids = donor_ids | acceptor_ids

    return {
        "unique_donor_count": len(
            donor_ids
        ),
        "unique_acceptor_count": len(
            acceptor_ids
        ),
        "unique_explicit_hydrogen_count": len(
            hydrogen_ids
        ),
        "unique_heavy_atom_count": len(
            all_heavy_atom_ids
        ),
    }


# -----------------------------------------------------------------------------
# Geometric value extraction
# -----------------------------------------------------------------------------

def collect_hydrogen_bond_distances(
    hydrogen_bonds: Iterable[
        HydrogenBond
    ],
) -> Dict[
    str,
    FloatArray,
]:
    """
    Collect hydrogen-bond distance measurements.

    Parameters
    ----------
    hydrogen_bonds : iterable of HydrogenBond
        Hydrogen bonds.

    Returns
    -------
    dict
        Arrays for D...A, H...A and D-H distances.
    """

    normalized_bonds = (
        _normalize_hydrogen_bond_collection(
            hydrogen_bonds
        )
    )

    return {
        "donor_acceptor": _finite_float_array(
            hydrogen_bond
            .donor_acceptor_distance
            for hydrogen_bond
            in normalized_bonds
        ),
        "hydrogen_acceptor": _finite_float_array(
            hydrogen_bond
            .hydrogen_acceptor_distance
            for hydrogen_bond
            in normalized_bonds
        ),
        "donor_hydrogen": _finite_float_array(
            hydrogen_bond
            .geometry
            .donor_hydrogen_distance
            for hydrogen_bond
            in normalized_bonds
        ),
    }


def collect_hydrogen_bond_angles(
    hydrogen_bonds: Iterable[
        HydrogenBond
    ],
) -> Dict[
    str,
    FloatArray,
]:
    """
    Collect hydrogen-bond angular measurements.

    Parameters
    ----------
    hydrogen_bonds : iterable of HydrogenBond
        Hydrogen bonds.

    Returns
    -------
    dict
        Arrays for explicit and inferred angles.
    """

    normalized_bonds = (
        _normalize_hydrogen_bond_collection(
            hydrogen_bonds
        )
    )

    return {
        "dha": _finite_float_array(
            hydrogen_bond.dha_angle
            for hydrogen_bond
            in normalized_bonds
        ),
        "donor_deviation": _finite_float_array(
            hydrogen_bond
            .geometry
            .donor_angle
            for hydrogen_bond
            in normalized_bonds
        ),
        "acceptor_deviation": _finite_float_array(
            hydrogen_bond
            .geometry
            .acceptor_angle
            for hydrogen_bond
            in normalized_bonds
        ),
    }


def collect_hydrogen_bond_strength_scores(
    hydrogen_bonds: Iterable[
        HydrogenBond
    ],
    *,
    config: HydrogenBondClassificationConfig = (
        DEFAULT_HBOND_CLASSIFICATION_CONFIG
    ),
    recalculate: bool = (
        DEFAULT_RECALCULATE_STRENGTHS
    ),
) -> FloatArray:
    """
    Collect geometric-strength scores.

    Parameters
    ----------
    hydrogen_bonds : iterable of HydrogenBond
        Hydrogen bonds.
    config : HydrogenBondClassificationConfig, optional
        Classification configuration.
    recalculate : bool, optional
        Whether scores should always be recalculated.

    Returns
    -------
    numpy.ndarray
        Finite strength scores.
    """

    normalized_bonds = (
        _normalize_hydrogen_bond_collection(
            hydrogen_bonds
        )
    )

    scores: List[
        np.float64
    ] = []

    for hydrogen_bond in normalized_bonds:
        stored_score = hydrogen_bond.metadata.get(
            "geometric_strength_score"
        )

        if (
            not recalculate
            and stored_score is not None
        ):
            try:
                normalized_score = np.float64(
                    stored_score
                )

            except (
                TypeError,
                ValueError,
                OverflowError,
            ):
                normalized_score = np.float64(
                    np.nan
                )

            if np.isfinite(
                normalized_score
            ):
                scores.append(
                    normalized_score
                )

                continue

        strength = get_hydrogen_bond_strength(
            hydrogen_bond,
            config=config,
        )

        scores.append(
            strength.score
        )

    return _finite_float_array(
        scores
    )


# -----------------------------------------------------------------------------
# Geometric statistics
# -----------------------------------------------------------------------------

def hydrogen_bond_distance_statistics(
    hydrogen_bonds: Iterable[
        HydrogenBond
    ],
    *,
    ddof: int = DEFAULT_STATISTICS_DDOF,
) -> Dict[
    str,
    Dict[
        str,
        Any,
    ],
]:
    """
    Calculate distance statistics for hydrogen bonds.

    Parameters
    ----------
    hydrogen_bonds : iterable of HydrogenBond
        Hydrogen bonds.
    ddof : int, optional
        Delta degrees of freedom.

    Returns
    -------
    dict
        Statistics for D...A, H...A and D-H distances.
    """

    distance_values = (
        collect_hydrogen_bond_distances(
            hydrogen_bonds
        )
    )

    return {
        distance_name: summarize_numeric_values(
            values,
            ddof=ddof,
        )
        for distance_name, values
        in distance_values.items()
    }


def hydrogen_bond_angle_statistics(
    hydrogen_bonds: Iterable[
        HydrogenBond
    ],
    *,
    ddof: int = DEFAULT_STATISTICS_DDOF,
) -> Dict[
    str,
    Dict[
        str,
        Any,
    ],
]:
    """
    Calculate angular statistics for hydrogen bonds.

    Parameters
    ----------
    hydrogen_bonds : iterable of HydrogenBond
        Hydrogen bonds.
    ddof : int, optional
        Delta degrees of freedom.

    Returns
    -------
    dict
        Statistics for explicit and inferred angles.
    """

    angle_values = (
        collect_hydrogen_bond_angles(
            hydrogen_bonds
        )
    )

    return {
        angle_name: summarize_numeric_values(
            values,
            ddof=ddof,
        )
        for angle_name, values
        in angle_values.items()
    }


def hydrogen_bond_score_statistics(
    hydrogen_bonds: Iterable[
        HydrogenBond
    ],
    *,
    config: HydrogenBondClassificationConfig = (
        DEFAULT_HBOND_CLASSIFICATION_CONFIG
    ),
    ddof: int = DEFAULT_STATISTICS_DDOF,
    recalculate: bool = (
        DEFAULT_RECALCULATE_STRENGTHS
    ),
) -> Dict[
    str,
    Any,
]:
    """
    Calculate geometric-strength score statistics.

    Parameters
    ----------
    hydrogen_bonds : iterable of HydrogenBond
        Hydrogen bonds.
    config : HydrogenBondClassificationConfig, optional
        Classification configuration.
    ddof : int, optional
        Delta degrees of freedom.
    recalculate : bool, optional
        Whether scores should be recalculated.

    Returns
    -------
    dict
        Strength-score statistics.
    """

    scores = collect_hydrogen_bond_strength_scores(
        hydrogen_bonds,
        config=config,
        recalculate=recalculate,
    )

    return summarize_numeric_values(
        scores,
        ddof=ddof,
    )


# -----------------------------------------------------------------------------
# Classification percentages
# -----------------------------------------------------------------------------

def hydrogen_bond_classification_percentages(
    hydrogen_bonds: Iterable[
        HydrogenBond
    ],
) -> Dict[
    HydrogenBondClassification,
    np.float64,
]:
    """
    Calculate classification percentages.

    Parameters
    ----------
    hydrogen_bonds : iterable of HydrogenBond
        Hydrogen bonds.

    Returns
    -------
    dict
        Percentage of interactions in each classification.
    """

    normalized_bonds = (
        _normalize_hydrogen_bond_collection(
            hydrogen_bonds
        )
    )

    counts = hydrogen_bond_classification_counts(
        normalized_bonds
    )

    total = len(
        normalized_bonds
    )

    if total == 0:
        return {
            classification: np.float64(
                0.0
            )
            for classification
            in VALID_HYDROGEN_BOND_CLASSIFICATIONS
        }

    return {
        classification: np.float64(
            100.0
            * count
            / total
        )
        for classification, count
        in counts.items()
    }


def hydrogen_bond_mode_percentages(
    hydrogen_bonds: Iterable[
        HydrogenBond
    ],
) -> Dict[
    HydrogenBondMode,
    np.float64,
]:
    """
    Calculate explicit and inferred percentages.

    Parameters
    ----------
    hydrogen_bonds : iterable of HydrogenBond
        Hydrogen bonds.

    Returns
    -------
    dict
        Mode percentages.
    """

    normalized_bonds = (
        _normalize_hydrogen_bond_collection(
            hydrogen_bonds
        )
    )

    counts = hydrogen_bond_mode_counts(
        normalized_bonds
    )

    total = len(
        normalized_bonds
    )

    if total == 0:
        return {
            mode: np.float64(
                0.0
            )
            for mode in counts
        }

    return {
        mode: np.float64(
            100.0
            * count
            / total
        )
        for mode, count
        in counts.items()
    }


def hydrogen_bond_direction_percentages(
    hydrogen_bonds: Iterable[
        HydrogenBond
    ],
) -> Dict[
    HydrogenBondDirection,
    np.float64,
]:
    """
    Calculate percentages by donor direction.

    Parameters
    ----------
    hydrogen_bonds : iterable of HydrogenBond
        Hydrogen bonds.

    Returns
    -------
    dict
        Direction percentages.
    """

    normalized_bonds = (
        _normalize_hydrogen_bond_collection(
            hydrogen_bonds
        )
    )

    counts = hydrogen_bond_direction_counts(
        normalized_bonds
    )

    total = len(
        normalized_bonds
    )

    if total == 0:
        return {
            direction: np.float64(
                0.0
            )
            for direction in counts
        }

    return {
        direction: np.float64(
            100.0
            * count
            / total
        )
        for direction, count
        in counts.items()
    }


# -----------------------------------------------------------------------------
# Hydrogen-bond ranking
# -----------------------------------------------------------------------------

def _get_stored_or_calculated_strength_score(
    hydrogen_bond: HydrogenBond,
    *,
    config: HydrogenBondClassificationConfig,
) -> np.float64:
    """
    Return a stored or calculated strength score.

    Parameters
    ----------
    hydrogen_bond : HydrogenBond
        Hydrogen bond.
    config : HydrogenBondClassificationConfig
        Classification configuration.

    Returns
    -------
    numpy.float64
        Strength score.
    """

    stored_score = hydrogen_bond.metadata.get(
        "geometric_strength_score"
    )

    if stored_score is not None:
        try:
            normalized_score = np.float64(
                stored_score
            )

        except (
            TypeError,
            ValueError,
            OverflowError,
        ):
            normalized_score = np.float64(
                np.nan
            )

        if np.isfinite(
            normalized_score
        ):
            return _clamp_unit_interval(
                normalized_score
            )

    return get_hydrogen_bond_strength(
        hydrogen_bond,
        config=config,
    ).score


def rank_hydrogen_bonds(
    hydrogen_bonds: Iterable[
        HydrogenBond
    ],
    *,
    config: HydrogenBondClassificationConfig = (
        DEFAULT_HBOND_CLASSIFICATION_CONFIG
    ),
    strongest_first: bool = True,
) -> Tuple[
    HydrogenBond,
    ...,
]:
    """
    Rank hydrogen bonds by geometric strength.

    Parameters
    ----------
    hydrogen_bonds : iterable of HydrogenBond
        Hydrogen bonds.
    config : HydrogenBondClassificationConfig, optional
        Classification configuration.
    strongest_first : bool, optional
        Whether strongest interactions should appear first.

    Returns
    -------
    tuple of HydrogenBond
        Ranked interactions.
    """

    normalized_bonds = (
        _normalize_hydrogen_bond_collection(
            hydrogen_bonds
        )
    )

    def ranking_key(
        hydrogen_bond: HydrogenBond,
    ) -> Tuple[
        float,
        float,
        float,
        str,
        str,
    ]:
        strength_score = (
            _get_stored_or_calculated_strength_score(
                hydrogen_bond,
                config=config,
            )
        )

        hydrogen_acceptor_distance = (
            hydrogen_bond
            .hydrogen_acceptor_distance
        )

        return (
            -float(
                strength_score
            ),
            float(
                hydrogen_bond
                .donor_acceptor_distance
            ),
            (
                float(
                    np.inf
                )
                if hydrogen_acceptor_distance
                is None
                else float(
                    hydrogen_acceptor_distance
                )
            ),
            (
                _safe_atom_identifier(
                    hydrogen_bond.donor
                )
                or ""
            ),
            (
                _safe_atom_identifier(
                    hydrogen_bond.acceptor
                )
                or ""
            ),
        )

    ranked = sorted(
        normalized_bonds,
        key=ranking_key,
    )

    if strongest_first:
        return tuple(
            ranked
        )

    return tuple(
        reversed(
            ranked
        )
    )


def get_top_hydrogen_bonds(
    hydrogen_bonds: Iterable[
        HydrogenBond
    ],
    *,
    limit: int = DEFAULT_TOP_HYDROGEN_BOND_COUNT,
    config: HydrogenBondClassificationConfig = (
        DEFAULT_HBOND_CLASSIFICATION_CONFIG
    ),
) -> Tuple[
    HydrogenBond,
    ...,
]:
    """
    Return the strongest hydrogen bonds.

    Parameters
    ----------
    hydrogen_bonds : iterable of HydrogenBond
        Hydrogen bonds.
    limit : int, optional
        Maximum number returned.
    config : HydrogenBondClassificationConfig, optional
        Classification configuration.

    Returns
    -------
    tuple of HydrogenBond
        Strongest interactions.
    """

    normalized_limit = _normalize_summary_limit(
        limit,
        name="top hydrogen-bond limit",
    )

    if normalized_limit == 0:
        return ()

    ranked = rank_hydrogen_bonds(
        hydrogen_bonds,
        config=config,
        strongest_first=True,
    )

    return ranked[
        :normalized_limit
    ]


# -----------------------------------------------------------------------------
# Atom and residue identifiers
# -----------------------------------------------------------------------------

def _serialize_residue_key(
    key: Optional[
        ResidueContactKey
    ],
) -> Optional[
    Dict[
        str,
        Any,
    ]
]:
    """
    Serialize a residue key.

    Parameters
    ----------
    key : ResidueContactKey or None
        Residue key.

    Returns
    -------
    dict or None
        Serialized residue identifier.
    """

    if key is None:
        return None

    normalized_key = _normalize_residue_key(
        key
    )

    if normalized_key is None:
        return None

    residue_name, residue_number, chain_id = (
        normalized_key
    )

    return {
        "residue_name": residue_name,
        "residue_number": residue_number,
        "chain_id": chain_id,
    }


def format_residue_key(
    key: Optional[
        ResidueContactKey
    ],
) -> str:
    """
    Format a residue key as a compact label.

    Parameters
    ----------
    key : ResidueContactKey or None
        Residue key.

    Returns
    -------
    str
        Compact residue label.
    """

    if key is None:
        return "unresolved"

    normalized_key = _normalize_residue_key(
        key
    )

    if normalized_key is None:
        return "unresolved"

    residue_name, residue_number, chain_id = (
        normalized_key
    )

    residue_label = (
        f"{residue_name}{residue_number}"
    )

    if chain_id:
        residue_label += f":{chain_id}"

    return residue_label


# -----------------------------------------------------------------------------
# Hydrogen-bond record generation
# -----------------------------------------------------------------------------

def hydrogen_bond_to_summary_record(
    hydrogen_bond: HydrogenBond,
    *,
    config: HydrogenBondClassificationConfig = (
        DEFAULT_HBOND_CLASSIFICATION_CONFIG
    ),
    include_metadata: bool = False,
) -> Dict[
    str,
    Any,
]:
    """
    Convert a hydrogen bond to a flat summary record.

    Parameters
    ----------
    hydrogen_bond : HydrogenBond
        Hydrogen bond.
    config : HydrogenBondClassificationConfig, optional
        Classification configuration.
    include_metadata : bool, optional
        Whether interaction metadata should be included.

    Returns
    -------
    dict
        Flat interaction record.
    """

    if not isinstance(
        hydrogen_bond,
        HydrogenBond,
    ):
        raise TypeError(
            "hydrogen_bond must be a HydrogenBond instance."
        )

    score = _get_stored_or_calculated_strength_score(
        hydrogen_bond,
        config=config,
    )

    donor_residue_key = (
        _get_hbond_donor_residue_key(
            hydrogen_bond
        )
    )

    acceptor_residue_key = (
        _get_hbond_acceptor_residue_key(
            hydrogen_bond
        )
    )

    receptor_residue_key = (
        get_hbond_receptor_residue_key(
            hydrogen_bond
        )
    )

    record: Dict[
        str,
        Any,
    ] = {
        "donor": _safe_atom_identifier(
            hydrogen_bond.donor
        ),
        "hydrogen": (
            None
            if hydrogen_bond.hydrogen is None
            else _safe_atom_identifier(
                hydrogen_bond.hydrogen
            )
        ),
        "acceptor": _safe_atom_identifier(
            hydrogen_bond.acceptor
        ),
        "mode": hydrogen_bond.mode,
        "direction": (
            hydrogen_bond.direction
        ),
        "classification": (
            hydrogen_bond.classification
        ),
        "strength_score": float(
            score
        ),
        "donor_acceptor_distance": float(
            hydrogen_bond
            .donor_acceptor_distance
        ),
        "hydrogen_acceptor_distance": (
            None
            if hydrogen_bond
            .hydrogen_acceptor_distance
            is None
            else float(
                hydrogen_bond
                .hydrogen_acceptor_distance
            )
        ),
        "donor_hydrogen_distance": (
            None
            if hydrogen_bond
            .geometry
            .donor_hydrogen_distance
            is None
            else float(
                hydrogen_bond
                .geometry
                .donor_hydrogen_distance
            )
        ),
        "dha_angle": (
            None
            if hydrogen_bond.dha_angle is None
            else float(
                hydrogen_bond.dha_angle
            )
        ),
        "donor_deviation_angle": (
            None
            if hydrogen_bond
            .geometry
            .donor_angle
            is None
            else float(
                hydrogen_bond
                .geometry
                .donor_angle
            )
        ),
        "acceptor_deviation_angle": (
            None
            if hydrogen_bond
            .geometry
            .acceptor_angle
            is None
            else float(
                hydrogen_bond
                .geometry
                .acceptor_angle
            )
        ),
        "donor_residue": format_residue_key(
            donor_residue_key
        ),
        "acceptor_residue": format_residue_key(
            acceptor_residue_key
        ),
        "receptor_residue": (
            format_residue_key(
                receptor_residue_key
            )
        ),
        "donor_index": (
            hydrogen_bond.donor_index
        ),
        "hydrogen_index": (
            hydrogen_bond.hydrogen_index
        ),
        "acceptor_index": (
            hydrogen_bond.acceptor_index
        ),
    }

    if include_metadata:
        record[
            "metadata"
        ] = dict(
            hydrogen_bond.metadata
        )

    return record


def hydrogen_bonds_to_summary_records(
    hydrogen_bonds: Iterable[
        HydrogenBond
    ],
    *,
    config: HydrogenBondClassificationConfig = (
        DEFAULT_HBOND_CLASSIFICATION_CONFIG
    ),
    include_metadata: bool = False,
    sort_by_strength: bool = False,
) -> Tuple[
    Dict[
        str,
        Any,
    ],
    ...,
]:
    """
    Convert hydrogen bonds to flat summary records.

    Parameters
    ----------
    hydrogen_bonds : iterable of HydrogenBond
        Hydrogen bonds.
    config : HydrogenBondClassificationConfig, optional
        Classification configuration.
    include_metadata : bool, optional
        Whether interaction metadata should be included.
    sort_by_strength : bool, optional
        Whether interactions should be strength-sorted.

    Returns
    -------
    tuple of dict
        Interaction records.
    """

    normalized_bonds = (
        _normalize_hydrogen_bond_collection(
            hydrogen_bonds
        )
    )

    if sort_by_strength:
        normalized_bonds = rank_hydrogen_bonds(
            normalized_bonds,
            config=config,
            strongest_first=True,
        )

    return tuple(
        hydrogen_bond_to_summary_record(
            hydrogen_bond,
            config=config,
            include_metadata=include_metadata,
        )
        for hydrogen_bond
        in normalized_bonds
    )


# -----------------------------------------------------------------------------
# Residue-level statistics
# -----------------------------------------------------------------------------

def residue_hydrogen_bond_strength_statistics(
    residue_group: ResidueHydrogenBond,
    *,
    config: HydrogenBondClassificationConfig = (
        DEFAULT_HBOND_CLASSIFICATION_CONFIG
    ),
    ddof: int = DEFAULT_STATISTICS_DDOF,
) -> Dict[
    str,
    Any,
]:
    """
    Calculate strength statistics for one residue group.

    Parameters
    ----------
    residue_group : ResidueHydrogenBond
        Residue group.
    config : HydrogenBondClassificationConfig, optional
        Classification configuration.
    ddof : int, optional
        Delta degrees of freedom.

    Returns
    -------
    dict
        Residue-level strength statistics.
    """

    if not isinstance(
        residue_group,
        ResidueHydrogenBond,
    ):
        raise TypeError(
            "residue_group must be a ResidueHydrogenBond instance."
        )

    scores = collect_hydrogen_bond_strength_scores(
        residue_group.hydrogen_bonds,
        config=config,
    )

    return summarize_numeric_values(
        scores,
        ddof=ddof,
    )


def residue_hydrogen_bond_summary(
    residue_group: ResidueHydrogenBond,
    *,
    config: HydrogenBondClassificationConfig = (
        DEFAULT_HBOND_CLASSIFICATION_CONFIG
    ),
    ddof: int = DEFAULT_STATISTICS_DDOF,
    include_bond_records: bool = False,
) -> Dict[
    str,
    Any,
]:
    """
    Summarize one residue-level hydrogen-bond group.

    Parameters
    ----------
    residue_group : ResidueHydrogenBond
        Residue group.
    config : HydrogenBondClassificationConfig, optional
        Classification configuration.
    ddof : int, optional
        Delta degrees of freedom.
    include_bond_records : bool, optional
        Whether detailed interaction records should be included.

    Returns
    -------
    dict
        Residue-level summary.
    """

    if not isinstance(
        residue_group,
        ResidueHydrogenBond,
    ):
        raise TypeError(
            "residue_group must be a ResidueHydrogenBond instance."
        )

    hydrogen_bonds = residue_group.hydrogen_bonds

    classification_counts = (
        hydrogen_bond_classification_counts(
            hydrogen_bonds
        )
    )

    mode_counts = hydrogen_bond_mode_counts(
        hydrogen_bonds
    )

    direction_counts = (
        hydrogen_bond_direction_counts(
            hydrogen_bonds
        )
    )

    distance_statistics = (
        hydrogen_bond_distance_statistics(
            hydrogen_bonds,
            ddof=ddof,
        )
    )

    angle_statistics = (
        hydrogen_bond_angle_statistics(
            hydrogen_bonds,
            ddof=ddof,
        )
    )

    score_statistics = (
        hydrogen_bond_score_statistics(
            hydrogen_bonds,
            config=config,
            ddof=ddof,
        )
    )

    strongest_bond: Optional[
        HydrogenBond
    ] = None

    ranked_bonds = rank_hydrogen_bonds(
        hydrogen_bonds,
        config=config,
    )

    if ranked_bonds:
        strongest_bond = ranked_bonds[
            0
        ]

    summary: Dict[
        str,
        Any,
    ] = {
        "residue_key": residue_group.key,
        "residue_label": format_residue_key(
            residue_group.key
        ),
        "side": residue_group.side,
        "hydrogen_bond_count": len(
            hydrogen_bonds
        ),
        "classification_counts": (
            classification_counts
        ),
        "mode_counts": mode_counts,
        "direction_counts": (
            direction_counts
        ),
        "distance_statistics": (
            distance_statistics
        ),
        "angle_statistics": (
            angle_statistics
        ),
        "strength_statistics": (
            score_statistics
        ),
        "strongest_hydrogen_bond": (
            None
            if strongest_bond is None
            else hydrogen_bond_to_summary_record(
                strongest_bond,
                config=config,
            )
        ),
        "has_strong_bond": (
            classification_counts.get(
                HBOND_TYPE_STRONG,
                0,
            )
            > 0
        ),
        "has_explicit_bond": (
            mode_counts.get(
                HBOND_MODE_EXPLICIT,
                0,
            )
            > 0
        ),
    }

    if include_bond_records:
        summary[
            "hydrogen_bonds"
        ] = hydrogen_bonds_to_summary_records(
            hydrogen_bonds,
            config=config,
            sort_by_strength=True,
        )

    return summary


def residue_hydrogen_bond_summaries(
    residue_groups: Iterable[
        ResidueHydrogenBond
    ],
    *,
    config: HydrogenBondClassificationConfig = (
        DEFAULT_HBOND_CLASSIFICATION_CONFIG
    ),
    ddof: int = DEFAULT_STATISTICS_DDOF,
    include_bond_records: bool = False,
    sort_by_count: bool = True,
) -> Tuple[
    Dict[
        str,
        Any,
    ],
    ...,
]:
    """
    Summarize multiple residue-level groups.

    Parameters
    ----------
    residue_groups : iterable of ResidueHydrogenBond
        Residue groups.
    config : HydrogenBondClassificationConfig, optional
        Classification configuration.
    ddof : int, optional
        Delta degrees of freedom.
    include_bond_records : bool, optional
        Whether interaction records should be included.
    sort_by_count : bool, optional
        Whether groups should be sorted by decreasing count.

    Returns
    -------
    tuple of dict
        Residue summaries.
    """

    normalized_groups = (
        _normalize_residue_hbond_collection(
            residue_groups
        )
    )

    summaries = [
        residue_hydrogen_bond_summary(
            residue_group,
            config=config,
            ddof=ddof,
            include_bond_records=(
                include_bond_records
            ),
        )
        for residue_group
        in normalized_groups
    ]

    if sort_by_count:
        summaries.sort(
            key=lambda summary: (
                -int(
                    summary[
                        "hydrogen_bond_count"
                    ]
                ),
                -float(
                    summary[
                        "strength_statistics"
                    ].get(
                        "maximum",
                        0.0,
                    )
                    or 0.0
                ),
                str(
                    summary[
                        "residue_label"
                    ]
                ),
            )
        )

    return tuple(
        summaries
    )


def residue_hydrogen_bond_global_statistics(
    residue_groups: Iterable[
        ResidueHydrogenBond
    ],
    *,
    config: HydrogenBondClassificationConfig = (
        DEFAULT_HBOND_CLASSIFICATION_CONFIG
    ),
    ddof: int = DEFAULT_STATISTICS_DDOF,
) -> Dict[
    str,
    Any,
]:
    """
    Calculate statistics across residue groups.

    Parameters
    ----------
    residue_groups : iterable of ResidueHydrogenBond
        Residue groups.
    config : HydrogenBondClassificationConfig, optional
        Classification configuration.
    ddof : int, optional
        Delta degrees of freedom.

    Returns
    -------
    dict
        Aggregate residue statistics.
    """

    normalized_groups = (
        _normalize_residue_hbond_collection(
            residue_groups
        )
    )

    bond_counts = [
        len(
            residue_group.hydrogen_bonds
        )
        for residue_group
        in normalized_groups
    ]

    maximum_scores: List[
        Optional[
            np.float64
        ]
    ] = []

    mean_scores: List[
        Optional[
            np.float64
        ]
    ] = []

    for residue_group in normalized_groups:
        score_statistics = (
            residue_hydrogen_bond_strength_statistics(
                residue_group,
                config=config,
                ddof=ddof,
            )
        )

        maximum_scores.append(
            score_statistics.get(
                "maximum"
            )
        )

        mean_scores.append(
            score_statistics.get(
                "mean"
            )
        )

    strong_residue_count = sum(
        any(
            hydrogen_bond.classification
            == HBOND_TYPE_STRONG
            for hydrogen_bond
            in residue_group.hydrogen_bonds
        )
        for residue_group
        in normalized_groups
    )

    explicit_residue_count = sum(
        any(
            hydrogen_bond.mode
            == HBOND_MODE_EXPLICIT
            for hydrogen_bond
            in residue_group.hydrogen_bonds
        )
        for residue_group
        in normalized_groups
    )

    return {
        "residue_group_count": len(
            normalized_groups
        ),
        "strong_residue_count": (
            strong_residue_count
        ),
        "explicit_residue_count": (
            explicit_residue_count
        ),
        "hydrogen_bonds_per_residue": (
            summarize_numeric_values(
                bond_counts,
                ddof=ddof,
            )
        ),
        "maximum_strength_per_residue": (
            summarize_numeric_values(
                maximum_scores,
                ddof=ddof,
            )
        ),
        "mean_strength_per_residue": (
            summarize_numeric_values(
                mean_scores,
                ddof=ddof,
            )
        ),
    }


# -----------------------------------------------------------------------------
# Complete statistical summary
# -----------------------------------------------------------------------------

def calculate_hydrogen_bond_statistics(
    hydrogen_bonds: Iterable[
        HydrogenBond
    ],
    *,
    residue_groups: Optional[
        Iterable[
            ResidueHydrogenBond
        ]
    ] = None,
    config: HydrogenBondClassificationConfig = (
        DEFAULT_HBOND_CLASSIFICATION_CONFIG
    ),
    ddof: int = DEFAULT_STATISTICS_DDOF,
    top_hydrogen_bond_count: int = (
        DEFAULT_TOP_HYDROGEN_BOND_COUNT
    ),
    top_residue_count: int = (
        DEFAULT_TOP_RESIDUE_COUNT
    ),
    include_hydrogen_bond_records: bool = (
        DEFAULT_INCLUDE_HYDROGEN_BOND_RECORDS
    ),
    include_residue_records: bool = (
        DEFAULT_INCLUDE_RESIDUE_RECORDS
    ),
) -> Dict[
    str,
    Any,
]:
    """
    Calculate a complete hydrogen-bond statistical summary.

    Parameters
    ----------
    hydrogen_bonds : iterable of HydrogenBond
        Hydrogen bonds.
    residue_groups : iterable of ResidueHydrogenBond or None, optional
        Existing residue groups.
    config : HydrogenBondClassificationConfig, optional
        Classification configuration.
    ddof : int, optional
        Delta degrees of freedom.
    top_hydrogen_bond_count : int, optional
        Number of strongest interactions included.
    top_residue_count : int, optional
        Number of top residues included.
    include_hydrogen_bond_records : bool, optional
        Whether all interaction records should be included.
    include_residue_records : bool, optional
        Whether all residue summaries should be included.

    Returns
    -------
    dict
        Complete statistical summary.
    """

    normalized_bonds = (
        _normalize_hydrogen_bond_collection(
            hydrogen_bonds
        )
    )

    normalized_top_bond_count = (
        _normalize_summary_limit(
            top_hydrogen_bond_count,
            name="top hydrogen-bond count",
        )
    )

    normalized_top_residue_count = (
        _normalize_summary_limit(
            top_residue_count,
            name="top residue count",
        )
    )

    if residue_groups is None:
        normalized_residue_groups: Tuple[
            ResidueHydrogenBond,
            ...,
        ] = ()

    else:
        normalized_residue_groups = (
            _normalize_residue_hbond_collection(
                residue_groups
            )
        )

    mode_counts = hydrogen_bond_mode_counts(
        normalized_bonds
    )

    direction_counts = (
        hydrogen_bond_direction_counts(
            normalized_bonds
        )
    )

    role_counts = hydrogen_bond_role_counts(
        normalized_bonds
    )

    classification_counts = (
        hydrogen_bond_classification_counts(
            normalized_bonds
        )
    )

    classification_percentages = (
        hydrogen_bond_classification_percentages(
            normalized_bonds
        )
    )

    mode_percentages = (
        hydrogen_bond_mode_percentages(
            normalized_bonds
        )
    )

    direction_percentages = (
        hydrogen_bond_direction_percentages(
            normalized_bonds
        )
    )

    unique_atom_counts = count_unique_hbond_atoms(
        normalized_bonds
    )

    distance_statistics = (
        hydrogen_bond_distance_statistics(
            normalized_bonds,
            ddof=ddof,
        )
    )

    angle_statistics = (
        hydrogen_bond_angle_statistics(
            normalized_bonds,
            ddof=ddof,
        )
    )

    strength_statistics = (
        hydrogen_bond_score_statistics(
            normalized_bonds,
            config=config,
            ddof=ddof,
        )
    )

    accepted_count = sum(
        classification_counts.get(
            classification,
            0,
        )
        for classification in (
            HBOND_TYPE_STRONG,
            HBOND_TYPE_MODERATE,
            HBOND_TYPE_WEAK,
            HBOND_TYPE_GEOMETRIC_ONLY,
        )
    )

    rejected_count = (
        classification_counts.get(
            HBOND_TYPE_REJECTED,
            0,
        )
    )

    top_hydrogen_bonds = (
        get_top_hydrogen_bonds(
            normalized_bonds,
            limit=normalized_top_bond_count,
            config=config,
        )
    )

    if normalized_residue_groups:
        ordered_residue_groups = (
            get_top_hbond_residues(
                normalized_residue_groups,
                limit=normalized_top_residue_count,
            )
        )

        residue_statistics = (
            residue_hydrogen_bond_global_statistics(
                normalized_residue_groups,
                config=config,
                ddof=ddof,
            )
        )

    else:
        ordered_residue_groups = ()
        residue_statistics = {
            "residue_group_count": 0,
            "strong_residue_count": 0,
            "explicit_residue_count": 0,
            "hydrogen_bonds_per_residue": (
                summarize_numeric_values(
                    (),
                    ddof=ddof,
                )
            ),
            "maximum_strength_per_residue": (
                summarize_numeric_values(
                    (),
                    ddof=ddof,
                )
            ),
            "mean_strength_per_residue": (
                summarize_numeric_values(
                    (),
                    ddof=ddof,
                )
            ),
        }

    statistics: Dict[
        str,
        Any,
    ] = {
        "hydrogen_bond_count": len(
            normalized_bonds
        ),
        "accepted_hydrogen_bond_count": (
            accepted_count
        ),
        "rejected_hydrogen_bond_count": (
            rejected_count
        ),
        "mode_counts": mode_counts,
        "mode_percentages": (
            mode_percentages
        ),
        "direction_counts": (
            direction_counts
        ),
        "direction_percentages": (
            direction_percentages
        ),
        "role_counts": role_counts,
        "classification_counts": (
            classification_counts
        ),
        "classification_percentages": (
            classification_percentages
        ),
        "unique_atom_counts": (
            unique_atom_counts
        ),
        "distance_statistics": (
            distance_statistics
        ),
        "angle_statistics": (
            angle_statistics
        ),
        "strength_statistics": (
            strength_statistics
        ),
        "residue_statistics": (
            residue_statistics
        ),
        "top_hydrogen_bonds": (
            hydrogen_bonds_to_summary_records(
                top_hydrogen_bonds,
                config=config,
                sort_by_strength=False,
            )
        ),
        "top_residues": (
            residue_hydrogen_bond_summaries(
                ordered_residue_groups,
                config=config,
                ddof=ddof,
                include_bond_records=False,
                sort_by_count=True,
            )
        ),
    }

    if include_hydrogen_bond_records:
        statistics[
            "hydrogen_bond_records"
        ] = hydrogen_bonds_to_summary_records(
            normalized_bonds,
            config=config,
            sort_by_strength=True,
        )

    if include_residue_records:
        statistics[
            "residue_records"
        ] = residue_hydrogen_bond_summaries(
            normalized_residue_groups,
            config=config,
            ddof=ddof,
            include_bond_records=False,
            sort_by_count=True,
        )

    return statistics


# -----------------------------------------------------------------------------
# Serializable statistics
# -----------------------------------------------------------------------------

def _serialize_hbond_statistics_value(
    value: Any,
) -> Any:
    """
    Recursively serialize statistical values.

    Parameters
    ----------
    value : Any
        Value to serialize.

    Returns
    -------
    Any
        JSON-compatible representation when possible.
    """

    if isinstance(
        value,
        Mapping,
    ):
        return {
            str(
                key
            ): _serialize_hbond_statistics_value(
                nested_value
            )
            for key, nested_value
            in value.items()
        }

    if isinstance(
        value,
        np.ndarray,
    ):
        return [
            _serialize_hbond_statistics_value(
                item
            )
            for item in value.tolist()
        ]

    if isinstance(
        value,
        (
            tuple,
            list,
            set,
            frozenset,
        ),
    ):
        return [
            _serialize_hbond_statistics_value(
                item
            )
            for item in value
        ]

    if isinstance(
        value,
        np.integer,
    ):
        return int(
            value
        )

    if isinstance(
        value,
        np.floating,
    ):
        return float(
            value
        )

    if isinstance(
        value,
        np.bool_,
    ):
        return bool(
            value
        )

    return value


def serialize_hydrogen_bond_statistics(
    statistics: Mapping[
        str,
        Any,
    ],
) -> Dict[
    str,
    Any,
]:
    """
    Convert hydrogen-bond statistics to serializable values.

    Parameters
    ----------
    statistics : mapping
        Statistical summary.

    Returns
    -------
    dict
        Serializable summary.
    """

    if not isinstance(
        statistics,
        Mapping,
    ):
        raise TypeError(
            "statistics must be a mapping."
        )

    return {
        str(
            key
        ): _serialize_hbond_statistics_value(
            value
        )
        for key, value
        in statistics.items()
    }


# -----------------------------------------------------------------------------
# Human-readable summaries
# -----------------------------------------------------------------------------

def _format_optional_number(
    value: Optional[
        Number
    ],
    *,
    decimal_places: int,
    suffix: str = "",
) -> str:
    """
    Format an optional numeric value.

    Parameters
    ----------
    value : Number or None
        Numeric value.
    decimal_places : int
        Number of decimal places.
    suffix : str, optional
        Value suffix.

    Returns
    -------
    str
        Formatted value.
    """

    if value is None:
        return "n/a"

    try:
        normalized_value = np.float64(
            value
        )

    except (
        TypeError,
        ValueError,
        OverflowError,
    ):
        return "n/a"

    if not np.isfinite(
        normalized_value
    ):
        return "n/a"

    return (
        f"{float(normalized_value):."
        f"{decimal_places}f}{suffix}"
    )


def format_hydrogen_bond_summary(
    statistics: Mapping[
        str,
        Any,
    ],
    *,
    decimal_places: int = (
        DEFAULT_SUMMARY_DECIMAL_PLACES
    ),
    multiline: bool = True,
) -> str:
    """
    Format hydrogen-bond statistics as readable text.

    Parameters
    ----------
    statistics : mapping
        Statistical summary.
    decimal_places : int, optional
        Decimal places for numeric values.
    multiline : bool, optional
        Whether the output should contain multiple lines.

    Returns
    -------
    str
        Human-readable summary.
    """

    normalized_decimal_places = (
        _optional_nonnegative_integer(
            decimal_places,
            name="summary decimal places",
        )
    )

    if normalized_decimal_places is None:
        normalized_decimal_places = (
            DEFAULT_SUMMARY_DECIMAL_PLACES
        )

    total_count = int(
        statistics.get(
            "hydrogen_bond_count",
            0,
        )
    )

    accepted_count = int(
        statistics.get(
            "accepted_hydrogen_bond_count",
            0,
        )
    )

    classification_counts = statistics.get(
        "classification_counts",
        {},
    )

    mode_counts = statistics.get(
        "mode_counts",
        {},
    )

    direction_counts = statistics.get(
        "direction_counts",
        {},
    )

    distance_statistics = statistics.get(
        "distance_statistics",
        {},
    )

    angle_statistics = statistics.get(
        "angle_statistics",
        {},
    )

    strength_statistics = statistics.get(
        "strength_statistics",
        {},
    )

    residue_statistics = statistics.get(
        "residue_statistics",
        {},
    )

    donor_acceptor_statistics = (
        distance_statistics.get(
            "donor_acceptor",
            {},
        )
    )

    dha_statistics = angle_statistics.get(
        "dha",
        {},
    )

    lines = [
        (
            "Hydrogen bonds: "
            f"{total_count} total, "
            f"{accepted_count} accepted."
        ),
        (
            "Classification: "
            f"{classification_counts.get(HBOND_TYPE_STRONG, 0)} strong, "
            f"{classification_counts.get(HBOND_TYPE_MODERATE, 0)} moderate, "
            f"{classification_counts.get(HBOND_TYPE_WEAK, 0)} weak, "
            f"{classification_counts.get(HBOND_TYPE_GEOMETRIC_ONLY, 0)} "
            "geometric-only, "
            f"{classification_counts.get(HBOND_TYPE_REJECTED, 0)} rejected."
        ),
        (
            "Mode: "
            f"{mode_counts.get(HBOND_MODE_EXPLICIT, 0)} explicit, "
            f"{mode_counts.get(HBOND_MODE_INFERRED, 0)} inferred."
        ),
        (
            "Direction: "
            f"{direction_counts.get(HBOND_DIRECTION_LIGAND_DONOR, 0)} "
            "ligand-donor, "
            f"{direction_counts.get(HBOND_DIRECTION_RECEPTOR_DONOR, 0)} "
            "receptor-donor."
        ),
        (
            "D...A distance: mean "
            f"{_format_optional_number(
                donor_acceptor_statistics.get('mean'),
                decimal_places=normalized_decimal_places,
                suffix=' Å',
            )}, median "
            f"{_format_optional_number(
                donor_acceptor_statistics.get('median'),
                decimal_places=normalized_decimal_places,
                suffix=' Å',
            )}."
        ),
        (
            "D-H...A angle: mean "
            f"{_format_optional_number(
                dha_statistics.get('mean'),
                decimal_places=normalized_decimal_places,
                suffix='°',
            )}."
        ),
        (
            "Geometric strength score: mean "
            f"{_format_optional_number(
                strength_statistics.get('mean'),
                decimal_places=normalized_decimal_places,
            )}, maximum "
            f"{_format_optional_number(
                strength_statistics.get('maximum'),
                decimal_places=normalized_decimal_places,
            )}."
        ),
        (
            "Interacting receptor residue groups: "
            f"{residue_statistics.get('residue_group_count', 0)}."
        ),
    ]

    separator = (
        "\n"
        if multiline
        else " "
    )

    return separator.join(
        lines
    )


def summarize_hydrogen_bond_analysis_result(
    result: HydrogenBondAnalysisResult,
    *,
    decimal_places: int = (
        DEFAULT_SUMMARY_DECIMAL_PLACES
    ),
    multiline: bool = True,
) -> str:
    """
    Format an analysis result as a readable summary.

    Parameters
    ----------
    result : HydrogenBondAnalysisResult
        Analysis result.
    decimal_places : int, optional
        Decimal places.
    multiline : bool, optional
        Whether the summary should span multiple lines.

    Returns
    -------
    str
        Human-readable summary.
    """

    if not isinstance(
        result,
        HydrogenBondAnalysisResult,
    ):
        raise TypeError(
            "result must be a HydrogenBondAnalysisResult instance."
        )

    if result.statistics:
        statistics = result.statistics

    else:
        statistics = (
            calculate_hydrogen_bond_statistics(
                result.hydrogen_bonds,
                residue_groups=(
                    result.residue_hydrogen_bonds
                ),
            )
        )

    return format_hydrogen_bond_summary(
        statistics,
        decimal_places=decimal_places,
        multiline=multiline,
    )


# -----------------------------------------------------------------------------
# Analysis-result integration
# -----------------------------------------------------------------------------

def attach_hydrogen_bond_statistics(
    result: HydrogenBondAnalysisResult,
    statistics: Mapping[
        str,
        Any,
    ],
    *,
    replace: bool = True,
    metadata: Optional[
        Mapping[
            str,
            Any,
        ]
    ] = None,
) -> HydrogenBondAnalysisResult:
    """
    Attach statistics to an immutable analysis result.

    Parameters
    ----------
    result : HydrogenBondAnalysisResult
        Original analysis result.
    statistics : mapping
        Statistics to attach.
    replace : bool, optional
        Whether existing statistics should be replaced.
    metadata : mapping or None, optional
        Additional result metadata.

    Returns
    -------
    HydrogenBondAnalysisResult
        Updated immutable result.
    """

    if not isinstance(
        result,
        HydrogenBondAnalysisResult,
    ):
        raise TypeError(
            "result must be a HydrogenBondAnalysisResult instance."
        )

    if not isinstance(
        statistics,
        Mapping,
    ):
        raise TypeError(
            "statistics must be a mapping."
        )

    if replace:
        final_statistics = dict(
            statistics
        )

    else:
        final_statistics = dict(
            result.statistics
        )

        final_statistics.update(
            statistics
        )

    updated_metadata = dict(
        result.metadata
    )

    updated_metadata.update(
        {
            "statistics_attached": True,
            "statistics_replaced": bool(
                replace
            ),
        }
    )

    if metadata:
        updated_metadata.update(
            metadata
        )

    return HydrogenBondAnalysisResult(
        hydrogen_bonds=(
            result.hydrogen_bonds
        ),
        residue_hydrogen_bonds=(
            result.residue_hydrogen_bonds
        ),
        ligand_atoms=result.ligand_atoms,
        receptor_atoms=(
            result.receptor_atoms
        ),
        donor_acceptor_cutoff=(
            result.donor_acceptor_cutoff
        ),
        hydrogen_acceptor_cutoff=(
            result.hydrogen_acceptor_cutoff
        ),
        minimum_dha_angle=(
            result.minimum_dha_angle
        ),
        minimum_inferred_angle=(
            result.minimum_inferred_angle
        ),
        statistics=final_statistics,
        metadata=updated_metadata,
    )


def calculate_analysis_hydrogen_bond_statistics(
    result: HydrogenBondAnalysisResult,
    *,
    config: HydrogenBondClassificationConfig = (
        DEFAULT_HBOND_CLASSIFICATION_CONFIG
    ),
    ddof: int = DEFAULT_STATISTICS_DDOF,
    top_hydrogen_bond_count: int = (
        DEFAULT_TOP_HYDROGEN_BOND_COUNT
    ),
    top_residue_count: int = (
        DEFAULT_TOP_RESIDUE_COUNT
    ),
    include_hydrogen_bond_records: bool = (
        DEFAULT_INCLUDE_HYDROGEN_BOND_RECORDS
    ),
    include_residue_records: bool = (
        DEFAULT_INCLUDE_RESIDUE_RECORDS
    ),
    replace: bool = True,
) -> HydrogenBondAnalysisResult:
    """
    Calculate and attach statistics to an analysis result.

    Parameters
    ----------
    result : HydrogenBondAnalysisResult
        Analysis result.
    config : HydrogenBondClassificationConfig, optional
        Classification configuration.
    ddof : int, optional
        Delta degrees of freedom.
    top_hydrogen_bond_count : int, optional
        Number of top interactions included.
    top_residue_count : int, optional
        Number of top residues included.
    include_hydrogen_bond_records : bool, optional
        Whether all interaction records should be included.
    include_residue_records : bool, optional
        Whether all residue records should be included.
    replace : bool, optional
        Whether existing statistics should be replaced.

    Returns
    -------
    HydrogenBondAnalysisResult
        Result containing statistics.
    """

    if not isinstance(
        result,
        HydrogenBondAnalysisResult,
    ):
        raise TypeError(
            "result must be a HydrogenBondAnalysisResult instance."
        )

    statistics = calculate_hydrogen_bond_statistics(
        result.hydrogen_bonds,
        residue_groups=(
            result.residue_hydrogen_bonds
        ),
        config=config,
        ddof=ddof,
        top_hydrogen_bond_count=(
            top_hydrogen_bond_count
        ),
        top_residue_count=(
            top_residue_count
        ),
        include_hydrogen_bond_records=(
            include_hydrogen_bond_records
        ),
        include_residue_records=(
            include_residue_records
        ),
    )

    return attach_hydrogen_bond_statistics(
        result,
        statistics,
        replace=replace,
        metadata={
            "statistics_ddof": int(
                ddof
            ),
            "top_hydrogen_bond_count": int(
                top_hydrogen_bond_count
            ),
            "top_residue_count": int(
                top_residue_count
            ),
        },
    )


def analyze_group_classify_and_summarize_hydrogen_bonds(
    ligand_atoms: Iterable[
        AtomLike
    ],
    receptor_atoms: Iterable[
        AtomLike
    ],
    *,
    residue_side: str = (
        DEFAULT_RESIDUE_GROUP_SIDE
    ),
    classification_config: HydrogenBondClassificationConfig = (
        DEFAULT_HBOND_CLASSIFICATION_CONFIG
    ),
    include_rejected: bool = False,
    sort_by_strength: bool = True,
    statistics_ddof: int = (
        DEFAULT_STATISTICS_DDOF
    ),
    top_hydrogen_bond_count: int = (
        DEFAULT_TOP_HYDROGEN_BOND_COUNT
    ),
    top_residue_count: int = (
        DEFAULT_TOP_RESIDUE_COUNT
    ),
    include_hydrogen_bond_records: bool = False,
    include_residue_records: bool = False,
    **analysis_kwargs: Any,
) -> HydrogenBondAnalysisResult:
    """
    Detect, group, classify and summarize hydrogen bonds.

    Parameters
    ----------
    ligand_atoms : iterable of atom-like
        Ligand atoms.
    receptor_atoms : iterable of atom-like
        Receptor atoms.
    residue_side : str, optional
        Residue grouping side.
    classification_config : HydrogenBondClassificationConfig, optional
        Classification configuration.
    include_rejected : bool, optional
        Whether rejected interactions should be retained.
    sort_by_strength : bool, optional
        Whether interactions should be strength-sorted.
    statistics_ddof : int, optional
        Delta degrees of freedom.
    top_hydrogen_bond_count : int, optional
        Number of strongest interactions included.
    top_residue_count : int, optional
        Number of top residue groups included.
    include_hydrogen_bond_records : bool, optional
        Whether all interaction records should be stored.
    include_residue_records : bool, optional
        Whether all residue summaries should be stored.
    **analysis_kwargs : Any
        Additional arguments forwarded to hydrogen-bond detection.

    Returns
    -------
    HydrogenBondAnalysisResult
        Complete result with groups, classifications and statistics.
    """

    classified_result = (
        analyze_group_and_classify_hydrogen_bonds(
            ligand_atoms,
            receptor_atoms,
            residue_side=residue_side,
            classification_config=(
                classification_config
            ),
            include_rejected=include_rejected,
            sort_by_strength=(
                sort_by_strength
            ),
            **analysis_kwargs,
        )
    )

    return calculate_analysis_hydrogen_bond_statistics(
        classified_result,
        config=classification_config,
        ddof=statistics_ddof,
        top_hydrogen_bond_count=(
            top_hydrogen_bond_count
        ),
        top_residue_count=(
            top_residue_count
        ),
        include_hydrogen_bond_records=(
            include_hydrogen_bond_records
        ),
        include_residue_records=(
            include_residue_records
        ),
        replace=True,
    )


# -----------------------------------------------------------------------------
# Public interface
# -----------------------------------------------------------------------------

_SECTION_10_PUBLIC_NAMES: Final[
    Tuple[
        str,
        ...,
    ]
] = (
    "DEFAULT_STATISTICS_DDOF",
    "DEFAULT_TOP_HYDROGEN_BOND_COUNT",
    "DEFAULT_TOP_RESIDUE_COUNT",
    "DEFAULT_SUMMARY_DECIMAL_PLACES",
    "DEFAULT_INCLUDE_EMPTY_STATISTICS",
    "DEFAULT_INCLUDE_HYDROGEN_BOND_RECORDS",
    "DEFAULT_INCLUDE_RESIDUE_RECORDS",
    "DEFAULT_RECALCULATE_STRENGTHS",
    "summarize_numeric_values",
    "hydrogen_bond_mode_counts",
    "hydrogen_bond_direction_counts",
    "hydrogen_bond_role_counts",
    "count_unique_hbond_atoms",
    "collect_hydrogen_bond_distances",
    "collect_hydrogen_bond_angles",
    "collect_hydrogen_bond_strength_scores",
    "hydrogen_bond_distance_statistics",
    "hydrogen_bond_angle_statistics",
    "hydrogen_bond_score_statistics",
    "hydrogen_bond_classification_percentages",
    "hydrogen_bond_mode_percentages",
    "hydrogen_bond_direction_percentages",
    "rank_hydrogen_bonds",
    "get_top_hydrogen_bonds",
    "format_residue_key",
    "hydrogen_bond_to_summary_record",
    "hydrogen_bonds_to_summary_records",
    "residue_hydrogen_bond_strength_statistics",
    "residue_hydrogen_bond_summary",
    "residue_hydrogen_bond_summaries",
    "residue_hydrogen_bond_global_statistics",
    "calculate_hydrogen_bond_statistics",
    "serialize_hydrogen_bond_statistics",
    "format_hydrogen_bond_summary",
    "summarize_hydrogen_bond_analysis_result",
    "attach_hydrogen_bond_statistics",
    "calculate_analysis_hydrogen_bond_statistics",
    "analyze_group_classify_and_summarize_hydrogen_bonds",
)

for public_name in _SECTION_10_PUBLIC_NAMES:
    if public_name not in __all__:
        __all__.append(
            public_name
        )


# =============================================================================
# End of Section 10
# =============================================================================


