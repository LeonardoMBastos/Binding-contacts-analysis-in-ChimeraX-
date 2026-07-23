# =============================================================================
# 1. IMPORTS, CONSTANTS AND CONFIGURATION
# =============================================================================
"""
Detection and characterization of pi-related molecular interactions.

This module identifies and evaluates interactions involving aromatic systems,
including:

    - pi-pi stacking;
    - cation-pi interactions;
    - anion-pi interactions;
    - amide-pi interactions;
    - sulfur-pi interactions, when enabled.

The implementation is designed to work both:

    1. inside UCSF ChimeraX, using native atomic objects; and
    2. outside ChimeraX, using compatible Python objects for testing,
       serialization, and batch analysis.

The module does not modify the internal behavior of DockModel. Results are
generated independently and can later be attached to a DockModel instance.
"""

from __future__ import annotations

# -----------------------------------------------------------------------------
# 1.1. Standard-library imports
# -----------------------------------------------------------------------------

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from enum import Enum
from itertools import combinations
from math import acos, degrees, isfinite, sqrt
from statistics import fmean
from typing import (
    Any,
    Callable,
    ClassVar,
    Collection,
    DefaultDict,
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

import json
import warnings


# -----------------------------------------------------------------------------
# 1.2. Optional NumPy support
# -----------------------------------------------------------------------------

try:
    import numpy as np
    from numpy.typing import NDArray

    NUMPY_AVAILABLE: Final[bool] = True

except ImportError:  # pragma: no cover - depends on the execution environment
    np = None  # type: ignore[assignment]
    NDArray = Any  # type: ignore[misc,assignment]

    NUMPY_AVAILABLE = False


# -----------------------------------------------------------------------------
# 1.3. Optional ChimeraX support
# -----------------------------------------------------------------------------

try:
    from chimerax.atomic import Atom as ChimeraXAtom
    from chimerax.atomic import AtomicStructure as ChimeraXAtomicStructure
    from chimerax.atomic import Residue as ChimeraXResidue

    CHIMERAX_AVAILABLE: Final[bool] = True

except ImportError:  # pragma: no cover - expected outside ChimeraX
    ChimeraXAtom = Any  # type: ignore[misc,assignment]
    ChimeraXAtomicStructure = Any  # type: ignore[misc,assignment]
    ChimeraXResidue = Any  # type: ignore[misc,assignment]

    CHIMERAX_AVAILABLE = False


# -----------------------------------------------------------------------------
# 1.4. Module metadata
# -----------------------------------------------------------------------------

MODULE_NAME: Final[str] = "pi"
MODULE_VERSION: Final[str] = "1.0.0"

MODULE_DESCRIPTION: Final[str] = (
    "Detection, classification, scoring, grouping, and serialization of "
    "pi-related protein-ligand interactions."
)


# -----------------------------------------------------------------------------
# 1.5. General type aliases
# -----------------------------------------------------------------------------

Number: TypeAlias = Union[int, float]

Coordinate3D: TypeAlias = Tuple[float, float, float]

Vector3D: TypeAlias = Tuple[float, float, float]

AtomCollection: TypeAlias = Sequence[Any]

ResidueCollection: TypeAlias = Sequence[Any]

ModelCollection: TypeAlias = Sequence[Any]

JSONPrimitive: TypeAlias = Union[str, int, float, bool, None]

JSONValue: TypeAlias = Union[
    JSONPrimitive,
    List["JSONValue"],
    Dict[str, "JSONValue"],
]

T = TypeVar("T")


# -----------------------------------------------------------------------------
# 1.6. Structural protocols
# -----------------------------------------------------------------------------

@runtime_checkable
class CoordinateLike(Protocol):
    """Protocol for coordinate-like objects."""

    def __getitem__(self, index: int) -> Number:
        ...


@runtime_checkable
class AtomLike(Protocol):
    """
    Minimum interface expected from atom-like objects.

    Native ChimeraX atoms provide more attributes than those described here.
    Test atoms and external molecular objects only need to expose equivalent
    information through attributes recognized by the normalization functions.
    """

    name: str


@runtime_checkable
class ResidueLike(Protocol):
    """Minimum interface expected from residue-like objects."""

    name: str


# -----------------------------------------------------------------------------
# 1.7. Numerical constants
# -----------------------------------------------------------------------------

EPSILON: Final[float] = 1.0e-12

ANGLE_EPSILON_DEGREES: Final[float] = 1.0e-6

MINIMUM_VECTOR_NORM: Final[float] = 1.0e-8

DEFAULT_FLOAT_PRECISION: Final[int] = 4

DEFAULT_SCORE_PRECISION: Final[int] = 4

DEFAULT_COORDINATE_PRECISION: Final[int] = 6

DEGREES_IN_HALF_ROTATION: Final[float] = 180.0

RIGHT_ANGLE_DEGREES: Final[float] = 90.0


# -----------------------------------------------------------------------------
# 1.8. Interaction type names
# -----------------------------------------------------------------------------

PI_PI: Final[str] = "pi_pi"

CATION_PI: Final[str] = "cation_pi"

ANION_PI: Final[str] = "anion_pi"

AMIDE_PI: Final[str] = "amide_pi"

SULFUR_PI: Final[str] = "sulfur_pi"


SUPPORTED_PI_INTERACTION_TYPES: Final[FrozenSet[str]] = frozenset(
    {
        PI_PI,
        CATION_PI,
        ANION_PI,
        AMIDE_PI,
        SULFUR_PI,
    }
)


DEFAULT_ENABLED_PI_INTERACTION_TYPES: Final[FrozenSet[str]] = frozenset(
    {
        PI_PI,
        CATION_PI,
        ANION_PI,
        AMIDE_PI,
    }
)


# -----------------------------------------------------------------------------
# 1.9. Interaction geometry labels
# -----------------------------------------------------------------------------

PI_PI_PARALLEL: Final[str] = "parallel"

PI_PI_FACE_TO_FACE: Final[str] = "face_to_face"

PI_PI_OFFSET_STACKED: Final[str] = "offset_stacked"

PI_PI_T_SHAPED: Final[str] = "t_shaped"

PI_PI_EDGE_TO_FACE: Final[str] = "edge_to_face"

PI_PI_INTERMEDIATE: Final[str] = "intermediate"

PI_PI_UNCLASSIFIED: Final[str] = "unclassified"


PI_PI_GEOMETRY_TYPES: Final[FrozenSet[str]] = frozenset(
    {
        PI_PI_PARALLEL,
        PI_PI_FACE_TO_FACE,
        PI_PI_OFFSET_STACKED,
        PI_PI_T_SHAPED,
        PI_PI_EDGE_TO_FACE,
        PI_PI_INTERMEDIATE,
        PI_PI_UNCLASSIFIED,
    }
)


# -----------------------------------------------------------------------------
# 1.10. Geometric quality and strength labels
# -----------------------------------------------------------------------------

GEOMETRY_OPTIMAL: Final[str] = "optimal"

GEOMETRY_FAVORABLE: Final[str] = "favorable"

GEOMETRY_WEAK: Final[str] = "weak"

GEOMETRY_BORDERLINE: Final[str] = "borderline"

GEOMETRY_REJECTED: Final[str] = "rejected"


GEOMETRY_CLASSES: Final[Tuple[str, ...]] = (
    GEOMETRY_OPTIMAL,
    GEOMETRY_FAVORABLE,
    GEOMETRY_WEAK,
    GEOMETRY_BORDERLINE,
    GEOMETRY_REJECTED,
)


STRENGTH_STRONG: Final[str] = "strong"

STRENGTH_MODERATE: Final[str] = "moderate"

STRENGTH_WEAK: Final[str] = "weak"

STRENGTH_UNCLASSIFIED: Final[str] = "unclassified"


STRENGTH_CLASSES: Final[Tuple[str, ...]] = (
    STRENGTH_STRONG,
    STRENGTH_MODERATE,
    STRENGTH_WEAK,
    STRENGTH_UNCLASSIFIED,
)


# -----------------------------------------------------------------------------
# 1.11. Molecular participant labels
# -----------------------------------------------------------------------------

PARTICIPANT_RECEPTOR: Final[str] = "receptor"

PARTICIPANT_LIGAND: Final[str] = "ligand"

PARTICIPANT_PROTEIN: Final[str] = "protein"

PARTICIPANT_NUCLEIC_ACID: Final[str] = "nucleic_acid"

PARTICIPANT_COFACTOR: Final[str] = "cofactor"

PARTICIPANT_UNKNOWN: Final[str] = "unknown"


# -----------------------------------------------------------------------------
# 1.12. Aromatic amino-acid residues
# -----------------------------------------------------------------------------

AROMATIC_RESIDUES: Final[FrozenSet[str]] = frozenset(
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


PHENYLALANINE_NAMES: Final[FrozenSet[str]] = frozenset(
    {
        "PHE",
    }
)


TYROSINE_NAMES: Final[FrozenSet[str]] = frozenset(
    {
        "TYR",
    }
)


TRYPTOPHAN_NAMES: Final[FrozenSet[str]] = frozenset(
    {
        "TRP",
    }
)


HISTIDINE_NAMES: Final[FrozenSet[str]] = frozenset(
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


PROTONATED_HISTIDINE_NAMES: Final[FrozenSet[str]] = frozenset(
    {
        "HIP",
        "HSP",
    }
)


# -----------------------------------------------------------------------------
# 1.13. Aromatic atom definitions for standard amino acids
# -----------------------------------------------------------------------------

PHE_AROMATIC_RING_ATOMS: Final[Tuple[str, ...]] = (
    "CG",
    "CD1",
    "CE1",
    "CZ",
    "CE2",
    "CD2",
)


TYR_AROMATIC_RING_ATOMS: Final[Tuple[str, ...]] = (
    "CG",
    "CD1",
    "CE1",
    "CZ",
    "CE2",
    "CD2",
)


HIS_AROMATIC_RING_ATOMS: Final[Tuple[str, ...]] = (
    "CG",
    "ND1",
    "CE1",
    "NE2",
    "CD2",
)


TRP_FIVE_MEMBER_RING_ATOMS: Final[Tuple[str, ...]] = (
    "CG",
    "CD1",
    "NE1",
    "CE2",
    "CD2",
)


TRP_SIX_MEMBER_RING_ATOMS: Final[Tuple[str, ...]] = (
    "CD2",
    "CE2",
    "CZ2",
    "CH2",
    "CZ3",
    "CE3",
)


TRP_FUSED_SYSTEM_ATOMS: Final[Tuple[str, ...]] = (
    "CG",
    "CD1",
    "NE1",
    "CE2",
    "CZ2",
    "CH2",
    "CZ3",
    "CE3",
    "CD2",
)


STANDARD_AROMATIC_RING_ATOMS: Final[
    Mapping[str, Tuple[Tuple[str, ...], ...]]
] = {
    "PHE": (
        PHE_AROMATIC_RING_ATOMS,
    ),
    "TYR": (
        TYR_AROMATIC_RING_ATOMS,
    ),
    "HIS": (
        HIS_AROMATIC_RING_ATOMS,
    ),
    "HID": (
        HIS_AROMATIC_RING_ATOMS,
    ),
    "HIE": (
        HIS_AROMATIC_RING_ATOMS,
    ),
    "HIP": (
        HIS_AROMATIC_RING_ATOMS,
    ),
    "HSD": (
        HIS_AROMATIC_RING_ATOMS,
    ),
    "HSE": (
        HIS_AROMATIC_RING_ATOMS,
    ),
    "HSP": (
        HIS_AROMATIC_RING_ATOMS,
    ),
    "TRP": (
        TRP_FIVE_MEMBER_RING_ATOMS,
        TRP_SIX_MEMBER_RING_ATOMS,
    ),
}


# -----------------------------------------------------------------------------
# 1.14. Nucleic-acid aromatic residues
# -----------------------------------------------------------------------------

PURINE_RESIDUES: Final[FrozenSet[str]] = frozenset(
    {
        "A",
        "ADE",
        "DA",
        "G",
        "GUA",
        "DG",
    }
)


PYRIMIDINE_RESIDUES: Final[FrozenSet[str]] = frozenset(
    {
        "C",
        "CYT",
        "DC",
        "T",
        "THY",
        "DT",
        "U",
        "URA",
        "DU",
    }
)


NUCLEIC_ACID_AROMATIC_RESIDUES: Final[FrozenSet[str]] = frozenset(
    PURINE_RESIDUES | PYRIMIDINE_RESIDUES
)


# -----------------------------------------------------------------------------
# 1.15. Aromatic elements and ring-size limits
# -----------------------------------------------------------------------------

COMMON_AROMATIC_ELEMENTS: Final[FrozenSet[str]] = frozenset(
    {
        "C",
        "N",
        "O",
        "S",
        "P",
        "B",
    }
)


DEFAULT_MINIMUM_RING_SIZE: Final[int] = 5

DEFAULT_MAXIMUM_RING_SIZE: Final[int] = 7

DEFAULT_MAXIMUM_FUSED_RING_SIZE: Final[int] = 12

DEFAULT_MINIMUM_AROMATIC_ATOMS: Final[int] = 5


# -----------------------------------------------------------------------------
# 1.16. Positively charged protein groups
# -----------------------------------------------------------------------------

CATIONIC_RESIDUES: Final[FrozenSet[str]] = frozenset(
    {
        "ARG",
        "LYS",
        "HIP",
        "HSP",
    }
)


ARG_CATIONIC_ATOMS: Final[Tuple[str, ...]] = (
    "CZ",
    "NE",
    "NH1",
    "NH2",
)


LYS_CATIONIC_ATOMS: Final[Tuple[str, ...]] = (
    "NZ",
)


PROTONATED_HIS_CATIONIC_ATOMS: Final[Tuple[str, ...]] = (
    "ND1",
    "CE1",
    "NE2",
    "CD2",
    "CG",
)


STANDARD_CATIONIC_GROUP_ATOMS: Final[
    Mapping[str, Tuple[str, ...]]
] = {
    "ARG": ARG_CATIONIC_ATOMS,
    "LYS": LYS_CATIONIC_ATOMS,
    "HIP": PROTONATED_HIS_CATIONIC_ATOMS,
    "HSP": PROTONATED_HIS_CATIONIC_ATOMS,
}


# -----------------------------------------------------------------------------
# 1.17. Negatively charged protein groups
# -----------------------------------------------------------------------------

ANIONIC_RESIDUES: Final[FrozenSet[str]] = frozenset(
    {
        "ASP",
        "GLU",
    }
)


ASP_ANIONIC_ATOMS: Final[Tuple[str, ...]] = (
    "CG",
    "OD1",
    "OD2",
)


GLU_ANIONIC_ATOMS: Final[Tuple[str, ...]] = (
    "CD",
    "OE1",
    "OE2",
)


STANDARD_ANIONIC_GROUP_ATOMS: Final[
    Mapping[str, Tuple[str, ...]]
] = {
    "ASP": ASP_ANIONIC_ATOMS,
    "GLU": GLU_ANIONIC_ATOMS,
}


# -----------------------------------------------------------------------------
# 1.18. Protein amide groups
# -----------------------------------------------------------------------------

AMIDE_RESIDUES: Final[FrozenSet[str]] = frozenset(
    {
        "ASN",
        "GLN",
    }
)


ASN_AMIDE_ATOMS: Final[Tuple[str, ...]] = (
    "CG",
    "OD1",
    "ND2",
)


GLN_AMIDE_ATOMS: Final[Tuple[str, ...]] = (
    "CD",
    "OE1",
    "NE2",
)


STANDARD_AMIDE_GROUP_ATOMS: Final[
    Mapping[str, Tuple[str, ...]]
] = {
    "ASN": ASN_AMIDE_ATOMS,
    "GLN": GLN_AMIDE_ATOMS,
}


BACKBONE_AMIDE_ATOMS: Final[Tuple[str, ...]] = (
    "C",
    "O",
    "N",
)


# -----------------------------------------------------------------------------
# 1.19. Sulfur-containing groups
# -----------------------------------------------------------------------------

SULFUR_CONTAINING_RESIDUES: Final[FrozenSet[str]] = frozenset(
    {
        "CYS",
        "CYM",
        "CYX",
        "MET",
    }
)


STANDARD_SULFUR_GROUP_ATOMS: Final[
    Mapping[str, Tuple[str, ...]]
] = {
    "CYS": ("SG",),
    "CYM": ("SG",),
    "CYX": ("SG",),
    "MET": ("SD",),
}


# -----------------------------------------------------------------------------
# 1.20. Atom-name and element conventions
# -----------------------------------------------------------------------------

HYDROGEN_ELEMENTS: Final[FrozenSet[str]] = frozenset(
    {
        "H",
        "D",
        "T",
    }
)


CARBON_ELEMENTS: Final[FrozenSet[str]] = frozenset(
    {
        "C",
    }
)


NITROGEN_ELEMENTS: Final[FrozenSet[str]] = frozenset(
    {
        "N",
    }
)


OXYGEN_ELEMENTS: Final[FrozenSet[str]] = frozenset(
    {
        "O",
    }
)


SULFUR_ELEMENTS: Final[FrozenSet[str]] = frozenset(
    {
        "S",
    }
)


HALOGEN_ELEMENTS: Final[FrozenSet[str]] = frozenset(
    {
        "F",
        "CL",
        "BR",
        "I",
    }
)


POSITIVE_NITROGEN_TYPE_HINTS: Final[FrozenSet[str]] = frozenset(
    {
        "N+",
        "N.4",
        "N4",
        "N.PL3+",
        "N.AR+",
        "N.GUAN+",
        "N.AMINE+",
    }
)


NEGATIVE_OXYGEN_TYPE_HINTS: Final[FrozenSet[str]] = frozenset(
    {
        "O-",
        "O.CO2",
        "O.2-",
        "O.CARBOXYLATE",
        "O.PHOSPHATE",
        "O.SULFATE",
    }
)


# -----------------------------------------------------------------------------
# 1.21. Aromatic-bond type hints
# -----------------------------------------------------------------------------

AROMATIC_BOND_TYPE_HINTS: Final[FrozenSet[str]] = frozenset(
    {
        "AR",
        "ARO",
        "AROMATIC",
        "1.5",
        ":",
    }
)


AROMATIC_ATOM_TYPE_HINTS: Final[FrozenSet[str]] = frozenset(
    {
        "C.AR",
        "N.AR",
        "N.PL3",
        "N.2",
        "S.AR",
        "CAR",
        "NAR",
    }
)


# -----------------------------------------------------------------------------
# 1.22. Default ring-quality thresholds
# -----------------------------------------------------------------------------

DEFAULT_MAXIMUM_RING_PLANARITY_RMSD: Final[float] = 0.20

DEFAULT_PREFERRED_RING_PLANARITY_RMSD: Final[float] = 0.10

DEFAULT_MAXIMUM_RING_ATOM_DEVIATION: Final[float] = 0.35

DEFAULT_MINIMUM_RING_RADIUS: Final[float] = 0.80

DEFAULT_MAXIMUM_RING_RADIUS: Final[float] = 2.50


# -----------------------------------------------------------------------------
# 1.23. Default pi-pi geometric thresholds
# -----------------------------------------------------------------------------
#
# Distances are expressed in ångströms.
# Angles are expressed in degrees.
#
# The thresholds are intentionally separated into:
#
#     - optimal;
#     - favorable;
#     - maximum accepted.
#
# The final classifier will use these limits together rather than evaluating
# any single parameter in isolation.
# -----------------------------------------------------------------------------

DEFAULT_PI_PI_MINIMUM_CENTROID_DISTANCE: Final[float] = 3.00

DEFAULT_PI_PI_OPTIMAL_CENTROID_DISTANCE: Final[float] = 4.50

DEFAULT_PI_PI_FAVORABLE_CENTROID_DISTANCE: Final[float] = 5.50

DEFAULT_PI_PI_MAXIMUM_CENTROID_DISTANCE: Final[float] = 6.00


DEFAULT_PI_PI_PARALLEL_OPTIMAL_ANGLE: Final[float] = 15.0

DEFAULT_PI_PI_PARALLEL_FAVORABLE_ANGLE: Final[float] = 30.0

DEFAULT_PI_PI_PARALLEL_MAXIMUM_ANGLE: Final[float] = 40.0


DEFAULT_PI_PI_T_SHAPED_OPTIMAL_MINIMUM_ANGLE: Final[float] = 70.0

DEFAULT_PI_PI_T_SHAPED_OPTIMAL_MAXIMUM_ANGLE: Final[float] = 90.0

DEFAULT_PI_PI_T_SHAPED_FAVORABLE_MINIMUM_ANGLE: Final[float] = 55.0

DEFAULT_PI_PI_T_SHAPED_FAVORABLE_MAXIMUM_ANGLE: Final[float] = 90.0


DEFAULT_PI_PI_FACE_TO_FACE_MAXIMUM_OFFSET: Final[float] = 1.50

DEFAULT_PI_PI_OFFSET_STACKING_MAXIMUM_OFFSET: Final[float] = 3.00

DEFAULT_PI_PI_MAXIMUM_LATERAL_OFFSET: Final[float] = 3.50


DEFAULT_PI_PI_MINIMUM_ATOMIC_DISTANCE: Final[float] = 2.50

DEFAULT_PI_PI_MAXIMUM_ATOMIC_DISTANCE: Final[float] = 5.00


# -----------------------------------------------------------------------------
# 1.24. Default cation-pi geometric thresholds
# -----------------------------------------------------------------------------

DEFAULT_CATION_PI_MINIMUM_DISTANCE: Final[float] = 2.50

DEFAULT_CATION_PI_OPTIMAL_DISTANCE: Final[float] = 4.50

DEFAULT_CATION_PI_FAVORABLE_DISTANCE: Final[float] = 5.50

DEFAULT_CATION_PI_MAXIMUM_DISTANCE: Final[float] = 6.00


DEFAULT_CATION_PI_MAXIMUM_RADIAL_OFFSET: Final[float] = 2.50

DEFAULT_CATION_PI_OPTIMAL_RADIAL_OFFSET: Final[float] = 1.50

DEFAULT_CATION_PI_MINIMUM_PLANE_HEIGHT: Final[float] = 1.50

DEFAULT_CATION_PI_MAXIMUM_PLANE_HEIGHT: Final[float] = 5.50


# -----------------------------------------------------------------------------
# 1.25. Default anion-pi geometric thresholds
# -----------------------------------------------------------------------------

DEFAULT_ANION_PI_MINIMUM_DISTANCE: Final[float] = 2.50

DEFAULT_ANION_PI_OPTIMAL_DISTANCE: Final[float] = 4.00

DEFAULT_ANION_PI_FAVORABLE_DISTANCE: Final[float] = 5.00

DEFAULT_ANION_PI_MAXIMUM_DISTANCE: Final[float] = 5.50


DEFAULT_ANION_PI_OPTIMAL_RADIAL_OFFSET: Final[float] = 1.25

DEFAULT_ANION_PI_MAXIMUM_RADIAL_OFFSET: Final[float] = 2.25

DEFAULT_ANION_PI_MINIMUM_PLANE_HEIGHT: Final[float] = 1.50

DEFAULT_ANION_PI_MAXIMUM_PLANE_HEIGHT: Final[float] = 5.00


# -----------------------------------------------------------------------------
# 1.26. Default amide-pi geometric thresholds
# -----------------------------------------------------------------------------

DEFAULT_AMIDE_PI_MINIMUM_DISTANCE: Final[float] = 2.50

DEFAULT_AMIDE_PI_OPTIMAL_DISTANCE: Final[float] = 4.00

DEFAULT_AMIDE_PI_FAVORABLE_DISTANCE: Final[float] = 5.00

DEFAULT_AMIDE_PI_MAXIMUM_DISTANCE: Final[float] = 5.50


DEFAULT_AMIDE_PI_PARALLEL_OPTIMAL_ANGLE: Final[float] = 20.0

DEFAULT_AMIDE_PI_PARALLEL_MAXIMUM_ANGLE: Final[float] = 40.0

DEFAULT_AMIDE_PI_PERPENDICULAR_MINIMUM_ANGLE: Final[float] = 55.0

DEFAULT_AMIDE_PI_PERPENDICULAR_MAXIMUM_ANGLE: Final[float] = 90.0

DEFAULT_AMIDE_PI_MAXIMUM_RADIAL_OFFSET: Final[float] = 2.50


# -----------------------------------------------------------------------------
# 1.27. Default sulfur-pi geometric thresholds
# -----------------------------------------------------------------------------

DEFAULT_SULFUR_PI_MINIMUM_DISTANCE: Final[float] = 2.50

DEFAULT_SULFUR_PI_OPTIMAL_DISTANCE: Final[float] = 4.00

DEFAULT_SULFUR_PI_FAVORABLE_DISTANCE: Final[float] = 5.00

DEFAULT_SULFUR_PI_MAXIMUM_DISTANCE: Final[float] = 5.50

DEFAULT_SULFUR_PI_MAXIMUM_RADIAL_OFFSET: Final[float] = 2.50


# -----------------------------------------------------------------------------
# 1.28. Charge thresholds
# -----------------------------------------------------------------------------

DEFAULT_POSITIVE_PARTIAL_CHARGE_THRESHOLD: Final[float] = 0.25

DEFAULT_NEGATIVE_PARTIAL_CHARGE_THRESHOLD: Final[float] = -0.25

DEFAULT_STRONG_POSITIVE_CHARGE_THRESHOLD: Final[float] = 0.75

DEFAULT_STRONG_NEGATIVE_CHARGE_THRESHOLD: Final[float] = -0.75


# -----------------------------------------------------------------------------
# 1.29. Deduplication thresholds
# -----------------------------------------------------------------------------

DEFAULT_CENTROID_DEDUPLICATION_TOLERANCE: Final[float] = 0.10

DEFAULT_NORMAL_DEDUPLICATION_ANGLE: Final[float] = 5.0

DEFAULT_INTERACTION_DISTANCE_TOLERANCE: Final[float] = 0.15

DEFAULT_INTERACTION_ANGLE_TOLERANCE: Final[float] = 3.0

DEFAULT_GROUP_CENTER_TOLERANCE: Final[float] = 0.15


# -----------------------------------------------------------------------------
# 1.30. Hotspot thresholds
# -----------------------------------------------------------------------------

DEFAULT_HOTSPOT_MINIMUM_INTERACTIONS: Final[int] = 2

DEFAULT_HOTSPOT_MINIMUM_SCORE: Final[float] = 1.00

DEFAULT_MULTIPOSE_HOTSPOT_FREQUENCY: Final[float] = 0.50


# -----------------------------------------------------------------------------
# 1.31. Default base scores
# -----------------------------------------------------------------------------

DEFAULT_INTERACTION_BASE_SCORES: Final[Mapping[str, float]] = {
    PI_PI: 1.00,
    CATION_PI: 1.20,
    ANION_PI: 0.85,
    AMIDE_PI: 0.80,
    SULFUR_PI: 0.70,
}


DEFAULT_GEOMETRY_SCORE_MULTIPLIERS: Final[Mapping[str, float]] = {
    GEOMETRY_OPTIMAL: 1.00,
    GEOMETRY_FAVORABLE: 0.80,
    GEOMETRY_WEAK: 0.55,
    GEOMETRY_BORDERLINE: 0.30,
    GEOMETRY_REJECTED: 0.00,
}


DEFAULT_STRENGTH_SCORE_MULTIPLIERS: Final[Mapping[str, float]] = {
    STRENGTH_STRONG: 1.00,
    STRENGTH_MODERATE: 0.70,
    STRENGTH_WEAK: 0.40,
    STRENGTH_UNCLASSIFIED: 0.00,
}


DEFAULT_PI_PI_GEOMETRY_MULTIPLIERS: Final[Mapping[str, float]] = {
    PI_PI_FACE_TO_FACE: 1.00,
    PI_PI_OFFSET_STACKED: 0.95,
    PI_PI_PARALLEL: 0.90,
    PI_PI_T_SHAPED: 0.90,
    PI_PI_EDGE_TO_FACE: 0.85,
    PI_PI_INTERMEDIATE: 0.60,
    PI_PI_UNCLASSIFIED: 0.40,
}


# -----------------------------------------------------------------------------
# 1.32. Scoring component weights
# -----------------------------------------------------------------------------

DEFAULT_PI_PI_SCORING_WEIGHTS: Final[Mapping[str, float]] = {
    "distance": 0.35,
    "angle": 0.25,
    "offset": 0.20,
    "planarity": 0.10,
    "atomic_contact": 0.10,
}


DEFAULT_CATION_PI_SCORING_WEIGHTS: Final[Mapping[str, float]] = {
    "distance": 0.40,
    "plane_height": 0.20,
    "radial_offset": 0.20,
    "charge": 0.15,
    "planarity": 0.05,
}


DEFAULT_ANION_PI_SCORING_WEIGHTS: Final[Mapping[str, float]] = {
    "distance": 0.35,
    "plane_height": 0.20,
    "radial_offset": 0.20,
    "charge": 0.15,
    "planarity": 0.10,
}


DEFAULT_AMIDE_PI_SCORING_WEIGHTS: Final[Mapping[str, float]] = {
    "distance": 0.35,
    "angle": 0.25,
    "radial_offset": 0.20,
    "planarity": 0.10,
    "group_planarity": 0.10,
}


DEFAULT_SULFUR_PI_SCORING_WEIGHTS: Final[Mapping[str, float]] = {
    "distance": 0.50,
    "plane_height": 0.20,
    "radial_offset": 0.20,
    "planarity": 0.10,
}


# -----------------------------------------------------------------------------
# 1.33. Strength-score boundaries
# -----------------------------------------------------------------------------

DEFAULT_STRONG_SCORE_THRESHOLD: Final[float] = 0.80

DEFAULT_MODERATE_SCORE_THRESHOLD: Final[float] = 0.55

DEFAULT_WEAK_SCORE_THRESHOLD: Final[float] = 0.25


# -----------------------------------------------------------------------------
# 1.34. Multipose-analysis defaults
# -----------------------------------------------------------------------------

DEFAULT_POSE_IDENTIFIER_PREFIX: Final[str] = "pose"

DEFAULT_MINIMUM_POSE_FREQUENCY: Final[float] = 0.10

DEFAULT_CONSENSUS_POSE_FREQUENCY: Final[float] = 0.50

DEFAULT_STRONG_CONSENSUS_POSE_FREQUENCY: Final[float] = 0.75


# -----------------------------------------------------------------------------
# 1.35. Serialization defaults
# -----------------------------------------------------------------------------

DEFAULT_INCLUDE_ATOM_DETAILS: Final[bool] = True

DEFAULT_INCLUDE_COORDINATES: Final[bool] = True

DEFAULT_INCLUDE_RAW_GEOMETRY: Final[bool] = True

DEFAULT_INCLUDE_EMPTY_RESULTS: Final[bool] = True

DEFAULT_SERIALIZATION_INDENT: Final[int] = 2

DEFAULT_JSON_SORT_KEYS: Final[bool] = True


# -----------------------------------------------------------------------------
# 1.36. Result and DockModel attribute names
# -----------------------------------------------------------------------------

DOCK_MODEL_PI_ATTRIBUTE: Final[str] = "pi"

DOCK_MODEL_STATISTICS_ATTRIBUTE: Final[str] = "statistics"

DOCK_MODEL_SCORE_ATTRIBUTE: Final[str] = "score"

DOCK_MODEL_METADATA_ATTRIBUTE: Final[str] = "metadata"


PI_RESULT_INTERACTIONS_KEY: Final[str] = "interactions"

PI_RESULT_RINGS_KEY: Final[str] = "rings"

PI_RESULT_CHARGED_GROUPS_KEY: Final[str] = "charged_groups"

PI_RESULT_AMIDE_GROUPS_KEY: Final[str] = "amide_groups"

PI_RESULT_STATISTICS_KEY: Final[str] = "statistics"

PI_RESULT_HOTSPOTS_KEY: Final[str] = "hotspots"

PI_RESULT_SCORE_KEY: Final[str] = "score"

PI_RESULT_METADATA_KEY: Final[str] = "metadata"


# -----------------------------------------------------------------------------
# 1.37. Common attribute-name fallbacks
# -----------------------------------------------------------------------------
#
# These names will later be used by normalization functions to support native
# ChimeraX atoms, test doubles, and third-party molecular objects.
# -----------------------------------------------------------------------------

ATOM_NAME_ATTRIBUTES: Final[Tuple[str, ...]] = (
    "name",
    "atom_name",
    "idatm_name",
)


ATOM_ELEMENT_ATTRIBUTES: Final[Tuple[str, ...]] = (
    "element",
    "element_name",
    "symbol",
    "atomic_symbol",
)


ATOM_COORDINATE_ATTRIBUTES: Final[Tuple[str, ...]] = (
    "scene_coord",
    "coord",
    "coords",
    "coordinate",
    "coordinates",
    "position",
    "xyz",
)


ATOM_CHARGE_ATTRIBUTES: Final[Tuple[str, ...]] = (
    "formal_charge",
    "charge",
    "partial_charge",
)


ATOM_TYPE_ATTRIBUTES: Final[Tuple[str, ...]] = (
    "idatm_type",
    "atom_type",
    "type",
    "sybyl_type",
)


ATOM_AROMATIC_ATTRIBUTES: Final[Tuple[str, ...]] = (
    "is_aromatic",
    "aromatic",
)


ATOM_BOND_ATTRIBUTES: Final[Tuple[str, ...]] = (
    "bonds",
    "neighbors",
    "bonded_atoms",
)


ATOM_RESIDUE_ATTRIBUTES: Final[Tuple[str, ...]] = (
    "residue",
    "parent_residue",
)


RESIDUE_NAME_ATTRIBUTES: Final[Tuple[str, ...]] = (
    "name",
    "resname",
    "residue_name",
)


RESIDUE_NUMBER_ATTRIBUTES: Final[Tuple[str, ...]] = (
    "number",
    "resnum",
    "residue_number",
    "index",
)


RESIDUE_CHAIN_ATTRIBUTES: Final[Tuple[str, ...]] = (
    "chain_id",
    "chain",
    "chain_name",
)


RESIDUE_ATOM_ATTRIBUTES: Final[Tuple[str, ...]] = (
    "atoms",
    "atom_list",
)


MODEL_ATOM_ATTRIBUTES: Final[Tuple[str, ...]] = (
    "atoms",
    "atom_list",
    "all_atoms",
)


MODEL_RESIDUE_ATTRIBUTES: Final[Tuple[str, ...]] = (
    "residues",
    "residue_list",
    "all_residues",
)


MODEL_IDENTIFIER_ATTRIBUTES: Final[Tuple[str, ...]] = (
    "id_string",
    "name",
    "model_id",
    "id",
)


# -----------------------------------------------------------------------------
# 1.38. Validation limits
# -----------------------------------------------------------------------------

MINIMUM_VALID_DISTANCE: Final[float] = 0.0

MAXIMUM_VALID_MOLECULAR_DISTANCE: Final[float] = 1000.0

MINIMUM_VALID_ANGLE: Final[float] = 0.0

MAXIMUM_VALID_ANGLE: Final[float] = 180.0

MINIMUM_VALID_SCORE: Final[float] = 0.0

MAXIMUM_NORMALIZED_SCORE: Final[float] = 1.0


# -----------------------------------------------------------------------------
# 1.39. Warning messages
# -----------------------------------------------------------------------------

WARNING_NUMPY_UNAVAILABLE: Final[str] = (
    "NumPy is unavailable. Geometry operations requiring NumPy may use "
    "slower pure-Python fallbacks."
)


WARNING_CHIMERAX_UNAVAILABLE: Final[str] = (
    "ChimeraX is unavailable. ChimeraX-specific visualization and atomic "
    "selection functions will be disabled."
)


WARNING_MISSING_COORDINATES: Final[str] = (
    "One or more atoms do not provide valid three-dimensional coordinates."
)


WARNING_INVALID_RING: Final[str] = (
    "The supplied atoms do not define a valid aromatic ring."
)


# -----------------------------------------------------------------------------
# 1.40. Internal configuration checks
# -----------------------------------------------------------------------------

def _validate_weight_mapping(
    weights: Mapping[str, float],
    *,
    name: str,
    tolerance: float = 1.0e-9,
) -> None:
    """
    Validate a scoring-weight mapping declared at module level.

    Parameters
    ----------
    weights
        Mapping containing scoring-component names and their corresponding
        weights.

    name
        Human-readable mapping name used in error messages.

    tolerance
        Maximum accepted deviation from a total weight of 1.0.

    Raises
    ------
    ValueError
        If the mapping is empty, contains invalid values, or does not sum
        approximately to 1.0.
    """

    if not weights:
        raise ValueError(f"{name} cannot be empty.")

    invalid_entries = {
        key: value
        for key, value in weights.items()
        if (
            not isinstance(key, str)
            or not key.strip()
            or not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not isfinite(float(value))
            or float(value) < 0.0
        )
    }

    if invalid_entries:
        raise ValueError(
            f"{name} contains invalid entries: {invalid_entries!r}."
        )

    total = sum(float(value) for value in weights.values())

    if abs(total - 1.0) > tolerance:
        raise ValueError(
            f"{name} must sum to 1.0; observed total: {total:.12f}."
        )


def _validate_module_constants() -> None:
    """
    Validate the internal consistency of module-level constants.

    This function only checks developer-defined constants. User-provided
    configuration values will be validated later by PiAnalysisConfig.
    """

    scoring_mappings = {
        "DEFAULT_PI_PI_SCORING_WEIGHTS": DEFAULT_PI_PI_SCORING_WEIGHTS,
        "DEFAULT_CATION_PI_SCORING_WEIGHTS": (
            DEFAULT_CATION_PI_SCORING_WEIGHTS
        ),
        "DEFAULT_ANION_PI_SCORING_WEIGHTS": DEFAULT_ANION_PI_SCORING_WEIGHTS,
        "DEFAULT_AMIDE_PI_SCORING_WEIGHTS": DEFAULT_AMIDE_PI_SCORING_WEIGHTS,
        "DEFAULT_SULFUR_PI_SCORING_WEIGHTS": (
            DEFAULT_SULFUR_PI_SCORING_WEIGHTS
        ),
    }

    for mapping_name, mapping in scoring_mappings.items():
        _validate_weight_mapping(
            mapping,
            name=mapping_name,
        )

    if not (
        DEFAULT_STRONG_SCORE_THRESHOLD
        > DEFAULT_MODERATE_SCORE_THRESHOLD
        > DEFAULT_WEAK_SCORE_THRESHOLD
        >= 0.0
    ):
        raise ValueError(
            "Score thresholds must satisfy: "
            "strong > moderate > weak >= 0."
        )

    if not (
        DEFAULT_PI_PI_MINIMUM_CENTROID_DISTANCE
        < DEFAULT_PI_PI_OPTIMAL_CENTROID_DISTANCE
        <= DEFAULT_PI_PI_FAVORABLE_CENTROID_DISTANCE
        <= DEFAULT_PI_PI_MAXIMUM_CENTROID_DISTANCE
    ):
        raise ValueError(
            "Invalid pi-pi centroid-distance threshold ordering."
        )

    if not (
        DEFAULT_CATION_PI_MINIMUM_DISTANCE
        < DEFAULT_CATION_PI_OPTIMAL_DISTANCE
        <= DEFAULT_CATION_PI_FAVORABLE_DISTANCE
        <= DEFAULT_CATION_PI_MAXIMUM_DISTANCE
    ):
        raise ValueError(
            "Invalid cation-pi distance threshold ordering."
        )

    if not (
        DEFAULT_ANION_PI_MINIMUM_DISTANCE
        < DEFAULT_ANION_PI_OPTIMAL_DISTANCE
        <= DEFAULT_ANION_PI_FAVORABLE_DISTANCE
        <= DEFAULT_ANION_PI_MAXIMUM_DISTANCE
    ):
        raise ValueError(
            "Invalid anion-pi distance threshold ordering."
        )

    if not (
        DEFAULT_AMIDE_PI_MINIMUM_DISTANCE
        < DEFAULT_AMIDE_PI_OPTIMAL_DISTANCE
        <= DEFAULT_AMIDE_PI_FAVORABLE_DISTANCE
        <= DEFAULT_AMIDE_PI_MAXIMUM_DISTANCE
    ):
        raise ValueError(
            "Invalid amide-pi distance threshold ordering."
        )

    if not (
        DEFAULT_SULFUR_PI_MINIMUM_DISTANCE
        < DEFAULT_SULFUR_PI_OPTIMAL_DISTANCE
        <= DEFAULT_SULFUR_PI_FAVORABLE_DISTANCE
        <= DEFAULT_SULFUR_PI_MAXIMUM_DISTANCE
    ):
        raise ValueError(
            "Invalid sulfur-pi distance threshold ordering."
        )

    if not (
        DEFAULT_MINIMUM_RING_SIZE
        <= DEFAULT_MAXIMUM_RING_SIZE
        <= DEFAULT_MAXIMUM_FUSED_RING_SIZE
    ):
        raise ValueError(
            "Invalid aromatic ring-size limits."
        )

    if not DEFAULT_ENABLED_PI_INTERACTION_TYPES.issubset(
        SUPPORTED_PI_INTERACTION_TYPES
    ):
        raise ValueError(
            "Default enabled interaction types must be supported."
        )


_validate_module_constants()


# =============================================================================
# End of section 1
# =============================================================================


# =============================================================================
# 2. DATACLASSES E MODELOS DE DADOS
# =============================================================================

# -----------------------------------------------------------------------------
# 2.1. Funções auxiliares internas para dataclasses
# -----------------------------------------------------------------------------

def _coerce_coordinate3d(
    value: Optional[Sequence[Number]],
    *,
    field_name: str,
    allow_none: bool = False,
) -> Optional[Coordinate3D]:
    """
    Convert a coordinate-like sequence into a validated Coordinate3D tuple.

    Parameters
    ----------
    value
        Sequence containing exactly three numeric values.

    field_name
        Name used in validation error messages.

    allow_none
        Whether ``None`` is accepted.

    Returns
    -------
    tuple of float or None
        Validated three-dimensional coordinate.

    Raises
    ------
    TypeError
        If the supplied value is not a valid coordinate sequence.

    ValueError
        If the sequence does not contain exactly three finite values.
    """

    if value is None:
        if allow_none:
            return None

        raise ValueError(
            f"{field_name} cannot be None."
        )

    if isinstance(value, (str, bytes)):
        raise TypeError(
            f"{field_name} must be a three-dimensional numeric sequence."
        )

    try:
        values = tuple(float(component) for component in value)

    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{field_name} must contain numeric values."
        ) from exc

    if len(values) != 3:
        raise ValueError(
            f"{field_name} must contain exactly three values; "
            f"received {len(values)}."
        )

    if not all(isfinite(component) for component in values):
        raise ValueError(
            f"{field_name} must contain only finite values."
        )

    return values


def _coerce_optional_float(
    value: Optional[Number],
    *,
    field_name: str,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> Optional[float]:
    """
    Convert an optional numeric value into a validated float.
    """

    if value is None:
        return None

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(
            f"{field_name} must be numeric or None."
        )

    converted = float(value)

    if not isfinite(converted):
        raise ValueError(
            f"{field_name} must be finite."
        )

    if minimum is not None and converted < minimum:
        raise ValueError(
            f"{field_name} must be >= {minimum}; received {converted}."
        )

    if maximum is not None and converted > maximum:
        raise ValueError(
            f"{field_name} must be <= {maximum}; received {converted}."
        )

    return converted


def _coerce_non_negative_float(
    value: Number,
    *,
    field_name: str,
) -> float:
    """
    Convert a numeric value into a finite non-negative float.
    """

    converted = _coerce_optional_float(
        value,
        field_name=field_name,
        minimum=0.0,
    )

    assert converted is not None

    return converted


def _coerce_fraction(
    value: Number,
    *,
    field_name: str,
) -> float:
    """
    Convert a value into a validated fraction in the interval [0, 1].
    """

    converted = _coerce_optional_float(
        value,
        field_name=field_name,
        minimum=0.0,
        maximum=1.0,
    )

    assert converted is not None

    return converted


def _coerce_string_tuple(
    value: Optional[Iterable[Any]],
    *,
    field_name: str,
    allow_empty: bool = True,
) -> Tuple[str, ...]:
    """
    Convert an iterable into a normalized tuple of non-empty strings.
    """

    if value is None:
        if allow_empty:
            return ()

        raise ValueError(
            f"{field_name} cannot be None."
        )

    if isinstance(value, str):
        raw_items = (value,)

    else:
        try:
            raw_items = tuple(value)

        except TypeError as exc:
            raise TypeError(
                f"{field_name} must be iterable."
            ) from exc

    normalized: List[str] = []

    for item in raw_items:
        text = str(item).strip()

        if text:
            normalized.append(text)

    if not normalized and not allow_empty:
        raise ValueError(
            f"{field_name} cannot be empty."
        )

    return tuple(normalized)


def _coerce_atom_tuple(
    value: Optional[Iterable[Any]],
    *,
    field_name: str,
    allow_empty: bool = False,
) -> Tuple[Any, ...]:
    """
    Convert an atom iterable into an immutable tuple.
    """

    if value is None:
        if allow_empty:
            return ()

        raise ValueError(
            f"{field_name} cannot be None."
        )

    if isinstance(value, (str, bytes)):
        raise TypeError(
            f"{field_name} must contain atom-like objects."
        )

    try:
        atoms = tuple(value)

    except TypeError as exc:
        raise TypeError(
            f"{field_name} must be iterable."
        ) from exc

    if not atoms and not allow_empty:
        raise ValueError(
            f"{field_name} cannot be empty."
        )

    return atoms


def _normalize_optional_text(
    value: Optional[Any],
) -> Optional[str]:
    """
    Normalize an optional textual value.
    """

    if value is None:
        return None

    text = str(value).strip()

    return text or None


def _copy_mapping(
    value: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """
    Return a shallow dictionary copy from an optional mapping.
    """

    if value is None:
        return {}

    if not isinstance(value, Mapping):
        raise TypeError(
            "Expected a mapping."
        )

    return dict(value)


def _validate_interaction_type(
    interaction_type: str,
) -> str:
    """
    Validate a supported pi-interaction type.
    """

    normalized = str(interaction_type).strip().lower()

    if normalized not in SUPPORTED_PI_INTERACTION_TYPES:
        raise ValueError(
            f"Unsupported pi interaction type: {interaction_type!r}. "
            f"Supported values: "
            f"{sorted(SUPPORTED_PI_INTERACTION_TYPES)!r}."
        )

    return normalized


def _validate_geometry_class(
    geometry_class: str,
) -> str:
    """
    Validate a geometry-quality class.
    """

    normalized = str(geometry_class).strip().lower()

    if normalized not in GEOMETRY_CLASSES:
        raise ValueError(
            f"Invalid geometry class: {geometry_class!r}."
        )

    return normalized


def _validate_strength_class(
    strength: str,
) -> str:
    """
    Validate an interaction-strength class.
    """

    normalized = str(strength).strip().lower()

    if normalized not in STRENGTH_CLASSES:
        raise ValueError(
            f"Invalid interaction strength: {strength!r}."
        )

    return normalized


def _validate_pi_pi_geometry(
    geometry_type: str,
) -> str:
    """
    Validate a pi-pi geometry subtype.
    """

    normalized = str(geometry_type).strip().lower()

    if normalized not in PI_PI_GEOMETRY_TYPES:
        raise ValueError(
            f"Invalid pi-pi geometry type: {geometry_type!r}."
        )

    return normalized


# -----------------------------------------------------------------------------
# 2.2. Identificação padronizada de átomos
# -----------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class PiAtomReference:
    """
    Serializable reference to an atom involved in a pi-related analysis.

    This object does not replace the native atom. It stores a normalized,
    immutable description that can safely be serialized and compared.
    """

    name: str
    element: str = ""
    atom_index: Optional[int] = None
    serial_number: Optional[int] = None
    residue_name: Optional[str] = None
    residue_number: Optional[Union[int, str]] = None
    chain_id: Optional[str] = None
    model_id: Optional[str] = None
    coordinate: Optional[Coordinate3D] = None
    formal_charge: Optional[float] = None
    partial_charge: Optional[float] = None
    atom_type: Optional[str] = None
    is_aromatic: Optional[bool] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized_name = str(self.name).strip()

        if not normalized_name:
            raise ValueError(
                "PiAtomReference.name cannot be empty."
            )

        object.__setattr__(
            self,
            "name",
            normalized_name,
        )

        object.__setattr__(
            self,
            "element",
            str(self.element or "").strip().upper(),
        )

        if self.atom_index is not None:
            if isinstance(self.atom_index, bool):
                raise TypeError(
                    "atom_index must be an integer or None."
                )

            object.__setattr__(
                self,
                "atom_index",
                int(self.atom_index),
            )

        if self.serial_number is not None:
            if isinstance(self.serial_number, bool):
                raise TypeError(
                    "serial_number must be an integer or None."
                )

            object.__setattr__(
                self,
                "serial_number",
                int(self.serial_number),
            )

        object.__setattr__(
            self,
            "residue_name",
            _normalize_optional_text(self.residue_name),
        )

        object.__setattr__(
            self,
            "chain_id",
            _normalize_optional_text(self.chain_id),
        )

        object.__setattr__(
            self,
            "model_id",
            _normalize_optional_text(self.model_id),
        )

        object.__setattr__(
            self,
            "atom_type",
            _normalize_optional_text(self.atom_type),
        )

        object.__setattr__(
            self,
            "coordinate",
            _coerce_coordinate3d(
                self.coordinate,
                field_name="coordinate",
                allow_none=True,
            ),
        )

        object.__setattr__(
            self,
            "formal_charge",
            _coerce_optional_float(
                self.formal_charge,
                field_name="formal_charge",
            ),
        )

        object.__setattr__(
            self,
            "partial_charge",
            _coerce_optional_float(
                self.partial_charge,
                field_name="partial_charge",
            ),
        )

        if self.is_aromatic is not None:
            object.__setattr__(
                self,
                "is_aromatic",
                bool(self.is_aromatic),
            )

        object.__setattr__(
            self,
            "metadata",
            _copy_mapping(self.metadata),
        )

    @property
    def residue_identifier(self) -> str:
        """
        Return a stable residue identifier.
        """

        parts: List[str] = []

        if self.chain_id:
            parts.append(self.chain_id)

        if self.residue_name:
            parts.append(self.residue_name)

        if self.residue_number is not None:
            parts.append(str(self.residue_number))

        return ":".join(parts) or "unknown"

    @property
    def atom_identifier(self) -> str:
        """
        Return a stable atom identifier.
        """

        residue_identifier = self.residue_identifier

        if self.serial_number is not None:
            suffix = f"{self.name}#{self.serial_number}"

        elif self.atom_index is not None:
            suffix = f"{self.name}@{self.atom_index}"

        else:
            suffix = self.name

        return f"{residue_identifier}:{suffix}"

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the atom reference into a serializable dictionary.
        """

        return {
            "name": self.name,
            "element": self.element,
            "atom_index": self.atom_index,
            "serial_number": self.serial_number,
            "residue_name": self.residue_name,
            "residue_number": self.residue_number,
            "chain_id": self.chain_id,
            "model_id": self.model_id,
            "coordinate": (
                list(self.coordinate)
                if self.coordinate is not None
                else None
            ),
            "formal_charge": self.formal_charge,
            "partial_charge": self.partial_charge,
            "atom_type": self.atom_type,
            "is_aromatic": self.is_aromatic,
            "metadata": dict(self.metadata),
        }


# -----------------------------------------------------------------------------
# 2.3. Representação de anel aromático
# -----------------------------------------------------------------------------

@dataclass(slots=True)
class PiRing:
    """
    Aromatic ring or fused aromatic system.

    Parameters
    ----------
    atoms
        Native atom-like objects composing the ring.

    atom_references
        Serializable normalized references corresponding to ``atoms``.

    centroid
        Geometric center of the ring.

    normal
        Unit vector normal to the fitted ring plane.

    planarity_rmsd
        Root-mean-square deviation of ring atoms from the fitted plane.

    maximum_plane_deviation
        Largest absolute atomic deviation from the fitted plane.
    """

    atoms: Tuple[Any, ...]
    atom_references: Tuple[PiAtomReference, ...] = ()
    centroid: Optional[Coordinate3D] = None
    normal: Optional[Vector3D] = None
    planarity_rmsd: Optional[float] = None
    maximum_plane_deviation: Optional[float] = None
    radius: Optional[float] = None

    ring_id: Optional[str] = None
    ring_index: Optional[int] = None
    ring_size: Optional[int] = None

    residue_name: Optional[str] = None
    residue_number: Optional[Union[int, str]] = None
    chain_id: Optional[str] = None
    model_id: Optional[str] = None
    participant_type: str = PARTICIPANT_UNKNOWN

    is_fused: bool = False
    is_heteroaromatic: bool = False
    is_protein_ring: bool = False
    is_ligand_ring: bool = False
    aromaticity_source: Optional[str] = None

    atom_names: Tuple[str, ...] = ()
    element_symbols: Tuple[str, ...] = ()

    valid: bool = True
    validation_messages: Tuple[str, ...] = ()

    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.atoms = _coerce_atom_tuple(
            self.atoms,
            field_name="PiRing.atoms",
            allow_empty=False,
        )

        if self.atom_references:
            self.atom_references = tuple(self.atom_references)

            if len(self.atom_references) != len(self.atoms):
                raise ValueError(
                    "PiRing.atom_references must have the same length "
                    "as PiRing.atoms."
                )

        self.centroid = _coerce_coordinate3d(
            self.centroid,
            field_name="PiRing.centroid",
            allow_none=True,
        )

        self.normal = _coerce_coordinate3d(
            self.normal,
            field_name="PiRing.normal",
            allow_none=True,
        )

        self.planarity_rmsd = _coerce_optional_float(
            self.planarity_rmsd,
            field_name="PiRing.planarity_rmsd",
            minimum=0.0,
        )

        self.maximum_plane_deviation = _coerce_optional_float(
            self.maximum_plane_deviation,
            field_name="PiRing.maximum_plane_deviation",
            minimum=0.0,
        )

        self.radius = _coerce_optional_float(
            self.radius,
            field_name="PiRing.radius",
            minimum=0.0,
        )

        if self.ring_index is not None:
            self.ring_index = int(self.ring_index)

        if self.ring_size is None:
            self.ring_size = len(self.atoms)

        else:
            self.ring_size = int(self.ring_size)

        if self.ring_size != len(self.atoms):
            raise ValueError(
                "PiRing.ring_size must match the number of atoms."
            )

        if self.ring_size < 3:
            raise ValueError(
                "A ring must contain at least three atoms."
            )

        self.ring_id = _normalize_optional_text(self.ring_id)
        self.residue_name = _normalize_optional_text(self.residue_name)
        self.chain_id = _normalize_optional_text(self.chain_id)
        self.model_id = _normalize_optional_text(self.model_id)

        self.participant_type = str(
            self.participant_type or PARTICIPANT_UNKNOWN
        ).strip().lower()

        self.aromaticity_source = _normalize_optional_text(
            self.aromaticity_source
        )

        self.atom_names = _coerce_string_tuple(
            self.atom_names,
            field_name="PiRing.atom_names",
        )

        self.element_symbols = tuple(
            symbol.upper()
            for symbol in _coerce_string_tuple(
                self.element_symbols,
                field_name="PiRing.element_symbols",
            )
        )

        self.validation_messages = _coerce_string_tuple(
            self.validation_messages,
            field_name="PiRing.validation_messages",
        )

        self.is_fused = bool(self.is_fused)
        self.is_heteroaromatic = bool(self.is_heteroaromatic)
        self.is_protein_ring = bool(self.is_protein_ring)
        self.is_ligand_ring = bool(self.is_ligand_ring)
        self.valid = bool(self.valid)

        self.metadata = _copy_mapping(self.metadata)

        if self.ring_id is None:
            self.ring_id = self.build_ring_id()

    def build_ring_id(self) -> str:
        """
        Build a deterministic identifier for the ring.
        """

        participant = self.participant_type or PARTICIPANT_UNKNOWN
        chain = self.chain_id or "?"
        residue = self.residue_name or "UNK"
        number = (
            str(self.residue_number)
            if self.residue_number is not None
            else "?"
        )

        index = (
            str(self.ring_index)
            if self.ring_index is not None
            else "0"
        )

        return (
            f"{participant}:{self.model_id or '?'}:"
            f"{chain}:{residue}:{number}:ring{index}"
        )

    @property
    def atom_count(self) -> int:
        """
        Return the number of atoms composing the ring.
        """

        return len(self.atoms)

    @property
    def has_complete_geometry(self) -> bool:
        """
        Indicate whether centroid and normal vector are available.
        """

        return (
            self.centroid is not None
            and self.normal is not None
        )

    @property
    def residue_identifier(self) -> str:
        """
        Return a stable residue identifier.
        """

        parts = [
            self.chain_id or "?",
            self.residue_name or "UNK",
            (
                str(self.residue_number)
                if self.residue_number is not None
                else "?"
            ),
        ]

        return ":".join(parts)

    def to_dict(
        self,
        *,
        include_atoms: bool = True,
        include_coordinates: bool = True,
    ) -> Dict[str, Any]:
        """
        Convert the ring into a serializable dictionary.
        """

        result: Dict[str, Any] = {
            "ring_id": self.ring_id,
            "ring_index": self.ring_index,
            "ring_size": self.ring_size,
            "residue_name": self.residue_name,
            "residue_number": self.residue_number,
            "chain_id": self.chain_id,
            "model_id": self.model_id,
            "participant_type": self.participant_type,
            "is_fused": self.is_fused,
            "is_heteroaromatic": self.is_heteroaromatic,
            "is_protein_ring": self.is_protein_ring,
            "is_ligand_ring": self.is_ligand_ring,
            "aromaticity_source": self.aromaticity_source,
            "atom_names": list(self.atom_names),
            "element_symbols": list(self.element_symbols),
            "planarity_rmsd": self.planarity_rmsd,
            "maximum_plane_deviation": self.maximum_plane_deviation,
            "radius": self.radius,
            "valid": self.valid,
            "validation_messages": list(self.validation_messages),
            "metadata": dict(self.metadata),
        }

        if include_coordinates:
            result["centroid"] = (
                list(self.centroid)
                if self.centroid is not None
                else None
            )

            result["normal"] = (
                list(self.normal)
                if self.normal is not None
                else None
            )

        if include_atoms:
            result["atoms"] = [
                atom_reference.to_dict()
                for atom_reference in self.atom_references
            ]

        return result


# -----------------------------------------------------------------------------
# 2.4. Grupo carregado
# -----------------------------------------------------------------------------

@dataclass(slots=True)
class PiChargedGroup:
    """
    Positively or negatively charged group involved in a pi interaction.
    """

    atoms: Tuple[Any, ...]
    charge_sign: int

    atom_references: Tuple[PiAtomReference, ...] = ()
    center: Optional[Coordinate3D] = None

    group_id: Optional[str] = None
    group_type: str = "unknown"
    formal_charge: Optional[float] = None
    partial_charge: Optional[float] = None

    residue_name: Optional[str] = None
    residue_number: Optional[Union[int, str]] = None
    chain_id: Optional[str] = None
    model_id: Optional[str] = None
    participant_type: str = PARTICIPANT_UNKNOWN

    charge_is_formal: bool = False
    charge_is_inferred: bool = False
    charge_is_delocalized: bool = False

    atom_names: Tuple[str, ...] = ()
    element_symbols: Tuple[str, ...] = ()

    valid: bool = True
    validation_messages: Tuple[str, ...] = ()

    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.atoms = _coerce_atom_tuple(
            self.atoms,
            field_name="PiChargedGroup.atoms",
            allow_empty=False,
        )

        if self.charge_sign not in (-1, 1):
            raise ValueError(
                "PiChargedGroup.charge_sign must be -1 or 1."
            )

        if self.atom_references:
            self.atom_references = tuple(self.atom_references)

            if len(self.atom_references) != len(self.atoms):
                raise ValueError(
                    "PiChargedGroup.atom_references must have the same "
                    "length as PiChargedGroup.atoms."
                )

        self.center = _coerce_coordinate3d(
            self.center,
            field_name="PiChargedGroup.center",
            allow_none=True,
        )

        self.group_id = _normalize_optional_text(self.group_id)
        self.group_type = str(self.group_type or "unknown").strip().lower()

        self.formal_charge = _coerce_optional_float(
            self.formal_charge,
            field_name="PiChargedGroup.formal_charge",
        )

        self.partial_charge = _coerce_optional_float(
            self.partial_charge,
            field_name="PiChargedGroup.partial_charge",
        )

        self.residue_name = _normalize_optional_text(self.residue_name)
        self.chain_id = _normalize_optional_text(self.chain_id)
        self.model_id = _normalize_optional_text(self.model_id)

        self.participant_type = str(
            self.participant_type or PARTICIPANT_UNKNOWN
        ).strip().lower()

        self.atom_names = _coerce_string_tuple(
            self.atom_names,
            field_name="PiChargedGroup.atom_names",
        )

        self.element_symbols = tuple(
            symbol.upper()
            for symbol in _coerce_string_tuple(
                self.element_symbols,
                field_name="PiChargedGroup.element_symbols",
            )
        )

        self.validation_messages = _coerce_string_tuple(
            self.validation_messages,
            field_name="PiChargedGroup.validation_messages",
        )

        self.charge_is_formal = bool(self.charge_is_formal)
        self.charge_is_inferred = bool(self.charge_is_inferred)
        self.charge_is_delocalized = bool(self.charge_is_delocalized)
        self.valid = bool(self.valid)

        self.metadata = _copy_mapping(self.metadata)

        if self.group_id is None:
            self.group_id = self.build_group_id()

    def build_group_id(self) -> str:
        """
        Build a deterministic charged-group identifier.
        """

        charge_label = "cation" if self.charge_sign > 0 else "anion"

        return (
            f"{charge_label}:{self.model_id or '?'}:"
            f"{self.chain_id or '?'}:"
            f"{self.residue_name or 'UNK'}:"
            f"{self.residue_number if self.residue_number is not None else '?'}:"
            f"{self.group_type}"
        )

    @property
    def is_cation(self) -> bool:
        """
        Return True when the group is positively charged.
        """

        return self.charge_sign > 0

    @property
    def is_anion(self) -> bool:
        """
        Return True when the group is negatively charged.
        """

        return self.charge_sign < 0

    @property
    def effective_charge(self) -> Optional[float]:
        """
        Return the best available charge estimate.
        """

        if self.formal_charge is not None:
            return self.formal_charge

        return self.partial_charge

    @property
    def residue_identifier(self) -> str:
        """
        Return a stable residue identifier.
        """

        return (
            f"{self.chain_id or '?'}:"
            f"{self.residue_name or 'UNK'}:"
            f"{self.residue_number if self.residue_number is not None else '?'}"
        )

    def to_dict(
        self,
        *,
        include_atoms: bool = True,
        include_coordinates: bool = True,
    ) -> Dict[str, Any]:
        """
        Convert the charged group into a serializable dictionary.
        """

        result: Dict[str, Any] = {
            "group_id": self.group_id,
            "group_type": self.group_type,
            "charge_sign": self.charge_sign,
            "formal_charge": self.formal_charge,
            "partial_charge": self.partial_charge,
            "effective_charge": self.effective_charge,
            "residue_name": self.residue_name,
            "residue_number": self.residue_number,
            "chain_id": self.chain_id,
            "model_id": self.model_id,
            "participant_type": self.participant_type,
            "charge_is_formal": self.charge_is_formal,
            "charge_is_inferred": self.charge_is_inferred,
            "charge_is_delocalized": self.charge_is_delocalized,
            "atom_names": list(self.atom_names),
            "element_symbols": list(self.element_symbols),
            "valid": self.valid,
            "validation_messages": list(self.validation_messages),
            "metadata": dict(self.metadata),
        }

        if include_coordinates:
            result["center"] = (
                list(self.center)
                if self.center is not None
                else None
            )

        if include_atoms:
            result["atoms"] = [
                atom_reference.to_dict()
                for atom_reference in self.atom_references
            ]

        return result


# -----------------------------------------------------------------------------
# 2.5. Grupo amida
# -----------------------------------------------------------------------------

@dataclass(slots=True)
class PiAmideGroup:
    """
    Amide group potentially involved in an amide-pi interaction.
    """

    atoms: Tuple[Any, ...]

    atom_references: Tuple[PiAtomReference, ...] = ()
    center: Optional[Coordinate3D] = None
    normal: Optional[Vector3D] = None

    carbonyl_carbon: Optional[Any] = None
    carbonyl_oxygen: Optional[Any] = None
    amide_nitrogen: Optional[Any] = None

    group_id: Optional[str] = None
    group_type: str = "amide"

    residue_name: Optional[str] = None
    residue_number: Optional[Union[int, str]] = None
    chain_id: Optional[str] = None
    model_id: Optional[str] = None
    participant_type: str = PARTICIPANT_UNKNOWN

    is_side_chain: bool = False
    is_backbone: bool = False
    is_ligand_group: bool = False

    planarity_rmsd: Optional[float] = None
    maximum_plane_deviation: Optional[float] = None

    atom_names: Tuple[str, ...] = ()

    valid: bool = True
    validation_messages: Tuple[str, ...] = ()

    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.atoms = _coerce_atom_tuple(
            self.atoms,
            field_name="PiAmideGroup.atoms",
            allow_empty=False,
        )

        if self.atom_references:
            self.atom_references = tuple(self.atom_references)

            if len(self.atom_references) != len(self.atoms):
                raise ValueError(
                    "PiAmideGroup.atom_references must have the same "
                    "length as PiAmideGroup.atoms."
                )

        self.center = _coerce_coordinate3d(
            self.center,
            field_name="PiAmideGroup.center",
            allow_none=True,
        )

        self.normal = _coerce_coordinate3d(
            self.normal,
            field_name="PiAmideGroup.normal",
            allow_none=True,
        )

        self.planarity_rmsd = _coerce_optional_float(
            self.planarity_rmsd,
            field_name="PiAmideGroup.planarity_rmsd",
            minimum=0.0,
        )

        self.maximum_plane_deviation = _coerce_optional_float(
            self.maximum_plane_deviation,
            field_name="PiAmideGroup.maximum_plane_deviation",
            minimum=0.0,
        )

        self.group_id = _normalize_optional_text(self.group_id)
        self.group_type = str(self.group_type or "amide").strip().lower()

        self.residue_name = _normalize_optional_text(self.residue_name)
        self.chain_id = _normalize_optional_text(self.chain_id)
        self.model_id = _normalize_optional_text(self.model_id)

        self.participant_type = str(
            self.participant_type or PARTICIPANT_UNKNOWN
        ).strip().lower()

        self.is_side_chain = bool(self.is_side_chain)
        self.is_backbone = bool(self.is_backbone)
        self.is_ligand_group = bool(self.is_ligand_group)
        self.valid = bool(self.valid)

        self.atom_names = _coerce_string_tuple(
            self.atom_names,
            field_name="PiAmideGroup.atom_names",
        )

        self.validation_messages = _coerce_string_tuple(
            self.validation_messages,
            field_name="PiAmideGroup.validation_messages",
        )

        self.metadata = _copy_mapping(self.metadata)

        if self.group_id is None:
            self.group_id = self.build_group_id()

    def build_group_id(self) -> str:
        """
        Build a deterministic amide-group identifier.
        """

        if self.is_backbone:
            location = "backbone"

        elif self.is_side_chain:
            location = "sidechain"

        elif self.is_ligand_group:
            location = "ligand"

        else:
            location = "unknown"

        return (
            f"amide:{self.model_id or '?'}:"
            f"{self.chain_id or '?'}:"
            f"{self.residue_name or 'UNK'}:"
            f"{self.residue_number if self.residue_number is not None else '?'}:"
            f"{location}"
        )

    @property
    def has_complete_geometry(self) -> bool:
        """
        Indicate whether center and normal vector are available.
        """

        return (
            self.center is not None
            and self.normal is not None
        )

    @property
    def residue_identifier(self) -> str:
        """
        Return a stable residue identifier.
        """

        return (
            f"{self.chain_id or '?'}:"
            f"{self.residue_name or 'UNK'}:"
            f"{self.residue_number if self.residue_number is not None else '?'}"
        )

    def to_dict(
        self,
        *,
        include_atoms: bool = True,
        include_coordinates: bool = True,
    ) -> Dict[str, Any]:
        """
        Convert the amide group into a serializable dictionary.
        """

        result: Dict[str, Any] = {
            "group_id": self.group_id,
            "group_type": self.group_type,
            "residue_name": self.residue_name,
            "residue_number": self.residue_number,
            "chain_id": self.chain_id,
            "model_id": self.model_id,
            "participant_type": self.participant_type,
            "is_side_chain": self.is_side_chain,
            "is_backbone": self.is_backbone,
            "is_ligand_group": self.is_ligand_group,
            "planarity_rmsd": self.planarity_rmsd,
            "maximum_plane_deviation": self.maximum_plane_deviation,
            "atom_names": list(self.atom_names),
            "valid": self.valid,
            "validation_messages": list(self.validation_messages),
            "metadata": dict(self.metadata),
        }

        if include_coordinates:
            result["center"] = (
                list(self.center)
                if self.center is not None
                else None
            )

            result["normal"] = (
                list(self.normal)
                if self.normal is not None
                else None
            )

        if include_atoms:
            result["atoms"] = [
                atom_reference.to_dict()
                for atom_reference in self.atom_references
            ]

        return result


# -----------------------------------------------------------------------------
# 2.6. Contato atômico associado a uma interação pi
# -----------------------------------------------------------------------------

@dataclass(slots=True)
class PiAtomicContact:
    """
    Pairwise atomic contact associated with a pi-related interaction.
    """

    atom_1: Any
    atom_2: Any
    distance: float

    atom_1_reference: Optional[PiAtomReference] = None
    atom_2_reference: Optional[PiAtomReference] = None

    interaction_type: str = PI_PI
    contact_id: Optional[str] = None

    atom_1_role: Optional[str] = None
    atom_2_role: Optional[str] = None

    within_cutoff: bool = True
    is_closest_contact: bool = False

    score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.distance = _coerce_non_negative_float(
            self.distance,
            field_name="PiAtomicContact.distance",
        )

        self.interaction_type = _validate_interaction_type(
            self.interaction_type
        )

        self.contact_id = _normalize_optional_text(self.contact_id)
        self.atom_1_role = _normalize_optional_text(self.atom_1_role)
        self.atom_2_role = _normalize_optional_text(self.atom_2_role)

        self.within_cutoff = bool(self.within_cutoff)
        self.is_closest_contact = bool(self.is_closest_contact)

        self.score = _coerce_optional_float(
            self.score,
            field_name="PiAtomicContact.score",
            minimum=0.0,
        )

        self.metadata = _copy_mapping(self.metadata)

        if self.contact_id is None:
            self.contact_id = self.build_contact_id()

    def build_contact_id(self) -> str:
        """
        Build a deterministic atomic-contact identifier.
        """

        atom_1_id = (
            self.atom_1_reference.atom_identifier
            if self.atom_1_reference is not None
            else str(id(self.atom_1))
        )

        atom_2_id = (
            self.atom_2_reference.atom_identifier
            if self.atom_2_reference is not None
            else str(id(self.atom_2))
        )

        ordered = sorted((atom_1_id, atom_2_id))

        return (
            f"{self.interaction_type}:"
            f"{ordered[0]}--{ordered[1]}"
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the contact into a serializable dictionary.
        """

        return {
            "contact_id": self.contact_id,
            "interaction_type": self.interaction_type,
            "distance": self.distance,
            "atom_1_role": self.atom_1_role,
            "atom_2_role": self.atom_2_role,
            "within_cutoff": self.within_cutoff,
            "is_closest_contact": self.is_closest_contact,
            "score": self.score,
            "atom_1": (
                self.atom_1_reference.to_dict()
                if self.atom_1_reference is not None
                else None
            ),
            "atom_2": (
                self.atom_2_reference.to_dict()
                if self.atom_2_reference is not None
                else None
            ),
            "metadata": dict(self.metadata),
        }


# -----------------------------------------------------------------------------
# 2.7. Interação pi consolidada
# -----------------------------------------------------------------------------

@dataclass(slots=True)
class PiInteraction:
    """
    Consolidated pi-related molecular interaction.

    The class supports pi-pi, cation-pi, anion-pi, amide-pi, and sulfur-pi
    interactions using a common representation.
    """

    interaction_type: str

    ring_1: PiRing
    ring_2: Optional[PiRing] = None
    charged_group: Optional[PiChargedGroup] = None
    amide_group: Optional[PiAmideGroup] = None
    sulfur_group: Optional[PiChargedGroup] = None

    interaction_id: Optional[str] = None
    geometry_type: str = PI_PI_UNCLASSIFIED
    geometry_class: str = GEOMETRY_REJECTED
    strength: str = STRENGTH_UNCLASSIFIED

    centroid_distance: Optional[float] = None
    minimum_atomic_distance: Optional[float] = None
    maximum_atomic_distance: Optional[float] = None

    normal_angle: Optional[float] = None
    plane_angle: Optional[float] = None
    lateral_offset: Optional[float] = None
    radial_offset: Optional[float] = None
    plane_height: Optional[float] = None

    ring_1_planarity: Optional[float] = None
    ring_2_planarity: Optional[float] = None
    group_planarity: Optional[float] = None

    distance_score: Optional[float] = None
    angle_score: Optional[float] = None
    offset_score: Optional[float] = None
    planarity_score: Optional[float] = None
    charge_score: Optional[float] = None
    geometry_score: Optional[float] = None

    base_score: Optional[float] = None
    raw_score: Optional[float] = None
    score: float = 0.0
    normalized_score: float = 0.0

    atomic_contacts: List[PiAtomicContact] = field(default_factory=list)

    receptor_residue: Optional[str] = None
    ligand_residue: Optional[str] = None

    pose_id: Optional[str] = None
    model_id: Optional[str] = None

    accepted: bool = False
    is_duplicate: bool = False
    duplicate_of: Optional[str] = None

    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.interaction_type = _validate_interaction_type(
            self.interaction_type
        )

        if not isinstance(self.ring_1, PiRing):
            raise TypeError(
                "PiInteraction.ring_1 must be a PiRing."
            )

        if self.ring_2 is not None and not isinstance(
            self.ring_2,
            PiRing,
        ):
            raise TypeError(
                "PiInteraction.ring_2 must be a PiRing or None."
            )

        if self.charged_group is not None and not isinstance(
            self.charged_group,
            PiChargedGroup,
        ):
            raise TypeError(
                "PiInteraction.charged_group must be "
                "a PiChargedGroup or None."
            )

        if self.amide_group is not None and not isinstance(
            self.amide_group,
            PiAmideGroup,
        ):
            raise TypeError(
                "PiInteraction.amide_group must be a PiAmideGroup or None."
            )

        self._validate_participant_combination()

        self.interaction_id = _normalize_optional_text(
            self.interaction_id
        )

        if self.interaction_type == PI_PI:
            self.geometry_type = _validate_pi_pi_geometry(
                self.geometry_type
            )

        else:
            self.geometry_type = str(
                self.geometry_type or "unclassified"
            ).strip().lower()

        self.geometry_class = _validate_geometry_class(
            self.geometry_class
        )

        self.strength = _validate_strength_class(
            self.strength
        )

        distance_fields = (
            "centroid_distance",
            "minimum_atomic_distance",
            "maximum_atomic_distance",
            "lateral_offset",
            "radial_offset",
            "plane_height",
            "ring_1_planarity",
            "ring_2_planarity",
            "group_planarity",
        )

        for field_name in distance_fields:
            setattr(
                self,
                field_name,
                _coerce_optional_float(
                    getattr(self, field_name),
                    field_name=f"PiInteraction.{field_name}",
                    minimum=0.0,
                ),
            )

        angle_fields = (
            "normal_angle",
            "plane_angle",
        )

        for field_name in angle_fields:
            setattr(
                self,
                field_name,
                _coerce_optional_float(
                    getattr(self, field_name),
                    field_name=f"PiInteraction.{field_name}",
                    minimum=0.0,
                    maximum=180.0,
                ),
            )

        component_score_fields = (
            "distance_score",
            "angle_score",
            "offset_score",
            "planarity_score",
            "charge_score",
            "geometry_score",
        )

        for field_name in component_score_fields:
            setattr(
                self,
                field_name,
                _coerce_optional_float(
                    getattr(self, field_name),
                    field_name=f"PiInteraction.{field_name}",
                    minimum=0.0,
                    maximum=1.0,
                ),
            )

        self.base_score = _coerce_optional_float(
            self.base_score,
            field_name="PiInteraction.base_score",
            minimum=0.0,
        )

        self.raw_score = _coerce_optional_float(
            self.raw_score,
            field_name="PiInteraction.raw_score",
            minimum=0.0,
        )

        self.score = _coerce_non_negative_float(
            self.score,
            field_name="PiInteraction.score",
        )

        self.normalized_score = _coerce_fraction(
            self.normalized_score,
            field_name="PiInteraction.normalized_score",
        )

        self.atomic_contacts = list(self.atomic_contacts)

        for contact in self.atomic_contacts:
            if not isinstance(contact, PiAtomicContact):
                raise TypeError(
                    "PiInteraction.atomic_contacts must contain only "
                    "PiAtomicContact objects."
                )

        self.receptor_residue = _normalize_optional_text(
            self.receptor_residue
        )

        self.ligand_residue = _normalize_optional_text(
            self.ligand_residue
        )

        self.pose_id = _normalize_optional_text(self.pose_id)
        self.model_id = _normalize_optional_text(self.model_id)

        self.accepted = bool(self.accepted)
        self.is_duplicate = bool(self.is_duplicate)

        self.duplicate_of = _normalize_optional_text(
            self.duplicate_of
        )

        self.warnings = list(
            _coerce_string_tuple(
                self.warnings,
                field_name="PiInteraction.warnings",
            )
        )

        self.metadata = _copy_mapping(self.metadata)

        if self.interaction_id is None:
            self.interaction_id = self.build_interaction_id()

    def _validate_participant_combination(self) -> None:
        """
        Validate participant fields according to the interaction type.
        """

        if self.interaction_type == PI_PI:
            if self.ring_2 is None:
                raise ValueError(
                    "A pi-pi interaction requires ring_2."
                )

        elif self.interaction_type in {CATION_PI, ANION_PI}:
            if self.charged_group is None:
                raise ValueError(
                    f"A {self.interaction_type} interaction requires "
                    "charged_group."
                )

            expected_sign = 1 if self.interaction_type == CATION_PI else -1

            if self.charged_group.charge_sign != expected_sign:
                raise ValueError(
                    f"{self.interaction_type} requires a charged group "
                    f"with sign {expected_sign}."
                )

        elif self.interaction_type == AMIDE_PI:
            if self.amide_group is None:
                raise ValueError(
                    "An amide-pi interaction requires amide_group."
                )

        elif self.interaction_type == SULFUR_PI:
            if (
                self.sulfur_group is None
                and self.charged_group is None
            ):
                raise ValueError(
                    "A sulfur-pi interaction requires sulfur_group "
                    "or charged_group."
                )

    def build_interaction_id(self) -> str:
        """
        Build a deterministic interaction identifier.
        """

        participant_ids = [self.ring_1.ring_id or "ring1"]

        if self.ring_2 is not None:
            participant_ids.append(
                self.ring_2.ring_id or "ring2"
            )

        elif self.charged_group is not None:
            participant_ids.append(
                self.charged_group.group_id or "charged_group"
            )

        elif self.amide_group is not None:
            participant_ids.append(
                self.amide_group.group_id or "amide_group"
            )

        elif self.sulfur_group is not None:
            participant_ids.append(
                self.sulfur_group.group_id or "sulfur_group"
            )

        ordered_ids = sorted(participant_ids)

        pose_suffix = (
            f":{self.pose_id}"
            if self.pose_id
            else ""
        )

        return (
            f"{self.interaction_type}:"
            f"{'--'.join(ordered_ids)}"
            f"{pose_suffix}"
        )

    @property
    def residue_identifiers(self) -> Tuple[str, ...]:
        """
        Return unique residue identifiers involved in the interaction.
        """

        identifiers: List[str] = [
            self.ring_1.residue_identifier,
        ]

        if self.ring_2 is not None:
            identifiers.append(
                self.ring_2.residue_identifier
            )

        if self.charged_group is not None:
            identifiers.append(
                self.charged_group.residue_identifier
            )

        if self.amide_group is not None:
            identifiers.append(
                self.amide_group.residue_identifier
            )

        if self.sulfur_group is not None:
            identifiers.append(
                self.sulfur_group.residue_identifier
            )

        return tuple(dict.fromkeys(identifiers))

    @property
    def atom_contact_count(self) -> int:
        """
        Return the number of associated atomic contacts.
        """

        return len(self.atomic_contacts)

    @property
    def is_valid_interaction(self) -> bool:
        """
        Return whether the interaction is accepted and not duplicated.
        """

        return (
            self.accepted
            and not self.is_duplicate
            and self.geometry_class != GEOMETRY_REJECTED
        )

    def to_dict(
        self,
        *,
        include_atoms: bool = True,
        include_coordinates: bool = True,
        include_raw_geometry: bool = True,
    ) -> Dict[str, Any]:
        """
        Convert the interaction into a serializable dictionary.
        """

        result: Dict[str, Any] = {
            "interaction_id": self.interaction_id,
            "interaction_type": self.interaction_type,
            "geometry_type": self.geometry_type,
            "geometry_class": self.geometry_class,
            "strength": self.strength,
            "score": self.score,
            "normalized_score": self.normalized_score,
            "base_score": self.base_score,
            "raw_score": self.raw_score,
            "receptor_residue": self.receptor_residue,
            "ligand_residue": self.ligand_residue,
            "pose_id": self.pose_id,
            "model_id": self.model_id,
            "accepted": self.accepted,
            "is_duplicate": self.is_duplicate,
            "duplicate_of": self.duplicate_of,
            "residue_identifiers": list(self.residue_identifiers),
            "atom_contact_count": self.atom_contact_count,
            "ring_1": self.ring_1.to_dict(
                include_atoms=include_atoms,
                include_coordinates=include_coordinates,
            ),
            "ring_2": (
                self.ring_2.to_dict(
                    include_atoms=include_atoms,
                    include_coordinates=include_coordinates,
                )
                if self.ring_2 is not None
                else None
            ),
            "charged_group": (
                self.charged_group.to_dict(
                    include_atoms=include_atoms,
                    include_coordinates=include_coordinates,
                )
                if self.charged_group is not None
                else None
            ),
            "amide_group": (
                self.amide_group.to_dict(
                    include_atoms=include_atoms,
                    include_coordinates=include_coordinates,
                )
                if self.amide_group is not None
                else None
            ),
            "atomic_contacts": [
                contact.to_dict()
                for contact in self.atomic_contacts
            ],
            "warnings": list(self.warnings),
            "metadata": dict(self.metadata),
        }

        if include_raw_geometry:
            result["geometry"] = {
                "centroid_distance": self.centroid_distance,
                "minimum_atomic_distance": self.minimum_atomic_distance,
                "maximum_atomic_distance": self.maximum_atomic_distance,
                "normal_angle": self.normal_angle,
                "plane_angle": self.plane_angle,
                "lateral_offset": self.lateral_offset,
                "radial_offset": self.radial_offset,
                "plane_height": self.plane_height,
                "ring_1_planarity": self.ring_1_planarity,
                "ring_2_planarity": self.ring_2_planarity,
                "group_planarity": self.group_planarity,
            }

            result["score_components"] = {
                "distance_score": self.distance_score,
                "angle_score": self.angle_score,
                "offset_score": self.offset_score,
                "planarity_score": self.planarity_score,
                "charge_score": self.charge_score,
                "geometry_score": self.geometry_score,
            }

        return result


# -----------------------------------------------------------------------------
# 2.8. Resumo por resíduo
# -----------------------------------------------------------------------------

@dataclass(slots=True)
class PiResidueSummary:
    """
    Summary of pi-related interactions associated with one residue.
    """

    residue_identifier: str

    residue_name: Optional[str] = None
    residue_number: Optional[Union[int, str]] = None
    chain_id: Optional[str] = None
    model_id: Optional[str] = None

    interaction_ids: List[str] = field(default_factory=list)
    interaction_types: Counter = field(default_factory=Counter)
    geometry_types: Counter = field(default_factory=Counter)
    strength_distribution: Counter = field(default_factory=Counter)

    total_interactions: int = 0
    accepted_interactions: int = 0

    total_score: float = 0.0
    mean_score: float = 0.0
    maximum_score: float = 0.0

    minimum_distance: Optional[float] = None
    mean_distance: Optional[float] = None
    maximum_distance: Optional[float] = None

    pose_ids: Set[str] = field(default_factory=set)
    pose_frequency: Optional[float] = None

    is_hotspot: bool = False
    hotspot_score: float = 0.0

    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.residue_identifier = str(
            self.residue_identifier
        ).strip()

        if not self.residue_identifier:
            raise ValueError(
                "PiResidueSummary.residue_identifier cannot be empty."
            )

        self.residue_name = _normalize_optional_text(self.residue_name)
        self.chain_id = _normalize_optional_text(self.chain_id)
        self.model_id = _normalize_optional_text(self.model_id)

        self.interaction_ids = [
            str(value)
            for value in self.interaction_ids
            if str(value).strip()
        ]

        self.interaction_types = Counter(self.interaction_types)
        self.geometry_types = Counter(self.geometry_types)
        self.strength_distribution = Counter(
            self.strength_distribution
        )

        self.total_interactions = int(self.total_interactions)
        self.accepted_interactions = int(
            self.accepted_interactions
        )

        if self.total_interactions < 0:
            raise ValueError(
                "total_interactions cannot be negative."
            )

        if not 0 <= self.accepted_interactions <= self.total_interactions:
            raise ValueError(
                "accepted_interactions must be between zero and "
                "total_interactions."
            )

        self.total_score = _coerce_non_negative_float(
            self.total_score,
            field_name="PiResidueSummary.total_score",
        )

        self.mean_score = _coerce_non_negative_float(
            self.mean_score,
            field_name="PiResidueSummary.mean_score",
        )

        self.maximum_score = _coerce_non_negative_float(
            self.maximum_score,
            field_name="PiResidueSummary.maximum_score",
        )

        for field_name in (
            "minimum_distance",
            "mean_distance",
            "maximum_distance",
        ):
            setattr(
                self,
                field_name,
                _coerce_optional_float(
                    getattr(self, field_name),
                    field_name=f"PiResidueSummary.{field_name}",
                    minimum=0.0,
                ),
            )

        self.pose_ids = {
            str(pose_id)
            for pose_id in self.pose_ids
            if str(pose_id).strip()
        }

        self.pose_frequency = _coerce_optional_float(
            self.pose_frequency,
            field_name="PiResidueSummary.pose_frequency",
            minimum=0.0,
            maximum=1.0,
        )

        self.is_hotspot = bool(self.is_hotspot)

        self.hotspot_score = _coerce_non_negative_float(
            self.hotspot_score,
            field_name="PiResidueSummary.hotspot_score",
        )

        self.metadata = _copy_mapping(self.metadata)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the residue summary into a serializable dictionary.
        """

        return {
            "residue_identifier": self.residue_identifier,
            "residue_name": self.residue_name,
            "residue_number": self.residue_number,
            "chain_id": self.chain_id,
            "model_id": self.model_id,
            "interaction_ids": list(self.interaction_ids),
            "interaction_types": dict(self.interaction_types),
            "geometry_types": dict(self.geometry_types),
            "strength_distribution": dict(
                self.strength_distribution
            ),
            "total_interactions": self.total_interactions,
            "accepted_interactions": self.accepted_interactions,
            "total_score": self.total_score,
            "mean_score": self.mean_score,
            "maximum_score": self.maximum_score,
            "minimum_distance": self.minimum_distance,
            "mean_distance": self.mean_distance,
            "maximum_distance": self.maximum_distance,
            "pose_ids": sorted(self.pose_ids),
            "pose_frequency": self.pose_frequency,
            "is_hotspot": self.is_hotspot,
            "hotspot_score": self.hotspot_score,
            "metadata": dict(self.metadata),
        }


# -----------------------------------------------------------------------------
# 2.9. Estatísticas gerais
# -----------------------------------------------------------------------------

@dataclass(slots=True)
class PiStatistics:
    """
    Global statistics generated from pi-related interactions.
    """

    total_interactions: int = 0
    accepted_interactions: int = 0
    rejected_interactions: int = 0
    duplicate_interactions: int = 0

    total_atomic_contacts: int = 0

    total_rings: int = 0
    protein_rings: int = 0
    ligand_rings: int = 0
    heteroaromatic_rings: int = 0
    fused_rings: int = 0

    interaction_type_distribution: Counter = field(
        default_factory=Counter
    )

    geometry_type_distribution: Counter = field(
        default_factory=Counter
    )

    geometry_class_distribution: Counter = field(
        default_factory=Counter
    )

    strength_distribution: Counter = field(
        default_factory=Counter
    )

    residue_distribution: Counter = field(
        default_factory=Counter
    )

    chain_distribution: Counter = field(
        default_factory=Counter
    )

    pose_distribution: Counter = field(
        default_factory=Counter
    )

    involved_residues: List[str] = field(default_factory=list)
    involved_chains: List[str] = field(default_factory=list)
    involved_poses: List[str] = field(default_factory=list)

    minimum_distance: Optional[float] = None
    mean_distance: Optional[float] = None
    median_distance: Optional[float] = None
    maximum_distance: Optional[float] = None

    minimum_score: Optional[float] = None
    mean_score: Optional[float] = None
    median_score: Optional[float] = None
    maximum_score: Optional[float] = None

    total_score: float = 0.0
    normalized_score: float = 0.0

    hotspot_count: int = 0
    hotspot_residues: List[str] = field(default_factory=list)

    strongest_interaction_id: Optional[str] = None
    predominant_interaction_type: Optional[str] = None
    predominant_geometry_type: Optional[str] = None

    total_poses: int = 0
    poses_with_interactions: int = 0
    pose_coverage: float = 0.0

    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        integer_fields = (
            "total_interactions",
            "accepted_interactions",
            "rejected_interactions",
            "duplicate_interactions",
            "total_atomic_contacts",
            "total_rings",
            "protein_rings",
            "ligand_rings",
            "heteroaromatic_rings",
            "fused_rings",
            "hotspot_count",
            "total_poses",
            "poses_with_interactions",
        )

        for field_name in integer_fields:
            converted = int(getattr(self, field_name))

            if converted < 0:
                raise ValueError(
                    f"PiStatistics.{field_name} cannot be negative."
                )

            setattr(self, field_name, converted)

        self.interaction_type_distribution = Counter(
            self.interaction_type_distribution
        )

        self.geometry_type_distribution = Counter(
            self.geometry_type_distribution
        )

        self.geometry_class_distribution = Counter(
            self.geometry_class_distribution
        )

        self.strength_distribution = Counter(
            self.strength_distribution
        )

        self.residue_distribution = Counter(
            self.residue_distribution
        )

        self.chain_distribution = Counter(
            self.chain_distribution
        )

        self.pose_distribution = Counter(
            self.pose_distribution
        )

        self.involved_residues = sorted(
            set(
                _coerce_string_tuple(
                    self.involved_residues,
                    field_name="PiStatistics.involved_residues",
                )
            )
        )

        self.involved_chains = sorted(
            set(
                _coerce_string_tuple(
                    self.involved_chains,
                    field_name="PiStatistics.involved_chains",
                )
            )
        )

        self.involved_poses = sorted(
            set(
                _coerce_string_tuple(
                    self.involved_poses,
                    field_name="PiStatistics.involved_poses",
                )
            )
        )

        self.hotspot_residues = sorted(
            set(
                _coerce_string_tuple(
                    self.hotspot_residues,
                    field_name="PiStatistics.hotspot_residues",
                )
            )
        )

        optional_non_negative_fields = (
            "minimum_distance",
            "mean_distance",
            "median_distance",
            "maximum_distance",
            "minimum_score",
            "mean_score",
            "median_score",
            "maximum_score",
        )

        for field_name in optional_non_negative_fields:
            setattr(
                self,
                field_name,
                _coerce_optional_float(
                    getattr(self, field_name),
                    field_name=f"PiStatistics.{field_name}",
                    minimum=0.0,
                ),
            )

        self.total_score = _coerce_non_negative_float(
            self.total_score,
            field_name="PiStatistics.total_score",
        )

        self.normalized_score = _coerce_fraction(
            self.normalized_score,
            field_name="PiStatistics.normalized_score",
        )

        self.pose_coverage = _coerce_fraction(
            self.pose_coverage,
            field_name="PiStatistics.pose_coverage",
        )

        self.strongest_interaction_id = _normalize_optional_text(
            self.strongest_interaction_id
        )

        self.predominant_interaction_type = _normalize_optional_text(
            self.predominant_interaction_type
        )

        self.predominant_geometry_type = _normalize_optional_text(
            self.predominant_geometry_type
        )

        self.metadata = _copy_mapping(self.metadata)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the statistics object into a serializable dictionary.
        """

        return {
            "total_interactions": self.total_interactions,
            "accepted_interactions": self.accepted_interactions,
            "rejected_interactions": self.rejected_interactions,
            "duplicate_interactions": self.duplicate_interactions,
            "total_atomic_contacts": self.total_atomic_contacts,
            "total_rings": self.total_rings,
            "protein_rings": self.protein_rings,
            "ligand_rings": self.ligand_rings,
            "heteroaromatic_rings": self.heteroaromatic_rings,
            "fused_rings": self.fused_rings,
            "interaction_type_distribution": dict(
                self.interaction_type_distribution
            ),
            "geometry_type_distribution": dict(
                self.geometry_type_distribution
            ),
            "geometry_class_distribution": dict(
                self.geometry_class_distribution
            ),
            "strength_distribution": dict(
                self.strength_distribution
            ),
            "residue_distribution": dict(
                self.residue_distribution
            ),
            "chain_distribution": dict(
                self.chain_distribution
            ),
            "pose_distribution": dict(
                self.pose_distribution
            ),
            "involved_residues": list(self.involved_residues),
            "involved_chains": list(self.involved_chains),
            "involved_poses": list(self.involved_poses),
            "minimum_distance": self.minimum_distance,
            "mean_distance": self.mean_distance,
            "median_distance": self.median_distance,
            "maximum_distance": self.maximum_distance,
            "minimum_score": self.minimum_score,
            "mean_score": self.mean_score,
            "median_score": self.median_score,
            "maximum_score": self.maximum_score,
            "total_score": self.total_score,
            "normalized_score": self.normalized_score,
            "hotspot_count": self.hotspot_count,
            "hotspot_residues": list(self.hotspot_residues),
            "strongest_interaction_id": self.strongest_interaction_id,
            "predominant_interaction_type": (
                self.predominant_interaction_type
            ),
            "predominant_geometry_type": (
                self.predominant_geometry_type
            ),
            "total_poses": self.total_poses,
            "poses_with_interactions": self.poses_with_interactions,
            "pose_coverage": self.pose_coverage,
            "metadata": dict(self.metadata),
        }


# -----------------------------------------------------------------------------
# 2.10. Configuração da análise
# -----------------------------------------------------------------------------

@dataclass(slots=True)
class PiAnalysisConfig:
    """
    Configuration object controlling pi-interaction analysis.
    """

    enabled_interaction_types: FrozenSet[str] = field(
        default_factory=lambda: DEFAULT_ENABLED_PI_INTERACTION_TYPES
    )

    include_protein_rings: bool = True
    include_ligand_rings: bool = True
    include_nucleic_acid_rings: bool = True

    include_backbone_amides: bool = False
    include_side_chain_amides: bool = True
    include_ligand_amides: bool = True

    infer_ligand_aromaticity: bool = True
    infer_ligand_charges: bool = True
    infer_protein_charges: bool = True

    allow_fused_rings: bool = True
    treat_fused_system_as_single_ring: bool = False
    allow_heteroaromatic_rings: bool = True

    minimum_ring_size: int = DEFAULT_MINIMUM_RING_SIZE
    maximum_ring_size: int = DEFAULT_MAXIMUM_RING_SIZE
    maximum_fused_ring_size: int = DEFAULT_MAXIMUM_FUSED_RING_SIZE

    preferred_ring_planarity_rmsd: float = (
        DEFAULT_PREFERRED_RING_PLANARITY_RMSD
    )

    maximum_ring_planarity_rmsd: float = (
        DEFAULT_MAXIMUM_RING_PLANARITY_RMSD
    )

    maximum_ring_atom_deviation: float = (
        DEFAULT_MAXIMUM_RING_ATOM_DEVIATION
    )

    pi_pi_minimum_centroid_distance: float = (
        DEFAULT_PI_PI_MINIMUM_CENTROID_DISTANCE
    )

    pi_pi_optimal_centroid_distance: float = (
        DEFAULT_PI_PI_OPTIMAL_CENTROID_DISTANCE
    )

    pi_pi_favorable_centroid_distance: float = (
        DEFAULT_PI_PI_FAVORABLE_CENTROID_DISTANCE
    )

    pi_pi_maximum_centroid_distance: float = (
        DEFAULT_PI_PI_MAXIMUM_CENTROID_DISTANCE
    )

    pi_pi_parallel_optimal_angle: float = (
        DEFAULT_PI_PI_PARALLEL_OPTIMAL_ANGLE
    )

    pi_pi_parallel_favorable_angle: float = (
        DEFAULT_PI_PI_PARALLEL_FAVORABLE_ANGLE
    )

    pi_pi_parallel_maximum_angle: float = (
        DEFAULT_PI_PI_PARALLEL_MAXIMUM_ANGLE
    )

    pi_pi_t_shaped_optimal_minimum_angle: float = (
        DEFAULT_PI_PI_T_SHAPED_OPTIMAL_MINIMUM_ANGLE
    )

    pi_pi_t_shaped_optimal_maximum_angle: float = (
        DEFAULT_PI_PI_T_SHAPED_OPTIMAL_MAXIMUM_ANGLE
    )

    pi_pi_t_shaped_favorable_minimum_angle: float = (
        DEFAULT_PI_PI_T_SHAPED_FAVORABLE_MINIMUM_ANGLE
    )

    pi_pi_t_shaped_favorable_maximum_angle: float = (
        DEFAULT_PI_PI_T_SHAPED_FAVORABLE_MAXIMUM_ANGLE
    )

    pi_pi_face_to_face_maximum_offset: float = (
        DEFAULT_PI_PI_FACE_TO_FACE_MAXIMUM_OFFSET
    )

    pi_pi_offset_stacking_maximum_offset: float = (
        DEFAULT_PI_PI_OFFSET_STACKING_MAXIMUM_OFFSET
    )

    pi_pi_maximum_lateral_offset: float = (
        DEFAULT_PI_PI_MAXIMUM_LATERAL_OFFSET
    )

    cation_pi_minimum_distance: float = (
        DEFAULT_CATION_PI_MINIMUM_DISTANCE
    )

    cation_pi_optimal_distance: float = (
        DEFAULT_CATION_PI_OPTIMAL_DISTANCE
    )

    cation_pi_favorable_distance: float = (
        DEFAULT_CATION_PI_FAVORABLE_DISTANCE
    )

    cation_pi_maximum_distance: float = (
        DEFAULT_CATION_PI_MAXIMUM_DISTANCE
    )

    cation_pi_optimal_radial_offset: float = (
        DEFAULT_CATION_PI_OPTIMAL_RADIAL_OFFSET
    )

    cation_pi_maximum_radial_offset: float = (
        DEFAULT_CATION_PI_MAXIMUM_RADIAL_OFFSET
    )

    anion_pi_minimum_distance: float = (
        DEFAULT_ANION_PI_MINIMUM_DISTANCE
    )

    anion_pi_optimal_distance: float = (
        DEFAULT_ANION_PI_OPTIMAL_DISTANCE
    )

    anion_pi_favorable_distance: float = (
        DEFAULT_ANION_PI_FAVORABLE_DISTANCE
    )

    anion_pi_maximum_distance: float = (
        DEFAULT_ANION_PI_MAXIMUM_DISTANCE
    )

    anion_pi_optimal_radial_offset: float = (
        DEFAULT_ANION_PI_OPTIMAL_RADIAL_OFFSET
    )

    anion_pi_maximum_radial_offset: float = (
        DEFAULT_ANION_PI_MAXIMUM_RADIAL_OFFSET
    )

    amide_pi_minimum_distance: float = (
        DEFAULT_AMIDE_PI_MINIMUM_DISTANCE
    )

    amide_pi_optimal_distance: float = (
        DEFAULT_AMIDE_PI_OPTIMAL_DISTANCE
    )

    amide_pi_favorable_distance: float = (
        DEFAULT_AMIDE_PI_FAVORABLE_DISTANCE
    )

    amide_pi_maximum_distance: float = (
        DEFAULT_AMIDE_PI_MAXIMUM_DISTANCE
    )

    amide_pi_parallel_optimal_angle: float = (
        DEFAULT_AMIDE_PI_PARALLEL_OPTIMAL_ANGLE
    )

    amide_pi_parallel_maximum_angle: float = (
        DEFAULT_AMIDE_PI_PARALLEL_MAXIMUM_ANGLE
    )

    amide_pi_maximum_radial_offset: float = (
        DEFAULT_AMIDE_PI_MAXIMUM_RADIAL_OFFSET
    )

    sulfur_pi_minimum_distance: float = (
        DEFAULT_SULFUR_PI_MINIMUM_DISTANCE
    )

    sulfur_pi_optimal_distance: float = (
        DEFAULT_SULFUR_PI_OPTIMAL_DISTANCE
    )

    sulfur_pi_favorable_distance: float = (
        DEFAULT_SULFUR_PI_FAVORABLE_DISTANCE
    )

    sulfur_pi_maximum_distance: float = (
        DEFAULT_SULFUR_PI_MAXIMUM_DISTANCE
    )

    positive_partial_charge_threshold: float = (
        DEFAULT_POSITIVE_PARTIAL_CHARGE_THRESHOLD
    )

    negative_partial_charge_threshold: float = (
        DEFAULT_NEGATIVE_PARTIAL_CHARGE_THRESHOLD
    )

    centroid_deduplication_tolerance: float = (
        DEFAULT_CENTROID_DEDUPLICATION_TOLERANCE
    )

    normal_deduplication_angle: float = (
        DEFAULT_NORMAL_DEDUPLICATION_ANGLE
    )

    interaction_distance_tolerance: float = (
        DEFAULT_INTERACTION_DISTANCE_TOLERANCE
    )

    interaction_angle_tolerance: float = (
        DEFAULT_INTERACTION_ANGLE_TOLERANCE
    )

    hotspot_minimum_interactions: int = (
        DEFAULT_HOTSPOT_MINIMUM_INTERACTIONS
    )

    hotspot_minimum_score: float = (
        DEFAULT_HOTSPOT_MINIMUM_SCORE
    )

    multipose_hotspot_frequency: float = (
        DEFAULT_MULTIPOSE_HOTSPOT_FREQUENCY
    )

    strong_score_threshold: float = (
        DEFAULT_STRONG_SCORE_THRESHOLD
    )

    moderate_score_threshold: float = (
        DEFAULT_MODERATE_SCORE_THRESHOLD
    )

    weak_score_threshold: float = (
        DEFAULT_WEAK_SCORE_THRESHOLD
    )

    interaction_base_scores: Mapping[str, float] = field(
        default_factory=lambda: dict(
            DEFAULT_INTERACTION_BASE_SCORES
        )
    )

    geometry_score_multipliers: Mapping[str, float] = field(
        default_factory=lambda: dict(
            DEFAULT_GEOMETRY_SCORE_MULTIPLIERS
        )
    )

    pi_pi_geometry_multipliers: Mapping[str, float] = field(
        default_factory=lambda: dict(
            DEFAULT_PI_PI_GEOMETRY_MULTIPLIERS
        )
    )

    pi_pi_scoring_weights: Mapping[str, float] = field(
        default_factory=lambda: dict(
            DEFAULT_PI_PI_SCORING_WEIGHTS
        )
    )

    cation_pi_scoring_weights: Mapping[str, float] = field(
        default_factory=lambda: dict(
            DEFAULT_CATION_PI_SCORING_WEIGHTS
        )
    )

    anion_pi_scoring_weights: Mapping[str, float] = field(
        default_factory=lambda: dict(
            DEFAULT_ANION_PI_SCORING_WEIGHTS
        )
    )

    amide_pi_scoring_weights: Mapping[str, float] = field(
        default_factory=lambda: dict(
            DEFAULT_AMIDE_PI_SCORING_WEIGHTS
        )
    )

    sulfur_pi_scoring_weights: Mapping[str, float] = field(
        default_factory=lambda: dict(
            DEFAULT_SULFUR_PI_SCORING_WEIGHTS
        )
    )

    include_atom_details: bool = DEFAULT_INCLUDE_ATOM_DETAILS
    include_coordinates: bool = DEFAULT_INCLUDE_COORDINATES
    include_raw_geometry: bool = DEFAULT_INCLUDE_RAW_GEOMETRY
    include_empty_results: bool = DEFAULT_INCLUDE_EMPTY_RESULTS

    preserve_previous_results: bool = True
    update_dock_model_statistics: bool = True
    update_dock_model_score: bool = True

    strict: bool = False
    emit_warnings: bool = True

    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        enabled = frozenset(
            _validate_interaction_type(interaction_type)
            for interaction_type in self.enabled_interaction_types
        )

        self.enabled_interaction_types = enabled

        boolean_fields = (
            "include_protein_rings",
            "include_ligand_rings",
            "include_nucleic_acid_rings",
            "include_backbone_amides",
            "include_side_chain_amides",
            "include_ligand_amides",
            "infer_ligand_aromaticity",
            "infer_ligand_charges",
            "infer_protein_charges",
            "allow_fused_rings",
            "treat_fused_system_as_single_ring",
            "allow_heteroaromatic_rings",
            "include_atom_details",
            "include_coordinates",
            "include_raw_geometry",
            "include_empty_results",
            "preserve_previous_results",
            "update_dock_model_statistics",
            "update_dock_model_score",
            "strict",
            "emit_warnings",
        )

        for field_name in boolean_fields:
            setattr(
                self,
                field_name,
                bool(getattr(self, field_name)),
            )

        self.minimum_ring_size = int(self.minimum_ring_size)
        self.maximum_ring_size = int(self.maximum_ring_size)
        self.maximum_fused_ring_size = int(
            self.maximum_fused_ring_size
        )

        if not (
            3
            <= self.minimum_ring_size
            <= self.maximum_ring_size
            <= self.maximum_fused_ring_size
        ):
            raise ValueError(
                "Ring-size limits must satisfy: "
                "3 <= minimum <= maximum <= maximum_fused."
            )

        numeric_non_negative_fields = (
            "preferred_ring_planarity_rmsd",
            "maximum_ring_planarity_rmsd",
            "maximum_ring_atom_deviation",
            "pi_pi_minimum_centroid_distance",
            "pi_pi_optimal_centroid_distance",
            "pi_pi_favorable_centroid_distance",
            "pi_pi_maximum_centroid_distance",
            "pi_pi_parallel_optimal_angle",
            "pi_pi_parallel_favorable_angle",
            "pi_pi_parallel_maximum_angle",
            "pi_pi_t_shaped_optimal_minimum_angle",
            "pi_pi_t_shaped_optimal_maximum_angle",
            "pi_pi_t_shaped_favorable_minimum_angle",
            "pi_pi_t_shaped_favorable_maximum_angle",
            "pi_pi_face_to_face_maximum_offset",
            "pi_pi_offset_stacking_maximum_offset",
            "pi_pi_maximum_lateral_offset",
            "cation_pi_minimum_distance",
            "cation_pi_optimal_distance",
            "cation_pi_favorable_distance",
            "cation_pi_maximum_distance",
            "cation_pi_optimal_radial_offset",
            "cation_pi_maximum_radial_offset",
            "anion_pi_minimum_distance",
            "anion_pi_optimal_distance",
            "anion_pi_favorable_distance",
            "anion_pi_maximum_distance",
            "anion_pi_optimal_radial_offset",
            "anion_pi_maximum_radial_offset",
            "amide_pi_minimum_distance",
            "amide_pi_optimal_distance",
            "amide_pi_favorable_distance",
            "amide_pi_maximum_distance",
            "amide_pi_parallel_optimal_angle",
            "amide_pi_parallel_maximum_angle",
            "amide_pi_maximum_radial_offset",
            "sulfur_pi_minimum_distance",
            "sulfur_pi_optimal_distance",
            "sulfur_pi_favorable_distance",
            "sulfur_pi_maximum_distance",
            "centroid_deduplication_tolerance",
            "normal_deduplication_angle",
            "interaction_distance_tolerance",
            "interaction_angle_tolerance",
            "hotspot_minimum_score",
        )

        for field_name in numeric_non_negative_fields:
            setattr(
                self,
                field_name,
                _coerce_non_negative_float(
                    getattr(self, field_name),
                    field_name=f"PiAnalysisConfig.{field_name}",
                ),
            )

        self.positive_partial_charge_threshold = (
            _coerce_optional_float(
                self.positive_partial_charge_threshold,
                field_name=(
                    "PiAnalysisConfig."
                    "positive_partial_charge_threshold"
                ),
            )
        )

        self.negative_partial_charge_threshold = (
            _coerce_optional_float(
                self.negative_partial_charge_threshold,
                field_name=(
                    "PiAnalysisConfig."
                    "negative_partial_charge_threshold"
                ),
            )
        )

        assert self.positive_partial_charge_threshold is not None
        assert self.negative_partial_charge_threshold is not None

        if self.positive_partial_charge_threshold <= 0.0:
            raise ValueError(
                "positive_partial_charge_threshold must be positive."
            )

        if self.negative_partial_charge_threshold >= 0.0:
            raise ValueError(
                "negative_partial_charge_threshold must be negative."
            )

        self.hotspot_minimum_interactions = int(
            self.hotspot_minimum_interactions
        )

        if self.hotspot_minimum_interactions < 1:
            raise ValueError(
                "hotspot_minimum_interactions must be at least 1."
            )

        self.multipose_hotspot_frequency = _coerce_fraction(
            self.multipose_hotspot_frequency,
            field_name=(
                "PiAnalysisConfig.multipose_hotspot_frequency"
            ),
        )

        self.strong_score_threshold = _coerce_fraction(
            self.strong_score_threshold,
            field_name="PiAnalysisConfig.strong_score_threshold",
        )

        self.moderate_score_threshold = _coerce_fraction(
            self.moderate_score_threshold,
            field_name="PiAnalysisConfig.moderate_score_threshold",
        )

        self.weak_score_threshold = _coerce_fraction(
            self.weak_score_threshold,
            field_name="PiAnalysisConfig.weak_score_threshold",
        )

        if not (
            self.strong_score_threshold
            > self.moderate_score_threshold
            > self.weak_score_threshold
            >= 0.0
        ):
            raise ValueError(
                "Score thresholds must satisfy: "
                "strong > moderate > weak >= 0."
            )

        self._validate_distance_ordering()
        self._validate_angle_limits()
        self._validate_score_mappings()

        self.metadata = _copy_mapping(self.metadata)

    def _validate_distance_ordering(self) -> None:
        """
        Validate minimum, optimal, favorable, and maximum distances.
        """

        groups = {
            "pi-pi": (
                self.pi_pi_minimum_centroid_distance,
                self.pi_pi_optimal_centroid_distance,
                self.pi_pi_favorable_centroid_distance,
                self.pi_pi_maximum_centroid_distance,
            ),
            "cation-pi": (
                self.cation_pi_minimum_distance,
                self.cation_pi_optimal_distance,
                self.cation_pi_favorable_distance,
                self.cation_pi_maximum_distance,
            ),
            "anion-pi": (
                self.anion_pi_minimum_distance,
                self.anion_pi_optimal_distance,
                self.anion_pi_favorable_distance,
                self.anion_pi_maximum_distance,
            ),
            "amide-pi": (
                self.amide_pi_minimum_distance,
                self.amide_pi_optimal_distance,
                self.amide_pi_favorable_distance,
                self.amide_pi_maximum_distance,
            ),
            "sulfur-pi": (
                self.sulfur_pi_minimum_distance,
                self.sulfur_pi_optimal_distance,
                self.sulfur_pi_favorable_distance,
                self.sulfur_pi_maximum_distance,
            ),
        }

        for label, values in groups.items():
            minimum, optimal, favorable, maximum = values

            if not (
                minimum
                < optimal
                <= favorable
                <= maximum
            ):
                raise ValueError(
                    f"Invalid {label} distance ordering."
                )

    def _validate_angle_limits(self) -> None:
        """
        Validate angular thresholds.
        """

        angle_values = {
            "pi_pi_parallel_optimal_angle": (
                self.pi_pi_parallel_optimal_angle
            ),
            "pi_pi_parallel_favorable_angle": (
                self.pi_pi_parallel_favorable_angle
            ),
            "pi_pi_parallel_maximum_angle": (
                self.pi_pi_parallel_maximum_angle
            ),
            "pi_pi_t_shaped_optimal_minimum_angle": (
                self.pi_pi_t_shaped_optimal_minimum_angle
            ),
            "pi_pi_t_shaped_optimal_maximum_angle": (
                self.pi_pi_t_shaped_optimal_maximum_angle
            ),
            "pi_pi_t_shaped_favorable_minimum_angle": (
                self.pi_pi_t_shaped_favorable_minimum_angle
            ),
            "pi_pi_t_shaped_favorable_maximum_angle": (
                self.pi_pi_t_shaped_favorable_maximum_angle
            ),
            "amide_pi_parallel_optimal_angle": (
                self.amide_pi_parallel_optimal_angle
            ),
            "amide_pi_parallel_maximum_angle": (
                self.amide_pi_parallel_maximum_angle
            ),
        }

        for name, value in angle_values.items():
            if not 0.0 <= value <= 180.0:
                raise ValueError(
                    f"{name} must be between 0 and 180 degrees."
                )

        if not (
            self.pi_pi_parallel_optimal_angle
            <= self.pi_pi_parallel_favorable_angle
            <= self.pi_pi_parallel_maximum_angle
        ):
            raise ValueError(
                "Invalid parallel pi-pi angle ordering."
            )

        if not (
            self.pi_pi_t_shaped_favorable_minimum_angle
            <= self.pi_pi_t_shaped_optimal_minimum_angle
            <= self.pi_pi_t_shaped_optimal_maximum_angle
            <= self.pi_pi_t_shaped_favorable_maximum_angle
        ):
            raise ValueError(
                "Invalid T-shaped pi-pi angle ordering."
            )

        if not (
            self.amide_pi_parallel_optimal_angle
            <= self.amide_pi_parallel_maximum_angle
        ):
            raise ValueError(
                "Invalid amide-pi angle ordering."
            )

    def _validate_score_mappings(self) -> None:
        """
        Validate scoring mappings.
        """

        self.interaction_base_scores = {
            _validate_interaction_type(key): (
                _coerce_non_negative_float(
                    value,
                    field_name=(
                        "PiAnalysisConfig.interaction_base_scores"
                    ),
                )
            )
            for key, value in self.interaction_base_scores.items()
        }

        self.geometry_score_multipliers = {
            _validate_geometry_class(key): _coerce_fraction(
                value,
                field_name=(
                    "PiAnalysisConfig.geometry_score_multipliers"
                ),
            )
            for key, value in (
                self.geometry_score_multipliers.items()
            )
        }

        self.pi_pi_geometry_multipliers = {
            _validate_pi_pi_geometry(key): _coerce_fraction(
                value,
                field_name=(
                    "PiAnalysisConfig.pi_pi_geometry_multipliers"
                ),
            )
            for key, value in (
                self.pi_pi_geometry_multipliers.items()
            )
        }

        weight_mappings = {
            "pi_pi_scoring_weights": self.pi_pi_scoring_weights,
            "cation_pi_scoring_weights": (
                self.cation_pi_scoring_weights
            ),
            "anion_pi_scoring_weights": (
                self.anion_pi_scoring_weights
            ),
            "amide_pi_scoring_weights": (
                self.amide_pi_scoring_weights
            ),
            "sulfur_pi_scoring_weights": (
                self.sulfur_pi_scoring_weights
            ),
        }

        for field_name, mapping in weight_mappings.items():
            copied_mapping = {
                str(key): float(value)
                for key, value in mapping.items()
            }

            _validate_weight_mapping(
                copied_mapping,
                name=f"PiAnalysisConfig.{field_name}",
            )

            setattr(
                self,
                field_name,
                copied_mapping,
            )

    def is_enabled(
        self,
        interaction_type: str,
    ) -> bool:
        """
        Return whether an interaction type is enabled.
        """

        return (
            _validate_interaction_type(interaction_type)
            in self.enabled_interaction_types
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the configuration into a serializable dictionary.
        """

        result: Dict[str, Any] = {}

        for dataclass_field in fields(self):
            value = getattr(self, dataclass_field.name)

            if isinstance(value, frozenset):
                result[dataclass_field.name] = sorted(value)

            elif isinstance(value, Mapping):
                result[dataclass_field.name] = dict(value)

            else:
                result[dataclass_field.name] = value

        return result


# -----------------------------------------------------------------------------
# 2.11. Resultado completo da análise
# -----------------------------------------------------------------------------

@dataclass(slots=True)
class PiAnalysisResult:
    """
    Complete output generated by a pi-interaction analysis.
    """

    interactions: List[PiInteraction] = field(default_factory=list)
    rings: List[PiRing] = field(default_factory=list)
    charged_groups: List[PiChargedGroup] = field(default_factory=list)
    amide_groups: List[PiAmideGroup] = field(default_factory=list)

    residue_summaries: Dict[str, PiResidueSummary] = field(
        default_factory=dict
    )

    hotspots: List[PiResidueSummary] = field(default_factory=list)

    statistics: PiStatistics = field(
        default_factory=PiStatistics
    )

    configuration: PiAnalysisConfig = field(
        default_factory=PiAnalysisConfig
    )

    score: float = 0.0
    normalized_score: float = 0.0

    pose_id: Optional[str] = None
    model_id: Optional[str] = None

    analyzed_receptor_atoms: int = 0
    analyzed_ligand_atoms: int = 0

    success: bool = True
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.interactions = list(self.interactions)
        self.rings = list(self.rings)
        self.charged_groups = list(self.charged_groups)
        self.amide_groups = list(self.amide_groups)

        if not all(
            isinstance(interaction, PiInteraction)
            for interaction in self.interactions
        ):
            raise TypeError(
                "PiAnalysisResult.interactions must contain only "
                "PiInteraction objects."
            )

        if not all(
            isinstance(ring, PiRing)
            for ring in self.rings
        ):
            raise TypeError(
                "PiAnalysisResult.rings must contain only PiRing objects."
            )

        if not all(
            isinstance(group, PiChargedGroup)
            for group in self.charged_groups
        ):
            raise TypeError(
                "PiAnalysisResult.charged_groups must contain only "
                "PiChargedGroup objects."
            )

        if not all(
            isinstance(group, PiAmideGroup)
            for group in self.amide_groups
        ):
            raise TypeError(
                "PiAnalysisResult.amide_groups must contain only "
                "PiAmideGroup objects."
            )

        self.residue_summaries = dict(self.residue_summaries)

        if not all(
            isinstance(summary, PiResidueSummary)
            for summary in self.residue_summaries.values()
        ):
            raise TypeError(
                "PiAnalysisResult.residue_summaries must contain only "
                "PiResidueSummary values."
            )

        self.hotspots = list(self.hotspots)

        if not all(
            isinstance(summary, PiResidueSummary)
            for summary in self.hotspots
        ):
            raise TypeError(
                "PiAnalysisResult.hotspots must contain only "
                "PiResidueSummary objects."
            )

        if not isinstance(self.statistics, PiStatistics):
            raise TypeError(
                "PiAnalysisResult.statistics must be a PiStatistics."
            )

        if not isinstance(self.configuration, PiAnalysisConfig):
            raise TypeError(
                "PiAnalysisResult.configuration must be "
                "a PiAnalysisConfig."
            )

        self.score = _coerce_non_negative_float(
            self.score,
            field_name="PiAnalysisResult.score",
        )

        self.normalized_score = _coerce_fraction(
            self.normalized_score,
            field_name="PiAnalysisResult.normalized_score",
        )

        self.pose_id = _normalize_optional_text(self.pose_id)
        self.model_id = _normalize_optional_text(self.model_id)

        self.analyzed_receptor_atoms = int(
            self.analyzed_receptor_atoms
        )

        self.analyzed_ligand_atoms = int(
            self.analyzed_ligand_atoms
        )

        if self.analyzed_receptor_atoms < 0:
            raise ValueError(
                "analyzed_receptor_atoms cannot be negative."
            )

        if self.analyzed_ligand_atoms < 0:
            raise ValueError(
                "analyzed_ligand_atoms cannot be negative."
            )

        self.success = bool(self.success)

        self.warnings = list(
            _coerce_string_tuple(
                self.warnings,
                field_name="PiAnalysisResult.warnings",
            )
        )

        self.errors = list(
            _coerce_string_tuple(
                self.errors,
                field_name="PiAnalysisResult.errors",
            )
        )

        self.metadata = _copy_mapping(self.metadata)

        if self.errors:
            self.success = False

    @property
    def accepted_interactions(self) -> List[PiInteraction]:
        """
        Return accepted and non-duplicate interactions.
        """

        return [
            interaction
            for interaction in self.interactions
            if interaction.is_valid_interaction
        ]

    @property
    def rejected_interactions(self) -> List[PiInteraction]:
        """
        Return rejected interactions.
        """

        return [
            interaction
            for interaction in self.interactions
            if not interaction.accepted
        ]

    @property
    def duplicate_interactions(self) -> List[PiInteraction]:
        """
        Return interactions marked as duplicates.
        """

        return [
            interaction
            for interaction in self.interactions
            if interaction.is_duplicate
        ]

    @property
    def interaction_count(self) -> int:
        """
        Return the total number of interactions.
        """

        return len(self.interactions)

    @property
    def accepted_interaction_count(self) -> int:
        """
        Return the number of valid accepted interactions.
        """

        return len(self.accepted_interactions)

    @property
    def has_interactions(self) -> bool:
        """
        Return whether at least one accepted interaction exists.
        """

        return bool(self.accepted_interactions)

    def get_interactions_by_type(
        self,
        interaction_type: str,
        *,
        accepted_only: bool = True,
    ) -> List[PiInteraction]:
        """
        Return interactions of a selected type.
        """

        normalized_type = _validate_interaction_type(
            interaction_type
        )

        source = (
            self.accepted_interactions
            if accepted_only
            else self.interactions
        )

        return [
            interaction
            for interaction in source
            if interaction.interaction_type == normalized_type
        ]

    def get_residue_summary(
        self,
        residue_identifier: str,
    ) -> Optional[PiResidueSummary]:
        """
        Return the summary associated with a residue identifier.
        """

        return self.residue_summaries.get(
            str(residue_identifier).strip()
        )

    def to_dict(
        self,
        *,
        include_configuration: bool = True,
    ) -> Dict[str, Any]:
        """
        Convert the complete analysis result into a dictionary.
        """

        include_atoms = self.configuration.include_atom_details
        include_coordinates = self.configuration.include_coordinates
        include_raw_geometry = (
            self.configuration.include_raw_geometry
        )

        result: Dict[str, Any] = {
            "module": MODULE_NAME,
            "module_version": MODULE_VERSION,
            "success": self.success,
            "pose_id": self.pose_id,
            "model_id": self.model_id,
            "score": self.score,
            "normalized_score": self.normalized_score,
            "analyzed_receptor_atoms": self.analyzed_receptor_atoms,
            "analyzed_ligand_atoms": self.analyzed_ligand_atoms,
            "interactions": [
                interaction.to_dict(
                    include_atoms=include_atoms,
                    include_coordinates=include_coordinates,
                    include_raw_geometry=include_raw_geometry,
                )
                for interaction in self.interactions
            ],
            "rings": [
                ring.to_dict(
                    include_atoms=include_atoms,
                    include_coordinates=include_coordinates,
                )
                for ring in self.rings
            ],
            "charged_groups": [
                group.to_dict(
                    include_atoms=include_atoms,
                    include_coordinates=include_coordinates,
                )
                for group in self.charged_groups
            ],
            "amide_groups": [
                group.to_dict(
                    include_atoms=include_atoms,
                    include_coordinates=include_coordinates,
                )
                for group in self.amide_groups
            ],
            "residue_summaries": {
                residue_identifier: summary.to_dict()
                for residue_identifier, summary
                in self.residue_summaries.items()
            },
            "hotspots": [
                hotspot.to_dict()
                for hotspot in self.hotspots
            ],
            "statistics": self.statistics.to_dict(),
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "metadata": dict(self.metadata),
        }

        if include_configuration:
            result["configuration"] = (
                self.configuration.to_dict()
            )

        return result


# -----------------------------------------------------------------------------
# 2.12. Resultado multipose
# -----------------------------------------------------------------------------

@dataclass(slots=True)
class PiMultiPoseResult:
    """
    Aggregated result generated from multiple poses or DockModel objects.
    """

    pose_results: List[PiAnalysisResult] = field(
        default_factory=list
    )

    consensus_interactions: List[PiInteraction] = field(
        default_factory=list
    )

    residue_summaries: Dict[str, PiResidueSummary] = field(
        default_factory=dict
    )

    hotspots: List[PiResidueSummary] = field(
        default_factory=list
    )

    statistics: PiStatistics = field(
        default_factory=PiStatistics
    )

    configuration: PiAnalysisConfig = field(
        default_factory=PiAnalysisConfig
    )

    total_score: float = 0.0
    mean_pose_score: float = 0.0
    normalized_score: float = 0.0

    successful_poses: int = 0
    failed_poses: int = 0

    success: bool = True
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.pose_results = list(self.pose_results)

        if not all(
            isinstance(result, PiAnalysisResult)
            for result in self.pose_results
        ):
            raise TypeError(
                "PiMultiPoseResult.pose_results must contain only "
                "PiAnalysisResult objects."
            )

        self.consensus_interactions = list(
            self.consensus_interactions
        )

        if not all(
            isinstance(interaction, PiInteraction)
            for interaction in self.consensus_interactions
        ):
            raise TypeError(
                "PiMultiPoseResult.consensus_interactions must contain "
                "only PiInteraction objects."
            )

        self.residue_summaries = dict(self.residue_summaries)
        self.hotspots = list(self.hotspots)

        if not isinstance(self.statistics, PiStatistics):
            raise TypeError(
                "PiMultiPoseResult.statistics must be a PiStatistics."
            )

        if not isinstance(self.configuration, PiAnalysisConfig):
            raise TypeError(
                "PiMultiPoseResult.configuration must be "
                "a PiAnalysisConfig."
            )

        self.total_score = _coerce_non_negative_float(
            self.total_score,
            field_name="PiMultiPoseResult.total_score",
        )

        self.mean_pose_score = _coerce_non_negative_float(
            self.mean_pose_score,
            field_name="PiMultiPoseResult.mean_pose_score",
        )

        self.normalized_score = _coerce_fraction(
            self.normalized_score,
            field_name="PiMultiPoseResult.normalized_score",
        )

        self.successful_poses = int(self.successful_poses)
        self.failed_poses = int(self.failed_poses)

        if self.successful_poses < 0 or self.failed_poses < 0:
            raise ValueError(
                "Pose counts cannot be negative."
            )

        self.success = bool(self.success)

        self.warnings = list(
            _coerce_string_tuple(
                self.warnings,
                field_name="PiMultiPoseResult.warnings",
            )
        )

        self.errors = list(
            _coerce_string_tuple(
                self.errors,
                field_name="PiMultiPoseResult.errors",
            )
        )

        self.metadata = _copy_mapping(self.metadata)

        if self.errors:
            self.success = False

    @property
    def total_poses(self) -> int:
        """
        Return the total number of analyzed poses.
        """

        return len(self.pose_results)

    @property
    def poses_with_interactions(self) -> int:
        """
        Return the number of poses containing accepted interactions.
        """

        return sum(
            1
            for result in self.pose_results
            if result.has_interactions
        )

    @property
    def pose_coverage(self) -> float:
        """
        Return the fraction of poses containing accepted interactions.
        """

        if not self.pose_results:
            return 0.0

        return (
            self.poses_with_interactions
            / len(self.pose_results)
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the multipose result into a serializable dictionary.
        """

        return {
            "module": MODULE_NAME,
            "module_version": MODULE_VERSION,
            "success": self.success,
            "total_poses": self.total_poses,
            "successful_poses": self.successful_poses,
            "failed_poses": self.failed_poses,
            "poses_with_interactions": self.poses_with_interactions,
            "pose_coverage": self.pose_coverage,
            "total_score": self.total_score,
            "mean_pose_score": self.mean_pose_score,
            "normalized_score": self.normalized_score,
            "pose_results": [
                result.to_dict()
                for result in self.pose_results
            ],
            "consensus_interactions": [
                interaction.to_dict(
                    include_atoms=(
                        self.configuration.include_atom_details
                    ),
                    include_coordinates=(
                        self.configuration.include_coordinates
                    ),
                    include_raw_geometry=(
                        self.configuration.include_raw_geometry
                    ),
                )
                for interaction in self.consensus_interactions
            ],
            "residue_summaries": {
                residue_identifier: summary.to_dict()
                for residue_identifier, summary
                in self.residue_summaries.items()
            },
            "hotspots": [
                hotspot.to_dict()
                for hotspot in self.hotspots
            ],
            "statistics": self.statistics.to_dict(),
            "configuration": self.configuration.to_dict(),
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "metadata": dict(self.metadata),
        }


# -----------------------------------------------------------------------------
# 2.13. Aliases públicos
# -----------------------------------------------------------------------------

PiInteractionCollection: TypeAlias = Sequence[PiInteraction]

PiRingCollection: TypeAlias = Sequence[PiRing]

PiChargedGroupCollection: TypeAlias = Sequence[PiChargedGroup]

PiAmideGroupCollection: TypeAlias = Sequence[PiAmideGroup]

PiResidueSummaryMapping: TypeAlias = Mapping[
    str,
    PiResidueSummary,
]


# -----------------------------------------------------------------------------
# 2.14. Configuração padrão reutilizável
# -----------------------------------------------------------------------------

def create_default_pi_config() -> PiAnalysisConfig:
    """
    Create a fresh default PiAnalysisConfig instance.

    A function is used instead of a shared module-level mutable instance to
    prevent state leakage between analyses.
    """

    return PiAnalysisConfig()

# -----------------------------------------------------------------------------
# End of section 2.
# -----------------------------------------------------------------------------

# =============================================================================
# 3. NORMALIZAÇÃO E ACESSO SEGURO AOS ÁTOMOS
# =============================================================================

# -----------------------------------------------------------------------------
# 3.1. Exceções específicas da camada de normalização
# -----------------------------------------------------------------------------

class PiNormalizationError(ValueError):
    """
    Base exception raised when molecular data cannot be normalized.
    """


class PiCoordinateError(PiNormalizationError):
    """
    Raised when valid three-dimensional coordinates cannot be obtained.
    """


class PiAtomAccessError(PiNormalizationError):
    """
    Raised when required atom information cannot be obtained.
    """


class PiResidueAccessError(PiNormalizationError):
    """
    Raised when required residue information cannot be obtained.
    """


# -----------------------------------------------------------------------------
# 3.2. Funções genéricas de acesso a atributos
# -----------------------------------------------------------------------------

_MISSING: Final[object] = object()


def _safe_call_zero_argument(
    value: Any,
    *,
    default: Any = _MISSING,
) -> Any:
    """
    Call a zero-argument callable when appropriate.

    Non-callable values are returned unchanged. Exceptions raised while calling
    the object are suppressed only when a default value was supplied.
    """

    if not callable(value):
        return value

    try:
        return value()

    except Exception:
        if default is not _MISSING:
            return default

        raise


def _safe_get_attribute(
    obj: Any,
    attribute_name: str,
    *,
    default: Any = None,
    call_if_callable: bool = True,
) -> Any:
    """
    Safely retrieve one attribute from an arbitrary object.

    Parameters
    ----------
    obj
        Object from which the attribute should be obtained.

    attribute_name
        Attribute name.

    default
        Value returned when the attribute is unavailable or raises an error.

    call_if_callable
        Whether zero-argument callable attributes should be evaluated.
    """

    if obj is None:
        return default

    try:
        value = getattr(obj, attribute_name)

    except Exception:
        return default

    if call_if_callable and callable(value):
        return _safe_call_zero_argument(
            value,
            default=default,
        )

    return value


def _safe_get_first_attribute(
    obj: Any,
    attribute_names: Iterable[str],
    *,
    default: Any = None,
    call_if_callable: bool = True,
    accept_none: bool = False,
) -> Any:
    """
    Return the first accessible attribute among several alternatives.
    """

    for attribute_name in attribute_names:
        value = _safe_get_attribute(
            obj,
            attribute_name,
            default=_MISSING,
            call_if_callable=call_if_callable,
        )

        if value is _MISSING:
            continue

        if value is None and not accept_none:
            continue

        return value

    return default


def _safe_get_mapping_value(
    obj: Any,
    keys: Iterable[str],
    *,
    default: Any = None,
    accept_none: bool = False,
) -> Any:
    """
    Return the first matching value when ``obj`` behaves like a mapping.
    """

    if not isinstance(obj, Mapping):
        return default

    for key in keys:
        try:
            if key not in obj:
                continue

            value = obj[key]

        except Exception:
            continue

        if value is None and not accept_none:
            continue

        return value

    return default


def _safe_get_value(
    obj: Any,
    names: Iterable[str],
    *,
    default: Any = None,
    call_if_callable: bool = True,
    accept_none: bool = False,
) -> Any:
    """
    Retrieve a value from either object attributes or mapping keys.
    """

    value = _safe_get_first_attribute(
        obj,
        names,
        default=_MISSING,
        call_if_callable=call_if_callable,
        accept_none=accept_none,
    )

    if value is not _MISSING:
        return value

    return _safe_get_mapping_value(
        obj,
        names,
        default=default,
        accept_none=accept_none,
    )


def _normalize_text(
    value: Any,
    *,
    default: str = "",
    uppercase: bool = False,
) -> str:
    """
    Convert an arbitrary value into normalized text.
    """

    if value is None:
        return default

    try:
        text = str(value).strip()

    except Exception:
        return default

    if not text:
        return default

    if uppercase:
        return text.upper()

    return text


def _normalize_optional_integer(
    value: Any,
) -> Optional[int]:
    """
    Convert an arbitrary value into an integer when possible.
    """

    if value is None or isinstance(value, bool):
        return None

    try:
        return int(value)

    except (TypeError, ValueError, OverflowError):
        return None


def _normalize_optional_numeric(
    value: Any,
) -> Optional[float]:
    """
    Convert an arbitrary value into a finite float when possible.
    """

    if value is None or isinstance(value, bool):
        return None

    try:
        converted = float(value)

    except (TypeError, ValueError, OverflowError):
        return None

    if not isfinite(converted):
        return None

    return converted


def _as_sequence(
    value: Any,
    *,
    allow_string: bool = False,
) -> Tuple[Any, ...]:
    """
    Convert an iterable-like value into a tuple.

    Scalar values are converted into a one-item tuple.
    """

    if value is None:
        return ()

    if isinstance(value, (str, bytes)):
        if allow_string:
            return (value,)

        return ()

    if isinstance(value, Mapping):
        return tuple(value.values())

    try:
        return tuple(value)

    except TypeError:
        return (value,)


# -----------------------------------------------------------------------------
# 3.3. Normalização de elementos químicos
# -----------------------------------------------------------------------------

_ELEMENT_ALIASES: Final[Mapping[str, str]] = {
    "HYDROGEN": "H",
    "DEUTERIUM": "D",
    "TRITIUM": "T",
    "CARBON": "C",
    "NITROGEN": "N",
    "OXYGEN": "O",
    "FLUORINE": "F",
    "PHOSPHORUS": "P",
    "SULFUR": "S",
    "CHLORINE": "CL",
    "BROMINE": "BR",
    "IODINE": "I",
    "BORON": "B",
    "SILICON": "SI",
    "SELENIUM": "SE",
}


_ATOMIC_NUMBER_TO_SYMBOL: Final[Mapping[int, str]] = {
    1: "H",
    5: "B",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
    14: "SI",
    15: "P",
    16: "S",
    17: "CL",
    34: "SE",
    35: "BR",
    53: "I",
}


_TWO_LETTER_ELEMENTS: Final[FrozenSet[str]] = frozenset(
    {
        "CL",
        "BR",
        "SI",
        "SE",
        "NA",
        "MG",
        "AL",
        "CA",
        "FE",
        "ZN",
        "CU",
        "MN",
        "CO",
        "NI",
        "CD",
        "HG",
        "PB",
    }
)


def normalize_element_symbol(
    value: Any,
    *,
    atom_name: Optional[str] = None,
) -> str:
    """
    Normalize an element representation into an uppercase chemical symbol.

    The function supports:

    - strings;
    - atomic numbers;
    - ChimeraX element objects;
    - objects exposing ``name``, ``symbol`` or ``number``;
    - fallback inference from the atom name.
    """

    if value is not None:
        if isinstance(value, bool):
            value = None

        elif isinstance(value, int):
            symbol = _ATOMIC_NUMBER_TO_SYMBOL.get(value)

            if symbol is not None:
                return symbol

        elif isinstance(value, float) and value.is_integer():
            symbol = _ATOMIC_NUMBER_TO_SYMBOL.get(int(value))

            if symbol is not None:
                return symbol

    if value is not None and not isinstance(value, str):
        nested_symbol = _safe_get_value(
            value,
            (
                "symbol",
                "name",
                "element_name",
            ),
            default=None,
        )

        if nested_symbol is not None and nested_symbol is not value:
            normalized = normalize_element_symbol(
                nested_symbol,
                atom_name=atom_name,
            )

            if normalized:
                return normalized

        atomic_number = _safe_get_value(
            value,
            (
                "number",
                "atomic_number",
            ),
            default=None,
        )

        normalized_number = _normalize_optional_integer(
            atomic_number
        )

        if normalized_number is not None:
            symbol = _ATOMIC_NUMBER_TO_SYMBOL.get(
                normalized_number
            )

            if symbol is not None:
                return symbol

    text = _normalize_text(
        value,
        uppercase=True,
    )

    if text:
        text = text.replace(" ", "")
        text = text.replace("_", "")
        text = text.replace("-", "")

        if text in _ELEMENT_ALIASES:
            return _ELEMENT_ALIASES[text]

        if text.isdigit():
            symbol = _ATOMIC_NUMBER_TO_SYMBOL.get(int(text))

            if symbol is not None:
                return symbol

        if len(text) == 1 and text.isalpha():
            return text

        if len(text) >= 2:
            first_two = text[:2]

            if first_two in _TWO_LETTER_ELEMENTS:
                return first_two

            if text[0].isalpha():
                return text[0]

    return infer_element_from_atom_name(atom_name)


def infer_element_from_atom_name(
    atom_name: Optional[str],
) -> str:
    """
    Infer a chemical element from an atom name.

    Digits and common PDB atom-name prefixes are removed before inference.
    """

    text = _normalize_text(
        atom_name,
        uppercase=True,
    )

    if not text:
        return ""

    cleaned = "".join(
        character
        for character in text
        if character.isalpha()
    )

    if not cleaned:
        return ""

    if len(cleaned) >= 2 and cleaned[:2] in _TWO_LETTER_ELEMENTS:
        return cleaned[:2]

    return cleaned[0]


# -----------------------------------------------------------------------------
# 3.4. Acesso a nomes e identificadores atômicos
# -----------------------------------------------------------------------------

def get_atom_name(
    atom: Any,
    *,
    default: str = "",
) -> str:
    """
    Return a normalized atom name.
    """

    value = _safe_get_value(
        atom,
        ATOM_NAME_ATTRIBUTES,
        default=None,
    )

    return _normalize_text(
        value,
        default=default,
    )


def get_atom_element(
    atom: Any,
    *,
    default: str = "",
) -> str:
    """
    Return the normalized chemical element of an atom.
    """

    atom_name = get_atom_name(atom)

    value = _safe_get_value(
        atom,
        ATOM_ELEMENT_ATTRIBUTES,
        default=None,
    )

    normalized = normalize_element_symbol(
        value,
        atom_name=atom_name,
    )

    return normalized or default


def get_atom_index(
    atom: Any,
) -> Optional[int]:
    """
    Return an atom index when available.
    """

    value = _safe_get_value(
        atom,
        (
            "index",
            "atom_index",
            "idx",
        ),
        default=None,
    )

    return _normalize_optional_integer(value)


def get_atom_serial_number(
    atom: Any,
) -> Optional[int]:
    """
    Return an atom serial number when available.
    """

    value = _safe_get_value(
        atom,
        (
            "serial_number",
            "serial",
            "serialNumber",
            "pdb_serial",
        ),
        default=None,
    )

    return _normalize_optional_integer(value)


def get_atom_type(
    atom: Any,
) -> Optional[str]:
    """
    Return the best available atom-type description.
    """

    value = _safe_get_value(
        atom,
        ATOM_TYPE_ATTRIBUTES,
        default=None,
    )

    return _normalize_optional_text(value)


def get_atom_model(
    atom: Any,
) -> Optional[Any]:
    """
    Return the molecular model associated with an atom.
    """

    model = _safe_get_value(
        atom,
        (
            "structure",
            "model",
            "molecule",
            "parent_model",
        ),
        default=None,
    )

    if model is not None:
        return model

    residue = get_atom_residue(atom)

    if residue is None:
        return None

    return _safe_get_value(
        residue,
        (
            "structure",
            "model",
            "molecule",
            "parent_model",
        ),
        default=None,
    )


# -----------------------------------------------------------------------------
# 3.5. Normalização de coordenadas
# -----------------------------------------------------------------------------

def _coordinate_components_from_object(
    value: Any,
) -> Optional[Coordinate3D]:
    """
    Extract x, y, and z components from an arbitrary coordinate object.
    """

    if value is None:
        return None

    if isinstance(value, Mapping):
        lowered = {
            str(key).lower(): component
            for key, component in value.items()
        }

        if all(axis in lowered for axis in ("x", "y", "z")):
            candidate = (
                lowered["x"],
                lowered["y"],
                lowered["z"],
            )

            try:
                return _coerce_coordinate3d(
                    candidate,
                    field_name="coordinate",
                )

            except (TypeError, ValueError):
                return None

    x = _safe_get_value(
        value,
        ("x",),
        default=_MISSING,
    )

    y = _safe_get_value(
        value,
        ("y",),
        default=_MISSING,
    )

    z = _safe_get_value(
        value,
        ("z",),
        default=_MISSING,
    )

    if (
        x is not _MISSING
        and y is not _MISSING
        and z is not _MISSING
    ):
        try:
            return _coerce_coordinate3d(
                (x, y, z),
                field_name="coordinate",
            )

        except (TypeError, ValueError):
            return None

    if NUMPY_AVAILABLE and isinstance(value, np.ndarray):
        flattened = np.asarray(
            value,
            dtype=float,
        ).reshape(-1)

        if flattened.size >= 3:
            candidate = tuple(
                float(component)
                for component in flattened[:3]
            )

            if all(isfinite(component) for component in candidate):
                return candidate  # type: ignore[return-value]

    try:
        sequence = tuple(value)

    except TypeError:
        return None

    if len(sequence) < 3:
        return None

    try:
        return _coerce_coordinate3d(
            sequence[:3],
            field_name="coordinate",
        )

    except (TypeError, ValueError):
        return None


def get_atom_coordinate(
    atom: Any,
    *,
    use_scene_coordinates: bool = True,
    strict: bool = False,
) -> Optional[Coordinate3D]:
    """
    Return the three-dimensional coordinate of an atom.

    Parameters
    ----------
    atom
        Atom-like object.

    use_scene_coordinates
        Prefer transformed ChimeraX scene coordinates when available.

    strict
        Raise ``PiCoordinateError`` instead of returning ``None``.
    """

    if atom is None:
        if strict:
            raise PiCoordinateError(
                "Cannot obtain coordinates from None."
            )

        return None

    coordinate_attributes = list(
        ATOM_COORDINATE_ATTRIBUTES
    )

    if not use_scene_coordinates:
        coordinate_attributes = [
            attribute_name
            for attribute_name in coordinate_attributes
            if attribute_name != "scene_coord"
        ]

        coordinate_attributes.insert(0, "coord")

    for attribute_name in coordinate_attributes:
        value = _safe_get_value(
            atom,
            (attribute_name,),
            default=None,
        )

        coordinate = _coordinate_components_from_object(value)

        if coordinate is not None:
            return coordinate

    direct_coordinate = _coordinate_components_from_object(atom)

    if direct_coordinate is not None:
        return direct_coordinate

    if strict:
        atom_name = get_atom_name(
            atom,
            default="<unknown>",
        )

        raise PiCoordinateError(
            f"Could not obtain valid coordinates for atom "
            f"{atom_name!r}."
        )

    return None


def require_atom_coordinate(
    atom: Any,
    *,
    use_scene_coordinates: bool = True,
) -> Coordinate3D:
    """
    Return an atom coordinate or raise ``PiCoordinateError``.
    """

    coordinate = get_atom_coordinate(
        atom,
        use_scene_coordinates=use_scene_coordinates,
        strict=True,
    )

    assert coordinate is not None

    return coordinate


def get_atom_coordinates(
    atoms: Iterable[Any],
    *,
    use_scene_coordinates: bool = True,
    skip_invalid: bool = False,
) -> Tuple[Coordinate3D, ...]:
    """
    Return coordinates from an atom collection.
    """

    coordinates: List[Coordinate3D] = []

    for atom in atoms:
        coordinate = get_atom_coordinate(
            atom,
            use_scene_coordinates=use_scene_coordinates,
            strict=not skip_invalid,
        )

        if coordinate is not None:
            coordinates.append(coordinate)

    return tuple(coordinates)


def atom_has_valid_coordinate(
    atom: Any,
    *,
    use_scene_coordinates: bool = True,
) -> bool:
    """
    Return whether an atom provides a valid three-dimensional coordinate.
    """

    return (
        get_atom_coordinate(
            atom,
            use_scene_coordinates=use_scene_coordinates,
        )
        is not None
    )


# -----------------------------------------------------------------------------
# 3.6. Acesso ao resíduo de um átomo
# -----------------------------------------------------------------------------

def get_atom_residue(
    atom: Any,
) -> Optional[Any]:
    """
    Return the residue associated with an atom.
    """

    return _safe_get_value(
        atom,
        ATOM_RESIDUE_ATTRIBUTES,
        default=None,
    )


def get_residue_name(
    residue_or_atom: Any,
    *,
    default: str = "",
) -> str:
    """
    Return a normalized residue name.

    Both residue-like and atom-like inputs are accepted.
    """

    if residue_or_atom is None:
        return default

    residue = get_atom_residue(residue_or_atom)

    if residue is None:
        residue = residue_or_atom

    value = _safe_get_value(
        residue,
        RESIDUE_NAME_ATTRIBUTES,
        default=None,
    )

    return _normalize_text(
        value,
        default=default,
        uppercase=True,
    )


def get_residue_number(
    residue_or_atom: Any,
) -> Optional[Union[int, str]]:
    """
    Return the residue number or identifier.
    """

    if residue_or_atom is None:
        return None

    residue = get_atom_residue(residue_or_atom)

    if residue is None:
        residue = residue_or_atom

    value = _safe_get_value(
        residue,
        RESIDUE_NUMBER_ATTRIBUTES,
        default=None,
    )

    if value is None:
        return None

    integer_value = _normalize_optional_integer(value)

    if integer_value is not None:
        return integer_value

    text = _normalize_optional_text(value)

    return text


def _normalize_chain_value(
    value: Any,
) -> Optional[str]:
    """
    Normalize chain-like values, including ChimeraX chain objects.
    """

    if value is None:
        return None

    if not isinstance(value, (str, int)):
        nested_value = _safe_get_value(
            value,
            (
                "chain_id",
                "id",
                "name",
            ),
            default=None,
        )

        if nested_value is not None and nested_value is not value:
            value = nested_value

    return _normalize_optional_text(value)


def get_residue_chain_id(
    residue_or_atom: Any,
) -> Optional[str]:
    """
    Return the chain identifier of a residue or atom.
    """

    if residue_or_atom is None:
        return None

    residue = get_atom_residue(residue_or_atom)

    if residue is None:
        residue = residue_or_atom

    value = _safe_get_value(
        residue,
        RESIDUE_CHAIN_ATTRIBUTES,
        default=None,
    )

    chain_id = _normalize_chain_value(value)

    if chain_id is not None:
        return chain_id

    atom_chain = _safe_get_value(
        residue_or_atom,
        (
            "chain_id",
            "chain",
        ),
        default=None,
    )

    return _normalize_chain_value(atom_chain)


def get_residue_insertion_code(
    residue_or_atom: Any,
) -> Optional[str]:
    """
    Return a PDB insertion code when available.
    """

    if residue_or_atom is None:
        return None

    residue = get_atom_residue(residue_or_atom)

    if residue is None:
        residue = residue_or_atom

    value = _safe_get_value(
        residue,
        (
            "insertion_code",
            "icode",
            "insert",
        ),
        default=None,
    )

    return _normalize_optional_text(value)


def get_residue_identifier(
    residue_or_atom: Any,
    *,
    include_model: bool = False,
    default: str = "unknown",
) -> str:
    """
    Build a stable residue identifier.

    The default format is::

        chain:residue_name:residue_number

    When ``include_model`` is true, the model identifier is added first.
    """

    residue_name = get_residue_name(
        residue_or_atom,
        default="UNK",
    )

    residue_number = get_residue_number(residue_or_atom)
    chain_id = get_residue_chain_id(residue_or_atom)
    insertion_code = get_residue_insertion_code(
        residue_or_atom
    )

    number_text = (
        str(residue_number)
        if residue_number is not None
        else "?"
    )

    if insertion_code:
        number_text = f"{number_text}{insertion_code}"

    parts = [
        chain_id or "?",
        residue_name,
        number_text,
    ]

    if include_model:
        model_id = get_model_identifier(
            get_atom_model(residue_or_atom),
        )

        parts.insert(0, model_id or "?")

    identifier = ":".join(parts)

    return identifier or default


# -----------------------------------------------------------------------------
# 3.7. Acesso a modelos, resíduos e coleções de átomos
# -----------------------------------------------------------------------------

def get_model_identifier(
    model: Any,
    *,
    default: Optional[str] = None,
) -> Optional[str]:
    """
    Return a normalized model identifier.
    """

    if model is None:
        return default

    value = _safe_get_value(
        model,
        MODEL_IDENTIFIER_ATTRIBUTES,
        default=None,
    )

    if isinstance(value, (tuple, list)):
        value = ".".join(
            str(component)
            for component in value
        )

    normalized = _normalize_optional_text(value)

    return normalized if normalized is not None else default


def get_residue_atoms(
    residue: Any,
    *,
    include_hydrogens: bool = True,
    valid_coordinates_only: bool = False,
) -> Tuple[Any, ...]:
    """
    Return atoms associated with a residue.
    """

    if residue is None:
        return ()

    value = _safe_get_value(
        residue,
        RESIDUE_ATOM_ATTRIBUTES,
        default=None,
    )

    atoms = _as_sequence(value)

    if not atoms:
        return ()

    filtered: List[Any] = []

    for atom in atoms:
        if not include_hydrogens and is_hydrogen_atom(atom):
            continue

        if (
            valid_coordinates_only
            and not atom_has_valid_coordinate(atom)
        ):
            continue

        filtered.append(atom)

    return tuple(filtered)


def get_model_atoms(
    model: Any,
    *,
    include_hydrogens: bool = True,
    valid_coordinates_only: bool = False,
) -> Tuple[Any, ...]:
    """
    Return atoms associated with a molecular model.
    """

    if model is None:
        return ()

    value = _safe_get_value(
        model,
        MODEL_ATOM_ATTRIBUTES,
        default=None,
    )

    atoms = _as_sequence(value)

    if not atoms:
        residues = get_model_residues(model)

        atoms = tuple(
            atom
            for residue in residues
            for atom in get_residue_atoms(residue)
        )

    filtered: List[Any] = []

    for atom in atoms:
        if not include_hydrogens and is_hydrogen_atom(atom):
            continue

        if (
            valid_coordinates_only
            and not atom_has_valid_coordinate(atom)
        ):
            continue

        filtered.append(atom)

    return tuple(filtered)


def get_model_residues(
    model: Any,
) -> Tuple[Any, ...]:
    """
    Return residues associated with a molecular model.
    """

    if model is None:
        return ()

    value = _safe_get_value(
        model,
        MODEL_RESIDUE_ATTRIBUTES,
        default=None,
    )

    residues = _as_sequence(value)

    if residues:
        return residues

    atoms = _as_sequence(
        _safe_get_value(
            model,
            MODEL_ATOM_ATTRIBUTES,
            default=None,
        )
    )

    unique_residues: List[Any] = []
    seen_ids: Set[int] = set()

    for atom in atoms:
        residue = get_atom_residue(atom)

        if residue is None:
            continue

        residue_object_id = id(residue)

        if residue_object_id in seen_ids:
            continue

        seen_ids.add(residue_object_id)
        unique_residues.append(residue)

    return tuple(unique_residues)


def normalize_atom_collection(
    value: Any,
    *,
    include_hydrogens: bool = True,
    valid_coordinates_only: bool = False,
    deduplicate: bool = True,
) -> Tuple[Any, ...]:
    """
    Normalize an atom, residue, model or atom iterable into an atom tuple.
    """

    if value is None:
        return ()

    direct_atom_name = get_atom_name(value)

    if direct_atom_name:
        candidates = (value,)

    else:
        model_atoms = get_model_atoms(
            value,
            include_hydrogens=include_hydrogens,
            valid_coordinates_only=valid_coordinates_only,
        )

        if model_atoms:
            candidates = model_atoms

        else:
            residue_atoms = get_residue_atoms(
                value,
                include_hydrogens=include_hydrogens,
                valid_coordinates_only=valid_coordinates_only,
            )

            if residue_atoms:
                candidates = residue_atoms

            else:
                candidates = _as_sequence(value)

    normalized: List[Any] = []
    seen_ids: Set[int] = set()

    for atom in candidates:
        if atom is None:
            continue

        if not include_hydrogens and is_hydrogen_atom(atom):
            continue

        if (
            valid_coordinates_only
            and not atom_has_valid_coordinate(atom)
        ):
            continue

        if deduplicate:
            object_id = id(atom)

            if object_id in seen_ids:
                continue

            seen_ids.add(object_id)

        normalized.append(atom)

    return tuple(normalized)


def normalize_residue_collection(
    value: Any,
    *,
    deduplicate: bool = True,
) -> Tuple[Any, ...]:
    """
    Normalize a residue, model, atom or iterable into a residue tuple.
    """

    if value is None:
        return ()

    atom_residue = get_atom_residue(value)

    if atom_residue is not None:
        candidates = (atom_residue,)

    else:
        model_residues = get_model_residues(value)

        if model_residues:
            candidates = model_residues

        elif get_residue_name(value):
            candidates = (value,)

        else:
            raw_values = _as_sequence(value)
            candidates_list: List[Any] = []

            for item in raw_values:
                residue = get_atom_residue(item)

                if residue is not None:
                    candidates_list.append(residue)

                elif get_residue_name(item):
                    candidates_list.append(item)

            candidates = tuple(candidates_list)

    if not deduplicate:
        return tuple(candidates)

    normalized: List[Any] = []
    seen_ids: Set[int] = set()

    for residue in candidates:
        object_id = id(residue)

        if object_id in seen_ids:
            continue

        seen_ids.add(object_id)
        normalized.append(residue)

    return tuple(normalized)


# -----------------------------------------------------------------------------
# 3.8. Cargas atômicas
# -----------------------------------------------------------------------------

def get_atom_formal_charge(
    atom: Any,
) -> Optional[float]:
    """
    Return the formal charge of an atom when available.
    """

    value = _safe_get_value(
        atom,
        (
            "formal_charge",
            "formalCharge",
        ),
        default=None,
    )

    return _normalize_optional_numeric(value)


def get_atom_partial_charge(
    atom: Any,
) -> Optional[float]:
    """
    Return the partial charge of an atom when available.
    """

    value = _safe_get_value(
        atom,
        (
            "partial_charge",
            "charge",
            "partialCharge",
            "gasteiger_charge",
        ),
        default=None,
    )

    return _normalize_optional_numeric(value)


def get_atom_effective_charge(
    atom: Any,
    *,
    prefer_formal: bool = True,
) -> Optional[float]:
    """
    Return the best available atomic charge.
    """

    formal_charge = get_atom_formal_charge(atom)
    partial_charge = get_atom_partial_charge(atom)

    if prefer_formal and formal_charge is not None:
        return formal_charge

    if partial_charge is not None:
        return partial_charge

    return formal_charge


def atom_has_positive_charge(
    atom: Any,
    *,
    partial_charge_threshold: float = (
        DEFAULT_POSITIVE_PARTIAL_CHARGE_THRESHOLD
    ),
    infer_from_type: bool = True,
) -> bool:
    """
    Return whether an atom is positively charged.
    """

    formal_charge = get_atom_formal_charge(atom)

    if formal_charge is not None and formal_charge > 0.0:
        return True

    partial_charge = get_atom_partial_charge(atom)

    if (
        partial_charge is not None
        and partial_charge >= partial_charge_threshold
    ):
        return True

    if infer_from_type:
        atom_type = _normalize_text(
            get_atom_type(atom),
            uppercase=True,
        )

        if atom_type in {
            value.upper()
            for value in POSITIVE_NITROGEN_TYPE_HINTS
        }:
            return True

    return False


def atom_has_negative_charge(
    atom: Any,
    *,
    partial_charge_threshold: float = (
        DEFAULT_NEGATIVE_PARTIAL_CHARGE_THRESHOLD
    ),
    infer_from_type: bool = True,
) -> bool:
    """
    Return whether an atom is negatively charged.
    """

    formal_charge = get_atom_formal_charge(atom)

    if formal_charge is not None and formal_charge < 0.0:
        return True

    partial_charge = get_atom_partial_charge(atom)

    if (
        partial_charge is not None
        and partial_charge <= partial_charge_threshold
    ):
        return True

    if infer_from_type:
        atom_type = _normalize_text(
            get_atom_type(atom),
            uppercase=True,
        )

        if atom_type in {
            value.upper()
            for value in NEGATIVE_OXYGEN_TYPE_HINTS
        }:
            return True

    return False


# -----------------------------------------------------------------------------
# 3.9. Aromaticidade e classificação atômica
# -----------------------------------------------------------------------------

def is_hydrogen_atom(
    atom: Any,
) -> bool:
    """
    Return whether an atom represents hydrogen or one of its isotopes.
    """

    return get_atom_element(atom) in HYDROGEN_ELEMENTS


def is_heavy_atom(
    atom: Any,
) -> bool:
    """
    Return whether an atom is not hydrogen.
    """

    element = get_atom_element(atom)

    return bool(element) and element not in HYDROGEN_ELEMENTS


def is_carbon_atom(
    atom: Any,
) -> bool:
    """
    Return whether an atom is carbon.
    """

    return get_atom_element(atom) in CARBON_ELEMENTS


def is_nitrogen_atom(
    atom: Any,
) -> bool:
    """
    Return whether an atom is nitrogen.
    """

    return get_atom_element(atom) in NITROGEN_ELEMENTS


def is_oxygen_atom(
    atom: Any,
) -> bool:
    """
    Return whether an atom is oxygen.
    """

    return get_atom_element(atom) in OXYGEN_ELEMENTS


def is_sulfur_atom(
    atom: Any,
) -> bool:
    """
    Return whether an atom is sulfur.
    """

    return get_atom_element(atom) in SULFUR_ELEMENTS


def is_halogen_atom(
    atom: Any,
) -> bool:
    """
    Return whether an atom is a common halogen.
    """

    return get_atom_element(atom) in HALOGEN_ELEMENTS


def get_atom_aromatic_flag(
    atom: Any,
) -> Optional[bool]:
    """
    Return an explicit aromaticity flag when available.
    """

    value = _safe_get_value(
        atom,
        ATOM_AROMATIC_ATTRIBUTES,
        default=None,
    )

    if value is None:
        return None

    if isinstance(value, str):
        normalized = value.strip().lower()

        if normalized in {
            "true",
            "yes",
            "1",
            "aromatic",
            "ar",
        }:
            return True

        if normalized in {
            "false",
            "no",
            "0",
            "nonaromatic",
            "non-aromatic",
        }:
            return False

    return bool(value)


def is_aromatic_atom(
    atom: Any,
    *,
    infer_from_residue: bool = True,
    infer_from_atom_type: bool = True,
) -> bool:
    """
    Return whether an atom should be considered aromatic.
    """

    explicit_flag = get_atom_aromatic_flag(atom)

    if explicit_flag is not None:
        return explicit_flag

    if infer_from_atom_type:
        atom_type = _normalize_text(
            get_atom_type(atom),
            uppercase=True,
        )

        aromatic_hints = {
            value.upper()
            for value in AROMATIC_ATOM_TYPE_HINTS
        }

        if atom_type in aromatic_hints:
            return True

        if ".AR" in atom_type or atom_type.endswith("AR"):
            return True

    if infer_from_residue:
        residue_name = get_residue_name(atom)
        atom_name = get_atom_name(atom).upper()

        ring_definitions = STANDARD_AROMATIC_RING_ATOMS.get(
            residue_name,
            (),
        )

        for ring_atom_names in ring_definitions:
            if atom_name in ring_atom_names:
                return True

    return False


# -----------------------------------------------------------------------------
# 3.10. Conectividade e ligações químicas
# -----------------------------------------------------------------------------

def get_atom_bonds(
    atom: Any,
) -> Tuple[Any, ...]:
    """
    Return bond-like objects connected to an atom.
    """

    if atom is None:
        return ()

    value = _safe_get_value(
        atom,
        ("bonds",),
        default=None,
    )

    return _as_sequence(value)


def _get_other_bond_atom(
    bond: Any,
    atom: Any,
) -> Optional[Any]:
    """
    Return the atom located at the opposite end of a bond.
    """

    if bond is None:
        return None

    other_atom_method = _safe_get_attribute(
        bond,
        "other_atom",
        default=None,
        call_if_callable=False,
    )

    if callable(other_atom_method):
        try:
            return other_atom_method(atom)

        except Exception:
            pass

    atoms = _safe_get_value(
        bond,
        (
            "atoms",
            "atom_pair",
            "endpoints",
        ),
        default=None,
    )

    atom_sequence = _as_sequence(atoms)

    if len(atom_sequence) >= 2:
        if atom_sequence[0] is atom:
            return atom_sequence[1]

        if atom_sequence[1] is atom:
            return atom_sequence[0]

        for candidate in atom_sequence:
            if candidate is not atom:
                return candidate

    atom_1 = _safe_get_value(
        bond,
        (
            "atom1",
            "atom_1",
            "begin_atom",
        ),
        default=None,
    )

    atom_2 = _safe_get_value(
        bond,
        (
            "atom2",
            "atom_2",
            "end_atom",
        ),
        default=None,
    )

    if atom_1 is atom:
        return atom_2

    if atom_2 is atom:
        return atom_1

    return None


def get_bonded_atoms(
    atom: Any,
    *,
    include_hydrogens: bool = True,
    deduplicate: bool = True,
) -> Tuple[Any, ...]:
    """
    Return atoms directly bonded to an atom.
    """

    if atom is None:
        return ()

    direct_neighbors = _safe_get_value(
        atom,
        (
            "neighbors",
            "bonded_atoms",
        ),
        default=None,
    )

    neighbors = list(_as_sequence(direct_neighbors))

    if not neighbors:
        for bond in get_atom_bonds(atom):
            other_atom = _get_other_bond_atom(
                bond,
                atom,
            )

            if other_atom is not None:
                neighbors.append(other_atom)

    filtered: List[Any] = []
    seen_ids: Set[int] = set()

    for neighbor in neighbors:
        if neighbor is None or neighbor is atom:
            continue

        if not include_hydrogens and is_hydrogen_atom(neighbor):
            continue

        if deduplicate:
            object_id = id(neighbor)

            if object_id in seen_ids:
                continue

            seen_ids.add(object_id)

        filtered.append(neighbor)

    return tuple(filtered)


def get_bond_between_atoms(
    atom_1: Any,
    atom_2: Any,
) -> Optional[Any]:
    """
    Return the bond connecting two atoms, when available.
    """

    if atom_1 is None or atom_2 is None:
        return None

    for bond in get_atom_bonds(atom_1):
        other_atom = _get_other_bond_atom(
            bond,
            atom_1,
        )

        if other_atom is atom_2:
            return bond

    return None


def atoms_are_bonded(
    atom_1: Any,
    atom_2: Any,
) -> bool:
    """
    Return whether two atoms are directly bonded.
    """

    return get_bond_between_atoms(atom_1, atom_2) is not None or any(
        neighbor is atom_2
        for neighbor in get_bonded_atoms(atom_1)
    )


def get_bond_order(
    bond_or_atom_1: Any,
    atom_2: Optional[Any] = None,
) -> Optional[float]:
    """
    Return a normalized bond order.

    The function accepts either a bond object or two atoms.
    """

    if atom_2 is not None:
        bond = get_bond_between_atoms(
            bond_or_atom_1,
            atom_2,
        )

    else:
        bond = bond_or_atom_1

    if bond is None:
        return None

    value = _safe_get_value(
        bond,
        (
            "order",
            "bond_order",
            "bondOrder",
        ),
        default=None,
    )

    if isinstance(value, str):
        normalized = value.strip().upper()

        symbolic_orders = {
            "SINGLE": 1.0,
            "S": 1.0,
            "DOUBLE": 2.0,
            "D": 2.0,
            "TRIPLE": 3.0,
            "T": 3.0,
            "AROMATIC": 1.5,
            "AR": 1.5,
            "ARO": 1.5,
            ":": 1.5,
        }

        if normalized in symbolic_orders:
            return symbolic_orders[normalized]

    return _normalize_optional_numeric(value)


def get_bond_type(
    bond_or_atom_1: Any,
    atom_2: Optional[Any] = None,
) -> Optional[str]:
    """
    Return a normalized bond-type label.
    """

    if atom_2 is not None:
        bond = get_bond_between_atoms(
            bond_or_atom_1,
            atom_2,
        )

    else:
        bond = bond_or_atom_1

    if bond is None:
        return None

    value = _safe_get_value(
        bond,
        (
            "type",
            "bond_type",
            "name",
        ),
        default=None,
    )

    return _normalize_optional_text(value)


def is_aromatic_bond(
    bond_or_atom_1: Any,
    atom_2: Optional[Any] = None,
) -> bool:
    """
    Return whether a bond should be considered aromatic.
    """

    if atom_2 is not None:
        bond = get_bond_between_atoms(
            bond_or_atom_1,
            atom_2,
        )

    else:
        bond = bond_or_atom_1

    if bond is None:
        return False

    explicit_flag = _safe_get_value(
        bond,
        (
            "is_aromatic",
            "aromatic",
        ),
        default=None,
    )

    if explicit_flag is not None:
        return bool(explicit_flag)

    bond_order = get_bond_order(bond)

    if bond_order is not None and abs(bond_order - 1.5) <= 0.1:
        return True

    bond_type = _normalize_text(
        get_bond_type(bond),
        uppercase=True,
    )

    return bond_type in {
        value.upper()
        for value in AROMATIC_BOND_TYPE_HINTS
    }


# -----------------------------------------------------------------------------
# 3.11. Seleção e busca de átomos
# -----------------------------------------------------------------------------

def find_atom_by_name(
    atoms_or_residue: Any,
    atom_name: str,
    *,
    case_sensitive: bool = False,
) -> Optional[Any]:
    """
    Return the first atom matching a name.
    """

    target_name = _normalize_text(atom_name)

    if not target_name:
        return None

    atoms = normalize_atom_collection(
        atoms_or_residue,
        deduplicate=False,
    )

    if not case_sensitive:
        target_name = target_name.upper()

    for atom in atoms:
        candidate_name = get_atom_name(atom)

        if not case_sensitive:
            candidate_name = candidate_name.upper()

        if candidate_name == target_name:
            return atom

    return None


def find_atoms_by_names(
    atoms_or_residue: Any,
    atom_names: Iterable[str],
    *,
    case_sensitive: bool = False,
    preserve_requested_order: bool = True,
) -> Tuple[Any, ...]:
    """
    Return atoms matching a collection of names.
    """

    requested_names = _coerce_string_tuple(
        atom_names,
        field_name="atom_names",
    )

    atoms = normalize_atom_collection(
        atoms_or_residue,
        deduplicate=False,
    )

    if preserve_requested_order:
        matched: List[Any] = []

        for atom_name in requested_names:
            atom = find_atom_by_name(
                atoms,
                atom_name,
                case_sensitive=case_sensitive,
            )

            if atom is not None:
                matched.append(atom)

        return tuple(matched)

    if case_sensitive:
        target_names = set(requested_names)

    else:
        target_names = {
            atom_name.upper()
            for atom_name in requested_names
        }

    return tuple(
        atom
        for atom in atoms
        if (
            get_atom_name(atom)
            if case_sensitive
            else get_atom_name(atom).upper()
        )
        in target_names
    )


def map_atoms_by_name(
    atoms_or_residue: Any,
    *,
    uppercase_keys: bool = True,
) -> Dict[str, Any]:
    """
    Return a dictionary mapping atom names to atom objects.

    The first atom with each name is retained.
    """

    result: Dict[str, Any] = {}

    for atom in normalize_atom_collection(
        atoms_or_residue,
        deduplicate=False,
    ):
        atom_name = get_atom_name(atom)

        if not atom_name:
            continue

        key = (
            atom_name.upper()
            if uppercase_keys
            else atom_name
        )

        result.setdefault(key, atom)

    return result


def filter_atoms_by_element(
    atoms: Iterable[Any],
    elements: Collection[str],
) -> Tuple[Any, ...]:
    """
    Return atoms whose element belongs to ``elements``.
    """

    normalized_elements = {
        normalize_element_symbol(element)
        for element in elements
    }

    return tuple(
        atom
        for atom in atoms
        if get_atom_element(atom) in normalized_elements
    )


def filter_heavy_atoms(
    atoms: Iterable[Any],
) -> Tuple[Any, ...]:
    """
    Return only non-hydrogen atoms.
    """

    return tuple(
        atom
        for atom in atoms
        if is_heavy_atom(atom)
    )


def filter_atoms_with_coordinates(
    atoms: Iterable[Any],
) -> Tuple[Any, ...]:
    """
    Return only atoms providing valid coordinates.
    """

    return tuple(
        atom
        for atom in atoms
        if atom_has_valid_coordinate(atom)
    )


# -----------------------------------------------------------------------------
# 3.12. Construção de referências atômicas serializáveis
# -----------------------------------------------------------------------------

def create_pi_atom_reference(
    atom: Any,
    *,
    use_scene_coordinates: bool = True,
    include_metadata: bool = False,
    strict: bool = False,
) -> PiAtomReference:
    """
    Create a normalized ``PiAtomReference`` from an atom-like object.
    """

    if atom is None:
        raise PiAtomAccessError(
            "Cannot create a PiAtomReference from None."
        )

    atom_name = get_atom_name(atom)

    if not atom_name:
        if strict:
            raise PiAtomAccessError(
                "The atom does not provide a valid name."
            )

        atom_name = "?"

    residue = get_atom_residue(atom)
    model = get_atom_model(atom)

    coordinate = get_atom_coordinate(
        atom,
        use_scene_coordinates=use_scene_coordinates,
        strict=strict,
    )

    aromatic_flag = get_atom_aromatic_flag(atom)

    metadata: Dict[str, Any] = {}

    if include_metadata:
        metadata.update(
            {
                "source_class": (
                    f"{atom.__class__.__module__}."
                    f"{atom.__class__.__name__}"
                ),
                "residue_identifier": get_residue_identifier(
                    residue or atom
                ),
            }
        )

    return PiAtomReference(
        name=atom_name,
        element=get_atom_element(atom),
        atom_index=get_atom_index(atom),
        serial_number=get_atom_serial_number(atom),
        residue_name=get_residue_name(
            residue or atom,
            default=None,  # type: ignore[arg-type]
        ),
        residue_number=get_residue_number(residue or atom),
        chain_id=get_residue_chain_id(residue or atom),
        model_id=get_model_identifier(model),
        coordinate=coordinate,
        formal_charge=get_atom_formal_charge(atom),
        partial_charge=get_atom_partial_charge(atom),
        atom_type=get_atom_type(atom),
        is_aromatic=aromatic_flag,
        metadata=metadata,
    )


def create_pi_atom_references(
    atoms: Iterable[Any],
    *,
    use_scene_coordinates: bool = True,
    include_metadata: bool = False,
    skip_invalid: bool = False,
) -> Tuple[PiAtomReference, ...]:
    """
    Create normalized references for multiple atoms.
    """

    references: List[PiAtomReference] = []

    for atom in atoms:
        try:
            reference = create_pi_atom_reference(
                atom,
                use_scene_coordinates=use_scene_coordinates,
                include_metadata=include_metadata,
                strict=not skip_invalid,
            )

        except (
            PiNormalizationError,
            TypeError,
            ValueError,
        ):
            if skip_invalid:
                continue

            raise

        references.append(reference)

    return tuple(references)


# -----------------------------------------------------------------------------
# 3.13. Identificação de participantes moleculares
# -----------------------------------------------------------------------------

_STANDARD_AMINO_ACIDS: Final[FrozenSet[str]] = frozenset(
    {
        "ALA",
        "ARG",
        "ASN",
        "ASP",
        "CYS",
        "CYM",
        "CYX",
        "GLN",
        "GLU",
        "GLY",
        "HIS",
        "HID",
        "HIE",
        "HIP",
        "HSD",
        "HSE",
        "HSP",
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


_COMMON_SOLVENT_RESIDUES: Final[FrozenSet[str]] = frozenset(
    {
        "HOH",
        "WAT",
        "H2O",
        "SOL",
        "TIP3",
        "TIP3P",
        "SPC",
        "SPCE",
    }
)


_COMMON_ION_RESIDUES: Final[FrozenSet[str]] = frozenset(
    {
        "NA",
        "CL",
        "K",
        "CA",
        "MG",
        "ZN",
        "FE",
        "MN",
        "CU",
        "CO",
    }
)


def is_standard_amino_acid_residue(
    residue_or_atom: Any,
) -> bool:
    """
    Return whether a residue is a standard or common protonation variant.
    """

    return get_residue_name(
        residue_or_atom
    ) in _STANDARD_AMINO_ACIDS


def is_nucleic_acid_residue(
    residue_or_atom: Any,
) -> bool:
    """
    Return whether a residue is a recognized nucleic-acid residue.
    """

    return get_residue_name(
        residue_or_atom
    ) in NUCLEIC_ACID_AROMATIC_RESIDUES


def is_solvent_residue(
    residue_or_atom: Any,
) -> bool:
    """
    Return whether a residue is a common explicit solvent molecule.
    """

    return get_residue_name(
        residue_or_atom
    ) in _COMMON_SOLVENT_RESIDUES


def is_ion_residue(
    residue_or_atom: Any,
) -> bool:
    """
    Return whether a residue is a common monoatomic ion.
    """

    return get_residue_name(
        residue_or_atom
    ) in _COMMON_ION_RESIDUES


def infer_participant_type(
    residue_or_atom: Any,
    *,
    ligand_residue_names: Optional[Collection[str]] = None,
    receptor_residue_names: Optional[Collection[str]] = None,
    default: str = PARTICIPANT_UNKNOWN,
) -> str:
    """
    Infer whether an atom or residue belongs to protein, nucleic acid, ligand,
    receptor, cofactor or another participant class.
    """

    residue_name = get_residue_name(residue_or_atom)

    normalized_ligand_names = {
        str(name).strip().upper()
        for name in (ligand_residue_names or ())
        if str(name).strip()
    }

    normalized_receptor_names = {
        str(name).strip().upper()
        for name in (receptor_residue_names or ())
        if str(name).strip()
    }

    if residue_name in normalized_ligand_names:
        return PARTICIPANT_LIGAND

    if residue_name in normalized_receptor_names:
        return PARTICIPANT_RECEPTOR

    explicit_value = _safe_get_value(
        residue_or_atom,
        (
            "participant_type",
            "role",
            "molecular_role",
        ),
        default=None,
    )

    explicit_text = _normalize_text(
        explicit_value,
        uppercase=False,
    ).lower()

    valid_explicit_types = {
        PARTICIPANT_RECEPTOR,
        PARTICIPANT_LIGAND,
        PARTICIPANT_PROTEIN,
        PARTICIPANT_NUCLEIC_ACID,
        PARTICIPANT_COFACTOR,
        PARTICIPANT_UNKNOWN,
    }

    if explicit_text in valid_explicit_types:
        return explicit_text

    if is_standard_amino_acid_residue(residue_or_atom):
        return PARTICIPANT_PROTEIN

    if is_nucleic_acid_residue(residue_or_atom):
        return PARTICIPANT_NUCLEIC_ACID

    if is_solvent_residue(residue_or_atom):
        return PARTICIPANT_UNKNOWN

    if is_ion_residue(residue_or_atom):
        return PARTICIPANT_COFACTOR

    if residue_name:
        return PARTICIPANT_LIGAND

    return default


# -----------------------------------------------------------------------------
# 3.14. Identificadores e chaves estáveis
# -----------------------------------------------------------------------------

def get_atom_identifier(
    atom: Any,
    *,
    include_model: bool = True,
) -> str:
    """
    Build a stable human-readable atom identifier.
    """

    residue_identifier = get_residue_identifier(
        atom,
        include_model=include_model,
    )

    atom_name = get_atom_name(
        atom,
        default="?",
    )

    serial_number = get_atom_serial_number(atom)
    atom_index = get_atom_index(atom)

    if serial_number is not None:
        atom_suffix = f"{atom_name}#{serial_number}"

    elif atom_index is not None:
        atom_suffix = f"{atom_name}@{atom_index}"

    else:
        atom_suffix = atom_name

    return f"{residue_identifier}:{atom_suffix}"


def get_atom_identity_key(
    atom: Any,
) -> Tuple[Any, ...]:
    """
    Return a hashable identity key suitable for deduplication.

    Native object identity is retained as the final component to prevent
    accidental merging of atoms that share identical labels.
    """

    return (
        get_model_identifier(get_atom_model(atom)),
        get_residue_chain_id(atom),
        get_residue_name(atom),
        get_residue_number(atom),
        get_residue_insertion_code(atom),
        get_atom_name(atom).upper(),
        get_atom_serial_number(atom),
        get_atom_index(atom),
        id(atom),
    )


def get_residue_identity_key(
    residue_or_atom: Any,
) -> Tuple[Any, ...]:
    """
    Return a hashable residue identity key.
    """

    residue = get_atom_residue(residue_or_atom)

    if residue is None:
        residue = residue_or_atom

    model = _safe_get_value(
        residue,
        (
            "structure",
            "model",
            "molecule",
        ),
        default=None,
    )

    return (
        get_model_identifier(model),
        get_residue_chain_id(residue),
        get_residue_name(residue),
        get_residue_number(residue),
        get_residue_insertion_code(residue),
        id(residue),
    )


def deduplicate_atoms(
    atoms: Iterable[Any],
) -> Tuple[Any, ...]:
    """
    Remove repeated atom objects while preserving order.
    """

    result: List[Any] = []
    seen: Set[Tuple[Any, ...]] = set()

    for atom in atoms:
        key = get_atom_identity_key(atom)

        if key in seen:
            continue

        seen.add(key)
        result.append(atom)

    return tuple(result)


def deduplicate_residues(
    residues: Iterable[Any],
) -> Tuple[Any, ...]:
    """
    Remove repeated residue objects while preserving order.
    """

    result: List[Any] = []
    seen: Set[Tuple[Any, ...]] = set()

    for residue in residues:
        key = get_residue_identity_key(residue)

        if key in seen:
            continue

        seen.add(key)
        result.append(residue)

    return tuple(result)


# -----------------------------------------------------------------------------
# 3.15. Validação de coleções moleculares
# -----------------------------------------------------------------------------

def validate_atom(
    atom: Any,
    *,
    require_name: bool = True,
    require_element: bool = True,
    require_coordinate: bool = True,
) -> Tuple[bool, Tuple[str, ...]]:
    """
    Validate the minimum information required from an atom.
    """

    messages: List[str] = []

    if atom is None:
        return False, ("Atom is None.",)

    if require_name and not get_atom_name(atom):
        messages.append(
            "Atom name is unavailable."
        )

    if require_element and not get_atom_element(atom):
        messages.append(
            "Atom element is unavailable."
        )

    if (
        require_coordinate
        and get_atom_coordinate(atom) is None
    ):
        messages.append(
            "Atom coordinates are unavailable or invalid."
        )

    return not messages, tuple(messages)


def validate_atom_collection(
    atoms: Iterable[Any],
    *,
    minimum_atoms: int = 1,
    require_coordinates: bool = True,
    skip_hydrogens: bool = False,
) -> Tuple[bool, Tuple[str, ...]]:
    """
    Validate an atom collection.
    """

    normalized_atoms = tuple(atoms)
    messages: List[str] = []

    if len(normalized_atoms) < minimum_atoms:
        messages.append(
            f"At least {minimum_atoms} atoms are required; "
            f"received {len(normalized_atoms)}."
        )

    for index, atom in enumerate(normalized_atoms):
        if skip_hydrogens and is_hydrogen_atom(atom):
            continue

        valid, atom_messages = validate_atom(
            atom,
            require_coordinate=require_coordinates,
        )

        if valid:
            continue

        atom_name = get_atom_name(
            atom,
            default=f"index_{index}",
        )

        for message in atom_messages:
            messages.append(
                f"{atom_name}: {message}"
            )

    return not messages, tuple(messages)


def require_valid_atom_collection(
    atoms: Iterable[Any],
    *,
    minimum_atoms: int = 1,
    require_coordinates: bool = True,
    context: str = "atom collection",
) -> Tuple[Any, ...]:
    """
    Validate an atom collection or raise ``PiAtomAccessError``.
    """

    normalized_atoms = tuple(atoms)

    valid, messages = validate_atom_collection(
        normalized_atoms,
        minimum_atoms=minimum_atoms,
        require_coordinates=require_coordinates,
    )

    if not valid:
        formatted_messages = "; ".join(messages)

        raise PiAtomAccessError(
            f"Invalid {context}: {formatted_messages}"
        )

    return normalized_atoms


# -----------------------------------------------------------------------------
# 3.16. Contexto normalizado de análise
# -----------------------------------------------------------------------------

@dataclass(slots=True)
class PiNormalizedInput:
    """
    Normalized receptor and ligand input used by the analysis pipeline.
    """

    receptor_atoms: Tuple[Any, ...] = ()
    ligand_atoms: Tuple[Any, ...] = ()

    receptor_residues: Tuple[Any, ...] = ()
    ligand_residues: Tuple[Any, ...] = ()

    receptor_model: Optional[Any] = None
    ligand_model: Optional[Any] = None

    receptor_model_id: Optional[str] = None
    ligand_model_id: Optional[str] = None

    pose_id: Optional[str] = None

    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.receptor_atoms = deduplicate_atoms(
            self.receptor_atoms
        )

        self.ligand_atoms = deduplicate_atoms(
            self.ligand_atoms
        )

        self.receptor_residues = deduplicate_residues(
            self.receptor_residues
        )

        self.ligand_residues = deduplicate_residues(
            self.ligand_residues
        )

        self.receptor_model_id = _normalize_optional_text(
            self.receptor_model_id
        )

        self.ligand_model_id = _normalize_optional_text(
            self.ligand_model_id
        )

        self.pose_id = _normalize_optional_text(self.pose_id)

        self.warnings = list(
            _coerce_string_tuple(
                self.warnings,
                field_name="PiNormalizedInput.warnings",
            )
        )

        self.metadata = _copy_mapping(self.metadata)

    @property
    def all_atoms(self) -> Tuple[Any, ...]:
        """
        Return all receptor and ligand atoms.
        """

        return deduplicate_atoms(
            self.receptor_atoms + self.ligand_atoms
        )

    @property
    def all_residues(self) -> Tuple[Any, ...]:
        """
        Return all receptor and ligand residues.
        """

        return deduplicate_residues(
            self.receptor_residues + self.ligand_residues
        )

    @property
    def receptor_atom_count(self) -> int:
        """
        Return the number of receptor atoms.
        """

        return len(self.receptor_atoms)

    @property
    def ligand_atom_count(self) -> int:
        """
        Return the number of ligand atoms.
        """

        return len(self.ligand_atoms)

    @property
    def is_empty(self) -> bool:
        """
        Return whether no molecular atoms were supplied.
        """

        return not self.receptor_atoms and not self.ligand_atoms


def normalize_pi_analysis_input(
    receptor: Any,
    ligand: Any,
    *,
    pose_id: Optional[str] = None,
    include_hydrogens: bool = True,
    valid_coordinates_only: bool = True,
    strict: bool = False,
) -> PiNormalizedInput:
    """
    Normalize receptor and ligand inputs for pi-interaction analysis.

    Parameters
    ----------
    receptor
        Receptor model, residue collection or atom collection.

    ligand
        Ligand model, residue collection or atom collection.

    pose_id
        Optional pose identifier.

    include_hydrogens
        Whether hydrogen atoms should remain in the normalized inputs.
        Ring detection will normally use heavy atoms only, but retaining
        hydrogens may be useful for charge and protonation inference.

    valid_coordinates_only
        Remove atoms without valid coordinates.

    strict
        Raise an exception when either receptor or ligand is empty.
    """

    receptor_atoms = normalize_atom_collection(
        receptor,
        include_hydrogens=include_hydrogens,
        valid_coordinates_only=valid_coordinates_only,
    )

    ligand_atoms = normalize_atom_collection(
        ligand,
        include_hydrogens=include_hydrogens,
        valid_coordinates_only=valid_coordinates_only,
    )

    receptor_residues = normalize_residue_collection(
        receptor_atoms
    )

    ligand_residues = normalize_residue_collection(
        ligand_atoms
    )

    warnings_list: List[str] = []

    if not receptor_atoms:
        message = (
            "No valid receptor atoms were found."
        )

        if strict:
            raise PiAtomAccessError(message)

        warnings_list.append(message)

    if not ligand_atoms:
        message = (
            "No valid ligand atoms were found."
        )

        if strict:
            raise PiAtomAccessError(message)

        warnings_list.append(message)

    receptor_model = (
        get_atom_model(receptor_atoms[0])
        if receptor_atoms
        else receptor
    )

    ligand_model = (
        get_atom_model(ligand_atoms[0])
        if ligand_atoms
        else ligand
    )

    return PiNormalizedInput(
        receptor_atoms=receptor_atoms,
        ligand_atoms=ligand_atoms,
        receptor_residues=receptor_residues,
        ligand_residues=ligand_residues,
        receptor_model=receptor_model,
        ligand_model=ligand_model,
        receptor_model_id=get_model_identifier(
            receptor_model
        ),
        ligand_model_id=get_model_identifier(
            ligand_model
        ),
        pose_id=pose_id,
        warnings=warnings_list,
        metadata={
            "include_hydrogens": include_hydrogens,
            "valid_coordinates_only": valid_coordinates_only,
        },
    )


# -----------------------------------------------------------------------------
# 3.17. Relatório resumido da normalização
# -----------------------------------------------------------------------------

def summarize_normalized_input(
    normalized_input: PiNormalizedInput,
) -> Dict[str, Any]:
    """
    Return a compact serializable summary of normalized molecular input.
    """

    if not isinstance(normalized_input, PiNormalizedInput):
        raise TypeError(
            "normalized_input must be a PiNormalizedInput."
        )

    receptor_elements = Counter(
        get_atom_element(atom)
        for atom in normalized_input.receptor_atoms
        if get_atom_element(atom)
    )

    ligand_elements = Counter(
        get_atom_element(atom)
        for atom in normalized_input.ligand_atoms
        if get_atom_element(atom)
    )

    receptor_residue_names = Counter(
        get_residue_name(residue)
        for residue in normalized_input.receptor_residues
        if get_residue_name(residue)
    )

    ligand_residue_names = Counter(
        get_residue_name(residue)
        for residue in normalized_input.ligand_residues
        if get_residue_name(residue)
    )

    return {
        "pose_id": normalized_input.pose_id,
        "receptor_model_id": normalized_input.receptor_model_id,
        "ligand_model_id": normalized_input.ligand_model_id,
        "receptor_atom_count": (
            normalized_input.receptor_atom_count
        ),
        "ligand_atom_count": (
            normalized_input.ligand_atom_count
        ),
        "receptor_residue_count": len(
            normalized_input.receptor_residues
        ),
        "ligand_residue_count": len(
            normalized_input.ligand_residues
        ),
        "receptor_element_distribution": dict(
            receptor_elements
        ),
        "ligand_element_distribution": dict(
            ligand_elements
        ),
        "receptor_residue_distribution": dict(
            receptor_residue_names
        ),
        "ligand_residue_distribution": dict(
            ligand_residue_names
        ),
        "warnings": list(normalized_input.warnings),
        "metadata": dict(normalized_input.metadata),
    }


# -----------------------------------------------------------------------------
# End of section 3.
# -----------------------------------------------------------------------------



# =============================================================================
# 4. RECONHECIMENTO DE ANÉIS AROMÁTICOS
# =============================================================================

# -----------------------------------------------------------------------------
# 4.1. Tipos auxiliares internos
# -----------------------------------------------------------------------------

RingAtomTuple: TypeAlias = Tuple[Any, ...]

RingIndexTuple: TypeAlias = Tuple[int, ...]

RingGraph: TypeAlias = Dict[int, Set[int]]


# -----------------------------------------------------------------------------
# 4.2. Normalização de ciclos
# -----------------------------------------------------------------------------

def _rotate_sequence_to_smallest(
    values: Sequence[int],
) -> Tuple[int, ...]:
    """
    Rotate a cyclic integer sequence so that its smallest value comes first.
    """

    if not values:
        return ()

    sequence = tuple(values)
    minimum_value = min(sequence)

    candidate_rotations = [
        sequence[index:] + sequence[:index]
        for index, value in enumerate(sequence)
        if value == minimum_value
    ]

    return min(candidate_rotations)


def canonicalize_ring_indices(
    ring_indices: Sequence[int],
) -> RingIndexTuple:
    """
    Return a direction-independent canonical representation of a ring cycle.

    The same ring traversed from different starting atoms or in reverse
    direction will therefore produce the same tuple.
    """

    if not ring_indices:
        return ()

    sequence = tuple(int(index) for index in ring_indices)

    if len(sequence) > 1 and sequence[0] == sequence[-1]:
        sequence = sequence[:-1]

    if len(sequence) < 3:
        return sequence

    forward = _rotate_sequence_to_smallest(sequence)
    reverse = _rotate_sequence_to_smallest(tuple(reversed(sequence)))

    return min(forward, reverse)


def canonicalize_ring_atoms(
    atoms: Sequence[Any],
) -> Tuple[Tuple[Any, ...], Tuple[int, ...]]:
    """
    Canonicalize an atom ring using object identity.

    Returns
    -------
    canonical_atoms
        Atoms ordered according to the canonical identity cycle.

    canonical_identity
        Canonical tuple containing ``id(atom)`` values.
    """

    atom_tuple = tuple(atoms)

    if len(atom_tuple) < 3:
        return atom_tuple, tuple(id(atom) for atom in atom_tuple)

    identities = tuple(id(atom) for atom in atom_tuple)
    canonical_identity = canonicalize_ring_indices(identities)

    identity_to_atom = {
        id(atom): atom
        for atom in atom_tuple
    }

    canonical_atoms = tuple(
        identity_to_atom[identity]
        for identity in canonical_identity
    )

    return canonical_atoms, canonical_identity


def deduplicate_atom_rings(
    rings: Iterable[Sequence[Any]],
) -> Tuple[RingAtomTuple, ...]:
    """
    Remove duplicate rings while preserving deterministic ordering.
    """

    unique: Dict[Tuple[int, ...], RingAtomTuple] = {}

    for ring in rings:
        atom_tuple = tuple(ring)

        if len(atom_tuple) < 3:
            continue

        canonical_atoms, canonical_identity = canonicalize_ring_atoms(
            atom_tuple
        )

        unique.setdefault(
            canonical_identity,
            canonical_atoms,
        )

    return tuple(
        unique[key]
        for key in sorted(unique)
    )


# -----------------------------------------------------------------------------
# 4.3. Utilitários de conectividade para anéis
# -----------------------------------------------------------------------------

def build_atom_adjacency_graph(
    atoms: Iterable[Any],
    *,
    heavy_atoms_only: bool = True,
) -> Tuple[Tuple[Any, ...], RingGraph]:
    """
    Build an undirected adjacency graph for an atom collection.

    Only bonds between atoms present in the supplied collection are included.
    """

    atom_tuple = deduplicate_atoms(atoms)

    if heavy_atoms_only:
        atom_tuple = tuple(
            atom
            for atom in atom_tuple
            if is_heavy_atom(atom)
        )

    identity_to_index = {
        id(atom): index
        for index, atom in enumerate(atom_tuple)
    }

    adjacency: RingGraph = {
        index: set()
        for index in range(len(atom_tuple))
    }

    for index, atom in enumerate(atom_tuple):
        for neighbor in get_bonded_atoms(
            atom,
            include_hydrogens=not heavy_atoms_only,
        ):
            neighbor_index = identity_to_index.get(id(neighbor))

            if neighbor_index is None or neighbor_index == index:
                continue

            adjacency[index].add(neighbor_index)
            adjacency[neighbor_index].add(index)

    return atom_tuple, adjacency


def _graph_edge_count(
    adjacency: Mapping[int, Set[int]],
) -> int:
    """
    Return the number of undirected edges in an adjacency graph.
    """

    return sum(
        len(neighbors)
        for neighbors in adjacency.values()
    ) // 2


def _ring_is_connected(
    indices: Sequence[int],
    adjacency: Mapping[int, Set[int]],
) -> bool:
    """
    Return whether all selected vertices belong to one connected component.
    """

    selected = set(indices)

    if not selected:
        return False

    stack = [next(iter(selected))]
    visited: Set[int] = set()

    while stack:
        current = stack.pop()

        if current in visited:
            continue

        visited.add(current)

        stack.extend(
            neighbor
            for neighbor in adjacency.get(current, ())
            if neighbor in selected and neighbor not in visited
        )

    return visited == selected


def _ring_subgraph_edge_count(
    indices: Sequence[int],
    adjacency: Mapping[int, Set[int]],
) -> int:
    """
    Count edges internal to a selected set of graph vertices.
    """

    selected = set(indices)

    return sum(
        1
        for index in selected
        for neighbor in adjacency.get(index, ())
        if neighbor in selected and index < neighbor
    )


def _is_simple_cycle_subgraph(
    indices: Sequence[int],
    adjacency: Mapping[int, Set[int]],
) -> bool:
    """
    Return whether selected vertices form a simple cycle subgraph.
    """

    selected = set(indices)

    if len(selected) < 3:
        return False

    if not _ring_is_connected(indices, adjacency):
        return False

    for index in selected:
        internal_degree = sum(
            1
            for neighbor in adjacency.get(index, ())
            if neighbor in selected
        )

        if internal_degree != 2:
            return False

    return (
        _ring_subgraph_edge_count(indices, adjacency)
        == len(selected)
    )


# -----------------------------------------------------------------------------
# 4.4. Busca de ciclos simples
# -----------------------------------------------------------------------------

def _enumerate_simple_cycles_depth_first(
    adjacency: Mapping[int, Set[int]],
    *,
    minimum_size: int,
    maximum_size: int,
) -> Tuple[RingIndexTuple, ...]:
    """
    Enumerate simple undirected cycles using bounded depth-first search.

    This implementation is intended for molecular graphs and therefore
    prioritizes correctness and deterministic output over very large graph
    performance.
    """

    if minimum_size < 3:
        raise ValueError(
            "minimum_size must be at least 3."
        )

    if maximum_size < minimum_size:
        raise ValueError(
            "maximum_size must be >= minimum_size."
        )

    found_cycles: Set[RingIndexTuple] = set()
    vertices = sorted(adjacency)

    for start in vertices:
        path: List[int] = [start]
        visited: Set[int] = {start}

        def visit(current: int) -> None:
            if len(path) > maximum_size:
                return

            for neighbor in sorted(adjacency.get(current, ())):
                if neighbor == start:
                    if len(path) >= minimum_size:
                        cycle = canonicalize_ring_indices(path)

                        if len(cycle) <= maximum_size:
                            found_cycles.add(cycle)

                    continue

                if neighbor in visited:
                    continue

                # Enforce start as the smallest vertex in the cycle.
                # This reduces duplicate traversal substantially.
                if neighbor < start:
                    continue

                if len(path) >= maximum_size:
                    continue

                visited.add(neighbor)
                path.append(neighbor)

                visit(neighbor)

                path.pop()
                visited.remove(neighbor)

        visit(start)

    return tuple(sorted(found_cycles))


def find_simple_cycles(
    atoms: Iterable[Any],
    *,
    minimum_size: int = DEFAULT_MINIMUM_RING_SIZE,
    maximum_size: int = DEFAULT_MAXIMUM_RING_SIZE,
    heavy_atoms_only: bool = True,
) -> Tuple[RingAtomTuple, ...]:
    """
    Find simple cycles in an atom collection.

    Parameters
    ----------
    atoms
        Atom collection with accessible bond connectivity.

    minimum_size
        Minimum accepted cycle size.

    maximum_size
        Maximum accepted cycle size.

    heavy_atoms_only
        Exclude hydrogen atoms before graph construction.
    """

    atom_tuple, adjacency = build_atom_adjacency_graph(
        atoms,
        heavy_atoms_only=heavy_atoms_only,
    )

    if len(atom_tuple) < minimum_size:
        return ()

    if _graph_edge_count(adjacency) < minimum_size:
        return ()

    index_cycles = _enumerate_simple_cycles_depth_first(
        adjacency,
        minimum_size=minimum_size,
        maximum_size=maximum_size,
    )

    atom_cycles = tuple(
        tuple(atom_tuple[index] for index in cycle)
        for cycle in index_cycles
        if _is_simple_cycle_subgraph(cycle, adjacency)
    )

    return deduplicate_atom_rings(atom_cycles)


# -----------------------------------------------------------------------------
# 4.5. Reconhecimento de anéis aromáticos proteicos conhecidos
# -----------------------------------------------------------------------------

def get_standard_residue_ring_definitions(
    residue_or_atom: Any,
) -> Tuple[Tuple[str, ...], ...]:
    """
    Return known aromatic ring definitions for a standard residue.
    """

    residue_name = get_residue_name(residue_or_atom)

    return STANDARD_AROMATIC_RING_ATOMS.get(
        residue_name,
        (),
    )


def find_standard_residue_aromatic_rings(
    residue: Any,
    *,
    require_complete: bool = True,
    include_fused_trp_system: bool = False,
    valid_coordinates_only: bool = True,
) -> Tuple[RingAtomTuple, ...]:
    """
    Identify aromatic rings from known protein residue atom names.

    Supported residues include phenylalanine, tyrosine, histidine protonation
    variants, and tryptophan.
    """

    residue_name = get_residue_name(residue)
    definitions = get_standard_residue_ring_definitions(residue)

    if not definitions:
        return ()

    atom_map = map_atoms_by_name(residue)
    detected: List[RingAtomTuple] = []

    for atom_names in definitions:
        atoms: List[Any] = []
        missing_names: List[str] = []

        for atom_name in atom_names:
            atom = atom_map.get(atom_name.upper())

            if atom is None:
                missing_names.append(atom_name)
                continue

            if (
                valid_coordinates_only
                and not atom_has_valid_coordinate(atom)
            ):
                missing_names.append(atom_name)
                continue

            atoms.append(atom)

        if missing_names and require_complete:
            continue

        if len(atoms) >= 3:
            detected.append(tuple(atoms))

    if (
        residue_name == "TRP"
        and include_fused_trp_system
    ):
        fused_atoms: List[Any] = []

        for atom_name in TRP_FUSED_SYSTEM_ATOMS:
            atom = atom_map.get(atom_name.upper())

            if atom is None:
                if require_complete:
                    fused_atoms = []
                    break

                continue

            if (
                valid_coordinates_only
                and not atom_has_valid_coordinate(atom)
            ):
                if require_complete:
                    fused_atoms = []
                    break

                continue

            fused_atoms.append(atom)

        if len(fused_atoms) >= 5:
            detected.append(tuple(fused_atoms))

    return deduplicate_atom_rings(detected)


# -----------------------------------------------------------------------------
# 4.6. Definições aromáticas para ácidos nucleicos
# -----------------------------------------------------------------------------

PURINE_FIVE_MEMBER_RING_ATOMS: Final[Tuple[str, ...]] = (
    "N7",
    "C8",
    "N9",
    "C4",
    "C5",
)


PURINE_SIX_MEMBER_RING_ATOMS: Final[Tuple[str, ...]] = (
    "N1",
    "C2",
    "N3",
    "C4",
    "C5",
    "C6",
)


PURINE_FUSED_SYSTEM_ATOMS: Final[Tuple[str, ...]] = (
    "N1",
    "C2",
    "N3",
    "C4",
    "C5",
    "C6",
    "N7",
    "C8",
    "N9",
)


PYRIMIDINE_RING_ATOMS: Final[Tuple[str, ...]] = (
    "N1",
    "C2",
    "N3",
    "C4",
    "C5",
    "C6",
)


STANDARD_NUCLEIC_ACID_RING_ATOMS: Final[
    Mapping[str, Tuple[Tuple[str, ...], ...]]
] = {
    "A": (
        PURINE_FIVE_MEMBER_RING_ATOMS,
        PURINE_SIX_MEMBER_RING_ATOMS,
    ),
    "ADE": (
        PURINE_FIVE_MEMBER_RING_ATOMS,
        PURINE_SIX_MEMBER_RING_ATOMS,
    ),
    "DA": (
        PURINE_FIVE_MEMBER_RING_ATOMS,
        PURINE_SIX_MEMBER_RING_ATOMS,
    ),
    "G": (
        PURINE_FIVE_MEMBER_RING_ATOMS,
        PURINE_SIX_MEMBER_RING_ATOMS,
    ),
    "GUA": (
        PURINE_FIVE_MEMBER_RING_ATOMS,
        PURINE_SIX_MEMBER_RING_ATOMS,
    ),
    "DG": (
        PURINE_FIVE_MEMBER_RING_ATOMS,
        PURINE_SIX_MEMBER_RING_ATOMS,
    ),
    "C": (
        PYRIMIDINE_RING_ATOMS,
    ),
    "CYT": (
        PYRIMIDINE_RING_ATOMS,
    ),
    "DC": (
        PYRIMIDINE_RING_ATOMS,
    ),
    "T": (
        PYRIMIDINE_RING_ATOMS,
    ),
    "THY": (
        PYRIMIDINE_RING_ATOMS,
    ),
    "DT": (
        PYRIMIDINE_RING_ATOMS,
    ),
    "U": (
        PYRIMIDINE_RING_ATOMS,
    ),
    "URA": (
        PYRIMIDINE_RING_ATOMS,
    ),
    "DU": (
        PYRIMIDINE_RING_ATOMS,
    ),
}


def find_nucleic_acid_aromatic_rings(
    residue: Any,
    *,
    require_complete: bool = True,
    include_fused_purine_system: bool = False,
    valid_coordinates_only: bool = True,
) -> Tuple[RingAtomTuple, ...]:
    """
    Identify aromatic rings in standard nucleic-acid bases.
    """

    residue_name = get_residue_name(residue)

    definitions = STANDARD_NUCLEIC_ACID_RING_ATOMS.get(
        residue_name,
        (),
    )

    if not definitions:
        return ()

    atom_map = map_atoms_by_name(residue)
    detected: List[RingAtomTuple] = []

    for atom_names in definitions:
        ring_atoms: List[Any] = []

        for atom_name in atom_names:
            atom = atom_map.get(atom_name.upper())

            if atom is None:
                if require_complete:
                    ring_atoms = []
                    break

                continue

            if (
                valid_coordinates_only
                and not atom_has_valid_coordinate(atom)
            ):
                if require_complete:
                    ring_atoms = []
                    break

                continue

            ring_atoms.append(atom)

        if len(ring_atoms) >= 3:
            detected.append(tuple(ring_atoms))

    if (
        residue_name in PURINE_RESIDUES
        and include_fused_purine_system
    ):
        fused_atoms: List[Any] = []

        for atom_name in PURINE_FUSED_SYSTEM_ATOMS:
            atom = atom_map.get(atom_name.upper())

            if atom is None:
                if require_complete:
                    fused_atoms = []
                    break

                continue

            if (
                valid_coordinates_only
                and not atom_has_valid_coordinate(atom)
            ):
                if require_complete:
                    fused_atoms = []
                    break

                continue

            fused_atoms.append(atom)

        if len(fused_atoms) >= 5:
            detected.append(tuple(fused_atoms))

    return deduplicate_atom_rings(detected)


# -----------------------------------------------------------------------------
# 4.7. Avaliação preliminar de aromaticidade de ciclos
# -----------------------------------------------------------------------------

def count_aromatic_atoms(
    atoms: Iterable[Any],
) -> int:
    """
    Count atoms explicitly or implicitly classified as aromatic.
    """

    return sum(
        1
        for atom in atoms
        if is_aromatic_atom(atom)
    )


def count_aromatic_bonds(
    ring_atoms: Sequence[Any],
) -> int:
    """
    Count aromatic bonds along a ring path.
    """

    atoms = tuple(ring_atoms)

    if len(atoms) < 3:
        return 0

    aromatic_bond_count = 0

    for index, atom in enumerate(atoms):
        next_atom = atoms[(index + 1) % len(atoms)]

        if is_aromatic_bond(atom, next_atom):
            aromatic_bond_count += 1

    return aromatic_bond_count


def count_double_bonds(
    ring_atoms: Sequence[Any],
) -> int:
    """
    Count double bonds along a ring path.
    """

    atoms = tuple(ring_atoms)

    if len(atoms) < 3:
        return 0

    double_bond_count = 0

    for index, atom in enumerate(atoms):
        next_atom = atoms[(index + 1) % len(atoms)]
        bond_order = get_bond_order(atom, next_atom)

        if bond_order is not None and bond_order >= 1.75:
            double_bond_count += 1

    return double_bond_count


def ring_has_alternating_bond_pattern(
    ring_atoms: Sequence[Any],
) -> bool:
    """
    Estimate whether a ring has a conjugated alternating bond pattern.

    Aromatic 1.5 bonds are treated as compatible with conjugation.
    """

    atoms = tuple(ring_atoms)

    if len(atoms) < 5:
        return False

    bond_orders: List[Optional[float]] = []

    for index, atom in enumerate(atoms):
        next_atom = atoms[(index + 1) % len(atoms)]
        bond_orders.append(
            get_bond_order(atom, next_atom)
        )

    known_orders = [
        order
        for order in bond_orders
        if order is not None
    ]

    if len(known_orders) < max(3, len(atoms) - 1):
        return False

    if all(
        abs(order - 1.5) <= 0.15
        for order in known_orders
    ):
        return True

    normalized_orders = [
        2 if order >= 1.75 else 1
        for order in known_orders
    ]

    transitions = sum(
        1
        for index, order in enumerate(normalized_orders)
        if order != normalized_orders[
            (index + 1) % len(normalized_orders)
        ]
    )

    return transitions >= len(normalized_orders) - 2


def ring_elements_are_aromatic_compatible(
    ring_atoms: Iterable[Any],
    *,
    allowed_elements: Collection[str] = COMMON_AROMATIC_ELEMENTS,
) -> bool:
    """
    Return whether all ring atoms use aromatic-compatible elements.
    """

    normalized_allowed = {
        normalize_element_symbol(element)
        for element in allowed_elements
    }

    elements = [
        get_atom_element(atom)
        for atom in ring_atoms
    ]

    return bool(elements) and all(
        element in normalized_allowed
        for element in elements
    )


def estimate_ring_aromaticity(
    ring_atoms: Sequence[Any],
    *,
    minimum_aromatic_atoms: Optional[int] = None,
    require_compatible_elements: bool = True,
    accept_known_residue_rings: bool = True,
) -> Tuple[bool, str, float]:
    """
    Estimate whether a molecular cycle is aromatic.

    Returns
    -------
    is_aromatic
        Final aromaticity classification.

    source
        Description of the evidence used.

    confidence
        Normalized confidence estimate in the interval [0, 1].
    """

    atoms = tuple(ring_atoms)
    ring_size = len(atoms)

    if ring_size < 3:
        return False, "invalid_ring", 0.0

    if require_compatible_elements and not (
        ring_elements_are_aromatic_compatible(atoms)
    ):
        return False, "incompatible_elements", 0.0

    residue_names = {
        get_residue_name(atom)
        for atom in atoms
        if get_residue_name(atom)
    }

    if accept_known_residue_rings and len(residue_names) == 1:
        residue_name = next(iter(residue_names))

        known_definitions = (
            STANDARD_AROMATIC_RING_ATOMS.get(
                residue_name,
                (),
            )
            or STANDARD_NUCLEIC_ACID_RING_ATOMS.get(
                residue_name,
                (),
            )
        )

        atom_name_set = {
            get_atom_name(atom).upper()
            for atom in atoms
        }

        for definition in known_definitions:
            if atom_name_set == set(definition):
                return True, "known_residue_definition", 1.0

    aromatic_atom_count = count_aromatic_atoms(atoms)
    aromatic_bond_count = count_aromatic_bonds(atoms)
    double_bond_count = count_double_bonds(atoms)

    required_aromatic_atoms = (
        minimum_aromatic_atoms
        if minimum_aromatic_atoms is not None
        else max(
            3,
            min(
                ring_size,
                DEFAULT_MINIMUM_AROMATIC_ATOMS,
            ),
        )
    )

    atom_fraction = aromatic_atom_count / ring_size
    bond_fraction = aromatic_bond_count / ring_size

    alternating_pattern = ring_has_alternating_bond_pattern(
        atoms
    )

    if aromatic_atom_count == ring_size:
        return True, "all_atoms_aromatic", 1.0

    if aromatic_bond_count == ring_size:
        return True, "all_bonds_aromatic", 1.0

    if (
        aromatic_atom_count >= required_aromatic_atoms
        and bond_fraction >= 0.50
    ):
        confidence = min(
            1.0,
            0.55 * atom_fraction
            + 0.45 * bond_fraction,
        )

        return True, "atom_and_bond_aromaticity", confidence

    if alternating_pattern:
        expected_double_bonds = ring_size // 2

        confidence = min(
            0.90,
            0.60
            + 0.30
            * min(
                1.0,
                double_bond_count
                / max(1, expected_double_bonds),
            ),
        )

        return True, "alternating_conjugation", confidence

    if atom_fraction >= 0.80:
        return True, "predominantly_aromatic_atoms", atom_fraction

    return False, "insufficient_aromatic_evidence", max(
        atom_fraction,
        bond_fraction,
    )


# -----------------------------------------------------------------------------
# 4.8. Filtragem de ciclos aromáticos
# -----------------------------------------------------------------------------

def filter_aromatic_cycles(
    cycles: Iterable[Sequence[Any]],
    *,
    minimum_size: int = DEFAULT_MINIMUM_RING_SIZE,
    maximum_size: int = DEFAULT_MAXIMUM_RING_SIZE,
    minimum_aromatic_atoms: Optional[int] = None,
    require_compatible_elements: bool = True,
) -> Tuple[Tuple[RingAtomTuple, str, float], ...]:
    """
    Filter molecular cycles according to aromaticity criteria.

    Each result contains:

    - ring atoms;
    - aromaticity source;
    - confidence score.
    """

    accepted: List[Tuple[RingAtomTuple, str, float]] = []
    seen: Set[Tuple[int, ...]] = set()

    for cycle in cycles:
        atoms, identity = canonicalize_ring_atoms(cycle)

        if identity in seen:
            continue

        seen.add(identity)

        if not minimum_size <= len(atoms) <= maximum_size:
            continue

        is_aromatic, source, confidence = estimate_ring_aromaticity(
            atoms,
            minimum_aromatic_atoms=minimum_aromatic_atoms,
            require_compatible_elements=require_compatible_elements,
        )

        if not is_aromatic:
            continue

        accepted.append(
            (
                atoms,
                source,
                confidence,
            )
        )

    accepted.sort(
        key=lambda item: tuple(
            id(atom)
            for atom in item[0]
        )
    )

    return tuple(accepted)


# -----------------------------------------------------------------------------
# 4.9. Detecção de sistemas fundidos
# -----------------------------------------------------------------------------

def rings_share_atoms(
    ring_1: Sequence[Any],
    ring_2: Sequence[Any],
    *,
    minimum_shared_atoms: int = 1,
) -> bool:
    """
    Return whether two rings share at least a selected number of atoms.
    """

    identities_1 = {
        id(atom)
        for atom in ring_1
    }

    identities_2 = {
        id(atom)
        for atom in ring_2
    }

    return (
        len(identities_1 & identities_2)
        >= minimum_shared_atoms
    )


def rings_are_fused(
    ring_1: Sequence[Any],
    ring_2: Sequence[Any],
) -> bool:
    """
    Return whether two rings share at least one covalent edge.
    """

    shared_atoms = [
        atom
        for atom in ring_1
        if any(atom is candidate for candidate in ring_2)
    ]

    if len(shared_atoms) < 2:
        return False

    for atom_1, atom_2 in combinations(shared_atoms, 2):
        if atoms_are_bonded(atom_1, atom_2):
            return True

    return False


def group_fused_rings(
    rings: Iterable[Sequence[Any]],
) -> Tuple[Tuple[RingAtomTuple, ...], ...]:
    """
    Group individual rings into fused aromatic systems.
    """

    ring_tuple = deduplicate_atom_rings(rings)

    if not ring_tuple:
        return ()

    adjacency: Dict[int, Set[int]] = {
        index: set()
        for index in range(len(ring_tuple))
    }

    for first_index, second_index in combinations(
        range(len(ring_tuple)),
        2,
    ):
        if rings_are_fused(
            ring_tuple[first_index],
            ring_tuple[second_index],
        ):
            adjacency[first_index].add(second_index)
            adjacency[second_index].add(first_index)

    groups: List[Tuple[RingAtomTuple, ...]] = []
    visited: Set[int] = set()

    for start_index in range(len(ring_tuple)):
        if start_index in visited:
            continue

        stack = [start_index]
        component_indices: List[int] = []

        while stack:
            current = stack.pop()

            if current in visited:
                continue

            visited.add(current)
            component_indices.append(current)

            stack.extend(
                neighbor
                for neighbor in adjacency[current]
                if neighbor not in visited
            )

        groups.append(
            tuple(
                ring_tuple[index]
                for index in sorted(component_indices)
            )
        )

    return tuple(groups)


def merge_fused_ring_atoms(
    fused_ring_group: Iterable[Sequence[Any]],
) -> RingAtomTuple:
    """
    Merge atoms from a fused-ring group while preserving stable order.
    """

    atoms: List[Any] = []
    seen_ids: Set[int] = set()

    for ring in fused_ring_group:
        for atom in ring:
            atom_id = id(atom)

            if atom_id in seen_ids:
                continue

            seen_ids.add(atom_id)
            atoms.append(atom)

    atoms.sort(
        key=lambda atom: (
            get_atom_serial_number(atom)
            if get_atom_serial_number(atom) is not None
            else float("inf"),
            get_atom_index(atom)
            if get_atom_index(atom) is not None
            else float("inf"),
            get_atom_name(atom),
            id(atom),
        )
    )

    return tuple(atoms)


# -----------------------------------------------------------------------------
# 4.10. Inferência de resíduos e propriedades do anel
# -----------------------------------------------------------------------------

def infer_ring_residue(
    ring_atoms: Sequence[Any],
) -> Optional[Any]:
    """
    Return the residue shared by all ring atoms, when one exists.
    """

    residues: List[Any] = []

    for atom in ring_atoms:
        residue = get_atom_residue(atom)

        if residue is not None:
            residues.append(residue)

    if not residues:
        return None

    first_residue = residues[0]

    if all(
        residue is first_residue
        for residue in residues
    ):
        return first_residue

    return None


def infer_ring_model(
    ring_atoms: Sequence[Any],
) -> Optional[Any]:
    """
    Return the molecular model shared by all ring atoms, when possible.
    """

    models = [
        get_atom_model(atom)
        for atom in ring_atoms
    ]

    models = [
        model
        for model in models
        if model is not None
    ]

    if not models:
        return None

    first_model = models[0]

    if all(model is first_model for model in models):
        return first_model

    return first_model


def ring_is_heteroaromatic(
    ring_atoms: Sequence[Any],
) -> bool:
    """
    Return whether an aromatic ring contains non-carbon atoms.
    """

    return any(
        get_atom_element(atom) != "C"
        for atom in ring_atoms
    )


def infer_ring_participant_type(
    ring_atoms: Sequence[Any],
    *,
    explicit_participant_type: Optional[str] = None,
    ligand_residue_names: Optional[Collection[str]] = None,
    receptor_residue_names: Optional[Collection[str]] = None,
) -> str:
    """
    Infer the molecular participant type associated with a ring.
    """

    if explicit_participant_type:
        return str(explicit_participant_type).strip().lower()

    residue = infer_ring_residue(ring_atoms)

    if residue is not None:
        return infer_participant_type(
            residue,
            ligand_residue_names=ligand_residue_names,
            receptor_residue_names=receptor_residue_names,
        )

    inferred_types = {
        infer_participant_type(
            atom,
            ligand_residue_names=ligand_residue_names,
            receptor_residue_names=receptor_residue_names,
        )
        for atom in ring_atoms
    }

    inferred_types.discard(PARTICIPANT_UNKNOWN)

    if len(inferred_types) == 1:
        return next(iter(inferred_types))

    return PARTICIPANT_UNKNOWN


# -----------------------------------------------------------------------------
# 4.11. Construção preliminar de PiRing
# -----------------------------------------------------------------------------

def create_preliminary_pi_ring(
    atoms: Sequence[Any],
    *,
    ring_index: Optional[int] = None,
    participant_type: Optional[str] = None,
    aromaticity_source: Optional[str] = None,
    is_fused: bool = False,
    valid_coordinates_only: bool = True,
    ligand_residue_names: Optional[Collection[str]] = None,
    receptor_residue_names: Optional[Collection[str]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> PiRing:
    """
    Create a preliminary ``PiRing`` before geometric fitting.

    Centroid, normal, radius, and planarity values remain unset. They will be
    calculated in the ring-geometry section.
    """

    atom_tuple = deduplicate_atoms(atoms)

    if valid_coordinates_only:
        atom_tuple = filter_atoms_with_coordinates(atom_tuple)

    if len(atom_tuple) < 3:
        raise ValueError(
            "At least three valid atoms are required to create a PiRing."
        )

    residue = infer_ring_residue(atom_tuple)
    model = infer_ring_model(atom_tuple)

    residue_name = (
        get_residue_name(residue)
        if residue is not None
        else None
    )

    residue_number = (
        get_residue_number(residue)
        if residue is not None
        else None
    )

    chain_id = (
        get_residue_chain_id(residue)
        if residue is not None
        else None
    )

    normalized_participant_type = infer_ring_participant_type(
        atom_tuple,
        explicit_participant_type=participant_type,
        ligand_residue_names=ligand_residue_names,
        receptor_residue_names=receptor_residue_names,
    )

    atom_references = create_pi_atom_references(
        atom_tuple,
        skip_invalid=not valid_coordinates_only,
    )

    element_symbols = tuple(
        get_atom_element(atom)
        for atom in atom_tuple
    )

    protein_ring = (
        normalized_participant_type
        in {
            PARTICIPANT_PROTEIN,
            PARTICIPANT_RECEPTOR,
        }
        and residue_name in AROMATIC_RESIDUES
    )

    ligand_ring = (
        normalized_participant_type == PARTICIPANT_LIGAND
    )

    return PiRing(
        atoms=atom_tuple,
        atom_references=atom_references,
        ring_index=ring_index,
        ring_size=len(atom_tuple),
        residue_name=residue_name,
        residue_number=residue_number,
        chain_id=chain_id,
        model_id=get_model_identifier(model),
        participant_type=normalized_participant_type,
        is_fused=is_fused,
        is_heteroaromatic=ring_is_heteroaromatic(
            atom_tuple
        ),
        is_protein_ring=protein_ring,
        is_ligand_ring=ligand_ring,
        aromaticity_source=aromaticity_source,
        atom_names=tuple(
            get_atom_name(atom)
            for atom in atom_tuple
        ),
        element_symbols=element_symbols,
        valid=True,
        metadata=_copy_mapping(metadata),
    )


# -----------------------------------------------------------------------------
# 4.12. Detecção de anéis aromáticos por resíduo
# -----------------------------------------------------------------------------

def detect_residue_aromatic_rings(
    residue: Any,
    *,
    config: Optional[PiAnalysisConfig] = None,
    participant_type: Optional[str] = None,
    use_known_definitions: bool = True,
    use_connectivity_detection: bool = True,
    include_fused_systems: Optional[bool] = None,
    ligand_residue_names: Optional[Collection[str]] = None,
    receptor_residue_names: Optional[Collection[str]] = None,
) -> List[PiRing]:
    """
    Detect aromatic rings in one residue.

    Known protein and nucleic-acid definitions are evaluated first. Generic
    connectivity detection is used as a fallback or complementary strategy.
    """

    analysis_config = (
        config
        if config is not None
        else create_default_pi_config()
    )

    if not isinstance(analysis_config, PiAnalysisConfig):
        raise TypeError(
            "config must be a PiAnalysisConfig or None."
        )

    residue_atoms = get_residue_atoms(
        residue,
        include_hydrogens=False,
        valid_coordinates_only=True,
    )

    if len(residue_atoms) < analysis_config.minimum_ring_size:
        return []

    residue_name = get_residue_name(residue)
    detected_cycles: List[
        Tuple[RingAtomTuple, str, float]
    ] = []

    if use_known_definitions:
        if residue_name in AROMATIC_RESIDUES:
            known_rings = find_standard_residue_aromatic_rings(
                residue,
                include_fused_trp_system=False,
            )

            for ring in known_rings:
                detected_cycles.append(
                    (
                        ring,
                        "known_protein_residue",
                        1.0,
                    )
                )

        elif residue_name in NUCLEIC_ACID_AROMATIC_RESIDUES:
            known_rings = find_nucleic_acid_aromatic_rings(
                residue,
                include_fused_purine_system=False,
            )

            for ring in known_rings:
                detected_cycles.append(
                    (
                        ring,
                        "known_nucleic_acid_residue",
                        1.0,
                    )
                )

    if use_connectivity_detection:
        generic_cycles = find_simple_cycles(
            residue_atoms,
            minimum_size=analysis_config.minimum_ring_size,
            maximum_size=analysis_config.maximum_ring_size,
        )

        aromatic_cycles = filter_aromatic_cycles(
            generic_cycles,
            minimum_size=analysis_config.minimum_ring_size,
            maximum_size=analysis_config.maximum_ring_size,
        )

        detected_cycles.extend(aromatic_cycles)

    unique_cycles: Dict[
        Tuple[int, ...],
        Tuple[RingAtomTuple, str, float],
    ] = {}

    for atoms, source, confidence in detected_cycles:
        canonical_atoms, identity = canonicalize_ring_atoms(
            atoms
        )

        current = unique_cycles.get(identity)

        if current is None or confidence > current[2]:
            unique_cycles[identity] = (
                canonical_atoms,
                source,
                confidence,
            )

    sorted_cycles = [
        unique_cycles[key]
        for key in sorted(unique_cycles)
    ]

    rings: List[PiRing] = []

    for ring_index, (
        atoms,
        source,
        confidence,
    ) in enumerate(sorted_cycles, start=1):
        ring = create_preliminary_pi_ring(
            atoms,
            ring_index=ring_index,
            participant_type=participant_type,
            aromaticity_source=source,
            ligand_residue_names=ligand_residue_names,
            receptor_residue_names=receptor_residue_names,
            metadata={
                "aromaticity_confidence": confidence,
                "detection_scope": "residue",
            },
        )

        if (
            ring.is_heteroaromatic
            and not analysis_config.allow_heteroaromatic_rings
        ):
            continue

        rings.append(ring)

    should_include_fused = (
        analysis_config.treat_fused_system_as_single_ring
        if include_fused_systems is None
        else bool(include_fused_systems)
    )

    if should_include_fused and analysis_config.allow_fused_rings:
        fused_groups = group_fused_rings(
            ring.atoms
            for ring in rings
        )

        for fused_group in fused_groups:
            if len(fused_group) < 2:
                continue

            fused_atoms = merge_fused_ring_atoms(
                fused_group
            )

            if (
                len(fused_atoms)
                > analysis_config.maximum_fused_ring_size
            ):
                continue

            fused_ring = create_preliminary_pi_ring(
                fused_atoms,
                ring_index=len(rings) + 1,
                participant_type=participant_type,
                aromaticity_source="fused_aromatic_system",
                is_fused=True,
                ligand_residue_names=ligand_residue_names,
                receptor_residue_names=receptor_residue_names,
                metadata={
                    "component_ring_count": len(fused_group),
                    "detection_scope": "residue",
                },
            )

            rings.append(fused_ring)

    return rings


# -----------------------------------------------------------------------------
# 4.13. Detecção de anéis aromáticos em coleções moleculares
# -----------------------------------------------------------------------------

def detect_aromatic_rings(
    molecular_input: Any,
    *,
    config: Optional[PiAnalysisConfig] = None,
    participant_type: Optional[str] = None,
    ligand_residue_names: Optional[Collection[str]] = None,
    receptor_residue_names: Optional[Collection[str]] = None,
    use_known_definitions: bool = True,
    use_connectivity_detection: bool = True,
    detect_cross_residue_cycles: bool = False,
) -> List[PiRing]:
    """
    Detect aromatic rings in a model, residue collection or atom collection.

    Residue-level detection is the default because most biochemical aromatic
    systems are contained within one residue. Optional cross-residue cycle
    detection supports unusual ligands or covalently connected structures.
    """

    analysis_config = (
        config
        if config is not None
        else create_default_pi_config()
    )

    if not isinstance(analysis_config, PiAnalysisConfig):
        raise TypeError(
            "config must be a PiAnalysisConfig or None."
        )

    atoms = normalize_atom_collection(
        molecular_input,
        include_hydrogens=False,
        valid_coordinates_only=True,
    )

    residues = normalize_residue_collection(atoms)

    detected_rings: List[PiRing] = []

    for residue in residues:
        inferred_type = (
            participant_type
            or infer_participant_type(
                residue,
                ligand_residue_names=ligand_residue_names,
                receptor_residue_names=receptor_residue_names,
            )
        )

        if (
            inferred_type in {
                PARTICIPANT_PROTEIN,
                PARTICIPANT_RECEPTOR,
            }
            and not analysis_config.include_protein_rings
        ):
            continue

        if (
            inferred_type == PARTICIPANT_LIGAND
            and not analysis_config.include_ligand_rings
        ):
            continue

        if (
            inferred_type == PARTICIPANT_NUCLEIC_ACID
            and not analysis_config.include_nucleic_acid_rings
        ):
            continue

        residue_rings = detect_residue_aromatic_rings(
            residue,
            config=analysis_config,
            participant_type=inferred_type,
            use_known_definitions=use_known_definitions,
            use_connectivity_detection=use_connectivity_detection,
            ligand_residue_names=ligand_residue_names,
            receptor_residue_names=receptor_residue_names,
        )

        detected_rings.extend(residue_rings)

    if detect_cross_residue_cycles:
        generic_cycles = find_simple_cycles(
            atoms,
            minimum_size=analysis_config.minimum_ring_size,
            maximum_size=analysis_config.maximum_ring_size,
        )

        aromatic_cycles = filter_aromatic_cycles(
            generic_cycles,
            minimum_size=analysis_config.minimum_ring_size,
            maximum_size=analysis_config.maximum_ring_size,
        )

        existing_identities = {
            canonicalize_ring_atoms(ring.atoms)[1]
            for ring in detected_rings
        }

        for atoms_in_ring, source, confidence in aromatic_cycles:
            identity = canonicalize_ring_atoms(
                atoms_in_ring
            )[1]

            if identity in existing_identities:
                continue

            ring = create_preliminary_pi_ring(
                atoms_in_ring,
                ring_index=len(detected_rings) + 1,
                participant_type=participant_type,
                aromaticity_source=source,
                ligand_residue_names=ligand_residue_names,
                receptor_residue_names=receptor_residue_names,
                metadata={
                    "aromaticity_confidence": confidence,
                    "detection_scope": "cross_residue",
                },
            )

            if (
                ring.is_heteroaromatic
                and not analysis_config.allow_heteroaromatic_rings
            ):
                continue

            detected_rings.append(ring)
            existing_identities.add(identity)

    return deduplicate_pi_rings(detected_rings)


# -----------------------------------------------------------------------------
# 4.14. Deduplicação de PiRing
# -----------------------------------------------------------------------------

def get_pi_ring_identity_key(
    ring: PiRing,
) -> Tuple[Any, ...]:
    """
    Return a stable identity key for a ``PiRing``.
    """

    if not isinstance(ring, PiRing):
        raise TypeError(
            "ring must be a PiRing."
        )

    atom_identity = tuple(
        sorted(
            id(atom)
            for atom in ring.atoms
        )
    )

    return (
        ring.model_id,
        ring.chain_id,
        ring.residue_name,
        ring.residue_number,
        atom_identity,
        bool(ring.is_fused),
    )


def deduplicate_pi_rings(
    rings: Iterable[PiRing],
) -> List[PiRing]:
    """
    Deduplicate ``PiRing`` objects while preserving the strongest metadata.
    """

    unique: Dict[
        Tuple[Any, ...],
        PiRing,
    ] = {}

    for ring in rings:
        if not isinstance(ring, PiRing):
            raise TypeError(
                "rings must contain only PiRing objects."
            )

        key = get_pi_ring_identity_key(ring)
        current = unique.get(key)

        if current is None:
            unique[key] = ring
            continue

        current_confidence = _normalize_optional_numeric(
            current.metadata.get(
                "aromaticity_confidence"
            )
        )

        candidate_confidence = _normalize_optional_numeric(
            ring.metadata.get(
                "aromaticity_confidence"
            )
        )

        if (
            candidate_confidence is not None
            and (
                current_confidence is None
                or candidate_confidence > current_confidence
            )
        ):
            unique[key] = ring

    result = list(unique.values())

    result.sort(
        key=lambda ring: (
            ring.model_id or "",
            ring.chain_id or "",
            ring.residue_number
            if isinstance(ring.residue_number, int)
            else str(ring.residue_number or ""),
            ring.residue_name or "",
            ring.ring_index
            if ring.ring_index is not None
            else 0,
            tuple(
                get_atom_name(atom)
                for atom in ring.atoms
            ),
        )
    )

    for ring_index, ring in enumerate(result, start=1):
        ring.ring_index = ring_index

        if not ring.ring_id:
            ring.ring_id = ring.build_ring_id()

    return result


# -----------------------------------------------------------------------------
# 4.15. Separação entre anéis do receptor e do ligante
# -----------------------------------------------------------------------------

def detect_receptor_aromatic_rings(
    receptor: Any,
    *,
    config: Optional[PiAnalysisConfig] = None,
    receptor_residue_names: Optional[Collection[str]] = None,
) -> List[PiRing]:
    """
    Detect aromatic rings associated with a receptor.
    """

    return detect_aromatic_rings(
        receptor,
        config=config,
        participant_type=PARTICIPANT_RECEPTOR,
        receptor_residue_names=receptor_residue_names,
    )


def detect_ligand_aromatic_rings(
    ligand: Any,
    *,
    config: Optional[PiAnalysisConfig] = None,
    ligand_residue_names: Optional[Collection[str]] = None,
) -> List[PiRing]:
    """
    Detect aromatic rings associated with a ligand.
    """

    return detect_aromatic_rings(
        ligand,
        config=config,
        participant_type=PARTICIPANT_LIGAND,
        ligand_residue_names=ligand_residue_names,
    )


def detect_pi_analysis_rings(
    normalized_input: PiNormalizedInput,
    *,
    config: Optional[PiAnalysisConfig] = None,
) -> Tuple[List[PiRing], List[PiRing]]:
    """
    Detect receptor and ligand aromatic rings from normalized analysis input.
    """

    if not isinstance(normalized_input, PiNormalizedInput):
        raise TypeError(
            "normalized_input must be a PiNormalizedInput."
        )

    analysis_config = (
        config
        if config is not None
        else create_default_pi_config()
    )

    receptor_rings = detect_receptor_aromatic_rings(
        normalized_input.receptor_atoms,
        config=analysis_config,
    )

    ligand_rings = detect_ligand_aromatic_rings(
        normalized_input.ligand_atoms,
        config=analysis_config,
    )

    return receptor_rings, ligand_rings


# -----------------------------------------------------------------------------
# 4.16. Validação preliminar de anéis detectados
# -----------------------------------------------------------------------------

def validate_detected_ring(
    ring: PiRing,
    *,
    config: Optional[PiAnalysisConfig] = None,
    require_aromaticity: bool = True,
    require_coordinates: bool = True,
) -> Tuple[bool, Tuple[str, ...]]:
    """
    Validate a detected ring before geometric calculations.
    """

    if not isinstance(ring, PiRing):
        raise TypeError(
            "ring must be a PiRing."
        )

    analysis_config = (
        config
        if config is not None
        else create_default_pi_config()
    )

    messages: List[str] = []

    minimum_size = analysis_config.minimum_ring_size
    maximum_size = (
        analysis_config.maximum_fused_ring_size
        if ring.is_fused
        else analysis_config.maximum_ring_size
    )

    if not minimum_size <= ring.atom_count <= maximum_size:
        messages.append(
            f"Ring size {ring.atom_count} is outside the accepted "
            f"range {minimum_size}-{maximum_size}."
        )

    atom_collection_valid, atom_messages = validate_atom_collection(
        ring.atoms,
        minimum_atoms=3,
        require_coordinates=require_coordinates,
        skip_hydrogens=True,
    )

    if not atom_collection_valid:
        messages.extend(atom_messages)

    if require_aromaticity and not ring.is_fused:
        aromatic, source, confidence = estimate_ring_aromaticity(
            ring.atoms
        )

        if not aromatic:
            messages.append(
                "Ring does not provide sufficient aromaticity evidence."
            )

        ring.metadata.setdefault(
            "validation_aromaticity_source",
            source,
        )

        ring.metadata.setdefault(
            "validation_aromaticity_confidence",
            confidence,
        )

    if (
        ring.is_heteroaromatic
        and not analysis_config.allow_heteroaromatic_rings
    ):
        messages.append(
            "Heteroaromatic rings are disabled by configuration."
        )

    ring.valid = not messages
    ring.validation_messages = tuple(messages)

    return ring.valid, ring.validation_messages


def validate_detected_rings(
    rings: Iterable[PiRing],
    *,
    config: Optional[PiAnalysisConfig] = None,
    remove_invalid: bool = False,
) -> List[PiRing]:
    """
    Validate multiple detected rings.
    """

    validated: List[PiRing] = []

    for ring in rings:
        valid, _ = validate_detected_ring(
            ring,
            config=config,
        )

        if remove_invalid and not valid:
            continue

        validated.append(ring)

    return validated


# -----------------------------------------------------------------------------
# 4.17. Resumo da detecção de anéis
# -----------------------------------------------------------------------------

def summarize_detected_rings(
    rings: Iterable[PiRing],
) -> Dict[str, Any]:
    """
    Generate a compact summary of detected aromatic rings.
    """

    ring_list = list(rings)

    participant_distribution = Counter(
        ring.participant_type
        for ring in ring_list
    )

    residue_distribution = Counter(
        ring.residue_name or "UNK"
        for ring in ring_list
    )

    ring_size_distribution = Counter(
        ring.atom_count
        for ring in ring_list
    )

    aromaticity_source_distribution = Counter(
        ring.aromaticity_source or "unknown"
        for ring in ring_list
    )

    valid_rings = [
        ring
        for ring in ring_list
        if ring.valid
    ]

    return {
        "total_rings": len(ring_list),
        "valid_rings": len(valid_rings),
        "invalid_rings": len(ring_list) - len(valid_rings),
        "protein_rings": sum(
            1
            for ring in ring_list
            if ring.is_protein_ring
        ),
        "ligand_rings": sum(
            1
            for ring in ring_list
            if ring.is_ligand_ring
        ),
        "heteroaromatic_rings": sum(
            1
            for ring in ring_list
            if ring.is_heteroaromatic
        ),
        "fused_rings": sum(
            1
            for ring in ring_list
            if ring.is_fused
        ),
        "participant_distribution": dict(
            participant_distribution
        ),
        "residue_distribution": dict(
            residue_distribution
        ),
        "ring_size_distribution": dict(
            ring_size_distribution
        ),
        "aromaticity_source_distribution": dict(
            aromaticity_source_distribution
        ),
        "ring_ids": [
            ring.ring_id
            for ring in ring_list
        ],
    }

# -----------------------------------------------------------------------------
# End of section 4.
# -----------------------------------------------------------------------------


# =============================================================================
# 5. GEOMETRIA DOS ANÉIS AROMÁTICOS
# =============================================================================

# -----------------------------------------------------------------------------
# 5.1. Exceções específicas de geometria
# -----------------------------------------------------------------------------

class PiGeometryError(ValueError):
    """
    Base exception raised when pi-interaction geometry cannot be calculated.
    """


class PiDegenerateGeometryError(PiGeometryError):
    """
    Raised when an atom set does not define a valid plane or direction.
    """


# -----------------------------------------------------------------------------
# 5.2. Tipos auxiliares
# -----------------------------------------------------------------------------

Matrix3x3: TypeAlias = Tuple[
    Tuple[float, float, float],
    Tuple[float, float, float],
    Tuple[float, float, float],
]


@dataclass(frozen=True, slots=True)
class PiPlaneGeometry:
    """
    Geometric description of a fitted molecular plane.

    Parameters
    ----------
    centroid
        Arithmetic mean of the supplied coordinates.

    normal
        Unit vector perpendicular to the fitted plane.

    planarity_rmsd
        Root-mean-square perpendicular deviation from the plane.

    maximum_deviation
        Largest absolute perpendicular deviation.

    signed_deviations
        Signed perpendicular deviations for each input coordinate.

    eigenvalues
        Eigenvalues of the coordinate covariance matrix, when available.

    valid
        Whether the plane fit was successful.

    method
        Plane-fitting method used.

    warnings
        Non-fatal geometric warnings.
    """

    centroid: Coordinate3D
    normal: Vector3D

    planarity_rmsd: float
    maximum_deviation: float

    signed_deviations: Tuple[float, ...] = ()
    eigenvalues: Tuple[float, ...] = ()

    valid: bool = True
    method: str = "unknown"

    warnings: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "centroid",
            _coerce_coordinate3d(
                self.centroid,
                field_name="PiPlaneGeometry.centroid",
            ),
        )

        normal = normalize_vector(
            self.normal,
            strict=True,
        )

        assert normal is not None

        object.__setattr__(
            self,
            "normal",
            normal,
        )

        object.__setattr__(
            self,
            "planarity_rmsd",
            _coerce_non_negative_float(
                self.planarity_rmsd,
                field_name="PiPlaneGeometry.planarity_rmsd",
            ),
        )

        object.__setattr__(
            self,
            "maximum_deviation",
            _coerce_non_negative_float(
                self.maximum_deviation,
                field_name="PiPlaneGeometry.maximum_deviation",
            ),
        )

        deviations = tuple(
            float(value)
            for value in self.signed_deviations
        )

        if not all(isfinite(value) for value in deviations):
            raise ValueError(
                "PiPlaneGeometry.signed_deviations must be finite."
            )

        object.__setattr__(
            self,
            "signed_deviations",
            deviations,
        )

        eigenvalues = tuple(
            float(value)
            for value in self.eigenvalues
        )

        if not all(isfinite(value) for value in eigenvalues):
            raise ValueError(
                "PiPlaneGeometry.eigenvalues must be finite."
            )

        object.__setattr__(
            self,
            "eigenvalues",
            eigenvalues,
        )

        object.__setattr__(
            self,
            "valid",
            bool(self.valid),
        )

        object.__setattr__(
            self,
            "method",
            str(self.method).strip() or "unknown",
        )

        object.__setattr__(
            self,
            "warnings",
            _coerce_string_tuple(
                self.warnings,
                field_name="PiPlaneGeometry.warnings",
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the plane geometry into a serializable dictionary.
        """

        return {
            "centroid": list(self.centroid),
            "normal": list(self.normal),
            "planarity_rmsd": self.planarity_rmsd,
            "maximum_deviation": self.maximum_deviation,
            "signed_deviations": list(self.signed_deviations),
            "eigenvalues": list(self.eigenvalues),
            "valid": self.valid,
            "method": self.method,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True, slots=True)
class PiRingPairGeometry:
    """
    Relative geometry between two aromatic rings.
    """

    centroid_distance: float
    normal_angle: float
    acute_normal_angle: float

    ring_1_to_ring_2_vector: Vector3D

    ring_1_plane_height: float
    ring_2_plane_height: float

    ring_1_lateral_offset: float
    ring_2_lateral_offset: float

    mean_plane_height: float
    mean_lateral_offset: float

    minimum_atomic_distance: Optional[float] = None
    maximum_atomic_distance: Optional[float] = None

    valid: bool = True
    warnings: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        numeric_non_negative = (
            "centroid_distance",
            "normal_angle",
            "acute_normal_angle",
            "ring_1_plane_height",
            "ring_2_plane_height",
            "ring_1_lateral_offset",
            "ring_2_lateral_offset",
            "mean_plane_height",
            "mean_lateral_offset",
        )

        for field_name in numeric_non_negative:
            object.__setattr__(
                self,
                field_name,
                _coerce_non_negative_float(
                    getattr(self, field_name),
                    field_name=f"PiRingPairGeometry.{field_name}",
                ),
            )

        for field_name in (
            "minimum_atomic_distance",
            "maximum_atomic_distance",
        ):
            object.__setattr__(
                self,
                field_name,
                _coerce_optional_float(
                    getattr(self, field_name),
                    field_name=f"PiRingPairGeometry.{field_name}",
                    minimum=0.0,
                ),
            )

        vector = _coerce_coordinate3d(
            self.ring_1_to_ring_2_vector,
            field_name=(
                "PiRingPairGeometry.ring_1_to_ring_2_vector"
            ),
        )

        assert vector is not None

        object.__setattr__(
            self,
            "ring_1_to_ring_2_vector",
            vector,
        )

        object.__setattr__(
            self,
            "valid",
            bool(self.valid),
        )

        object.__setattr__(
            self,
            "warnings",
            _coerce_string_tuple(
                self.warnings,
                field_name="PiRingPairGeometry.warnings",
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the ring-pair geometry into a serializable dictionary.
        """

        return {
            "centroid_distance": self.centroid_distance,
            "normal_angle": self.normal_angle,
            "acute_normal_angle": self.acute_normal_angle,
            "ring_1_to_ring_2_vector": list(
                self.ring_1_to_ring_2_vector
            ),
            "ring_1_plane_height": self.ring_1_plane_height,
            "ring_2_plane_height": self.ring_2_plane_height,
            "ring_1_lateral_offset": self.ring_1_lateral_offset,
            "ring_2_lateral_offset": self.ring_2_lateral_offset,
            "mean_plane_height": self.mean_plane_height,
            "mean_lateral_offset": self.mean_lateral_offset,
            "minimum_atomic_distance": self.minimum_atomic_distance,
            "maximum_atomic_distance": self.maximum_atomic_distance,
            "valid": self.valid,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True, slots=True)
class PiPointRingGeometry:
    """
    Relative geometry between a point and an aromatic ring.

    This representation will later be used for cation-pi, anion-pi and
    sulfur-pi interactions.
    """

    point: Coordinate3D
    ring_centroid: Coordinate3D

    center_distance: float
    signed_plane_distance: float
    absolute_plane_distance: float
    radial_offset: float

    direction_vector: Vector3D
    direction_angle: Optional[float] = None

    valid: bool = True
    warnings: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "point",
            _coerce_coordinate3d(
                self.point,
                field_name="PiPointRingGeometry.point",
            ),
        )

        object.__setattr__(
            self,
            "ring_centroid",
            _coerce_coordinate3d(
                self.ring_centroid,
                field_name="PiPointRingGeometry.ring_centroid",
            ),
        )

        object.__setattr__(
            self,
            "direction_vector",
            _coerce_coordinate3d(
                self.direction_vector,
                field_name="PiPointRingGeometry.direction_vector",
            ),
        )

        for field_name in (
            "center_distance",
            "absolute_plane_distance",
            "radial_offset",
        ):
            object.__setattr__(
                self,
                field_name,
                _coerce_non_negative_float(
                    getattr(self, field_name),
                    field_name=f"PiPointRingGeometry.{field_name}",
                ),
            )

        signed_distance = float(self.signed_plane_distance)

        if not isfinite(signed_distance):
            raise ValueError(
                "signed_plane_distance must be finite."
            )

        object.__setattr__(
            self,
            "signed_plane_distance",
            signed_distance,
        )

        object.__setattr__(
            self,
            "direction_angle",
            _coerce_optional_float(
                self.direction_angle,
                field_name="PiPointRingGeometry.direction_angle",
                minimum=0.0,
                maximum=180.0,
            ),
        )

        object.__setattr__(
            self,
            "valid",
            bool(self.valid),
        )

        object.__setattr__(
            self,
            "warnings",
            _coerce_string_tuple(
                self.warnings,
                field_name="PiPointRingGeometry.warnings",
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the point-ring geometry into a dictionary.
        """

        return {
            "point": list(self.point),
            "ring_centroid": list(self.ring_centroid),
            "center_distance": self.center_distance,
            "signed_plane_distance": self.signed_plane_distance,
            "absolute_plane_distance": self.absolute_plane_distance,
            "radial_offset": self.radial_offset,
            "direction_vector": list(self.direction_vector),
            "direction_angle": self.direction_angle,
            "valid": self.valid,
            "warnings": list(self.warnings),
        }


# -----------------------------------------------------------------------------
# 5.3. Operações vetoriais básicas
# -----------------------------------------------------------------------------

def add_vectors(
    vector_1: Sequence[Number],
    vector_2: Sequence[Number],
) -> Vector3D:
    """
    Add two three-dimensional vectors.
    """

    first = _coerce_coordinate3d(
        vector_1,
        field_name="vector_1",
    )

    second = _coerce_coordinate3d(
        vector_2,
        field_name="vector_2",
    )

    assert first is not None
    assert second is not None

    return (
        first[0] + second[0],
        first[1] + second[1],
        first[2] + second[2],
    )


def subtract_vectors(
    vector_1: Sequence[Number],
    vector_2: Sequence[Number],
) -> Vector3D:
    """
    Subtract ``vector_2`` from ``vector_1``.
    """

    first = _coerce_coordinate3d(
        vector_1,
        field_name="vector_1",
    )

    second = _coerce_coordinate3d(
        vector_2,
        field_name="vector_2",
    )

    assert first is not None
    assert second is not None

    return (
        first[0] - second[0],
        first[1] - second[1],
        first[2] - second[2],
    )


def scale_vector(
    vector: Sequence[Number],
    scalar: Number,
) -> Vector3D:
    """
    Multiply a vector by a scalar.
    """

    normalized_vector = _coerce_coordinate3d(
        vector,
        field_name="vector",
    )

    assert normalized_vector is not None

    scalar_value = float(scalar)

    if not isfinite(scalar_value):
        raise ValueError(
            "scalar must be finite."
        )

    return (
        normalized_vector[0] * scalar_value,
        normalized_vector[1] * scalar_value,
        normalized_vector[2] * scalar_value,
    )


def dot_product(
    vector_1: Sequence[Number],
    vector_2: Sequence[Number],
) -> float:
    """
    Return the dot product of two vectors.
    """

    first = _coerce_coordinate3d(
        vector_1,
        field_name="vector_1",
    )

    second = _coerce_coordinate3d(
        vector_2,
        field_name="vector_2",
    )

    assert first is not None
    assert second is not None

    return (
        first[0] * second[0]
        + first[1] * second[1]
        + first[2] * second[2]
    )


def cross_product(
    vector_1: Sequence[Number],
    vector_2: Sequence[Number],
) -> Vector3D:
    """
    Return the cross product of two vectors.
    """

    first = _coerce_coordinate3d(
        vector_1,
        field_name="vector_1",
    )

    second = _coerce_coordinate3d(
        vector_2,
        field_name="vector_2",
    )

    assert first is not None
    assert second is not None

    return (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )


def vector_norm(
    vector: Sequence[Number],
) -> float:
    """
    Return the Euclidean norm of a vector.
    """

    normalized_vector = _coerce_coordinate3d(
        vector,
        field_name="vector",
    )

    assert normalized_vector is not None

    return sqrt(
        normalized_vector[0] ** 2
        + normalized_vector[1] ** 2
        + normalized_vector[2] ** 2
    )


def normalize_vector(
    vector: Sequence[Number],
    *,
    tolerance: float = DEFAULT_VECTOR_TOLERANCE,
    strict: bool = False,
) -> Optional[Vector3D]:
    """
    Convert a vector into a unit vector.

    Degenerate vectors return ``None`` unless ``strict`` is enabled.
    """

    normalized_vector = _coerce_coordinate3d(
        vector,
        field_name="vector",
    )

    assert normalized_vector is not None

    tolerance_value = _coerce_non_negative_float(
        tolerance,
        field_name="tolerance",
    )

    magnitude = vector_norm(normalized_vector)

    if magnitude <= tolerance_value:
        if strict:
            raise PiDegenerateGeometryError(
                "Cannot normalize a zero-length or degenerate vector."
            )

        return None

    return (
        normalized_vector[0] / magnitude,
        normalized_vector[1] / magnitude,
        normalized_vector[2] / magnitude,
    )


def distance_between_points(
    point_1: Sequence[Number],
    point_2: Sequence[Number],
) -> float:
    """
    Return the Euclidean distance between two points.
    """

    return vector_norm(
        subtract_vectors(
            point_1,
            point_2,
        )
    )


def squared_distance_between_points(
    point_1: Sequence[Number],
    point_2: Sequence[Number],
) -> float:
    """
    Return the squared Euclidean distance between two points.
    """

    delta = subtract_vectors(
        point_1,
        point_2,
    )

    return dot_product(delta, delta)


def midpoint(
    point_1: Sequence[Number],
    point_2: Sequence[Number],
) -> Coordinate3D:
    """
    Return the midpoint between two coordinates.
    """

    first = _coerce_coordinate3d(
        point_1,
        field_name="point_1",
    )

    second = _coerce_coordinate3d(
        point_2,
        field_name="point_2",
    )

    assert first is not None
    assert second is not None

    return (
        (first[0] + second[0]) / 2.0,
        (first[1] + second[1]) / 2.0,
        (first[2] + second[2]) / 2.0,
    )


# -----------------------------------------------------------------------------
# 5.4. Ângulos vetoriais
# -----------------------------------------------------------------------------

def angle_between_vectors(
    vector_1: Sequence[Number],
    vector_2: Sequence[Number],
    *,
    degrees: bool = True,
    acute: bool = False,
    strict: bool = False,
) -> Optional[float]:
    """
    Return the angle between two vectors.

    Parameters
    ----------
    acute
        Treat opposite normal directions as geometrically equivalent.
        When enabled, the returned angle is restricted to [0, 90].
    """

    first = normalize_vector(
        vector_1,
        strict=strict,
    )

    second = normalize_vector(
        vector_2,
        strict=strict,
    )

    if first is None or second is None:
        return None

    cosine_value = dot_product(first, second)

    if acute:
        cosine_value = abs(cosine_value)

    cosine_value = max(
        -1.0,
        min(1.0, cosine_value),
    )

    angle_radians = acos(cosine_value)

    if degrees:
        return angle_radians * 180.0 / pi

    return angle_radians


def acute_angle_between_vectors(
    vector_1: Sequence[Number],
    vector_2: Sequence[Number],
    *,
    strict: bool = False,
) -> Optional[float]:
    """
    Return the orientation-independent acute angle between vectors.
    """

    return angle_between_vectors(
        vector_1,
        vector_2,
        degrees=True,
        acute=True,
        strict=strict,
    )


def angle_between_planes(
    normal_1: Sequence[Number],
    normal_2: Sequence[Number],
    *,
    strict: bool = False,
) -> Optional[float]:
    """
    Return the acute angle between two planes.
    """

    return acute_angle_between_vectors(
        normal_1,
        normal_2,
        strict=strict,
    )


# -----------------------------------------------------------------------------
# 5.5. Centroide e dispersão espacial
# -----------------------------------------------------------------------------

def calculate_centroid(
    coordinates: Iterable[Sequence[Number]],
) -> Coordinate3D:
    """
    Calculate the arithmetic centroid of three-dimensional coordinates.
    """

    normalized_coordinates = tuple(
        _coerce_coordinate3d(
            coordinate,
            field_name="coordinate",
        )
        for coordinate in coordinates
    )

    normalized_coordinates = tuple(
        coordinate
        for coordinate in normalized_coordinates
        if coordinate is not None
    )

    if not normalized_coordinates:
        raise PiGeometryError(
            "At least one coordinate is required to calculate a centroid."
        )

    count = float(len(normalized_coordinates))

    return (
        sum(coordinate[0] for coordinate in normalized_coordinates) / count,
        sum(coordinate[1] for coordinate in normalized_coordinates) / count,
        sum(coordinate[2] for coordinate in normalized_coordinates) / count,
    )


def calculate_atom_centroid(
    atoms: Iterable[Any],
    *,
    use_scene_coordinates: bool = True,
    skip_invalid: bool = False,
) -> Coordinate3D:
    """
    Calculate the centroid of an atom collection.
    """

    coordinates = get_atom_coordinates(
        atoms,
        use_scene_coordinates=use_scene_coordinates,
        skip_invalid=skip_invalid,
    )

    return calculate_centroid(coordinates)


def calculate_radius_from_centroid(
    coordinates: Iterable[Sequence[Number]],
    centroid: Optional[Sequence[Number]] = None,
    *,
    method: str = "maximum",
) -> float:
    """
    Calculate a radial size estimate from a coordinate centroid.

    Supported methods are:

    - ``maximum``;
    - ``mean``;
    - ``rms``.
    """

    normalized_coordinates = tuple(
        _coerce_coordinate3d(
            coordinate,
            field_name="coordinate",
        )
        for coordinate in coordinates
    )

    normalized_coordinates = tuple(
        coordinate
        for coordinate in normalized_coordinates
        if coordinate is not None
    )

    if not normalized_coordinates:
        raise PiGeometryError(
            "At least one coordinate is required to calculate a radius."
        )

    normalized_centroid = (
        _coerce_coordinate3d(
            centroid,
            field_name="centroid",
        )
        if centroid is not None
        else calculate_centroid(normalized_coordinates)
    )

    assert normalized_centroid is not None

    distances = tuple(
        distance_between_points(
            coordinate,
            normalized_centroid,
        )
        for coordinate in normalized_coordinates
    )

    normalized_method = str(method).strip().lower()

    if normalized_method == "maximum":
        return max(distances)

    if normalized_method == "mean":
        return sum(distances) / len(distances)

    if normalized_method == "rms":
        return sqrt(
            sum(distance ** 2 for distance in distances)
            / len(distances)
        )

    raise ValueError(
        "method must be 'maximum', 'mean' or 'rms'."
    )


# -----------------------------------------------------------------------------
# 5.6. Matriz de covariância
# -----------------------------------------------------------------------------

def calculate_coordinate_covariance_matrix(
    coordinates: Iterable[Sequence[Number]],
    *,
    centroid: Optional[Sequence[Number]] = None,
) -> Matrix3x3:
    """
    Calculate the unweighted 3x3 covariance matrix of coordinates.
    """

    coordinate_tuple = tuple(
        _coerce_coordinate3d(
            coordinate,
            field_name="coordinate",
        )
        for coordinate in coordinates
    )

    coordinate_tuple = tuple(
        coordinate
        for coordinate in coordinate_tuple
        if coordinate is not None
    )

    if len(coordinate_tuple) < 2:
        raise PiGeometryError(
            "At least two coordinates are required for covariance."
        )

    normalized_centroid = (
        _coerce_coordinate3d(
            centroid,
            field_name="centroid",
        )
        if centroid is not None
        else calculate_centroid(coordinate_tuple)
    )

    assert normalized_centroid is not None

    xx = xy = xz = yy = yz = zz = 0.0

    for coordinate in coordinate_tuple:
        x = coordinate[0] - normalized_centroid[0]
        y = coordinate[1] - normalized_centroid[1]
        z = coordinate[2] - normalized_centroid[2]

        xx += x * x
        xy += x * y
        xz += x * z
        yy += y * y
        yz += y * z
        zz += z * z

    divisor = float(len(coordinate_tuple))

    return (
        (
            xx / divisor,
            xy / divisor,
            xz / divisor,
        ),
        (
            xy / divisor,
            yy / divisor,
            yz / divisor,
        ),
        (
            xz / divisor,
            yz / divisor,
            zz / divisor,
        ),
    )


# -----------------------------------------------------------------------------
# 5.7. Ajuste de plano com NumPy
# -----------------------------------------------------------------------------

def _fit_plane_with_numpy(
    coordinates: Sequence[Coordinate3D],
) -> PiPlaneGeometry:
    """
    Fit a plane using eigenvalue decomposition of the covariance matrix.
    """

    if not NUMPY_AVAILABLE:
        raise RuntimeError(
            "NumPy is unavailable."
        )

    coordinate_array = np.asarray(
        coordinates,
        dtype=float,
    )

    if coordinate_array.ndim != 2 or coordinate_array.shape[1] != 3:
        raise PiGeometryError(
            "Coordinates must define an N x 3 array."
        )

    centroid_array = np.mean(
        coordinate_array,
        axis=0,
    )

    centered = coordinate_array - centroid_array

    covariance = np.matmul(
        centered.T,
        centered,
    ) / float(len(coordinate_array))

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    minimum_index = int(np.argmin(eigenvalues))
    normal_array = eigenvectors[:, minimum_index]

    normal = normalize_vector(
        tuple(float(value) for value in normal_array),
        strict=True,
    )

    assert normal is not None

    centroid = tuple(
        float(value)
        for value in centroid_array
    )

    signed_deviations = tuple(
        dot_product(
            subtract_vectors(coordinate, centroid),
            normal,
        )
        for coordinate in coordinates
    )

    rmsd = sqrt(
        sum(value ** 2 for value in signed_deviations)
        / len(signed_deviations)
    )

    maximum_deviation = max(
        abs(value)
        for value in signed_deviations
    )

    return PiPlaneGeometry(
        centroid=centroid,
        normal=normal,
        planarity_rmsd=rmsd,
        maximum_deviation=maximum_deviation,
        signed_deviations=signed_deviations,
        eigenvalues=tuple(
            float(value)
            for value in eigenvalues
        ),
        valid=True,
        method="covariance_eigendecomposition",
    )


# -----------------------------------------------------------------------------
# 5.8. Ajuste de plano sem NumPy
# -----------------------------------------------------------------------------

def _select_stable_plane_vectors(
    coordinates: Sequence[Coordinate3D],
    centroid: Coordinate3D,
) -> Tuple[Vector3D, Vector3D]:
    """
    Select two non-collinear coordinate vectors around a centroid.
    """

    centered_vectors = [
        subtract_vectors(coordinate, centroid)
        for coordinate in coordinates
    ]

    centered_vectors.sort(
        key=vector_norm,
        reverse=True,
    )

    if not centered_vectors:
        raise PiDegenerateGeometryError(
            "No centered vectors are available."
        )

    first_vector = centered_vectors[0]

    best_second: Optional[Vector3D] = None
    best_cross_norm = 0.0

    for candidate in centered_vectors[1:]:
        cross = cross_product(
            first_vector,
            candidate,
        )

        cross_norm = vector_norm(cross)

        if cross_norm > best_cross_norm:
            best_cross_norm = cross_norm
            best_second = candidate

    if (
        best_second is None
        or best_cross_norm <= DEFAULT_VECTOR_TOLERANCE
    ):
        raise PiDegenerateGeometryError(
            "Coordinates are collinear or geometrically degenerate."
        )

    return first_vector, best_second


def _fit_plane_without_numpy(
    coordinates: Sequence[Coordinate3D],
) -> PiPlaneGeometry:
    """
    Fit an approximate plane using stable cross products.

    This fallback does not require NumPy. For ordinary aromatic rings it
    provides a reliable plane normal, while NumPy remains the preferred method.
    """

    centroid = calculate_centroid(coordinates)

    first_vector, second_vector = _select_stable_plane_vectors(
        coordinates,
        centroid,
    )

    normal = normalize_vector(
        cross_product(
            first_vector,
            second_vector,
        ),
        strict=True,
    )

    assert normal is not None

    signed_deviations = tuple(
        dot_product(
            subtract_vectors(coordinate, centroid),
            normal,
        )
        for coordinate in coordinates
    )

    rmsd = sqrt(
        sum(value ** 2 for value in signed_deviations)
        / len(signed_deviations)
    )

    maximum_deviation = max(
        abs(value)
        for value in signed_deviations
    )

    return PiPlaneGeometry(
        centroid=centroid,
        normal=normal,
        planarity_rmsd=rmsd,
        maximum_deviation=maximum_deviation,
        signed_deviations=signed_deviations,
        eigenvalues=(),
        valid=True,
        method="stable_cross_product",
        warnings=(
            "NumPy was unavailable; plane fitting used the "
            "cross-product fallback.",
        ),
    )


# -----------------------------------------------------------------------------
# 5.9. Ajuste público de plano
# -----------------------------------------------------------------------------

def fit_plane_to_coordinates(
    coordinates: Iterable[Sequence[Number]],
    *,
    prefer_numpy: bool = True,
    strict: bool = True,
) -> Optional[PiPlaneGeometry]:
    """
    Fit a best-estimate plane to three-dimensional coordinates.

    At least three non-collinear points are required.
    """

    coordinate_tuple = tuple(
        _coerce_coordinate3d(
            coordinate,
            field_name="coordinate",
        )
        for coordinate in coordinates
    )

    coordinate_tuple = tuple(
        coordinate
        for coordinate in coordinate_tuple
        if coordinate is not None
    )

    if len(coordinate_tuple) < 3:
        if strict:
            raise PiGeometryError(
                "At least three coordinates are required to fit a plane."
            )

        return None

    try:
        if prefer_numpy and NUMPY_AVAILABLE:
            return _fit_plane_with_numpy(coordinate_tuple)

        return _fit_plane_without_numpy(coordinate_tuple)

    except (
        PiGeometryError,
        PiDegenerateGeometryError,
        ValueError,
        ArithmeticError,
    ):
        if strict:
            raise

        return None


def fit_plane_to_atoms(
    atoms: Iterable[Any],
    *,
    use_scene_coordinates: bool = True,
    skip_invalid: bool = False,
    prefer_numpy: bool = True,
    strict: bool = True,
) -> Optional[PiPlaneGeometry]:
    """
    Fit a plane to an atom collection.
    """

    coordinates = get_atom_coordinates(
        atoms,
        use_scene_coordinates=use_scene_coordinates,
        skip_invalid=skip_invalid,
    )

    return fit_plane_to_coordinates(
        coordinates,
        prefer_numpy=prefer_numpy,
        strict=strict,
    )


# -----------------------------------------------------------------------------
# 5.10. Orientação determinística de vetores normais
# -----------------------------------------------------------------------------

def orient_normal_deterministically(
    normal: Sequence[Number],
    *,
    reference_vector: Optional[Sequence[Number]] = None,
) -> Vector3D:
    """
    Orient a plane normal deterministically.

    Plane normals ``n`` and ``-n`` describe the same plane. This function
    prevents unstable serialization by selecting one orientation consistently.

    When ``reference_vector`` is provided, the normal is oriented toward it.
    Otherwise, the first non-negligible component is made positive.
    """

    normalized = normalize_vector(
        normal,
        strict=True,
    )

    assert normalized is not None

    if reference_vector is not None:
        reference = normalize_vector(reference_vector)

        if (
            reference is not None
            and dot_product(normalized, reference) < 0.0
        ):
            return scale_vector(normalized, -1.0)

        return normalized

    for component in normalized:
        if abs(component) <= DEFAULT_VECTOR_TOLERANCE:
            continue

        if component < 0.0:
            return scale_vector(normalized, -1.0)

        break

    return normalized


def align_normal_to_reference(
    normal: Sequence[Number],
    reference_normal: Sequence[Number],
) -> Vector3D:
    """
    Orient a normal vector in the same hemisphere as a reference normal.
    """

    normalized = normalize_vector(
        normal,
        strict=True,
    )

    reference = normalize_vector(
        reference_normal,
        strict=True,
    )

    assert normalized is not None
    assert reference is not None

    if dot_product(normalized, reference) < 0.0:
        return scale_vector(normalized, -1.0)

    return normalized


# -----------------------------------------------------------------------------
# 5.11. Distância de ponto a plano
# -----------------------------------------------------------------------------

def signed_distance_to_plane(
    point: Sequence[Number],
    plane_centroid: Sequence[Number],
    plane_normal: Sequence[Number],
) -> float:
    """
    Return the signed perpendicular distance from a point to a plane.
    """

    normalized_normal = normalize_vector(
        plane_normal,
        strict=True,
    )

    assert normalized_normal is not None

    displacement = subtract_vectors(
        point,
        plane_centroid,
    )

    return dot_product(
        displacement,
        normalized_normal,
    )


def absolute_distance_to_plane(
    point: Sequence[Number],
    plane_centroid: Sequence[Number],
    plane_normal: Sequence[Number],
) -> float:
    """
    Return the absolute perpendicular distance from a point to a plane.
    """

    return abs(
        signed_distance_to_plane(
            point,
            plane_centroid,
            plane_normal,
        )
    )


def project_point_onto_plane(
    point: Sequence[Number],
    plane_centroid: Sequence[Number],
    plane_normal: Sequence[Number],
) -> Coordinate3D:
    """
    Orthogonally project a point onto a plane.
    """

    signed_distance = signed_distance_to_plane(
        point,
        plane_centroid,
        plane_normal,
    )

    normalized_normal = normalize_vector(
        plane_normal,
        strict=True,
    )

    assert normalized_normal is not None

    return subtract_vectors(
        point,
        scale_vector(
            normalized_normal,
            signed_distance,
        ),
    )


def calculate_radial_offset_from_plane_axis(
    point: Sequence[Number],
    plane_centroid: Sequence[Number],
    plane_normal: Sequence[Number],
) -> float:
    """
    Return the lateral displacement of a point from a plane-normal axis.
    """

    projected_point = project_point_onto_plane(
        point,
        plane_centroid,
        plane_normal,
    )

    return distance_between_points(
        projected_point,
        plane_centroid,
    )


# -----------------------------------------------------------------------------
# 5.12. Distâncias atômicas
# -----------------------------------------------------------------------------

def calculate_pairwise_atomic_distances(
    atoms_1: Iterable[Any],
    atoms_2: Iterable[Any],
    *,
    use_scene_coordinates: bool = True,
    skip_invalid: bool = False,
) -> Tuple[float, ...]:
    """
    Calculate all pairwise distances between two atom collections.
    """

    first_coordinates = get_atom_coordinates(
        atoms_1,
        use_scene_coordinates=use_scene_coordinates,
        skip_invalid=skip_invalid,
    )

    second_coordinates = get_atom_coordinates(
        atoms_2,
        use_scene_coordinates=use_scene_coordinates,
        skip_invalid=skip_invalid,
    )

    if not first_coordinates or not second_coordinates:
        return ()

    return tuple(
        distance_between_points(
            first_coordinate,
            second_coordinate,
        )
        for first_coordinate in first_coordinates
        for second_coordinate in second_coordinates
    )


def calculate_minimum_atomic_distance(
    atoms_1: Iterable[Any],
    atoms_2: Iterable[Any],
    *,
    use_scene_coordinates: bool = True,
    skip_invalid: bool = False,
) -> Optional[float]:
    """
    Return the shortest pairwise atomic distance.
    """

    distances = calculate_pairwise_atomic_distances(
        atoms_1,
        atoms_2,
        use_scene_coordinates=use_scene_coordinates,
        skip_invalid=skip_invalid,
    )

    if not distances:
        return None

    return min(distances)


def calculate_maximum_atomic_distance(
    atoms_1: Iterable[Any],
    atoms_2: Iterable[Any],
    *,
    use_scene_coordinates: bool = True,
    skip_invalid: bool = False,
) -> Optional[float]:
    """
    Return the largest pairwise atomic distance.
    """

    distances = calculate_pairwise_atomic_distances(
        atoms_1,
        atoms_2,
        use_scene_coordinates=use_scene_coordinates,
        skip_invalid=skip_invalid,
    )

    if not distances:
        return None

    return max(distances)


def find_closest_atom_pair(
    atoms_1: Iterable[Any],
    atoms_2: Iterable[Any],
    *,
    use_scene_coordinates: bool = True,
    skip_invalid: bool = False,
) -> Optional[Tuple[Any, Any, float]]:
    """
    Return the closest atom pair and its distance.
    """

    first_atoms = tuple(atoms_1)
    second_atoms = tuple(atoms_2)

    closest_pair: Optional[Tuple[Any, Any, float]] = None

    for atom_1 in first_atoms:
        coordinate_1 = get_atom_coordinate(
            atom_1,
            use_scene_coordinates=use_scene_coordinates,
            strict=not skip_invalid,
        )

        if coordinate_1 is None:
            continue

        for atom_2 in second_atoms:
            coordinate_2 = get_atom_coordinate(
                atom_2,
                use_scene_coordinates=use_scene_coordinates,
                strict=not skip_invalid,
            )

            if coordinate_2 is None:
                continue

            distance = distance_between_points(
                coordinate_1,
                coordinate_2,
            )

            if (
                closest_pair is None
                or distance < closest_pair[2]
            ):
                closest_pair = (
                    atom_1,
                    atom_2,
                    distance,
                )

    return closest_pair


# -----------------------------------------------------------------------------
# 5.13. Geometria completa de um PiRing
# -----------------------------------------------------------------------------

def calculate_pi_ring_geometry(
    ring: PiRing,
    *,
    config: Optional[PiAnalysisConfig] = None,
    use_scene_coordinates: bool = True,
    prefer_numpy: bool = True,
    orient_normal: bool = True,
    strict: bool = False,
    update_ring: bool = True,
) -> Optional[PiPlaneGeometry]:
    """
    Calculate and optionally attach geometric properties to a ``PiRing``.

    The following fields are updated:

    - ``centroid``;
    - ``normal``;
    - ``planarity_rmsd``;
    - ``maximum_plane_deviation``;
    - ``radius``;
    - ``valid``;
    - ``validation_messages``.
    """

    if not isinstance(ring, PiRing):
        raise TypeError(
            "ring must be a PiRing."
        )

    analysis_config = (
        config
        if config is not None
        else create_default_pi_config()
    )

    warnings_list: List[str] = []

    try:
        atoms = require_valid_atom_collection(
            ring.atoms,
            minimum_atoms=3,
            require_coordinates=True,
            context=f"ring {ring.ring_id or '<unknown>'}",
        )

        plane_geometry = fit_plane_to_atoms(
            atoms,
            use_scene_coordinates=use_scene_coordinates,
            skip_invalid=False,
            prefer_numpy=prefer_numpy,
            strict=True,
        )

        assert plane_geometry is not None

        normal = plane_geometry.normal

        if orient_normal:
            normal = orient_normal_deterministically(normal)

        coordinates = get_atom_coordinates(
            atoms,
            use_scene_coordinates=use_scene_coordinates,
            skip_invalid=False,
        )

        radius = calculate_radius_from_centroid(
            coordinates,
            plane_geometry.centroid,
            method="maximum",
        )

        if (
            plane_geometry.planarity_rmsd
            > analysis_config.preferred_ring_planarity_rmsd
        ):
            warnings_list.append(
                "Ring planarity RMSD exceeds the preferred threshold."
            )

        if (
            plane_geometry.maximum_deviation
            > analysis_config.maximum_ring_atom_deviation
        ):
            warnings_list.append(
                "Maximum atomic plane deviation exceeds the "
                "configured threshold."
            )

        valid = (
            plane_geometry.planarity_rmsd
            <= analysis_config.maximum_ring_planarity_rmsd
            and plane_geometry.maximum_deviation
            <= analysis_config.maximum_ring_atom_deviation
        )

        if not valid:
            warnings_list.append(
                "Ring geometry was rejected because its planarity "
                "exceeds the configured limits."
            )

        if update_ring:
            ring.centroid = plane_geometry.centroid
            ring.normal = normal
            ring.planarity_rmsd = (
                plane_geometry.planarity_rmsd
            )
            ring.maximum_plane_deviation = (
                plane_geometry.maximum_deviation
            )
            ring.radius = radius
            ring.valid = valid

            combined_messages = list(
                ring.validation_messages
            )

            combined_messages.extend(
                message
                for message in warnings_list
                if message not in combined_messages
            )

            ring.validation_messages = tuple(
                combined_messages
            )

            ring.metadata.update(
                {
                    "plane_fit_method": plane_geometry.method,
                    "signed_plane_deviations": list(
                        plane_geometry.signed_deviations
                    ),
                    "plane_eigenvalues": list(
                        plane_geometry.eigenvalues
                    ),
                    "geometry_calculated": True,
                }
            )

        return PiPlaneGeometry(
            centroid=plane_geometry.centroid,
            normal=normal,
            planarity_rmsd=plane_geometry.planarity_rmsd,
            maximum_deviation=plane_geometry.maximum_deviation,
            signed_deviations=plane_geometry.signed_deviations,
            eigenvalues=plane_geometry.eigenvalues,
            valid=valid,
            method=plane_geometry.method,
            warnings=tuple(
                list(plane_geometry.warnings)
                + warnings_list
            ),
        )

    except (
        PiGeometryError,
        PiAtomAccessError,
        PiCoordinateError,
        TypeError,
        ValueError,
        ArithmeticError,
    ) as exc:
        if update_ring:
            ring.valid = False

            messages = list(
                ring.validation_messages
            )

            messages.append(
                f"Geometry calculation failed: {exc}"
            )

            ring.validation_messages = tuple(messages)

            ring.metadata["geometry_calculated"] = False
            ring.metadata["geometry_error"] = str(exc)

        if strict:
            raise

        return None


def calculate_pi_ring_geometries(
    rings: Iterable[PiRing],
    *,
    config: Optional[PiAnalysisConfig] = None,
    use_scene_coordinates: bool = True,
    prefer_numpy: bool = True,
    remove_invalid: bool = False,
    strict: bool = False,
) -> List[PiRing]:
    """
    Calculate geometry for multiple rings.
    """

    processed: List[PiRing] = []

    for ring in rings:
        geometry = calculate_pi_ring_geometry(
            ring,
            config=config,
            use_scene_coordinates=use_scene_coordinates,
            prefer_numpy=prefer_numpy,
            strict=strict,
            update_ring=True,
        )

        if remove_invalid and (
            geometry is None
            or not ring.valid
        ):
            continue

        processed.append(ring)

    return processed


# -----------------------------------------------------------------------------
# 5.14. Garantia de geometria disponível
# -----------------------------------------------------------------------------

def ensure_pi_ring_geometry(
    ring: PiRing,
    *,
    config: Optional[PiAnalysisConfig] = None,
    strict: bool = True,
) -> PiRing:
    """
    Ensure that a ring has centroid, normal and planarity information.
    """

    if not isinstance(ring, PiRing):
        raise TypeError(
            "ring must be a PiRing."
        )

    if ring.has_complete_geometry:
        return ring

    geometry = calculate_pi_ring_geometry(
        ring,
        config=config,
        strict=strict,
        update_ring=True,
    )

    if geometry is None or not ring.has_complete_geometry:
        raise PiGeometryError(
            f"Could not calculate geometry for ring "
            f"{ring.ring_id or '<unknown>'}."
        )

    return ring


def ensure_pi_ring_geometries(
    rings: Iterable[PiRing],
    *,
    config: Optional[PiAnalysisConfig] = None,
    remove_invalid: bool = True,
    strict: bool = False,
) -> List[PiRing]:
    """
    Ensure complete geometry for a ring collection.
    """

    result: List[PiRing] = []

    for ring in rings:
        try:
            ensure_pi_ring_geometry(
                ring,
                config=config,
                strict=True,
            )

        except PiGeometryError:
            if strict:
                raise

            if remove_invalid:
                continue

        if remove_invalid and not ring.valid:
            continue

        result.append(ring)

    return result


# -----------------------------------------------------------------------------
# 5.15. Geometria relativa entre dois anéis
# -----------------------------------------------------------------------------

def calculate_pi_ring_pair_geometry(
    ring_1: PiRing,
    ring_2: PiRing,
    *,
    config: Optional[PiAnalysisConfig] = None,
    calculate_atomic_distances: bool = True,
    strict: bool = False,
) -> Optional[PiRingPairGeometry]:
    """
    Calculate the relative geometry between two aromatic rings.

    The lateral offset is calculated separately relative to each ring plane,
    then averaged. This is more robust than using only one ring as reference.
    """

    if not isinstance(ring_1, PiRing):
        raise TypeError(
            "ring_1 must be a PiRing."
        )

    if not isinstance(ring_2, PiRing):
        raise TypeError(
            "ring_2 must be a PiRing."
        )

    try:
        ensure_pi_ring_geometry(
            ring_1,
            config=config,
            strict=True,
        )

        ensure_pi_ring_geometry(
            ring_2,
            config=config,
            strict=True,
        )

        assert ring_1.centroid is not None
        assert ring_2.centroid is not None
        assert ring_1.normal is not None
        assert ring_2.normal is not None

        displacement = subtract_vectors(
            ring_2.centroid,
            ring_1.centroid,
        )

        centroid_distance = vector_norm(displacement)

        normal_angle = angle_between_vectors(
            ring_1.normal,
            ring_2.normal,
            acute=False,
            strict=True,
        )

        acute_normal_angle = acute_angle_between_vectors(
            ring_1.normal,
            ring_2.normal,
            strict=True,
        )

        assert normal_angle is not None
        assert acute_normal_angle is not None

        ring_1_signed_height = signed_distance_to_plane(
            ring_2.centroid,
            ring_1.centroid,
            ring_1.normal,
        )

        ring_2_signed_height = signed_distance_to_plane(
            ring_1.centroid,
            ring_2.centroid,
            ring_2.normal,
        )

        ring_1_plane_height = abs(ring_1_signed_height)
        ring_2_plane_height = abs(ring_2_signed_height)

        ring_1_lateral_offset = (
            calculate_radial_offset_from_plane_axis(
                ring_2.centroid,
                ring_1.centroid,
                ring_1.normal,
            )
        )

        ring_2_lateral_offset = (
            calculate_radial_offset_from_plane_axis(
                ring_1.centroid,
                ring_2.centroid,
                ring_2.normal,
            )
        )

        minimum_atomic_distance: Optional[float] = None
        maximum_atomic_distance: Optional[float] = None

        if calculate_atomic_distances:
            minimum_atomic_distance = (
                calculate_minimum_atomic_distance(
                    ring_1.atoms,
                    ring_2.atoms,
                    skip_invalid=False,
                )
            )

            maximum_atomic_distance = (
                calculate_maximum_atomic_distance(
                    ring_1.atoms,
                    ring_2.atoms,
                    skip_invalid=False,
                )
            )

        warnings_list: List[str] = []

        if centroid_distance <= DEFAULT_VECTOR_TOLERANCE:
            warnings_list.append(
                "Ring centroids are coincident or nearly coincident."
            )

        return PiRingPairGeometry(
            centroid_distance=centroid_distance,
            normal_angle=normal_angle,
            acute_normal_angle=acute_normal_angle,
            ring_1_to_ring_2_vector=displacement,
            ring_1_plane_height=ring_1_plane_height,
            ring_2_plane_height=ring_2_plane_height,
            ring_1_lateral_offset=ring_1_lateral_offset,
            ring_2_lateral_offset=ring_2_lateral_offset,
            mean_plane_height=(
                ring_1_plane_height
                + ring_2_plane_height
            ) / 2.0,
            mean_lateral_offset=(
                ring_1_lateral_offset
                + ring_2_lateral_offset
            ) / 2.0,
            minimum_atomic_distance=minimum_atomic_distance,
            maximum_atomic_distance=maximum_atomic_distance,
            valid=True,
            warnings=tuple(warnings_list),
        )

    except (
        PiGeometryError,
        PiAtomAccessError,
        PiCoordinateError,
        TypeError,
        ValueError,
        ArithmeticError,
    ):
        if strict:
            raise

        return None


# -----------------------------------------------------------------------------
# 5.16. Geometria entre um ponto e um anel
# -----------------------------------------------------------------------------

def calculate_point_ring_geometry(
    point: Sequence[Number],
    ring: PiRing,
    *,
    direction_vector: Optional[Sequence[Number]] = None,
    config: Optional[PiAnalysisConfig] = None,
    strict: bool = False,
) -> Optional[PiPointRingGeometry]:
    """
    Calculate the geometry between a point and an aromatic ring.

    Parameters
    ----------
    point
        Center of a charged, amide or sulfur-containing group.

    ring
        Aromatic ring.

    direction_vector
        Optional chemically meaningful group direction. When supplied, its
        angle relative to the ring normal is calculated.
    """

    if not isinstance(ring, PiRing):
        raise TypeError(
            "ring must be a PiRing."
        )

    try:
        normalized_point = _coerce_coordinate3d(
            point,
            field_name="point",
        )

        assert normalized_point is not None

        ensure_pi_ring_geometry(
            ring,
            config=config,
            strict=True,
        )

        assert ring.centroid is not None
        assert ring.normal is not None

        center_vector = subtract_vectors(
            normalized_point,
            ring.centroid,
        )

        center_distance = vector_norm(center_vector)

        signed_plane_distance = signed_distance_to_plane(
            normalized_point,
            ring.centroid,
            ring.normal,
        )

        absolute_plane_distance = abs(
            signed_plane_distance
        )

        radial_offset = (
            calculate_radial_offset_from_plane_axis(
                normalized_point,
                ring.centroid,
                ring.normal,
            )
        )

        direction_angle: Optional[float] = None

        if direction_vector is not None:
            direction_angle = acute_angle_between_vectors(
                direction_vector,
                ring.normal,
                strict=False,
            )

        warnings_list: List[str] = []

        if center_distance <= DEFAULT_VECTOR_TOLERANCE:
            warnings_list.append(
                "The supplied point is coincident with the ring centroid."
            )

        return PiPointRingGeometry(
            point=normalized_point,
            ring_centroid=ring.centroid,
            center_distance=center_distance,
            signed_plane_distance=signed_plane_distance,
            absolute_plane_distance=absolute_plane_distance,
            radial_offset=radial_offset,
            direction_vector=center_vector,
            direction_angle=direction_angle,
            valid=True,
            warnings=tuple(warnings_list),
        )

    except (
        PiGeometryError,
        PiAtomAccessError,
        PiCoordinateError,
        TypeError,
        ValueError,
        ArithmeticError,
    ):
        if strict:
            raise

        return None


# -----------------------------------------------------------------------------
# 5.17. Geometria entre grupo planar e anel
# -----------------------------------------------------------------------------

def calculate_planar_group_ring_geometry(
    group_center: Sequence[Number],
    group_normal: Sequence[Number],
    ring: PiRing,
    *,
    config: Optional[PiAnalysisConfig] = None,
    strict: bool = False,
) -> Optional[Dict[str, float]]:
    """
    Calculate geometry between a planar functional group and a ring.

    This helper will be used primarily for amide-pi analysis.
    """

    try:
        normalized_center = _coerce_coordinate3d(
            group_center,
            field_name="group_center",
        )

        normalized_group_normal = normalize_vector(
            group_normal,
            strict=True,
        )

        assert normalized_center is not None
        assert normalized_group_normal is not None

        ensure_pi_ring_geometry(
            ring,
            config=config,
            strict=True,
        )

        assert ring.centroid is not None
        assert ring.normal is not None

        point_geometry = calculate_point_ring_geometry(
            normalized_center,
            ring,
            config=config,
            strict=True,
        )

        assert point_geometry is not None

        plane_angle = angle_between_planes(
            normalized_group_normal,
            ring.normal,
            strict=True,
        )

        assert plane_angle is not None

        return {
            "centroid_distance": (
                point_geometry.center_distance
            ),
            "plane_height": (
                point_geometry.absolute_plane_distance
            ),
            "radial_offset": (
                point_geometry.radial_offset
            ),
            "plane_angle": plane_angle,
        }

    except (
        PiGeometryError,
        TypeError,
        ValueError,
        ArithmeticError,
    ):
        if strict:
            raise

        return None


# -----------------------------------------------------------------------------
# 5.18. Comparação geométrica entre anéis
# -----------------------------------------------------------------------------

def pi_rings_have_similar_geometry(
    ring_1: PiRing,
    ring_2: PiRing,
    *,
    centroid_tolerance: float = (
        DEFAULT_CENTROID_DEDUPLICATION_TOLERANCE
    ),
    normal_angle_tolerance: float = (
        DEFAULT_NORMAL_DEDUPLICATION_ANGLE
    ),
    config: Optional[PiAnalysisConfig] = None,
) -> bool:
    """
    Return whether two rings occupy nearly equivalent geometric positions.
    """

    try:
        ensure_pi_ring_geometry(
            ring_1,
            config=config,
            strict=True,
        )

        ensure_pi_ring_geometry(
            ring_2,
            config=config,
            strict=True,
        )

    except PiGeometryError:
        return False

    assert ring_1.centroid is not None
    assert ring_2.centroid is not None
    assert ring_1.normal is not None
    assert ring_2.normal is not None

    centroid_distance = distance_between_points(
        ring_1.centroid,
        ring_2.centroid,
    )

    if centroid_distance > centroid_tolerance:
        return False

    normal_angle = acute_angle_between_vectors(
        ring_1.normal,
        ring_2.normal,
    )

    if normal_angle is None:
        return False

    return normal_angle <= normal_angle_tolerance


def deduplicate_pi_rings_by_geometry(
    rings: Iterable[PiRing],
    *,
    config: Optional[PiAnalysisConfig] = None,
    centroid_tolerance: Optional[float] = None,
    normal_angle_tolerance: Optional[float] = None,
    preserve_fused_distinction: bool = True,
) -> List[PiRing]:
    """
    Remove rings occupying geometrically equivalent positions.
    """

    analysis_config = (
        config
        if config is not None
        else create_default_pi_config()
    )

    centroid_limit = (
        centroid_tolerance
        if centroid_tolerance is not None
        else analysis_config.centroid_deduplication_tolerance
    )

    angle_limit = (
        normal_angle_tolerance
        if normal_angle_tolerance is not None
        else analysis_config.normal_deduplication_angle
    )

    prepared_rings = ensure_pi_ring_geometries(
        rings,
        config=analysis_config,
        remove_invalid=True,
        strict=False,
    )

    unique: List[PiRing] = []

    for candidate in prepared_rings:
        duplicate_found = False

        for existing in unique:
            if (
                preserve_fused_distinction
                and candidate.is_fused != existing.is_fused
            ):
                continue

            if pi_rings_have_similar_geometry(
                candidate,
                existing,
                centroid_tolerance=centroid_limit,
                normal_angle_tolerance=angle_limit,
                config=analysis_config,
            ):
                duplicate_found = True

                candidate.metadata[
                    "geometry_duplicate_of"
                ] = existing.ring_id

                break

        if not duplicate_found:
            unique.append(candidate)

    for ring_index, ring in enumerate(unique, start=1):
        ring.ring_index = ring_index

    return unique


# -----------------------------------------------------------------------------
# 5.19. Classificação preliminar da planaridade
# -----------------------------------------------------------------------------

def classify_ring_planarity(
    ring_or_rmsd: Union[PiRing, Number],
    *,
    config: Optional[PiAnalysisConfig] = None,
) -> str:
    """
    Classify ring planar quality as preferred, acceptable or rejected.
    """

    analysis_config = (
        config
        if config is not None
        else create_default_pi_config()
    )

    if isinstance(ring_or_rmsd, PiRing):
        rmsd = ring_or_rmsd.planarity_rmsd

    else:
        rmsd = _normalize_optional_numeric(
            ring_or_rmsd
        )

    if rmsd is None:
        return GEOMETRY_REJECTED

    if rmsd <= analysis_config.preferred_ring_planarity_rmsd:
        return GEOMETRY_OPTIMAL

    if rmsd <= analysis_config.maximum_ring_planarity_rmsd:
        return GEOMETRY_FAVORABLE

    return GEOMETRY_REJECTED


def calculate_ring_planarity_score(
    ring_or_rmsd: Union[PiRing, Number],
    *,
    config: Optional[PiAnalysisConfig] = None,
) -> float:
    """
    Convert ring planarity RMSD into a normalized score.

    The score equals 1.0 for ideal planar geometry and decreases linearly to
    zero at the configured maximum accepted RMSD.
    """

    analysis_config = (
        config
        if config is not None
        else create_default_pi_config()
    )

    if isinstance(ring_or_rmsd, PiRing):
        rmsd = ring_or_rmsd.planarity_rmsd

    else:
        rmsd = _normalize_optional_numeric(
            ring_or_rmsd
        )

    if rmsd is None:
        return 0.0

    maximum_rmsd = (
        analysis_config.maximum_ring_planarity_rmsd
    )

    if maximum_rmsd <= 0.0:
        return 1.0 if rmsd <= 0.0 else 0.0

    return max(
        0.0,
        min(
            1.0,
            1.0 - rmsd / maximum_rmsd,
        ),
    )


# -----------------------------------------------------------------------------
# 5.20. Validação geométrica de anéis
# -----------------------------------------------------------------------------

def validate_pi_ring_geometry(
    ring: PiRing,
    *,
    config: Optional[PiAnalysisConfig] = None,
    calculate_if_missing: bool = True,
) -> Tuple[bool, Tuple[str, ...]]:
    """
    Validate centroid, normal, planarity and radius of a ring.
    """

    if not isinstance(ring, PiRing):
        raise TypeError(
            "ring must be a PiRing."
        )

    analysis_config = (
        config
        if config is not None
        else create_default_pi_config()
    )

    messages: List[str] = []

    if (
        calculate_if_missing
        and not ring.has_complete_geometry
    ):
        calculate_pi_ring_geometry(
            ring,
            config=analysis_config,
            strict=False,
            update_ring=True,
        )

    if ring.centroid is None:
        messages.append(
            "Ring centroid is unavailable."
        )

    if ring.normal is None:
        messages.append(
            "Ring normal is unavailable."
        )

    elif normalize_vector(ring.normal) is None:
        messages.append(
            "Ring normal is degenerate."
        )

    if ring.planarity_rmsd is None:
        messages.append(
            "Ring planarity RMSD is unavailable."
        )

    elif (
        ring.planarity_rmsd
        > analysis_config.maximum_ring_planarity_rmsd
    ):
        messages.append(
            "Ring planarity RMSD exceeds the maximum threshold."
        )

    if ring.maximum_plane_deviation is None:
        messages.append(
            "Maximum ring-plane deviation is unavailable."
        )

    elif (
        ring.maximum_plane_deviation
        > analysis_config.maximum_ring_atom_deviation
    ):
        messages.append(
            "Maximum ring-plane deviation exceeds the threshold."
        )

    if ring.radius is None or ring.radius <= 0.0:
        messages.append(
            "Ring radius is unavailable or invalid."
        )

    ring.valid = not messages

    combined_messages = list(
        ring.validation_messages
    )

    combined_messages.extend(
        message
        for message in messages
        if message not in combined_messages
    )

    ring.validation_messages = tuple(
        combined_messages
    )

    return ring.valid, tuple(messages)


def validate_pi_ring_geometries(
    rings: Iterable[PiRing],
    *,
    config: Optional[PiAnalysisConfig] = None,
    remove_invalid: bool = False,
) -> List[PiRing]:
    """
    Validate the geometry of multiple rings.
    """

    result: List[PiRing] = []

    for ring in rings:
        valid, _ = validate_pi_ring_geometry(
            ring,
            config=config,
            calculate_if_missing=True,
        )

        if remove_invalid and not valid:
            continue

        result.append(ring)

    return result


# -----------------------------------------------------------------------------
# 5.21. Processamento geométrico integrado
# -----------------------------------------------------------------------------

def prepare_pi_rings_geometry(
    rings: Iterable[PiRing],
    *,
    config: Optional[PiAnalysisConfig] = None,
    calculate_geometry: bool = True,
    validate_geometry: bool = True,
    deduplicate_geometry: bool = True,
    remove_invalid: bool = True,
) -> List[PiRing]:
    """
    Run the complete geometric preparation pipeline for aromatic rings.
    """

    analysis_config = (
        config
        if config is not None
        else create_default_pi_config()
    )

    prepared = list(rings)

    if calculate_geometry:
        prepared = calculate_pi_ring_geometries(
            prepared,
            config=analysis_config,
            remove_invalid=False,
            strict=False,
        )

    if validate_geometry:
        prepared = validate_pi_ring_geometries(
            prepared,
            config=analysis_config,
            remove_invalid=remove_invalid,
        )

    if deduplicate_geometry:
        prepared = deduplicate_pi_rings_by_geometry(
            prepared,
            config=analysis_config,
        )

    if remove_invalid:
        prepared = [
            ring
            for ring in prepared
            if ring.valid
        ]

    return prepared


def prepare_pi_analysis_ring_geometries(
    receptor_rings: Iterable[PiRing],
    ligand_rings: Iterable[PiRing],
    *,
    config: Optional[PiAnalysisConfig] = None,
) -> Tuple[List[PiRing], List[PiRing]]:
    """
    Prepare receptor and ligand ring geometries independently.
    """

    analysis_config = (
        config
        if config is not None
        else create_default_pi_config()
    )

    prepared_receptor_rings = prepare_pi_rings_geometry(
        receptor_rings,
        config=analysis_config,
    )

    prepared_ligand_rings = prepare_pi_rings_geometry(
        ligand_rings,
        config=analysis_config,
    )

    return (
        prepared_receptor_rings,
        prepared_ligand_rings,
    )


# -----------------------------------------------------------------------------
# 5.22. Atualização de uma PiInteraction com geometria ring-ring
# -----------------------------------------------------------------------------

def attach_ring_pair_geometry_to_interaction(
    interaction: PiInteraction,
    geometry: PiRingPairGeometry,
) -> PiInteraction:
    """
    Attach calculated ring-pair geometry to a ``PiInteraction``.
    """

    if not isinstance(interaction, PiInteraction):
        raise TypeError(
            "interaction must be a PiInteraction."
        )

    if not isinstance(geometry, PiRingPairGeometry):
        raise TypeError(
            "geometry must be a PiRingPairGeometry."
        )

    interaction.centroid_distance = (
        geometry.centroid_distance
    )

    interaction.minimum_atomic_distance = (
        geometry.minimum_atomic_distance
    )

    interaction.maximum_atomic_distance = (
        geometry.maximum_atomic_distance
    )

    interaction.normal_angle = (
        geometry.acute_normal_angle
    )

    interaction.plane_angle = (
        geometry.acute_normal_angle
    )

    interaction.lateral_offset = (
        geometry.mean_lateral_offset
    )

    interaction.radial_offset = (
        geometry.mean_lateral_offset
    )

    interaction.plane_height = (
        geometry.mean_plane_height
    )

    interaction.ring_1_planarity = (
        interaction.ring_1.planarity_rmsd
    )

    interaction.ring_2_planarity = (
        interaction.ring_2.planarity_rmsd
        if interaction.ring_2 is not None
        else None
    )

    interaction.metadata[
        "ring_pair_geometry"
    ] = geometry.to_dict()

    for warning_message in geometry.warnings:
        if warning_message not in interaction.warnings:
            interaction.warnings.append(
                warning_message
            )

    return interaction


# -----------------------------------------------------------------------------
# 5.23. Atualização de PiInteraction com geometria point-ring
# -----------------------------------------------------------------------------

def attach_point_ring_geometry_to_interaction(
    interaction: PiInteraction,
    geometry: PiPointRingGeometry,
) -> PiInteraction:
    """
    Attach point-ring geometry to a ``PiInteraction``.
    """

    if not isinstance(interaction, PiInteraction):
        raise TypeError(
            "interaction must be a PiInteraction."
        )

    if not isinstance(geometry, PiPointRingGeometry):
        raise TypeError(
            "geometry must be a PiPointRingGeometry."
        )

    interaction.centroid_distance = (
        geometry.center_distance
    )

    interaction.plane_height = (
        geometry.absolute_plane_distance
    )

    interaction.radial_offset = (
        geometry.radial_offset
    )

    interaction.lateral_offset = (
        geometry.radial_offset
    )

    if geometry.direction_angle is not None:
        interaction.normal_angle = (
            geometry.direction_angle
        )

    interaction.ring_1_planarity = (
        interaction.ring_1.planarity_rmsd
    )

    interaction.metadata[
        "point_ring_geometry"
    ] = geometry.to_dict()

    for warning_message in geometry.warnings:
        if warning_message not in interaction.warnings:
            interaction.warnings.append(
                warning_message
            )

    return interaction


# -----------------------------------------------------------------------------
# 5.24. Resumo geométrico dos anéis
# -----------------------------------------------------------------------------

def summarize_pi_ring_geometries(
    rings: Iterable[PiRing],
) -> Dict[str, Any]:
    """
    Generate statistics for calculated aromatic-ring geometries.
    """

    ring_list = list(rings)

    valid_rings = [
        ring
        for ring in ring_list
        if ring.valid and ring.has_complete_geometry
    ]

    planarity_values = [
        ring.planarity_rmsd
        for ring in valid_rings
        if ring.planarity_rmsd is not None
    ]

    maximum_deviations = [
        ring.maximum_plane_deviation
        for ring in valid_rings
        if ring.maximum_plane_deviation is not None
    ]

    radius_values = [
        ring.radius
        for ring in valid_rings
        if ring.radius is not None
    ]

    def summarize_values(
        values: Sequence[float],
    ) -> Dict[str, Optional[float]]:
        if not values:
            return {
                "minimum": None,
                "mean": None,
                "maximum": None,
            }

        return {
            "minimum": min(values),
            "mean": sum(values) / len(values),
            "maximum": max(values),
        }

    planarity_class_distribution = Counter(
        classify_ring_planarity(ring)
        for ring in ring_list
    )

    fit_method_distribution = Counter(
        str(
            ring.metadata.get(
                "plane_fit_method",
                "unknown",
            )
        )
        for ring in ring_list
    )

    return {
        "total_rings": len(ring_list),
        "rings_with_complete_geometry": sum(
            1
            for ring in ring_list
            if ring.has_complete_geometry
        ),
        "valid_geometries": len(valid_rings),
        "invalid_geometries": (
            len(ring_list) - len(valid_rings)
        ),
        "planarity_rmsd": summarize_values(
            planarity_values
        ),
        "maximum_plane_deviation": summarize_values(
            maximum_deviations
        ),
        "radius": summarize_values(
            radius_values
        ),
        "planarity_class_distribution": dict(
            planarity_class_distribution
        ),
        "fit_method_distribution": dict(
            fit_method_distribution
        ),
        "invalid_ring_ids": [
            ring.ring_id
            for ring in ring_list
            if not ring.valid
        ],
    }

# -----------------------------------------------------------------------------
# End of section 5.
# -----------------------------------------------------------------------------


# =============================================================================
# 6. DETECÇÃO E GEOMETRIA DE GRUPOS CARREGADOS
# =============================================================================

# -----------------------------------------------------------------------------
# 6.1. Tipos e constantes auxiliares
# -----------------------------------------------------------------------------

ChargedGroupAtomTuple: TypeAlias = Tuple[Any, ...]


CHARGE_POSITIVE: Final[str] = "positive"
CHARGE_NEGATIVE: Final[str] = "negative"
CHARGE_NEUTRAL: Final[str] = "neutral"
CHARGE_UNKNOWN: Final[str] = "unknown"


SUPPORTED_CHARGE_SIGNS: Final[FrozenSet[str]] = frozenset(
    {
        CHARGE_POSITIVE,
        CHARGE_NEGATIVE,
        CHARGE_NEUTRAL,
        CHARGE_UNKNOWN,
    }
)


POSITIVE_GROUP_AMMONIUM: Final[str] = "ammonium"
POSITIVE_GROUP_GUANIDINIUM: Final[str] = "guanidinium"
POSITIVE_GROUP_IMIDAZOLIUM: Final[str] = "imidazolium"
POSITIVE_GROUP_METAL: Final[str] = "metal_cation"
POSITIVE_GROUP_GENERIC: Final[str] = "generic_cation"


NEGATIVE_GROUP_CARBOXYLATE: Final[str] = "carboxylate"
NEGATIVE_GROUP_PHOSPHATE: Final[str] = "phosphate"
NEGATIVE_GROUP_SULFATE: Final[str] = "sulfate"
NEGATIVE_GROUP_SULFONATE: Final[str] = "sulfonate"
NEGATIVE_GROUP_PHENOLATE: Final[str] = "phenolate"
NEGATIVE_GROUP_GENERIC: Final[str] = "generic_anion"


STANDARD_POSITIVE_RESIDUE_GROUPS: Final[
    Mapping[str, Tuple[Mapping[str, Any], ...]]
] = {
    "LYS": (
        {
            "group_type": POSITIVE_GROUP_AMMONIUM,
            "atom_names": ("NZ",),
            "support_atom_names": ("CE",),
            "formal_charge": 1.0,
        },
    ),
    "LYN": (),
    "ARG": (
        {
            "group_type": POSITIVE_GROUP_GUANIDINIUM,
            "atom_names": ("CZ", "NH1", "NH2"),
            "support_atom_names": ("NE",),
            "formal_charge": 1.0,
        },
    ),
    "HIP": (
        {
            "group_type": POSITIVE_GROUP_IMIDAZOLIUM,
            "atom_names": ("CG", "ND1", "CE1", "NE2", "CD2"),
            "support_atom_names": ("CB",),
            "formal_charge": 1.0,
        },
    ),
    "HSP": (
        {
            "group_type": POSITIVE_GROUP_IMIDAZOLIUM,
            "atom_names": ("CG", "ND1", "CE1", "NE2", "CD2"),
            "support_atom_names": ("CB",),
            "formal_charge": 1.0,
        },
    ),
}


STANDARD_NEGATIVE_RESIDUE_GROUPS: Final[
    Mapping[str, Tuple[Mapping[str, Any], ...]]
] = {
    "ASP": (
        {
            "group_type": NEGATIVE_GROUP_CARBOXYLATE,
            "atom_names": ("CG", "OD1", "OD2"),
            "charge_atom_names": ("OD1", "OD2"),
            "support_atom_names": ("CB",),
            "formal_charge": -1.0,
        },
    ),
    "ASH": (),
    "GLU": (
        {
            "group_type": NEGATIVE_GROUP_CARBOXYLATE,
            "atom_names": ("CD", "OE1", "OE2"),
            "charge_atom_names": ("OE1", "OE2"),
            "support_atom_names": ("CG",),
            "formal_charge": -1.0,
        },
    ),
    "GLH": (),
    "TYM": (
        {
            "group_type": NEGATIVE_GROUP_PHENOLATE,
            "atom_names": ("CZ", "OH"),
            "charge_atom_names": ("OH",),
            "support_atom_names": ("CE1", "CE2"),
            "formal_charge": -1.0,
        },
    ),
    "CYM": (
        {
            "group_type": NEGATIVE_GROUP_GENERIC,
            "atom_names": ("SG",),
            "charge_atom_names": ("SG",),
            "support_atom_names": ("CB",),
            "formal_charge": -1.0,
        },
    ),
}


NUCLEIC_ACID_PHOSPHATE_ATOM_NAMES: Final[Tuple[str, ...]] = (
    "P",
    "OP1",
    "OP2",
    "O1P",
    "O2P",
)


COMMON_METAL_ELEMENTS: Final[FrozenSet[str]] = frozenset(
    {
        "LI",
        "NA",
        "K",
        "MG",
        "CA",
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


POSITIVE_GROUP_ELEMENTS: Final[FrozenSet[str]] = frozenset(
    {
        "N",
        "P",
        "S",
    }
)


NEGATIVE_GROUP_ELEMENTS: Final[FrozenSet[str]] = frozenset(
    {
        "O",
        "S",
        "P",
        "N",
    }
)


DEFAULT_CHARGE_GROUP_BOND_DEPTH: Final[int] = 1

DEFAULT_MINIMUM_GROUP_CHARGE_MAGNITUDE: Final[float] = 0.25

DEFAULT_LOCAL_CHARGE_NEIGHBOR_DISTANCE: Final[float] = 1.95

DEFAULT_CHARGE_CENTER_WEIGHT_FLOOR: Final[float] = 0.05


# -----------------------------------------------------------------------------
# 6.2. Normalização de sinais de carga
# -----------------------------------------------------------------------------

def normalize_charge_sign(
    value: Any,
    *,
    default: str = CHARGE_UNKNOWN,
) -> str:
    """
    Normalize a charge-sign description.
    """

    if value is None:
        return default

    if isinstance(value, bool):
        return default

    if isinstance(value, (int, float)):
        numeric_value = _normalize_optional_numeric(value)

        if numeric_value is None:
            return default

        if numeric_value > 0.0:
            return CHARGE_POSITIVE

        if numeric_value < 0.0:
            return CHARGE_NEGATIVE

        return CHARGE_NEUTRAL

    normalized = str(value).strip().lower()

    aliases = {
        "+": CHARGE_POSITIVE,
        "positive": CHARGE_POSITIVE,
        "cation": CHARGE_POSITIVE,
        "cationic": CHARGE_POSITIVE,
        "pos": CHARGE_POSITIVE,
        "-": CHARGE_NEGATIVE,
        "negative": CHARGE_NEGATIVE,
        "anion": CHARGE_NEGATIVE,
        "anionic": CHARGE_NEGATIVE,
        "neg": CHARGE_NEGATIVE,
        "0": CHARGE_NEUTRAL,
        "neutral": CHARGE_NEUTRAL,
        "unknown": CHARGE_UNKNOWN,
        "none": CHARGE_UNKNOWN,
    }

    return aliases.get(normalized, default)


def charge_sign_from_value(
    charge: Optional[Number],
    *,
    tolerance: float = 1.0e-8,
) -> str:
    """
    Return the charge sign associated with a numeric charge.
    """

    numeric_charge = _normalize_optional_numeric(charge)

    if numeric_charge is None:
        return CHARGE_UNKNOWN

    tolerance_value = _coerce_non_negative_float(
        tolerance,
        field_name="tolerance",
    )

    if numeric_charge > tolerance_value:
        return CHARGE_POSITIVE

    if numeric_charge < -tolerance_value:
        return CHARGE_NEGATIVE

    return CHARGE_NEUTRAL


# -----------------------------------------------------------------------------
# 6.3. Cálculo de cargas agregadas
# -----------------------------------------------------------------------------

def calculate_group_formal_charge(
    atoms: Iterable[Any],
) -> Optional[float]:
    """
    Sum explicit formal charges for a group.

    ``None`` is returned when no atom exposes formal-charge information.
    """

    charges = [
        get_atom_formal_charge(atom)
        for atom in atoms
    ]

    known_charges = [
        charge
        for charge in charges
        if charge is not None
    ]

    if not known_charges:
        return None

    return float(sum(known_charges))


def calculate_group_partial_charge(
    atoms: Iterable[Any],
) -> Optional[float]:
    """
    Sum explicit partial charges for a group.
    """

    charges = [
        get_atom_partial_charge(atom)
        for atom in atoms
    ]

    known_charges = [
        charge
        for charge in charges
        if charge is not None
    ]

    if not known_charges:
        return None

    return float(sum(known_charges))


def calculate_group_effective_charge(
    atoms: Iterable[Any],
    *,
    inferred_charge: Optional[float] = None,
    prefer_formal: bool = True,
) -> Optional[float]:
    """
    Return the best available charge estimate for a group.
    """

    atom_tuple = tuple(atoms)

    formal_charge = calculate_group_formal_charge(atom_tuple)
    partial_charge = calculate_group_partial_charge(atom_tuple)

    if prefer_formal and formal_charge is not None:
        if abs(formal_charge) > 1.0e-8:
            return formal_charge

    if partial_charge is not None:
        if abs(partial_charge) > 1.0e-8:
            return partial_charge

    if formal_charge is not None:
        return formal_charge

    return _normalize_optional_numeric(inferred_charge)


def group_has_positive_charge(
    atoms: Iterable[Any],
    *,
    inferred_charge: Optional[float] = None,
    minimum_magnitude: float = DEFAULT_MINIMUM_GROUP_CHARGE_MAGNITUDE,
) -> bool:
    """
    Return whether a group has a sufficiently positive effective charge.
    """

    charge = calculate_group_effective_charge(
        atoms,
        inferred_charge=inferred_charge,
    )

    return (
        charge is not None
        and charge >= minimum_magnitude
    )


def group_has_negative_charge(
    atoms: Iterable[Any],
    *,
    inferred_charge: Optional[float] = None,
    minimum_magnitude: float = DEFAULT_MINIMUM_GROUP_CHARGE_MAGNITUDE,
) -> bool:
    """
    Return whether a group has a sufficiently negative effective charge.
    """

    charge = calculate_group_effective_charge(
        atoms,
        inferred_charge=inferred_charge,
    )

    return (
        charge is not None
        and charge <= -minimum_magnitude
    )


# -----------------------------------------------------------------------------
# 6.4. Centro geométrico e centro ponderado por carga
# -----------------------------------------------------------------------------

def calculate_charge_weighted_center(
    atoms: Iterable[Any],
    *,
    charge_sign: Optional[str] = None,
    use_scene_coordinates: bool = True,
    weight_floor: float = DEFAULT_CHARGE_CENTER_WEIGHT_FLOOR,
    fallback_to_centroid: bool = True,
) -> Optional[Coordinate3D]:
    """
    Calculate a center weighted by the magnitude of atomic charge.

    For positive groups, only positive atomic contributions are preferred.
    For negative groups, only negative contributions are preferred.
    """

    atom_tuple = tuple(atoms)

    if not atom_tuple:
        return None

    normalized_sign = normalize_charge_sign(
        charge_sign,
        default=CHARGE_UNKNOWN,
    )

    weighted_coordinates: List[
        Tuple[Coordinate3D, float]
    ] = []

    for atom in atom_tuple:
        coordinate = get_atom_coordinate(
            atom,
            use_scene_coordinates=use_scene_coordinates,
        )

        if coordinate is None:
            continue

        formal_charge = get_atom_formal_charge(atom)
        partial_charge = get_atom_partial_charge(atom)

        charge = (
            formal_charge
            if formal_charge is not None
            else partial_charge
        )

        if charge is None:
            continue

        if normalized_sign == CHARGE_POSITIVE:
            weight = max(0.0, charge)

        elif normalized_sign == CHARGE_NEGATIVE:
            weight = max(0.0, -charge)

        else:
            weight = abs(charge)

        if weight < weight_floor:
            continue

        weighted_coordinates.append(
            (
                coordinate,
                weight,
            )
        )

    total_weight = sum(
        weight
        for _, weight in weighted_coordinates
    )

    if total_weight > 0.0:
        return (
            sum(
                coordinate[0] * weight
                for coordinate, weight in weighted_coordinates
            ) / total_weight,
            sum(
                coordinate[1] * weight
                for coordinate, weight in weighted_coordinates
            ) / total_weight,
            sum(
                coordinate[2] * weight
                for coordinate, weight in weighted_coordinates
            ) / total_weight,
        )

    if not fallback_to_centroid:
        return None

    coordinates = get_atom_coordinates(
        atom_tuple,
        use_scene_coordinates=use_scene_coordinates,
        skip_invalid=True,
    )

    if not coordinates:
        return None

    return calculate_centroid(coordinates)


def calculate_charged_group_center(
    group_atoms: Iterable[Any],
    *,
    charge_atoms: Optional[Iterable[Any]] = None,
    charge_sign: Optional[str] = None,
    use_scene_coordinates: bool = True,
) -> Optional[Coordinate3D]:
    """
    Calculate the chemically relevant center of a charged group.
    """

    atom_tuple = tuple(group_atoms)
    charge_atom_tuple = tuple(charge_atoms or ())

    preferred_atoms = (
        charge_atom_tuple
        if charge_atom_tuple
        else atom_tuple
    )

    center = calculate_charge_weighted_center(
        preferred_atoms,
        charge_sign=charge_sign,
        use_scene_coordinates=use_scene_coordinates,
        fallback_to_centroid=True,
    )

    if center is not None:
        return center

    coordinates = get_atom_coordinates(
        atom_tuple,
        use_scene_coordinates=use_scene_coordinates,
        skip_invalid=True,
    )

    if not coordinates:
        return None

    return calculate_centroid(coordinates)


# -----------------------------------------------------------------------------
# 6.5. Vetor direcional de grupos carregados
# -----------------------------------------------------------------------------

def calculate_group_direction_vector(
    center: Sequence[Number],
    support_atoms: Iterable[Any],
    *,
    use_scene_coordinates: bool = True,
    point_away_from_support: bool = True,
) -> Optional[Vector3D]:
    """
    Calculate a direction vector for a charged functional group.

    The vector usually points from the supporting molecular scaffold toward
    the charged center.
    """

    normalized_center = _coerce_coordinate3d(
        center,
        field_name="center",
    )

    assert normalized_center is not None

    support_coordinates = get_atom_coordinates(
        support_atoms,
        use_scene_coordinates=use_scene_coordinates,
        skip_invalid=True,
    )

    if not support_coordinates:
        return None

    support_center = calculate_centroid(
        support_coordinates
    )

    if point_away_from_support:
        vector = subtract_vectors(
            normalized_center,
            support_center,
        )

    else:
        vector = subtract_vectors(
            support_center,
            normalized_center,
        )

    return normalize_vector(vector)


def calculate_group_plane_normal(
    atoms: Iterable[Any],
    *,
    use_scene_coordinates: bool = True,
) -> Optional[Vector3D]:
    """
    Fit a plane normal to a planar charged group.
    """

    atom_tuple = tuple(atoms)

    if len(atom_tuple) < 3:
        return None

    plane = fit_plane_to_atoms(
        atom_tuple,
        use_scene_coordinates=use_scene_coordinates,
        skip_invalid=True,
        strict=False,
    )

    if plane is None:
        return None

    return orient_normal_deterministically(
        plane.normal
    )


# -----------------------------------------------------------------------------
# 6.6. Construção de grupos carregados conhecidos
# -----------------------------------------------------------------------------

def _create_known_residue_charged_group(
    residue: Any,
    definition: Mapping[str, Any],
    *,
    charge_sign: str,
    participant_type: Optional[str] = None,
    group_index: Optional[int] = None,
    use_scene_coordinates: bool = True,
    ligand_residue_names: Optional[Collection[str]] = None,
    receptor_residue_names: Optional[Collection[str]] = None,
) -> Optional[PiChargedGroup]:
    """
    Create a charged group from a residue-specific definition.
    """

    atom_map = map_atoms_by_name(residue)

    required_names = tuple(
        str(name).strip().upper()
        for name in definition.get(
            "atom_names",
            (),
        )
    )

    charge_names = tuple(
        str(name).strip().upper()
        for name in definition.get(
            "charge_atom_names",
            required_names,
        )
    )

    support_names = tuple(
        str(name).strip().upper()
        for name in definition.get(
            "support_atom_names",
            (),
        )
    )

    atoms = tuple(
        atom_map[name]
        for name in required_names
        if name in atom_map
    )

    if len(atoms) != len(required_names):
        return None

    if not all(
        atom_has_valid_coordinate(
            atom,
            use_scene_coordinates=use_scene_coordinates,
        )
        for atom in atoms
    ):
        return None

    charge_atoms = tuple(
        atom_map[name]
        for name in charge_names
        if name in atom_map
    )

    support_atoms = tuple(
        atom_map[name]
        for name in support_names
        if name in atom_map
    )

    inferred_charge = _normalize_optional_numeric(
        definition.get("formal_charge")
    )

    center = calculate_charged_group_center(
        atoms,
        charge_atoms=charge_atoms,
        charge_sign=charge_sign,
        use_scene_coordinates=use_scene_coordinates,
    )

    if center is None:
        return None

    direction = calculate_group_direction_vector(
        center,
        support_atoms,
        use_scene_coordinates=use_scene_coordinates,
    )

    plane_normal = calculate_group_plane_normal(
        atoms,
        use_scene_coordinates=use_scene_coordinates,
    )

    residue_name = get_residue_name(residue)
    residue_number = get_residue_number(residue)
    chain_id = get_residue_chain_id(residue)

    model = _safe_get_value(
        residue,
        (
            "structure",
            "model",
            "molecule",
        ),
        default=None,
    )

    normalized_participant_type = (
        participant_type
        or infer_participant_type(
            residue,
            ligand_residue_names=ligand_residue_names,
            receptor_residue_names=receptor_residue_names,
        )
    )

    formal_charge = calculate_group_formal_charge(atoms)
    partial_charge = calculate_group_partial_charge(atoms)

    effective_charge = calculate_group_effective_charge(
        atoms,
        inferred_charge=inferred_charge,
    )

    return PiChargedGroup(
        atoms=atoms,
        atom_references=create_pi_atom_references(
            atoms,
            skip_invalid=False,
        ),
        charge_atoms=charge_atoms,
        support_atoms=support_atoms,
        group_index=group_index,
        group_type=str(
            definition.get(
                "group_type",
                (
                    POSITIVE_GROUP_GENERIC
                    if charge_sign == CHARGE_POSITIVE
                    else NEGATIVE_GROUP_GENERIC
                ),
            )
        ),
        charge_sign=charge_sign,
        center=center,
        direction=direction,
        plane_normal=plane_normal,
        formal_charge=formal_charge,
        partial_charge=partial_charge,
        effective_charge=effective_charge,
        residue_name=residue_name,
        residue_number=residue_number,
        chain_id=chain_id,
        model_id=get_model_identifier(model),
        participant_type=normalized_participant_type,
        valid=True,
        metadata={
            "detection_method": "known_residue_definition",
            "definition": dict(definition),
        },
    )


def detect_known_residue_charged_groups(
    residue: Any,
    *,
    include_positive: bool = True,
    include_negative: bool = True,
    participant_type: Optional[str] = None,
    use_scene_coordinates: bool = True,
    ligand_residue_names: Optional[Collection[str]] = None,
    receptor_residue_names: Optional[Collection[str]] = None,
) -> List[PiChargedGroup]:
    """
    Detect known charged groups in protein residues.
    """

    residue_name = get_residue_name(residue)
    groups: List[PiChargedGroup] = []

    if include_positive:
        positive_definitions = (
            STANDARD_POSITIVE_RESIDUE_GROUPS.get(
                residue_name,
                (),
            )
        )

        for definition in positive_definitions:
            group = _create_known_residue_charged_group(
                residue,
                definition,
                charge_sign=CHARGE_POSITIVE,
                participant_type=participant_type,
                group_index=len(groups) + 1,
                use_scene_coordinates=use_scene_coordinates,
                ligand_residue_names=ligand_residue_names,
                receptor_residue_names=receptor_residue_names,
            )

            if group is not None:
                groups.append(group)

    if include_negative:
        negative_definitions = (
            STANDARD_NEGATIVE_RESIDUE_GROUPS.get(
                residue_name,
                (),
            )
        )

        for definition in negative_definitions:
            group = _create_known_residue_charged_group(
                residue,
                definition,
                charge_sign=CHARGE_NEGATIVE,
                participant_type=participant_type,
                group_index=len(groups) + 1,
                use_scene_coordinates=use_scene_coordinates,
                ligand_residue_names=ligand_residue_names,
                receptor_residue_names=receptor_residue_names,
            )

            if group is not None:
                groups.append(group)

    return groups


# -----------------------------------------------------------------------------
# 6.7. Detecção de grupos fosfato
# -----------------------------------------------------------------------------

def detect_phosphate_groups(
    atoms_or_residue: Any,
    *,
    participant_type: Optional[str] = None,
    require_negative_charge: bool = False,
    use_scene_coordinates: bool = True,
) -> List[PiChargedGroup]:
    """
    Detect phosphate groups centered on phosphorus atoms.
    """

    atoms = normalize_atom_collection(
        atoms_or_residue,
        include_hydrogens=False,
        valid_coordinates_only=True,
    )

    atom_identity_set = {
        id(atom)
        for atom in atoms
    }

    groups: List[PiChargedGroup] = []

    for phosphorus in atoms:
        if get_atom_element(phosphorus) != "P":
            continue

        bonded_oxygens = tuple(
            neighbor
            for neighbor in get_bonded_atoms(
                phosphorus,
                include_hydrogens=False,
            )
            if (
                id(neighbor) in atom_identity_set
                and get_atom_element(neighbor) == "O"
            )
        )

        if len(bonded_oxygens) < 3:
            continue

        group_atoms = (
            phosphorus,
            *bonded_oxygens,
        )

        inferred_charge = calculate_group_effective_charge(
            group_atoms,
            inferred_charge=-1.0,
        )

        if (
            require_negative_charge
            and (
                inferred_charge is None
                or inferred_charge >= 0.0
            )
        ):
            continue

        negative_oxygens = tuple(
            oxygen
            for oxygen in bonded_oxygens
            if atom_has_negative_charge(
                oxygen,
                infer_from_type=True,
            )
        )

        charge_atoms = (
            negative_oxygens
            if negative_oxygens
            else bonded_oxygens
        )

        center = calculate_charged_group_center(
            group_atoms,
            charge_atoms=charge_atoms,
            charge_sign=CHARGE_NEGATIVE,
            use_scene_coordinates=use_scene_coordinates,
        )

        if center is None:
            continue

        residue = get_atom_residue(phosphorus)
        model = get_atom_model(phosphorus)

        group = PiChargedGroup(
            atoms=tuple(group_atoms),
            atom_references=create_pi_atom_references(
                group_atoms
            ),
            charge_atoms=charge_atoms,
            support_atoms=(phosphorus,),
            group_index=len(groups) + 1,
            group_type=NEGATIVE_GROUP_PHOSPHATE,
            charge_sign=CHARGE_NEGATIVE,
            center=center,
            direction=None,
            plane_normal=calculate_group_plane_normal(
                bonded_oxygens,
                use_scene_coordinates=use_scene_coordinates,
            ),
            formal_charge=calculate_group_formal_charge(
                group_atoms
            ),
            partial_charge=calculate_group_partial_charge(
                group_atoms
            ),
            effective_charge=inferred_charge,
            residue_name=get_residue_name(
                residue or phosphorus
            ),
            residue_number=get_residue_number(
                residue or phosphorus
            ),
            chain_id=get_residue_chain_id(
                residue or phosphorus
            ),
            model_id=get_model_identifier(model),
            participant_type=(
                participant_type
                or infer_participant_type(
                    residue or phosphorus
                )
            ),
            valid=True,
            metadata={
                "detection_method": "phosphorus_connectivity",
                "oxygen_count": len(bonded_oxygens),
            },
        )

        groups.append(group)

    return groups


# -----------------------------------------------------------------------------
# 6.8. Detecção de carboxilatos genéricos
# -----------------------------------------------------------------------------

def detect_carboxylate_groups(
    atoms_or_residue: Any,
    *,
    participant_type: Optional[str] = None,
    require_negative_charge: bool = False,
    use_scene_coordinates: bool = True,
) -> List[PiChargedGroup]:
    """
    Detect generic carboxylate groups from carbon–oxygen connectivity.
    """

    atoms = normalize_atom_collection(
        atoms_or_residue,
        include_hydrogens=False,
        valid_coordinates_only=True,
    )

    atom_identity_set = {
        id(atom)
        for atom in atoms
    }

    groups: List[PiChargedGroup] = []

    for carbon in atoms:
        if get_atom_element(carbon) != "C":
            continue

        bonded_oxygens = tuple(
            neighbor
            for neighbor in get_bonded_atoms(
                carbon,
                include_hydrogens=False,
            )
            if (
                id(neighbor) in atom_identity_set
                and get_atom_element(neighbor) == "O"
            )
        )

        if len(bonded_oxygens) != 2:
            continue

        bond_orders = tuple(
            get_bond_order(carbon, oxygen)
            for oxygen in bonded_oxygens
        )

        aromatic_like = any(
            order is not None
            and abs(order - 1.5) <= 0.20
            for order in bond_orders
        )

        double_bond_count = sum(
            1
            for order in bond_orders
            if order is not None
            and order >= 1.75
        )

        negative_oxygen_count = sum(
            1
            for oxygen in bonded_oxygens
            if atom_has_negative_charge(oxygen)
        )

        carboxylate_like = (
            aromatic_like
            or double_bond_count == 1
            or negative_oxygen_count >= 1
        )

        if not carboxylate_like:
            continue

        group_atoms = (
            carbon,
            *bonded_oxygens,
        )

        inferred_charge = calculate_group_effective_charge(
            group_atoms,
            inferred_charge=-1.0,
        )

        if (
            require_negative_charge
            and (
                inferred_charge is None
                or inferred_charge >= 0.0
            )
        ):
            continue

        negative_oxygens = tuple(
            oxygen
            for oxygen in bonded_oxygens
            if atom_has_negative_charge(oxygen)
        )

        charge_atoms = (
            negative_oxygens
            if negative_oxygens
            else bonded_oxygens
        )

        center = calculate_charged_group_center(
            group_atoms,
            charge_atoms=charge_atoms,
            charge_sign=CHARGE_NEGATIVE,
            use_scene_coordinates=use_scene_coordinates,
        )

        if center is None:
            continue

        support_atoms = tuple(
            neighbor
            for neighbor in get_bonded_atoms(
                carbon,
                include_hydrogens=False,
            )
            if (
                id(neighbor) in atom_identity_set
                and neighbor not in bonded_oxygens
            )
        )

        residue = get_atom_residue(carbon)
        model = get_atom_model(carbon)

        groups.append(
            PiChargedGroup(
                atoms=group_atoms,
                atom_references=create_pi_atom_references(
                    group_atoms
                ),
                charge_atoms=charge_atoms,
                support_atoms=support_atoms,
                group_index=len(groups) + 1,
                group_type=NEGATIVE_GROUP_CARBOXYLATE,
                charge_sign=CHARGE_NEGATIVE,
                center=center,
                direction=calculate_group_direction_vector(
                    center,
                    support_atoms,
                    use_scene_coordinates=use_scene_coordinates,
                ),
                plane_normal=calculate_group_plane_normal(
                    group_atoms,
                    use_scene_coordinates=use_scene_coordinates,
                ),
                formal_charge=calculate_group_formal_charge(
                    group_atoms
                ),
                partial_charge=calculate_group_partial_charge(
                    group_atoms
                ),
                effective_charge=inferred_charge,
                residue_name=get_residue_name(
                    residue or carbon
                ),
                residue_number=get_residue_number(
                    residue or carbon
                ),
                chain_id=get_residue_chain_id(
                    residue or carbon
                ),
                model_id=get_model_identifier(model),
                participant_type=(
                    participant_type
                    or infer_participant_type(
                        residue or carbon
                    )
                ),
                valid=True,
                metadata={
                    "detection_method": (
                        "carboxylate_connectivity"
                    ),
                    "bond_orders": list(bond_orders),
                },
            )
        )

    return groups


# -----------------------------------------------------------------------------
# 6.9. Detecção de amônio e nitrogênios catiônicos
# -----------------------------------------------------------------------------

def _infer_nitrogen_positive_group_type(
    nitrogen: Any,
    bonded_atoms: Sequence[Any],
) -> str:
    """
    Infer a positive nitrogen group class.
    """

    carbon_neighbors = [
        atom
        for atom in bonded_atoms
        if get_atom_element(atom) == "C"
    ]

    nitrogen_neighbors = [
        atom
        for atom in bonded_atoms
        if get_atom_element(atom) == "N"
    ]

    if len(carbon_neighbors) >= 4:
        return POSITIVE_GROUP_AMMONIUM

    if nitrogen_neighbors:
        return POSITIVE_GROUP_GUANIDINIUM

    return POSITIVE_GROUP_GENERIC


def detect_positive_nitrogen_groups(
    atoms_or_residue: Any,
    *,
    participant_type: Optional[str] = None,
    require_explicit_charge: bool = False,
    use_scene_coordinates: bool = True,
) -> List[PiChargedGroup]:
    """
    Detect positively charged nitrogen centers.
    """

    atoms = normalize_atom_collection(
        atoms_or_residue,
        include_hydrogens=True,
        valid_coordinates_only=True,
    )

    atom_identity_set = {
        id(atom)
        for atom in atoms
    }

    groups: List[PiChargedGroup] = []

    for nitrogen in atoms:
        if get_atom_element(nitrogen) != "N":
            continue

        explicit_positive = atom_has_positive_charge(
            nitrogen,
            infer_from_type=True,
        )

        bonded_atoms = tuple(
            neighbor
            for neighbor in get_bonded_atoms(
                nitrogen,
                include_hydrogens=True,
            )
            if id(neighbor) in atom_identity_set
        )

        heavy_neighbors = tuple(
            atom
            for atom in bonded_atoms
            if is_heavy_atom(atom)
        )

        hydrogen_neighbors = tuple(
            atom
            for atom in bonded_atoms
            if is_hydrogen_atom(atom)
        )

        valence_estimate = (
            len(heavy_neighbors)
            + len(hydrogen_neighbors)
        )

        quaternary_like = len(heavy_neighbors) >= 4

        protonated_like = (
            valence_estimate >= 4
            and len(hydrogen_neighbors) >= 1
        )

        if require_explicit_charge:
            if not explicit_positive:
                continue

        elif not (
            explicit_positive
            or quaternary_like
            or protonated_like
        ):
            continue

        support_atoms = heavy_neighbors

        center = get_atom_coordinate(
            nitrogen,
            use_scene_coordinates=use_scene_coordinates,
        )

        if center is None:
            continue

        residue = get_atom_residue(nitrogen)
        model = get_atom_model(nitrogen)

        effective_charge = calculate_group_effective_charge(
            (nitrogen,),
            inferred_charge=1.0,
        )

        groups.append(
            PiChargedGroup(
                atoms=(nitrogen,),
                atom_references=create_pi_atom_references(
                    (nitrogen,)
                ),
                charge_atoms=(nitrogen,),
                support_atoms=support_atoms,
                group_index=len(groups) + 1,
                group_type=_infer_nitrogen_positive_group_type(
                    nitrogen,
                    bonded_atoms,
                ),
                charge_sign=CHARGE_POSITIVE,
                center=center,
                direction=calculate_group_direction_vector(
                    center,
                    support_atoms,
                    use_scene_coordinates=use_scene_coordinates,
                ),
                plane_normal=None,
                formal_charge=get_atom_formal_charge(
                    nitrogen
                ),
                partial_charge=get_atom_partial_charge(
                    nitrogen
                ),
                effective_charge=effective_charge,
                residue_name=get_residue_name(
                    residue or nitrogen
                ),
                residue_number=get_residue_number(
                    residue or nitrogen
                ),
                chain_id=get_residue_chain_id(
                    residue or nitrogen
                ),
                model_id=get_model_identifier(model),
                participant_type=(
                    participant_type
                    or infer_participant_type(
                        residue or nitrogen
                    )
                ),
                valid=True,
                metadata={
                    "detection_method": (
                        "positive_nitrogen_connectivity"
                    ),
                    "explicit_positive_charge": (
                        explicit_positive
                    ),
                    "heavy_neighbor_count": len(
                        heavy_neighbors
                    ),
                    "hydrogen_neighbor_count": len(
                        hydrogen_neighbors
                    ),
                },
            )
        )

    return groups


# -----------------------------------------------------------------------------
# 6.10. Detecção de metais catiônicos
# -----------------------------------------------------------------------------

def detect_metal_cations(
    atoms_or_model: Any,
    *,
    participant_type: Optional[str] = None,
    require_positive_charge: bool = False,
    use_scene_coordinates: bool = True,
) -> List[PiChargedGroup]:
    """
    Detect monoatomic metal cations.
    """

    atoms = normalize_atom_collection(
        atoms_or_model,
        include_hydrogens=False,
        valid_coordinates_only=True,
    )

    groups: List[PiChargedGroup] = []

    for atom in atoms:
        element = get_atom_element(atom)

        if element not in COMMON_METAL_ELEMENTS:
            continue

        formal_charge = get_atom_formal_charge(atom)
        partial_charge = get_atom_partial_charge(atom)

        explicit_positive = (
            formal_charge is not None
            and formal_charge > 0.0
        ) or (
            partial_charge is not None
            and partial_charge > 0.0
        )

        if require_positive_charge and not explicit_positive:
            continue

        inferred_charge = (
            formal_charge
            if formal_charge is not None
            else partial_charge
        )

        if inferred_charge is None:
            inferred_charge = 2.0 if element in {
                "MG",
                "CA",
                "ZN",
                "FE",
                "MN",
                "CU",
                "CO",
                "NI",
                "CD",
                "HG",
            } else 1.0

        center = get_atom_coordinate(
            atom,
            use_scene_coordinates=use_scene_coordinates,
        )

        if center is None:
            continue

        residue = get_atom_residue(atom)
        model = get_atom_model(atom)

        groups.append(
            PiChargedGroup(
                atoms=(atom,),
                atom_references=create_pi_atom_references(
                    (atom,)
                ),
                charge_atoms=(atom,),
                support_atoms=(),
                group_index=len(groups) + 1,
                group_type=POSITIVE_GROUP_METAL,
                charge_sign=CHARGE_POSITIVE,
                center=center,
                direction=None,
                plane_normal=None,
                formal_charge=formal_charge,
                partial_charge=partial_charge,
                effective_charge=inferred_charge,
                residue_name=get_residue_name(
                    residue or atom
                ),
                residue_number=get_residue_number(
                    residue or atom
                ),
                chain_id=get_residue_chain_id(
                    residue or atom
                ),
                model_id=get_model_identifier(model),
                participant_type=(
                    participant_type
                    or infer_participant_type(
                        residue or atom
                    )
                ),
                valid=True,
                metadata={
                    "detection_method": "metal_element",
                    "element": element,
                },
            )
        )

    return groups


# -----------------------------------------------------------------------------
# 6.11. Detecção genérica por cargas atômicas
# -----------------------------------------------------------------------------

def _expand_local_charged_environment(
    atom: Any,
    allowed_atom_ids: Set[int],
    *,
    bond_depth: int = DEFAULT_CHARGE_GROUP_BOND_DEPTH,
) -> Tuple[Any, ...]:
    """
    Expand a charged atom into a local bonded environment.
    """

    if bond_depth < 0:
        raise ValueError(
            "bond_depth must be non-negative."
        )

    collected: List[Any] = [atom]
    visited: Set[int] = {id(atom)}
    frontier: List[Tuple[Any, int]] = [
        (
            atom,
            0,
        )
    ]

    while frontier:
        current, depth = frontier.pop(0)

        if depth >= bond_depth:
            continue

        for neighbor in get_bonded_atoms(
            current,
            include_hydrogens=False,
        ):
            neighbor_id = id(neighbor)

            if neighbor_id not in allowed_atom_ids:
                continue

            if neighbor_id in visited:
                continue

            visited.add(neighbor_id)
            collected.append(neighbor)
            frontier.append(
                (
                    neighbor,
                    depth + 1,
                )
            )

    return tuple(collected)


def detect_explicitly_charged_atom_groups(
    atoms_or_model: Any,
    *,
    charge_sign: Optional[str] = None,
    participant_type: Optional[str] = None,
    bond_depth: int = DEFAULT_CHARGE_GROUP_BOND_DEPTH,
    minimum_charge_magnitude: float = (
        DEFAULT_MINIMUM_GROUP_CHARGE_MAGNITUDE
    ),
    use_scene_coordinates: bool = True,
) -> List[PiChargedGroup]:
    """
    Detect generic groups centered on explicitly charged atoms.
    """

    atoms = normalize_atom_collection(
        atoms_or_model,
        include_hydrogens=False,
        valid_coordinates_only=True,
    )

    allowed_atom_ids = {
        id(atom)
        for atom in atoms
    }

    requested_sign = normalize_charge_sign(
        charge_sign,
        default=CHARGE_UNKNOWN,
    )

    groups: List[PiChargedGroup] = []

    for atom in atoms:
        effective_charge = get_atom_effective_charge(atom)

        if effective_charge is None:
            continue

        detected_sign = charge_sign_from_value(
            effective_charge
        )

        if detected_sign not in {
            CHARGE_POSITIVE,
            CHARGE_NEGATIVE,
        }:
            continue

        if (
            abs(effective_charge)
            < minimum_charge_magnitude
        ):
            continue

        if (
            requested_sign != CHARGE_UNKNOWN
            and detected_sign != requested_sign
        ):
            continue

        group_atoms = _expand_local_charged_environment(
            atom,
            allowed_atom_ids,
            bond_depth=bond_depth,
        )

        center = calculate_charged_group_center(
            group_atoms,
            charge_atoms=(atom,),
            charge_sign=detected_sign,
            use_scene_coordinates=use_scene_coordinates,
        )

        if center is None:
            continue

        support_atoms = tuple(
            candidate
            for candidate in group_atoms
            if candidate is not atom
        )

        residue = get_atom_residue(atom)
        model = get_atom_model(atom)

        groups.append(
            PiChargedGroup(
                atoms=group_atoms,
                atom_references=create_pi_atom_references(
                    group_atoms
                ),
                charge_atoms=(atom,),
                support_atoms=support_atoms,
                group_index=len(groups) + 1,
                group_type=(
                    POSITIVE_GROUP_GENERIC
                    if detected_sign == CHARGE_POSITIVE
                    else NEGATIVE_GROUP_GENERIC
                ),
                charge_sign=detected_sign,
                center=center,
                direction=calculate_group_direction_vector(
                    center,
                    support_atoms,
                    use_scene_coordinates=use_scene_coordinates,
                ),
                plane_normal=calculate_group_plane_normal(
                    group_atoms,
                    use_scene_coordinates=use_scene_coordinates,
                ),
                formal_charge=calculate_group_formal_charge(
                    group_atoms
                ),
                partial_charge=calculate_group_partial_charge(
                    group_atoms
                ),
                effective_charge=calculate_group_effective_charge(
                    group_atoms,
                    inferred_charge=effective_charge,
                ),
                residue_name=get_residue_name(
                    residue or atom
                ),
                residue_number=get_residue_number(
                    residue or atom
                ),
                chain_id=get_residue_chain_id(
                    residue or atom
                ),
                model_id=get_model_identifier(model),
                participant_type=(
                    participant_type
                    or infer_participant_type(
                        residue or atom
                    )
                ),
                valid=True,
                metadata={
                    "detection_method": (
                        "explicit_atomic_charge"
                    ),
                    "central_atom": get_atom_identifier(
                        atom
                    ),
                    "bond_depth": bond_depth,
                },
            )
        )

    return groups


# -----------------------------------------------------------------------------
# 6.12. Identidade e deduplicação de grupos carregados
# -----------------------------------------------------------------------------

def get_charged_group_identity_key(
    group: PiChargedGroup,
) -> Tuple[Any, ...]:
    """
    Return a hashable identity key for a charged group.
    """

    if not isinstance(group, PiChargedGroup):
        raise TypeError(
            "group must be a PiChargedGroup."
        )

    atom_ids = tuple(
        sorted(
            id(atom)
            for atom in group.atoms
        )
    )

    charge_atom_ids = tuple(
        sorted(
            id(atom)
            for atom in group.charge_atoms
        )
    )

    return (
        group.model_id,
        group.chain_id,
        group.residue_name,
        group.residue_number,
        group.group_type,
        group.charge_sign,
        atom_ids,
        charge_atom_ids,
    )


def _charged_group_priority(
    group: PiChargedGroup,
) -> Tuple[int, float, int]:
    """
    Return a priority tuple used during deduplication.
    """

    method = str(
        group.metadata.get(
            "detection_method",
            "",
        )
    )

    method_priority = {
        "known_residue_definition": 5,
        "phosphorus_connectivity": 4,
        "carboxylate_connectivity": 4,
        "positive_nitrogen_connectivity": 4,
        "metal_element": 4,
        "explicit_atomic_charge": 2,
    }.get(method, 1)

    charge_magnitude = abs(
        group.effective_charge
        if group.effective_charge is not None
        else 0.0
    )

    return (
        method_priority,
        charge_magnitude,
        len(group.atoms),
    )


def charged_groups_overlap(
    group_1: PiChargedGroup,
    group_2: PiChargedGroup,
    *,
    minimum_shared_fraction: float = 0.50,
) -> bool:
    """
    Return whether two charged groups substantially overlap.
    """

    atom_ids_1 = {
        id(atom)
        for atom in group_1.atoms
    }

    atom_ids_2 = {
        id(atom)
        for atom in group_2.atoms
    }

    if not atom_ids_1 or not atom_ids_2:
        return False

    shared_count = len(
        atom_ids_1 & atom_ids_2
    )

    denominator = min(
        len(atom_ids_1),
        len(atom_ids_2),
    )

    return (
        shared_count / denominator
        >= minimum_shared_fraction
    )


def deduplicate_charged_groups(
    groups: Iterable[PiChargedGroup],
) -> List[PiChargedGroup]:
    """
    Remove identical or substantially overlapping charged groups.
    """

    group_list = list(groups)
    unique: List[PiChargedGroup] = []

    group_list.sort(
        key=_charged_group_priority,
        reverse=True,
    )

    for candidate in group_list:
        duplicate_index: Optional[int] = None

        for index, existing in enumerate(unique):
            same_sign = (
                candidate.charge_sign
                == existing.charge_sign
            )

            same_model = (
                candidate.model_id
                == existing.model_id
            )

            same_residue = (
                candidate.chain_id
                == existing.chain_id
                and candidate.residue_name
                == existing.residue_name
                and candidate.residue_number
                == existing.residue_number
            )

            if not (
                same_sign
                and same_model
                and same_residue
            ):
                continue

            if (
                get_charged_group_identity_key(candidate)
                == get_charged_group_identity_key(existing)
            ) or charged_groups_overlap(
                candidate,
                existing,
            ):
                duplicate_index = index
                break

        if duplicate_index is None:
            unique.append(candidate)
            continue

        existing = unique[duplicate_index]

        if (
            _charged_group_priority(candidate)
            > _charged_group_priority(existing)
        ):
            unique[duplicate_index] = candidate

    unique.sort(
        key=lambda group: (
            group.model_id or "",
            group.chain_id or "",
            str(group.residue_number or ""),
            group.residue_name or "",
            group.charge_sign,
            group.group_type,
        )
    )

    for group_index, group in enumerate(
        unique,
        start=1,
    ):
        group.group_index = group_index

        if not group.group_id:
            group.group_id = group.build_group_id()

    return unique


# -----------------------------------------------------------------------------
# 6.13. Validação de grupos carregados
# -----------------------------------------------------------------------------

def validate_charged_group(
    group: PiChargedGroup,
    *,
    minimum_charge_magnitude: float = (
        DEFAULT_MINIMUM_GROUP_CHARGE_MAGNITUDE
    ),
    require_center: bool = True,
    require_charge_evidence: bool = True,
) -> Tuple[bool, Tuple[str, ...]]:
    """
    Validate a charged group.
    """

    if not isinstance(group, PiChargedGroup):
        raise TypeError(
            "group must be a PiChargedGroup."
        )

    messages: List[str] = []

    if not group.atoms:
        messages.append(
            "Charged group contains no atoms."
        )

    if group.charge_sign not in {
        CHARGE_POSITIVE,
        CHARGE_NEGATIVE,
    }:
        messages.append(
            "Charged group sign must be positive or negative."
        )

    if require_center and group.center is None:
        messages.append(
            "Charged group center is unavailable."
        )

    if group.center is not None:
        try:
            _coerce_coordinate3d(
                group.center,
                field_name="group.center",
            )

        except (TypeError, ValueError):
            messages.append(
                "Charged group center is invalid."
            )

    effective_charge = group.effective_charge

    if effective_charge is None:
        effective_charge = calculate_group_effective_charge(
            group.atoms
        )

        group.effective_charge = effective_charge

    if require_charge_evidence:
        if effective_charge is None:
            messages.append(
                "No explicit or inferred charge is available."
            )

        elif (
            abs(effective_charge)
            < minimum_charge_magnitude
        ):
            messages.append(
                "Charge magnitude is below the configured threshold."
            )

        elif (
            group.charge_sign == CHARGE_POSITIVE
            and effective_charge <= 0.0
        ):
            messages.append(
                "Effective charge is inconsistent with a positive group."
            )

        elif (
            group.charge_sign == CHARGE_NEGATIVE
            and effective_charge >= 0.0
        ):
            messages.append(
                "Effective charge is inconsistent with a negative group."
            )

    group.valid = not messages

    existing_messages = list(
        group.validation_messages
    )

    existing_messages.extend(
        message
        for message in messages
        if message not in existing_messages
    )

    group.validation_messages = tuple(
        existing_messages
    )

    return group.valid, tuple(messages)


def validate_charged_groups(
    groups: Iterable[PiChargedGroup],
    *,
    minimum_charge_magnitude: float = (
        DEFAULT_MINIMUM_GROUP_CHARGE_MAGNITUDE
    ),
    remove_invalid: bool = False,
) -> List[PiChargedGroup]:
    """
    Validate multiple charged groups.
    """

    validated: List[PiChargedGroup] = []

    for group in groups:
        valid, _ = validate_charged_group(
            group,
            minimum_charge_magnitude=(
                minimum_charge_magnitude
            ),
        )

        if remove_invalid and not valid:
            continue

        validated.append(group)

    return validated


# -----------------------------------------------------------------------------
# 6.14. Detecção integrada de grupos carregados
# -----------------------------------------------------------------------------

def detect_charged_groups(
    molecular_input: Any,
    *,
    config: Optional[PiAnalysisConfig] = None,
    participant_type: Optional[str] = None,
    charge_sign: Optional[str] = None,
    include_known_residue_groups: bool = True,
    include_generic_carboxylates: bool = True,
    include_phosphates: bool = True,
    include_positive_nitrogens: bool = True,
    include_metals: bool = True,
    include_explicit_atomic_charges: bool = True,
    ligand_residue_names: Optional[Collection[str]] = None,
    receptor_residue_names: Optional[Collection[str]] = None,
) -> List[PiChargedGroup]:
    """
    Detect charged groups in a model, residue or atom collection.
    """

    analysis_config = (
        config
        if config is not None
        else create_default_pi_config()
    )

    if not isinstance(
        analysis_config,
        PiAnalysisConfig,
    ):
        raise TypeError(
            "config must be a PiAnalysisConfig or None."
        )

    requested_sign = normalize_charge_sign(
        charge_sign,
        default=CHARGE_UNKNOWN,
    )

    atoms = normalize_atom_collection(
        molecular_input,
        include_hydrogens=True,
        valid_coordinates_only=True,
    )

    residues = normalize_residue_collection(
        atoms
    )

    detected: List[PiChargedGroup] = []

    if include_known_residue_groups:
        for residue in residues:
            residue_participant_type = (
                participant_type
                or infer_participant_type(
                    residue,
                    ligand_residue_names=ligand_residue_names,
                    receptor_residue_names=receptor_residue_names,
                )
            )

            detected.extend(
                detect_known_residue_charged_groups(
                    residue,
                    include_positive=(
                        requested_sign
                        in {
                            CHARGE_POSITIVE,
                            CHARGE_UNKNOWN,
                        }
                    ),
                    include_negative=(
                        requested_sign
                        in {
                            CHARGE_NEGATIVE,
                            CHARGE_UNKNOWN,
                        }
                    ),
                    participant_type=(
                        residue_participant_type
                    ),
                    ligand_residue_names=(
                        ligand_residue_names
                    ),
                    receptor_residue_names=(
                        receptor_residue_names
                    ),
                )
            )

    if (
        include_generic_carboxylates
        and requested_sign
        in {
            CHARGE_NEGATIVE,
            CHARGE_UNKNOWN,
        }
    ):
        detected.extend(
            detect_carboxylate_groups(
                atoms,
                participant_type=participant_type,
            )
        )

    if (
        include_phosphates
        and requested_sign
        in {
            CHARGE_NEGATIVE,
            CHARGE_UNKNOWN,
        }
    ):
        detected.extend(
            detect_phosphate_groups(
                atoms,
                participant_type=participant_type,
            )
        )

    if (
        include_positive_nitrogens
        and requested_sign
        in {
            CHARGE_POSITIVE,
            CHARGE_UNKNOWN,
        }
    ):
        detected.extend(
            detect_positive_nitrogen_groups(
                atoms,
                participant_type=participant_type,
            )
        )

    if (
        include_metals
        and requested_sign
        in {
            CHARGE_POSITIVE,
            CHARGE_UNKNOWN,
        }
    ):
        detected.extend(
            detect_metal_cations(
                atoms,
                participant_type=participant_type,
            )
        )

    if include_explicit_atomic_charges:
        detected.extend(
            detect_explicitly_charged_atom_groups(
                atoms,
                charge_sign=(
                    None
                    if requested_sign == CHARGE_UNKNOWN
                    else requested_sign
                ),
                participant_type=participant_type,
                minimum_charge_magnitude=(
                    analysis_config
                    .minimum_group_charge_magnitude
                ),
            )
        )

    deduplicated = deduplicate_charged_groups(
        detected
    )

    validated = validate_charged_groups(
        deduplicated,
        minimum_charge_magnitude=(
            analysis_config
            .minimum_group_charge_magnitude
        ),
        remove_invalid=True,
    )

    return validated


# -----------------------------------------------------------------------------
# 6.15. Funções específicas para cátions e ânions
# -----------------------------------------------------------------------------

def detect_cationic_groups(
    molecular_input: Any,
    *,
    config: Optional[PiAnalysisConfig] = None,
    participant_type: Optional[str] = None,
    ligand_residue_names: Optional[Collection[str]] = None,
    receptor_residue_names: Optional[Collection[str]] = None,
) -> List[PiChargedGroup]:
    """
    Detect positively charged groups.
    """

    return detect_charged_groups(
        molecular_input,
        config=config,
        participant_type=participant_type,
        charge_sign=CHARGE_POSITIVE,
        ligand_residue_names=ligand_residue_names,
        receptor_residue_names=receptor_residue_names,
    )


def detect_anionic_groups(
    molecular_input: Any,
    *,
    config: Optional[PiAnalysisConfig] = None,
    participant_type: Optional[str] = None,
    ligand_residue_names: Optional[Collection[str]] = None,
    receptor_residue_names: Optional[Collection[str]] = None,
) -> List[PiChargedGroup]:
    """
    Detect negatively charged groups.
    """

    return detect_charged_groups(
        molecular_input,
        config=config,
        participant_type=participant_type,
        charge_sign=CHARGE_NEGATIVE,
        ligand_residue_names=ligand_residue_names,
        receptor_residue_names=receptor_residue_names,
    )


# -----------------------------------------------------------------------------
# 6.16. Detecção para receptor e ligante
# -----------------------------------------------------------------------------

def detect_receptor_charged_groups(
    receptor: Any,
    *,
    config: Optional[PiAnalysisConfig] = None,
    charge_sign: Optional[str] = None,
) -> List[PiChargedGroup]:
    """
    Detect charged groups associated with a receptor.
    """

    return detect_charged_groups(
        receptor,
        config=config,
        participant_type=PARTICIPANT_RECEPTOR,
        charge_sign=charge_sign,
    )


def detect_ligand_charged_groups(
    ligand: Any,
    *,
    config: Optional[PiAnalysisConfig] = None,
    charge_sign: Optional[str] = None,
) -> List[PiChargedGroup]:
    """
    Detect charged groups associated with a ligand.
    """

    return detect_charged_groups(
        ligand,
        config=config,
        participant_type=PARTICIPANT_LIGAND,
        charge_sign=charge_sign,
    )


def detect_pi_analysis_charged_groups(
    normalized_input: PiNormalizedInput,
    *,
    config: Optional[PiAnalysisConfig] = None,
) -> Tuple[
    List[PiChargedGroup],
    List[PiChargedGroup],
]:
    """
    Detect receptor and ligand charged groups.
    """

    if not isinstance(
        normalized_input,
        PiNormalizedInput,
    ):
        raise TypeError(
            "normalized_input must be a PiNormalizedInput."
        )

    analysis_config = (
        config
        if config is not None
        else create_default_pi_config()
    )

    receptor_groups = detect_receptor_charged_groups(
        normalized_input.receptor_atoms,
        config=analysis_config,
    )

    ligand_groups = detect_ligand_charged_groups(
        normalized_input.ligand_atoms,
        config=analysis_config,
    )

    return (
        receptor_groups,
        ligand_groups,
    )


# -----------------------------------------------------------------------------
# 6.17. Separação por sinal
# -----------------------------------------------------------------------------

def split_charged_groups_by_sign(
    groups: Iterable[PiChargedGroup],
) -> Tuple[
    List[PiChargedGroup],
    List[PiChargedGroup],
]:
    """
    Split charged groups into cationic and anionic collections.
    """

    cations: List[PiChargedGroup] = []
    anions: List[PiChargedGroup] = []

    for group in groups:
        if group.charge_sign == CHARGE_POSITIVE:
            cations.append(group)

        elif group.charge_sign == CHARGE_NEGATIVE:
            anions.append(group)

    return cations, anions


def filter_charged_groups_by_participant(
    groups: Iterable[PiChargedGroup],
    participant_type: str,
) -> List[PiChargedGroup]:
    """
    Filter charged groups by molecular participant type.
    """

    normalized_type = str(
        participant_type
    ).strip().lower()

    return [
        group
        for group in groups
        if group.participant_type == normalized_type
    ]


# -----------------------------------------------------------------------------
# 6.18. Geometria entre grupos carregados e anéis
# -----------------------------------------------------------------------------

def calculate_charged_group_ring_geometry(
    group: PiChargedGroup,
    ring: PiRing,
    *,
    config: Optional[PiAnalysisConfig] = None,
    strict: bool = False,
) -> Optional[PiPointRingGeometry]:
    """
    Calculate point-ring geometry for a charged group.
    """

    if not isinstance(
        group,
        PiChargedGroup,
    ):
        raise TypeError(
            "group must be a PiChargedGroup."
        )

    if group.center is None:
        if strict:
            raise PiGeometryError(
                "Charged group center is unavailable."
            )

        return None

    direction = (
        group.direction
        if group.direction is not None
        else group.plane_normal
    )

    return calculate_point_ring_geometry(
        group.center,
        ring,
        direction_vector=direction,
        config=config,
        strict=strict,
    )


def charged_group_is_on_ring_face(
    group: PiChargedGroup,
    ring: PiRing,
    *,
    maximum_radial_offset: float,
    maximum_plane_distance: float,
    config: Optional[PiAnalysisConfig] = None,
) -> bool:
    """
    Return whether a charged group lies above or below a ring face.
    """

    geometry = calculate_charged_group_ring_geometry(
        group,
        ring,
        config=config,
        strict=False,
    )

    if geometry is None:
        return False

    return (
        geometry.radial_offset
        <= maximum_radial_offset
        and geometry.absolute_plane_distance
        <= maximum_plane_distance
    )


# -----------------------------------------------------------------------------
# 6.19. Criação de contatos atômicos preliminares
# -----------------------------------------------------------------------------

def create_charged_group_atomic_contacts(
    group: PiChargedGroup,
    ring: PiRing,
    *,
    maximum_distance: float,
) -> Tuple[PiAtomicContact, ...]:
    """
    Create atomic contacts between charged-group atoms and ring atoms.
    """

    maximum_distance_value = (
        _coerce_non_negative_float(
            maximum_distance,
            field_name="maximum_distance",
        )
    )

    contacts: List[PiAtomicContact] = []

    source_atoms = (
        group.charge_atoms
        if group.charge_atoms
        else group.atoms
    )

    for group_atom in source_atoms:
        group_coordinate = get_atom_coordinate(
            group_atom
        )

        if group_coordinate is None:
            continue

        for ring_atom in ring.atoms:
            ring_coordinate = get_atom_coordinate(
                ring_atom
            )

            if ring_coordinate is None:
                continue

            distance = distance_between_points(
                group_coordinate,
                ring_coordinate,
            )

            if distance > maximum_distance_value:
                continue

            contacts.append(
                PiAtomicContact(
                    atom_1=create_pi_atom_reference(
                        group_atom
                    ),
                    atom_2=create_pi_atom_reference(
                        ring_atom
                    ),
                    distance=distance,
                    contact_type=(
                        CATION_PI
                        if group.charge_sign
                        == CHARGE_POSITIVE
                        else ANION_PI
                    ),
                    metadata={
                        "charged_group_id": (
                            group.group_id
                        ),
                        "ring_id": ring.ring_id,
                    },
                )
            )

    contacts.sort(
        key=lambda contact: contact.distance
    )

    return tuple(contacts)


# -----------------------------------------------------------------------------
# 6.20. Resumo de grupos carregados
# -----------------------------------------------------------------------------

def summarize_charged_groups(
    groups: Iterable[PiChargedGroup],
) -> Dict[str, Any]:
    """
    Generate a serializable summary of charged groups.
    """

    group_list = list(groups)

    positive_groups = [
        group
        for group in group_list
        if group.charge_sign == CHARGE_POSITIVE
    ]

    negative_groups = [
        group
        for group in group_list
        if group.charge_sign == CHARGE_NEGATIVE
    ]

    effective_charges = [
        group.effective_charge
        for group in group_list
        if group.effective_charge is not None
    ]

    charge_magnitudes = [
        abs(charge)
        for charge in effective_charges
    ]

    group_type_distribution = Counter(
        group.group_type
        for group in group_list
    )

    participant_distribution = Counter(
        group.participant_type
        for group in group_list
    )

    residue_distribution = Counter(
        group.residue_name or "UNK"
        for group in group_list
    )

    detection_method_distribution = Counter(
        str(
            group.metadata.get(
                "detection_method",
                "unknown",
            )
        )
        for group in group_list
    )

    return {
        "total_groups": len(group_list),
        "positive_groups": len(positive_groups),
        "negative_groups": len(negative_groups),
        "valid_groups": sum(
            1
            for group in group_list
            if group.valid
        ),
        "invalid_groups": sum(
            1
            for group in group_list
            if not group.valid
        ),
        "group_type_distribution": dict(
            group_type_distribution
        ),
        "participant_distribution": dict(
            participant_distribution
        ),
        "residue_distribution": dict(
            residue_distribution
        ),
        "detection_method_distribution": dict(
            detection_method_distribution
        ),
        "effective_charge": {
            "minimum": (
                min(effective_charges)
                if effective_charges
                else None
            ),
            "mean": (
                sum(effective_charges)
                / len(effective_charges)
                if effective_charges
                else None
            ),
            "maximum": (
                max(effective_charges)
                if effective_charges
                else None
            ),
        },
        "charge_magnitude": {
            "minimum": (
                min(charge_magnitudes)
                if charge_magnitudes
                else None
            ),
            "mean": (
                sum(charge_magnitudes)
                / len(charge_magnitudes)
                if charge_magnitudes
                else None
            ),
            "maximum": (
                max(charge_magnitudes)
                if charge_magnitudes
                else None
            ),
        },
        "group_ids": [
            group.group_id
            for group in group_list
        ],
    }


# -----------------------------------------------------------------------------
# 6.21. Preparação integrada
# -----------------------------------------------------------------------------

def prepare_charged_groups(
    molecular_input: Any,
    *,
    config: Optional[PiAnalysisConfig] = None,
    participant_type: Optional[str] = None,
    charge_sign: Optional[str] = None,
) -> List[PiChargedGroup]:
    """
    Run the complete charged-group preparation pipeline.
    """

    groups = detect_charged_groups(
        molecular_input,
        config=config,
        participant_type=participant_type,
        charge_sign=charge_sign,
    )

    groups = deduplicate_charged_groups(
        groups
    )

    groups = validate_charged_groups(
        groups,
        minimum_charge_magnitude=(
            (
                config
                if config is not None
                else create_default_pi_config()
            ).minimum_group_charge_magnitude
        ),
        remove_invalid=True,
    )

    return groups


def prepare_pi_analysis_charged_groups(
    normalized_input: PiNormalizedInput,
    *,
    config: Optional[PiAnalysisConfig] = None,
) -> Dict[str, List[PiChargedGroup]]:
    """
    Prepare and separate receptor and ligand cations and anions.
    """

    receptor_groups, ligand_groups = (
        detect_pi_analysis_charged_groups(
            normalized_input,
            config=config,
        )
    )

    receptor_cations, receptor_anions = (
        split_charged_groups_by_sign(
            receptor_groups
        )
    )

    ligand_cations, ligand_anions = (
        split_charged_groups_by_sign(
            ligand_groups
        )
    )

    return {
        "receptor_groups": receptor_groups,
        "ligand_groups": ligand_groups,
        "receptor_cations": receptor_cations,
        "receptor_anions": receptor_anions,
        "ligand_cations": ligand_cations,
        "ligand_anions": ligand_anions,
    }

# -----------------------------------------------------------------------------
# End of section 6. 
# -----------------------------------------------------------------------------


# =============================================================================
# 7. DETECÇÃO E GEOMETRIA DE GRUPOS AMIDA
# =============================================================================

# -----------------------------------------------------------------------------
# 7.1. Tipos e constantes auxiliares
# -----------------------------------------------------------------------------

AmideGroupAtomTuple: TypeAlias = Tuple[Any, ...]


AMIDE_GROUP_PEPTIDE: Final[str] = "peptide_amide"
AMIDE_GROUP_PRIMARY: Final[str] = "primary_amide"
AMIDE_GROUP_SECONDARY: Final[str] = "secondary_amide"
AMIDE_GROUP_TERTIARY: Final[str] = "tertiary_amide"
AMIDE_GROUP_CYCLIC: Final[str] = "cyclic_amide"
AMIDE_GROUP_UREA: Final[str] = "urea"
AMIDE_GROUP_CARBAMATE: Final[str] = "carbamate"
AMIDE_GROUP_IMIDE: Final[str] = "imide"
AMIDE_GROUP_LACTAM: Final[str] = "lactam"
AMIDE_GROUP_GENERIC: Final[str] = "generic_amide"


SUPPORTED_AMIDE_GROUP_TYPES: Final[FrozenSet[str]] = frozenset(
    {
        AMIDE_GROUP_PEPTIDE,
        AMIDE_GROUP_PRIMARY,
        AMIDE_GROUP_SECONDARY,
        AMIDE_GROUP_TERTIARY,
        AMIDE_GROUP_CYCLIC,
        AMIDE_GROUP_UREA,
        AMIDE_GROUP_CARBAMATE,
        AMIDE_GROUP_IMIDE,
        AMIDE_GROUP_LACTAM,
        AMIDE_GROUP_GENERIC,
    }
)


DEFAULT_AMIDE_CARBONYL_BOND_MINIMUM: Final[float] = 1.60
DEFAULT_AMIDE_CARBONYL_BOND_MAXIMUM: Final[float] = 2.20

DEFAULT_AMIDE_CN_BOND_MINIMUM: Final[float] = 0.80
DEFAULT_AMIDE_CN_BOND_MAXIMUM: Final[float] = 1.60

DEFAULT_AMIDE_PLANARITY_RMSD: Final[float] = 0.20
DEFAULT_MAXIMUM_AMIDE_PLANARITY_RMSD: Final[float] = 0.35
DEFAULT_MAXIMUM_AMIDE_ATOM_DEVIATION: Final[float] = 0.50

DEFAULT_AMIDE_CARBONYL_DISTANCE_MINIMUM: Final[float] = 1.15
DEFAULT_AMIDE_CARBONYL_DISTANCE_MAXIMUM: Final[float] = 1.40

DEFAULT_AMIDE_CN_DISTANCE_MINIMUM: Final[float] = 1.25
DEFAULT_AMIDE_CN_DISTANCE_MAXIMUM: Final[float] = 1.55

DEFAULT_AMIDE_NEIGHBOR_DISTANCE_MAXIMUM: Final[float] = 1.90

DEFAULT_MINIMUM_AMIDE_ATOM_COUNT: Final[int] = 3

DEFAULT_AMIDE_NORMAL_ORIENTATION_TOLERANCE: Final[float] = 1.0e-8


PROTEIN_BACKBONE_AMIDE_ATOM_NAMES: Final[Tuple[str, ...]] = (
    "C",
    "O",
    "N",
)


PROTEIN_SIDECHAIN_AMIDE_DEFINITIONS: Final[
    Mapping[str, Tuple[Mapping[str, Any], ...]]
] = {
    "ASN": (
        {
            "group_type": AMIDE_GROUP_PRIMARY,
            "carbonyl_carbon": "CG",
            "carbonyl_oxygen": "OD1",
            "amide_nitrogen": "ND2",
            "support_atom_names": ("CB",),
        },
    ),
    "GLN": (
        {
            "group_type": AMIDE_GROUP_PRIMARY,
            "carbonyl_carbon": "CD",
            "carbonyl_oxygen": "OE1",
            "amide_nitrogen": "NE2",
            "support_atom_names": ("CG",),
        },
    ),
}


COMMON_AMIDE_ATOM_TYPE_PATTERNS: Final[Tuple[str, ...]] = (
    "AM",
    "NPL",
    "NPL3",
    "N.AM",
    "N_AM",
    "NAM",
    "C.AM",
    "C_AM",
)


# -----------------------------------------------------------------------------
# 7.2. Normalização do tipo de amida
# -----------------------------------------------------------------------------

def normalize_amide_group_type(
    value: Any,
    *,
    default: str = AMIDE_GROUP_GENERIC,
) -> str:
    """
    Normalize an amide-group type.
    """

    if value is None:
        return default

    normalized = str(value).strip().lower()

    aliases = {
        "peptide": AMIDE_GROUP_PEPTIDE,
        "peptide_amide": AMIDE_GROUP_PEPTIDE,
        "backbone": AMIDE_GROUP_PEPTIDE,
        "backbone_amide": AMIDE_GROUP_PEPTIDE,
        "primary": AMIDE_GROUP_PRIMARY,
        "primary_amide": AMIDE_GROUP_PRIMARY,
        "secondary": AMIDE_GROUP_SECONDARY,
        "secondary_amide": AMIDE_GROUP_SECONDARY,
        "tertiary": AMIDE_GROUP_TERTIARY,
        "tertiary_amide": AMIDE_GROUP_TERTIARY,
        "cyclic": AMIDE_GROUP_CYCLIC,
        "cyclic_amide": AMIDE_GROUP_CYCLIC,
        "lactam": AMIDE_GROUP_LACTAM,
        "urea": AMIDE_GROUP_UREA,
        "carbamate": AMIDE_GROUP_CARBAMATE,
        "imide": AMIDE_GROUP_IMIDE,
        "generic": AMIDE_GROUP_GENERIC,
        "amide": AMIDE_GROUP_GENERIC,
    }

    normalized = aliases.get(
        normalized,
        normalized,
    )

    if normalized not in SUPPORTED_AMIDE_GROUP_TYPES:
        return default

    return normalized


# -----------------------------------------------------------------------------
# 7.3. Reconhecimento de ligações carbonila
# -----------------------------------------------------------------------------

def is_carbonyl_carbon(
    atom: Any,
    *,
    allow_distance_inference: bool = True,
) -> bool:
    """
    Return whether an atom behaves as a carbonyl carbon.
    """

    if get_atom_element(atom) != "C":
        return False

    bonded_oxygens = tuple(
        neighbor
        for neighbor in get_bonded_atoms(
            atom,
            include_hydrogens=False,
        )
        if get_atom_element(neighbor) == "O"
    )

    if not bonded_oxygens:
        return False

    for oxygen in bonded_oxygens:
        bond_order = get_bond_order(
            atom,
            oxygen,
        )

        if (
            bond_order is not None
            and DEFAULT_AMIDE_CARBONYL_BOND_MINIMUM
            <= bond_order
            <= DEFAULT_AMIDE_CARBONYL_BOND_MAXIMUM
        ):
            return True

        if is_aromatic_bond(atom, oxygen):
            return True

        if not allow_distance_inference:
            continue

        carbon_coordinate = get_atom_coordinate(atom)
        oxygen_coordinate = get_atom_coordinate(oxygen)

        if (
            carbon_coordinate is None
            or oxygen_coordinate is None
        ):
            continue

        distance = distance_between_points(
            carbon_coordinate,
            oxygen_coordinate,
        )

        if (
            DEFAULT_AMIDE_CARBONYL_DISTANCE_MINIMUM
            <= distance
            <= DEFAULT_AMIDE_CARBONYL_DISTANCE_MAXIMUM
        ):
            return True

    return False


def get_carbonyl_oxygens(
    carbon: Any,
    *,
    allow_distance_inference: bool = True,
) -> Tuple[Any, ...]:
    """
    Return oxygen atoms behaving as carbonyl oxygens.
    """

    if get_atom_element(carbon) != "C":
        return ()

    carbon_coordinate = get_atom_coordinate(carbon)

    oxygens: List[Any] = []

    for neighbor in get_bonded_atoms(
        carbon,
        include_hydrogens=False,
    ):
        if get_atom_element(neighbor) != "O":
            continue

        bond_order = get_bond_order(
            carbon,
            neighbor,
        )

        accepted = (
            bond_order is not None
            and DEFAULT_AMIDE_CARBONYL_BOND_MINIMUM
            <= bond_order
            <= DEFAULT_AMIDE_CARBONYL_BOND_MAXIMUM
        )

        if is_aromatic_bond(carbon, neighbor):
            accepted = True

        if (
            not accepted
            and allow_distance_inference
            and carbon_coordinate is not None
        ):
            oxygen_coordinate = get_atom_coordinate(neighbor)

            if oxygen_coordinate is not None:
                distance = distance_between_points(
                    carbon_coordinate,
                    oxygen_coordinate,
                )

                accepted = (
                    DEFAULT_AMIDE_CARBONYL_DISTANCE_MINIMUM
                    <= distance
                    <= DEFAULT_AMIDE_CARBONYL_DISTANCE_MAXIMUM
                )

        if accepted:
            oxygens.append(neighbor)

    return tuple(oxygens)


# -----------------------------------------------------------------------------
# 7.4. Reconhecimento de nitrogênio amídico
# -----------------------------------------------------------------------------

def nitrogen_has_amide_atom_type(
    nitrogen: Any,
) -> bool:
    """
    Return whether an atom-type label indicates an amide nitrogen.
    """

    if get_atom_element(nitrogen) != "N":
        return False

    atom_type = get_atom_type(nitrogen)

    if atom_type is None:
        return False

    normalized = str(atom_type).strip().upper()

    return any(
        pattern in normalized
        for pattern in COMMON_AMIDE_ATOM_TYPE_PATTERNS
    )


def is_amide_carbon_nitrogen_bond(
    carbon: Any,
    nitrogen: Any,
    *,
    allow_distance_inference: bool = True,
) -> bool:
    """
    Return whether a carbon–nitrogen bond is compatible with an amide.
    """

    if get_atom_element(carbon) != "C":
        return False

    if get_atom_element(nitrogen) != "N":
        return False

    if not atoms_are_bonded(
        carbon,
        nitrogen,
    ):
        return False

    bond_order = get_bond_order(
        carbon,
        nitrogen,
    )

    if bond_order is not None:
        if (
            DEFAULT_AMIDE_CN_BOND_MINIMUM
            <= bond_order
            <= DEFAULT_AMIDE_CN_BOND_MAXIMUM
        ):
            return True

    if nitrogen_has_amide_atom_type(nitrogen):
        return True

    if not allow_distance_inference:
        return False

    carbon_coordinate = get_atom_coordinate(carbon)
    nitrogen_coordinate = get_atom_coordinate(nitrogen)

    if (
        carbon_coordinate is None
        or nitrogen_coordinate is None
    ):
        return False

    distance = distance_between_points(
        carbon_coordinate,
        nitrogen_coordinate,
    )

    return (
        DEFAULT_AMIDE_CN_DISTANCE_MINIMUM
        <= distance
        <= DEFAULT_AMIDE_CN_DISTANCE_MAXIMUM
    )


def get_amide_nitrogens_for_carbonyl(
    carbonyl_carbon: Any,
    *,
    allow_distance_inference: bool = True,
) -> Tuple[Any, ...]:
    """
    Return nitrogen atoms bound to a carbonyl carbon in an amide-like pattern.
    """

    if not is_carbonyl_carbon(
        carbonyl_carbon,
        allow_distance_inference=allow_distance_inference,
    ):
        return ()

    nitrogens = tuple(
        neighbor
        for neighbor in get_bonded_atoms(
            carbonyl_carbon,
            include_hydrogens=False,
        )
        if (
            get_atom_element(neighbor) == "N"
            and is_amide_carbon_nitrogen_bond(
                carbonyl_carbon,
                neighbor,
                allow_distance_inference=allow_distance_inference,
            )
        )
    )

    return nitrogens


# -----------------------------------------------------------------------------
# 7.5. Classificação estrutural de grupos amida
# -----------------------------------------------------------------------------

def count_nitrogen_non_hydrogen_substituents(
    nitrogen: Any,
    *,
    exclude_atoms: Optional[Collection[Any]] = None,
) -> int:
    """
    Count heavy-atom substituents attached to an amide nitrogen.
    """

    excluded_ids = {
        id(atom)
        for atom in (
            exclude_atoms or ()
        )
    }

    return sum(
        1
        for neighbor in get_bonded_atoms(
            nitrogen,
            include_hydrogens=False,
        )
        if id(neighbor) not in excluded_ids
    )


def detect_amide_hydrogen_count(
    nitrogen: Any,
) -> int:
    """
    Count explicit hydrogens bound to an amide nitrogen.
    """

    return sum(
        1
        for neighbor in get_bonded_atoms(
            nitrogen,
            include_hydrogens=True,
        )
        if is_hydrogen_atom(neighbor)
    )


def carbonyl_carbon_has_second_heteroatom(
    carbonyl_carbon: Any,
    *,
    excluded_atoms: Optional[Collection[Any]] = None,
) -> Tuple[bool, Optional[str]]:
    """
    Detect additional nitrogen or oxygen substitution at a carbonyl carbon.
    """

    excluded_ids = {
        id(atom)
        for atom in (
            excluded_atoms or ()
        )
    }

    heteroatoms = tuple(
        neighbor
        for neighbor in get_bonded_atoms(
            carbonyl_carbon,
            include_hydrogens=False,
        )
        if (
            id(neighbor) not in excluded_ids
            and get_atom_element(neighbor)
            in {
                "N",
                "O",
                "S",
            }
        )
    )

    if not heteroatoms:
        return False, None

    elements = {
        get_atom_element(atom)
        for atom in heteroatoms
    }

    if "N" in elements:
        return True, "nitrogen"

    if "O" in elements:
        return True, "oxygen"

    if "S" in elements:
        return True, "sulfur"

    return True, "other"


def infer_amide_group_type(
    carbonyl_carbon: Any,
    carbonyl_oxygen: Any,
    amide_nitrogen: Any,
    *,
    residue: Optional[Any] = None,
    is_peptide: bool = False,
    is_cyclic: bool = False,
) -> str:
    """
    Infer the most likely amide-group class.
    """

    if is_peptide:
        return AMIDE_GROUP_PEPTIDE

    if is_cyclic:
        return AMIDE_GROUP_LACTAM

    has_second_heteroatom, heteroatom_type = (
        carbonyl_carbon_has_second_heteroatom(
            carbonyl_carbon,
            excluded_atoms=(
                carbonyl_oxygen,
                amide_nitrogen,
            ),
        )
    )

    if has_second_heteroatom:
        if heteroatom_type == "nitrogen":
            return AMIDE_GROUP_UREA

        if heteroatom_type == "oxygen":
            return AMIDE_GROUP_CARBAMATE

    substituent_count = count_nitrogen_non_hydrogen_substituents(
        amide_nitrogen,
        exclude_atoms=(carbonyl_carbon,),
    )

    hydrogen_count = detect_amide_hydrogen_count(
        amide_nitrogen
    )

    if hydrogen_count >= 2:
        return AMIDE_GROUP_PRIMARY

    if hydrogen_count == 1:
        return AMIDE_GROUP_SECONDARY

    if substituent_count >= 2:
        return AMIDE_GROUP_TERTIARY

    if substituent_count == 1:
        return AMIDE_GROUP_SECONDARY

    return AMIDE_GROUP_PRIMARY


# -----------------------------------------------------------------------------
# 7.6. Identificação de ligação peptídica
# -----------------------------------------------------------------------------

def residues_are_sequential(
    residue_1: Any,
    residue_2: Any,
) -> bool:
    """
    Return whether two residues appear sequential in the same chain.
    """

    chain_1 = get_residue_chain_id(residue_1)
    chain_2 = get_residue_chain_id(residue_2)

    if chain_1 != chain_2:
        return False

    number_1 = get_residue_number(residue_1)
    number_2 = get_residue_number(residue_2)

    if (
        not isinstance(number_1, int)
        or not isinstance(number_2, int)
    ):
        return True

    return abs(number_1 - number_2) == 1


def is_protein_peptide_amide(
    carbonyl_carbon: Any,
    carbonyl_oxygen: Any,
    amide_nitrogen: Any,
) -> bool:
    """
    Return whether atoms form a protein backbone peptide bond.
    """

    carbon_name = get_atom_name(
        carbonyl_carbon
    ).upper()

    oxygen_name = get_atom_name(
        carbonyl_oxygen
    ).upper()

    nitrogen_name = get_atom_name(
        amide_nitrogen
    ).upper()

    if (
        carbon_name != "C"
        or oxygen_name != "O"
        or nitrogen_name != "N"
    ):
        return False

    carbon_residue = get_atom_residue(
        carbonyl_carbon
    )

    nitrogen_residue = get_atom_residue(
        amide_nitrogen
    )

    if (
        carbon_residue is None
        or nitrogen_residue is None
    ):
        return False

    if not (
        is_standard_amino_acid_residue(
            carbon_residue
        )
        and is_standard_amino_acid_residue(
            nitrogen_residue
        )
    ):
        return False

    return (
        carbon_residue is nitrogen_residue
        or residues_are_sequential(
            carbon_residue,
            nitrogen_residue,
        )
    )


# -----------------------------------------------------------------------------
# 7.7. Detecção de ciclos contendo amida
# -----------------------------------------------------------------------------

def atoms_belong_to_same_cycle(
    atom_1: Any,
    atom_2: Any,
    molecular_atoms: Iterable[Any],
    *,
    minimum_size: int = 4,
    maximum_size: int = 12,
) -> bool:
    """
    Return whether two atoms occur in at least one common molecular cycle.
    """

    cycles = find_simple_cycles(
        molecular_atoms,
        minimum_size=minimum_size,
        maximum_size=maximum_size,
        heavy_atoms_only=True,
    )

    atom_1_id = id(atom_1)
    atom_2_id = id(atom_2)

    return any(
        atom_1_id in {
            id(atom)
            for atom in cycle
        }
        and atom_2_id in {
            id(atom)
            for atom in cycle
        }
        for cycle in cycles
    )


# -----------------------------------------------------------------------------
# 7.8. Seleção de átomos de suporte
# -----------------------------------------------------------------------------

def get_amide_support_atoms(
    carbonyl_carbon: Any,
    carbonyl_oxygen: Any,
    amide_nitrogen: Any,
    *,
    include_nitrogen_substituents: bool = True,
    include_carbonyl_substituents: bool = True,
) -> Tuple[Any, ...]:
    """
    Return atoms attached to the amide core that define its orientation.
    """

    excluded_ids = {
        id(carbonyl_carbon),
        id(carbonyl_oxygen),
        id(amide_nitrogen),
    }

    support_atoms: List[Any] = []

    if include_carbonyl_substituents:
        support_atoms.extend(
            neighbor
            for neighbor in get_bonded_atoms(
                carbonyl_carbon,
                include_hydrogens=False,
            )
            if id(neighbor) not in excluded_ids
        )

    if include_nitrogen_substituents:
        support_atoms.extend(
            neighbor
            for neighbor in get_bonded_atoms(
                amide_nitrogen,
                include_hydrogens=False,
            )
            if id(neighbor) not in excluded_ids
        )

    return deduplicate_atoms(
        support_atoms
    )


def get_complete_amide_geometry_atoms(
    carbonyl_carbon: Any,
    carbonyl_oxygen: Any,
    amide_nitrogen: Any,
    *,
    support_atoms: Optional[Iterable[Any]] = None,
) -> Tuple[Any, ...]:
    """
    Return the atom set used for amide-plane fitting.
    """

    atoms = [
        carbonyl_carbon,
        carbonyl_oxygen,
        amide_nitrogen,
    ]

    atoms.extend(
        support_atoms or ()
    )

    return tuple(
        atom
        for atom in deduplicate_atoms(atoms)
        if (
            is_heavy_atom(atom)
            and atom_has_valid_coordinate(atom)
        )
    )


# -----------------------------------------------------------------------------
# 7.9. Centro geométrico do grupo amida
# -----------------------------------------------------------------------------

def calculate_amide_group_center(
    carbonyl_carbon: Any,
    carbonyl_oxygen: Any,
    amide_nitrogen: Any,
    *,
    method: str = "core_centroid",
) -> Coordinate3D:
    """
    Calculate the chemically relevant center of an amide group.
    """

    normalized_method = str(
        method
    ).strip().lower()

    carbon_coordinate = require_atom_coordinate(
        carbonyl_carbon
    )

    oxygen_coordinate = require_atom_coordinate(
        carbonyl_oxygen
    )

    nitrogen_coordinate = require_atom_coordinate(
        amide_nitrogen
    )

    if normalized_method == "carbonyl_carbon":
        return carbon_coordinate

    if normalized_method == "carbonyl_midpoint":
        return midpoint(
            carbon_coordinate,
            oxygen_coordinate,
        )

    if normalized_method == "cn_midpoint":
        return midpoint(
            carbon_coordinate,
            nitrogen_coordinate,
        )

    if normalized_method == "core_centroid":
        return calculate_centroid(
            (
                carbon_coordinate,
                oxygen_coordinate,
                nitrogen_coordinate,
            )
        )

    raise ValueError(
        "method must be 'carbonyl_carbon', "
        "'carbonyl_midpoint', 'cn_midpoint' or "
        "'core_centroid'."
    )


# -----------------------------------------------------------------------------
# 7.10. Vetores direcionais do grupo amida
# -----------------------------------------------------------------------------

def calculate_carbonyl_direction(
    carbonyl_carbon: Any,
    carbonyl_oxygen: Any,
) -> Vector3D:
    """
    Return the unit vector from carbonyl carbon toward oxygen.
    """

    carbon_coordinate = require_atom_coordinate(
        carbonyl_carbon
    )

    oxygen_coordinate = require_atom_coordinate(
        carbonyl_oxygen
    )

    direction = normalize_vector(
        subtract_vectors(
            oxygen_coordinate,
            carbon_coordinate,
        ),
        strict=True,
    )

    assert direction is not None

    return direction


def calculate_amide_cn_direction(
    carbonyl_carbon: Any,
    amide_nitrogen: Any,
) -> Vector3D:
    """
    Return the unit vector from carbonyl carbon toward amide nitrogen.
    """

    carbon_coordinate = require_atom_coordinate(
        carbonyl_carbon
    )

    nitrogen_coordinate = require_atom_coordinate(
        amide_nitrogen
    )

    direction = normalize_vector(
        subtract_vectors(
            nitrogen_coordinate,
            carbon_coordinate,
        ),
        strict=True,
    )

    assert direction is not None

    return direction


def calculate_amide_bisector_direction(
    carbonyl_carbon: Any,
    carbonyl_oxygen: Any,
    amide_nitrogen: Any,
) -> Optional[Vector3D]:
    """
    Calculate the in-plane bisector between C=O and C–N directions.
    """

    carbonyl_direction = calculate_carbonyl_direction(
        carbonyl_carbon,
        carbonyl_oxygen,
    )

    cn_direction = calculate_amide_cn_direction(
        carbonyl_carbon,
        amide_nitrogen,
    )

    bisector = add_vectors(
        carbonyl_direction,
        cn_direction,
    )

    return normalize_vector(
        bisector
    )


# -----------------------------------------------------------------------------
# 7.11. Ajuste do plano da amida
# -----------------------------------------------------------------------------

def fit_amide_plane(
    carbonyl_carbon: Any,
    carbonyl_oxygen: Any,
    amide_nitrogen: Any,
    *,
    support_atoms: Optional[Iterable[Any]] = None,
    prefer_numpy: bool = True,
    strict: bool = False,
) -> Optional[PiPlaneGeometry]:
    """
    Fit a plane to an amide group.
    """

    geometry_atoms = get_complete_amide_geometry_atoms(
        carbonyl_carbon,
        carbonyl_oxygen,
        amide_nitrogen,
        support_atoms=support_atoms,
    )

    if len(geometry_atoms) < 3:
        if strict:
            raise PiGeometryError(
                "At least three valid atoms are required "
                "to fit an amide plane."
            )

        return None

    return fit_plane_to_atoms(
        geometry_atoms,
        skip_invalid=False,
        prefer_numpy=prefer_numpy,
        strict=strict,
    )


def orient_amide_normal(
    normal: Sequence[Number],
    carbonyl_carbon: Any,
    carbonyl_oxygen: Any,
    amide_nitrogen: Any,
) -> Vector3D:
    """
    Orient an amide normal deterministically using its core geometry.
    """

    normalized_normal = normalize_vector(
        normal,
        strict=True,
    )

    assert normalized_normal is not None

    carbonyl_direction = (
        calculate_carbonyl_direction(
            carbonyl_carbon,
            carbonyl_oxygen,
        )
    )

    cn_direction = calculate_amide_cn_direction(
        carbonyl_carbon,
        amide_nitrogen,
    )

    local_cross = normalize_vector(
        cross_product(
            carbonyl_direction,
            cn_direction,
        )
    )

    if local_cross is None:
        return orient_normal_deterministically(
            normalized_normal
        )

    return align_normal_to_reference(
        normalized_normal,
        local_cross,
    )


# -----------------------------------------------------------------------------
# 7.12. Planaridade da amida
# -----------------------------------------------------------------------------

def calculate_amide_planarity_score(
    planarity_rmsd: Optional[Number],
    *,
    maximum_rmsd: float = (
        DEFAULT_MAXIMUM_AMIDE_PLANARITY_RMSD
    ),
) -> float:
    """
    Convert amide planarity RMSD into a normalized score.
    """

    rmsd = _normalize_optional_numeric(
        planarity_rmsd
    )

    if rmsd is None:
        return 0.0

    maximum = _coerce_non_negative_float(
        maximum_rmsd,
        field_name="maximum_rmsd",
    )

    if maximum <= 0.0:
        return 1.0 if rmsd <= 0.0 else 0.0

    return max(
        0.0,
        min(
            1.0,
            1.0 - rmsd / maximum,
        ),
    )


def classify_amide_planarity(
    planarity_rmsd: Optional[Number],
    *,
    preferred_rmsd: float = DEFAULT_AMIDE_PLANARITY_RMSD,
    maximum_rmsd: float = (
        DEFAULT_MAXIMUM_AMIDE_PLANARITY_RMSD
    ),
) -> str:
    """
    Classify amide planarity.
    """

    rmsd = _normalize_optional_numeric(
        planarity_rmsd
    )

    if rmsd is None:
        return GEOMETRY_REJECTED

    if rmsd <= preferred_rmsd:
        return GEOMETRY_OPTIMAL

    if rmsd <= maximum_rmsd:
        return GEOMETRY_FAVORABLE

    return GEOMETRY_REJECTED


# -----------------------------------------------------------------------------
# 7.13. Construção de PiAmideGroup
# -----------------------------------------------------------------------------

def create_pi_amide_group(
    carbonyl_carbon: Any,
    carbonyl_oxygen: Any,
    amide_nitrogen: Any,
    *,
    group_index: Optional[int] = None,
    group_type: Optional[str] = None,
    participant_type: Optional[str] = None,
    support_atoms: Optional[Iterable[Any]] = None,
    is_peptide: bool = False,
    is_cyclic: bool = False,
    prefer_numpy: bool = True,
    ligand_residue_names: Optional[Collection[str]] = None,
    receptor_residue_names: Optional[Collection[str]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    strict: bool = False,
) -> Optional[PiAmideGroup]:
    """
    Create a complete ``PiAmideGroup`` from an amide core.
    """

    try:
        if get_atom_element(
            carbonyl_carbon
        ) != "C":
            raise ValueError(
                "carbonyl_carbon must be carbon."
            )

        if get_atom_element(
            carbonyl_oxygen
        ) != "O":
            raise ValueError(
                "carbonyl_oxygen must be oxygen."
            )

        if get_atom_element(
            amide_nitrogen
        ) != "N":
            raise ValueError(
                "amide_nitrogen must be nitrogen."
            )

        if not atoms_are_bonded(
            carbonyl_carbon,
            carbonyl_oxygen,
        ):
            raise ValueError(
                "Carbonyl carbon and oxygen are not bonded."
            )

        if not atoms_are_bonded(
            carbonyl_carbon,
            amide_nitrogen,
        ):
            raise ValueError(
                "Carbonyl carbon and amide nitrogen are not bonded."
            )

        normalized_support_atoms = (
            tuple(support_atoms)
            if support_atoms is not None
            else get_amide_support_atoms(
                carbonyl_carbon,
                carbonyl_oxygen,
                amide_nitrogen,
            )
        )

        normalized_support_atoms = tuple(
            atom
            for atom in deduplicate_atoms(
                normalized_support_atoms
            )
            if atom_has_valid_coordinate(atom)
        )

        core_atoms = (
            carbonyl_carbon,
            carbonyl_oxygen,
            amide_nitrogen,
        )

        group_atoms = get_complete_amide_geometry_atoms(
            carbonyl_carbon,
            carbonyl_oxygen,
            amide_nitrogen,
            support_atoms=normalized_support_atoms,
        )

        plane = fit_amide_plane(
            carbonyl_carbon,
            carbonyl_oxygen,
            amide_nitrogen,
            support_atoms=normalized_support_atoms,
            prefer_numpy=prefer_numpy,
            strict=True,
        )

        assert plane is not None

        oriented_normal = orient_amide_normal(
            plane.normal,
            carbonyl_carbon,
            carbonyl_oxygen,
            amide_nitrogen,
        )

        center = calculate_amide_group_center(
            carbonyl_carbon,
            carbonyl_oxygen,
            amide_nitrogen,
            method="core_centroid",
        )

        carbonyl_direction = (
            calculate_carbonyl_direction(
                carbonyl_carbon,
                carbonyl_oxygen,
            )
        )

        cn_direction = calculate_amide_cn_direction(
            carbonyl_carbon,
            amide_nitrogen,
        )

        bisector_direction = (
            calculate_amide_bisector_direction(
                carbonyl_carbon,
                carbonyl_oxygen,
                amide_nitrogen,
            )
        )

        carbonyl_residue = get_atom_residue(
            carbonyl_carbon
        )

        nitrogen_residue = get_atom_residue(
            amide_nitrogen
        )

        residue = (
            carbonyl_residue
            if carbonyl_residue is not None
            else nitrogen_residue
        )

        model = get_atom_model(
            carbonyl_carbon
        )

        normalized_participant_type = (
            participant_type
            or infer_participant_type(
                residue or carbonyl_carbon,
                ligand_residue_names=(
                    ligand_residue_names
                ),
                receptor_residue_names=(
                    receptor_residue_names
                ),
            )
        )

        inferred_group_type = (
            normalize_amide_group_type(
                group_type
            )
            if group_type is not None
            else infer_amide_group_type(
                carbonyl_carbon,
                carbonyl_oxygen,
                amide_nitrogen,
                residue=residue,
                is_peptide=is_peptide,
                is_cyclic=is_cyclic,
            )
        )

        carbonyl_bond_order = get_bond_order(
            carbonyl_carbon,
            carbonyl_oxygen,
        )

        cn_bond_order = get_bond_order(
            carbonyl_carbon,
            amide_nitrogen,
        )

        carbonyl_distance = distance_between_points(
            require_atom_coordinate(
                carbonyl_carbon
            ),
            require_atom_coordinate(
                carbonyl_oxygen
            ),
        )

        cn_distance = distance_between_points(
            require_atom_coordinate(
                carbonyl_carbon
            ),
            require_atom_coordinate(
                amide_nitrogen
            ),
        )

        group_metadata = _copy_mapping(
            metadata
        )

        group_metadata.update(
            {
                "plane_fit_method": plane.method,
                "plane_eigenvalues": list(
                    plane.eigenvalues
                ),
                "signed_plane_deviations": list(
                    plane.signed_deviations
                ),
                "carbonyl_bond_order": (
                    carbonyl_bond_order
                ),
                "cn_bond_order": cn_bond_order,
                "carbonyl_distance": (
                    carbonyl_distance
                ),
                "cn_distance": cn_distance,
                "carbonyl_direction": list(
                    carbonyl_direction
                ),
                "cn_direction": list(
                    cn_direction
                ),
                "bisector_direction": (
                    list(bisector_direction)
                    if bisector_direction is not None
                    else None
                ),
                "is_peptide": bool(is_peptide),
                "is_cyclic": bool(is_cyclic),
            }
        )

        group = PiAmideGroup(
            atoms=group_atoms,
            atom_references=create_pi_atom_references(
                group_atoms,
                skip_invalid=False,
            ),
            core_atoms=core_atoms,
            support_atoms=normalized_support_atoms,
            carbonyl_carbon=carbonyl_carbon,
            carbonyl_oxygen=carbonyl_oxygen,
            amide_nitrogen=amide_nitrogen,
            group_index=group_index,
            group_type=inferred_group_type,
            center=center,
            normal=oriented_normal,
            direction=carbonyl_direction,
            carbonyl_direction=carbonyl_direction,
            cn_direction=cn_direction,
            planarity_rmsd=plane.planarity_rmsd,
            maximum_plane_deviation=(
                plane.maximum_deviation
            ),
            carbonyl_bond_order=carbonyl_bond_order,
            cn_bond_order=cn_bond_order,
            carbonyl_distance=carbonyl_distance,
            cn_distance=cn_distance,
            residue_name=get_residue_name(
                residue or carbonyl_carbon
            ),
            residue_number=get_residue_number(
                residue or carbonyl_carbon
            ),
            chain_id=get_residue_chain_id(
                residue or carbonyl_carbon
            ),
            model_id=get_model_identifier(model),
            participant_type=normalized_participant_type,
            is_peptide=is_peptide,
            is_cyclic=is_cyclic,
            valid=True,
            metadata=group_metadata,
        )

        if not group.group_id:
            group.group_id = group.build_group_id()

        return group

    except (
        PiGeometryError,
        PiAtomAccessError,
        PiCoordinateError,
        TypeError,
        ValueError,
        ArithmeticError,
    ):
        if strict:
            raise

        return None


# -----------------------------------------------------------------------------
# 7.14. Detecção por definição conhecida de resíduos
# -----------------------------------------------------------------------------

def detect_sidechain_amide_groups(
    residue: Any,
    *,
    participant_type: Optional[str] = None,
    prefer_numpy: bool = True,
    ligand_residue_names: Optional[Collection[str]] = None,
    receptor_residue_names: Optional[Collection[str]] = None,
) -> List[PiAmideGroup]:
    """
    Detect side-chain amides in standard amino acids.
    """

    residue_name = get_residue_name(
        residue
    )

    definitions = (
        PROTEIN_SIDECHAIN_AMIDE_DEFINITIONS.get(
            residue_name,
            (),
        )
    )

    if not definitions:
        return []

    atom_map = map_atoms_by_name(
        residue
    )

    groups: List[PiAmideGroup] = []

    for definition in definitions:
        carbon = atom_map.get(
            str(
                definition[
                    "carbonyl_carbon"
                ]
            ).upper()
        )

        oxygen = atom_map.get(
            str(
                definition[
                    "carbonyl_oxygen"
                ]
            ).upper()
        )

        nitrogen = atom_map.get(
            str(
                definition[
                    "amide_nitrogen"
                ]
            ).upper()
        )

        if (
            carbon is None
            or oxygen is None
            or nitrogen is None
        ):
            continue

        support_atoms = tuple(
            atom_map[name.upper()]
            for name in definition.get(
                "support_atom_names",
                (),
            )
            if name.upper() in atom_map
        )

        group = create_pi_amide_group(
            carbon,
            oxygen,
            nitrogen,
            group_index=len(groups) + 1,
            group_type=definition.get(
                "group_type",
                AMIDE_GROUP_PRIMARY,
            ),
            participant_type=participant_type,
            support_atoms=support_atoms,
            is_peptide=False,
            is_cyclic=False,
            prefer_numpy=prefer_numpy,
            ligand_residue_names=(
                ligand_residue_names
            ),
            receptor_residue_names=(
                receptor_residue_names
            ),
            metadata={
                "detection_method": (
                    "known_sidechain_definition"
                ),
                "residue_definition": dict(
                    definition
                ),
            },
        )

        if group is not None:
            groups.append(group)

    return groups


# -----------------------------------------------------------------------------
# 7.15. Detecção de ligações peptídicas
# -----------------------------------------------------------------------------

def detect_peptide_amide_groups(
    molecular_input: Any,
    *,
    participant_type: Optional[str] = None,
    prefer_numpy: bool = True,
    include_proline: bool = True,
) -> List[PiAmideGroup]:
    """
    Detect backbone peptide bonds in a protein structure.
    """

    atoms = normalize_atom_collection(
        molecular_input,
        include_hydrogens=True,
        valid_coordinates_only=True,
    )

    atom_ids = {
        id(atom)
        for atom in atoms
    }

    groups: List[PiAmideGroup] = []

    for carbon in atoms:
        if get_atom_name(carbon).upper() != "C":
            continue

        carbon_residue = get_atom_residue(
            carbon
        )

        if (
            carbon_residue is None
            or not is_standard_amino_acid_residue(
                carbon_residue
            )
        ):
            continue

        oxygen_candidates = tuple(
            neighbor
            for neighbor in get_bonded_atoms(
                carbon,
                include_hydrogens=False,
            )
            if (
                id(neighbor) in atom_ids
                and get_atom_name(
                    neighbor
                ).upper() == "O"
            )
        )

        if not oxygen_candidates:
            continue

        nitrogen_candidates = tuple(
            neighbor
            for neighbor in get_bonded_atoms(
                carbon,
                include_hydrogens=False,
            )
            if (
                id(neighbor) in atom_ids
                and get_atom_name(
                    neighbor
                ).upper() == "N"
            )
        )

        for oxygen in oxygen_candidates:
            for nitrogen in nitrogen_candidates:
                nitrogen_residue = get_atom_residue(
                    nitrogen
                )

                if nitrogen_residue is None:
                    continue

                if (
                    not include_proline
                    and get_residue_name(
                        nitrogen_residue
                    ) == "PRO"
                ):
                    continue

                if not is_protein_peptide_amide(
                    carbon,
                    oxygen,
                    nitrogen,
                ):
                    continue

                support_atoms = (
                    get_amide_support_atoms(
                        carbon,
                        oxygen,
                        nitrogen,
                    )
                )

                group = create_pi_amide_group(
                    carbon,
                    oxygen,
                    nitrogen,
                    group_index=len(groups) + 1,
                    group_type=AMIDE_GROUP_PEPTIDE,
                    participant_type=(
                        participant_type
                        or PARTICIPANT_RECEPTOR
                    ),
                    support_atoms=support_atoms,
                    is_peptide=True,
                    prefer_numpy=prefer_numpy,
                    metadata={
                        "detection_method": (
                            "protein_backbone_connectivity"
                        ),
                        "carbonyl_residue": (
                            get_residue_identifier(
                                carbon_residue
                            )
                        ),
                        "nitrogen_residue": (
                            get_residue_identifier(
                                nitrogen_residue
                            )
                        ),
                    },
                )

                if group is not None:
                    groups.append(group)

    return groups


# -----------------------------------------------------------------------------
# 7.16. Detecção genérica por conectividade
# -----------------------------------------------------------------------------

def detect_generic_amide_groups(
    molecular_input: Any,
    *,
    participant_type: Optional[str] = None,
    allow_distance_inference: bool = True,
    prefer_numpy: bool = True,
    detect_cyclic_amides: bool = True,
    ligand_residue_names: Optional[Collection[str]] = None,
    receptor_residue_names: Optional[Collection[str]] = None,
) -> List[PiAmideGroup]:
    """
    Detect amide groups using generic carbonyl–nitrogen connectivity.
    """

    atoms = normalize_atom_collection(
        molecular_input,
        include_hydrogens=True,
        valid_coordinates_only=True,
    )

    atom_ids = {
        id(atom)
        for atom in atoms
    }

    groups: List[PiAmideGroup] = []

    for carbon in atoms:
        if not is_carbonyl_carbon(
            carbon,
            allow_distance_inference=(
                allow_distance_inference
            ),
        ):
            continue

        oxygens = tuple(
            oxygen
            for oxygen in get_carbonyl_oxygens(
                carbon,
                allow_distance_inference=(
                    allow_distance_inference
                ),
            )
            if id(oxygen) in atom_ids
        )

        nitrogens = tuple(
            nitrogen
            for nitrogen in get_amide_nitrogens_for_carbonyl(
                carbon,
                allow_distance_inference=(
                    allow_distance_inference
                ),
            )
            if id(nitrogen) in atom_ids
        )

        if not oxygens or not nitrogens:
            continue

        for oxygen in oxygens:
            for nitrogen in nitrogens:
                peptide = is_protein_peptide_amide(
                    carbon,
                    oxygen,
                    nitrogen,
                )

                cyclic = False

                if detect_cyclic_amides:
                    cyclic = atoms_belong_to_same_cycle(
                        carbon,
                        nitrogen,
                        atoms,
                    )

                support_atoms = get_amide_support_atoms(
                    carbon,
                    oxygen,
                    nitrogen,
                )

                group = create_pi_amide_group(
                    carbon,
                    oxygen,
                    nitrogen,
                    group_index=len(groups) + 1,
                    participant_type=participant_type,
                    support_atoms=support_atoms,
                    is_peptide=peptide,
                    is_cyclic=cyclic,
                    prefer_numpy=prefer_numpy,
                    ligand_residue_names=(
                        ligand_residue_names
                    ),
                    receptor_residue_names=(
                        receptor_residue_names
                    ),
                    metadata={
                        "detection_method": (
                            "generic_amide_connectivity"
                        ),
                        "distance_inference_enabled": (
                            allow_distance_inference
                        ),
                    },
                )

                if group is not None:
                    groups.append(group)

    return groups


# -----------------------------------------------------------------------------
# 7.17. Identidade e deduplicação de grupos amida
# -----------------------------------------------------------------------------

def get_amide_group_identity_key(
    group: PiAmideGroup,
) -> Tuple[Any, ...]:
    """
    Return a stable identity key for a ``PiAmideGroup``.
    """

    if not isinstance(
        group,
        PiAmideGroup,
    ):
        raise TypeError(
            "group must be a PiAmideGroup."
        )

    return (
        group.model_id,
        id(group.carbonyl_carbon),
        id(group.carbonyl_oxygen),
        id(group.amide_nitrogen),
    )


def _amide_group_priority(
    group: PiAmideGroup,
) -> Tuple[int, float, int]:
    """
    Return a priority tuple for amide-group deduplication.
    """

    detection_method = str(
        group.metadata.get(
            "detection_method",
            "",
        )
    )

    method_priority = {
        "known_sidechain_definition": 5,
        "protein_backbone_connectivity": 5,
        "generic_amide_connectivity": 3,
    }.get(
        detection_method,
        1,
    )

    planarity_score = (
        calculate_amide_planarity_score(
            group.planarity_rmsd
        )
    )

    return (
        method_priority,
        planarity_score,
        len(group.atoms),
    )


def amide_groups_overlap(
    group_1: PiAmideGroup,
    group_2: PiAmideGroup,
) -> bool:
    """
    Return whether two amide groups describe the same amide core.
    """

    core_ids_1 = {
        id(group_1.carbonyl_carbon),
        id(group_1.carbonyl_oxygen),
        id(group_1.amide_nitrogen),
    }

    core_ids_2 = {
        id(group_2.carbonyl_carbon),
        id(group_2.carbonyl_oxygen),
        id(group_2.amide_nitrogen),
    }

    return len(
        core_ids_1 & core_ids_2
    ) >= 2


def deduplicate_amide_groups(
    groups: Iterable[PiAmideGroup],
) -> List[PiAmideGroup]:
    """
    Remove duplicate or overlapping amide groups.
    """

    group_list = list(groups)

    group_list.sort(
        key=_amide_group_priority,
        reverse=True,
    )

    unique: List[PiAmideGroup] = []

    for candidate in group_list:
        duplicate_index: Optional[int] = None

        for index, existing in enumerate(unique):
            if (
                get_amide_group_identity_key(
                    candidate
                )
                == get_amide_group_identity_key(
                    existing
                )
            ) or amide_groups_overlap(
                candidate,
                existing,
            ):
                duplicate_index = index
                break

        if duplicate_index is None:
            unique.append(candidate)
            continue

        existing = unique[
            duplicate_index
        ]

        if (
            _amide_group_priority(candidate)
            > _amide_group_priority(existing)
        ):
            unique[
                duplicate_index
            ] = candidate

    unique.sort(
        key=lambda group: (
            group.model_id or "",
            group.chain_id or "",
            str(
                group.residue_number or ""
            ),
            group.residue_name or "",
            get_atom_name(
                group.carbonyl_carbon
            ),
            get_atom_name(
                group.amide_nitrogen
            ),
        )
    )

    for group_index, group in enumerate(
        unique,
        start=1,
    ):
        group.group_index = group_index

        if not group.group_id:
            group.group_id = (
                group.build_group_id()
            )

    return unique


# -----------------------------------------------------------------------------
# 7.18. Validação estrutural de grupos amida
# -----------------------------------------------------------------------------

def validate_amide_group(
    group: PiAmideGroup,
    *,
    maximum_planarity_rmsd: float = (
        DEFAULT_MAXIMUM_AMIDE_PLANARITY_RMSD
    ),
    maximum_atom_deviation: float = (
        DEFAULT_MAXIMUM_AMIDE_ATOM_DEVIATION
    ),
    require_complete_geometry: bool = True,
) -> Tuple[bool, Tuple[str, ...]]:
    """
    Validate the structure and geometry of an amide group.
    """

    if not isinstance(
        group,
        PiAmideGroup,
    ):
        raise TypeError(
            "group must be a PiAmideGroup."
        )

    messages: List[str] = []

    if len(group.core_atoms) < 3:
        messages.append(
            "Amide group must contain three core atoms."
        )

    if get_atom_element(
        group.carbonyl_carbon
    ) != "C":
        messages.append(
            "Carbonyl carbon is invalid."
        )

    if get_atom_element(
        group.carbonyl_oxygen
    ) != "O":
        messages.append(
            "Carbonyl oxygen is invalid."
        )

    if get_atom_element(
        group.amide_nitrogen
    ) != "N":
        messages.append(
            "Amide nitrogen is invalid."
        )

    if not atoms_are_bonded(
        group.carbonyl_carbon,
        group.carbonyl_oxygen,
    ):
        messages.append(
            "Carbonyl carbon and oxygen are not bonded."
        )

    if not atoms_are_bonded(
        group.carbonyl_carbon,
        group.amide_nitrogen,
    ):
        messages.append(
            "Carbonyl carbon and nitrogen are not bonded."
        )

    if require_complete_geometry:
        if group.center is None:
            messages.append(
                "Amide center is unavailable."
            )

        if group.normal is None:
            messages.append(
                "Amide plane normal is unavailable."
            )

        elif normalize_vector(
            group.normal
        ) is None:
            messages.append(
                "Amide plane normal is degenerate."
            )

        if group.direction is None:
            messages.append(
                "Amide carbonyl direction is unavailable."
            )

    if group.planarity_rmsd is None:
        messages.append(
            "Amide planarity RMSD is unavailable."
        )

    elif (
        group.planarity_rmsd
        > maximum_planarity_rmsd
    ):
        messages.append(
            "Amide planarity RMSD exceeds the maximum threshold."
        )

    if (
        group.maximum_plane_deviation
        is None
    ):
        messages.append(
            "Maximum amide-plane deviation is unavailable."
        )

    elif (
        group.maximum_plane_deviation
        > maximum_atom_deviation
    ):
        messages.append(
            "Maximum amide-plane deviation exceeds the threshold."
        )

    if group.carbonyl_distance is not None:
        if not (
            DEFAULT_AMIDE_CARBONYL_DISTANCE_MINIMUM
            <= group.carbonyl_distance
            <= DEFAULT_AMIDE_CARBONYL_DISTANCE_MAXIMUM
        ):
            messages.append(
                "Carbonyl bond distance is outside the expected range."
            )

    if group.cn_distance is not None:
        if not (
            DEFAULT_AMIDE_CN_DISTANCE_MINIMUM
            <= group.cn_distance
            <= DEFAULT_AMIDE_CN_DISTANCE_MAXIMUM
        ):
            messages.append(
                "Amide C–N distance is outside the expected range."
            )

    group.valid = not messages

    existing_messages = list(
        group.validation_messages
    )

    existing_messages.extend(
        message
        for message in messages
        if message not in existing_messages
    )

    group.validation_messages = tuple(
        existing_messages
    )

    return (
        group.valid,
        tuple(messages),
    )


def validate_amide_groups(
    groups: Iterable[PiAmideGroup],
    *,
    maximum_planarity_rmsd: float = (
        DEFAULT_MAXIMUM_AMIDE_PLANARITY_RMSD
    ),
    maximum_atom_deviation: float = (
        DEFAULT_MAXIMUM_AMIDE_ATOM_DEVIATION
    ),
    remove_invalid: bool = False,
) -> List[PiAmideGroup]:
    """
    Validate multiple amide groups.
    """

    validated: List[PiAmideGroup] = []

    for group in groups:
        valid, _ = validate_amide_group(
            group,
            maximum_planarity_rmsd=(
                maximum_planarity_rmsd
            ),
            maximum_atom_deviation=(
                maximum_atom_deviation
            ),
        )

        if remove_invalid and not valid:
            continue

        validated.append(group)

    return validated


# -----------------------------------------------------------------------------
# 7.19. Detecção integrada
# -----------------------------------------------------------------------------

def detect_amide_groups(
    molecular_input: Any,
    *,
    config: Optional[PiAnalysisConfig] = None,
    participant_type: Optional[str] = None,
    include_peptide_amides: bool = True,
    include_sidechain_amides: bool = True,
    include_generic_amides: bool = True,
    include_cyclic_amides: bool = True,
    prefer_numpy: bool = True,
    ligand_residue_names: Optional[Collection[str]] = None,
    receptor_residue_names: Optional[Collection[str]] = None,
) -> List[PiAmideGroup]:
    """
    Detect amide groups in a model, residue or atom collection.
    """

    analysis_config = (
        config
        if config is not None
        else create_default_pi_config()
    )

    if not isinstance(
        analysis_config,
        PiAnalysisConfig,
    ):
        raise TypeError(
            "config must be a PiAnalysisConfig or None."
        )

    atoms = normalize_atom_collection(
        molecular_input,
        include_hydrogens=True,
        valid_coordinates_only=True,
    )

    residues = normalize_residue_collection(
        atoms
    )

    detected: List[PiAmideGroup] = []

    if include_peptide_amides:
        detected.extend(
            detect_peptide_amide_groups(
                atoms,
                participant_type=participant_type,
                prefer_numpy=prefer_numpy,
            )
        )

    if include_sidechain_amides:
        for residue in residues:
            residue_participant_type = (
                participant_type
                or infer_participant_type(
                    residue,
                    ligand_residue_names=(
                        ligand_residue_names
                    ),
                    receptor_residue_names=(
                        receptor_residue_names
                    ),
                )
            )

            detected.extend(
                detect_sidechain_amide_groups(
                    residue,
                    participant_type=(
                        residue_participant_type
                    ),
                    prefer_numpy=prefer_numpy,
                    ligand_residue_names=(
                        ligand_residue_names
                    ),
                    receptor_residue_names=(
                        receptor_residue_names
                    ),
                )
            )

    if include_generic_amides:
        detected.extend(
            detect_generic_amide_groups(
                atoms,
                participant_type=participant_type,
                prefer_numpy=prefer_numpy,
                detect_cyclic_amides=(
                    include_cyclic_amides
                ),
                ligand_residue_names=(
                    ligand_residue_names
                ),
                receptor_residue_names=(
                    receptor_residue_names
                ),
            )
        )

    deduplicated = deduplicate_amide_groups(
        detected
    )

    maximum_planarity_rmsd = getattr(
        analysis_config,
        "maximum_amide_planarity_rmsd",
        DEFAULT_MAXIMUM_AMIDE_PLANARITY_RMSD,
    )

    maximum_atom_deviation = getattr(
        analysis_config,
        "maximum_amide_atom_deviation",
        DEFAULT_MAXIMUM_AMIDE_ATOM_DEVIATION,
    )

    return validate_amide_groups(
        deduplicated,
        maximum_planarity_rmsd=(
            maximum_planarity_rmsd
        ),
        maximum_atom_deviation=(
            maximum_atom_deviation
        ),
        remove_invalid=True,
    )


# -----------------------------------------------------------------------------
# 7.20. Detecção específica para receptor e ligante
# -----------------------------------------------------------------------------

def detect_receptor_amide_groups(
    receptor: Any,
    *,
    config: Optional[PiAnalysisConfig] = None,
    include_peptide_amides: bool = True,
) -> List[PiAmideGroup]:
    """
    Detect amide groups associated with a receptor.
    """

    return detect_amide_groups(
        receptor,
        config=config,
        participant_type=PARTICIPANT_RECEPTOR,
        include_peptide_amides=(
            include_peptide_amides
        ),
    )


def detect_ligand_amide_groups(
    ligand: Any,
    *,
    config: Optional[PiAnalysisConfig] = None,
) -> List[PiAmideGroup]:
    """
    Detect amide groups associated with a ligand.
    """

    return detect_amide_groups(
        ligand,
        config=config,
        participant_type=PARTICIPANT_LIGAND,
        include_peptide_amides=False,
    )


def detect_pi_analysis_amide_groups(
    normalized_input: PiNormalizedInput,
    *,
    config: Optional[PiAnalysisConfig] = None,
) -> Tuple[
    List[PiAmideGroup],
    List[PiAmideGroup],
]:
    """
    Detect receptor and ligand amide groups.
    """

    if not isinstance(
        normalized_input,
        PiNormalizedInput,
    ):
        raise TypeError(
            "normalized_input must be a PiNormalizedInput."
        )

    receptor_groups = (
        detect_receptor_amide_groups(
            normalized_input.receptor_atoms,
            config=config,
        )
    )

    ligand_groups = (
        detect_ligand_amide_groups(
            normalized_input.ligand_atoms,
            config=config,
        )
    )

    return (
        receptor_groups,
        ligand_groups,
    )


# -----------------------------------------------------------------------------
# 7.21. Geometria entre grupo amida e anel aromático
# -----------------------------------------------------------------------------

def calculate_amide_group_ring_geometry(
    group: PiAmideGroup,
    ring: PiRing,
    *,
    config: Optional[PiAnalysisConfig] = None,
    strict: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Calculate complete geometry between an amide group and an aromatic ring.
    """

    if not isinstance(
        group,
        PiAmideGroup,
    ):
        raise TypeError(
            "group must be a PiAmideGroup."
        )

    if not isinstance(
        ring,
        PiRing,
    ):
        raise TypeError(
            "ring must be a PiRing."
        )

    try:
        if (
            group.center is None
            or group.normal is None
        ):
            raise PiGeometryError(
                "Amide group geometry is incomplete."
            )

        ensure_pi_ring_geometry(
            ring,
            config=config,
            strict=True,
        )

        point_geometry = calculate_point_ring_geometry(
            group.center,
            ring,
            direction_vector=group.direction,
            config=config,
            strict=True,
        )

        assert point_geometry is not None
        assert ring.normal is not None

        plane_angle = angle_between_planes(
            group.normal,
            ring.normal,
            strict=True,
        )

        assert plane_angle is not None

        carbonyl_angle: Optional[float] = None

        if group.carbonyl_direction is not None:
            carbonyl_angle = (
                acute_angle_between_vectors(
                    group.carbonyl_direction,
                    ring.normal,
                    strict=False,
                )
            )

        cn_angle: Optional[float] = None

        if group.cn_direction is not None:
            cn_angle = (
                acute_angle_between_vectors(
                    group.cn_direction,
                    ring.normal,
                    strict=False,
                )
            )

        minimum_atomic_distance = (
            calculate_minimum_atomic_distance(
                group.core_atoms,
                ring.atoms,
                skip_invalid=False,
            )
        )

        maximum_atomic_distance = (
            calculate_maximum_atomic_distance(
                group.core_atoms,
                ring.atoms,
                skip_invalid=False,
            )
        )

        closest_pair = find_closest_atom_pair(
            group.core_atoms,
            ring.atoms,
            skip_invalid=False,
        )

        closest_contact: Optional[
            Dict[str, Any]
        ] = None

        if closest_pair is not None:
            atom_1, atom_2, distance = (
                closest_pair
            )

            closest_contact = {
                "amide_atom": (
                    get_atom_identifier(
                        atom_1
                    )
                ),
                "ring_atom": (
                    get_atom_identifier(
                        atom_2
                    )
                ),
                "distance": distance,
            }

        return {
            "centroid_distance": (
                point_geometry.center_distance
            ),
            "plane_height": (
                point_geometry.absolute_plane_distance
            ),
            "signed_plane_height": (
                point_geometry.signed_plane_distance
            ),
            "radial_offset": (
                point_geometry.radial_offset
            ),
            "plane_angle": plane_angle,
            "carbonyl_angle": carbonyl_angle,
            "cn_angle": cn_angle,
            "minimum_atomic_distance": (
                minimum_atomic_distance
            ),
            "maximum_atomic_distance": (
                maximum_atomic_distance
            ),
            "closest_atomic_contact": (
                closest_contact
            ),
            "amide_planarity_rmsd": (
                group.planarity_rmsd
            ),
            "ring_planarity_rmsd": (
                ring.planarity_rmsd
            ),
            "valid": True,
            "warnings": list(
                point_geometry.warnings
            ),
        }

    except (
        PiGeometryError,
        PiAtomAccessError,
        PiCoordinateError,
        TypeError,
        ValueError,
        ArithmeticError,
    ):
        if strict:
            raise

        return None


# -----------------------------------------------------------------------------
# 7.22. Criação de contatos atômicos amide–π
# -----------------------------------------------------------------------------

def create_amide_ring_atomic_contacts(
    group: PiAmideGroup,
    ring: PiRing,
    *,
    maximum_distance: float,
) -> Tuple[PiAtomicContact, ...]:
    """
    Create atomic contacts between an amide core and an aromatic ring.
    """

    maximum_distance_value = (
        _coerce_non_negative_float(
            maximum_distance,
            field_name="maximum_distance",
        )
    )

    contacts: List[PiAtomicContact] = []

    for amide_atom in group.core_atoms:
        amide_coordinate = get_atom_coordinate(
            amide_atom
        )

        if amide_coordinate is None:
            continue

        for ring_atom in ring.atoms:
            ring_coordinate = get_atom_coordinate(
                ring_atom
            )

            if ring_coordinate is None:
                continue

            distance = distance_between_points(
                amide_coordinate,
                ring_coordinate,
            )

            if distance > maximum_distance_value:
                continue

            contacts.append(
                PiAtomicContact(
                    atom_1=create_pi_atom_reference(
                        amide_atom
                    ),
                    atom_2=create_pi_atom_reference(
                        ring_atom
                    ),
                    distance=distance,
                    contact_type=AMIDE_PI,
                    metadata={
                        "amide_group_id": (
                            group.group_id
                        ),
                        "ring_id": ring.ring_id,
                        "amide_atom_role": (
                            "carbonyl_carbon"
                            if amide_atom
                            is group.carbonyl_carbon
                            else (
                                "carbonyl_oxygen"
                                if amide_atom
                                is group.carbonyl_oxygen
                                else "amide_nitrogen"
                            )
                        ),
                    },
                )
            )

    contacts.sort(
        key=lambda contact: (
            contact.distance,
            contact.atom_1.atom_name,
            contact.atom_2.atom_name,
        )
    )

    return tuple(contacts)


# -----------------------------------------------------------------------------
# 7.23. Atualização de PiInteraction
# -----------------------------------------------------------------------------

def attach_amide_ring_geometry_to_interaction(
    interaction: PiInteraction,
    geometry: Mapping[str, Any],
) -> PiInteraction:
    """
    Attach amide–ring geometry to a ``PiInteraction``.
    """

    if not isinstance(
        interaction,
        PiInteraction,
    ):
        raise TypeError(
            "interaction must be a PiInteraction."
        )

    interaction.centroid_distance = (
        _normalize_optional_numeric(
            geometry.get(
                "centroid_distance"
            )
        )
    )

    interaction.plane_height = (
        _normalize_optional_numeric(
            geometry.get(
                "plane_height"
            )
        )
    )

    interaction.radial_offset = (
        _normalize_optional_numeric(
            geometry.get(
                "radial_offset"
            )
        )
    )

    interaction.lateral_offset = (
        interaction.radial_offset
    )

    interaction.plane_angle = (
        _normalize_optional_numeric(
            geometry.get(
                "plane_angle"
            )
        )
    )

    interaction.normal_angle = (
        interaction.plane_angle
    )

    interaction.minimum_atomic_distance = (
        _normalize_optional_numeric(
            geometry.get(
                "minimum_atomic_distance"
            )
        )
    )

    interaction.maximum_atomic_distance = (
        _normalize_optional_numeric(
            geometry.get(
                "maximum_atomic_distance"
            )
        )
    )

    interaction.ring_1_planarity = (
        _normalize_optional_numeric(
            geometry.get(
                "ring_planarity_rmsd"
            )
        )
    )

    interaction.metadata[
        "amide_ring_geometry"
    ] = dict(geometry)

    for warning in geometry.get(
        "warnings",
        (),
    ):
        if warning not in interaction.warnings:
            interaction.warnings.append(
                str(warning)
            )

    return interaction


# -----------------------------------------------------------------------------
# 7.24. Filtragem preliminar por proximidade
# -----------------------------------------------------------------------------

def filter_amide_ring_candidates(
    amide_groups: Iterable[PiAmideGroup],
    rings: Iterable[PiRing],
    *,
    maximum_centroid_distance: float,
    maximum_minimum_atomic_distance: Optional[float] = None,
    config: Optional[PiAnalysisConfig] = None,
) -> List[
    Tuple[
        PiAmideGroup,
        PiRing,
        Dict[str, Any],
    ]
]:
    """
    Return geometrically plausible amide–ring candidate pairs.
    """

    centroid_limit = _coerce_non_negative_float(
        maximum_centroid_distance,
        field_name="maximum_centroid_distance",
    )

    atomic_limit = (
        _coerce_non_negative_float(
            maximum_minimum_atomic_distance,
            field_name=(
                "maximum_minimum_atomic_distance"
            ),
        )
        if maximum_minimum_atomic_distance
        is not None
        else None
    )

    candidates: List[
        Tuple[
            PiAmideGroup,
            PiRing,
            Dict[str, Any],
        ]
    ] = []

    for group in amide_groups:
        if not group.valid:
            continue

        for ring in rings:
            if not ring.valid:
                continue

            geometry = (
                calculate_amide_group_ring_geometry(
                    group,
                    ring,
                    config=config,
                    strict=False,
                )
            )

            if geometry is None:
                continue

            centroid_distance = (
                geometry[
                    "centroid_distance"
                ]
            )

            if (
                centroid_distance
                > centroid_limit
            ):
                continue

            minimum_distance = (
                geometry.get(
                    "minimum_atomic_distance"
                )
            )

            if (
                atomic_limit is not None
                and minimum_distance is not None
                and minimum_distance
                > atomic_limit
            ):
                continue

            candidates.append(
                (
                    group,
                    ring,
                    geometry,
                )
            )

    candidates.sort(
        key=lambda item: (
            item[2][
                "centroid_distance"
            ],
            (
                item[2].get(
                    "minimum_atomic_distance"
                )
                if item[2].get(
                    "minimum_atomic_distance"
                )
                is not None
                else float("inf")
            ),
        )
    )

    return candidates


# -----------------------------------------------------------------------------
# 7.25. Preparação integrada
# -----------------------------------------------------------------------------

def prepare_amide_groups(
    molecular_input: Any,
    *,
    config: Optional[PiAnalysisConfig] = None,
    participant_type: Optional[str] = None,
    include_peptide_amides: bool = True,
) -> List[PiAmideGroup]:
    """
    Run the complete amide-group preparation pipeline.
    """

    groups = detect_amide_groups(
        molecular_input,
        config=config,
        participant_type=participant_type,
        include_peptide_amides=(
            include_peptide_amides
        ),
    )

    groups = deduplicate_amide_groups(
        groups
    )

    analysis_config = (
        config
        if config is not None
        else create_default_pi_config()
    )

    maximum_planarity_rmsd = getattr(
        analysis_config,
        "maximum_amide_planarity_rmsd",
        DEFAULT_MAXIMUM_AMIDE_PLANARITY_RMSD,
    )

    maximum_atom_deviation = getattr(
        analysis_config,
        "maximum_amide_atom_deviation",
        DEFAULT_MAXIMUM_AMIDE_ATOM_DEVIATION,
    )

    return validate_amide_groups(
        groups,
        maximum_planarity_rmsd=(
            maximum_planarity_rmsd
        ),
        maximum_atom_deviation=(
            maximum_atom_deviation
        ),
        remove_invalid=True,
    )


def prepare_pi_analysis_amide_groups(
    normalized_input: PiNormalizedInput,
    *,
    config: Optional[PiAnalysisConfig] = None,
) -> Dict[str, List[PiAmideGroup]]:
    """
    Prepare receptor and ligand amide groups.
    """

    receptor_groups, ligand_groups = (
        detect_pi_analysis_amide_groups(
            normalized_input,
            config=config,
        )
    )

    return {
        "receptor_amide_groups": (
            receptor_groups
        ),
        "ligand_amide_groups": (
            ligand_groups
        ),
        "all_amide_groups": (
            receptor_groups
            + ligand_groups
        ),
    }


# -----------------------------------------------------------------------------
# 7.26. Resumo de grupos amida
# -----------------------------------------------------------------------------

def summarize_amide_groups(
    groups: Iterable[PiAmideGroup],
) -> Dict[str, Any]:
    """
    Generate a serializable summary of detected amide groups.
    """

    group_list = list(groups)

    valid_groups = [
        group
        for group in group_list
        if group.valid
    ]

    planarity_values = [
        group.planarity_rmsd
        for group in group_list
        if group.planarity_rmsd is not None
    ]

    maximum_deviations = [
        group.maximum_plane_deviation
        for group in group_list
        if (
            group.maximum_plane_deviation
            is not None
        )
    ]

    carbonyl_distances = [
        group.carbonyl_distance
        for group in group_list
        if group.carbonyl_distance is not None
    ]

    cn_distances = [
        group.cn_distance
        for group in group_list
        if group.cn_distance is not None
    ]

    def summarize_numeric_values(
        values: Sequence[float],
    ) -> Dict[str, Optional[float]]:
        if not values:
            return {
                "minimum": None,
                "mean": None,
                "maximum": None,
            }

        return {
            "minimum": min(values),
            "mean": (
                sum(values)
                / len(values)
            ),
            "maximum": max(values),
        }

    group_type_distribution = Counter(
        group.group_type
        for group in group_list
    )

    participant_distribution = Counter(
        group.participant_type
        for group in group_list
    )

    residue_distribution = Counter(
        group.residue_name or "UNK"
        for group in group_list
    )

    detection_method_distribution = Counter(
        str(
            group.metadata.get(
                "detection_method",
                "unknown",
            )
        )
        for group in group_list
    )

    planarity_distribution = Counter(
        classify_amide_planarity(
            group.planarity_rmsd
        )
        for group in group_list
    )

    return {
        "total_groups": len(group_list),
        "valid_groups": len(valid_groups),
        "invalid_groups": (
            len(group_list)
            - len(valid_groups)
        ),
        "peptide_amides": sum(
            1
            for group in group_list
            if group.is_peptide
        ),
        "cyclic_amides": sum(
            1
            for group in group_list
            if group.is_cyclic
        ),
        "group_type_distribution": dict(
            group_type_distribution
        ),
        "participant_distribution": dict(
            participant_distribution
        ),
        "residue_distribution": dict(
            residue_distribution
        ),
        "detection_method_distribution": dict(
            detection_method_distribution
        ),
        "planarity_distribution": dict(
            planarity_distribution
        ),
        "planarity_rmsd": (
            summarize_numeric_values(
                planarity_values
            )
        ),
        "maximum_plane_deviation": (
            summarize_numeric_values(
                maximum_deviations
            )
        ),
        "carbonyl_distance": (
            summarize_numeric_values(
                carbonyl_distances
            )
        ),
        "cn_distance": (
            summarize_numeric_values(
                cn_distances
            )
        ),
        "group_ids": [
            group.group_id
            for group in group_list
        ],
    }


# -----------------------------------------------------------------------------
# End of section 7.
# -----------------------------------------------------------------------------



