# =============================================================================
# DockAnalyzer — Molecular Geometry Utilities
# =============================================================================
#
# File:
#     geometry.py
#
# Description:
#     Geometric calculations used by DockAnalyzer for molecular docking
#     analysis, including coordinate validation, vector operations, molecular
#     distances, angular measurements, molecular planes, aromatic-ring
#     geometry, interaction geometry, RMSD and structural alignment.
#
# Notes:
#     This module is designed to operate with both plain Python and NumPy
#     coordinate objects and ChimeraX-like atomic objects through duck typing.
#
#     Basic general-purpose functions such as distance(), centroid(), angle()
#     and normalize() are imported from utils.py rather than reimplemented.
#
# =============================================================================


# -----------------------------------------------------------------------------
# Standard-library imports
# -----------------------------------------------------------------------------

from __future__ import annotations

import math
import warnings

from collections.abc import (
    Iterable,
    Mapping,
    Sequence,
)

from dataclasses import (
    dataclass,
    field,
)

from typing import (
    Any,
    Dict,
    List,
    Literal,
    NamedTuple,
    Optional,
    Tuple,
    TypeAlias,
    Union,
    overload,
)


# -----------------------------------------------------------------------------
# Third-party imports
# -----------------------------------------------------------------------------

import numpy as np

from numpy.typing import (
    ArrayLike,
    NDArray,
)


# -----------------------------------------------------------------------------
# DockAnalyzer imports
# -----------------------------------------------------------------------------

try:
    from .utils import (
        angle,
        centroid,
        distance,
        normalize,
    )

except ImportError:
    # Allows direct execution:
    #
    #     python geometry.py
    #
    # while preserving package-relative imports during normal use.
    from utils import (
        angle,
        centroid,
        distance,
        normalize,
    )


# -----------------------------------------------------------------------------
# Type aliases
# -----------------------------------------------------------------------------

FloatArray: TypeAlias = NDArray[
    np.float64
]

Coordinate: TypeAlias = Union[
    ArrayLike,
    Sequence[float],
    Any,
]

CoordinateCollection: TypeAlias = Union[
    ArrayLike,
    Sequence[Coordinate],
    Any,
]

Vector3D: TypeAlias = FloatArray

Matrix3D: TypeAlias = FloatArray

GeometryMetadata: TypeAlias = Dict[
    str,
    Any,
]

AngleUnit: TypeAlias = Literal[
    "degrees",
    "radians",
]

DistanceMethod: TypeAlias = Literal[
    "euclidean",
    "squared",
]

AlignmentMethod: TypeAlias = Literal[
    "kabsch",
    "none",
]


# -----------------------------------------------------------------------------
# Public interface
# -----------------------------------------------------------------------------

__all__ = [
    # Basic geometry imported from utils.py
    "distance",
    "centroid",
    "angle",
    "normalize",

    # Public types
    "Coordinate",
    "CoordinateCollection",
    "FloatArray",
    "Vector3D",
    "Matrix3D",
    "GeometryMetadata",
    "AngleUnit",
    "DistanceMethod",
    "AlignmentMethod",
]

# -----------------------------------------------------------------------------
# End section 1
# -----------------------------------------------------------------------------





