"""Molecular geometry utilities for DockAnalyzer 0.1.0.

This module has no file-system side effects and does not require ChimeraX at
import time. ChimeraX atoms and collections are supported through duck typing.
NumPy is the only required third-party dependency.
"""

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
#     and normalize() are implemented locally to keep imports side-effect free.
#
# =============================================================================


# -----------------------------------------------------------------------------
# Standard-library imports
# -----------------------------------------------------------------------------

from __future__ import annotations

import math

from collections.abc import (
    Mapping,
    Sequence,
)

from dataclasses import (
    dataclass,
    field,
    fields,
    is_dataclass,
)

from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

try:
    from typing import (
        Literal,
        Protocol,
        TypeAlias,
        runtime_checkable,
    )
except ImportError:  # pragma: no cover - compatibility with Python 3.7
    class _SubscriptableAny:
        """Fallback used only to evaluate type-alias expressions."""

        def __class_getitem__(cls, item):
            return Any

    Literal = _SubscriptableAny
    TypeAlias = Any

    class Protocol:
        """Minimal runtime fallback for a typing-only protocol base."""

    def runtime_checkable(cls):
        return cls


# -----------------------------------------------------------------------------
# Third-party imports
# -----------------------------------------------------------------------------

import numpy as np

try:
    from numpy.typing import (
        ArrayLike,
        NDArray,
    )
except ImportError:  # pragma: no cover - compatibility with older NumPy
    ArrayLike = Any

    class NDArray:
        """Fallback used only to evaluate type-alias expressions."""

        def __class_getitem__(cls, item):
            return np.ndarray


# This foundation module intentionally does not import ``utils``. The four
# basic geometry functions are implemented locally so importing ``geometry``
# cannot initialize logging or create output directories indirectly.

from ._version import __version__


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
# Side-effect-free basic geometry
# -----------------------------------------------------------------------------

def normalize(
    vector: Any,
    *,
    zero_tolerance: float = 1.0e-12,
) -> FloatArray:
    """Return a three-dimensional vector normalized to unit length."""

    vector_array = as_coordinate(
        vector,
        name="vector",
    )
    vector_norm = float(
        np.linalg.norm(vector_array)
    )

    if vector_norm <= float(zero_tolerance):
        raise ValueError(
            "A zero-length vector cannot be normalized."
        )

    return vector_array / vector_norm


def distance(
    point_a: Any,
    point_b: Any,
) -> float:
    """Return the Euclidean distance between two 3D points or atoms."""

    coordinate_a = as_coordinate(
        point_a,
        name="point A",
    )
    coordinate_b = as_coordinate(
        point_b,
        name="point B",
    )
    return float(
        np.linalg.norm(coordinate_b - coordinate_a)
    )


def centroid(
    coordinates: Any,
    *,
    weights: Optional[Any] = None,
) -> FloatArray:
    """Return the arithmetic or weighted centroid of 3D coordinates."""

    coordinate_matrix = as_coordinate_matrix(
        coordinates,
        name="coordinates",
    )

    if weights is None:
        return np.mean(
            coordinate_matrix,
            axis=0,
        )

    try:
        weight_array = np.asarray(
            weights,
            dtype=np.float64,
        )
    except (TypeError, ValueError, OverflowError) as error:
        raise TypeError(
            "Weights must contain numeric values."
        ) from error

    weight_array = np.squeeze(weight_array)
    if weight_array.ndim != 1:
        raise ValueError(
            "Weights must be a one-dimensional sequence."
        )
    if weight_array.size != coordinate_matrix.shape[0]:
        raise ValueError(
            "The number of weights must match the number of coordinates."
        )
    if not np.all(np.isfinite(weight_array)):
        raise ValueError(
            "Weights contain NaN or infinite values."
        )
    if np.isclose(float(np.sum(weight_array)), 0.0):
        raise ValueError(
            "The sum of the centroid weights cannot be zero."
        )

    return np.average(
        coordinate_matrix,
        axis=0,
        weights=weight_array,
    )


def angle(
    point_a: Any,
    vertex: Any,
    point_c: Any,
    *,
    degrees: bool = True,
    zero_tolerance: float = 1.0e-12,
) -> float:
    """Return the angle formed by ``point_a - vertex - point_c``."""

    coordinate_a = as_coordinate(
        point_a,
        name="point A",
    )
    vertex_coordinate = as_coordinate(
        vertex,
        name="vertex",
    )
    coordinate_c = as_coordinate(
        point_c,
        name="point C",
    )
    vector_a = coordinate_a - vertex_coordinate
    vector_c = coordinate_c - vertex_coordinate
    norm_a = float(np.linalg.norm(vector_a))
    norm_c = float(np.linalg.norm(vector_c))

    if norm_a <= zero_tolerance or norm_c <= zero_tolerance:
        raise ValueError(
            "An angle cannot be calculated when an endpoint "
            "coincides with the vertex."
        )

    cosine_value = float(
        np.dot(vector_a, vector_c) / (norm_a * norm_c)
    )
    angle_radians = float(
        np.arccos(np.clip(cosine_value, -1.0, 1.0))
    )
    if degrees:
        return float(np.degrees(angle_radians))
    return angle_radians


# -----------------------------------------------------------------------------
# Public interface
# -----------------------------------------------------------------------------

__all__ = [
    # Side-effect-free basic geometry
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


def _extend_public_names(
    names: Sequence[str],
) -> None:
    """Append unique public names while preserving declaration order."""

    for name in names:
        if name not in __all__:
            __all__.append(name)

# -----------------------------------------------------------------------------
# End of Section 1
# -----------------------------------------------------------------------------


# =============================================================================
# Section 2 — Geometric Constants
# =============================================================================


# -----------------------------------------------------------------------------
# Numerical tolerances
# -----------------------------------------------------------------------------

DEFAULT_TOLERANCE: float = 1.0e-8
"""
Default numerical tolerance used in general geometric comparisons.

This value is intended for operations involving normalized vectors, matrix
comparisons, degeneracy checks and floating-point equality tests.
"""


DEFAULT_DISTANCE_TOLERANCE: float = 1.0e-6
"""
Default tolerance for distance comparisons, expressed in the same coordinate
unit used by the input data.

For molecular structures, coordinates are normally expressed in ångströms.
This constant is intended primarily for numerical comparisons rather than
physical interaction cutoffs.
"""


DEFAULT_ANGLE_TOLERANCE: float = 1.0e-6
"""
Default tolerance for angular comparisons.

The tolerance is expressed in degrees unless a function explicitly operates
in radians. It is intended for floating-point comparisons and not for
classifying molecular interaction geometries.
"""


# -----------------------------------------------------------------------------
# Angular conversion constants
# -----------------------------------------------------------------------------

DEGREES_PER_RADIAN: float = 180.0 / math.pi
"""
Number of degrees in one radian.

Equivalent to approximately 57.295779513 degrees.
"""


RADIANS_PER_DEGREE: float = math.pi / 180.0
"""
Number of radians in one degree.

Equivalent to approximately 0.01745329252 radians.
"""


# -----------------------------------------------------------------------------
# Public module interface
# -----------------------------------------------------------------------------

_SECTION_2_PUBLIC_NAMES = [
    "DEFAULT_TOLERANCE",
    "DEFAULT_DISTANCE_TOLERANCE",
    "DEFAULT_ANGLE_TOLERANCE",
    "DEGREES_PER_RADIAN",
    "RADIANS_PER_DEGREE",
]

_extend_public_names(_SECTION_2_PUBLIC_NAMES)


# =============================================================================
# End of Section 2
# =============================================================================


# =============================================================================
# Section 3 — Coordinate Conversion and Validation
# =============================================================================


# -----------------------------------------------------------------------------
# Coordinate attribute priorities
# -----------------------------------------------------------------------------

_SINGLE_COORDINATE_ATTRIBUTES: Tuple[str, ...] = (
    "scene_coord",
    "coord",
    "coordinate",
    "position",
)

_MULTIPLE_COORDINATE_ATTRIBUTES: Tuple[str, ...] = (
    "scene_coords",
    "coords",
    "coordinates",
    "positions",
)


# -----------------------------------------------------------------------------
# Internal coordinate helpers
# -----------------------------------------------------------------------------

def _is_scalar_coordinate_component(
    value: Any,
) -> bool:
    """
    Return whether a value can represent one coordinate component.

    Parameters
    ----------
    value : Any
        Value to inspect.

    Returns
    -------
    bool
        ``True`` when the value is a finite numeric scalar.
    """

    if isinstance(
        value,
        (
            bool,
            np.bool_,
        ),
    ):
        return False

    try:
        numeric_value = float(
            value
        )

    except (
        TypeError,
        ValueError,
        OverflowError,
    ):
        return False

    return math.isfinite(
        numeric_value
    )


def _extract_xyz_attributes(
    value: Any,
) -> Optional[FloatArray]:
    """
    Extract coordinates from ``x``, ``y`` and ``z`` attributes.

    Parameters
    ----------
    value : Any
        Object potentially exposing Cartesian components.

    Returns
    -------
    numpy.ndarray or None
        Coordinate vector when extraction succeeds.
    """

    if not all(
        hasattr(
            value,
            attribute_name,
        )
        for attribute_name in (
            "x",
            "y",
            "z",
        )
    ):
        return None

    try:
        components = [
            getattr(
                value,
                "x",
            ),
            getattr(
                value,
                "y",
            ),
            getattr(
                value,
                "z",
            ),
        ]

    except Exception:
        return None

    if not all(
        _is_scalar_coordinate_component(
            component
        )
        for component in components
    ):
        return None

    return np.asarray(
        components,
        dtype=np.float64,
    )


def _extract_coordinate_attribute(
    value: Any,
    *,
    scene: bool = True,
) -> Optional[Any]:
    """
    Extract one coordinate-like attribute from an object.

    Parameters
    ----------
    value : Any
        Object to inspect.
    scene : bool, optional
        Whether scene-transformed coordinates should be preferred.

    Returns
    -------
    Any or None
        Extracted coordinate-like value.
    """

    if scene:
        attribute_order = (
            "scene_coord",
            "coord",
            "coordinate",
            "position",
        )

    else:
        attribute_order = (
            "coord",
            "scene_coord",
            "coordinate",
            "position",
        )

    for attribute_name in attribute_order:
        if not hasattr(
            value,
            attribute_name,
        ):
            continue

        try:
            attribute_value = getattr(
                value,
                attribute_name,
            )

        except Exception:
            continue

        if callable(
            attribute_value
        ):
            try:
                attribute_value = (
                    attribute_value()
                )

            except Exception:
                continue

        if attribute_value is not None:
            return attribute_value

    return None


def _extract_coordinate_collection_attribute(
    value: Any,
    *,
    scene: bool = True,
) -> Optional[Any]:
    """
    Extract a coordinate-matrix-like attribute from an object.

    Parameters
    ----------
    value : Any
        Object to inspect.
    scene : bool, optional
        Whether scene-transformed coordinates should be preferred.

    Returns
    -------
    Any or None
        Extracted coordinate collection.
    """

    if scene:
        attribute_order = (
            "scene_coords",
            "coords",
            "coordinates",
            "positions",
        )

    else:
        attribute_order = (
            "coords",
            "scene_coords",
            "coordinates",
            "positions",
        )

    for attribute_name in attribute_order:
        if not hasattr(
            value,
            attribute_name,
        ):
            continue

        try:
            attribute_value = getattr(
                value,
                attribute_name,
            )

        except Exception:
            continue

        if callable(
            attribute_value
        ):
            try:
                attribute_value = (
                    attribute_value()
                )

            except Exception:
                continue

        if attribute_value is not None:
            return attribute_value

    return None


def _coordinate_error_prefix(
    name: Optional[str],
) -> str:
    """
    Build a readable coordinate validation prefix.

    Parameters
    ----------
    name : str or None
        User-facing value name.

    Returns
    -------
    str
        Error-message prefix.
    """

    if name is None:
        return "Coordinate"

    normalized_name = str(
        name
    ).strip()

    if not normalized_name:
        return "Coordinate"

    return normalized_name


# -----------------------------------------------------------------------------
# Coordinate validation
# -----------------------------------------------------------------------------

def validate_coordinate(
    coordinate: Any,
    *,
    name: Optional[str] = None,
    require_finite: bool = True,
    copy: bool = False,
) -> FloatArray:
    """
    Validate a three-dimensional Cartesian coordinate.

    Parameters
    ----------
    coordinate : Any
        Coordinate-like value already convertible to a NumPy array.
    name : str, optional
        User-facing name used in validation messages.
    require_finite : bool, optional
        Whether NaN and infinite values should be rejected.
    copy : bool, optional
        Whether a new array must always be returned.

    Returns
    -------
    numpy.ndarray
        One-dimensional ``float64`` array with shape ``(3,)``.

    Raises
    ------
    TypeError
        If the value cannot be converted to a numeric array.
    ValueError
        If the coordinate does not contain exactly three components or
        contains invalid values.

    Examples
    --------
    >>> validate_coordinate([1, 2, 3])
    array([1., 2., 3.])
    """

    value_name = _coordinate_error_prefix(
        name
    )

    try:
        array = np.asarray(
            coordinate,
            dtype=np.float64,
        )

    except (
        TypeError,
        ValueError,
        OverflowError,
    ) as error:
        raise TypeError(
            f"{value_name} must be convertible "
            "to three numeric components."
        ) from error

    if array.ndim == 0:
        raise ValueError(
            f"{value_name} must contain exactly "
            "three components; a scalar was provided."
        )

    array = np.ravel(
        array
    )

    if array.size != 3:
        raise ValueError(
            f"{value_name} must contain exactly "
            f"three components; received {array.size}."
        )

    if require_finite and not np.all(
        np.isfinite(
            array
        )
    ):
        raise ValueError(
            f"{value_name} contains NaN or "
            "infinite values."
        )

    if copy:
        return np.array(
            array,
            dtype=np.float64,
            copy=True,
        )

    return array.astype(
        np.float64,
        copy=False,
    )


def validate_coordinate_matrix(
    coordinates: Any,
    *,
    name: Optional[str] = None,
    minimum_rows: int = 1,
    allow_empty: bool = False,
    require_finite: bool = True,
    copy: bool = False,
) -> FloatArray:
    """
    Validate a matrix of three-dimensional coordinates.

    Parameters
    ----------
    coordinates : Any
        Coordinate collection convertible to an ``N × 3`` array.
    name : str, optional
        User-facing name used in validation messages.
    minimum_rows : int, optional
        Minimum number of coordinate rows.
    allow_empty : bool, optional
        Whether an empty ``(0, 3)`` array is accepted.
    require_finite : bool, optional
        Whether NaN and infinite values should be rejected.
    copy : bool, optional
        Whether a new array must always be returned.

    Returns
    -------
    numpy.ndarray
        Two-dimensional ``float64`` array with shape ``(N, 3)``.

    Raises
    ------
    TypeError
        If the value is not numeric or ``minimum_rows`` is invalid.
    ValueError
        If the array shape or numerical values are invalid.
    """

    value_name = (
        str(name).strip()
        if name is not None
        else "Coordinate matrix"
    )

    if not value_name:
        value_name = "Coordinate matrix"

    if isinstance(
        minimum_rows,
        (
            bool,
            np.bool_,
        ),
    ) or not isinstance(
        minimum_rows,
        (
            int,
            np.integer,
        ),
    ):
        raise TypeError(
            "minimum_rows must be an integer."
        )

    minimum_rows = int(
        minimum_rows
    )

    if minimum_rows < 0:
        raise ValueError(
            "minimum_rows cannot be negative."
        )

    try:
        array = np.asarray(
            coordinates,
            dtype=np.float64,
        )

    except (
        TypeError,
        ValueError,
        OverflowError,
    ) as error:
        raise TypeError(
            f"{value_name} must be convertible "
            "to a numeric N × 3 array."
        ) from error

    if array.ndim == 1:
        if array.size == 0:
            array = np.empty(
                (
                    0,
                    3,
                ),
                dtype=np.float64,
            )

        elif array.size == 3:
            array = array.reshape(
                1,
                3,
            )

        else:
            raise ValueError(
                f"{value_name} must have shape "
                f"(N, 3); received {array.shape}."
            )

    elif array.ndim != 2:
        raise ValueError(
            f"{value_name} must be two-dimensional; "
            f"received an array with {array.ndim} dimensions."
        )

    if array.shape[1] != 3:
        raise ValueError(
            f"{value_name} must have exactly "
            f"three columns; received shape {array.shape}."
        )

    row_count = int(
        array.shape[0]
    )

    if row_count == 0 and not allow_empty:
        raise ValueError(
            f"{value_name} cannot be empty."
        )

    effective_minimum = (
        0
        if allow_empty
        else minimum_rows
    )

    if row_count < effective_minimum:
        raise ValueError(
            f"{value_name} must contain at least "
            f"{effective_minimum} coordinate rows; "
            f"received {row_count}."
        )

    if require_finite and not np.all(
        np.isfinite(
            array
        )
    ):
        raise ValueError(
            f"{value_name} contains NaN or "
            "infinite values."
        )

    if copy:
        return np.array(
            array,
            dtype=np.float64,
            copy=True,
        )

    return array.astype(
        np.float64,
        copy=False,
    )


# -----------------------------------------------------------------------------
# Coordinate conversion
# -----------------------------------------------------------------------------

def as_coordinate(
    value: Coordinate,
    *,
    scene: bool = True,
    name: Optional[str] = None,
    require_finite: bool = True,
    copy: bool = False,
) -> FloatArray:
    """
    Convert a coordinate-like object to a validated Cartesian vector.

    Parameters
    ----------
    value : Coordinate
        Coordinate-like object. Supported inputs include:

        - lists, tuples and NumPy arrays;
        - objects exposing ``scene_coord``;
        - objects exposing ``coord``;
        - objects exposing ``coordinate`` or ``position``;
        - objects exposing numeric ``x``, ``y`` and ``z`` attributes.
    scene : bool, optional
        Whether ``scene_coord`` should be preferred over ``coord``.
    name : str, optional
        User-facing name used in validation messages.
    require_finite : bool, optional
        Whether NaN and infinite values should be rejected.
    copy : bool, optional
        Whether a new array must always be returned.

    Returns
    -------
    numpy.ndarray
        Coordinate array with shape ``(3,)`` and dtype ``float64``.

    Raises
    ------
    TypeError
        If no coordinate representation can be extracted.
    ValueError
        If the extracted coordinate is invalid.

    Examples
    --------
    >>> as_coordinate((1, 2, 3))
    array([1., 2., 3.])

    >>> as_coordinate(atom)
    array([12.4, 18.1,  7.3])
    """

    if value is None:
        raise TypeError(
            f"{_coordinate_error_prefix(name)} "
            "cannot be None."
        )

    extracted_value = (
        _extract_coordinate_attribute(
            value,
            scene=scene,
        )
    )

    if extracted_value is not None:
        return validate_coordinate(
            extracted_value,
            name=name,
            require_finite=(
                require_finite
            ),
            copy=copy,
        )

    xyz_coordinate = (
        _extract_xyz_attributes(
            value
        )
    )

    if xyz_coordinate is not None:
        return validate_coordinate(
            xyz_coordinate,
            name=name,
            require_finite=(
                require_finite
            ),
            copy=copy,
        )

    try:
        return validate_coordinate(
            value,
            name=name,
            require_finite=(
                require_finite
            ),
            copy=copy,
        )

    except (
        TypeError,
        ValueError,
    ) as error:
        raise TypeError(
            f"{_coordinate_error_prefix(name)} "
            "could not be extracted from "
            f"{type(value).__name__}."
        ) from error


def as_coordinate_matrix(
    values: CoordinateCollection,
    *,
    scene: bool = True,
    name: Optional[str] = None,
    minimum_rows: int = 1,
    allow_empty: bool = False,
    require_finite: bool = True,
    copy: bool = False,
) -> FloatArray:
    """
    Convert coordinate-like values to a validated ``N × 3`` matrix.

    Parameters
    ----------
    values : CoordinateCollection
        Collection of coordinates or object exposing coordinate arrays.
        Supported inputs include:

        - NumPy ``N × 3`` arrays;
        - lists or tuples of coordinates;
        - atom collections exposing ``scene_coords`` or ``coords``;
        - iterables of atom-like objects.
    scene : bool, optional
        Whether scene-transformed coordinates should be preferred.
    name : str, optional
        User-facing name used in validation messages.
    minimum_rows : int, optional
        Minimum accepted number of coordinates.
    allow_empty : bool, optional
        Whether an empty coordinate matrix is accepted.
    require_finite : bool, optional
        Whether NaN and infinite values should be rejected.
    copy : bool, optional
        Whether a new matrix must always be returned.

    Returns
    -------
    numpy.ndarray
        Coordinate matrix with shape ``(N, 3)`` and dtype ``float64``.

    Raises
    ------
    TypeError
        If coordinates cannot be extracted.
    ValueError
        If the resulting coordinate matrix is invalid.
    """

    if values is None:
        raise TypeError(
            f"{name or 'Coordinate collection'} "
            "cannot be None."
        )

    extracted_values = (
        _extract_coordinate_collection_attribute(
            values,
            scene=scene,
        )
    )

    if extracted_values is not None:
        return validate_coordinate_matrix(
            extracted_values,
            name=name,
            minimum_rows=minimum_rows,
            allow_empty=allow_empty,
            require_finite=(
                require_finite
            ),
            copy=copy,
        )

    try:
        return validate_coordinate_matrix(
            values,
            name=name,
            minimum_rows=minimum_rows,
            allow_empty=allow_empty,
            require_finite=(
                require_finite
            ),
            copy=copy,
        )

    except (
        TypeError,
        ValueError,
    ):
        pass

    if isinstance(
        values,
        (
            str,
            bytes,
            bytearray,
        ),
    ):
        raise TypeError(
            f"{name or 'Coordinate collection'} "
            "cannot be created from a string-like object."
        )

    try:
        value_list = list(
            values
        )

    except TypeError as error:
        raise TypeError(
            f"{name or 'Coordinate collection'} "
            "must be an iterable of coordinate-like objects."
        ) from error

    if not value_list:
        return validate_coordinate_matrix(
            np.empty(
                (
                    0,
                    3,
                ),
                dtype=np.float64,
            ),
            name=name,
            minimum_rows=minimum_rows,
            allow_empty=allow_empty,
            require_finite=(
                require_finite
            ),
            copy=copy,
        )

    converted_coordinates: List[
        FloatArray
    ] = []

    for index, item in enumerate(
        value_list
    ):
        try:
            converted_coordinate = (
                as_coordinate(
                    item,
                    scene=scene,
                    name=(
                        f"{name or 'Coordinate collection'}"
                        f"[{index}]"
                    ),
                    require_finite=(
                        require_finite
                    ),
                    copy=False,
                )
            )

        except (
            TypeError,
            ValueError,
        ) as error:
            raise TypeError(
                f"Could not convert item {index} "
                f"of {name or 'coordinate collection'} "
                "to a three-dimensional coordinate."
            ) from error

        converted_coordinates.append(
            converted_coordinate
        )

    matrix = np.vstack(
        converted_coordinates
    ).astype(
        np.float64,
        copy=False,
    )

    return validate_coordinate_matrix(
        matrix,
        name=name,
        minimum_rows=minimum_rows,
        allow_empty=allow_empty,
        require_finite=(
            require_finite
        ),
        copy=copy,
    )


# -----------------------------------------------------------------------------
# Atom-coordinate extraction
# -----------------------------------------------------------------------------

def get_atom_coordinate(
    atom: Any,
    *,
    scene: bool = True,
    name: Optional[str] = None,
    require_finite: bool = True,
    copy: bool = False,
) -> FloatArray:
    """
    Return the Cartesian coordinate of an atom-like object.

    Parameters
    ----------
    atom : Any
        Atom-like object or coordinate-like value.
    scene : bool, optional
        Whether scene-transformed coordinates should be preferred.
    name : str, optional
        User-facing atom name used in validation messages.
    require_finite : bool, optional
        Whether NaN and infinite coordinates should be rejected.
    copy : bool, optional
        Whether a new coordinate array must be returned.

    Returns
    -------
    numpy.ndarray
        Atom coordinate with shape ``(3,)``.

    Raises
    ------
    TypeError
        If the atom is missing or has no usable coordinate.
    ValueError
        If the extracted coordinate is invalid.

    Notes
    -----
    This function intentionally accepts plain coordinates as well as atom-like
    objects. This makes geometry functions testable outside ChimeraX.
    """

    if atom is None:
        raise TypeError(
            f"{name or 'Atom'} cannot be None."
        )

    atom_name = name

    if atom_name is None:
        for attribute_name in (
            "atomspec",
            "name",
        ):
            if not hasattr(
                atom,
                attribute_name,
            ):
                continue

            try:
                attribute_value = getattr(
                    atom,
                    attribute_name,
                )

            except Exception:
                continue

            if attribute_value:
                atom_name = (
                    f"Atom {attribute_value}"
                )
                break

    if atom_name is None:
        atom_name = "Atom coordinate"

    try:
        return as_coordinate(
            atom,
            scene=scene,
            name=atom_name,
            require_finite=(
                require_finite
            ),
            copy=copy,
        )

    except (
        TypeError,
        ValueError,
    ) as error:
        raise TypeError(
            f"Could not obtain coordinates "
            f"from {atom_name}."
        ) from error


def get_coordinates(
    objects: Any,
    *,
    scene: bool = True,
    name: Optional[str] = None,
    minimum_rows: int = 1,
    allow_empty: bool = False,
    require_finite: bool = True,
    copy: bool = False,
    ignore_none: bool = False,
) -> FloatArray:
    """
    Return coordinates from atoms, atom collections or coordinate values.

    Parameters
    ----------
    objects : Any
        Coordinate matrix, atom collection, iterable of atoms, or iterable of
        coordinate-like values.
    scene : bool, optional
        Whether scene-transformed coordinates should be preferred.
    name : str, optional
        User-facing collection name used in validation messages.
    minimum_rows : int, optional
        Minimum accepted number of coordinate rows.
    allow_empty : bool, optional
        Whether an empty result is accepted.
    require_finite : bool, optional
        Whether NaN and infinite values should be rejected.
    copy : bool, optional
        Whether a new matrix must always be returned.
    ignore_none : bool, optional
        Whether ``None`` items in iterables should be skipped.

    Returns
    -------
    numpy.ndarray
        Coordinate matrix with shape ``(N, 3)``.

    Raises
    ------
    TypeError
        If coordinates cannot be extracted.
    ValueError
        If the resulting coordinate matrix is invalid.

    Examples
    --------
    >>> get_coordinates([[0, 0, 0], [1, 1, 1]])
    array([[0., 0., 0.],
           [1., 1., 1.]])

    >>> get_coordinates(atoms)
    array([[...], [...], ...])
    """

    collection_name = (
        str(name).strip()
        if name is not None
        else "Coordinates"
    )

    if not collection_name:
        collection_name = "Coordinates"

    if objects is None:
        if allow_empty:
            return validate_coordinate_matrix(
                np.empty(
                    (
                        0,
                        3,
                    ),
                    dtype=np.float64,
                ),
                name=collection_name,
                minimum_rows=minimum_rows,
                allow_empty=True,
                require_finite=(
                    require_finite
                ),
                copy=copy,
            )

        raise TypeError(
            f"{collection_name} cannot be None."
        )

    extracted_values = (
        _extract_coordinate_collection_attribute(
            objects,
            scene=scene,
        )
    )

    if extracted_values is not None:
        return validate_coordinate_matrix(
            extracted_values,
            name=collection_name,
            minimum_rows=minimum_rows,
            allow_empty=allow_empty,
            require_finite=(
                require_finite
            ),
            copy=copy,
        )

    try:
        return as_coordinate_matrix(
            objects,
            scene=scene,
            name=collection_name,
            minimum_rows=minimum_rows,
            allow_empty=allow_empty,
            require_finite=(
                require_finite
            ),
            copy=copy,
        )

    except (
        TypeError,
        ValueError,
    ) as direct_error:
        if isinstance(
            objects,
            (
                str,
                bytes,
                bytearray,
            ),
        ):
            raise TypeError(
                f"{collection_name} cannot be "
                "extracted from a string-like object."
            ) from direct_error

    try:
        object_list = list(
            objects
        )

    except TypeError as error:
        raise TypeError(
            f"{collection_name} must be a "
            "coordinate collection or iterable "
            "of atom-like objects."
        ) from error

    coordinates: List[
        FloatArray
    ] = []

    for index, object_value in enumerate(
        object_list
    ):
        if object_value is None:
            if ignore_none:
                continue

            raise TypeError(
                f"{collection_name}[{index}] "
                "is None."
            )

        try:
            coordinate = get_atom_coordinate(
                object_value,
                scene=scene,
                name=(
                    f"{collection_name}[{index}]"
                ),
                require_finite=(
                    require_finite
                ),
                copy=False,
            )

        except (
            TypeError,
            ValueError,
        ) as error:
            raise TypeError(
                f"Could not extract a coordinate "
                f"from {collection_name}[{index}]."
            ) from error

        coordinates.append(
            coordinate
        )

    if coordinates:
        coordinate_matrix = np.vstack(
            coordinates
        )

    else:
        coordinate_matrix = np.empty(
            (
                0,
                3,
            ),
            dtype=np.float64,
        )

    return validate_coordinate_matrix(
        coordinate_matrix,
        name=collection_name,
        minimum_rows=minimum_rows,
        allow_empty=allow_empty,
        require_finite=(
            require_finite
        ),
        copy=copy,
    )


# -----------------------------------------------------------------------------
# Public module interface
# -----------------------------------------------------------------------------

_SECTION_3_PUBLIC_NAMES = [
    "as_coordinate",
    "as_coordinate_matrix",
    "validate_coordinate",
    "validate_coordinate_matrix",
    "get_atom_coordinate",
    "get_coordinates",
]

_extend_public_names(_SECTION_3_PUBLIC_NAMES)


# =============================================================================
# End of Section 3
# =============================================================================




# =============================================================================
# Section 4 — Vector Operations
# =============================================================================


# -----------------------------------------------------------------------------
# Vector construction
# -----------------------------------------------------------------------------

def vector_between(
    start: Coordinate,
    end: Coordinate,
    *,
    scene: bool = True,
    normalize_result: bool = False,
    tolerance: float = DEFAULT_TOLERANCE,
    copy: bool = False,
) -> Vector3D:
    """
    Return the vector directed from one point to another.

    The vector is calculated as:

    ``end - start``

    Parameters
    ----------
    start : Coordinate
        Starting point or coordinate-like object.
    end : Coordinate
        Ending point or coordinate-like object.
    scene : bool, optional
        Whether scene-transformed coordinates should be preferred when
        coordinate-like objects expose both ``scene_coord`` and ``coord``.
    normalize_result : bool, optional
        Whether the resulting vector should be normalized.
    tolerance : float, optional
        Minimum accepted vector norm when ``normalize_result=True``.
    copy : bool, optional
        Whether a new array must always be returned.

    Returns
    -------
    numpy.ndarray
        Vector with shape ``(3,)`` and dtype ``float64``.

    Raises
    ------
    TypeError
        If either input cannot be converted to a coordinate.
    ValueError
        If normalization is requested for a near-zero vector.

    Examples
    --------
    >>> vector_between([0, 0, 0], [1, 2, 3])
    array([1., 2., 3.])

    >>> vector_between(
    ...     [0, 0, 0],
    ...     [2, 0, 0],
    ...     normalize_result=True,
    ... )
    array([1., 0., 0.])
    """

    start_coordinate = as_coordinate(
        start,
        scene=scene,
        name="Start coordinate",
        copy=False,
    )

    end_coordinate = as_coordinate(
        end,
        scene=scene,
        name="End coordinate",
        copy=False,
    )

    result = (
        end_coordinate
        - start_coordinate
    ).astype(
        np.float64,
        copy=False,
    )

    if normalize_result:
        result = unit_vector(
            result,
            tolerance=tolerance,
            copy=False,
        )

    if copy:
        return np.array(
            result,
            dtype=np.float64,
            copy=True,
        )

    return result


# -----------------------------------------------------------------------------
# Vector magnitude and normalization
# -----------------------------------------------------------------------------

def vector_norm(
    vector: Coordinate,
    *,
    squared: bool = False,
    scene: bool = True,
) -> float:
    """
    Return the Euclidean norm of a three-dimensional vector.

    Parameters
    ----------
    vector : Coordinate
        Vector or coordinate-like object.
    squared : bool, optional
        Whether the squared norm should be returned without calculating the
        square root.
    scene : bool, optional
        Whether scene-transformed coordinates should be preferred.

    Returns
    -------
    float
        Vector magnitude or squared magnitude.

    Examples
    --------
    >>> vector_norm([3, 4, 0])
    5.0

    >>> vector_norm([3, 4, 0], squared=True)
    25.0
    """

    vector_array = as_coordinate(
        vector,
        scene=scene,
        name="Vector",
        copy=False,
    )

    squared_norm = float(
        np.dot(
            vector_array,
            vector_array,
        )
    )

    if squared:
        return squared_norm

    return float(
        math.sqrt(
            max(
                squared_norm,
                0.0,
            )
        )
    )


def unit_vector(
    vector: Coordinate,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
    scene: bool = True,
    copy: bool = False,
) -> Vector3D:
    """
    Return the normalized form of a three-dimensional vector.

    Parameters
    ----------
    vector : Coordinate
        Vector or coordinate-like object.
    tolerance : float, optional
        Minimum norm required for normalization.
    scene : bool, optional
        Whether scene-transformed coordinates should be preferred.
    copy : bool, optional
        Whether a new array must always be returned.

    Returns
    -------
    numpy.ndarray
        Unit vector with shape ``(3,)``.

    Raises
    ------
    TypeError
        If ``tolerance`` is not numeric.
    ValueError
        If ``tolerance`` is negative or the vector norm is too small.

    Notes
    -----
    This function provides the specialized public interface for vector
    normalization in ``geometry.py``. The imported ``normalize()`` function
    remains available for compatibility with ``utils.py``.
    """

    if isinstance(
        tolerance,
        (
            bool,
            np.bool_,
        ),
    ):
        raise TypeError(
            "tolerance must be a numeric value."
        )

    try:
        numeric_tolerance = float(
            tolerance
        )

    except (
        TypeError,
        ValueError,
        OverflowError,
    ) as error:
        raise TypeError(
            "tolerance must be a numeric value."
        ) from error

    if not math.isfinite(
        numeric_tolerance
    ):
        raise ValueError(
            "tolerance must be finite."
        )

    if numeric_tolerance < 0.0:
        raise ValueError(
            "tolerance cannot be negative."
        )

    vector_array = as_coordinate(
        vector,
        scene=scene,
        name="Vector",
        copy=False,
    )

    magnitude = vector_norm(
        vector_array,
        squared=False,
        scene=False,
    )

    if magnitude <= numeric_tolerance:
        raise ValueError(
            "Cannot normalize a zero or near-zero "
            f"vector with norm {magnitude:.6g}."
        )

    normalized_vector = (
        vector_array
        / magnitude
    ).astype(
        np.float64,
        copy=False,
    )

    if copy:
        return np.array(
            normalized_vector,
            dtype=np.float64,
            copy=True,
        )

    return normalized_vector


# -----------------------------------------------------------------------------
# Vector products
# -----------------------------------------------------------------------------

def dot_product(
    vector_1: Coordinate,
    vector_2: Coordinate,
    *,
    scene: bool = True,
) -> float:
    """
    Return the scalar dot product of two three-dimensional vectors.

    Parameters
    ----------
    vector_1 : Coordinate
        First vector.
    vector_2 : Coordinate
        Second vector.
    scene : bool, optional
        Whether scene-transformed coordinates should be preferred.

    Returns
    -------
    float
        Scalar dot product.

    Examples
    --------
    >>> dot_product([1, 0, 0], [0, 1, 0])
    0.0

    >>> dot_product([1, 2, 3], [4, 5, 6])
    32.0
    """

    first_vector = as_coordinate(
        vector_1,
        scene=scene,
        name="First vector",
        copy=False,
    )

    second_vector = as_coordinate(
        vector_2,
        scene=scene,
        name="Second vector",
        copy=False,
    )

    return float(
        np.dot(
            first_vector,
            second_vector,
        )
    )


def cross_product(
    vector_1: Coordinate,
    vector_2: Coordinate,
    *,
    scene: bool = True,
    normalize_result: bool = False,
    tolerance: float = DEFAULT_TOLERANCE,
    copy: bool = False,
) -> Vector3D:
    """
    Return the cross product of two three-dimensional vectors.

    Parameters
    ----------
    vector_1 : Coordinate
        First vector.
    vector_2 : Coordinate
        Second vector.
    scene : bool, optional
        Whether scene-transformed coordinates should be preferred.
    normalize_result : bool, optional
        Whether the resulting vector should be normalized.
    tolerance : float, optional
        Minimum norm accepted when normalization is requested.
    copy : bool, optional
        Whether a new array must always be returned.

    Returns
    -------
    numpy.ndarray
        Cross-product vector with shape ``(3,)``.

    Raises
    ------
    ValueError
        If normalization is requested for parallel or near-parallel vectors.

    Examples
    --------
    >>> cross_product([1, 0, 0], [0, 1, 0])
    array([0., 0., 1.])
    """

    first_vector = as_coordinate(
        vector_1,
        scene=scene,
        name="First vector",
        copy=False,
    )

    second_vector = as_coordinate(
        vector_2,
        scene=scene,
        name="Second vector",
        copy=False,
    )

    result = np.cross(
        first_vector,
        second_vector,
    ).astype(
        np.float64,
        copy=False,
    )

    if normalize_result:
        result = unit_vector(
            result,
            tolerance=tolerance,
            scene=False,
            copy=False,
        )

    if copy:
        return np.array(
            result,
            dtype=np.float64,
            copy=True,
        )

    return result


# -----------------------------------------------------------------------------
# Vector projection and rejection
# -----------------------------------------------------------------------------

def project_vector(
    vector: Coordinate,
    onto: Coordinate,
    *,
    scene: bool = True,
    tolerance: float = DEFAULT_TOLERANCE,
    copy: bool = False,
) -> Vector3D:
    """
    Project one vector onto another vector.

    Parameters
    ----------
    vector : Coordinate
        Vector being projected.
    onto : Coordinate
        Vector defining the projection direction.
    scene : bool, optional
        Whether scene-transformed coordinates should be preferred.
    tolerance : float, optional
        Minimum squared norm accepted for the projection direction.
    copy : bool, optional
        Whether a new array must always be returned.

    Returns
    -------
    numpy.ndarray
        Component of ``vector`` parallel to ``onto``.

    Raises
    ------
    TypeError
        If ``tolerance`` is not numeric.
    ValueError
        If ``onto`` is zero or near zero.

    Notes
    -----
    The projection is calculated as:

    ``dot(vector, onto) / dot(onto, onto) * onto``

    Examples
    --------
    >>> project_vector([2, 2, 0], [1, 0, 0])
    array([2., 0., 0.])
    """

    vector_array = as_coordinate(
        vector,
        scene=scene,
        name="Projected vector",
        copy=False,
    )

    projection_axis = as_coordinate(
        onto,
        scene=scene,
        name="Projection vector",
        copy=False,
    )

    try:
        numeric_tolerance = float(
            tolerance
        )

    except (
        TypeError,
        ValueError,
        OverflowError,
    ) as error:
        raise TypeError(
            "tolerance must be a numeric value."
        ) from error

    if not math.isfinite(
        numeric_tolerance
    ):
        raise ValueError(
            "tolerance must be finite."
        )

    if numeric_tolerance < 0.0:
        raise ValueError(
            "tolerance cannot be negative."
        )

    axis_squared_norm = dot_product(
        projection_axis,
        projection_axis,
        scene=False,
    )

    if axis_squared_norm <= (
        numeric_tolerance ** 2
    ):
        raise ValueError(
            "Cannot project onto a zero or "
            "near-zero vector."
        )

    scalar_component = (
        dot_product(
            vector_array,
            projection_axis,
            scene=False,
        )
        / axis_squared_norm
    )

    projection = (
        scalar_component
        * projection_axis
    ).astype(
        np.float64,
        copy=False,
    )

    if copy:
        return np.array(
            projection,
            dtype=np.float64,
            copy=True,
        )

    return projection


def reject_vector(
    vector: Coordinate,
    from_vector: Coordinate,
    *,
    scene: bool = True,
    tolerance: float = DEFAULT_TOLERANCE,
    copy: bool = False,
) -> Vector3D:
    """
    Return the component of a vector perpendicular to another vector.

    Vector rejection is calculated as:

    ``vector - project_vector(vector, from_vector)``

    Parameters
    ----------
    vector : Coordinate
        Vector being decomposed.
    from_vector : Coordinate
        Vector defining the parallel direction to remove.
    scene : bool, optional
        Whether scene-transformed coordinates should be preferred.
    tolerance : float, optional
        Minimum norm accepted for ``from_vector``.
    copy : bool, optional
        Whether a new array must always be returned.

    Returns
    -------
    numpy.ndarray
        Component of ``vector`` perpendicular to ``from_vector``.

    Raises
    ------
    ValueError
        If ``from_vector`` is zero or near zero.

    Examples
    --------
    >>> reject_vector([2, 2, 0], [1, 0, 0])
    array([0., 2., 0.])
    """

    vector_array = as_coordinate(
        vector,
        scene=scene,
        name="Rejected vector",
        copy=False,
    )

    parallel_component = project_vector(
        vector_array,
        from_vector,
        scene=scene,
        tolerance=tolerance,
        copy=False,
    )

    rejection = (
        vector_array
        - parallel_component
    ).astype(
        np.float64,
        copy=False,
    )

    if copy:
        return np.array(
            rejection,
            dtype=np.float64,
            copy=True,
        )

    return rejection


# -----------------------------------------------------------------------------
# Point projection onto a line
# -----------------------------------------------------------------------------

def project_point_on_line(
    point: Coordinate,
    line_start: Coordinate,
    line_end: Optional[Coordinate] = None,
    *,
    direction: Optional[Coordinate] = None,
    scene: bool = True,
    tolerance: float = DEFAULT_TOLERANCE,
    clamp_to_segment: bool = False,
    return_parameter: bool = False,
    copy: bool = False,
) -> Union[
    Vector3D,
    Tuple[
        Vector3D,
        float,
    ],
]:
    """
    Project a point orthogonally onto a three-dimensional line.

    The line may be defined using either:

    - ``line_start`` and ``line_end``; or
    - ``line_start`` and ``direction``.

    Parameters
    ----------
    point : Coordinate
        Point to project.
    line_start : Coordinate
        Point located on the line.
    line_end : Coordinate, optional
        Second point defining the line.
    direction : Coordinate, optional
        Explicit line-direction vector. Exactly one of ``line_end`` and
        ``direction`` must be provided.
    scene : bool, optional
        Whether scene-transformed coordinates should be preferred.
    tolerance : float, optional
        Minimum accepted norm for the line direction.
    clamp_to_segment : bool, optional
        Whether the projection parameter should be restricted to the interval
        from zero to one. This converts projection onto an infinite line into
        projection onto the finite segment from ``line_start`` to
        ``line_end``.
    return_parameter : bool, optional
        Whether the scalar line parameter should also be returned.
    copy : bool, optional
        Whether a new coordinate array must always be returned.

    Returns
    -------
    numpy.ndarray
        Projected coordinate.

    tuple
        ``(projected_coordinate, parameter)`` when
        ``return_parameter=True``.

    Raises
    ------
    ValueError
        If neither or both line definitions are supplied, if the direction is
        near zero, or if segment clamping is requested without ``line_end``.

    Notes
    -----
    The projected point is calculated as:

    ``line_start + t * line_direction``

    where:

    ``t = dot(point - line_start, line_direction) / |line_direction|²``

    For a line defined by two points:

    - ``t = 0`` corresponds to ``line_start``;
    - ``t = 1`` corresponds to ``line_end``;
    - values outside this interval lie beyond the finite segment.

    Examples
    --------
    >>> project_point_on_line(
    ...     [1, 2, 0],
    ...     [0, 0, 0],
    ...     [3, 0, 0],
    ... )
    array([1., 0., 0.])

    >>> project_point_on_line(
    ...     [5, 2, 0],
    ...     [0, 0, 0],
    ...     [3, 0, 0],
    ...     clamp_to_segment=True,
    ... )
    array([3., 0., 0.])
    """

    has_line_end = (
        line_end is not None
    )

    has_direction = (
        direction is not None
    )

    if has_line_end == has_direction:
        raise ValueError(
            "Provide exactly one of line_end "
            "or direction."
        )

    if (
        clamp_to_segment
        and not has_line_end
    ):
        raise ValueError(
            "clamp_to_segment=True requires "
            "line_end to define a finite segment."
        )

    point_coordinate = as_coordinate(
        point,
        scene=scene,
        name="Projected point",
        copy=False,
    )

    start_coordinate = as_coordinate(
        line_start,
        scene=scene,
        name="Line start",
        copy=False,
    )

    if has_line_end:
        end_coordinate = as_coordinate(
            line_end,
            scene=scene,
            name="Line end",
            copy=False,
        )

        line_direction = vector_between(
            start_coordinate,
            end_coordinate,
            scene=False,
            normalize_result=False,
            copy=False,
        )

    else:
        line_direction = as_coordinate(
            direction,
            scene=scene,
            name="Line direction",
            copy=False,
        )

    try:
        numeric_tolerance = float(
            tolerance
        )

    except (
        TypeError,
        ValueError,
        OverflowError,
    ) as error:
        raise TypeError(
            "tolerance must be a numeric value."
        ) from error

    if not math.isfinite(
        numeric_tolerance
    ):
        raise ValueError(
            "tolerance must be finite."
        )

    if numeric_tolerance < 0.0:
        raise ValueError(
            "tolerance cannot be negative."
        )

    direction_squared_norm = (
        vector_norm(
            line_direction,
            squared=True,
            scene=False,
        )
    )

    if direction_squared_norm <= (
        numeric_tolerance ** 2
    ):
        raise ValueError(
            "Cannot project onto a line with "
            "a zero or near-zero direction."
        )

    point_offset = vector_between(
        start_coordinate,
        point_coordinate,
        scene=False,
        normalize_result=False,
        copy=False,
    )

    parameter = (
        dot_product(
            point_offset,
            line_direction,
            scene=False,
        )
        / direction_squared_norm
    )

    if clamp_to_segment:
        parameter = float(
            np.clip(
                parameter,
                0.0,
                1.0,
            )
        )

    projected_coordinate = (
        start_coordinate
        + parameter
        * line_direction
    ).astype(
        np.float64,
        copy=False,
    )

    if copy:
        projected_coordinate = np.array(
            projected_coordinate,
            dtype=np.float64,
            copy=True,
        )

    if return_parameter:
        return (
            projected_coordinate,
            float(parameter),
        )

    return projected_coordinate


# -----------------------------------------------------------------------------
# Public module interface
# -----------------------------------------------------------------------------

_SECTION_4_PUBLIC_NAMES = [
    "vector_between",
    "vector_norm",
    "unit_vector",
    "dot_product",
    "cross_product",
    "project_vector",
    "reject_vector",
    "project_point_on_line",
]

_extend_public_names(_SECTION_4_PUBLIC_NAMES)


# =============================================================================
# End of Section 4
# =============================================================================


# =============================================================================
# Section 5 — Distance Operations
# =============================================================================


# -----------------------------------------------------------------------------
# Point and atom distances
# -----------------------------------------------------------------------------

def squared_distance(
    point_1: Coordinate,
    point_2: Coordinate,
    *,
    scene: bool = True,
) -> float:
    """
    Return the squared Euclidean distance between two points.

    Parameters
    ----------
    point_1 : Coordinate
        First point or coordinate-like object.
    point_2 : Coordinate
        Second point or coordinate-like object.
    scene : bool, optional
        Whether scene-transformed coordinates should be preferred when the
        inputs expose both ``scene_coord`` and ``coord``.

    Returns
    -------
    float
        Squared Euclidean distance between the points.

    Notes
    -----
    This function avoids calculating a square root and is therefore useful
    for distance comparisons and cutoff screening.

    The squared distance is calculated as:

    ``dot(point_2 - point_1, point_2 - point_1)``

    Examples
    --------
    >>> squared_distance([0, 0, 0], [3, 4, 0])
    25.0
    """

    first_coordinate = as_coordinate(
        point_1,
        scene=scene,
        name="First point",
        copy=False,
    )

    second_coordinate = as_coordinate(
        point_2,
        scene=scene,
        name="Second point",
        copy=False,
    )

    displacement = (
        second_coordinate
        - first_coordinate
    )

    result = float(
        np.dot(
            displacement,
            displacement,
        )
    )

    # Floating-point roundoff should not produce a physically meaningful
    # negative squared distance. The maximum protects against values such as
    # -1e-16 arising from future alternative implementations.
    return max(
        result,
        0.0,
    )


def atom_distance(
    atom_1: Any,
    atom_2: Any,
    *,
    scene: bool = True,
    squared: bool = False,
) -> float:
    """
    Return the Euclidean distance between two atom-like objects.

    Parameters
    ----------
    atom_1 : Any
        First atom-like object or coordinate-like value.
    atom_2 : Any
        Second atom-like object or coordinate-like value.
    scene : bool, optional
        Whether scene-transformed coordinates should be preferred.
    squared : bool, optional
        Whether the squared distance should be returned.

    Returns
    -------
    float
        Distance between the atoms. For molecular structures, the unit is
        normally ångströms.

    Raises
    ------
    TypeError
        If either atom has no usable coordinate.
    ValueError
        If an extracted coordinate is invalid.

    Notes
    -----
    Plain coordinate arrays are accepted to keep this function testable
    outside ChimeraX.

    Examples
    --------
    >>> atom_distance([0, 0, 0], [0, 0, 2])
    2.0
    """

    first_coordinate = get_atom_coordinate(
        atom_1,
        scene=scene,
        name="First atom",
        copy=False,
    )

    second_coordinate = get_atom_coordinate(
        atom_2,
        scene=scene,
        name="Second atom",
        copy=False,
    )

    distance_squared = squared_distance(
        first_coordinate,
        second_coordinate,
        scene=False,
    )

    if squared:
        return distance_squared

    return float(
        math.sqrt(
            distance_squared
        )
    )


# -----------------------------------------------------------------------------
# Pairwise distance matrices
# -----------------------------------------------------------------------------

def distance_matrix(
    points_1: CoordinateCollection,
    points_2: Optional[
        CoordinateCollection
    ] = None,
    *,
    scene: bool = True,
    squared: bool = False,
    minimum_rows: int = 1,
    allow_empty: bool = False,
    copy: bool = False,
) -> FloatArray:
    """
    Calculate pairwise distances between two coordinate collections.

    Parameters
    ----------
    points_1 : CoordinateCollection
        First coordinate collection containing ``N`` points.
    points_2 : CoordinateCollection, optional
        Second coordinate collection containing ``M`` points. When omitted,
        distances are calculated between all pairs in ``points_1``.
    scene : bool, optional
        Whether scene-transformed coordinates should be preferred.
    squared : bool, optional
        Whether squared distances should be returned.
    minimum_rows : int, optional
        Minimum number of coordinates required in each collection.
    allow_empty : bool, optional
        Whether empty coordinate collections should be accepted.
    copy : bool, optional
        Whether a new result array must always be returned.

    Returns
    -------
    numpy.ndarray
        Distance matrix with shape ``(N, M)``. When ``points_2`` is omitted,
        the result has shape ``(N, N)``.

    Raises
    ------
    TypeError
        If either collection cannot be converted to coordinates.
    ValueError
        If either coordinate matrix is invalid.

    Notes
    -----
    Squared distances use the Gram-matrix identity after translating both
    collections to a shared origin. This avoids an intermediate ``N × M × 3``
    displacement array while retaining numerical stability for molecular data.

    Examples
    --------
    >>> distance_matrix(
    ...     [[0, 0, 0], [1, 0, 0]],
    ...     [[0, 1, 0], [2, 0, 0]],
    ... )
    array([[1.        , 2.        ],
           [1.41421356, 1.        ]])
    """

    first_coordinates = get_coordinates(
        points_1,
        scene=scene,
        name="First coordinate collection",
        minimum_rows=minimum_rows,
        allow_empty=allow_empty,
        copy=False,
    )

    if points_2 is None:
        second_coordinates = (
            first_coordinates
        )

    else:
        second_coordinates = get_coordinates(
            points_2,
            scene=scene,
            name="Second coordinate collection",
            minimum_rows=minimum_rows,
            allow_empty=allow_empty,
            copy=False,
        )

    if first_coordinates.size and second_coordinates.size:
        origin = (
            first_coordinates[0]
            + second_coordinates[0]
        ) / 2.0
        first_centered = first_coordinates - origin
        second_centered = second_coordinates - origin
    else:
        first_centered = first_coordinates
        second_centered = second_coordinates

    first_squared_norms = np.einsum(
        "ij,ij->i",
        first_centered,
        first_centered,
        optimize=True,
    )[:, np.newaxis]

    second_squared_norms = np.einsum(
        "ij,ij->i",
        second_centered,
        second_centered,
        optimize=True,
    )[np.newaxis, :]

    squared_distances = (
        first_squared_norms
        + second_squared_norms
        - 2.0
        * (
            first_centered
            @ second_centered.T
        )
    ).astype(
        np.float64,
        copy=False,
    )

    # Remove negligible negative roundoff before square-root conversion.
    np.maximum(
        squared_distances,
        0.0,
        out=squared_distances,
    )

    if squared:
        result = squared_distances

    else:
        result = np.sqrt(
            squared_distances
        )

    if copy:
        return np.array(
            result,
            dtype=np.float64,
            copy=True,
        )

    return result.astype(
        np.float64,
        copy=False,
    )


# -----------------------------------------------------------------------------
# Minimum distances and closest pairs
# -----------------------------------------------------------------------------

def closest_point_pair(
    points_1: CoordinateCollection,
    points_2: CoordinateCollection,
    *,
    scene: bool = True,
    return_indices: bool = False,
    return_distance: bool = False,
    squared: bool = False,
    copy: bool = False,
) -> Union[
    Tuple[
        Vector3D,
        Vector3D,
    ],
    Tuple[
        Vector3D,
        Vector3D,
        float,
    ],
    Tuple[
        Vector3D,
        Vector3D,
        Tuple[int, int],
    ],
    Tuple[
        Vector3D,
        Vector3D,
        float,
        Tuple[int, int],
    ],
]:
    """
    Return the closest pair of points from two coordinate collections.

    Parameters
    ----------
    points_1 : CoordinateCollection
        First non-empty coordinate collection.
    points_2 : CoordinateCollection
        Second non-empty coordinate collection.
    scene : bool, optional
        Whether scene-transformed coordinates should be preferred.
    return_indices : bool, optional
        Whether the source indices of the closest points should be returned.
    return_distance : bool, optional
        Whether the minimum distance should also be returned.
    squared : bool, optional
        Whether the returned distance should be squared. This option only
        affects the value returned when ``return_distance=True``.
    copy : bool, optional
        Whether the returned coordinate arrays must be copied.

    Returns
    -------
    tuple
        By default, returns:

        ``(closest_point_1, closest_point_2)``

        With ``return_distance=True``:

        ``(closest_point_1, closest_point_2, distance)``

        With ``return_indices=True``:

        ``(closest_point_1, closest_point_2, (index_1, index_2))``

        With both options enabled:

        ``(closest_point_1, closest_point_2, distance, (index_1, index_2))``

    Raises
    ------
    TypeError
        If either collection cannot be converted to coordinates.
    ValueError
        If either collection is empty or invalid.

    Notes
    -----
    When multiple point pairs have the same minimum distance, NumPy's
    row-major ordering selects the first pair.

    Examples
    --------
    >>> closest_point_pair(
    ...     [[0, 0, 0], [5, 0, 0]],
    ...     [[2, 0, 0], [8, 0, 0]],
    ...     return_distance=True,
    ...     return_indices=True,
    ... )
    (array([0., 0., 0.]),
     array([2., 0., 0.]),
     2.0,
     (0, 0))
    """

    first_coordinates = get_coordinates(
        points_1,
        scene=scene,
        name="First coordinate collection",
        minimum_rows=1,
        allow_empty=False,
        copy=False,
    )

    second_coordinates = get_coordinates(
        points_2,
        scene=scene,
        name="Second coordinate collection",
        minimum_rows=1,
        allow_empty=False,
        copy=False,
    )

    squared_distances = distance_matrix(
        first_coordinates,
        second_coordinates,
        scene=False,
        squared=True,
        minimum_rows=1,
        allow_empty=False,
        copy=False,
    )

    flat_index = int(
        np.argmin(
            squared_distances
        )
    )

    first_index, second_index = (
        np.unravel_index(
            flat_index,
            squared_distances.shape,
        )
    )

    first_index = int(
        first_index
    )

    second_index = int(
        second_index
    )

    first_point = first_coordinates[
        first_index
    ]

    second_point = second_coordinates[
        second_index
    ]

    if copy:
        first_point = np.array(
            first_point,
            dtype=np.float64,
            copy=True,
        )

        second_point = np.array(
            second_point,
            dtype=np.float64,
            copy=True,
        )

    output: List[Any] = [
        first_point,
        second_point,
    ]

    if return_distance:
        minimum_squared_distance = float(
            squared_distances[
                first_index,
                second_index,
            ]
        )

        if squared:
            minimum_value = (
                minimum_squared_distance
            )

        else:
            minimum_value = float(
                math.sqrt(
                    minimum_squared_distance
                )
            )

        output.append(
            minimum_value
        )

    if return_indices:
        output.append(
            (
                first_index,
                second_index,
            )
        )

    return tuple(
        output
    )


def minimum_distance(
    points_1: CoordinateCollection,
    points_2: CoordinateCollection,
    *,
    scene: bool = True,
    squared: bool = False,
    return_indices: bool = False,
    return_points: bool = False,
    copy: bool = False,
) -> Union[
    float,
    Tuple[
        float,
        Tuple[int, int],
    ],
    Tuple[
        float,
        Vector3D,
        Vector3D,
    ],
    Tuple[
        float,
        Vector3D,
        Vector3D,
        Tuple[int, int],
    ],
]:
    """
    Return the minimum distance between two coordinate collections.

    Parameters
    ----------
    points_1 : CoordinateCollection
        First non-empty coordinate collection.
    points_2 : CoordinateCollection
        Second non-empty coordinate collection.
    scene : bool, optional
        Whether scene-transformed coordinates should be preferred.
    squared : bool, optional
        Whether the squared minimum distance should be returned.
    return_indices : bool, optional
        Whether the indices of the closest pair should also be returned.
    return_points : bool, optional
        Whether the two closest coordinates should also be returned.
    copy : bool, optional
        Whether returned point arrays must be copied.

    Returns
    -------
    float
        Minimum distance by default.

    tuple
        Optional return forms are:

        - ``(distance, (index_1, index_2))``;
        - ``(distance, point_1, point_2)``;
        - ``(distance, point_1, point_2, (index_1, index_2))``.

    Raises
    ------
    TypeError
        If either collection cannot be converted to coordinates.
    ValueError
        If either collection is empty or invalid.

    Examples
    --------
    >>> minimum_distance(
    ...     [[0, 0, 0], [5, 0, 0]],
    ...     [[2, 0, 0], [8, 0, 0]],
    ... )
    2.0
    """

    closest_result = closest_point_pair(
        points_1,
        points_2,
        scene=scene,
        return_indices=True,
        return_distance=True,
        squared=squared,
        copy=copy,
    )

    (
        first_point,
        second_point,
        minimum_value,
        indices,
    ) = closest_result

    if not return_points and not return_indices:
        return float(
            minimum_value
        )

    output: List[Any] = [
        float(
            minimum_value
        ),
    ]

    if return_points:
        output.extend(
            [
                first_point,
                second_point,
            ]
        )

    if return_indices:
        output.append(
            indices
        )

    return tuple(
        output
    )


# -----------------------------------------------------------------------------
# Internal spatial-neighbor index
# -----------------------------------------------------------------------------

@dataclass
class _SpatialNeighborIndex:
    """Reusable radius-search index with SciPy and cell-list backends.

    This is intentionally private. It accelerates sparse molecular neighbor
    searches without changing the public geometry API. Source indices are
    always returned in ascending order so callers can preserve legacy pair
    ordering exactly.
    """

    coordinates: FloatArray
    backend: str = "cell_list"
    cell_size: float = 4.0
    tree: Any = field(default=None, repr=False, compare=False)
    cells: Dict[Tuple[int, int, int], Tuple[int, ...]] = field(
        default_factory=dict, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        coordinates = validate_coordinate_matrix(
            self.coordinates,
            name="spatial index coordinates",
            minimum_rows=0,
            allow_empty=True,
            require_finite=True,
            copy=True,
        )
        self.coordinates = coordinates
        self.cell_size = float(self.cell_size)
        if not math.isfinite(self.cell_size) or self.cell_size <= 0.0:
            raise ValueError("cell_size must be a positive finite number.")

    def query_ball_points(
        self,
        points: CoordinateCollection,
        radius: float,
    ) -> Tuple[Tuple[int, ...], ...]:
        query = as_coordinate_matrix(
            points,
            name="spatial query coordinates",
            minimum_rows=0,
            allow_empty=True,
            require_finite=True,
            copy=False,
        )
        radius_value = float(radius)
        if not math.isfinite(radius_value) or radius_value < 0.0:
            raise ValueError("radius must be a non-negative finite number.")
        if query.shape[0] == 0 or self.coordinates.shape[0] == 0:
            return tuple(() for _ in range(query.shape[0]))

        if self.backend == "scipy_ckdtree" and self.tree is not None:
            raw = self.tree.query_ball_point(query, radius_value)
            return tuple(tuple(sorted(int(i) for i in indices)) for indices in raw)

        radius_squared = radius_value * radius_value
        cell_size = self.cell_size
        reach = int(math.ceil(radius_value / cell_size))
        results: List[Tuple[int, ...]] = []
        for point in query:
            base = tuple(int(v) for v in np.floor(point / cell_size))
            candidate_indices: List[int] = []
            for dx in range(-reach, reach + 1):
                for dy in range(-reach, reach + 1):
                    for dz in range(-reach, reach + 1):
                        candidate_indices.extend(
                            self.cells.get(
                                (base[0] + dx, base[1] + dy, base[2] + dz),
                                (),
                            )
                        )
            if not candidate_indices:
                results.append(())
                continue
            unique = np.asarray(sorted(set(candidate_indices)), dtype=np.int64)
            offsets = self.coordinates[unique] - point
            squared = np.einsum("ij,ij->i", offsets, offsets)
            accepted = unique[squared <= radius_squared + DEFAULT_TOLERANCE]
            results.append(tuple(int(i) for i in accepted))
        return tuple(results)

    def query_unique_indices(
        self,
        points: CoordinateCollection,
        radius: float,
    ) -> Tuple[int, ...]:
        return tuple(
            sorted(
                {
                    index
                    for indices in self.query_ball_points(points, radius)
                    for index in indices
                }
            )
        )


def _build_spatial_neighbor_index(
    coordinates: CoordinateCollection,
    *,
    prefer_scipy: bool = True,
    cell_size: float = 4.0,
) -> _SpatialNeighborIndex:
    """Build a reusable radius-search index without requiring SciPy."""

    matrix = as_coordinate_matrix(
        coordinates,
        name="spatial index coordinates",
        minimum_rows=0,
        allow_empty=True,
        require_finite=True,
        copy=True,
    )
    if prefer_scipy and matrix.shape[0]:
        try:
            from scipy.spatial import cKDTree  # type: ignore
        except Exception:
            pass
        else:
            try:
                return _SpatialNeighborIndex(
                    matrix,
                    backend="scipy_ckdtree",
                    cell_size=cell_size,
                    tree=cKDTree(matrix),
                )
            except Exception:
                pass

    normalized_cell_size = float(cell_size)
    if not math.isfinite(normalized_cell_size) or normalized_cell_size <= 0.0:
        normalized_cell_size = 4.0
    mutable_cells: Dict[Tuple[int, int, int], List[int]] = {}
    if matrix.shape[0]:
        cell_coordinates = np.floor(matrix / normalized_cell_size).astype(np.int64)
        for index, cell in enumerate(cell_coordinates):
            key = (int(cell[0]), int(cell[1]), int(cell[2]))
            mutable_cells.setdefault(key, []).append(index)
    cells = {key: tuple(values) for key, values in mutable_cells.items()}
    return _SpatialNeighborIndex(
        matrix,
        backend="cell_list",
        cell_size=normalized_cell_size,
        cells=cells,
    )


# -----------------------------------------------------------------------------
# Point-to-line distance
# -----------------------------------------------------------------------------

def point_line_distance(
    point: Coordinate,
    line_start: Coordinate,
    line_end: Optional[Coordinate] = None,
    *,
    direction: Optional[Coordinate] = None,
    scene: bool = True,
    tolerance: float = DEFAULT_TOLERANCE,
    clamp_to_segment: bool = False,
    squared: bool = False,
    return_projection: bool = False,
    return_parameter: bool = False,
    copy: bool = False,
) -> Union[
    float,
    Tuple[
        float,
        Vector3D,
    ],
    Tuple[
        float,
        float,
    ],
    Tuple[
        float,
        Vector3D,
        float,
    ],
]:
    """
    Return the shortest distance from a point to a line or segment.

    The line may be defined using either:

    - ``line_start`` and ``line_end``; or
    - ``line_start`` and ``direction``.

    Parameters
    ----------
    point : Coordinate
        Point whose distance should be calculated.
    line_start : Coordinate
        Point located on the line.
    line_end : Coordinate, optional
        Second point defining the line or segment.
    direction : Coordinate, optional
        Explicit line-direction vector. Exactly one of ``line_end`` and
        ``direction`` must be provided.
    scene : bool, optional
        Whether scene-transformed coordinates should be preferred.
    tolerance : float, optional
        Minimum accepted norm for the line direction.
    clamp_to_segment : bool, optional
        Whether the closest point should be restricted to the segment between
        ``line_start`` and ``line_end``.
    squared : bool, optional
        Whether the squared distance should be returned.
    return_projection : bool, optional
        Whether the closest projected point should also be returned.
    return_parameter : bool, optional
        Whether the scalar line parameter should also be returned.
    copy : bool, optional
        Whether the returned projected coordinate must be copied.

    Returns
    -------
    float
        Point-to-line distance by default.

    tuple
        Optional return forms are:

        - ``(distance, projected_point)``;
        - ``(distance, parameter)``;
        - ``(distance, projected_point, parameter)``.

    Raises
    ------
    TypeError
        If a coordinate or tolerance value is invalid.
    ValueError
        If the line definition is ambiguous or degenerate.

    Notes
    -----
    For a line defined by two points, the parameter has the following
    interpretation:

    - ``t = 0`` at ``line_start``;
    - ``t = 1`` at ``line_end``;
    - ``0 < t < 1`` inside the segment.

    When ``clamp_to_segment=True``, the parameter is restricted to
    ``[0, 1]``.

    Examples
    --------
    >>> point_line_distance(
    ...     [1, 2, 0],
    ...     [0, 0, 0],
    ...     [3, 0, 0],
    ... )
    2.0
    """

    point_coordinate = as_coordinate(
        point,
        scene=scene,
        name="Point",
        copy=False,
    )

    projected_point, parameter = (
        project_point_on_line(
            point_coordinate,
            line_start,
            line_end,
            direction=direction,
            scene=scene,
            tolerance=tolerance,
            clamp_to_segment=(
                clamp_to_segment
            ),
            return_parameter=True,
            copy=copy,
        )
    )

    distance_squared = squared_distance(
        point_coordinate,
        projected_point,
        scene=False,
    )

    if squared:
        distance_value = (
            distance_squared
        )

    else:
        distance_value = float(
            math.sqrt(
                distance_squared
            )
        )

    if (
        not return_projection
        and not return_parameter
    ):
        return distance_value

    output: List[Any] = [
        distance_value,
    ]

    if return_projection:
        output.append(
            projected_point
        )

    if return_parameter:
        output.append(
            float(
                parameter
            )
        )

    return tuple(
        output
    )


# -----------------------------------------------------------------------------
# Public module interface
# -----------------------------------------------------------------------------

_SECTION_5_PUBLIC_NAMES = [
    "atom_distance",
    "squared_distance",
    "distance_matrix",
    "minimum_distance",
    "closest_point_pair",
    "point_line_distance",
]

_extend_public_names(_SECTION_5_PUBLIC_NAMES)


# =============================================================================
# End of Section 5
# =============================================================================



# =============================================================================
# Section 6 — Angles and Dihedrals
# =============================================================================


# -----------------------------------------------------------------------------
# Internal angular helpers
# -----------------------------------------------------------------------------

def _validate_angle_unit(
    unit: AngleUnit,
) -> AngleUnit:
    """
    Validate and normalize an angular unit.

    Parameters
    ----------
    unit : {"degrees", "radians"}
        Angular unit to validate.

    Returns
    -------
    {"degrees", "radians"}
        Normalized angular unit.

    Raises
    ------
    TypeError
        If ``unit`` is not a string.
    ValueError
        If ``unit`` is not ``"degrees"`` or ``"radians"``.
    """

    if not isinstance(
        unit,
        str,
    ):
        raise TypeError(
            "unit must be a string."
        )

    normalized_unit = unit.strip().lower()

    if normalized_unit not in {
        "degrees",
        "radians",
    }:
        raise ValueError(
            "unit must be either "
            "'degrees' or 'radians'."
        )

    return normalized_unit  # type: ignore[return-value]


def _validate_angular_tolerance(
    tolerance: float,
    *,
    name: str = "tolerance",
) -> float:
    """
    Validate a non-negative finite tolerance.

    Parameters
    ----------
    tolerance : float
        Tolerance value.
    name : str, optional
        Parameter name used in validation messages.

    Returns
    -------
    float
        Validated tolerance.

    Raises
    ------
    TypeError
        If the value is not numeric.
    ValueError
        If the value is negative or non-finite.
    """

    if isinstance(
        tolerance,
        (
            bool,
            np.bool_,
        ),
    ):
        raise TypeError(
            f"{name} must be a numeric value."
        )

    try:
        numeric_tolerance = float(
            tolerance
        )

    except (
        TypeError,
        ValueError,
        OverflowError,
    ) as error:
        raise TypeError(
            f"{name} must be a numeric value."
        ) from error

    if not math.isfinite(
        numeric_tolerance
    ):
        raise ValueError(
            f"{name} must be finite."
        )

    if numeric_tolerance < 0.0:
        raise ValueError(
            f"{name} cannot be negative."
        )

    return numeric_tolerance


def _convert_angle_from_radians(
    angle_radians: float,
    *,
    unit: AngleUnit,
) -> float:
    """
    Convert an angle in radians to the requested unit.

    Parameters
    ----------
    angle_radians : float
        Angle expressed in radians.
    unit : {"degrees", "radians"}
        Requested output unit.

    Returns
    -------
    float
        Converted angle.
    """

    normalized_unit = _validate_angle_unit(
        unit
    )

    if normalized_unit == "radians":
        return float(
            angle_radians
        )

    return float(
        angle_radians
        * DEGREES_PER_RADIAN
    )


def _wrap_signed_angle(
    angle: float,
    *,
    unit: AngleUnit,
) -> float:
    """
    Wrap an angle to its conventional signed interval.

    Parameters
    ----------
    angle : float
        Angle to wrap.
    unit : {"degrees", "radians"}
        Angular unit.

    Returns
    -------
    float
        Angle wrapped to ``[-180, 180]`` degrees or ``[-π, π]`` radians.
    """

    normalized_unit = _validate_angle_unit(
        unit
    )

    if normalized_unit == "degrees":
        period = 360.0
        half_period = 180.0

    else:
        period = 2.0 * math.pi
        half_period = math.pi

    wrapped = (
        angle + half_period
    ) % period - half_period

    # Preserve the positive half-turn boundary.
    if (
        math.isclose(
            wrapped,
            -half_period,
            abs_tol=DEFAULT_ANGLE_TOLERANCE,
        )
        and angle > 0.0
    ):
        wrapped = half_period

    return float(
        wrapped
    )


def _normalize_positive_angle(
    angle: float,
    *,
    unit: AngleUnit,
) -> float:
    """
    Convert a signed angle to a non-negative full-circle interval.

    Parameters
    ----------
    angle : float
        Signed angle.
    unit : {"degrees", "radians"}
        Angular unit.

    Returns
    -------
    float
        Angle in ``[0, 360)`` degrees or ``[0, 2π)`` radians.
    """

    normalized_unit = _validate_angle_unit(
        unit
    )

    period = (
        360.0
        if normalized_unit == "degrees"
        else 2.0 * math.pi
    )

    normalized_angle = angle % period

    if math.isclose(
        normalized_angle,
        period,
        abs_tol=DEFAULT_ANGLE_TOLERANCE,
    ):
        normalized_angle = 0.0

    return float(
        normalized_angle
    )


# -----------------------------------------------------------------------------
# Angle between vectors
# -----------------------------------------------------------------------------

def vector_angle(
    vector_1: Coordinate,
    vector_2: Coordinate,
    *,
    unit: AngleUnit = "degrees",
    scene: bool = True,
    tolerance: float = DEFAULT_TOLERANCE,
) -> float:
    """
    Return the smallest angle between two three-dimensional vectors.

    Parameters
    ----------
    vector_1 : Coordinate
        First vector.
    vector_2 : Coordinate
        Second vector.
    unit : {"degrees", "radians"}, optional
        Unit used for the returned angle.
    scene : bool, optional
        Whether scene-transformed coordinates should be preferred.
    tolerance : float, optional
        Minimum vector norm accepted for the calculation.

    Returns
    -------
    float
        Smallest angle between the vectors, in the interval ``[0, 180]``
        degrees or ``[0, π]`` radians.

    Raises
    ------
    TypeError
        If an input or parameter has an invalid type.
    ValueError
        If either vector is zero or numerically degenerate.

    Notes
    -----
    The angle is calculated using ``atan2``:

    ``atan2(|v1 × v2|, v1 · v2)``

    This formulation is generally more numerically stable near zero and
    180 degrees than calculating the inverse cosine directly.

    Examples
    --------
    >>> vector_angle([1, 0, 0], [0, 1, 0])
    90.0

    >>> vector_angle([1, 0, 0], [-1, 0, 0])
    180.0
    """

    normalized_unit = _validate_angle_unit(
        unit
    )

    numeric_tolerance = (
        _validate_angular_tolerance(
            tolerance
        )
    )

    first_vector = as_coordinate(
        vector_1,
        scene=scene,
        name="First vector",
        copy=False,
    )

    second_vector = as_coordinate(
        vector_2,
        scene=scene,
        name="Second vector",
        copy=False,
    )

    first_norm = vector_norm(
        first_vector,
        scene=False,
    )

    second_norm = vector_norm(
        second_vector,
        scene=False,
    )

    if first_norm <= numeric_tolerance:
        raise ValueError(
            "Cannot calculate an angle using "
            "a zero or near-zero first vector."
        )

    if second_norm <= numeric_tolerance:
        raise ValueError(
            "Cannot calculate an angle using "
            "a zero or near-zero second vector."
        )

    cross_magnitude = vector_norm(
        cross_product(
            first_vector,
            second_vector,
            scene=False,
            copy=False,
        ),
        scene=False,
    )

    scalar_product = dot_product(
        first_vector,
        second_vector,
        scene=False,
    )

    angle_radians = math.atan2(
        cross_magnitude,
        scalar_product,
    )

    return _convert_angle_from_radians(
        angle_radians,
        unit=normalized_unit,
    )


# -----------------------------------------------------------------------------
# Bond angles
# -----------------------------------------------------------------------------

def bond_angle(
    point_1: Coordinate,
    vertex: Coordinate,
    point_3: Coordinate,
    *,
    unit: AngleUnit = "degrees",
    scene: bool = True,
    tolerance: float = DEFAULT_TOLERANCE,
) -> float:
    """
    Return the angle formed by three points.

    The angle is measured at ``vertex`` using the vectors:

    ``point_1 - vertex``

    and:

    ``point_3 - vertex``

    Parameters
    ----------
    point_1 : Coordinate
        First endpoint or atom-like object.
    vertex : Coordinate
        Central point or atom where the angle is measured.
    point_3 : Coordinate
        Second endpoint or atom-like object.
    unit : {"degrees", "radians"}, optional
        Unit used for the returned angle.
    scene : bool, optional
        Whether scene-transformed coordinates should be preferred.
    tolerance : float, optional
        Minimum accepted length of either bond vector.

    Returns
    -------
    float
        Bond angle in ``[0, 180]`` degrees or ``[0, π]`` radians.

    Raises
    ------
    TypeError
        If a coordinate or parameter is invalid.
    ValueError
        If an endpoint coincides with the central point.

    Examples
    --------
    >>> bond_angle(
    ...     [1, 0, 0],
    ...     [0, 0, 0],
    ...     [0, 1, 0],
    ... )
    90.0
    """

    first_coordinate = as_coordinate(
        point_1,
        scene=scene,
        name="First angle point",
        copy=False,
    )

    vertex_coordinate = as_coordinate(
        vertex,
        scene=scene,
        name="Angle vertex",
        copy=False,
    )

    third_coordinate = as_coordinate(
        point_3,
        scene=scene,
        name="Third angle point",
        copy=False,
    )

    first_bond_vector = vector_between(
        vertex_coordinate,
        first_coordinate,
        scene=False,
        copy=False,
    )

    second_bond_vector = vector_between(
        vertex_coordinate,
        third_coordinate,
        scene=False,
        copy=False,
    )

    return vector_angle(
        first_bond_vector,
        second_bond_vector,
        unit=unit,
        scene=False,
        tolerance=tolerance,
    )


# -----------------------------------------------------------------------------
# Dihedral and torsion angles
# -----------------------------------------------------------------------------

def dihedral_angle(
    point_1: Coordinate,
    point_2: Coordinate,
    point_3: Coordinate,
    point_4: Coordinate,
    *,
    unit: AngleUnit = "degrees",
    scene: bool = True,
    tolerance: float = DEFAULT_TOLERANCE,
    signed: bool = True,
) -> float:
    """
    Return the dihedral angle defined by four ordered points.

    The angle describes the rotation between the planes formed by:

    - ``point_1, point_2, point_3``;
    - ``point_2, point_3, point_4``.

    Parameters
    ----------
    point_1 : Coordinate
        First point or atom-like object.
    point_2 : Coordinate
        Second point, defining the start of the central bond.
    point_3 : Coordinate
        Third point, defining the end of the central bond.
    point_4 : Coordinate
        Fourth point or atom-like object.
    unit : {"degrees", "radians"}, optional
        Unit used for the returned angle.
    scene : bool, optional
        Whether scene-transformed coordinates should be preferred.
    tolerance : float, optional
        Minimum accepted norm for the central bond and projected vectors.
    signed : bool, optional
        Whether the oriented angle should be returned.

        When ``True``, the result lies in ``[-180, 180]`` degrees or
        ``[-π, π]`` radians.

        When ``False``, the absolute dihedral lies in ``[0, 180]`` degrees
        or ``[0, π]`` radians.

    Returns
    -------
    float
        Signed or unsigned dihedral angle.

    Raises
    ------
    TypeError
        If a coordinate or parameter is invalid.
    ValueError
        If the central bond is degenerate or either plane cannot be defined.

    Notes
    -----
    The implementation projects the two outer bond vectors onto the plane
    perpendicular to the central bond and calculates the oriented angle using
    ``atan2``.

    Reversing the order of the four points reverses the sign convention only
    when the resulting orientation changes under the selected ordering.

    Examples
    --------
    >>> dihedral_angle(
    ...     [1, 0, 0],
    ...     [0, 0, 0],
    ...     [0, 1, 0],
    ...     [0, 1, 1],
    ... )
    -90.0
    """

    normalized_unit = _validate_angle_unit(
        unit
    )

    numeric_tolerance = (
        _validate_angular_tolerance(
            tolerance
        )
    )

    first_coordinate = as_coordinate(
        point_1,
        scene=scene,
        name="First dihedral point",
        copy=False,
    )

    second_coordinate = as_coordinate(
        point_2,
        scene=scene,
        name="Second dihedral point",
        copy=False,
    )

    third_coordinate = as_coordinate(
        point_3,
        scene=scene,
        name="Third dihedral point",
        copy=False,
    )

    fourth_coordinate = as_coordinate(
        point_4,
        scene=scene,
        name="Fourth dihedral point",
        copy=False,
    )

    first_bond = vector_between(
        first_coordinate,
        second_coordinate,
        scene=False,
        copy=False,
    )

    central_bond = vector_between(
        second_coordinate,
        third_coordinate,
        scene=False,
        copy=False,
    )

    third_bond = vector_between(
        third_coordinate,
        fourth_coordinate,
        scene=False,
        copy=False,
    )

    central_norm = vector_norm(
        central_bond,
        scene=False,
    )

    if central_norm <= numeric_tolerance:
        raise ValueError(
            "Cannot calculate a dihedral angle "
            "because the central bond is zero or near zero."
        )

    central_unit = unit_vector(
        central_bond,
        tolerance=numeric_tolerance,
        scene=False,
        copy=False,
    )

    # Project the outer bonds onto the plane normal to the rotation axis.
    first_projected = reject_vector(
        first_bond,
        central_unit,
        scene=False,
        tolerance=numeric_tolerance,
        copy=False,
    )

    third_projected = reject_vector(
        third_bond,
        central_unit,
        scene=False,
        tolerance=numeric_tolerance,
        copy=False,
    )

    first_projected_norm = vector_norm(
        first_projected,
        scene=False,
    )

    third_projected_norm = vector_norm(
        third_projected,
        scene=False,
    )

    if first_projected_norm <= numeric_tolerance:
        raise ValueError(
            "Cannot define the first dihedral plane: "
            "the first three points are collinear or degenerate."
        )

    if third_projected_norm <= numeric_tolerance:
        raise ValueError(
            "Cannot define the second dihedral plane: "
            "the last three points are collinear or degenerate."
        )

    first_unit = unit_vector(
        first_projected,
        tolerance=numeric_tolerance,
        scene=False,
        copy=False,
    )

    third_unit = unit_vector(
        third_projected,
        tolerance=numeric_tolerance,
        scene=False,
        copy=False,
    )

    x_component = dot_product(
        first_unit,
        third_unit,
        scene=False,
    )

    y_component = dot_product(
        cross_product(
            first_unit,
            third_unit,
            scene=False,
            copy=False,
        ),
        central_unit,
        scene=False,
    )

    # Clip negligible dot-product drift before atan2.
    x_component = float(
        np.clip(
            x_component,
            -1.0,
            1.0,
        )
    )

    angle_radians = math.atan2(
        y_component,
        x_component,
    )

    angle_value = _convert_angle_from_radians(
        angle_radians,
        unit=normalized_unit,
    )

    angle_value = _wrap_signed_angle(
        angle_value,
        unit=normalized_unit,
    )

    if signed:
        return angle_value

    return abs(
        angle_value
    )


def torsion_angle(
    point_1: Coordinate,
    point_2: Coordinate,
    point_3: Coordinate,
    point_4: Coordinate,
    *,
    unit: AngleUnit = "degrees",
    scene: bool = True,
    tolerance: float = DEFAULT_TOLERANCE,
    signed: bool = True,
    positive: bool = False,
) -> float:
    """
    Return the torsion angle defined by four ordered points.

    Parameters
    ----------
    point_1 : Coordinate
        First point or atom-like object.
    point_2 : Coordinate
        Second point.
    point_3 : Coordinate
        Third point.
    point_4 : Coordinate
        Fourth point.
    unit : {"degrees", "radians"}, optional
        Unit used for the returned angle.
    scene : bool, optional
        Whether scene-transformed coordinates should be preferred.
    tolerance : float, optional
        Minimum accepted norm for the vectors defining the torsion.
    signed : bool, optional
        Whether the signed torsion should be preserved.
    positive : bool, optional
        Whether a signed result should be converted to a full positive-circle
        interval.

        In degrees, the result is converted from ``[-180, 180]`` to
        ``[0, 360)``.

        In radians, the result is converted from ``[-π, π]`` to
        ``[0, 2π)``.

        This option requires ``signed=True``.

    Returns
    -------
    float
        Torsion angle in the requested representation.

    Raises
    ------
    TypeError
        If a coordinate or parameter is invalid.
    ValueError
        If the geometry is degenerate or incompatible options are selected.

    Notes
    -----
    In molecular geometry, ``torsion_angle`` and ``dihedral_angle`` usually
    describe the same geometric quantity. This function provides a
    domain-specific interface and optionally converts signed negative angles
    to a positive full-circle representation.

    Examples
    --------
    Signed representation:

    >>> torsion_angle(
    ...     [1, 0, 0],
    ...     [0, 0, 0],
    ...     [0, 1, 0],
    ...     [0, 1, 1],
    ... )
    -90.0

    Positive full-circle representation:

    >>> torsion_angle(
    ...     [1, 0, 0],
    ...     [0, 0, 0],
    ...     [0, 1, 0],
    ...     [0, 1, 1],
    ...     positive=True,
    ... )
    270.0
    """

    normalized_unit = _validate_angle_unit(
        unit
    )

    if positive and not signed:
        raise ValueError(
            "positive=True requires signed=True."
        )

    angle_value = dihedral_angle(
        point_1,
        point_2,
        point_3,
        point_4,
        unit=normalized_unit,
        scene=scene,
        tolerance=tolerance,
        signed=signed,
    )

    if positive:
        return _normalize_positive_angle(
            angle_value,
            unit=normalized_unit,
        )

    return angle_value


# -----------------------------------------------------------------------------
# Public module interface
# -----------------------------------------------------------------------------

_SECTION_6_PUBLIC_NAMES = [
    "vector_angle",
    "bond_angle",
    "dihedral_angle",
    "torsion_angle",
]

_extend_public_names(_SECTION_6_PUBLIC_NAMES)


# =============================================================================
# End of Section 6
# =============================================================================


# =============================================================================
# Section 7 — Molecular Planes
# =============================================================================


# -----------------------------------------------------------------------------
# Plane representation
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Plane:
    """
    Represent a three-dimensional geometric plane.

    A plane is defined by one point and one unit normal vector. Its implicit
    Cartesian equation is:

    ``normal · x + offset = 0``

    where:

    ``offset = -normal · point``

    Parameters
    ----------
    point : Coordinate
        Any point located on the plane.
    normal : Coordinate
        Vector perpendicular to the plane. It is normalized automatically.
    rmsd : float, optional
        Root-mean-square distance of the fitted input points from the plane.
        This value is normally populated by :func:`fit_plane`.
    maximum_deviation : float, optional
        Maximum absolute distance of any fitted input point from the plane.
    singular_values : array-like, optional
        Singular values obtained during plane fitting.
    point_count : int, optional
        Number of points used to construct or fit the plane.
    metadata : Mapping[str, Any], optional
        Additional descriptive information.

    Attributes
    ----------
    point : numpy.ndarray
        Reference point on the plane with shape ``(3,)``.
    normal : numpy.ndarray
        Unit normal vector with shape ``(3,)``.
    rmsd : float or None
        RMS deviation of fitted points from the plane.
    maximum_deviation : float or None
        Maximum absolute deviation from the plane.
    singular_values : numpy.ndarray or None
        Singular values from the fitting procedure.
    point_count : int or None
        Number of points used in the fit.
    metadata : dict
        Additional metadata.

    Notes
    -----
    The normal direction is oriented but geometrically ambiguous: ``normal``
    and ``-normal`` describe the same unoriented plane. Functions comparing
    planes account for this ambiguity unless an oriented comparison is
    explicitly requested.
    """

    point: Coordinate
    normal: Coordinate
    rmsd: Optional[float] = None
    maximum_deviation: Optional[float] = None
    singular_values: Optional[ArrayLike] = None
    point_count: Optional[int] = None
    metadata: GeometryMetadata = field(
        default_factory=dict
    )

    def __post_init__(
        self,
    ) -> None:
        """
        Validate and normalize plane attributes.
        """

        validated_point = as_coordinate(
            self.point,
            scene=False,
            name="Plane point",
            copy=True,
        )

        validated_normal = unit_vector(
            self.normal,
            scene=False,
            tolerance=DEFAULT_TOLERANCE,
            copy=True,
        )

        validated_point.setflags(
            write=False
        )

        validated_normal.setflags(
            write=False
        )

        object.__setattr__(
            self,
            "point",
            validated_point,
        )

        object.__setattr__(
            self,
            "normal",
            validated_normal,
        )

        if self.rmsd is not None:
            rmsd_value = _validate_nonnegative_finite_value(
                self.rmsd,
                name="rmsd",
            )

            object.__setattr__(
                self,
                "rmsd",
                rmsd_value,
            )

        if self.maximum_deviation is not None:
            maximum_deviation_value = (
                _validate_nonnegative_finite_value(
                    self.maximum_deviation,
                    name="maximum_deviation",
                )
            )

            object.__setattr__(
                self,
                "maximum_deviation",
                maximum_deviation_value,
            )

        if self.singular_values is not None:
            try:
                singular_values_array = np.asarray(
                    self.singular_values,
                    dtype=np.float64,
                )

            except (
                TypeError,
                ValueError,
                OverflowError,
            ) as error:
                raise TypeError(
                    "singular_values must be convertible "
                    "to a numeric one-dimensional array."
                ) from error

            singular_values_array = np.ravel(
                singular_values_array
            )

            if singular_values_array.size == 0:
                raise ValueError(
                    "singular_values cannot be empty."
                )

            if not np.all(
                np.isfinite(
                    singular_values_array
                )
            ):
                raise ValueError(
                    "singular_values contains NaN or "
                    "infinite values."
                )

            if np.any(
                singular_values_array < 0.0
            ):
                raise ValueError(
                    "singular_values cannot contain "
                    "negative values."
                )

            singular_values_array = np.array(
                singular_values_array,
                dtype=np.float64,
                copy=True,
            )

            singular_values_array.setflags(
                write=False
            )

            object.__setattr__(
                self,
                "singular_values",
                singular_values_array,
            )

        if self.point_count is not None:
            if isinstance(
                self.point_count,
                (
                    bool,
                    np.bool_,
                ),
            ) or not isinstance(
                self.point_count,
                (
                    int,
                    np.integer,
                ),
            ):
                raise TypeError(
                    "point_count must be an integer."
                )

            point_count_value = int(
                self.point_count
            )

            if point_count_value < 1:
                raise ValueError(
                    "point_count must be at least 1."
                )

            object.__setattr__(
                self,
                "point_count",
                point_count_value,
            )

        if self.metadata is None:
            metadata_value: Dict[str, Any] = {}

        elif isinstance(
            self.metadata,
            Mapping,
        ):
            metadata_value = dict(
                self.metadata
            )

        else:
            raise TypeError(
                "metadata must be a mapping or None."
            )

        object.__setattr__(
            self,
            "metadata",
            metadata_value,
        )

    @property
    def offset(
        self,
    ) -> float:
        """
        Return the constant term of the implicit plane equation.

        Returns
        -------
        float
            Value ``d`` in ``ax + by + cz + d = 0``.
        """

        return -dot_product(
            self.normal,
            self.point,
            scene=False,
        )

    @property
    def coefficients(
        self,
    ) -> Tuple[
        float,
        float,
        float,
        float,
    ]:
        """
        Return the normalized Cartesian plane coefficients.

        Returns
        -------
        tuple of float
            Coefficients ``(a, b, c, d)`` from:

            ``ax + by + cz + d = 0``.
        """

        return (
            float(
                self.normal[0]
            ),
            float(
                self.normal[1]
            ),
            float(
                self.normal[2]
            ),
            self.offset,
        )

    def signed_distance(
        self,
        point: Coordinate,
        *,
        scene: bool = True,
    ) -> float:
        """
        Return the oriented distance from a point to this plane.

        Parameters
        ----------
        point : Coordinate
            Point or coordinate-like object.
        scene : bool, optional
            Whether scene-transformed coordinates should be preferred.

        Returns
        -------
        float
            Signed distance. Positive and negative values correspond to
            opposite sides of the plane according to the normal direction.
        """

        point_coordinate = as_coordinate(
            point,
            scene=scene,
            name="Point",
            copy=False,
        )

        displacement = vector_between(
            self.point,
            point_coordinate,
            scene=False,
            copy=False,
        )

        return dot_product(
            displacement,
            self.normal,
            scene=False,
        )

    def distance(
        self,
        point: Coordinate,
        *,
        scene: bool = True,
    ) -> float:
        """
        Return the absolute distance from a point to this plane.

        Parameters
        ----------
        point : Coordinate
            Point or coordinate-like object.
        scene : bool, optional
            Whether scene-transformed coordinates should be preferred.

        Returns
        -------
        float
            Non-negative point-to-plane distance.
        """

        return abs(
            self.signed_distance(
                point,
                scene=scene,
            )
        )

    def project(
        self,
        point: Coordinate,
        *,
        scene: bool = True,
        copy: bool = False,
    ) -> Vector3D:
        """
        Project a point orthogonally onto this plane.

        Parameters
        ----------
        point : Coordinate
            Point or coordinate-like object.
        scene : bool, optional
            Whether scene-transformed coordinates should be preferred.
        copy : bool, optional
            Whether a copied result must be returned.

        Returns
        -------
        numpy.ndarray
            Projected point with shape ``(3,)``.
        """

        return project_point_on_plane(
            point,
            self,
            scene=scene,
            copy=copy,
        )

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        """
        Convert the plane to a JSON-compatible dictionary.

        Returns
        -------
        dict
            Serialized plane data.
        """

        return {
            "point": self.point.tolist(),
            "normal": self.normal.tolist(),
            "offset": self.offset,
            "coefficients": list(
                self.coefficients
            ),
            "rmsd": self.rmsd,
            "maximum_deviation": (
                self.maximum_deviation
            ),
            "singular_values": (
                None
                if self.singular_values is None
                else self.singular_values.tolist()
            ),
            "point_count": self.point_count,
            "metadata": dict(
                self.metadata
            ),
        }


# -----------------------------------------------------------------------------
# Internal plane helpers
# -----------------------------------------------------------------------------

def _validate_nonnegative_finite_value(
    value: Any,
    *,
    name: str,
) -> float:
    """
    Validate a finite, non-negative numeric value.

    Parameters
    ----------
    value : Any
        Value to validate.
    name : str
        Parameter name used in error messages.

    Returns
    -------
    float
        Validated numeric value.
    """

    if isinstance(
        value,
        (
            bool,
            np.bool_,
        ),
    ):
        raise TypeError(
            f"{name} must be numeric."
        )

    try:
        numeric_value = float(
            value
        )

    except (
        TypeError,
        ValueError,
        OverflowError,
    ) as error:
        raise TypeError(
            f"{name} must be numeric."
        ) from error

    if not math.isfinite(
        numeric_value
    ):
        raise ValueError(
            f"{name} must be finite."
        )

    if numeric_value < 0.0:
        raise ValueError(
            f"{name} cannot be negative."
        )

    return numeric_value


def _coerce_plane(
    plane: Any,
    *,
    point: Optional[Coordinate] = None,
    normal: Optional[Coordinate] = None,
    scene: bool = True,
    tolerance: float = DEFAULT_TOLERANCE,
    name: str = "Plane",
) -> Plane:
    """
    Convert a plane-like definition to a :class:`Plane`.

    Parameters
    ----------
    plane : Any
        Existing ``Plane`` instance or plane-like object.
    point : Coordinate, optional
        Explicit point on the plane.
    normal : Coordinate, optional
        Explicit plane normal.
    scene : bool, optional
        Whether scene coordinates should be preferred.
    tolerance : float, optional
        Minimum accepted normal magnitude.
    name : str, optional
        Name used in validation errors.

    Returns
    -------
    Plane
        Validated plane object.
    """

    if isinstance(
        plane,
        Plane,
    ):
        if point is not None or normal is not None:
            raise ValueError(
                f"{name} was provided as a Plane instance; "
                "point and normal must therefore be omitted."
            )

        return plane

    if plane is not None:
        if point is not None or normal is not None:
            raise ValueError(
                f"{name} cannot be combined with explicit "
                "point or normal arguments."
            )

        if isinstance(
            plane,
            Mapping,
        ):
            if (
                "point" not in plane
                or "normal" not in plane
            ):
                raise ValueError(
                    f"{name} mapping must contain "
                    "'point' and 'normal'."
                )

            point = plane[
                "point"
            ]

            normal = plane[
                "normal"
            ]

        elif (
            hasattr(
                plane,
                "point",
            )
            and hasattr(
                plane,
                "normal",
            )
        ):
            point = getattr(
                plane,
                "point"
            )

            normal = getattr(
                plane,
                "normal"
            )

        else:
            raise TypeError(
                f"{name} must be a Plane, a mapping with "
                "'point' and 'normal', or an object exposing "
                "point and normal attributes."
            )

    if point is None or normal is None:
        raise ValueError(
            f"{name} requires both a point and a normal."
        )

    point_coordinate = as_coordinate(
        point,
        scene=scene,
        name=f"{name} point",
        copy=True,
    )

    normal_vector = unit_vector(
        normal,
        scene=scene,
        tolerance=tolerance,
        copy=True,
    )

    return Plane(
        point=point_coordinate,
        normal=normal_vector,
    )


def _orient_plane_normal(
    normal: Coordinate,
    *,
    reference_normal: Optional[
        Coordinate
    ] = None,
    scene: bool = True,
    tolerance: float = DEFAULT_TOLERANCE,
) -> Vector3D:
    """
    Apply a deterministic or reference-based orientation to a plane normal.

    Parameters
    ----------
    normal : Coordinate
        Normal vector to orient.
    reference_normal : Coordinate, optional
        Preferred orientation. The fitted normal is flipped when its dot
        product with this vector is negative.
    scene : bool, optional
        Whether scene coordinates should be preferred.
    tolerance : float, optional
        Minimum vector norm.

    Returns
    -------
    numpy.ndarray
        Oriented unit normal.
    """

    oriented_normal = unit_vector(
        normal,
        scene=scene,
        tolerance=tolerance,
        copy=True,
    )

    if reference_normal is not None:
        reference = unit_vector(
            reference_normal,
            scene=scene,
            tolerance=tolerance,
            copy=False,
        )

        if dot_product(
            oriented_normal,
            reference,
            scene=False,
        ) < 0.0:
            oriented_normal *= -1.0

        return oriented_normal

    # Use a deterministic sign when no reference normal is supplied.
    for component in oriented_normal:
        if abs(
            float(component)
        ) <= tolerance:
            continue

        if component < 0.0:
            oriented_normal *= -1.0

        break

    return oriented_normal


# -----------------------------------------------------------------------------
# Plane fitting
# -----------------------------------------------------------------------------

def fit_plane(
    points: CoordinateCollection,
    *,
    scene: bool = True,
    tolerance: float = DEFAULT_TOLERANCE,
    reference_normal: Optional[
        Coordinate
    ] = None,
    weights: Optional[ArrayLike] = None,
    metadata: Optional[
        Mapping[str, Any]
    ] = None,
) -> Plane:
    """
    Fit a least-squares plane to three or more points.

    Parameters
    ----------
    points : CoordinateCollection
        Coordinate collection containing at least three non-collinear points.
    scene : bool, optional
        Whether scene-transformed coordinates should be preferred.
    tolerance : float, optional
        Numerical tolerance used to detect degenerate geometries.
    reference_normal : Coordinate, optional
        Vector used to orient the fitted normal. The normal is flipped when
        necessary to point toward the same hemisphere as this reference.
    weights : array-like, optional
        Non-negative weights for the input points. The array must contain one
        weight per coordinate and have a positive total.
    metadata : Mapping[str, Any], optional
        Additional metadata stored in the returned ``Plane``.

    Returns
    -------
    Plane
        Best-fitting plane.

    Raises
    ------
    TypeError
        If coordinates, weights or metadata have invalid types.
    ValueError
        If fewer than three points are supplied, the points are collinear,
        or the weighting scheme is invalid.

    Notes
    -----
    The fit uses singular value decomposition of the centered coordinate
    matrix. The right singular vector associated with the smallest singular
    value is the plane normal.

    For weighted fitting, each centered coordinate is multiplied by the square
    root of its normalized weight before SVD.
    """

    numeric_tolerance = (
        _validate_angular_tolerance(
            tolerance,
            name="tolerance",
        )
    )

    coordinates = get_coordinates(
        points,
        scene=scene,
        name="Plane fitting points",
        minimum_rows=3,
        allow_empty=False,
        copy=False,
    )

    point_count = int(
        coordinates.shape[0]
    )

    normalized_weights: Optional[
        FloatArray
    ]

    if weights is None:
        normalized_weights = None

        plane_point = np.mean(
            coordinates,
            axis=0,
            dtype=np.float64,
        )

        centered_coordinates = (
            coordinates
            - plane_point
        )

        fitting_matrix = (
            centered_coordinates
        )

    else:
        try:
            weight_array = np.asarray(
                weights,
                dtype=np.float64,
            )

        except (
            TypeError,
            ValueError,
            OverflowError,
        ) as error:
            raise TypeError(
                "weights must be convertible to a "
                "numeric one-dimensional array."
            ) from error

        weight_array = np.ravel(
            weight_array
        )

        if weight_array.size != point_count:
            raise ValueError(
                "weights must contain exactly one value "
                f"per point; expected {point_count}, "
                f"received {weight_array.size}."
            )

        if not np.all(
            np.isfinite(
                weight_array
            )
        ):
            raise ValueError(
                "weights contains NaN or infinite values."
            )

        if np.any(
            weight_array < 0.0
        ):
            raise ValueError(
                "weights cannot contain negative values."
            )

        total_weight = float(
            np.sum(
                weight_array
            )
        )

        if total_weight <= numeric_tolerance:
            raise ValueError(
                "weights must have a positive total."
            )

        positive_weight_count = int(
            np.count_nonzero(
                weight_array
                > numeric_tolerance
            )
        )

        if positive_weight_count < 3:
            raise ValueError(
                "At least three points must have "
                "positive weights."
            )

        normalized_weights = (
            weight_array
            / total_weight
        )

        plane_point = np.average(
            coordinates,
            axis=0,
            weights=normalized_weights,
        )

        centered_coordinates = (
            coordinates
            - plane_point
        )

        fitting_matrix = (
            centered_coordinates
            * np.sqrt(
                normalized_weights
            )[:, np.newaxis]
        )

    try:
        (
            _,
            singular_values,
            right_singular_vectors,
        ) = np.linalg.svd(
            fitting_matrix,
            full_matrices=False,
        )

    except np.linalg.LinAlgError as error:
        raise ValueError(
            "Plane fitting failed because singular "
            "value decomposition did not converge."
        ) from error

    if singular_values.size < 2:
        raise ValueError(
            "Plane fitting did not produce enough "
            "independent geometric directions."
        )

    largest_singular_value = float(
        singular_values[0]
    )

    second_singular_value = float(
        singular_values[1]
    )

    rank_threshold = max(
        numeric_tolerance,
        largest_singular_value
        * numeric_tolerance,
    )

    if second_singular_value <= rank_threshold:
        raise ValueError(
            "Cannot fit a unique plane because the "
            "input points are collinear or degenerate."
        )

    fitted_normal = (
        right_singular_vectors[-1]
    )

    fitted_normal = _orient_plane_normal(
        fitted_normal,
        reference_normal=reference_normal,
        scene=scene,
        tolerance=numeric_tolerance,
    )

    signed_deviations = (
        centered_coordinates
        @ fitted_normal
    )

    absolute_deviations = np.abs(
        signed_deviations
    )

    if normalized_weights is None:
        mean_squared_deviation = float(
            np.mean(
                signed_deviations ** 2
            )
        )

    else:
        mean_squared_deviation = float(
            np.sum(
                normalized_weights
                * signed_deviations ** 2
            )
        )

    rmsd_value = float(
        math.sqrt(
            max(
                mean_squared_deviation,
                0.0,
            )
        )
    )

    maximum_deviation_value = float(
        np.max(
            absolute_deviations
        )
    )

    plane_metadata: Dict[str, Any] = {}

    if metadata is not None:
        if not isinstance(
            metadata,
            Mapping,
        ):
            raise TypeError(
                "metadata must be a mapping or None."
            )

        plane_metadata.update(
            metadata
        )

    plane_metadata.setdefault(
        "fit_method",
        (
            "weighted_svd"
            if normalized_weights is not None
            else "svd"
        ),
    )

    plane_metadata.setdefault(
        "weighted",
        normalized_weights is not None,
    )

    return Plane(
        point=plane_point,
        normal=fitted_normal,
        rmsd=rmsd_value,
        maximum_deviation=(
            maximum_deviation_value
        ),
        singular_values=singular_values,
        point_count=point_count,
        metadata=plane_metadata,
    )


# -----------------------------------------------------------------------------
# Plane normal
# -----------------------------------------------------------------------------

def plane_normal(
    points: CoordinateCollection,
    *,
    scene: bool = True,
    tolerance: float = DEFAULT_TOLERANCE,
    reference_normal: Optional[
        Coordinate
    ] = None,
    weights: Optional[ArrayLike] = None,
    copy: bool = False,
) -> Vector3D:
    """
    Return the unit normal of a plane fitted to a coordinate collection.

    Parameters
    ----------
    points : CoordinateCollection
        Three or more points defining or approximating a plane.
    scene : bool, optional
        Whether scene-transformed coordinates should be preferred.
    tolerance : float, optional
        Numerical tolerance used to detect degenerate point arrangements.
    reference_normal : Coordinate, optional
        Preferred orientation for the returned normal.
    weights : array-like, optional
        Optional non-negative fitting weights.
    copy : bool, optional
        Whether a copied normal vector must be returned.

    Returns
    -------
    numpy.ndarray
        Unit plane normal with shape ``(3,)``.

    Raises
    ------
    ValueError
        If the points cannot define a unique plane.
    """

    fitted_plane = fit_plane(
        points,
        scene=scene,
        tolerance=tolerance,
        reference_normal=reference_normal,
        weights=weights,
    )

    if copy:
        return np.array(
            fitted_plane.normal,
            dtype=np.float64,
            copy=True,
        )

    return fitted_plane.normal


# -----------------------------------------------------------------------------
# Point-to-plane distance
# -----------------------------------------------------------------------------

def point_plane_distance(
    point: Coordinate,
    plane: Optional[Any] = None,
    *,
    plane_point: Optional[
        Coordinate
    ] = None,
    plane_normal_vector: Optional[
        Coordinate
    ] = None,
    scene: bool = True,
    tolerance: float = DEFAULT_TOLERANCE,
    signed: bool = False,
    squared: bool = False,
) -> float:
    """
    Return the shortest distance from a point to a plane.

    The plane may be supplied as:

    - a :class:`Plane` instance;
    - an object or mapping containing ``point`` and ``normal``;
    - explicit ``plane_point`` and ``plane_normal_vector`` arguments.

    Parameters
    ----------
    point : Coordinate
        Point or coordinate-like object.
    plane : Any, optional
        Plane-like object.
    plane_point : Coordinate, optional
        Explicit point located on the plane.
    plane_normal_vector : Coordinate, optional
        Explicit normal vector.
    scene : bool, optional
        Whether scene-transformed coordinates should be preferred.
    tolerance : float, optional
        Minimum accepted normal magnitude.
    signed : bool, optional
        Whether the oriented distance should be returned.
    squared : bool, optional
        Whether the squared distance should be returned. Squaring removes the
        sign, so ``signed=True`` and ``squared=True`` cannot be combined.

    Returns
    -------
    float
        Point-to-plane distance.

    Raises
    ------
    ValueError
        If the plane definition is missing, ambiguous or degenerate, or if
        incompatible options are selected.

    Examples
    --------
    >>> plane = Plane(
    ...     point=[0, 0, 0],
    ...     normal=[0, 0, 1],
    ... )
    >>> point_plane_distance(
    ...     [1, 2, 3],
    ...     plane,
    ... )
    3.0
    """

    if signed and squared:
        raise ValueError(
            "signed=True cannot be combined with "
            "squared=True because squaring removes "
            "the distance sign."
        )

    plane_object = _coerce_plane(
        plane,
        point=plane_point,
        normal=plane_normal_vector,
        scene=scene,
        tolerance=tolerance,
        name="Plane",
    )

    point_coordinate = as_coordinate(
        point,
        scene=scene,
        name="Point",
        copy=False,
    )

    signed_distance_value = (
        plane_object.signed_distance(
            point_coordinate,
            scene=False,
        )
    )

    if signed:
        return float(
            signed_distance_value
        )

    absolute_distance = abs(
        signed_distance_value
    )

    if squared:
        return float(
            absolute_distance
            * absolute_distance
        )

    return float(
        absolute_distance
    )


# -----------------------------------------------------------------------------
# Point projection onto a plane
# -----------------------------------------------------------------------------

def project_point_on_plane(
    point: Coordinate,
    plane: Optional[Any] = None,
    *,
    plane_point: Optional[
        Coordinate
    ] = None,
    plane_normal_vector: Optional[
        Coordinate
    ] = None,
    scene: bool = True,
    tolerance: float = DEFAULT_TOLERANCE,
    return_distance: bool = False,
    copy: bool = False,
) -> Union[
    Vector3D,
    Tuple[
        Vector3D,
        float,
    ],
]:
    """
    Project a point orthogonally onto a plane.

    Parameters
    ----------
    point : Coordinate
        Point to project.
    plane : Any, optional
        Plane-like object.
    plane_point : Coordinate, optional
        Explicit point located on the plane.
    plane_normal_vector : Coordinate, optional
        Explicit plane normal.
    scene : bool, optional
        Whether scene-transformed coordinates should be preferred.
    tolerance : float, optional
        Minimum accepted normal magnitude.
    return_distance : bool, optional
        Whether the signed displacement from the plane to the original point
        should also be returned.
    copy : bool, optional
        Whether a copied coordinate must be returned.

    Returns
    -------
    numpy.ndarray
        Projected coordinate.

    tuple
        ``(projected_coordinate, signed_distance)`` when
        ``return_distance=True``.

    Notes
    -----
    The projection is calculated as:

    ``projected = point - signed_distance * normal``

    where the plane normal is a unit vector.
    """

    plane_object = _coerce_plane(
        plane,
        point=plane_point,
        normal=plane_normal_vector,
        scene=scene,
        tolerance=tolerance,
        name="Plane",
    )

    point_coordinate = as_coordinate(
        point,
        scene=scene,
        name="Projected point",
        copy=False,
    )

    signed_distance_value = (
        plane_object.signed_distance(
            point_coordinate,
            scene=False,
        )
    )

    projected_coordinate = (
        point_coordinate
        - signed_distance_value
        * plane_object.normal
    ).astype(
        np.float64,
        copy=False,
    )

    if copy:
        projected_coordinate = np.array(
            projected_coordinate,
            dtype=np.float64,
            copy=True,
        )

    if return_distance:
        return (
            projected_coordinate,
            float(
                signed_distance_value
            ),
        )

    return projected_coordinate


# -----------------------------------------------------------------------------
# Angle between planes
# -----------------------------------------------------------------------------

def angle_between_planes(
    plane_1: Any,
    plane_2: Any,
    *,
    unit: AngleUnit = "degrees",
    scene: bool = True,
    tolerance: float = DEFAULT_TOLERANCE,
    oriented: bool = False,
) -> float:
    """
    Return the angle between two planes.

    Parameters
    ----------
    plane_1 : Any
        First plane-like object.
    plane_2 : Any
        Second plane-like object.
    unit : {"degrees", "radians"}, optional
        Unit used for the returned angle.
    scene : bool, optional
        Whether scene-transformed coordinates should be preferred.
    tolerance : float, optional
        Minimum accepted normal magnitude.
    oriented : bool, optional
        Whether the orientations of the normal vectors should be preserved.

        When ``False``, opposite normals are treated as representing the same
        plane orientation and the result lies in ``[0, 90]`` degrees or
        ``[0, π/2]`` radians.

        When ``True``, the result is the ordinary angle between the oriented
        normals and lies in ``[0, 180]`` degrees or ``[0, π]`` radians.

    Returns
    -------
    float
        Angle between the planes.

    Notes
    -----
    Molecular plane comparisons normally use ``oriented=False`` because a
    plane does not intrinsically distinguish between ``normal`` and
    ``-normal``. This is appropriate for aromatic-ring plane comparisons.
    """

    first_plane = _coerce_plane(
        plane_1,
        scene=scene,
        tolerance=tolerance,
        name="First plane",
    )

    second_plane = _coerce_plane(
        plane_2,
        scene=scene,
        tolerance=tolerance,
        name="Second plane",
    )

    normal_angle = vector_angle(
        first_plane.normal,
        second_plane.normal,
        unit=unit,
        scene=False,
        tolerance=tolerance,
    )

    if oriented:
        return normal_angle

    normalized_unit = _validate_angle_unit(
        unit
    )

    half_turn = (
        180.0
        if normalized_unit == "degrees"
        else math.pi
    )

    acute_plane_angle = min(
        normal_angle,
        half_turn - normal_angle,
    )

    # Remove tiny negative values caused by floating-point rounding.
    if acute_plane_angle < 0.0 and math.isclose(
        acute_plane_angle,
        0.0,
        abs_tol=DEFAULT_ANGLE_TOLERANCE,
    ):
        acute_plane_angle = 0.0

    return float(
        acute_plane_angle
    )


# -----------------------------------------------------------------------------
# Public module interface
# -----------------------------------------------------------------------------

_SECTION_7_PUBLIC_NAMES = [
    "Plane",
    "fit_plane",
    "plane_normal",
    "point_plane_distance",
    "project_point_on_plane",
    "angle_between_planes",
]

_extend_public_names(_SECTION_7_PUBLIC_NAMES)


# =============================================================================
# End of Section 7
# =============================================================================



# =============================================================================
# Section 8 — Aromatic Ring Geometry
# =============================================================================


# -----------------------------------------------------------------------------
# Internal ring helpers
# -----------------------------------------------------------------------------

def _validate_ring_coordinates(
    points: CoordinateCollection,
    *,
    scene: bool = True,
    minimum_atoms: int = 3,
    tolerance: float = DEFAULT_TOLERANCE,
    name: str = "Ring coordinates",
    copy: bool = False,
) -> FloatArray:
    """
    Validate a coordinate collection representing a molecular ring.

    Parameters
    ----------
    points : CoordinateCollection
        Ring atoms or coordinate-like values.
    scene : bool, optional
        Whether scene-transformed coordinates should be preferred.
    minimum_atoms : int, optional
        Minimum number of atoms required.
    tolerance : float, optional
        Numerical tolerance used to detect coincident coordinates.
    name : str, optional
        User-facing collection name.
    copy : bool, optional
        Whether the resulting coordinate matrix must be copied.

    Returns
    -------
    numpy.ndarray
        Validated coordinate matrix with shape ``(N, 3)``.

    Raises
    ------
    TypeError
        If ``minimum_atoms`` is not an integer.
    ValueError
        If too few atoms are supplied, coordinates are duplicated, or the
        geometry cannot define a ring plane.
    """

    if isinstance(
        minimum_atoms,
        (
            bool,
            np.bool_,
        ),
    ) or not isinstance(
        minimum_atoms,
        (
            int,
            np.integer,
        ),
    ):
        raise TypeError(
            "minimum_atoms must be an integer."
        )

    minimum_atoms = int(
        minimum_atoms
    )

    if minimum_atoms < 3:
        raise ValueError(
            "minimum_atoms must be at least 3."
        )

    numeric_tolerance = (
        _validate_angular_tolerance(
            tolerance,
            name="tolerance",
        )
    )

    coordinates = get_coordinates(
        points,
        scene=scene,
        name=name,
        minimum_rows=minimum_atoms,
        allow_empty=False,
        require_finite=True,
        copy=copy,
    )

    if coordinates.shape[0] < minimum_atoms:
        raise ValueError(
            f"{name} must contain at least "
            f"{minimum_atoms} atoms."
        )

    if coordinates.shape[0] > 1:
        pairwise_squared = distance_matrix(
            coordinates,
            scene=False,
            squared=True,
            minimum_rows=1,
            allow_empty=False,
            copy=False,
        )

        upper_triangle = np.triu_indices(
            coordinates.shape[0],
            k=1,
        )

        duplicate_mask = (
            pairwise_squared[
                upper_triangle
            ]
            <= numeric_tolerance ** 2
        )

        if np.any(
            duplicate_mask
        ):
            first_duplicate_position = int(
                np.flatnonzero(
                    duplicate_mask
                )[0]
            )

            first_index = int(
                upper_triangle[0][
                    first_duplicate_position
                ]
            )

            second_index = int(
                upper_triangle[1][
                    first_duplicate_position
                ]
            )

            raise ValueError(
                f"{name} contains coincident or "
                "near-coincident coordinates at indices "
                f"{first_index} and {second_index}."
            )

    # Fitting also validates that the coordinates are not collinear.
    fit_plane(
        coordinates,
        scene=False,
        tolerance=numeric_tolerance,
    )

    return coordinates


def _ring_radial_distances(
    coordinates: CoordinateCollection,
    *,
    center: Optional[Coordinate] = None,
    plane: Optional[Plane] = None,
    scene: bool = True,
    projected: bool = True,
    tolerance: float = DEFAULT_TOLERANCE,
) -> FloatArray:
    """
    Return radial distances of ring atoms from a ring center.

    Parameters
    ----------
    coordinates : CoordinateCollection
        Ring coordinate collection.
    center : Coordinate, optional
        Center used for the radial calculation. When omitted, the coordinate
        centroid is used.
    plane : Plane, optional
        Ring plane. When omitted and ``projected=True``, a plane is fitted.
    scene : bool, optional
        Whether scene-transformed coordinates should be preferred.
    projected : bool, optional
        Whether coordinates should be projected onto the ring plane before
        calculating radial distances.
    tolerance : float, optional
        Numerical tolerance used during plane fitting.

    Returns
    -------
    numpy.ndarray
        One-dimensional array containing one radial distance per atom.
    """

    coordinate_matrix = get_coordinates(
        coordinates,
        scene=scene,
        name="Ring coordinates",
        minimum_rows=3,
        allow_empty=False,
        copy=False,
    )

    if center is None:
        center_coordinate = np.mean(
            coordinate_matrix,
            axis=0,
            dtype=np.float64,
        )

    else:
        center_coordinate = as_coordinate(
            center,
            scene=scene,
            name="Ring center",
            copy=False,
        )

    if projected:
        if plane is None:
            plane_object = fit_plane(
                coordinate_matrix,
                scene=False,
                tolerance=tolerance,
            )

        elif isinstance(
            plane,
            Plane,
        ):
            plane_object = plane

        else:
            plane_object = _coerce_plane(
                plane,
                scene=scene,
                tolerance=tolerance,
                name="Ring plane",
            )

        normal = plane_object.normal
        plane_point = plane_object.point

        center_deviation = dot_product(
            center_coordinate - plane_point,
            normal,
            scene=False,
        )
        projected_center = (
            center_coordinate
            - center_deviation * normal
        )

        coordinate_deviations = (
            (coordinate_matrix - plane_point)
            @ normal
        )
        projected_coordinates = (
            coordinate_matrix
            - coordinate_deviations[:, np.newaxis]
            * normal
        )
        displacements = (
            projected_coordinates
            - projected_center
        )

    else:
        displacements = (
            coordinate_matrix
            - center_coordinate
        )

    squared_radii = np.einsum(
        "ij,ij->i",
        displacements,
        displacements,
        optimize=True,
    )

    np.maximum(
        squared_radii,
        0.0,
        out=squared_radii,
    )

    return np.sqrt(
        squared_radii
    ).astype(
        np.float64,
        copy=False,
    )


# -----------------------------------------------------------------------------
# Ring geometry representation
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class RingGeometry:
    """
    Represent the geometric properties of a molecular ring.

    Parameters
    ----------
    coordinates : CoordinateCollection
        Coordinates of the atoms forming the ring.
    centroid : Coordinate, optional
        Ring centroid. When omitted, it is calculated from ``coordinates``.
    plane : Plane, optional
        Best-fitting ring plane. When omitted, it is calculated automatically.
    radius : float, optional
        Mean projected radial distance of the ring atoms from the centroid.
    minimum_radius : float, optional
        Minimum projected radial distance.
    maximum_radius : float, optional
        Maximum projected radial distance.
    radius_std : float, optional
        Standard deviation of projected radial distances.
    planarity_rmsd : float, optional
        Root-mean-square atom deviation from the fitted ring plane.
    maximum_planarity_deviation : float, optional
        Maximum absolute atom deviation from the ring plane.
    atom_count : int, optional
        Number of atoms defining the ring.
    metadata : Mapping[str, Any], optional
        Additional ring information, such as residue name, ring type or atom
        identifiers.

    Attributes
    ----------
    coordinates : numpy.ndarray
        Ring coordinate matrix with shape ``(N, 3)``.
    centroid : numpy.ndarray
        Ring centroid.
    plane : Plane
        Best-fitting plane.
    normal : numpy.ndarray
        Unit normal of the fitted plane.
    radius : float
        Mean projected ring radius.
    planarity_rmsd : float
        RMS deviation from the fitted plane.

    Notes
    -----
    This class describes ring geometry only. Aromaticity perception and ring
    atom detection should be performed by a separate chemical-topology layer.
    """

    coordinates: CoordinateCollection
    centroid: Optional[Coordinate] = None
    plane: Optional[Plane] = None
    radius: Optional[float] = None
    minimum_radius: Optional[float] = None
    maximum_radius: Optional[float] = None
    radius_std: Optional[float] = None
    planarity_rmsd: Optional[float] = None
    maximum_planarity_deviation: Optional[float] = None
    atom_count: Optional[int] = None
    metadata: GeometryMetadata = field(
        default_factory=dict
    )

    def __post_init__(
        self,
    ) -> None:
        """
        Validate coordinates and calculate missing ring properties.
        """

        coordinate_matrix = (
            _validate_ring_coordinates(
                self.coordinates,
                scene=False,
                minimum_atoms=3,
                tolerance=DEFAULT_TOLERANCE,
                name="Ring coordinates",
                copy=True,
            )
        )

        coordinate_matrix.setflags(
            write=False
        )

        object.__setattr__(
            self,
            "coordinates",
            coordinate_matrix,
        )

        calculated_atom_count = int(
            coordinate_matrix.shape[0]
        )

        if self.atom_count is None:
            atom_count_value = (
                calculated_atom_count
            )

        else:
            if isinstance(
                self.atom_count,
                (
                    bool,
                    np.bool_,
                ),
            ) or not isinstance(
                self.atom_count,
                (
                    int,
                    np.integer,
                ),
            ):
                raise TypeError(
                    "atom_count must be an integer."
                )

            atom_count_value = int(
                self.atom_count
            )

            if (
                atom_count_value
                != calculated_atom_count
            ):
                raise ValueError(
                    "atom_count does not match the "
                    "number of coordinate rows."
                )

        object.__setattr__(
            self,
            "atom_count",
            atom_count_value,
        )

        if self.plane is None:
            plane_object = fit_plane(
                coordinate_matrix,
                scene=False,
                tolerance=DEFAULT_TOLERANCE,
                metadata={
                    "geometry_type": (
                        "molecular_ring"
                    ),
                },
            )

        elif isinstance(
            self.plane,
            Plane,
        ):
            plane_object = self.plane

        else:
            plane_object = _coerce_plane(
                self.plane,
                scene=False,
                tolerance=DEFAULT_TOLERANCE,
                name="Ring plane",
            )

        object.__setattr__(
            self,
            "plane",
            plane_object,
        )

        if self.centroid is None:
            centroid_coordinate = np.mean(
                coordinate_matrix,
                axis=0,
                dtype=np.float64,
            )

            # Ensure that the stored ring center belongs exactly to the
            # best-fitting plane, even for slightly non-planar rings.
            centroid_coordinate = (
                project_point_on_plane(
                    centroid_coordinate,
                    plane_object,
                    scene=False,
                    copy=True,
                )
            )

        else:
            centroid_coordinate = as_coordinate(
                self.centroid,
                scene=False,
                name="Ring centroid",
                copy=True,
            )

            centroid_coordinate = (
                project_point_on_plane(
                    centroid_coordinate,
                    plane_object,
                    scene=False,
                    copy=True,
                )
            )

        centroid_coordinate.setflags(
            write=False
        )

        object.__setattr__(
            self,
            "centroid",
            centroid_coordinate,
        )

        radial_distances = (
            _ring_radial_distances(
                coordinate_matrix,
                center=centroid_coordinate,
                plane=plane_object,
                scene=False,
                projected=True,
            )
        )

        calculated_radius = float(
            np.mean(
                radial_distances
            )
        )

        calculated_minimum_radius = float(
            np.min(
                radial_distances
            )
        )

        calculated_maximum_radius = float(
            np.max(
                radial_distances
            )
        )

        calculated_radius_std = float(
            np.std(
                radial_distances,
                ddof=0,
            )
        )

        radius_values = {
            "radius": (
                self.radius,
                calculated_radius,
            ),
            "minimum_radius": (
                self.minimum_radius,
                calculated_minimum_radius,
            ),
            "maximum_radius": (
                self.maximum_radius,
                calculated_maximum_radius,
            ),
            "radius_std": (
                self.radius_std,
                calculated_radius_std,
            ),
        }

        for (
            attribute_name,
            (
                supplied_value,
                calculated_value,
            ),
        ) in radius_values.items():
            if supplied_value is None:
                final_value = calculated_value

            else:
                final_value = (
                    _validate_nonnegative_finite_value(
                        supplied_value,
                        name=attribute_name,
                    )
                )

            object.__setattr__(
                self,
                attribute_name,
                final_value,
            )

        signed_deviations = (
            (
                coordinate_matrix
                - plane_object.point
            )
            @ plane_object.normal
        )

        absolute_deviations = np.abs(
            signed_deviations
        )

        calculated_planarity_rmsd = float(
            math.sqrt(
                max(
                    float(
                        np.mean(
                            signed_deviations ** 2
                        )
                    ),
                    0.0,
                )
            )
        )

        calculated_maximum_deviation = float(
            np.max(
                absolute_deviations
            )
        )

        if self.planarity_rmsd is None:
            planarity_rmsd_value = (
                calculated_planarity_rmsd
            )

        else:
            planarity_rmsd_value = (
                _validate_nonnegative_finite_value(
                    self.planarity_rmsd,
                    name="planarity_rmsd",
                )
            )

        if (
            self.maximum_planarity_deviation
            is None
        ):
            maximum_planarity_value = (
                calculated_maximum_deviation
            )

        else:
            maximum_planarity_value = (
                _validate_nonnegative_finite_value(
                    self.maximum_planarity_deviation,
                    name=(
                        "maximum_planarity_deviation"
                    ),
                )
            )

        object.__setattr__(
            self,
            "planarity_rmsd",
            planarity_rmsd_value,
        )

        object.__setattr__(
            self,
            "maximum_planarity_deviation",
            maximum_planarity_value,
        )

        if self.metadata is None:
            metadata_value: Dict[str, Any] = {}

        elif isinstance(
            self.metadata,
            Mapping,
        ):
            metadata_value = dict(
                self.metadata
            )

        else:
            raise TypeError(
                "metadata must be a mapping or None."
            )

        metadata_value.setdefault(
            "geometry_type",
            "aromatic_ring",
        )

        object.__setattr__(
            self,
            "metadata",
            metadata_value,
        )

    @property
    def normal(
        self,
    ) -> Vector3D:
        """
        Return the unit normal of the ring plane.

        Returns
        -------
        numpy.ndarray
            Read-only plane normal.
        """

        return self.plane.normal

    @property
    def diameter(
        self,
    ) -> float:
        """
        Return twice the mean ring radius.

        Returns
        -------
        float
            Approximate ring diameter.
        """

        return 2.0 * float(
            self.radius
        )

    @property
    def radius_range(
        self,
    ) -> float:
        """
        Return the range of projected ring radii.

        Returns
        -------
        float
            Difference between maximum and minimum radii.
        """

        return float(
            self.maximum_radius
            - self.minimum_radius
        )

    @property
    def is_planar(
        self,
    ) -> bool:
        """
        Return whether the ring satisfies a default planarity criterion.

        Returns
        -------
        bool
            ``True`` when the RMS deviation is no greater than
            ``DEFAULT_DISTANCE_TOLERANCE``.

        Notes
        -----
        ``DEFAULT_DISTANCE_TOLERANCE`` is a numerical tolerance, not a
        chemically validated aromatic-ring cutoff. Interaction analyses
        should normally use :meth:`check_planarity` with an explicit cutoff.
        """

        return bool(
            self.planarity_rmsd
            <= DEFAULT_DISTANCE_TOLERANCE
        )

    def check_planarity(
        self,
        *,
        rmsd_cutoff: Optional[float] = None,
        maximum_deviation_cutoff: Optional[
            float
        ] = None,
    ) -> bool:
        """
        Test the ring against explicit planarity cutoffs.

        Parameters
        ----------
        rmsd_cutoff : float, optional
            Maximum accepted RMS deviation.
        maximum_deviation_cutoff : float, optional
            Maximum accepted single-atom deviation.

        Returns
        -------
        bool
            Whether all supplied criteria are satisfied.

        Raises
        ------
        ValueError
            If neither cutoff is supplied.
        """

        if (
            rmsd_cutoff is None
            and maximum_deviation_cutoff is None
        ):
            raise ValueError(
                "Provide rmsd_cutoff, "
                "maximum_deviation_cutoff, or both."
            )

        results: List[bool] = []

        if rmsd_cutoff is not None:
            validated_rmsd_cutoff = (
                _validate_nonnegative_finite_value(
                    rmsd_cutoff,
                    name="rmsd_cutoff",
                )
            )

            results.append(
                self.planarity_rmsd
                <= validated_rmsd_cutoff
            )

        if (
            maximum_deviation_cutoff
            is not None
        ):
            validated_maximum_cutoff = (
                _validate_nonnegative_finite_value(
                    maximum_deviation_cutoff,
                    name=(
                        "maximum_deviation_cutoff"
                    ),
                )
            )

            results.append(
                self.maximum_planarity_deviation
                <= validated_maximum_cutoff
            )

        return all(
            results
        )

    def to_dict(
        self,
        *,
        include_coordinates: bool = True,
    ) -> Dict[str, Any]:
        """
        Convert the ring geometry to a JSON-compatible dictionary.

        Parameters
        ----------
        include_coordinates : bool, optional
            Whether atomic coordinates should be included.

        Returns
        -------
        dict
            Serialized ring geometry.
        """

        result: Dict[str, Any] = {
            "centroid": self.centroid.tolist(),
            "normal": self.normal.tolist(),
            "radius": self.radius,
            "minimum_radius": (
                self.minimum_radius
            ),
            "maximum_radius": (
                self.maximum_radius
            ),
            "radius_std": self.radius_std,
            "diameter": self.diameter,
            "radius_range": (
                self.radius_range
            ),
            "planarity_rmsd": (
                self.planarity_rmsd
            ),
            "maximum_planarity_deviation": (
                self.maximum_planarity_deviation
            ),
            "atom_count": self.atom_count,
            "plane": self.plane.to_dict(),
            "metadata": dict(
                self.metadata
            ),
        }

        if include_coordinates:
            result[
                "coordinates"
            ] = self.coordinates.tolist()

        return result


# -----------------------------------------------------------------------------
# Ring centroid
# -----------------------------------------------------------------------------

def ring_centroid(
    ring: Union[
        RingGeometry,
        CoordinateCollection,
    ],
    *,
    scene: bool = True,
    project_to_plane: bool = True,
    tolerance: float = DEFAULT_TOLERANCE,
    copy: bool = False,
) -> Vector3D:
    """
    Return the geometric centroid of a molecular ring.

    Parameters
    ----------
    ring : RingGeometry or CoordinateCollection
        Existing ring geometry or ring atoms.
    scene : bool, optional
        Whether scene-transformed coordinates should be preferred.
    project_to_plane : bool, optional
        Whether the arithmetic centroid should be projected onto the fitted
        ring plane.
    tolerance : float, optional
        Numerical tolerance used during plane fitting.
    copy : bool, optional
        Whether a copied coordinate must be returned.

    Returns
    -------
    numpy.ndarray
        Ring centroid with shape ``(3,)``.
    """

    if isinstance(
        ring,
        RingGeometry,
    ):
        result = ring.centroid

    else:
        coordinates = (
            _validate_ring_coordinates(
                ring,
                scene=scene,
                minimum_atoms=3,
                tolerance=tolerance,
                name="Ring coordinates",
                copy=False,
            )
        )

        result = np.mean(
            coordinates,
            axis=0,
            dtype=np.float64,
        )

        if project_to_plane:
            fitted_plane = fit_plane(
                coordinates,
                scene=False,
                tolerance=tolerance,
            )

            result = project_point_on_plane(
                result,
                fitted_plane,
                scene=False,
                copy=False,
            )

    if copy:
        return np.array(
            result,
            dtype=np.float64,
            copy=True,
        )

    return np.asarray(
        result,
        dtype=np.float64,
    )


# -----------------------------------------------------------------------------
# Ring normal
# -----------------------------------------------------------------------------

def ring_normal(
    ring: Union[
        RingGeometry,
        CoordinateCollection,
    ],
    *,
    scene: bool = True,
    tolerance: float = DEFAULT_TOLERANCE,
    reference_normal: Optional[
        Coordinate
    ] = None,
    copy: bool = False,
) -> Vector3D:
    """
    Return the unit normal of a molecular ring.

    Parameters
    ----------
    ring : RingGeometry or CoordinateCollection
        Existing ring geometry or ring atoms.
    scene : bool, optional
        Whether scene-transformed coordinates should be preferred.
    tolerance : float, optional
        Numerical tolerance used during plane fitting.
    reference_normal : Coordinate, optional
        Preferred normal orientation.
    copy : bool, optional
        Whether a copied vector must be returned.

    Returns
    -------
    numpy.ndarray
        Unit ring normal.
    """

    if isinstance(
        ring,
        RingGeometry,
    ):
        normal_vector = ring.normal

        if reference_normal is not None:
            normal_vector = (
                _orient_plane_normal(
                    normal_vector,
                    reference_normal=(
                        reference_normal
                    ),
                    scene=scene,
                    tolerance=tolerance,
                )
            )

    else:
        normal_vector = plane_normal(
            ring,
            scene=scene,
            tolerance=tolerance,
            reference_normal=(
                reference_normal
            ),
            copy=False,
        )

    if copy:
        return np.array(
            normal_vector,
            dtype=np.float64,
            copy=True,
        )

    return np.asarray(
        normal_vector,
        dtype=np.float64,
    )


# -----------------------------------------------------------------------------
# Ring radius
# -----------------------------------------------------------------------------

def ring_radius(
    ring: Union[
        RingGeometry,
        CoordinateCollection,
    ],
    *,
    scene: bool = True,
    method: Literal[
        "mean",
        "median",
        "minimum",
        "maximum",
        "rms",
    ] = "mean",
    projected: bool = True,
    tolerance: float = DEFAULT_TOLERANCE,
    return_all: bool = False,
) -> Union[
    float,
    Tuple[
        float,
        FloatArray,
    ],
]:
    """
    Return a representative molecular-ring radius.

    Parameters
    ----------
    ring : RingGeometry or CoordinateCollection
        Existing ring geometry or ring atoms.
    scene : bool, optional
        Whether scene-transformed coordinates should be preferred.
    method : {"mean", "median", "minimum", "maximum", "rms"}, optional
        Reduction applied to atom-centroid radial distances.
    projected : bool, optional
        Whether atoms and centroid should be projected onto the ring plane
        before calculating distances.
    tolerance : float, optional
        Numerical tolerance used during plane fitting.
    return_all : bool, optional
        Whether all atom-specific radial distances should also be returned.

    Returns
    -------
    float
        Representative ring radius.

    tuple
        ``(radius, radial_distances)`` when ``return_all=True``.

    Raises
    ------
    ValueError
        If ``method`` is unsupported.
    """

    if not isinstance(
        method,
        str,
    ):
        raise TypeError(
            "method must be a string."
        )

    normalized_method = (
        method.strip().lower()
    )

    supported_methods = {
        "mean",
        "median",
        "minimum",
        "maximum",
        "rms",
    }

    if (
        normalized_method
        not in supported_methods
    ):
        raise ValueError(
            "method must be one of: "
            "'mean', 'median', 'minimum', "
            "'maximum' or 'rms'."
        )

    if (
        isinstance(
            ring,
            RingGeometry,
        )
        and projected
    ):
        coordinates = ring.coordinates
        center_coordinate = ring.centroid
        plane_object = ring.plane

    else:
        coordinates = (
            _validate_ring_coordinates(
                ring.coordinates
                if isinstance(
                    ring,
                    RingGeometry,
                )
                else ring,
                scene=scene,
                minimum_atoms=3,
                tolerance=tolerance,
                name="Ring coordinates",
                copy=False,
            )
        )

        plane_object = (
            fit_plane(
                coordinates,
                scene=False,
                tolerance=tolerance,
            )
            if projected
            else None
        )

        center_coordinate = np.mean(
            coordinates,
            axis=0,
            dtype=np.float64,
        )

    radial_distances = (
        _ring_radial_distances(
            coordinates,
            center=center_coordinate,
            plane=plane_object,
            scene=False,
            projected=projected,
            tolerance=tolerance,
        )
    )

    if normalized_method == "mean":
        radius_value = float(
            np.mean(
                radial_distances
            )
        )

    elif normalized_method == "median":
        radius_value = float(
            np.median(
                radial_distances
            )
        )

    elif normalized_method == "minimum":
        radius_value = float(
            np.min(
                radial_distances
            )
        )

    elif normalized_method == "maximum":
        radius_value = float(
            np.max(
                radial_distances
            )
        )

    else:
        radius_value = float(
            math.sqrt(
                float(
                    np.mean(
                        radial_distances ** 2
                    )
                )
            )
        )

    if return_all:
        return (
            radius_value,
            radial_distances,
        )

    return radius_value


# -----------------------------------------------------------------------------
# Ring planarity
# -----------------------------------------------------------------------------

def ring_planarity(
    ring: Union[
        RingGeometry,
        CoordinateCollection,
    ],
    *,
    scene: bool = True,
    tolerance: float = DEFAULT_TOLERANCE,
    metric: Literal[
        "rmsd",
        "maximum",
        "mean",
        "median",
    ] = "rmsd",
    return_deviations: bool = False,
) -> Union[
    float,
    Tuple[
        float,
        FloatArray,
    ],
]:
    """
    Quantify the deviation of ring atoms from their best-fitting plane.

    Parameters
    ----------
    ring : RingGeometry or CoordinateCollection
        Existing ring geometry or ring atoms.
    scene : bool, optional
        Whether scene-transformed coordinates should be preferred.
    tolerance : float, optional
        Numerical tolerance used during plane fitting.
    metric : {"rmsd", "maximum", "mean", "median"}, optional
        Statistic calculated from absolute atom-to-plane deviations.
    return_deviations : bool, optional
        Whether individual absolute deviations should also be returned.

    Returns
    -------
    float
        Ring planarity metric. Smaller values indicate greater planarity.

    tuple
        ``(planarity_value, absolute_deviations)`` when
        ``return_deviations=True``.

    Notes
    -----
    This function returns a geometric deviation, not a boolean aromaticity or
    planarity classification. Chemical cutoffs should be selected explicitly
    by the interaction-analysis layer.
    """

    if not isinstance(
        metric,
        str,
    ):
        raise TypeError(
            "metric must be a string."
        )

    normalized_metric = (
        metric.strip().lower()
    )

    supported_metrics = {
        "rmsd",
        "maximum",
        "mean",
        "median",
    }

    if (
        normalized_metric
        not in supported_metrics
    ):
        raise ValueError(
            "metric must be one of: "
            "'rmsd', 'maximum', 'mean' "
            "or 'median'."
        )

    if isinstance(
        ring,
        RingGeometry,
    ):
        coordinates = ring.coordinates
        plane_object = ring.plane

    else:
        coordinates = (
            _validate_ring_coordinates(
                ring,
                scene=scene,
                minimum_atoms=3,
                tolerance=tolerance,
                name="Ring coordinates",
                copy=False,
            )
        )

        plane_object = fit_plane(
            coordinates,
            scene=False,
            tolerance=tolerance,
        )

    signed_deviations = (
        (
            coordinates
            - plane_object.point
        )
        @ plane_object.normal
    )

    absolute_deviations = np.abs(
        signed_deviations
    ).astype(
        np.float64,
        copy=False,
    )

    if normalized_metric == "rmsd":
        planarity_value = float(
            math.sqrt(
                max(
                    float(
                        np.mean(
                            signed_deviations ** 2
                        )
                    ),
                    0.0,
                )
            )
        )

    elif normalized_metric == "maximum":
        planarity_value = float(
            np.max(
                absolute_deviations
            )
        )

    elif normalized_metric == "mean":
        planarity_value = float(
            np.mean(
                absolute_deviations
            )
        )

    else:
        planarity_value = float(
            np.median(
                absolute_deviations
            )
        )

    if return_deviations:
        return (
            planarity_value,
            absolute_deviations,
        )

    return planarity_value


# -----------------------------------------------------------------------------
# Public module interface
# -----------------------------------------------------------------------------

_SECTION_8_PUBLIC_NAMES = [
    "RingGeometry",
    "ring_centroid",
    "ring_normal",
    "ring_radius",
    "ring_planarity",
]

_extend_public_names(_SECTION_8_PUBLIC_NAMES)


# =============================================================================
# End of Section 8
# =============================================================================



# =============================================================================
# Section 9 — Pi Interactions
# =============================================================================


# -----------------------------------------------------------------------------
# Pi-stacking geometry representation
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class PiStackGeometry:
    """
    Represent the geometric relationship between two molecular rings.

    Parameters
    ----------
    ring_1 : RingGeometry
        First ring geometry.
    ring_2 : RingGeometry
        Second ring geometry.
    centroid_distance : float
        Euclidean distance between ring centroids.
    plane_angle : float
        Unoriented angle between the ring planes.
    normal_angle : float
        Oriented angle between the ring normals.
    interplanar_distance_1 : float
        Absolute distance from the second-ring centroid to the first-ring
        plane.
    interplanar_distance_2 : float
        Absolute distance from the first-ring centroid to the second-ring
        plane.
    lateral_offset_1 : float
        Distance between the second-ring centroid projection on the first-ring
        plane and the first-ring centroid.
    lateral_offset_2 : float
        Distance between the first-ring centroid projection on the second-ring
        plane and the second-ring centroid.
    mean_interplanar_distance : float
        Mean of the two centroid-to-plane distances.
    mean_lateral_offset : float
        Mean of the two lateral offsets.
    minimum_atom_distance : float, optional
        Minimum distance between atoms from the two rings.
    classification : str
        Geometric classification of the interaction.
    distance_compatible : bool
        Whether the centroid distance satisfies the supplied distance cutoff.
    metadata : Mapping[str, Any], optional
        Additional interaction metadata.

    Notes
    -----
    The geometric classification is based primarily on the angle between the
    ring planes and the lateral displacement between their centroids.

    Distance compatibility is stored separately because geometric orientation
    and interaction plausibility are distinct concepts. Two distant rings may
    be parallel without constituting a physically relevant pi interaction.
    """

    ring_1: RingGeometry
    ring_2: RingGeometry

    centroid_distance: float
    plane_angle: float
    normal_angle: float

    interplanar_distance_1: float
    interplanar_distance_2: float

    lateral_offset_1: float
    lateral_offset_2: float

    mean_interplanar_distance: float
    mean_lateral_offset: float

    minimum_atom_distance: Optional[
        float
    ] = None

    classification: Literal[
        "parallel",
        "parallel-displaced",
        "T-shaped",
        "intermediate",
    ] = "intermediate"

    distance_compatible: bool = True

    metadata: GeometryMetadata = field(
        default_factory=dict
    )

    def __post_init__(
        self,
    ) -> None:
        """
        Validate interaction geometry attributes.
        """

        if not isinstance(
            self.ring_1,
            RingGeometry,
        ):
            raise TypeError(
                "ring_1 must be a RingGeometry instance."
            )

        if not isinstance(
            self.ring_2,
            RingGeometry,
        ):
            raise TypeError(
                "ring_2 must be a RingGeometry instance."
            )

        nonnegative_attributes = (
            "centroid_distance",
            "plane_angle",
            "normal_angle",
            "interplanar_distance_1",
            "interplanar_distance_2",
            "lateral_offset_1",
            "lateral_offset_2",
            "mean_interplanar_distance",
            "mean_lateral_offset",
        )

        for attribute_name in (
            nonnegative_attributes
        ):
            validated_value = (
                _validate_nonnegative_finite_value(
                    getattr(
                        self,
                        attribute_name,
                    ),
                    name=attribute_name,
                )
            )

            object.__setattr__(
                self,
                attribute_name,
                validated_value,
            )

        if self.minimum_atom_distance is not None:
            minimum_atom_distance_value = (
                _validate_nonnegative_finite_value(
                    self.minimum_atom_distance,
                    name="minimum_atom_distance",
                )
            )

            object.__setattr__(
                self,
                "minimum_atom_distance",
                minimum_atom_distance_value,
            )

        supported_classifications = {
            "parallel",
            "parallel-displaced",
            "T-shaped",
            "intermediate",
        }

        if (
            self.classification
            not in supported_classifications
        ):
            raise ValueError(
                "classification must be one of: "
                "'parallel', 'parallel-displaced', "
                "'T-shaped' or 'intermediate'."
            )

        if not isinstance(
            self.distance_compatible,
            (
                bool,
                np.bool_,
            ),
        ):
            raise TypeError(
                "distance_compatible must be boolean."
            )

        object.__setattr__(
            self,
            "distance_compatible",
            bool(
                self.distance_compatible
            ),
        )

        if self.metadata is None:
            metadata_value: Dict[str, Any] = {}

        elif isinstance(
            self.metadata,
            Mapping,
        ):
            metadata_value = dict(
                self.metadata
            )

        else:
            raise TypeError(
                "metadata must be a mapping or None."
            )

        metadata_value.setdefault(
            "geometry_type",
            "pi_stacking",
        )

        object.__setattr__(
            self,
            "metadata",
            metadata_value,
        )

    @property
    def is_parallel_like(
        self,
    ) -> bool:
        """
        Return whether the rings have a parallel-like arrangement.

        Returns
        -------
        bool
            ``True`` for parallel and parallel-displaced geometries.
        """

        return self.classification in {
            "parallel",
            "parallel-displaced",
        }

    @property
    def is_t_shaped(
        self,
    ) -> bool:
        """
        Return whether the rings have a T-shaped arrangement.

        Returns
        -------
        bool
            ``True`` when the interaction is classified as T-shaped.
        """

        return (
            self.classification
            == "T-shaped"
        )

    @property
    def is_interaction_candidate(
        self,
    ) -> bool:
        """
        Return whether the geometry is distance-compatible.

        Returns
        -------
        bool
            Distance compatibility flag.

        Notes
        -----
        This property does not independently validate chemical aromaticity,
        atom types or energetic favorability.
        """

        return self.distance_compatible

    def to_dict(
        self,
        *,
        include_rings: bool = False,
        include_coordinates: bool = False,
    ) -> Dict[str, Any]:
        """
        Convert the pi-stacking geometry to a serializable dictionary.

        Parameters
        ----------
        include_rings : bool, optional
            Whether serialized ring geometries should be included.
        include_coordinates : bool, optional
            Whether ring coordinates should be included when
            ``include_rings=True``.

        Returns
        -------
        dict
            Serialized interaction geometry.
        """

        result: Dict[str, Any] = {
            "classification": (
                self.classification
            ),
            "distance_compatible": (
                self.distance_compatible
            ),
            "centroid_distance": (
                self.centroid_distance
            ),
            "plane_angle": (
                self.plane_angle
            ),
            "normal_angle": (
                self.normal_angle
            ),
            "interplanar_distance_1": (
                self.interplanar_distance_1
            ),
            "interplanar_distance_2": (
                self.interplanar_distance_2
            ),
            "mean_interplanar_distance": (
                self.mean_interplanar_distance
            ),
            "lateral_offset_1": (
                self.lateral_offset_1
            ),
            "lateral_offset_2": (
                self.lateral_offset_2
            ),
            "mean_lateral_offset": (
                self.mean_lateral_offset
            ),
            "minimum_atom_distance": (
                self.minimum_atom_distance
            ),
            "metadata": dict(
                self.metadata
            ),
        }

        if include_rings:
            result["ring_1"] = (
                self.ring_1.to_dict(
                    include_coordinates=(
                        include_coordinates
                    ),
                )
            )

            result["ring_2"] = (
                self.ring_2.to_dict(
                    include_coordinates=(
                        include_coordinates
                    ),
                )
            )

        return result


# -----------------------------------------------------------------------------
# Internal pi-interaction helpers
# -----------------------------------------------------------------------------

def _coerce_ring_geometry(
    ring: Union[
        RingGeometry,
        CoordinateCollection,
    ],
    *,
    scene: bool = True,
    tolerance: float = DEFAULT_TOLERANCE,
    name: str = "Ring",
) -> RingGeometry:
    """
    Convert a ring-like input to :class:`RingGeometry`.

    Parameters
    ----------
    ring : RingGeometry or CoordinateCollection
        Existing ring geometry or ring coordinates.
    scene : bool, optional
        Whether scene-transformed coordinates should be preferred.
    tolerance : float, optional
        Numerical geometry tolerance.
    name : str, optional
        Name used in validation messages.

    Returns
    -------
    RingGeometry
        Validated ring geometry.
    """

    if isinstance(
        ring,
        RingGeometry,
    ):
        return ring

    try:
        coordinates = (
            _validate_ring_coordinates(
                ring,
                scene=scene,
                minimum_atoms=3,
                tolerance=tolerance,
                name=f"{name} coordinates",
                copy=True,
            )
        )

    except (
        TypeError,
        ValueError,
    ) as error:
        raise type(error)(
            f"Invalid {name.lower()}: {error}"
        ) from error

    return RingGeometry(
        coordinates=coordinates,
        metadata={
            "source": "coordinate_collection",
        },
    )


def _classify_pi_stack(
    plane_angle: float,
    lateral_offset: float,
    *,
    parallel_angle_cutoff: float,
    parallel_offset_cutoff: float,
    t_shaped_angle_cutoff: float,
) -> Literal[
    "parallel",
    "parallel-displaced",
    "T-shaped",
    "intermediate",
]:
    """
    Classify a ring-ring orientation.

    Parameters
    ----------
    plane_angle : float
        Unoriented angle between planes in degrees.
    lateral_offset : float
        Representative lateral centroid displacement.
    parallel_angle_cutoff : float
        Maximum plane angle accepted as parallel-like.
    parallel_offset_cutoff : float
        Maximum lateral displacement accepted as directly parallel.
    t_shaped_angle_cutoff : float
        Minimum plane angle accepted as T-shaped.

    Returns
    -------
    str
        Pi-stacking classification.
    """

    if plane_angle <= parallel_angle_cutoff:
        if (
            lateral_offset
            <= parallel_offset_cutoff
        ):
            return "parallel"

        return "parallel-displaced"

    if plane_angle >= t_shaped_angle_cutoff:
        return "T-shaped"

    return "intermediate"


def _validate_pi_angle_cutoff(
    value: Any,
    *,
    name: str,
) -> float:
    """
    Validate an angular cutoff in degrees.

    Parameters
    ----------
    value : Any
        Angular cutoff.
    name : str
        Parameter name.

    Returns
    -------
    float
        Validated angle in degrees.
    """

    validated_value = (
        _validate_nonnegative_finite_value(
            value,
            name=name,
        )
    )

    if validated_value > 90.0:
        raise ValueError(
            f"{name} cannot exceed 90 degrees "
            "for an unoriented plane angle."
        )

    return validated_value


# -----------------------------------------------------------------------------
# Pi-stacking geometry
# -----------------------------------------------------------------------------

def pi_stack_geometry(
    ring_1: Union[
        RingGeometry,
        CoordinateCollection,
    ],
    ring_2: Union[
        RingGeometry,
        CoordinateCollection,
    ],
    *,
    scene: bool = True,
    tolerance: float = DEFAULT_TOLERANCE,
    parallel_angle_cutoff: float = 30.0,
    parallel_offset_cutoff: float = 1.5,
    t_shaped_angle_cutoff: float = 60.0,
    maximum_centroid_distance: Optional[
        float
    ] = 6.0,
    calculate_minimum_atom_distance: bool = True,
    metadata: Optional[
        Mapping[str, Any]
    ] = None,
) -> PiStackGeometry:
    """
    Calculate and classify the geometry between two molecular rings.

    Parameters
    ----------
    ring_1 : RingGeometry or CoordinateCollection
        First ring.
    ring_2 : RingGeometry or CoordinateCollection
        Second ring.
    scene : bool, optional
        Whether scene-transformed coordinates should be preferred.
    tolerance : float, optional
        Numerical tolerance used in geometric operations.
    parallel_angle_cutoff : float, optional
        Maximum angle between planes for a parallel-like arrangement.
    parallel_offset_cutoff : float, optional
        Maximum representative lateral displacement for classification as
        directly parallel. Larger displacements remain parallel-like but are
        classified as parallel-displaced.
    t_shaped_angle_cutoff : float, optional
        Minimum plane angle for a T-shaped arrangement.
    maximum_centroid_distance : float, optional
        Maximum centroid distance considered compatible with a candidate
        interaction. Set to ``None`` to disable distance screening.
    calculate_minimum_atom_distance : bool, optional
        Whether the minimum interatomic ring-ring distance should be
        calculated.
    metadata : Mapping[str, Any], optional
        Additional interaction metadata.

    Returns
    -------
    PiStackGeometry
        Structured ring-ring interaction geometry.

    Raises
    ------
    ValueError
        If cutoffs are invalid or geometrically inconsistent.

    Notes
    -----
    Classification rules are:

    - ``parallel``:
      plane angle is at most ``parallel_angle_cutoff`` and lateral offset is
      at most ``parallel_offset_cutoff``;
    - ``parallel-displaced``:
      plane angle is at most ``parallel_angle_cutoff`` but lateral offset is
      larger than ``parallel_offset_cutoff``;
    - ``T-shaped``:
      plane angle is at least ``t_shaped_angle_cutoff``;
    - ``intermediate``:
      geometry lies between the parallel-like and T-shaped angular regions.

    The default cutoffs are configurable geometric defaults. They should not
    be interpreted as universal physicochemical criteria.
    """

    numeric_tolerance = (
        _validate_angular_tolerance(
            tolerance,
            name="tolerance",
        )
    )

    parallel_angle_value = (
        _validate_pi_angle_cutoff(
            parallel_angle_cutoff,
            name="parallel_angle_cutoff",
        )
    )

    t_shaped_angle_value = (
        _validate_pi_angle_cutoff(
            t_shaped_angle_cutoff,
            name="t_shaped_angle_cutoff",
        )
    )

    if (
        parallel_angle_value
        >= t_shaped_angle_value
    ):
        raise ValueError(
            "parallel_angle_cutoff must be smaller "
            "than t_shaped_angle_cutoff."
        )

    parallel_offset_value = (
        _validate_nonnegative_finite_value(
            parallel_offset_cutoff,
            name="parallel_offset_cutoff",
        )
    )

    if maximum_centroid_distance is None:
        maximum_centroid_distance_value = None

    else:
        maximum_centroid_distance_value = (
            _validate_nonnegative_finite_value(
                maximum_centroid_distance,
                name="maximum_centroid_distance",
            )
        )

    first_ring = _coerce_ring_geometry(
        ring_1,
        scene=scene,
        tolerance=numeric_tolerance,
        name="First ring",
    )

    second_ring = _coerce_ring_geometry(
        ring_2,
        scene=scene,
        tolerance=numeric_tolerance,
        name="Second ring",
    )

    centroid_distance_value = float(
        distance(
            first_ring.centroid,
            second_ring.centroid,
        )
    )

    plane_angle_value = (
        angle_between_planes(
            first_ring.plane,
            second_ring.plane,
            unit="degrees",
            scene=False,
            tolerance=numeric_tolerance,
            oriented=False,
        )
    )

    normal_angle_value = vector_angle(
        first_ring.normal,
        second_ring.normal,
        unit="degrees",
        scene=False,
        tolerance=numeric_tolerance,
    )

    projected_second_on_first = (
        project_point_on_plane(
            second_ring.centroid,
            first_ring.plane,
            scene=False,
            copy=False,
        )
    )

    projected_first_on_second = (
        project_point_on_plane(
            first_ring.centroid,
            second_ring.plane,
            scene=False,
            copy=False,
        )
    )

    interplanar_distance_1 = (
        point_plane_distance(
            second_ring.centroid,
            first_ring.plane,
            scene=False,
            signed=False,
        )
    )

    interplanar_distance_2 = (
        point_plane_distance(
            first_ring.centroid,
            second_ring.plane,
            scene=False,
            signed=False,
        )
    )

    lateral_offset_1 = float(
        distance(
            first_ring.centroid,
            projected_second_on_first,
        )
    )

    lateral_offset_2 = float(
        distance(
            second_ring.centroid,
            projected_first_on_second,
        )
    )

    mean_interplanar_distance = float(
        (
            interplanar_distance_1
            + interplanar_distance_2
        )
        / 2.0
    )

    mean_lateral_offset = float(
        (
            lateral_offset_1
            + lateral_offset_2
        )
        / 2.0
    )

    classification = _classify_pi_stack(
        plane_angle_value,
        mean_lateral_offset,
        parallel_angle_cutoff=(
            parallel_angle_value
        ),
        parallel_offset_cutoff=(
            parallel_offset_value
        ),
        t_shaped_angle_cutoff=(
            t_shaped_angle_value
        ),
    )

    if maximum_centroid_distance_value is None:
        distance_compatible = True

    else:
        distance_compatible = (
            centroid_distance_value
            <= maximum_centroid_distance_value
        )

    if calculate_minimum_atom_distance:
        minimum_atom_distance_value = (
            minimum_distance(
                first_ring.coordinates,
                second_ring.coordinates,
                scene=False,
                squared=False,
            )
        )

    else:
        minimum_atom_distance_value = None

    interaction_metadata: Dict[
        str,
        Any,
    ] = {}

    if metadata is not None:
        if not isinstance(
            metadata,
            Mapping,
        ):
            raise TypeError(
                "metadata must be a mapping or None."
            )

        interaction_metadata.update(
            metadata
        )

    interaction_metadata.setdefault(
        "parallel_angle_cutoff",
        parallel_angle_value,
    )

    interaction_metadata.setdefault(
        "parallel_offset_cutoff",
        parallel_offset_value,
    )

    interaction_metadata.setdefault(
        "t_shaped_angle_cutoff",
        t_shaped_angle_value,
    )

    interaction_metadata.setdefault(
        "maximum_centroid_distance",
        maximum_centroid_distance_value,
    )

    return PiStackGeometry(
        ring_1=first_ring,
        ring_2=second_ring,
        centroid_distance=(
            centroid_distance_value
        ),
        plane_angle=plane_angle_value,
        normal_angle=normal_angle_value,
        interplanar_distance_1=(
            interplanar_distance_1
        ),
        interplanar_distance_2=(
            interplanar_distance_2
        ),
        lateral_offset_1=(
            lateral_offset_1
        ),
        lateral_offset_2=(
            lateral_offset_2
        ),
        mean_interplanar_distance=(
            mean_interplanar_distance
        ),
        mean_lateral_offset=(
            mean_lateral_offset
        ),
        minimum_atom_distance=(
            minimum_atom_distance_value
        ),
        classification=classification,
        distance_compatible=(
            distance_compatible
        ),
        metadata=interaction_metadata,
    )


# -----------------------------------------------------------------------------
# Cation-pi geometry
# -----------------------------------------------------------------------------

def cation_pi_geometry(
    ring: Union[
        RingGeometry,
        CoordinateCollection,
    ],
    cation: Coordinate,
    *,
    scene: bool = True,
    tolerance: float = DEFAULT_TOLERANCE,
    maximum_distance: Optional[
        float
    ] = 6.0,
    maximum_lateral_offset: Optional[
        float
    ] = None,
    return_projection: bool = True,
    metadata: Optional[
        Mapping[str, Any]
    ] = None,
) -> Dict[str, Any]:
    """
    Calculate the geometry between a cation and a molecular ring.

    Parameters
    ----------
    ring : RingGeometry or CoordinateCollection
        Aromatic or pi-system ring.
    cation : Coordinate
        Cation position or atom-like object.
    scene : bool, optional
        Whether scene-transformed coordinates should be preferred.
    tolerance : float, optional
        Numerical geometry tolerance.
    maximum_distance : float, optional
        Maximum centroid-to-cation distance considered compatible with a
        candidate interaction. Set to ``None`` to disable this criterion.
    maximum_lateral_offset : float, optional
        Maximum accepted in-plane displacement from the ring centroid. When
        omitted, the ring's maximum radius is used.
    return_projection : bool, optional
        Whether the projected cation coordinate should be included.
    metadata : Mapping[str, Any], optional
        Additional metadata.

    Returns
    -------
    dict
        Dictionary containing:

        - ``centroid_distance``;
        - ``plane_distance``;
        - ``lateral_offset``;
        - ``approach_angle``;
        - ``signed_plane_distance``;
        - ``within_ring_projection``;
        - ``distance_compatible``;
        - ``geometry_compatible``;
        - ``projected_point`` when requested.

    Notes
    -----
    ``approach_angle`` is the acute angle between the centroid-to-cation
    vector and the ring normal:

    - values near 0 degrees indicate a face-on approach;
    - values near 90 degrees indicate an edge-on approach.
    """

    numeric_tolerance = (
        _validate_angular_tolerance(
            tolerance,
            name="tolerance",
        )
    )

    ring_geometry = _coerce_ring_geometry(
        ring,
        scene=scene,
        tolerance=numeric_tolerance,
        name="Ring",
    )

    cation_coordinate = as_coordinate(
        cation,
        scene=scene,
        name="Cation coordinate",
        copy=False,
    )

    centroid_to_cation = vector_between(
        ring_geometry.centroid,
        cation_coordinate,
        scene=False,
        copy=False,
    )

    centroid_distance_value = vector_norm(
        centroid_to_cation,
        scene=False,
    )

    if centroid_distance_value <= numeric_tolerance:
        approach_angle_value = 0.0

    else:
        oriented_angle = vector_angle(
            centroid_to_cation,
            ring_geometry.normal,
            unit="degrees",
            scene=False,
            tolerance=numeric_tolerance,
        )

        approach_angle_value = min(
            oriented_angle,
            180.0 - oriented_angle,
        )

    (
        projected_point,
        signed_plane_distance,
    ) = project_point_on_plane(
        cation_coordinate,
        ring_geometry.plane,
        scene=False,
        return_distance=True,
        copy=True,
    )

    plane_distance_value = abs(
        signed_plane_distance
    )

    lateral_offset_value = float(
        distance(
            projected_point,
            ring_geometry.centroid,
        )
    )

    if maximum_distance is None:
        maximum_distance_value = None
        distance_compatible = True

    else:
        maximum_distance_value = (
            _validate_nonnegative_finite_value(
                maximum_distance,
                name="maximum_distance",
            )
        )

        distance_compatible = (
            centroid_distance_value
            <= maximum_distance_value
        )

    if maximum_lateral_offset is None:
        maximum_lateral_offset_value = (
            float(
                ring_geometry.maximum_radius
            )
        )

    else:
        maximum_lateral_offset_value = (
            _validate_nonnegative_finite_value(
                maximum_lateral_offset,
                name="maximum_lateral_offset",
            )
        )

    within_ring_projection = (
        lateral_offset_value
        <= maximum_lateral_offset_value
    )

    geometry_compatible = (
        distance_compatible
        and within_ring_projection
    )

    interaction_metadata: Dict[
        str,
        Any,
    ] = {}

    if metadata is not None:
        if not isinstance(
            metadata,
            Mapping,
        ):
            raise TypeError(
                "metadata must be a mapping or None."
            )

        interaction_metadata.update(
            metadata
        )

    interaction_metadata.setdefault(
        "geometry_type",
        "cation_pi",
    )

    interaction_metadata.setdefault(
        "maximum_distance",
        maximum_distance_value,
    )

    interaction_metadata.setdefault(
        "maximum_lateral_offset",
        maximum_lateral_offset_value,
    )

    result: Dict[str, Any] = {
        "ring_centroid": (
            ring_geometry.centroid.tolist()
        ),
        "ring_normal": (
            ring_geometry.normal.tolist()
        ),
        "cation_coordinate": (
            cation_coordinate.tolist()
        ),
        "centroid_distance": float(
            centroid_distance_value
        ),
        "plane_distance": float(
            plane_distance_value
        ),
        "signed_plane_distance": float(
            signed_plane_distance
        ),
        "lateral_offset": float(
            lateral_offset_value
        ),
        "approach_angle": float(
            approach_angle_value
        ),
        "within_ring_projection": bool(
            within_ring_projection
        ),
        "distance_compatible": bool(
            distance_compatible
        ),
        "geometry_compatible": bool(
            geometry_compatible
        ),
        "metadata": (
            interaction_metadata
        ),
    }

    if return_projection:
        result["projected_point"] = (
            projected_point.tolist()
        )

    return result


# -----------------------------------------------------------------------------
# Public module interface
# -----------------------------------------------------------------------------

_SECTION_9_PUBLIC_NAMES = [
    "PiStackGeometry",
    "pi_stack_geometry",
    "cation_pi_geometry",
]

_extend_public_names(_SECTION_9_PUBLIC_NAMES)


# =============================================================================
# End of Section 9
# =============================================================================




# =============================================================================
# Section 10 — Hydrogen-Bond Geometry
# =============================================================================


# -----------------------------------------------------------------------------
# Hydrogen-bond geometry representation
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class HydrogenBondGeometry:
    """
    Represent the geometry of a potential hydrogen bond.

    A conventional hydrogen bond is represented as:

    ``donor — hydrogen ··· acceptor``

    Parameters
    ----------
    donor_coordinate : Coordinate
        Coordinate of the donor atom.
    acceptor_coordinate : Coordinate
        Coordinate of the acceptor atom.
    hydrogen_coordinate : Coordinate, optional
        Coordinate of the donor-bound hydrogen. This may be omitted for
        structures without explicit hydrogens.
    donor_acceptor_distance : float
        Distance between donor and acceptor atoms.
    hydrogen_acceptor_distance : float, optional
        Distance between hydrogen and acceptor atoms.
    donor_hydrogen_distance : float, optional
        Distance between donor and hydrogen atoms.
    donor_hydrogen_acceptor_angle : float, optional
        D-H···A angle in degrees.
    distance_compatible : bool
        Whether all available distance criteria are satisfied.
    angle_compatible : bool or None
        Whether the angular criterion is satisfied. ``None`` means that the
        angle could not be evaluated because no hydrogen was supplied.
    geometry_compatible : bool
        Whether the complete set of available geometric criteria is
        satisfied.
    has_explicit_hydrogen : bool
        Whether an explicit hydrogen coordinate was used.
    metadata : Mapping[str, Any], optional
        Additional interaction metadata.

    Notes
    -----
    This class represents geometry only. It does not determine whether the
    atoms are chemically valid donors or acceptors.
    """

    donor_coordinate: Coordinate
    acceptor_coordinate: Coordinate
    hydrogen_coordinate: Optional[
        Coordinate
    ] = None

    donor_acceptor_distance: float = 0.0
    hydrogen_acceptor_distance: Optional[
        float
    ] = None
    donor_hydrogen_distance: Optional[
        float
    ] = None
    donor_hydrogen_acceptor_angle: Optional[
        float
    ] = None

    distance_compatible: bool = False
    angle_compatible: Optional[bool] = None
    geometry_compatible: bool = False
    has_explicit_hydrogen: bool = False

    metadata: GeometryMetadata = field(
        default_factory=dict
    )

    def __post_init__(
        self,
    ) -> None:
        """
        Validate and normalize hydrogen-bond geometry attributes.
        """

        donor_coordinate = as_coordinate(
            self.donor_coordinate,
            scene=False,
            name="Donor coordinate",
            copy=True,
        )

        acceptor_coordinate = as_coordinate(
            self.acceptor_coordinate,
            scene=False,
            name="Acceptor coordinate",
            copy=True,
        )

        donor_coordinate.setflags(
            write=False
        )

        acceptor_coordinate.setflags(
            write=False
        )

        object.__setattr__(
            self,
            "donor_coordinate",
            donor_coordinate,
        )

        object.__setattr__(
            self,
            "acceptor_coordinate",
            acceptor_coordinate,
        )

        if self.hydrogen_coordinate is None:
            hydrogen_coordinate = None

        else:
            hydrogen_coordinate = as_coordinate(
                self.hydrogen_coordinate,
                scene=False,
                name="Hydrogen coordinate",
                copy=True,
            )

            hydrogen_coordinate.setflags(
                write=False
            )

        object.__setattr__(
            self,
            "hydrogen_coordinate",
            hydrogen_coordinate,
        )

        donor_acceptor_distance = (
            _validate_nonnegative_finite_value(
                self.donor_acceptor_distance,
                name="donor_acceptor_distance",
            )
        )

        object.__setattr__(
            self,
            "donor_acceptor_distance",
            donor_acceptor_distance,
        )

        optional_nonnegative_attributes = (
            "hydrogen_acceptor_distance",
            "donor_hydrogen_distance",
            "donor_hydrogen_acceptor_angle",
        )

        for attribute_name in (
            optional_nonnegative_attributes
        ):
            attribute_value = getattr(
                self,
                attribute_name,
            )

            if attribute_value is None:
                continue

            validated_value = (
                _validate_nonnegative_finite_value(
                    attribute_value,
                    name=attribute_name,
                )
            )

            if (
                attribute_name
                == "donor_hydrogen_acceptor_angle"
                and validated_value > 180.0
            ):
                raise ValueError(
                    "donor_hydrogen_acceptor_angle "
                    "cannot exceed 180 degrees."
                )

            object.__setattr__(
                self,
                attribute_name,
                validated_value,
            )

        boolean_attributes = (
            "distance_compatible",
            "geometry_compatible",
            "has_explicit_hydrogen",
        )

        for attribute_name in boolean_attributes:
            attribute_value = getattr(
                self,
                attribute_name,
            )

            if not isinstance(
                attribute_value,
                (
                    bool,
                    np.bool_,
                ),
            ):
                raise TypeError(
                    f"{attribute_name} must be boolean."
                )

            object.__setattr__(
                self,
                attribute_name,
                bool(
                    attribute_value
                ),
            )

        if self.angle_compatible is not None:
            if not isinstance(
                self.angle_compatible,
                (
                    bool,
                    np.bool_,
                ),
            ):
                raise TypeError(
                    "angle_compatible must be boolean "
                    "or None."
                )

            object.__setattr__(
                self,
                "angle_compatible",
                bool(
                    self.angle_compatible
                ),
            )

        coordinate_has_hydrogen = (
            hydrogen_coordinate is not None
        )

        if (
            self.has_explicit_hydrogen
            != coordinate_has_hydrogen
        ):
            raise ValueError(
                "has_explicit_hydrogen does not match "
                "the presence of hydrogen_coordinate."
            )

        hydrogen_dependent_attributes = (
            self.hydrogen_acceptor_distance,
            self.donor_hydrogen_distance,
            self.donor_hydrogen_acceptor_angle,
        )

        if not coordinate_has_hydrogen:
            if any(
                value is not None
                for value in hydrogen_dependent_attributes
            ):
                raise ValueError(
                    "Hydrogen-dependent geometric values "
                    "cannot be provided without a hydrogen "
                    "coordinate."
                )

            if self.angle_compatible is not None:
                raise ValueError(
                    "angle_compatible must be None when "
                    "no explicit hydrogen is available."
                )

        else:
            if any(
                value is None
                for value in hydrogen_dependent_attributes
            ):
                raise ValueError(
                    "All hydrogen-dependent geometric "
                    "values must be supplied when an "
                    "explicit hydrogen is present."
                )

            if self.angle_compatible is None:
                raise ValueError(
                    "angle_compatible cannot be None when "
                    "an explicit hydrogen is present."
                )

        if self.metadata is None:
            metadata_value: Dict[str, Any] = {}

        elif isinstance(
            self.metadata,
            Mapping,
        ):
            metadata_value = dict(
                self.metadata
            )

        else:
            raise TypeError(
                "metadata must be a mapping or None."
            )

        metadata_value.setdefault(
            "geometry_type",
            "hydrogen_bond",
        )

        object.__setattr__(
            self,
            "metadata",
            metadata_value,
        )

    @property
    def angle_deviation_from_linear(
        self,
    ) -> Optional[float]:
        """
        Return the deviation of the D-H···A angle from linearity.

        Returns
        -------
        float or None
            ``180 - angle`` in degrees, or ``None`` when no explicit
            hydrogen is available.
        """

        if (
            self.donor_hydrogen_acceptor_angle
            is None
        ):
            return None

        return float(
            180.0
            - self.donor_hydrogen_acceptor_angle
        )

    @property
    def is_complete(
        self,
    ) -> bool:
        """
        Return whether an explicit hydrogen was used.

        Returns
        -------
        bool
            ``True`` when all D-H···A geometric terms are available.
        """

        return self.has_explicit_hydrogen

    @property
    def is_distance_only(
        self,
    ) -> bool:
        """
        Return whether only heavy-atom geometry was evaluated.

        Returns
        -------
        bool
            ``True`` when no explicit hydrogen was supplied.
        """

        return not self.has_explicit_hydrogen

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        """
        Convert the geometry to a JSON-compatible dictionary.

        Returns
        -------
        dict
            Serialized hydrogen-bond geometry.
        """

        return {
            "donor_coordinate": (
                self.donor_coordinate.tolist()
            ),
            "hydrogen_coordinate": (
                None
                if self.hydrogen_coordinate is None
                else self.hydrogen_coordinate.tolist()
            ),
            "acceptor_coordinate": (
                self.acceptor_coordinate.tolist()
            ),
            "donor_acceptor_distance": (
                self.donor_acceptor_distance
            ),
            "hydrogen_acceptor_distance": (
                self.hydrogen_acceptor_distance
            ),
            "donor_hydrogen_distance": (
                self.donor_hydrogen_distance
            ),
            "donor_hydrogen_acceptor_angle": (
                self.donor_hydrogen_acceptor_angle
            ),
            "angle_deviation_from_linear": (
                self.angle_deviation_from_linear
            ),
            "distance_compatible": (
                self.distance_compatible
            ),
            "angle_compatible": (
                self.angle_compatible
            ),
            "geometry_compatible": (
                self.geometry_compatible
            ),
            "has_explicit_hydrogen": (
                self.has_explicit_hydrogen
            ),
            "metadata": dict(
                self.metadata
            ),
        }


# -----------------------------------------------------------------------------
# Donor-hydrogen-acceptor angle
# -----------------------------------------------------------------------------

def donor_hydrogen_acceptor_angle(
    donor: Coordinate,
    hydrogen: Coordinate,
    acceptor: Coordinate,
    *,
    unit: AngleUnit = "degrees",
    scene: bool = True,
    tolerance: float = DEFAULT_TOLERANCE,
) -> float:
    """
    Return the donor-hydrogen-acceptor angle.

    The angle is measured at the hydrogen atom:

    ``donor — hydrogen ··· acceptor``

    Parameters
    ----------
    donor : Coordinate
        Donor atom or coordinate-like object.
    hydrogen : Coordinate
        Donor-bound hydrogen atom or coordinate-like object.
    acceptor : Coordinate
        Acceptor atom or coordinate-like object.
    unit : {"degrees", "radians"}, optional
        Unit used for the returned angle.
    scene : bool, optional
        Whether scene-transformed coordinates should be preferred.
    tolerance : float, optional
        Minimum accepted length of the H-D and H-A vectors.

    Returns
    -------
    float
        D-H···A angle in the interval ``[0, 180]`` degrees or
        ``[0, π]`` radians.

    Raises
    ------
    TypeError
        If coordinates or parameters are invalid.
    ValueError
        If the hydrogen coincides with the donor or acceptor.

    Notes
    -----
    A more linear hydrogen bond has an angle closer to 180 degrees.

    Examples
    --------
    >>> donor_hydrogen_acceptor_angle(
    ...     [0, 0, 0],
    ...     [1, 0, 0],
    ...     [2, 0, 0],
    ... )
    180.0
    """

    return bond_angle(
        donor,
        hydrogen,
        acceptor,
        unit=unit,
        scene=scene,
        tolerance=tolerance,
    )


# -----------------------------------------------------------------------------
# Hydrogen-bond geometry calculation
# -----------------------------------------------------------------------------

def hydrogen_bond_geometry(
    donor: Coordinate,
    acceptor: Coordinate,
    hydrogen: Optional[
        Coordinate
    ] = None,
    *,
    scene: bool = True,
    tolerance: float = DEFAULT_TOLERANCE,
    maximum_donor_acceptor_distance: Optional[
        float
    ] = 3.5,
    maximum_hydrogen_acceptor_distance: Optional[
        float
    ] = 2.5,
    minimum_donor_hydrogen_acceptor_angle: Optional[
        float
    ] = 120.0,
    donor_hydrogen_distance_range: Optional[
        Tuple[
            float,
            float,
        ]
    ] = None,
    require_explicit_hydrogen: bool = False,
    metadata: Optional[
        Mapping[str, Any]
    ] = None,
) -> HydrogenBondGeometry:
    """
    Calculate the geometry of a potential hydrogen bond.

    Parameters
    ----------
    donor : Coordinate
        Donor atom.
    acceptor : Coordinate
        Acceptor atom.
    hydrogen : Coordinate, optional
        Hydrogen bonded to the donor. When omitted, only the donor-acceptor
        distance can be evaluated.
    scene : bool, optional
        Whether scene-transformed coordinates should be preferred.
    tolerance : float, optional
        Numerical tolerance used for degenerate-vector detection.
    maximum_donor_acceptor_distance : float, optional
        Maximum accepted D···A distance. Set to ``None`` to disable this
        criterion.
    maximum_hydrogen_acceptor_distance : float, optional
        Maximum accepted H···A distance. This criterion is only evaluated
        when a hydrogen is supplied. Set to ``None`` to disable it.
    minimum_donor_hydrogen_acceptor_angle : float, optional
        Minimum accepted D-H···A angle in degrees. This criterion is only
        evaluated when a hydrogen is supplied. Set to ``None`` to disable it.
    donor_hydrogen_distance_range : tuple of float, optional
        Optional inclusive ``(minimum, maximum)`` range for the D-H bond
        distance.
    require_explicit_hydrogen : bool, optional
        Whether absence of an explicit hydrogen should raise an error.
    metadata : Mapping[str, Any], optional
        Additional interaction metadata.

    Returns
    -------
    HydrogenBondGeometry
        Structured hydrogen-bond geometry.

    Raises
    ------
    TypeError
        If coordinates, cutoffs or metadata are invalid.
    ValueError
        If cutoffs are inconsistent, a required hydrogen is absent, or the
        geometry is degenerate.

    Notes
    -----
    With an explicit hydrogen, ``geometry_compatible`` requires all enabled
    D···A, H···A, D-H and angular criteria to pass.

    Without an explicit hydrogen, ``geometry_compatible`` reflects only the
    enabled D···A heavy-atom criterion. The result is marked as
    ``is_distance_only`` and ``angle_compatible`` is ``None``.

    Default cutoff values are configurable geometric defaults rather than
    universal definitions of hydrogen bonding.
    """

    numeric_tolerance = (
        _validate_angular_tolerance(
            tolerance,
            name="tolerance",
        )
    )

    if not isinstance(
        require_explicit_hydrogen,
        (
            bool,
            np.bool_,
        ),
    ):
        raise TypeError(
            "require_explicit_hydrogen must be boolean."
        )

    require_explicit_hydrogen = bool(
        require_explicit_hydrogen
    )

    donor_coordinate = get_atom_coordinate(
        donor,
        scene=scene,
        name="Donor atom",
        copy=True,
    )

    acceptor_coordinate = get_atom_coordinate(
        acceptor,
        scene=scene,
        name="Acceptor atom",
        copy=True,
    )

    donor_acceptor_distance = atom_distance(
        donor_coordinate,
        acceptor_coordinate,
        scene=False,
    )

    if (
        donor_acceptor_distance
        <= numeric_tolerance
    ):
        raise ValueError(
            "Donor and acceptor coordinates are "
            "coincident or near-coincident."
        )

    if (
        maximum_donor_acceptor_distance
        is None
    ):
        maximum_da_distance = None
        donor_acceptor_compatible = True

    else:
        maximum_da_distance = (
            _validate_nonnegative_finite_value(
                maximum_donor_acceptor_distance,
                name=(
                    "maximum_donor_acceptor_distance"
                ),
            )
        )

        donor_acceptor_compatible = (
            donor_acceptor_distance
            <= maximum_da_distance
        )

    if (
        maximum_hydrogen_acceptor_distance
        is None
    ):
        maximum_ha_distance = None

    else:
        maximum_ha_distance = (
            _validate_nonnegative_finite_value(
                maximum_hydrogen_acceptor_distance,
                name=(
                    "maximum_hydrogen_acceptor_distance"
                ),
            )
        )

    if (
        minimum_donor_hydrogen_acceptor_angle
        is None
    ):
        minimum_dha_angle = None

    else:
        minimum_dha_angle = (
            _validate_nonnegative_finite_value(
                minimum_donor_hydrogen_acceptor_angle,
                name=(
                    "minimum_donor_hydrogen_acceptor_angle"
                ),
            )
        )

        if minimum_dha_angle > 180.0:
            raise ValueError(
                "minimum_donor_hydrogen_acceptor_angle "
                "cannot exceed 180 degrees."
            )

    if donor_hydrogen_distance_range is None:
        minimum_dh_distance = None
        maximum_dh_distance = None

    else:
        if (
            not isinstance(
                donor_hydrogen_distance_range,
                (
                    tuple,
                    list,
                ),
            )
            or len(
                donor_hydrogen_distance_range
            ) != 2
        ):
            raise TypeError(
                "donor_hydrogen_distance_range must "
                "be a two-value sequence."
            )

        minimum_dh_distance = (
            _validate_nonnegative_finite_value(
                donor_hydrogen_distance_range[0],
                name=(
                    "minimum donor-hydrogen distance"
                ),
            )
        )

        maximum_dh_distance = (
            _validate_nonnegative_finite_value(
                donor_hydrogen_distance_range[1],
                name=(
                    "maximum donor-hydrogen distance"
                ),
            )
        )

        if (
            minimum_dh_distance
            > maximum_dh_distance
        ):
            raise ValueError(
                "The minimum donor-hydrogen distance "
                "cannot exceed the maximum distance."
            )

    interaction_metadata: Dict[
        str,
        Any,
    ] = {}

    if metadata is not None:
        if not isinstance(
            metadata,
            Mapping,
        ):
            raise TypeError(
                "metadata must be a mapping or None."
            )

        interaction_metadata.update(
            metadata
        )

    interaction_metadata.setdefault(
        "maximum_donor_acceptor_distance",
        maximum_da_distance,
    )

    interaction_metadata.setdefault(
        "maximum_hydrogen_acceptor_distance",
        maximum_ha_distance,
    )

    interaction_metadata.setdefault(
        "minimum_donor_hydrogen_acceptor_angle",
        minimum_dha_angle,
    )

    interaction_metadata.setdefault(
        "donor_hydrogen_distance_range",
        (
            None
            if minimum_dh_distance is None
            else [
                minimum_dh_distance,
                maximum_dh_distance,
            ]
        ),
    )

    if hydrogen is None:
        if require_explicit_hydrogen:
            raise ValueError(
                "An explicit hydrogen coordinate is "
                "required for this calculation."
            )

        interaction_metadata.setdefault(
            "evaluation_mode",
            "heavy_atom_distance_only",
        )

        return HydrogenBondGeometry(
            donor_coordinate=donor_coordinate,
            hydrogen_coordinate=None,
            acceptor_coordinate=(
                acceptor_coordinate
            ),
            donor_acceptor_distance=(
                donor_acceptor_distance
            ),
            hydrogen_acceptor_distance=None,
            donor_hydrogen_distance=None,
            donor_hydrogen_acceptor_angle=None,
            distance_compatible=(
                donor_acceptor_compatible
            ),
            angle_compatible=None,
            geometry_compatible=(
                donor_acceptor_compatible
            ),
            has_explicit_hydrogen=False,
            metadata=interaction_metadata,
        )

    hydrogen_coordinate = get_atom_coordinate(
        hydrogen,
        scene=scene,
        name="Hydrogen atom",
        copy=True,
    )

    donor_hydrogen_distance = atom_distance(
        donor_coordinate,
        hydrogen_coordinate,
        scene=False,
    )

    hydrogen_acceptor_distance = atom_distance(
        hydrogen_coordinate,
        acceptor_coordinate,
        scene=False,
    )

    if (
        donor_hydrogen_distance
        <= numeric_tolerance
    ):
        raise ValueError(
            "Donor and hydrogen coordinates are "
            "coincident or near-coincident."
        )

    if (
        hydrogen_acceptor_distance
        <= numeric_tolerance
    ):
        raise ValueError(
            "Hydrogen and acceptor coordinates are "
            "coincident or near-coincident."
        )

    dha_angle = (
        donor_hydrogen_acceptor_angle(
            donor_coordinate,
            hydrogen_coordinate,
            acceptor_coordinate,
            unit="degrees",
            scene=False,
            tolerance=numeric_tolerance,
        )
    )

    if maximum_ha_distance is None:
        hydrogen_acceptor_compatible = True

    else:
        hydrogen_acceptor_compatible = (
            hydrogen_acceptor_distance
            <= maximum_ha_distance
        )

    if minimum_dh_distance is None:
        donor_hydrogen_compatible = True

    else:
        donor_hydrogen_compatible = (
            minimum_dh_distance
            <= donor_hydrogen_distance
            <= maximum_dh_distance
        )

    if minimum_dha_angle is None:
        angle_compatible = True

    else:
        angle_compatible = (
            dha_angle
            >= minimum_dha_angle
        )

    distance_compatible = all(
        (
            donor_acceptor_compatible,
            hydrogen_acceptor_compatible,
            donor_hydrogen_compatible,
        )
    )

    geometry_compatible = (
        distance_compatible
        and angle_compatible
    )

    interaction_metadata.setdefault(
        "evaluation_mode",
        "explicit_hydrogen",
    )

    interaction_metadata.setdefault(
        "donor_acceptor_compatible",
        bool(
            donor_acceptor_compatible
        ),
    )

    interaction_metadata.setdefault(
        "hydrogen_acceptor_compatible",
        bool(
            hydrogen_acceptor_compatible
        ),
    )

    interaction_metadata.setdefault(
        "donor_hydrogen_compatible",
        bool(
            donor_hydrogen_compatible
        ),
    )

    return HydrogenBondGeometry(
        donor_coordinate=donor_coordinate,
        hydrogen_coordinate=(
            hydrogen_coordinate
        ),
        acceptor_coordinate=(
            acceptor_coordinate
        ),
        donor_acceptor_distance=(
            donor_acceptor_distance
        ),
        hydrogen_acceptor_distance=(
            hydrogen_acceptor_distance
        ),
        donor_hydrogen_distance=(
            donor_hydrogen_distance
        ),
        donor_hydrogen_acceptor_angle=(
            dha_angle
        ),
        distance_compatible=(
            distance_compatible
        ),
        angle_compatible=(
            angle_compatible
        ),
        geometry_compatible=(
            geometry_compatible
        ),
        has_explicit_hydrogen=True,
        metadata=interaction_metadata,
    )


# -----------------------------------------------------------------------------
# Public module interface
# -----------------------------------------------------------------------------

_SECTION_10_PUBLIC_NAMES = [
    "HydrogenBondGeometry",
    "hydrogen_bond_geometry",
    "donor_hydrogen_acceptor_angle",
]

_extend_public_names(_SECTION_10_PUBLIC_NAMES)


# =============================================================================
# End of Section 10
# =============================================================================



# =============================================================================
# Section 11 — Molecular Contacts
# =============================================================================


# -----------------------------------------------------------------------------
# Contact geometry representation
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class ContactGeometry:
    """
    Represent the geometric relationship between two atoms or points.

    Parameters
    ----------
    atom_1 : Any
        First original atom or coordinate-like object.
    atom_2 : Any
        Second original atom or coordinate-like object.
    coordinate_1 : Coordinate
        Coordinate of the first atom.
    coordinate_2 : Coordinate
        Coordinate of the second atom.
    distance : float
        Euclidean distance between the two coordinates.
    cutoff : float, optional
        Maximum distance used to classify the pair as a contact.
    contact_compatible : bool or None, optional
        Whether the distance satisfies ``cutoff``. This must be ``None`` when
        no cutoff is supplied.
    index_1 : int, optional
        Index of the first atom in its original collection.
    index_2 : int, optional
        Index of the second atom in its original collection.
    metadata : Mapping[str, Any], optional
        Additional contact information.

    Attributes
    ----------
    atom_1 : Any
        Original first object.
    atom_2 : Any
        Original second object.
    coordinate_1 : numpy.ndarray
        Read-only first coordinate.
    coordinate_2 : numpy.ndarray
        Read-only second coordinate.
    distance : float
        Euclidean separation.
    cutoff : float or None
        Contact cutoff.
    contact_compatible : bool or None
        Contact classification based on the cutoff.

    Notes
    -----
    This class describes geometric proximity only. A short distance does not
    by itself establish a favorable molecular interaction.
    """

    atom_1: Any
    atom_2: Any

    coordinate_1: Coordinate
    coordinate_2: Coordinate

    distance: float

    cutoff: Optional[float] = None
    contact_compatible: Optional[bool] = None

    index_1: Optional[int] = None
    index_2: Optional[int] = None

    metadata: GeometryMetadata = field(
        default_factory=dict
    )

    def __post_init__(
        self,
    ) -> None:
        """
        Validate and normalize contact attributes.
        """

        first_coordinate = as_coordinate(
            self.coordinate_1,
            scene=False,
            name="First contact coordinate",
            copy=True,
        )

        second_coordinate = as_coordinate(
            self.coordinate_2,
            scene=False,
            name="Second contact coordinate",
            copy=True,
        )

        first_coordinate.setflags(
            write=False
        )

        second_coordinate.setflags(
            write=False
        )

        object.__setattr__(
            self,
            "coordinate_1",
            first_coordinate,
        )

        object.__setattr__(
            self,
            "coordinate_2",
            second_coordinate,
        )

        distance_value = (
            _validate_nonnegative_finite_value(
                self.distance,
                name="distance",
            )
        )

        calculated_distance = float(
            distance(
                first_coordinate,
                second_coordinate,
            )
        )

        consistency_tolerance = max(
            DEFAULT_DISTANCE_TOLERANCE,
            calculated_distance
            * DEFAULT_DISTANCE_TOLERANCE,
        )

        if not math.isclose(
            distance_value,
            calculated_distance,
            rel_tol=DEFAULT_DISTANCE_TOLERANCE,
            abs_tol=consistency_tolerance,
        ):
            raise ValueError(
                "distance is inconsistent with "
                "coordinate_1 and coordinate_2."
            )

        object.__setattr__(
            self,
            "distance",
            distance_value,
        )

        if self.cutoff is None:
            cutoff_value = None

            if self.contact_compatible is not None:
                raise ValueError(
                    "contact_compatible must be None "
                    "when cutoff is not supplied."
                )

            contact_compatible_value = None

        else:
            cutoff_value = (
                _validate_nonnegative_finite_value(
                    self.cutoff,
                    name="cutoff",
                )
            )

            if not isinstance(
                self.contact_compatible,
                (
                    bool,
                    np.bool_,
                ),
            ):
                raise TypeError(
                    "contact_compatible must be boolean "
                    "when cutoff is supplied."
                )

            contact_compatible_value = bool(
                self.contact_compatible
            )

            expected_compatibility = (
                distance_value
                <= cutoff_value
            )

            if (
                contact_compatible_value
                != expected_compatibility
            ):
                raise ValueError(
                    "contact_compatible is inconsistent "
                    "with distance and cutoff."
                )

        object.__setattr__(
            self,
            "cutoff",
            cutoff_value,
        )

        object.__setattr__(
            self,
            "contact_compatible",
            contact_compatible_value,
        )

        for attribute_name in (
            "index_1",
            "index_2",
        ):
            index_value = getattr(
                self,
                attribute_name,
            )

            if index_value is None:
                continue

            if isinstance(
                index_value,
                (
                    bool,
                    np.bool_,
                ),
            ) or not isinstance(
                index_value,
                (
                    int,
                    np.integer,
                ),
            ):
                raise TypeError(
                    f"{attribute_name} must be an "
                    "integer or None."
                )

            index_value = int(
                index_value
            )

            if index_value < 0:
                raise ValueError(
                    f"{attribute_name} cannot be negative."
                )

            object.__setattr__(
                self,
                attribute_name,
                index_value,
            )

        if self.metadata is None:
            metadata_value: Dict[str, Any] = {}

        elif isinstance(
            self.metadata,
            Mapping,
        ):
            metadata_value = dict(
                self.metadata
            )

        else:
            raise TypeError(
                "metadata must be a mapping or None."
            )

        metadata_value.setdefault(
            "geometry_type",
            "molecular_contact",
        )

        object.__setattr__(
            self,
            "metadata",
            metadata_value,
        )

    @property
    def squared_distance(
        self,
    ) -> float:
        """
        Return the squared contact distance.

        Returns
        -------
        float
            Squared Euclidean distance.
        """

        return float(
            self.distance
            * self.distance
        )

    @property
    def displacement(
        self,
    ) -> Vector3D:
        """
        Return the vector from the first coordinate to the second.

        Returns
        -------
        numpy.ndarray
            Vector ``coordinate_2 - coordinate_1``.
        """

        return vector_between(
            self.coordinate_1,
            self.coordinate_2,
            scene=False,
            copy=True,
        )

    @property
    def direction(
        self,
    ) -> Optional[Vector3D]:
        """
        Return the unit vector from the first atom to the second.

        Returns
        -------
        numpy.ndarray or None
            Unit direction vector, or ``None`` when both coordinates
            coincide.
        """

        if (
            self.distance
            <= DEFAULT_TOLERANCE
        ):
            return None

        return unit_vector(
            self.displacement,
            scene=False,
            tolerance=DEFAULT_TOLERANCE,
            copy=True,
        )

    @property
    def midpoint(
        self,
    ) -> Vector3D:
        """
        Return the midpoint between the two coordinates.

        Returns
        -------
        numpy.ndarray
            Contact midpoint.
        """

        return (
            (
                self.coordinate_1
                + self.coordinate_2
            )
            / 2.0
        ).astype(
            np.float64,
            copy=True,
        )

    @property
    def margin_to_cutoff(
        self,
    ) -> Optional[float]:
        """
        Return the distance margin relative to the contact cutoff.

        Returns
        -------
        float or None
            ``cutoff - distance`` when a cutoff is available.

            Positive values indicate that the pair lies inside the cutoff.
            Negative values indicate that the pair lies outside it.
        """

        if self.cutoff is None:
            return None

        return float(
            self.cutoff
            - self.distance
        )

    @property
    def is_contact(
        self,
    ) -> Optional[bool]:
        """
        Return the cutoff-based contact classification.

        Returns
        -------
        bool or None
            Contact compatibility or ``None`` when no cutoff was evaluated.
        """

        return self.contact_compatible

    @property
    def has_collection_indices(
        self,
    ) -> bool:
        """
        Return whether indices from both collections are available.

        Returns
        -------
        bool
            ``True`` when both indices are stored.
        """

        return (
            self.index_1 is not None
            and self.index_2 is not None
        )

    def to_dict(
        self,
        *,
        include_coordinates: bool = True,
        include_atoms: bool = False,
    ) -> Dict[str, Any]:
        """
        Convert the contact geometry to a serializable dictionary.

        Parameters
        ----------
        include_coordinates : bool, optional
            Whether coordinates should be included.
        include_atoms : bool, optional
            Whether string representations of the original objects should be
            included.

        Returns
        -------
        dict
            Serialized contact geometry.
        """

        direction = self.direction

        result: Dict[str, Any] = {
            "distance": self.distance,
            "squared_distance": (
                self.squared_distance
            ),
            "cutoff": self.cutoff,
            "contact_compatible": (
                self.contact_compatible
            ),
            "margin_to_cutoff": (
                self.margin_to_cutoff
            ),
            "index_1": self.index_1,
            "index_2": self.index_2,
            "midpoint": self.midpoint.tolist(),
            "direction": (
                None
                if direction is None
                else direction.tolist()
            ),
            "metadata": dict(
                self.metadata
            ),
        }

        if include_coordinates:
            result["coordinate_1"] = (
                self.coordinate_1.tolist()
            )

            result["coordinate_2"] = (
                self.coordinate_2.tolist()
            )

        if include_atoms:
            result["atom_1"] = repr(
                self.atom_1
            )

            result["atom_2"] = repr(
                self.atom_2
            )

        return result


# -----------------------------------------------------------------------------
# Internal contact helpers
# -----------------------------------------------------------------------------

def _validate_optional_contact_cutoff(
    cutoff: Optional[float],
    *,
    name: str = "cutoff",
) -> Optional[float]:
    """
    Validate an optional non-negative distance cutoff.

    Parameters
    ----------
    cutoff : float or None
        Distance cutoff.
    name : str, optional
        Parameter name used in validation messages.

    Returns
    -------
    float or None
        Validated cutoff.
    """

    if cutoff is None:
        return None

    return _validate_nonnegative_finite_value(
        cutoff,
        name=name,
    )


def _validate_optional_collection_index(
    index: Optional[int],
    *,
    name: str,
) -> Optional[int]:
    """
    Validate an optional non-negative collection index.

    Parameters
    ----------
    index : int or None
        Index to validate.
    name : str
        Parameter name.

    Returns
    -------
    int or None
        Validated index.
    """

    if index is None:
        return None

    if isinstance(
        index,
        (
            bool,
            np.bool_,
        ),
    ) or not isinstance(
        index,
        (
            int,
            np.integer,
        ),
    ):
        raise TypeError(
            f"{name} must be an integer or None."
        )

    index_value = int(
        index
    )

    if index_value < 0:
        raise ValueError(
            f"{name} cannot be negative."
        )

    return index_value


def _materialize_contact_collection(
    collection: Any,
    *,
    name: str,
) -> List[Any]:
    """
    Convert an atom or coordinate collection to a concrete list.

    Parameters
    ----------
    collection : Any
        Iterable containing atoms or coordinates.
    name : str
        Collection name used in validation messages.

    Returns
    -------
    list
        Materialized collection preserving the original objects.

    Raises
    ------
    TypeError
        If the input is not a valid iterable collection.
    ValueError
        If the collection is empty.
    """

    if collection is None:
        raise TypeError(
            f"{name} cannot be None."
        )

    if isinstance(
        collection,
        (
            str,
            bytes,
            bytearray,
        ),
    ):
        raise TypeError(
            f"{name} must be an atom or coordinate "
            "collection, not text."
        )

    if isinstance(
        collection,
        np.ndarray,
    ):
        array = np.asarray(
            collection
        )

        if (
            array.ndim == 1
            and array.shape[0] == 3
        ):
            items = [
                array
            ]

        elif (
            array.ndim == 2
            and array.shape[1] == 3
        ):
            items = [
                array[index]
                for index in range(
                    array.shape[0]
                )
            ]

        else:
            try:
                items = list(
                    collection
                )

            except TypeError as error:
                raise TypeError(
                    f"{name} must be iterable."
                ) from error

    else:
        try:
            items = list(
                collection
            )

        except TypeError as error:
            raise TypeError(
                f"{name} must be iterable."
            ) from error

    if not items:
        raise ValueError(
            f"{name} cannot be empty."
        )

    return items


def _contact_metadata(
    metadata: Optional[
        Mapping[str, Any]
    ],
    **defaults: Any,
) -> Dict[str, Any]:
    """
    Construct contact metadata while preserving caller-supplied values.

    Parameters
    ----------
    metadata : Mapping[str, Any], optional
        User metadata.
    **defaults : Any
        Default values inserted only when absent.

    Returns
    -------
    dict
        Contact metadata.
    """

    if metadata is None:
        result: Dict[str, Any] = {}

    elif isinstance(
        metadata,
        Mapping,
    ):
        result = dict(
            metadata
        )

    else:
        raise TypeError(
            "metadata must be a mapping or None."
        )

    for key, value in defaults.items():
        result.setdefault(
            key,
            value,
        )

    return result


# -----------------------------------------------------------------------------
# Contact geometry
# -----------------------------------------------------------------------------

def contact_geometry(
    atom_1: Any,
    atom_2: Any,
    *,
    scene: bool = True,
    cutoff: Optional[float] = None,
    index_1: Optional[int] = None,
    index_2: Optional[int] = None,
    metadata: Optional[
        Mapping[str, Any]
    ] = None,
) -> ContactGeometry:
    """
    Calculate the geometric relationship between two atoms or points.

    Parameters
    ----------
    atom_1 : Any
        First atom or coordinate-like object.
    atom_2 : Any
        Second atom or coordinate-like object.
    scene : bool, optional
        Whether scene-transformed coordinates should be preferred.
    cutoff : float, optional
        Maximum distance used to classify the pair as a contact. When omitted,
        the distance is measured without assigning a contact classification.
    index_1 : int, optional
        Index of the first object in an original collection.
    index_2 : int, optional
        Index of the second object in an original collection.
    metadata : Mapping[str, Any], optional
        Additional contact metadata.

    Returns
    -------
    ContactGeometry
        Structured contact geometry.

    Notes
    -----
    ``contact_compatible`` indicates only whether the pair lies within the
    supplied geometric cutoff. Chemical compatibility must be assessed by a
    separate interaction-analysis layer.

    Examples
    --------
    Measure a distance without classification:

    >>> geometry = contact_geometry(
    ...     atom_a,
    ...     atom_b,
    ... )
    >>> geometry.distance
    3.24

    Apply a contact cutoff:

    >>> geometry = contact_geometry(
    ...     atom_a,
    ...     atom_b,
    ...     cutoff=4.0,
    ... )
    >>> geometry.is_contact
    True
    """

    cutoff_value = (
        _validate_optional_contact_cutoff(
            cutoff
        )
    )

    first_index = (
        _validate_optional_collection_index(
            index_1,
            name="index_1",
        )
    )

    second_index = (
        _validate_optional_collection_index(
            index_2,
            name="index_2",
        )
    )

    first_coordinate = get_atom_coordinate(
        atom_1,
        scene=scene,
        name="First contact atom",
        copy=True,
    )

    second_coordinate = get_atom_coordinate(
        atom_2,
        scene=scene,
        name="Second contact atom",
        copy=True,
    )

    distance_value = atom_distance(
        first_coordinate,
        second_coordinate,
        scene=False,
    )

    if cutoff_value is None:
        contact_compatible = None

    else:
        contact_compatible = (
            distance_value
            <= cutoff_value
        )

    contact_metadata = _contact_metadata(
        metadata,
        evaluation_mode="single_pair",
    )

    return ContactGeometry(
        atom_1=atom_1,
        atom_2=atom_2,
        coordinate_1=first_coordinate,
        coordinate_2=second_coordinate,
        distance=distance_value,
        cutoff=cutoff_value,
        contact_compatible=(
            contact_compatible
        ),
        index_1=first_index,
        index_2=second_index,
        metadata=contact_metadata,
    )


# -----------------------------------------------------------------------------
# Closest atoms between collections
# -----------------------------------------------------------------------------

def closest_atoms(
    atoms_1: Any,
    atoms_2: Any,
    *,
    scene: bool = True,
    cutoff: Optional[float] = None,
    exclude_identical_objects: bool = False,
    exclude_same_index: bool = False,
    metadata: Optional[
        Mapping[str, Any]
    ] = None,
    return_distance_matrix: bool = False,
) -> Union[
    ContactGeometry,
    Tuple[
        ContactGeometry,
        FloatArray,
    ],
]:
    """
    Return the closest atom pair between two collections.

    Parameters
    ----------
    atoms_1 : Any
        First atom or coordinate collection.
    atoms_2 : Any
        Second atom or coordinate collection.
    scene : bool, optional
        Whether scene-transformed coordinates should be preferred.
    cutoff : float, optional
        Maximum distance used to classify the closest pair as a contact.
    exclude_identical_objects : bool, optional
        Whether pairs containing the exact same Python object should be
        excluded. This is useful when both collections overlap.
    exclude_same_index : bool, optional
        Whether pairs having equal collection indices should be excluded.
        This is mainly useful when comparing a collection with itself.
    metadata : Mapping[str, Any], optional
        Additional contact metadata.
    return_distance_matrix : bool, optional
        Whether the complete pairwise distance matrix should also be returned.

    Returns
    -------
    ContactGeometry
        Closest atom-pair geometry.

    tuple
        ``(contact_geometry, distance_matrix)`` when
        ``return_distance_matrix=True``.

    Raises
    ------
    TypeError
        If a collection or parameter has an invalid type.
    ValueError
        If either collection is empty or every possible pair is excluded.

    Notes
    -----
    Pairwise distances are calculated using NumPy broadcasting. The original
    atom objects are preserved in the returned ``ContactGeometry``.

    When the same collection is supplied twice, set at least one exclusion
    option to prevent each atom from being selected as its own closest pair.

    Examples
    --------
    Find the closest receptor-ligand atom pair:

    >>> closest = closest_atoms(
    ...     receptor_atoms,
    ...     ligand_atoms,
    ...     cutoff=4.0,
    ... )

    Compare one collection with itself:

    >>> closest = closest_atoms(
    ...     atoms,
    ...     atoms,
    ...     exclude_identical_objects=True,
    ...     exclude_same_index=True,
    ... )
    """

    if not isinstance(
        exclude_identical_objects,
        (
            bool,
            np.bool_,
        ),
    ):
        raise TypeError(
            "exclude_identical_objects must be boolean."
        )

    if not isinstance(
        exclude_same_index,
        (
            bool,
            np.bool_,
        ),
    ):
        raise TypeError(
            "exclude_same_index must be boolean."
        )

    if not isinstance(
        return_distance_matrix,
        (
            bool,
            np.bool_,
        ),
    ):
        raise TypeError(
            "return_distance_matrix must be boolean."
        )

    cutoff_value = (
        _validate_optional_contact_cutoff(
            cutoff
        )
    )

    first_items = (
        _materialize_contact_collection(
            atoms_1,
            name="First atom collection",
        )
    )

    second_items = (
        _materialize_contact_collection(
            atoms_2,
            name="Second atom collection",
        )
    )

    first_coordinates = get_coordinates(
        first_items,
        scene=scene,
        name="First atom collection",
        minimum_rows=1,
        allow_empty=False,
        ignore_none=False,
        require_finite=True,
        copy=False,
    )

    second_coordinates = get_coordinates(
        second_items,
        scene=scene,
        name="Second atom collection",
        minimum_rows=1,
        allow_empty=False,
        ignore_none=False,
        require_finite=True,
        copy=False,
    )

    pairwise_squared_distances = distance_matrix(
        first_coordinates,
        second_coordinates,
        scene=False,
        squared=True,
        minimum_rows=1,
        allow_empty=False,
        copy=False,
    )

    valid_pairs = np.ones(
        pairwise_squared_distances.shape,
        dtype=bool,
    )

    if exclude_same_index:
        shared_count = min(
            len(
                first_items
            ),
            len(
                second_items
            ),
        )

        diagonal_indices = np.arange(
            shared_count
        )

        valid_pairs[
            diagonal_indices,
            diagonal_indices,
        ] = False

    if exclude_identical_objects:
        second_indices_by_identity: Dict[
            int,
            List[int],
        ] = {}

        for second_index, second_atom in enumerate(
            second_items
        ):
            second_indices_by_identity.setdefault(
                id(second_atom),
                [],
            ).append(second_index)

        for first_index, first_atom in enumerate(
            first_items
        ):
            for second_index in second_indices_by_identity.get(
                id(first_atom),
                (),
            ):
                if first_atom is second_items[second_index]:
                    valid_pairs[
                        first_index,
                        second_index,
                    ] = False

    if not np.any(
        valid_pairs
    ):
        raise ValueError(
            "No valid atom pairs remain after applying "
            "the exclusion criteria."
        )

    searchable_squared_distances = np.where(
        valid_pairs,
        pairwise_squared_distances,
        np.inf,
    )

    flat_index = int(
        np.argmin(
            searchable_squared_distances
        )
    )

    first_index, second_index = (
        np.unravel_index(
            flat_index,
            searchable_squared_distances.shape,
        )
    )

    first_index = int(
        first_index
    )

    second_index = int(
        second_index
    )

    minimum_squared_distance = float(
        searchable_squared_distances[
            first_index,
            second_index,
        ]
    )

    if not math.isfinite(
        minimum_squared_distance
    ):
        raise ValueError(
            "No finite atom pair distance could be found."
        )

    minimum_distance_value = float(
        math.sqrt(
            minimum_squared_distance
        )
    )

    if cutoff_value is None:
        contact_compatible = None

    else:
        contact_compatible = (
            minimum_distance_value
            <= cutoff_value
        )

    contact_metadata = _contact_metadata(
        metadata,
        evaluation_mode="closest_pair",
        first_collection_size=len(
            first_items
        ),
        second_collection_size=len(
            second_items
        ),
        exclude_identical_objects=bool(
            exclude_identical_objects
        ),
        exclude_same_index=bool(
            exclude_same_index
        ),
    )

    result = ContactGeometry(
        atom_1=first_items[
            first_index
        ],
        atom_2=second_items[
            second_index
        ],
        coordinate_1=first_coordinates[
            first_index
        ],
        coordinate_2=second_coordinates[
            second_index
        ],
        distance=minimum_distance_value,
        cutoff=cutoff_value,
        contact_compatible=(
            contact_compatible
        ),
        index_1=first_index,
        index_2=second_index,
        metadata=contact_metadata,
    )

    if return_distance_matrix:
        return (
            result,
            np.sqrt(
                pairwise_squared_distances
            ),
        )

    return result


# -----------------------------------------------------------------------------
# Public module interface
# -----------------------------------------------------------------------------

_SECTION_11_PUBLIC_NAMES = [
    "ContactGeometry",
    "contact_geometry",
    "closest_atoms",
]

_extend_public_names(_SECTION_11_PUBLIC_NAMES)


# =============================================================================
# End of Section 11
# =============================================================================


# =============================================================================
# Section 12 — RMSD and Alignment
# =============================================================================


# -----------------------------------------------------------------------------
# Alignment result representation
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class AlignmentResult:
    """
    Represent the result of a rigid-body coordinate alignment.

    The transformation follows the row-vector convention:

    ``aligned = mobile @ rotation + translation``

    Parameters
    ----------
    reference_coordinates : CoordinateCollection
        Fixed reference coordinates.
    mobile_coordinates : CoordinateCollection
        Original coordinates transformed during alignment.
    aligned_coordinates : CoordinateCollection
        Mobile coordinates after rotation and translation.
    rotation : array-like
        Rotation matrix with shape ``(3, 3)``.
    translation : Coordinate
        Translation vector applied after rotation.
    reference_centroid : Coordinate
        Centroid of the reference coordinates.
    mobile_centroid : Coordinate
        Centroid of the original mobile coordinates.
    initial_rmsd : float
        RMSD before alignment.
    final_rmsd : float
        RMSD after alignment.
    point_count : int
        Number of corresponding coordinate pairs.
    weights : array-like, optional
        Normalized alignment weights.
    singular_values : array-like, optional
        Singular values obtained from the covariance matrix.
    determinant : float, optional
        Determinant of the final rotation matrix.
    reflection_corrected : bool, optional
        Whether an improper rotation was corrected.
    metadata : Mapping[str, Any], optional
        Additional alignment information.

    Attributes
    ----------
    rotation : numpy.ndarray
        Read-only ``(3, 3)`` rotation matrix.
    translation : numpy.ndarray
        Read-only translation vector.
    initial_rmsd : float
        Unaligned RMSD.
    final_rmsd : float
        Aligned RMSD.

    Notes
    -----
    Coordinate correspondence is positional. Row ``i`` of the mobile
    collection is aligned with row ``i`` of the reference collection.
    """

    reference_coordinates: CoordinateCollection
    mobile_coordinates: CoordinateCollection
    aligned_coordinates: CoordinateCollection

    rotation: ArrayLike
    translation: Coordinate

    reference_centroid: Coordinate
    mobile_centroid: Coordinate

    initial_rmsd: float
    final_rmsd: float

    point_count: int

    weights: Optional[ArrayLike] = None
    singular_values: Optional[ArrayLike] = None
    determinant: Optional[float] = None

    reflection_corrected: bool = False

    metadata: GeometryMetadata = field(
        default_factory=dict
    )

    def __post_init__(
        self,
    ) -> None:
        """
        Validate and normalize alignment attributes.
        """

        reference_matrix = as_coordinate_matrix(
            self.reference_coordinates,
            scene=False,
            name="Reference coordinates",
            minimum_rows=1,
            allow_empty=False,
            require_finite=True,
            copy=True,
        )

        mobile_matrix = as_coordinate_matrix(
            self.mobile_coordinates,
            scene=False,
            name="Mobile coordinates",
            minimum_rows=1,
            allow_empty=False,
            require_finite=True,
            copy=True,
        )

        aligned_matrix = as_coordinate_matrix(
            self.aligned_coordinates,
            scene=False,
            name="Aligned coordinates",
            minimum_rows=1,
            allow_empty=False,
            require_finite=True,
            copy=True,
        )

        if (
            reference_matrix.shape
            != mobile_matrix.shape
        ):
            raise ValueError(
                "reference_coordinates and "
                "mobile_coordinates must have "
                "the same shape."
            )

        if (
            reference_matrix.shape
            != aligned_matrix.shape
        ):
            raise ValueError(
                "aligned_coordinates must have the "
                "same shape as reference_coordinates."
            )

        calculated_point_count = int(
            reference_matrix.shape[0]
        )

        if isinstance(
            self.point_count,
            (
                bool,
                np.bool_,
            ),
        ) or not isinstance(
            self.point_count,
            (
                int,
                np.integer,
            ),
        ):
            raise TypeError(
                "point_count must be an integer."
            )

        point_count_value = int(
            self.point_count
        )

        if (
            point_count_value
            != calculated_point_count
        ):
            raise ValueError(
                "point_count does not match the "
                "number of coordinate rows."
            )

        rotation_matrix = _validate_rotation_matrix(
            self.rotation,
            name="rotation",
            require_proper=False,
            copy=True,
        )

        translation_vector = as_coordinate(
            self.translation,
            scene=False,
            name="Translation vector",
            copy=True,
        )

        reference_centroid = as_coordinate(
            self.reference_centroid,
            scene=False,
            name="Reference centroid",
            copy=True,
        )

        mobile_centroid = as_coordinate(
            self.mobile_centroid,
            scene=False,
            name="Mobile centroid",
            copy=True,
        )

        initial_rmsd_value = (
            _validate_nonnegative_finite_value(
                self.initial_rmsd,
                name="initial_rmsd",
            )
        )

        final_rmsd_value = (
            _validate_nonnegative_finite_value(
                self.final_rmsd,
                name="final_rmsd",
            )
        )

        if not isinstance(
            self.reflection_corrected,
            (
                bool,
                np.bool_,
            ),
        ):
            raise TypeError(
                "reflection_corrected must be boolean."
            )

        reflection_corrected_value = bool(
            self.reflection_corrected
        )

        if self.weights is None:
            normalized_weights = None

        else:
            normalized_weights = (
                _validate_alignment_weights(
                    self.weights,
                    point_count=point_count_value,
                    normalize=True,
                    name="weights",
                )
            )

            normalized_weights.setflags(
                write=False
            )

        if self.singular_values is None:
            singular_values_array = None

        else:
            try:
                singular_values_array = np.asarray(
                    self.singular_values,
                    dtype=np.float64,
                )

            except (
                TypeError,
                ValueError,
                OverflowError,
            ) as error:
                raise TypeError(
                    "singular_values must be "
                    "convertible to a numeric array."
                ) from error

            singular_values_array = np.ravel(
                singular_values_array
            )

            if singular_values_array.size == 0:
                raise ValueError(
                    "singular_values cannot be empty."
                )

            if not np.all(
                np.isfinite(
                    singular_values_array
                )
            ):
                raise ValueError(
                    "singular_values contains NaN "
                    "or infinite values."
                )

            if np.any(
                singular_values_array < 0.0
            ):
                raise ValueError(
                    "singular_values cannot contain "
                    "negative values."
                )

            singular_values_array = np.array(
                singular_values_array,
                dtype=np.float64,
                copy=True,
            )

            singular_values_array.setflags(
                write=False
            )

        calculated_determinant = float(
            np.linalg.det(
                rotation_matrix
            )
        )

        if self.determinant is None:
            determinant_value = (
                calculated_determinant
            )

        else:
            if isinstance(
                self.determinant,
                (
                    bool,
                    np.bool_,
                ),
            ):
                raise TypeError(
                    "determinant must be numeric."
                )

            try:
                determinant_value = float(
                    self.determinant
                )

            except (
                TypeError,
                ValueError,
                OverflowError,
            ) as error:
                raise TypeError(
                    "determinant must be numeric."
                ) from error

            if not math.isfinite(
                determinant_value
            ):
                raise ValueError(
                    "determinant must be finite."
                )

            if not math.isclose(
                determinant_value,
                calculated_determinant,
                rel_tol=DEFAULT_DISTANCE_TOLERANCE,
                abs_tol=DEFAULT_DISTANCE_TOLERANCE,
            ):
                raise ValueError(
                    "determinant is inconsistent "
                    "with the rotation matrix."
                )

        reconstructed_coordinates = (
            mobile_matrix
            @ rotation_matrix
            + translation_vector
        )

        if not np.allclose(
            reconstructed_coordinates,
            aligned_matrix,
            rtol=DEFAULT_DISTANCE_TOLERANCE,
            atol=DEFAULT_DISTANCE_TOLERANCE,
        ):
            raise ValueError(
                "aligned_coordinates are inconsistent "
                "with mobile_coordinates, rotation and "
                "translation."
            )

        reference_matrix.setflags(
            write=False
        )

        mobile_matrix.setflags(
            write=False
        )

        aligned_matrix.setflags(
            write=False
        )

        rotation_matrix.setflags(
            write=False
        )

        translation_vector.setflags(
            write=False
        )

        reference_centroid.setflags(
            write=False
        )

        mobile_centroid.setflags(
            write=False
        )

        object.__setattr__(
            self,
            "reference_coordinates",
            reference_matrix,
        )

        object.__setattr__(
            self,
            "mobile_coordinates",
            mobile_matrix,
        )

        object.__setattr__(
            self,
            "aligned_coordinates",
            aligned_matrix,
        )

        object.__setattr__(
            self,
            "rotation",
            rotation_matrix,
        )

        object.__setattr__(
            self,
            "translation",
            translation_vector,
        )

        object.__setattr__(
            self,
            "reference_centroid",
            reference_centroid,
        )

        object.__setattr__(
            self,
            "mobile_centroid",
            mobile_centroid,
        )

        object.__setattr__(
            self,
            "initial_rmsd",
            initial_rmsd_value,
        )

        object.__setattr__(
            self,
            "final_rmsd",
            final_rmsd_value,
        )

        object.__setattr__(
            self,
            "point_count",
            point_count_value,
        )

        object.__setattr__(
            self,
            "weights",
            normalized_weights,
        )

        object.__setattr__(
            self,
            "singular_values",
            singular_values_array,
        )

        object.__setattr__(
            self,
            "determinant",
            determinant_value,
        )

        object.__setattr__(
            self,
            "reflection_corrected",
            reflection_corrected_value,
        )

        if self.metadata is None:
            metadata_value: Dict[str, Any] = {}

        elif isinstance(
            self.metadata,
            Mapping,
        ):
            metadata_value = dict(
                self.metadata
            )

        else:
            raise TypeError(
                "metadata must be a mapping or None."
            )

        metadata_value.setdefault(
            "geometry_type",
            "rigid_alignment",
        )

        metadata_value.setdefault(
            "alignment_method",
            "kabsch",
        )

        object.__setattr__(
            self,
            "metadata",
            metadata_value,
        )

    @property
    def rmsd_improvement(
        self,
    ) -> float:
        """
        Return the absolute RMSD reduction after alignment.

        Returns
        -------
        float
            ``initial_rmsd - final_rmsd``.
        """

        return float(
            self.initial_rmsd
            - self.final_rmsd
        )

    @property
    def relative_rmsd_improvement(
        self,
    ) -> float:
        """
        Return the relative RMSD reduction.

        Returns
        -------
        float
            Fractional RMSD reduction. Returns ``0.0`` when the initial RMSD
            is numerically zero.
        """

        if (
            self.initial_rmsd
            <= DEFAULT_TOLERANCE
        ):
            return 0.0

        return float(
            self.rmsd_improvement
            / self.initial_rmsd
        )

    @property
    def is_proper_rotation(
        self,
    ) -> bool:
        """
        Return whether the transformation uses a proper rotation.

        Returns
        -------
        bool
            ``True`` when the determinant is approximately ``+1``.
        """

        return math.isclose(
            self.determinant,
            1.0,
            rel_tol=DEFAULT_DISTANCE_TOLERANCE,
            abs_tol=DEFAULT_DISTANCE_TOLERANCE,
        )

    @property
    def transformation_matrix(
        self,
    ) -> FloatArray:
        """
        Return a homogeneous transformation matrix.

        Returns
        -------
        numpy.ndarray
            Matrix with shape ``(4, 4)``.

        Notes
        -----
        This matrix follows the row-vector convention used by this module:

        ``[x, y, z, 1] @ transformation_matrix``

        The translation is therefore stored in the final matrix row.
        """

        transformation = np.eye(
            4,
            dtype=np.float64,
        )

        transformation[
            :3,
            :3,
        ] = self.rotation

        transformation[
            3,
            :3,
        ] = self.translation

        return transformation

    def transform(
        self,
        coordinates: CoordinateCollection,
        *,
        scene: bool = True,
        copy: bool = False,
    ) -> FloatArray:
        """
        Apply the fitted transformation to another coordinate collection.

        Parameters
        ----------
        coordinates : CoordinateCollection
            Coordinates to transform.
        scene : bool, optional
            Whether scene coordinates should be preferred.
        copy : bool, optional
            Whether the returned matrix must be copied.

        Returns
        -------
        numpy.ndarray
            Transformed coordinates with shape ``(N, 3)``.
        """

        coordinate_matrix = get_coordinates(
            coordinates,
            scene=scene,
            name="Coordinates to transform",
            minimum_rows=1,
            allow_empty=False,
            require_finite=True,
            copy=False,
        )

        transformed = (
            coordinate_matrix
            @ self.rotation
            + self.translation
        ).astype(
            np.float64,
            copy=False,
        )

        if copy:
            return np.array(
                transformed,
                dtype=np.float64,
                copy=True,
            )

        return transformed

    def inverse_transform(
        self,
        coordinates: CoordinateCollection,
        *,
        scene: bool = True,
        copy: bool = False,
    ) -> FloatArray:
        """
        Apply the inverse alignment transformation.

        Parameters
        ----------
        coordinates : CoordinateCollection
            Coordinates in the aligned reference frame.
        scene : bool, optional
            Whether scene coordinates should be preferred.
        copy : bool, optional
            Whether the returned matrix must be copied.

        Returns
        -------
        numpy.ndarray
            Coordinates transformed back to the mobile frame.
        """

        coordinate_matrix = get_coordinates(
            coordinates,
            scene=scene,
            name="Coordinates to inverse-transform",
            minimum_rows=1,
            allow_empty=False,
            require_finite=True,
            copy=False,
        )

        inverse_rotation = (
            self.rotation.T
        )

        restored = (
            coordinate_matrix
            - self.translation
        ) @ inverse_rotation

        restored = restored.astype(
            np.float64,
            copy=False,
        )

        if copy:
            return np.array(
                restored,
                dtype=np.float64,
                copy=True,
            )

        return restored

    def to_dict(
        self,
        *,
        include_coordinates: bool = False,
    ) -> Dict[str, Any]:
        """
        Convert the alignment result to a serializable dictionary.

        Parameters
        ----------
        include_coordinates : bool, optional
            Whether coordinate matrices should be included.

        Returns
        -------
        dict
            Serialized alignment result.
        """

        result: Dict[str, Any] = {
            "rotation": self.rotation.tolist(),
            "translation": (
                self.translation.tolist()
            ),
            "reference_centroid": (
                self.reference_centroid.tolist()
            ),
            "mobile_centroid": (
                self.mobile_centroid.tolist()
            ),
            "initial_rmsd": (
                self.initial_rmsd
            ),
            "final_rmsd": (
                self.final_rmsd
            ),
            "rmsd_improvement": (
                self.rmsd_improvement
            ),
            "relative_rmsd_improvement": (
                self.relative_rmsd_improvement
            ),
            "point_count": self.point_count,
            "weights": (
                None
                if self.weights is None
                else self.weights.tolist()
            ),
            "singular_values": (
                None
                if self.singular_values is None
                else self.singular_values.tolist()
            ),
            "determinant": self.determinant,
            "is_proper_rotation": (
                self.is_proper_rotation
            ),
            "reflection_corrected": (
                self.reflection_corrected
            ),
            "transformation_matrix": (
                self.transformation_matrix.tolist()
            ),
            "metadata": dict(
                self.metadata
            ),
        }

        if include_coordinates:
            result[
                "reference_coordinates"
            ] = self.reference_coordinates.tolist()

            result[
                "mobile_coordinates"
            ] = self.mobile_coordinates.tolist()

            result[
                "aligned_coordinates"
            ] = self.aligned_coordinates.tolist()

        return result


# -----------------------------------------------------------------------------
# Internal alignment helpers
# -----------------------------------------------------------------------------

def _validate_alignment_weights(
    weights: ArrayLike,
    *,
    point_count: int,
    normalize: bool = True,
    name: str = "weights",
) -> FloatArray:
    """
    Validate coordinate-pair weights.

    Parameters
    ----------
    weights : array-like
        Weight collection.
    point_count : int
        Required number of weights.
    normalize : bool, optional
        Whether weights should sum to one.
    name : str, optional
        Parameter name used in validation messages.

    Returns
    -------
    numpy.ndarray
        Validated one-dimensional weights.
    """

    if isinstance(
        point_count,
        (
            bool,
            np.bool_,
        ),
    ) or not isinstance(
        point_count,
        (
            int,
            np.integer,
        ),
    ):
        raise TypeError(
            "point_count must be an integer."
        )

    point_count_value = int(
        point_count
    )

    if point_count_value < 1:
        raise ValueError(
            "point_count must be at least 1."
        )

    try:
        weight_array = np.asarray(
            weights,
            dtype=np.float64,
        )

    except (
        TypeError,
        ValueError,
        OverflowError,
    ) as error:
        raise TypeError(
            f"{name} must be convertible to a "
            "numeric one-dimensional array."
        ) from error

    weight_array = np.ravel(
        weight_array
    )

    if (
        weight_array.size
        != point_count_value
    ):
        raise ValueError(
            f"{name} must contain exactly "
            f"{point_count_value} values; received "
            f"{weight_array.size}."
        )

    if not np.all(
        np.isfinite(
            weight_array
        )
    ):
        raise ValueError(
            f"{name} contains NaN or infinite values."
        )

    if np.any(
        weight_array < 0.0
    ):
        raise ValueError(
            f"{name} cannot contain negative values."
        )

    total_weight = float(
        np.sum(
            weight_array
        )
    )

    if (
        total_weight
        <= DEFAULT_TOLERANCE
    ):
        raise ValueError(
            f"{name} must have a positive total."
        )

    result = np.array(
        weight_array,
        dtype=np.float64,
        copy=True,
    )

    if normalize:
        result /= total_weight

    return result


def _validate_rotation_matrix(
    rotation: ArrayLike,
    *,
    name: str = "rotation",
    require_proper: bool = False,
    tolerance: float = DEFAULT_DISTANCE_TOLERANCE,
    copy: bool = False,
) -> FloatArray:
    """
    Validate a three-dimensional orthogonal rotation matrix.

    Parameters
    ----------
    rotation : array-like
        Candidate matrix.
    name : str, optional
        Parameter name.
    require_proper : bool, optional
        Whether the determinant must be approximately ``+1``.
    tolerance : float, optional
        Orthogonality tolerance.
    copy : bool, optional
        Whether a copied matrix must be returned.

    Returns
    -------
    numpy.ndarray
        Validated matrix with shape ``(3, 3)``.
    """

    numeric_tolerance = (
        _validate_angular_tolerance(
            tolerance,
            name="tolerance",
        )
    )

    try:
        rotation_matrix = np.asarray(
            rotation,
            dtype=np.float64,
        )

    except (
        TypeError,
        ValueError,
        OverflowError,
    ) as error:
        raise TypeError(
            f"{name} must be convertible to a "
            "numeric 3 x 3 matrix."
        ) from error

    if rotation_matrix.shape != (
        3,
        3,
    ):
        raise ValueError(
            f"{name} must have shape (3, 3); "
            f"received {rotation_matrix.shape}."
        )

    if not np.all(
        np.isfinite(
            rotation_matrix
        )
    ):
        raise ValueError(
            f"{name} contains NaN or infinite values."
        )

    orthogonality_product = (
        rotation_matrix.T
        @ rotation_matrix
    )

    if not np.allclose(
        orthogonality_product,
        np.eye(
            3,
            dtype=np.float64,
        ),
        rtol=numeric_tolerance,
        atol=numeric_tolerance,
    ):
        raise ValueError(
            f"{name} is not an orthogonal matrix."
        )

    determinant_value = float(
        np.linalg.det(
            rotation_matrix
        )
    )

    if not math.isclose(
        abs(
            determinant_value
        ),
        1.0,
        rel_tol=numeric_tolerance,
        abs_tol=numeric_tolerance,
    ):
        raise ValueError(
            f"{name} determinant must have "
            "absolute value approximately equal to 1."
        )

    if require_proper and not math.isclose(
        determinant_value,
        1.0,
        rel_tol=numeric_tolerance,
        abs_tol=numeric_tolerance,
    ):
        raise ValueError(
            f"{name} must be a proper rotation "
            "with determinant +1."
        )

    if copy:
        return np.array(
            rotation_matrix,
            dtype=np.float64,
            copy=True,
        )

    return rotation_matrix


def _validate_alignment_pair(
    reference: CoordinateCollection,
    mobile: CoordinateCollection,
    *,
    scene: bool,
    minimum_rows: int = 1,
    copy: bool = False,
) -> Tuple[
    FloatArray,
    FloatArray,
]:
    """
    Validate two corresponding coordinate collections.

    Parameters
    ----------
    reference : CoordinateCollection
        Fixed reference coordinates.
    mobile : CoordinateCollection
        Coordinates to compare or align.
    scene : bool
        Whether scene coordinates should be preferred.
    minimum_rows : int, optional
        Minimum number of coordinate pairs.
    copy : bool, optional
        Whether matrices must be copied.

    Returns
    -------
    tuple of numpy.ndarray
        ``(reference_coordinates, mobile_coordinates)``.
    """

    reference_matrix = get_coordinates(
        reference,
        scene=scene,
        name="Reference coordinates",
        minimum_rows=minimum_rows,
        allow_empty=False,
        ignore_none=False,
        require_finite=True,
        copy=copy,
    )

    mobile_matrix = get_coordinates(
        mobile,
        scene=scene,
        name="Mobile coordinates",
        minimum_rows=minimum_rows,
        allow_empty=False,
        ignore_none=False,
        require_finite=True,
        copy=copy,
    )

    if (
        reference_matrix.shape
        != mobile_matrix.shape
    ):
        raise ValueError(
            "Reference and mobile coordinates "
            "must have identical shapes; received "
            f"{reference_matrix.shape} and "
            f"{mobile_matrix.shape}."
        )

    return (
        reference_matrix,
        mobile_matrix,
    )


# -----------------------------------------------------------------------------
# RMSD calculation
# -----------------------------------------------------------------------------

def calculate_rmsd(
    coordinates_1: CoordinateCollection,
    coordinates_2: CoordinateCollection,
    *,
    scene: bool = True,
    weights: Optional[ArrayLike] = None,
    squared: bool = False,
    return_distances: bool = False,
) -> Union[
    float,
    Tuple[
        float,
        FloatArray,
    ],
]:
    """
    Calculate RMSD between corresponding coordinates.

    Parameters
    ----------
    coordinates_1 : CoordinateCollection
        First coordinate collection.
    coordinates_2 : CoordinateCollection
        Second coordinate collection.
    scene : bool, optional
        Whether scene-transformed coordinates should be preferred.
    weights : array-like, optional
        Non-negative weights for corresponding coordinate pairs.
    squared : bool, optional
        Whether the mean squared displacement should be returned without
        taking the square root.
    return_distances : bool, optional
        Whether individual Euclidean pair distances should also be returned.

    Returns
    -------
    float
        RMSD or mean squared displacement.

    tuple
        ``(value, pair_distances)`` when ``return_distances=True``.

    Notes
    -----
    This function does not align the coordinates. Each row is compared
    directly with the corresponding row in the other collection.
    """

    (
        first_coordinates,
        second_coordinates,
    ) = _validate_alignment_pair(
        coordinates_1,
        coordinates_2,
        scene=scene,
        minimum_rows=1,
        copy=False,
    )

    point_count = int(
        first_coordinates.shape[0]
    )

    differences = (
        first_coordinates
        - second_coordinates
    )

    squared_distances = np.einsum(
        "ij,ij->i",
        differences,
        differences,
        optimize=True,
    )

    np.maximum(
        squared_distances,
        0.0,
        out=squared_distances,
    )

    if weights is None:
        mean_squared_displacement = float(
            np.mean(
                squared_distances
            )
        )

    else:
        normalized_weights = (
            _validate_alignment_weights(
                weights,
                point_count=point_count,
                normalize=True,
            )
        )

        mean_squared_displacement = float(
            np.sum(
                normalized_weights
                * squared_distances
            )
        )

    mean_squared_displacement = max(
        mean_squared_displacement,
        0.0,
    )

    if squared:
        result_value = (
            mean_squared_displacement
        )

    else:
        result_value = float(
            math.sqrt(
                mean_squared_displacement
            )
        )

    if return_distances:
        pair_distances = np.sqrt(
            squared_distances
        ).astype(
            np.float64,
            copy=False,
        )

        return (
            result_value,
            pair_distances,
        )

    return result_value


# -----------------------------------------------------------------------------
# Coordinate centering
# -----------------------------------------------------------------------------

def center_coordinates(
    coordinates: CoordinateCollection,
    *,
    scene: bool = True,
    weights: Optional[ArrayLike] = None,
    center: Optional[Coordinate] = None,
    return_centroid: bool = False,
    copy: bool = False,
) -> Union[
    FloatArray,
    Tuple[
        FloatArray,
        Vector3D,
    ],
]:
    """
    Center a coordinate collection around a selected point.

    Parameters
    ----------
    coordinates : CoordinateCollection
        Coordinates to center.
    scene : bool, optional
        Whether scene-transformed coordinates should be preferred.
    weights : array-like, optional
        Weights used to calculate the centroid.
    center : Coordinate, optional
        Explicit center to subtract. When provided, ``weights`` must be
        omitted.
    return_centroid : bool, optional
        Whether the subtracted center should also be returned.
    copy : bool, optional
        Whether the centered matrix must be copied.

    Returns
    -------
    numpy.ndarray
        Centered coordinate matrix.

    tuple
        ``(centered_coordinates, centroid)`` when
        ``return_centroid=True``.
    """

    coordinate_matrix = get_coordinates(
        coordinates,
        scene=scene,
        name="Coordinates",
        minimum_rows=1,
        allow_empty=False,
        ignore_none=False,
        require_finite=True,
        copy=False,
    )

    point_count = int(
        coordinate_matrix.shape[0]
    )

    if center is not None:
        if weights is not None:
            raise ValueError(
                "weights cannot be combined with "
                "an explicit center."
            )

        centroid_coordinate = as_coordinate(
            center,
            scene=scene,
            name="Center",
            copy=True,
        )

    elif weights is None:
        centroid_coordinate = np.mean(
            coordinate_matrix,
            axis=0,
            dtype=np.float64,
        )

    else:
        normalized_weights = (
            _validate_alignment_weights(
                weights,
                point_count=point_count,
                normalize=True,
            )
        )

        centroid_coordinate = np.sum(
            coordinate_matrix
            * normalized_weights[
                :,
                np.newaxis,
            ],
            axis=0,
            dtype=np.float64,
        )

    centered_matrix = (
        coordinate_matrix
        - centroid_coordinate
    ).astype(
        np.float64,
        copy=False,
    )

    if copy:
        centered_matrix = np.array(
            centered_matrix,
            dtype=np.float64,
            copy=True,
        )

    if return_centroid:
        return (
            centered_matrix,
            np.array(
                centroid_coordinate,
                dtype=np.float64,
                copy=True,
            ),
        )

    return centered_matrix


# -----------------------------------------------------------------------------
# Kabsch rotation
# -----------------------------------------------------------------------------

def kabsch_rotation(
    reference: CoordinateCollection,
    mobile: CoordinateCollection,
    *,
    scene: bool = True,
    weights: Optional[ArrayLike] = None,
    centered: bool = False,
    allow_reflection: bool = False,
    tolerance: float = DEFAULT_TOLERANCE,
    return_details: bool = False,
) -> Union[
    FloatArray,
    Tuple[
        FloatArray,
        FloatArray,
        bool,
    ],
]:
    """
    Calculate the optimal Kabsch rotation.

    The returned matrix rotates the mobile coordinates toward the reference
    coordinates using the row-vector convention:

    ``aligned_mobile = mobile @ rotation``

    Parameters
    ----------
    reference : CoordinateCollection
        Fixed reference coordinates.
    mobile : CoordinateCollection
        Coordinates to rotate.
    scene : bool, optional
        Whether scene-transformed coordinates should be preferred.
    weights : array-like, optional
        Non-negative coordinate-pair weights.
    centered : bool, optional
        Whether both collections are already centered.
    allow_reflection : bool, optional
        Whether an improper transformation with determinant ``-1`` may be
        returned.
    tolerance : float, optional
        Numerical tolerance used for degeneracy detection.
    return_details : bool, optional
        Whether singular values and the reflection-correction flag should
        also be returned.

    Returns
    -------
    numpy.ndarray
        Optimal ``(3, 3)`` transformation matrix.

    tuple
        ``(rotation, singular_values, reflection_corrected)`` when
        ``return_details=True``.

    Raises
    ------
    ValueError
        If fewer than two coordinate pairs are supplied or the covariance
        matrix is degenerate.

    Notes
    -----
    By default, reflections are prevented so the determinant of the returned
    rotation is approximately ``+1``.
    """

    numeric_tolerance = (
        _validate_angular_tolerance(
            tolerance,
            name="tolerance",
        )
    )

    if not isinstance(
        centered,
        (
            bool,
            np.bool_,
        ),
    ):
        raise TypeError(
            "centered must be boolean."
        )

    if not isinstance(
        allow_reflection,
        (
            bool,
            np.bool_,
        ),
    ):
        raise TypeError(
            "allow_reflection must be boolean."
        )

    if not isinstance(
        return_details,
        (
            bool,
            np.bool_,
        ),
    ):
        raise TypeError(
            "return_details must be boolean."
        )

    (
        reference_matrix,
        mobile_matrix,
    ) = _validate_alignment_pair(
        reference,
        mobile,
        scene=scene,
        minimum_rows=2,
        copy=False,
    )

    point_count = int(
        reference_matrix.shape[0]
    )

    if weights is None:
        normalized_weights = None

    else:
        normalized_weights = (
            _validate_alignment_weights(
                weights,
                point_count=point_count,
                normalize=True,
            )
        )

    if centered:
        reference_centered = (
            reference_matrix
        )

        mobile_centered = mobile_matrix

    else:
        reference_centered = (
            center_coordinates(
                reference_matrix,
                scene=False,
                weights=normalized_weights,
                copy=False,
            )
        )

        mobile_centered = (
            center_coordinates(
                mobile_matrix,
                scene=False,
                weights=normalized_weights,
                copy=False,
            )
        )

    if normalized_weights is None:
        covariance_matrix = (
            mobile_centered.T
            @ reference_centered
        )

    else:
        covariance_matrix = (
            mobile_centered.T
            @ (
                reference_centered
                * normalized_weights[
                    :,
                    np.newaxis,
                ]
            )
        )

    covariance_norm = float(
        np.linalg.norm(
            covariance_matrix
        )
    )

    if (
        covariance_norm
        <= numeric_tolerance
    ):
        raise ValueError(
            "Cannot calculate a stable Kabsch "
            "rotation from a degenerate covariance "
            "matrix."
        )

    try:
        (
            left_singular_vectors,
            singular_values,
            right_singular_vectors_transposed,
        ) = np.linalg.svd(
            covariance_matrix,
            full_matrices=True,
        )

    except np.linalg.LinAlgError as error:
        raise ValueError(
            "Kabsch alignment failed because singular "
            "value decomposition did not converge."
        ) from error

    initial_rotation = (
        left_singular_vectors
        @ right_singular_vectors_transposed
    )

    initial_determinant = float(
        np.linalg.det(
            initial_rotation
        )
    )

    reflection_corrected = False

    if (
        initial_determinant < 0.0
        and not allow_reflection
    ):
        corrected_left_vectors = np.array(
            left_singular_vectors,
            dtype=np.float64,
            copy=True,
        )

        corrected_left_vectors[
            :,
            -1,
        ] *= -1.0

        rotation_matrix = (
            corrected_left_vectors
            @ right_singular_vectors_transposed
        )

        reflection_corrected = True

    else:
        rotation_matrix = initial_rotation

    rotation_matrix = (
        _validate_rotation_matrix(
            rotation_matrix,
            name="Kabsch rotation",
            require_proper=(
                not allow_reflection
            ),
            tolerance=max(
                numeric_tolerance,
                DEFAULT_DISTANCE_TOLERANCE,
            ),
            copy=True,
        )
    )

    if return_details:
        return (
            rotation_matrix,
            np.array(
                singular_values,
                dtype=np.float64,
                copy=True,
            ),
            reflection_corrected,
        )

    return rotation_matrix


# -----------------------------------------------------------------------------
# Complete Kabsch alignment
# -----------------------------------------------------------------------------

def kabsch_alignment(
    reference: CoordinateCollection,
    mobile: CoordinateCollection,
    *,
    scene: bool = True,
    weights: Optional[ArrayLike] = None,
    allow_reflection: bool = False,
    tolerance: float = DEFAULT_TOLERANCE,
    metadata: Optional[
        Mapping[str, Any]
    ] = None,
) -> AlignmentResult:
    """
    Align mobile coordinates to a reference using the Kabsch algorithm.

    Parameters
    ----------
    reference : CoordinateCollection
        Fixed reference coordinates.
    mobile : CoordinateCollection
        Coordinates to rotate and translate.
    scene : bool, optional
        Whether scene-transformed coordinates should be preferred.
    weights : array-like, optional
        Non-negative coordinate-pair weights.
    allow_reflection : bool, optional
        Whether improper transformations may be used.
    tolerance : float, optional
        Numerical tolerance used during rotation calculation.
    metadata : Mapping[str, Any], optional
        Additional alignment metadata.

    Returns
    -------
    AlignmentResult
        Complete rigid-body alignment result.

    Notes
    -----
    The final transformation is:

    ``aligned = mobile @ rotation + translation``

    with:

    ``translation = reference_centroid - mobile_centroid @ rotation``
    """

    (
        reference_matrix,
        mobile_matrix,
    ) = _validate_alignment_pair(
        reference,
        mobile,
        scene=scene,
        minimum_rows=2,
        copy=True,
    )

    point_count = int(
        reference_matrix.shape[0]
    )

    if weights is None:
        normalized_weights = None

    else:
        normalized_weights = (
            _validate_alignment_weights(
                weights,
                point_count=point_count,
                normalize=True,
            )
        )

    (
        reference_centered,
        reference_centroid,
    ) = center_coordinates(
        reference_matrix,
        scene=False,
        weights=normalized_weights,
        return_centroid=True,
        copy=False,
    )

    (
        mobile_centered,
        mobile_centroid,
    ) = center_coordinates(
        mobile_matrix,
        scene=False,
        weights=normalized_weights,
        return_centroid=True,
        copy=False,
    )

    (
        rotation_matrix,
        singular_values,
        reflection_corrected,
    ) = kabsch_rotation(
        reference_centered,
        mobile_centered,
        scene=False,
        weights=normalized_weights,
        centered=True,
        allow_reflection=allow_reflection,
        tolerance=tolerance,
        return_details=True,
    )

    translation_vector = (
        reference_centroid
        - mobile_centroid
        @ rotation_matrix
    )

    aligned_coordinates = (
        mobile_matrix
        @ rotation_matrix
        + translation_vector
    ).astype(
        np.float64,
        copy=False,
    )

    initial_rmsd_value = calculate_rmsd(
        reference_matrix,
        mobile_matrix,
        scene=False,
        weights=normalized_weights,
    )

    final_rmsd_value = calculate_rmsd(
        reference_matrix,
        aligned_coordinates,
        scene=False,
        weights=normalized_weights,
    )

    determinant_value = float(
        np.linalg.det(
            rotation_matrix
        )
    )

    alignment_metadata: Dict[
        str,
        Any,
    ] = {}

    if metadata is not None:
        if not isinstance(
            metadata,
            Mapping,
        ):
            raise TypeError(
                "metadata must be a mapping or None."
            )

        alignment_metadata.update(
            metadata
        )

    alignment_metadata.setdefault(
        "weighted",
        normalized_weights is not None,
    )

    alignment_metadata.setdefault(
        "allow_reflection",
        bool(
            allow_reflection
        ),
    )

    alignment_metadata.setdefault(
        "coordinate_convention",
        "row_vectors",
    )

    return AlignmentResult(
        reference_coordinates=(
            reference_matrix
        ),
        mobile_coordinates=mobile_matrix,
        aligned_coordinates=(
            aligned_coordinates
        ),
        rotation=rotation_matrix,
        translation=translation_vector,
        reference_centroid=(
            reference_centroid
        ),
        mobile_centroid=mobile_centroid,
        initial_rmsd=initial_rmsd_value,
        final_rmsd=final_rmsd_value,
        point_count=point_count,
        weights=normalized_weights,
        singular_values=singular_values,
        determinant=determinant_value,
        reflection_corrected=(
            reflection_corrected
        ),
        metadata=alignment_metadata,
    )


# -----------------------------------------------------------------------------
# Aligned RMSD
# -----------------------------------------------------------------------------

def aligned_rmsd(
    reference: CoordinateCollection,
    mobile: CoordinateCollection,
    *,
    scene: bool = True,
    weights: Optional[ArrayLike] = None,
    allow_reflection: bool = False,
    tolerance: float = DEFAULT_TOLERANCE,
    return_alignment: bool = False,
) -> Union[
    float,
    Tuple[
        float,
        AlignmentResult,
    ],
]:
    """
    Calculate RMSD after optimal Kabsch alignment.

    Parameters
    ----------
    reference : CoordinateCollection
        Fixed reference coordinates.
    mobile : CoordinateCollection
        Coordinates to align.
    scene : bool, optional
        Whether scene-transformed coordinates should be preferred.
    weights : array-like, optional
        Non-negative coordinate-pair weights.
    allow_reflection : bool, optional
        Whether improper transformations may be used.
    tolerance : float, optional
        Numerical alignment tolerance.
    return_alignment : bool, optional
        Whether the complete alignment result should also be returned.

    Returns
    -------
    float
        RMSD after optimal alignment.

    tuple
        ``(rmsd, alignment_result)`` when
        ``return_alignment=True``.
    """

    alignment_result = kabsch_alignment(
        reference,
        mobile,
        scene=scene,
        weights=weights,
        allow_reflection=allow_reflection,
        tolerance=tolerance,
    )

    if return_alignment:
        return (
            alignment_result.final_rmsd,
            alignment_result,
        )

    return alignment_result.final_rmsd


# -----------------------------------------------------------------------------
# Public module interface
# -----------------------------------------------------------------------------

_SECTION_12_PUBLIC_NAMES = [
    "AlignmentResult",
    "calculate_rmsd",
    "center_coordinates",
    "kabsch_rotation",
    "kabsch_alignment",
    "aligned_rmsd",
]

_extend_public_names(_SECTION_12_PUBLIC_NAMES)


# =============================================================================
# End of Section 12
# =============================================================================


# =============================================================================
# Section 13 — Bounding Geometry
# =============================================================================


# -----------------------------------------------------------------------------
# Internal bounding-geometry helpers
# -----------------------------------------------------------------------------

def _validate_padding(
    padding: Union[
        float,
        Coordinate,
    ],
    *,
    name: str = "padding",
) -> Vector3D:
    """
    Validate scalar or axis-specific bounding-box padding.

    Parameters
    ----------
    padding : float or Coordinate
        Non-negative scalar padding or three axis-specific values.
    name : str, optional
        Parameter name used in validation messages.

    Returns
    -------
    numpy.ndarray
        Three-component padding vector.

    Raises
    ------
    TypeError
        If the value cannot be interpreted as numeric padding.
    ValueError
        If padding contains negative or non-finite values.
    """

    if isinstance(
        padding,
        (
            bool,
            np.bool_,
        ),
    ):
        raise TypeError(
            f"{name} must be numeric, not boolean."
        )

    if np.isscalar(
        padding
    ):
        try:
            scalar_padding = float(
                padding
            )

        except (
            TypeError,
            ValueError,
            OverflowError,
        ) as error:
            raise TypeError(
                f"{name} must be a numeric scalar "
                "or a three-component sequence."
            ) from error

        if not math.isfinite(
            scalar_padding
        ):
            raise ValueError(
                f"{name} must be finite."
            )

        if scalar_padding < 0.0:
            raise ValueError(
                f"{name} cannot be negative."
            )

        return np.full(
            3,
            scalar_padding,
            dtype=np.float64,
        )

    try:
        padding_vector = np.asarray(
            padding,
            dtype=np.float64,
        )

    except (
        TypeError,
        ValueError,
        OverflowError,
    ) as error:
        raise TypeError(
            f"{name} must be a numeric scalar "
            "or a three-component sequence."
        ) from error

    padding_vector = np.ravel(
        padding_vector
    )

    if padding_vector.shape != (
        3,
    ):
        raise ValueError(
            f"{name} must be a scalar or contain "
            "exactly three values."
        )

    if not np.all(
        np.isfinite(
            padding_vector
        )
    ):
        raise ValueError(
            f"{name} contains NaN or infinite values."
        )

    if np.any(
        padding_vector < 0.0
    ):
        raise ValueError(
            f"{name} cannot contain negative values."
        )

    return np.array(
        padding_vector,
        dtype=np.float64,
        copy=True,
    )


def _validate_bounding_limits(
    minimum: Coordinate,
    maximum: Coordinate,
    *,
    scene: bool = False,
    name: str = "Bounding box",
) -> Tuple[
    Vector3D,
    Vector3D,
]:
    """
    Validate minimum and maximum bounding-box limits.

    Parameters
    ----------
    minimum : Coordinate
        Minimum coordinate on each axis.
    maximum : Coordinate
        Maximum coordinate on each axis.
    scene : bool, optional
        Whether scene coordinates should be preferred.
    name : str, optional
        Name used in validation messages.

    Returns
    -------
    tuple of numpy.ndarray
        ``(minimum, maximum)``.

    Raises
    ------
    ValueError
        If any minimum coordinate exceeds its corresponding maximum.
    """

    minimum_coordinate = as_coordinate(
        minimum,
        scene=scene,
        name=f"{name} minimum",
        copy=True,
    )

    maximum_coordinate = as_coordinate(
        maximum,
        scene=scene,
        name=f"{name} maximum",
        copy=True,
    )

    if np.any(
        minimum_coordinate
        > maximum_coordinate
    ):
        raise ValueError(
            f"{name} minimum coordinates cannot "
            "exceed maximum coordinates."
        )

    return (
        minimum_coordinate,
        maximum_coordinate,
    )


def _validate_mass_array(
    masses: ArrayLike,
    *,
    atom_count: int,
    normalize: bool = False,
    allow_zero: bool = True,
    name: str = "masses",
) -> FloatArray:
    """
    Validate atomic masses or generic point weights.

    Parameters
    ----------
    masses : array-like
        Mass assigned to each coordinate.
    atom_count : int
        Required number of mass values.
    normalize : bool, optional
        Whether values should be normalized to sum to one.
    allow_zero : bool, optional
        Whether individual zero masses are allowed.
    name : str, optional
        Parameter name used in validation messages.

    Returns
    -------
    numpy.ndarray
        Validated one-dimensional mass array.

    Raises
    ------
    TypeError
        If masses cannot be converted to a numeric array.
    ValueError
        If the array has an invalid length, contains invalid values or has
        zero total mass.
    """

    if isinstance(
        atom_count,
        (
            bool,
            np.bool_,
        ),
    ) or not isinstance(
        atom_count,
        (
            int,
            np.integer,
        ),
    ):
        raise TypeError(
            "atom_count must be an integer."
        )

    atom_count_value = int(
        atom_count
    )

    if atom_count_value < 1:
        raise ValueError(
            "atom_count must be at least 1."
        )

    try:
        mass_array = np.asarray(
            masses,
            dtype=np.float64,
        )

    except (
        TypeError,
        ValueError,
        OverflowError,
    ) as error:
        raise TypeError(
            f"{name} must be convertible to a "
            "numeric one-dimensional array."
        ) from error

    mass_array = np.ravel(
        mass_array
    )

    if (
        mass_array.size
        != atom_count_value
    ):
        raise ValueError(
            f"{name} must contain exactly "
            f"{atom_count_value} values; received "
            f"{mass_array.size}."
        )

    if not np.all(
        np.isfinite(
            mass_array
        )
    ):
        raise ValueError(
            f"{name} contains NaN or infinite values."
        )

    if allow_zero:
        invalid_mask = (
            mass_array < 0.0
        )

        error_message = (
            f"{name} cannot contain negative values."
        )

    else:
        invalid_mask = (
            mass_array <= 0.0
        )

        error_message = (
            f"{name} must contain only positive values."
        )

    if np.any(
        invalid_mask
    ):
        raise ValueError(
            error_message
        )

    total_mass = float(
        np.sum(
            mass_array
        )
    )

    if (
        total_mass
        <= DEFAULT_TOLERANCE
    ):
        raise ValueError(
            f"{name} must have a positive total."
        )

    result = np.array(
        mass_array,
        dtype=np.float64,
        copy=True,
    )

    if normalize:
        result /= total_mass

    return result


def _extract_atomic_mass(
    atom: Any,
    *,
    name: str = "Atom",
) -> float:
    """
    Extract atomic mass from an atom-like object.

    Parameters
    ----------
    atom : Any
        ChimeraX atom or another atom-like object.
    name : str, optional
        Object name used in validation messages.

    Returns
    -------
    float
        Positive atomic mass.

    Raises
    ------
    TypeError
        If no supported mass representation is available.
    ValueError
        If the extracted mass is invalid.

    Notes
    -----
    Supported representations include:

    - ``atom.mass``;
    - ``atom.element.mass``;
    - ``atom.atomic_mass``;
    - mapping keys with equivalent names.
    """

    candidate_values: List[
        Tuple[
            str,
            Any,
        ]
    ] = []

    if isinstance(
        atom,
        Mapping,
    ):
        for key in (
            "mass",
            "atomic_mass",
            "element_mass",
        ):
            if key in atom:
                candidate_values.append(
                    (
                        key,
                        atom[key],
                    )
                )

        element_value = atom.get(
            "element"
        )

        if element_value is not None:
            if isinstance(
                element_value,
                Mapping,
            ):
                for key in (
                    "mass",
                    "atomic_mass",
                ):
                    if key in element_value:
                        candidate_values.append(
                            (
                                f"element.{key}",
                                element_value[key],
                            )
                        )

            else:
                for attribute_name in (
                    "mass",
                    "atomic_mass",
                ):
                    if hasattr(
                        element_value,
                        attribute_name,
                    ):
                        candidate_values.append(
                            (
                                f"element.{attribute_name}",
                                getattr(
                                    element_value,
                                    attribute_name,
                                ),
                            )
                        )

    else:
        for attribute_name in (
            "mass",
            "atomic_mass",
        ):
            if hasattr(
                atom,
                attribute_name,
            ):
                candidate_values.append(
                    (
                        attribute_name,
                        getattr(
                            atom,
                            attribute_name,
                        ),
                    )
                )

        element_value = getattr(
            atom,
            "element",
            None,
        )

        if element_value is not None:
            for attribute_name in (
                "mass",
                "atomic_mass",
            ):
                if hasattr(
                    element_value,
                    attribute_name,
                ):
                    candidate_values.append(
                        (
                            f"element.{attribute_name}",
                            getattr(
                                element_value,
                                attribute_name,
                            ),
                        )
                    )

    for source_name, candidate in candidate_values:
        if callable(
            candidate
        ):
            try:
                candidate = candidate()

            except Exception:
                continue

        if isinstance(
            candidate,
            (
                bool,
                np.bool_,
            ),
        ):
            continue

        try:
            mass_value = float(
                candidate
            )

        except (
            TypeError,
            ValueError,
            OverflowError,
        ):
            continue

        if not math.isfinite(
            mass_value
        ):
            raise ValueError(
                f"{name} mass extracted from "
                f"{source_name!r} is not finite."
            )

        if mass_value <= 0.0:
            raise ValueError(
                f"{name} mass extracted from "
                f"{source_name!r} must be positive."
            )

        return mass_value

    raise TypeError(
        f"Could not determine the mass of {name.lower()}. "
        "Provide masses explicitly or use atom-like "
        "objects exposing mass or element.mass."
    )


def _extract_collection_masses(
    atoms: Any,
    *,
    name: str = "Atoms",
) -> FloatArray:
    """
    Extract masses from an atom collection.

    Parameters
    ----------
    atoms : Any
        Iterable of atom-like objects.
    name : str, optional
        Collection name used in validation messages.

    Returns
    -------
    numpy.ndarray
        One mass per atom.
    """

    atom_items = _materialize_contact_collection(
        atoms,
        name=name,
    )

    masses = np.empty(
        len(
            atom_items
        ),
        dtype=np.float64,
    )

    for index, atom in enumerate(
        atom_items
    ):
        masses[index] = _extract_atomic_mass(
            atom,
            name=f"{name}[{index}]",
        )

    return masses


def _bounding_box_corners(
    minimum: Coordinate,
    maximum: Coordinate,
) -> FloatArray:
    """
    Construct all eight corners of an axis-aligned bounding box.

    Parameters
    ----------
    minimum : Coordinate
        Minimum axis coordinates.
    maximum : Coordinate
        Maximum axis coordinates.

    Returns
    -------
    numpy.ndarray
        Corner matrix with shape ``(8, 3)``.
    """

    minimum_coordinate, maximum_coordinate = (
        _validate_bounding_limits(
            minimum,
            maximum,
            scene=False,
        )
    )

    return np.asarray(
        [
            [
                minimum_coordinate[0],
                minimum_coordinate[1],
                minimum_coordinate[2],
            ],
            [
                minimum_coordinate[0],
                minimum_coordinate[1],
                maximum_coordinate[2],
            ],
            [
                minimum_coordinate[0],
                maximum_coordinate[1],
                minimum_coordinate[2],
            ],
            [
                minimum_coordinate[0],
                maximum_coordinate[1],
                maximum_coordinate[2],
            ],
            [
                maximum_coordinate[0],
                minimum_coordinate[1],
                minimum_coordinate[2],
            ],
            [
                maximum_coordinate[0],
                minimum_coordinate[1],
                maximum_coordinate[2],
            ],
            [
                maximum_coordinate[0],
                maximum_coordinate[1],
                minimum_coordinate[2],
            ],
            [
                maximum_coordinate[0],
                maximum_coordinate[1],
                maximum_coordinate[2],
            ],
        ],
        dtype=np.float64,
    )


# -----------------------------------------------------------------------------
# Axis-aligned bounding box
# -----------------------------------------------------------------------------

def bounding_box(
    coordinates: CoordinateCollection,
    *,
    scene: bool = True,
    padding: Union[
        float,
        Coordinate,
    ] = 0.0,
    return_corners: bool = False,
    copy: bool = False,
) -> Union[
    Tuple[
        Vector3D,
        Vector3D,
    ],
    Tuple[
        Vector3D,
        Vector3D,
        FloatArray,
    ],
]:
    """
    Calculate an axis-aligned bounding box.

    Parameters
    ----------
    coordinates : CoordinateCollection
        Atoms or coordinate-like values.
    scene : bool, optional
        Whether scene-transformed coordinates should be preferred.
    padding : float or Coordinate, optional
        Non-negative padding added on both sides of each axis. A scalar
        applies equal padding to all axes; three values apply axis-specific
        padding.
    return_corners : bool, optional
        Whether all eight box corners should also be returned.
    copy : bool, optional
        Whether returned arrays must be copied.

    Returns
    -------
    tuple
        ``(minimum, maximum)``.

    tuple
        ``(minimum, maximum, corners)`` when ``return_corners=True``.

    Notes
    -----
    The box is aligned with the global Cartesian axes. It is not an oriented
    minimum-volume bounding box.
    """

    if not isinstance(
        return_corners,
        (
            bool,
            np.bool_,
        ),
    ):
        raise TypeError(
            "return_corners must be boolean."
        )

    coordinate_matrix = get_coordinates(
        coordinates,
        scene=scene,
        name="Bounding coordinates",
        minimum_rows=1,
        allow_empty=False,
        ignore_none=False,
        require_finite=True,
        copy=False,
    )

    padding_vector = _validate_padding(
        padding
    )

    minimum_coordinate = (
        np.min(
            coordinate_matrix,
            axis=0,
        )
        - padding_vector
    ).astype(
        np.float64,
        copy=False,
    )

    maximum_coordinate = (
        np.max(
            coordinate_matrix,
            axis=0,
        )
        + padding_vector
    ).astype(
        np.float64,
        copy=False,
    )

    if copy:
        minimum_coordinate = np.array(
            minimum_coordinate,
            dtype=np.float64,
            copy=True,
        )

        maximum_coordinate = np.array(
            maximum_coordinate,
            dtype=np.float64,
            copy=True,
        )

    if return_corners:
        corners = _bounding_box_corners(
            minimum_coordinate,
            maximum_coordinate,
        )

        return (
            minimum_coordinate,
            maximum_coordinate,
            corners,
        )

    return (
        minimum_coordinate,
        maximum_coordinate,
    )


# -----------------------------------------------------------------------------
# Bounding-box center
# -----------------------------------------------------------------------------

def bounding_box_center(
    coordinates: Optional[
        CoordinateCollection
    ] = None,
    *,
    minimum: Optional[
        Coordinate
    ] = None,
    maximum: Optional[
        Coordinate
    ] = None,
    scene: bool = True,
    padding: Union[
        float,
        Coordinate,
    ] = 0.0,
    copy: bool = False,
) -> Vector3D:
    """
    Return the center of an axis-aligned bounding box.

    Parameters
    ----------
    coordinates : CoordinateCollection, optional
        Atoms or coordinates used to calculate the box.
    minimum : Coordinate, optional
        Precomputed minimum box coordinate.
    maximum : Coordinate, optional
        Precomputed maximum box coordinate.
    scene : bool, optional
        Whether scene-transformed coordinates should be preferred.
    padding : float or Coordinate, optional
        Padding used when calculating the box from ``coordinates``.
    copy : bool, optional
        Whether a copied vector must be returned.

    Returns
    -------
    numpy.ndarray
        Bounding-box center.

    Raises
    ------
    ValueError
        If coordinates are combined with explicit limits or if only one
        explicit limit is supplied.
    """

    explicit_limits = (
        minimum is not None
        or maximum is not None
    )

    if (
        coordinates is not None
        and explicit_limits
    ):
        raise ValueError(
            "Provide either coordinates or explicit "
            "minimum and maximum limits, not both."
        )

    if coordinates is not None:
        (
            minimum_coordinate,
            maximum_coordinate,
        ) = bounding_box(
            coordinates,
            scene=scene,
            padding=padding,
            return_corners=False,
            copy=False,
        )

    else:
        if (
            minimum is None
            or maximum is None
        ):
            raise ValueError(
                "Provide coordinates or both minimum "
                "and maximum limits."
            )

        (
            minimum_coordinate,
            maximum_coordinate,
        ) = _validate_bounding_limits(
            minimum,
            maximum,
            scene=scene,
        )

    center = (
        (
            minimum_coordinate
            + maximum_coordinate
        )
        / 2.0
    ).astype(
        np.float64,
        copy=False,
    )

    if copy:
        return np.array(
            center,
            dtype=np.float64,
            copy=True,
        )

    return center


# -----------------------------------------------------------------------------
# Bounding-box size
# -----------------------------------------------------------------------------

def bounding_box_size(
    coordinates: Optional[
        CoordinateCollection
    ] = None,
    *,
    minimum: Optional[
        Coordinate
    ] = None,
    maximum: Optional[
        Coordinate
    ] = None,
    scene: bool = True,
    padding: Union[
        float,
        Coordinate,
    ] = 0.0,
    return_volume: bool = False,
    return_diagonal: bool = False,
    copy: bool = False,
) -> Union[
    Vector3D,
    Tuple[
        Vector3D,
        float,
    ],
    Tuple[
        Vector3D,
        float,
        float,
    ],
]:
    """
    Return the Cartesian dimensions of an axis-aligned bounding box.

    Parameters
    ----------
    coordinates : CoordinateCollection, optional
        Atoms or coordinates used to calculate the box.
    minimum : Coordinate, optional
        Precomputed minimum coordinate.
    maximum : Coordinate, optional
        Precomputed maximum coordinate.
    scene : bool, optional
        Whether scene-transformed coordinates should be preferred.
    padding : float or Coordinate, optional
        Padding used when calculating limits from coordinates.
    return_volume : bool, optional
        Whether box volume should also be returned.
    return_diagonal : bool, optional
        Whether the box diagonal length should also be returned.
    copy : bool, optional
        Whether a copied size vector must be returned.

    Returns
    -------
    numpy.ndarray
        Box dimensions ``[size_x, size_y, size_z]``.

    tuple
        Additional values are returned in the order ``volume``, then
        ``diagonal`` when requested.
    """

    explicit_limits = (
        minimum is not None
        or maximum is not None
    )

    if (
        coordinates is not None
        and explicit_limits
    ):
        raise ValueError(
            "Provide either coordinates or explicit "
            "minimum and maximum limits, not both."
        )

    if coordinates is not None:
        (
            minimum_coordinate,
            maximum_coordinate,
        ) = bounding_box(
            coordinates,
            scene=scene,
            padding=padding,
            return_corners=False,
            copy=False,
        )

    else:
        if (
            minimum is None
            or maximum is None
        ):
            raise ValueError(
                "Provide coordinates or both minimum "
                "and maximum limits."
            )

        (
            minimum_coordinate,
            maximum_coordinate,
        ) = _validate_bounding_limits(
            minimum,
            maximum,
            scene=scene,
        )

    size = (
        maximum_coordinate
        - minimum_coordinate
    ).astype(
        np.float64,
        copy=False,
    )

    if copy:
        size = np.array(
            size,
            dtype=np.float64,
            copy=True,
        )

    result: List[Any] = [
        size
    ]

    if return_volume:
        volume = float(
            np.prod(
                size
            )
        )

        result.append(
            volume
        )

    if return_diagonal:
        diagonal = float(
            vector_norm(
                size,
                scene=False,
            )
        )

        result.append(
            diagonal
        )

    if len(
        result
    ) == 1:
        return size

    return tuple(
        result
    )


# -----------------------------------------------------------------------------
# Geometric center
# -----------------------------------------------------------------------------

def geometric_center(
    coordinates: CoordinateCollection,
    *,
    scene: bool = True,
    copy: bool = False,
) -> Vector3D:
    """
    Return the arithmetic mean of a coordinate collection.

    Parameters
    ----------
    coordinates : CoordinateCollection
        Atoms or coordinates.
    scene : bool, optional
        Whether scene-transformed coordinates should be preferred.
    copy : bool, optional
        Whether a copied coordinate must be returned.

    Returns
    -------
    numpy.ndarray
        Arithmetic mean coordinate.

    Notes
    -----
    Every point contributes equally, independent of atomic mass.
    """

    coordinate_matrix = get_coordinates(
        coordinates,
        scene=scene,
        name="Coordinates",
        minimum_rows=1,
        allow_empty=False,
        ignore_none=False,
        require_finite=True,
        copy=False,
    )

    center = np.mean(
        coordinate_matrix,
        axis=0,
        dtype=np.float64,
    )

    if copy:
        return np.array(
            center,
            dtype=np.float64,
            copy=True,
        )

    return np.asarray(
        center,
        dtype=np.float64,
    )


# -----------------------------------------------------------------------------
# Center of mass
# -----------------------------------------------------------------------------

def center_of_mass(
    atoms_or_coordinates: Any,
    *,
    masses: Optional[
        ArrayLike
    ] = None,
    scene: bool = True,
    copy: bool = False,
    return_total_mass: bool = False,
) -> Union[
    Vector3D,
    Tuple[
        Vector3D,
        float,
    ],
]:
    """
    Calculate the mass-weighted center of a molecular structure.

    Parameters
    ----------
    atoms_or_coordinates : Any
        Atom collection or coordinate collection.
    masses : array-like, optional
        Mass assigned to each coordinate. When omitted, masses are extracted
        from the original atom objects.
    scene : bool, optional
        Whether scene-transformed coordinates should be preferred.
    copy : bool, optional
        Whether a copied coordinate must be returned.
    return_total_mass : bool, optional
        Whether the total mass should also be returned.

    Returns
    -------
    numpy.ndarray
        Center of mass.

    tuple
        ``(center_of_mass, total_mass)`` when
        ``return_total_mass=True``.

    Raises
    ------
    TypeError
        If masses are omitted and cannot be extracted from the input atoms.
    ValueError
        If masses are invalid or do not match the coordinate count.

    Notes
    -----
    The center of mass is calculated as:

    ``sum(mass_i * coordinate_i) / sum(mass_i)``.
    """

    original_items = (
        _materialize_contact_collection(
            atoms_or_coordinates,
            name="Atoms or coordinates",
        )
    )

    coordinate_matrix = get_coordinates(
        original_items,
        scene=scene,
        name="Atoms or coordinates",
        minimum_rows=1,
        allow_empty=False,
        ignore_none=False,
        require_finite=True,
        copy=False,
    )

    atom_count = int(
        coordinate_matrix.shape[0]
    )

    if masses is None:
        mass_array = (
            _extract_collection_masses(
                original_items,
                name="Atoms",
            )
        )

    else:
        mass_array = _validate_mass_array(
            masses,
            atom_count=atom_count,
            normalize=False,
            allow_zero=True,
            name="masses",
        )

    total_mass = float(
        np.sum(
            mass_array
        )
    )

    center = np.sum(
        coordinate_matrix
        * mass_array[
            :,
            np.newaxis,
        ],
        axis=0,
        dtype=np.float64,
    ) / total_mass

    center = center.astype(
        np.float64,
        copy=False,
    )

    if copy:
        center = np.array(
            center,
            dtype=np.float64,
            copy=True,
        )

    if return_total_mass:
        return (
            center,
            total_mass,
        )

    return center


# -----------------------------------------------------------------------------
# Radius of gyration
# -----------------------------------------------------------------------------

def radius_of_gyration(
    atoms_or_coordinates: Any,
    *,
    masses: Optional[
        ArrayLike
    ] = None,
    center: Optional[
        Coordinate
    ] = None,
    mass_weighted: bool = True,
    scene: bool = True,
    squared: bool = False,
    return_center: bool = False,
) -> Union[
    float,
    Tuple[
        float,
        Vector3D,
    ],
]:
    """
    Calculate the radius of gyration of a coordinate collection.

    Parameters
    ----------
    atoms_or_coordinates : Any
        Atom collection or coordinate collection.
    masses : array-like, optional
        Mass assigned to each atom. When omitted and ``mass_weighted=True``,
        masses are extracted from the atom objects.
    center : Coordinate, optional
        Explicit reference center. When omitted, the center of mass is used
        for mass-weighted calculations and the geometric center otherwise.
    mass_weighted : bool, optional
        Whether squared distances should be mass weighted.
    scene : bool, optional
        Whether scene-transformed coordinates should be preferred.
    squared : bool, optional
        Whether squared radius of gyration should be returned.
    return_center : bool, optional
        Whether the center used in the calculation should also be returned.

    Returns
    -------
    float
        Radius of gyration or squared radius of gyration.

    tuple
        ``(radius_of_gyration, center)`` when ``return_center=True``.

    Notes
    -----
    For a mass-weighted calculation:

    ``Rg² = sum(mass_i * ||coordinate_i - center||²) / sum(mass_i)``.

    For an unweighted calculation:

    ``Rg² = mean(||coordinate_i - center||²)``.
    """

    if not isinstance(
        mass_weighted,
        (
            bool,
            np.bool_,
        ),
    ):
        raise TypeError(
            "mass_weighted must be boolean."
        )

    if not isinstance(
        squared,
        (
            bool,
            np.bool_,
        ),
    ):
        raise TypeError(
            "squared must be boolean."
        )

    if not isinstance(
        return_center,
        (
            bool,
            np.bool_,
        ),
    ):
        raise TypeError(
            "return_center must be boolean."
        )

    original_items = (
        _materialize_contact_collection(
            atoms_or_coordinates,
            name="Atoms or coordinates",
        )
    )

    coordinate_matrix = get_coordinates(
        original_items,
        scene=scene,
        name="Atoms or coordinates",
        minimum_rows=1,
        allow_empty=False,
        ignore_none=False,
        require_finite=True,
        copy=False,
    )

    point_count = int(
        coordinate_matrix.shape[0]
    )

    if mass_weighted:
        if masses is None:
            mass_array = (
                _extract_collection_masses(
                    original_items,
                    name="Atoms",
                )
            )

        else:
            mass_array = (
                _validate_mass_array(
                    masses,
                    atom_count=point_count,
                    normalize=False,
                    allow_zero=True,
                    name="masses",
                )
            )

        total_mass = float(
            np.sum(
                mass_array
            )
        )

        if center is None:
            center_coordinate = np.sum(
                coordinate_matrix
                * mass_array[
                    :,
                    np.newaxis,
                ],
                axis=0,
                dtype=np.float64,
            ) / total_mass

        else:
            center_coordinate = (
                as_coordinate(
                    center,
                    scene=scene,
                    name="Radius-of-gyration center",
                    copy=True,
                )
            )

    else:
        if masses is not None:
            raise ValueError(
                "masses cannot be supplied when "
                "mass_weighted=False."
            )

        mass_array = None
        total_mass = None

        if center is None:
            center_coordinate = np.mean(
                coordinate_matrix,
                axis=0,
                dtype=np.float64,
            )

        else:
            center_coordinate = (
                as_coordinate(
                    center,
                    scene=scene,
                    name="Radius-of-gyration center",
                    copy=True,
                )
            )

    displacements = (
        coordinate_matrix
        - center_coordinate
    )

    squared_distances = np.einsum(
        "ij,ij->i",
        displacements,
        displacements,
        optimize=True,
    )

    np.maximum(
        squared_distances,
        0.0,
        out=squared_distances,
    )

    if mass_weighted:
        squared_radius = float(
            np.sum(
                mass_array
                * squared_distances
            )
            / total_mass
        )

    else:
        squared_radius = float(
            np.mean(
                squared_distances
            )
        )

    squared_radius = max(
        squared_radius,
        0.0,
    )

    if squared:
        result_value = squared_radius

    else:
        result_value = float(
            math.sqrt(
                squared_radius
            )
        )

    if return_center:
        return (
            result_value,
            np.array(
                center_coordinate,
                dtype=np.float64,
                copy=True,
            ),
        )

    return result_value


# -----------------------------------------------------------------------------
# Public module interface
# -----------------------------------------------------------------------------

_SECTION_13_PUBLIC_NAMES = [
    "bounding_box",
    "bounding_box_center",
    "bounding_box_size",
    "geometric_center",
    "center_of_mass",
    "radius_of_gyration",
]

_extend_public_names(_SECTION_13_PUBLIC_NAMES)


# =============================================================================
# End of Section 13
# =============================================================================


# =============================================================================
# Section 14 — Structured Results
# =============================================================================

# -----------------------------------------------------------------------------
# Structured-result protocol
# -----------------------------------------------------------------------------

@runtime_checkable
class StructuralResult(
    Protocol
):
    """
    Protocol implemented by structured geometry results.

    Any class satisfying this protocol must provide a ``to_dict()`` method
    returning a dictionary suitable for further serialization.

    Notes
    -----
    The protocol provides structural typing. A class does not need to inherit
    from ``StructuralResult`` explicitly; implementing a compatible
    ``to_dict()`` method is sufficient.
    """

    def to_dict(
        self,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Convert the structured result to a dictionary.

        Parameters
        ----------
        **kwargs : Any
            Class-specific serialization options.

        Returns
        -------
        dict
            Serialized result.
        """

        ...


# -----------------------------------------------------------------------------
# Generic structured-result serialization
# -----------------------------------------------------------------------------

def _serialize_structural_value(
    value: Any,
    *,
    include_private: bool = False,
    stringify_unknown: bool = False,
) -> Any:
    """
    Recursively convert a value to a serialization-friendly representation.

    Parameters
    ----------
    value : Any
        Value to serialize.
    include_private : bool, optional
        Whether mapping or dataclass fields whose names begin with an
        underscore should be included.
    stringify_unknown : bool, optional
        Whether unsupported objects should be converted with ``repr()``.
        When ``False``, unsupported values are returned unchanged.

    Returns
    -------
    Any
        Recursively serialized value.

    Notes
    -----
    This helper recognizes:

    - Python scalar values;
    - NumPy scalar values;
    - NumPy arrays;
    - mappings;
    - lists, tuples and sets;
    - objects implementing ``to_dict()``;
    - arbitrary dataclass instances.
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

    if isinstance(
        value,
        np.generic,
    ):
        return value.item()

    if isinstance(
        value,
        np.ndarray,
    ):
        return value.tolist()

    if isinstance(
        value,
        Mapping,
    ):
        serialized_mapping: Dict[
            str,
            Any,
        ] = {}

        for key, item in value.items():
            key_string = str(
                key
            )

            if (
                not include_private
                and key_string.startswith(
                    "_"
                )
            ):
                continue

            serialized_mapping[
                key_string
            ] = _serialize_structural_value(
                item,
                include_private=include_private,
                stringify_unknown=(
                    stringify_unknown
                ),
            )

        return serialized_mapping

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
            _serialize_structural_value(
                item,
                include_private=include_private,
                stringify_unknown=(
                    stringify_unknown
                ),
            )
            for item in value
        ]

    to_dict_method = getattr(
        value,
        "to_dict",
        None,
    )

    if callable(
        to_dict_method
    ):
        try:
            serialized_value = (
                to_dict_method()
            )

        except TypeError:
            serialized_value = None

        if serialized_value is not None:
            return _serialize_structural_value(
                serialized_value,
                include_private=include_private,
                stringify_unknown=(
                    stringify_unknown
                ),
            )

    if is_dataclass(
        value
    ):
        serialized_dataclass: Dict[
            str,
            Any,
        ] = {}

        for dataclass_field in fields(
            value
        ):
            field_name = (
                dataclass_field.name
            )

            if (
                not include_private
                and field_name.startswith(
                    "_"
                )
            ):
                continue

            serialized_dataclass[
                field_name
            ] = _serialize_structural_value(
                getattr(
                    value,
                    field_name,
                ),
                include_private=include_private,
                stringify_unknown=(
                    stringify_unknown
                ),
            )

        return serialized_dataclass

    if stringify_unknown:
        return repr(
            value
        )

    return value


def structural_result_to_dict(
    result: Any,
    *,
    include_private: bool = False,
    stringify_unknown: bool = False,
    serialization_options: Optional[
        Mapping[str, Any]
    ] = None,
) -> Dict[str, Any]:
    """
    Serialize a structured geometry result.

    Parameters
    ----------
    result : Any
        Structured result implementing ``to_dict()`` or an arbitrary
        dataclass instance.
    include_private : bool, optional
        Whether private fields beginning with an underscore should be
        included when generic dataclass serialization is used.
    stringify_unknown : bool, optional
        Whether unsupported nested values should be converted with ``repr()``.
    serialization_options : Mapping[str, Any], optional
        Keyword arguments passed to the result's own ``to_dict()`` method.

    Returns
    -------
    dict
        Serialized result.

    Raises
    ------
    TypeError
        If the object is neither a dataclass nor an object implementing a
        compatible ``to_dict()`` method.
    ValueError
        If ``to_dict()`` does not return a mapping.
    """

    if serialization_options is None:
        options: Dict[str, Any] = {}

    elif isinstance(
        serialization_options,
        Mapping,
    ):
        options = dict(
            serialization_options
        )

    else:
        raise TypeError(
            "serialization_options must be a "
            "mapping or None."
        )

    to_dict_method = getattr(
        result,
        "to_dict",
        None,
    )

    if callable(
        to_dict_method
    ):
        try:
            raw_result = to_dict_method(
                **options
            )

        except TypeError as error:
            if options:
                raise TypeError(
                    "The supplied serialization options "
                    "are not accepted by the result's "
                    "to_dict() method."
                ) from error

            raise

        if not isinstance(
            raw_result,
            Mapping,
        ):
            raise ValueError(
                "The result's to_dict() method must "
                "return a mapping."
            )

        serialized = _serialize_structural_value(
            raw_result,
            include_private=include_private,
            stringify_unknown=(
                stringify_unknown
            ),
        )

        return dict(
            serialized
        )

    if is_dataclass(
        result
    ):
        serialized = _serialize_structural_value(
            result,
            include_private=include_private,
            stringify_unknown=(
                stringify_unknown
            ),
        )

        if not isinstance(
            serialized,
            Mapping,
        ):
            raise ValueError(
                "Dataclass serialization did not "
                "produce a mapping."
            )

        return dict(
            serialized
        )

    raise TypeError(
        "result must be a dataclass or implement "
        "a to_dict() method."
    )


# -----------------------------------------------------------------------------
# Bounding-box structured result
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class BoundingBox:
    """
    Represent a three-dimensional axis-aligned bounding box.

    Parameters
    ----------
    minimum : Coordinate
        Minimum coordinate along each Cartesian axis.
    maximum : Coordinate
        Maximum coordinate along each Cartesian axis.
    padding : Coordinate, optional
        Padding previously applied along each axis.
    point_count : int, optional
        Number of points used to construct the box.
    metadata : Mapping[str, Any], optional
        Additional bounding-box information.

    Attributes
    ----------
    minimum : numpy.ndarray
        Read-only minimum coordinate.
    maximum : numpy.ndarray
        Read-only maximum coordinate.
    padding : numpy.ndarray
        Read-only axis-specific padding.
    point_count : int or None
        Number of source points.

    Notes
    -----
    The represented box is aligned with the global Cartesian axes. It is not
    an oriented minimum-volume bounding box.
    """

    minimum: Coordinate
    maximum: Coordinate

    padding: Coordinate = field(
        default_factory=lambda: np.zeros(
            3,
            dtype=np.float64,
        )
    )

    point_count: Optional[int] = None

    metadata: GeometryMetadata = field(
        default_factory=dict
    )

    def __post_init__(
        self,
    ) -> None:
        """
        Validate and normalize bounding-box attributes.
        """

        (
            minimum_coordinate,
            maximum_coordinate,
        ) = _validate_bounding_limits(
            self.minimum,
            self.maximum,
            scene=False,
            name="BoundingBox",
        )

        padding_vector = _validate_padding(
            self.padding,
            name="padding",
        )

        if self.point_count is None:
            point_count_value = None

        else:
            if isinstance(
                self.point_count,
                (
                    bool,
                    np.bool_,
                ),
            ) or not isinstance(
                self.point_count,
                (
                    int,
                    np.integer,
                ),
            ):
                raise TypeError(
                    "point_count must be an integer "
                    "or None."
                )

            point_count_value = int(
                self.point_count
            )

            if point_count_value < 1:
                raise ValueError(
                    "point_count must be at least 1."
                )

        minimum_coordinate = np.array(
            minimum_coordinate,
            dtype=np.float64,
            copy=True,
        )

        maximum_coordinate = np.array(
            maximum_coordinate,
            dtype=np.float64,
            copy=True,
        )

        padding_vector = np.array(
            padding_vector,
            dtype=np.float64,
            copy=True,
        )

        minimum_coordinate.setflags(
            write=False
        )

        maximum_coordinate.setflags(
            write=False
        )

        padding_vector.setflags(
            write=False
        )

        object.__setattr__(
            self,
            "minimum",
            minimum_coordinate,
        )

        object.__setattr__(
            self,
            "maximum",
            maximum_coordinate,
        )

        object.__setattr__(
            self,
            "padding",
            padding_vector,
        )

        object.__setattr__(
            self,
            "point_count",
            point_count_value,
        )

        if self.metadata is None:
            metadata_value: Dict[str, Any] = {}

        elif isinstance(
            self.metadata,
            Mapping,
        ):
            metadata_value = dict(
                self.metadata
            )

        else:
            raise TypeError(
                "metadata must be a mapping or None."
            )

        metadata_value.setdefault(
            "geometry_type",
            "axis_aligned_bounding_box",
        )

        object.__setattr__(
            self,
            "metadata",
            metadata_value,
        )

    @property
    def center(
        self,
    ) -> Vector3D:
        """
        Return the center of the bounding box.

        Returns
        -------
        numpy.ndarray
            Box center.
        """

        return (
            (
                self.minimum
                + self.maximum
            )
            / 2.0
        ).astype(
            np.float64,
            copy=True,
        )

    @property
    def size(
        self,
    ) -> Vector3D:
        """
        Return the box size along each axis.

        Returns
        -------
        numpy.ndarray
            Dimensions ``[size_x, size_y, size_z]``.
        """

        return (
            self.maximum
            - self.minimum
        ).astype(
            np.float64,
            copy=True,
        )

    @property
    def half_size(
        self,
    ) -> Vector3D:
        """
        Return half of the box size along each axis.

        Returns
        -------
        numpy.ndarray
            Half-dimensions.
        """

        return (
            self.size
            / 2.0
        ).astype(
            np.float64,
            copy=False,
        )

    @property
    def volume(
        self,
    ) -> float:
        """
        Return the bounding-box volume.

        Returns
        -------
        float
            Cartesian box volume.
        """

        return float(
            np.prod(
                self.size
            )
        )

    @property
    def diagonal(
        self,
    ) -> float:
        """
        Return the box diagonal length.

        Returns
        -------
        float
            Distance between opposite corners.
        """

        return float(
            vector_norm(
                self.size,
                scene=False,
            )
        )

    @property
    def corners(
        self,
    ) -> FloatArray:
        """
        Return all eight bounding-box corners.

        Returns
        -------
        numpy.ndarray
            Corner matrix with shape ``(8, 3)``.
        """

        return _bounding_box_corners(
            self.minimum,
            self.maximum,
        )

    @property
    def is_degenerate(
        self,
    ) -> bool:
        """
        Return whether at least one box dimension is numerically zero.

        Returns
        -------
        bool
            ``True`` for a planar, linear or point-like box.
        """

        return bool(
            np.any(
                self.size
                <= DEFAULT_TOLERANCE
            )
        )

    def contains(
        self,
        coordinate: Coordinate,
        *,
        scene: bool = True,
        tolerance: float = DEFAULT_DISTANCE_TOLERANCE,
    ) -> bool:
        """
        Test whether a coordinate lies inside the box.

        Parameters
        ----------
        coordinate : Coordinate
            Point to test.
        scene : bool, optional
            Whether scene coordinates should be preferred.
        tolerance : float, optional
            Numerical boundary tolerance.

        Returns
        -------
        bool
            ``True`` when the coordinate is inside or on the box boundary.
        """

        tolerance_value = (
            _validate_nonnegative_finite_value(
                tolerance,
                name="tolerance",
            )
        )

        point = as_coordinate(
            coordinate,
            scene=scene,
            name="Coordinate",
            copy=False,
        )

        return bool(
            np.all(
                point
                >= (
                    self.minimum
                    - tolerance_value
                )
            )
            and np.all(
                point
                <= (
                    self.maximum
                    + tolerance_value
                )
            )
        )

    def to_dict(
        self,
        *,
        include_corners: bool = False,
    ) -> Dict[str, Any]:
        """
        Convert the bounding box to a serializable dictionary.

        Parameters
        ----------
        include_corners : bool, optional
            Whether all eight corner coordinates should be included.

        Returns
        -------
        dict
            Serialized bounding box.
        """

        result: Dict[str, Any] = {
            "minimum": self.minimum.tolist(),
            "maximum": self.maximum.tolist(),
            "center": self.center.tolist(),
            "size": self.size.tolist(),
            "half_size": (
                self.half_size.tolist()
            ),
            "padding": self.padding.tolist(),
            "volume": self.volume,
            "diagonal": self.diagonal,
            "point_count": self.point_count,
            "is_degenerate": (
                self.is_degenerate
            ),
            "metadata": dict(
                self.metadata
            ),
        }

        if include_corners:
            result["corners"] = (
                self.corners.tolist()
            )

        return result


# -----------------------------------------------------------------------------
# BoundingBox construction
# -----------------------------------------------------------------------------

def create_bounding_box(
    coordinates: CoordinateCollection,
    *,
    scene: bool = True,
    padding: Union[
        float,
        Coordinate,
    ] = 0.0,
    metadata: Optional[
        Mapping[str, Any]
    ] = None,
) -> BoundingBox:
    """
    Construct a structured bounding-box result.

    Parameters
    ----------
    coordinates : CoordinateCollection
        Atoms or coordinate-like values.
    scene : bool, optional
        Whether scene-transformed coordinates should be preferred.
    padding : float or Coordinate, optional
        Non-negative scalar or axis-specific padding.
    metadata : Mapping[str, Any], optional
        Additional bounding-box metadata.

    Returns
    -------
    BoundingBox
        Structured axis-aligned bounding box.
    """

    coordinate_matrix = get_coordinates(
        coordinates,
        scene=scene,
        name="Bounding coordinates",
        minimum_rows=1,
        allow_empty=False,
        ignore_none=False,
        require_finite=True,
        copy=False,
    )

    padding_vector = _validate_padding(
        padding
    )

    minimum_coordinate = (
        np.min(
            coordinate_matrix,
            axis=0,
        )
        - padding_vector
    )

    maximum_coordinate = (
        np.max(
            coordinate_matrix,
            axis=0,
        )
        + padding_vector
    )

    if metadata is None:
        metadata_value: Dict[str, Any] = {}

    elif isinstance(
        metadata,
        Mapping,
    ):
        metadata_value = dict(
            metadata
        )

    else:
        raise TypeError(
            "metadata must be a mapping or None."
        )

    metadata_value.setdefault(
        "coordinate_space",
        (
            "scene"
            if scene
            else "model"
        ),
    )

    return BoundingBox(
        minimum=minimum_coordinate,
        maximum=maximum_coordinate,
        padding=padding_vector,
        point_count=int(
            coordinate_matrix.shape[0]
        ),
        metadata=metadata_value,
    )


# -----------------------------------------------------------------------------
# Structured-result validation
# -----------------------------------------------------------------------------

def is_structural_result(
    value: Any,
) -> bool:
    """
    Return whether an object behaves as a structured geometry result.

    Parameters
    ----------
    value : Any
        Object to inspect.

    Returns
    -------
    bool
        ``True`` when the object implements a callable ``to_dict()`` method.
    """

    return callable(
        getattr(
            value,
            "to_dict",
            None,
        )
    )


def validate_structural_result(
    value: Any,
    *,
    name: str = "result",
) -> StructuralResult:
    """
    Validate that an object provides structured-result serialization.

    Parameters
    ----------
    value : Any
        Object to validate.
    name : str, optional
        Parameter name used in validation messages.

    Returns
    -------
    StructuralResult
        Validated result object.

    Raises
    ------
    TypeError
        If the object does not implement ``to_dict()``.
    """

    if not is_structural_result(
        value
    ):
        raise TypeError(
            f"{name} must implement a callable "
            "to_dict() method."
        )

    return value


# -----------------------------------------------------------------------------
# Public module interface
# -----------------------------------------------------------------------------

_SECTION_14_PUBLIC_NAMES = [
    "StructuralResult",
    "BoundingBox",
    "create_bounding_box",
    "structural_result_to_dict",
    "is_structural_result",
    "validate_structural_result",
]

_extend_public_names(_SECTION_14_PUBLIC_NAMES)


# =============================================================================
# End of Section 14
# =============================================================================



# =============================================================================
# Section 15 — Self Tests
# =============================================================================


# -----------------------------------------------------------------------------
# Self-test helpers
# -----------------------------------------------------------------------------

def _assert_true(
    condition: Any,
    message: str,
) -> None:
    """
    Assert that a condition is true.

    Parameters
    ----------
    condition : Any
        Condition to evaluate.
    message : str
        Error message raised when the assertion fails.

    Raises
    ------
    AssertionError
        If the condition evaluates to ``False``.
    """

    if not bool(
        condition
    ):
        raise AssertionError(
            message
        )


def _assert_equal(
    actual: Any,
    expected: Any,
    message: str,
) -> None:
    """
    Assert exact equality between two values.

    Parameters
    ----------
    actual : Any
        Observed value.
    expected : Any
        Expected value.
    message : str
        Error message raised when values differ.

    Raises
    ------
    AssertionError
        If the values are not equal.
    """

    if actual != expected:
        raise AssertionError(
            f"{message}\n"
            f"Expected: {expected!r}\n"
            f"Actual:   {actual!r}"
        )


def _assert_close(
    actual: Any,
    expected: Any,
    *,
    tolerance: float = 1e-7,
    message: str = "Values are not sufficiently close.",
) -> None:
    """
    Assert numerical closeness between scalar or array-like values.

    Parameters
    ----------
    actual : Any
        Observed numeric value.
    expected : Any
        Expected numeric value.
    tolerance : float, optional
        Absolute and relative comparison tolerance.
    message : str, optional
        Error message raised when values differ.

    Raises
    ------
    AssertionError
        If values are not numerically close.
    """

    try:
        actual_array = np.asarray(
            actual,
            dtype=np.float64,
        )

        expected_array = np.asarray(
            expected,
            dtype=np.float64,
        )

    except (
        TypeError,
        ValueError,
        OverflowError,
    ) as error:
        raise AssertionError(
            f"{message}\n"
            "Values could not be converted to "
            "numeric arrays."
        ) from error

    if not np.allclose(
        actual_array,
        expected_array,
        rtol=tolerance,
        atol=tolerance,
        equal_nan=False,
    ):
        maximum_difference = float(
            np.max(
                np.abs(
                    actual_array
                    - expected_array
                )
            )
        )

        raise AssertionError(
            f"{message}\n"
            f"Expected: {expected_array!r}\n"
            f"Actual:   {actual_array!r}\n"
            f"Maximum difference: "
            f"{maximum_difference:.12g}"
        )


def _assert_shape(
    value: Any,
    expected_shape: Tuple[
        int,
        ...,
    ],
    message: str,
) -> None:
    """
    Assert the shape of an array-like value.

    Parameters
    ----------
    value : Any
        Array-like value.
    expected_shape : tuple of int
        Required shape.
    message : str
        Error message raised when the shape differs.

    Raises
    ------
    AssertionError
        If the observed shape differs from the expected shape.
    """

    actual_shape = np.asarray(
        value
    ).shape

    if actual_shape != expected_shape:
        raise AssertionError(
            f"{message}\n"
            f"Expected shape: {expected_shape}\n"
            f"Actual shape:   {actual_shape}"
        )


def _assert_raises(
    exception_type: type,
    function: Callable[
        ...,
        Any,
    ],
    *args: Any,
    message: str = (
        "Expected exception was not raised."
    ),
    **kwargs: Any,
) -> BaseException:
    """
    Assert that a callable raises a selected exception.

    Parameters
    ----------
    exception_type : type
        Expected exception class.
    function : callable
        Function to execute.
    *args : Any
        Positional arguments passed to the function.
    message : str, optional
        Error message used when no expected exception is raised.
    **kwargs : Any
        Keyword arguments passed to the function.

    Returns
    -------
    BaseException
        Captured exception.

    Raises
    ------
    AssertionError
        If no exception is raised or a different exception type is raised.
    """

    try:
        function(
            *args,
            **kwargs,
        )

    except exception_type as error:
        return error

    except Exception as error:
        raise AssertionError(
            f"{message}\n"
            f"Expected exception: "
            f"{exception_type.__name__}\n"
            f"Actual exception: "
            f"{type(error).__name__}: {error}"
        ) from error

    raise AssertionError(
        f"{message}\n"
        f"Expected exception: "
        f"{exception_type.__name__}"
    )


def _run_test_group(
    name: str,
    function: Callable[
        [],
        None,
    ],
    *,
    verbose: bool = True,
) -> Tuple[
    str,
    bool,
    Optional[str],
]:
    """
    Execute one self-test group.

    Parameters
    ----------
    name : str
        Test-group name.
    function : callable
        Test function requiring no arguments.
    verbose : bool, optional
        Whether status messages should be printed.

    Returns
    -------
    tuple
        ``(name, passed, error_message)``.
    """

    try:
        function()

    except Exception as error:
        if verbose:
            print(
                f"[FAIL] {name}"
            )

            print(
                f"       {type(error).__name__}: "
                f"{error}"
            )

        return (
            name,
            False,
            (
                f"{type(error).__name__}: "
                f"{error}"
            ),
        )

    if verbose:
        print(
            f"[PASS] {name}"
        )

    return (
        name,
        True,
        None,
    )


# -----------------------------------------------------------------------------
# Coordinate and vector tests
# -----------------------------------------------------------------------------

def _test_coordinates_and_vectors() -> None:
    """
    Test coordinate conversion and vector operations.
    """

    coordinate = as_coordinate(
        [
            1.0,
            2.0,
            3.0,
        ]
    )

    _assert_shape(
        coordinate,
        (
            3,
        ),
        "Coordinate conversion returned "
        "an invalid shape.",
    )

    _assert_true(
        coordinate.dtype
        == np.float64,
        "Coordinates must use numpy.float64.",
    )

    coordinate_matrix = (
        as_coordinate_matrix(
            [
                [
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    1.0,
                    2.0,
                    3.0,
                ],
            ]
        )
    )

    _assert_shape(
        coordinate_matrix,
        (
            2,
            3,
        ),
        "Coordinate matrix conversion returned "
        "an invalid shape.",
    )

    vector = vector_between(
        [
            1.0,
            1.0,
            1.0,
        ],
        [
            4.0,
            5.0,
            1.0,
        ],
    )

    _assert_close(
        vector,
        [
            3.0,
            4.0,
            0.0,
        ],
        message="vector_between() returned "
        "an incorrect vector.",
    )

    _assert_close(
        vector_norm(
            vector
        ),
        5.0,
        message="vector_norm() returned "
        "an incorrect magnitude.",
    )

    _assert_close(
        vector_norm(
            vector,
            squared=True,
        ),
        25.0,
        message="Squared vector norm is incorrect.",
    )

    normalized = unit_vector(
        vector
    )

    _assert_close(
        vector_norm(
            normalized
        ),
        1.0,
        message="unit_vector() did not produce "
        "a unit-length vector.",
    )

    _assert_close(
        dot_product(
            [
                1.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                1.0,
                0.0,
            ],
        ),
        0.0,
        message="Orthogonal vectors must have "
        "zero dot product.",
    )

    _assert_close(
        cross_product(
            [
                1.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                1.0,
                0.0,
            ],
        ),
        [
            0.0,
            0.0,
            1.0,
        ],
        message="cross_product() returned "
        "an incorrect orientation.",
    )

    projected = project_vector(
        [
            2.0,
            2.0,
            0.0,
        ],
        [
            1.0,
            0.0,
            0.0,
        ],
    )

    rejected = reject_vector(
        [
            2.0,
            2.0,
            0.0,
        ],
        [
            1.0,
            0.0,
            0.0,
        ],
    )

    _assert_close(
        projected,
        [
            2.0,
            0.0,
            0.0,
        ],
        message="Vector projection is incorrect.",
    )

    _assert_close(
        rejected,
        [
            0.0,
            2.0,
            0.0,
        ],
        message="Vector rejection is incorrect.",
    )

    projected_point = project_point_on_line(
        [
            1.0,
            2.0,
            0.0,
        ],
        [
            0.0,
            0.0,
            0.0,
        ],
        line_end=[
            3.0,
            0.0,
            0.0,
        ],
    )

    _assert_close(
        projected_point,
        [
            1.0,
            0.0,
            0.0,
        ],
        message="Point projection onto line "
        "is incorrect.",
    )

    _assert_raises(
        ValueError,
        unit_vector,
        [
            0.0,
            0.0,
            0.0,
        ],
        message="unit_vector() must reject "
        "a zero-length vector.",
    )


# -----------------------------------------------------------------------------
# Distance and angle tests
# -----------------------------------------------------------------------------

def _test_distances_and_angles() -> None:
    """
    Test distance, angle and dihedral calculations.
    """

    first_point = np.asarray(
        [
            0.0,
            0.0,
            0.0,
        ],
        dtype=np.float64,
    )

    second_point = np.asarray(
        [
            3.0,
            4.0,
            0.0,
        ],
        dtype=np.float64,
    )

    _assert_close(
        squared_distance(
            first_point,
            second_point,
        ),
        25.0,
        message="squared_distance() returned "
        "an incorrect value.",
    )

    _assert_close(
        atom_distance(
            first_point,
            second_point,
        ),
        5.0,
        message="atom_distance() returned "
        "an incorrect value.",
    )

    first_collection = np.asarray(
        [
            [
                0.0,
                0.0,
                0.0,
            ],
            [
                2.0,
                0.0,
                0.0,
            ],
        ],
        dtype=np.float64,
    )

    second_collection = np.asarray(
        [
            [
                1.0,
                0.0,
                0.0,
            ],
            [
                5.0,
                0.0,
                0.0,
            ],
        ],
        dtype=np.float64,
    )

    distances = distance_matrix(
        first_collection,
        second_collection,
    )

    _assert_close(
        distances,
        [
            [
                1.0,
                5.0,
            ],
            [
                1.0,
                3.0,
            ],
        ],
        message="distance_matrix() returned "
        "incorrect distances.",
    )

    (
        closest_first,
        closest_second,
        closest_distance_value,
        closest_indices,
    ) = closest_point_pair(
        first_collection,
        second_collection,
        return_distance=True,
        return_indices=True,
    )

    _assert_close(
        closest_distance_value,
        1.0,
        message="closest_point_pair() returned "
        "an incorrect minimum distance.",
    )

    _assert_equal(
        closest_indices,
        (
            0,
            0,
        ),
        "closest_point_pair() returned "
        "unexpected indices.",
    )

    _assert_close(
        closest_first,
        first_collection[0],
        message="Incorrect first closest point.",
    )

    _assert_close(
        closest_second,
        second_collection[0],
        message="Incorrect second closest point.",
    )

    _assert_close(
        minimum_distance(
            first_collection,
            second_collection,
        ),
        1.0,
        message="minimum_distance() returned "
        "an incorrect value.",
    )

    _assert_close(
        point_line_distance(
            [
                1.0,
                2.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
            ],
            line_end=[
                3.0,
                0.0,
                0.0,
            ],
        ),
        2.0,
        message="point_line_distance() returned "
        "an incorrect value.",
    )

    _assert_close(
        vector_angle(
            [
                1.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                1.0,
                0.0,
            ],
        ),
        90.0,
        message="vector_angle() returned "
        "an incorrect right angle.",
    )

    _assert_close(
        bond_angle(
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
        ),
        90.0,
        message="bond_angle() returned "
        "an incorrect angle.",
    )

    dihedral = dihedral_angle(
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
        [
            0.0,
            1.0,
            1.0,
        ],
    )

    _assert_close(
        abs(
            dihedral
        ),
        90.0,
        message="dihedral_angle() returned "
        "an incorrect magnitude.",
    )

    torsion = torsion_angle(
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
        [
            0.0,
            1.0,
            1.0,
        ],
        signed=False,
        positive=False,
    )

    _assert_close(
        torsion,
        90.0,
        message="torsion_angle() returned "
        "an incorrect unsigned angle.",
    )


# -----------------------------------------------------------------------------
# Plane tests
# -----------------------------------------------------------------------------

def _test_planes() -> None:
    """
    Test plane fitting and plane-related operations.
    """

    planar_points = np.asarray(
        [
            [
                -1.0,
                -1.0,
                2.0,
            ],
            [
                1.0,
                -1.0,
                2.0,
            ],
            [
                1.0,
                1.0,
                2.0,
            ],
            [
                -1.0,
                1.0,
                2.0,
            ],
        ],
        dtype=np.float64,
    )

    fitted_plane = fit_plane(
        planar_points,
        reference_normal=[
            0.0,
            0.0,
            1.0,
        ],
    )

    _assert_true(
        isinstance(
            fitted_plane,
            Plane,
        ),
        "fit_plane() must return Plane.",
    )

    _assert_close(
        fitted_plane.point,
        [
            0.0,
            0.0,
            2.0,
        ],
        message="Plane point is incorrect.",
    )

    _assert_close(
        fitted_plane.normal,
        [
            0.0,
            0.0,
            1.0,
        ],
        message="Plane normal is incorrect.",
    )

    _assert_close(
        fitted_plane.rmsd,
        0.0,
        message="Perfectly planar coordinates "
        "must have zero plane RMSD.",
    )

    _assert_close(
        plane_normal(
            planar_points,
            reference_normal=[
                0.0,
                0.0,
                1.0,
            ],
        ),
        [
            0.0,
            0.0,
            1.0,
        ],
        message="plane_normal() returned "
        "an incorrect normal.",
    )

    _assert_close(
        point_plane_distance(
            [
                0.0,
                0.0,
                5.0,
            ],
            fitted_plane,
        ),
        3.0,
        message="point_plane_distance() returned "
        "an incorrect value.",
    )

    _assert_close(
        project_point_on_plane(
            [
                1.0,
                1.0,
                5.0,
            ],
            fitted_plane,
        ),
        [
            1.0,
            1.0,
            2.0,
        ],
        message="project_point_on_plane() returned "
        "an incorrect point.",
    )

    second_plane = Plane(
        point=np.zeros(
            3,
            dtype=np.float64,
        ),
        normal=[
            1.0,
            0.0,
            0.0,
        ],
        point_count=3,
    )

    _assert_close(
        angle_between_planes(
            fitted_plane,
            second_plane,
        ),
        90.0,
        message="angle_between_planes() returned "
        "an incorrect angle.",
    )

    plane_dictionary = (
        fitted_plane.to_dict()
    )

    _assert_true(
        isinstance(
            plane_dictionary,
            dict,
        ),
        "Plane.to_dict() must return a dictionary.",
    )


# -----------------------------------------------------------------------------
# Ring and pi-interaction tests
# -----------------------------------------------------------------------------

def _synthetic_hexagonal_ring(
    *,
    center: Coordinate,
    radius: float = 1.4,
    plane: str = "xy",
) -> FloatArray:
    """
    Construct a synthetic regular six-membered ring.

    Parameters
    ----------
    center : Coordinate
        Ring center.
    radius : float, optional
        Ring radius.
    plane : {"xy", "xz", "yz"}, optional
        Ring plane.

    Returns
    -------
    numpy.ndarray
        Ring coordinates with shape ``(6, 3)``.
    """

    center_coordinate = as_coordinate(
        center,
        scene=False,
        name="Ring center",
        copy=False,
    )

    angles = np.linspace(
        0.0,
        2.0 * np.pi,
        num=6,
        endpoint=False,
        dtype=np.float64,
    )

    first_component = (
        radius
        * np.cos(
            angles
        )
    )

    second_component = (
        radius
        * np.sin(
            angles
        )
    )

    coordinates = np.zeros(
        (
            6,
            3,
        ),
        dtype=np.float64,
    )

    if plane == "xy":
        coordinates[
            :,
            0,
        ] = first_component

        coordinates[
            :,
            1,
        ] = second_component

    elif plane == "xz":
        coordinates[
            :,
            0,
        ] = first_component

        coordinates[
            :,
            2,
        ] = second_component

    elif plane == "yz":
        coordinates[
            :,
            1,
        ] = first_component

        coordinates[
            :,
            2,
        ] = second_component

    else:
        raise ValueError(
            "plane must be 'xy', 'xz' or 'yz'."
        )

    coordinates += center_coordinate

    return coordinates


def _test_rings_and_pi_interactions() -> None:
    """
    Test aromatic-ring and pi-interaction geometry.
    """

    first_ring_coordinates = (
        _synthetic_hexagonal_ring(
            center=[
                0.0,
                0.0,
                0.0,
            ],
        )
    )

    second_ring_coordinates = (
        _synthetic_hexagonal_ring(
            center=[
                0.0,
                0.0,
                3.5,
            ],
        )
    )

    first_ring = RingGeometry(
        coordinates=(
            first_ring_coordinates
        )
    )

    second_ring = RingGeometry(
        coordinates=(
            second_ring_coordinates
        )
    )

    _assert_close(
        first_ring.centroid,
        [
            0.0,
            0.0,
            0.0,
        ],
        message="Ring centroid is incorrect.",
    )

    _assert_close(
        first_ring.radius,
        1.4,
        message="Ring radius is incorrect.",
    )

    _assert_close(
        first_ring.planarity_rmsd,
        0.0,
        message="Synthetic ring must be planar.",
    )

    _assert_close(
        ring_centroid(
            first_ring_coordinates
        ),
        [
            0.0,
            0.0,
            0.0,
        ],
        message="ring_centroid() returned "
        "an incorrect result.",
    )

    _assert_close(
        abs(
            ring_normal(
                first_ring_coordinates
            )[2]
        ),
        1.0,
        message="ring_normal() returned "
        "an incorrect orientation.",
    )

    _assert_close(
        ring_radius(
            first_ring_coordinates
        ),
        1.4,
        message="ring_radius() returned "
        "an incorrect value.",
    )

    _assert_close(
        ring_planarity(
            first_ring_coordinates
        ),
        0.0,
        message="ring_planarity() returned "
        "an incorrect value.",
    )

    stacking = pi_stack_geometry(
        first_ring,
        second_ring,
        maximum_centroid_distance=5.0,
    )

    _assert_true(
        isinstance(
            stacking,
            PiStackGeometry,
        ),
        "pi_stack_geometry() must return "
        "PiStackGeometry.",
    )

    _assert_close(
        stacking.centroid_distance,
        3.5,
        message="Pi-stack centroid distance "
        "is incorrect.",
    )

    _assert_close(
        stacking.plane_angle,
        0.0,
        message="Parallel rings must have "
        "zero plane angle.",
    )

    _assert_equal(
        stacking.classification,
        "parallel",
        "Synthetic stacked rings should be "
        "classified as parallel.",
    )

    _assert_true(
        stacking.distance_compatible,
        "Synthetic stacked rings should satisfy "
        "the distance criterion.",
    )

    cation_geometry = cation_pi_geometry(
        first_ring,
        [
            0.0,
            0.0,
            2.0,
        ],
        maximum_distance=6.0,
        maximum_lateral_offset=2.0,
    )

    _assert_close(
        cation_geometry[
            "centroid_distance"
        ],
        2.0,
        message="Cation-pi centroid distance "
        "is incorrect.",
    )

    _assert_close(
        cation_geometry[
            "lateral_offset"
        ],
        0.0,
        message="Centered cation must have "
        "zero lateral offset.",
    )

    _assert_true(
        cation_geometry[
            "geometry_compatible"
        ],
        "Centered synthetic cation-pi geometry "
        "should be compatible.",
    )


# -----------------------------------------------------------------------------
# Hydrogen-bond tests
# -----------------------------------------------------------------------------

def _test_hydrogen_bonds() -> None:
    """
    Test hydrogen-bond geometry.
    """

    donor = np.asarray(
        [
            0.0,
            0.0,
            0.0,
        ],
        dtype=np.float64,
    )

    hydrogen = np.asarray(
        [
            1.0,
            0.0,
            0.0,
        ],
        dtype=np.float64,
    )

    acceptor = np.asarray(
        [
            2.8,
            0.0,
            0.0,
        ],
        dtype=np.float64,
    )

    angle_value = (
        donor_hydrogen_acceptor_angle(
            donor,
            hydrogen,
            acceptor,
        )
    )

    _assert_close(
        angle_value,
        180.0,
        message="Linear hydrogen bond must have "
        "a 180-degree angle.",
    )

    geometry = hydrogen_bond_geometry(
        donor,
        acceptor,
        hydrogen,
    )

    _assert_true(
        isinstance(
            geometry,
            HydrogenBondGeometry,
        ),
        "hydrogen_bond_geometry() must return "
        "HydrogenBondGeometry.",
    )

    _assert_close(
        geometry.donor_acceptor_distance,
        2.8,
        message="D-A distance is incorrect.",
    )

    _assert_close(
        geometry.hydrogen_acceptor_distance,
        1.8,
        message="H-A distance is incorrect.",
    )

    _assert_close(
        geometry.donor_hydrogen_distance,
        1.0,
        message="D-H distance is incorrect.",
    )

    _assert_close(
        geometry.donor_hydrogen_acceptor_angle,
        180.0,
        message="D-H-A angle is incorrect.",
    )

    _assert_true(
        geometry.geometry_compatible,
        "Synthetic hydrogen bond should be "
        "geometrically compatible.",
    )

    distance_only_geometry = (
        hydrogen_bond_geometry(
            donor,
            acceptor,
        )
    )

    _assert_true(
        distance_only_geometry.is_distance_only,
        "Hydrogen-free analysis must be marked "
        "as distance-only.",
    )

    _assert_true(
        distance_only_geometry.angle_compatible
        is None,
        "Angle compatibility must be None "
        "without an explicit hydrogen.",
    )

    _assert_raises(
        ValueError,
        hydrogen_bond_geometry,
        donor,
        acceptor,
        require_explicit_hydrogen=True,
        message="Explicit-hydrogen mode must reject "
        "a missing hydrogen.",
    )


# -----------------------------------------------------------------------------
# Contact tests
# -----------------------------------------------------------------------------

def _test_contacts() -> None:
    """
    Test contact geometry and closest-pair detection.
    """

    first_atom = np.asarray(
        [
            0.0,
            0.0,
            0.0,
        ],
        dtype=np.float64,
    )

    second_atom = np.asarray(
        [
            3.0,
            0.0,
            0.0,
        ],
        dtype=np.float64,
    )

    contact = contact_geometry(
        first_atom,
        second_atom,
        cutoff=4.0,
    )

    _assert_true(
        isinstance(
            contact,
            ContactGeometry,
        ),
        "contact_geometry() must return "
        "ContactGeometry.",
    )

    _assert_close(
        contact.distance,
        3.0,
        message="Contact distance is incorrect.",
    )

    _assert_true(
        contact.is_contact,
        "The pair should lie inside "
        "the contact cutoff.",
    )

    _assert_close(
        contact.margin_to_cutoff,
        1.0,
        message="Contact cutoff margin is incorrect.",
    )

    _assert_close(
        contact.midpoint,
        [
            1.5,
            0.0,
            0.0,
        ],
        message="Contact midpoint is incorrect.",
    )

    first_collection = [
        np.asarray(
            [
                0.0,
                0.0,
                0.0,
            ],
            dtype=np.float64,
        ),
        np.asarray(
            [
                10.0,
                0.0,
                0.0,
            ],
            dtype=np.float64,
        ),
    ]

    second_collection = [
        np.asarray(
            [
                3.0,
                0.0,
                0.0,
            ],
            dtype=np.float64,
        ),
        np.asarray(
            [
                20.0,
                0.0,
                0.0,
            ],
            dtype=np.float64,
        ),
    ]

    closest_contact = closest_atoms(
        first_collection,
        second_collection,
        cutoff=4.0,
    )

    _assert_close(
        closest_contact.distance,
        3.0,
        message="closest_atoms() returned "
        "an incorrect distance.",
    )

    _assert_equal(
        closest_contact.index_1,
        0,
        "closest_atoms() returned "
        "an incorrect first index.",
    )

    _assert_equal(
        closest_contact.index_2,
        0,
        "closest_atoms() returned "
        "an incorrect second index.",
    )

    (
        closest_with_matrix,
        complete_matrix,
    ) = closest_atoms(
        first_collection,
        second_collection,
        return_distance_matrix=True,
    )

    _assert_shape(
        complete_matrix,
        (
            2,
            2,
        ),
        "Closest-contact distance matrix "
        "has an invalid shape.",
    )

    _assert_close(
        closest_with_matrix.distance,
        np.min(
            complete_matrix
        ),
        message="Closest contact does not match "
        "the minimum matrix value.",
    )

    self_contact = closest_atoms(
        first_collection,
        first_collection,
        exclude_identical_objects=True,
        exclude_same_index=True,
    )

    _assert_true(
        self_contact.distance > 0.0,
        "Self-comparison exclusions must prevent "
        "zero-distance self-pairs.",
    )


# -----------------------------------------------------------------------------
# RMSD and alignment tests
# -----------------------------------------------------------------------------

def _test_rmsd_and_alignment() -> None:
    """
    Test RMSD, centering and Kabsch alignment.
    """

    reference = np.asarray(
        [
            [
                0.0,
                0.0,
                0.0,
            ],
            [
                1.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                1.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                1.0,
            ],
        ],
        dtype=np.float64,
    )

    known_rotation = np.asarray(
        [
            [
                0.0,
                -1.0,
                0.0,
            ],
            [
                1.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                1.0,
            ],
        ],
        dtype=np.float64,
    )

    known_translation = np.asarray(
        [
            5.0,
            -3.0,
            2.0,
        ],
        dtype=np.float64,
    )

    mobile = (
        reference
        @ known_rotation
        + known_translation
    )

    initial_rmsd = calculate_rmsd(
        reference,
        mobile,
    )

    _assert_true(
        initial_rmsd > 1.0,
        "Synthetic unaligned RMSD should "
        "be clearly non-zero.",
    )

    centered_reference, centroid = (
        center_coordinates(
            reference,
            return_centroid=True,
        )
    )

    _assert_close(
        np.mean(
            centered_reference,
            axis=0,
        ),
        np.zeros(
            3,
            dtype=np.float64,
        ),
        message="Centered coordinates must have "
        "zero geometric mean.",
    )

    _assert_close(
        centroid,
        np.mean(
            reference,
            axis=0,
        ),
        message="center_coordinates() returned "
        "an incorrect centroid.",
    )

    rotation = kabsch_rotation(
        reference,
        mobile,
    )

    _assert_shape(
        rotation,
        (
            3,
            3,
        ),
        "Kabsch rotation has an invalid shape.",
    )

    _assert_close(
        rotation.T
        @ rotation,
        np.eye(
            3,
            dtype=np.float64,
        ),
        message="Kabsch rotation must be orthogonal.",
    )

    _assert_close(
        np.linalg.det(
            rotation
        ),
        1.0,
        message="Kabsch rotation must have "
        "determinant +1.",
    )

    alignment = kabsch_alignment(
        reference,
        mobile,
    )

    _assert_true(
        isinstance(
            alignment,
            AlignmentResult,
        ),
        "kabsch_alignment() must return "
        "AlignmentResult.",
    )

    _assert_close(
        alignment.aligned_coordinates,
        reference,
        tolerance=1e-6,
        message="Kabsch alignment did not recover "
        "the reference coordinates.",
    )

    _assert_close(
        alignment.final_rmsd,
        0.0,
        tolerance=1e-6,
        message="Rigidly transformed coordinates "
        "must align with near-zero RMSD.",
    )

    _assert_true(
        alignment.initial_rmsd
        > alignment.final_rmsd,
        "Alignment must reduce RMSD.",
    )

    transformed = alignment.transform(
        mobile
    )

    _assert_close(
        transformed,
        reference,
        tolerance=1e-6,
        message="AlignmentResult.transform() "
        "returned incorrect coordinates.",
    )

    restored = alignment.inverse_transform(
        transformed
    )

    _assert_close(
        restored,
        mobile,
        tolerance=1e-6,
        message="Alignment inverse transformation "
        "did not restore mobile coordinates.",
    )

    aligned_value = aligned_rmsd(
        reference,
        mobile,
    )

    _assert_close(
        aligned_value,
        0.0,
        tolerance=1e-6,
        message="aligned_rmsd() returned "
        "an incorrect value.",
    )

    weighted_value = aligned_rmsd(
        reference,
        mobile,
        weights=[
            1.0,
            2.0,
            3.0,
            4.0,
        ],
    )

    _assert_close(
        weighted_value,
        0.0,
        tolerance=1e-6,
        message="Weighted aligned RMSD "
        "should be near zero.",
    )


# -----------------------------------------------------------------------------
# Bounding-geometry tests
# -----------------------------------------------------------------------------

def _test_bounding_geometry() -> None:
    """
    Test bounding boxes, centers and radius of gyration.
    """

    coordinates = np.asarray(
        [
            [
                -1.0,
                -2.0,
                -3.0,
            ],
            [
                3.0,
                4.0,
                5.0,
            ],
            [
                1.0,
                0.0,
                2.0,
            ],
        ],
        dtype=np.float64,
    )

    minimum, maximum = bounding_box(
        coordinates
    )

    _assert_close(
        minimum,
        [
            -1.0,
            -2.0,
            -3.0,
        ],
        message="Bounding-box minimum is incorrect.",
    )

    _assert_close(
        maximum,
        [
            3.0,
            4.0,
            5.0,
        ],
        message="Bounding-box maximum is incorrect.",
    )

    (
        padded_minimum,
        padded_maximum,
        corners,
    ) = bounding_box(
        coordinates,
        padding=1.0,
        return_corners=True,
    )

    _assert_close(
        padded_minimum,
        [
            -2.0,
            -3.0,
            -4.0,
        ],
        message="Padded bounding-box minimum "
        "is incorrect.",
    )

    _assert_close(
        padded_maximum,
        [
            4.0,
            5.0,
            6.0,
        ],
        message="Padded bounding-box maximum "
        "is incorrect.",
    )

    _assert_shape(
        corners,
        (
            8,
            3,
        ),
        "Bounding-box corners have "
        "an invalid shape.",
    )

    box_center = bounding_box_center(
        minimum=minimum,
        maximum=maximum,
    )

    _assert_close(
        box_center,
        [
            1.0,
            1.0,
            1.0,
        ],
        message="Bounding-box center is incorrect.",
    )

    size, volume, diagonal = (
        bounding_box_size(
            minimum=minimum,
            maximum=maximum,
            return_volume=True,
            return_diagonal=True,
        )
    )

    _assert_close(
        size,
        [
            4.0,
            6.0,
            8.0,
        ],
        message="Bounding-box size is incorrect.",
    )

    _assert_close(
        volume,
        192.0,
        message="Bounding-box volume is incorrect.",
    )

    _assert_close(
        diagonal,
        math.sqrt(
            116.0
        ),
        message="Bounding-box diagonal is incorrect.",
    )

    geometric_center_value = (
        geometric_center(
            coordinates
        )
    )

    _assert_close(
        geometric_center_value,
        np.mean(
            coordinates,
            axis=0,
        ),
        message="geometric_center() returned "
        "an incorrect value.",
    )

    masses = np.asarray(
        [
            1.0,
            2.0,
            1.0,
        ],
        dtype=np.float64,
    )

    (
        mass_center,
        total_mass,
    ) = center_of_mass(
        coordinates,
        masses=masses,
        return_total_mass=True,
    )

    expected_mass_center = np.sum(
        coordinates
        * masses[
            :,
            np.newaxis,
        ],
        axis=0,
    ) / np.sum(
        masses
    )

    _assert_close(
        mass_center,
        expected_mass_center,
        message="center_of_mass() returned "
        "an incorrect value.",
    )

    _assert_close(
        total_mass,
        4.0,
        message="Total mass is incorrect.",
    )

    simple_coordinates = np.asarray(
        [
            [
                -1.0,
                0.0,
                0.0,
            ],
            [
                1.0,
                0.0,
                0.0,
            ],
        ],
        dtype=np.float64,
    )

    unweighted_rg = radius_of_gyration(
        simple_coordinates,
        mass_weighted=False,
    )

    _assert_close(
        unweighted_rg,
        1.0,
        message="Unweighted radius of gyration "
        "is incorrect.",
    )

    weighted_rg = radius_of_gyration(
        simple_coordinates,
        masses=[
            1.0,
            1.0,
        ],
        mass_weighted=True,
    )

    _assert_close(
        weighted_rg,
        1.0,
        message="Mass-weighted radius of gyration "
        "is incorrect.",
    )


# -----------------------------------------------------------------------------
# Structured-result tests
# -----------------------------------------------------------------------------

def _test_structured_results() -> None:
    """
    Test structured result serialization.
    """

    coordinates = np.asarray(
        [
            [
                -1.0,
                -2.0,
                -3.0,
            ],
            [
                3.0,
                4.0,
                5.0,
            ],
        ],
        dtype=np.float64,
    )

    structured_box = create_bounding_box(
        coordinates,
        padding=1.0,
        metadata={
            "test": True,
        },
    )

    _assert_true(
        isinstance(
            structured_box,
            BoundingBox,
        ),
        "create_bounding_box() must return "
        "BoundingBox.",
    )

    _assert_close(
        structured_box.minimum,
        [
            -2.0,
            -3.0,
            -4.0,
        ],
        message="Structured bounding-box minimum "
        "is incorrect.",
    )

    _assert_close(
        structured_box.maximum,
        [
            4.0,
            5.0,
            6.0,
        ],
        message="Structured bounding-box maximum "
        "is incorrect.",
    )

    _assert_close(
        structured_box.center,
        [
            1.0,
            1.0,
            1.0,
        ],
        message="Structured bounding-box center "
        "is incorrect.",
    )

    _assert_true(
        structured_box.contains(
            [
                0.0,
                0.0,
                0.0,
            ]
        ),
        "BoundingBox.contains() rejected "
        "an internal point.",
    )

    _assert_true(
        not structured_box.contains(
            [
                20.0,
                0.0,
                0.0,
            ]
        ),
        "BoundingBox.contains() accepted "
        "an external point.",
    )

    box_dictionary = (
        structured_box.to_dict(
            include_corners=True,
        )
    )

    _assert_true(
        isinstance(
            box_dictionary,
            dict,
        ),
        "BoundingBox.to_dict() must return "
        "a dictionary.",
    )

    _assert_true(
        "corners"
        in box_dictionary,
        "BoundingBox.to_dict() did not include "
        "requested corners.",
    )

    serialized_box = (
        structural_result_to_dict(
            structured_box,
            serialization_options={
                "include_corners": True,
            },
        )
    )

    _assert_true(
        isinstance(
            serialized_box[
                "minimum"
            ],
            list,
        ),
        "Structural serialization must convert "
        "NumPy coordinates to lists.",
    )

    _assert_true(
        is_structural_result(
            structured_box
        ),
        "BoundingBox must satisfy the "
        "structured-result interface.",
    )

    validated_result = (
        validate_structural_result(
            structured_box
        )
    )

    _assert_true(
        validated_result
        is structured_box,
        "validate_structural_result() must "
        "return the original object.",
    )

    _assert_raises(
        TypeError,
        validate_structural_result,
        object(),
        message="Objects without to_dict() must "
        "be rejected.",
    )


# -----------------------------------------------------------------------------
# Integration and compatibility tests
# -----------------------------------------------------------------------------

class _SyntheticChimeraXAtom:
    """Minimal ChimeraX-like atom used by integration tests."""

    def __init__(
        self,
        coordinate: Coordinate,
        *,
        scene_offset: Coordinate = (
            10.0,
            0.0,
            0.0,
        ),
        name: str = "C",
    ) -> None:
        self.coord = as_coordinate(
            coordinate,
            scene=False,
            copy=True,
        )
        self.scene_coord = (
            self.coord
            + as_coordinate(
                scene_offset,
                scene=False,
                copy=False,
            )
        )
        self.name = str(name)
        self.atomspec = f"@{self.name}"


class _SyntheticChimeraXAtoms:
    """Minimal ChimeraX-like atom collection used by integration tests."""

    def __init__(
        self,
        atoms: Sequence[
            _SyntheticChimeraXAtom
        ],
    ) -> None:
        self._atoms = list(
            atoms
        )

    @property
    def coords(
        self,
    ) -> FloatArray:
        return np.vstack(
            [
                atom.coord
                for atom in self._atoms
            ]
        )

    @property
    def scene_coords(
        self,
    ) -> FloatArray:
        return np.vstack(
            [
                atom.scene_coord
                for atom in self._atoms
            ]
        )

    def __iter__(
        self,
    ) -> Any:
        return iter(
            self._atoms
        )


@dataclass
class _SyntheticDockModel:
    """Minimal DockModel contract used by integration tests."""

    name: str
    pose: Any = None
    receptor: Any = None
    ligand: Any = None
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
    files: Dict[str, Any] = field(
        default_factory=dict
    )
    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    @staticmethod
    def _serialize_value(
        value: Any,
    ) -> Any:
        if value is None or isinstance(
            value,
            (
                str,
                int,
                float,
                bool,
            ),
        ):
            return value

        if isinstance(
            value,
            np.generic,
        ):
            return value.item()

        if isinstance(
            value,
            np.ndarray,
        ):
            return value.tolist()

        if isinstance(
            value,
            Mapping,
        ):
            return {
                str(key): (
                    _SyntheticDockModel
                    ._serialize_value(
                        item
                    )
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
                _SyntheticDockModel
                ._serialize_value(
                    item
                )
                for item in value
            ]

        to_dict_method = getattr(
            value,
            "to_dict",
            None,
        )

        if callable(
            to_dict_method
        ):
            return (
                _SyntheticDockModel
                ._serialize_value(
                    to_dict_method()
                )
            )

        if hasattr(
            value,
            "__dict__",
        ):
            return {
                str(key): (
                    _SyntheticDockModel
                    ._serialize_value(
                        item
                    )
                )
                for key, item in vars(
                    value
                ).items()
                if not str(key).startswith(
                    "_"
                )
            }

        return str(
            value
        )

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        fields_to_serialize = (
            "name",
            "contacts",
            "hbonds",
            "hydrophobic",
            "pi",
            "score",
            "statistics",
            "files",
            "metadata",
        )

        return {
            field_name: self._serialize_value(
                getattr(
                    self,
                    field_name,
                )
            )
            for field_name in fields_to_serialize
        }


def _test_integration_and_compatibility() -> None:
    """Test DockModel, serialization and ChimeraX-like integration."""

    import json

    atoms = _SyntheticChimeraXAtoms(
        [
            _SyntheticChimeraXAtom(
                [
                    0.0,
                    0.0,
                    0.0,
                ]
            ),
            _SyntheticChimeraXAtom(
                [
                    1.0,
                    0.0,
                    0.0,
                ]
            ),
            _SyntheticChimeraXAtom(
                [
                    0.0,
                    1.0,
                    0.0,
                ]
            ),
        ]
    )

    _assert_close(
        get_coordinates(
            atoms,
            scene=False,
        ),
        [
            [
                0.0,
                0.0,
                0.0,
            ],
            [
                1.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                1.0,
                0.0,
            ],
        ],
        message=(
            "ChimeraX-like local coordinates "
            "were not extracted correctly."
        ),
    )

    _assert_close(
        get_coordinates(
            atoms,
            scene=True,
        ),
        [
            [
                10.0,
                0.0,
                0.0,
            ],
            [
                11.0,
                0.0,
                0.0,
            ],
            [
                10.0,
                1.0,
                0.0,
            ],
        ],
        message=(
            "ChimeraX-like scene coordinates "
            "were not preferred correctly."
        ),
    )

    contact = contact_geometry(
        [
            0.0,
            0.0,
            0.0,
        ],
        [
            0.0,
            0.0,
            3.0,
        ],
        cutoff=4.0,
    )

    hydrogen_bond = (
        hydrogen_bond_geometry(
            [
                0.0,
                0.0,
                0.0,
            ],
            [
                2.8,
                0.0,
                0.0,
            ],
            [
                1.0,
                0.0,
                0.0,
            ],
        )
    )

    first_ring = RingGeometry(
        _synthetic_hexagonal_ring(
            center=[
                0.0,
                0.0,
                0.0,
            ]
        )
    )

    second_ring = RingGeometry(
        _synthetic_hexagonal_ring(
            center=[
                0.0,
                0.0,
                3.5,
            ]
        )
    )

    stacking = pi_stack_geometry(
        first_ring,
        second_ring,
    )

    cation = cation_pi_geometry(
        first_ring,
        [
            0.0,
            0.0,
            2.0,
        ],
    )

    bounding_box = create_bounding_box(
        [
            [
                0.0,
                0.0,
                0.0,
            ],
            [
                1.0,
                2.0,
                3.0,
            ],
        ]
    )

    structured_results = (
        contact,
        hydrogen_bond,
        first_ring,
        stacking,
        bounding_box,
    )

    for structured_result in (
        structured_results
    ):
        serialized_result = (
            structural_result_to_dict(
                structured_result
            )
        )

        json.dumps(
            serialized_result,
            allow_nan=False,
        )

    dock_model = _SyntheticDockModel(
        name="synthetic_pose"
    )

    dock_model.contacts.append(
        contact
    )

    dock_model.hbonds.append(
        hydrogen_bond
    )

    dock_model.pi[
        "stacking"
    ].append(
        stacking
    )

    dock_model.pi[
        "cation"
    ].append(
        cation
    )

    dock_model.metadata[
        "bounding_box"
    ] = bounding_box

    serialized_model = (
        dock_model.to_dict()
    )

    json.dumps(
        serialized_model,
        allow_nan=False,
    )

    _assert_true(
        serialized_model[
            "contacts"
        ][0][
            "contact_compatible"
        ],
        "DockModel contact serialization failed.",
    )

    _assert_true(
        serialized_model[
            "hbonds"
        ][0][
            "geometry_compatible"
        ],
        "DockModel hydrogen-bond serialization failed.",
    )

    _assert_true(
        serialized_model[
            "pi"
        ][
            "stacking"
        ][0][
            "classification"
        ] == "parallel",
        "DockModel pi-stacking serialization failed.",
    )

    empty_coordinates = get_coordinates(
        None,
        allow_empty=True,
    )

    _assert_true(
        empty_coordinates.shape
        == (
            0,
            3,
        ),
        "Empty coordinate integration returned "
        "an invalid shape.",
    )

    empty_distances = distance_matrix(
        empty_coordinates,
        allow_empty=True,
    )

    _assert_true(
        empty_distances.shape
        == (
            0,
            0,
        ),
        "Empty distance matrix returned an "
        "invalid shape.",
    )

    _assert_raises(
        ValueError,
        validate_coordinate,
        [
            1.0,
            2.0,
        ],
        message=(
            "Invalid coordinate shape must be "
            "rejected during integration."
        ),
    )

    _assert_raises(
        ValueError,
        validate_coordinate,
        [
            1.0,
            np.nan,
            3.0,
        ],
        message=(
            "Non-finite coordinates must be "
            "rejected during integration."
        ),
    )

    _assert_raises(
        ValueError,
        fit_plane,
        [
            [
                0.0,
                0.0,
                0.0,
            ],
            [
                1.0,
                0.0,
                0.0,
            ],
            [
                2.0,
                0.0,
                0.0,
            ],
        ],
        message=(
            "Degenerate synthetic planes must "
            "be rejected."
        ),
    )

    _assert_raises(
        ValueError,
        closest_atoms,
        [],
        [
            [
                0.0,
                0.0,
                0.0,
            ]
        ],
        message=(
            "Empty atom collections must be "
            "rejected by closest_atoms()."
        ),
    )


# -----------------------------------------------------------------------------
# Public self-test runner
# -----------------------------------------------------------------------------

def run_self_tests(
    *,
    verbose: bool = True,
    raise_on_failure: bool = True,
) -> Dict[str, Any]:
    """
    Run the geometry module's internal synthetic tests.

    Parameters
    ----------
    verbose : bool, optional
        Whether individual test-group results should be printed.
    raise_on_failure : bool, optional
        Whether an ``AssertionError`` should be raised when one or more test
        groups fail.

    Returns
    -------
    dict
        Test summary containing:

        - ``passed``;
        - ``failed``;
        - ``total``;
        - ``success``;
        - ``results``.

    Raises
    ------
    AssertionError
        If at least one group fails and ``raise_on_failure=True``.

    Notes
    -----
    These tests use only synthetic coordinates and NumPy. ChimeraX is not
    required.
    """

    test_groups: List[
        Tuple[
            str,
            Callable[
                [],
                None,
            ],
        ]
    ] = [
        (
            "Coordinates and vectors",
            _test_coordinates_and_vectors,
        ),
        (
            "Distances and angles",
            _test_distances_and_angles,
        ),
        (
            "Planes",
            _test_planes,
        ),
        (
            "Rings and pi interactions",
            _test_rings_and_pi_interactions,
        ),
        (
            "Hydrogen bonds",
            _test_hydrogen_bonds,
        ),
        (
            "Contacts",
            _test_contacts,
        ),
        (
            "RMSD and alignment",
            _test_rmsd_and_alignment,
        ),
        (
            "Bounding geometry",
            _test_bounding_geometry,
        ),
        (
            "Structured results",
            _test_structured_results,
        ),
        (
            "Integration and compatibility",
            _test_integration_and_compatibility,
        ),
    ]

    if verbose:
        print()
        print(
            "=" * 72
        )
        print(
            "DockAnalyzer geometry.py self-test"
        )
        print(
            "=" * 72
        )

    results: List[
        Dict[str, Any]
    ] = []

    for test_name, test_function in test_groups:
        (
            result_name,
            passed,
            error_message,
        ) = _run_test_group(
            test_name,
            test_function,
            verbose=verbose,
        )

        results.append(
            {
                "name": result_name,
                "passed": passed,
                "error": error_message,
            }
        )

    passed_count = sum(
        1
        for result in results
        if result[
            "passed"
        ]
    )

    failed_count = (
        len(
            results
        )
        - passed_count
    )

    success = (
        failed_count == 0
    )

    summary: Dict[str, Any] = {
        "passed": passed_count,
        "failed": failed_count,
        "total": len(
            results
        ),
        "success": success,
        "results": results,
    }

    if verbose:
        print(
            "-" * 72
        )

        print(
            f"Passed: {passed_count}"
        )

        print(
            f"Failed: {failed_count}"
        )

        print(
            f"Total:  {len(results)}"
        )

        if success:
            print(
                "Result: ALL TESTS PASSED"
            )

        else:
            print(
                "Result: TEST FAILURE"
            )

        print(
            "=" * 72
        )
        print()

    if (
        not success
        and raise_on_failure
    ):
        failed_names = [
            result[
                "name"
            ]
            for result in results
            if not result[
                "passed"
            ]
        ]

        raise AssertionError(
            "geometry.py self-test failed in: "
            + ", ".join(
                failed_names
            )
        )

    return summary


# -----------------------------------------------------------------------------
# Public module interface
# -----------------------------------------------------------------------------

_SECTION_15_PUBLIC_NAMES = [
    "run_self_tests",
]

_extend_public_names(_SECTION_15_PUBLIC_NAMES)


# -----------------------------------------------------------------------------
# Public API validation
# -----------------------------------------------------------------------------

def _validate_public_api() -> None:
    """Validate the exported public-name contract."""

    if not isinstance(__all__, list):
        raise TypeError("__all__ must be a list of public names.")

    invalid_names = [
        name
        for name in __all__
        if not isinstance(name, str) or not name
    ]

    if invalid_names:
        raise TypeError(
            "__all__ must contain only non-empty strings."
        )

    seen: set[str] = set()
    duplicate_names: List[str] = []

    for name in __all__:
        if name in seen and name not in duplicate_names:
            duplicate_names.append(name)
        seen.add(name)

    if duplicate_names:
        raise RuntimeError(
            "Duplicate public names in __all__: "
            + ", ".join(duplicate_names)
        )

    missing_names = [
        name
        for name in __all__
        if name not in globals()
    ]

    if missing_names:
        raise RuntimeError(
            "Missing public names declared in __all__: "
            + ", ".join(missing_names)
        )


_validate_public_api()


# -----------------------------------------------------------------------------
# Standalone execution
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    run_self_tests(
        verbose=True,
        raise_on_failure=True,
    )


# =============================================================================
# End of Section 15
# =============================================================================
